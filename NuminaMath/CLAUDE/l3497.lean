import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_l3497_349723

/-- The radius of the circle described by the equation x^2 + y^2 - 6x + 8y = 0 is 5 -/
theorem circle_radius (x y : ℝ) : x^2 + y^2 - 6*x + 8*y = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 5^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3497_349723


namespace NUMINAMATH_CALUDE_ninth_triangle_shaded_fraction_l3497_349789

/- Define the sequence of shaded triangles -/
def shaded_triangles (n : ℕ) : ℕ := 2 * n - 1

/- Define the sequence of total triangles -/
def total_triangles (n : ℕ) : ℕ := 4^(n - 1)

/- Theorem statement -/
theorem ninth_triangle_shaded_fraction :
  shaded_triangles 9 / total_triangles 9 = 17 / 65536 := by
  sorry

end NUMINAMATH_CALUDE_ninth_triangle_shaded_fraction_l3497_349789


namespace NUMINAMATH_CALUDE_range_of_a_l3497_349791

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) ∧ 
  (∃ x : ℝ, |x - a| < 2 ∧ (x < 1 ∨ x > 3)) ↔ 
  1 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3497_349791


namespace NUMINAMATH_CALUDE_tan_triple_angle_l3497_349781

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_tan_triple_angle_l3497_349781


namespace NUMINAMATH_CALUDE_conic_eccentricity_l3497_349731

/-- Given that 4, m, 1 form a geometric sequence, 
    the eccentricity of x²/m + y² = 1 is √2/2 or √3 -/
theorem conic_eccentricity (m : ℝ) : 
  (4 * 1 = m^2) →  -- Geometric sequence condition
  (∃ (e : ℝ), (e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3) ∧
   ∀ (x y : ℝ), x^2 / m + y^2 = 1 → 
   (∃ (a b : ℝ), 
     (m > 0 → x^2 / a^2 + y^2 / b^2 = 1 ∧ e = Real.sqrt (1 - b^2 / a^2)) ∧
     (m < 0 → y^2 / a^2 - x^2 / b^2 = 1 ∧ e = Real.sqrt (1 + a^2 / b^2)))) :=
by sorry

end NUMINAMATH_CALUDE_conic_eccentricity_l3497_349731


namespace NUMINAMATH_CALUDE_dance_theorem_l3497_349767

/-- Represents a dance function with boys and girls -/
structure DanceFunction where
  boys : ℕ
  girls : ℕ
  first_boy_dances : ℕ
  last_boy_dances_all : Prop

/-- The relationship between boys and girls in the dance function -/
def dance_relationship (df : DanceFunction) : Prop :=
  df.boys = df.girls - df.first_boy_dances + 1

theorem dance_theorem (df : DanceFunction) 
  (h1 : df.first_boy_dances = 6)
  (h2 : df.last_boy_dances_all)
  : df.boys = df.girls - 5 := by
  sorry

end NUMINAMATH_CALUDE_dance_theorem_l3497_349767


namespace NUMINAMATH_CALUDE_problem_statement_l3497_349724

theorem problem_statement (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) :
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3497_349724


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3497_349719

def C : Set Nat := {54, 56, 59, 63, 65}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∃ (p : Nat), Nat.Prime p ∧ p ∣ n ∧
    ∀ (m : Nat) (q : Nat), m ∈ C → Nat.Prime q → q ∣ m → p ≤ q) ∧
  (∀ (m : Nat) (q : Nat), m ∈ C → Nat.Prime q → q ∣ m → 2 ≤ q) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3497_349719


namespace NUMINAMATH_CALUDE_radical_product_simplification_l3497_349720

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 63 * q * Real.sqrt (2 * q) := by
  sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l3497_349720


namespace NUMINAMATH_CALUDE_gcd_problem_l3497_349716

theorem gcd_problem (a b c : ℕ) : 
  a * b * c = 2^4 * 3^2 * 5^3 →
  Nat.gcd a b = 15 →
  Nat.gcd a c = 5 →
  Nat.gcd b c = 20 →
  (a = 15 ∧ b = 60 ∧ c = 20) := by
sorry

end NUMINAMATH_CALUDE_gcd_problem_l3497_349716


namespace NUMINAMATH_CALUDE_min_value_expression_l3497_349792

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 8*x₀ + 6*y₀ + 25 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3497_349792


namespace NUMINAMATH_CALUDE_undefined_fraction_l3497_349727

theorem undefined_fraction (a : ℝ) : 
  ¬ (∃ x : ℝ, x = (a + 3) / (a^2 - 9)) ↔ a = -3 ∨ a = 3 := by
  sorry

end NUMINAMATH_CALUDE_undefined_fraction_l3497_349727


namespace NUMINAMATH_CALUDE_orange_weight_change_l3497_349756

theorem orange_weight_change (initial_weight : ℝ) (initial_water_percent : ℝ) (water_decrease : ℝ) : 
  initial_weight = 5 →
  initial_water_percent = 95 →
  water_decrease = 5 →
  let non_water_weight := initial_weight * (100 - initial_water_percent) / 100
  let new_water_percent := initial_water_percent - water_decrease
  let new_total_weight := non_water_weight / ((100 - new_water_percent) / 100)
  new_total_weight = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_orange_weight_change_l3497_349756


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3497_349798

theorem slope_angle_of_line (x y : ℝ) : 
  x - y + 3 = 0 → Real.arctan 1 = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3497_349798


namespace NUMINAMATH_CALUDE_max_value_quadratic_l3497_349769

theorem max_value_quadratic (x : ℝ) :
  let y : ℝ → ℝ := λ x => -3 * x^2 + 4 * x + 6
  ∃ (max_y : ℝ), ∀ (x : ℝ), y x ≤ max_y ∧ max_y = 22/3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l3497_349769


namespace NUMINAMATH_CALUDE_rotation_90_degrees_l3497_349715

theorem rotation_90_degrees (z : ℂ) : z = -4 - I → z * I = 1 - 4*I := by
  sorry

end NUMINAMATH_CALUDE_rotation_90_degrees_l3497_349715


namespace NUMINAMATH_CALUDE_product_one_plus_minus_sqrt_three_l3497_349712

theorem product_one_plus_minus_sqrt_three : (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_product_one_plus_minus_sqrt_three_l3497_349712


namespace NUMINAMATH_CALUDE_base_b_square_theorem_l3497_349772

/-- Converts a number from base b representation to base 10 -/
def base_b_to_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => b * acc + digit) 0

/-- Theorem: If 1325 in base b is the square of 35 in base b, then b = 7 in base 10 -/
theorem base_b_square_theorem :
  ∀ b : Nat,
  (base_b_to_10 [1, 3, 2, 5] b = (base_b_to_10 [3, 5] b) ^ 2) →
  b = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_b_square_theorem_l3497_349772


namespace NUMINAMATH_CALUDE_wallace_jerky_production_l3497_349730

/-- Represents the jerky production scenario -/
structure JerkyProduction where
  total_order : ℕ
  already_made : ℕ
  days_to_fulfill : ℕ
  batches_per_day : ℕ

/-- Calculates the number of bags one batch can make -/
def bags_per_batch (jp : JerkyProduction) : ℕ :=
  ((jp.total_order - jp.already_made) / jp.days_to_fulfill) / jp.batches_per_day

/-- Theorem stating that under the given conditions, one batch makes 10 bags -/
theorem wallace_jerky_production :
  ∀ (jp : JerkyProduction),
    jp.total_order = 60 →
    jp.already_made = 20 →
    jp.days_to_fulfill = 4 →
    jp.batches_per_day = 1 →
    bags_per_batch jp = 10 := by
  sorry

end NUMINAMATH_CALUDE_wallace_jerky_production_l3497_349730


namespace NUMINAMATH_CALUDE_sin_70_degrees_l3497_349725

theorem sin_70_degrees (a : ℝ) (h : Real.sin (10 * π / 180) = a) : 
  Real.sin (70 * π / 180) = 1 - 2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_70_degrees_l3497_349725


namespace NUMINAMATH_CALUDE_unique_prime_base_l3497_349793

theorem unique_prime_base (b : ℕ) : 
  Prime b ∧ (b + 5)^2 = 3*b^2 + 6*b + 1 → b = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_base_l3497_349793


namespace NUMINAMATH_CALUDE_trajectory_of_M_equation_of_l_area_of_POM_smallest_circle_l3497_349710

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0

-- Define point P
def P : ℝ × ℝ := (2, 2)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the midpoint M of chord AB
def M (x y : ℝ) : Prop := ∃ (a b : ℝ × ℝ), 
  circle_C a.1 a.2 ∧ circle_C b.1 b.2 ∧ 
  x = (a.1 + b.1) / 2 ∧ y = (a.2 + b.2) / 2

-- Theorem 1: Trajectory of M
theorem trajectory_of_M : 
  ∀ x y : ℝ, M x y → (x - 1)^2 + (y - 3)^2 = 2 :=
sorry

-- Theorem 2a: Equation of line l when |OP| = |OM|
theorem equation_of_l : 
  ∃ x y : ℝ, M x y ∧ (x^2 + y^2 = P.1^2 + P.2^2) → 
  ∀ x y : ℝ, y = -1/3 * x + 8/3 :=
sorry

-- Theorem 2b: Area of triangle POM when |OP| = |OM|
theorem area_of_POM : 
  ∃ x y : ℝ, M x y ∧ (x^2 + y^2 = P.1^2 + P.2^2) → 
  (1/2) * |P.1 * y - P.2 * x| = 16/5 :=
sorry

-- Theorem 3: Equation of smallest circle through intersection of C and l
theorem smallest_circle : 
  ∃ x y : ℝ, circle_C x y ∧ y = -1/3 * x + 8/3 → 
  ∀ x y : ℝ, (x + 2/5)^2 + (y - 14/5)^2 = 72/5 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_equation_of_l_area_of_POM_smallest_circle_l3497_349710


namespace NUMINAMATH_CALUDE_list_price_is_40_l3497_349788

/-- The list price of the item. -/
def list_price : ℝ := 40

/-- Alice's selling price. -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price. -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Alice's commission rate. -/
def alice_rate : ℝ := 0.15

/-- Bob's commission rate. -/
def bob_rate : ℝ := 0.25

/-- Alice's commission. -/
def alice_commission (x : ℝ) : ℝ := alice_rate * alice_price x

/-- Bob's commission. -/
def bob_commission (x : ℝ) : ℝ := bob_rate * bob_price x

theorem list_price_is_40 :
  alice_commission list_price = bob_commission list_price ∧
  list_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_list_price_is_40_l3497_349788


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3497_349766

/-- A quadratic equation ax² + bx + c = 0 -/
structure QuadraticEq where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The discriminant of a quadratic equation -/
def discriminant (q : QuadraticEq) : ℝ := q.b^2 - 4*q.a*q.c

/-- A quadratic equation has real roots iff its discriminant is non-negative -/
def has_real_roots (q : QuadraticEq) : Prop := discriminant q ≥ 0

theorem sufficient_but_not_necessary_condition 
  (q1 q2 : QuadraticEq) 
  (h1 : has_real_roots q1)
  (h2 : has_real_roots q2)
  (h3 : q1.a ≠ q2.a) :
  (∀ w c, w * c > 0 → 
    has_real_roots ⟨q2.a, q1.b, q1.c⟩ ∨ has_real_roots ⟨q1.a, q2.b, q2.c⟩) ∧ 
  (∃ w c, w * c ≤ 0 ∧ 
    (has_real_roots ⟨q2.a, q1.b, q1.c⟩ ∨ has_real_roots ⟨q1.a, q2.b, q2.c⟩)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3497_349766


namespace NUMINAMATH_CALUDE_milk_cost_proof_l3497_349718

def total_cost : ℕ := 42
def banana_cost : ℕ := 12
def bread_cost : ℕ := 9
def apple_cost : ℕ := 14

theorem milk_cost_proof :
  total_cost - (banana_cost + bread_cost + apple_cost) = 7 := by
sorry

end NUMINAMATH_CALUDE_milk_cost_proof_l3497_349718


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_nonagon_diagonal_intersection_probability_proof_l3497_349714

/-- The probability of two randomly chosen diagonals intersecting inside a regular nonagon -/
theorem nonagon_diagonal_intersection_probability : ℚ :=
  14 / 39

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of diagonals in a nonagon -/
def nonagon_diagonals : ℕ := (nonagon_sides.choose 2) - nonagon_sides

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def diagonal_pairs : ℕ := nonagon_diagonals.choose 2

/-- The number of ways to choose 4 points from the nonagon vertices -/
def intersecting_diagonal_sets : ℕ := nonagon_sides.choose 4

theorem nonagon_diagonal_intersection_probability_proof :
  (intersecting_diagonal_sets : ℚ) / diagonal_pairs = nonagon_diagonal_intersection_probability :=
sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_nonagon_diagonal_intersection_probability_proof_l3497_349714


namespace NUMINAMATH_CALUDE_novel_writing_speed_l3497_349790

/-- Calculates the average writing speed given the total number of words and hours spent writing. -/
def average_writing_speed (total_words : ℕ) (total_hours : ℕ) : ℚ :=
  total_words / total_hours

/-- Theorem stating that for a novel with 60,000 words completed in 120 hours, 
    the average writing speed is 500 words per hour. -/
theorem novel_writing_speed :
  average_writing_speed 60000 120 = 500 := by
  sorry

end NUMINAMATH_CALUDE_novel_writing_speed_l3497_349790


namespace NUMINAMATH_CALUDE_subway_speed_comparison_l3497_349709

-- Define the speed function
def speed (s : ℝ) : ℝ := s^2 + 2*s

-- Define the theorem
theorem subway_speed_comparison :
  ∃! t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ speed 5 = speed t + 20 ∧ t = 3 := by
  sorry

end NUMINAMATH_CALUDE_subway_speed_comparison_l3497_349709


namespace NUMINAMATH_CALUDE_f_strictly_decreasing_l3497_349795

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem f_strictly_decreasing : ∀ x ∈ Set.Ioo 0 2, StrictMonoOn f (Set.Ioo 0 2) := by
  sorry

end NUMINAMATH_CALUDE_f_strictly_decreasing_l3497_349795


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l3497_349702

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (incorrect_value : ℚ) 
  (correct_value : ℚ) : 
  n = 10 → 
  initial_average = 15 → 
  incorrect_value = 26 → 
  correct_value = 36 → 
  (n : ℚ) * initial_average + (correct_value - incorrect_value) = n * 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l3497_349702


namespace NUMINAMATH_CALUDE_intersection_union_sets_l3497_349759

theorem intersection_union_sets : 
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {1, 2, 3, 4}
  let P : Set ℕ := {2, 3, 4, 5}
  (M ∩ N) ∪ P = {1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_union_sets_l3497_349759


namespace NUMINAMATH_CALUDE_problem_statement_l3497_349701

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (n : ℕ), a^5 % 10 = b^5 % 10 → a - b = 10 * n) ∧
  (a^2 - b^2 = 1940 → a = 102 ∧ b = 92) ∧
  (a^2 - b^2 = 1920 → 
    ((a = 101 ∧ b = 91) ∨ 
     (a = 58 ∧ b = 38) ∨ 
     (a = 47 ∧ b = 17) ∨ 
     (a = 44 ∧ b = 4))) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3497_349701


namespace NUMINAMATH_CALUDE_bobs_driving_speed_l3497_349728

/-- Bob's driving problem -/
theorem bobs_driving_speed (initial_speed : ℝ) (initial_time : ℝ) (construction_time : ℝ) (total_time : ℝ) (total_distance : ℝ) :
  initial_speed = 60 →
  initial_time = 1.5 →
  construction_time = 2 →
  total_time = 3.5 →
  total_distance = 180 →
  (initial_speed * initial_time + construction_time * ((total_distance - initial_speed * initial_time) / construction_time) = total_distance) →
  (total_distance - initial_speed * initial_time) / construction_time = 45 :=
by sorry

end NUMINAMATH_CALUDE_bobs_driving_speed_l3497_349728


namespace NUMINAMATH_CALUDE_szilveszter_age_l3497_349735

def birth_year (a b : ℕ) := 1900 + 10 * a + b

def grandfather_birth_year (a b : ℕ) := 1910 + a + b

def current_year := 1999

theorem szilveszter_age (a b : ℕ) 
  (h1 : a < 10 ∧ b < 10) 
  (h2 : 1 + 9 + a + b = current_year - grandfather_birth_year a b) 
  (h3 : 10 * a + b = current_year - grandfather_birth_year a b) :
  current_year - birth_year a b = 23 := by
sorry

end NUMINAMATH_CALUDE_szilveszter_age_l3497_349735


namespace NUMINAMATH_CALUDE_april_spending_l3497_349785

def initial_savings : ℕ := 11000
def february_percentage : ℚ := 20 / 100
def march_percentage : ℚ := 40 / 100
def remaining_savings : ℕ := 2900

theorem april_spending :
  let february_spending := (february_percentage * initial_savings).floor
  let march_spending := (march_percentage * initial_savings).floor
  let total_spent := initial_savings - remaining_savings
  total_spent - february_spending - march_spending = 1500 := by sorry

end NUMINAMATH_CALUDE_april_spending_l3497_349785


namespace NUMINAMATH_CALUDE_greater_number_problem_l3497_349784

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x > y) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l3497_349784


namespace NUMINAMATH_CALUDE_count_solutions_l3497_349754

/-- The number of ordered pairs of complex numbers satisfying the given equations -/
def num_solutions : ℕ := 29

/-- Predicate to check if a pair of complex numbers satisfies the equations -/
def satisfies_equations (a b : ℂ) : Prop :=
  a^3 * b^5 = 1 ∧ a^7 * b^2 = 1

theorem count_solutions :
  (∃! (s : Finset (ℂ × ℂ)), s.card = num_solutions ∧ 
    ∀ (p : ℂ × ℂ), p ∈ s ↔ satisfies_equations p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_solutions_l3497_349754


namespace NUMINAMATH_CALUDE_movie_profit_calculation_l3497_349721

/-- Calculate profit for a movie given its earnings and costs -/
def movie_profit (
  opening_weekend : ℝ
  ) (
  domestic_multiplier : ℝ
  ) (
  international_multiplier : ℝ
  ) (
  domestic_tax_rate : ℝ
  ) (
  international_tax_rate : ℝ
  ) (
  royalty_rate : ℝ
  ) (
  production_cost : ℝ
  ) (
  marketing_cost : ℝ
  ) : ℝ :=
  let domestic_earnings := opening_weekend * domestic_multiplier
  let international_earnings := domestic_earnings * international_multiplier
  let domestic_after_tax := domestic_earnings * domestic_tax_rate
  let international_after_tax := international_earnings * international_tax_rate
  let total_after_tax := domestic_after_tax + international_after_tax
  let total_earnings := domestic_earnings + international_earnings
  let royalties := total_earnings * royalty_rate
  total_after_tax - royalties - production_cost - marketing_cost

/-- The profit calculation for the given movie is correct -/
theorem movie_profit_calculation :
  movie_profit 120 3.5 1.8 0.6 0.45 0.05 60 40 = 433.4 :=
by sorry

end NUMINAMATH_CALUDE_movie_profit_calculation_l3497_349721


namespace NUMINAMATH_CALUDE_simplify_expression_l3497_349734

theorem simplify_expression (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 4) * (8 * p - 12) = 40 * p - 42 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3497_349734


namespace NUMINAMATH_CALUDE_trapezium_other_side_length_l3497_349742

/-- Theorem: In a trapezium with one parallel side of 18 cm, a distance between parallel sides of 10 cm,
    and an area of 190 square centimeters, the length of the other parallel side is 20 cm. -/
theorem trapezium_other_side_length (a b h : ℝ) (h1 : a = 18) (h2 : h = 10) (h3 : (a + b) * h / 2 = 190) :
  b = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_other_side_length_l3497_349742


namespace NUMINAMATH_CALUDE_may_scarf_count_l3497_349799

/-- Represents the number of scarves that can be made from one yarn of a given color -/
def scarvesPerYarn (color : String) : ℕ :=
  match color with
  | "red" => 3
  | "blue" => 2
  | "yellow" => 4
  | "green" => 5
  | "purple" => 6
  | _ => 0

/-- Represents the number of yarns May has for each color -/
def yarnCount (color : String) : ℕ :=
  match color with
  | "red" => 1
  | "blue" => 1
  | "yellow" => 1
  | "green" => 3
  | "purple" => 2
  | _ => 0

/-- The list of colors May has yarn for -/
def colors : List String := ["red", "blue", "yellow", "green", "purple"]

/-- The total number of scarves May can make -/
def totalScarves : ℕ := (colors.map (fun c => scarvesPerYarn c * yarnCount c)).sum

theorem may_scarf_count : totalScarves = 36 := by
  sorry

end NUMINAMATH_CALUDE_may_scarf_count_l3497_349799


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l3497_349722

theorem sum_of_reciprocal_squares (p q r : ℝ) : 
  p^3 - 9*p^2 + 8*p + 2 = 0 →
  q^3 - 9*q^2 + 8*q + 2 = 0 →
  r^3 - 9*r^2 + 8*r + 2 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l3497_349722


namespace NUMINAMATH_CALUDE_intersection_range_a_l3497_349771

/-- The range of a for which the curves x^2 + 4(y - a)^2 = 4 and x^2 = 4y intersect -/
theorem intersection_range_a :
  ∀ a : ℝ,
  (∃ x y : ℝ, x^2 + 4*(y - a)^2 = 4 ∧ x^2 = 4*y) →
  -1 ≤ a ∧ a < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_a_l3497_349771


namespace NUMINAMATH_CALUDE_shopping_problem_l3497_349762

theorem shopping_problem (total : ℕ) (stores : ℕ) (initial_amount : ℕ) :
  total = stores ∧ 
  initial_amount = 100 ∧ 
  stores = 6 → 
  ∃ (spent_per_store : ℕ), 
    spent_per_store * stores ≤ initial_amount ∧ 
    spent_per_store > 0 ∧
    initial_amount - spent_per_store * stores ≤ 28 :=
by sorry

#check shopping_problem

end NUMINAMATH_CALUDE_shopping_problem_l3497_349762


namespace NUMINAMATH_CALUDE_linear_function_comparison_inverse_proportion_comparison_l3497_349711

-- Linear function
theorem linear_function_comparison (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 1) 
  (h2 : y₂ = -2 * x₂ + 1) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by sorry

-- Inverse proportion function
theorem inverse_proportion_comparison (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_linear_function_comparison_inverse_proportion_comparison_l3497_349711


namespace NUMINAMATH_CALUDE_perpendicular_line_through_circle_center_l3497_349758

/-- Given a circle with equation x^2 + 2x + y^2 = 0 and a line x + y = 0,
    prove that x - y + 1 = 0 is the equation of the line passing through
    the center of the circle and perpendicular to the given line. -/
theorem perpendicular_line_through_circle_center :
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + 2*p.1 + p.2^2 = 0
  let given_line : ℝ × ℝ → Prop := λ p => p.1 + p.2 = 0
  let perpendicular_line : ℝ × ℝ → Prop := λ p => p.1 - p.2 + 1 = 0
  let center : ℝ × ℝ := (-1, 0)
  (∀ p, circle p ↔ (p.1 + 1)^2 + p.2^2 = 1) →
  perpendicular_line center ∧
  (∀ p q : ℝ × ℝ, p ≠ q →
    given_line p ∧ given_line q →
    perpendicular_line p ∧ perpendicular_line q →
    (p.1 - q.1) * (p.1 - q.1 + q.2 - p.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_circle_center_l3497_349758


namespace NUMINAMATH_CALUDE_irrational_minus_rational_is_irrational_pi_minus_3_14_irrational_l3497_349764

/-- π is irrational -/
axiom pi_irrational : Irrational Real.pi

/-- 3.14 is rational -/
axiom rational_3_14 : ∃ (q : ℚ), (q : ℝ) = 3.14

/-- The difference of an irrational number and a rational number is irrational -/
theorem irrational_minus_rational_is_irrational (x y : ℝ) (hx : Irrational x) (hy : ∃ (q : ℚ), (q : ℝ) = y) :
  Irrational (x - y) :=
sorry

/-- π - 3.14 is irrational -/
theorem pi_minus_3_14_irrational : Irrational (Real.pi - 3.14) :=
  irrational_minus_rational_is_irrational Real.pi 3.14 pi_irrational rational_3_14

end NUMINAMATH_CALUDE_irrational_minus_rational_is_irrational_pi_minus_3_14_irrational_l3497_349764


namespace NUMINAMATH_CALUDE_remainder_of_large_number_l3497_349794

theorem remainder_of_large_number (n : ℕ) (d : ℕ) (h : n = 123456789012 ∧ d = 210) :
  n % d = 17 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_large_number_l3497_349794


namespace NUMINAMATH_CALUDE_mariams_neighborhood_houses_l3497_349780

/-- The number of houses in Mariam's neighborhood -/
def total_houses (houses_one_side : ℕ) (multiplier : ℕ) : ℕ :=
  houses_one_side + houses_one_side * multiplier

/-- Theorem stating the total number of houses in Mariam's neighborhood -/
theorem mariams_neighborhood_houses : 
  total_houses 40 3 = 160 := by sorry

end NUMINAMATH_CALUDE_mariams_neighborhood_houses_l3497_349780


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l3497_349732

theorem wrong_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (correct_avg : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 185)
  (h3 : actual_height = 106)
  (h4 : correct_avg = 183) :
  ∃ wrong_height : ℝ, 
    wrong_height = n * initial_avg - (n * correct_avg - actual_height) := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l3497_349732


namespace NUMINAMATH_CALUDE_parallelepiped_volume_and_lateral_area_l3497_349763

/-- 
Given a right parallelepiped with a rhombus base of area Q and diagonal section areas S₁ and S₂,
this theorem proves the formulas for its volume and lateral surface area.
-/
theorem parallelepiped_volume_and_lateral_area (Q S₁ S₂ : ℝ) 
  (hQ : Q > 0) (hS₁ : S₁ > 0) (hS₂ : S₂ > 0) :
  ∃ (V LSA : ℝ),
    V = Real.sqrt ((S₁ * S₂ * Q) / 2) ∧ 
    LSA = 2 * Real.sqrt (S₁^2 + S₂^2) := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_volume_and_lateral_area_l3497_349763


namespace NUMINAMATH_CALUDE_probability_five_heads_seven_flips_l3497_349770

theorem probability_five_heads_seven_flips :
  let n : ℕ := 7  -- total number of flips
  let k : ℕ := 5  -- number of heads we want
  let p : ℚ := 1/2  -- probability of heads on a single flip (fair coin)
  Nat.choose n k * p^k * (1 - p)^(n - k) = 21/128 :=
by sorry

end NUMINAMATH_CALUDE_probability_five_heads_seven_flips_l3497_349770


namespace NUMINAMATH_CALUDE_monthly_expenses_calculation_l3497_349768

/-- Calculates monthly expenses given initial investment, monthly revenue, and payback period. -/
def calculate_monthly_expenses (initial_investment : ℕ) (monthly_revenue : ℕ) (payback_months : ℕ) : ℕ :=
  (monthly_revenue * payback_months - initial_investment) / payback_months

theorem monthly_expenses_calculation (initial_investment monthly_revenue payback_months : ℕ) 
  (h1 : initial_investment = 25000)
  (h2 : monthly_revenue = 4000)
  (h3 : payback_months = 10) :
  calculate_monthly_expenses initial_investment monthly_revenue payback_months = 1500 := by
  sorry

#eval calculate_monthly_expenses 25000 4000 10

end NUMINAMATH_CALUDE_monthly_expenses_calculation_l3497_349768


namespace NUMINAMATH_CALUDE_download_time_is_450_minutes_l3497_349751

-- Define the problem parameters
def min_speed : ℝ := 20
def max_speed : ℝ := 40
def avg_speed : ℝ := 30
def program_a_size : ℝ := 450
def program_b_size : ℝ := 240
def program_c_size : ℝ := 120
def mb_per_gb : ℝ := 1000
def seconds_per_minute : ℝ := 60

-- State the theorem
theorem download_time_is_450_minutes :
  let total_size := (program_a_size + program_b_size + program_c_size) * mb_per_gb
  let download_time_seconds := total_size / avg_speed
  let download_time_minutes := download_time_seconds / seconds_per_minute
  download_time_minutes = 450 := by
sorry

end NUMINAMATH_CALUDE_download_time_is_450_minutes_l3497_349751


namespace NUMINAMATH_CALUDE_consecutive_roots_quadratic_l3497_349776

theorem consecutive_roots_quadratic (n : ℕ) (hn : n > 1) :
  let f : ℝ → ℝ := λ x => x^2 - (2*n - 1)*x + n*(n-1)
  (f (n - 1) = 0) ∧ (f n = 0) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_roots_quadratic_l3497_349776


namespace NUMINAMATH_CALUDE_M_is_empty_l3497_349705

def M : Set ℝ := {x | x^4 + 4*x^2 - 12*x + 8 = 0 ∧ x > 0}

theorem M_is_empty : M = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_is_empty_l3497_349705


namespace NUMINAMATH_CALUDE_diamond_computation_l3497_349787

-- Define the ⋄ operation
def diamond (a b : ℚ) : ℚ := a - 1 / b

-- Theorem statement
theorem diamond_computation :
  (diamond (diamond 4 5) 6) - (diamond 4 (diamond 5 6)) = -139 / 870 := by
  sorry

end NUMINAMATH_CALUDE_diamond_computation_l3497_349787


namespace NUMINAMATH_CALUDE_simplify_expression_l3497_349752

theorem simplify_expression (x : ℝ) (h : x ≠ -2) :
  4 / (x + 2) + x - 2 = x^2 / (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3497_349752


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l3497_349775

/-- Represents the number of seats in each row -/
def seats : Fin 2 → Nat
  | 0 => 6  -- front row
  | 1 => 7  -- back row

/-- Calculates the number of ways to arrange 2 people in two rows of seats
    such that they are not sitting next to each other -/
def seating_arrangements : Nat :=
  let different_rows := seats 0 * seats 1 * 2
  let front_row := 2 * 4 + 4 * 3
  let back_row := 2 * 5 + 5 * 4
  different_rows + front_row + back_row

/-- Theorem stating that the number of seating arrangements is 134 -/
theorem seating_arrangements_count : seating_arrangements = 134 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l3497_349775


namespace NUMINAMATH_CALUDE_conference_seating_arrangements_l3497_349760

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def bench_arrangements (g1 g2 g3 g4 : ℕ) : ℕ :=
  factorial g1 * factorial g2 * factorial g3 * factorial g4 * factorial 4

def table_arrangements (g1 g2 g3 g4 : ℕ) : ℕ :=
  factorial g1 * factorial g2 * factorial g3 * factorial g4 * factorial 3

theorem conference_seating_arrangements :
  bench_arrangements 4 2 3 4 = 165888 ∧
  table_arrangements 4 2 3 4 = 41472 := by
  sorry

end NUMINAMATH_CALUDE_conference_seating_arrangements_l3497_349760


namespace NUMINAMATH_CALUDE_longer_strap_length_l3497_349773

theorem longer_strap_length (short long : ℕ) : 
  long = short + 72 →
  short + long = 348 →
  long = 210 :=
by sorry

end NUMINAMATH_CALUDE_longer_strap_length_l3497_349773


namespace NUMINAMATH_CALUDE_sum_odd_numbers_eq_square_last_term_eq_2n_minus_1_sum_odd_numbers_40_times_3_eq_4800_l3497_349706

/-- The sum of the first n odd numbers -/
def sum_odd_numbers (n : ℕ) : ℕ := (Finset.range n).sum (fun i => 2 * i + 1)

theorem sum_odd_numbers_eq_square (n : ℕ) : sum_odd_numbers n = n^2 :=
  by sorry

theorem last_term_eq_2n_minus_1 (n : ℕ) : 2 * n - 1 = sum_odd_numbers n - sum_odd_numbers (n - 1) :=
  by sorry

theorem sum_odd_numbers_40_times_3_eq_4800 : 3 * sum_odd_numbers 40 = 4800 :=
  by sorry

end NUMINAMATH_CALUDE_sum_odd_numbers_eq_square_last_term_eq_2n_minus_1_sum_odd_numbers_40_times_3_eq_4800_l3497_349706


namespace NUMINAMATH_CALUDE_total_people_needed_l3497_349749

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars to be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks to be lifted -/
def num_trucks : ℕ := 3

/-- Theorem stating the total number of people needed to lift the given vehicles -/
theorem total_people_needed : 
  num_cars * people_per_car + num_trucks * people_per_truck = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_people_needed_l3497_349749


namespace NUMINAMATH_CALUDE_quadratic_roots_l3497_349707

theorem quadratic_roots (a c : ℝ) (h1 : a ≠ 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 - 2*a*x + c
  (f (-1) = 0) →
  (∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 3 ∧
    ∀ x : ℝ, (a * x^2 - 2*a*x + c = 0) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3497_349707


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3497_349747

theorem linear_equation_solution (x y m : ℝ) 
  (hx : x = -1)
  (hy : y = 2)
  (hm : 5 * x + 3 * y = m) : 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3497_349747


namespace NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_9_starting_with_7_l3497_349733

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

def starts_with_7 (n : ℕ) : Prop := ∃ m : ℕ, n = 70000 + m ∧ m < 30000

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

theorem smallest_five_digit_multiple_of_9_starting_with_7 :
  ∀ n : ℕ, is_five_digit n → starts_with_7 n → is_multiple_of_9 n → n ≥ 70002 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_multiple_of_9_starting_with_7_l3497_349733


namespace NUMINAMATH_CALUDE_money_distribution_l3497_349750

theorem money_distribution (x : ℝ) (x_pos : x > 0) : 
  let adriano_initial := 5 * x
  let bruno_initial := 4 * x
  let cesar_initial := 3 * x
  let total_initial := adriano_initial + bruno_initial + cesar_initial
  let daniel_received := x + x + x
  daniel_received / total_initial = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_money_distribution_l3497_349750


namespace NUMINAMATH_CALUDE_equation_solution_l3497_349755

theorem equation_solution : ∃ x : ℝ, (10 - x)^2 = (x - 2)^2 + 8 ∧ x = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3497_349755


namespace NUMINAMATH_CALUDE_smallest_prime_minister_l3497_349745

/-- A positive integer is primer if it has a prime number of distinct prime factors. -/
def isPrimer (n : ℕ+) : Prop := sorry

/-- A positive integer is primest if it has a primer number of distinct primer factors. -/
def isPrimest (n : ℕ+) : Prop := sorry

/-- A positive integer is prime-minister if it has a primest number of distinct primest factors. -/
def isPrimeMinister (n : ℕ+) : Prop := sorry

/-- The smallest prime-minister number -/
def smallestPrimeMinister : ℕ+ := 378000

theorem smallest_prime_minister :
  isPrimeMinister smallestPrimeMinister ∧
  ∀ n : ℕ+, n < smallestPrimeMinister → ¬isPrimeMinister n := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_minister_l3497_349745


namespace NUMINAMATH_CALUDE_only_D_in_second_quadrant_l3497_349738

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (2, -3)
def point_C : ℝ × ℝ := (-2, -3)
def point_D : ℝ × ℝ := (-2, 3)

theorem only_D_in_second_quadrant :
  ¬(second_quadrant point_A.1 point_A.2) ∧
  ¬(second_quadrant point_B.1 point_B.2) ∧
  ¬(second_quadrant point_C.1 point_C.2) ∧
  second_quadrant point_D.1 point_D.2 := by sorry

end NUMINAMATH_CALUDE_only_D_in_second_quadrant_l3497_349738


namespace NUMINAMATH_CALUDE_michelle_gas_usage_l3497_349748

theorem michelle_gas_usage (start_gas end_gas : Real) 
  (h1 : start_gas = 0.5)
  (h2 : end_gas = 0.16666666666666666) :
  start_gas - end_gas = 0.33333333333333334 := by
  sorry

end NUMINAMATH_CALUDE_michelle_gas_usage_l3497_349748


namespace NUMINAMATH_CALUDE_square_tablecloth_side_length_l3497_349778

-- Define a square tablecloth
structure SquareTablecloth where
  side : ℝ
  area : ℝ
  is_square : area = side * side

-- Theorem statement
theorem square_tablecloth_side_length 
  (tablecloth : SquareTablecloth) 
  (h : tablecloth.area = 5) : 
  tablecloth.side = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_tablecloth_side_length_l3497_349778


namespace NUMINAMATH_CALUDE_circular_students_l3497_349700

/-- Given a circular arrangement of students, if the 8th student is directly opposite
    the 33rd student, then there are 52 students in total. -/
theorem circular_students (n : ℕ) : n ≥ 33 → (8 + n / 2 = 33) → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_circular_students_l3497_349700


namespace NUMINAMATH_CALUDE_difference_of_squares_l3497_349777

theorem difference_of_squares (a b : ℝ) (h1 : a + b = 5) (h2 : a - b = 3) :
  a^2 - b^2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3497_349777


namespace NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3497_349740

/-- Given a quadratic function f(x) = ax^2 - 4x + c with range [0,+∞),
    prove that the minimum value of 1/c + 9/a is 3 -/
theorem min_value_of_fraction_sum (a c : ℝ) (h₁ : a > 0) (h₂ : c > 0)
    (h₃ : ∀ x, ax^2 - 4*x + c ≥ 0) : 
    ∃ (m : ℝ), m = 3 ∧ ∀ a c, a > 0 → c > 0 → (∀ x, ax^2 - 4*x + c ≥ 0) → 1/c + 9/a ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_sum_l3497_349740


namespace NUMINAMATH_CALUDE_trucks_left_l3497_349743

-- Define the initial number of trucks Sarah had
def initial_trucks : ℕ := 51

-- Define the number of trucks Sarah gave away
def trucks_given_away : ℕ := 13

-- Theorem to prove
theorem trucks_left : initial_trucks - trucks_given_away = 38 := by
  sorry

end NUMINAMATH_CALUDE_trucks_left_l3497_349743


namespace NUMINAMATH_CALUDE_remaining_volume_is_five_sixths_l3497_349797

/-- The volume of a tetrahedron formed by planes passing through the midpoints
    of three edges sharing a vertex in a unit cube --/
def tetrahedron_volume : ℚ := 1 / 24

/-- The number of tetrahedra removed from the cube --/
def num_tetrahedra : ℕ := 8

/-- The volume of the remaining solid after removing tetrahedra from a unit cube --/
def remaining_volume : ℚ := 1 - num_tetrahedra * tetrahedron_volume

theorem remaining_volume_is_five_sixths :
  remaining_volume = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_remaining_volume_is_five_sixths_l3497_349797


namespace NUMINAMATH_CALUDE_waitress_income_fraction_l3497_349739

theorem waitress_income_fraction (salary : ℚ) (salary_pos : salary > 0) :
  let first_week_tips := (11 / 4) * salary
  let second_week_tips := (7 / 3) * salary
  let total_salary := 2 * salary
  let total_tips := first_week_tips + second_week_tips
  let total_income := total_salary + total_tips
  (total_tips / total_income) = 61 / 85 := by
  sorry

end NUMINAMATH_CALUDE_waitress_income_fraction_l3497_349739


namespace NUMINAMATH_CALUDE_log_equation_solution_l3497_349704

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x / Real.log 16) + (Real.log x / Real.log 4) + (Real.log x / Real.log 2) = 7 →
  x = 16 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3497_349704


namespace NUMINAMATH_CALUDE_equation_transformation_l3497_349736

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = x^2 * (y^2 + y - 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_transformation_l3497_349736


namespace NUMINAMATH_CALUDE_rectangle_area_arithmetic_progression_l3497_349708

/-- The area of a rectangle with sides in arithmetic progression -/
theorem rectangle_area_arithmetic_progression (a d : ℚ) :
  let shorter_side := a
  let longer_side := a + d
  shorter_side > 0 → longer_side > shorter_side →
  (shorter_side * longer_side : ℚ) = a^2 + a*d :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_arithmetic_progression_l3497_349708


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l3497_349726

theorem angle_sum_is_pi_over_four (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan α = 1/7 →
  Real.sin β = Real.sqrt 10/10 →
  α + 2*β = π/4 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_four_l3497_349726


namespace NUMINAMATH_CALUDE_nickel_probability_l3497_349782

/-- Represents the types of coins in the jar -/
inductive Coin
| Dime
| Nickel
| Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Dime => 10
| Coin.Nickel => 5
| Coin.Penny => 1

/-- The total value of each coin type in cents -/
def total_value : Coin → ℕ
| Coin.Dime => 500
| Coin.Nickel => 300
| Coin.Penny => 200

/-- The number of coins of each type -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly choosing a nickel from the jar -/
theorem nickel_probability : 
  (coin_count Coin.Nickel : ℚ) / total_coins = 6 / 31 := by sorry

end NUMINAMATH_CALUDE_nickel_probability_l3497_349782


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3497_349741

/-- Proves that given a loan of 1000 for 5 years, where the interest amount is 750 less than the sum lent, the interest rate per annum is 5% -/
theorem interest_rate_calculation (sum_lent : ℝ) (time_period : ℝ) (interest_amount : ℝ) 
  (h1 : sum_lent = 1000)
  (h2 : time_period = 5)
  (h3 : interest_amount = sum_lent - 750) :
  (interest_amount * 100) / (sum_lent * time_period) = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3497_349741


namespace NUMINAMATH_CALUDE_johns_roommates_multiple_l3497_349757

/-- Given that Bob has 10 roommates and John has 25 roommates, 
    prove that the multiple of Bob's roommates that John has five more than is 2. -/
theorem johns_roommates_multiple (bob_roommates john_roommates : ℕ) : 
  bob_roommates = 10 → john_roommates = 25 → 
  ∃ (x : ℕ), john_roommates = x * bob_roommates + 5 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_johns_roommates_multiple_l3497_349757


namespace NUMINAMATH_CALUDE_firm_employee_count_l3497_349717

-- Define the initial number of Democrats and Republicans
def initial_democrats : ℕ := sorry
def initial_republicans : ℕ := sorry

-- Define the conditions
axiom condition1 : initial_democrats + 1 = initial_republicans - 1
axiom condition2 : initial_democrats + 4 = 2 * (initial_republicans - 4)

-- Define the total number of employees
def total_employees : ℕ := initial_democrats + initial_republicans

-- Theorem to prove
theorem firm_employee_count : total_employees = 18 := by
  sorry

end NUMINAMATH_CALUDE_firm_employee_count_l3497_349717


namespace NUMINAMATH_CALUDE_snake_count_theorem_l3497_349746

/-- Represents the number of pet owners for different combinations of pets --/
structure PetOwners where
  total : Nat
  onlyDogs : Nat
  onlyCats : Nat
  catsAndDogs : Nat
  catsDogsSnakes : Nat

/-- Given the pet ownership data, proves that the minimum number of snakes is 3
    and that the total number of snakes cannot be determined --/
theorem snake_count_theorem (po : PetOwners)
  (h1 : po.total = 79)
  (h2 : po.onlyDogs = 15)
  (h3 : po.onlyCats = 10)
  (h4 : po.catsAndDogs = 5)
  (h5 : po.catsDogsSnakes = 3) :
  ∃ (minSnakes : Nat), minSnakes = 3 ∧ 
  ¬∃ (totalSnakes : Nat), ∀ (n : Nat), n ≥ minSnakes → n = totalSnakes :=
by sorry

end NUMINAMATH_CALUDE_snake_count_theorem_l3497_349746


namespace NUMINAMATH_CALUDE_solution_count_l3497_349737

/-- The number of distinct divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The number of positive integer solutions (x, y) to the equation 1/n = 1/x + 1/y where x ≠ y -/
def num_solutions (n : ℕ+) : ℕ := sorry

theorem solution_count (n : ℕ+) : num_solutions n = num_divisors (n^2) - 1 := by sorry

end NUMINAMATH_CALUDE_solution_count_l3497_349737


namespace NUMINAMATH_CALUDE_magician_reappeared_count_l3497_349703

/-- Represents the magician's performance statistics --/
structure MagicianStats where
  total_shows : ℕ
  min_audience : ℕ
  max_audience : ℕ
  disappear_ratio : ℕ
  no_reappear_prob : ℚ
  double_reappear_prob : ℚ
  triple_reappear_prob : ℚ

/-- Calculates the total number of people who reappeared in the magician's performances --/
def total_reappeared (stats : MagicianStats) : ℕ :=
  sorry

/-- Theorem stating that given the magician's performance statistics, 
    the total number of people who reappeared is 640 --/
theorem magician_reappeared_count (stats : MagicianStats) 
  (h1 : stats.total_shows = 100)
  (h2 : stats.min_audience = 50)
  (h3 : stats.max_audience = 500)
  (h4 : stats.disappear_ratio = 50)
  (h5 : stats.no_reappear_prob = 1/10)
  (h6 : stats.double_reappear_prob = 1/5)
  (h7 : stats.triple_reappear_prob = 1/20) :
  total_reappeared stats = 640 :=
sorry

end NUMINAMATH_CALUDE_magician_reappeared_count_l3497_349703


namespace NUMINAMATH_CALUDE_division_and_addition_of_fractions_l3497_349713

theorem division_and_addition_of_fractions : 
  (2 : ℚ) / 3 / ((4 : ℚ) / 5) + (1 : ℚ) / 2 = (4 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_and_addition_of_fractions_l3497_349713


namespace NUMINAMATH_CALUDE_souvenir_store_problem_l3497_349783

/-- Souvenir store problem -/
theorem souvenir_store_problem 
  (total_souvenirs : ℕ)
  (cost_40A_30B cost_10A_50B : ℕ)
  (sell_price_A sell_price_B : ℕ)
  (m : ℚ)
  (h_total : total_souvenirs = 300)
  (h_cost1 : cost_40A_30B = 5000)
  (h_cost2 : cost_10A_50B = 3800)
  (h_sell_A : sell_price_A = 120)
  (h_sell_B : sell_price_B = 80)
  (h_m_range : 4 < m ∧ m < 8) :
  ∃ (cost_A cost_B max_profit : ℕ) (reduced_profit : ℚ),
    cost_A = 80 ∧ 
    cost_B = 60 ∧
    max_profit = 7500 ∧
    reduced_profit = 5720 ∧
    ∀ (a : ℕ), 
      a ≤ total_souvenirs →
      (total_souvenirs - a) ≥ 3 * a →
      (sell_price_A - cost_A) * a + (sell_price_B - cost_B) * (total_souvenirs - a) ≥ 7400 →
      (sell_price_A - cost_A) * a + (sell_price_B - cost_B) * (total_souvenirs - a) ≤ max_profit ∧
      ((sell_price_A - 5 * m - cost_A) * 70 + (sell_price_B - cost_B) * (total_souvenirs - 70) : ℚ) = reduced_profit →
      m = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_store_problem_l3497_349783


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3497_349761

theorem inequality_solution_set : 
  {x : ℝ | (x - 2) * (2 * x + 1) > 0} = 
  {x : ℝ | x < -1/2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3497_349761


namespace NUMINAMATH_CALUDE_inequality_proof_l3497_349786

theorem inequality_proof (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) : b/a + a/b > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3497_349786


namespace NUMINAMATH_CALUDE_train_length_proof_l3497_349774

/-- Proves that a train with given speed and crossing time has a specific length -/
theorem train_length_proof (speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed = 90 → -- speed in km/hr
  crossing_time = 1 / 60 → -- crossing time in hours (1 minute = 1/60 hour)
  train_length = speed * crossing_time / 2 → -- length calculation
  train_length = 750 / 1000 -- length in km (750 m = 0.75 km)
  := by sorry

end NUMINAMATH_CALUDE_train_length_proof_l3497_349774


namespace NUMINAMATH_CALUDE_pencil_sale_problem_l3497_349796

theorem pencil_sale_problem (total_students : ℕ) (total_pencils : ℕ) 
  (h_total_students : total_students = 10)
  (h_total_pencils : total_pencils = 24)
  (h_first_two : 2 * 2 = 4)  -- First two students bought 2 pencils each
  (h_last_two : 2 * 1 = 2)   -- Last two students bought 1 pencil each
  : ∃ (middle_group : ℕ), 
    middle_group = 6 ∧ 
    middle_group * 3 + 4 + 2 = total_pencils ∧ 
    2 + middle_group + 2 = total_students :=
by sorry

end NUMINAMATH_CALUDE_pencil_sale_problem_l3497_349796


namespace NUMINAMATH_CALUDE_fourth_root_of_polynomial_l3497_349744

theorem fourth_root_of_polynomial (a b : ℝ) : 
  (∀ x : ℝ, b * x^3 + (3*b + a) * x^2 + (a - 2*b) * x + (5 - b) = 0 ↔ 
    x = -1 ∨ x = 2 ∨ x = 4 ∨ x = -8) → 
  ∃ x : ℝ, x = -8 ∧ b * x^3 + (3*b + a) * x^2 + (a - 2*b) * x + (5 - b) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_of_polynomial_l3497_349744


namespace NUMINAMATH_CALUDE_fraction_simplification_l3497_349765

theorem fraction_simplification : (210 : ℚ) / 21 * 7 / 98 * 6 / 4 = 15 / 14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3497_349765


namespace NUMINAMATH_CALUDE_no_integer_solution_l3497_349729

theorem no_integer_solution : ¬∃ (x y : ℤ), x * y + 4 = 40 ∧ x + y = 14 := by sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3497_349729


namespace NUMINAMATH_CALUDE_gcd_105_90_l3497_349753

theorem gcd_105_90 : Nat.gcd 105 90 = 15 := by sorry

end NUMINAMATH_CALUDE_gcd_105_90_l3497_349753


namespace NUMINAMATH_CALUDE_jacks_books_l3497_349779

/-- Calculates the number of books in a stack given the stack thickness,
    pages per inch, and pages per book. -/
def number_of_books (stack_thickness : ℕ) (pages_per_inch : ℕ) (pages_per_book : ℕ) : ℕ :=
  (stack_thickness * pages_per_inch) / pages_per_book

/-- Theorem stating that Jack's stack of 12 inches with 80 pages per inch
    and 160 pages per book contains 6 books. -/
theorem jacks_books :
  number_of_books 12 80 160 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacks_books_l3497_349779
