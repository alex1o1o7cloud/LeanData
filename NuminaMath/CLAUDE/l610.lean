import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_d_value_l610_61033

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of d for which the lines y = 3x + 5 and y = (4d)x + 3 are parallel -/
theorem parallel_lines_d_value :
  (∀ x y : ℝ, y = 3 * x + 5 ↔ y = (4 * d) * x + 3) → d = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_d_value_l610_61033


namespace NUMINAMATH_CALUDE_prime_sum_seven_power_l610_61086

theorem prime_sum_seven_power (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 7 → (p^q = 32 ∨ p^q = 25) := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_seven_power_l610_61086


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l610_61083

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (original_price_positive : original_price > 0)
  (original_quantity_positive : original_quantity > 0) :
  let new_price := 1.30 * original_price
  let new_revenue := 1.04 * (original_price * original_quantity)
  let sales_decrease_percentage := 
    100 * (1 - (new_revenue / new_price) / original_quantity)
  sales_decrease_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l610_61083


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l610_61014

theorem boat_speed_ratio (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : current_speed = 4)
  (h3 : distance = 2) : 
  (2 * distance) / ((distance / (boat_speed + current_speed)) + (distance / (boat_speed - current_speed))) / boat_speed = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l610_61014


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l610_61028

theorem regular_polygon_sides (n : ℕ) (h : n > 2) :
  (∀ θ : ℝ, θ = 160 * π / 180 → (n - 2) * π = n * θ) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l610_61028


namespace NUMINAMATH_CALUDE_fly_path_distance_l610_61019

theorem fly_path_distance (r : ℝ) (s : ℝ) (h1 : r = 58) (h2 : s = 80) : 
  let d := 2 * r
  let x := Real.sqrt (d^2 - s^2)
  d + x + s = 280 :=
by sorry

end NUMINAMATH_CALUDE_fly_path_distance_l610_61019


namespace NUMINAMATH_CALUDE_third_number_proof_l610_61030

theorem third_number_proof (A B C : ℕ+) : 
  A = 600 → 
  B = 840 → 
  Nat.gcd A (Nat.gcd B C) = 60 →
  Nat.lcm A (Nat.lcm B C) = 50400 →
  C = 6 := by
sorry

end NUMINAMATH_CALUDE_third_number_proof_l610_61030


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l610_61055

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x | -1 < x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l610_61055


namespace NUMINAMATH_CALUDE_people_needed_for_two_hours_l610_61061

/-- Represents the rate at which water enters the boat (units per hour) -/
def water_entry_rate : ℝ := 2

/-- Represents the amount of water one person can bail out per hour -/
def bailing_rate : ℝ := 1

/-- Represents the total amount of water to be bailed out -/
def total_water : ℝ := 30

/-- Given the conditions from the problem, proves that 14 people are needed to bail out the water in 2 hours -/
theorem people_needed_for_two_hours : 
  (∀ (p : ℕ), p = 10 → p * bailing_rate * 3 = total_water + water_entry_rate * 3) →
  (∀ (p : ℕ), p = 5 → p * bailing_rate * 8 = total_water + water_entry_rate * 8) →
  ∃ (p : ℕ), p = 14 ∧ p * bailing_rate * 2 = total_water + water_entry_rate * 2 :=
by sorry

end NUMINAMATH_CALUDE_people_needed_for_two_hours_l610_61061


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l610_61000

/-- An arithmetic sequence -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l610_61000


namespace NUMINAMATH_CALUDE_equation_solution_l610_61095

theorem equation_solution : 
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 3) = 1 / (x + 6) + 1 / (x + 2)) ∧ (x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l610_61095


namespace NUMINAMATH_CALUDE_min_tangent_length_l610_61004

/-- Circle C with equation x^2 + y^2 - 2x - 4y + 1 = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 - 4*p.2 + 1 = 0}

/-- Line l -/
def line_l : Set (ℝ × ℝ) := sorry

/-- Maximum distance from any point on C to line l is 6 -/
axiom max_distance_to_l (p : ℝ × ℝ) :
  p ∈ circle_C → ∃ (q : ℝ × ℝ), q ∈ line_l ∧ dist p q ≤ 6

/-- Tangent line from a point on l to C -/
def tangent_length (a : ℝ × ℝ) : ℝ := sorry

theorem min_tangent_length :
  ∃ (a : ℝ × ℝ), a ∈ line_l ∧
  (∀ (b : ℝ × ℝ), b ∈ line_l → tangent_length a ≤ tangent_length b) ∧
  tangent_length a = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_tangent_length_l610_61004


namespace NUMINAMATH_CALUDE_marys_marbles_count_l610_61035

def dans_marbles : ℕ := 5
def marys_marbles_multiplier : ℕ := 2

theorem marys_marbles_count : dans_marbles * marys_marbles_multiplier = 10 := by
  sorry

end NUMINAMATH_CALUDE_marys_marbles_count_l610_61035


namespace NUMINAMATH_CALUDE_unique_a_for_perpendicular_chords_l610_61036

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define a point on the x-axis
def point_on_x_axis (a : ℝ) : ℝ × ℝ := (a, 0)

-- Define the property of perpendicular lines intersecting the hyperbola
def perpendicular_lines_property (a : ℝ) : Prop :=
  ∀ (l₁ l₂ : ℝ → ℝ → Prop),
    (∀ x y, l₁ x y ↔ l₂ y (-x)) →  -- l₁ and l₂ are perpendicular
    (l₁ a 0 ∧ l₂ a 0) →  -- both lines pass through (a, 0)
    ∀ (px py qx qy rx ry sx sy : ℝ),
      (hyperbola px py ∧ hyperbola qx qy ∧ l₁ px py ∧ l₁ qx qy) →  -- P and Q on l₁ and hyperbola
      (hyperbola rx ry ∧ hyperbola sx sy ∧ l₂ rx ry ∧ l₂ sx sy) →  -- R and S on l₂ and hyperbola
      (px - qx)^2 + (py - qy)^2 = (rx - sx)^2 + (ry - sy)^2  -- |PQ| = |RS|

-- The main theorem
theorem unique_a_for_perpendicular_chords :
  ∃! (a : ℝ), a > 1 ∧ perpendicular_lines_property a ∧ a = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_for_perpendicular_chords_l610_61036


namespace NUMINAMATH_CALUDE_flu_infection_rate_l610_61039

theorem flu_infection_rate : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (1 + x + x * (1 + x) = 196) ∧ 
  (x = 13) :=
sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l610_61039


namespace NUMINAMATH_CALUDE_modulus_of_z_l610_61003

def z : ℂ := (2 + Complex.I) * (1 - Complex.I)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l610_61003


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l610_61089

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47, the 8th term is 71. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h4 : a 4 = 23) (h6 : a 6 = 47) : a 8 = 71 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l610_61089


namespace NUMINAMATH_CALUDE_solution_set_inequality_l610_61042

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo 50 60 : Set ℝ) = {x | (x - 50) * (60 - x) > 0} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l610_61042


namespace NUMINAMATH_CALUDE_product_of_roots_plus_one_l610_61012

theorem product_of_roots_plus_one (a b c : ℂ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) → 
  (b^3 - 15*b^2 + 22*b - 8 = 0) → 
  (c^3 - 15*c^2 + 22*c - 8 = 0) → 
  (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_plus_one_l610_61012


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l610_61067

/-- Represents a card with a digit -/
structure Card where
  digit : Fin 10

/-- Represents an arrangement of cards -/
def Arrangement := List Card

/-- Checks if an arrangement satisfies the problem conditions -/
def satisfiesConditions (arr : Arrangement) : Prop :=
  ∀ i : Fin 10, ∃ pos1 pos2 : Nat,
    pos1 < pos2 ∧
    pos2 < arr.length ∧
    (arr.get ⟨pos1, by sorry⟩).digit = i ∧
    (arr.get ⟨pos2, by sorry⟩).digit = i ∧
    pos2 - pos1 - 1 = i.val

theorem no_valid_arrangement :
  ¬∃ (arr : Arrangement),
    arr.length = 20 ∧
    (∀ i : Fin 10, (arr.filter (λ c => c.digit = i)).length = 2) ∧
    satisfiesConditions arr := by
  sorry


end NUMINAMATH_CALUDE_no_valid_arrangement_l610_61067


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l610_61029

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def primeFactorExponents (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  (primeFactorExponents (largestPerfectSquareDivisor (factorial 15))).sum = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l610_61029


namespace NUMINAMATH_CALUDE_lowest_score_problem_l610_61073

theorem lowest_score_problem (scores : List ℝ) (highest_score lowest_score : ℝ) : 
  scores.length = 15 →
  scores.sum / scores.length = 75 →
  highest_score ∈ scores →
  lowest_score ∈ scores →
  highest_score = 95 →
  (scores.sum - highest_score - lowest_score) / (scores.length - 2) = 78 →
  lowest_score = 16 :=
by sorry

end NUMINAMATH_CALUDE_lowest_score_problem_l610_61073


namespace NUMINAMATH_CALUDE_floor_sqrt_99_l610_61088

theorem floor_sqrt_99 : ⌊Real.sqrt 99⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_99_l610_61088


namespace NUMINAMATH_CALUDE_pardee_road_length_is_12000_l610_61080

/-- The length of Pardee Road in meters, given the conditions of the problem -/
def pardee_road_length : ℕ :=
  let telegraph_road_km : ℕ := 162
  let difference_km : ℕ := 150
  let meters_per_km : ℕ := 1000
  (telegraph_road_km - difference_km) * meters_per_km

theorem pardee_road_length_is_12000 : pardee_road_length = 12000 := by
  sorry

end NUMINAMATH_CALUDE_pardee_road_length_is_12000_l610_61080


namespace NUMINAMATH_CALUDE_probability_black_white_balls_l610_61052

theorem probability_black_white_balls (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_balls : ℕ) :
  total_balls = black_balls + white_balls + green_balls →
  black_balls = 3 →
  white_balls = 3 →
  green_balls = 1 →
  (black_balls * white_balls : ℚ) / ((total_balls * (total_balls - 1)) / 2) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_black_white_balls_l610_61052


namespace NUMINAMATH_CALUDE_total_jars_is_72_l610_61071

/-- Represents the number of jars of each size -/
def num_jars : ℕ := 24

/-- Represents the total volume of water in gallons -/
def total_volume : ℚ := 42

/-- Represents the volume of a quart jar in gallons -/
def quart_volume : ℚ := 1/4

/-- Represents the volume of a half-gallon jar in gallons -/
def half_gallon_volume : ℚ := 1/2

/-- Represents the volume of a one-gallon jar in gallons -/
def gallon_volume : ℚ := 1

/-- The theorem stating that given the conditions, the total number of jars is 72 -/
theorem total_jars_is_72 :
  (num_jars : ℚ) * (quart_volume + half_gallon_volume + gallon_volume) = total_volume ∧
  num_jars * 3 = 72 := by
  sorry

#check total_jars_is_72

end NUMINAMATH_CALUDE_total_jars_is_72_l610_61071


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l610_61027

/-- Given a triangle with side lengths 8, 10, and 12, the surface area of the circumscribed sphere
    of the triangular prism formed by connecting the midpoints of the triangle's sides
    is equal to 77π/2. -/
theorem circumscribed_sphere_surface_area (A₁ A₂ A₃ B C D : ℝ × ℝ × ℝ) : 
  let side_lengths := [8, 10, 12]
  ∀ (a b c : ℝ), 
    a ∈ side_lengths ∧ 
    b ∈ side_lengths ∧ 
    c ∈ side_lengths ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    (‖A₁ - A₂‖ = a ∧ ‖A₂ - A₃‖ = b ∧ ‖A₃ - A₁‖ = c) →
    (B = (A₁ + A₂) / 2 ∧ C = (A₂ + A₃) / 2 ∧ D = (A₃ + A₁) / 2) →
    let R := Real.sqrt (77 / 8)
    4 * π * R^2 = 77 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l610_61027


namespace NUMINAMATH_CALUDE_train_crossing_time_l610_61062

/-- Proves that a train crossing a signal pole takes 18 seconds given specific conditions -/
theorem train_crossing_time (train_length : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_length = 300 →
  platform_length = 600.0000000000001 →
  platform_crossing_time = 54 →
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l610_61062


namespace NUMINAMATH_CALUDE_blocks_left_l610_61058

/-- Given that Randy had 59 blocks initially and used 36 blocks to build a tower,
    prove that he has 23 blocks left. -/
theorem blocks_left (initial_blocks : ℕ) (used_blocks : ℕ) 
  (h1 : initial_blocks = 59)
  (h2 : used_blocks = 36) :
  initial_blocks - used_blocks = 23 :=
by sorry

end NUMINAMATH_CALUDE_blocks_left_l610_61058


namespace NUMINAMATH_CALUDE_spinster_cat_ratio_l610_61002

theorem spinster_cat_ratio : 
  ∀ (spinsters cats : ℕ),
  spinsters = 18 →
  cats = spinsters + 63 →
  ∃ (n : ℕ), spinsters * n = 2 * cats →
  (spinsters : ℚ) / (cats : ℚ) = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_spinster_cat_ratio_l610_61002


namespace NUMINAMATH_CALUDE_union_and_complement_when_a_is_one_intersection_equals_b_iff_a_in_range_l610_61090

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part 1
theorem union_and_complement_when_a_is_one :
  (A ∪ B 1 = {x | -1 ≤ x ∧ x ≤ 3}) ∧
  (Set.univ \ B 1 = {x | x < 0 ∨ x > 3}) := by sorry

-- Theorem for part 2
theorem intersection_equals_b_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ∈ Set.Ioi (-2) ∪ Set.Icc 0 1 := by sorry

end NUMINAMATH_CALUDE_union_and_complement_when_a_is_one_intersection_equals_b_iff_a_in_range_l610_61090


namespace NUMINAMATH_CALUDE_prob_through_C_eq_25_63_l610_61013

/-- Represents a point in the city grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of choosing either direction at an intersection -/
def choice_prob : ℚ := 1/2

/-- The starting point A -/
def A : Point := ⟨0, 0⟩

/-- The intermediate point C -/
def C : Point := ⟨3, 2⟩

/-- The ending point B -/
def B : Point := ⟨5, 5⟩

/-- Calculates the number of paths between two points -/
def num_paths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of walking from A to B through C -/
def prob_through_C : ℚ :=
  (num_paths A C * num_paths C B : ℚ) / num_paths A B

theorem prob_through_C_eq_25_63 : prob_through_C = 25/63 := by
  sorry

end NUMINAMATH_CALUDE_prob_through_C_eq_25_63_l610_61013


namespace NUMINAMATH_CALUDE_snack_package_average_l610_61059

theorem snack_package_average (cookie_counts : List ℕ)
  (candy_counts : List ℕ) (pie_counts : List ℕ) :
  cookie_counts.length = 4 →
  candy_counts.length = 3 →
  pie_counts.length = 2 →
  cookie_counts.sum + candy_counts.sum + pie_counts.sum = 153 →
  cookie_counts.sum / cookie_counts.length = 17 →
  (cookie_counts.sum + candy_counts.sum + pie_counts.sum) /
    (cookie_counts.length + candy_counts.length + pie_counts.length) = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_snack_package_average_l610_61059


namespace NUMINAMATH_CALUDE_sculpture_base_height_l610_61006

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  12 * feet + inches

/-- Calculates the total height of a sculpture and its base -/
theorem sculpture_base_height 
  (sculpture_feet : ℕ) 
  (sculpture_inches : ℕ) 
  (base_inches : ℕ) : 
  feet_inches_to_inches sculpture_feet sculpture_inches + base_inches = 38 :=
by
  sorry

#check sculpture_base_height 2 10 4

end NUMINAMATH_CALUDE_sculpture_base_height_l610_61006


namespace NUMINAMATH_CALUDE_intersection_A_B_l610_61091

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l610_61091


namespace NUMINAMATH_CALUDE_quadratic_sum_of_d_and_e_l610_61070

/-- Given a quadratic polynomial x^2 - 16x + 15, when written in the form (x+d)^2 + e,
    the sum of d and e is -57. -/
theorem quadratic_sum_of_d_and_e : ∃ d e : ℝ, 
  (∀ x, x^2 - 16*x + 15 = (x+d)^2 + e) ∧ d + e = -57 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_d_and_e_l610_61070


namespace NUMINAMATH_CALUDE_parabola_properties_l610_61011

-- Define the parabolas
def parabola_G (x y : ℝ) : Prop := x^2 = y
def parabola_M (x y : ℝ) : Prop := y^2 = 4*x

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the focus of parabola M
def focus_M : ℝ × ℝ := (1, 0)

-- Define the property of being inscribed in a parabola
def inscribed_in_parabola (t : Triangle) (p : ℝ → ℝ → Prop) : Prop :=
  p t.A.1 t.A.2 ∧ p t.B.1 t.B.2 ∧ p t.C.1 t.C.2

-- Define the property of a line being tangent to a parabola
def line_tangent_to_parabola (p q : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) : Prop :=
  ∃ (t : ℝ), parabola ((1-t)*p.1 + t*q.1) ((1-t)*p.2 + t*q.2)

-- Define the property of points being concyclic
def concyclic (p q r s : ℝ × ℝ) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧
    (q.1 - center.1)^2 + (q.2 - center.2)^2 = radius^2 ∧
    (r.1 - center.1)^2 + (r.2 - center.2)^2 = radius^2 ∧
    (s.1 - center.1)^2 + (s.2 - center.2)^2 = radius^2

theorem parabola_properties (t : Triangle) 
  (h1 : inscribed_in_parabola t parabola_G)
  (h2 : line_tangent_to_parabola t.A t.B parabola_M)
  (h3 : line_tangent_to_parabola t.A t.C parabola_M) :
  line_tangent_to_parabola t.B t.C parabola_M ∧
  concyclic t.A t.C t.B focus_M := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l610_61011


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l610_61066

theorem fraction_to_decimal : (5 : ℚ) / 125 = (4 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l610_61066


namespace NUMINAMATH_CALUDE_nabla_calculation_l610_61092

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_calculation : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l610_61092


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l610_61065

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 1 + a 5 = 10) 
  (h2 : a 4 = 7) 
  (h3 : arithmetic_sequence a d) : 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l610_61065


namespace NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l610_61057

theorem function_range_contained_in_unit_interval 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, x > y → (f x)^2 ≤ f y) : 
  ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_contained_in_unit_interval_l610_61057


namespace NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l610_61064

/-- Two integers are consecutive odd numbers if their difference is 2 and they are both odd -/
def ConsecutiveOddNumbers (m n : ℤ) : Prop :=
  m - n = 2 ∧ Odd m ∧ Odd n

/-- The largest divisor of m^2 - n^2 for consecutive odd numbers m and n where n < m is 8 -/
theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (h1 : ConsecutiveOddNumbers m n) (h2 : n < m) : 
  (∃ (k : ℤ), m^2 - n^2 = 8 * k) ∧ 
  (∀ (d : ℤ), d > 8 → ¬(∀ (j : ℤ), m^2 - n^2 = d * j)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_consecutive_odd_squares_l610_61064


namespace NUMINAMATH_CALUDE_dandelion_game_strategy_l610_61097

/-- The dandelion mowing game -/
def has_winning_strategy (m n : ℕ+) : Prop :=
  (m.val + n.val) % 2 = 1 ∨ min m.val n.val = 1

theorem dandelion_game_strategy (m n : ℕ+) :
  has_winning_strategy m n ↔ (m.val + n.val) % 2 = 1 ∨ min m.val n.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_game_strategy_l610_61097


namespace NUMINAMATH_CALUDE_parallel_lines_circle_chord_l610_61018

theorem parallel_lines_circle_chord (r : ℝ) (d : ℝ) : 
  r > 0 ∧ d > 0 ∧ 
  36 * r^2 = 648 + 9 * d^2 ∧ 
  40 * r^2 = 800 + 90 * d^2 → 
  d = 67/10 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_chord_l610_61018


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l610_61034

/-- Represents the color of a frog -/
inductive FrogColor
  | Green
  | Red
  | Blue

/-- Represents a row of frogs -/
def FrogRow := List FrogColor

/-- Checks if a frog arrangement is valid -/
def is_valid_arrangement (row : FrogRow) : Bool :=
  sorry

/-- Counts the number of frogs of each color in a row -/
def count_frogs (row : FrogRow) : Nat × Nat × Nat :=
  sorry

/-- Generates all possible arrangements of frogs -/
def all_arrangements : List FrogRow :=
  sorry

/-- Counts the number of valid arrangements -/
def count_valid_arrangements : Nat :=
  sorry

theorem frog_arrangement_count :
  count_valid_arrangements = 96 :=
sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l610_61034


namespace NUMINAMATH_CALUDE_jerry_age_l610_61016

/-- Given that Mickey's age is 5 years less than 200% of Jerry's age and Mickey is 19 years old,
    prove that Jerry is 12 years old. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 2 * jerry_age - 5)
  (h2 : mickey_age = 19) : 
  jerry_age = 12 := by
  sorry

end NUMINAMATH_CALUDE_jerry_age_l610_61016


namespace NUMINAMATH_CALUDE_video_game_expenditure_l610_61010

theorem video_game_expenditure (total : ℝ) (books snacks stationery shoes : ℝ) :
  total = 50 →
  books = (1 / 4) * total →
  snacks = (1 / 5) * total →
  stationery = (1 / 10) * total →
  shoes = (3 / 10) * total →
  total - (books + snacks + stationery + shoes) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l610_61010


namespace NUMINAMATH_CALUDE_w_squared_value_l610_61085

theorem w_squared_value (w : ℝ) (h : (w + 15)^2 = (4*w + 9)*(w + 6)) :
  w^2 = 57.5 - 0.5 * Real.sqrt 229 := by
  sorry

end NUMINAMATH_CALUDE_w_squared_value_l610_61085


namespace NUMINAMATH_CALUDE_hybrid_car_trip_length_l610_61005

theorem hybrid_car_trip_length 
  (battery_distance : ℝ) 
  (gasoline_consumption_rate : ℝ) 
  (average_efficiency : ℝ) :
  battery_distance = 75 →
  gasoline_consumption_rate = 0.05 →
  average_efficiency = 50 →
  ∃ (total_distance : ℝ),
    total_distance = 125 ∧
    average_efficiency = total_distance / (gasoline_consumption_rate * (total_distance - battery_distance)) :=
by
  sorry

end NUMINAMATH_CALUDE_hybrid_car_trip_length_l610_61005


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l610_61044

def Alex : ℚ := 1/6
def Beth : ℚ := 2/5
def Cyril : ℚ := 1/3
def Dan : ℚ := 3/10
def Ella : ℚ := 1 - (Alex + Beth + Cyril + Dan)

def siblings : List ℚ := [Beth, Cyril, Dan, Alex, Ella]

theorem pizza_consumption_order : 
  List.Sorted (fun a b => a ≥ b) siblings ∧ 
  siblings = [Beth, Cyril, Dan, Alex, Ella] :=
sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l610_61044


namespace NUMINAMATH_CALUDE_fruit_box_composition_l610_61038

/-- Represents the contents of the fruit box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ

/-- The total number of fruits in the box -/
def FruitBox.total (box : FruitBox) : ℕ := box.apples + box.pears

/-- Predicate to check if selecting n fruits always includes at least one apple -/
def always_includes_apple (box : FruitBox) (n : ℕ) : Prop :=
  box.pears < n

/-- Predicate to check if selecting n fruits always includes at least one pear -/
def always_includes_pear (box : FruitBox) (n : ℕ) : Prop :=
  box.apples < n

/-- The main theorem stating the unique composition of the fruit box -/
theorem fruit_box_composition :
  ∃! (box : FruitBox),
    box.total ≥ 5 ∧
    always_includes_apple box 3 ∧
    always_includes_pear box 4 :=
  sorry

end NUMINAMATH_CALUDE_fruit_box_composition_l610_61038


namespace NUMINAMATH_CALUDE_new_gross_profit_percentage_l610_61051

theorem new_gross_profit_percentage
  (old_selling_price : ℝ)
  (old_gross_profit_percentage : ℝ)
  (new_selling_price : ℝ)
  (h1 : old_selling_price = 88)
  (h2 : old_gross_profit_percentage = 10)
  (h3 : new_selling_price = 92) :
  let cost := old_selling_price / (1 + old_gross_profit_percentage / 100)
  let new_gross_profit := new_selling_price - cost
  new_gross_profit / cost * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_new_gross_profit_percentage_l610_61051


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l610_61047

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = n * (a 1 + a n) / 2

/-- Theorem: For an arithmetic sequence with S₂ = 9 and S₄ = 22, S₈ = 60 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence)
    (h₂ : seq.S 2 = 9)
    (h₄ : seq.S 4 = 22) :
    seq.S 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l610_61047


namespace NUMINAMATH_CALUDE_track_width_l610_61087

-- Define the radii of the two circles
variable (r₁ r₂ : ℝ)

-- Define the condition that the circles are concentric and r₁ > r₂
variable (h₁ : r₁ > r₂)

-- Define the condition that the difference in circumferences is 16π
variable (h₂ : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 16 * Real.pi)

-- Theorem statement
theorem track_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) (h₂ : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 16 * Real.pi) :
  r₁ - r₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_track_width_l610_61087


namespace NUMINAMATH_CALUDE_ball_hits_ground_time_l610_61076

theorem ball_hits_ground_time :
  ∃ t : ℝ, t > 0 ∧ -8 * t^2 - 12 * t + 72 = 0 ∧ abs (t - 2.34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ball_hits_ground_time_l610_61076


namespace NUMINAMATH_CALUDE_sqrt_three_irrational_l610_61074

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_irrational_l610_61074


namespace NUMINAMATH_CALUDE_wire_length_proof_l610_61021

theorem wire_length_proof (shorter_piece longer_piece total_length : ℝ) : 
  shorter_piece = 20 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 70 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l610_61021


namespace NUMINAMATH_CALUDE_horse_cloth_problem_l610_61099

/-- Represents the system of equations for the horse and cloth problem -/
def horse_cloth_system (m n : ℚ) : Prop :=
  m + n = 100 ∧ 3 * m + n / 3 = 100

/-- The horse and cloth problem statement -/
theorem horse_cloth_problem :
  ∃ m n : ℚ, 
    m ≥ 0 ∧ n ≥ 0 ∧  -- Ensuring non-negative numbers of horses
    horse_cloth_system m n :=
by sorry

end NUMINAMATH_CALUDE_horse_cloth_problem_l610_61099


namespace NUMINAMATH_CALUDE_probability_two_of_each_color_l610_61008

theorem probability_two_of_each_color (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) 
  (drawn_balls : ℕ) (h1 : total_balls = black_balls + white_balls) (h2 : total_balls = 17) 
  (h3 : black_balls = 9) (h4 : white_balls = 8) (h5 : drawn_balls = 4) : 
  (Nat.choose black_balls 2 * Nat.choose white_balls 2) / Nat.choose total_balls drawn_balls = 168 / 397 :=
sorry

end NUMINAMATH_CALUDE_probability_two_of_each_color_l610_61008


namespace NUMINAMATH_CALUDE_f_2020_is_sin_l610_61046

noncomputable def f (n : ℕ) : ℝ → ℝ := 
  match n with
  | 0 => Real.sin
  | n + 1 => deriv (f n)

theorem f_2020_is_sin : f 2020 = Real.sin := by
  sorry

end NUMINAMATH_CALUDE_f_2020_is_sin_l610_61046


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l610_61001

theorem cube_less_than_triple : ∃! (x : ℤ), x^3 < 3*x :=
sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l610_61001


namespace NUMINAMATH_CALUDE_log_sum_difference_l610_61075

theorem log_sum_difference (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 - Real.log 4 / Real.log 10 = 2 + Real.log 2.5 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_difference_l610_61075


namespace NUMINAMATH_CALUDE_conference_handshakes_l610_61054

/-- The number of people at the conference -/
def n : ℕ := 27

/-- The number of people who don't shake hands with each other -/
def k : ℕ := 3

/-- The maximum number of handshakes possible under the given conditions -/
def max_handshakes : ℕ := n.choose 2 - k.choose 2

/-- Theorem stating the maximum number of handshakes at the conference -/
theorem conference_handshakes :
  max_handshakes = 348 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l610_61054


namespace NUMINAMATH_CALUDE_polynomial_division_l610_61094

theorem polynomial_division (x : ℂ) : 
  ∃! (a : ℤ), ∃ (p : ℂ → ℂ), (x^2 - x + a) * p x = x^15 + x^2 + 100 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l610_61094


namespace NUMINAMATH_CALUDE_c_death_year_l610_61084

structure Mathematician where
  name : String
  birth_year : ℕ
  death_year : ℕ

def arithmetic_sequence (a b c : ℕ) : Prop :=
  b - a = c - b

theorem c_death_year (a b c : Mathematician) (d : String) :
  a.name = "A" →
  b.name = "B" →
  c.name = "C" →
  d = "D" →
  a.death_year = 1980 →
  a.death_year - a.birth_year = 50 →
  b.death_year - b.birth_year < 50 →
  c.death_year - c.birth_year = 60 →
  a.death_year - b.death_year < 10 →
  a.death_year - b.death_year > 0 →
  b.death_year - b.birth_year = c.death_year - b.death_year →
  arithmetic_sequence a.birth_year b.birth_year c.birth_year →
  c.death_year = 1986 := by
  sorry

#check c_death_year

end NUMINAMATH_CALUDE_c_death_year_l610_61084


namespace NUMINAMATH_CALUDE_prob_A_union_B_l610_61093

-- Define the sample space for a fair six-sided die
def Ω : Finset Nat := Finset.range 6

-- Define the probability measure
def P (S : Finset Nat) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Define event A: getting a 3
def A : Finset Nat := {2}

-- Define event B: getting an even number
def B : Finset Nat := {1, 3, 5}

-- Theorem statement
theorem prob_A_union_B : P (A ∪ B) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_union_B_l610_61093


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l610_61017

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l610_61017


namespace NUMINAMATH_CALUDE_calculation_proof_l610_61048

theorem calculation_proof : (-1)^2024 - 1/2 * (8 - (-2)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l610_61048


namespace NUMINAMATH_CALUDE_one_plus_i_fourth_power_l610_61053

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The fourth power of (1 + i) equals -4 -/
theorem one_plus_i_fourth_power : (1 + i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_one_plus_i_fourth_power_l610_61053


namespace NUMINAMATH_CALUDE_swimmer_problem_l610_61063

/-- Swimmer problem -/
theorem swimmer_problem (a s r : ℝ) (ha : a > 0) (hs : s > 0) (hr : r > 0) 
  (h_order : s < r ∧ r < (100 * s) / (50 + s)) :
  ∃ (x z : ℝ),
    x = (100 * s - 50 * r - r * s) / ((3 * s - r) * a) ∧
    z = (100 * s - 50 * r - r * s) / ((r - s) * a) ∧
    x > 0 ∧ z > 0 ∧
    ∃ (y t : ℝ),
      y > 0 ∧ t > 0 ∧
      t * z = (t + a) * y ∧
      t * z = (t + 2 * a) * x ∧
      (50 + r) / z = (50 - r) / x - 2 * a ∧
      (50 + s) / z = (50 - s) / y - a :=
by
  sorry

end NUMINAMATH_CALUDE_swimmer_problem_l610_61063


namespace NUMINAMATH_CALUDE_apple_cost_calculation_l610_61078

/-- The cost of apples in dollars per 3 pounds -/
def apple_cost_per_3_pounds : ℚ := 3

/-- The weight of apples in pounds that we want to calculate the cost for -/
def apple_weight : ℚ := 18

/-- Theorem stating that the cost of 18 pounds of apples is 18 dollars -/
theorem apple_cost_calculation : 
  (apple_weight / 3) * apple_cost_per_3_pounds = 18 := by
  sorry


end NUMINAMATH_CALUDE_apple_cost_calculation_l610_61078


namespace NUMINAMATH_CALUDE_estate_distribution_l610_61009

theorem estate_distribution (a b c d : ℝ) : 
  a > 0 ∧ 
  b = 1.20 * a ∧ 
  c = 1.20 * b ∧ 
  d = 1.20 * c ∧ 
  d - a = 19520 →
  (b = 32176 ∨ c = 32176 ∨ d = 32176) := by
sorry

end NUMINAMATH_CALUDE_estate_distribution_l610_61009


namespace NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l610_61020

/-- The slope angle of a line with parametric equations x = 2 + t and y = 1 + (√3/3)t is π/6 -/
theorem slope_angle_of_parametric_line : 
  ∀ (t : ℝ), 
  let x := 2 + t
  let y := 1 + (Real.sqrt 3 / 3) * t
  let slope := (Real.sqrt 3 / 3)
  let slope_angle := Real.arctan slope
  slope_angle = π / 6 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l610_61020


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l610_61081

-- Problem 1
theorem problem_1 : (3 + Real.sqrt 5) * (Real.sqrt 5 - 2) = Real.sqrt 5 - 1 := by
  sorry

-- Problem 2
theorem problem_2 : (Real.sqrt 12 + Real.sqrt 27) / Real.sqrt 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l610_61081


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l610_61098

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distanceToXAxis (y : ℝ) : ℝ := |y|

/-- Point P in the Cartesian coordinate system -/
def P : ℝ × ℝ := (4, -3)

/-- Theorem: The distance from point P(4, -3) to the x-axis is 3 -/
theorem distance_P_to_x_axis :
  distanceToXAxis P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l610_61098


namespace NUMINAMATH_CALUDE_range_of_z_l610_61031

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) : 
  ∃ z, z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l610_61031


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l610_61024

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x - 2*a = 0)
  (h4 : 4*y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l610_61024


namespace NUMINAMATH_CALUDE_soccer_league_games_l610_61015

theorem soccer_league_games (n : ℕ) (h : n = 25) : n * (n - 1) / 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l610_61015


namespace NUMINAMATH_CALUDE_conic_sections_from_equation_l610_61025

/-- The equation y^4 - 8x^4 = 4y^2 - 4 represents two conic sections -/
theorem conic_sections_from_equation :
  ∃ (f g : ℝ → ℝ → Prop),
    (∀ x y, y^4 - 8*x^4 = 4*y^2 - 4 ↔ f x y ∨ g x y) ∧
    (∃ a b c d e : ℝ, ∀ x y, f x y ↔ (x^2 / a^2) - (y^2 / b^2) = 1) ∧
    (∃ a b c d e : ℝ, ∀ x y, g x y ↔ (x^2 / c^2) + (y^2 / d^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_conic_sections_from_equation_l610_61025


namespace NUMINAMATH_CALUDE_xyz_sum_product_bounds_l610_61068

theorem xyz_sum_product_bounds (x y z : ℝ) 
  (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  let f := x*y + x*z + y*z
  ∃ (M m : ℝ), 
    (∀ a b c : ℝ, 3*(a + b + c) = a^2 + b^2 + c^2 → a*b + a*c + b*c ≤ M) ∧
    (∀ a b c : ℝ, 3*(a + b + c) = a^2 + b^2 + c^2 → m ≤ a*b + a*c + b*c) ∧
    f ≤ M ∧ 
    m ≤ f ∧
    M = 27 ∧ 
    m = -9/8 ∧ 
    M + 5*m = 126/8 :=
by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_product_bounds_l610_61068


namespace NUMINAMATH_CALUDE_game_result_l610_61026

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 2 ≠ 0 then 8
  else if n % 2 = 0 ∧ n % 3 ≠ 0 then 3
  else 0

def chris_rolls : List ℕ := [5, 2, 1, 6]
def dana_rolls : List ℕ := [6, 2, 3, 3]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem game_result :
  let chris_points := total_points chris_rolls
  let dana_points := total_points dana_rolls
  dana_points = 27 ∧ chris_points * dana_points = 297 := by sorry

end NUMINAMATH_CALUDE_game_result_l610_61026


namespace NUMINAMATH_CALUDE_largest_negative_angle_solution_l610_61040

theorem largest_negative_angle_solution :
  let θ : ℝ := -π/2
  let eq (x : ℝ) := (1 - Real.sin x + Real.cos x) / (1 - Real.sin x - Real.cos x) +
                    (1 - Real.sin x - Real.cos x) / (1 - Real.sin x + Real.cos x) = 2
  (eq θ) ∧ 
  (∀ φ, φ < 0 → φ > θ → ¬(eq φ)) :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_angle_solution_l610_61040


namespace NUMINAMATH_CALUDE_removed_digit_not_power_of_two_l610_61072

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def remove_middle_digit (n : ℕ) : ℕ := 
  -- Implementation details omitted
  sorry

theorem removed_digit_not_power_of_two (N : ℕ) (h : is_power_of_two N) :
  ¬ is_power_of_two (remove_middle_digit N) := by
  sorry

end NUMINAMATH_CALUDE_removed_digit_not_power_of_two_l610_61072


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l610_61045

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 8*x + 2

-- Define the roots
variable (p q r : ℝ)

-- State that p, q, and r are roots of f
axiom root_p : f p = 0
axiom root_q : f q = 0
axiom root_r : f r = 0

-- State that p, q, and r are distinct
axiom distinct_roots : p ≠ q ∧ q ≠ r ∧ p ≠ r

-- Theorem to prove
theorem sum_of_reciprocal_squares : 1/p^2 + 1/q^2 + 1/r^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_squares_l610_61045


namespace NUMINAMATH_CALUDE_friday_temperature_l610_61082

/-- Temperatures for Tuesday, Wednesday, Thursday, and Friday --/
structure WeekTemperatures where
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Theorem stating that Friday's temperature is 53°C given the conditions --/
theorem friday_temperature (t : WeekTemperatures) : t.friday = 53 :=
  by
  have h1 : (t.tuesday + t.wednesday + t.thursday) / 3 = 45 := by sorry
  have h2 : (t.wednesday + t.thursday + t.friday) / 3 = 50 := by sorry
  have h3 : t.tuesday = 38 := by sorry
  have h4 : t.tuesday = 38 ∨ t.wednesday = 53 ∨ t.thursday = 53 ∨ t.friday = 53 := by sorry
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l610_61082


namespace NUMINAMATH_CALUDE_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l610_61079

theorem integer_divisible_by_15_with_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), n > 0 ∧ 15 ∣ n ∧ 30 < Real.sqrt n ∧ Real.sqrt n < 30.5 ∧
  (n = 900 ∨ n = 915 ∨ n = 930) := by
sorry

end NUMINAMATH_CALUDE_integer_divisible_by_15_with_sqrt_between_30_and_30_5_l610_61079


namespace NUMINAMATH_CALUDE_equation_two_solutions_l610_61007

theorem equation_two_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, Real.sqrt (9 - x) = x * Real.sqrt (9 - x)) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_two_solutions_l610_61007


namespace NUMINAMATH_CALUDE_fundraising_ratio_l610_61041

-- Define the fundraising goal
def goal : ℕ := 4000

-- Define Ken's collection
def ken_collection : ℕ := 600

-- Define the amount they exceeded the goal by
def excess : ℕ := 600

-- Define the total amount collected
def total_collected : ℕ := goal + excess

-- Define Mary's collection as a function of Ken's
def mary_collection (x : ℚ) : ℚ := x * ken_collection

-- Define Scott's collection as a function of Mary's
def scott_collection (x : ℚ) : ℚ := (1 / 3) * mary_collection x

-- State the theorem
theorem fundraising_ratio : 
  ∃ x : ℚ, 
    scott_collection x + mary_collection x + ken_collection = total_collected ∧ 
    mary_collection x / ken_collection = 5 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_ratio_l610_61041


namespace NUMINAMATH_CALUDE_quadrilateral_exists_for_four_lines_l610_61037

/-- A line in a plane --/
structure Line :=
  (a b c : ℝ)

/-- A point in a plane --/
structure Point :=
  (x y : ℝ)

/-- A region in a plane --/
structure Region :=
  (vertices : List Point)

/-- Checks if a region is a quadrilateral --/
def isQuadrilateral (r : Region) : Prop :=
  r.vertices.length = 4

/-- The set of all regions formed by the intersections of the given lines --/
def regionsFormedByLines (lines : List Line) : Set Region :=
  sorry

/-- The theorem stating that among the regions formed by 4 intersecting lines, 
    there exists at least one quadrilateral --/
theorem quadrilateral_exists_for_four_lines 
  (lines : List Line) 
  (h : lines.length = 4) : 
  ∃ r ∈ regionsFormedByLines lines, isQuadrilateral r :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_exists_for_four_lines_l610_61037


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l610_61032

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon (nonagon) contains 27 diagonals -/
theorem nonagon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l610_61032


namespace NUMINAMATH_CALUDE_car_rate_problem_l610_61077

/-- Given two cars starting at the same time and point, with one car traveling at 60 mph,
    if after 3 hours the distance between them is 30 miles,
    then the rate of the other car is 50 mph. -/
theorem car_rate_problem (rate1 : ℝ) : 
  (60 * 3 = rate1 * 3 + 30) → rate1 = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_rate_problem_l610_61077


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l610_61060

theorem fraction_equals_zero (x : ℝ) :
  (x + 2) / (3 - x) = 0 ∧ 3 - x ≠ 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l610_61060


namespace NUMINAMATH_CALUDE_linear_regression_coefficient_l610_61096

theorem linear_regression_coefficient
  (x : Fin 4 → ℝ)
  (y : Fin 4 → ℝ)
  (h_x : x = ![6, 8, 10, 12])
  (h_y : y = ![6, 5, 3, 2])
  (a : ℝ)
  (h_reg : ∀ i, y i = a * x i + 10.3) :
  a = -0.7 := by
sorry

end NUMINAMATH_CALUDE_linear_regression_coefficient_l610_61096


namespace NUMINAMATH_CALUDE_number_percentage_problem_l610_61056

theorem number_percentage_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 14 → (40/100 : ℝ) * N = 168 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_problem_l610_61056


namespace NUMINAMATH_CALUDE_arithmetic_sequence_of_powers_no_infinite_arithmetic_sequence_of_powers_l610_61022

/-- For any positive integer n, there exists an arithmetic sequence of n different elements 
    where every term is a power of a positive integer greater than 1. -/
theorem arithmetic_sequence_of_powers (n : ℕ+) : 
  ∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i, ∃ (b c : ℕ), c > 1 ∧ a i = b^c) ∧
    (∀ i, a (i + 1) = a i + d) :=
sorry

/-- There does not exist an infinite arithmetic sequence where every term is a power 
    of a positive integer greater than 1. -/
theorem no_infinite_arithmetic_sequence_of_powers : 
  ¬∃ (a : ℕ → ℕ) (d : ℕ), 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ i, ∃ (b c : ℕ), c > 1 ∧ a i = b^c) ∧
    (∀ i, a (i + 1) = a i + d) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_of_powers_no_infinite_arithmetic_sequence_of_powers_l610_61022


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l610_61069

/-- In an isosceles triangle XYZ, where angle X is congruent to angle Z,
    and angle Z is five times angle Y, the measure of angle X is 900/11 degrees. -/
theorem isosceles_triangle_angle_measure (X Y Z : ℝ) : 
  X = Z →                   -- Angle X is congruent to angle Z
  Z = 5 * Y →               -- Angle Z is five times angle Y
  X + Y + Z = 180 →         -- Sum of angles in a triangle is 180 degrees
  X = 900 / 11 :=           -- Measure of angle X is 900/11 degrees
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l610_61069


namespace NUMINAMATH_CALUDE_half_power_inequality_l610_61023

theorem half_power_inequality (a b : ℝ) (h : a > b) : (1/2 : ℝ)^a < (1/2 : ℝ)^b := by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l610_61023


namespace NUMINAMATH_CALUDE_top_card_after_74_shuffles_l610_61049

/-- Represents the order of cards -/
inductive Card
| A
| B
| C
| D
| E

/-- Represents the stack of cards -/
def Stack := List Card

/-- The initial configuration of cards -/
def initial_stack : Stack := [Card.A, Card.B, Card.C, Card.D, Card.E]

/-- Performs one shuffle operation on the stack -/
def shuffle (s : Stack) : Stack :=
  match s with
  | x :: y :: rest => rest ++ [y, x]
  | _ => s

/-- Performs n shuffle operations on the stack -/
def n_shuffles (n : Nat) (s : Stack) : Stack :=
  match n with
  | 0 => s
  | n + 1 => shuffle (n_shuffles n s)

theorem top_card_after_74_shuffles :
  (n_shuffles 74 initial_stack).head? = some Card.E := by
  sorry

end NUMINAMATH_CALUDE_top_card_after_74_shuffles_l610_61049


namespace NUMINAMATH_CALUDE_unique_solution_implies_k_equals_one_l610_61050

/-- The set of real solutions to the quadratic equation kx^2 + 4x + 4 = 0 -/
def A (k : ℝ) : Set ℝ := {x : ℝ | k * x^2 + 4 * x + 4 = 0}

/-- Theorem: If the set A contains only one element, then k = 1 -/
theorem unique_solution_implies_k_equals_one (k : ℝ) : (∃! x, x ∈ A k) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_k_equals_one_l610_61050


namespace NUMINAMATH_CALUDE_triangle_properties_l610_61043

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a / (Real.sin A) = b / (Real.sin B) ∧ 
  b / (Real.sin B) = c / (Real.sin C) →
  -- Given condition: b^2 + c^2 - a^2 = bc
  b^2 + c^2 - a^2 = b * c →
  -- Area of triangle is 3√3/2
  (1/2) * a * b * (Real.sin C) = (3 * Real.sqrt 3) / 2 →
  -- Given condition: sin C + √3 cos C = 2
  Real.sin C + Real.sqrt 3 * Real.cos C = 2 →
  -- Prove: A = π/3 and a = 3
  A = π/3 ∧ a = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l610_61043
