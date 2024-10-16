import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_iff_m_in_range_l949_94981

/-- The equation represents a hyperbola -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 + m) + y^2 / (m + 1) = 1

/-- The range of m for which the equation represents a hyperbola -/
def m_range : Set ℝ := {m | -2 < m ∧ m < -1}

/-- Theorem: The equation represents a hyperbola if and only if m is in the range (-2, -1) -/
theorem hyperbola_iff_m_in_range :
  ∀ m : ℝ, is_hyperbola m ↔ m ∈ m_range :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_in_range_l949_94981


namespace NUMINAMATH_CALUDE_sets_intersection_and_union_l949_94928

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | (x+2)*(x-3) < 0}

theorem sets_intersection_and_union :
  (A ∩ B = {x : ℝ | -2 < x ∧ x < 1}) ∧
  (A ∪ B = {x : ℝ | -3 < x ∧ x < 3}) := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_and_union_l949_94928


namespace NUMINAMATH_CALUDE_ben_hit_seven_l949_94905

-- Define the set of friends
inductive Friend
| Alice | Ben | Cindy | Dave | Ellen | Frank

-- Define the scores for each friend
def score (f : Friend) : ℕ :=
  match f with
  | Friend.Alice => 18
  | Friend.Ben => 13
  | Friend.Cindy => 19
  | Friend.Dave => 16
  | Friend.Ellen => 20
  | Friend.Frank => 5

-- Define the set of possible target scores
def targetScores : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

-- Define a function to check if a pair of scores is valid
def validPair (a b : ℕ) : Prop :=
  a ∈ targetScores ∧ b ∈ targetScores ∧ a ≠ b ∧ a + b = score Friend.Ben

-- Theorem statement
theorem ben_hit_seven :
  ∃ (a b : ℕ), validPair a b ∧ (a = 7 ∨ b = 7) ∧
  (∀ (f : Friend), f ≠ Friend.Ben → ¬∃ (x y : ℕ), validPair x y ∧ (x = 7 ∨ y = 7)) :=
sorry

end NUMINAMATH_CALUDE_ben_hit_seven_l949_94905


namespace NUMINAMATH_CALUDE_train_lateness_l949_94955

/-- Proves that a train with reduced speed arrives 9 minutes late -/
theorem train_lateness (usual_time : ℝ) (speed_ratio : ℝ) :
  usual_time = 12 →
  speed_ratio = 4 / 7 →
  (usual_time / speed_ratio) - usual_time = 9 :=
by
  sorry

#check train_lateness

end NUMINAMATH_CALUDE_train_lateness_l949_94955


namespace NUMINAMATH_CALUDE_tank_fill_time_l949_94932

-- Define the fill/drain rates for each pipe
def rate_A : ℚ := 1 / 10
def rate_B : ℚ := 1 / 20
def rate_C : ℚ := -(1 / 30)  -- Negative because it's draining

-- Define the combined rate
def combined_rate : ℚ := rate_A + rate_B + rate_C

-- Theorem to prove
theorem tank_fill_time :
  (1 : ℚ) / combined_rate = 60 / 7 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_l949_94932


namespace NUMINAMATH_CALUDE_ac_length_l949_94925

/-- Right triangle ABC with altitude AH and circle through A and H -/
structure RightTriangleWithCircle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  X : ℝ × ℝ
  Y : ℝ × ℝ
  -- ABC is a right triangle with right angle at A
  right_angle_A : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0
  -- AH is altitude
  altitude_AH : (H.1 - A.1) * (C.1 - B.1) + (H.2 - A.2) * (C.2 - B.2) = 0
  -- Circle passes through A, H, X, Y
  on_circle : ∃ (center : ℝ × ℝ) (radius : ℝ),
    (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2 ∧
    (H.1 - center.1)^2 + (H.2 - center.2)^2 = radius^2 ∧
    (X.1 - center.1)^2 + (X.2 - center.2)^2 = radius^2 ∧
    (Y.1 - center.1)^2 + (Y.2 - center.2)^2 = radius^2
  -- X is on AB
  X_on_AB : ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ X = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))
  -- Y is on AC
  Y_on_AC : ∃ (s : ℝ), 0 ≤ s ∧ s ≤ 1 ∧ Y = (A.1 + s * (C.1 - A.1), A.2 + s * (C.2 - A.2))
  -- Given lengths
  AX_length : Real.sqrt ((X.1 - A.1)^2 + (X.2 - A.2)^2) = 5
  AY_length : Real.sqrt ((Y.1 - A.1)^2 + (Y.2 - A.2)^2) = 6
  AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 9

/-- The main theorem to prove -/
theorem ac_length (triangle : RightTriangleWithCircle) :
  Real.sqrt ((triangle.C.1 - triangle.A.1)^2 + (triangle.C.2 - triangle.A.2)^2) = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l949_94925


namespace NUMINAMATH_CALUDE_fourth_term_value_l949_94921

def S (n : ℕ+) : ℤ := 2 * n.val ^ 2 - 3 * n.val

theorem fourth_term_value : ∃ (a : ℕ+ → ℤ), a 4 = 11 :=
  sorry

end NUMINAMATH_CALUDE_fourth_term_value_l949_94921


namespace NUMINAMATH_CALUDE_transfer_amount_christinas_transfer_l949_94994

/-- The amount transferred out of a bank account is equal to the difference
    between the initial balance and the final balance. -/
theorem transfer_amount (initial_balance final_balance : ℕ) 
  (h : initial_balance ≥ final_balance) :
  initial_balance - final_balance = 
  (initial_balance : ℤ) - (final_balance : ℤ) :=
by sorry

/-- Christina's bank transfer problem -/
theorem christinas_transfer : 
  (27004 : ℕ) - (26935 : ℕ) = (69 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_transfer_amount_christinas_transfer_l949_94994


namespace NUMINAMATH_CALUDE_number_greater_than_fraction_l949_94944

theorem number_greater_than_fraction : ∃ x : ℝ, x = 40 ∧ 0.8 * x > (2 / 5) * 25 + 22 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_fraction_l949_94944


namespace NUMINAMATH_CALUDE_median_to_mean_l949_94939

theorem median_to_mean (m : ℝ) : 
  let set := [m, m + 3, m + 7, m + 10, m + 12]
  m + 7 = 12 → 
  (set.sum / set.length : ℝ) = 11.4 := by
sorry

end NUMINAMATH_CALUDE_median_to_mean_l949_94939


namespace NUMINAMATH_CALUDE_resistor_value_l949_94952

/-- Given two identical resistors connected in series to a DC voltage source,
    if the voltage across one resistor is 2 V and the current through the circuit is 4 A,
    then the resistance of each resistor is 0.5 Ω. -/
theorem resistor_value (R₀ : ℝ) (U V I : ℝ) : 
  U = 2 → -- Voltage across one resistor
  V = 2 * U → -- Total voltage
  I = 4 → -- Current through the circuit
  V = I * (2 * R₀) → -- Ohm's law
  R₀ = 0.5 := by
  sorry

#check resistor_value

end NUMINAMATH_CALUDE_resistor_value_l949_94952


namespace NUMINAMATH_CALUDE_unique_prime_perfect_power_l949_94957

def is_perfect_power (x : ℕ) : Prop :=
  ∃ m n, m > 1 ∧ n ≥ 2 ∧ x = m^n

theorem unique_prime_perfect_power :
  ∀ p : ℕ, p ≤ 1000 → Prime p → is_perfect_power (2*p + 1) → p = 13 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_perfect_power_l949_94957


namespace NUMINAMATH_CALUDE_quadratic_a_value_l949_94930

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_a_value 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : f = QuadraticFunction a b c) 
  (h2 : f 0 = 3) 
  (h3 : ∀ x, f x ≤ f 2) 
  (h4 : f 2 = 5) : 
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_a_value_l949_94930


namespace NUMINAMATH_CALUDE_binomial_22_10_l949_94988

theorem binomial_22_10 (h1 : Nat.choose 20 8 = 125970)
                       (h2 : Nat.choose 20 9 = 167960)
                       (h3 : Nat.choose 20 10 = 184756) :
  Nat.choose 22 10 = 646646 := by
  sorry

end NUMINAMATH_CALUDE_binomial_22_10_l949_94988


namespace NUMINAMATH_CALUDE_polyhedron_property_l949_94985

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  euler_formula : V - E + F = 2
  face_count : F = 42
  face_types : t + h = F
  edge_formula : E = (3 * t + 6 * h) / 2
  vertex_face_relation : 3 * t + 2 * h = V

/-- The main theorem to be proved -/
theorem polyhedron_property (p : ConvexPolyhedron) : 
  100 * 2 + 10 * 3 + p.V = 328 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l949_94985


namespace NUMINAMATH_CALUDE_double_negation_and_abs_value_l949_94923

theorem double_negation_and_abs_value : 
  (-(-2) = 2) ∧ (-(abs (-2)) = -2) := by sorry

end NUMINAMATH_CALUDE_double_negation_and_abs_value_l949_94923


namespace NUMINAMATH_CALUDE_cube_diff_divisibility_l949_94950

theorem cube_diff_divisibility (m n k : ℕ) (hm : Odd m) (hn : Odd n) (hk : k > 0) :
  (2^k ∣ m^3 - n^3) ↔ (2^k ∣ m - n) := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_divisibility_l949_94950


namespace NUMINAMATH_CALUDE_triangle_properties_l949_94916

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about the measure of angle C and the value of side b in a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C)
  (h2 : t.a^2 - t.c^2 = 2 * t.b^2)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 21 * Real.sqrt 3) :
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l949_94916


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l949_94902

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l949_94902


namespace NUMINAMATH_CALUDE_ellipse_point_distance_to_y_axis_l949_94929

/-- Given an ellipse with equation x²/4 + y² = 1 and foci at (-√3, 0) and (√3, 0),
    if a point M(x,y) on the ellipse satisfies the condition that the vectors from
    the foci to M are perpendicular, then the absolute value of x is 2√6/3. -/
theorem ellipse_point_distance_to_y_axis 
  (x y : ℝ) 
  (h_ellipse : x^2/4 + y^2 = 1) 
  (h_perpendicular : (x + Real.sqrt 3) * (x - Real.sqrt 3) + y * y = 0) : 
  |x| = 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_point_distance_to_y_axis_l949_94929


namespace NUMINAMATH_CALUDE_midpoint_ratio_range_l949_94982

/-- Given two lines and a point M that is the midpoint of two points on these lines,
    prove that the ratio of y₀/x₀ falls within a specific range. -/
theorem midpoint_ratio_range (P Q : ℝ × ℝ) (x₀ y₀ : ℝ) :
  (P.1 + 2 * P.2 - 1 = 0) →  -- P is on the line x + 2y - 1 = 0
  (Q.1 + 2 * Q.2 + 3 = 0) →  -- Q is on the line x + 2y + 3 = 0
  ((x₀, y₀) = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →  -- M(x₀, y₀) is the midpoint of PQ
  (y₀ > x₀ + 2) →  -- Given condition
  (-1/2 < y₀ / x₀) ∧ (y₀ / x₀ < -1/5) :=  -- The range of y₀/x₀
by sorry

end NUMINAMATH_CALUDE_midpoint_ratio_range_l949_94982


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l949_94933

-- Define the sets A and B
def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {5, 7, 8, 9}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l949_94933


namespace NUMINAMATH_CALUDE_fruit_difference_l949_94917

theorem fruit_difference (watermelons peaches plums : ℕ) : 
  watermelons = 1 →
  peaches > watermelons →
  plums = 3 * peaches →
  watermelons + peaches + plums = 53 →
  peaches - watermelons = 12 :=
by sorry

end NUMINAMATH_CALUDE_fruit_difference_l949_94917


namespace NUMINAMATH_CALUDE_N_mod_500_l949_94962

/-- A function that counts the number of 1s in the binary representation of a natural number -/
def countOnes (n : ℕ) : ℕ := sorry

/-- The sequence of positive integers whose binary representation has exactly 7 ones -/
def S : List ℕ := sorry

/-- The 500th number in the sequence S -/
def N : ℕ := sorry

theorem N_mod_500 : N % 500 = 375 := by sorry

end NUMINAMATH_CALUDE_N_mod_500_l949_94962


namespace NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l949_94960

/-- Given real functions f and g defined on ℝ, satisfying certain conditions,
    prove that the absolute value of g is bounded by 1 for all real numbers. -/
theorem bounded_g_given_bounded_f (f g : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x : ℝ, f x ≠ 0)
  (h3 : ∀ x : ℝ, |f x| ≤ 1) :
  ∀ y : ℝ, |g y| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_bounded_g_given_bounded_f_l949_94960


namespace NUMINAMATH_CALUDE_log_36_in_terms_of_a_b_l949_94904

theorem log_36_in_terms_of_a_b (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 36 = 2 * a + 2 * b := by
  sorry

end NUMINAMATH_CALUDE_log_36_in_terms_of_a_b_l949_94904


namespace NUMINAMATH_CALUDE_cube_surface_area_l949_94910

/-- Given a cube with vertices A, B, and C, prove that its surface area is 150 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (2, 5, 3) → B = (2, 10, 3) → C = (2, 5, 8) → 
  (let surface_area := 6 * (dist A B) ^ 2
   surface_area = 150) := by
  sorry

#check cube_surface_area

end NUMINAMATH_CALUDE_cube_surface_area_l949_94910


namespace NUMINAMATH_CALUDE_rick_cards_count_l949_94996

theorem rick_cards_count : ℕ := by
  -- Define the number of cards Rick kept
  let cards_kept : ℕ := 15

  -- Define the number of cards given to Miguel
  let cards_to_miguel : ℕ := 13

  -- Define the number of friends and cards given to each friend
  let num_friends : ℕ := 8
  let cards_per_friend : ℕ := 12

  -- Define the number of sisters and cards given to each sister
  let num_sisters : ℕ := 2
  let cards_per_sister : ℕ := 3

  -- Calculate the total number of cards
  let total_cards : ℕ := 
    cards_kept + 
    cards_to_miguel + 
    (num_friends * cards_per_friend) + 
    (num_sisters * cards_per_sister)

  -- Prove that the total number of cards is 130
  have h : total_cards = 130 := by sorry

  -- Return the result
  exact 130

end NUMINAMATH_CALUDE_rick_cards_count_l949_94996


namespace NUMINAMATH_CALUDE_area_difference_l949_94915

/-- The difference in combined area (front and back) between two rectangular sheets of paper -/
theorem area_difference (sheet1_length sheet1_width sheet2_length sheet2_width : ℝ) 
  (h1 : sheet1_length = 11) 
  (h2 : sheet1_width = 13) 
  (h3 : sheet2_length = 6.5) 
  (h4 : sheet2_width = 11) : 
  2 * (sheet1_length * sheet1_width) - 2 * (sheet2_length * sheet2_width) = 143 := by
  sorry

end NUMINAMATH_CALUDE_area_difference_l949_94915


namespace NUMINAMATH_CALUDE_garden_perimeter_l949_94997

/-- 
A rectangular garden has a diagonal of 34 meters and an area of 240 square meters.
This theorem proves that the perimeter of such a garden is 80 meters.
-/
theorem garden_perimeter : 
  ∀ (a b : ℝ), 
  a > 0 → b > 0 →  -- Ensure positive dimensions
  a * b = 240 →    -- Area condition
  a^2 + b^2 = 34^2 →  -- Diagonal condition
  2 * (a + b) = 80 :=  -- Perimeter calculation
by
  sorry

#check garden_perimeter

end NUMINAMATH_CALUDE_garden_perimeter_l949_94997


namespace NUMINAMATH_CALUDE_negative_a_cubed_times_a_squared_l949_94935

theorem negative_a_cubed_times_a_squared (a : ℝ) : (-a)^3 * a^2 = -a^5 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_cubed_times_a_squared_l949_94935


namespace NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l949_94927

theorem definite_integral_exp_plus_2x : ∫ x in (0:ℝ)..1, (Real.exp x + 2 * x) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_definite_integral_exp_plus_2x_l949_94927


namespace NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_at_m_one_l949_94980

theorem min_sum_squares (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂ →
  (∃ D : ℝ, D ≥ 0 ∧ D = (m + 3)^2) →
  x₁ + x₂ = -(m + 1) →
  x₁ * x₂ = 2*m - 2 →
  x₁^2 + x₂^2 ≥ 4 :=
by sorry

theorem min_sum_squares_at_m_one (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 + x₂^2 = (x₁ + x₂)^2 - 2*x₁*x₂ →
  (∃ D : ℝ, D ≥ 0 ∧ D = (m + 3)^2) →
  x₁ + x₂ = -(m + 1) →
  x₁ * x₂ = 2*m - 2 →
  m = 1 →
  x₁^2 + x₂^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_min_sum_squares_at_m_one_l949_94980


namespace NUMINAMATH_CALUDE_fixed_points_of_f_composition_l949_94987

def f (x : ℝ) : ℝ := x^2 - 4*x

theorem fixed_points_of_f_composition (x : ℝ) : 
  f (f x) = f x ↔ x ∈ ({-1, 0, 4, 5} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_composition_l949_94987


namespace NUMINAMATH_CALUDE_complex_equality_implies_sum_zero_l949_94948

theorem complex_equality_implies_sum_zero (z : ℂ) (x y : ℝ) :
  Complex.abs (z + 1) = Complex.abs (z - Complex.I) →
  z = Complex.mk x y →
  x + y = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_implies_sum_zero_l949_94948


namespace NUMINAMATH_CALUDE_slower_train_speed_l949_94964

theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 80) 
  (h2 : faster_speed = 52) 
  (h3 : passing_time = 36) : 
  ∃ slower_speed : ℝ, 
    slower_speed = 36 ∧ 
    (faster_speed - slower_speed) * passing_time / 3600 * 1000 = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l949_94964


namespace NUMINAMATH_CALUDE_hydrochloric_acid_solution_l949_94914

/-- Represents the volume of pure hydrochloric acid needed to be added -/
def x : ℝ := sorry

/-- The initial volume of the solution in liters -/
def initial_volume : ℝ := 60

/-- The initial concentration of hydrochloric acid as a decimal -/
def initial_concentration : ℝ := 0.10

/-- The target concentration of hydrochloric acid as a decimal -/
def target_concentration : ℝ := 0.15

theorem hydrochloric_acid_solution :
  initial_concentration * initial_volume + x = target_concentration * (initial_volume + x) := by
  sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_solution_l949_94914


namespace NUMINAMATH_CALUDE_congruence_problem_l949_94967

theorem congruence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1234 ≡ n [ZMOD 9] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l949_94967


namespace NUMINAMATH_CALUDE_stone_197_is_5_and_prime_l949_94984

/-- The number of stones in the line -/
def num_stones : ℕ := 13

/-- The length of one full cycle in the counting pattern -/
def cycle_length : ℕ := 24

/-- The count we're interested in -/
def target_count : ℕ := 197

/-- Function to determine which stone corresponds to a given count -/
def stone_for_count (count : ℕ) : ℕ :=
  (count - 1) % cycle_length + 1

/-- Primality check -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem stone_197_is_5_and_prime :
  stone_for_count target_count = 5 ∧ is_prime 5 := by
  sorry


end NUMINAMATH_CALUDE_stone_197_is_5_and_prime_l949_94984


namespace NUMINAMATH_CALUDE_exists_special_function_l949_94978

theorem exists_special_function :
  ∃ f : ℕ+ → ℕ+,
    (∀ m n : ℕ+, m < n → f m < f n) ∧
    f 1 = 2 ∧
    ∀ n : ℕ+, f (f n) = f n + n :=
by sorry

end NUMINAMATH_CALUDE_exists_special_function_l949_94978


namespace NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_sum_of_squares_l949_94924

theorem quadratic_equation_from_sum_and_sum_of_squares 
  (x₁ x₂ : ℝ) 
  (h_sum : x₁ + x₂ = 3) 
  (h_sum_squares : x₁^2 + x₂^2 = 5) :
  ∀ x : ℝ, x^2 - 3*x + 2 = 0 ↔ (x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_sum_and_sum_of_squares_l949_94924


namespace NUMINAMATH_CALUDE_sum_simplification_l949_94979

theorem sum_simplification :
  (296 + 297 + 298 + 299 + 1 + 2 + 3 + 4 = 1200) ∧
  (457 + 458 + 459 + 460 + 461 + 462 + 463 = 3220) := by
  sorry

end NUMINAMATH_CALUDE_sum_simplification_l949_94979


namespace NUMINAMATH_CALUDE_line_segment_polar_equation_l949_94973

theorem line_segment_polar_equation :
  ∀ (ρ θ : ℝ), 
    0 ≤ θ ∧ θ ≤ π/2 →
    ρ = 1 / (Real.cos θ + Real.sin θ) →
    ∃ (x y : ℝ),
      x = ρ * Real.cos θ ∧
      y = ρ * Real.sin θ ∧
      y = 1 - x ∧
      0 ≤ x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_polar_equation_l949_94973


namespace NUMINAMATH_CALUDE_equation_solutions_l949_94993

theorem equation_solutions : 
  (∀ x : ℝ, (x - 1)^2 = 25 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 ↔ x = 1 ∨ x = 3) ∧
  (∀ x : ℝ, (2*x + 1)^2 = 2*(2*x + 1) ↔ x = -1/2 ∨ x = 1/2) ∧
  (∀ x : ℝ, 2*x^2 - 5*x + 3 = 0 ↔ x = 1 ∨ x = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l949_94993


namespace NUMINAMATH_CALUDE_three_questions_uniquely_identify_l949_94956

-- Define the set of geometric figures
inductive GeometricFigure
  | Circle
  | Ellipse
  | Triangle
  | Square
  | Rectangle
  | Parallelogram
  | Trapezoid

-- Define the properties of the figures
def is_curve (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Circle => true
  | GeometricFigure.Ellipse => true
  | _ => false

def has_axis_symmetry (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Circle => true
  | GeometricFigure.Ellipse => true
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true
  | _ => false

def has_center_symmetry (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Circle => true
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true
  | GeometricFigure.Parallelogram => true
  | _ => false

-- Theorem statement
theorem three_questions_uniquely_identify (f1 f2 : GeometricFigure) :
  (is_curve f1 = is_curve f2) →
  (has_axis_symmetry f1 = has_axis_symmetry f2) →
  (has_center_symmetry f1 = has_center_symmetry f2) →
  f1 = f2 :=
by sorry

end NUMINAMATH_CALUDE_three_questions_uniquely_identify_l949_94956


namespace NUMINAMATH_CALUDE_no_candies_to_remove_for_30_and_5_l949_94907

/-- Given a number of candies and sisters, calculate the minimum number of candies to remove for even distribution -/
def min_candies_to_remove (candies : ℕ) (sisters : ℕ) : ℕ :=
  candies % sisters

/-- Prove that for 30 candies and 5 sisters, no candies need to be removed for even distribution -/
theorem no_candies_to_remove_for_30_and_5 :
  min_candies_to_remove 30 5 = 0 := by
  sorry

#eval min_candies_to_remove 30 5

end NUMINAMATH_CALUDE_no_candies_to_remove_for_30_and_5_l949_94907


namespace NUMINAMATH_CALUDE_complex_expression_equality_l949_94934

theorem complex_expression_equality (z : ℂ) (h : z = 1 + Complex.I) :
  5 / z + z^2 = 5/2 - (1/2) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l949_94934


namespace NUMINAMATH_CALUDE_red_shirt_pairs_l949_94954

theorem red_shirt_pairs (total_students : ℕ) (green_students : ℕ) (red_students : ℕ) 
  (total_pairs : ℕ) (green_green_pairs : ℕ) :
  total_students = 132 →
  green_students = 63 →
  red_students = 69 →
  total_pairs = 66 →
  green_green_pairs = 27 →
  total_students = green_students + red_students →
  2 * total_pairs = total_students →
  ∃ (red_red_pairs : ℕ),
    red_red_pairs = 30 ∧
    red_red_pairs + green_green_pairs + (green_students + red_students - 2 * (red_red_pairs + green_green_pairs)) = total_pairs :=
by sorry

end NUMINAMATH_CALUDE_red_shirt_pairs_l949_94954


namespace NUMINAMATH_CALUDE_business_income_calculation_l949_94942

theorem business_income_calculation (spending income : ℚ) (profit : ℚ) : 
  spending / income = 5 / 9 →
  profit = income - spending →
  profit = 48000 →
  income = 108000 := by
sorry

end NUMINAMATH_CALUDE_business_income_calculation_l949_94942


namespace NUMINAMATH_CALUDE_obtuse_triangles_in_17gon_l949_94903

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A triangle formed by three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) (polygon : RegularPolygon n) where
  v1 : Fin n
  v2 : Fin n
  v3 : Fin n

/-- Predicate to determine if a triangle is obtuse -/
def isObtuseTriangle (n : ℕ) (polygon : RegularPolygon n) (triangle : PolygonTriangle n polygon) : Prop :=
  sorry

/-- Count the number of obtuse triangles in a regular polygon -/
def countObtuseTriangles (n : ℕ) (polygon : RegularPolygon n) : ℕ :=
  sorry

theorem obtuse_triangles_in_17gon :
  ∀ (polygon : RegularPolygon 17),
  countObtuseTriangles 17 polygon = 476 :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangles_in_17gon_l949_94903


namespace NUMINAMATH_CALUDE_letters_with_both_dot_and_line_l949_94968

/-- Represents the number of letters in the alphabet -/
def total_letters : ℕ := 40

/-- Represents the number of letters with only a straight line -/
def straight_line_only : ℕ := 24

/-- Represents the number of letters with only a dot -/
def dot_only : ℕ := 7

/-- Represents the number of letters with both a dot and a straight line -/
def both : ℕ := total_letters - straight_line_only - dot_only

theorem letters_with_both_dot_and_line :
  both = 9 :=
sorry

end NUMINAMATH_CALUDE_letters_with_both_dot_and_line_l949_94968


namespace NUMINAMATH_CALUDE_strawberry_sales_l949_94998

/-- The number of pints of strawberries sold by a supermarket -/
def pints_sold : ℕ := 54

/-- The revenue from selling strawberries on sale -/
def sale_revenue : ℕ := 216

/-- The revenue that would have been made without the sale -/
def non_sale_revenue : ℕ := 324

/-- The price difference between non-sale and sale price per pint -/
def price_difference : ℕ := 2

theorem strawberry_sales :
  ∃ (sale_price : ℚ),
    sale_price > 0 ∧
    sale_price * pints_sold = sale_revenue ∧
    (sale_price + price_difference) * pints_sold = non_sale_revenue :=
by sorry

end NUMINAMATH_CALUDE_strawberry_sales_l949_94998


namespace NUMINAMATH_CALUDE_closest_multiple_of_18_to_3050_l949_94947

-- Define a function to check if a number is divisible by both 2 and 9
def is_multiple_of_18 (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 9 = 0

-- Define a function to calculate the absolute difference between two numbers
def abs_diff (a b : ℕ) : ℕ :=
  if a ≥ b then a - b else b - a

-- State the theorem
theorem closest_multiple_of_18_to_3050 :
  ∀ n : ℕ, is_multiple_of_18 n → abs_diff n 3050 ≥ abs_diff 3042 3050 :=
by sorry

end NUMINAMATH_CALUDE_closest_multiple_of_18_to_3050_l949_94947


namespace NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l949_94999

def tomatoes_problem (yesterday today total : ℕ) : Prop :=
  (yesterday = 120) ∧
  (today = yesterday + 50) ∧
  (total = yesterday + today)

theorem uncle_jerry_tomatoes : ∃ yesterday today total : ℕ,
  tomatoes_problem yesterday today total ∧ total = 290 := by sorry

end NUMINAMATH_CALUDE_uncle_jerry_tomatoes_l949_94999


namespace NUMINAMATH_CALUDE_calculation_proof_l949_94972

theorem calculation_proof : 3 * 8 * 9 + 18 / 3 - 2^3 = 214 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l949_94972


namespace NUMINAMATH_CALUDE_sammy_bottle_caps_l949_94976

theorem sammy_bottle_caps :
  ∀ (billie janine sammy : ℕ),
    billie = 2 →
    janine = 3 * billie →
    sammy = janine + 2 →
    sammy = 8 := by
  sorry

end NUMINAMATH_CALUDE_sammy_bottle_caps_l949_94976


namespace NUMINAMATH_CALUDE_log_inequality_l949_94941

theorem log_inequality : 
  Real.log 2 / Real.log 3 < 2/3 ∧ 
  2/3 < Real.log 75 / Real.log 625 ∧ 
  Real.log 75 / Real.log 625 < Real.log 3 / Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l949_94941


namespace NUMINAMATH_CALUDE_total_defective_rate_is_correct_l949_94965

/-- The defective rate of worker x -/
def worker_x_rate : ℝ := 0.005

/-- The defective rate of worker y -/
def worker_y_rate : ℝ := 0.008

/-- The fraction of products checked by worker y -/
def worker_y_fraction : ℝ := 0.8

/-- The fraction of products checked by worker x -/
def worker_x_fraction : ℝ := 1 - worker_y_fraction

/-- The total defective rate of all products -/
def total_defective_rate : ℝ := worker_x_rate * worker_x_fraction + worker_y_rate * worker_y_fraction

theorem total_defective_rate_is_correct :
  total_defective_rate = 0.0074 := by sorry

end NUMINAMATH_CALUDE_total_defective_rate_is_correct_l949_94965


namespace NUMINAMATH_CALUDE_oneSeventhIncreaseAfterRemoval_l949_94971

/-- The decimal representation of 1/7 -/
def oneSeventhDecimal : ℚ := 1 / 7

/-- The position of the digit to be removed -/
def digitPosition : ℕ := 2021

/-- The function that removes the digit at the specified position and shifts subsequent digits -/
def removeDigitAndShift (q : ℚ) (pos : ℕ) : ℚ :=
  sorry -- Implementation details omitted

/-- Theorem stating that removing the 2021st digit after the decimal point in 1/7 increases the value -/
theorem oneSeventhIncreaseAfterRemoval :
  removeDigitAndShift oneSeventhDecimal digitPosition > oneSeventhDecimal :=
sorry

end NUMINAMATH_CALUDE_oneSeventhIncreaseAfterRemoval_l949_94971


namespace NUMINAMATH_CALUDE_min_cos_sum_sin_triangle_angles_l949_94975

theorem min_cos_sum_sin_triangle_angles (A B C : Real) : 
  A + B + C = π → 
  A > 0 → B > 0 → C > 0 →
  ∃ (m : Real), m = -2 * Real.sqrt 6 / 9 ∧ 
    ∀ (X Y Z : Real), X + Y + Z = π → X > 0 → Y > 0 → Z > 0 → 
      m ≤ Real.cos X * (Real.sin Y + Real.sin Z) :=
by sorry

end NUMINAMATH_CALUDE_min_cos_sum_sin_triangle_angles_l949_94975


namespace NUMINAMATH_CALUDE_prime_divisor_bound_l949_94991

theorem prime_divisor_bound (p : ℕ) : 
  Prime p → 
  (Finset.card (Nat.divisors (p^2 + 71)) ≤ 10) → 
  p = 2 ∨ p = 3 := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_bound_l949_94991


namespace NUMINAMATH_CALUDE_sqrt_a_sqrt_a_l949_94992

theorem sqrt_a_sqrt_a (a : ℝ) (ha : 0 < a) : Real.sqrt (a * Real.sqrt a) = a^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_sqrt_a_l949_94992


namespace NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l949_94974

theorem unique_prime_sum_and_difference : 
  ∃! p : ℕ, 
    Prime p ∧ 
    (∃ q₁ q₂ : ℕ, Prime q₁ ∧ Prime q₂ ∧ p = q₁ + q₂) ∧
    (∃ q₃ q₄ : ℕ, Prime q₃ ∧ Prime q₄ ∧ q₃ > q₄ ∧ p = q₃ - q₄) ∧
    p = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_and_difference_l949_94974


namespace NUMINAMATH_CALUDE_square_sum_given_product_and_sum_square_l949_94970

theorem square_sum_given_product_and_sum_square (x y : ℝ) 
  (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_product_and_sum_square_l949_94970


namespace NUMINAMATH_CALUDE_spinner_probability_l949_94963

theorem spinner_probability (pA pB pC pD pE : ℚ) : 
  pA = 1/3 →
  pB = 1/6 →
  pC = 2*pE →
  pD = 2*pE →
  pA + pB + pC + pD + pE = 1 →
  pE = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l949_94963


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l949_94912

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l949_94912


namespace NUMINAMATH_CALUDE_function_growth_l949_94995

/-- For any differentiable function f: ℝ → ℝ, if f'(x) > f(x) for all x ∈ ℝ,
    then f(a) > e^a * f(0) for any a > 0. -/
theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) :
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l949_94995


namespace NUMINAMATH_CALUDE_sugar_packet_weight_l949_94977

-- Define the number of packets sold per week
def packets_per_week : ℕ := 20

-- Define the total weight of sugar sold per week in kilograms
def total_weight_kg : ℕ := 2

-- Define the conversion factor from kilograms to grams
def kg_to_g : ℕ := 1000

-- Theorem stating that each packet weighs 100 grams
theorem sugar_packet_weight :
  (total_weight_kg * kg_to_g) / packets_per_week = 100 := by
sorry

end NUMINAMATH_CALUDE_sugar_packet_weight_l949_94977


namespace NUMINAMATH_CALUDE_unique_solution_l949_94908

def satisfies_equation (x y : ℕ+) : Prop :=
  (x.val ^ 4) * (y.val ^ 4) - 16 * (x.val ^ 2) * (y.val ^ 2) + 15 = 0

theorem unique_solution : 
  ∃! p : ℕ+ × ℕ+, satisfies_equation p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l949_94908


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l949_94901

/-- The y-intercept of the line 3x - 5y = 10 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 3*x - 5*y = 10 → x = 0 → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l949_94901


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l949_94922

theorem inequality_system_solutions : 
  {x : ℤ | x ≥ 0 ∧ 4 * (x + 1) ≤ 7 * x + 10 ∧ x - 5 < (x - 8) / 3} = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l949_94922


namespace NUMINAMATH_CALUDE_janous_inequality_janous_equality_l949_94919

theorem janous_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 := by
  sorry

theorem janous_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ y = z ∧ x = 2 * y := by
  sorry

end NUMINAMATH_CALUDE_janous_inequality_janous_equality_l949_94919


namespace NUMINAMATH_CALUDE_cubic_function_root_condition_l949_94983

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem cubic_function_root_condition (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ < 0) → a > 2 := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_root_condition_l949_94983


namespace NUMINAMATH_CALUDE_red_marble_fraction_l949_94909

theorem red_marble_fraction (total : ℝ) (h_total_pos : total > 0) : 
  let blue := (2/3) * total
  let red := (1/3) * total
  let new_blue := 3 * blue
  let new_total := new_blue + red
  red / new_total = 1/7 := by
sorry

end NUMINAMATH_CALUDE_red_marble_fraction_l949_94909


namespace NUMINAMATH_CALUDE_find_carols_number_l949_94940

/-- A prime number between 10 and 99, inclusive. -/
def TwoDigitPrime := {p : Nat // p.Prime ∧ 10 ≤ p ∧ p ≤ 99}

/-- The problem statement -/
theorem find_carols_number 
  (a b c : TwoDigitPrime) 
  (h1 : b.val + c.val = 14)
  (h2 : a.val + c.val = 20)
  (h3 : a.val + b.val = 18)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  c.val = 11 := by
  sorry

#check find_carols_number

end NUMINAMATH_CALUDE_find_carols_number_l949_94940


namespace NUMINAMATH_CALUDE_triangle_intersection_area_l949_94913

/-- Given a triangle PQR with vertices P(0, 10), Q(3, 0), R(9, 0),
    and a horizontal line y=s intersecting PQ at V and PR at W,
    if the area of triangle PVW is 18, then s = 10 - 2√15. -/
theorem triangle_intersection_area (s : ℝ) : 
  let P : ℝ × ℝ := (0, 10)
  let Q : ℝ × ℝ := (3, 0)
  let R : ℝ × ℝ := (9, 0)
  let V : ℝ × ℝ := ((3/10) * (10 - s), s)
  let W : ℝ × ℝ := ((9/10) * (10 - s), s)
  let area_PVW : ℝ := (1/2) * ((W.1 - V.1) * (P.2 - V.2))
  area_PVW = 18 → s = 10 - 2 * Real.sqrt 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_intersection_area_l949_94913


namespace NUMINAMATH_CALUDE_cube_sum_of_equal_ratios_l949_94966

theorem cube_sum_of_equal_ratios (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) :
  a^3 + b^3 + c^3 = -27 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_equal_ratios_l949_94966


namespace NUMINAMATH_CALUDE_system_of_equations_range_l949_94961

theorem system_of_equations_range (x y k : ℝ) : 
  x - y = k - 1 →
  3 * x + 2 * y = 4 * k + 5 →
  2 * x + 3 * y > 7 →
  k > 1/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_range_l949_94961


namespace NUMINAMATH_CALUDE_complete_square_l949_94936

theorem complete_square (x : ℝ) : x^2 - 6*x + 10 = (x - 3)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_l949_94936


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_implies_m_less_than_two_l949_94938

/-- A quadratic equation with parameter m -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4 - 4*m

theorem quadratic_real_solutions_implies_m_less_than_two (m : ℝ) :
  (∃ x : ℝ, quadratic_equation x m) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_implies_m_less_than_two_l949_94938


namespace NUMINAMATH_CALUDE_james_works_six_hours_l949_94989

/-- Calculates the time James spends working on chores given the following conditions:
  * There are 3 bedrooms, 1 living room, and 2 bathrooms to clean
  * Bedrooms each take 20 minutes to clean
  * Living room takes as long as the 3 bedrooms combined
  * Bathroom takes twice as long as the living room
  * Outside cleaning takes twice as long as cleaning the house
  * Chores are split with 2 siblings who are just as fast -/
def james_working_time : ℕ :=
  let num_bedrooms : ℕ := 3
  let num_livingrooms : ℕ := 1
  let num_bathrooms : ℕ := 2
  let bedroom_cleaning_time : ℕ := 20
  let livingroom_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time
  let bathroom_cleaning_time : ℕ := 2 * livingroom_cleaning_time
  let inside_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time +
                                  num_livingrooms * livingroom_cleaning_time +
                                  num_bathrooms * bathroom_cleaning_time
  let outside_cleaning_time : ℕ := 2 * inside_cleaning_time
  let total_cleaning_time : ℕ := inside_cleaning_time + outside_cleaning_time
  let num_siblings : ℕ := 2
  let james_time_minutes : ℕ := total_cleaning_time / (num_siblings + 1)
  james_time_minutes / 60

theorem james_works_six_hours : james_working_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_james_works_six_hours_l949_94989


namespace NUMINAMATH_CALUDE_lcm_gcd_48_180_l949_94920

theorem lcm_gcd_48_180 : 
  (Nat.lcm 48 180 = 720) ∧ (Nat.gcd 48 180 = 12) := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_48_180_l949_94920


namespace NUMINAMATH_CALUDE_min_points_for_top_two_l949_94918

/-- Represents a soccer tournament --/
structure Tournament :=
  (num_teams : Nat)
  (scoring_system : List Nat)

/-- Calculates the total number of matches in a round-robin tournament --/
def total_matches (t : Tournament) : Nat :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Calculates the maximum total points possible in the tournament --/
def max_total_points (t : Tournament) : Nat :=
  (total_matches t) * (t.scoring_system.head!)

/-- Theorem: In a 4-team round-robin tournament with the given scoring system,
    a team needs at least 7 points to guarantee a top-two finish --/
theorem min_points_for_top_two (t : Tournament) 
  (h1 : t.num_teams = 4)
  (h2 : t.scoring_system = [3, 1, 0]) : 
  ∃ (min_points : Nat), 
    (min_points = 7) ∧ 
    (∀ (team_points : Nat), 
      team_points ≥ min_points → 
      (max_total_points t - team_points) / (t.num_teams - 1) < team_points) :=
by sorry

end NUMINAMATH_CALUDE_min_points_for_top_two_l949_94918


namespace NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l949_94911

theorem largest_inscribed_square_side_length 
  (outer_square_side : ℝ) 
  (triangle_side : ℝ) 
  (inscribed_square_side : ℝ) :
  outer_square_side = 12 →
  triangle_side = 6 * Real.sqrt 2 * (Real.sqrt 3 - 1) →
  2 * inscribed_square_side * Real.sqrt 2 + triangle_side = 12 * Real.sqrt 2 →
  inscribed_square_side = 9 - 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_largest_inscribed_square_side_length_l949_94911


namespace NUMINAMATH_CALUDE_car_speed_problem_l949_94953

theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) 
  (h1 : speed_second_hour = 55)
  (h2 : average_speed = 72.5) : 
  ∃ speed_first_hour : ℝ, 
    speed_first_hour = 90 ∧ 
    (speed_first_hour + speed_second_hour) / 2 = average_speed :=
by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l949_94953


namespace NUMINAMATH_CALUDE_median_longest_side_right_triangle_l949_94906

theorem median_longest_side_right_triangle (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let median := (max a (max b c)) / 2
  median = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_longest_side_right_triangle_l949_94906


namespace NUMINAMATH_CALUDE_jack_mopping_time_l949_94926

/-- Calculates the total time Jack spends mopping and resting given the room sizes and mopping speeds -/
def total_mopping_time (bathroom_size kitchen_size living_room_size : ℕ) 
                       (bathroom_speed kitchen_speed living_room_speed : ℕ) : ℕ :=
  let bathroom_time := (bathroom_size + bathroom_speed - 1) / bathroom_speed
  let kitchen_time := (kitchen_size + kitchen_speed - 1) / kitchen_speed
  let living_room_time := (living_room_size + living_room_speed - 1) / living_room_speed
  let mopping_time := bathroom_time + kitchen_time + living_room_time
  let break_time := 3 * 5 + (bathroom_size + kitchen_size + living_room_size) / 40
  mopping_time + break_time

theorem jack_mopping_time :
  total_mopping_time 24 80 120 8 10 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_jack_mopping_time_l949_94926


namespace NUMINAMATH_CALUDE_expression_value_l949_94969

theorem expression_value (x y z : ℤ) (hx : x = -3) (hy : y = 5) (hz : z = -4) :
  x^2 + y^2 - z^2 + 2*x*y = -12 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l949_94969


namespace NUMINAMATH_CALUDE_division_remainder_l949_94959

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 15 →
  divisor = 3 →
  quotient = 4 →
  remainder = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l949_94959


namespace NUMINAMATH_CALUDE_nines_squared_zeros_l949_94951

theorem nines_squared_zeros (n : ℕ) :
  ∃ m : ℕ, (10^9 - 1)^2 = m * 10^8 ∧ m % 10 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_nines_squared_zeros_l949_94951


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l949_94949

theorem opposite_of_negative_2023 : -((-2023) : ℤ) = (2023 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l949_94949


namespace NUMINAMATH_CALUDE_range_of_a_l949_94931

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 ≤ a) → a ∈ Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l949_94931


namespace NUMINAMATH_CALUDE_initial_sweets_count_prove_initial_sweets_count_l949_94946

theorem initial_sweets_count : ℕ → Prop :=
  fun S => 
    (S / 2 + 4 + 7 = S) → 
    (S = 22)

-- Proof
theorem prove_initial_sweets_count : initial_sweets_count 22 := by
  sorry

end NUMINAMATH_CALUDE_initial_sweets_count_prove_initial_sweets_count_l949_94946


namespace NUMINAMATH_CALUDE_initial_cartons_processed_l949_94945

/-- Proves that the initial number of cartons processed is 400 --/
theorem initial_cartons_processed (num_customers : ℕ) (returned_cartons : ℕ) (total_accepted : ℕ) :
  num_customers = 4 →
  returned_cartons = 60 →
  total_accepted = 160 →
  (num_customers * (total_accepted / num_customers + returned_cartons)) = 400 := by
sorry

end NUMINAMATH_CALUDE_initial_cartons_processed_l949_94945


namespace NUMINAMATH_CALUDE_stock_percentage_l949_94986

/-- The percentage of a stock given certain conditions -/
theorem stock_percentage (income : ℝ) (investment : ℝ) (percentage : ℝ) : 
  income = 1000 →
  investment = 10000 →
  income = (percentage * investment) / 100 →
  percentage = 10 := by
sorry

end NUMINAMATH_CALUDE_stock_percentage_l949_94986


namespace NUMINAMATH_CALUDE_complex_sum_equals_two_l949_94937

theorem complex_sum_equals_two (z : ℂ) (h : z = Complex.exp (2 * Real.pi * I / 5)) :
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_two_l949_94937


namespace NUMINAMATH_CALUDE_fourth_grade_agreement_l949_94900

theorem fourth_grade_agreement (third_grade : ℕ) (total : ℕ) (h1 : third_grade = 154) (h2 : total = 391) :
  total - third_grade = 237 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_agreement_l949_94900


namespace NUMINAMATH_CALUDE_tiles_needed_to_complete_pool_l949_94943

/-- Given a pool with blue and red tiles, calculate the number of additional tiles needed to complete it. -/
theorem tiles_needed_to_complete_pool 
  (blue_tiles : ℕ) 
  (red_tiles : ℕ) 
  (total_required : ℕ) 
  (h1 : blue_tiles = 48)
  (h2 : red_tiles = 32)
  (h3 : total_required = 100) :
  total_required - (blue_tiles + red_tiles) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tiles_needed_to_complete_pool_l949_94943


namespace NUMINAMATH_CALUDE_original_statement_converse_is_false_inverse_is_false_neither_converse_nor_inverse_true_l949_94958

-- Define the properties of triangles
def is_equilateral (t : Triangle) : Prop := sorry
def is_isosceles (t : Triangle) : Prop := sorry

-- The original statement
theorem original_statement (t : Triangle) : is_equilateral t → is_isosceles t := sorry

-- The converse is false
theorem converse_is_false : ¬(∀ t : Triangle, is_isosceles t → is_equilateral t) := sorry

-- The inverse is false
theorem inverse_is_false : ¬(∀ t : Triangle, ¬is_equilateral t → ¬is_isosceles t) := sorry

-- Main theorem: Neither the converse nor the inverse is true
theorem neither_converse_nor_inverse_true : 
  (¬(∀ t : Triangle, is_isosceles t → is_equilateral t)) ∧ 
  (¬(∀ t : Triangle, ¬is_equilateral t → ¬is_isosceles t)) := sorry

end NUMINAMATH_CALUDE_original_statement_converse_is_false_inverse_is_false_neither_converse_nor_inverse_true_l949_94958


namespace NUMINAMATH_CALUDE_candidate_votes_l949_94990

theorem candidate_votes (total_votes : ℕ) (invalid_percentage : ℚ) (candidate_percentage : ℚ) : 
  total_votes = 560000 →
  invalid_percentage = 15 / 100 →
  candidate_percentage = 75 / 100 →
  ⌊(total_votes : ℚ) * (1 - invalid_percentage) * candidate_percentage⌋ = 357000 := by
sorry

end NUMINAMATH_CALUDE_candidate_votes_l949_94990
