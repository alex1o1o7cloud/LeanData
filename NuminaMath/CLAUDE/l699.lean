import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_digits_divisibility_l699_69925

/-- Sum of digits function -/
def sum_of_digits (a : ℕ) : ℕ := sorry

/-- Theorem: If the sum of digits of a equals the sum of digits of 2a, then a is divisible by 9 -/
theorem sum_of_digits_divisibility (a : ℕ) : sum_of_digits a = sum_of_digits (2 * a) → 9 ∣ a := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_divisibility_l699_69925


namespace NUMINAMATH_CALUDE_exists_appropriate_ratio_in_small_interval_l699_69908

-- Define a type for the cutting ratio
def CuttingRatio := {a : ℝ // 0 < a ∧ a < 1}

-- Define a predicate for appropriate cutting ratios
def isAppropriate (a : CuttingRatio) : Prop :=
  ∃ (n : ℕ), ∀ (w : ℝ), w > 0 → ∃ (w1 w2 : ℝ), w1 = w2 ∧ w1 + w2 = w ∧
  ∃ (cuts : List ℝ), cuts.length ≤ n ∧ 
    (∀ c ∈ cuts, c = a.val * w ∨ c = (1 - a.val) * w)

-- State the theorem
theorem exists_appropriate_ratio_in_small_interval :
  ∀ x : ℝ, 0 < x → x < 0.999 →
  ∃ a : CuttingRatio, x < a.val ∧ a.val < x + 0.001 ∧ isAppropriate a :=
sorry

end NUMINAMATH_CALUDE_exists_appropriate_ratio_in_small_interval_l699_69908


namespace NUMINAMATH_CALUDE_jenny_pokemon_cards_l699_69938

theorem jenny_pokemon_cards (J : ℕ) : 
  J + (J + 2) + 3 * (J + 2) = 38 → J = 6 := by
  sorry

end NUMINAMATH_CALUDE_jenny_pokemon_cards_l699_69938


namespace NUMINAMATH_CALUDE_smallest_angle_theorem_l699_69909

/-- The smallest positive angle θ, in degrees, that satisfies the given equation is 50°. -/
theorem smallest_angle_theorem : 
  ∃ θ : ℝ, θ > 0 ∧ θ < 360 ∧ 
  Real.cos (θ * π / 180) = Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - 
                           Real.sin (20 * π / 180) - Real.cos (10 * π / 180) ∧
  θ = 50 ∧ 
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → 
    Real.cos (φ * π / 180) ≠ Real.sin (70 * π / 180) + Real.cos (50 * π / 180) - 
                             Real.sin (20 * π / 180) - Real.cos (10 * π / 180) :=
by sorry


end NUMINAMATH_CALUDE_smallest_angle_theorem_l699_69909


namespace NUMINAMATH_CALUDE_sum_of_five_variables_l699_69974

theorem sum_of_five_variables (a b c d e : ℚ) : 
  (a + 1 = b + 2) ∧ 
  (b + 2 = c + 3) ∧ 
  (c + 3 = d + 4) ∧ 
  (d + 4 = e + 5) ∧ 
  (e + 5 = a + b + c + d + e + 10) → 
  a + b + c + d + e = -35/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_variables_l699_69974


namespace NUMINAMATH_CALUDE_sector_central_angle_l699_69980

theorem sector_central_angle (l : ℝ) (S : ℝ) (h1 : l = 6) (h2 : S = 18) : ∃ (r : ℝ) (α : ℝ), 
  S = (1/2) * l * r ∧ l = r * α ∧ α = 1 :=
sorry

end NUMINAMATH_CALUDE_sector_central_angle_l699_69980


namespace NUMINAMATH_CALUDE_problem_statement_l699_69957

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 3)^2 = 0) :
  (a + b)^2015 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l699_69957


namespace NUMINAMATH_CALUDE_range_of_a_for_two_roots_l699_69916

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then (1/4) * x + 1 else Real.log x

theorem range_of_a_for_two_roots :
  ∃ (a_min a_max : ℝ), a_min = (1/4) ∧ a_max = (1/Real.exp 1) ∧
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ ∧ f x₂ = a * x₂ ∧
              ∀ (x : ℝ), f x = a * x → (x = x₁ ∨ x = x₂)) ↔
              (a_min ≤ a ∧ a < a_max) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_roots_l699_69916


namespace NUMINAMATH_CALUDE_min_sum_cube_relation_l699_69955

theorem min_sum_cube_relation (m n : ℕ+) (h : 108 * m = n ^ 3) :
  ∃ (m₀ n₀ : ℕ+), 108 * m₀ = n₀ ^ 3 ∧ m₀ + n₀ = 8 ∧ ∀ (m' n' : ℕ+), 108 * m' = n' ^ 3 → m' + n' ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_cube_relation_l699_69955


namespace NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_7_9_and_8_l699_69948

theorem unique_number_divisible_by_24_with_cube_root_between_7_9_and_8 :
  ∃! (n : ℕ), n > 0 ∧ 24 ∣ n ∧ (7.9 : ℝ) < n^(1/3) ∧ n^(1/3) < 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_7_9_and_8_l699_69948


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l699_69941

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces of any rectangular prism is 26 -/
theorem rectangular_prism_sum (p : RectangularPrism) :
  num_edges p + num_corners p + num_faces p = 26 := by
  sorry

#check rectangular_prism_sum

end NUMINAMATH_CALUDE_rectangular_prism_sum_l699_69941


namespace NUMINAMATH_CALUDE_bag_weight_problem_l699_69962

theorem bag_weight_problem (sugar_weight salt_weight removed_weight : ℕ) 
  (h1 : sugar_weight = 16)
  (h2 : salt_weight = 30)
  (h3 : removed_weight = 4) :
  sugar_weight + salt_weight - removed_weight = 42 := by
  sorry

end NUMINAMATH_CALUDE_bag_weight_problem_l699_69962


namespace NUMINAMATH_CALUDE_sum_of_y_coordinates_is_negative_six_l699_69918

/-- A circle passes through points (2,0) and (4,0) and is tangent to the line y = x. -/
def CircleThroughPointsAndTangentToLine : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (center.1 - 2)^2 + center.2^2 = radius^2 ∧
    (center.1 - 4)^2 + center.2^2 = radius^2 ∧
    (|center.1 - center.2| / Real.sqrt 2) = radius

/-- The sum of all possible y-coordinates of the center of the circle is -6. -/
theorem sum_of_y_coordinates_is_negative_six
  (h : CircleThroughPointsAndTangentToLine) :
  ∃ (y₁ y₂ : ℝ), y₁ + y₂ = -6 ∧
    ∀ (center : ℝ × ℝ) (radius : ℝ),
      (center.1 - 2)^2 + center.2^2 = radius^2 →
      (center.1 - 4)^2 + center.2^2 = radius^2 →
      (|center.1 - center.2| / Real.sqrt 2) = radius →
      center.2 = y₁ ∨ center.2 = y₂ :=
sorry

end NUMINAMATH_CALUDE_sum_of_y_coordinates_is_negative_six_l699_69918


namespace NUMINAMATH_CALUDE_polynomial_simplification_l699_69992

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 + 5 * q^3 - 7 * q + 8) + (3 - 9 * q^3 + 5 * q^2 - 2 * q) =
  4 * q^4 - 4 * q^3 + 5 * q^2 - 9 * q + 11 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l699_69992


namespace NUMINAMATH_CALUDE_afternoon_bundles_burned_eq_three_l699_69983

/-- Given the number of wood bundles burned in the morning, at the start of the day, and at the end of the day, 
    calculate the number of wood bundles burned in the afternoon. -/
def afternoon_bundles_burned (morning_burned start_of_day end_of_day : ℕ) : ℕ :=
  (start_of_day - end_of_day) - morning_burned

theorem afternoon_bundles_burned_eq_three : 
  afternoon_bundles_burned 4 10 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_bundles_burned_eq_three_l699_69983


namespace NUMINAMATH_CALUDE_celebrity_baby_picture_matching_probability_l699_69997

theorem celebrity_baby_picture_matching_probability :
  ∀ (n : ℕ), n = 5 →
  (1 : ℚ) / (n.factorial : ℚ) = 1 / 120 :=
by sorry

end NUMINAMATH_CALUDE_celebrity_baby_picture_matching_probability_l699_69997


namespace NUMINAMATH_CALUDE_carpenter_logs_needed_l699_69991

/-- A carpenter building a house needs additional logs. -/
theorem carpenter_logs_needed
  (total_woodblocks_needed : ℕ)
  (logs_available : ℕ)
  (woodblocks_per_log : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5) :
  total_woodblocks_needed - logs_available * woodblocks_per_log = 8 * woodblocks_per_log :=
by sorry

end NUMINAMATH_CALUDE_carpenter_logs_needed_l699_69991


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l699_69903

theorem fraction_to_decimal : (4 : ℚ) / 5 = (0.8 : ℚ) := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l699_69903


namespace NUMINAMATH_CALUDE_percentage_given_away_l699_69951

def total_amount : ℝ := 100
def amount_kept : ℝ := 80

theorem percentage_given_away : 
  (total_amount - amount_kept) / total_amount * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_percentage_given_away_l699_69951


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l699_69954

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 5}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(4, -1)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l699_69954


namespace NUMINAMATH_CALUDE_max_m_value_inequality_proof_l699_69945

-- Define the function representing |x-2| - |x+3| ≥ |m+1|
def f (x m : ℝ) : Prop := |x - 2| - |x + 3| ≥ |m + 1|

-- Part I: Maximum value of m
theorem max_m_value : 
  (∃ M : ℝ, (∀ m : ℝ, (∃ x : ℝ, f x m) → m ≤ M) ∧ 
   (∃ x : ℝ, f x M)) → 
  (∃ M : ℝ, M = 4 ∧ (∀ m : ℝ, (∃ x : ℝ, f x m) → m ≤ M) ∧ 
   (∃ x : ℝ, f x M)) :=
sorry

-- Part II: Inequality proof
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + 2*b + c = 4) : 
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_max_m_value_inequality_proof_l699_69945


namespace NUMINAMATH_CALUDE_simplify_expression_l699_69930

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 - 2*b + 4) - 2*b^2 = 9*b^3 - 8*b^2 + 12*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l699_69930


namespace NUMINAMATH_CALUDE_building_height_l699_69985

/-- Given a flagpole and a building casting shadows under similar conditions,
    prove that the height of the building is 20 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 50)
  : (flagpole_height / flagpole_shadow) * building_shadow = 20 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l699_69985


namespace NUMINAMATH_CALUDE_joan_sofa_cost_l699_69923

theorem joan_sofa_cost (joan_cost karl_cost : ℝ) 
  (sum_condition : joan_cost + karl_cost = 600)
  (price_relation : 2 * joan_cost = karl_cost + 90) : 
  joan_cost = 230 := by
sorry

end NUMINAMATH_CALUDE_joan_sofa_cost_l699_69923


namespace NUMINAMATH_CALUDE_backyard_area_l699_69959

/-- Proves that the area of a rectangular backyard with given conditions is 400 square meters -/
theorem backyard_area (length walk_length perimeter : ℝ) 
  (h1 : length * 30 = 1200)
  (h2 : perimeter * 12 = 1200)
  (h3 : perimeter = 2 * length + 2 * (perimeter / 2 - length)) : 
  length * (perimeter / 2 - length) = 400 := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l699_69959


namespace NUMINAMATH_CALUDE_last_digit_of_one_third_to_tenth_l699_69900

theorem last_digit_of_one_third_to_tenth (n : ℕ) : 
  (1 : ℚ) / 3^10 * 10^n % 10 = 5 :=
sorry

end NUMINAMATH_CALUDE_last_digit_of_one_third_to_tenth_l699_69900


namespace NUMINAMATH_CALUDE_flowerpot_problem_l699_69910

/-- Given a row of flowerpots, calculates the number of pots between two specific pots. -/
def pots_between (total : ℕ) (a_from_right : ℕ) (b_from_left : ℕ) : ℕ :=
  b_from_left - (total - a_from_right + 1) - 1

/-- Theorem stating that there are 8 flowerpots between A and B under the given conditions. -/
theorem flowerpot_problem :
  pots_between 33 14 29 = 8 := by
  sorry

end NUMINAMATH_CALUDE_flowerpot_problem_l699_69910


namespace NUMINAMATH_CALUDE_race_total_time_l699_69939

theorem race_total_time (total_runners : Nat) (fast_runners : Nat) (fast_time : Nat) (extra_time : Nat) :
  total_runners = 8 →
  fast_runners = 5 →
  fast_time = 8 →
  extra_time = 2 →
  (fast_runners * fast_time) + ((total_runners - fast_runners) * (fast_time + extra_time)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_race_total_time_l699_69939


namespace NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l699_69942

theorem smallest_prime_twelve_less_than_square : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → ¬(Nat.Prime m ∧ ∃ k : ℕ, m = k^2 - 12)) ∧
  Nat.Prime n ∧ ∃ k : ℕ, n = k^2 - 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_twelve_less_than_square_l699_69942


namespace NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_proof_l699_69970

/-- The smallest positive angle (in degrees) with the same terminal side as -2002° -/
def smallest_positive_equivalent_angle : ℝ := 158

theorem smallest_positive_equivalent_angle_proof :
  ∃ (k : ℤ), smallest_positive_equivalent_angle = -2002 + 360 * k ∧
  0 < smallest_positive_equivalent_angle ∧
  smallest_positive_equivalent_angle < 360 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_equivalent_angle_proof_l699_69970


namespace NUMINAMATH_CALUDE_vector_subtraction_l699_69917

def a : Fin 3 → ℝ := ![5, -3, 2]
def b : Fin 3 → ℝ := ![-2, 4, 1]

theorem vector_subtraction :
  (fun i => a i - 2 * b i) = ![9, -11, 0] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l699_69917


namespace NUMINAMATH_CALUDE_five_double_prime_value_l699_69990

-- Define the prime operation
noncomputable def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem five_double_prime_value : prime (prime 5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_five_double_prime_value_l699_69990


namespace NUMINAMATH_CALUDE_not_all_tv_owners_have_gellert_pass_l699_69960

-- Define the universe of discourse
variable (Person : Type)

-- Define predicates
variable (isTelevisionOwner : Person → Prop)
variable (isPainter : Person → Prop)
variable (hasGellertPass : Person → Prop)

-- State the theorem
theorem not_all_tv_owners_have_gellert_pass
  (h1 : ∃ x, isTelevisionOwner x ∧ ¬isPainter x)
  (h2 : ∀ x, hasGellertPass x ∧ ¬isPainter x → ¬isTelevisionOwner x) :
  ∃ x, isTelevisionOwner x ∧ ¬hasGellertPass x :=
by sorry

end NUMINAMATH_CALUDE_not_all_tv_owners_have_gellert_pass_l699_69960


namespace NUMINAMATH_CALUDE_sphere_surface_area_l699_69987

/-- The surface area of a sphere with radius 14 meters is 4 * π * 14^2 square meters. -/
theorem sphere_surface_area :
  let r : ℝ := 14
  4 * Real.pi * r^2 = 4 * Real.pi * 14^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l699_69987


namespace NUMINAMATH_CALUDE_min_max_f_l699_69946

def f (x : ℝ) : ℝ := x^2 + 4*x + 3

theorem min_max_f (t : ℝ) :
  (t < -3 → (∀ x ∈ Set.Icc t (t+1), f x ≥ t^2 + 6*t + 8) ∧ 
            (∀ x ∈ Set.Icc t (t+1), f x ≤ t^2 + 4*t + 3)) ∧
  (t > -2 → (∀ x ∈ Set.Icc t (t+1), f x ≥ t^2 + 4*t + 3) ∧ 
            (∀ x ∈ Set.Icc t (t+1), f x ≤ t^2 + 6*t + 8)) ∧
  (-2.5 < t ∧ t < -2 → (∀ x ∈ Set.Icc t (t+1), f x ≥ -1) ∧ 
                       (∀ x ∈ Set.Icc t (t+1), f x ≤ t^2 + 6*t + 8)) ∧
  (-3 < t ∧ t ≤ -2.5 → (∀ x ∈ Set.Icc t (t+1), f x ≥ -1) ∧ 
                       (∀ x ∈ Set.Icc t (t+1), f x ≤ t^2 + 4*t + 3)) :=
by sorry

end NUMINAMATH_CALUDE_min_max_f_l699_69946


namespace NUMINAMATH_CALUDE_inequality_solution_range_l699_69982

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, abs (x + 2) - abs (x + 3) > m) → m < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l699_69982


namespace NUMINAMATH_CALUDE_brick_surface_area_l699_69967

/-- The surface area of a rectangular prism -/
def surface_area (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a 8 cm x 6 cm x 2 cm brick is 152 cm² -/
theorem brick_surface_area :
  surface_area 8 6 2 = 152 := by
  sorry

end NUMINAMATH_CALUDE_brick_surface_area_l699_69967


namespace NUMINAMATH_CALUDE_inequalities_not_always_satisfied_l699_69984

theorem inequalities_not_always_satisfied :
  ∃ (a b c x y z : ℝ), 
    x ≤ a ∧ y ≤ b ∧ z ≤ c ∧
    ((x^2 * y + y^2 * z + z^2 * x ≥ a^2 * b + b^2 * c + c^2 * a) ∨
     (x^3 + y^3 + z^3 ≥ a^3 + b^3 + c^3)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_always_satisfied_l699_69984


namespace NUMINAMATH_CALUDE_third_circle_radius_l699_69926

/-- Given two internally tangent circles with radii R and r (R > r),
    the radius x of a third circle tangent to both circles and their common diameter
    is given by x = 4Rr / (R + r). -/
theorem third_circle_radius (R r : ℝ) (h : R > r) (h_pos_R : R > 0) (h_pos_r : r > 0) :
  ∃ x : ℝ, x > 0 ∧ x = (4 * R * r) / (R + r) ∧
    (∀ y : ℝ, y > 0 → y ≠ x →
      ¬(∃ p q : ℝ × ℝ,
        (p.1 - q.1)^2 + (p.2 - q.2)^2 = (R - r)^2 ∧
        (p.1 - 0)^2 + (p.2 - 0)^2 = R^2 ∧
        (q.1 - 0)^2 + (q.2 - 0)^2 = r^2 ∧
        ((p.1 + q.1)/2 - 0)^2 + ((p.2 + q.2)/2 - y)^2 = y^2)) :=
by sorry

end NUMINAMATH_CALUDE_third_circle_radius_l699_69926


namespace NUMINAMATH_CALUDE_rectangle_division_distinctness_l699_69989

theorem rectangle_division_distinctness (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  a * c = b * d →
  a + c = b + d →
  ¬(a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_division_distinctness_l699_69989


namespace NUMINAMATH_CALUDE_gcd_462_330_l699_69961

theorem gcd_462_330 : Nat.gcd 462 330 = 66 := by
  sorry

end NUMINAMATH_CALUDE_gcd_462_330_l699_69961


namespace NUMINAMATH_CALUDE_second_day_speed_l699_69968

/-- Proves that given the climbing conditions, the speed on the second day is 4 km/h -/
theorem second_day_speed (total_time : ℝ) (speed_difference : ℝ) (time_difference : ℝ) (total_distance : ℝ)
  (h1 : total_time = 14)
  (h2 : speed_difference = 0.5)
  (h3 : time_difference = 2)
  (h4 : total_distance = 52) :
  let first_day_time := (total_time + time_difference) / 2
  let second_day_time := total_time - first_day_time
  let first_day_speed := (total_distance - speed_difference * second_day_time) / total_time
  let second_day_speed := first_day_speed + speed_difference
  second_day_speed = 4 := by sorry

end NUMINAMATH_CALUDE_second_day_speed_l699_69968


namespace NUMINAMATH_CALUDE_negation_of_existence_l699_69936

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 - a*x + 1 < 0) ↔ (∀ x : ℝ, x^2 - a*x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l699_69936


namespace NUMINAMATH_CALUDE_kayla_waiting_time_l699_69928

/-- The number of years Kayla needs to wait before reaching the minimum driving age -/
def years_until_driving (minimum_age : ℕ) (kimiko_age : ℕ) : ℕ :=
  minimum_age - kimiko_age / 2

/-- Proof that Kayla needs to wait 5 years before she can start driving -/
theorem kayla_waiting_time :
  years_until_driving 18 26 = 5 := by
  sorry

end NUMINAMATH_CALUDE_kayla_waiting_time_l699_69928


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l699_69958

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l699_69958


namespace NUMINAMATH_CALUDE_ryan_spanish_hours_l699_69932

/-- Ryan's daily study hours -/
structure StudyHours where
  english : ℕ
  chinese : ℕ
  spanish : ℕ

/-- Ryan's study schedule satisfies the given conditions -/
def validSchedule (h : StudyHours) : Prop :=
  h.english = 7 ∧ h.chinese = 2 ∧ h.english = h.spanish + 3

theorem ryan_spanish_hours (h : StudyHours) (hvalid : validSchedule h) : h.spanish = 4 := by
  sorry

end NUMINAMATH_CALUDE_ryan_spanish_hours_l699_69932


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l699_69956

-- Part 1
theorem calculate_expression : (-2)^2 + (Real.sqrt 3 - Real.pi)^0 + abs (1 - Real.sqrt 3) = 4 + Real.sqrt 3 := by
  sorry

-- Part 2
theorem solve_system_of_equations :
  ∃ x y : ℝ, 2*x + y = 1 ∧ x - 2*y = 3 ∧ x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l699_69956


namespace NUMINAMATH_CALUDE_investment_equalizes_profits_l699_69911

/-- The investment amount in yuan that equalizes profits from two selling methods -/
def investment : ℝ := 20000

/-- The profit rate when selling at the beginning of the month -/
def early_profit_rate : ℝ := 0.15

/-- The profit rate for reinvestment -/
def reinvestment_profit_rate : ℝ := 0.10

/-- The profit rate when selling at the end of the month -/
def late_profit_rate : ℝ := 0.30

/-- The storage fee in yuan -/
def storage_fee : ℝ := 700

/-- Theorem stating that the investment amount equalizes profits from both selling methods -/
theorem investment_equalizes_profits :
  investment * (1 + early_profit_rate) * (1 + reinvestment_profit_rate) =
  investment * (1 + late_profit_rate) - storage_fee := by
  sorry

#eval investment -- Should output 20000

end NUMINAMATH_CALUDE_investment_equalizes_profits_l699_69911


namespace NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_is_ten_l699_69920

theorem expansion_terms_count : ℕ :=
  let expression := (fun (x y : ℝ) => ((x + 5*y)^3 * (x - 5*y)^3)^3)
  let simplified := (fun (x y : ℝ) => (x^2 - 25*y^2)^9)
  let distinct_terms_count := 10
  distinct_terms_count

#check expansion_terms_count

theorem expansion_terms_count_is_ten : expansion_terms_count = 10 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_expansion_terms_count_is_ten_l699_69920


namespace NUMINAMATH_CALUDE_garden_length_l699_69943

/-- Proves that a rectangular garden with length twice its width and 300 yards of fencing has a length of 100 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- length is twice the width
  2 * length + 2 * width = 300 →  -- 300 yards of fencing encloses the garden
  length = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l699_69943


namespace NUMINAMATH_CALUDE_photographer_arrangement_exists_l699_69934

-- Define a type for photographers
def Photographer := Fin 6

-- Define a type for positions in the plane
def Position := ℝ × ℝ

-- Define a function to check if a photographer is between two others
def isBetween (p₁ p₂ p₃ : Position) : Prop := sorry

-- Define a function to check if two photographers can see each other
def canSee (positions : Photographer → Position) (p₁ p₂ : Photographer) : Prop :=
  ∀ p₃, p₃ ≠ p₁ ∧ p₃ ≠ p₂ → ¬ isBetween (positions p₁) (positions p₃) (positions p₂)

-- State the theorem
theorem photographer_arrangement_exists :
  ∃ (positions : Photographer → Position),
    ∀ p, (∃! (s : Finset Photographer), s.card = 4 ∧ ∀ p' ∈ s, canSee positions p p') :=
sorry

end NUMINAMATH_CALUDE_photographer_arrangement_exists_l699_69934


namespace NUMINAMATH_CALUDE_line_through_circle_center_l699_69949

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop := 3 * x + y + a = 0

/-- The theorem stating that if the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0, then a = 1 -/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l699_69949


namespace NUMINAMATH_CALUDE_greatest_prime_base_angle_l699_69947

-- Define the triangle and its properties
def IsoscelesTriangle (a b c : ℕ) : Prop :=
  a = b ∧ a + b + c = 180 ∧ c = 60 ∧ a < 90 ∧ Nat.Prime a

-- State the theorem
theorem greatest_prime_base_angle :
  ∃ (a : ℕ), IsoscelesTriangle a a 60 ∧
  ∀ (x : ℕ), IsoscelesTriangle x x 60 → x ≤ a :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_base_angle_l699_69947


namespace NUMINAMATH_CALUDE_equation_solution_l699_69904

-- Define the equation
def equation (y : ℝ) : Prop :=
  (15 : ℝ)^(3*2) * (7^4 - 3*2) / 5670 = y

-- State the theorem
theorem equation_solution : 
  ∃ y : ℝ, equation y ∧ abs (y - 4812498.20123) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l699_69904


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l699_69935

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_for_inequality_l699_69935


namespace NUMINAMATH_CALUDE_largest_valid_coloring_l699_69950

/-- A coloring of an n × n grid with two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle in the grid has all four corners the same color. -/
def hasMonochromaticRectangle (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i < k ∧ j < l ∧
    c i j = c i l ∧ c i l = c k j ∧ c k j = c k l

/-- The largest n for which a valid coloring exists. -/
def largestValidN : ℕ := 4

theorem largest_valid_coloring :
  (∃ (c : Coloring largestValidN), ¬hasMonochromaticRectangle c) ∧
  (∀ (m : ℕ), m > largestValidN →
    ∀ (c : Coloring m), hasMonochromaticRectangle c) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_coloring_l699_69950


namespace NUMINAMATH_CALUDE_beach_visitors_beach_visitors_proof_l699_69978

theorem beach_visitors (initial_people : ℕ) (people_left : ℕ) (total_if_stayed : ℕ) : ℕ :=
  let total_before_leaving := total_if_stayed + people_left
  total_before_leaving - initial_people

#check beach_visitors 3 40 63 = 100

/- Proof
theorem beach_visitors_proof :
  beach_visitors 3 40 63 = 100 := by
  sorry
-/

end NUMINAMATH_CALUDE_beach_visitors_beach_visitors_proof_l699_69978


namespace NUMINAMATH_CALUDE_carpet_shaded_area_carpet_specific_shaded_area_l699_69969

/-- Calculates the total shaded area of a rectangular carpet with specific dimensions and shaded areas. -/
theorem carpet_shaded_area (carpet_length carpet_width : ℝ) 
  (num_small_squares : ℕ) (ratio_long_to_R ratio_R_to_S : ℝ) : ℝ :=
  let R := carpet_length / ratio_long_to_R
  let S := R / ratio_R_to_S
  let area_R := R * R
  let area_S := S * S
  let total_area := area_R + (num_small_squares : ℝ) * area_S
  total_area

/-- Proves that the total shaded area of the carpet with given specifications is 141.75 square feet. -/
theorem carpet_specific_shaded_area : 
  carpet_shaded_area 18 12 12 2 4 = 141.75 := by
  sorry

end NUMINAMATH_CALUDE_carpet_shaded_area_carpet_specific_shaded_area_l699_69969


namespace NUMINAMATH_CALUDE_melanie_marbles_l699_69995

def sandy_marbles : ℕ := 56 * 12

theorem melanie_marbles : ∃ m : ℕ, m * 8 = sandy_marbles ∧ m = 84 := by
  sorry

end NUMINAMATH_CALUDE_melanie_marbles_l699_69995


namespace NUMINAMATH_CALUDE_max_sum_of_three_numbers_l699_69976

theorem max_sum_of_three_numbers (a b c : ℕ) : 
  a + b = 1014 → c - b = 497 → a > b → (∀ S : ℕ, S = a + b + c → S ≤ 2017) ∧ (∃ S : ℕ, S = a + b + c ∧ S = 2017) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_three_numbers_l699_69976


namespace NUMINAMATH_CALUDE_remainder_55_pow_55_plus_15_mod_8_l699_69924

theorem remainder_55_pow_55_plus_15_mod_8 : (55^55 + 15) % 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_55_pow_55_plus_15_mod_8_l699_69924


namespace NUMINAMATH_CALUDE_total_turtles_is_100_l699_69963

/-- The number of turtles Martha received -/
def martha_turtles : ℕ := 40

/-- The difference between Marion's and Martha's turtles -/
def difference : ℕ := 20

/-- The number of turtles Marion received -/
def marion_turtles : ℕ := martha_turtles + difference

/-- The total number of turtles received by Martha and Marion -/
def total_turtles : ℕ := martha_turtles + marion_turtles

theorem total_turtles_is_100 : total_turtles = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_turtles_is_100_l699_69963


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l699_69919

theorem right_triangle_acute_angle_theorem :
  ∀ (x y : ℝ),
  x > 0 ∧ y > 0 →
  x + y = 90 →
  y = 5 * x →
  y = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_theorem_l699_69919


namespace NUMINAMATH_CALUDE_reflect_center_of_circle_l699_69975

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

theorem reflect_center_of_circle : reflect_about_y_eq_neg_x (3, -7) = (7, -3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_center_of_circle_l699_69975


namespace NUMINAMATH_CALUDE_find_y_l699_69988

-- Define the function F
def F (a b c d : ℕ) : ℕ := a^b + c * d

-- State the theorem
theorem find_y : ∃ y : ℕ, F 3 y 5 15 = 490 ∧ ∀ z : ℕ, F 3 z 5 15 = 490 → y = z :=
  sorry

end NUMINAMATH_CALUDE_find_y_l699_69988


namespace NUMINAMATH_CALUDE_concert_attendance_l699_69913

/-- The number of students from School A who went to the concert -/
def school_a_students : ℕ := 15 * 30

/-- The number of students from School B who went to the concert -/
def school_b_students : ℕ := 18 * 7 + 5 * 6

/-- The number of students from School C who went to the concert -/
def school_c_students : ℕ := 13 * 33 + 10 * 4

/-- The total number of students who went to the concert -/
def total_students : ℕ := school_a_students + school_b_students + school_c_students

theorem concert_attendance : total_students = 1075 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l699_69913


namespace NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l699_69979

theorem negation_of_divisible_by_two_is_even :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, 2 ∣ n ∧ ¬Even n) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_divisible_by_two_is_even_l699_69979


namespace NUMINAMATH_CALUDE_bananas_multiple_of_three_l699_69901

/-- Represents the number of fruit baskets that can be made -/
def num_baskets : ℕ := 3

/-- Represents the number of oranges Peter has -/
def oranges : ℕ := 18

/-- Represents the number of pears Peter has -/
def pears : ℕ := 27

/-- Represents the number of bananas Peter has -/
def bananas : ℕ := sorry

/-- Theorem stating that the number of bananas must be a multiple of 3 -/
theorem bananas_multiple_of_three :
  ∃ k : ℕ, bananas = 3 * k ∧
  oranges % num_baskets = 0 ∧
  pears % num_baskets = 0 ∧
  bananas % num_baskets = 0 :=
sorry

end NUMINAMATH_CALUDE_bananas_multiple_of_three_l699_69901


namespace NUMINAMATH_CALUDE_coefficient_a3b3_is_1400_l699_69998

/-- The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 -/
def coefficient_a3b3 : ℕ :=
  (Nat.choose 6 3) * (Nat.choose 8 4)

/-- Theorem: The coefficient of a^3b^3 in (a+b)^6(c+1/c)^8 is 1400 -/
theorem coefficient_a3b3_is_1400 : coefficient_a3b3 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_a3b3_is_1400_l699_69998


namespace NUMINAMATH_CALUDE_calculate_boxes_l699_69996

/-- Given the number of blocks and blocks per box, calculate the number of boxes -/
theorem calculate_boxes (total_blocks : ℕ) (blocks_per_box : ℕ) (h : blocks_per_box > 0) :
  total_blocks / blocks_per_box = total_blocks / blocks_per_box :=
by sorry

/-- George's specific case -/
def george_boxes : ℕ :=
  let total_blocks : ℕ := 12
  let blocks_per_box : ℕ := 6
  total_blocks / blocks_per_box

#eval george_boxes

end NUMINAMATH_CALUDE_calculate_boxes_l699_69996


namespace NUMINAMATH_CALUDE_gmat_scores_l699_69972

theorem gmat_scores (u v w : ℝ) 
  (h_order : u > v ∧ v > w)
  (h_avg : u - w = (u + v + w) / 3)
  (h_diff : u - v = 2 * (v - w)) :
  v / u = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_gmat_scores_l699_69972


namespace NUMINAMATH_CALUDE_line_intersects_circle_l699_69906

theorem line_intersects_circle (a : ℝ) : ∃ (x y : ℝ), 
  y = a * x - a + 1 ∧ x^2 + y^2 = 8 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_line_intersects_circle_l699_69906


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l699_69981

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  interval : Nat

/-- Checks if a student number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = k * s.interval + (n % s.interval)

/-- Main theorem: If students 6, 32, and 45 are in a systematic sample of 4 from 52 students,
    then student 19 must also be in the sample -/
theorem systematic_sampling_theorem (s : SystematicSample) 
  (h1 : s.total_students = 52)
  (h2 : s.sample_size = 4)
  (h3 : in_sample s 6)
  (h4 : in_sample s 32)
  (h5 : in_sample s 45) :
  in_sample s 19 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l699_69981


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l699_69929

/-- Given an arithmetic sequence with first three terms a-1, a+1, 2a+3, 
    prove that its general formula is a_n = 2n - 3 -/
theorem arithmetic_sequence_formula 
  (a : ℝ) 
  (seq : ℕ → ℝ) 
  (h1 : seq 1 = a - 1) 
  (h2 : seq 2 = a + 1) 
  (h3 : seq 3 = 2*a + 3) 
  (h_arithmetic : ∀ n : ℕ, seq (n + 1) - seq n = seq (n + 2) - seq (n + 1)) :
  ∀ n : ℕ, seq n = 2*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l699_69929


namespace NUMINAMATH_CALUDE_regular_polygon_interior_twice_exterior_has_six_sides_l699_69902

/-- A regular polygon where the sum of interior angles is twice the sum of exterior angles has 6 sides. -/
theorem regular_polygon_interior_twice_exterior_has_six_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 2) * 180 = 2 * 360 →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_twice_exterior_has_six_sides_l699_69902


namespace NUMINAMATH_CALUDE_sim_tetrahedron_volume_l699_69912

/-- A tetrahedron with similar but not all equal triangular faces -/
structure SimTetrahedron where
  /-- The faces are similar triangles -/
  similar_faces : Bool
  /-- Not all faces are equal -/
  not_all_equal : Bool
  /-- Any two faces share at least one pair of equal edges, not counting the common edge -/
  shared_equal_edges : Bool
  /-- Two edges in one face have lengths 3 and 5 -/
  edge_lengths : (ℝ × ℝ)

/-- The volume of a SimTetrahedron is either (55 * √6) / 18 or (11 * √10) / 10 -/
theorem sim_tetrahedron_volume (t : SimTetrahedron) : 
  t.similar_faces ∧ t.not_all_equal ∧ t.shared_equal_edges ∧ t.edge_lengths = (3, 5) →
  (∃ v : ℝ, v = (55 * Real.sqrt 6) / 18 ∨ v = (11 * Real.sqrt 10) / 10) :=
by sorry

end NUMINAMATH_CALUDE_sim_tetrahedron_volume_l699_69912


namespace NUMINAMATH_CALUDE_fisherman_daily_earnings_l699_69966

/-- Calculates the daily earnings of a fisherman based on their catch and fish prices -/
theorem fisherman_daily_earnings (red_snapper_count : ℕ) (tuna_count : ℕ) (red_snapper_price : ℕ) (tuna_price : ℕ) : 
  red_snapper_count = 8 → 
  tuna_count = 14 → 
  red_snapper_price = 3 → 
  tuna_price = 2 → 
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 := by
sorry

end NUMINAMATH_CALUDE_fisherman_daily_earnings_l699_69966


namespace NUMINAMATH_CALUDE_complement_of_union_is_five_l699_69927

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 3}

-- Define set B
def B : Set Nat := {2, 4}

-- Theorem statement
theorem complement_of_union_is_five :
  (U \ (A ∪ B)) = {5} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_is_five_l699_69927


namespace NUMINAMATH_CALUDE_joan_balloons_l699_69952

theorem joan_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (remaining_balloons : ℕ) : 
  initial_balloons = 9 → lost_balloons = 2 → remaining_balloons = initial_balloons - lost_balloons →
  remaining_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l699_69952


namespace NUMINAMATH_CALUDE_figure_to_square_possible_l699_69921

/-- Represents a geometric figure on a grid paper -/
structure GridFigure where
  -- Add necessary fields to represent the figure

/-- Represents a part of the figure after cutting -/
structure FigurePart where
  -- Add necessary fields to represent a part

/-- Represents a square -/
structure Square where
  -- Add necessary fields to represent a square

/-- Function to check if a list of parts can be reassembled into a square -/
def can_form_square (parts : List FigurePart) : Bool :=
  sorry

/-- Function to check if all parts are triangles -/
def all_triangles (parts : List FigurePart) : Bool :=
  sorry

/-- Theorem stating that the figure can be cut and reassembled into a square under given conditions -/
theorem figure_to_square_possible (fig : GridFigure) : 
  (∃ (parts : List FigurePart), parts.length ≤ 4 ∧ can_form_square parts) ∧
  (∃ (parts : List FigurePart), parts.length ≤ 5 ∧ all_triangles parts ∧ can_form_square parts) :=
by
  sorry

end NUMINAMATH_CALUDE_figure_to_square_possible_l699_69921


namespace NUMINAMATH_CALUDE_expression_evaluation_l699_69999

theorem expression_evaluation : -1^4 - (1/6) * (|(-2)| - (-3)^2) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l699_69999


namespace NUMINAMATH_CALUDE_triangle_problem_l699_69933

/-- Given a triangle ABC with vertex A at (5,1), altitude CH from AB with equation x-2y-5=0,
    and median BM from AC with equation 2x-y-1=0, prove the coordinates of B and the equation
    of the perpendicular bisector of BC. -/
theorem triangle_problem (B : ℝ × ℝ) (perpBisectorBC : ℝ → ℝ → ℝ) : 
  let A : ℝ × ℝ := (5, 1)
  let altitude_CH (x y : ℝ) := x - 2*y - 5 = 0
  let median_BM (x y : ℝ) := 2*x - y - 1 = 0
  B = (3, 5) ∧ 
  (∀ x y, perpBisectorBC x y = 0 ↔ 21*x + 24*y + 43 = 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l699_69933


namespace NUMINAMATH_CALUDE_vasya_meeting_time_l699_69986

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the difference in minutes between two times -/
def timeDifference (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem vasya_meeting_time :
  let normalArrival : Time := ⟨18, 0, by norm_num, by norm_num⟩
  let earlyArrival : Time := ⟨17, 0, by norm_num, by norm_num⟩
  let meetingTime : Time := ⟨17, 50, by norm_num, by norm_num⟩
  let normalHomeArrival : Time := ⟨19, 0, by norm_num, by norm_num⟩  -- Assuming normal home arrival is at 19:00
  let earlyHomeArrival : Time := ⟨18, 40, by norm_num, by norm_num⟩  -- 20 minutes earlier than normal

  -- Vasya arrives 1 hour early
  timeDifference normalArrival earlyArrival = 60 →
  -- They arrive home 20 minutes earlier than usual
  timeDifference normalHomeArrival earlyHomeArrival = 20 →
  -- The meeting time is 10 minutes before the normal arrival time
  timeDifference normalArrival meetingTime = 10 →
  -- The meeting time is 50 minutes after the early arrival time
  timeDifference meetingTime earlyArrival = 50 →
  meetingTime = ⟨17, 50, by norm_num, by norm_num⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_vasya_meeting_time_l699_69986


namespace NUMINAMATH_CALUDE_percent_relation_l699_69993

theorem percent_relation (x y z : ℝ) (h1 : 0.45 * z = 0.39 * y) (h2 : z = 0.65 * x) :
  y = 0.75 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l699_69993


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_77_over_6_l699_69994

/-- Represents the arrangement of three squares -/
structure SquareArrangement where
  small_side : ℝ
  medium_side : ℝ
  large_side : ℝ
  coplanar : Prop
  side_by_side : Prop

/-- Calculates the area of the quadrilateral formed in the square arrangement -/
def quadrilateral_area (arr : SquareArrangement) : ℝ :=
  sorry

/-- The main theorem stating that the quadrilateral area is 77/6 -/
theorem quadrilateral_area_is_77_over_6 (arr : SquareArrangement) 
  (h1 : arr.small_side = 3)
  (h2 : arr.medium_side = 5)
  (h3 : arr.large_side = 7)
  (h4 : arr.coplanar)
  (h5 : arr.side_by_side) :
  quadrilateral_area arr = 77 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_77_over_6_l699_69994


namespace NUMINAMATH_CALUDE_triangle_abc_equilateral_l699_69940

theorem triangle_abc_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * a = b + c) 
  (h2 : Real.sin A ^ 2 = Real.sin B * Real.sin C) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_equilateral_l699_69940


namespace NUMINAMATH_CALUDE_special_operation_l699_69964

theorem special_operation (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum : a + b = 12) (product : a * b = 35) : 
  (1 : ℚ) / a + (1 : ℚ) / b = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_special_operation_l699_69964


namespace NUMINAMATH_CALUDE_jackie_free_time_l699_69914

/-- Calculates the free time given the time spent on various activities and the total time available. -/
def free_time (work_hours exercise_hours sleep_hours total_hours : ℕ) : ℕ :=
  total_hours - (work_hours + exercise_hours + sleep_hours)

/-- Proves that Jackie has 5 hours of free time given her daily schedule. -/
theorem jackie_free_time :
  let work_hours : ℕ := 8
  let exercise_hours : ℕ := 3
  let sleep_hours : ℕ := 8
  let total_hours : ℕ := 24
  free_time work_hours exercise_hours sleep_hours total_hours = 5 := by
  sorry

end NUMINAMATH_CALUDE_jackie_free_time_l699_69914


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l699_69915

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 3 * x + 2 > 0) ↔ (b < x ∧ x < 1)) →
  (a = -5 ∧ b = -2/5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l699_69915


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l699_69977

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 4 * x^2 - k * x + 6 = 0 ∧ x = 2) → 
  (k = 11 ∧ ∃ y : ℝ, 4 * y^2 - k * y + 6 = 0 ∧ y = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l699_69977


namespace NUMINAMATH_CALUDE_normal_distribution_mean_l699_69944

/-- 
Given a normal distribution with standard deviation σ,
if the value that is exactly k standard deviations less than the mean is x,
then the arithmetic mean μ of the distribution is x + k * σ.
-/
theorem normal_distribution_mean 
  (σ : ℝ) (k : ℝ) (x : ℝ) (μ : ℝ) 
  (hσ : σ = 1.5) 
  (hk : k = 2) 
  (hx : x = 11.5) 
  (h : x = μ - k * σ) : 
  μ = 14.5 := by
  sorry

#check normal_distribution_mean

end NUMINAMATH_CALUDE_normal_distribution_mean_l699_69944


namespace NUMINAMATH_CALUDE_exists_abc_factorial_sum_l699_69971

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem exists_abc_factorial_sum :
  ∃ (a b c : ℕ), 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧
    100 * a + 10 * b + c = factorial a + factorial b + factorial c :=
by sorry

end NUMINAMATH_CALUDE_exists_abc_factorial_sum_l699_69971


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_eight_l699_69905

theorem ceiling_neg_sqrt_eight : ⌈-Real.sqrt 8⌉ = -2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_eight_l699_69905


namespace NUMINAMATH_CALUDE_min_value_on_interval_l699_69953

def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem min_value_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ 
  (∀ y ∈ Set.Icc 0 3, f y ≥ f x) ∧
  f x = -15 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l699_69953


namespace NUMINAMATH_CALUDE_volume_of_midpoint_set_l699_69973

/-- A regular tetrahedron -/
structure RegularTetrahedron :=
  (edge_length : ℝ)

/-- The set of midpoints of segments whose endpoints belong to different tetrahedra -/
def midpoint_set (t1 t2 : RegularTetrahedron) : Set (Fin 3 → ℝ) :=
  sorry

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Central symmetry transformation -/
def central_symmetry (t : RegularTetrahedron) : RegularTetrahedron :=
  sorry

theorem volume_of_midpoint_set :
  ∀ t : RegularTetrahedron,
  t.edge_length = Real.sqrt 2 →
  volume (midpoint_set t (central_symmetry t)) = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_midpoint_set_l699_69973


namespace NUMINAMATH_CALUDE_inscribed_triangle_regular_polygon_sides_l699_69965

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle :=
  (A B C : ℝ × ℝ)  -- Vertices of the triangle
  (center : ℝ × ℝ)  -- Center of the circle
  (radius : ℝ)  -- Radius of the circle

/-- Calculates the angle at a vertex of a triangle -/
def angle (t : InscribedTriangle) (v : Fin 3) : ℝ :=
  sorry  -- Definition of angle calculation

/-- Represents a regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (n : ℕ)  -- Number of sides
  (center : ℝ × ℝ)  -- Center of the circle
  (radius : ℝ)  -- Radius of the circle

/-- Checks if two points are adjacent vertices of a regular polygon -/
def areAdjacentVertices (p : RegularPolygon) (v1 v2 : ℝ × ℝ) : Prop :=
  sorry  -- Definition of adjacency check

theorem inscribed_triangle_regular_polygon_sides 
  (t : InscribedTriangle) 
  (p : RegularPolygon) 
  (h1 : angle t 1 = angle t 2)  -- ∠B = ∠C
  (h2 : angle t 1 = 3 * angle t 0)  -- ∠B = 3∠A
  (h3 : t.center = p.center ∧ t.radius = p.radius)  -- Same circle
  (h4 : areAdjacentVertices p t.B t.C)  -- B and C are adjacent vertices
  : p.n = 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_regular_polygon_sides_l699_69965


namespace NUMINAMATH_CALUDE_right_triangle_area_l699_69922

/-- Right triangle XYZ with altitude foot W -/
structure RightTriangle where
  -- Point X
  X : ℝ × ℝ
  -- Point Y (right angle)
  Y : ℝ × ℝ
  -- Point Z
  Z : ℝ × ℝ
  -- Point W (foot of altitude from Y to XZ)
  W : ℝ × ℝ
  -- XW length
  xw_length : ℝ
  -- WZ length
  wz_length : ℝ
  -- Constraint: XYZ is a right triangle with right angle at Y
  right_angle_at_Y : (X.1 - Y.1) * (Z.1 - Y.1) + (X.2 - Y.2) * (Z.2 - Y.2) = 0
  -- Constraint: W is on XZ
  w_on_xz : ∃ t : ℝ, W = (t * X.1 + (1 - t) * Z.1, t * X.2 + (1 - t) * Z.2)
  -- Constraint: YW is perpendicular to XZ
  yw_perpendicular_xz : (Y.1 - W.1) * (X.1 - Z.1) + (Y.2 - W.2) * (X.2 - Z.2) = 0
  -- Constraint: XW length is 5
  xw_is_5 : xw_length = 5
  -- Constraint: WZ length is 7
  wz_is_7 : wz_length = 7

/-- The area of the right triangle XYZ -/
def triangleArea (t : RightTriangle) : ℝ := sorry

/-- Theorem: The area of the right triangle XYZ is 6√35 -/
theorem right_triangle_area (t : RightTriangle) : triangleArea t = 6 * Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l699_69922


namespace NUMINAMATH_CALUDE_alicia_remaining_masks_l699_69931

/-- The number of sets of masks Alicia had initially -/
def initial_sets : ℕ := 90

/-- The number of sets of masks Alicia gave away -/
def given_away : ℕ := 51

/-- The number of sets of masks left in Alicia's collection -/
def remaining_sets : ℕ := initial_sets - given_away

theorem alicia_remaining_masks : remaining_sets = 39 := by
  sorry

end NUMINAMATH_CALUDE_alicia_remaining_masks_l699_69931


namespace NUMINAMATH_CALUDE_angle_complement_quadrant_l699_69937

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_fourth_quadrant (α : Real) : Prop :=
  270 < α ∧ α < 360

/-- An angle is in the third quadrant if it's between 180° and 270° (exclusive) -/
def is_third_quadrant (α : Real) : Prop :=
  180 < α ∧ α < 270

theorem angle_complement_quadrant (α : Real) :
  is_fourth_quadrant α → is_third_quadrant (180 - α) := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_quadrant_l699_69937


namespace NUMINAMATH_CALUDE_veggies_per_day_l699_69907

/-- The number of servings of veggies eaten in a week -/
def weekly_servings : ℕ := 21

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of servings of veggies eaten per day -/
def daily_servings : ℕ := weekly_servings / days_in_week

theorem veggies_per_day : daily_servings = 3 := by
  sorry

end NUMINAMATH_CALUDE_veggies_per_day_l699_69907
