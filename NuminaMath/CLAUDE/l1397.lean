import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1397_139731

theorem inscribed_cube_volume (large_cube_edge : ℝ) (sphere_diameter : ℝ) (small_cube_edge : ℝ) :
  large_cube_edge = 12 →
  sphere_diameter = large_cube_edge →
  sphere_diameter = small_cube_edge * Real.sqrt 3 →
  small_cube_edge^3 = 192 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1397_139731


namespace NUMINAMATH_CALUDE_current_rate_calculation_l1397_139791

/-- Given a boat with speed in still water and its downstream travel distance and time,
    calculate the rate of the current. -/
theorem current_rate_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (travel_time : ℝ) :
  boat_speed = 20 →
  downstream_distance = 6.25 →
  travel_time = 0.25 →
  ∃ (current_rate : ℝ),
    current_rate = 5 ∧
    downstream_distance = (boat_speed + current_rate) * travel_time :=
by
  sorry


end NUMINAMATH_CALUDE_current_rate_calculation_l1397_139791


namespace NUMINAMATH_CALUDE_log_identity_l1397_139781

-- Define the logarithm base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem log_identity : (log5 (3 * log5 25))^2 = (1 + log5 1.2)^2 := by sorry

end NUMINAMATH_CALUDE_log_identity_l1397_139781


namespace NUMINAMATH_CALUDE_ratio_to_ten_l1397_139783

theorem ratio_to_ten : ∃ x : ℚ, (15 : ℚ) / 1 = x / 10 ∧ x = 150 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_ten_l1397_139783


namespace NUMINAMATH_CALUDE_max_cars_quotient_l1397_139736

/-- Represents the maximum number of cars that can pass a point on the highway in one hour -/
def M : ℕ :=
  -- Definition to be proved
  2000

/-- The length of each car in meters -/
def car_length : ℝ := 5

/-- Theorem stating that M divided by 10 equals 200 -/
theorem max_cars_quotient :
  M / 10 = 200 := by sorry

end NUMINAMATH_CALUDE_max_cars_quotient_l1397_139736


namespace NUMINAMATH_CALUDE_mouse_jump_difference_l1397_139770

/-- Proves that the mouse jumped 12 inches less than the frog in the jumping contest. -/
theorem mouse_jump_difference (grasshopper_jump : ℕ) (grasshopper_frog_diff : ℕ) (mouse_jump : ℕ)
  (h1 : grasshopper_jump = 39)
  (h2 : grasshopper_frog_diff = 19)
  (h3 : mouse_jump = 8) :
  grasshopper_jump - grasshopper_frog_diff - mouse_jump = 12 := by
  sorry

end NUMINAMATH_CALUDE_mouse_jump_difference_l1397_139770


namespace NUMINAMATH_CALUDE_weight_of_grapes_l1397_139767

/-- Given the weights of fruits ordered by Tommy, prove the weight of grapes. -/
theorem weight_of_grapes (total weight_apples weight_oranges weight_strawberries : ℕ) 
  (h_total : total = 10)
  (h_apples : weight_apples = 3)
  (h_oranges : weight_oranges = 1)
  (h_strawberries : weight_strawberries = 3) :
  total - (weight_apples + weight_oranges + weight_strawberries) = 3 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_grapes_l1397_139767


namespace NUMINAMATH_CALUDE_gold_coins_per_hour_l1397_139711

def scuba_diving_hours : ℕ := 8
def treasure_chest_coins : ℕ := 100
def smaller_bags_count : ℕ := 2

def smaller_bag_coins : ℕ := treasure_chest_coins / 2

def total_coins : ℕ := treasure_chest_coins + smaller_bags_count * smaller_bag_coins

theorem gold_coins_per_hour :
  total_coins / scuba_diving_hours = 25 := by sorry

end NUMINAMATH_CALUDE_gold_coins_per_hour_l1397_139711


namespace NUMINAMATH_CALUDE_range_of_m_l1397_139700

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, (m + 1) * (x^2 + 1) ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 > 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m)) → (m ≤ -2 ∨ m > -1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1397_139700


namespace NUMINAMATH_CALUDE_festival_groups_l1397_139728

theorem festival_groups (n : ℕ) (h : n = 7) : 
  (Nat.choose n 4 = 35) ∧ (Nat.choose n 3 = 35) := by
  sorry

#check festival_groups

end NUMINAMATH_CALUDE_festival_groups_l1397_139728


namespace NUMINAMATH_CALUDE_G_equals_4F_l1397_139760

noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

noncomputable def G (x : ℝ) : ℝ := F ((4 * x - x^3) / (1 + 4 * x^2))

theorem G_equals_4F (x : ℝ) : G x = 4 * F x :=
  sorry

end NUMINAMATH_CALUDE_G_equals_4F_l1397_139760


namespace NUMINAMATH_CALUDE_price_after_decrease_l1397_139795

/-- The original price of an article given its reduced price after a percentage decrease -/
def original_price (reduced_price : ℚ) (decrease_percentage : ℚ) : ℚ :=
  reduced_price / (1 - decrease_percentage)

/-- Theorem stating that if an article's price after a 56% decrease is Rs. 4400, 
    then its original price was Rs. 10000 -/
theorem price_after_decrease (reduced_price : ℚ) (h : reduced_price = 4400) :
  original_price reduced_price (56/100) = 10000 := by
  sorry

end NUMINAMATH_CALUDE_price_after_decrease_l1397_139795


namespace NUMINAMATH_CALUDE_intersection_area_is_zero_l1397_139772

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three vertices -/
structure Triangle :=
  (v1 : Point)
  (v2 : Point)
  (v3 : Point)

/-- Calculate the area of intersection between two triangles -/
def areaOfIntersection (t1 t2 : Triangle) : ℝ := sorry

/-- The main theorem stating that the area of intersection is zero -/
theorem intersection_area_is_zero :
  let t1 := Triangle.mk (Point.mk 0 2) (Point.mk 2 1) (Point.mk 0 0)
  let t2 := Triangle.mk (Point.mk 2 2) (Point.mk 0 1) (Point.mk 2 0)
  areaOfIntersection t1 t2 = 0 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_zero_l1397_139772


namespace NUMINAMATH_CALUDE_special_triangle_side_length_l1397_139743

/-- An equilateral triangle with a special interior point -/
structure SpecialTriangle where
  -- The side length of the equilateral triangle
  s : ℝ
  -- The coordinates of the interior point P
  P : ℝ × ℝ
  -- Condition that the triangle is equilateral
  equilateral : s > 0
  -- Conditions for distances from P to vertices
  dist_AP : Real.sqrt ((P.1 - s/2)^2 + (P.2 - Real.sqrt 3 * s/2)^2) = Real.sqrt 2
  dist_BP : Real.sqrt ((P.1 - s)^2 + P.2^2) = 2
  dist_CP : Real.sqrt P.1^2 + P.2^2 = 1

/-- The side length of a special triangle is 5 -/
theorem special_triangle_side_length (t : SpecialTriangle) : t.s = 5 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_side_length_l1397_139743


namespace NUMINAMATH_CALUDE_clock_angle_division_theorem_l1397_139769

/-- The time when the second hand divides the angle between hour and minute hands -/
def clock_division_time (n : ℕ) (k : ℚ) : ℚ :=
  (43200 * (1 + k) * n) / (719 + 708 * k)

/-- Theorem stating the time when the second hand divides the angle between hour and minute hands -/
theorem clock_angle_division_theorem (n : ℕ) (k : ℚ) :
  let t := clock_division_time n k
  let second_pos := t
  let minute_pos := t / 60
  let hour_pos := t / 720
  (second_pos - hour_pos) / (minute_pos - second_pos) = k ∧
  t = (43200 * (1 + k) * n) / (719 + 708 * k) := by
  sorry


end NUMINAMATH_CALUDE_clock_angle_division_theorem_l1397_139769


namespace NUMINAMATH_CALUDE_complement_A_union_B_l1397_139737

def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}
def U : Set Int := Set.univ

theorem complement_A_union_B :
  (A ∪ B)ᶜ = {x : Int | ∃ k : Int, x = 3 * k} :=
sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l1397_139737


namespace NUMINAMATH_CALUDE_johnny_age_multiple_l1397_139751

theorem johnny_age_multiple (current_age : ℕ) (m : ℕ+) : current_age = 8 →
  (current_age + 2 : ℕ) = m * (current_age - 3) →
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_johnny_age_multiple_l1397_139751


namespace NUMINAMATH_CALUDE_tetrahedron_has_six_edges_l1397_139746

/-- A tetrahedron is a three-dimensional geometric shape with four triangular faces. -/
structure Tetrahedron where
  vertices : Finset (Fin 4)
  faces : Finset (Finset (Fin 3))
  is_valid : faces.card = 4 ∧ ∀ f ∈ faces, f.card = 3

/-- The number of edges in a tetrahedron -/
def num_edges (t : Tetrahedron) : ℕ := sorry

/-- Theorem: A tetrahedron has exactly 6 edges -/
theorem tetrahedron_has_six_edges (t : Tetrahedron) : num_edges t = 6 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_has_six_edges_l1397_139746


namespace NUMINAMATH_CALUDE_cubeTowerSurfaceArea_8_l1397_139741

/-- Calculates the surface area of a cube tower -/
def cubeTowerSurfaceArea (n : Nat) : Nat :=
  let sideAreas : Nat → Nat := fun i => 6 * i^2
  let bottomAreas : Nat → Nat := fun i => i^2
  let adjustedAreas : List Nat := (List.range n).map (fun i =>
    if i = 0 then sideAreas (i + 1)
    else sideAreas (i + 1) - bottomAreas (i + 1))
  adjustedAreas.sum

/-- The surface area of a tower of 8 cubes with side lengths 1 to 8 is 1021 -/
theorem cubeTowerSurfaceArea_8 :
  cubeTowerSurfaceArea 8 = 1021 := by
  sorry

end NUMINAMATH_CALUDE_cubeTowerSurfaceArea_8_l1397_139741


namespace NUMINAMATH_CALUDE_prime_form_and_infinitude_l1397_139717

theorem prime_form_and_infinitude (p : ℕ) :
  (Prime p ∧ p ≥ 3) →
  (∃! k : ℕ, k ≥ 1 ∧ (p = 4*k - 1 ∨ p = 4*k + 1)) ∧
  Set.Infinite {p : ℕ | Prime p ∧ ∃ k : ℕ, p = 4*k - 1} :=
by sorry

end NUMINAMATH_CALUDE_prime_form_and_infinitude_l1397_139717


namespace NUMINAMATH_CALUDE_negation_of_all_students_prepared_l1397_139750

variable (α : Type)
variable (student : α → Prop)
variable (prepared : α → Prop)

theorem negation_of_all_students_prepared :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_students_prepared_l1397_139750


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1397_139796

theorem constant_term_expansion (x : ℝ) : 
  ∃ (c : ℝ), (x + 1/x + 2)^4 = c + (terms_with_x : ℝ) ∧ c = 70 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1397_139796


namespace NUMINAMATH_CALUDE_arrangement_count_l1397_139766

/-- The number of ways to choose 2 items from a set of 4 items -/
def choose_2_from_4 : ℕ := 6

/-- The number of ways to arrange 3 items -/
def arrange_3 : ℕ := 6

/-- The total number of arrangements -/
def total_arrangements : ℕ := choose_2_from_4 * arrange_3

/-- Theorem: The number of ways to arrange 4 letters from the set {a, b, c, d, e, f},
    where a and b must be selected and adjacent (with a in front of b), is equal to 36 -/
theorem arrangement_count : total_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1397_139766


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_mixture_l1397_139709

/-- Represents a solution with a specific ratio of alcohol to water -/
structure Solution :=
  (alcohol : ℚ)
  (water : ℚ)

/-- Calculates the percentage of alcohol in a solution -/
def alcoholPercentage (s : Solution) : ℚ :=
  s.alcohol / (s.alcohol + s.water)

/-- Represents the mixing of two solutions in a specific ratio -/
structure Mixture :=
  (s1 : Solution)
  (s2 : Solution)
  (ratio1 : ℚ)
  (ratio2 : ℚ)

/-- Calculates the percentage of alcohol in a mixture -/
def mixtureAlcoholPercentage (m : Mixture) : ℚ :=
  (alcoholPercentage m.s1 * m.ratio1 + alcoholPercentage m.s2 * m.ratio2) / (m.ratio1 + m.ratio2)

theorem alcohol_percentage_in_mixture :
  let solutionA : Solution := ⟨21, 4⟩
  let solutionB : Solution := ⟨2, 3⟩
  let mixture : Mixture := ⟨solutionA, solutionB, 5, 6⟩
  mixtureAlcoholPercentage mixture = 3/5 := by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_mixture_l1397_139709


namespace NUMINAMATH_CALUDE_f_value_at_107_5_l1397_139714

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_value_at_107_5 (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_period : ∀ x, f (x + 3) = -1 / f x)
  (h_neg : ∀ x, x < 0 → f x = 4 * x) :
  f 107.5 = 1/10 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_107_5_l1397_139714


namespace NUMINAMATH_CALUDE_expression_bounds_l1397_139774

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  2 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) ∧
  Real.sqrt (a^2 + (1-b)^2) + Real.sqrt (b^2 + (1-c)^2) + 
    Real.sqrt (c^2 + (1-d)^2) + Real.sqrt (d^2 + (1-a)^2) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_bounds_l1397_139774


namespace NUMINAMATH_CALUDE_bus_stop_time_l1397_139733

/-- Proves that a bus with given speeds stops for 10 minutes per hour -/
theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 54) 
  (h2 : speed_with_stops = 45) : ℝ :=
by
  sorry

#check bus_stop_time

end NUMINAMATH_CALUDE_bus_stop_time_l1397_139733


namespace NUMINAMATH_CALUDE_distinct_primes_in_product_l1397_139720

theorem distinct_primes_in_product : ∃ (S : Finset Nat), 
  (∀ p ∈ S, Nat.Prime p) ∧ 
  (∀ p : Nat, Nat.Prime p → (p ∣ (85 * 87 * 90 * 92) ↔ p ∈ S)) ∧ 
  Finset.card S = 6 := by
  sorry

end NUMINAMATH_CALUDE_distinct_primes_in_product_l1397_139720


namespace NUMINAMATH_CALUDE_trajectory_equation_l1397_139721

/-- The trajectory of point M satisfying the given conditions -/
def trajectory_of_M (x y : ℝ) : Prop :=
  x ≠ 0 ∧ y > 0 ∧ y = (1/4) * x^2

/-- Point P -/
def P : ℝ × ℝ := (0, -3)

/-- Point A on the x-axis -/
def A (a : ℝ) : ℝ × ℝ := (a, 0)

/-- Point Q on the positive y-axis -/
def Q (b : ℝ) : ℝ × ℝ := (0, b)

/-- Condition: Q is on the positive half of y-axis -/
def Q_positive (b : ℝ) : Prop := b > 0

/-- Vector PA -/
def vec_PA (a : ℝ) : ℝ × ℝ := (a - P.1, 0 - P.2)

/-- Vector AM -/
def vec_AM (a x y : ℝ) : ℝ × ℝ := (x - a, y)

/-- Vector MQ -/
def vec_MQ (x y b : ℝ) : ℝ × ℝ := (0 - x, b - y)

/-- Dot product of PA and AM is zero -/
def PA_dot_AM_zero (a x y : ℝ) : Prop :=
  (vec_PA a).1 * (vec_AM a x y).1 + (vec_PA a).2 * (vec_AM a x y).2 = 0

/-- AM = -3/2 * MQ -/
def AM_eq_neg_three_half_MQ (a x y b : ℝ) : Prop :=
  vec_AM a x y = (-3/2 : ℝ) • vec_MQ x y b

/-- The main theorem: given the conditions, prove that M follows the trajectory equation -/
theorem trajectory_equation (x y a b : ℝ) : 
  Q_positive b →
  PA_dot_AM_zero a x y →
  AM_eq_neg_three_half_MQ a x y b →
  trajectory_of_M x y :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1397_139721


namespace NUMINAMATH_CALUDE_absolute_value_equation_simplification_l1397_139702

theorem absolute_value_equation_simplification
  (a b c : ℝ)
  (h1 : ∀ x : ℝ, |5*x - 4| + a ≠ 0)
  (h2 : ∃ x y : ℝ, x ≠ y ∧ |4*x - 3| + b = 0 ∧ |4*y - 3| + b = 0)
  (h3 : ∃! x : ℝ, |3*x - 2| + c = 0) :
  |a - c| + |c - b| - |a - b| = 0 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_simplification_l1397_139702


namespace NUMINAMATH_CALUDE_quotient_problem_l1397_139793

theorem quotient_problem (k : ℤ) (h : k = 4) : 16 / k = 4 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l1397_139793


namespace NUMINAMATH_CALUDE_largest_number_proof_l1397_139797

theorem largest_number_proof (w x y z : ℕ) : 
  w + x + y = 190 ∧ 
  w + x + z = 210 ∧ 
  w + y + z = 220 ∧ 
  x + y + z = 235 → 
  max w (max x (max y z)) = 95 := by
sorry

end NUMINAMATH_CALUDE_largest_number_proof_l1397_139797


namespace NUMINAMATH_CALUDE_divisible_by_1968_l1397_139784

theorem divisible_by_1968 (n : ℕ) : ∃ k : ℤ, 
  (-1)^(2*n) + 9^(4*n) - 6^(8*n) + 8^(16*n) = 1968 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_1968_l1397_139784


namespace NUMINAMATH_CALUDE_no_rain_probability_l1397_139763

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l1397_139763


namespace NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l1397_139761

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_n_for_irreducible_fractions : 
  ∃ (n : ℕ), n = 28 ∧ 
  (∀ k : ℕ, 5 ≤ k → k ≤ 24 → is_coprime k (n + k + 1)) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, 5 ≤ k ∧ k ≤ 24 ∧ ¬is_coprime k (m + k + 1)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_irreducible_fractions_l1397_139761


namespace NUMINAMATH_CALUDE_roots_sum_powers_l1397_139742

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 3*a + 2 = 0 → b^2 - 3*b + 2 = 0 → a^3 + a^4*b^2 + a^2*b^4 + b^3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l1397_139742


namespace NUMINAMATH_CALUDE_range_of_a_l1397_139768

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - a| > 5) ↔ 
  (a > 8 ∨ a < -2) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1397_139768


namespace NUMINAMATH_CALUDE_fred_change_is_correct_l1397_139718

/-- The change Fred received after paying for movie tickets and borrowing a movie --/
def fred_change : ℝ :=
  let ticket_price : ℝ := 5.92
  let num_tickets : ℕ := 2
  let borrowed_movie_cost : ℝ := 6.79
  let payment : ℝ := 20
  let total_cost : ℝ := ticket_price * num_tickets + borrowed_movie_cost
  payment - total_cost

/-- Theorem stating that Fred's change is $1.37 --/
theorem fred_change_is_correct : fred_change = 1.37 := by
  sorry

end NUMINAMATH_CALUDE_fred_change_is_correct_l1397_139718


namespace NUMINAMATH_CALUDE_certain_number_problem_l1397_139727

theorem certain_number_problem : 
  ∃ x : ℝ, 0.60 * x = 0.42 * 30 + 17.4 ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1397_139727


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1397_139786

theorem sum_of_fractions_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + c) / (a + b) + (b + d) / (b + c) + (c + a) / (c + d) + (d + b) / (d + a) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l1397_139786


namespace NUMINAMATH_CALUDE_factors_of_81_l1397_139787

theorem factors_of_81 : Finset.card (Nat.divisors 81) = 5 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_81_l1397_139787


namespace NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1397_139708

/-- Given a regular triangular prism with an inscribed sphere of radius r
    and a circumscribed sphere of radius R, prove that the ratio of their
    surface areas is 5:1 -/
theorem sphere_surface_area_ratio (r R : ℝ) :
  r > 0 →
  R = r * Real.sqrt 5 →
  (4 * Real.pi * R^2) / (4 * Real.pi * r^2) = 5 :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_ratio_l1397_139708


namespace NUMINAMATH_CALUDE_square_area_problem_l1397_139782

theorem square_area_problem (s : ℝ) : 
  (0.8 * s) * (5 * s) = s^2 + 15.18 → s^2 = 5.06 := by sorry

end NUMINAMATH_CALUDE_square_area_problem_l1397_139782


namespace NUMINAMATH_CALUDE_find_n_l1397_139749

theorem find_n (n : ℕ) (h1 : Nat.lcm n 12 = 42) (h2 : Nat.gcd n 12 = 6) : n = 21 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l1397_139749


namespace NUMINAMATH_CALUDE_range_of_a_l1397_139738

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Define the statement that ¬p is a necessary but not sufficient condition for ¬q
def not_p_necessary_not_sufficient_for_not_q (a : ℝ) : Prop :=
  (∀ x, q x → p x a) ∧ (∃ x, ¬q x ∧ p x a)

-- Main theorem
theorem range_of_a :
  ∀ a : ℝ, a > 0 → not_p_necessary_not_sufficient_for_not_q a → 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1397_139738


namespace NUMINAMATH_CALUDE_valid_parameterization_l1397_139735

/-- A vector parameterization of a line --/
structure VectorParam where
  v : ℝ × ℝ  -- point vector
  d : ℝ × ℝ  -- direction vector

/-- The line y = 2x - 5 --/
def line (x : ℝ) : ℝ := 2 * x - 5

/-- Check if a point lies on the line --/
def on_line (p : ℝ × ℝ) : Prop :=
  p.2 = line p.1

/-- Check if a vector is a scalar multiple of (1, 2) --/
def is_valid_direction (v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k, 2 * k)

/-- A parameterization is valid if it satisfies both conditions --/
def is_valid_param (param : VectorParam) : Prop :=
  on_line param.v ∧ is_valid_direction param.d

theorem valid_parameterization (param : VectorParam) :
  is_valid_param param ↔ 
    (∀ (t : ℝ), on_line (param.v.1 + t * param.d.1, param.v.2 + t * param.d.2)) :=
sorry

end NUMINAMATH_CALUDE_valid_parameterization_l1397_139735


namespace NUMINAMATH_CALUDE_inequality_proof_l1397_139780

theorem inequality_proof (x y z : ℝ) (h : x + 2*y + 3*z + 8 = 0) :
  (x - 1)^2 + (y + 2)^2 + (z - 3)^2 ≥ 14 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1397_139780


namespace NUMINAMATH_CALUDE_bracelet_bead_ratio_l1397_139734

/-- Proves that the ratio of small beads to large beads in each bracelet is 1:1 --/
theorem bracelet_bead_ratio
  (total_beads : ℕ)
  (bracelets : ℕ)
  (large_beads_per_bracelet : ℕ)
  (h1 : total_beads = 528)
  (h2 : bracelets = 11)
  (h3 : large_beads_per_bracelet = 12)
  (h4 : total_beads % 2 = 0)  -- Equal amounts of small and large beads
  (h5 : (total_beads / 2) ≥ (bracelets * large_beads_per_bracelet)) :
  (total_beads / 2 - bracelets * large_beads_per_bracelet) / bracelets = large_beads_per_bracelet :=
by sorry

end NUMINAMATH_CALUDE_bracelet_bead_ratio_l1397_139734


namespace NUMINAMATH_CALUDE_tabletop_coverage_fraction_l1397_139704

-- Define the radius of the circular mat
def mat_radius : ℝ := 10

-- Define the side length of the square tabletop
def table_side : ℝ := 24

-- Theorem to prove the fraction of the tabletop covered by the mat
theorem tabletop_coverage_fraction :
  (π * mat_radius^2) / (table_side^2) = 100 * π / 576 := by
  sorry

end NUMINAMATH_CALUDE_tabletop_coverage_fraction_l1397_139704


namespace NUMINAMATH_CALUDE_cotton_planting_rate_l1397_139726

/-- Calculates the required acres per tractor per day to plant cotton --/
theorem cotton_planting_rate (total_acres : ℕ) (total_days : ℕ) 
  (tractors_first_period : ℕ) (days_first_period : ℕ)
  (tractors_second_period : ℕ) (days_second_period : ℕ) :
  total_acres = 1700 →
  total_days = 5 →
  tractors_first_period = 2 →
  days_first_period = 2 →
  tractors_second_period = 7 →
  days_second_period = 3 →
  (total_acres : ℚ) / ((tractors_first_period * days_first_period + 
    tractors_second_period * days_second_period) : ℚ) = 68 := by
  sorry

#eval (1700 : ℚ) / 25  -- Should output 68

end NUMINAMATH_CALUDE_cotton_planting_rate_l1397_139726


namespace NUMINAMATH_CALUDE_ellipse_properties_l1397_139712

-- Define the ellipse
def ellipse (x y m : ℝ) : Prop := x^2 / 25 + y^2 / m^2 = 1

-- Define the focus
def left_focus (x y : ℝ) : Prop := x = -4 ∧ y = 0

-- Define eccentricity
def eccentricity (e : ℝ) (a c : ℝ) : Prop := e = c / a

theorem ellipse_properties (m : ℝ) (h : m > 0) :
  (∃ x y, ellipse x y m ∧ left_focus x y) →
  m = 3 ∧ ∃ e, eccentricity e 5 4 ∧ e = 4/5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1397_139712


namespace NUMINAMATH_CALUDE_right_triangle_properties_l1397_139753

/-- A right triangle with sides 9 cm and 12 cm -/
structure RightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  area : ℝ
  side1_eq : side1 = 9
  side2_eq : side2 = 12
  pythagorean : side1^2 + side2^2 = hypotenuse^2
  area_formula : area = (1/2) * side1 * side2

/-- The hypotenuse of the right triangle is 15 cm and its area is 54 cm² -/
theorem right_triangle_properties (t : RightTriangle) : t.hypotenuse = 15 ∧ t.area = 54 := by
  sorry

#check right_triangle_properties

end NUMINAMATH_CALUDE_right_triangle_properties_l1397_139753


namespace NUMINAMATH_CALUDE_larger_cube_volume_l1397_139765

-- Define the number of smaller cubes
def num_small_cubes : ℕ := 343

-- Define the volume of each smaller cube
def small_cube_volume : ℝ := 1

-- Define the surface area difference
def surface_area_difference : ℝ := 1764

-- Theorem statement
theorem larger_cube_volume :
  let large_cube_side : ℝ := (num_small_cubes : ℝ) ^ (1/3)
  let small_cube_side : ℝ := small_cube_volume ^ (1/3)
  let large_cube_volume : ℝ := large_cube_side ^ 3
  (num_small_cubes : ℝ) * (6 * small_cube_side ^ 2) - (6 * large_cube_side ^ 2) = surface_area_difference →
  large_cube_volume = num_small_cubes * small_cube_volume :=
by
  sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l1397_139765


namespace NUMINAMATH_CALUDE_triangle_expression_positive_l1397_139789

theorem triangle_expression_positive (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  0 < 4 * b^2 * c^2 - (b^2 + c^2 - a^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_expression_positive_l1397_139789


namespace NUMINAMATH_CALUDE_common_root_theorem_l1397_139745

theorem common_root_theorem (a b c d : ℝ) (h1 : a + d = 2017) (h2 : b + c = 2017) :
  ∃ x : ℝ, x = 2017 / 2 ∧ (x - a) * (x - b) = (x - c) * (x - d) :=
by sorry

end NUMINAMATH_CALUDE_common_root_theorem_l1397_139745


namespace NUMINAMATH_CALUDE_remaining_sticker_sheets_l1397_139729

theorem remaining_sticker_sheets 
  (initial_stickers : ℕ) 
  (shared_stickers : ℕ) 
  (stickers_per_sheet : ℕ) 
  (h1 : initial_stickers = 150) 
  (h2 : shared_stickers = 100) 
  (h3 : stickers_per_sheet = 10) 
  (h4 : stickers_per_sheet > 0) : 
  (initial_stickers - shared_stickers) / stickers_per_sheet = 5 := by
sorry

end NUMINAMATH_CALUDE_remaining_sticker_sheets_l1397_139729


namespace NUMINAMATH_CALUDE_calculate_expression_l1397_139790

theorem calculate_expression : (-7)^7 / 7^4 + 2^8 - 10^1 = -97 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1397_139790


namespace NUMINAMATH_CALUDE_sqrt_square_negative_two_l1397_139775

theorem sqrt_square_negative_two : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_negative_two_l1397_139775


namespace NUMINAMATH_CALUDE_camp_cedar_counselors_l1397_139730

/-- Calculates the number of counselors needed for a camp --/
def counselors_needed (num_boys : ℕ) (girls_to_boys_ratio : ℕ) (children_per_counselor : ℕ) : ℕ :=
  let num_girls := num_boys * girls_to_boys_ratio
  let total_children := num_boys + num_girls
  total_children / children_per_counselor

/-- Proves that Camp Cedar needs 20 counselors --/
theorem camp_cedar_counselors :
  counselors_needed 40 3 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_camp_cedar_counselors_l1397_139730


namespace NUMINAMATH_CALUDE_odometer_reading_before_trip_l1397_139756

theorem odometer_reading_before_trip 
  (odometer_at_lunch : ℝ) 
  (miles_traveled : ℝ) 
  (h1 : odometer_at_lunch = 372.0)
  (h2 : miles_traveled = 159.7) :
  odometer_at_lunch - miles_traveled = 212.3 := by
sorry

end NUMINAMATH_CALUDE_odometer_reading_before_trip_l1397_139756


namespace NUMINAMATH_CALUDE_minimize_distance_sum_l1397_139723

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculate the angle at vertex B of a triangle -/
def angle_at_vertex (t : Triangle) (v : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def is_inside (p : Point) (t : Triangle) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- The main theorem about the point that minimizes the sum of distances -/
theorem minimize_distance_sum (t : Triangle) : 
  (∀ v, angle_at_vertex t v < 120) → 
    ∃ O, is_inside O t ∧ 
      ∀ P, is_inside P t → 
        distance O t.A + distance O t.B + distance O t.C ≤ 
        distance P t.A + distance P t.B + distance P t.C 
  ∧ 
  (∃ v, angle_at_vertex t v ≥ 120) → 
    ∃ v, angle_at_vertex t v ≥ 120 ∧ 
      ∀ P, is_inside P t → 
        distance v t.A + distance v t.B + distance v t.C ≤ 
        distance P t.A + distance P t.B + distance P t.C :=
sorry

end NUMINAMATH_CALUDE_minimize_distance_sum_l1397_139723


namespace NUMINAMATH_CALUDE_initial_members_family_b_l1397_139762

/-- Represents the number of members in each family in Indira Nagar -/
structure FamilyMembers where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ

/-- The theorem stating the initial number of members in family b -/
theorem initial_members_family_b (fm : FamilyMembers) : 
  fm.a = 7 ∧ fm.c = 10 ∧ fm.d = 13 ∧ fm.e = 6 ∧ fm.f = 10 ∧
  (fm.a + fm.b + fm.c + fm.d + fm.e + fm.f - 6) / 6 = 8 →
  fm.b = 8 := by
  sorry

#check initial_members_family_b

end NUMINAMATH_CALUDE_initial_members_family_b_l1397_139762


namespace NUMINAMATH_CALUDE_max_faces_convex_polyhedron_l1397_139725

/-- A convex polyhedron with n congruent triangular faces, each having angles 36°, 72°, and 72° -/
structure ConvexPolyhedron where
  n : ℕ  -- number of faces
  convex : Bool
  congruentFaces : Bool
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

/-- The maximum number of faces for the given polyhedron is 36 -/
theorem max_faces_convex_polyhedron (p : ConvexPolyhedron) 
  (h1 : p.convex = true)
  (h2 : p.congruentFaces = true)
  (h3 : p.angleA = 36)
  (h4 : p.angleB = 72)
  (h5 : p.angleC = 72) :
  p.n ≤ 36 :=
sorry

end NUMINAMATH_CALUDE_max_faces_convex_polyhedron_l1397_139725


namespace NUMINAMATH_CALUDE_simplify_expression_l1397_139748

theorem simplify_expression (w x : ℝ) :
  3*w + 6*w + 9*w + 12*w + 15*w + 20*x + 24 = 45*w + 20*x + 24 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1397_139748


namespace NUMINAMATH_CALUDE_fraction_equality_l1397_139771

theorem fraction_equality (p q r : ℕ+) 
  (h : (p : ℚ) + 1 / ((q : ℚ) + 1 / (r : ℚ)) = 25 / 19) : 
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1397_139771


namespace NUMINAMATH_CALUDE_cone_vertex_angle_l1397_139739

noncomputable def vertex_angle_third_cone : ℝ := 2 * Real.arcsin (1/4)

theorem cone_vertex_angle 
  (first_two_cones_identical : Bool)
  (fourth_cone_internal : Bool)
  (first_two_cones_half_fourth : Bool) :
  ∃ (α : ℝ), 
    α = π/6 + Real.arcsin (1/4) ∧ 
    α > 0 ∧ 
    α < π/2 ∧
    2 * α = vertex_angle_third_cone ∧
    first_two_cones_identical = true ∧
    fourth_cone_internal = true ∧
    first_two_cones_half_fourth = true :=
by sorry

end NUMINAMATH_CALUDE_cone_vertex_angle_l1397_139739


namespace NUMINAMATH_CALUDE_reciprocal_roots_identity_l1397_139710

theorem reciprocal_roots_identity (p q r s : ℝ) : 
  (∃ a : ℝ, a^2 + p*a + q = 0 ∧ (1/a)^2 + r*(1/a) + s = 0) →
  (p*s - r)*(q*r - p) = (q*s - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_roots_identity_l1397_139710


namespace NUMINAMATH_CALUDE_interest_group_signups_l1397_139752

theorem interest_group_signups :
  let num_students : ℕ := 5
  let num_groups : ℕ := 3
  num_groups ^ num_students = 243 :=
by sorry

end NUMINAMATH_CALUDE_interest_group_signups_l1397_139752


namespace NUMINAMATH_CALUDE_square_root_problem_l1397_139747

theorem square_root_problem (x y z a : ℝ) : 
  (∃ (x : ℝ), x > 0 ∧ (a - 3)^2 = x ∧ (2*a + 15)^2 = x) →
  y = (-3)^3 →
  z = Int.floor (Real.sqrt 13) →
  Real.sqrt (x + y - 2*z) = 4 ∨ Real.sqrt (x + y - 2*z) = -4 := by
  sorry

#check square_root_problem

end NUMINAMATH_CALUDE_square_root_problem_l1397_139747


namespace NUMINAMATH_CALUDE_expression_undefined_at_ten_expression_undefined_when_denominator_zero_l1397_139703

/-- The expression is not defined when x = 10 -/
theorem expression_undefined_at_ten : 
  ∀ x : ℝ, x = 10 → (x^3 - 30*x^2 + 300*x - 1000 = 0) := by
  sorry

/-- The denominator of the expression -/
def denominator (x : ℝ) : ℝ := x^3 - 30*x^2 + 300*x - 1000

/-- The expression is undefined when the denominator is zero -/
theorem expression_undefined_when_denominator_zero (x : ℝ) : 
  denominator x = 0 → ¬∃y : ℝ, y = (3*x^4 + 2*x + 6) / (x^3 - 30*x^2 + 300*x - 1000) := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_at_ten_expression_undefined_when_denominator_zero_l1397_139703


namespace NUMINAMATH_CALUDE_simplify_expressions_l1397_139773

/-- Prove the simplification of two algebraic expressions -/
theorem simplify_expressions (x y : ℝ) :
  (7 * x + 3 * (x^2 - 2) - 3 * (1/2 * x^2 - x + 3) = 3/2 * x^2 + 10 * x - 15) ∧
  (3 * (2 * x^2 * y - x * y^2) - 4 * (-x * y^2 + 3 * x^2 * y) = -6 * x^2 * y + x * y^2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1397_139773


namespace NUMINAMATH_CALUDE_units_digit_37_pow_37_l1397_139776

theorem units_digit_37_pow_37 : 37^37 ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_37_pow_37_l1397_139776


namespace NUMINAMATH_CALUDE_cube_surface_area_l1397_139779

/-- The surface area of a cube, given the distance between non-intersecting diagonals of adjacent faces -/
theorem cube_surface_area (d : ℝ) (h : d = 8) : 
  let a := d * 3 / Real.sqrt 3
  6 * a^2 = 1152 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_l1397_139779


namespace NUMINAMATH_CALUDE_class_average_theorem_l1397_139754

theorem class_average_theorem (total_students : ℝ) (h_total : total_students > 0) :
  let group1_percent : ℝ := 25
  let group1_average : ℝ := 80
  let group2_percent : ℝ := 50
  let group2_average : ℝ := 65
  let group3_percent : ℝ := 100 - group1_percent - group2_percent
  let group3_average : ℝ := 90
  let overall_average : ℝ := (group1_percent * group1_average + group2_percent * group2_average + group3_percent * group3_average) / 100
  overall_average = 75 := by
  sorry


end NUMINAMATH_CALUDE_class_average_theorem_l1397_139754


namespace NUMINAMATH_CALUDE_square_diff_eq_three_implies_product_eq_nine_l1397_139755

theorem square_diff_eq_three_implies_product_eq_nine (x y : ℝ) :
  x^2 - y^2 = 3 → (x + y)^2 * (x - y)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_eq_three_implies_product_eq_nine_l1397_139755


namespace NUMINAMATH_CALUDE_speedster_roadster_convertibles_l1397_139799

/-- Represents the inventory of an automobile company -/
structure Inventory where
  total : ℕ
  speedsters : ℕ
  roadsters : ℕ
  cruisers : ℕ
  speedster_convertibles : ℕ
  roadster_convertibles : ℕ
  cruiser_convertibles : ℕ

/-- Theorem stating the number of Speedster and Roadster convertibles -/
theorem speedster_roadster_convertibles (inv : Inventory) : 
  inv.speedster_convertibles + inv.roadster_convertibles = 52 :=
  by
  have h1 : inv.total = 100 := by sorry
  have h2 : inv.speedsters = inv.total * 2 / 5 := by sorry
  have h3 : inv.roadsters = inv.total * 3 / 10 := by sorry
  have h4 : inv.cruisers = inv.total - inv.speedsters - inv.roadsters := by sorry
  have h5 : inv.speedster_convertibles = inv.speedsters * 4 / 5 := by sorry
  have h6 : inv.roadster_convertibles = inv.roadsters * 2 / 3 := by sorry
  have h7 : inv.cruiser_convertibles = inv.cruisers * 1 / 4 := by sorry
  have h8 : inv.total - inv.speedsters = 60 := by sorry
  sorry

end NUMINAMATH_CALUDE_speedster_roadster_convertibles_l1397_139799


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1397_139713

theorem sum_of_a_and_b_is_one :
  ∀ (a b : ℝ),
  (∃ (x : ℝ), x = a + Real.sqrt b) →
  (a + Real.sqrt b + (a - Real.sqrt b) = -4) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 1) →
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l1397_139713


namespace NUMINAMATH_CALUDE_tangent_trapezoid_EQ_length_l1397_139798

/-- Represents a trapezoid with a circle tangent to two sides --/
structure TangentTrapezoid where
  EF : ℝ
  FG : ℝ
  GH : ℝ
  HE : ℝ
  EQ : ℝ
  QF : ℝ
  EQ_QF_ratio : EQ / QF = 5 / 3

/-- The theorem stating the length of EQ in the given trapezoid --/
theorem tangent_trapezoid_EQ_length (t : TangentTrapezoid) 
  (h1 : t.EF = 150)
  (h2 : t.FG = 65)
  (h3 : t.GH = 35)
  (h4 : t.HE = 90)
  (h5 : t.EF = t.EQ + t.QF) :
  t.EQ = 375 / 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_trapezoid_EQ_length_l1397_139798


namespace NUMINAMATH_CALUDE_distance_las_vegas_to_los_angeles_l1397_139777

/-- Calculates the distance from Las Vegas to Los Angeles given the total drive time,
    average speed, and distance from Salt Lake City to Las Vegas. -/
theorem distance_las_vegas_to_los_angeles
  (total_time : ℝ)
  (average_speed : ℝ)
  (distance_salt_lake_to_vegas : ℝ)
  (h1 : total_time = 11)
  (h2 : average_speed = 63)
  (h3 : distance_salt_lake_to_vegas = 420) :
  total_time * average_speed - distance_salt_lake_to_vegas = 273 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_las_vegas_to_los_angeles_l1397_139777


namespace NUMINAMATH_CALUDE_simplify_expression_l1397_139724

theorem simplify_expression (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1397_139724


namespace NUMINAMATH_CALUDE_divisor_sum_equality_implies_prime_power_l1397_139701

/-- σ(N) is the sum of the positive integer divisors of N -/
def sigma (N : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem divisor_sum_equality_implies_prime_power (m n : ℕ) :
  m ≥ n → n ≥ 2 →
  (sigma m - 1) / (m - 1) = (sigma n - 1) / (n - 1) →
  (sigma m - 1) / (m - 1) = (sigma (m * n) - 1) / (m * n - 1) →
  ∃ (p : ℕ) (e f : ℕ), Prime p ∧ e ≥ f ∧ f ≥ 1 ∧ m = p^e ∧ n = p^f :=
sorry

end NUMINAMATH_CALUDE_divisor_sum_equality_implies_prime_power_l1397_139701


namespace NUMINAMATH_CALUDE_tank_capacity_l1397_139732

theorem tank_capacity : ∃ (capacity : ℚ), 
  capacity > 0 ∧ 
  (1/3 : ℚ) * capacity + 180 = (2/3 : ℚ) * capacity ∧ 
  capacity = 540 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l1397_139732


namespace NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l1397_139740

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x > f y

theorem even_periodic_function_monotonicity (f : ℝ → ℝ)
  (h_even : is_even f) (h_period : has_period f 2) :
  increasing_on f 0 1 ↔ decreasing_on f 3 4 := by sorry

end NUMINAMATH_CALUDE_even_periodic_function_monotonicity_l1397_139740


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1397_139778

theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a - 3 ∧ P.2 = 2 * a + 1 ∧ P.2 = 0) → a = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1397_139778


namespace NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l1397_139706

/-- The smallest positive integer M such that M and M^2 end in the same sequence of three non-zero digits in base 10 -/
def smallest_matching_end_digits : ℕ := 376

/-- Check if a number ends with the given three digits -/
def ends_with (n : ℕ) (xyz : ℕ) : Prop :=
  n % 1000 = xyz

/-- The property that M and M^2 end with the same three non-zero digits -/
def has_matching_end_digits (M : ℕ) : Prop :=
  ∃ (xyz : ℕ), xyz ≥ 100 ∧ xyz < 1000 ∧ ends_with M xyz ∧ ends_with (M^2) xyz

theorem smallest_matching_end_digits_correct :
  has_matching_end_digits smallest_matching_end_digits ∧
  ∀ M : ℕ, M < smallest_matching_end_digits → ¬(has_matching_end_digits M) :=
sorry

end NUMINAMATH_CALUDE_smallest_matching_end_digits_correct_l1397_139706


namespace NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1397_139758

theorem quadratic_inequality_empty_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) → (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_empty_solution_l1397_139758


namespace NUMINAMATH_CALUDE_square_root_equation_l1397_139788

theorem square_root_equation (x : ℝ) : 
  Real.sqrt (x - 3) = 5 → x = 28 := by sorry

end NUMINAMATH_CALUDE_square_root_equation_l1397_139788


namespace NUMINAMATH_CALUDE_total_face_masks_produced_l1397_139715

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the duration of Manolo's shift in hours -/
def shift_duration : ℕ := 4

/-- Represents the time to make one face-mask in the first hour (in minutes) -/
def first_hour_rate : ℕ := 4

/-- Represents the time to make one face-mask after the first hour (in minutes) -/
def subsequent_rate : ℕ := 6

/-- Calculates the number of face-masks made in the first hour -/
def first_hour_production : ℕ := minutes_per_hour / first_hour_rate

/-- Calculates the number of face-masks made in the subsequent hours -/
def subsequent_hours_production : ℕ := (shift_duration - 1) * minutes_per_hour / subsequent_rate

/-- Theorem: The total number of face-masks produced in a four-hour shift is 45 -/
theorem total_face_masks_produced :
  first_hour_production + subsequent_hours_production = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_face_masks_produced_l1397_139715


namespace NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l1397_139785

/-- Parabola type -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Line passing through a point with given slope angle -/
structure Line where
  slope_angle : ℝ
  point : ℝ × ℝ

/-- Intersection points of a line and a parabola -/
structure Intersection where
  A : ℝ × ℝ
  B : ℝ × ℝ

/-- Theorem stating the ratio of distances from intersection points to focus -/
theorem parabola_line_intersection_ratio
  (C : Parabola)
  (l : Line)
  (i : Intersection)
  (h1 : l.slope_angle = π / 3) -- 60 degrees in radians
  (h2 : l.point = (C.p / 2, 0)) -- Focus of the parabola
  (h3 : i.A.1 > 0 ∧ i.A.2 > 0) -- A in first quadrant
  (h4 : i.B.1 > 0 ∧ i.B.2 < 0) -- B in fourth quadrant
  (h5 : i.A.2^2 = 2 * C.p * i.A.1) -- A satisfies parabola equation
  (h6 : i.B.2^2 = 2 * C.p * i.B.1) -- B satisfies parabola equation
  (h7 : i.A.2 - 0 = Real.sqrt 3 * (i.A.1 - C.p / 2)) -- A satisfies line equation
  (h8 : i.B.2 - 0 = Real.sqrt 3 * (i.B.1 - C.p / 2)) -- B satisfies line equation
  : (Real.sqrt ((i.A.1 - C.p / 2)^2 + i.A.2^2)) / (Real.sqrt ((i.B.1 - C.p / 2)^2 + i.B.2^2)) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_ratio_l1397_139785


namespace NUMINAMATH_CALUDE_game_terminates_l1397_139759

/-- Represents the state of knowledge for each player -/
structure PlayerKnowledge where
  lower : Nat
  upper : Nat

/-- Represents the game state -/
structure GameState where
  player1 : PlayerKnowledge
  player2 : PlayerKnowledge
  turn : Nat

/-- Updates the game state based on a negative response -/
def updateGameState (state : GameState) : GameState :=
  sorry

/-- Checks if a player knows the other's number -/
def knowsNumber (knowledge : PlayerKnowledge) : Bool :=
  sorry

/-- Simulates the game for a given initial state -/
def playGame (initialState : GameState) : Nat :=
  sorry

/-- Theorem stating that the game will terminate -/
theorem game_terminates (n : Nat) :
  ∃ (k : Nat), ∀ (m : Nat),
    let initialState : GameState := {
      player1 := { lower := 1, upper := n + 1 },
      player2 := { lower := 1, upper := n + 1 },
      turn := 0
    }
    playGame initialState ≤ k :=
  sorry

end NUMINAMATH_CALUDE_game_terminates_l1397_139759


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l1397_139744

/-- Represents a right regular prism with external diagonal lengths a, b, and c --/
structure RightRegularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Theorem stating that {3, 4, 6} cannot be the lengths of external diagonals of a right regular prism --/
theorem invalid_external_diagonals (p : RightRegularPrism) :
  p.a = 3 ∧ p.b = 4 ∧ p.c = 6 → False := by
  sorry

#check invalid_external_diagonals

end NUMINAMATH_CALUDE_invalid_external_diagonals_l1397_139744


namespace NUMINAMATH_CALUDE_parallelogram_reflection_l1397_139716

/-- Reflect a point across the x-axis -/
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- Reflect a point across the line y = -x -/
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ := (-p.2, -p.1)

/-- The final position of point C after two reflections -/
def final_position (C : ℝ × ℝ) : ℝ × ℝ :=
  reflect_y_neg_x (reflect_x_axis C)

theorem parallelogram_reflection :
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (4, 9)
  let C : ℝ × ℝ := (6, 5)
  let D : ℝ × ℝ := (4, 1)
  final_position C = (5, -6) := by sorry

end NUMINAMATH_CALUDE_parallelogram_reflection_l1397_139716


namespace NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l1397_139719

/-- 
Given an arithmetic sequence {a_n} with a₁ = -8 and a₂ = -6,
if x is added to a₁, a₄, and a₅ to form a geometric sequence,
then x = -1.
-/
theorem arithmetic_to_geometric_sequence (a : ℕ → ℤ) (x : ℤ) : 
  a 1 = -8 →
  a 2 = -6 →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  ((-8 + x) * x = (-2 + x)^2) →
  ((-2 + x)^2 = x * x) →
  x = -1 := by sorry

end NUMINAMATH_CALUDE_arithmetic_to_geometric_sequence_l1397_139719


namespace NUMINAMATH_CALUDE_log_problem_l1397_139707

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_problem (x : ℝ) 
  (h1 : x < 1) 
  (h2 : (log10 x)^3 - 3 * log10 x = 522) : 
  (log10 x)^4 - log10 (x^4) = 6597 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1397_139707


namespace NUMINAMATH_CALUDE_symmetry_implies_t_zero_l1397_139792

/-- Line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop :=
  8 * x + 6 * y + 1 = 0

/-- Circle C₁ in the Cartesian coordinate system -/
def circle_C1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 2*y + 13 = 0

/-- Circle C₂ in the Cartesian coordinate system -/
def circle_C2 (t x y : ℝ) : Prop :=
  x^2 + y^2 + 8*t*x - 8*y + 16*t + 12 = 0

/-- The center of circle C₁ -/
def center_C1 : ℝ × ℝ :=
  (-4, 1)

/-- The center of circle C₂ -/
def center_C2 (t : ℝ) : ℝ × ℝ :=
  (-4*t, 4)

/-- Theorem: When circle C₁ and circle C₂ are symmetric about line l, t = 0 -/
theorem symmetry_implies_t_zero :
  ∀ t : ℝ, (∃ x y : ℝ, line_l x y ∧ 
    ((x - (-4))^2 + (y - 1)^2 = (x - (-4*t))^2 + (y - 4)^2) ∧
    ((8*x + 6*y + 1 = 0) → 
      ((-4 + (-4*t))/2 = x ∧ (1 + 4)/2 = y))) →
  t = 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_t_zero_l1397_139792


namespace NUMINAMATH_CALUDE_largest_m_for_cubic_quintic_inequality_l1397_139794

theorem largest_m_for_cubic_quintic_inequality :
  ∃ (m : ℝ), m = 9 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    10 * (a^3 + b^3 + c^3) - m * (a^5 + b^5 + c^5) ≥ 1) ∧
  (∀ (m' : ℝ), m' > m →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
      10 * (a^3 + b^3 + c^3) - m' * (a^5 + b^5 + c^5) < 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_cubic_quintic_inequality_l1397_139794


namespace NUMINAMATH_CALUDE_new_bill_is_35_l1397_139722

/-- Calculates the new total bill after substitutions and delivery/tip --/
def calculate_new_bill (original_order : ℝ) 
                       (tomato_old tomato_new : ℝ) 
                       (lettuce_old lettuce_new : ℝ) 
                       (celery_old celery_new : ℝ) 
                       (delivery_tip : ℝ) : ℝ :=
  original_order + 
  (tomato_new - tomato_old) + 
  (lettuce_new - lettuce_old) + 
  (celery_new - celery_old) + 
  delivery_tip

/-- Theorem stating that the new bill is $35.00 --/
theorem new_bill_is_35 : 
  calculate_new_bill 25 0.99 2.20 1.00 1.75 1.96 2.00 8.00 = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_new_bill_is_35_l1397_139722


namespace NUMINAMATH_CALUDE_profit_percentage_previous_year_l1397_139764

theorem profit_percentage_previous_year 
  (R : ℝ) -- Revenues in the previous year
  (P : ℝ) -- Profits in the previous year
  (h1 : 0.95 * R * 0.10 = 0.95 * P) -- Condition relating 2009 profits to previous year
  : P / R = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_previous_year_l1397_139764


namespace NUMINAMATH_CALUDE_fraction_irreducible_l1397_139705

theorem fraction_irreducible (n : ℤ) : Int.gcd (21*n + 4) (14*n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l1397_139705


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l1397_139757

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x + 2 * Real.sqrt y)^5 = a₀*x^5 + a₁*x^4*(Real.sqrt y) + a₂*x^3*y + a₃*x^2*y*(Real.sqrt y) + a₄*x*y^2 + a₅*y^(5/2)) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l1397_139757
