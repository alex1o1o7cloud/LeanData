import Mathlib

namespace NUMINAMATH_CALUDE_train_crossing_time_l1046_104657

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 1162.5)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1046_104657


namespace NUMINAMATH_CALUDE_f_properties_l1046_104619

/-- The function f(x) = x^2 - kx + (2k-3) -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + (2*k - 3)

theorem f_properties (k : ℝ) :
  /- (1) When k = 3/2, f(x) > 0 if and only if x < 0 or x > 3/2 -/
  (k = 3/2 → ∀ x, f k x > 0 ↔ (x < 0 ∨ x > 3/2)) ∧
  /- (2) f(x) > 0 for all x ∈ R if and only if k ∈ (2, 6) -/
  ((∀ x, f k x > 0) ↔ (k > 2 ∧ k < 6)) ∧
  /- (3) f(x) has two distinct zeros both greater than 5/2 if and only if k ∈ (6, 13/2) -/
  (∃ x y, x ≠ y ∧ x > 5/2 ∧ y > 5/2 ∧ f k x = 0 ∧ f k y = 0) ↔ (k > 6 ∧ k < 13/2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1046_104619


namespace NUMINAMATH_CALUDE_s_one_eq_one_l1046_104602

/-- s(n) is a function that returns the n-digit number formed by attaching
    the first n perfect squares in order. -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem: s(1) equals 1 -/
theorem s_one_eq_one : s 1 = 1 := by sorry

end NUMINAMATH_CALUDE_s_one_eq_one_l1046_104602


namespace NUMINAMATH_CALUDE_cookie_distribution_l1046_104606

theorem cookie_distribution (total_cookies : ℕ) (num_children : ℕ) 
  (h1 : total_cookies = 28) (h2 : num_children = 6) : 
  total_cookies - (num_children * (total_cookies / num_children)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l1046_104606


namespace NUMINAMATH_CALUDE_derivative_limit_equality_l1046_104627

theorem derivative_limit_equality (f : ℝ → ℝ) (h : HasDerivAt f (-2) 2) :
  Filter.Tendsto (fun x => (f x - f 2) / (x - 2)) (Filter.atTop.comap (fun x => |x - 2|)) (nhds (-2)) := by
  sorry

end NUMINAMATH_CALUDE_derivative_limit_equality_l1046_104627


namespace NUMINAMATH_CALUDE_A_power_98_l1046_104653

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_98 : A^98 = !![0, 0, 0; 0, -1, 0; 0, 0, -1] := by
  sorry

end NUMINAMATH_CALUDE_A_power_98_l1046_104653


namespace NUMINAMATH_CALUDE_M_equals_interval_l1046_104673

/-- The set of real numbers m for which there exists an x in (-1, 1) satisfying x^2 - x - m = 0 -/
def M : Set ℝ := {m : ℝ | ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - x - m = 0}

/-- The theorem stating that M is equal to [-1/4, 2) -/
theorem M_equals_interval : M = Set.Icc (-1/4) 2 := by
  sorry

end NUMINAMATH_CALUDE_M_equals_interval_l1046_104673


namespace NUMINAMATH_CALUDE_shooting_stars_count_l1046_104645

theorem shooting_stars_count (bridget reginald sam emma max : ℕ) : 
  bridget = 14 →
  reginald = bridget - 2 →
  sam = reginald + 4 →
  emma = sam + 3 →
  max = bridget - 7 →
  sam - ((bridget + reginald + sam + emma + max) / 5 : ℚ) = 2.4 :=
by sorry

end NUMINAMATH_CALUDE_shooting_stars_count_l1046_104645


namespace NUMINAMATH_CALUDE_prob_adjacent_is_half_l1046_104649

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange four students with two specific students adjacent. -/
def adjacent_arrangements : ℕ := 2 * (permutations 3)

/-- The total number of ways to arrange four students. -/
def total_arrangements : ℕ := permutations 4

/-- The probability of two specific students being adjacent in a line of four students. -/
def prob_adjacent : ℚ := adjacent_arrangements / total_arrangements

theorem prob_adjacent_is_half : prob_adjacent = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_adjacent_is_half_l1046_104649


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1046_104663

/-- An isosceles right triangle with legs of length 8 units -/
structure IsoscelesRightTriangle where
  /-- The length of each leg -/
  leg_length : ℝ
  /-- The leg length is 8 units -/
  leg_is_eight : leg_length = 8

/-- The inscribed circle of the isosceles right triangle -/
def inscribed_circle (t : IsoscelesRightTriangle) : ℝ := sorry

/-- Theorem: The radius of the inscribed circle in an isosceles right triangle
    with legs of length 8 units is 8 - 4√2 -/
theorem inscribed_circle_radius (t : IsoscelesRightTriangle) :
  inscribed_circle t = 8 - 4 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1046_104663


namespace NUMINAMATH_CALUDE_problem_solution_l1046_104624

def star (a b : ℚ) : ℚ := 2 * a - b

theorem problem_solution (x : ℚ) (h : star x (star 1 3) = 2) : x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1046_104624


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1046_104628

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

/-- The theorem stating the properties of the quadratic function and the range of m -/
theorem quadratic_function_properties :
  (∀ x ∈ Set.Icc (-3 : ℝ) 1, f x ≤ 0) ∧
  (∀ x ∈ Set.Ioi 1 ∪ Set.Iio (-3 : ℝ), f x > 0) ∧
  (f 2 = 5) ∧
  (∀ m : ℝ, (∃ x : ℝ, f x = 9*m + 3) ↔ m ≥ -7/9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1046_104628


namespace NUMINAMATH_CALUDE_pyramid_frustum_volume_ratio_l1046_104668

theorem pyramid_frustum_volume_ratio : 
  let base_edge : ℝ := 24
  let altitude : ℝ := 18
  let small_altitude : ℝ := altitude / 3
  let original_volume : ℝ := (1 / 3) * (base_edge ^ 2) * altitude
  let small_volume : ℝ := (1 / 3) * ((small_altitude / altitude) * base_edge) ^ 2 * small_altitude
  let frustum_volume : ℝ := original_volume - small_volume
  frustum_volume / original_volume = 32 / 33 := by sorry

end NUMINAMATH_CALUDE_pyramid_frustum_volume_ratio_l1046_104668


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1046_104611

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : a - b = 7 * ((a + b) / 2)) : a / b = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1046_104611


namespace NUMINAMATH_CALUDE_distributive_law_l1046_104640

theorem distributive_law (a b c : ℝ) : (a + b) * c = a * c + b * c := by
  sorry

end NUMINAMATH_CALUDE_distributive_law_l1046_104640


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1046_104658

theorem ratio_w_to_y (w x y z : ℝ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1046_104658


namespace NUMINAMATH_CALUDE_terrell_weight_lifting_l1046_104655

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 15

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 2

/-- The total weight Terrell lifts with the original weights -/
def total_original_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of times Terrell needs to lift the new weights to match the original total weight -/
def new_lifts : ℚ := total_original_weight / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 37.5 := by sorry

end NUMINAMATH_CALUDE_terrell_weight_lifting_l1046_104655


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l1046_104679

theorem cube_sum_divisibility (a b c : ℤ) 
  (h1 : 6 ∣ (a^2 + b^2 + c^2)) 
  (h2 : 3 ∣ (a*b + b*c + c*a)) : 
  6 ∣ (a^3 + b^3 + c^3) :=
by sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l1046_104679


namespace NUMINAMATH_CALUDE_spheres_theorem_l1046_104699

/-- The configuration of four spheres -/
structure SpheresConfiguration where
  r : ℝ  -- radius of the three smaller spheres
  R : ℝ  -- radius of the larger sphere
  h : R > r  -- condition that R is greater than r

/-- The condition for the configuration to be possible -/
def configuration_possible (c : SpheresConfiguration) : Prop :=
  c.R ≥ (2 / Real.sqrt 3 - 1) * c.r

/-- The radius of the sphere tangent to all four spheres -/
noncomputable def tangent_sphere_radius (c : SpheresConfiguration) : ℝ :=
  let numerator := c.R * (c.R + c.r - Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3))
  let denominator := c.r + Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3) - c.R
  numerator / denominator

/-- The main theorem stating the conditions and the radius of the tangent sphere -/
theorem spheres_theorem (c : SpheresConfiguration) :
  configuration_possible c ∧
  tangent_sphere_radius c = (c.R * (c.R + c.r - Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3))) /
                            (c.r + Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3) - c.R) := by
  sorry

end NUMINAMATH_CALUDE_spheres_theorem_l1046_104699


namespace NUMINAMATH_CALUDE_room_width_to_perimeter_ratio_l1046_104625

theorem room_width_to_perimeter_ratio :
  let length : ℝ := 22
  let width : ℝ := 15
  let perimeter : ℝ := 2 * (length + width)
  (width / perimeter) = (15 / 74) := by
sorry

end NUMINAMATH_CALUDE_room_width_to_perimeter_ratio_l1046_104625


namespace NUMINAMATH_CALUDE_equation_solutions_l1046_104675

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2) ∧
  (∀ x l : ℝ, 64 * (x + l)^3 = 27 → x = -1/4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1046_104675


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l1046_104608

theorem value_of_a_minus_b (a b : ℚ) 
  (eq1 : 2020 * a + 2024 * b = 2025)
  (eq2 : 2022 * a + 2026 * b = 2030) : 
  a - b = 1515 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l1046_104608


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1046_104677

theorem trigonometric_identities :
  ∀ α : ℝ,
  (((Real.sqrt 3 * Real.sin (-1200 * π / 180)) / Real.tan (11 * π / 3)) - 
   (Real.cos (585 * π / 180) * Real.tan (-37 * π / 4)) = 
   Real.sqrt 3 / 2 - Real.sqrt 2 / 2) ∧
  ((Real.cos (α - π / 2) / Real.sin (5 * π / 2 + α)) * 
   Real.sin (α - 2 * π) * Real.cos (2 * π - α) = 
   Real.sin α ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1046_104677


namespace NUMINAMATH_CALUDE_sara_bouquets_l1046_104654

theorem sara_bouquets (red_flowers yellow_flowers : ℕ) 
  (h1 : red_flowers = 16) 
  (h2 : yellow_flowers = 24) : 
  (Nat.gcd red_flowers yellow_flowers) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sara_bouquets_l1046_104654


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1046_104644

theorem opposite_of_2023 : ∀ x : ℤ, x + 2023 = 0 ↔ x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1046_104644


namespace NUMINAMATH_CALUDE_solve_equation_l1046_104642

theorem solve_equation : ∃ y : ℕ, 400 + 2 * 20 * 5 + 25 = y ∧ y = 625 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1046_104642


namespace NUMINAMATH_CALUDE_alloy_mixing_theorem_l1046_104656

/-- Represents an alloy with two metals -/
structure Alloy where
  ratio1 : ℚ
  ratio2 : ℚ

/-- Creates a new alloy by mixing two existing alloys -/
def mixAlloys (a1 : Alloy) (p1 : ℚ) (a2 : Alloy) (p2 : ℚ) : Alloy :=
  { ratio1 := (a1.ratio1 * p1 + a2.ratio1 * p2) / (p1 + p2),
    ratio2 := (a1.ratio2 * p1 + a2.ratio2 * p2) / (p1 + p2) }

theorem alloy_mixing_theorem :
  let alloy1 : Alloy := { ratio1 := 1, ratio2 := 2 }
  let alloy2 : Alloy := { ratio1 := 2, ratio2 := 3 }
  let mixedAlloy := mixAlloys alloy1 9 alloy2 35
  mixedAlloy.ratio1 / mixedAlloy.ratio2 = 17 / 27 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixing_theorem_l1046_104656


namespace NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1046_104618

theorem smallest_number_with_given_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 3 = 2) ∧ 
  (x % 7 = 6) ∧ 
  (x % 8 = 7) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 3 = 2 ∧ y % 7 = 6 ∧ y % 8 = 7 → x ≤ y) ∧
  x = 167 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_given_remainders_l1046_104618


namespace NUMINAMATH_CALUDE_notebooks_given_to_paula_notebooks_given_to_paula_is_five_l1046_104623

theorem notebooks_given_to_paula (gerald_notebooks : ℕ) (jack_initial_extra : ℕ) 
  (given_to_mike : ℕ) (jack_remaining : ℕ) : ℕ :=
  let jack_initial := gerald_notebooks + jack_initial_extra
  let jack_after_paula := jack_remaining + given_to_mike
  let given_to_paula := jack_initial - jack_after_paula
  given_to_paula

theorem notebooks_given_to_paula_is_five :
  notebooks_given_to_paula 8 13 6 10 = 5 := by sorry

end NUMINAMATH_CALUDE_notebooks_given_to_paula_notebooks_given_to_paula_is_five_l1046_104623


namespace NUMINAMATH_CALUDE_legs_product_ge_parallel_sides_product_l1046_104662

/-- A trapezoid with perpendicular diagonals -/
structure PerpDiagonalTrapezoid where
  -- Parallel sides
  a : ℝ
  c : ℝ
  -- Legs
  b : ℝ
  d : ℝ
  -- All sides are positive
  a_pos : 0 < a
  c_pos : 0 < c
  b_pos : 0 < b
  d_pos : 0 < d
  -- Diagonals are perpendicular (using the property from the solution)
  perp_diag : b^2 + d^2 = a^2 + c^2

/-- 
  The product of the legs is at least as large as 
  the product of the parallel sides in a trapezoid 
  with perpendicular diagonals
-/
theorem legs_product_ge_parallel_sides_product (t : PerpDiagonalTrapezoid) : 
  t.b * t.d ≥ t.a * t.c := by
  sorry

end NUMINAMATH_CALUDE_legs_product_ge_parallel_sides_product_l1046_104662


namespace NUMINAMATH_CALUDE_class_size_problem_l1046_104669

theorem class_size_problem (x : ℕ) (n : ℕ) : 
  20 < x ∧ x < 30 ∧ 
  n = (0.20 : ℝ) * (5 * n) ∧
  n = (0.25 : ℝ) * (4 * n) ∧
  x = 8 * n + 2 →
  x = 26 := by sorry

end NUMINAMATH_CALUDE_class_size_problem_l1046_104669


namespace NUMINAMATH_CALUDE_square_sum_equals_five_l1046_104604

theorem square_sum_equals_five (a b : ℝ) 
  (h1 : a^3 - 3*a*b^2 = 11) 
  (h2 : b^3 - 3*a^2*b = 2) : 
  a^2 + b^2 = 5 := by sorry

end NUMINAMATH_CALUDE_square_sum_equals_five_l1046_104604


namespace NUMINAMATH_CALUDE_petes_number_l1046_104630

theorem petes_number : ∃ x : ℝ, 4 * (2 * x + 20) = 200 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_petes_number_l1046_104630


namespace NUMINAMATH_CALUDE_dani_pants_count_l1046_104674

/-- Calculate the final number of pants after receiving a certain number of pairs each year for a given period. -/
def final_pants_count (initial_pants : ℕ) (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) : ℕ :=
  initial_pants + pairs_per_year * pants_per_pair * years

/-- Theorem stating that Dani will have 90 pants after 5 years -/
theorem dani_pants_count : final_pants_count 50 4 2 5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_dani_pants_count_l1046_104674


namespace NUMINAMATH_CALUDE_rectangle_to_circle_area_l1046_104690

/-- Given a rectangle with area 200 and length twice its width, 
    the area of the largest circle that can be formed from a string 
    equal to the rectangle's perimeter is 900/π. -/
theorem rectangle_to_circle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let area_rect := w * l
  let perimeter := 2 * (w + l)
  area_rect = 200 → (perimeter^2) / (4 * π) = 900 / π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_circle_area_l1046_104690


namespace NUMINAMATH_CALUDE_set_operations_l1046_104660

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≤ 8 - 2 * x}

-- State the theorem
theorem set_operations :
  (B = {x : ℝ | x ≤ 3}) ∧
  (A ∪ B = {x : ℝ | x < 4}) ∧
  ((Aᶜ) ∩ B = {x : ℝ | x < -1}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1046_104660


namespace NUMINAMATH_CALUDE_incircle_median_intersection_l1046_104620

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  -- Right angle at C
  (B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2) = 0 ∧
  -- AC = 1
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 1

-- Define the median AM
def Median (A B C M : ℝ × ℝ) : Prop :=
  M.1 = (B.1 + C.1) / 2 ∧ M.2 = (B.2 + C.2) / 2

-- Define the incircle
def Incircle (O : ℝ × ℝ) (r : ℝ) (A B C : ℝ × ℝ) : Prop :=
  -- The incircle touches all three sides of the triangle
  ∃ (D E F : ℝ × ℝ),
    ((D.1 - O.1)^2 + (D.2 - O.2)^2 = r^2) ∧
    ((E.1 - O.1)^2 + (E.2 - O.2)^2 = r^2) ∧
    ((F.1 - O.1)^2 + (F.2 - O.2)^2 = r^2) ∧
    -- D is on BC, E is on CA, F is on AB
    ((D.1 - B.1) * (C.2 - B.2) = (D.2 - B.2) * (C.1 - B.1)) ∧
    ((E.1 - C.1) * (A.2 - C.2) = (E.2 - C.2) * (A.1 - C.1)) ∧
    ((F.1 - A.1) * (B.2 - A.2) = (F.2 - A.2) * (B.1 - A.1))

-- Theorem statement
theorem incircle_median_intersection
  (A B C M P Q : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) :
  Triangle A B C →
  Median A B C M →
  Incircle O r A B C →
  -- P and Q are on AM and inside the incircle
  ((P.1 - A.1) * (M.2 - A.2) = (P.2 - A.2) * (M.1 - A.1)) ∧
  ((Q.1 - A.1) * (M.2 - A.2) = (Q.2 - A.2) * (M.1 - A.1)) ∧
  ((P.1 - O.1)^2 + (P.2 - O.2)^2 ≤ r^2) ∧
  ((Q.1 - O.1)^2 + (Q.2 - O.2)^2 ≤ r^2) →
  -- P is between A and Q
  ((P.1 - A.1) * (Q.1 - P.1) + (P.2 - A.2) * (Q.2 - P.2) ≥ 0) ∧
  ((Q.1 - P.1) * (M.1 - Q.1) + (Q.2 - P.2) * (M.2 - Q.2) ≥ 0) →
  -- AP = QM
  ((P.1 - A.1)^2 + (P.2 - A.2)^2 = (M.1 - Q.1)^2 + (M.2 - Q.2)^2) →
  -- Then PQ = √(2√5 - 4)
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 2 * Real.sqrt 5 - 4 :=
sorry

end NUMINAMATH_CALUDE_incircle_median_intersection_l1046_104620


namespace NUMINAMATH_CALUDE_intersection_of_odd_integers_and_open_interval_l1046_104643

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_odd_integers_and_open_interval :
  A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_odd_integers_and_open_interval_l1046_104643


namespace NUMINAMATH_CALUDE_area_of_triangle_QPO_l1046_104641

-- Define the points
variable (A B C D P Q O N M : Point)
-- Define the area of the parallelogram
variable (k : ℝ)

-- Define the conditions
def is_parallelogram (A B C D : Point) : Prop := sorry

def bisects (P Q R : Point) : Prop := sorry

def intersects (L₁ L₂ P : Point) : Prop := sorry

def area (shape : Set Point) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_QPO 
  (h1 : is_parallelogram A B C D)
  (h2 : bisects D N C)
  (h3 : intersects D P B)
  (h4 : bisects C M D)
  (h5 : intersects C Q A)
  (h6 : intersects D P O)
  (h7 : intersects C Q O)
  (h8 : area {A, B, C, D} = k) :
  area {Q, P, O} = 9/8 * k := sorry

end NUMINAMATH_CALUDE_area_of_triangle_QPO_l1046_104641


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1046_104670

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_geometric_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 3 = Real.sqrt (3^x * 3^y) → 1/x + 1/y ≥ 1/a + 1/b) →
  1/a + 1/b = 4 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1046_104670


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l1046_104636

theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l1046_104636


namespace NUMINAMATH_CALUDE_total_savings_theorem_l1046_104665

/-- Represents the savings of a child in various currencies and denominations -/
structure Savings where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  one_dollar_bills : ℕ
  five_dollar_bills : ℕ
  two_dollar_canadian_coins : ℕ
  one_dollar_canadian_coins : ℕ
  five_dollar_canadian_bills : ℕ
  one_pound_uk_coins : ℕ

/-- Conversion rates for different currencies -/
structure ConversionRates where
  british_pound_to_usd : ℚ
  canadian_dollar_to_usd : ℚ

/-- Calculates the total savings in US dollars -/
def calculate_total_savings (teagan_savings : Savings) (rex_savings : Savings) (toni_savings : Savings) (rates : ConversionRates) : ℚ :=
  sorry

/-- Theorem stating the total savings of the three kids -/
theorem total_savings_theorem (teagan_savings rex_savings toni_savings : Savings) (rates : ConversionRates) :
  teagan_savings.pennies = 200 ∧
  teagan_savings.one_dollar_bills = 15 ∧
  teagan_savings.two_dollar_canadian_coins = 13 ∧
  rex_savings.nickels = 100 ∧
  rex_savings.quarters = 45 ∧
  rex_savings.one_pound_uk_coins = 8 ∧
  rex_savings.one_dollar_canadian_coins = 20 ∧
  toni_savings.dimes = 330 ∧
  toni_savings.five_dollar_bills = 12 ∧
  toni_savings.five_dollar_canadian_bills = 7 ∧
  rates.british_pound_to_usd = 138/100 ∧
  rates.canadian_dollar_to_usd = 76/100 →
  calculate_total_savings teagan_savings rex_savings toni_savings rates = 19885/100 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_theorem_l1046_104665


namespace NUMINAMATH_CALUDE_probability_is_four_twentysevenths_l1046_104629

/-- A regular tetrahedron with painted stripes -/
structure StripedTetrahedron where
  /-- The number of faces in a tetrahedron -/
  num_faces : Nat
  /-- The number of possible stripe configurations per face -/
  stripes_per_face : Nat
  /-- The total number of possible stripe configurations -/
  total_configurations : Nat
  /-- The number of configurations that form a continuous stripe -/
  continuous_configurations : Nat

/-- The probability of a continuous stripe encircling the tetrahedron -/
def probability_continuous_stripe (t : StripedTetrahedron) : Rat :=
  t.continuous_configurations / t.total_configurations

/-- Theorem stating the probability of a continuous stripe encircling the tetrahedron -/
theorem probability_is_four_twentysevenths (t : StripedTetrahedron) 
  (h1 : t.num_faces = 4)
  (h2 : t.stripes_per_face = 3)
  (h3 : t.total_configurations = t.stripes_per_face ^ t.num_faces)
  (h4 : t.continuous_configurations = 12) : 
  probability_continuous_stripe t = 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_four_twentysevenths_l1046_104629


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1046_104688

/-- The perimeter of an equilateral triangle, given an isosceles triangle with specific properties -/
theorem equilateral_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun perimeter_isosceles base_isosceles perimeter_equilateral =>
    perimeter_isosceles = 40 ∧
    base_isosceles = 10 ∧
    ∃ (side : ℝ), 
      2 * side + base_isosceles = perimeter_isosceles ∧
      3 * side = perimeter_equilateral ∧
      perimeter_equilateral = 45

/-- Proof of the theorem -/
theorem equilateral_triangle_perimeter_proof :
  ∃ (perimeter_equilateral : ℝ),
    equilateral_triangle_perimeter 40 10 perimeter_equilateral :=
by
  sorry

#check equilateral_triangle_perimeter
#check equilateral_triangle_perimeter_proof

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1046_104688


namespace NUMINAMATH_CALUDE_perpendicular_chords_diameter_l1046_104600

theorem perpendicular_chords_diameter (r : ℝ) (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  a + b = 7 →
  c + d = 8 →
  (a * b = r^2) ∧ (c * d = r^2) →
  2 * r = Real.sqrt 65 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_chords_diameter_l1046_104600


namespace NUMINAMATH_CALUDE_potato_distribution_l1046_104622

theorem potato_distribution (total : ℕ) (gina : ℕ) (left : ℕ) :
  total = 300 →
  gina = 69 →
  left = 47 →
  ∃ (tom : ℕ) (k : ℕ),
    tom = k * gina ∧
    total = gina + tom + (tom / 3) + left →
    tom / gina = 2 :=
by sorry

end NUMINAMATH_CALUDE_potato_distribution_l1046_104622


namespace NUMINAMATH_CALUDE_weight_of_four_cakes_l1046_104678

/-- The weight of a cake in grams -/
def cake_weight : ℕ := sorry

/-- The weight of a piece of bread in grams -/
def bread_weight : ℕ := sorry

/-- Theorem stating the weight of 4 cakes -/
theorem weight_of_four_cakes : 4 * cake_weight = 800 :=
by
  have h1 : 3 * cake_weight + 5 * bread_weight = 1100 := sorry
  have h2 : cake_weight = bread_weight + 100 := sorry
  sorry

#check weight_of_four_cakes

end NUMINAMATH_CALUDE_weight_of_four_cakes_l1046_104678


namespace NUMINAMATH_CALUDE_total_supervisors_count_l1046_104633

/-- The number of buses -/
def num_buses : ℕ := 7

/-- The number of supervisors per bus -/
def supervisors_per_bus : ℕ := 3

/-- The total number of supervisors -/
def total_supervisors : ℕ := num_buses * supervisors_per_bus

theorem total_supervisors_count : total_supervisors = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_supervisors_count_l1046_104633


namespace NUMINAMATH_CALUDE_skew_perpendicular_plane_skew_parallel_perpendicular_l1046_104601

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (skew : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Theorem 1
theorem skew_perpendicular_plane 
  (a b : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : perpendicular a α) : 
  ¬ perpendicular b α := by sorry

-- Theorem 2
theorem skew_parallel_perpendicular 
  (a b l : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : parallel a α) 
  (h3 : parallel b α) 
  (h4 : perpendicular l α) : 
  perpendicularLines l a ∧ perpendicularLines l b := by sorry

end NUMINAMATH_CALUDE_skew_perpendicular_plane_skew_parallel_perpendicular_l1046_104601


namespace NUMINAMATH_CALUDE_fraction_invariance_l1046_104651

theorem fraction_invariance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2008 * (2 * x)) / (2007 * (2 * y)) = (2008 * x) / (2007 * y) := by
sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1046_104651


namespace NUMINAMATH_CALUDE_logarithmic_scales_imply_ohms_law_l1046_104603

/-- Represents a point on the logarithmic scale for resistance, current, or voltage -/
structure LogPoint where
  value : ℝ
  coordinate : ℝ

/-- Represents the scales for resistance, current, and voltage -/
structure Circuit where
  resistance : LogPoint
  current : LogPoint
  voltage : LogPoint

/-- The relationship between the coordinates of resistance, current, and voltage -/
def coordinate_relation (c : Circuit) : Prop :=
  c.current.coordinate + c.voltage.coordinate = 2 * c.resistance.coordinate

/-- The relationship between resistance, current, and voltage values -/
def ohms_law (c : Circuit) : Prop :=
  c.voltage.value = c.current.value * c.resistance.value

/-- The logarithmic scale relationship for resistance -/
def resistance_scale (r : LogPoint) : Prop :=
  r.value = 10^(-2 * r.coordinate)

/-- The logarithmic scale relationship for current -/
def current_scale (i : LogPoint) : Prop :=
  i.value = 10^(i.coordinate)

/-- The logarithmic scale relationship for voltage -/
def voltage_scale (v : LogPoint) : Prop :=
  v.value = 10^(-v.coordinate)

/-- Theorem stating that the logarithmic scales and coordinate relation imply Ohm's law -/
theorem logarithmic_scales_imply_ohms_law (c : Circuit) :
  resistance_scale c.resistance →
  current_scale c.current →
  voltage_scale c.voltage →
  coordinate_relation c →
  ohms_law c :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_scales_imply_ohms_law_l1046_104603


namespace NUMINAMATH_CALUDE_expand_product_l1046_104687

theorem expand_product (x : ℝ) : (3*x - 4) * (2*x + 9) = 6*x^2 + 19*x - 36 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1046_104687


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l1046_104667

/-- Represents the measurement of a square's side length to the nearest centimeter. -/
def MeasuredSideLength : ℝ := 4

/-- The minimum possible actual side length given the measured side length. -/
def MinActualSideLength : ℝ := MeasuredSideLength - 0.5

/-- Calculates the area of a square given its side length. -/
def SquareArea (sideLength : ℝ) : ℝ := sideLength * sideLength

/-- The least possible actual area of the square. -/
def LeastPossibleArea : ℝ := SquareArea MinActualSideLength

theorem least_possible_area_of_square :
  LeastPossibleArea = 12.25 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l1046_104667


namespace NUMINAMATH_CALUDE_buttons_needed_for_shirts_l1046_104685

theorem buttons_needed_for_shirts 
  (shirts_per_kid : ℕ)
  (num_kids : ℕ)
  (buttons_per_shirt : ℕ)
  (h1 : shirts_per_kid = 3)
  (h2 : num_kids = 3)
  (h3 : buttons_per_shirt = 7) :
  shirts_per_kid * num_kids * buttons_per_shirt = 63 :=
by sorry

end NUMINAMATH_CALUDE_buttons_needed_for_shirts_l1046_104685


namespace NUMINAMATH_CALUDE_not_always_both_false_l1046_104666

theorem not_always_both_false (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∧ ¬q) → False :=
sorry

end NUMINAMATH_CALUDE_not_always_both_false_l1046_104666


namespace NUMINAMATH_CALUDE_tan_equation_solution_l1046_104647

theorem tan_equation_solution (x : ℝ) : 
  x = 30 * Real.pi / 180 → 
  Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l1046_104647


namespace NUMINAMATH_CALUDE_factorial_ratio_l1046_104681

theorem factorial_ratio : (Nat.factorial 10) / ((Nat.factorial 7) * (Nat.factorial 3)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1046_104681


namespace NUMINAMATH_CALUDE_carol_carrots_l1046_104609

-- Define the variables
def total_carrots : ℕ := 38 + 7
def mom_carrots : ℕ := 16

-- State the theorem
theorem carol_carrots : total_carrots - mom_carrots = 29 := by
  sorry

end NUMINAMATH_CALUDE_carol_carrots_l1046_104609


namespace NUMINAMATH_CALUDE_pascal_triangle_symmetry_and_sum_l1046_104652

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem pascal_triangle_symmetry_and_sum (n : ℕ) :
  pascal_triangle 48 46 = pascal_triangle 48 2 ∧
  pascal_triangle 48 46 + pascal_triangle 48 2 = 2256 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_symmetry_and_sum_l1046_104652


namespace NUMINAMATH_CALUDE_min_value_sum_l1046_104613

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - x - 2 * y = 0) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l1046_104613


namespace NUMINAMATH_CALUDE_arcade_tickets_l1046_104637

theorem arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) :
  initial_tickets ≥ spent_tickets →
  (initial_tickets - spent_tickets + additional_tickets) = 
    initial_tickets - spent_tickets + additional_tickets :=
by
  sorry

end NUMINAMATH_CALUDE_arcade_tickets_l1046_104637


namespace NUMINAMATH_CALUDE_daves_weight_l1046_104691

/-- Proves Dave's weight given the conditions from the problem -/
theorem daves_weight (dave_weight : ℝ) (dave_bench : ℝ) (craig_bench : ℝ) (mark_bench : ℝ) :
  dave_bench = 3 * dave_weight →
  craig_bench = 0.2 * dave_bench →
  mark_bench = craig_bench - 50 →
  mark_bench = 55 →
  dave_weight = 175 := by
sorry

end NUMINAMATH_CALUDE_daves_weight_l1046_104691


namespace NUMINAMATH_CALUDE_intersection_exists_l1046_104689

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem intersection_exists : ∃ z, z ∈ A ∩ B := by
  sorry

end NUMINAMATH_CALUDE_intersection_exists_l1046_104689


namespace NUMINAMATH_CALUDE_hoseok_addition_l1046_104682

theorem hoseok_addition (x : ℤ) : x + 56 = 110 → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_addition_l1046_104682


namespace NUMINAMATH_CALUDE_freshman_percentage_l1046_104684

theorem freshman_percentage (total_students : ℝ) (freshman : ℝ) 
  (h1 : freshman > 0)
  (h2 : total_students > 0)
  (h3 : (0.2 * 0.4 * freshman) / total_students = 0.04) :
  freshman / total_students = 0.5 := by
sorry

end NUMINAMATH_CALUDE_freshman_percentage_l1046_104684


namespace NUMINAMATH_CALUDE_line_parameterization_l1046_104695

/-- Given a line y = 2x - 30 parameterized by (x, y) = (f(t), 20t - 10), 
    prove that f(t) = 10t + 10 -/
theorem line_parameterization (f : ℝ → ℝ) : 
  (∀ t : ℝ, 2 * (f t) - 30 = 20 * t - 10) → 
  (∀ t : ℝ, f t = 10 * t + 10) := by
  sorry

end NUMINAMATH_CALUDE_line_parameterization_l1046_104695


namespace NUMINAMATH_CALUDE_gaskets_sold_l1046_104607

/-- Calculates the total cost of gasket packages --/
def totalCost (packages : ℕ) : ℚ :=
  if packages ≤ 10 then
    25 * packages
  else
    250 + 20 * (packages - 10)

/-- Proves that 65 packages of gaskets were sold given the conditions --/
theorem gaskets_sold : ∃ (packages : ℕ), packages > 10 ∧ totalCost packages = 1340 := by
  sorry

#eval totalCost 65

end NUMINAMATH_CALUDE_gaskets_sold_l1046_104607


namespace NUMINAMATH_CALUDE_emily_garden_seeds_l1046_104698

theorem emily_garden_seeds (total_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : small_gardens = 3)
  (h4 : small_gardens > 0) :
  (total_seeds - big_garden_seeds) / small_gardens = 4 := by
sorry

end NUMINAMATH_CALUDE_emily_garden_seeds_l1046_104698


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1046_104650

/-- A parabola is a function of the form f(x) = a(x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifting a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k - dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 2 ∧ p.k = 1 →
  let p' := shift p 2 1
  p'.a = 3 ∧ p'.h = 0 ∧ p'.k = 0 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1046_104650


namespace NUMINAMATH_CALUDE_tom_purchases_amount_l1046_104646

/-- Calculates the amount available for purchases given hourly rate, work hours, and savings rate. -/
def amountAvailableForPurchases (hourlyRate : ℚ) (workHours : ℕ) (savingsRate : ℚ) : ℚ :=
  let totalEarnings := hourlyRate * workHours
  let savingsAmount := savingsRate * totalEarnings
  totalEarnings - savingsAmount

/-- Proves that Tom's amount available for purchases is $181.35 -/
theorem tom_purchases_amount :
  let hourlyRate : ℚ := 13/2  -- $6.50
  let workHours : ℕ := 31
  let savingsRate : ℚ := 1/10  -- 10%
  amountAvailableForPurchases hourlyRate workHours savingsRate = 36270/200  -- $181.35
  := by sorry

end NUMINAMATH_CALUDE_tom_purchases_amount_l1046_104646


namespace NUMINAMATH_CALUDE_trailing_zeros_of_product_trailing_zeros_of_product_is_90_l1046_104605

/-- The number of trailing zeros in the product of 20^50 and 50^20 -/
theorem trailing_zeros_of_product : ℕ :=
  let a := 20^50
  let b := 50^20
  let product := a * b
  90

/-- Proof that the number of trailing zeros in the product of 20^50 and 50^20 is 90 -/
theorem trailing_zeros_of_product_is_90 :
  trailing_zeros_of_product = 90 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_of_product_trailing_zeros_of_product_is_90_l1046_104605


namespace NUMINAMATH_CALUDE_total_cookies_l1046_104632

theorem total_cookies (cookies_per_bag : ℕ) (number_of_bags : ℕ) : 
  cookies_per_bag = 41 → number_of_bags = 53 → cookies_per_bag * number_of_bags = 2173 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1046_104632


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l1046_104692

theorem consecutive_integers_average (c d : ℤ) : 
  (c > 0) →
  (d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7) →
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7 = c + 6) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l1046_104692


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1046_104680

/-- Given an arithmetic sequence with first four terms a, x, b, 2x, prove that a/b = 1/3 -/
theorem arithmetic_sequence_ratio (a x b : ℝ) :
  (x - a = b - x) ∧ (b - x = 2 * x - b) → a / b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1046_104680


namespace NUMINAMATH_CALUDE_prime_between_squares_l1046_104661

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ x : ℕ, p = x^2 + 5 ∧ p + 9 = (x + 1)^2 := by
sorry

end NUMINAMATH_CALUDE_prime_between_squares_l1046_104661


namespace NUMINAMATH_CALUDE_M_in_fourth_quadrant_l1046_104664

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point M -/
def M : Point :=
  { x := 2, y := -5 }

/-- Theorem stating that M is in the fourth quadrant -/
theorem M_in_fourth_quadrant : is_in_fourth_quadrant M := by
  sorry


end NUMINAMATH_CALUDE_M_in_fourth_quadrant_l1046_104664


namespace NUMINAMATH_CALUDE_third_grade_students_l1046_104610

/-- The number of story books to be distributed -/
def total_books : ℕ := 90

/-- Proves that the number of third-grade students is 60 -/
theorem third_grade_students :
  ∃ n : ℕ, n > 0 ∧ n < total_books ∧ total_books - n = n / 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_third_grade_students_l1046_104610


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1046_104612

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^12 + 8*x^11 + 15*x^10 + 2023*x^9 - 1500*x^8

-- State the theorem
theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1046_104612


namespace NUMINAMATH_CALUDE_six_digit_divisibility_difference_l1046_104616

def six_digit_lower_bound : Nat := 100000
def six_digit_upper_bound : Nat := 999999

def count_divisible (n : Nat) : Nat :=
  (six_digit_upper_bound / n) - (six_digit_lower_bound / n)

def a : Nat := count_divisible 13 - count_divisible (13 * 17)
def b : Nat := count_divisible 17 - count_divisible (13 * 17)

theorem six_digit_divisibility_difference : a - b = 16290 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_difference_l1046_104616


namespace NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l1046_104693

/-- Proves that the ratio of a dishwasher's hourly wage to a manager's hourly wage is 0.5 -/
theorem dishwasher_manager_wage_ratio :
  ∀ (manager_wage chef_wage dishwasher_wage : ℝ),
    manager_wage = 8.5 →
    chef_wage = manager_wage - 3.4 →
    chef_wage = dishwasher_wage * 1.2 →
    dishwasher_wage / manager_wage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_dishwasher_manager_wage_ratio_l1046_104693


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l1046_104676

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ 11) ^ (1/8) := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l1046_104676


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1046_104614

theorem smallest_sum_of_sequence (X Y Z W : ℕ) : 
  X > 0 → Y > 0 → Z > 0 →
  (∃ d : ℤ, Z - Y = Y - X) →  -- arithmetic sequence condition
  (∃ r : ℚ, Z = r * Y ∧ W = r * Z) →  -- geometric sequence condition
  (Z : ℚ) / Y = 7 / 4 →
  X + Y + Z + W ≥ 97 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l1046_104614


namespace NUMINAMATH_CALUDE_average_run_time_l1046_104634

/-- Represents the average minutes run per day for each grade -/
structure GradeRunTime where
  sixth : ℝ
  seventh : ℝ
  eighth : ℝ

/-- Represents the number of students in each grade -/
structure GradePopulation where
  seventh : ℝ
  sixth : ℝ
  eighth : ℝ

/-- Represents the number of days each grade runs per week -/
structure RunDays where
  sixth : ℕ
  seventh : ℕ
  eighth : ℕ

theorem average_run_time 
  (run_time : GradeRunTime)
  (population : GradePopulation)
  (days : RunDays)
  (h1 : run_time.sixth = 10)
  (h2 : run_time.seventh = 12)
  (h3 : run_time.eighth = 8)
  (h4 : population.sixth = 3 * population.seventh)
  (h5 : population.eighth = population.seventh / 2)
  (h6 : days.sixth = 2)
  (h7 : days.seventh = 2)
  (h8 : days.eighth = 1) :
  (run_time.sixth * population.sixth * days.sixth +
   run_time.seventh * population.seventh * days.seventh +
   run_time.eighth * population.eighth * days.eighth) /
  (population.sixth + population.seventh + population.eighth) /
  7 = 176 / 9 := by
  sorry


end NUMINAMATH_CALUDE_average_run_time_l1046_104634


namespace NUMINAMATH_CALUDE_function_bound_l1046_104635

/-- Given real-valued functions f and g defined on the real line,
    if f(x + y) + f(x - y) = 2f(x)g(y) for all x and y,
    f is not identically zero, and |f(x)| ≤ 1 for all x,
    then |g(x)| ≤ 1 for all x. -/
theorem function_bound (f g : ℝ → ℝ)
    (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
    (h2 : ∃ x, f x ≠ 0)
    (h3 : ∀ x, |f x| ≤ 1) :
    ∀ x, |g x| ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_bound_l1046_104635


namespace NUMINAMATH_CALUDE_find_d_l1046_104631

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x - 3

-- State the theorem
theorem find_d (c : ℝ) :
  (∃ d : ℝ, ∀ x : ℝ, f c (g c x) = 15 * x + d) →
  (∃ d : ℝ, ∀ x : ℝ, f c (g c x) = 15 * x + d ∧ d = -12) :=
by sorry

end NUMINAMATH_CALUDE_find_d_l1046_104631


namespace NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l1046_104686

theorem z_greater_than_w_by_50_percent 
  (w x y z : ℝ) 
  (hw : w = 0.6 * x) 
  (hx : x = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
sorry

end NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l1046_104686


namespace NUMINAMATH_CALUDE_system_solution_l1046_104696

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -8) ∧
    (5 * x + 9 * y = -18) ∧
    (x = -14/3) ∧
    (y = -32/9) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1046_104696


namespace NUMINAMATH_CALUDE_rational_power_floor_theorem_l1046_104672

theorem rational_power_floor_theorem (x : ℚ) : 
  (∃ (a : ℤ), a ≥ 1 ∧ x^(⌊x⌋) = a / 2) ↔ (∃ (n : ℤ), x = n) ∨ x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_rational_power_floor_theorem_l1046_104672


namespace NUMINAMATH_CALUDE_geometric_series_equality_l1046_104648

def C (n : ℕ) : ℚ := 512 * (1 - (1/2)^n) / (1 - 1/2)

def D (n : ℕ) : ℚ := 1536 * (1 - (1/(-2))^n) / (1 + 1/2)

theorem geometric_series_equality :
  ∃ (n : ℕ), n > 0 ∧ C n = D n ∧ ∀ (m : ℕ), 0 < m ∧ m < n → C m ≠ D m :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l1046_104648


namespace NUMINAMATH_CALUDE_remainder_problem_l1046_104694

theorem remainder_problem (x : ℤ) : x % 62 = 7 → (x + 11) % 31 = 18 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1046_104694


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l1046_104626

/-- Given a canoe's upstream speed and the stream speed, calculates the downstream speed. -/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  2 * upstream_speed + 3 * stream_speed

/-- Theorem stating that for a canoe with upstream speed 8 km/hr and stream speed 2 km/hr, 
    the downstream speed is 12 km/hr. -/
theorem canoe_downstream_speed :
  let upstream_speed := 8
  let stream_speed := 2
  downstream_speed upstream_speed stream_speed = 12 := by
  sorry

end NUMINAMATH_CALUDE_canoe_downstream_speed_l1046_104626


namespace NUMINAMATH_CALUDE_move_left_result_l1046_104639

/-- Moving a point 2 units to the left in a Cartesian coordinate system. -/
def moveLeft (x y : ℝ) : ℝ × ℝ := (x - 2, y)

/-- The theorem stating that moving (-2, 3) 2 units to the left results in (-4, 3). -/
theorem move_left_result : moveLeft (-2) 3 = (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_move_left_result_l1046_104639


namespace NUMINAMATH_CALUDE_absent_present_probability_l1046_104671

theorem absent_present_probability (course_days : ℕ) (avg_absent_days : ℕ) : 
  course_days = 40 → 
  avg_absent_days = 1 → 
  (39 : ℚ) / 800 = (course_days - avg_absent_days) / (course_days^2) * 2 := by
  sorry

end NUMINAMATH_CALUDE_absent_present_probability_l1046_104671


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1046_104621

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}

theorem complement_of_A_wrt_U : 
  {x ∈ U | x ∉ A} = {4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_U_l1046_104621


namespace NUMINAMATH_CALUDE_min_area_triangle_AOB_l1046_104638

/-- Given a line l: mx + ny - 1 = 0 intersecting the x-axis at A and y-axis at B,
    and forming a chord of length 2 with the circle x² + y² = 4,
    the minimum area of triangle AOB is 3. -/
theorem min_area_triangle_AOB (m n : ℝ) :
  ∃ (A B : ℝ × ℝ),
    (m * A.1 + n * A.2 - 1 = 0) ∧
    (m * B.1 + n * B.2 - 1 = 0) ∧
    (A.2 = 0) ∧
    (B.1 = 0) ∧
    (∃ (C D : ℝ × ℝ),
      (m * C.1 + n * C.2 - 1 = 0) ∧
      (m * D.1 + n * D.2 - 1 = 0) ∧
      (C.1^2 + C.2^2 = 4) ∧
      (D.1^2 + D.2^2 = 4) ∧
      ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 4)) →
  ∃ (area_min : ℝ),
    (∀ (A' B' : ℝ × ℝ),
      (m * A'.1 + n * A'.2 - 1 = 0) →
      (A'.2 = 0) →
      (m * B'.1 + n * B'.2 - 1 = 0) →
      (B'.1 = 0) →
      area_min ≤ (1/2) * A'.1 * B'.2) ∧
    area_min = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_area_triangle_AOB_l1046_104638


namespace NUMINAMATH_CALUDE_sqrt_two_thirds_same_type_as_sqrt6_l1046_104617

-- Define what it means for a real number to be of the same type as √6
def same_type_as_sqrt6 (x : ℝ) : Prop :=
  ∃ (a b : ℚ), x = a * Real.sqrt 2 * b * Real.sqrt 3

-- State the theorem
theorem sqrt_two_thirds_same_type_as_sqrt6 :
  same_type_as_sqrt6 (Real.sqrt (2/3)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_two_thirds_same_type_as_sqrt6_l1046_104617


namespace NUMINAMATH_CALUDE_two_element_subsets_of_three_element_set_l1046_104659

theorem two_element_subsets_of_three_element_set :
  let S : Finset Int := {-1, 0, 2}
  (Finset.filter (fun M => M.card = 2) (S.powerset)).card = 3 := by
  sorry

end NUMINAMATH_CALUDE_two_element_subsets_of_three_element_set_l1046_104659


namespace NUMINAMATH_CALUDE_relay_team_arrangements_l1046_104683

/-- The number of possible arrangements for a relay team --/
def relay_arrangements (n : ℕ) : ℕ :=
  Nat.factorial (n - 1)

/-- Theorem: For a 5-person relay team with one fixed runner,
    there are 24 possible arrangements --/
theorem relay_team_arrangements :
  relay_arrangements 5 = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_arrangements_l1046_104683


namespace NUMINAMATH_CALUDE_count_squarish_numbers_l1046_104697

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_two_digit_perfect_square (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_perfect_square n

def is_squarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  is_perfect_square n ∧
  n % 16 = 0 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  is_two_digit_perfect_square (n / 10000) ∧
  is_two_digit_perfect_square ((n / 100) % 100) ∧
  is_two_digit_perfect_square (n % 100)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_squarish_numbers_l1046_104697


namespace NUMINAMATH_CALUDE_binomial_60_2_l1046_104615

theorem binomial_60_2 : Nat.choose 60 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_2_l1046_104615
