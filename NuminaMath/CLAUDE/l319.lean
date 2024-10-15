import Mathlib

namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l319_31919

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l319_31919


namespace NUMINAMATH_CALUDE_power_of_product_cube_l319_31916

theorem power_of_product_cube (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_cube_l319_31916


namespace NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l319_31992

theorem unit_circle_point_x_coordinate 
  (P : ℝ × ℝ) (α : ℝ) 
  (h1 : P.1^2 + P.2^2 = 1) 
  (h2 : P.1 = Real.cos α) 
  (h3 : P.2 = Real.sin α) 
  (h4 : π/3 < α ∧ α < 5*π/6) 
  (h5 : Real.sin (α + π/6) = 3/5) : 
  P.1 = (3 - 4*Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_unit_circle_point_x_coordinate_l319_31992


namespace NUMINAMATH_CALUDE_f_properties_l319_31925

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem f_properties :
  ∃ (T : ℝ) (max_val : ℝ) (k : ℤ → ℝ),
    (∀ x, f (x + T) = f x) ∧ 
    (∀ y, y > 0 → (∀ x, f (x + y) = f x) → y ≥ T) ∧
    (T = 2 * Real.pi) ∧
    (∀ x, f x ≤ max_val) ∧
    (max_val = 2) ∧
    (∀ n, f (k n) = max_val) ∧
    (∀ n, k n = 2 * n * Real.pi + Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l319_31925


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l319_31980

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l319_31980


namespace NUMINAMATH_CALUDE_kelly_found_games_l319_31961

def initial_games : ℕ := 80
def games_to_give_away : ℕ := 105
def games_left : ℕ := 6

theorem kelly_found_games : 
  ∃ (found_games : ℕ), 
    initial_games + found_games = games_to_give_away + games_left ∧ 
    found_games = 31 :=
by sorry

end NUMINAMATH_CALUDE_kelly_found_games_l319_31961


namespace NUMINAMATH_CALUDE_slope_of_line_l319_31958

/-- The slope of a line given by the equation y/4 - x/5 = 2 is 4/5 -/
theorem slope_of_line (x y : ℝ) :
  y / 4 - x / 5 = 2 → (∃ b : ℝ, y = 4 / 5 * x + b) :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l319_31958


namespace NUMINAMATH_CALUDE_average_marks_of_passed_candidates_l319_31987

theorem average_marks_of_passed_candidates 
  (total_candidates : ℕ) 
  (overall_average : ℚ) 
  (failed_average : ℚ) 
  (passed_count : ℕ) 
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : failed_average = 15)
  (h4 : passed_count = 100) :
  (total_candidates * overall_average - (total_candidates - passed_count) * failed_average) / passed_count = 39 := by
sorry

end NUMINAMATH_CALUDE_average_marks_of_passed_candidates_l319_31987


namespace NUMINAMATH_CALUDE_remainder_invariance_l319_31966

theorem remainder_invariance (S A : ℤ) (K : ℤ) : 
  S % A = (S + A * K) % A := by sorry

end NUMINAMATH_CALUDE_remainder_invariance_l319_31966


namespace NUMINAMATH_CALUDE_purely_imaginary_z_and_z_plus_one_squared_l319_31965

theorem purely_imaginary_z_and_z_plus_one_squared (z : ℂ) :
  (z.re = 0) → ((z + 1)^2).re = 0 → (z = Complex.I ∨ z = -Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_and_z_plus_one_squared_l319_31965


namespace NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l319_31991

def hulk_jump (n : ℕ) : ℝ :=
  2 * (3 : ℝ) ^ (n - 1)

theorem hulk_jump_exceeds_500 :
  (∀ k < 7, hulk_jump k ≤ 500) ∧ hulk_jump 7 > 500 := by
  sorry

end NUMINAMATH_CALUDE_hulk_jump_exceeds_500_l319_31991


namespace NUMINAMATH_CALUDE_divisor_sum_ratio_l319_31996

def N : ℕ := 48 * 49 * 75 * 343

def sum_of_divisors (n : ℕ) : ℕ := sorry

def sum_of_divisors_multiple_of_three (n : ℕ) : ℕ := sorry

def sum_of_divisors_not_multiple_of_three (n : ℕ) : ℕ := sorry

theorem divisor_sum_ratio :
  ∃ (a b : ℕ), 
    (sum_of_divisors_multiple_of_three N) * b = (sum_of_divisors_not_multiple_of_three N) * a ∧
    a ≠ 0 ∧ b ≠ 0 ∧ 
    (∀ (c d : ℕ), c ≠ 0 → d ≠ 0 → 
      (sum_of_divisors_multiple_of_three N) * d = (sum_of_divisors_not_multiple_of_three N) * c →
      a ≤ c ∧ b ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_divisor_sum_ratio_l319_31996


namespace NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l319_31939

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_times_abs_even_is_odd
  (f g : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_even : isEven g) :
  isOdd (fun x ↦ f x * |g x|) := by
  sorry

end NUMINAMATH_CALUDE_odd_times_abs_even_is_odd_l319_31939


namespace NUMINAMATH_CALUDE_candy_distribution_l319_31993

theorem candy_distribution (x y n : ℕ) : 
  y + n = 4 * (x - n) →
  x + 90 = 5 * (y - 90) →
  y ≥ 115 →
  (∀ y' : ℕ, y' ≥ 115 → y' + n = 4 * (x - n) → x + 90 = 5 * (y' - 90) → y ≤ y') →
  y = 115 ∧ x = 35 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l319_31993


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l319_31917

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a ≠ 0 ∧ b ≠ 0) →  -- y is reverse of x
  (∃ a b : ℕ, x = 10 * a + b ∧ a + b = 8) →  -- sum of digits of x is 8
  x^2 - y^2 = n^2 →  -- x^2 - y^2 = n^2
  x + y + n = 144 := by
sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l319_31917


namespace NUMINAMATH_CALUDE_ranch_minimum_animals_l319_31967

theorem ranch_minimum_animals : ∀ (ponies horses : ℕ),
  ponies > 0 →
  horses = ponies + 4 →
  (5 * ponies) % 6 = 0 →
  (10 * ponies) % 18 = 0 →
  ponies + horses ≥ 40 ∧
  ∀ (p h : ℕ), p > 0 → h = p + 4 → (5 * p) % 6 = 0 → (10 * p) % 18 = 0 → p + h ≥ ponies + horses :=
by sorry

end NUMINAMATH_CALUDE_ranch_minimum_animals_l319_31967


namespace NUMINAMATH_CALUDE_adam_figurines_l319_31931

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of basswood blocks Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines

theorem adam_figurines : total_figurines = 245 := by
  sorry

end NUMINAMATH_CALUDE_adam_figurines_l319_31931


namespace NUMINAMATH_CALUDE_inequality_proof_l319_31943

theorem inequality_proof (a b c : ℝ) (ha : a = (Real.log 2) / 2) 
  (hb : b = (Real.log Real.pi) / Real.pi) (hc : c = (Real.log 5) / 5) : 
  b > a ∧ a > c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l319_31943


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l319_31900

theorem power_of_product_equals_product_of_powers (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l319_31900


namespace NUMINAMATH_CALUDE_square_area_with_circles_l319_31904

/-- The area of a square containing six circles arranged in two rows and three columns, 
    where each circle has a radius of 3 units. -/
theorem square_area_with_circles (radius : ℝ) (h : radius = 3) : 
  (3 * (2 * radius))^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_circles_l319_31904


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l319_31937

theorem greatest_x_given_lcm (x : ℕ+) : 
  (Nat.lcm x (Nat.lcm 12 18) = 180) → x ≤ 180 ∧ ∃ y : ℕ+, y > 180 → Nat.lcm y (Nat.lcm 12 18) > 180 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l319_31937


namespace NUMINAMATH_CALUDE_line_slope_is_two_l319_31949

/-- The slope of a line given by the equation 3y - 6x = 9 is 2 -/
theorem line_slope_is_two : 
  ∀ (x y : ℝ), 3 * y - 6 * x = 9 → (∃ b : ℝ, y = 2 * x + b) :=
by sorry

end NUMINAMATH_CALUDE_line_slope_is_two_l319_31949


namespace NUMINAMATH_CALUDE_vector_equality_transitivity_l319_31905

variable {V : Type*} [AddCommGroup V]

theorem vector_equality_transitivity (a b c : V) :
  a = b → b = c → a = c := by sorry

end NUMINAMATH_CALUDE_vector_equality_transitivity_l319_31905


namespace NUMINAMATH_CALUDE_intersection_product_l319_31908

/-- Curve C in polar coordinates -/
def curve_C (ρ θ : ℝ) : Prop :=
  ρ^2 - 2*ρ*Real.cos θ - 3 = 0

/-- Line l in polar coordinates -/
def line_l (k : ℝ) (θ : ℝ) : Prop :=
  k > 0 ∧ θ ∈ Set.Ioo 0 (Real.pi / 2)

/-- Intersection points of curve C and line l -/
def intersection_points (ρ₁ ρ₂ θ : ℝ) : Prop :=
  curve_C ρ₁ θ ∧ curve_C ρ₂ θ ∧ ∃ k, line_l k θ

theorem intersection_product (ρ₁ ρ₂ θ : ℝ) :
  intersection_points ρ₁ ρ₂ θ → |ρ₁ * ρ₂| = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_l319_31908


namespace NUMINAMATH_CALUDE_dans_tshirt_production_rate_l319_31906

/-- The time it takes Dan to make one t-shirt in the first hour -/
def time_per_shirt_first_hour (total_shirts : ℕ) (second_hour_rate : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  let second_hour_shirts := minutes_per_hour / second_hour_rate
  let first_hour_shirts := total_shirts - second_hour_shirts
  minutes_per_hour / first_hour_shirts

/-- Theorem stating that it takes Dan 12 minutes to make one t-shirt in the first hour -/
theorem dans_tshirt_production_rate :
  time_per_shirt_first_hour 15 6 60 = 12 :=
by
  sorry

#eval time_per_shirt_first_hour 15 6 60

end NUMINAMATH_CALUDE_dans_tshirt_production_rate_l319_31906


namespace NUMINAMATH_CALUDE_no_solution_equation_l319_31901

theorem no_solution_equation :
  ¬∃ (x : ℝ), (1 / (x + 11) + 1 / (x + 8) = 1 / (x + 12) + 1 / (x + 7)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_equation_l319_31901


namespace NUMINAMATH_CALUDE_sports_club_overlap_l319_31945

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) : 
  total = 28 → badminton = 17 → tennis = 19 → neither = 2 →
  badminton + tennis - total + neither = 10 := by
sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l319_31945


namespace NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_D_to_D_l319_31934

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) : 
  let D : ℝ × ℝ := (x, y)
  let D' : ℝ × ℝ := (x, -y)
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point D(2, 4) --/
theorem distance_D_to_D'_is_8 : 
  let D : ℝ × ℝ := (2, 4)
  let D' : ℝ × ℝ := (2, -4)
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_reflection_over_x_axis_distance_D_to_D_l319_31934


namespace NUMINAMATH_CALUDE_f_bounded_iff_alpha_in_unit_interval_l319_31959

/-- The function f defined on pairs of nonnegative integers -/
noncomputable def f (α : ℝ) : ℕ → ℕ → ℝ
| 0, 0 => 1
| m, 0 => 0
| 0, n => 0
| (m+1), (n+1) => α * f α m (n+1) + (1 - α) * f α m n

/-- The theorem statement -/
theorem f_bounded_iff_alpha_in_unit_interval (α : ℝ) :
  (∀ m n : ℕ, |f α m n| < 1989) ↔ 0 < α ∧ α < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_bounded_iff_alpha_in_unit_interval_l319_31959


namespace NUMINAMATH_CALUDE_cistern_filling_time_l319_31915

/-- The time it takes to fill a cistern without a leak, given that:
    1. With a leak, it takes T + 2 hours to fill
    2. When full, it takes 24 hours to empty due to the leak -/
theorem cistern_filling_time (T : ℝ) : 
  (∀ (t : ℝ), t > 0 → (1 / T - 1 / (T + 2) = 1 / 24)) → 
  T = 6 := by
sorry

end NUMINAMATH_CALUDE_cistern_filling_time_l319_31915


namespace NUMINAMATH_CALUDE_sum_and_count_equals_1271_l319_31941

/-- The sum of integers from 50 to 70, inclusive -/
def x : ℕ := (List.range 21).map (· + 50) |>.sum

/-- The number of even integers from 50 to 70, inclusive -/
def y : ℕ := (List.range 21).map (· + 50) |>.filter (· % 2 = 0) |>.length

/-- The theorem stating that x + y equals 1271 -/
theorem sum_and_count_equals_1271 : x + y = 1271 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_equals_1271_l319_31941


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_neg_one_l319_31973

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x+m)(x+1) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (x + m) * (x + 1)

/-- If f(x) = (x+m)(x+1) is an even function, then m = -1 -/
theorem even_function_implies_m_equals_neg_one :
  ∀ m : ℝ, IsEven (f m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_neg_one_l319_31973


namespace NUMINAMATH_CALUDE_min_value_sum_l319_31995

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 19 / x + 98 / y = 1) :
  x + y ≥ 203 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_l319_31995


namespace NUMINAMATH_CALUDE_cylinder_height_l319_31954

/-- Given a cylinder with the following properties:
  * AB is the diameter of the lower base
  * A₁B₁ is a chord of the upper base, parallel to AB
  * The plane passing through AB and A₁B₁ forms an acute angle α with the lower base
  * The line AB₁ forms an angle β with the lower base
  * R is the radius of the base of the cylinder
  * A and A₁ lie on the same side of the line passing through the midpoints of AB and A₁B₁

  Prove that the height of the cylinder is equal to the given expression. -/
theorem cylinder_height (R α β : ℝ) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) :
  ∃ (height : ℝ), height = 2 * R * Real.tan β * (Real.sqrt (Real.sin (α + β) * Real.sin (α - β))) / (Real.sin α * Real.cos β) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_l319_31954


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l319_31922

-- Define a tetrahedron with edge length 2
def Tetrahedron := {edge_length : ℝ // edge_length = 2}

-- Define the surface area of a tetrahedron
noncomputable def surfaceArea (t : Tetrahedron) : ℝ :=
  4 * Real.sqrt 3

-- Theorem statement
theorem tetrahedron_surface_area (t : Tetrahedron) :
  surfaceArea t = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l319_31922


namespace NUMINAMATH_CALUDE_rabbit_speed_l319_31927

/-- Proves that a rabbit catching up to a cat in 1 hour, given the cat's speed and head start, has a speed of 25 mph. -/
theorem rabbit_speed (cat_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  cat_speed = 20 →
  head_start = 0.25 →
  catch_up_time = 1 →
  let rabbit_speed := (cat_speed * (catch_up_time + head_start)) / catch_up_time
  rabbit_speed = 25 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_speed_l319_31927


namespace NUMINAMATH_CALUDE_bobby_average_increase_l319_31990

/-- Represents Bobby's deadlift capabilities and progress --/
structure DeadliftProgress where
  initial_weight : ℕ  -- Initial deadlift weight at age 13
  final_weight : ℕ    -- Final deadlift weight at age 18
  initial_age : ℕ     -- Age when initial weight was lifted
  final_age : ℕ       -- Age when final weight was lifted

/-- Calculates the average yearly increase in deadlift weight --/
def average_yearly_increase (progress : DeadliftProgress) : ℚ :=
  (progress.final_weight - progress.initial_weight : ℚ) / (progress.final_age - progress.initial_age)

/-- Bobby's actual deadlift progress --/
def bobby_progress : DeadliftProgress := {
  initial_weight := 300,
  final_weight := 850,
  initial_age := 13,
  final_age := 18
}

/-- Theorem stating that Bobby's average yearly increase in deadlift weight is 110 pounds --/
theorem bobby_average_increase : 
  average_yearly_increase bobby_progress = 110 := by
  sorry

end NUMINAMATH_CALUDE_bobby_average_increase_l319_31990


namespace NUMINAMATH_CALUDE_power_multiplication_l319_31969

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l319_31969


namespace NUMINAMATH_CALUDE_slope_base_extension_l319_31928

/-- Extension of slope base to change inclination angle -/
theorem slope_base_extension (slope_length : ℝ) (initial_angle final_angle : ℝ) 
  (h_slope : slope_length = 1)
  (h_initial : initial_angle = 20 * π / 180)
  (h_final : final_angle = 10 * π / 180) :
  let extension := slope_length
  extension = 1 := by sorry

end NUMINAMATH_CALUDE_slope_base_extension_l319_31928


namespace NUMINAMATH_CALUDE_set_equality_l319_31953

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 3)}
def B : Set ℝ := {x | x ≤ -1}

-- Define the set we want to prove is equal to the complement of A ∪ B
def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- State the theorem
theorem set_equality : C = (Set.univ : Set ℝ) \ (A ∪ B) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l319_31953


namespace NUMINAMATH_CALUDE_officer_selection_with_past_officer_l319_31963

/- Given conditions -/
def total_candidates : ℕ := 20
def past_officers : ℕ := 8
def positions_available : ℕ := 10

/- Theorem to prove -/
theorem officer_selection_with_past_officer :
  (Nat.choose total_candidates positions_available) - 
  (Nat.choose (total_candidates - past_officers) positions_available) = 184690 := by
  sorry

end NUMINAMATH_CALUDE_officer_selection_with_past_officer_l319_31963


namespace NUMINAMATH_CALUDE_function_range_l319_31999

theorem function_range (x : ℝ) (h : x > 1) : 
  let y := x + 1 / (x - 1)
  (∀ x > 1, y ≥ 3) ∧ (∃ x > 1, y = 3) :=
by sorry

end NUMINAMATH_CALUDE_function_range_l319_31999


namespace NUMINAMATH_CALUDE_min_grid_size_l319_31932

theorem min_grid_size (k : ℝ) (h : k > 0.9999) :
  (∃ n : ℕ, n ≥ 51 ∧ 4 * n * (n - 1) * k * (1 - k) = k) ∧
  (∀ m : ℕ, m < 51 → 4 * m * (m - 1) * k * (1 - k) ≠ k) := by
  sorry

end NUMINAMATH_CALUDE_min_grid_size_l319_31932


namespace NUMINAMATH_CALUDE_third_day_temp_is_two_l319_31985

/-- The temperature on the third day of a sequence of 8 days, given other temperatures and the mean -/
def third_day_temperature (t1 t2 t4 t5 t6 t7 t8 mean : ℚ) : ℚ :=
  let sum := t1 + t2 + t4 + t5 + t6 + t7 + t8
  8 * mean - sum

theorem third_day_temp_is_two :
  let t1 := -6
  let t2 := -3
  let t4 := -6
  let t5 := 2
  let t6 := 4
  let t7 := 3
  let t8 := 0
  let mean := -0.5
  third_day_temperature t1 t2 t4 t5 t6 t7 t8 mean = 2 := by
  sorry

#eval third_day_temperature (-6) (-3) (-6) 2 4 3 0 (-0.5)

end NUMINAMATH_CALUDE_third_day_temp_is_two_l319_31985


namespace NUMINAMATH_CALUDE_percent_difference_z_w_l319_31936

theorem percent_difference_z_w (y x w z : ℝ) 
  (hw : w = 0.6 * x)
  (hx : x = 0.6 * y)
  (hz : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_z_w_l319_31936


namespace NUMINAMATH_CALUDE_factor_w4_minus_81_factors_are_monic_real_polynomials_l319_31962

theorem factor_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by sorry

theorem factors_are_monic_real_polynomials :
  ∀ w : ℝ, 
    (∃ a b c : ℝ, (w - 3) = w + a ∧ (w + 3) = w + b ∧ (w^2 + 9) = w^2 + c) := by sorry

end NUMINAMATH_CALUDE_factor_w4_minus_81_factors_are_monic_real_polynomials_l319_31962


namespace NUMINAMATH_CALUDE_third_trial_point_l319_31940

/-- The 0.618 method for optimization --/
def golden_ratio : ℝ := 0.618

/-- The lower bound of the initial range --/
def lower_bound : ℝ := 100

/-- The upper bound of the initial range --/
def upper_bound : ℝ := 1100

/-- Calculate the first trial point --/
def x₁ : ℝ := lower_bound + golden_ratio * (upper_bound - lower_bound)

/-- Calculate the second trial point --/
def x₂ : ℝ := lower_bound + (upper_bound - x₁)

/-- Calculate the third trial point --/
def x₃ : ℝ := lower_bound + golden_ratio * (x₂ - lower_bound)

/-- The theorem to be proved --/
theorem third_trial_point : ⌊x₃⌋ = 336 := by sorry

end NUMINAMATH_CALUDE_third_trial_point_l319_31940


namespace NUMINAMATH_CALUDE_apple_grape_worth_l319_31971

theorem apple_grape_worth (apple_value grape_value : ℚ) :
  (3/4 * 16) * apple_value = 10 * grape_value →
  (1/3 * 9) * apple_value = (5/2) * grape_value := by
  sorry

end NUMINAMATH_CALUDE_apple_grape_worth_l319_31971


namespace NUMINAMATH_CALUDE_max_squared_ratio_is_four_thirds_l319_31983

/-- The maximum squared ratio of a to b satisfying the given conditions -/
def max_squared_ratio (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b ∧
  ∃ ρ : ℝ, ρ > 0 ∧
    ∀ x y : ℝ, 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < b ∧
      a^2 + y^2 = b^2 + x^2 ∧
      b^2 + x^2 = (a - x)^2 + (b - y)^2 →
      (a / b)^2 ≤ ρ^2 ∧
    ρ^2 = 4/3

theorem max_squared_ratio_is_four_thirds (a b : ℝ) :
  max_squared_ratio a b :=
sorry

end NUMINAMATH_CALUDE_max_squared_ratio_is_four_thirds_l319_31983


namespace NUMINAMATH_CALUDE_sum_of_four_cubes_equals_three_l319_31914

theorem sum_of_four_cubes_equals_three (k : ℤ) :
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_cubes_equals_three_l319_31914


namespace NUMINAMATH_CALUDE_investment_average_rate_l319_31960

theorem investment_average_rate (total : ℝ) (rate1 rate2 : ℝ) (x : ℝ) :
  total = 6000 →
  rate1 = 0.03 →
  rate2 = 0.07 →
  rate1 * (total - x) = rate2 * x →
  (rate1 * (total - x) + rate2 * x) / total = 0.042 :=
by sorry

end NUMINAMATH_CALUDE_investment_average_rate_l319_31960


namespace NUMINAMATH_CALUDE_chocolate_bars_left_l319_31912

theorem chocolate_bars_left (initial_bars : ℕ) (thomas_friends : ℕ) (piper_return : ℕ) (paul_extra : ℕ) : 
  initial_bars = 500 →
  thomas_friends = 7 →
  piper_return = 7 →
  paul_extra = 5 →
  ∃ (thomas_take piper_take paul_take : ℕ),
    thomas_take = (initial_bars / 3 / thomas_friends) * thomas_friends + 2 ∧
    piper_take = initial_bars / 4 - piper_return ∧
    paul_take = piper_take + paul_extra ∧
    initial_bars - (thomas_take + piper_take + paul_take) = 96 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_left_l319_31912


namespace NUMINAMATH_CALUDE_triangle_area_l319_31930

/-- Given a triangle ABC with angles A, B, C forming an arithmetic sequence,
    side b = √3, and f(x) = 2√3 sin²x + 2sin x cos x - √3 reaching its maximum at x = A,
    prove that the area of triangle ABC is (3 + √3) / 4 -/
theorem triangle_area (A B C : Real) (b : Real) (f : Real → Real) :
  (∃ d : Real, B = A - d ∧ C = A + d) →  -- Angles form arithmetic sequence
  b = Real.sqrt 3 →  -- Side b equals √3
  (∀ x, f x = 2 * Real.sqrt 3 * Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3) →  -- Definition of f
  (∀ x, f x ≤ f A) →  -- f reaches maximum at A
  A + B + C = π →  -- Angle sum in triangle
  (∃ a c : Real, a * Real.sin B = b * Real.sin A ∧ c * Real.sin A = b * Real.sin C) →  -- Sine law
  1 / 2 * b * Real.sin A * Real.sin C / Real.sin B = (3 + Real.sqrt 3) / 4 :=  -- Area formula
by sorry

end NUMINAMATH_CALUDE_triangle_area_l319_31930


namespace NUMINAMATH_CALUDE_probability_is_correct_l319_31955

/-- The set of numbers from which we're selecting -/
def number_set : Set Nat := {n | 60 ≤ n ∧ n ≤ 1000}

/-- Predicate for a number being two-digit and divisible by 3 -/
def is_two_digit_div_by_three (n : Nat) : Prop := 60 ≤ n ∧ n ≤ 99 ∧ n % 3 = 0

/-- The count of numbers in the set -/
def total_count : Nat := 941

/-- The count of two-digit numbers divisible by 3 in the set -/
def favorable_count : Nat := 14

/-- The probability of selecting a two-digit number divisible by 3 from the set -/
def probability : Rat := favorable_count / total_count

theorem probability_is_correct : probability = 14 / 941 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_correct_l319_31955


namespace NUMINAMATH_CALUDE_golden_ratio_problem_l319_31944

theorem golden_ratio_problem (m n : ℝ) : 
  m = 2 * Real.sin (18 * π / 180) →
  m^2 + n = 4 →
  (m * Real.sqrt n) / (2 * Real.cos (27 * π / 180)^2 - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_golden_ratio_problem_l319_31944


namespace NUMINAMATH_CALUDE_complex_equation_sum_l319_31935

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 + Complex.I) * (2 - Complex.I) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l319_31935


namespace NUMINAMATH_CALUDE_probability_of_no_three_consecutive_ones_l319_31977

/-- Represents the number of valid sequences of length n -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 4
| n + 3 => b (n + 2) + b (n + 1) + b n

/-- The probability of a 12-element sequence not containing three consecutive 1s -/
def probability : ℚ := b 12 / 2^12

theorem probability_of_no_three_consecutive_ones : probability = 281 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_no_three_consecutive_ones_l319_31977


namespace NUMINAMATH_CALUDE_probability_one_common_course_is_two_thirds_l319_31910

def total_courses : ℕ := 4
def courses_per_person : ℕ := 2

def probability_one_common_course : ℚ :=
  let total_selections := Nat.choose total_courses courses_per_person * Nat.choose total_courses courses_per_person
  let no_common_courses := Nat.choose total_courses courses_per_person
  let all_common_courses := Nat.choose total_courses courses_per_person
  let one_common_course := total_selections - no_common_courses - all_common_courses
  ↑one_common_course / ↑total_selections

theorem probability_one_common_course_is_two_thirds :
  probability_one_common_course = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_common_course_is_two_thirds_l319_31910


namespace NUMINAMATH_CALUDE_prob_two_math_books_l319_31956

def total_books : ℕ := 5
def math_books : ℕ := 3

def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem prob_two_math_books : 
  (choose math_books 2 : ℚ) / (choose total_books 2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_math_books_l319_31956


namespace NUMINAMATH_CALUDE_vector_problem_l319_31984

/-- Given two planar vectors a and b, where a is orthogonal to b and their sum with a third vector c is zero, 
    prove that the first component of a is 2 and the magnitude of c is 5. -/
theorem vector_problem (m : ℝ) :
  let a : ℝ × ℝ := (m, -1)
  let b : ℝ × ℝ := (2, 4)
  let c : ℝ × ℝ := (-a.1 - b.1, -a.2 - b.2)
  (a.1 * b.1 + a.2 * b.2 = 0) →  -- a ⊥ b
  (m = 2 ∧ Real.sqrt ((c.1 ^ 2) + (c.2 ^ 2)) = 5) := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l319_31984


namespace NUMINAMATH_CALUDE_perpendicular_bisector_correct_parallel_line_correct_l319_31951

-- Define points A and B
def A : ℝ × ℝ := (7, -4)
def B : ℝ × ℝ := (-5, 6)

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x + y - 8 = 0
def line2 (x y : ℝ) : Prop := x - 2*y + 1 = 0
def line3 (x y : ℝ) : Prop := 4*x - 3*y - 7 = 0

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 6*x - 5*y - 1 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop := 4*x - 3*y - 6 = 0

-- Theorem for the perpendicular bisector
theorem perpendicular_bisector_correct : 
  perp_bisector = λ x y => (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

-- Theorem for the parallel line
theorem parallel_line_correct : 
  ∃ x y : ℝ, line1 x y ∧ line2 x y ∧ parallel_line x y ∧
  ∃ k : ℝ, ∀ x' y' : ℝ, parallel_line x' y' ↔ line3 x' y' ∧ (y' - y = k * (x' - x)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_correct_parallel_line_correct_l319_31951


namespace NUMINAMATH_CALUDE_complex_multiplication_l319_31952

/-- Given that i² = -1, prove that (4-5i)(-5+5i) = 5 + 45i --/
theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) :
  (4 - 5*i) * (-5 + 5*i) = 5 + 45*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l319_31952


namespace NUMINAMATH_CALUDE_geometric_sum_is_60_l319_31907

/-- The sum of a geometric sequence with 4 terms, first term 4, and common ratio 2 -/
def geometric_sum : ℕ := 
  let a := 4  -- first term
  let r := 2  -- common ratio
  let n := 4  -- number of terms
  a * (r^n - 1) / (r - 1)

/-- Theorem stating that the geometric sum is equal to 60 -/
theorem geometric_sum_is_60 : geometric_sum = 60 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_is_60_l319_31907


namespace NUMINAMATH_CALUDE_black_triangles_2008_l319_31957

/-- Given a sequence of triangles in the pattern ▲▲△△▲△, 
    this function returns the number of black triangles in n triangles -/
def black_triangles (n : ℕ) : ℕ :=
  (n - n % 6) / 2 + min 2 (n % 6)

/-- Theorem: In a sequence of 2008 triangles following the pattern ▲▲△△▲△,
    there are 1004 black triangles -/
theorem black_triangles_2008 : black_triangles 2008 = 1004 := by
  sorry

end NUMINAMATH_CALUDE_black_triangles_2008_l319_31957


namespace NUMINAMATH_CALUDE_pure_imaginary_quotient_condition_l319_31975

theorem pure_imaginary_quotient_condition (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 3 - 4 * Complex.I
  (∃ (b : ℝ), z₁ / z₂ = b * Complex.I) → a = 4/3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_quotient_condition_l319_31975


namespace NUMINAMATH_CALUDE_heather_blocks_l319_31933

/-- Given that Heather starts with 86 blocks and shares 41 blocks,
    prove that she ends up with 45 blocks. -/
theorem heather_blocks (initial_blocks : ℕ) (shared_blocks : ℕ) (final_blocks : ℕ)
  (h1 : initial_blocks = 86)
  (h2 : shared_blocks = 41)
  (h3 : final_blocks = initial_blocks - shared_blocks) :
  final_blocks = 45 := by
  sorry

end NUMINAMATH_CALUDE_heather_blocks_l319_31933


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l319_31997

theorem decimal_arithmetic : 25.3 - 0.432 + 1.25 = 26.118 := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l319_31997


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l319_31926

theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) (h1 : total_players = 24) (h2 : goalies = 4) (h3 : goalies < total_players) :
  (total_players - 1) * goalies = 92 :=
by sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l319_31926


namespace NUMINAMATH_CALUDE_parabola_coefficient_l319_31913

/-- Given a parabola y = ax^2 + bx + c with vertex at (q, q) and y-intercept at (0, -2q),
    where q ≠ 0, the value of b is 6/q. -/
theorem parabola_coefficient (a b c q : ℝ) : q ≠ 0 →
  (∀ x y, y = a * x^2 + b * x + c ↔ 
    ((x - q)^2 = 0 → y = q) ∧ 
    (x = 0 → y = -2 * q)) →
  b = 6 / q := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l319_31913


namespace NUMINAMATH_CALUDE_population_percentage_l319_31989

theorem population_percentage (W M : ℝ) (h : M = 1.1111111111111111 * W) : 
  W = 0.9 * M := by
sorry

end NUMINAMATH_CALUDE_population_percentage_l319_31989


namespace NUMINAMATH_CALUDE_class_configuration_exists_l319_31986

theorem class_configuration_exists (n : ℕ) (hn : n = 30) :
  ∃ (b g : ℕ),
    b + g = n ∧
    b = g ∧
    (∀ i j : ℕ, i < b → j < b → i ≠ j → ∃ k : ℕ, k < g ∧ (∃ f : ℕ → ℕ → Prop, f i k ≠ f j k)) ∧
    (∀ i j : ℕ, i < g → j < g → i ≠ j → ∃ k : ℕ, k < b ∧ (∃ f : ℕ → ℕ → Prop, f k i ≠ f k j)) :=
by
  sorry

end NUMINAMATH_CALUDE_class_configuration_exists_l319_31986


namespace NUMINAMATH_CALUDE_circle_radius_l319_31974

/-- Given a circle with diameter 26 centimeters, prove that its radius is 13 centimeters. -/
theorem circle_radius (diameter : ℝ) (h : diameter = 26) : diameter / 2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l319_31974


namespace NUMINAMATH_CALUDE_revenue_percent_change_l319_31924

/-- Calculates the percent change in revenue given initial conditions and tax changes -/
theorem revenue_percent_change 
  (initial_consumption : ℝ)
  (initial_tax_rate : ℝ)
  (tax_decrease_percent : ℝ)
  (consumption_increase_percent : ℝ)
  (additional_tax_decrease_percent : ℝ)
  (h1 : initial_consumption = 150)
  (h2 : tax_decrease_percent = 0.2)
  (h3 : consumption_increase_percent = 0.2)
  (h4 : additional_tax_decrease_percent = 0.02)
  (h5 : initial_consumption * (1 + consumption_increase_percent) < 200) :
  let new_consumption := initial_consumption * (1 + consumption_increase_percent)
  let new_tax_rate := initial_tax_rate * (1 - tax_decrease_percent - additional_tax_decrease_percent)
  let initial_revenue := initial_consumption * initial_tax_rate
  let new_revenue := new_consumption * new_tax_rate
  let percent_change := (new_revenue - initial_revenue) / initial_revenue * 100
  percent_change = -6.4 := by
sorry

end NUMINAMATH_CALUDE_revenue_percent_change_l319_31924


namespace NUMINAMATH_CALUDE_hyperbola_properties_l319_31981

/-- Properties of a hyperbola M with equation x²/4 - y²/2 = 1 -/
theorem hyperbola_properties :
  let M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 4 - p.2^2 / 2 = 1}
  ∃ (a b c : ℝ) (e : ℝ),
    a = 2 ∧
    b = Real.sqrt 2 ∧
    c = Real.sqrt 6 ∧
    e = Real.sqrt 6 / 2 ∧
    (2 * a = 4) ∧  -- Length of real axis
    (2 * b = 2 * Real.sqrt 2) ∧  -- Length of imaginary axis
    (2 * c = 2 * Real.sqrt 6) ∧  -- Focal distance
    (e = Real.sqrt 6 / 2)  -- Eccentricity
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l319_31981


namespace NUMINAMATH_CALUDE_snack_eaters_final_count_l319_31938

/-- Calculates the final number of snack eaters after a series of events --/
def finalSnackEaters (initialGathering : ℕ) (initialSnackEaters : ℕ)
  (firstNewGroup : ℕ) (secondNewGroup : ℕ) (thirdLeaving : ℕ) : ℕ :=
  let afterFirst := initialSnackEaters + firstNewGroup
  let afterHalfLeft := afterFirst / 2
  let afterSecondNew := afterHalfLeft + secondNewGroup
  let afterThirdLeft := afterSecondNew - thirdLeaving
  afterThirdLeft / 2

/-- Theorem stating that given the initial conditions and sequence of events,
    the final number of snack eaters is 20 --/
theorem snack_eaters_final_count :
  finalSnackEaters 200 100 20 10 30 = 20 := by
  sorry

#eval finalSnackEaters 200 100 20 10 30

end NUMINAMATH_CALUDE_snack_eaters_final_count_l319_31938


namespace NUMINAMATH_CALUDE_chicken_entree_cost_l319_31998

/-- Calculates the cost of each chicken entree given the wedding catering constraints. -/
theorem chicken_entree_cost
  (total_guests : ℕ)
  (steak_to_chicken_ratio : ℕ)
  (steak_cost : ℕ)
  (total_budget : ℕ)
  (h_total_guests : total_guests = 80)
  (h_ratio : steak_to_chicken_ratio = 3)
  (h_steak_cost : steak_cost = 25)
  (h_total_budget : total_budget = 1860) :
  (total_budget - steak_cost * (steak_to_chicken_ratio * total_guests / (steak_to_chicken_ratio + 1))) /
  (total_guests / (steak_to_chicken_ratio + 1)) = 18 := by
  sorry

#check chicken_entree_cost

end NUMINAMATH_CALUDE_chicken_entree_cost_l319_31998


namespace NUMINAMATH_CALUDE_angles_on_squared_paper_l319_31970

/-- Three angles marked on squared paper sum to 90 degrees -/
theorem angles_on_squared_paper (α β γ : ℝ) : α + β + γ = 90 := by
  sorry

end NUMINAMATH_CALUDE_angles_on_squared_paper_l319_31970


namespace NUMINAMATH_CALUDE_point_not_on_ln_graph_l319_31920

theorem point_not_on_ln_graph (a b : ℝ) (h : b = Real.log a) :
  ¬(1 + b = Real.log (a + Real.exp 1)) := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_ln_graph_l319_31920


namespace NUMINAMATH_CALUDE_smallest_block_volume_l319_31994

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Checks if the given dimensions satisfy the problem conditions. -/
def satisfiesConditions (d : BlockDimensions) : Prop :=
  (d.length - 1) * (d.width - 1) * (d.height - 1) = 288 ∧
  (d.length + d.width + d.height) % 10 = 0

/-- The volume of the block given its dimensions. -/
def blockVolume (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The theorem stating the smallest possible value of N. -/
theorem smallest_block_volume :
  ∃ (d : BlockDimensions), satisfiesConditions d ∧
    blockVolume d = 455 ∧
    ∀ (d' : BlockDimensions), satisfiesConditions d' → blockVolume d' ≥ 455 := by
  sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l319_31994


namespace NUMINAMATH_CALUDE_square_area_9cm_l319_31968

/-- The area of a square with side length 9 cm is 81 cm² -/
theorem square_area_9cm (square : Real → Real) (h : ∀ x, square x = x * x) :
  square 9 = 81 :=
by sorry

end NUMINAMATH_CALUDE_square_area_9cm_l319_31968


namespace NUMINAMATH_CALUDE_smallest_angle_WYZ_l319_31909

-- Define the angles
def angle_XYZ : ℝ := 36
def angle_XYW : ℝ := 15

-- Theorem to prove
theorem smallest_angle_WYZ : 
  let angle_WYZ := angle_XYZ - angle_XYW
  angle_WYZ = 21 := by sorry

end NUMINAMATH_CALUDE_smallest_angle_WYZ_l319_31909


namespace NUMINAMATH_CALUDE_sqrt_inequality_triangle_inequality_l319_31979

-- Problem 1
theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

-- Problem 2
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_triangle_inequality_l319_31979


namespace NUMINAMATH_CALUDE_inequality_proof_l319_31902

theorem inequality_proof (x y : ℝ) :
  (x + y) / 2 * (x^2 + y^2) / 2 * (x^3 + y^3) / 2 ≤ (x^6 + y^6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l319_31902


namespace NUMINAMATH_CALUDE_exponent_of_five_in_forty_factorial_l319_31964

theorem exponent_of_five_in_forty_factorial :
  ∃ k : ℕ, (40 : ℕ).factorial = 5^10 * k ∧ ¬(5 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_forty_factorial_l319_31964


namespace NUMINAMATH_CALUDE_taehyung_age_l319_31948

theorem taehyung_age (taehyung_age uncle_age : ℕ) 
  (h1 : uncle_age = taehyung_age + 17)
  (h2 : (taehyung_age + 4) + (uncle_age + 4) = 43) : 
  taehyung_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_age_l319_31948


namespace NUMINAMATH_CALUDE_three_special_lines_l319_31942

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has intercepts on both axes with equal absolute values -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ (l.a * t + l.c = 0 ∧ l.b * t + l.c = 0)

/-- The set of lines passing through (1, 2) with equal intercepts -/
def specialLines : Set Line :=
  {l : Line | l.contains 1 2 ∧ l.hasEqualIntercepts}

theorem three_special_lines :
  ∃ (l₁ l₂ l₃ : Line),
    l₁ ∈ specialLines ∧
    l₂ ∈ specialLines ∧
    l₃ ∈ specialLines ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ ∧
    ∀ l : Line, l ∈ specialLines → l = l₁ ∨ l = l₂ ∨ l = l₃ :=
  sorry

end NUMINAMATH_CALUDE_three_special_lines_l319_31942


namespace NUMINAMATH_CALUDE_epipen_insurance_coverage_l319_31972

/-- Calculates the insurance coverage percentage for EpiPens -/
theorem epipen_insurance_coverage 
  (frequency : ℕ) -- Number of EpiPens per year
  (cost : ℝ) -- Cost of each EpiPen in dollars
  (annual_payment : ℝ) -- John's annual payment in dollars
  (h1 : frequency = 2) -- John gets 2 EpiPens per year
  (h2 : cost = 500) -- Each EpiPen costs $500
  (h3 : annual_payment = 250) -- John pays $250 per year
  : (1 - annual_payment / (frequency * cost)) * 100 = 75 := by
  sorry


end NUMINAMATH_CALUDE_epipen_insurance_coverage_l319_31972


namespace NUMINAMATH_CALUDE_min_value_product_l319_31929

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 1 → (a + 1) * (b + 1) * (c + 1) ≤ (x + 1) * (y + 1) * (z + 1)) ∧
  (a + 1) * (b + 1) * (c + 1) = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_product_l319_31929


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l319_31976

noncomputable def f (x : ℝ) := Real.exp x * (x^2 - 2*x - 1)

theorem tangent_line_at_one (x y : ℝ) :
  let p := (1, f 1)
  let m := (Real.exp 1) * ((1:ℝ)^2 - 3)
  (y - f 1 = m * (x - 1)) ↔ (2 * Real.exp 1 * x + y = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l319_31976


namespace NUMINAMATH_CALUDE_coffee_price_percentage_increase_l319_31947

def highest_price : ℝ := 45
def lowest_price : ℝ := 30

theorem coffee_price_percentage_increase :
  (highest_price - lowest_price) / lowest_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_coffee_price_percentage_increase_l319_31947


namespace NUMINAMATH_CALUDE_gina_rose_cups_per_hour_l319_31946

theorem gina_rose_cups_per_hour :
  let lily_cups_per_hour : ℕ := 7
  let order_rose_cups : ℕ := 6
  let order_lily_cups : ℕ := 14
  let total_payment : ℕ := 90
  let hourly_rate : ℕ := 30
  let rose_cups_per_hour : ℕ := order_rose_cups / (total_payment / hourly_rate - order_lily_cups / lily_cups_per_hour)
  rose_cups_per_hour = 6 := by
sorry

end NUMINAMATH_CALUDE_gina_rose_cups_per_hour_l319_31946


namespace NUMINAMATH_CALUDE_rayden_extra_birds_l319_31911

def lily_ducks : ℕ := 20
def lily_geese : ℕ := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def rayden_geese : ℕ := 4 * lily_geese

theorem rayden_extra_birds : 
  (rayden_ducks + rayden_geese) - (lily_ducks + lily_geese) = 70 := by
  sorry

end NUMINAMATH_CALUDE_rayden_extra_birds_l319_31911


namespace NUMINAMATH_CALUDE_estate_distribution_l319_31921

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

end NUMINAMATH_CALUDE_estate_distribution_l319_31921


namespace NUMINAMATH_CALUDE_hannah_mugs_problem_l319_31982

/-- The number of mugs Hannah has of a color other than blue, red, or yellow -/
def other_color_mugs (total : ℕ) (blue red yellow : ℕ) : ℕ :=
  total - (blue + red + yellow)

theorem hannah_mugs_problem :
  ∀ (total blue red yellow : ℕ),
  total = 40 →
  blue = 3 * red →
  yellow = 12 →
  red = yellow / 2 →
  other_color_mugs total blue red yellow = 4 := by
sorry

end NUMINAMATH_CALUDE_hannah_mugs_problem_l319_31982


namespace NUMINAMATH_CALUDE_min_ships_proof_l319_31950

/-- The number of passengers to accommodate -/
def total_passengers : ℕ := 792

/-- The maximum capacity of each cruise ship -/
def ship_capacity : ℕ := 55

/-- The minimum number of cruise ships required -/
def min_ships : ℕ := (total_passengers + ship_capacity - 1) / ship_capacity

theorem min_ships_proof : min_ships = 15 := by
  sorry

end NUMINAMATH_CALUDE_min_ships_proof_l319_31950


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l319_31903

theorem quadratic_equation_roots_ratio (m : ℝ) : 
  (∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 / 2 ∧ 
   r + s = 5 ∧ r * s = m ∧ 
   ∀ x : ℝ, x^2 - 5*x + m = 0 ↔ (x = r ∨ x = s)) → 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l319_31903


namespace NUMINAMATH_CALUDE_set_intersection_equality_l319_31978

def S : Set Int := {s | ∃ n : Int, s = 2*n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4*n + 1}

theorem set_intersection_equality : S ∩ T = T := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_equality_l319_31978


namespace NUMINAMATH_CALUDE_m_values_l319_31923

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem m_values (m : ℝ) (h : 2 ∈ A m) : m = 0 ∨ m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_m_values_l319_31923


namespace NUMINAMATH_CALUDE_not_all_parallel_lines_in_plane_l319_31918

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in_plane : Line → Plane → Prop)

-- State the theorem
theorem not_all_parallel_lines_in_plane 
  (b : Line) (a : Line) (α : Plane)
  (h1 : parallel_line_plane b α)
  (h2 : contained_in_plane a α) :
  ¬ (∀ (l : Line), parallel_line_plane l α → ∀ (m : Line), contained_in_plane m α → parallel_lines l m) :=
sorry

end NUMINAMATH_CALUDE_not_all_parallel_lines_in_plane_l319_31918


namespace NUMINAMATH_CALUDE_x_values_l319_31988

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- State the theorem
theorem x_values (x : ℝ) (h : A x ∩ B x = B x) : x = 0 ∨ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l319_31988
