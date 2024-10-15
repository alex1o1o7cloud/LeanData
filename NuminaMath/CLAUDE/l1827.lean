import Mathlib

namespace NUMINAMATH_CALUDE_sin_cos_transformation_l1827_182759

/-- The transformation between sin and cos functions -/
theorem sin_cos_transformation (f g : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 4)) →
  (∀ x, g x = Real.cos (2 * x)) →
  (∀ θ, Real.sin θ = Real.cos (θ - π / 2)) →
  f x = g (x + 3 * π / 8) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_transformation_l1827_182759


namespace NUMINAMATH_CALUDE_sin_585_degrees_l1827_182705

theorem sin_585_degrees : Real.sin (585 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l1827_182705


namespace NUMINAMATH_CALUDE_product_quality_probability_l1827_182742

theorem product_quality_probability (p_B p_C : ℝ) 
  (h_B : p_B = 0.03) 
  (h_C : p_C = 0.02) : 
  1 - (p_B + p_C) = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_product_quality_probability_l1827_182742


namespace NUMINAMATH_CALUDE_percentage_equality_l1827_182768

theorem percentage_equality (x y : ℝ) (p : ℝ) (h1 : x / y = 4) (h2 : p / 100 * x = 20 / 100 * y) : p = 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l1827_182768


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_80_nines_80_sevens_l1827_182732

/-- A function that returns a natural number consisting of n repetitions of a given digit --/
def repeatDigit (digit : Nat) (n : Nat) : Nat :=
  if n = 0 then 0 else digit + 10 * repeatDigit digit (n - 1)

/-- A function that calculates the sum of digits of a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem --/
theorem sum_of_digits_of_product_80_nines_80_sevens :
  sumOfDigits (repeatDigit 9 80 * repeatDigit 7 80) = 720 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_of_product_80_nines_80_sevens_l1827_182732


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1827_182712

theorem sphere_volume_from_surface_area (O : Set ℝ) (surface_area : ℝ) (volume : ℝ) :
  (∃ (r : ℝ), surface_area = 4 * Real.pi * r^2) →
  surface_area = 4 * Real.pi →
  (∃ (r : ℝ), volume = (4 / 3) * Real.pi * r^3) →
  volume = (4 / 3) * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1827_182712


namespace NUMINAMATH_CALUDE_factorial_fraction_simplification_l1827_182723

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 1) * N^2) / Nat.factorial (N + 2) = N / (N + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_simplification_l1827_182723


namespace NUMINAMATH_CALUDE_shaded_square_area_fraction_l1827_182779

/-- The fraction of a 6x6 grid's area occupied by a square with vertices at midpoints of grid lines along the diagonal -/
theorem shaded_square_area_fraction (grid_size : ℕ) (shaded_square_side : ℝ) : 
  grid_size = 6 → 
  shaded_square_side = 1 / Real.sqrt 2 →
  (shaded_square_side^2) / (grid_size^2 : ℝ) = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_shaded_square_area_fraction_l1827_182779


namespace NUMINAMATH_CALUDE_schedule_ways_eq_840_l1827_182701

/-- The number of periods in a day -/
def num_periods : ℕ := 8

/-- The number of mathematics courses -/
def num_courses : ℕ := 4

/-- The number of ways to schedule the mathematics courses -/
def schedule_ways : ℕ := (num_periods - 1).choose num_courses * num_courses.factorial

/-- Theorem stating that the number of ways to schedule the mathematics courses is 840 -/
theorem schedule_ways_eq_840 : schedule_ways = 840 := by sorry

end NUMINAMATH_CALUDE_schedule_ways_eq_840_l1827_182701


namespace NUMINAMATH_CALUDE_ken_kept_pencils_l1827_182790

def pencil_problem (initial_pencils : ℕ) (given_to_manny : ℕ) (extra_to_nilo : ℕ) : Prop :=
  let given_to_nilo : ℕ := given_to_manny + extra_to_nilo
  let total_given : ℕ := given_to_manny + given_to_nilo
  let kept : ℕ := initial_pencils - total_given
  kept = 20

theorem ken_kept_pencils :
  pencil_problem 50 10 10 :=
sorry

end NUMINAMATH_CALUDE_ken_kept_pencils_l1827_182790


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1827_182756

/-- The function f(x) = x³ - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x² + a -/
def g (a x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 2 * x

/-- The tangent line of f at x₁ -/
def tangent_f (x₁ x : ℝ) : ℝ := f' x₁ * (x - x₁) + f x₁

/-- The tangent line of g at x₂ -/
def tangent_g (a x₂ x : ℝ) : ℝ := g' x₂ * (x - x₂) + g a x₂

theorem tangent_line_intersection (a : ℝ) :
  (∃ x₁ x₂ : ℝ, ∀ x : ℝ, tangent_f x₁ x = tangent_g a x₂ x) →
  (x₁ = -1 → a = 3) ∧ (a ≥ -1) := by sorry

end NUMINAMATH_CALUDE_tangent_line_intersection_l1827_182756


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l1827_182766

theorem least_three_digit_multiple_of_eight : ∀ n : ℕ, 
  n ≥ 100 ∧ n < 1000 ∧ n % 8 = 0 → n ≥ 104 := by
  sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l1827_182766


namespace NUMINAMATH_CALUDE_circle_area_ratio_false_l1827_182722

theorem circle_area_ratio_false : 
  ¬ (∀ (r : ℝ), r > 0 → (π * r^2) / (π * (2*r)^2) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_false_l1827_182722


namespace NUMINAMATH_CALUDE_library_books_count_l1827_182707

theorem library_books_count : ∀ (total_books : ℕ), 
  (35 : ℚ) / 100 * total_books + 104 = total_books → total_books = 160 :=
by
  sorry

end NUMINAMATH_CALUDE_library_books_count_l1827_182707


namespace NUMINAMATH_CALUDE_f_2014_value_l1827_182725

def N0 : Set ℕ := {n : ℕ | n ≥ 0}

def is_valid_f (f : ℕ → ℕ) : Prop :=
  f 2 = 0 ∧
  f 3 > 0 ∧
  f 6042 = 2014 ∧
  ∀ m n : ℕ, (f (m + n) - f m - f n) ∈ ({0, 1} : Set ℕ)

theorem f_2014_value (f : ℕ → ℕ) (h : is_valid_f f) : f 2014 = 671 := by
  sorry

end NUMINAMATH_CALUDE_f_2014_value_l1827_182725


namespace NUMINAMATH_CALUDE_two_isosceles_triangles_l1827_182782

/-- Represents a point in 2D space with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- Calculates the squared distance between two points -/
def squaredDistance (p1 p2 : Point) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if a triangle is isosceles based on its vertices -/
def isIsosceles (a b c : Point) : Bool :=
  let d1 := squaredDistance a b
  let d2 := squaredDistance b c
  let d3 := squaredDistance c a
  d1 = d2 || d2 = d3 || d3 = d1

/-- The four triangles given in the problem -/
def triangle1 : (Point × Point × Point) := ({x := 0, y := 0}, {x := 4, y := 0}, {x := 2, y := 3})
def triangle2 : (Point × Point × Point) := ({x := 1, y := 1}, {x := 1, y := 4}, {x := 4, y := 1})
def triangle3 : (Point × Point × Point) := ({x := 3, y := 0}, {x := 6, y := 0}, {x := 4, y := 3})
def triangle4 : (Point × Point × Point) := ({x := 5, y := 2}, {x := 8, y := 2}, {x := 7, y := 5})

theorem two_isosceles_triangles :
  let triangles := [triangle1, triangle2, triangle3, triangle4]
  (triangles.filter (fun t => isIsosceles t.1 t.2.1 t.2.2)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_isosceles_triangles_l1827_182782


namespace NUMINAMATH_CALUDE_midpoint_set_properties_l1827_182745

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  perimeter : ℝ

/-- The set of midpoints of segments with one end in F and the other in G -/
def midpoint_set (F G : ConvexPolygon) : Set (ℝ × ℝ) := sorry

theorem midpoint_set_properties (F G : ConvexPolygon) :
  let H := midpoint_set F G
  ∃ (sides_H : ℕ) (perimeter_H : ℝ),
    (ConvexPolygon.sides F).max (ConvexPolygon.sides G) ≤ sides_H ∧
    sides_H ≤ (ConvexPolygon.sides F) + (ConvexPolygon.sides G) ∧
    perimeter_H = (ConvexPolygon.perimeter F + ConvexPolygon.perimeter G) / 2 ∧
    (∀ (x y : ℝ × ℝ), x ∈ H → y ∈ H → (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → (1 - t) • x + t • y ∈ H)) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_set_properties_l1827_182745


namespace NUMINAMATH_CALUDE_min_workers_theorem_l1827_182733

/-- Represents the company's keychain production and sales model -/
structure KeychainCompany where
  maintenance_fee : ℝ  -- Daily maintenance fee
  worker_wage : ℝ      -- Hourly wage per worker
  keychains_per_hour : ℝ  -- Keychains produced per worker per hour
  keychain_price : ℝ   -- Price of each keychain
  work_hours : ℝ       -- Hours in a workday

/-- Calculates the minimum number of workers needed for profit -/
def min_workers_for_profit (company : KeychainCompany) : ℕ :=
  sorry

/-- Theorem stating the minimum number of workers needed for profit -/
theorem min_workers_theorem (company : KeychainCompany) 
  (h1 : company.maintenance_fee = 500)
  (h2 : company.worker_wage = 15)
  (h3 : company.keychains_per_hour = 5)
  (h4 : company.keychain_price = 3.10)
  (h5 : company.work_hours = 8) :
  min_workers_for_profit company = 126 :=
sorry

end NUMINAMATH_CALUDE_min_workers_theorem_l1827_182733


namespace NUMINAMATH_CALUDE_stating_raptors_score_l1827_182757

/-- 
Represents the scores of three teams in a cricket match, 
where the total score is 48 and one team wins over another by 18 points.
-/
structure CricketScores where
  eagles : ℕ
  raptors : ℕ
  hawks : ℕ
  total_is_48 : eagles + raptors + hawks = 48
  eagles_margin : eagles = raptors + 18

/-- 
Theorem stating that the Raptors' score is (30 - hawks) / 2
given the conditions of the cricket match.
-/
theorem raptors_score (scores : CricketScores) : 
  scores.raptors = (30 - scores.hawks) / 2 := by
  sorry

#check raptors_score

end NUMINAMATH_CALUDE_stating_raptors_score_l1827_182757


namespace NUMINAMATH_CALUDE_extended_triangle_PQ_length_l1827_182710

/-- Triangle ABC with extended sides and intersection points -/
structure ExtendedTriangle where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  CA : ℝ
  -- Extended segments
  DA : ℝ
  BE : ℝ
  -- Intersection points with circumcircle of CDE
  PQ : ℝ

/-- Theorem stating the length of PQ in the given configuration -/
theorem extended_triangle_PQ_length 
  (triangle : ExtendedTriangle)
  (h1 : triangle.AB = 15)
  (h2 : triangle.BC = 18)
  (h3 : triangle.CA = 20)
  (h4 : triangle.DA = triangle.AB)
  (h5 : triangle.BE = triangle.AB)
  : triangle.PQ = 37 := by
  sorry

#check extended_triangle_PQ_length

end NUMINAMATH_CALUDE_extended_triangle_PQ_length_l1827_182710


namespace NUMINAMATH_CALUDE_correct_division_l1827_182775

theorem correct_division (n : ℕ) : 
  n % 8 = 2 ∧ n / 8 = 156 → n / 5 = 250 := by
  sorry

end NUMINAMATH_CALUDE_correct_division_l1827_182775


namespace NUMINAMATH_CALUDE_line_slope_one_implies_a_equals_one_l1827_182700

/-- Given a line passing through points (-2, a) and (a, 4) with slope 1, prove that a = 1 -/
theorem line_slope_one_implies_a_equals_one (a : ℝ) :
  (4 - a) / (a + 2) = 1 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_one_implies_a_equals_one_l1827_182700


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l1827_182750

/-- Given two vectors a and b in ℝ², where b = (-1, 2) and a + b = (1, 3),
    prove that the magnitude of a - 2b is 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) : 
  b = (-1, 2) → a + b = (1, 3) → ‖a - 2 • b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l1827_182750


namespace NUMINAMATH_CALUDE_soccer_ball_surface_area_l1827_182714

/-- The surface area of a sphere with circumference 69 cm is 4761/π square cm. -/
theorem soccer_ball_surface_area :
  let circumference : ℝ := 69
  let radius : ℝ := circumference / (2 * Real.pi)
  let surface_area : ℝ := 4 * Real.pi * radius^2
  surface_area = 4761 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_soccer_ball_surface_area_l1827_182714


namespace NUMINAMATH_CALUDE_algebra_test_female_students_l1827_182751

theorem algebra_test_female_students 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 83)
  (h4 : female_average = 92) :
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_algebra_test_female_students_l1827_182751


namespace NUMINAMATH_CALUDE_tall_blonde_is_swedish_l1827_182719

/-- Represents the nationality of a racer -/
inductive Nationality
| Italian
| Swedish

/-- Represents the physical characteristics of a racer -/
structure Characteristics where
  height : Bool  -- true for tall, false for short
  hair : Bool    -- true for blonde, false for brunette

/-- Represents a racer -/
structure Racer where
  nationality : Nationality
  characteristics : Characteristics

def is_tall_blonde (r : Racer) : Prop :=
  r.characteristics.height ∧ r.characteristics.hair

def is_short_brunette (r : Racer) : Prop :=
  ¬r.characteristics.height ∧ ¬r.characteristics.hair

theorem tall_blonde_is_swedish (racers : Finset Racer) : 
  (∀ r : Racer, r ∈ racers → (is_tall_blonde r → r.nationality = Nationality.Swedish)) :=
by
  sorry

#check tall_blonde_is_swedish

end NUMINAMATH_CALUDE_tall_blonde_is_swedish_l1827_182719


namespace NUMINAMATH_CALUDE_unique_root_of_cubic_l1827_182736

/-- The function f(x) = (x-3)(x^2+2x+3) has exactly one real root. -/
theorem unique_root_of_cubic (x : ℝ) : ∃! a : ℝ, (a - 3) * (a^2 + 2*a + 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_cubic_l1827_182736


namespace NUMINAMATH_CALUDE_equal_reading_time_l1827_182764

/-- The total number of pages in the novel -/
def total_pages : ℕ := 760

/-- Bob's reading speed in seconds per page -/
def bob_speed : ℕ := 45

/-- Chandra's reading speed in seconds per page -/
def chandra_speed : ℕ := 30

/-- The number of pages Chandra reads -/
def chandra_pages : ℕ := 456

/-- The number of pages Bob reads -/
def bob_pages : ℕ := total_pages - chandra_pages

theorem equal_reading_time : chandra_speed * chandra_pages = bob_speed * bob_pages := by
  sorry

end NUMINAMATH_CALUDE_equal_reading_time_l1827_182764


namespace NUMINAMATH_CALUDE_bucket_fill_time_l1827_182704

/-- Given that two-thirds of a bucket is filled in 100 seconds,
    prove that it takes 150 seconds to fill the bucket completely. -/
theorem bucket_fill_time :
  let partial_fill_time : ℝ := 100
  let partial_fill_fraction : ℝ := 2/3
  let complete_fill_time : ℝ := 150
  (partial_fill_fraction * complete_fill_time = partial_fill_time) →
  complete_fill_time = 150 :=
by sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l1827_182704


namespace NUMINAMATH_CALUDE_career_preference_representation_l1827_182784

theorem career_preference_representation (total_students : ℕ) 
  (male_ratio female_ratio : ℕ) (male_preference female_preference : ℕ) : 
  total_students = 30 →
  male_ratio = 2 →
  female_ratio = 3 →
  male_preference = 2 →
  female_preference = 3 →
  (((male_preference + female_preference : ℝ) / total_students) * 360 : ℝ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_representation_l1827_182784


namespace NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l1827_182758

theorem complex_root_modulus_one_iff_divisible_by_six (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_modulus_one_iff_divisible_by_six_l1827_182758


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1827_182748

theorem consecutive_numbers_sum (n : ℕ) : 
  (∃ a : ℕ, (∀ k : ℕ, k < n → ∃ i j l m : ℕ, 
    i < j ∧ j < l ∧ l < m ∧ m < n ∧ 
    a + i + (a + j) + (a + l) + (a + m) = k + (4 * a + 6))) ∧
  (∀ k : ℕ, k ≥ 385 → ¬∃ i j l m : ℕ, 
    i < j ∧ j < l ∧ l < m ∧ m < n ∧ 
    ∃ a : ℕ, a + i + (a + j) + (a + l) + (a + m) = k + (4 * a + 6)) →
  n = 100 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1827_182748


namespace NUMINAMATH_CALUDE_even_function_characterization_l1827_182770

def M (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  {t | ∃ x ≥ a, t = f x - f a}

def L (f : ℝ → ℝ) (a : ℝ) : Set ℝ :=
  {t | ∃ x ≤ a, t = f x - f a}

def has_minimum (f : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, f m ≤ f x

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_characterization (f : ℝ → ℝ) (h : has_minimum f) :
  is_even_function f ↔ ∀ c > 0, M f (-c) = L f c := by
  sorry

end NUMINAMATH_CALUDE_even_function_characterization_l1827_182770


namespace NUMINAMATH_CALUDE_correct_average_after_error_correction_l1827_182711

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (incorrect_avg : ℚ) 
  (incorrect_num correct_num : ℚ) :
  n = 10 ∧ 
  incorrect_avg = 16 ∧ 
  incorrect_num = 25 ∧ 
  correct_num = 55 →
  (n * incorrect_avg - incorrect_num + correct_num) / n = 19 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_after_error_correction_l1827_182711


namespace NUMINAMATH_CALUDE_two_sqrt_five_less_than_five_l1827_182794

theorem two_sqrt_five_less_than_five : 2 * Real.sqrt 5 < 5 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_five_less_than_five_l1827_182794


namespace NUMINAMATH_CALUDE_married_men_fraction_l1827_182746

-- Define the faculty
structure Faculty where
  total : ℕ
  women : ℕ
  married : ℕ
  men : ℕ

-- Define the conditions
def faculty_conditions (f : Faculty) : Prop :=
  f.women = (70 * f.total) / 100 ∧
  f.married = (40 * f.total) / 100 ∧
  f.men = f.total - f.women

-- Define the fraction of single men
def single_men_fraction (f : Faculty) : ℚ :=
  1 / 3

-- Theorem to prove
theorem married_men_fraction (f : Faculty) 
  (h : faculty_conditions f) : 
  (f.married - (f.women - (f.total - f.married))) / f.men = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l1827_182746


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1827_182767

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularLinePlane : Line → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (m n : Line) (α β : Plane)
  (h1 : parallelPlanes α β)
  (h2 : perpendicularLinePlane m α)
  (h3 : perpendicularLinePlane n β) :
  parallelLinePlane m β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l1827_182767


namespace NUMINAMATH_CALUDE_boat_speed_is_54_l1827_182731

/-- Represents the speed of a boat in still water -/
def boat_speed (v : ℝ) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  (v - 18) * (2 * t) = (v + 18) * t

/-- Theorem: The speed of the boat in still water is 54 kmph -/
theorem boat_speed_is_54 : boat_speed 54 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_is_54_l1827_182731


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1827_182786

theorem smallest_max_sum (a b c d e f g : ℕ+) 
  (sum_eq : a + b + c + d + e + f + g = 2024) : 
  (∃ M : ℕ, 
    (M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g)))))) ∧ 
    (∀ M' : ℕ, 
      (M' = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g)))))) → 
      M ≤ M') ∧
    M = 338) := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1827_182786


namespace NUMINAMATH_CALUDE_circular_path_area_l1827_182777

/-- The area of a circular path around a circular lawn -/
theorem circular_path_area (r : ℝ) (w : ℝ) (h_r : r > 0) (h_w : w > 0) :
  let R := r + w
  (π * R^2 - π * r^2) = π * (R^2 - r^2) := by sorry

#check circular_path_area

end NUMINAMATH_CALUDE_circular_path_area_l1827_182777


namespace NUMINAMATH_CALUDE_terms_are_like_l1827_182716

-- Define a structure for algebraic terms
structure AlgebraicTerm where
  coefficient : ℤ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define a function to check if two terms are like terms
def are_like_terms (t1 t2 : AlgebraicTerm) : Prop :=
  t1.x_exponent = t2.x_exponent ∧ t1.y_exponent = t2.y_exponent

-- Define the two terms we want to compare
def term1 : AlgebraicTerm := { coefficient := -4, x_exponent := 1, y_exponent := 2 }
def term2 : AlgebraicTerm := { coefficient := 4, x_exponent := 1, y_exponent := 2 }

-- State the theorem
theorem terms_are_like : are_like_terms term1 term2 := by
  sorry

end NUMINAMATH_CALUDE_terms_are_like_l1827_182716


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1827_182730

/-- Given a hyperbola with equation y²/16 - x²/m = 1 and eccentricity e = 2, prove that m = 48 -/
theorem hyperbola_eccentricity (m : ℝ) (e : ℝ) :
  (∀ x y : ℝ, y^2 / 16 - x^2 / m = 1) →
  e = 2 →
  m = 48 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1827_182730


namespace NUMINAMATH_CALUDE_logarithm_sum_equals_two_l1827_182796

theorem logarithm_sum_equals_two : Real.log 25 / Real.log 10 + (Real.log 2 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_equals_two_l1827_182796


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1827_182729

theorem trigonometric_identity (x y z a : ℝ) 
  (h1 : (Real.cos x + Real.cos y + Real.cos z) / Real.cos (x + y + z) = a)
  (h2 : (Real.sin x + Real.sin y + Real.sin z) / Real.sin (x + y + z) = a) :
  Real.cos (x + y) + Real.cos (y + z) + Real.cos (z + x) = a :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1827_182729


namespace NUMINAMATH_CALUDE_recurring_decimal_sum_l1827_182738

/-- The sum of 0.3̄, 0.04̄, and 0.005̄ is equal to 112386/296703 -/
theorem recurring_decimal_sum : 
  (1 : ℚ) / 3 + 4 / 99 + 5 / 999 = 112386 / 296703 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_sum_l1827_182738


namespace NUMINAMATH_CALUDE_smallest_consecutive_integer_l1827_182709

theorem smallest_consecutive_integer (a b c d e : ℤ) : 
  (a + b + c + d + e = 2015) → 
  (b = a + 1) → 
  (c = b + 1) → 
  (d = c + 1) → 
  (e = d + 1) → 
  (a = 401) := by
  sorry

end NUMINAMATH_CALUDE_smallest_consecutive_integer_l1827_182709


namespace NUMINAMATH_CALUDE_line_through_M_and_P_line_through_M_perp_to_line_l1827_182747

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0
def l₂ (x y : ℝ) : Prop := 2*x + 3*y - 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (1, 2)

-- Define point P
def P : ℝ × ℝ := (3, 1)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 3*x + 2*y + 5 = 0

-- Part 1: Line equation through M and P
theorem line_through_M_and_P :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ (l₁ x y ∧ l₂ x y) ∨ (x = P.1 ∧ y = P.2)) →
    a = 1 ∧ b = 2 ∧ c = -5 :=
sorry

-- Part 2: Line equation through M and perpendicular to perp_line
theorem line_through_M_perp_to_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, a*x + b*y + c = 0 ↔ (l₁ x y ∧ l₂ x y) ∨ 
      (∃ (k : ℝ), a*3 + b*2 = 0 ∧ x = M.1 + k*2 ∧ y = M.2 - k*3)) →
    a = 2 ∧ b = -3 ∧ c = 4 :=
sorry

end NUMINAMATH_CALUDE_line_through_M_and_P_line_through_M_perp_to_line_l1827_182747


namespace NUMINAMATH_CALUDE_max_x_value_l1827_182740

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (prod_sum_eq : x*y + x*z + y*z = 9) :
  x ≤ 4 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ + y₀ + z₀ = 6 ∧ x₀*y₀ + x₀*z₀ + y₀*z₀ = 9 ∧ x₀ = 4 :=
by sorry

end NUMINAMATH_CALUDE_max_x_value_l1827_182740


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1827_182708

theorem absolute_value_simplification (x : ℝ) (h : x < 0) : |3*x + Real.sqrt (x^2)| = -2*x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1827_182708


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1827_182721

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), 
  r₁ > 0 → r₂ > 0 → r₂ = 2 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1827_182721


namespace NUMINAMATH_CALUDE_smaller_number_expression_l1827_182713

theorem smaller_number_expression (m n t s : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : t > 1) 
  (h4 : m / n = t) 
  (h5 : m + n = s) : 
  n = s / (1 + t) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_expression_l1827_182713


namespace NUMINAMATH_CALUDE_work_project_solution_l1827_182774

/-- Represents a work project with a number of workers and days to complete. -/
structure WorkProject where
  workers : ℕ
  days : ℕ

/-- The condition when 2 workers are removed. -/
def condition1 (wp : WorkProject) : Prop :=
  (wp.workers - 2) * (wp.days + 4) = wp.workers * wp.days

/-- The condition when 3 workers are added. -/
def condition2 (wp : WorkProject) : Prop :=
  (wp.workers + 3) * (wp.days - 2) > wp.workers * wp.days

/-- The condition when 4 workers are added. -/
def condition3 (wp : WorkProject) : Prop :=
  (wp.workers + 4) * (wp.days - 3) > wp.workers * wp.days

/-- The main theorem stating the solution to the work project problem. -/
theorem work_project_solution :
  ∃ (wp : WorkProject),
    condition1 wp ∧
    condition2 wp ∧
    condition3 wp ∧
    wp.workers = 6 ∧
    wp.days = 8 := by
  sorry


end NUMINAMATH_CALUDE_work_project_solution_l1827_182774


namespace NUMINAMATH_CALUDE_rachel_brownies_l1827_182792

theorem rachel_brownies (total : ℚ) : 
  (3 / 5 : ℚ) * total = 18 → total = 30 := by
  sorry

end NUMINAMATH_CALUDE_rachel_brownies_l1827_182792


namespace NUMINAMATH_CALUDE_newspaper_selling_price_l1827_182799

theorem newspaper_selling_price 
  (total_newspapers : ℕ) 
  (sold_percentage : ℚ)
  (buying_discount : ℚ)
  (total_profit : ℚ) :
  total_newspapers = 500 →
  sold_percentage = 80 / 100 →
  buying_discount = 75 / 100 →
  total_profit = 550 →
  ∃ (selling_price : ℚ),
    selling_price = 2 ∧
    (sold_percentage * total_newspapers : ℚ) * selling_price -
    (1 - buying_discount) * (total_newspapers : ℚ) * selling_price = total_profit :=
by sorry

end NUMINAMATH_CALUDE_newspaper_selling_price_l1827_182799


namespace NUMINAMATH_CALUDE_red_apples_sold_l1827_182789

theorem red_apples_sold (red green : ℕ) : 
  (red : ℚ) / green = 8 / 3 → 
  red + green = 44 → 
  red = 32 := by
sorry

end NUMINAMATH_CALUDE_red_apples_sold_l1827_182789


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_6770_l1827_182754

/-- Calculate the cost of white washing a room with given dimensions and openings. -/
def whitewashingCost (roomLength roomWidth roomHeight : ℝ)
                     (doorLength doorWidth : ℝ)
                     (windowLength windowWidth : ℝ)
                     (numDoors numWindows : ℕ)
                     (costPerSqFt : ℝ) : ℝ :=
  let wallArea := 2 * (roomLength * roomHeight + roomWidth * roomHeight)
  let doorArea := numDoors * (doorLength * doorWidth)
  let windowArea := numWindows * (windowLength * windowWidth)
  let paintableArea := wallArea - doorArea - windowArea
  paintableArea * costPerSqFt

/-- Theorem stating that the cost of white washing the room with given specifications is 6770 Rs. -/
theorem whitewashing_cost_is_6770 :
  whitewashingCost 30 20 15 7 4 5 3 2 6 5 = 6770 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_6770_l1827_182754


namespace NUMINAMATH_CALUDE_triangle_properties_l1827_182743

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  b * c * (Real.cos A) = 4 ∧
  a * c * (Real.sin B) = 8 * (Real.sin A) →
  A = π / 3 ∧ 
  0 < Real.sin A * Real.sin B * Real.sin C ∧ 
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1827_182743


namespace NUMINAMATH_CALUDE_job_completion_time_l1827_182737

theorem job_completion_time (x : ℝ) : 
  x > 0 → 
  5 * (1/x + 1/20) = 1 - 0.41666666666666663 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l1827_182737


namespace NUMINAMATH_CALUDE_tom_seashells_l1827_182728

def seashells_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

theorem tom_seashells : seashells_remaining 5 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_seashells_l1827_182728


namespace NUMINAMATH_CALUDE_square_function_is_even_l1827_182785

/-- The function f(x) = x^2 is an even function for all real numbers x. -/
theorem square_function_is_even : ∀ x : ℝ, (fun x => x^2) (-x) = (fun x => x^2) x := by
  sorry

end NUMINAMATH_CALUDE_square_function_is_even_l1827_182785


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l1827_182727

theorem reciprocal_inequality (a b : ℝ) (ha : a < 0) (hb : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l1827_182727


namespace NUMINAMATH_CALUDE_derivative_of_e_squared_l1827_182795

theorem derivative_of_e_squared :
  (deriv (λ _ : ℝ => Real.exp 2)) = (λ _ => 0) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_e_squared_l1827_182795


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1827_182793

/-- Given a hyperbola with eccentricity 2 and foci coinciding with those of a specific ellipse,
    prove that its equation is x²/4 - y²/12 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h : ℝ × ℝ → Prop) (e : ℝ × ℝ → Prop) :
  (∀ x y, h (x, y) ↔ x^2/a^2 - y^2/b^2 = 1) →
  (∀ x y, e (x, y) ↔ x^2/25 + y^2/9 = 1) →
  (∀ x y, h (x, y) → (x/a)^2 - (y/b)^2 = 1) →
  (∀ x, e (x, 0) → x = 4 ∨ x = -4) →
  (∀ x, h (x, 0) → x = 4 ∨ x = -4) →
  (a / Real.sqrt (a^2 - b^2) = 2) →
  (∀ x y, h (x, y) ↔ x^2/4 - y^2/12 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1827_182793


namespace NUMINAMATH_CALUDE_total_unique_customers_l1827_182762

/-- Represents the number of customers who had meals with both ham and cheese -/
def ham_cheese : ℕ := 80

/-- Represents the number of customers who had meals with both ham and tomatoes -/
def ham_tomato : ℕ := 90

/-- Represents the number of customers who had meals with both tomatoes and cheese -/
def tomato_cheese : ℕ := 100

/-- Represents the number of customers who had meals with all three ingredients -/
def all_three : ℕ := 20

/-- Theorem stating that the total number of unique customers is 230 -/
theorem total_unique_customers : 
  ham_cheese + ham_tomato + tomato_cheese - 2 * all_three = 230 := by
  sorry

end NUMINAMATH_CALUDE_total_unique_customers_l1827_182762


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l1827_182715

theorem polar_to_rectangular (ρ : ℝ) (θ : ℝ) :
  ρ = 2 ∧ θ = π / 6 →
  ∃ x y : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ x = Real.sqrt 3 ∧ y = 1 := by
sorry


end NUMINAMATH_CALUDE_polar_to_rectangular_l1827_182715


namespace NUMINAMATH_CALUDE_dasha_number_l1827_182783

-- Define a function to calculate the product of digits
def digitProduct (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digitProduct (n / 10)

-- Define a function to check if a number is single-digit
def isSingleDigit (n : ℕ) : Prop := n < 10

-- Theorem statement
theorem dasha_number (n : ℕ) :
  n ≤ digitProduct n → isSingleDigit n :=
by sorry

end NUMINAMATH_CALUDE_dasha_number_l1827_182783


namespace NUMINAMATH_CALUDE_hundred_with_fewer_threes_l1827_182772

-- Define a datatype for arithmetic expressions
inductive Expr
  | Num : ℕ → Expr
  | Add : Expr → Expr → Expr
  | Sub : Expr → Expr → Expr
  | Mul : Expr → Expr → Expr
  | Div : Expr → Expr → Expr

-- Function to count the number of 3's in an expression
def countThrees : Expr → ℕ
  | Expr.Num n => if n = 3 then 1 else 0
  | Expr.Add e1 e2 => countThrees e1 + countThrees e2
  | Expr.Sub e1 e2 => countThrees e1 + countThrees e2
  | Expr.Mul e1 e2 => countThrees e1 + countThrees e2
  | Expr.Div e1 e2 => countThrees e1 + countThrees e2

-- Function to evaluate an expression
def evaluate : Expr → ℚ
  | Expr.Num n => n
  | Expr.Add e1 e2 => evaluate e1 + evaluate e2
  | Expr.Sub e1 e2 => evaluate e1 - evaluate e2
  | Expr.Mul e1 e2 => evaluate e1 * evaluate e2
  | Expr.Div e1 e2 => evaluate e1 / evaluate e2

-- Theorem statement
theorem hundred_with_fewer_threes : 
  ∃ e : Expr, evaluate e = 100 ∧ countThrees e < 10 :=
sorry

end NUMINAMATH_CALUDE_hundred_with_fewer_threes_l1827_182772


namespace NUMINAMATH_CALUDE_tan_pi_plus_alpha_problem_l1827_182753

theorem tan_pi_plus_alpha_problem (α : Real) (h : Real.tan (Real.pi + α) = -1/2) :
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) /
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3/2 * Real.pi - α)) = -7/9 ∧
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_plus_alpha_problem_l1827_182753


namespace NUMINAMATH_CALUDE_tan_equality_integer_l1827_182788

theorem tan_equality_integer (n : ℤ) : 
  -90 < n ∧ n < 90 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = -30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_integer_l1827_182788


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1827_182771

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

-- Theorem for the first equation
theorem solutions_equation1 : 
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1/3 ∧ equation1 x₁ ∧ equation1 x₂ ∧ 
  ∀ x : ℝ, equation1 x → x = x₁ ∨ x = x₂ := by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -4 ∧ equation2 x₁ ∧ equation2 x₂ ∧ 
  ∀ x : ℝ, equation2 x → x = x₁ ∨ x = x₂ := by sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1827_182771


namespace NUMINAMATH_CALUDE_parabola_vertex_l1827_182763

/-- The vertex of the parabola y = -3x^2 + 6x + 4 is (1, 7) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -3 * x^2 + 6 * x + 4 → (1, 7) = (x, y) ∧ ∀ (x' : ℝ), y ≥ -3 * x'^2 + 6 * x' + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1827_182763


namespace NUMINAMATH_CALUDE_parallel_transitivity_l1827_182787

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the "outside of plane" relation
variable (outside_of_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (m n : Line) (α : Plane) 
  (h1 : outside_of_plane m α)
  (h2 : outside_of_plane n α)
  (h3 : parallel_lines m n)
  (h4 : parallel_line_plane m α) :
  parallel_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l1827_182787


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l1827_182720

theorem rectangle_area_increase (L W : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let original_area := L * W
  let new_area := (1.1 * L) * (1.1 * W)
  (new_area - original_area) / original_area = 0.21 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l1827_182720


namespace NUMINAMATH_CALUDE_fruit_condition_percentage_l1827_182752

/-- Calculates the percentage of fruits in good condition given the number of oranges and bananas and their rotten percentages -/
def percentageGoodFruits (totalOranges totalBananas : ℕ) (rottenOrangesPercent rottenBananasPercent : ℚ) : ℚ :=
  let goodOranges : ℚ := totalOranges * (1 - rottenOrangesPercent)
  let goodBananas : ℚ := totalBananas * (1 - rottenBananasPercent)
  let totalFruits : ℚ := totalOranges + totalBananas
  (goodOranges + goodBananas) / totalFruits * 100

/-- Theorem stating that given 600 oranges with 15% rotten and 400 bananas with 4% rotten, 
    the percentage of fruits in good condition is 89.4% -/
theorem fruit_condition_percentage : 
  percentageGoodFruits 600 400 (15/100) (4/100) = 89.4 := by
  sorry

end NUMINAMATH_CALUDE_fruit_condition_percentage_l1827_182752


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l1827_182798

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x - 1

-- Theorem 1
theorem theorem_1 (a : ℝ) :
  f a 1 = 2 → a = 1 ∧ ∀ x, f 1 x ≥ -2 :=
sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
sorry

-- Theorem 3
theorem theorem_3 (a : ℝ) :
  (∀ x ≤ 4, ∀ y ≤ x, f a x ≤ f a y) → a ≤ -4 :=
sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_theorem_3_l1827_182798


namespace NUMINAMATH_CALUDE_mary_shirts_problem_l1827_182741

theorem mary_shirts_problem (blue_shirts : ℕ) (brown_shirts : ℕ) : 
  brown_shirts = 36 →
  blue_shirts / 2 + brown_shirts * 2 / 3 = 37 →
  blue_shirts = 26 := by
sorry

end NUMINAMATH_CALUDE_mary_shirts_problem_l1827_182741


namespace NUMINAMATH_CALUDE_julie_lawns_mowed_l1827_182718

def bike_cost : ℕ := 2345
def initial_savings : ℕ := 1500
def newspapers_delivered : ℕ := 600
def newspaper_pay : ℚ := 0.4
def dogs_walked : ℕ := 24
def dog_walking_pay : ℕ := 15
def lawn_mowing_pay : ℕ := 20
def money_left : ℕ := 155

def total_earned (lawns_mowed : ℕ) : ℚ :=
  initial_savings + newspapers_delivered * newspaper_pay + dogs_walked * dog_walking_pay + lawns_mowed * lawn_mowing_pay

theorem julie_lawns_mowed :
  ∃ (lawns_mowed : ℕ), total_earned lawns_mowed = bike_cost + money_left ∧ lawns_mowed = 20 :=
sorry

end NUMINAMATH_CALUDE_julie_lawns_mowed_l1827_182718


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l1827_182735

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- State the theorem
theorem solution_set_equivalence 
  (h1 : ∀ x, 3 * f x + f' x < 0)
  (h2 : f (log 2) = 1) :
  ∀ x, f x > 8 * exp (-3 * x) ↔ x < log 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l1827_182735


namespace NUMINAMATH_CALUDE_roots_are_zero_neg_five_and_a_l1827_182749

variable (a : ℝ)

def roots : Set ℝ := {x : ℝ | x * (x + 5)^2 * (a - x) = 0}

theorem roots_are_zero_neg_five_and_a : roots a = {0, -5, a} := by
  sorry

end NUMINAMATH_CALUDE_roots_are_zero_neg_five_and_a_l1827_182749


namespace NUMINAMATH_CALUDE_cube_diff_prime_mod_six_l1827_182739

theorem cube_diff_prime_mod_six (a b p : ℕ) : 
  0 < a → 0 < b → Prime p → a^3 - b^3 = p → p % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_diff_prime_mod_six_l1827_182739


namespace NUMINAMATH_CALUDE_second_set_is_twenty_feet_l1827_182726

/-- The length of the first set of wood in feet -/
def first_set_length : ℝ := 4

/-- The factor by which the second set is longer than the first set -/
def length_factor : ℝ := 5

/-- The length of the second set of wood in feet -/
def second_set_length : ℝ := first_set_length * length_factor

/-- Theorem stating that the second set of wood is 20 feet long -/
theorem second_set_is_twenty_feet : second_set_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_set_is_twenty_feet_l1827_182726


namespace NUMINAMATH_CALUDE_discount_percentage_l1827_182761

theorem discount_percentage (original_price sale_price : ℝ) 
  (h1 : original_price = 600)
  (h2 : sale_price = 480) : 
  (original_price - sale_price) / original_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l1827_182761


namespace NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_sixteenth_l1827_182724

/-- Represents a fair 12-sided die -/
def TwelveSidedDie := Fin 12

/-- The probability of getting a number divisible by 4 on a 12-sided die -/
def prob_divisible_by_four (die : TwelveSidedDie) : ℚ :=
  3 / 12

/-- The probability of getting two numbers divisible by 4 when tossing two 12-sided dice -/
def prob_both_divisible_by_four (die1 die2 : TwelveSidedDie) : ℚ :=
  (prob_divisible_by_four die1) * (prob_divisible_by_four die2)

theorem prob_both_divisible_by_four_is_one_sixteenth :
  ∀ (die1 die2 : TwelveSidedDie), prob_both_divisible_by_four die1 die2 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_divisible_by_four_is_one_sixteenth_l1827_182724


namespace NUMINAMATH_CALUDE_root_equation_problem_l1827_182744

/-- Given two polynomial equations with constants c and d, prove that 100c + d = 359 -/
theorem root_equation_problem (c d : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    ((x + c) * (x + d) * (x + 10)) / ((x + 5) * (x + 5)) = 0 ∧
    ((y + c) * (y + d) * (y + 10)) / ((y + 5) * (y + 5)) = 0 ∧
    ((z + c) * (z + d) * (z + 10)) / ((z + 5) * (z + 5)) = 0) ∧
  (∃! w : ℝ, ((w + 2*c) * (w + 7) * (w + 9)) / ((w + d) * (w + 10)) = 0) →
  100 * c + d = 359 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l1827_182744


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l1827_182776

theorem impossible_coin_probabilities : ¬∃ (p₁ p₂ : ℝ), 
  0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧ 
  (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧ 
  p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) := by
  sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l1827_182776


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1827_182791

/-- A sequence a : ℕ → ℝ is geometric if there exists a common ratio r such that
    for all n, a(n+1) = r * a(n) -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) (h : IsGeometric a) 
    (h1 : a 1 * a 99 = 16) : a 20 * a 80 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1827_182791


namespace NUMINAMATH_CALUDE_debate_team_group_size_l1827_182778

/-- The size of each group in a debate team -/
def group_size (num_boys num_girls num_groups : ℕ) : ℕ :=
  (num_boys + num_girls) / num_groups

/-- Theorem: The size of each group in the debate team is 7 -/
theorem debate_team_group_size :
  group_size 11 45 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l1827_182778


namespace NUMINAMATH_CALUDE_father_sees_boy_less_than_half_time_l1827_182797

/-- Represents a point on the perimeter of the square school -/
structure PerimeterPoint where
  side : Fin 4
  position : ℝ
  h_position : 0 ≤ position ∧ position ≤ 1

/-- Represents the movement of a person around the square school -/
structure Movement where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise
  start_point : PerimeterPoint

/-- Defines when two points are on the same side of the square -/
def on_same_side (p1 p2 : PerimeterPoint) : Prop :=
  p1.side = p2.side

/-- The boy's movement around the school -/
def boy_movement : Movement :=
  { speed := 10
  , direction := true  -- Always clockwise
  , start_point := { side := 0, position := 0, h_position := ⟨by norm_num, by norm_num⟩ } }

/-- The father's movement around the school -/
def father_movement : Movement :=
  { speed := 5
  , direction := true  -- Initial direction (can change)
  , start_point := { side := 0, position := 0, h_position := ⟨by norm_num, by norm_num⟩ } }

/-- Theorem stating that the father cannot see the boy for more than half the time -/
theorem father_sees_boy_less_than_half_time :
  ∀ (t : ℝ) (t_pos : 0 < t),
  ∃ (boy_pos father_pos : PerimeterPoint),
  (∀ τ : ℝ, 0 ≤ τ ∧ τ ≤ t →
    (on_same_side (boy_pos) (father_pos)) →
    (∃ (see_time : ℝ), see_time ≤ t / 2)) :=
sorry

end NUMINAMATH_CALUDE_father_sees_boy_less_than_half_time_l1827_182797


namespace NUMINAMATH_CALUDE_expected_defective_60000_l1827_182773

/-- Represents a shipment of computer chips -/
structure Shipment where
  defective : ℕ
  total : ℕ

/-- Calculates the expected number of defective chips in a future shipment -/
def expectedDefective (shipments : List Shipment) (futureTotal : ℕ) : ℕ :=
  let totalDefective := shipments.map (·.defective) |>.sum
  let totalChips := shipments.map (·.total) |>.sum
  (totalDefective * futureTotal) / totalChips

/-- Theorem stating the expected number of defective chips in a shipment of 60,000 -/
theorem expected_defective_60000 (shipments : List Shipment) 
    (h1 : shipments = [
      ⟨2, 5000⟩, 
      ⟨4, 12000⟩, 
      ⟨2, 15000⟩, 
      ⟨4, 16000⟩
    ]) : 
    expectedDefective shipments 60000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_expected_defective_60000_l1827_182773


namespace NUMINAMATH_CALUDE_dads_dimes_count_l1827_182703

/-- The number of dimes Tom's dad gave him -/
def dimes_from_dad (initial_dimes final_dimes : ℕ) : ℕ :=
  final_dimes - initial_dimes

/-- Proof that Tom's dad gave him 33 dimes -/
theorem dads_dimes_count : dimes_from_dad 15 48 = 33 := by
  sorry

end NUMINAMATH_CALUDE_dads_dimes_count_l1827_182703


namespace NUMINAMATH_CALUDE_negation_equivalence_l1827_182717

theorem negation_equivalence (m : ℤ) :
  (¬ ∃ x : ℤ, 2 * x^2 + x + m ≤ 0) ↔ (∀ x : ℤ, 2 * x^2 + x + m > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1827_182717


namespace NUMINAMATH_CALUDE_larger_box_jellybeans_l1827_182781

def jellybeans_in_box (length width height : ℕ) : ℕ := length * width * height * 20

theorem larger_box_jellybeans (l w h : ℕ) :
  jellybeans_in_box l w h = 200 →
  jellybeans_in_box (3 * l) (3 * w) (3 * h) = 5400 :=
by
  sorry

#check larger_box_jellybeans

end NUMINAMATH_CALUDE_larger_box_jellybeans_l1827_182781


namespace NUMINAMATH_CALUDE_tomato_growth_l1827_182765

theorem tomato_growth (initial_tomatoes : ℕ) (increase_factor : ℕ) 
  (h1 : initial_tomatoes = 36) 
  (h2 : increase_factor = 100) : 
  initial_tomatoes * increase_factor = 3600 := by
sorry

end NUMINAMATH_CALUDE_tomato_growth_l1827_182765


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_solutions_specific_equation_l1827_182760

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = -b / a :=
by sorry

theorem sum_of_solutions_specific_equation :
  let a : ℝ := -48
  let b : ℝ := 66
  let c : ℝ := 195
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  r₁ + r₂ = 11/8 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_solutions_specific_equation_l1827_182760


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1827_182780

/-- Given two quadratic equations where the roots of one are three times the roots of the other, 
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (∃ s₁ s₂ : ℝ, (s₁ + s₂ = -c ∧ s₁ * s₂ = a) ∧
               (3 * s₁ + 3 * s₂ = -a ∧ 9 * s₁ * s₂ = b)) →
  b / c = 27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l1827_182780


namespace NUMINAMATH_CALUDE_second_company_base_rate_l1827_182769

/-- Represents the base rate and per-minute charge of a telephone company --/
structure TelephoneCharge where
  baseRate : ℝ
  perMinuteRate : ℝ

/-- Calculates the total charge for a given number of minutes --/
def totalCharge (tc : TelephoneCharge) (minutes : ℝ) : ℝ :=
  tc.baseRate + tc.perMinuteRate * minutes

theorem second_company_base_rate :
  let unitedTelephone : TelephoneCharge := { baseRate := 9, perMinuteRate := 0.25 }
  let otherCompany : TelephoneCharge := { baseRate := x, perMinuteRate := 0.20 }
  totalCharge unitedTelephone 60 = totalCharge otherCompany 60 →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_second_company_base_rate_l1827_182769


namespace NUMINAMATH_CALUDE_race_time_difference_l1827_182734

def malcolm_speed : ℝ := 5.5
def joshua_speed : ℝ := 7.5
def race_distance : ℝ := 12

theorem race_time_difference : 
  (joshua_speed * race_distance) - (malcolm_speed * race_distance) = 24 := by
  sorry

end NUMINAMATH_CALUDE_race_time_difference_l1827_182734


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1827_182706

theorem cube_root_simplification : 
  (50^3 + 60^3 + 70^3 : ℝ)^(1/3) = 10 * 684^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1827_182706


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1827_182755

/-- For a hyperbola with equation x^2/45 - y^2/5 = 1, the distance between its foci is 10√2 -/
theorem hyperbola_foci_distance :
  ∀ x y : ℝ,
  (x^2 / 45) - (y^2 / 5) = 1 →
  ∃ f₁ f₂ : ℝ × ℝ,
  (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 200 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1827_182755


namespace NUMINAMATH_CALUDE_square_difference_l1827_182702

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1827_182702
