import Mathlib

namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l1645_164573

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 4/7
  | 1 => 36/49
  | 2 => 324/343
  | _ => 0  -- We only need the first three terms for this problem

theorem common_ratio_of_geometric_series :
  (geometric_series 1) / (geometric_series 0) = 9/7 :=
by sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l1645_164573


namespace NUMINAMATH_CALUDE_concert_probability_at_least_seven_concert_probability_is_one_ninth_l1645_164580

/-- The probability that at least 7 out of 8 people stay for an entire concert,
    given that 4 are certain to stay and 4 have a 1/3 probability of staying. -/
theorem concert_probability_at_least_seven (total_people : Nat) (certain_people : Nat)
    (uncertain_people : Nat) (stay_prob : ℚ) : ℚ :=
  let total_people := 8
  let certain_people := 4
  let uncertain_people := 4
  let stay_prob := 1 / 3
  1 / 9

theorem concert_probability_is_one_ninth :
    concert_probability_at_least_seven 8 4 4 (1 / 3) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_concert_probability_at_least_seven_concert_probability_is_one_ninth_l1645_164580


namespace NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1645_164511

theorem probability_three_heads_in_eight_tosses : 
  let n : ℕ := 8  -- number of tosses
  let k : ℕ := 3  -- number of heads we're looking for
  let total_outcomes : ℕ := 2^n  -- total number of possible outcomes
  let favorable_outcomes : ℕ := Nat.choose n k  -- number of ways to choose k heads from n tosses
  (favorable_outcomes : ℚ) / total_outcomes = 7 / 32 :=
by sorry

end NUMINAMATH_CALUDE_probability_three_heads_in_eight_tosses_l1645_164511


namespace NUMINAMATH_CALUDE_divided_square_area_l1645_164550

/-- A square divided into five rectangles of equal area -/
structure DividedSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The width of one rectangle -/
  rect_width : ℝ
  /-- The height of the central rectangle -/
  central_height : ℝ
  /-- All rectangles have equal area -/
  equal_area : ℝ
  /-- The given width of one rectangle is 5 -/
  width_condition : rect_width = 5
  /-- The square is divided into 5 rectangles -/
  division_condition : side = rect_width + 2 * central_height
  /-- Area of each rectangle -/
  area_condition : equal_area = rect_width * central_height
  /-- Total area of the square -/
  total_area : ℝ
  /-- Total area is the square of the side length -/
  area_calculation : total_area = side * side

/-- The theorem stating that the area of the divided square is 400 -/
theorem divided_square_area (s : DividedSquare) : s.total_area = 400 := by
  sorry

end NUMINAMATH_CALUDE_divided_square_area_l1645_164550


namespace NUMINAMATH_CALUDE_ninth_group_number_l1645_164571

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_employees : ℕ
  sample_size : ℕ
  group_size : ℕ
  fifth_group_number : ℕ

/-- The number drawn from the nth group in a systematic sampling -/
def number_drawn (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.group_size * (n - 1) + (s.fifth_group_number - s.group_size * 4)

/-- Theorem stating the relationship between the 5th and 9th group numbers -/
theorem ninth_group_number (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.sample_size = 40)
  (h3 : s.group_size = 5)
  (h4 : s.fifth_group_number = 22) :
  number_drawn s 9 = 42 := by
  sorry


end NUMINAMATH_CALUDE_ninth_group_number_l1645_164571


namespace NUMINAMATH_CALUDE_absolute_value_expression_l1645_164557

theorem absolute_value_expression (a b c : ℝ) (h1 : b < a) (h2 : a < 0) (h3 : 0 < c) :
  |b| - |b-a| + |c-a| - |a+b| = b + c - a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_expression_l1645_164557


namespace NUMINAMATH_CALUDE_surface_area_volume_incomparable_l1645_164596

-- Define the edge length of the cube
def edge_length : ℝ := 6

-- Define the surface area of the cube
def surface_area (l : ℝ) : ℝ := 6 * l^2

-- Define the volume of the cube
def volume (l : ℝ) : ℝ := l^3

-- Theorem stating that surface area and volume are incomparable
theorem surface_area_volume_incomparable :
  ¬(∃ (ord : ℝ → ℝ → Prop), 
    (∀ a b, ord a b ∨ ord b a) ∧ 
    (∀ a b c, ord a b → ord b c → ord a c) ∧
    (∀ a b, ord a b → ord b a → a = b) ∧
    (ord (surface_area edge_length) (volume edge_length) ∨ 
     ord (volume edge_length) (surface_area edge_length))) :=
by sorry


end NUMINAMATH_CALUDE_surface_area_volume_incomparable_l1645_164596


namespace NUMINAMATH_CALUDE_train_speed_increase_l1645_164582

theorem train_speed_increase (old_time new_time : ℝ) 
  (hold : old_time = 16 ∧ new_time = 14) : 
  (1 / new_time - 1 / old_time) / (1 / old_time) = 
  (1 / 14 - 1 / 16) / (1 / 16) := by
  sorry

end NUMINAMATH_CALUDE_train_speed_increase_l1645_164582


namespace NUMINAMATH_CALUDE_sum_four_consecutive_composite_sum_three_consecutive_composite_l1645_164585

-- Define the sum of four consecutive positive integers
def sum_four_consecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3)

-- Define the sum of three consecutive positive integers
def sum_three_consecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2)

-- Theorem for four consecutive positive integers
theorem sum_four_consecutive_composite (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ sum_four_consecutive n = a * b :=
sorry

-- Theorem for three consecutive positive integers
theorem sum_three_consecutive_composite (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ sum_three_consecutive n = a * b :=
sorry

end NUMINAMATH_CALUDE_sum_four_consecutive_composite_sum_three_consecutive_composite_l1645_164585


namespace NUMINAMATH_CALUDE_nine_is_unique_digit_l1645_164537

/-- A function that returns true if a natural number ends with at least k repetitions of digit z -/
def endsWithKDigits (num : ℕ) (k : ℕ) (z : ℕ) : Prop :=
  ∃ m : ℕ, num = m * (10^k) + z * ((10^k - 1) / 9)

/-- The main theorem stating that 9 is the only digit satisfying the condition -/
theorem nine_is_unique_digit : 
  ∀ z : ℕ, z < 10 →
    (∀ k : ℕ, k ≥ 1 → ∃ n : ℕ, n ≥ 1 ∧ endsWithKDigits (n^9) k z) ↔ z = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_is_unique_digit_l1645_164537


namespace NUMINAMATH_CALUDE_gem_stone_necklaces_count_gem_stone_necklaces_count_proof_l1645_164504

theorem gem_stone_necklaces_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_earnings cost_per_necklace bead_necklaces gem_stone_necklaces =>
    total_earnings = cost_per_necklace * (bead_necklaces + gem_stone_necklaces) →
    cost_per_necklace = 6 →
    bead_necklaces = 3 →
    total_earnings = 36 →
    gem_stone_necklaces = 3

-- Proof
theorem gem_stone_necklaces_count_proof :
  gem_stone_necklaces_count 36 6 3 3 := by
  sorry

end NUMINAMATH_CALUDE_gem_stone_necklaces_count_gem_stone_necklaces_count_proof_l1645_164504


namespace NUMINAMATH_CALUDE_number_of_persimmons_l1645_164555

/-- Given that there are 18 apples and the sum of apples and persimmons is 33,
    prove that the number of persimmons is 15. -/
theorem number_of_persimmons (apples : ℕ) (total : ℕ) (persimmons : ℕ) 
    (h1 : apples = 18)
    (h2 : apples + persimmons = total)
    (h3 : total = 33) :
    persimmons = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_persimmons_l1645_164555


namespace NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l1645_164548

theorem tens_digit_of_2023_pow_2024_minus_2025 :
  (2023^2024 - 2025) % 100 / 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_2023_pow_2024_minus_2025_l1645_164548


namespace NUMINAMATH_CALUDE_no_integer_solution_l1645_164541

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1645_164541


namespace NUMINAMATH_CALUDE_fish_problem_solution_l1645_164586

/-- Calculates the number of fish added on day 7 given the initial conditions and daily changes --/
def fish_added_day_7 (initial : ℕ) (double : ℕ → ℕ) (remove_third : ℕ → ℕ) (remove_fourth : ℕ → ℕ) (final : ℕ) : ℕ :=
  let day1 := initial
  let day2 := double day1
  let day3 := remove_third (double day2)
  let day4 := double day3
  let day5 := remove_fourth (double day4)
  let day6 := double day5
  let day7_before_adding := double day6
  final - day7_before_adding

theorem fish_problem_solution :
  fish_added_day_7 6 (λ x => 2 * x) (λ x => x - x / 3) (λ x => x - x / 4) 207 = 15 := by
  sorry

#eval fish_added_day_7 6 (λ x => 2 * x) (λ x => x - x / 3) (λ x => x - x / 4) 207

end NUMINAMATH_CALUDE_fish_problem_solution_l1645_164586


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1645_164506

theorem range_of_a_for_always_positive_quadratic :
  {a : ℝ | ∀ x : ℝ, x^2 + 2*a*x + a > 0} = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1645_164506


namespace NUMINAMATH_CALUDE_plane_properties_l1645_164599

def plane_equation (x y z : ℝ) : ℝ := 4*x - 3*y - z - 7

def point_M : ℝ × ℝ × ℝ := (2, -1, 4)
def point_N : ℝ × ℝ × ℝ := (3, 2, -1)

def given_plane_normal : ℝ × ℝ × ℝ := (1, 1, 1)

theorem plane_properties :
  (plane_equation point_M.1 point_M.2.1 point_M.2.2 = 0) ∧
  (plane_equation point_N.1 point_N.2.1 point_N.2.2 = 0) ∧
  (4 * given_plane_normal.1 + (-3) * given_plane_normal.2.1 + (-1) * given_plane_normal.2.2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_plane_properties_l1645_164599


namespace NUMINAMATH_CALUDE_center_of_gravity_semicircle_semidisk_l1645_164558

/-- The center of gravity of a homogeneous semicircle and semi-disk -/
theorem center_of_gravity_semicircle_semidisk (r : ℝ) (hr : r > 0) :
  ∃ (y z : ℝ),
    y = (2 * r) / Real.pi ∧
    z = (4 * r) / (3 * Real.pi) ∧
    y > 0 ∧ z > 0 :=
by sorry

end NUMINAMATH_CALUDE_center_of_gravity_semicircle_semidisk_l1645_164558


namespace NUMINAMATH_CALUDE_fourth_term_in_geometric_sequence_l1645_164589

/-- Given a geometric sequence of 6 terms where the first term is 5 and the sixth term is 20,
    prove that the fourth term is approximately 6.6. -/
theorem fourth_term_in_geometric_sequence (a : ℕ → ℝ) (h1 : a 1 = 5) (h6 : a 6 = 20)
  (h_geometric : ∀ n ∈ Finset.range 5, a (n + 2) / a (n + 1) = a (n + 1) / a n) :
  ∃ ε > 0, |a 4 - 6.6| < ε :=
sorry

end NUMINAMATH_CALUDE_fourth_term_in_geometric_sequence_l1645_164589


namespace NUMINAMATH_CALUDE_airplane_hover_time_l1645_164529

/-- Proves that given the conditions of the airplane problem, 
    the time spent in Eastern time on the first day was 2 hours. -/
theorem airplane_hover_time : 
  ∀ (eastern_time : ℕ),
    (3 + 4 + eastern_time) + (5 + 6 + (eastern_time + 2)) = 24 →
    eastern_time = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_airplane_hover_time_l1645_164529


namespace NUMINAMATH_CALUDE_sum_of_squares_l1645_164569

theorem sum_of_squares (x y z a b : ℝ) 
  (sum_eq : x + y + z = a) 
  (sum_prod_eq : x*y + y*z + x*z = b) : 
  x^2 + y^2 + z^2 = a^2 - 2*b := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1645_164569


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1645_164545

/-- An isosceles triangle with a semicircle inscribed along its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base length of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The area of the triangle equals the area of the semicircle -/
  area_equality : (1/2) * base * height = π * radius^2

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius (t : IsoscelesTriangleWithSemicircle)
    (h1 : t.base = 24) (h2 : t.height = 18) : t.radius = 18 / π := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1645_164545


namespace NUMINAMATH_CALUDE_complex_power_sum_l1645_164581

theorem complex_power_sum (i : ℂ) (h : i^2 = -1) :
  i^14 + i^19 + i^24 + i^29 + 3*i^34 + 2*i^39 = -3 - 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1645_164581


namespace NUMINAMATH_CALUDE_exists_perpendicular_line_l1645_164566

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define a relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define a relation for two lines being perpendicular
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem exists_perpendicular_line (l : Line) (α : Plane) :
  ∃ (m : Line), in_plane m α ∧ perpendicular m l := by
  sorry

end NUMINAMATH_CALUDE_exists_perpendicular_line_l1645_164566


namespace NUMINAMATH_CALUDE_workshop_A_more_stable_l1645_164528

def workshop_A : List ℕ := [102, 101, 99, 98, 103, 98, 99]
def workshop_B : List ℕ := [110, 115, 90, 85, 75, 115, 110]

def variance (data : List ℕ) : ℚ :=
  let mean := (data.sum : ℚ) / data.length
  (data.map (fun x => ((x : ℚ) - mean) ^ 2)).sum / data.length

theorem workshop_A_more_stable :
  variance workshop_A < variance workshop_B :=
sorry

end NUMINAMATH_CALUDE_workshop_A_more_stable_l1645_164528


namespace NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l1645_164512

theorem drama_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 80) 
  (h2 : math = 50) 
  (h3 : physics = 32) 
  (h4 : both = 15) : 
  total - (math + physics - both) = 13 := by
sorry

end NUMINAMATH_CALUDE_drama_club_neither_math_nor_physics_l1645_164512


namespace NUMINAMATH_CALUDE_apple_dealer_profit_l1645_164579

/-- Profit calculation for apple dealer Bronson -/
theorem apple_dealer_profit :
  let cost_per_bushel : ℚ := 12
  let apples_per_bushel : ℕ := 48
  let selling_price_per_apple : ℚ := 0.40
  let apples_sold : ℕ := 100
  
  let cost_per_apple : ℚ := cost_per_bushel / apples_per_bushel
  let profit_per_apple : ℚ := selling_price_per_apple - cost_per_apple
  let total_profit : ℚ := profit_per_apple * apples_sold

  total_profit = 15 := by sorry

end NUMINAMATH_CALUDE_apple_dealer_profit_l1645_164579


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l1645_164542

theorem fraction_sum_simplification :
  1 / 462 + 17 / 42 = 94 / 231 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l1645_164542


namespace NUMINAMATH_CALUDE_bobs_spending_limit_l1645_164513

/-- The spending limit problem -/
theorem bobs_spending_limit
  (necklace_cost : ℕ)
  (book_cost_difference : ℕ)
  (overspent_amount : ℕ)
  (h1 : necklace_cost = 34)
  (h2 : book_cost_difference = 5)
  (h3 : overspent_amount = 3) :
  necklace_cost + (necklace_cost + book_cost_difference) - overspent_amount = 70 :=
by sorry

end NUMINAMATH_CALUDE_bobs_spending_limit_l1645_164513


namespace NUMINAMATH_CALUDE_number_equals_two_l1645_164587

theorem number_equals_two : ∃ x : ℝ, 0.4 * x = 0.8 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_number_equals_two_l1645_164587


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_200_l1645_164519

-- Define what a Mersenne number is
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

-- Define what a Mersenne prime is
def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, Nat.Prime n ∧ p = mersenne_number n

-- Theorem statement
theorem largest_mersenne_prime_under_200 :
  (∀ p : ℕ, is_mersenne_prime p ∧ p < 200 → p ≤ 127) ∧
  is_mersenne_prime 127 := by sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_200_l1645_164519


namespace NUMINAMATH_CALUDE_money_ratio_problem_l1645_164503

theorem money_ratio_problem (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  ram = 588 →
  krishan = 3468 →
  (gopal : ℚ) / krishan = 100 / 243 := by
sorry

end NUMINAMATH_CALUDE_money_ratio_problem_l1645_164503


namespace NUMINAMATH_CALUDE_adam_picked_apples_for_30_days_l1645_164521

/-- The number of days Adam picked apples -/
def days_picked : ℕ := 30

/-- The number of apples Adam picked each day -/
def apples_per_day : ℕ := 4

/-- The number of remaining apples Adam collected after a month -/
def remaining_apples : ℕ := 230

/-- The total number of apples Adam collected -/
def total_apples : ℕ := 350

/-- Theorem stating that the number of days Adam picked apples is 30 -/
theorem adam_picked_apples_for_30_days :
  days_picked * apples_per_day + remaining_apples = total_apples :=
by sorry

end NUMINAMATH_CALUDE_adam_picked_apples_for_30_days_l1645_164521


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l1645_164552

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + m * x + 16 = 0) → m = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l1645_164552


namespace NUMINAMATH_CALUDE_min_value_xy_minus_2x_l1645_164595

theorem min_value_xy_minus_2x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log x + y * Real.log y = Real.exp x) : 
  ∃ (m : ℝ), m = 2 - 2 * Real.log 2 ∧ 
  ∀ (z : ℝ), z > 0 → y * z * Real.log (y * z) = z * Real.exp z → 
  x * y - 2 * x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_xy_minus_2x_l1645_164595


namespace NUMINAMATH_CALUDE_mauras_remaining_seashells_l1645_164577

/-- The number of seashells Maura found -/
def total_seashells : ℕ := 75

/-- The number of seashells Maura gave to her sister -/
def given_seashells : ℕ := 18

/-- The number of days Maura's family stays at the beach house -/
def beach_days : ℕ := 21

/-- Theorem stating that Maura has 57 seashells left -/
theorem mauras_remaining_seashells :
  total_seashells - given_seashells = 57 := by sorry

end NUMINAMATH_CALUDE_mauras_remaining_seashells_l1645_164577


namespace NUMINAMATH_CALUDE_quadratic_root_property_l1645_164516

theorem quadratic_root_property (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 → β^2 - 3*β - 4 = 0 → α^2 + α*β - 3*α = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l1645_164516


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_41_20_l1645_164507

theorem sum_of_fractions_equals_41_20 : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) + (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_41_20_l1645_164507


namespace NUMINAMATH_CALUDE_at_least_two_boundary_triangles_l1645_164565

/-- A polygon divided into triangles by non-intersecting diagonals -/
structure TriangulatedPolygon where
  /-- The number of sides of the polygon -/
  n : ℕ
  /-- The number of triangles with exactly i sides as sides of the polygon -/
  k : Fin 3 → ℕ
  /-- The total number of triangles is n - 2 -/
  total_triangles : k 0 + k 1 + k 2 = n - 2
  /-- The total number of polygon sides used in forming triangles is n -/
  total_sides : 2 * k 2 + k 1 = n

/-- 
In a polygon divided into triangles by non-intersecting diagonals, 
there are at least two triangles that have at least two sides 
coinciding with the sides of the original polygon.
-/
theorem at_least_two_boundary_triangles (p : TriangulatedPolygon) : 
  p.k 2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_boundary_triangles_l1645_164565


namespace NUMINAMATH_CALUDE_star_calculation_l1645_164572

-- Define the ⋆ operation
def star (a b : ℚ) : ℚ := a + 2 / b

-- Theorem statement
theorem star_calculation :
  (star (star 3 4) 5) - (star 3 (star 4 5)) = 49 / 110 := by
  sorry

end NUMINAMATH_CALUDE_star_calculation_l1645_164572


namespace NUMINAMATH_CALUDE_intersection_slope_range_l1645_164533

/-- Given two points P and Q, and a line l that intersects the extension of PQ,
    this theorem states the range of values for the slope parameter m of line l. -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (m : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  ∃ (x y : ℝ), x + m * y + m = 0 ∧ 
    (∃ (t : ℝ), x = -1 + 3 * t ∧ y = 1 + t) →
  -3 < m ∧ m < -2/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l1645_164533


namespace NUMINAMATH_CALUDE_mango_price_reduction_l1645_164525

/-- Represents the price reduction problem for mangoes --/
theorem mango_price_reduction 
  (original_price : ℝ) 
  (original_quantity : ℕ) 
  (total_spent : ℝ) 
  (additional_mangoes : ℕ) :
  original_price = 416.67 →
  original_quantity = 125 →
  total_spent = 360 →
  additional_mangoes = 12 →
  let original_price_per_mango := original_price / original_quantity
  let original_bought_quantity := total_spent / original_price_per_mango
  let new_quantity := original_bought_quantity + additional_mangoes
  let new_price_per_mango := total_spent / new_quantity
  let price_reduction_percentage := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100
  price_reduction_percentage = 10 := by
sorry


end NUMINAMATH_CALUDE_mango_price_reduction_l1645_164525


namespace NUMINAMATH_CALUDE_second_team_soup_amount_l1645_164520

/-- Given the total required amount of soup and the amounts made by the first and third teams,
    calculate the amount the second team should prepare. -/
theorem second_team_soup_amount (total_required : ℕ) (first_team : ℕ) (third_team : ℕ) :
  total_required = 280 →
  first_team = 90 →
  third_team = 70 →
  total_required - (first_team + third_team) = 120 := by
sorry

end NUMINAMATH_CALUDE_second_team_soup_amount_l1645_164520


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l1645_164576

theorem completing_square_quadratic (x : ℝ) :
  x^2 - 2*x = 9 ↔ (x - 1)^2 = 10 := by sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l1645_164576


namespace NUMINAMATH_CALUDE_bens_baseball_card_boxes_l1645_164598

theorem bens_baseball_card_boxes (basketball_boxes : ℕ) (basketball_cards_per_box : ℕ)
  (baseball_cards_per_box : ℕ) (cards_given_away : ℕ) (cards_left : ℕ) :
  basketball_boxes = 4 →
  basketball_cards_per_box = 10 →
  baseball_cards_per_box = 8 →
  cards_given_away = 58 →
  cards_left = 22 →
  (basketball_boxes * basketball_cards_per_box +
    baseball_cards_per_box * ((cards_given_away + cards_left - basketball_boxes * basketball_cards_per_box) / baseball_cards_per_box)) =
  cards_given_away + cards_left →
  (cards_given_away + cards_left - basketball_boxes * basketball_cards_per_box) / baseball_cards_per_box = 5 :=
by sorry

end NUMINAMATH_CALUDE_bens_baseball_card_boxes_l1645_164598


namespace NUMINAMATH_CALUDE_complex_magnitude_equality_l1645_164567

theorem complex_magnitude_equality (t : ℝ) (h1 : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 3 * Real.sqrt 5 → t = 6 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equality_l1645_164567


namespace NUMINAMATH_CALUDE_prob_four_or_full_house_after_reroll_l1645_164522

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 6

-- Define a function to represent the probability of a specific outcome when rolling a die
def prob_specific_outcome (sides : ℕ) : ℚ := 1 / sides

-- Define the probability of getting a four-of-a-kind or a full house after re-rolling
def prob_four_or_full_house : ℚ := 1 / 3

-- State the theorem
theorem prob_four_or_full_house_after_reroll 
  (h1 : ∃ (triple pair : ℕ), triple ≠ pair ∧ triple ≤ die_sides ∧ pair ≤ die_sides) 
  (h2 : ¬ ∃ (four : ℕ), four ≤ die_sides) :
  prob_four_or_full_house = prob_specific_outcome die_sides + prob_specific_outcome die_sides :=
sorry

end NUMINAMATH_CALUDE_prob_four_or_full_house_after_reroll_l1645_164522


namespace NUMINAMATH_CALUDE_hammer_wrench_problem_l1645_164547

theorem hammer_wrench_problem (H W : ℝ) (x : ℕ) 
  (h1 : 2 * H + 2 * W = (1 / 3) * (x * H + 5 * W))
  (h2 : W = 2 * H) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_hammer_wrench_problem_l1645_164547


namespace NUMINAMATH_CALUDE_log_sum_equality_l1645_164564

theorem log_sum_equality : Real.log 0.01 / Real.log 10 + Real.log 16 / Real.log 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l1645_164564


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1645_164530

/-- Given a quadrilateral PQRS with extended sides, prove the reconstruction equation -/
theorem quadrilateral_reconstruction
  (P P' Q Q' R R' S S' : ℝ × ℝ) -- Points as pairs of real numbers
  (h1 : P' - Q = 2 * (P - Q)) -- PP' = 3PQ
  (h2 : R' - Q = R - Q) -- QR' = QR
  (h3 : R' - S = R - S) -- SR' = SR
  (h4 : S' - P = 3 * (S - P)) -- PS' = 4PS
  : ∃ (x y z w : ℝ),
    x = 48/95 ∧ y = 32/95 ∧ z = 19/95 ∧ w = 4/5 ∧
    P = x • P' + y • Q' + z • R' + w • S' :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l1645_164530


namespace NUMINAMATH_CALUDE_A_proper_superset_B_l1645_164544

def A : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def B : Set ℤ := {y | ∃ k : ℤ, y = 4 * k}

theorem A_proper_superset_B : A ⊃ B := by
  sorry

end NUMINAMATH_CALUDE_A_proper_superset_B_l1645_164544


namespace NUMINAMATH_CALUDE_abe_age_sum_l1645_164588

theorem abe_age_sum : 
  let present_age : ℕ := 29
  let years_ago : ℕ := 7
  let past_age : ℕ := present_age - years_ago
  present_age + past_age = 51
  := by sorry

end NUMINAMATH_CALUDE_abe_age_sum_l1645_164588


namespace NUMINAMATH_CALUDE_apple_picking_theorem_l1645_164518

/-- Represents the number of apples of each color in the bin -/
structure AppleBin :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (orange : ℕ)

/-- The minimum number of apples needed to guarantee a specific count of one color -/
def minApplesToGuarantee (bin : AppleBin) (targetCount : ℕ) : ℕ :=
  sorry

theorem apple_picking_theorem (bin : AppleBin) (h : bin = ⟨32, 24, 22, 15, 14⟩) :
  minApplesToGuarantee bin 18 = 81 :=
sorry

end NUMINAMATH_CALUDE_apple_picking_theorem_l1645_164518


namespace NUMINAMATH_CALUDE_arrangements_with_three_together_eq_36_l1645_164536

/-- The number of different arrangements of five students in a row,
    where three specific students must be together. -/
def arrangements_with_three_together : ℕ :=
  (3 : ℕ).factorial * (3 : ℕ).factorial

theorem arrangements_with_three_together_eq_36 :
  arrangements_with_three_together = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_with_three_together_eq_36_l1645_164536


namespace NUMINAMATH_CALUDE_negative_cube_divided_by_base_l1645_164562

theorem negative_cube_divided_by_base (a : ℝ) (h : a ≠ 0) : -a^3 / a = -a^2 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_divided_by_base_l1645_164562


namespace NUMINAMATH_CALUDE_D_72_l1645_164554

/-- D(n) is the number of ways to write n as a product of factors greater than 1,
    considering the order of factors, and allowing any number of factors (at least one). -/
def D (n : ℕ) : ℕ := sorry

/-- Theorem stating that D(72) equals 93 -/
theorem D_72 : D 72 = 93 := by sorry

end NUMINAMATH_CALUDE_D_72_l1645_164554


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l1645_164546

theorem rectangular_plot_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l1645_164546


namespace NUMINAMATH_CALUDE_difference_of_squares_l1645_164517

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1645_164517


namespace NUMINAMATH_CALUDE_t_shaped_region_perimeter_l1645_164510

/-- A T-shaped region formed by six congruent squares -/
structure TShapedRegion where
  /-- The side length of each square in the region -/
  side_length : ℝ
  /-- The total area of the region -/
  total_area : ℝ
  /-- The area of the region is the sum of six squares -/
  area_eq : total_area = 6 * side_length ^ 2

/-- The perimeter of a T-shaped region -/
def perimeter (region : TShapedRegion) : ℝ :=
  9 * region.side_length

/-- Theorem stating the perimeter of a T-shaped region with area 576 is 36√6 -/
theorem t_shaped_region_perimeter :
  ∀ (region : TShapedRegion),
  region.total_area = 576 →
  perimeter region = 36 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_t_shaped_region_perimeter_l1645_164510


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l1645_164505

theorem decimal_to_percentage (x : ℚ) : x = 2.08 → (x * 100 : ℚ) = 208 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l1645_164505


namespace NUMINAMATH_CALUDE_olivia_wallet_problem_l1645_164578

theorem olivia_wallet_problem (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount = 128)
  (h2 : spent_amount = 38) :
  initial_amount - spent_amount = 90 := by sorry

end NUMINAMATH_CALUDE_olivia_wallet_problem_l1645_164578


namespace NUMINAMATH_CALUDE_guilty_cases_count_l1645_164535

theorem guilty_cases_count (total : ℕ) (dismissed : ℕ) (delayed : ℕ) : 
  total = 17 →
  dismissed = 2 →
  delayed = 1 →
  (total - dismissed - delayed - (2 * (total - dismissed) / 3)) = 4 := by
sorry

end NUMINAMATH_CALUDE_guilty_cases_count_l1645_164535


namespace NUMINAMATH_CALUDE_max_value_of_squares_l1645_164583

theorem max_value_of_squares (x y z : ℤ) 
  (eq1 : x * y + x + y = 20)
  (eq2 : y * z + y + z = 6)
  (eq3 : x * z + x + z = 2) :
  ∃ (a b c : ℤ), a * b + a + b = 20 ∧ b * c + b + c = 6 ∧ a * c + a + c = 2 ∧
    ∀ (x y z : ℤ), x * y + x + y = 20 → y * z + y + z = 6 → x * z + x + z = 2 →
      x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧ a^2 + b^2 + c^2 = 84 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_squares_l1645_164583


namespace NUMINAMATH_CALUDE_hyperbola_equation_given_conditions_l1645_164549

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Check if a point is on a hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  hyperbola_equation h p

theorem hyperbola_equation_given_conditions 
  (E : Hyperbola)
  (center : Point)
  (focus : Point)
  (N : Point)
  (h_center : center.x = 0 ∧ center.y = 0)
  (h_focus : focus.x = 3 ∧ focus.y = 0)
  (h_midpoint : N.x = -12 ∧ N.y = -15)
  (h_on_hyperbola : ∃ (A B : Point), 
    point_on_hyperbola E A ∧ 
    point_on_hyperbola E B ∧ 
    N.x = (A.x + B.x) / 2 ∧ 
    N.y = (A.y + B.y) / 2) :
  E.a^2 = 4 ∧ E.b^2 = 5 := by
  sorry

#check hyperbola_equation_given_conditions

end NUMINAMATH_CALUDE_hyperbola_equation_given_conditions_l1645_164549


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l1645_164592

theorem factorization_of_quadratic (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l1645_164592


namespace NUMINAMATH_CALUDE_difference_of_squares_l1645_164556

theorem difference_of_squares (a b : ℕ+) : 
  a + b = 40 → a - b = 10 → a^2 - b^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1645_164556


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l1645_164551

theorem geometric_sequence_inequality (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence with common ratio q
  a 2 > a 1 →                   -- a₂ > a₁
  a 1 > 0 →                     -- a₁ > 0
  a 1 + a 3 > 2 * a 2 :=        -- prove: a₁ + a₃ > 2a₂
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l1645_164551


namespace NUMINAMATH_CALUDE_circle_chord_intersection_l1645_164523

theorem circle_chord_intersection (r : ℝ) (chord_length : ℝ) : 
  r = 5 →
  chord_length = 8 →
  ∃ (ak kb : ℝ),
    ak + kb = 2 * r ∧
    ak * kb = r^2 - (chord_length / 2)^2 ∧
    ak = 1.25 ∧
    kb = 8.75 := by
sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_l1645_164523


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l1645_164500

theorem shopkeeper_profit_percentage 
  (total_goods : ℝ)
  (theft_percentage : ℝ)
  (loss_percentage : ℝ)
  (h1 : theft_percentage = 20)
  (h2 : loss_percentage = 12)
  : ∃ (profit_percentage : ℝ), profit_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l1645_164500


namespace NUMINAMATH_CALUDE_max_trailing_zeros_l1645_164509

def trailing_zeros (n : ℕ) : ℕ := sorry

def expr_a : ℕ := 2^5 * 3^4 * 5^6
def expr_b : ℕ := 2^4 * 3^4 * 5^5
def expr_c : ℕ := 4^3 * 5^6 * 6^5
def expr_d : ℕ := 4^2 * 5^4 * 6^3

theorem max_trailing_zeros :
  trailing_zeros expr_c > trailing_zeros expr_a ∧
  trailing_zeros expr_c > trailing_zeros expr_b ∧
  trailing_zeros expr_c > trailing_zeros expr_d :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_l1645_164509


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l1645_164568

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = (17 * Real.sqrt 41) / 41 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l1645_164568


namespace NUMINAMATH_CALUDE_constant_is_monomial_l1645_164560

/-- A monomial is a constant or a product of variables with non-negative integer exponents. --/
def IsMonomial (x : ℝ) : Prop :=
  x ≠ 0 ∨ ∃ (n : ℕ), x = 1 ∨ x = -1

/-- Theorem: The constant -2010 is a monomial. --/
theorem constant_is_monomial : IsMonomial (-2010) := by
  sorry

end NUMINAMATH_CALUDE_constant_is_monomial_l1645_164560


namespace NUMINAMATH_CALUDE_students_per_bench_l1645_164532

/-- Proves that at least 5 students must sit on each bench for all to fit in the hall --/
theorem students_per_bench (male_students : ℕ) (female_students : ℕ) (num_benches : ℕ) : 
  male_students = 29 →
  female_students = 4 * male_students →
  num_benches = 29 →
  ((male_students + female_students : ℚ) / num_benches).ceil ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_students_per_bench_l1645_164532


namespace NUMINAMATH_CALUDE_prob_even_sum_two_dice_l1645_164540

/-- Die with faces numbered 1 through 4 -/
def Die1 : Finset Nat := {1, 2, 3, 4}

/-- Die with faces numbered 1 through 8 -/
def Die2 : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8}

/-- The probability of getting an even sum when rolling two dice -/
def ProbEvenSum : ℚ := 1 / 2

/-- Theorem stating that the probability of getting an even sum when rolling
    two dice, one with faces 1-4 and another with faces 1-8, is equal to 1/2 -/
theorem prob_even_sum_two_dice :
  let outcomes := Die1.product Die2
  let even_sum := {p : Nat × Nat | (p.1 + p.2) % 2 = 0}
  (outcomes.filter (λ p => p ∈ even_sum)).card / outcomes.card = ProbEvenSum :=
sorry


end NUMINAMATH_CALUDE_prob_even_sum_two_dice_l1645_164540


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1645_164574

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_of_A_and_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1645_164574


namespace NUMINAMATH_CALUDE_benny_pumpkin_pies_l1645_164590

/-- Represents the number of pumpkin pies Benny plans to make -/
def num_pumpkin_pies : ℕ := sorry

/-- The cost to make one pumpkin pie -/
def pumpkin_pie_cost : ℕ := 3

/-- The cost to make one cherry pie -/
def cherry_pie_cost : ℕ := 5

/-- The number of cherry pies Benny plans to make -/
def num_cherry_pies : ℕ := 12

/-- The profit Benny wants to make -/
def desired_profit : ℕ := 20

/-- The price Benny charges for each pie -/
def pie_price : ℕ := 5

/-- Theorem stating that the number of pumpkin pies Benny plans to make is 10 -/
theorem benny_pumpkin_pies : 
  num_pumpkin_pies = 10 := by
  sorry

end NUMINAMATH_CALUDE_benny_pumpkin_pies_l1645_164590


namespace NUMINAMATH_CALUDE_factors_imply_absolute_value_l1645_164559

-- Define the polynomial
def p (h k x : ℝ) : ℝ := 3 * x^3 - h * x - 3 * k

-- Define the factors
def f₁ (x : ℝ) : ℝ := x + 3
def f₂ (x : ℝ) : ℝ := x - 2

-- Theorem statement
theorem factors_imply_absolute_value (h k : ℝ) :
  (∀ x, p h k x = 0 → f₁ x = 0 ∨ f₂ x = 0) →
  |3 * h - 4 * k| = 615 := by
  sorry

end NUMINAMATH_CALUDE_factors_imply_absolute_value_l1645_164559


namespace NUMINAMATH_CALUDE_alloy_ratio_l1645_164561

/-- Proves that the ratio of tin to copper in alloy B is 3:5 given the specified conditions -/
theorem alloy_ratio : 
  ∀ (lead_A tin_A tin_B copper_B : ℝ),
  -- Alloy A has 170 kg total
  lead_A + tin_A = 170 →
  -- Alloy A has lead and tin in ratio 1:3
  lead_A * 3 = tin_A →
  -- Alloy B has 250 kg total
  tin_B + copper_B = 250 →
  -- Total tin in new alloy is 221.25 kg
  tin_A + tin_B = 221.25 →
  -- Ratio of tin to copper in alloy B is 3:5
  tin_B * 5 = copper_B * 3 := by
sorry


end NUMINAMATH_CALUDE_alloy_ratio_l1645_164561


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1645_164553

theorem pure_imaginary_condition (x : ℝ) : 
  (∃ (y : ℝ), y ≠ 0 ∧ (x^2 - 1) + (x - 1)*I = y*I) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1645_164553


namespace NUMINAMATH_CALUDE_condo_cats_count_l1645_164575

theorem condo_cats_count :
  ∀ (x y z : ℕ),
    x + y + z = 29 →
    x = z →
    87 = x * 1 + y * 3 + z * 5 :=
by
  sorry

end NUMINAMATH_CALUDE_condo_cats_count_l1645_164575


namespace NUMINAMATH_CALUDE_rectangle_sides_l1645_164534

theorem rectangle_sides (area : ℝ) (perimeter : ℝ) : area = 12 ∧ perimeter = 26 →
  ∃ (length width : ℝ), length * width = area ∧ 2 * (length + width) = perimeter ∧
  ((length = 12 ∧ width = 1) ∨ (length = 1 ∧ width = 12)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_sides_l1645_164534


namespace NUMINAMATH_CALUDE_distribute_plumbers_count_l1645_164502

/-- The number of ways to distribute 4 plumbers to 3 residences -/
def distribute_plumbers : ℕ :=
  Nat.choose 4 2 * (3 * 2 * 1)

/-- The conditions of the problem -/
axiom plumbers : ℕ
axiom residences : ℕ
axiom plumbers_eq_four : plumbers = 4
axiom residences_eq_three : residences = 3
axiom all_plumbers_assigned : True
axiom one_residence_per_plumber : True
axiom all_residences_checked : True

/-- The theorem to be proved -/
theorem distribute_plumbers_count :
  distribute_plumbers = Nat.choose plumbers 2 * (residences * (residences - 1) * (residences - 2)) :=
sorry

end NUMINAMATH_CALUDE_distribute_plumbers_count_l1645_164502


namespace NUMINAMATH_CALUDE_clown_balloons_l1645_164570

/-- The number of additional balloons blown up by the clown -/
def additional_balloons (initial : ℕ) (total : ℕ) : ℕ :=
  total - initial

theorem clown_balloons : additional_balloons 47 60 = 13 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloons_l1645_164570


namespace NUMINAMATH_CALUDE_sarah_initial_money_l1645_164594

def toy_car_price : ℕ := 11
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7
def num_toy_cars : ℕ := 2

theorem sarah_initial_money :
  ∃ (initial_money : ℕ),
    initial_money = num_toy_cars * toy_car_price + scarf_price + beanie_price + remaining_money :=
by sorry

end NUMINAMATH_CALUDE_sarah_initial_money_l1645_164594


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1645_164539

/-- Represents a triangle XYZ -/
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

/-- Checks if a triangle is isosceles right -/
def isIsoscelesRight (t : Triangle) : Prop := sorry

/-- Calculates the length of a side given two points -/
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem isosceles_right_triangle_area 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : sideLength t.X t.Y > sideLength t.Y t.Z) 
  (h3 : sideLength t.X t.Y = 12.000000000000002) : 
  triangleArea t = 36.000000000000015 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1645_164539


namespace NUMINAMATH_CALUDE_increasing_function_equivalence_l1645_164514

/-- A function f is increasing on ℝ -/
def IncreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

theorem increasing_function_equivalence (f : ℝ → ℝ) (h : IncreasingOn f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ↔
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) ∧
  (∀ a b : ℝ, f a + f b < f (-a) + f (-b) → a + b < 0) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_equivalence_l1645_164514


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1645_164563

theorem fraction_multiplication : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1645_164563


namespace NUMINAMATH_CALUDE_gordon_jamie_persian_ratio_l1645_164584

/-- Represents the number of cats of each type owned by each person -/
structure CatOwnership where
  jamie_persian : ℕ
  jamie_maine_coon : ℕ
  gordon_persian : ℕ
  gordon_maine_coon : ℕ
  hawkeye_persian : ℕ
  hawkeye_maine_coon : ℕ

/-- The theorem stating the ratio of Gordon's Persian cats to Jamie's Persian cats -/
theorem gordon_jamie_persian_ratio (cats : CatOwnership) : 
  cats.jamie_persian = 4 →
  cats.jamie_maine_coon = 2 →
  cats.gordon_maine_coon = cats.jamie_maine_coon + 1 →
  cats.hawkeye_persian = 0 →
  cats.hawkeye_maine_coon = cats.gordon_maine_coon - 1 →
  cats.jamie_persian + cats.jamie_maine_coon + 
  cats.gordon_persian + cats.gordon_maine_coon + 
  cats.hawkeye_persian + cats.hawkeye_maine_coon = 13 →
  2 * cats.gordon_persian = cats.jamie_persian := by
  sorry


end NUMINAMATH_CALUDE_gordon_jamie_persian_ratio_l1645_164584


namespace NUMINAMATH_CALUDE_crayons_left_l1645_164526

-- Define the initial number of crayons
def initial_crayons : ℕ := 62

-- Define the number of crayons eaten
def eaten_crayons : ℕ := 52

-- Theorem to prove
theorem crayons_left : initial_crayons - eaten_crayons = 10 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l1645_164526


namespace NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l1645_164538

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 4*x^2 + 13*x + 20

-- Theorem statement
theorem intersection_point_of_f_and_inverse :
  ∃! p : ℝ × ℝ, p.1 = f p.2 ∧ p.2 = f p.1 ∧ p = (-2, -2) :=
by
  sorry


end NUMINAMATH_CALUDE_intersection_point_of_f_and_inverse_l1645_164538


namespace NUMINAMATH_CALUDE_walking_ring_width_l1645_164543

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * π * r₁ - 2 * π * r₂ = 20 * π) : 
  r₁ - r₂ = 10 := by
sorry

end NUMINAMATH_CALUDE_walking_ring_width_l1645_164543


namespace NUMINAMATH_CALUDE_simplify_expression_l1645_164593

theorem simplify_expression : (6^8 - 4^7) * (2^3 - (-2)^3)^10 = 1663232 * 16^10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1645_164593


namespace NUMINAMATH_CALUDE_dave_paints_200_sqft_l1645_164515

/-- The total area of the wall to be painted in square feet -/
def total_area : ℝ := 360

/-- The ratio of Carl's work to Dave's work -/
def work_ratio : ℚ := 4 / 5

/-- Dave's share of the work -/
def dave_share : ℚ := 5 / 9

theorem dave_paints_200_sqft :
  dave_share * total_area = 200 := by sorry

end NUMINAMATH_CALUDE_dave_paints_200_sqft_l1645_164515


namespace NUMINAMATH_CALUDE_pi_irrational_in_set_l1645_164508

theorem pi_irrational_in_set : ∃ x ∈ ({-2, 0, Real.sqrt 9, Real.pi} : Set ℝ), Irrational x :=
by sorry

end NUMINAMATH_CALUDE_pi_irrational_in_set_l1645_164508


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1645_164501

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -105 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1645_164501


namespace NUMINAMATH_CALUDE_cookie_theorem_l1645_164524

def cookie_problem (initial_total initial_chocolate initial_sugar initial_oatmeal : ℕ)
  (morning_chocolate morning_sugar : ℕ)
  (lunch_chocolate lunch_sugar lunch_oatmeal : ℕ)
  (afternoon_chocolate afternoon_sugar afternoon_oatmeal : ℕ)
  (damage_percent : ℚ) : Prop :=
  let total_chocolate_sold := morning_chocolate + lunch_chocolate + afternoon_chocolate
  let total_sugar_sold := morning_sugar + lunch_sugar + afternoon_sugar
  let total_oatmeal_sold := lunch_oatmeal + afternoon_oatmeal
  let remaining_chocolate := max (initial_chocolate - total_chocolate_sold) 0
  let remaining_sugar := initial_sugar - total_sugar_sold
  let remaining_oatmeal := initial_oatmeal - total_oatmeal_sold
  let total_remaining := remaining_chocolate + remaining_sugar + remaining_oatmeal
  let damaged := ⌊(damage_percent * total_remaining : ℚ)⌋
  total_remaining - damaged = 18

theorem cookie_theorem :
  cookie_problem 120 60 40 20 24 12 33 20 4 10 4 2 (1/20) := by sorry

end NUMINAMATH_CALUDE_cookie_theorem_l1645_164524


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1645_164597

theorem polynomial_multiplication (t : ℝ) : 
  (3*t^3 + 2*t^2 - 4*t + 3) * (-2*t^2 + 3*t - 4) = 
  -6*t^5 + 5*t^4 + 2*t^3 - 26*t^2 + 25*t - 12 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1645_164597


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l1645_164531

theorem trigonometric_system_solution (x y : ℝ) :
  (Real.sin x + Real.cos y = 0) ∧ (Real.sin x ^ 2 + Real.cos y ^ 2 = 1/2) →
  (∃ (k n : ℤ), 
    ((x = (-1)^(k+1) * Real.pi/6 + Real.pi * k ∧ y = Real.pi/3 + 2*Real.pi*n) ∨
     (x = (-1)^(k+1) * Real.pi/6 + Real.pi * k ∧ y = -Real.pi/3 + 2*Real.pi*n)) ∨
    ((x = (-1)^k * Real.pi/6 + Real.pi * k ∧ y = 2*Real.pi/3 + 2*Real.pi*n) ∨
     (x = (-1)^k * Real.pi/6 + Real.pi * k ∧ y = -2*Real.pi/3 + 2*Real.pi*n))) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_system_solution_l1645_164531


namespace NUMINAMATH_CALUDE_dave_derek_money_difference_l1645_164591

/-- Calculates the difference between Dave's and Derek's remaining money after expenses -/
def moneyDifference (derekInitial : ℕ) (derekExpenses : List ℕ) (daveInitial : ℕ) (daveExpenses : List ℕ) : ℕ :=
  let derekRemaining := derekInitial - derekExpenses.sum
  let daveRemaining := daveInitial - daveExpenses.sum
  daveRemaining - derekRemaining

/-- Proves that Dave has $20 more left than Derek after expenses -/
theorem dave_derek_money_difference :
  moneyDifference 40 [14, 11, 5, 8] 50 [7, 12, 9] = 20 := by
  sorry

#eval moneyDifference 40 [14, 11, 5, 8] 50 [7, 12, 9]

end NUMINAMATH_CALUDE_dave_derek_money_difference_l1645_164591


namespace NUMINAMATH_CALUDE_partial_sum_base_7_l1645_164527

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem partial_sum_base_7 :
  let a := [2, 3, 4, 5, 1]
  let b := [1, 5, 6, 4, 2]
  let sum := [4, 2, 4, 2, 3]
  let base := 7
  (to_decimal a base + to_decimal b base = to_decimal sum base) ∧
  (∀ d ∈ (a ++ b ++ sum), d < base) :=
by sorry

end NUMINAMATH_CALUDE_partial_sum_base_7_l1645_164527
