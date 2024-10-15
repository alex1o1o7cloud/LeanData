import Mathlib

namespace NUMINAMATH_CALUDE_polygon_with_16_diagonals_has_7_sides_l2174_217400

/-- The number of sides in a regular polygon with 16 diagonals -/
def num_sides_of_polygon_with_16_diagonals : ℕ := 7

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_with_16_diagonals_has_7_sides :
  num_diagonals num_sides_of_polygon_with_16_diagonals = 16 :=
by sorry

end NUMINAMATH_CALUDE_polygon_with_16_diagonals_has_7_sides_l2174_217400


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l2174_217462

/-- Given two rectangles with equal area, where one rectangle has dimensions 5 inches by W inches,
    and the other has dimensions 8 inches by 15 inches, prove that W equals 24 inches. -/
theorem equal_area_rectangles_width (W : ℝ) : 
  (5 * W = 8 * 15) → W = 24 := by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l2174_217462


namespace NUMINAMATH_CALUDE_billy_sam_money_difference_l2174_217412

theorem billy_sam_money_difference (sam_money : ℕ) (total_money : ℕ) (billy_money : ℕ) :
  sam_money = 75 →
  total_money = 200 →
  billy_money = total_money - sam_money →
  billy_money < 2 * sam_money →
  2 * sam_money - billy_money = 25 := by
  sorry

end NUMINAMATH_CALUDE_billy_sam_money_difference_l2174_217412


namespace NUMINAMATH_CALUDE_intersection_A_B_l2174_217443

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_A_B : A ∩ B = {2, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2174_217443


namespace NUMINAMATH_CALUDE_min_value_reciprocal_product_l2174_217418

theorem min_value_reciprocal_product (a b : ℝ) 
  (h1 : a + a * b + 2 * b = 30) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x + x * y + 2 * y = 30 → 1 / (a * b) ≤ 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_product_l2174_217418


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2174_217490

theorem imaginary_part_of_z (z : ℂ) (h : (z - Complex.I) / (z - 2) = Complex.I) :
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2174_217490


namespace NUMINAMATH_CALUDE_linda_rings_sold_l2174_217437

/-- Proves that Linda sold 8 rings given the conditions of the problem -/
theorem linda_rings_sold :
  let necklaces_sold : ℕ := 4
  let total_sales : ℕ := 80
  let necklace_price : ℕ := 12
  let ring_price : ℕ := 4
  let rings_sold : ℕ := (total_sales - necklaces_sold * necklace_price) / ring_price
  rings_sold = 8 := by
  sorry

end NUMINAMATH_CALUDE_linda_rings_sold_l2174_217437


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2174_217435

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 6*p^2 + 8*p - 3 = 0 →
  q^3 - 6*q^2 + 8*q - 3 = 0 →
  r^3 - 6*r^2 + 8*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 6/5 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2174_217435


namespace NUMINAMATH_CALUDE_jia_zi_second_occurrence_l2174_217401

/-- The number of Heavenly Stems -/
def heavenly_stems : ℕ := 10

/-- The number of Earthly Branches -/
def earthly_branches : ℕ := 12

/-- The column number when Jia and Zi are in the same column for the second time -/
def second_occurrence : ℕ := 61

/-- Proves that the column number when Jia and Zi are in the same column for the second time is 61 -/
theorem jia_zi_second_occurrence :
  second_occurrence = Nat.lcm heavenly_stems earthly_branches + 1 := by
  sorry

end NUMINAMATH_CALUDE_jia_zi_second_occurrence_l2174_217401


namespace NUMINAMATH_CALUDE_initial_number_proof_l2174_217491

theorem initial_number_proof (x : ℝ) : 
  ((5 * x - 20) / 2 - 100 = 4) → x = 45.6 := by
sorry

end NUMINAMATH_CALUDE_initial_number_proof_l2174_217491


namespace NUMINAMATH_CALUDE_part_one_part_two_l2174_217473

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 2

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x > 1} = {x : ℝ | x < 1/2 ∨ x > 1} := by sorry

-- Part II
theorem part_two : 
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ 0) → 
  1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2174_217473


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l2174_217493

theorem abs_sum_inequality (x : ℝ) : |x - 2| + |x + 3| < 7 ↔ -6 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l2174_217493


namespace NUMINAMATH_CALUDE_total_friends_count_l2174_217486

-- Define the number of friends who can pay Rs. 60 each
def standard_payers : ℕ := 10

-- Define the amount each standard payer would pay
def standard_payment : ℕ := 60

-- Define the extra amount paid by one friend
def extra_payment : ℕ := 50

-- Define the total amount paid by the friend who paid extra
def total_extra_payer_amount : ℕ := 115

-- Theorem to prove
theorem total_friends_count : 
  ∃ (n : ℕ), 
    n = standard_payers + 1 ∧ 
    n * (total_extra_payer_amount - extra_payment) = 
      standard_payers * standard_payment + extra_payment :=
by
  sorry

#check total_friends_count

end NUMINAMATH_CALUDE_total_friends_count_l2174_217486


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2174_217414

theorem inequality_solution_set : 
  {x : ℕ | 1 + x ≥ 2 * x - 1} = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2174_217414


namespace NUMINAMATH_CALUDE_distance_polar_point_to_circle_center_distance_specific_point_to_specific_circle_l2174_217425

/-- The distance between a point in polar coordinates and the center of a circle defined by a polar equation --/
theorem distance_polar_point_to_circle_center 
  (r : ℝ) (θ : ℝ) (circle_eq : ℝ → ℝ → Prop) : Prop :=
  let p_rect := (r * Real.cos θ, r * Real.sin θ)
  let circle_center := (1, 0)
  Real.sqrt ((p_rect.1 - circle_center.1)^2 + (p_rect.2 - circle_center.2)^2) = Real.sqrt 3

/-- The main theorem to be proved --/
theorem distance_specific_point_to_specific_circle : 
  distance_polar_point_to_circle_center 2 (Real.pi / 3) (fun ρ θ ↦ ρ = 2 * Real.cos θ) :=
sorry

end NUMINAMATH_CALUDE_distance_polar_point_to_circle_center_distance_specific_point_to_specific_circle_l2174_217425


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2174_217461

theorem min_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2*m + 2*n = 1) :
  1/m + 1/n ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2174_217461


namespace NUMINAMATH_CALUDE_area_of_closed_figure_l2174_217498

/-- The area of the closed figure bounded by y = 1/2, y = 2, y = 1/x, and the y-axis is 2ln(2) -/
theorem area_of_closed_figure : 
  let lower_bound : ℝ := 1/2
  let upper_bound : ℝ := 2
  let curve (x : ℝ) : ℝ := 1/x
  ∫ y in lower_bound..upper_bound, (1/y) = 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_closed_figure_l2174_217498


namespace NUMINAMATH_CALUDE_unique_right_triangle_with_2021_leg_l2174_217440

theorem unique_right_triangle_with_2021_leg : 
  ∃! (a b c : ℕ+), (a = 2021 ∨ b = 2021) ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_right_triangle_with_2021_leg_l2174_217440


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2174_217483

theorem max_value_quadratic (q : ℝ) : -3 * q^2 + 18 * q + 5 ≤ 32 ∧ ∃ q₀ : ℝ, -3 * q₀^2 + 18 * q₀ + 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2174_217483


namespace NUMINAMATH_CALUDE_probability_red_then_blue_l2174_217455

def red_marbles : ℕ := 4
def white_marbles : ℕ := 5
def blue_marbles : ℕ := 3

def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

theorem probability_red_then_blue : 
  (red_marbles : ℚ) / total_marbles * blue_marbles / (total_marbles - 1) = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_then_blue_l2174_217455


namespace NUMINAMATH_CALUDE_broken_stick_triangle_area_l2174_217495

/-- Given three sticks of length 24, if one is broken to form a right triangle with the others,
    the area of the resulting triangle is 216. -/
theorem broken_stick_triangle_area : 
  ∀ a : ℝ, 0 < a → a < 24 →
  (a^2 + 24^2 = (48 - a)^2) →
  (1/2 * a * 24 = 216) :=
by sorry

end NUMINAMATH_CALUDE_broken_stick_triangle_area_l2174_217495


namespace NUMINAMATH_CALUDE_problem_solution_l2174_217460

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 591 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2174_217460


namespace NUMINAMATH_CALUDE_triangle_reconstruction_l2174_217484

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle) : Point := sorry

/-- Represents the foot of altitude from C to AB -/
def altitudeFootC (t : Triangle) : Point := sorry

/-- Represents the C-excenter of a triangle -/
def excenterC (t : Triangle) : Point := sorry

/-- Theorem: Given the incenter, foot of altitude from C, and C-excenter, 
    a unique triangle can be reconstructed -/
theorem triangle_reconstruction 
  (I : Point) (H : Point) (I_c : Point) : 
  ∃! (t : Triangle), 
    incenter t = I ∧ 
    altitudeFootC t = H ∧ 
    excenterC t = I_c := by sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_l2174_217484


namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l2174_217497

theorem stratified_sampling_female_count
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = male_students + female_students)
  (h2 : total_students = 49)
  (h3 : male_students = 28)
  (h4 : female_students = 21)
  (h5 : sample_size = 14) :
  (sample_size : ℚ) / total_students * female_students = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l2174_217497


namespace NUMINAMATH_CALUDE_lake_width_correct_l2174_217432

/-- The width of the lake in miles -/
def lake_width : ℝ := 60

/-- The speed of the faster boat in miles per hour -/
def fast_boat_speed : ℝ := 30

/-- The speed of the slower boat in miles per hour -/
def slow_boat_speed : ℝ := 12

/-- The time difference in hours between the arrivals of the two boats -/
def time_difference : ℝ := 3

/-- Theorem stating that the lake width is correct given the boat speeds and time difference -/
theorem lake_width_correct :
  lake_width / slow_boat_speed = lake_width / fast_boat_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_lake_width_correct_l2174_217432


namespace NUMINAMATH_CALUDE_binary_1011_is_11_l2174_217475

def binary_to_decimal (b : List Bool) : ℕ :=
  List.foldl (λ acc d => 2 * acc + if d then 1 else 0) 0 b

theorem binary_1011_is_11 :
  binary_to_decimal [true, false, true, true] = 11 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011_is_11_l2174_217475


namespace NUMINAMATH_CALUDE_store_payback_time_l2174_217487

/-- Calculates the time required to pay back an initial investment given monthly revenue and expenses -/
def payback_time (initial_cost : ℕ) (monthly_revenue : ℕ) (monthly_expenses : ℕ) : ℕ :=
  let monthly_profit := monthly_revenue - monthly_expenses
  initial_cost / monthly_profit

theorem store_payback_time :
  payback_time 25000 4000 1500 = 10 := by
  sorry

end NUMINAMATH_CALUDE_store_payback_time_l2174_217487


namespace NUMINAMATH_CALUDE_factory_equation_correctness_l2174_217413

/-- Represents the factory worker assignment problem -/
def factory_problem (x y : ℕ) : Prop :=
  -- Total number of workers is 95
  x + y = 95 ∧
  -- Production ratio for sets (2 nuts : 1 screw)
  16 * x = 22 * y

/-- The system of linear equations correctly represents the factory problem -/
theorem factory_equation_correctness :
  ∀ x y : ℕ,
  factory_problem x y ↔ 
  (x + y = 95 ∧ 16 * x - 22 * y = 0) :=
by sorry

end NUMINAMATH_CALUDE_factory_equation_correctness_l2174_217413


namespace NUMINAMATH_CALUDE_josh_marbles_l2174_217430

/-- The number of marbles Josh has -/
def total_marbles (blue red yellow : ℕ) : ℕ := blue + red + yellow

/-- The problem statement -/
theorem josh_marbles : 
  ∀ (blue red yellow : ℕ),
  blue = 3 * red →
  red = 14 →
  yellow = 29 →
  total_marbles blue red yellow = 85 := by
sorry

end NUMINAMATH_CALUDE_josh_marbles_l2174_217430


namespace NUMINAMATH_CALUDE_mrs_hilt_pizzas_l2174_217451

/-- The number of slices in each pizza -/
def slices_per_pizza : ℕ := 8

/-- The total number of slices Mrs. Hilt had -/
def total_slices : ℕ := 16

/-- The number of pizzas Mrs. Hilt bought -/
def pizzas_bought : ℕ := total_slices / slices_per_pizza

theorem mrs_hilt_pizzas : pizzas_bought = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pizzas_l2174_217451


namespace NUMINAMATH_CALUDE_weight_of_b_l2174_217404

-- Define the weights as real numbers
variable (a b c : ℝ)

-- Define the conditions
def average_abc : Prop := (a + b + c) / 3 = 30
def average_ab : Prop := (a + b) / 2 = 25
def average_bc : Prop := (b + c) / 2 = 28

-- Theorem statement
theorem weight_of_b (h1 : average_abc a b c) (h2 : average_ab a b) (h3 : average_bc b c) : b = 16 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2174_217404


namespace NUMINAMATH_CALUDE_math_city_intersections_l2174_217448

/-- Represents a city layout with streets and intersections -/
structure CityLayout where
  num_streets : ℕ
  num_nonintersecting_pairs : ℕ

/-- Calculates the number of intersections in a city layout -/
def num_intersections (layout : CityLayout) : ℕ :=
  (layout.num_streets.choose 2) - layout.num_nonintersecting_pairs

/-- Theorem: In a city with 10 streets and 3 non-intersecting pairs, there are 42 intersections -/
theorem math_city_intersections :
  let layout : CityLayout := ⟨10, 3⟩
  num_intersections layout = 42 := by
  sorry

#eval num_intersections ⟨10, 3⟩

end NUMINAMATH_CALUDE_math_city_intersections_l2174_217448


namespace NUMINAMATH_CALUDE_square_cylinder_volume_l2174_217436

/-- A cylinder with a square cross-section and lateral area 4π has volume 2π -/
theorem square_cylinder_volume (h : ℝ) (lateral_area : ℝ) (volume : ℝ) 
  (h_positive : h > 0)
  (lateral_area_eq : lateral_area = 4 * Real.pi)
  (lateral_area_def : lateral_area = h * h * Real.pi)
  (volume_def : volume = h * h * h / 4) : 
  volume = 2 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_square_cylinder_volume_l2174_217436


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2174_217458

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The theorem stating that among the given sets, only (5, 12, 13) forms a right-angled triangle -/
theorem right_triangle_sets : 
  is_right_triangle 5 12 13 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 5 6 ∧
  ¬is_right_triangle 3 4 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2174_217458


namespace NUMINAMATH_CALUDE_natural_number_representation_l2174_217411

theorem natural_number_representation (n : ℕ) : 
  (∃ a b c : ℕ, (a + b + c)^2 = n * a * b * c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ↔ 
  n ∈ ({1, 2, 3, 4, 5, 6, 8, 9} : Set ℕ) := by
sorry

end NUMINAMATH_CALUDE_natural_number_representation_l2174_217411


namespace NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l2174_217423

/-- A regular hendecagon is an 11-sided regular polygon -/
def RegularHendecagon : Type := Unit

/-- The number of vertices in a regular hendecagon -/
def num_vertices : ℕ := 11

/-- The number of diagonals in a regular hendecagon -/
def num_diagonals (h : RegularHendecagon) : ℕ := 44

/-- The number of pairs of diagonals in a regular hendecagon -/
def num_diagonal_pairs (h : RegularHendecagon) : ℕ := Nat.choose (num_diagonals h) 2

/-- The number of intersecting diagonal pairs inside a regular hendecagon -/
def num_intersecting_pairs (h : RegularHendecagon) : ℕ := Nat.choose num_vertices 4

/-- The probability that two randomly chosen diagonals intersect inside the hendecagon -/
def intersection_probability (h : RegularHendecagon) : ℚ :=
  (num_intersecting_pairs h : ℚ) / (num_diagonal_pairs h : ℚ)

theorem hendecagon_diagonal_intersection_probability (h : RegularHendecagon) :
  intersection_probability h = 165 / 473 := by
  sorry

end NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l2174_217423


namespace NUMINAMATH_CALUDE_algae_growth_theorem_l2174_217454

/-- The time (in hours) for an algae population to grow from 200 to 145,800 cells, tripling every 3 hours. -/
def algae_growth_time : ℕ :=
  18

theorem algae_growth_theorem (initial_population : ℕ) (final_population : ℕ) (growth_factor : ℕ) (growth_interval : ℕ) :
  initial_population = 200 →
  final_population = 145800 →
  growth_factor = 3 →
  growth_interval = 3 →
  (growth_factor ^ (algae_growth_time / growth_interval)) * initial_population = final_population :=
by
  sorry

#check algae_growth_theorem

end NUMINAMATH_CALUDE_algae_growth_theorem_l2174_217454


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l2174_217431

/-- A circle tangent to the x-axis, y-axis, and hypotenuse of a 45°-45°-90° triangle --/
structure TangentCircle where
  /-- The radius of the circle --/
  radius : ℝ
  /-- The circle is tangent to the x-axis --/
  tangent_x : True
  /-- The circle is tangent to the y-axis --/
  tangent_y : True
  /-- The circle is tangent to the hypotenuse of a 45°-45°-90° triangle --/
  tangent_hypotenuse : True
  /-- The length of a leg of the 45°-45°-90° triangle is 2 --/
  triangle_leg : ℝ := 2

/-- The radius of the TangentCircle is equal to 2 + √2 --/
theorem tangent_circle_radius (c : TangentCircle) : c.radius = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l2174_217431


namespace NUMINAMATH_CALUDE_two_tangent_lines_l2174_217427

/-- The number of tangent lines to a circle passing through a point -/
def num_tangent_lines (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : ℕ :=
  sorry

/-- The theorem stating that there are exactly two tangent lines from (2,3) to x^2 + y^2 = 4 -/
theorem two_tangent_lines : num_tangent_lines (0, 0) 2 (2, 3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_tangent_lines_l2174_217427


namespace NUMINAMATH_CALUDE_negative_square_to_fourth_power_l2174_217429

theorem negative_square_to_fourth_power (a : ℝ) : (-a^2)^4 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_to_fourth_power_l2174_217429


namespace NUMINAMATH_CALUDE_special_circle_distances_l2174_217477

/-- A circle with specific properties and a point on its circumference -/
structure SpecialCircle where
  r : ℕ
  u : ℕ
  v : ℕ
  p : ℕ
  q : ℕ
  m : ℕ
  n : ℕ
  h_r_odd : Odd r
  h_circle_eq : u^2 + v^2 = r^2
  h_u_prime_power : u = p^m
  h_v_prime_power : v = q^n
  h_p_prime : Nat.Prime p
  h_q_prime : Nat.Prime q
  h_u_gt_v : u > v

/-- The theorem to be proved -/
theorem special_circle_distances (c : SpecialCircle) :
  let A : ℝ × ℝ := (c.r, 0)
  let B : ℝ × ℝ := (-c.r, 0)
  let C : ℝ × ℝ := (0, -c.r)
  let D : ℝ × ℝ := (0, c.r)
  let P : ℝ × ℝ := (c.u, c.v)
  let M : ℝ × ℝ := (c.u, 0)
  let N : ℝ × ℝ := (0, c.v)
  |A.1 - M.1| = 1 ∧
  |B.1 - M.1| = 9 ∧
  |C.2 - N.2| = 8 ∧
  |D.2 - N.2| = 2 :=
by sorry

end NUMINAMATH_CALUDE_special_circle_distances_l2174_217477


namespace NUMINAMATH_CALUDE_negative_cube_squared_l2174_217476

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l2174_217476


namespace NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_probability_nine_heads_in_twelve_flips_proof_l2174_217416

/-- The probability of getting exactly 9 heads in 12 flips of a fair coin -/
theorem probability_nine_heads_in_twelve_flips : ℚ :=
  55 / 1024

/-- Proof that the probability of getting exactly 9 heads in 12 flips of a fair coin is 55/1024 -/
theorem probability_nine_heads_in_twelve_flips_proof :
  probability_nine_heads_in_twelve_flips = 55 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_nine_heads_in_twelve_flips_probability_nine_heads_in_twelve_flips_proof_l2174_217416


namespace NUMINAMATH_CALUDE_thousand_factorial_zeroes_l2174_217485

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

/-- n! is the product of integers from 1 to n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem thousand_factorial_zeroes :
  trailingZeroes 1000 = 249 :=
sorry

end NUMINAMATH_CALUDE_thousand_factorial_zeroes_l2174_217485


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2174_217442

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_sequence a q n * q

theorem geometric_sequence_sum (a q : ℝ) :
  let seq := geometric_sequence a q
  (seq 0 + seq 1 = 2) →
  (seq 4 + seq 5 = 4) →
  (seq 8 + seq 9 = 8) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2174_217442


namespace NUMINAMATH_CALUDE_school_problem_l2174_217459

/-- Represents a school with a specific number of classes and students. -/
structure School where
  num_classes : Nat
  largest_class : Nat
  difference : Nat
  total_students : Nat

/-- Calculates the total number of students in the school. -/
def calculate_total (s : School) : Nat :=
  let series := List.range s.num_classes
  series.foldr (fun i acc => acc + s.largest_class - i * s.difference) 0

/-- Theorem stating the properties of the school in the problem. -/
theorem school_problem :
  ∃ (s : School),
    s.num_classes = 5 ∧
    s.largest_class = 32 ∧
    s.difference = 2 ∧
    s.total_students = 140 ∧
    calculate_total s = s.total_students :=
  sorry

end NUMINAMATH_CALUDE_school_problem_l2174_217459


namespace NUMINAMATH_CALUDE_max_product_l2174_217410

def Digits : Finset Nat := {3, 5, 8, 9, 1}

def valid_two_digit (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ Digits ∧ (n % 10) ∈ Digits ∧ (n / 10) ≠ (n % 10)

def valid_three_digit (n : Nat) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧ (n / 100) ∈ Digits ∧ ((n / 10) % 10) ∈ Digits ∧ (n % 10) ∈ Digits ∧
  (n / 100) ≠ ((n / 10) % 10) ∧ (n / 100) ≠ (n % 10) ∧ ((n / 10) % 10) ≠ (n % 10)

def valid_pair (a b : Nat) : Prop :=
  valid_two_digit a ∧ valid_three_digit b ∧
  (∀ d : Nat, d ∈ Digits → (d = (a / 10) ∨ d = (a % 10) ∨ d = (b / 100) ∨ d = ((b / 10) % 10) ∨ d = (b % 10)))

theorem max_product :
  ∀ a b : Nat, valid_pair a b → a * b ≤ 91 * 853 :=
sorry

end NUMINAMATH_CALUDE_max_product_l2174_217410


namespace NUMINAMATH_CALUDE_triangle_AOB_properties_l2174_217479

def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

def OA : ℝ × ℝ := A
def OB : ℝ × ℝ := B
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem triangle_AOB_properties :
  let magnitude_AB := Real.sqrt ((AB.1)^2 + (AB.2)^2)
  let dot_product_AB_OA := AB.1 * OA.1 + AB.2 * OA.2
  let cos_angle_OA_OB := (OA.1 * OB.1 + OA.2 * OB.2) / 
    (Real.sqrt (OA.1^2 + OA.2^2) * Real.sqrt (OB.1^2 + OB.2^2))
  (magnitude_AB = Real.sqrt 5) ∧
  (dot_product_AB_OA = 0) ∧
  (cos_angle_OA_OB = Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_AOB_properties_l2174_217479


namespace NUMINAMATH_CALUDE_students_not_excelling_l2174_217467

theorem students_not_excelling (total : ℕ) (basketball : ℕ) (soccer : ℕ) (both : ℕ) : 
  total = 40 → basketball = 12 → soccer = 18 → both = 6 → 
  total - (basketball + soccer - both) = 16 := by
  sorry

end NUMINAMATH_CALUDE_students_not_excelling_l2174_217467


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l2174_217447

/-- A quadratic function with integer coefficients -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  eval : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The theorem statement -/
theorem quadratic_coefficient (f : QuadraticFunction) 
  (h1 : f.eval 1 = 4)
  (h2 : f.eval (-2) = 3)
  (h3 : f.eval (-1) = 2)
  (h4 : ∀ x : ℝ, f.eval x ≤ f.eval (-1)) :
  f.a = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l2174_217447


namespace NUMINAMATH_CALUDE_cubic_function_b_value_l2174_217444

/-- A cubic function f(x) = ax³ + bx² + cx + d with specific properties -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_function_b_value (a b c d : ℝ) :
  (cubic_function a b c d (-1) = 0) →
  (cubic_function a b c d 1 = 0) →
  (cubic_function a b c d 0 = 2) →
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_b_value_l2174_217444


namespace NUMINAMATH_CALUDE_orchestra_members_count_l2174_217405

theorem orchestra_members_count : ∃! n : ℕ, 
  150 < n ∧ n < 250 ∧ 
  n % 8 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_orchestra_members_count_l2174_217405


namespace NUMINAMATH_CALUDE_complement_of_M_l2174_217419

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}

theorem complement_of_M : 
  (U \ M) = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2174_217419


namespace NUMINAMATH_CALUDE_nates_matches_l2174_217496

/-- The number of matches Nate started with -/
def initial_matches : ℕ := 70

/-- The number of matches Nate dropped in the creek -/
def dropped_matches : ℕ := 10

/-- The number of matches eaten by the dog -/
def eaten_matches : ℕ := 2 * dropped_matches

/-- The number of matches Nate has left -/
def remaining_matches : ℕ := 40

theorem nates_matches :
  initial_matches = remaining_matches + dropped_matches + eaten_matches :=
by sorry

end NUMINAMATH_CALUDE_nates_matches_l2174_217496


namespace NUMINAMATH_CALUDE_platform_length_calculation_l2174_217434

/-- Calculates the length of a platform given train length and crossing times -/
theorem platform_length_calculation (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 33 →
  pole_time = 18 →
  ∃ (platform_length : ℝ),
    platform_length = platform_time * (train_length / pole_time) - train_length ∧
    platform_length = 250 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l2174_217434


namespace NUMINAMATH_CALUDE_cube_preserves_order_l2174_217465

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l2174_217465


namespace NUMINAMATH_CALUDE_barycentric_centroid_relation_l2174_217439

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable {A B C X M : V}
variable (α β γ : ℝ)

/-- Given a triangle ABC and a point X with barycentric coordinates (α : β : γ),
    where α + β + γ = 1, and M is the centroid of triangle ABC,
    prove that 3 * vector(XM) = (α - β) * vector(AB) + (β - γ) * vector(BC) + (γ - α) * vector(CA) -/
theorem barycentric_centroid_relation
  (h1 : X = α • A + β • B + γ • C)
  (h2 : α + β + γ = 1)
  (h3 : M = (1/3 : ℝ) • (A + B + C)) :
  3 • (X - M) = (α - β) • (A - B) + (β - γ) • (B - C) + (γ - α) • (C - A) := by
  sorry

end NUMINAMATH_CALUDE_barycentric_centroid_relation_l2174_217439


namespace NUMINAMATH_CALUDE_dividend_calculation_l2174_217421

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h_remainder : remainder = 8)
  (h_quotient : quotient = 43)
  (h_divisor : divisor = 23) :
  divisor * quotient + remainder = 997 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2174_217421


namespace NUMINAMATH_CALUDE_marble_sculpture_weight_l2174_217464

theorem marble_sculpture_weight (original_weight : ℝ) : 
  original_weight > 0 →
  (0.75 * (0.80 * (0.70 * original_weight))) = 105 →
  original_weight = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_sculpture_weight_l2174_217464


namespace NUMINAMATH_CALUDE_smallest_n_for_non_prime_2n_plus_1_l2174_217489

theorem smallest_n_for_non_prime_2n_plus_1 :
  ∃ n : ℕ+, (∀ k < n, Nat.Prime (2 * k + 1)) ∧ ¬Nat.Prime (2 * n + 1) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_non_prime_2n_plus_1_l2174_217489


namespace NUMINAMATH_CALUDE_system_solution_l2174_217456

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 6) (eq2 : 2*x + y = 21) : x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2174_217456


namespace NUMINAMATH_CALUDE_flowers_to_grandma_vs_mom_l2174_217468

theorem flowers_to_grandma_vs_mom (total : ℕ) (to_mom : ℕ) (in_vase : ℕ) :
  total = 52 →
  to_mom = 15 →
  in_vase = 16 →
  total - to_mom - in_vase - to_mom = 6 := by
  sorry

end NUMINAMATH_CALUDE_flowers_to_grandma_vs_mom_l2174_217468


namespace NUMINAMATH_CALUDE_beka_miles_l2174_217409

/-- The number of miles Jackson flew -/
def jackson_miles : ℕ := 563

/-- The additional miles Beka flew compared to Jackson -/
def additional_miles : ℕ := 310

/-- Theorem: Given the conditions, Beka flew 873 miles -/
theorem beka_miles : jackson_miles + additional_miles = 873 := by
  sorry

end NUMINAMATH_CALUDE_beka_miles_l2174_217409


namespace NUMINAMATH_CALUDE_smallest_variable_l2174_217426

theorem smallest_variable (p q r s : ℝ) 
  (h : p + 3 = q - 1 ∧ p + 3 = r + 5 ∧ p + 3 = s - 2) : 
  r ≤ p ∧ r ≤ q ∧ r ≤ s := by
  sorry

end NUMINAMATH_CALUDE_smallest_variable_l2174_217426


namespace NUMINAMATH_CALUDE_yellow_sweets_count_l2174_217449

theorem yellow_sweets_count (green_sweets blue_sweets total_sweets : ℕ) 
  (h1 : green_sweets = 212)
  (h2 : blue_sweets = 310)
  (h3 : total_sweets = 1024) : 
  total_sweets - (green_sweets + blue_sweets) = 502 := by
  sorry

end NUMINAMATH_CALUDE_yellow_sweets_count_l2174_217449


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2174_217406

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 84 ∧ y = x + 12 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2174_217406


namespace NUMINAMATH_CALUDE_c_investment_value_l2174_217420

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that given the conditions of the partnership,
    c's investment is 50,000. -/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 30000)
  (h2 : p.b_investment = 45000)
  (h3 : p.total_profit = 90000)
  (h4 : p.c_profit = 36000)
  (h5 : p.c_investment * p.total_profit = p.c_profit * (p.a_investment + p.b_investment + p.c_investment)) :
  p.c_investment = 50000 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_value_l2174_217420


namespace NUMINAMATH_CALUDE_triangle_sides_simplification_l2174_217422

theorem triangle_sides_simplification (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  |a + b - c| - |a - c - b| = 2*a - 2*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_simplification_l2174_217422


namespace NUMINAMATH_CALUDE_adam_total_earnings_l2174_217457

/-- Calculates Adam's earnings given task rates, completion numbers, and exchange rates -/
def adam_earnings (dollar_per_lawn : ℝ) (euro_per_car : ℝ) (peso_per_dog : ℝ)
                  (lawns_total : ℕ) (cars_total : ℕ) (dogs_total : ℕ)
                  (lawns_forgot : ℕ) (cars_forgot : ℕ) (dogs_forgot : ℕ)
                  (euro_to_dollar : ℝ) (peso_to_dollar : ℝ) : ℝ :=
  let lawns_done := lawns_total - lawns_forgot
  let cars_done := cars_total - cars_forgot
  let dogs_done := dogs_total - dogs_forgot
  
  let lawn_earnings := dollar_per_lawn * lawns_done
  let car_earnings := euro_per_car * cars_done * euro_to_dollar
  let dog_earnings := peso_per_dog * dogs_done * peso_to_dollar
  
  lawn_earnings + car_earnings + dog_earnings

/-- Theorem stating Adam's earnings based on given conditions -/
theorem adam_total_earnings :
  adam_earnings 9 10 50 12 6 4 8 2 1 1.1 0.05 = 87.5 := by
  sorry

#eval adam_earnings 9 10 50 12 6 4 8 2 1 1.1 0.05

end NUMINAMATH_CALUDE_adam_total_earnings_l2174_217457


namespace NUMINAMATH_CALUDE_stickers_in_red_folder_l2174_217417

/-- The number of stickers on each sheet in the red folder -/
def red_stickers : ℕ := 3

/-- The number of sheets in each folder -/
def sheets_per_folder : ℕ := 10

/-- The number of stickers on each sheet in the green folder -/
def green_stickers : ℕ := 2

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers : ℕ := 1

/-- The total number of stickers used -/
def total_stickers : ℕ := 60

theorem stickers_in_red_folder :
  red_stickers * sheets_per_folder +
  green_stickers * sheets_per_folder +
  blue_stickers * sheets_per_folder = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_stickers_in_red_folder_l2174_217417


namespace NUMINAMATH_CALUDE_min_pairs_for_flashlight_l2174_217452

/-- Represents the minimum number of pairs to test to guarantee finding a working pair of batteries -/
def min_pairs_to_test (total_batteries : ℕ) (working_batteries : ℕ) : ℕ :=
  total_batteries / 2 - working_batteries / 2 + 1

/-- Theorem stating the minimum number of pairs to test for the given problem -/
theorem min_pairs_for_flashlight :
  min_pairs_to_test 8 4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_pairs_for_flashlight_l2174_217452


namespace NUMINAMATH_CALUDE_stick_swap_theorem_l2174_217415

/-- Represents a set of three sticks --/
structure StickSet where
  stick1 : Real
  stick2 : Real
  stick3 : Real
  sum_is_one : stick1 + stick2 + stick3 = 1
  all_positive : stick1 > 0 ∧ stick2 > 0 ∧ stick3 > 0

/-- Checks if a triangle can be formed from a set of sticks --/
def can_form_triangle (s : StickSet) : Prop :=
  s.stick1 + s.stick2 > s.stick3 ∧
  s.stick1 + s.stick3 > s.stick2 ∧
  s.stick2 + s.stick3 > s.stick1

theorem stick_swap_theorem (vintik_initial shpuntik_initial vintik_final shpuntik_final : StickSet) :
  can_form_triangle vintik_initial →
  can_form_triangle shpuntik_initial →
  ¬can_form_triangle vintik_final →
  (∃ (x y : Real), 
    vintik_final.stick1 = vintik_initial.stick1 ∧
    vintik_final.stick2 = vintik_initial.stick2 ∧
    vintik_final.stick3 = y ∧
    shpuntik_final.stick1 = shpuntik_initial.stick1 ∧
    shpuntik_final.stick2 = shpuntik_initial.stick2 ∧
    shpuntik_final.stick3 = x ∧
    x + y = vintik_initial.stick3 + shpuntik_initial.stick3) →
  can_form_triangle shpuntik_final := by
  sorry

end NUMINAMATH_CALUDE_stick_swap_theorem_l2174_217415


namespace NUMINAMATH_CALUDE_intersection_bounds_l2174_217441

theorem intersection_bounds (m : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x < 8}
  let B : Set ℝ := {x | 2*m - 1 < x ∧ x < m + 3}
  let U : Set ℝ := Set.univ
  ∃ (a b : ℝ), A ∩ B = {x | a < x ∧ x < b} ∧ b - a = 3 → m = -2 ∨ m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_bounds_l2174_217441


namespace NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l2174_217471

def sumOfDigits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

def satisfiesCondition (n : ℕ) : Prop :=
  isTwoDigit n ∧ Nat.Prime (n - 7 * sumOfDigits n)

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | satisfiesCondition n} = {10, 31, 52, 73, 94} := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l2174_217471


namespace NUMINAMATH_CALUDE_two_diggers_two_hours_l2174_217438

/-- The rate at which diggers dig pits -/
def digging_rate (diggers : ℚ) (pits : ℚ) (hours : ℚ) : ℚ :=
  pits / (diggers * hours)

/-- The number of pits dug given a rate, number of diggers, and hours -/
def pits_dug (rate : ℚ) (diggers : ℚ) (hours : ℚ) : ℚ :=
  rate * diggers * hours

theorem two_diggers_two_hours 
  (h : digging_rate (3/2) (3/2) (3/2) = digging_rate 2 x 2) : x = 8/3 := by
  sorry

#check two_diggers_two_hours

end NUMINAMATH_CALUDE_two_diggers_two_hours_l2174_217438


namespace NUMINAMATH_CALUDE_graph_is_two_lines_l2174_217450

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 25 * y^2 - 20 * x + 100 = 0

/-- Definition of a line in slope-intercept form -/
def is_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

/-- The graph represents two lines -/
theorem graph_is_two_lines :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, equation x y ↔ (is_line m₁ b₁ x y ∨ is_line m₂ b₂ x y)) ∧
    m₁ ≠ m₂ :=
sorry

end NUMINAMATH_CALUDE_graph_is_two_lines_l2174_217450


namespace NUMINAMATH_CALUDE_age_difference_robert_elizabeth_l2174_217478

theorem age_difference_robert_elizabeth : 
  ∀ (robert_age patrick_age elizabeth_age : ℕ),
  robert_age = 28 →
  patrick_age = robert_age / 2 →
  elizabeth_age = patrick_age - 4 →
  robert_age - elizabeth_age = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_robert_elizabeth_l2174_217478


namespace NUMINAMATH_CALUDE_probability_three_students_l2174_217494

/-- The probability of having students participate on both Saturday and Sunday -/
def probability_both_days (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (2^n - 2) / 2^n

theorem probability_three_students :
  probability_both_days 3 = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_students_l2174_217494


namespace NUMINAMATH_CALUDE_largest_less_than_point_seven_l2174_217428

def numbers : Set ℚ := {8/10, 1/2, 9/10, 1/3}

theorem largest_less_than_point_seven :
  (∃ (x : ℚ), x ∈ numbers ∧ x < 7/10 ∧ ∀ (y : ℚ), y ∈ numbers ∧ y < 7/10 → y ≤ x) ∧
  (∀ (z : ℚ), z ∈ numbers ∧ z < 7/10 → z ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_largest_less_than_point_seven_l2174_217428


namespace NUMINAMATH_CALUDE_smallest_student_count_l2174_217470

/-- Represents the number of students in each grade --/
structure GradeCount where
  ninth : ℕ
  tenth : ℕ
  eleventh : ℕ

/-- Checks if the given grade counts satisfy the required ratios --/
def satisfies_ratios (gc : GradeCount) : Prop :=
  4 * gc.ninth = 3 * gc.tenth ∧ 6 * gc.tenth = 5 * gc.eleventh

/-- The total number of students across the three grades --/
def total_students (gc : GradeCount) : ℕ :=
  gc.ninth + gc.tenth + gc.eleventh

/-- Theorem stating that 59 is the smallest number of students satisfying the ratios --/
theorem smallest_student_count : 
  ∃ (gc : GradeCount), satisfies_ratios gc ∧ total_students gc = 59 ∧
  ∀ (gc' : GradeCount), satisfies_ratios gc' → total_students gc' ≥ 59 :=
sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2174_217470


namespace NUMINAMATH_CALUDE_solution_k_l2174_217474

theorem solution_k (h : 2 * k - (-4) = 2) : k = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_k_l2174_217474


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l2174_217472

theorem angle_in_first_quadrant (α : Real) 
  (h1 : Real.tan α > 0) 
  (h2 : Real.sin α + Real.cos α > 0) : 
  0 < α ∧ α < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l2174_217472


namespace NUMINAMATH_CALUDE_min_value_quadratic_function_l2174_217481

theorem min_value_quadratic_function :
  ∀ (x y z : ℝ), 
    x^2 + 4*x*y + 3*y^2 + 2*z^2 - 8*x - 4*y + 6*z ≥ -13.5 ∧
    (x^2 + 4*x*y + 3*y^2 + 2*z^2 - 8*x - 4*y + 6*z = -13.5 ↔ x = 1 ∧ y = 3/2 ∧ z = -3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_function_l2174_217481


namespace NUMINAMATH_CALUDE_total_length_QP_PL_l2174_217466

-- Define the triangle XYZ
def X : ℝ × ℝ := (1, 4)
def Y : ℝ × ℝ := (0, 0)
def Z : ℝ × ℝ := (3, 0)

-- Define the altitudes
def XK : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = X.1 ∧ p.2 ≤ X.2}
def YL : Set (ℝ × ℝ) := {p : ℝ × ℝ | (Z.1 - X.1) * (p.1 - X.1) = (Z.2 - X.2) * (p.2 - X.2)}

-- Define the angle bisectors
def ZD : Set (ℝ × ℝ) := {p : ℝ × ℝ | (X.1 - Z.1) * (p.2 - Z.2) = (X.2 - Z.2) * (p.1 - Z.1)}
def XE : Set (ℝ × ℝ) := {p : ℝ × ℝ | (Y.1 - X.1) * (p.2 - X.2) = (Y.2 - X.2) * (p.1 - X.1)}

-- Define Q and P
def Q : ℝ × ℝ := (1, 1)
noncomputable def P : ℝ × ℝ := (0.5, 3)

-- Theorem statement
theorem total_length_QP_PL : 
  let qp_length := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let pl_length := Real.sqrt ((P.1 - (3/4))^2 + (P.2 - 3)^2)
  qp_length + pl_length = 1.5 := by sorry

end NUMINAMATH_CALUDE_total_length_QP_PL_l2174_217466


namespace NUMINAMATH_CALUDE_problem_solution_l2174_217488

-- Define the ⊗ operation
def otimes (a b : ℕ) : ℕ := sorry

-- Define the main property of ⊗
axiom otimes_prop (a b c : ℕ) : otimes a b = c ↔ a^c = b

theorem problem_solution :
  (∀ x, otimes 3 81 = x → x = 4) ∧
  (∀ a b c, otimes 3 5 = a → otimes 3 6 = b → otimes 3 10 = c → a < b ∧ b < c) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2174_217488


namespace NUMINAMATH_CALUDE_circle_polar_rectangular_equivalence_l2174_217408

/-- The polar coordinate equation of a circle is equivalent to its rectangular coordinate equation -/
theorem circle_polar_rectangular_equivalence (x y ρ θ : ℝ) :
  (x^2 + y^2 - 2*x = 0) ↔ (ρ = 2*Real.cos θ ∧ x = ρ*Real.cos θ ∧ y = ρ*Real.sin θ) :=
sorry

end NUMINAMATH_CALUDE_circle_polar_rectangular_equivalence_l2174_217408


namespace NUMINAMATH_CALUDE_max_silver_tokens_l2174_217482

/-- Represents the state of tokens --/
structure TokenState where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth --/
structure ExchangeBooth where
  redIn : ℕ
  blueIn : ℕ
  redOut : ℕ
  blueOut : ℕ
  silverOut : ℕ

/-- Defines if an exchange is possible given a token state and an exchange booth --/
def canExchange (state : TokenState) (booth : ExchangeBooth) : Prop :=
  state.red ≥ booth.redIn ∧ state.blue ≥ booth.blueIn

/-- Defines the result of an exchange --/
def exchangeResult (state : TokenState) (booth : ExchangeBooth) : TokenState :=
  { red := state.red - booth.redIn + booth.redOut,
    blue := state.blue - booth.blueIn + booth.blueOut,
    silver := state.silver + booth.silverOut }

/-- Defines if a state is final (no more exchanges possible) --/
def isFinalState (state : TokenState) (booths : List ExchangeBooth) : Prop :=
  ∀ booth ∈ booths, ¬(canExchange state booth)

/-- The main theorem --/
theorem max_silver_tokens : 
  ∃ (finalState : TokenState),
    let initialState : TokenState := { red := 100, blue := 50, silver := 0 }
    let booth1 : ExchangeBooth := { redIn := 4, blueIn := 0, redOut := 0, blueOut := 3, silverOut := 1 }
    let booth2 : ExchangeBooth := { redIn := 0, blueIn := 2, redOut := 1, blueOut := 0, silverOut := 1 }
    let booths : List ExchangeBooth := [booth1, booth2]
    (isFinalState finalState booths) ∧
    (finalState.silver = 143) ∧
    (∀ (otherFinalState : TokenState),
      (isFinalState otherFinalState booths) →
      (otherFinalState.silver ≤ finalState.silver)) := by
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l2174_217482


namespace NUMINAMATH_CALUDE_simons_blueberry_pies_l2174_217445

/-- Simon's blueberry pie problem -/
theorem simons_blueberry_pies :
  ∀ (own_blueberries nearby_blueberries blueberries_per_pie : ℕ),
    own_blueberries = 100 →
    nearby_blueberries = 200 →
    blueberries_per_pie = 100 →
    (own_blueberries + nearby_blueberries) / blueberries_per_pie = 3 :=
by
  sorry

#check simons_blueberry_pies

end NUMINAMATH_CALUDE_simons_blueberry_pies_l2174_217445


namespace NUMINAMATH_CALUDE_s_equals_2012_l2174_217469

/-- S(n, k) is the number of coefficients in the expansion of (x+1)^n that are not divisible by k -/
def S (n k : ℕ) : ℕ := sorry

/-- Theorem stating that S(2012^2011, 2011) equals 2012 -/
theorem s_equals_2012 : S (2012^2011) 2011 = 2012 := by sorry

end NUMINAMATH_CALUDE_s_equals_2012_l2174_217469


namespace NUMINAMATH_CALUDE_four_different_suits_count_l2174_217499

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents a suit in a deck of cards -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Number of suits in a standard deck -/
def num_suits : Nat := 4

/-- Number of cards in each suit -/
def cards_per_suit : Nat := 13

/-- 
Theorem: The number of ways to choose 4 cards from a standard deck of 52 cards, 
where all four cards must be of different suits and the order doesn't matter, 
is equal to 28561.
-/
theorem four_different_suits_count (d : Deck) : 
  (cards_per_suit ^ num_suits) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_four_different_suits_count_l2174_217499


namespace NUMINAMATH_CALUDE_spring_bud_cup_value_l2174_217424

theorem spring_bud_cup_value : ∃ x : ℕ, x + x = 578 ∧ x = 289 := by sorry

end NUMINAMATH_CALUDE_spring_bud_cup_value_l2174_217424


namespace NUMINAMATH_CALUDE_cube_guessing_game_l2174_217480

/-- The maximum amount Alexei can guarantee himself in the cube guessing game -/
def maxGuaranteedAmount (m : ℕ) (n : ℕ) : ℚ :=
  2^m / (Nat.choose m n)

/-- The problem statement for the cube guessing game -/
theorem cube_guessing_game (n : ℕ) (hn : n ≤ 100) :
  /- Part a: One blue cube -/
  maxGuaranteedAmount 100 1 = 2^100 / 100 ∧
  /- Part b: n blue cubes -/
  maxGuaranteedAmount 100 n = 2^100 / (Nat.choose 100 n) :=
sorry

end NUMINAMATH_CALUDE_cube_guessing_game_l2174_217480


namespace NUMINAMATH_CALUDE_equation_solution_l2174_217402

theorem equation_solution :
  ∀ a b c : ℤ,
  (∀ x : ℝ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔
  ((a = 2 ∧ b = -3 ∧ c = -4) ∨ (a = 8 ∧ b = -6 ∧ c = -7)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2174_217402


namespace NUMINAMATH_CALUDE_max_value_of_f_l2174_217407

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := 2^n - 1

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n-1)

/-- The expression to be maximized -/
def f (n : ℕ) : ℚ :=
  (a n : ℚ) / ((a n * S n : ℕ) + a 6 : ℚ)

theorem max_value_of_f :
  ∀ n : ℕ, n ≥ 1 → f n ≤ 1/15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2174_217407


namespace NUMINAMATH_CALUDE_unique_n_value_l2174_217463

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem unique_n_value (m n : ℕ) 
  (h1 : m > 0)
  (h2 : is_three_digit n)
  (h3 : Nat.lcm m n = 690)
  (h4 : ¬(3 ∣ n))
  (h5 : ¬(2 ∣ m)) :
  n = 230 := by
sorry

end NUMINAMATH_CALUDE_unique_n_value_l2174_217463


namespace NUMINAMATH_CALUDE_roots_polynomial_sum_l2174_217492

theorem roots_polynomial_sum (α β : ℝ) : 
  α^2 - 3*α - 4 = 0 ∧ β^2 - 3*β - 4 = 0 → 3*α^4 + 8*β^3 = 876 := by
  sorry

end NUMINAMATH_CALUDE_roots_polynomial_sum_l2174_217492


namespace NUMINAMATH_CALUDE_floor_sqrt_27_squared_l2174_217433

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_27_squared_l2174_217433


namespace NUMINAMATH_CALUDE_mangoes_in_basket_B_mangoes_in_basket_B_is_30_l2174_217453

theorem mangoes_in_basket_B (total_baskets : ℕ) (avg_fruits : ℕ) 
  (apples_A : ℕ) (peaches_C : ℕ) (pears_D : ℕ) (bananas_E : ℕ) : ℕ :=
  let total_fruits := total_baskets * avg_fruits
  let accounted_fruits := apples_A + peaches_C + pears_D + bananas_E
  total_fruits - accounted_fruits

#check mangoes_in_basket_B 5 25 15 20 25 35 = 30

theorem mangoes_in_basket_B_is_30 :
  mangoes_in_basket_B 5 25 15 20 25 35 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_in_basket_B_mangoes_in_basket_B_is_30_l2174_217453


namespace NUMINAMATH_CALUDE_completePassage_correct_l2174_217446

/-- Represents an incomplete sentence or passage -/
inductive IncompleteSentence : Type
| Wei : IncompleteSentence
| Zhuangzi : IncompleteSentence
| TaoYuanming : IncompleteSentence
| LiBai : IncompleteSentence
| SuShi : IncompleteSentence
| XinQiji : IncompleteSentence
| Analects : IncompleteSentence
| LiuYuxi : IncompleteSentence

/-- Represents the correct completion for a sentence -/
def Completion : Type := String

/-- A function that returns the correct completion for a given incomplete sentence -/
def completePassage : IncompleteSentence → Completion
| IncompleteSentence.Wei => "垝垣"
| IncompleteSentence.Zhuangzi => "水之积也不厚"
| IncompleteSentence.TaoYuanming => "仰而视之"
| IncompleteSentence.LiBai => "扶疏荫初上"
| IncompleteSentence.SuShi => "举匏樽"
| IncompleteSentence.XinQiji => "骑鲸鱼"
| IncompleteSentence.Analects => "切问而近思"
| IncompleteSentence.LiuYuxi => "莫是银屏"

/-- Theorem stating that the completePassage function returns the correct completion for each incomplete sentence -/
theorem completePassage_correct :
  ∀ (s : IncompleteSentence), 
    (s = IncompleteSentence.Wei → completePassage s = "垝垣") ∧
    (s = IncompleteSentence.Zhuangzi → completePassage s = "水之积也不厚") ∧
    (s = IncompleteSentence.TaoYuanming → completePassage s = "仰而视之") ∧
    (s = IncompleteSentence.LiBai → completePassage s = "扶疏荫初上") ∧
    (s = IncompleteSentence.SuShi → completePassage s = "举匏樽") ∧
    (s = IncompleteSentence.XinQiji → completePassage s = "骑鲸鱼") ∧
    (s = IncompleteSentence.Analects → completePassage s = "切问而近思") ∧
    (s = IncompleteSentence.LiuYuxi → completePassage s = "莫是银屏") :=
by sorry


end NUMINAMATH_CALUDE_completePassage_correct_l2174_217446


namespace NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2174_217403

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents a way to cut the plywood --/
structure CutPattern where
  piece : Rectangle
  num_pieces : ℕ

theorem plywood_cut_perimeter_difference :
  ∀ (cuts : List CutPattern),
    (∀ c ∈ cuts, c.num_pieces = 6 ∧ c.piece.length * c.piece.width * 6 = 54) →
    (∃ c ∈ cuts, perimeter c.piece = 20) →
    (∀ c ∈ cuts, perimeter c.piece ≥ 15) →
    (∃ c ∈ cuts, perimeter c.piece = 15) →
    20 - 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_plywood_cut_perimeter_difference_l2174_217403
