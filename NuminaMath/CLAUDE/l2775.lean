import Mathlib

namespace line_equation_l2775_277536

/-- A line passing through point A(1,4) with zero sum of intercepts on coordinate axes -/
structure LineWithZeroSumIntercepts where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point A(1,4) -/
  passes_through_A : 4 = slope * 1 + y_intercept
  /-- The sum of intercepts on coordinate axes is zero -/
  zero_sum_intercepts : 1 - (4 - y_intercept) / slope + y_intercept = 0

/-- The equation of the line is either 4x-y=0 or x-y+3=0 -/
theorem line_equation (l : LineWithZeroSumIntercepts) :
  (l.slope = 4 ∧ l.y_intercept = 0) ∨ (l.slope = 1 ∧ l.y_intercept = 3) :=
sorry

end line_equation_l2775_277536


namespace tom_july_books_l2775_277589

/-- The number of books Tom read in May -/
def may_books : ℕ := 2

/-- The number of books Tom read in June -/
def june_books : ℕ := 6

/-- The total number of books Tom read -/
def total_books : ℕ := 18

/-- The number of books Tom read in July -/
def july_books : ℕ := total_books - may_books - june_books

theorem tom_july_books : july_books = 10 := by
  sorry

end tom_july_books_l2775_277589


namespace lcm_20_45_75_l2775_277547

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end lcm_20_45_75_l2775_277547


namespace difference_of_squares_25_7_l2775_277569

theorem difference_of_squares_25_7 : (25 + 7)^2 - (25 - 7)^2 = 700 := by
  sorry

end difference_of_squares_25_7_l2775_277569


namespace percentage_of_sum_l2775_277513

theorem percentage_of_sum (x y : ℝ) (P : ℝ) : 
  (0.5 * (x - y) = (P / 100) * (x + y)) → 
  (y = (11.11111111111111 / 100) * x) → 
  P = 40 := by
  sorry

end percentage_of_sum_l2775_277513


namespace last_two_digits_product_l2775_277595

theorem last_two_digits_product (A B : ℕ) : 
  A < 10 → B < 10 → A + B = 12 → (10 * A + B) % 3 = 0 → A * B = 35 :=
by sorry

end last_two_digits_product_l2775_277595


namespace two_rational_solutions_l2775_277565

-- Define the system of equations
def system (x y z : ℚ) : Prop :=
  x + y + z = 0 ∧ x * y * z + z = 0 ∧ x * y + y * z + x * z + y = 0

-- Theorem stating that there are exactly two rational solutions
theorem two_rational_solutions :
  ∃! (s : Finset (ℚ × ℚ × ℚ)), s.card = 2 ∧ ∀ (x y z : ℚ), (x, y, z) ∈ s ↔ system x y z :=
sorry

end two_rational_solutions_l2775_277565


namespace city_a_sand_amount_l2775_277533

/-- The amount of sand received by City A, given the total amount and amounts received by other cities -/
theorem city_a_sand_amount (total sand_b sand_c sand_d : ℝ) (h1 : total = 95) 
  (h2 : sand_b = 26) (h3 : sand_c = 24.5) (h4 : sand_d = 28) : 
  total - (sand_b + sand_c + sand_d) = 16.5 := by
  sorry

end city_a_sand_amount_l2775_277533


namespace trapezoid_inner_quadrilateral_area_l2775_277526

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- A trapezoid is a quadrilateral with two parallel sides -/
structure Trapezoid extends Quadrilateral :=
  (parallel : (A.y - B.y) / (A.x - B.x) = (D.y - C.y) / (D.x - C.x))

/-- Calculate the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Check if a point lies on a line segment -/
def onSegment (P Q R : Point) : Prop := sorry

/-- Find the intersection point of two line segments -/
def intersectionPoint (P Q R S : Point) : Point := sorry

/-- Theorem: Area of inner quadrilateral is at most 1/4 of trapezoid area -/
theorem trapezoid_inner_quadrilateral_area 
  (ABCD : Trapezoid) 
  (E : Point) 
  (F : Point) 
  (H : Point) 
  (G : Point)
  (hE : onSegment ABCD.A ABCD.B E)
  (hF : onSegment ABCD.C ABCD.D F)
  (hH : H = intersectionPoint ABCD.C E ABCD.B F)
  (hG : G = intersectionPoint E ABCD.D ABCD.A F) :
  area ⟨E, H, F, G⟩ ≤ (1/4 : ℝ) * area ABCD.toQuadrilateral := by
  sorry

end trapezoid_inner_quadrilateral_area_l2775_277526


namespace system_solution_is_correct_l2775_277599

/-- The solution set of the system of inequalities {2x + 3 ≤ x + 2, (x + 1) / 3 > x - 1} -/
def solution_set : Set ℝ := {x : ℝ | x ≤ -1}

/-- The first inequality of the system -/
def inequality1 (x : ℝ) : Prop := 2 * x + 3 ≤ x + 2

/-- The second inequality of the system -/
def inequality2 (x : ℝ) : Prop := (x + 1) / 3 > x - 1

theorem system_solution_is_correct :
  ∀ x : ℝ, x ∈ solution_set ↔ (inequality1 x ∧ inequality2 x) := by
  sorry

end system_solution_is_correct_l2775_277599


namespace yellow_face_probability_l2775_277504

/-- The probability of rolling a yellow face on a 12-sided die with 4 yellow faces is 1/3 -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) 
  (h1 : total_faces = 12) (h2 : yellow_faces = 4) : 
  (yellow_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end yellow_face_probability_l2775_277504


namespace mabels_daisies_l2775_277528

theorem mabels_daisies (petals_per_daisy : ℕ) (remaining_petals : ℕ) (daisies_given_away : ℕ) : 
  petals_per_daisy = 8 →
  daisies_given_away = 2 →
  remaining_petals = 24 →
  ∃ (initial_daisies : ℕ), 
    initial_daisies * petals_per_daisy = 
      remaining_petals + daisies_given_away * petals_per_daisy ∧
    initial_daisies = 5 :=
by sorry

end mabels_daisies_l2775_277528


namespace pentadecagon_diagonals_l2775_277548

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A pentadecagon is a 15-sided polygon -/
def pentadecagon_sides : ℕ := 15

/-- Theorem: The number of diagonals in a convex pentadecagon is 90 -/
theorem pentadecagon_diagonals : 
  num_diagonals pentadecagon_sides = 90 := by sorry

end pentadecagon_diagonals_l2775_277548


namespace num_plane_line_pairs_is_48_l2775_277520

/-- A rectangular box -/
structure RectangularBox where
  -- We don't need to define the specifics of the box for this problem

/-- A line determined by two vertices of the box -/
structure BoxLine where
  box : RectangularBox
  -- We don't need to specify how the line is determined

/-- A plane containing four vertices of the box -/
structure BoxPlane where
  box : RectangularBox
  -- We don't need to specify how the plane is determined

/-- A plane-line pair in the box -/
structure PlaneLine where
  box : RectangularBox
  line : BoxLine
  plane : BoxPlane
  is_parallel : Bool -- Indicates if the line and plane are parallel

/-- The number of plane-line pairs in a rectangular box -/
def num_plane_line_pairs (box : RectangularBox) : Nat :=
  -- The actual implementation is not needed for the statement
  sorry

/-- Theorem stating that the number of plane-line pairs in a rectangular box is 48 -/
theorem num_plane_line_pairs_is_48 (box : RectangularBox) :
  num_plane_line_pairs box = 48 := by
  sorry

end num_plane_line_pairs_is_48_l2775_277520


namespace right_triangle_altitude_reciprocal_squares_l2775_277562

/-- In a right triangle with sides a and b, hypotenuse c, and altitude x drawn on the hypotenuse,
    the following equation holds: 1/x² = 1/a² + 1/b² -/
theorem right_triangle_altitude_reciprocal_squares 
  (a b c x : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_altitude : a * b = c * x) : 
  1 / x^2 = 1 / a^2 + 1 / b^2 := by
sorry

end right_triangle_altitude_reciprocal_squares_l2775_277562


namespace triangle_side_length_l2775_277576

theorem triangle_side_length (X Y Z : ℝ) : 
  -- Triangle XYZ with right angle at X
  X^2 + Y^2 = Z^2 →
  -- YZ = 20
  Z = 20 →
  -- tan Z = 3 cos Y
  (Real.tan Z) = 3 * (Real.cos Y) →
  -- XY = (40√2)/3
  Y = (40 * Real.sqrt 2) / 3 := by
sorry

end triangle_side_length_l2775_277576


namespace desired_line_equation_l2775_277559

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def line2 (x y : ℝ) : Prop := x - y + 5 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- The theorem to prove
theorem desired_line_equation : 
  ∃ (x y : ℝ), intersection x y ∧ 
  (∃ (m : ℝ), perpendicular m (1/2) ∧ 
  (∀ (x' y' : ℝ), 2 * x' + y' - 8 = 0 ↔ y' - y = m * (x' - x))) :=
sorry

end desired_line_equation_l2775_277559


namespace total_animals_in_community_l2775_277535

theorem total_animals_in_community (total_families : ℕ) 
  (families_with_two_dogs : ℕ) (families_with_one_dog : ℕ) 
  (h1 : total_families = 50)
  (h2 : families_with_two_dogs = 15)
  (h3 : families_with_one_dog = 20) :
  (families_with_two_dogs * 2 + families_with_one_dog * 1 + 
   (total_families - families_with_two_dogs - families_with_one_dog) * 2) = 80 := by
  sorry

end total_animals_in_community_l2775_277535


namespace log_8641_between_consecutive_integers_l2775_277527

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_8641_between_consecutive_integers : 
  ∃ (c d : ℤ), c + 1 = d ∧ 
  (log10 1000 : ℝ) = 3 ∧
  (log10 10000 : ℝ) = 4 ∧
  1000 < 8641 ∧ 8641 < 10000 ∧
  Monotone log10 ∧
  (c : ℝ) < log10 8641 ∧ log10 8641 < (d : ℝ) ∧
  c + d = 7 := by
  sorry

end log_8641_between_consecutive_integers_l2775_277527


namespace completing_square_equivalence_l2775_277578

theorem completing_square_equivalence :
  ∀ x : ℝ, 3 * x^2 + 4 * x + 1 = 0 ↔ (x + 2/3)^2 = 1/9 :=
by sorry

end completing_square_equivalence_l2775_277578


namespace sqrt_form_existence_l2775_277534

def has_sqrt_form (a : ℕ) : Prop :=
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x^2 + 2*y^2 = a) ∧ (2*x*y = 12)

theorem sqrt_form_existence :
  has_sqrt_form 17 ∧
  has_sqrt_form 22 ∧
  has_sqrt_form 38 ∧
  has_sqrt_form 73 ∧
  ¬(has_sqrt_form 54) :=
sorry

end sqrt_form_existence_l2775_277534


namespace plastic_rings_total_weight_l2775_277585

theorem plastic_rings_total_weight 
  (orange : ℝ) (purple : ℝ) (white : ℝ) (blue : ℝ) (red : ℝ) (green : ℝ)
  (h_orange : orange = 0.08)
  (h_purple : purple = 0.33)
  (h_white : white = 0.42)
  (h_blue : blue = 0.59)
  (h_red : red = 0.24)
  (h_green : green = 0.16) :
  orange + purple + white + blue + red + green = 1.82 := by
sorry

end plastic_rings_total_weight_l2775_277585


namespace amphibian_count_l2775_277543

/-- The total number of amphibians observed in the pond -/
def total_amphibians (frogs salamanders tadpoles newts : ℕ) : ℕ :=
  frogs + salamanders + tadpoles + newts

/-- Theorem stating that the total number of amphibians is 42 -/
theorem amphibian_count : 
  total_amphibians 7 4 30 1 = 42 := by sorry

end amphibian_count_l2775_277543


namespace sum_series_eq_factorial_minus_one_l2775_277577

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_series (n : ℕ) : ℕ := 
  Finset.sum (Finset.range (n + 1)) (λ k => k * factorial k)

theorem sum_series_eq_factorial_minus_one (n : ℕ) : 
  sum_series n = factorial (n + 1) - 1 := by
  sorry

end sum_series_eq_factorial_minus_one_l2775_277577


namespace infinite_primes_solution_l2775_277502

theorem infinite_primes_solution (f : ℕ → ℕ) (k : ℕ) 
  (h_inj : Function.Injective f) 
  (h_bound : ∀ n, f n ≤ n^k) :
  ∃ S : Set ℕ, Set.Infinite S ∧ 
    (∀ q ∈ S, Nat.Prime q ∧ 
      ∃ p, Nat.Prime p ∧ f p ≡ 0 [MOD q]) :=
sorry

end infinite_primes_solution_l2775_277502


namespace quadratic_equal_roots_l2775_277550

theorem quadratic_equal_roots (b : ℝ) :
  (∃ x : ℝ, b * x^2 + 2 * b * x + 4 = 0 ∧
   ∀ y : ℝ, b * y^2 + 2 * b * y + 4 = 0 → y = x) →
  b = 4 :=
sorry

end quadratic_equal_roots_l2775_277550


namespace train_length_l2775_277582

/-- Given a train that crosses a tunnel and a platform, calculate its length -/
theorem train_length (tunnel_length platform_length : ℝ) 
                     (tunnel_time platform_time : ℝ) 
                     (h1 : tunnel_length = 1200)
                     (h2 : platform_length = 180)
                     (h3 : tunnel_time = 45)
                     (h4 : platform_time = 15) : 
  ∃ (train_length : ℝ), 
    (train_length + tunnel_length) / tunnel_time = 
    (train_length + platform_length) / platform_time ∧ 
    train_length = 330 := by
  sorry

end train_length_l2775_277582


namespace min_value_expression_l2775_277574

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ m : ℝ, m = -2031948.5 ∧ 
  ∀ x y : ℝ, x > 0 → y > 0 → 
    (x + 1/y) * (x + 1/y - 2023) + (y + 1/x) * (y + 1/x - 2023) ≥ m ∧
    (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023) = m :=
by sorry

end min_value_expression_l2775_277574


namespace triangle_inequality_l2775_277584

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c ∧ b + c > a ∧ a + c > b →
  ¬(a = 3 ∧ b = 4 ∧ c = 7) := by
  sorry

#check triangle_inequality

end triangle_inequality_l2775_277584


namespace correct_num_cars_l2775_277592

/-- Represents the number of cars taken on the hike -/
def num_cars : ℕ := 3

/-- Represents the number of taxis taken on the hike -/
def num_taxis : ℕ := 6

/-- Represents the number of vans taken on the hike -/
def num_vans : ℕ := 2

/-- Represents the number of people in each car -/
def people_per_car : ℕ := 4

/-- Represents the number of people in each taxi -/
def people_per_taxi : ℕ := 6

/-- Represents the number of people in each van -/
def people_per_van : ℕ := 5

/-- Represents the total number of people on the hike -/
def total_people : ℕ := 58

/-- Theorem stating that the number of cars is correct given the conditions -/
theorem correct_num_cars :
  num_cars * people_per_car +
  num_taxis * people_per_taxi +
  num_vans * people_per_van = total_people :=
by sorry

end correct_num_cars_l2775_277592


namespace james_travel_distance_l2775_277570

/-- Calculates the total distance traveled during a road trip with multiple legs -/
def total_distance (speeds : List ℝ) (durations : List ℝ) : ℝ :=
  (List.zip speeds durations).map (fun (s, t) => s * t) |>.sum

/-- Theorem: James' total travel distance is 995.0 miles -/
theorem james_travel_distance : 
  let speeds : List ℝ := [80.0, 65.0, 75.0, 70.0]
  let durations : List ℝ := [2.0, 4.0, 3.0, 5.0]
  total_distance speeds durations = 995.0 := by
  sorry


end james_travel_distance_l2775_277570


namespace complex_fraction_simplification_l2775_277521

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 7*I
  z₁ / z₂ = (29:ℝ)/53 - (31:ℝ)/53 * I := by
  sorry

end complex_fraction_simplification_l2775_277521


namespace water_level_rise_l2775_277590

/-- Calculates the rise in water level when a cube is fully immersed in a rectangular vessel. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  cube_edge = 17 →
  vessel_length = 20 →
  vessel_width = 15 →
  ∃ (water_rise : ℝ), abs (water_rise - (cube_edge^3 / (vessel_length * vessel_width))) < 0.01 :=
by
  sorry

end water_level_rise_l2775_277590


namespace tank_capacity_l2775_277575

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity (t : Tank) 
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 3 * 60)  -- 3 liters per minute converted to per hour
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 4320 / 7 := by
  sorry

#check tank_capacity

end tank_capacity_l2775_277575


namespace samuel_travel_distance_l2775_277538

/-- The total distance Samuel needs to travel to reach the hotel -/
def total_distance (speed1 speed2 : ℝ) (time1 time2 : ℝ) (remaining_distance : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2 + remaining_distance

/-- Theorem stating that Samuel needs to travel 600 miles to reach the hotel -/
theorem samuel_travel_distance :
  total_distance 50 80 3 4 130 = 600 := by
  sorry

end samuel_travel_distance_l2775_277538


namespace polynomial_division_remainder_l2775_277553

/-- The polynomial P(x) = x + x^3 + x^9 + x^27 + x^81 + x^243 -/
def P (x : ℝ) : ℝ := x + x^3 + x^9 + x^27 + x^81 + x^243

theorem polynomial_division_remainder :
  (∃ Q₁ : ℝ → ℝ, P = fun x ↦ (x - 1) * Q₁ x + 6) ∧
  (∃ Q₂ : ℝ → ℝ, P = fun x ↦ (x^2 - 1) * Q₂ x + 6*x) := by
  sorry

end polynomial_division_remainder_l2775_277553


namespace cube_sum_minus_triple_product_l2775_277551

theorem cube_sum_minus_triple_product (x y z : ℝ) 
  (h1 : x + y + z = 8) 
  (h2 : x*y + y*z + z*x = 20) : 
  x^3 + y^3 + z^3 - 3*x*y*z = 32 := by
sorry

end cube_sum_minus_triple_product_l2775_277551


namespace fraction_meaningful_condition_l2775_277519

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = x / (x + 1)) ↔ x ≠ -1 := by sorry

end fraction_meaningful_condition_l2775_277519


namespace calendar_reuse_2052_l2775_277531

def is_leap_year (year : ℕ) : Prop :=
  year % 4 = 0 ∧ (year % 100 ≠ 0 ∨ year % 400 = 0)

def calendar_repeats (year1 year2 : ℕ) : Prop :=
  is_leap_year year1 ∧ is_leap_year year2 ∧ (year2 - year1) % 28 = 0

theorem calendar_reuse_2052 :
  ∀ y : ℕ, y > 1912 → y < 2052 → ¬(calendar_repeats y 2052) →
  calendar_repeats 1912 2052 ∧ is_leap_year 1912 ∧ is_leap_year 2052 :=
sorry

end calendar_reuse_2052_l2775_277531


namespace tan_seven_pi_fourth_l2775_277588

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end tan_seven_pi_fourth_l2775_277588


namespace third_layer_sugar_l2775_277556

def sugar_for_cake (smallest_layer sugar_second_layer sugar_third_layer : ℕ) : Prop :=
  (sugar_second_layer = 2 * smallest_layer) ∧ 
  (sugar_third_layer = 3 * sugar_second_layer)

theorem third_layer_sugar : ∀ (smallest_layer sugar_second_layer sugar_third_layer : ℕ),
  smallest_layer = 2 →
  sugar_for_cake smallest_layer sugar_second_layer sugar_third_layer →
  sugar_third_layer = 12 := by
  sorry

end third_layer_sugar_l2775_277556


namespace smallest_number_with_conditions_l2775_277598

def alice_number : ℕ := 30

def has_all_prime_factors_except_7 (n : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ alice_number → p ≠ 7 → p ∣ n

theorem smallest_number_with_conditions :
  ∃ (bob_number : ℕ), bob_number > 0 ∧
  has_all_prime_factors_except_7 bob_number ∧
  ∀ m : ℕ, m > 0 → has_all_prime_factors_except_7 m → bob_number ≤ m :=
by sorry

end smallest_number_with_conditions_l2775_277598


namespace infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l2775_277539

/-- A positive integer n is lovely if there exists a positive integer k and 
    positive integers d₁, d₂, ..., dₖ such that n = d₁d₂...dₖ and d_i² | n+d_i for all i ∈ {1, ..., k}. -/
def IsLovely (n : ℕ+) : Prop :=
  ∃ k : ℕ+, ∃ d : Fin k → ℕ+, 
    (n = (Finset.univ.prod (λ i => d i))) ∧ 
    (∀ i : Fin k, (d i)^2 ∣ (n + d i))

/-- There are infinitely many lovely numbers. -/
theorem infinitely_many_lovely_numbers : ∀ N : ℕ, ∃ n : ℕ+, n > N ∧ IsLovely n :=
sorry

/-- There does not exist a lovely number greater than 1 which is a square of an integer. -/
theorem no_lovely_square_greater_than_one : ¬∃ n : ℕ+, n > 1 ∧ ∃ m : ℕ+, n = m^2 ∧ IsLovely n :=
sorry

end infinitely_many_lovely_numbers_no_lovely_square_greater_than_one_l2775_277539


namespace half_angle_quadrant_l2775_277579

def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  ∃ k : Int, k * Real.pi - Real.pi / 4 < α ∧ α < k * Real.pi

theorem half_angle_quadrant (α : Real) :
  is_in_fourth_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by sorry

end half_angle_quadrant_l2775_277579


namespace min_degree_of_g_l2775_277566

variable (x : ℝ)
variable (f g h : ℝ → ℝ)

def is_polynomial (p : ℝ → ℝ) : Prop := sorry

def degree (p : ℝ → ℝ) : ℕ := sorry

theorem min_degree_of_g 
  (hpoly : is_polynomial f ∧ is_polynomial g ∧ is_polynomial h)
  (heq : ∀ x, 2 * f x + 5 * g x = h x)
  (hf : degree f = 7)
  (hh : degree h = 10) :
  degree g ≥ 10 :=
sorry

end min_degree_of_g_l2775_277566


namespace tan_sum_identity_l2775_277507

theorem tan_sum_identity (x : ℝ) : 
  Real.tan (18 * π / 180 - x) * Real.tan (12 * π / 180 + x) + 
  Real.sqrt 3 * (Real.tan (18 * π / 180 - x) + Real.tan (12 * π / 180 + x)) = 1 := by
sorry

end tan_sum_identity_l2775_277507


namespace find_m_value_l2775_277542

/-- Given two functions f and g, prove that m equals 10/7 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 - 3*x + m) →
  (∀ x, g x = x^2 - 3*x + 5*m) →
  3 * f 5 = 2 * g 5 →
  m = 10/7 := by
  sorry

end find_m_value_l2775_277542


namespace age_ratio_l2775_277544

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- Conditions on the ages -/
def validAges (a : Ages) : Prop :=
  a.roy = a.julia + 8 ∧
  a.roy + 2 = 3 * (a.julia + 2) ∧
  (a.roy + 2) * (a.kelly + 2) = 96

/-- The theorem to be proved -/
theorem age_ratio (a : Ages) (h : validAges a) :
  (a.roy - a.julia) / (a.roy - a.kelly) = 2 := by
  sorry

end age_ratio_l2775_277544


namespace remainder_8_pow_1996_mod_5_l2775_277515

theorem remainder_8_pow_1996_mod_5 : 8^1996 % 5 = 1 := by
  sorry

end remainder_8_pow_1996_mod_5_l2775_277515


namespace product_of_fractions_and_powers_of_two_l2775_277546

theorem product_of_fractions_and_powers_of_two : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * 
  (1 / 1024 : ℚ) * 2048 * (1 / 4096 : ℚ) * 8192 = 64 := by
  sorry

end product_of_fractions_and_powers_of_two_l2775_277546


namespace previous_salary_calculation_l2775_277523

-- Define the salary increase rate
def salary_increase_rate : ℝ := 1.05

-- Define the new salary
def new_salary : ℝ := 2100

-- Theorem statement
theorem previous_salary_calculation :
  ∃ (previous_salary : ℝ),
    salary_increase_rate * previous_salary = new_salary ∧
    previous_salary = 2000 := by
  sorry

end previous_salary_calculation_l2775_277523


namespace primitive_triples_theorem_l2775_277555

/-- A triple of positive integers (a, b, c) is primitive if they have no common prime factors -/
def isPrimitive (a b c : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p ∣ a.val ∧ p ∣ b.val ∧ p ∣ c.val)

/-- Each number in the triple divides the sum of the other two -/
def eachDividesSumOfOthers (a b c : ℕ+) : Prop :=
  a ∣ (b + c) ∧ b ∣ (a + c) ∧ c ∣ (a + b)

/-- The main theorem -/
theorem primitive_triples_theorem :
  ∀ a b c : ℕ+, a ≤ b → b ≤ c →
  isPrimitive a b c → eachDividesSumOfOthers a b c →
  (a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 1, 2) ∨ (a, b, c) = (1, 2, 3) :=
sorry

end primitive_triples_theorem_l2775_277555


namespace stating_constant_sum_of_products_l2775_277557

/-- 
Represents the sum of all products of pile sizes during the division process
for n balls.
-/
def f (n : ℕ) : ℕ := sorry

/-- 
Theorem stating that the sum of all products of pile sizes during the division
process is constant for any division strategy.
-/
theorem constant_sum_of_products (n : ℕ) (h : n > 0) :
  ∀ (strategy1 strategy2 : ℕ → ℕ × ℕ),
  (∀ k, k ≤ n → (strategy1 k).1 + (strategy1 k).2 = k) →
  (∀ k, k ≤ n → (strategy2 k).1 + (strategy2 k).2 = k) →
  f n = f n :=
by sorry

/--
Lemma showing that f(n) equals n(n-1)/2 for all positive integers n.
This represents the insight from the solution, but is not directly
assumed from the problem statement.
-/
lemma f_equals_combinations (n : ℕ) (h : n > 0) :
  f n = n * (n - 1) / 2 :=
by sorry

end stating_constant_sum_of_products_l2775_277557


namespace prime_sum_squares_l2775_277587

theorem prime_sum_squares (a b c d : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d →
  a > 3 →
  b > 6 →
  c > 12 →
  a^2 - b^2 + c^2 - d^2 = 1749 →
  a^2 + b^2 + c^2 + d^2 = 1999 := by
  sorry

end prime_sum_squares_l2775_277587


namespace no_positive_integer_solution_l2775_277518

theorem no_positive_integer_solution : 
  ¬ ∃ (n k : ℕ+), (5 + 3 * Real.sqrt 2) ^ n.val = (3 + 5 * Real.sqrt 2) ^ k.val := by
sorry

end no_positive_integer_solution_l2775_277518


namespace range_of_q_l2775_277529

def q (x : ℝ) : ℝ := x^4 - 4*x^2 + 4

theorem range_of_q :
  Set.range q = Set.Icc 0 4 :=
sorry

end range_of_q_l2775_277529


namespace jean_needs_four_more_packs_l2775_277525

/-- Represents the number of cupcakes in a small pack -/
def small_pack : ℕ := 10

/-- Represents the number of cupcakes in a large pack -/
def large_pack : ℕ := 15

/-- Represents the number of large packs Jean initially bought -/
def initial_packs : ℕ := 4

/-- Represents the total number of children in the orphanage -/
def total_children : ℕ := 100

/-- Calculates the number of additional packs of 10 cupcakes Jean needs to buy -/
def additional_packs_needed : ℕ :=
  (total_children - initial_packs * large_pack) / small_pack

theorem jean_needs_four_more_packs :
  additional_packs_needed = 4 :=
sorry

end jean_needs_four_more_packs_l2775_277525


namespace coefficient_x6y4_in_expansion_l2775_277563

theorem coefficient_x6y4_in_expansion : ∀ x y : ℝ,
  (Nat.choose 10 4 : ℝ) = 210 :=
by
  sorry

end coefficient_x6y4_in_expansion_l2775_277563


namespace hyperbola_foci_distance_l2775_277514

/-- The distance between the foci of a hyperbola defined by xy = 4 is 4√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 4 → (x - f₁.1)^2 / (f₂.1 - f₁.1)^2 - 
                               (y - f₁.2)^2 / (f₂.2 - f₁.2)^2 = 1) ∧
    Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) = 4 * Real.sqrt 2 :=
by sorry

end hyperbola_foci_distance_l2775_277514


namespace jerry_showers_l2775_277596

/-- Represents the water usage scenario for Jerry's household --/
structure WaterUsage where
  total_allowance : ℕ
  drinking_cooking : ℕ
  shower_usage : ℕ
  pool_length : ℕ
  pool_width : ℕ
  pool_height : ℕ
  gallon_to_cubic_foot : ℕ

/-- Calculates the number of showers Jerry can take in July --/
def calculate_showers (w : WaterUsage) : ℕ :=
  let pool_volume := w.pool_length * w.pool_width * w.pool_height
  let remaining_water := w.total_allowance - w.drinking_cooking - pool_volume
  remaining_water / w.shower_usage

/-- Theorem stating that Jerry can take 15 showers in July --/
theorem jerry_showers :
  let w : WaterUsage := {
    total_allowance := 1000,
    drinking_cooking := 100,
    shower_usage := 20,
    pool_length := 10,
    pool_width := 10,
    pool_height := 6,
    gallon_to_cubic_foot := 1
  }
  calculate_showers w = 15 := by
  sorry

#eval calculate_showers {
  total_allowance := 1000,
  drinking_cooking := 100,
  shower_usage := 20,
  pool_length := 10,
  pool_width := 10,
  pool_height := 6,
  gallon_to_cubic_foot := 1
}

end jerry_showers_l2775_277596


namespace square_of_negative_integer_is_positive_l2775_277501

theorem square_of_negative_integer_is_positive (P : Int) (h : P < 0) : P^2 > 0 := by
  sorry

end square_of_negative_integer_is_positive_l2775_277501


namespace senior_to_child_ratio_l2775_277593

theorem senior_to_child_ratio 
  (adults : ℕ) 
  (children : ℕ) 
  (seniors : ℕ) 
  (total : ℕ) 
  (h1 : adults = 58)
  (h2 : children = adults - 35)
  (h3 : total = adults + children + seniors)
  (h4 : total = 127) :
  (seniors : ℚ) / children = 2 / 1 :=
by sorry

end senior_to_child_ratio_l2775_277593


namespace minimum_cut_length_l2775_277564

theorem minimum_cut_length (longer_strip shorter_strip : ℝ) 
  (h1 : longer_strip = 23)
  (h2 : shorter_strip = 15) : 
  ∃ x : ℝ, x ≥ 7 ∧ ∀ y : ℝ, y ≥ 0 → longer_strip - y ≥ 2 * (shorter_strip - y) → y ≥ x :=
by sorry

end minimum_cut_length_l2775_277564


namespace ginger_water_usage_l2775_277508

/-- The amount of water Ginger drank and used in her garden --/
def water_used (hours_worked : ℕ) (cups_per_bottle : ℕ) (extra_bottles : ℕ) : ℕ :=
  (hours_worked * cups_per_bottle) + (extra_bottles * cups_per_bottle)

/-- Theorem stating the total amount of water Ginger used --/
theorem ginger_water_usage :
  water_used 8 2 5 = 26 := by
  sorry

end ginger_water_usage_l2775_277508


namespace units_digit_of_k_squared_plus_two_to_k_l2775_277552

def k : ℕ := 2010^2 + 2^2010

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := 2010^2 + 2^2010) : 
  (k^2 + 2^k) % 10 = 7 := by
  sorry

end units_digit_of_k_squared_plus_two_to_k_l2775_277552


namespace prime_pairs_divisibility_l2775_277522

theorem prime_pairs_divisibility : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (6 * p * q ∣ p^3 + q^2 + 38) → 
    ((p = 3 ∧ q = 5) ∨ (p = 3 ∧ q = 13)) := by
  sorry

end prime_pairs_divisibility_l2775_277522


namespace triangle_ratio_l2775_277510

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end triangle_ratio_l2775_277510


namespace andy_initial_minks_l2775_277503

/-- The number of mink skins required to make one coat -/
def skins_per_coat : ℕ := 15

/-- The number of babies each mink has -/
def babies_per_mink : ℕ := 6

/-- The fraction of minks set free by activists -/
def fraction_set_free : ℚ := 1/2

/-- The number of coats Andy can make -/
def coats_made : ℕ := 7

/-- Theorem stating that given the conditions, Andy must have bought 30 minks initially -/
theorem andy_initial_minks :
  ∀ x : ℕ,
  (x + x * babies_per_mink) * (1 - fraction_set_free) = coats_made * skins_per_coat →
  x = 30 := by
  sorry

end andy_initial_minks_l2775_277503


namespace constant_point_on_graph_unique_constant_point_l2775_277505

/-- The quadratic function f(x) that passes through a constant point for any real m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

/-- The constant point that lies on the graph of f(x) for all real m -/
def constant_point : ℝ × ℝ := (2, 13)

/-- Theorem stating that the constant_point lies on the graph of f(x) for all real m -/
theorem constant_point_on_graph :
  ∀ m : ℝ, f m (constant_point.1) = constant_point.2 :=
by sorry

/-- Theorem stating that constant_point is the unique point satisfying the condition -/
theorem unique_constant_point :
  ∀ p : ℝ × ℝ, (∀ m : ℝ, f m p.1 = p.2) → p = constant_point :=
by sorry

end constant_point_on_graph_unique_constant_point_l2775_277505


namespace not_tileable_rectangles_l2775_277524

/-- A domino is a 1x2 rectangle -/
structure Domino :=
  (width : Nat := 2)
  (height : Nat := 1)

/-- A rectangle with given width and height -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Predicate to check if a rectangle is (1,2)-tileable -/
def is_tileable (r : Rectangle) : Prop := sorry

/-- Theorem stating that 1xk and 2xn (where 4 ∤ n) rectangles are not (1,2)-tileable -/
theorem not_tileable_rectangles :
  ∀ (k n : Nat), 
    (¬ is_tileable ⟨1, k⟩) ∧ 
    ((¬ (4 ∣ n)) → ¬ is_tileable ⟨2, n⟩) :=
by sorry

end not_tileable_rectangles_l2775_277524


namespace cosine_A_in_special_triangle_l2775_277549

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem cosine_A_in_special_triangle (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)  -- Sum of angles in a triangle
  (h2 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0)  -- Positive side lengths
  (h3 : Real.sin t.A / 4 = Real.sin t.B / 5)  -- Given ratio
  (h4 : Real.sin t.B / 5 = Real.sin t.C / 6)  -- Given ratio
  : Real.cos t.A = 3/4 := by
  sorry

end cosine_A_in_special_triangle_l2775_277549


namespace toms_age_ratio_l2775_277567

theorem toms_age_ratio (T M : ℚ) : 
  (∃ (children_sum : ℚ), 
    children_sum = T ∧ 
    T - M = 3 * (children_sum - 4 * M)) → 
  T / M = 11 / 2 := by
sorry

end toms_age_ratio_l2775_277567


namespace bottles_left_on_shelf_l2775_277506

theorem bottles_left_on_shelf (initial_bottles : ℕ) (jason_purchase : ℕ) (harry_purchase : ℕ)
  (h1 : initial_bottles = 35)
  (h2 : jason_purchase = 5)
  (h3 : harry_purchase = 6) :
  initial_bottles - (jason_purchase + harry_purchase) = 24 :=
by sorry

end bottles_left_on_shelf_l2775_277506


namespace common_factor_is_gcf_l2775_277594

-- Define the expression
def expression (a b c : ℤ) : ℤ := 8 * a^3 * b^2 - 12 * a * b^3 * c + 2 * a * b

-- Define the common factor
def common_factor (a b : ℤ) : ℤ := 2 * a * b

-- Theorem statement
theorem common_factor_is_gcf (a b c : ℤ) :
  (∃ k₁ k₂ k₃ : ℤ, 
    expression a b c = common_factor a b * (k₁ + k₂ + k₃) ∧
    k₁ = 4 * a^2 * b ∧
    k₂ = -6 * b^2 * c ∧
    k₃ = 1) ∧
  (∀ d : ℤ, d ∣ expression a b c → d ∣ common_factor a b ∨ d = 1 ∨ d = -1) :=
sorry

end common_factor_is_gcf_l2775_277594


namespace relationship_between_x_and_z_l2775_277581

theorem relationship_between_x_and_z (x y z : ℝ) 
  (h1 : x = y * 1.027)  -- x is 2.7% greater than y
  (h2 : y = z * 0.45)   -- y is 55% less than z
  : x = z * (1 - 0.53785) :=  -- x is 53.785% less than z
by sorry

end relationship_between_x_and_z_l2775_277581


namespace madeline_utilities_l2775_277500

/-- Calculates the amount left for utilities given expenses and income --/
def amount_for_utilities (rent groceries medical emergency hourly_wage hours : ℕ) : ℕ :=
  hourly_wage * hours - (rent + groceries + medical + emergency)

/-- Proves that Madeline's amount left for utilities is $70 --/
theorem madeline_utilities : amount_for_utilities 1200 400 200 200 15 138 = 70 := by
  sorry

end madeline_utilities_l2775_277500


namespace line_passes_through_fixed_point_l2775_277568

/-- The line y = mx + (2m + 1), where m ∈ ℝ, always passes through the point (-2, 1). -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), ((-2 : ℝ) : ℝ) * m + (2 * m + 1) = 1 := by sorry

end line_passes_through_fixed_point_l2775_277568


namespace solution_set_max_value_min_value_l2775_277558

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |2*x - 2|

-- Theorem 1: Solution set of f(x) ≥ x-1
theorem solution_set (x : ℝ) : f x ≥ x - 1 ↔ 0 ≤ x ∧ x ≤ 2 :=
sorry

-- Theorem 2: Maximum value of f
theorem max_value : ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 2 :=
sorry

-- Theorem 3: Minimum value of expression
theorem min_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 2) :
  (b^2 / a) + (c^2 / b) + (a^2 / c) ≥ 2 :=
sorry

end solution_set_max_value_min_value_l2775_277558


namespace smallest_inverse_undefined_l2775_277537

theorem smallest_inverse_undefined (a : ℕ) : a = 6 ↔ 
  a > 0 ∧ 
  (∀ k < a, k > 0 → (Nat.gcd k 72 = 1 ∨ Nat.gcd k 90 = 1)) ∧
  Nat.gcd a 72 > 1 ∧ 
  Nat.gcd a 90 > 1 :=
sorry

end smallest_inverse_undefined_l2775_277537


namespace average_difference_l2775_277591

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((x + 80 + 15) / 3) + 5 → x = 10 := by
  sorry

end average_difference_l2775_277591


namespace square_of_negative_product_l2775_277545

theorem square_of_negative_product (b : ℝ) : (-3 * b)^2 = 9 * b^2 := by
  sorry

end square_of_negative_product_l2775_277545


namespace cone_lateral_surface_area_l2775_277580

/-- The lateral surface area of a cone with base radius 6 and slant height 15 is 90π. -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 6 → l = 15 → π * r * l = 90 * π := by
  sorry

end cone_lateral_surface_area_l2775_277580


namespace polynomial_simplification_l2775_277530

theorem polynomial_simplification (x : ℝ) :
  3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 =
  2*x^3 - x^2 + 23*x - 3 := by
  sorry

end polynomial_simplification_l2775_277530


namespace arithmetic_sequence_before_four_l2775_277516

/-- An arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem arithmetic_sequence_before_four :
  ∀ n : ℕ, n ≤ 30 → arithmetic_sequence 92 (-3) n > 4 ∧
  arithmetic_sequence 92 (-3) 31 ≤ 4 := by
  sorry

end arithmetic_sequence_before_four_l2775_277516


namespace identify_coefficients_l2775_277573

-- Define the coefficients of a quadratic equation ax^2 + bx + c = 0
structure QuadraticCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define our specific quadratic equation 2x^2 - x - 5 = 0
def our_quadratic : QuadraticCoefficients := ⟨2, -1, -5⟩

-- Theorem to prove
theorem identify_coefficients :
  our_quadratic.a = 2 ∧ our_quadratic.b = -1 := by
  sorry

end identify_coefficients_l2775_277573


namespace eighth_group_sample_digit_l2775_277560

/-- Represents the systematic sampling method described in the problem -/
def systematicSample (t : ℕ) (k : ℕ) : ℕ :=
  (t + k) % 10

/-- The theorem to prove -/
theorem eighth_group_sample_digit (t : ℕ) (h : t = 7) : systematicSample t 8 = 5 := by
  sorry

end eighth_group_sample_digit_l2775_277560


namespace hyperbola_sum_l2775_277561

/-- The asymptotes of the hyperbola -/
def asymptote1 (x : ℝ) : ℝ := 3 * x + 6
def asymptote2 (x : ℝ) : ℝ := -3 * x + 4

/-- The point through which the hyperbola passes -/
def point : ℝ × ℝ := (1, 10)

/-- The standard form of the hyperbola equation -/
def hyperbola_equation (x y h k a b : ℝ) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem hyperbola_sum (h k a b : ℝ) :
  (∀ x, asymptote1 x = asymptote2 x → x = -1/3 ∧ asymptote1 x = 5) →
  hyperbola_equation point.1 point.2 h k a b →
  a > 0 ∧ b > 0 →
  a + h = 8/3 :=
sorry

end hyperbola_sum_l2775_277561


namespace arithmetic_problem_l2775_277532

theorem arithmetic_problem : (40 + 5 * 12) / (180 / 3^2) + Real.sqrt 49 = 12 := by
  sorry

end arithmetic_problem_l2775_277532


namespace tangent_line_at_origin_l2775_277512

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_at_origin :
  let p : ℝ × ℝ := (0, 0)  -- The origin point
  let m : ℝ := f' p.1      -- The slope of the tangent line at the origin
  ∀ x y : ℝ, y = m * (x - p.1) + f p.1 → y = 0 :=
by sorry


end tangent_line_at_origin_l2775_277512


namespace least_subtraction_for_divisibility_l2775_277586

theorem least_subtraction_for_divisibility : ∃! k : ℕ, 
  k ≤ 16 ∧ (762429836 - k) % 17 = 0 ∧ 
  ∀ m : ℕ, m < k → (762429836 - m) % 17 ≠ 0 :=
by
  sorry

end least_subtraction_for_divisibility_l2775_277586


namespace p_sufficient_not_necessary_for_q_l2775_277540

-- Define p and q as predicates on real numbers x and y
def p (x y : ℝ) : Prop := x + y ≠ 4
def q (x y : ℝ) : Prop := x ≠ 1 ∨ y ≠ 3

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x y : ℝ, p x y → q x y) ∧ 
  (∃ x y : ℝ, q x y ∧ ¬(p x y)) :=
sorry

end p_sufficient_not_necessary_for_q_l2775_277540


namespace sum_of_bases_equals_1188_l2775_277541

/-- Converts a number from base 8 to base 10 -/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 13 to base 10 -/
def base13ToBase10 (n : ℕ) : ℕ := sorry

/-- Represents the digit C in base 13 -/
def C : ℕ := 12

theorem sum_of_bases_equals_1188 :
  base8ToBase10 537 + base13ToBase10 (4 * 13^2 + C * 13 + 5) = 1188 := by sorry

end sum_of_bases_equals_1188_l2775_277541


namespace total_pets_is_45_l2775_277509

/-- The total number of pets given the specified conditions -/
def total_pets : ℕ :=
  let taylor_cats := 4
  let friends_with_double_pets := 3
  let friend1_dogs := 3
  let friend1_birds := 1
  let friend2_dogs := 5
  let friend2_cats := 2
  let friend3_reptiles := 2
  let friend3_birds := 3
  let friend3_cats := 1

  let total_cats := taylor_cats + friends_with_double_pets * (2 * taylor_cats) + friend2_cats + friend3_cats
  let total_dogs := friend1_dogs + friend2_dogs
  let total_birds := friend1_birds + friend3_birds
  let total_reptiles := friend3_reptiles

  total_cats + total_dogs + total_birds + total_reptiles

theorem total_pets_is_45 : total_pets = 45 := by
  sorry

end total_pets_is_45_l2775_277509


namespace negative_sum_l2775_277511

theorem negative_sum (u v w : ℝ) 
  (hu : -1 < u ∧ u < 0) 
  (hv : 0 < v ∧ v < 1) 
  (hw : -2 < w ∧ w < -1) : 
  v + w < 0 := by
  sorry

end negative_sum_l2775_277511


namespace birds_in_tree_l2775_277517

theorem birds_in_tree (initial_birds final_birds : ℕ) : 
  initial_birds = 231 → final_birds = 312 → 
  final_birds - initial_birds = 81 := by sorry

end birds_in_tree_l2775_277517


namespace x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four_l2775_277554

theorem x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four :
  (∀ x : ℝ, x^2 < 4 → x > -2) ∧
  (∃ x : ℝ, x > -2 ∧ x^2 ≥ 4) :=
by sorry

end x_gt_neg_two_necessary_not_sufficient_for_x_sq_lt_four_l2775_277554


namespace tangent_line_x_intercept_l2775_277572

-- Define the function f(x) = x³ - 2x² + 3x + 1
def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x + 3

theorem tangent_line_x_intercept :
  let slope : ℝ := f' 1
  let y_intercept : ℝ := f 1 - slope * 1
  let x_intercept : ℝ := -y_intercept / slope
  x_intercept = -1/2 := by sorry

end tangent_line_x_intercept_l2775_277572


namespace comic_book_collection_comparison_l2775_277583

/-- Represents the number of comic books in a collection after a given number of months -/
def comic_books (initial : ℕ) (monthly_addition : ℕ) (months : ℕ) : ℕ :=
  initial + monthly_addition * months

/-- The month when LaShawn's collection becomes at least three times Kymbrea's -/
def target_month : ℕ := 33

theorem comic_book_collection_comparison :
  ∀ m : ℕ, m < target_month →
    3 * comic_books 50 3 m > comic_books 20 5 m ∧
    3 * comic_books 50 3 target_month ≤ comic_books 20 5 target_month :=
by sorry

end comic_book_collection_comparison_l2775_277583


namespace rectangle_side_sum_l2775_277571

theorem rectangle_side_sum (x y : ℝ) : 
  (2 * x + 4 = 10) → (8 * y - 2 = 10) → x + y = 4.5 := by
  sorry

end rectangle_side_sum_l2775_277571


namespace smallest_prime_factor_in_C_l2775_277597

def C : Set Nat := {64, 66, 67, 68, 71}

def has_smallest_prime_factor (n : Nat) (s : Set Nat) : Prop :=
  n ∈ s ∧ ∀ m ∈ s, ∃ p : Nat, Prime p ∧ p ∣ n ∧ ∀ q : Nat, Prime q → q ∣ m → p ≤ q

theorem smallest_prime_factor_in_C :
  has_smallest_prime_factor 64 C ∧
  has_smallest_prime_factor 66 C ∧
  has_smallest_prime_factor 68 C :=
sorry

end smallest_prime_factor_in_C_l2775_277597
