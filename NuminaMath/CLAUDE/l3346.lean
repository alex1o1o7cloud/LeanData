import Mathlib

namespace NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_255_sin_165_l3346_334606

theorem cos_75_cos_15_minus_sin_255_sin_165 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.sin (255 * π / 180) * Real.sin (165 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_cos_15_minus_sin_255_sin_165_l3346_334606


namespace NUMINAMATH_CALUDE_mean_problem_l3346_334620

theorem mean_problem (x : ℝ) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 → 
  (28 + x + 42 + 78 + 104) / 5 = 62 := by
sorry

end NUMINAMATH_CALUDE_mean_problem_l3346_334620


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3346_334665

def A : Set ℝ := {x | x^2 - 1 = 0}
def B : Set ℝ := {-1, 2, 5}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3346_334665


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3346_334615

/-- Given vectors a and b in ℝ², prove that the cosine of the angle between them is 2√13/13 -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (5, -10)) 
  (h2 : a - b = (3, 6)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3346_334615


namespace NUMINAMATH_CALUDE_polynomial_coefficient_F_l3346_334679

def polynomial (x E F G H : ℤ) : ℤ := x^6 - 14*x^5 + E*x^4 + F*x^3 + G*x^2 + H*x + 36

def roots : List ℤ := [3, 3, 2, 2, 2, 2]

theorem polynomial_coefficient_F (E F G H : ℤ) :
  (∀ r ∈ roots, polynomial r E F G H = 0) →
  (List.sum roots = 14) →
  (∀ r ∈ roots, r > 0) →
  F = -248 := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_F_l3346_334679


namespace NUMINAMATH_CALUDE_parabola_equation_l3346_334634

/-- Represents a parabola in standard form -/
structure Parabola where
  p : ℝ
  axis : Bool  -- True for vertical axis, False for horizontal axis

/-- The hyperbola from the problem statement -/
def hyperbola : Set (ℝ × ℝ) :=
  {(x, y) | 16 * x^2 - 9 * y^2 = 144}

/-- The theorem statement -/
theorem parabola_equation (P : Parabola) :
  P.axis = true ∧  -- Vertical axis of symmetry
  (∀ (x y : ℝ), (x, y) ∈ hyperbola → x^2 = 9 ∧ y^2 = 16) ∧  -- Hyperbola properties
  (0, 0) ∈ hyperbola ∧  -- Vertex at origin
  (-3, 0) ∈ hyperbola ∧  -- Left vertex of hyperbola
  P.p = 6  -- Distance from vertex to directrix is 3
  →
  ∀ (x y : ℝ), y^2 = 2 * P.p * x ↔ y^2 = 12 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l3346_334634


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3346_334686

/-- Represents the number of ways to arrange books of two subjects -/
def arrange_books (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

/-- Theorem: The number of ways to arrange 3 math books and 3 Chinese books
    on a shelf, such that no two books of the same subject are adjacent,
    is equal to 72 -/
theorem book_arrangement_theorem :
  arrange_books 3 = 72 := by
  sorry

#eval arrange_books 3  -- This should output 72

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3346_334686


namespace NUMINAMATH_CALUDE_square_containing_circle_l3346_334675

/-- The area and perimeter of the smallest square containing a circle --/
theorem square_containing_circle (r : ℝ) (h : r = 6) :
  ∃ (area perimeter : ℝ),
    area = (2 * r) ^ 2 ∧
    perimeter = 4 * (2 * r) ∧
    area = 144 ∧
    perimeter = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_containing_circle_l3346_334675


namespace NUMINAMATH_CALUDE_sum_with_rearrangement_not_all_nines_l3346_334628

def digit_sum (n : ℕ) : ℕ := sorry

def is_digit_rearrangement (n m : ℕ) : Prop :=
  digit_sum n = digit_sum m

def repeated_nines (k : ℕ) : ℕ := sorry

theorem sum_with_rearrangement_not_all_nines (n : ℕ) :
  ∀ m : ℕ, is_digit_rearrangement n m → n + m ≠ repeated_nines 125 := by sorry

end NUMINAMATH_CALUDE_sum_with_rearrangement_not_all_nines_l3346_334628


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l3346_334684

theorem ratio_sum_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  x / y = 3 / 4 → x + y + 100 = 500 → y = 1600 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l3346_334684


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3346_334656

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a^2 + b^2 = 2*a*b → a^2 = b^2) ∧
  (∃ a b : ℝ, a^2 = b^2 ∧ a^2 + b^2 ≠ 2*a*b) := by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l3346_334656


namespace NUMINAMATH_CALUDE_angle_DAC_measure_l3346_334622

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the point D
def D (t : Triangle) : Point := sorry

-- Define the angles
def angle_BAC (t : Triangle) : ℝ := sorry
def angle_ABC (t : Triangle) : ℝ := sorry
def angle_DAC (t : Triangle) : ℝ := sorry

-- Define the lengths
def length_DA (t : Triangle) : ℝ := sorry
def length_CB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_DAC_measure (t : Triangle) 
  (h1 : length_DA t = length_CB t)
  (h2 : angle_BAC t = 70)
  (h3 : angle_ABC t = 55) :
  angle_DAC t = 100 := by sorry

end NUMINAMATH_CALUDE_angle_DAC_measure_l3346_334622


namespace NUMINAMATH_CALUDE_function_value_at_negative_two_l3346_334643

theorem function_value_at_negative_two :
  Real.sqrt (4 * (-2) + 9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_two_l3346_334643


namespace NUMINAMATH_CALUDE_trapezium_side_length_l3346_334624

theorem trapezium_side_length 
  (x : ℝ) 
  (h : x > 0) 
  (area : ℝ) 
  (height : ℝ) 
  (other_side : ℝ) 
  (h_area : area = 228) 
  (h_height : height = 12) 
  (h_other_side : other_side = 18) 
  (h_trapezium_area : area = (1/2) * (x + other_side) * height) : 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l3346_334624


namespace NUMINAMATH_CALUDE_dining_bill_calculation_l3346_334636

theorem dining_bill_calculation (total_bill : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) 
  (h1 : total_bill = 198)
  (h2 : tax_rate = 0.1)
  (h3 : tip_rate = 0.2) :
  ∃ (food_price : ℝ), 
    food_price * (1 + tax_rate) * (1 + tip_rate) = total_bill ∧ 
    food_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_dining_bill_calculation_l3346_334636


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l3346_334694

/-- The cost per foot of fencing a square plot -/
theorem fence_cost_per_foot 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h1 : area = 25) 
  (h2 : total_cost = 1160) : 
  total_cost / (4 * Real.sqrt area) = 58 := by
sorry


end NUMINAMATH_CALUDE_fence_cost_per_foot_l3346_334694


namespace NUMINAMATH_CALUDE_probability_of_exact_successes_l3346_334649

def probability_of_success : ℚ := 1/3

def number_of_trials : ℕ := 3

def number_of_successes : ℕ := 2

theorem probability_of_exact_successes :
  (number_of_trials.choose number_of_successes) *
  probability_of_success ^ number_of_successes *
  (1 - probability_of_success) ^ (number_of_trials - number_of_successes) =
  2/9 :=
sorry

end NUMINAMATH_CALUDE_probability_of_exact_successes_l3346_334649


namespace NUMINAMATH_CALUDE_rainfall_problem_l3346_334678

/-- Rainfall problem -/
theorem rainfall_problem (day1 day2 day3 : ℝ) : 
  day1 = 4 →
  day2 = 5 * day1 →
  day3 = day1 + day2 - 6 →
  day3 = 18 := by
sorry

end NUMINAMATH_CALUDE_rainfall_problem_l3346_334678


namespace NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3346_334662

/-- A color representing red, green, or blue -/
inductive Color
| Red
| Green
| Blue

/-- A point with integer coordinates on a plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A coloring of points on a plane -/
def Coloring := Point → Color

/-- The area of a triangle formed by three points -/
def triangleArea (p1 p2 p3 : Point) : ℚ :=
  |p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)| / 2

theorem monochromatic_unit_area_triangle_exists (c : Coloring) :
  ∃ (p1 p2 p3 : Point), c p1 = c p2 ∧ c p2 = c p3 ∧ triangleArea p1 p2 p3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_monochromatic_unit_area_triangle_exists_l3346_334662


namespace NUMINAMATH_CALUDE_milburg_grown_ups_l3346_334668

/-- The number of grown-ups in Milburg -/
def grown_ups (total_population children : ℕ) : ℕ :=
  total_population - children

/-- Proof that the number of grown-ups in Milburg is 5256 -/
theorem milburg_grown_ups :
  grown_ups 8243 2987 = 5256 := by
  sorry

end NUMINAMATH_CALUDE_milburg_grown_ups_l3346_334668


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l3346_334642

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143) (h2 : num_nieces = 11) :
  total_sandwiches / num_nieces = 13 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l3346_334642


namespace NUMINAMATH_CALUDE_problem_statement_l3346_334645

theorem problem_statement (a b c d k m : ℕ) 
  (h1 : d * a = b * c)
  (h2 : a + d = 2^k)
  (h3 : b + c = 2^m) :
  a = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3346_334645


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l3346_334653

theorem shaded_area_fraction (total_squares : ℕ) (shaded_squares : ℕ) :
  total_squares = 6 →
  shaded_squares = 2 →
  (shaded_squares : ℚ) / total_squares = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l3346_334653


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l3346_334672

theorem four_digit_divisible_by_eleven : 
  ∃ (B : ℕ), B < 10 ∧ (4000 + 100 * B + 10 * B + 2) % 11 = 0 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_eleven_l3346_334672


namespace NUMINAMATH_CALUDE_union_equals_A_l3346_334635

def A : Set ℤ := {-1, 0, 1}
def B (a : ℤ) : Set ℤ := {a, a^2}

theorem union_equals_A (a : ℤ) : A ∪ B a = A ↔ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3346_334635


namespace NUMINAMATH_CALUDE_journey_average_speed_l3346_334621

/-- Calculates the average speed given distances in meters and times in minutes -/
def average_speed (distances : List Float) (times : List Float) : Float :=
  let total_distance := (distances.sum / 1000)  -- Convert to km
  let total_time := (times.sum / 60)  -- Convert to hours
  total_distance / total_time

/-- Theorem: The average speed for the given journey is 6 km/h -/
theorem journey_average_speed :
  let distances := [1000, 1500, 2000]
  let times := [10, 15, 20]
  average_speed distances times = 6 := by
sorry

#eval average_speed [1000, 1500, 2000] [10, 15, 20]

end NUMINAMATH_CALUDE_journey_average_speed_l3346_334621


namespace NUMINAMATH_CALUDE_colors_wash_time_l3346_334688

/-- Represents the time in minutes for a laundry load in the washing machine and dryer -/
structure LaundryTime where
  wash : ℕ
  dry : ℕ

/-- The total time for all three loads of laundry -/
def totalTime : ℕ := 344

/-- The laundry time for the whites -/
def whites : LaundryTime := { wash := 72, dry := 50 }

/-- The laundry time for the darks -/
def darks : LaundryTime := { wash := 58, dry := 65 }

/-- The drying time for the colors -/
def colorsDryTime : ℕ := 54

/-- Theorem stating that the washing time for the colors is 45 minutes -/
theorem colors_wash_time :
  ∃ (colorsWashTime : ℕ),
    colorsWashTime = totalTime - (whites.wash + whites.dry + darks.wash + darks.dry + colorsDryTime) ∧
    colorsWashTime = 45 := by
  sorry

end NUMINAMATH_CALUDE_colors_wash_time_l3346_334688


namespace NUMINAMATH_CALUDE_high_school_sampling_l3346_334640

/-- Represents a stratified sampling scenario in a high school -/
structure StratifiedSampling where
  total_students : ℕ
  freshmen : ℕ
  sampled_freshmen : ℕ

/-- Calculates the total number of students to be sampled in a stratified sampling scenario -/
def total_sampled (s : StratifiedSampling) : ℚ :=
  (s.total_students : ℚ) * s.sampled_freshmen / s.freshmen

/-- Theorem stating that for the given high school scenario, the total number of students
    to be sampled is 80 -/
theorem high_school_sampling :
  let s : StratifiedSampling := { total_students := 2400, freshmen := 600, sampled_freshmen := 20 }
  total_sampled s = 80 := by
  sorry


end NUMINAMATH_CALUDE_high_school_sampling_l3346_334640


namespace NUMINAMATH_CALUDE_log_equation_solution_l3346_334619

theorem log_equation_solution : 
  ∃! x : ℝ, (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 4)) ∧ 
  (x = 11 / 2) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3346_334619


namespace NUMINAMATH_CALUDE_gcd_18_30_l3346_334685

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l3346_334685


namespace NUMINAMATH_CALUDE_least_positive_angle_theta_l3346_334696

theorem least_positive_angle_theta (θ : Real) : 
  (θ > 0) → 
  (Real.cos (15 * Real.pi / 180) = Real.sin (45 * Real.pi / 180) + Real.sin θ) → 
  θ = 15 * Real.pi / 180 :=
by
  sorry

end NUMINAMATH_CALUDE_least_positive_angle_theta_l3346_334696


namespace NUMINAMATH_CALUDE_x_difference_l3346_334689

theorem x_difference (x₁ x₂ : ℝ) : 
  ((x₁ + 3)^2 / (2*x₁ + 15) = 3) →
  ((x₂ + 3)^2 / (2*x₂ + 15) = 3) →
  x₁ ≠ x₂ →
  |x₁ - x₂| = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_x_difference_l3346_334689


namespace NUMINAMATH_CALUDE_milk_packet_cost_l3346_334629

theorem milk_packet_cost (total_packets : Nat) (remaining_packets : Nat) 
  (avg_price_all : ℚ) (avg_price_remaining : ℚ) :
  total_packets = 10 →
  remaining_packets = 7 →
  avg_price_all = 25 →
  avg_price_remaining = 20 →
  (total_packets * avg_price_all - remaining_packets * avg_price_remaining : ℚ) = 110 := by
  sorry

end NUMINAMATH_CALUDE_milk_packet_cost_l3346_334629


namespace NUMINAMATH_CALUDE_simplify_expression_l3346_334626

theorem simplify_expression : 1 + 1 / (1 + Real.sqrt 5) + 1 / (1 - Real.sqrt 5) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3346_334626


namespace NUMINAMATH_CALUDE_isosceles_triangle_square_equal_area_l3346_334616

/-- 
Given an isosceles triangle with base s and height h, and a square with side length s,
if their areas are equal, then the height of the triangle is twice the side length of the square.
-/
theorem isosceles_triangle_square_equal_area (s h : ℝ) (s_pos : s > 0) :
  (1 / 2) * s * h = s^2 → h = 2 * s := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_square_equal_area_l3346_334616


namespace NUMINAMATH_CALUDE_builders_hired_for_houses_l3346_334660

/-- The number of builders hired to build houses given specific conditions -/
def builders_hired (days_per_floor : ℕ) (builders_per_floor : ℕ) (pay_per_builder : ℕ) 
  (num_houses : ℕ) (floors_per_house : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_per_floor := days_per_floor * builders_per_floor * pay_per_builder
  let total_floors := num_houses * floors_per_house
  total_cost / cost_per_floor

/-- Theorem stating the number of builders hired under given conditions -/
theorem builders_hired_for_houses :
  builders_hired 30 3 100 5 6 270000 = 30 := by
  sorry

end NUMINAMATH_CALUDE_builders_hired_for_houses_l3346_334660


namespace NUMINAMATH_CALUDE_mauve_red_parts_l3346_334604

/-- Represents the composition of paint mixtures -/
structure PaintMixture where
  red : ℝ
  blue : ℝ

/-- Defines the fuchsia paint mixture -/
def fuchsia : PaintMixture := { red := 5, blue := 3 }

/-- Defines the mauve paint mixture with unknown red parts -/
def mauve (x : ℝ) : PaintMixture := { red := x, blue := 6 }

/-- Theorem stating the number of red parts in mauve paint -/
theorem mauve_red_parts : 
  ∃ (x : ℝ), 
    (16 * (fuchsia.red / (fuchsia.red + fuchsia.blue))) = 
    (x * 20 / (x + (mauve x).blue)) ∧ 
    x = 3 := by sorry

end NUMINAMATH_CALUDE_mauve_red_parts_l3346_334604


namespace NUMINAMATH_CALUDE_average_salary_proof_l3346_334602

theorem average_salary_proof (salary_a salary_b salary_c salary_d salary_e : ℕ)
  (h1 : salary_a = 10000)
  (h2 : salary_b = 5000)
  (h3 : salary_c = 11000)
  (h4 : salary_d = 7000)
  (h5 : salary_e = 9000) :
  (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8600 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_proof_l3346_334602


namespace NUMINAMATH_CALUDE_cup_stacking_l3346_334697

theorem cup_stacking (a₁ a₂ a₄ a₅ : ℕ) (h1 : a₁ = 17) (h2 : a₂ = 21) (h4 : a₄ = 29) (h5 : a₅ = 33)
  (h_pattern : ∃ d : ℕ, a₂ = a₁ + d ∧ a₄ = a₂ + 2*d ∧ a₅ = a₄ + d) :
  ∃ a₃ : ℕ, a₃ = 25 ∧ a₃ = a₂ + (a₂ - a₁) := by
  sorry

end NUMINAMATH_CALUDE_cup_stacking_l3346_334697


namespace NUMINAMATH_CALUDE_floor_product_equals_48_l3346_334670

theorem floor_product_equals_48 (x : ℝ) :
  ⌊x * ⌊x⌋⌋ = 48 ↔ x ∈ Set.Icc 8 (49/6) :=
sorry

end NUMINAMATH_CALUDE_floor_product_equals_48_l3346_334670


namespace NUMINAMATH_CALUDE_count_valid_pairs_l3346_334611

def has_one_solution (b c : ℕ) : Prop :=
  b^2 = 4*c ∨ c^2 = 4*b

def valid_pair (b c : ℕ) : Prop :=
  1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ has_one_solution b c

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_pair p.1 p.2) ∧ Finset.card S = 3 :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l3346_334611


namespace NUMINAMATH_CALUDE_percentage_calculation_l3346_334673

theorem percentage_calculation (x : ℝ) : 
  (0.08 : ℝ) * x = (0.6 : ℝ) * ((0.3 : ℝ) * x) - (0.1 : ℝ) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3346_334673


namespace NUMINAMATH_CALUDE_characterize_solutions_l3346_334692

/-- Given a system of equations with real parameters a, b, c and variables x, y, z,
    this theorem characterizes all possible solutions. -/
theorem characterize_solutions (a b c x y z : ℝ) :
  x^2 * y^2 + x^2 * z^2 = a * x * y * z ∧
  y^2 * z^2 + y^2 * x^2 = b * x * y * z ∧
  z^2 * x^2 + z^2 * y^2 = c * x * y * z →
  (∃ t : ℝ, (x = t ∧ y = 0 ∧ z = 0) ∨ (x = 0 ∧ y = t ∧ z = 0) ∨ (x = 0 ∧ y = 0 ∧ z = t)) ∨
  (∃ s : ℝ, s = (a + b + c) / 2 ∧
    ((x^2 = (s - b) * (s - c) ∧ y^2 = (s - a) * (s - c) ∧ z^2 = (s - a) * (s - b)) ∧
     (0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∨
     (0 < -a ∧ 0 < -b ∧ 0 < -c ∧ -a + -b > -c ∧ -a + -c > -b ∧ -b + -c > -a))) :=
by sorry

end NUMINAMATH_CALUDE_characterize_solutions_l3346_334692


namespace NUMINAMATH_CALUDE_conic_section_type_l3346_334637

/-- The equation √((x-2)² + y²) + √((x+2)² + y²) = 12 represents an ellipse -/
theorem conic_section_type : ∃ (a b : ℝ) (h : 0 < a ∧ 0 < b),
  {(x, y) : ℝ × ℝ | Real.sqrt ((x - 2)^2 + y^2) + Real.sqrt ((x + 2)^2 + y^2) = 12} =
  {(x, y) : ℝ × ℝ | (x^2 / a^2) + (y^2 / b^2) = 1} :=
by sorry

end NUMINAMATH_CALUDE_conic_section_type_l3346_334637


namespace NUMINAMATH_CALUDE_sams_ribbon_length_l3346_334641

/-- The total length of a ribbon cut into equal pieces -/
def total_ribbon_length (piece_length : ℕ) (num_pieces : ℕ) : ℕ :=
  piece_length * num_pieces

/-- Theorem: The total length of Sam's ribbon is 3723 cm -/
theorem sams_ribbon_length : 
  total_ribbon_length 73 51 = 3723 := by
  sorry

end NUMINAMATH_CALUDE_sams_ribbon_length_l3346_334641


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l3346_334652

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt ((x + 3)^2) = |x + 3| := by sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l3346_334652


namespace NUMINAMATH_CALUDE_A_symmetry_l3346_334609

/-- A(n, k, r) is the number of integer tuples (x₁, x₂, ..., xₖ) satisfying:
    - x₁ ≥ x₂ ≥ ... ≥ xₖ ≥ 0
    - x₁ + x₂ + ... + xₖ = n
    - x₁ - xₖ ≤ r -/
def A (n k r : ℕ+) : ℕ :=
  sorry

/-- For all positive integers m, s, t, A(m, s, t) = A(m, t, s) -/
theorem A_symmetry (m s t : ℕ+) : A m s t = A m t s := by
  sorry

end NUMINAMATH_CALUDE_A_symmetry_l3346_334609


namespace NUMINAMATH_CALUDE_oranges_per_box_l3346_334666

/-- Given 56 oranges and 8 boxes, prove that the number of oranges per box is 7 -/
theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 56) (h2 : num_boxes = 8) :
  total_oranges / num_boxes = 7 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l3346_334666


namespace NUMINAMATH_CALUDE_kate_change_l3346_334683

def candy1_cost : ℚ := 54/100
def candy2_cost : ℚ := 35/100
def candy3_cost : ℚ := 68/100
def amount_paid : ℚ := 5

theorem kate_change : 
  amount_paid - (candy1_cost + candy2_cost + candy3_cost) = 343/100 := by
  sorry

end NUMINAMATH_CALUDE_kate_change_l3346_334683


namespace NUMINAMATH_CALUDE_S_is_specific_set_l3346_334625

/-- A set of complex numbers satisfying certain conditions -/
def S : Set ℂ :=
  {z : ℂ | ∃ (n : ℕ), 2 < n ∧ n < 6 ∧ Complex.abs z = 1}

/-- The condition that 1 is in S -/
axiom one_in_S : (1 : ℂ) ∈ S

/-- The closure property of S -/
axiom S_closure (z₁ z₂ : ℂ) (h₁ : z₁ ∈ S) (h₂ : z₂ ∈ S) :
  z₁ - 2 * z₂ * Complex.cos (Complex.arg (z₁ / z₂)) ∈ S

/-- The theorem to be proved -/
theorem S_is_specific_set : S = {-1, 1, -Complex.I, Complex.I} := by
  sorry

end NUMINAMATH_CALUDE_S_is_specific_set_l3346_334625


namespace NUMINAMATH_CALUDE_prime_remainder_theorem_l3346_334690

theorem prime_remainder_theorem (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (p^3 + 17) % 24 = 0 ∨ (p^3 + 17) % 24 = 16 := by
  sorry

end NUMINAMATH_CALUDE_prime_remainder_theorem_l3346_334690


namespace NUMINAMATH_CALUDE_statement_equivalence_l3346_334698

/-- Represents the property of being happy -/
def happy : Prop := sorry

/-- Represents the property of possessing the food item -/
def possess : Prop := sorry

/-- The statement "Happy people all possess it" -/
def original_statement : Prop := happy → possess

/-- The statement "People who do not possess it are unhappy" -/
def equivalent_statement : Prop := ¬possess → ¬happy

/-- Theorem stating that the original statement is logically equivalent to the equivalent statement -/
theorem statement_equivalence : original_statement ↔ equivalent_statement :=
  sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3346_334698


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3346_334613

theorem absolute_value_inequality (a : ℝ) : 
  (∀ x : ℝ, |2*x - 3| - 2*a > |x + a|) ↔ -3/2 ≤ a ∧ a < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3346_334613


namespace NUMINAMATH_CALUDE_domain_of_shifted_f_l3346_334680

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 1 2

-- Define the property that f is defined only on its domain
axiom f_defined_on_domain : ∀ x, x ∈ domain_f → f x ≠ 0

-- State the theorem
theorem domain_of_shifted_f :
  {x | f (x + 1) ≠ 0} = Set.Icc 0 1 :=
sorry

end NUMINAMATH_CALUDE_domain_of_shifted_f_l3346_334680


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3346_334650

theorem cyclic_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

#check cyclic_inequality

end NUMINAMATH_CALUDE_cyclic_inequality_l3346_334650


namespace NUMINAMATH_CALUDE_munchausen_polygon_exists_l3346_334655

/-- A polygon in a 2D plane --/
structure Polygon :=
  (vertices : Set (ℝ × ℝ))

/-- A point in a 2D plane --/
def Point := ℝ × ℝ

/-- A line in a 2D plane --/
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

/-- Predicate to check if a point is inside a polygon --/
def IsInside (p : Point) (poly : Polygon) : Prop := sorry

/-- Predicate to check if a line passes through a point --/
def PassesThrough (l : Line) (p : Point) : Prop := sorry

/-- Predicate to check if a line divides a polygon into three parts --/
def DividesIntoThree (l : Line) (poly : Polygon) : Prop := sorry

/-- The main theorem --/
theorem munchausen_polygon_exists :
  ∃ (poly : Polygon) (p : Point),
    IsInside p poly ∧
    ∀ (l : Line), PassesThrough l p → DividesIntoThree l poly := by
  sorry

end NUMINAMATH_CALUDE_munchausen_polygon_exists_l3346_334655


namespace NUMINAMATH_CALUDE_calculator_cost_proof_l3346_334695

theorem calculator_cost_proof (basic scientific graphing : ℝ) 
  (h1 : scientific = 2 * basic)
  (h2 : graphing = 3 * scientific)
  (h3 : basic + scientific + graphing = 72) :
  basic = 8 := by
sorry

end NUMINAMATH_CALUDE_calculator_cost_proof_l3346_334695


namespace NUMINAMATH_CALUDE_rectangle_area_l3346_334603

/-- A rectangle divided into three identical squares with a perimeter of 120 cm has an area of 675 square centimeters. -/
theorem rectangle_area (side : ℝ) : 
  (8 * side = 120) →  -- perimeter condition
  (3 * side * side = 675) -- area calculation
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3346_334603


namespace NUMINAMATH_CALUDE_cubic_roots_sum_series_l3346_334687

def cubic_polynomial (x : ℝ) : ℝ := 30 * x^3 - 50 * x^2 + 22 * x - 1

theorem cubic_roots_sum_series : 
  ∃ (a b c : ℝ),
    (∀ x : ℝ, cubic_polynomial x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) ∧ (0 < c ∧ c < 1) →
    (∑' n : ℕ, (a^n + b^n + c^n)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_series_l3346_334687


namespace NUMINAMATH_CALUDE_jenna_reading_schedule_l3346_334663

/-- Represents Jenna's reading schedule for September --/
structure ReadingSchedule where
  total_days : Nat
  total_pages : Nat
  busy_days : Nat
  special_day_pages : Nat

/-- Calculates the number of pages Jenna needs to read per day on regular reading days --/
def pages_per_day (schedule : ReadingSchedule) : Nat :=
  let regular_reading_days := schedule.total_days - schedule.busy_days - 1
  let regular_pages := schedule.total_pages - schedule.special_day_pages
  regular_pages / regular_reading_days

/-- Theorem stating that Jenna needs to read 20 pages per day on regular reading days --/
theorem jenna_reading_schedule :
  let schedule := ReadingSchedule.mk 30 600 4 100
  pages_per_day schedule = 20 := by
  sorry

end NUMINAMATH_CALUDE_jenna_reading_schedule_l3346_334663


namespace NUMINAMATH_CALUDE_shaded_area_circle_in_square_l3346_334674

/-- The area of the shaded region between a circle inscribed in a square,
    where the circle touches the midpoints of the square's sides. -/
theorem shaded_area_circle_in_square (side_length : ℝ) (h : side_length = 12) :
  side_length ^ 2 - π * (side_length / 2) ^ 2 = side_length ^ 2 - π * 36 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_circle_in_square_l3346_334674


namespace NUMINAMATH_CALUDE_f_properties_l3346_334605

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * (a ^ x) - (a ^ (2 * x))

theorem f_properties (a : ℝ) (h_a : a > 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y ∧ y < 1) ∧
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2) 1 ∧ f a x₀ = -7 →
    a = 2 ∧ ∃ x_max : ℝ, x_max ∈ Set.Icc (-2) 1 ∧ f a x_max = 7/16 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f a x ≤ 7/16) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3346_334605


namespace NUMINAMATH_CALUDE_largest_similar_triangle_exists_l3346_334627

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the properties of the triangles
axiom similar_triangles (T1 T2 : Triangle) : Prop
axiom point_on_line (P Q R : Point) : Prop
axiom triangle_area (T : Triangle) : ℝ

-- Define the given triangles
variable (A B : Triangle)

-- Define the constructed triangle
variable (M : Triangle)

-- Define the conditions
variable (h1 : point_on_line (A.1) (M.2.1) (M.2.2))
variable (h2 : point_on_line (A.2.1) (A.1) (A.2.2))
variable (h3 : point_on_line (A.2.2) (A.1) (A.2.1))
variable (h4 : similar_triangles M B)

-- State the theorem
theorem largest_similar_triangle_exists :
  ∃ (M : Triangle), 
    point_on_line (A.1) (M.2.1) (M.2.2) ∧
    point_on_line (A.2.1) (A.1) (A.2.2) ∧
    point_on_line (A.2.2) (A.1) (A.2.1) ∧
    similar_triangles M B ∧
    ∀ (M' : Triangle), 
      (point_on_line (A.1) (M'.2.1) (M'.2.2) ∧
       point_on_line (A.2.1) (A.1) (A.2.2) ∧
       point_on_line (A.2.2) (A.1) (A.2.1) ∧
       similar_triangles M' B) →
      triangle_area M ≥ triangle_area M' :=
sorry

end NUMINAMATH_CALUDE_largest_similar_triangle_exists_l3346_334627


namespace NUMINAMATH_CALUDE_range_of_a_l3346_334658

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3346_334658


namespace NUMINAMATH_CALUDE_least_possible_value_l3346_334644

theorem least_possible_value (x y z : ℤ) : 
  Even x → Odd y → Odd z → x < y → y < z → y - x > 5 → (∀ w, w - x ≥ 9 → w ≥ z) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_value_l3346_334644


namespace NUMINAMATH_CALUDE_percent_of_self_l3346_334648

theorem percent_of_self (y : ℝ) (h1 : y > 0) (h2 : y * (y / 100) = 9) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_self_l3346_334648


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3346_334661

theorem negation_of_proposition (P : ℝ → Prop) :
  (∀ x : ℝ, x^2 + 1 > 1) ↔ ¬(∃ x₀ : ℝ, x₀^2 + 1 ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3346_334661


namespace NUMINAMATH_CALUDE_systematic_sample_sequence_l3346_334614

/-- Represents a systematic sample of students -/
structure SystematicSample where
  total_students : Nat
  sample_size : Nat
  first_number : Nat

/-- Calculates the next numbers in a systematic sample sequence -/
def next_numbers (s : SystematicSample) : List Nat :=
  let step := s.total_students / s.sample_size
  [1, 2, 3, 4].map (fun i => s.first_number + i * step)

theorem systematic_sample_sequence (s : SystematicSample) 
  (h1 : s.total_students = 60)
  (h2 : s.sample_size = 5)
  (h3 : s.first_number = 4) :
  next_numbers s = [16, 28, 40, 52] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_sequence_l3346_334614


namespace NUMINAMATH_CALUDE_largest_odd_between_1_and_7_l3346_334647

theorem largest_odd_between_1_and_7 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 7 ∧ Odd n → n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_between_1_and_7_l3346_334647


namespace NUMINAMATH_CALUDE_B_equals_D_l3346_334639

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

-- Define set D
def D : Set ℝ := {y : ℝ | y ≥ 1}

-- Theorem stating that B and D are equal
theorem B_equals_D : B = D := by sorry

end NUMINAMATH_CALUDE_B_equals_D_l3346_334639


namespace NUMINAMATH_CALUDE_min_cubes_to_remove_l3346_334699

/-- Represents the dimensions of a rectangular block. -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular block given its dimensions. -/
def blockVolume (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the side length of the largest cube that can fit within the block. -/
def largestCubeSide (d : BlockDimensions) : ℕ :=
  min d.length (min d.width d.height)

/-- Calculates the volume of the largest cube that can fit within the block. -/
def largestCubeVolume (d : BlockDimensions) : ℕ :=
  let side := largestCubeSide d
  side * side * side

/-- The main theorem stating the minimum number of cubes to remove. -/
theorem min_cubes_to_remove (d : BlockDimensions) 
    (h1 : d.length = 4) (h2 : d.width = 5) (h3 : d.height = 6) : 
    blockVolume d - largestCubeVolume d = 56 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_to_remove_l3346_334699


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3346_334693

theorem quadratic_factorization (p : ℕ+) :
  (∃ a b : ℤ, ∀ x : ℤ, x^2 - 5*x + p.val = (x - a) * (x - b)) →
  p.val = 4 ∨ p.val = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3346_334693


namespace NUMINAMATH_CALUDE_like_terms_imply_exponents_l3346_334633

/-- Two terms are like terms if they have the same variables with the same exponents -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∃ (c₁ c₂ : ℚ), ∀ (x y : ℕ), term1 x y = c₁ * term2 x y

/-- The first term in our problem -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^m * y^2

/-- The second term in our problem -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := (2/3) * x * y^n

/-- If term1 and term2 are like terms, then m = 1 and n = 2 -/
theorem like_terms_imply_exponents (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m = 1 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_exponents_l3346_334633


namespace NUMINAMATH_CALUDE_inequality_range_l3346_334667

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3*a) ↔ (a ≤ -1 ∨ a ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3346_334667


namespace NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3346_334600

theorem largest_gold_coins_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 15 * k + 3) → 
  n < 150 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 15 * j + 3) → m < 150 → m ≤ n) →
  n = 138 := by
sorry

end NUMINAMATH_CALUDE_largest_gold_coins_distribution_l3346_334600


namespace NUMINAMATH_CALUDE_room_width_calculation_l3346_334601

theorem room_width_calculation (length width total_area : ℝ) 
  (h1 : length = 4)
  (h2 : total_area = 80)
  (h3 : total_area = length * width) :
  width = 20 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l3346_334601


namespace NUMINAMATH_CALUDE_torn_pages_fine_l3346_334671

/-- Calculates the fine for tearing out pages from a book -/
def calculate_fine (start_page end_page : ℕ) (cost_per_sheet : ℕ) : ℕ :=
  let total_pages := end_page - start_page + 1
  let total_sheets := (total_pages + 1) / 2
  total_sheets * cost_per_sheet

/-- The fine for tearing out pages 15 to 30 is 128 yuan -/
theorem torn_pages_fine :
  calculate_fine 15 30 16 = 128 := by
  sorry

end NUMINAMATH_CALUDE_torn_pages_fine_l3346_334671


namespace NUMINAMATH_CALUDE_parabola_proof_l3346_334618

def parabola (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

theorem parabola_proof :
  ∃ (b c : ℝ),
    (parabola 3 b c = 0) ∧
    (parabola 0 b c = -3) ∧
    (∀ x, parabola x b c = x^2 - 2*x - 3) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 4 → parabola x b c ≤ 5) ∧
    (∀ x, -1 ≤ x ∧ x ≤ 4 → parabola x b c ≥ -4) ∧
    (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ parabola x b c = 5) ∧
    (∃ x, -1 ≤ x ∧ x ≤ 4 ∧ parabola x b c = -4) :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_proof_l3346_334618


namespace NUMINAMATH_CALUDE_tan_inequality_solution_set_l3346_334631

open Real

theorem tan_inequality_solution_set (x : ℝ) :
  (3 * tan x + Real.sqrt 3 > 0) ↔
  ∃ k : ℤ, x ∈ Set.Ioo ((-(π / 6) : ℝ) + k * π) ((π / 6 : ℝ) + k * π) :=
by sorry

end NUMINAMATH_CALUDE_tan_inequality_solution_set_l3346_334631


namespace NUMINAMATH_CALUDE_expansion_equality_l3346_334657

theorem expansion_equality (x y : ℝ) : 25 * (3 * x + 7 - 4 * y) = 75 * x + 175 - 100 * y := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l3346_334657


namespace NUMINAMATH_CALUDE_remainder_seven_to_63_mod_8_l3346_334677

theorem remainder_seven_to_63_mod_8 :
  7^63 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_to_63_mod_8_l3346_334677


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3346_334646

-- Problem 1
theorem problem_1 (f : ℝ → ℝ) (x₀ : ℝ) :
  (∀ x, f x = 13 - 8*x + Real.sqrt 2 * x^2) →
  (deriv f x₀ = 4) →
  x₀ = 3 * Real.sqrt 2 := by sorry

-- Problem 2
theorem problem_2 (f : ℝ → ℝ) :
  (∀ x, f x = x^2 + 2*x*(deriv f 0)) →
  ¬∃ y, deriv f 0 = y := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3346_334646


namespace NUMINAMATH_CALUDE_f_composition_value_l3346_334623

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_value : f (f (f (-1))) = Real.pi + 1 := by sorry

end NUMINAMATH_CALUDE_f_composition_value_l3346_334623


namespace NUMINAMATH_CALUDE_probability_at_least_one_two_l3346_334632

def num_sides : ℕ := 8

def total_outcomes : ℕ := num_sides * num_sides

def outcomes_without_two : ℕ := (num_sides - 1) * (num_sides - 1)

def outcomes_with_at_least_one_two : ℕ := total_outcomes - outcomes_without_two

theorem probability_at_least_one_two :
  (outcomes_with_at_least_one_two : ℚ) / total_outcomes = 15 / 64 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_two_l3346_334632


namespace NUMINAMATH_CALUDE_room_area_difference_l3346_334608

-- Define the dimensions of the rooms
def largest_room_width : ℝ := 45
def largest_room_length : ℝ := 30
def smallest_room_width : ℝ := 15
def smallest_room_length : ℝ := 8

-- Define the area calculation function
def area (width : ℝ) (length : ℝ) : ℝ := width * length

-- Theorem statement
theorem room_area_difference :
  area largest_room_width largest_room_length - area smallest_room_width smallest_room_length = 1230 := by
  sorry

end NUMINAMATH_CALUDE_room_area_difference_l3346_334608


namespace NUMINAMATH_CALUDE_square_polygon_area_l3346_334659

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon represented by a list of its vertices -/
def Polygon := List Point

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (p : Polygon) : ℝ := sorry

/-- The specific polygon described in the problem -/
def squarePolygon : Polygon := [
  { x := 0, y := 0 },
  { x := 6, y := 0 },
  { x := 6, y := 6 },
  { x := 0, y := 6 }
]

/-- Theorem stating that the area of the given polygon is 36 square units -/
theorem square_polygon_area :
  polygonArea squarePolygon = 36 := by sorry

end NUMINAMATH_CALUDE_square_polygon_area_l3346_334659


namespace NUMINAMATH_CALUDE_average_student_height_l3346_334654

/-- The average height of all students given specific conditions -/
theorem average_student_height 
  (avg_female_height : ℝ) 
  (avg_male_height : ℝ) 
  (male_to_female_ratio : ℝ) 
  (h1 : avg_female_height = 170) 
  (h2 : avg_male_height = 182) 
  (h3 : male_to_female_ratio = 5) : 
  (male_to_female_ratio * avg_male_height + avg_female_height) / (male_to_female_ratio + 1) = 180 := by
  sorry

end NUMINAMATH_CALUDE_average_student_height_l3346_334654


namespace NUMINAMATH_CALUDE_roses_age_l3346_334682

theorem roses_age (rose_age mother_age : ℕ) : 
  rose_age = mother_age / 3 →
  rose_age + mother_age = 100 →
  rose_age = 25 := by
sorry

end NUMINAMATH_CALUDE_roses_age_l3346_334682


namespace NUMINAMATH_CALUDE_square_of_cube_third_smallest_prime_l3346_334664

def third_smallest_prime : Nat := 5

theorem square_of_cube_third_smallest_prime : 
  (third_smallest_prime ^ 3) ^ 2 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_third_smallest_prime_l3346_334664


namespace NUMINAMATH_CALUDE_intersection_count_l3346_334638

/-- The number of distinct intersection points between a circle and a parabola -/
def numIntersectionPoints (r : ℝ) (a b : ℝ) : ℕ :=
  let circle (x y : ℝ) := x^2 + y^2 = r^2
  let parabola (x y : ℝ) := y = a * x^2 + b
  -- Definition of the function to count intersection points
  sorry

/-- Theorem stating that the number of intersection points is 3 for the given equations -/
theorem intersection_count : numIntersectionPoints 4 1 (-4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_count_l3346_334638


namespace NUMINAMATH_CALUDE_petya_wins_against_sasha_l3346_334630

/-- Represents a player in the knockout tennis tournament -/
inductive Player : Type
| Petya : Player
| Sasha : Player
| Misha : Player

/-- The number of rounds played by each player -/
def rounds_played (p : Player) : ℕ :=
  match p with
  | Player.Petya => 12
  | Player.Sasha => 7
  | Player.Misha => 11

/-- The total number of games played in the tournament -/
def total_games : ℕ := (rounds_played Player.Petya + rounds_played Player.Sasha + rounds_played Player.Misha) / 2

/-- The number of games a player did not play -/
def games_not_played (p : Player) : ℕ := total_games - rounds_played p

/-- Theorem stating that Petya won 4 times against Sasha -/
theorem petya_wins_against_sasha : 
  games_not_played Player.Misha = 4 ∧ 
  (∀ p : Player, games_not_played p + rounds_played p = total_games) ∧
  (rounds_played Player.Sasha = 7 → games_not_played Player.Sasha = 8) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_against_sasha_l3346_334630


namespace NUMINAMATH_CALUDE_condition_relationship_l3346_334651

theorem condition_relationship :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3346_334651


namespace NUMINAMATH_CALUDE_distance_point_to_line_l3346_334610

/-- Given a line in polar form and a point in polar coordinates, 
    calculate the distance from the point to the line. -/
theorem distance_point_to_line 
  (ρ θ : ℝ) -- polar coordinates of the point
  (h_line : ∀ (ρ' θ' : ℝ), 2 * ρ' * Real.sin (θ' - π/4) = Real.sqrt 2) -- line equation
  (h_point : ρ = 2 * Real.sqrt 2 ∧ θ = 7 * π/4) -- point coordinates
  : let x := ρ * Real.cos θ
    let y := ρ * Real.sin θ
    (y - x - 1) / Real.sqrt 2 = 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l3346_334610


namespace NUMINAMATH_CALUDE_triangle_equilateral_l3346_334612

theorem triangle_equilateral (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_equation : 2 * (a * b^2 + b * c^2 + c * a^2) = a^2 * b + b^2 * c + c^2 * a + 3 * a * b * c) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_l3346_334612


namespace NUMINAMATH_CALUDE_eliot_account_balance_l3346_334617

/-- Represents the bank account balances of Al and Eliot -/
structure BankAccounts where
  al : ℝ
  eliot : ℝ

/-- The conditions of the problem -/
def satisfiesConditions (accounts : BankAccounts) : Prop :=
  accounts.al > accounts.eliot ∧
  accounts.al - accounts.eliot = (1 / 12) * (accounts.al + accounts.eliot) ∧
  1.10 * accounts.al - 1.15 * accounts.eliot = 22

/-- The theorem stating Eliot's account balance -/
theorem eliot_account_balance :
  ∀ accounts : BankAccounts, satisfiesConditions accounts → accounts.eliot = 146.67 := by
  sorry

end NUMINAMATH_CALUDE_eliot_account_balance_l3346_334617


namespace NUMINAMATH_CALUDE_difference_at_negative_five_l3346_334669

-- Define the functions f and g
def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g (k : ℤ) (x : ℝ) : ℝ := x^3 - k * x - 10

-- State the theorem
theorem difference_at_negative_five (k : ℤ) : f (-5) - g k (-5) = -24 → k = 61 := by
  sorry

end NUMINAMATH_CALUDE_difference_at_negative_five_l3346_334669


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3346_334691

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  x^4 * y^3 * z^2 ≥ 1/9^9 :=
by sorry

theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 1/x + 1/y + 1/z = 9 ∧
  x^4 * y^3 * z^2 < 1/9^9 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l3346_334691


namespace NUMINAMATH_CALUDE_system_solutions_l3346_334681

theorem system_solutions :
  let eq1 := (fun (x y : ℝ) => x + 3*y + 3*x*y = -1)
  let eq2 := (fun (x y : ℝ) => x^2*y + 3*x*y^2 = -4)
  ∃! (s : Set (ℝ × ℝ)), s = {(-3, -1/3), (-1, -1), (-1, 4/3), (4, -1/3)} ∧
    ∀ (p : ℝ × ℝ), p ∈ s ↔ (eq1 p.1 p.2 ∧ eq2 p.1 p.2) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3346_334681


namespace NUMINAMATH_CALUDE_milly_extra_balloons_l3346_334607

theorem milly_extra_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (floretta_balloons : ℕ) : 
  total_packs = 5 →
  balloons_per_pack = 6 →
  floretta_balloons = 8 →
  (total_packs * balloons_per_pack) / 2 - floretta_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_milly_extra_balloons_l3346_334607


namespace NUMINAMATH_CALUDE_quadratic_properties_l3346_334676

/-- A quadratic function y = ax² + bx + c with given points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_neq_0 : a ≠ 0
  h_point_neg1 : a * (-1)^2 + b * (-1) + c = -1
  h_point_0 : c = 3
  h_point_1 : a + b + c = 5
  h_point_3 : 9 * a + 3 * b + c = 3

/-- Theorem stating the properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  f.a * f.c < 0 ∧ f.a * 3^2 + (f.b - 1) * 3 + f.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3346_334676
