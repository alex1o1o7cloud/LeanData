import Mathlib

namespace NUMINAMATH_CALUDE_minimal_circle_and_intersecting_line_l637_63726

-- Define the right-angled triangle
def triangle : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 / 4 + p.2 / 2 ≤ 1}

-- Define the circle equation
def circle_equation (center : ℝ × ℝ) (radius : ℝ) (point : ℝ × ℝ) : Prop :=
  (point.1 - center.1)^2 + (point.2 - center.2)^2 = radius^2

-- Define the line equation
def line_equation (slope : ℝ) (intercept : ℝ) (point : ℝ × ℝ) : Prop :=
  point.2 = slope * point.1 + intercept

theorem minimal_circle_and_intersecting_line :
  ∃ (center : ℝ × ℝ) (radius : ℝ) (intercept : ℝ),
    (∀ p ∈ triangle, circle_equation center radius p) ∧
    (∀ c, ∀ r, (∀ p ∈ triangle, circle_equation c r p) → r ≥ radius) ∧
    center = (2, 1) ∧
    radius^2 = 5 ∧
    (intercept = -1 - Real.sqrt 5 ∨ intercept = -1 + Real.sqrt 5) ∧
    (∃ A B : ℝ × ℝ,
      A ≠ B ∧
      circle_equation center radius A ∧
      circle_equation center radius B ∧
      line_equation 1 intercept A ∧
      line_equation 1 intercept B ∧
      ((A.1 - center.1) * (B.1 - center.1) + (A.2 - center.2) * (B.2 - center.2) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_minimal_circle_and_intersecting_line_l637_63726


namespace NUMINAMATH_CALUDE_course_selection_theorem_l637_63750

/-- The number of available courses. -/
def n : ℕ := 4

/-- The number of courses each person selects. -/
def k : ℕ := 2

/-- The number of ways to select courses with at least one difference. -/
def different_selections : ℕ := 30

/-- Theorem stating the number of ways to select courses with at least one difference. -/
theorem course_selection_theorem :
  (Finset.univ.filter (fun s : Finset (Fin n) => s.card = k)).card *
  (Finset.univ.filter (fun s : Finset (Fin n) => s.card = k)).card -
  (Finset.univ.filter (fun s : Finset (Fin n) => s.card = k)).card = different_selections :=
sorry


end NUMINAMATH_CALUDE_course_selection_theorem_l637_63750


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l637_63740

theorem inequality_and_equality_condition (a b : ℝ) : 
  (a^2 + 4*b^2 + 4*b - 4*a + 5 ≥ 0) ∧ 
  (a^2 + 4*b^2 + 4*b - 4*a + 5 = 0 ↔ a = 2 ∧ b = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l637_63740


namespace NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l637_63717

-- Define a fifth degree polynomial with leading coefficient 1
def FifthDegreePolynomial (a b c d e : ℝ) : ℝ → ℝ := λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

-- Theorem statement
theorem max_intersections_fifth_degree_polynomials 
  (p q : ℝ → ℝ) 
  (hp : ∃ a b c d e, p = FifthDegreePolynomial a b c d e) 
  (hq : ∃ a' b' c' d' e', q = FifthDegreePolynomial a' b' c' d' e') 
  (hpq_diff : p ≠ q) :
  ∃ S : Finset ℝ, (∀ x ∈ S, p x = q x) ∧ S.card ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_fifth_degree_polynomials_l637_63717


namespace NUMINAMATH_CALUDE_commodity_price_increase_l637_63710

/-- The annual price increase of commodity Y -/
def y : ℝ := sorry

/-- The year we're interested in -/
def target_year : ℝ := 1999.18

/-- The reference year -/
def reference_year : ℝ := 2001

/-- The price of commodity X in the reference year -/
def price_x_reference : ℝ := 5.20

/-- The price of commodity Y in the reference year -/
def price_y_reference : ℝ := 7.30

/-- The annual price increase of commodity X -/
def x_increase : ℝ := 0.45

/-- The price difference between X and Y in the target year -/
def price_difference : ℝ := 0.90

/-- The number of years between the target year and the reference year -/
def years_difference : ℝ := reference_year - target_year

theorem commodity_price_increase : 
  abs (y - 0.021) < 0.001 :=
by
  sorry

end NUMINAMATH_CALUDE_commodity_price_increase_l637_63710


namespace NUMINAMATH_CALUDE_intersection_in_sphere_l637_63778

/-- Given three unit cylinders with pairwise perpendicular axes, 
    their intersection is contained in a sphere of radius √(3/2) --/
theorem intersection_in_sphere (a b c d e f : ℝ) :
  ∀ x y z : ℝ, 
  (x - a)^2 + (y - b)^2 ≤ 1 →
  (y - c)^2 + (z - d)^2 ≤ 1 →
  (z - e)^2 + (x - f)^2 ≤ 1 →
  ∃ center_x center_y center_z : ℝ, 
    (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_in_sphere_l637_63778


namespace NUMINAMATH_CALUDE_interior_angle_sum_for_polygon_with_60_degree_exterior_angles_l637_63721

theorem interior_angle_sum_for_polygon_with_60_degree_exterior_angles :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 60 →
    n * exterior_angle = 360 →
    (n - 2) * 180 = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_interior_angle_sum_for_polygon_with_60_degree_exterior_angles_l637_63721


namespace NUMINAMATH_CALUDE_wedge_volume_l637_63759

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (log_diameter : ℝ) (cut_angle : ℝ) : 
  log_diameter = 12 →
  cut_angle = 45 →
  (π * (log_diameter / 2)^2 * log_diameter) / 2 = 216 * π := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_l637_63759


namespace NUMINAMATH_CALUDE_angle_C_range_l637_63743

theorem angle_C_range (A B C : Real) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : AB = 1) (h6 : BC = 2) : 
  0 < C ∧ C ≤ π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_range_l637_63743


namespace NUMINAMATH_CALUDE_bus_seating_solution_l637_63728

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  back_seat_capacity : Nat
  total_capacity : Nat

/-- Calculates the number of people each regular seat can hold -/
def seats_capacity (bus : BusSeating) : Nat :=
  let regular_seats := bus.left_seats + bus.right_seats
  let regular_capacity := bus.total_capacity - bus.back_seat_capacity
  regular_capacity / regular_seats

/-- Theorem stating the solution to the bus seating problem -/
theorem bus_seating_solution :
  let bus := BusSeating.mk 15 12 11 92
  seats_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seating_solution_l637_63728


namespace NUMINAMATH_CALUDE_max_distance_to_line_l637_63774

theorem max_distance_to_line (a b c : ℝ) (h : a - b - c = 0) :
  ∃ (x y : ℝ), a * x + b * y + c = 0 ∧
  ∀ (x' y' : ℝ), a * x' + b * y' + c = 0 →
  (x' ^ 2 + y' ^ 2 : ℝ) ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l637_63774


namespace NUMINAMATH_CALUDE_jungs_youngest_sibling_age_l637_63752

/-- Represents the ages of the people in the problem -/
structure Ages where
  li : ℕ
  zhang : ℕ
  jung : ℕ
  mei : ℕ
  youngest : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.li = 12 ∧
  ages.zhang = 2 * ages.li ∧
  ages.jung = ages.zhang + 2 ∧
  ages.mei = ages.jung / 2 ∧
  ages.zhang + ages.jung + ages.mei + ages.youngest = 66

/-- The theorem stating that Jung's youngest sibling is 3 years old -/
theorem jungs_youngest_sibling_age (ages : Ages) 
  (h : problem_conditions ages) : ages.youngest = 3 := by
  sorry


end NUMINAMATH_CALUDE_jungs_youngest_sibling_age_l637_63752


namespace NUMINAMATH_CALUDE_ratio_A_to_B_in_X_l637_63754

/-- Represents a compound with two elements -/
structure Compound where
  totalWeight : ℝ
  weightB : ℝ

/-- Calculates the ratio of element A to element B in a compound -/
def ratioAtoB (c : Compound) : ℝ × ℝ :=
  let weightA := c.totalWeight - c.weightB
  (weightA, c.weightB)

/-- Theorem: The ratio of A to B in compound X is 1:5 -/
theorem ratio_A_to_B_in_X :
  let compoundX : Compound := { totalWeight := 300, weightB := 250 }
  let (a, b) := ratioAtoB compoundX
  a / b = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_ratio_A_to_B_in_X_l637_63754


namespace NUMINAMATH_CALUDE_proportional_sum_ratio_l637_63741

theorem proportional_sum_ratio (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4 ∧ z / 4 ≠ 0) : 
  (2 * x + 3 * y) / z = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_proportional_sum_ratio_l637_63741


namespace NUMINAMATH_CALUDE_corn_height_after_ten_weeks_l637_63729

/-- Represents the growth of corn plants over 10 weeks -/
def corn_growth : List ℝ := [
  2,       -- Week 1
  4,       -- Week 2
  16,      -- Week 3
  22,      -- Week 4
  8,       -- Week 5
  16,      -- Week 6
  12.33,   -- Week 7
  7.33,    -- Week 8
  24,      -- Week 9
  36       -- Week 10
]

/-- The total height of the corn plants after 10 weeks -/
def total_height : ℝ := corn_growth.sum

/-- Theorem stating that the total height of the corn plants after 10 weeks is 147.66 inches -/
theorem corn_height_after_ten_weeks : total_height = 147.66 := by
  sorry

end NUMINAMATH_CALUDE_corn_height_after_ten_weeks_l637_63729


namespace NUMINAMATH_CALUDE_prob_same_heads_m_plus_n_l637_63737

def fair_coin_prob : ℚ := 1/2
def biased_coin_prob : ℚ := 3/5

def same_heads_prob : ℚ :=
  (1 - fair_coin_prob) * (1 - biased_coin_prob) +
  fair_coin_prob * biased_coin_prob +
  fair_coin_prob * (1 - biased_coin_prob) * biased_coin_prob * (1 - fair_coin_prob)

theorem prob_same_heads :
  same_heads_prob = 19/50 := by sorry

#eval Nat.gcd 19 50  -- To verify that 19 and 50 are relatively prime

def m : ℕ := 19
def n : ℕ := 50

theorem m_plus_n : m + n = 69 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_m_plus_n_l637_63737


namespace NUMINAMATH_CALUDE_square_of_two_plus_i_l637_63787

theorem square_of_two_plus_i : (2 + Complex.I) ^ 2 = 3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_plus_i_l637_63787


namespace NUMINAMATH_CALUDE_dissimilar_terms_expansion_l637_63781

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^12 -/
def dissimilar_terms : ℕ := 455

/-- The number of variables in the expansion -/
def num_variables : ℕ := 4

/-- The power to which the sum is raised -/
def power : ℕ := 12

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^12 is 455 -/
theorem dissimilar_terms_expansion :
  dissimilar_terms = Nat.choose (power + num_variables - 1) (num_variables - 1) := by
  sorry

end NUMINAMATH_CALUDE_dissimilar_terms_expansion_l637_63781


namespace NUMINAMATH_CALUDE_right_triangle_area_l637_63735

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b = 24 →
  c = 24 →
  a^2 + c^2 = (a + b + c)^2 →
  (1/2) * a * c = 216 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l637_63735


namespace NUMINAMATH_CALUDE_latest_departure_time_l637_63709

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem latest_departure_time 
  (flight_time : Time)
  (check_in_time : Nat)
  (drive_time : Nat)
  (park_walk_time : Nat)
  (h1 : flight_time = ⟨20, 0⟩)  -- 8:00 pm
  (h2 : check_in_time = 120)    -- 2 hours
  (h3 : drive_time = 45)        -- 45 minutes
  (h4 : park_walk_time = 15)    -- 15 minutes
  : 
  let latest_departure := Time.mk 17 0  -- 5:00 pm
  timeDifferenceInMinutes flight_time latest_departure = 
    check_in_time + drive_time + park_walk_time :=
by sorry

end NUMINAMATH_CALUDE_latest_departure_time_l637_63709


namespace NUMINAMATH_CALUDE_cyclical_sequence_value_of_3_cyclical_sequence_properties_l637_63762

def cyclical_sequence (n : ℕ) : ℕ :=
  match n % 5 with
  | 1 => 6
  | 2 => 12
  | 3 => 18  -- This is what we want to prove
  | 4 => 24
  | 0 => 30
  | _ => 0   -- This case should never occur

theorem cyclical_sequence_value_of_3 :
  cyclical_sequence 3 = 18 :=
by
  sorry

theorem cyclical_sequence_properties :
  (cyclical_sequence 1 = 6) ∧
  (cyclical_sequence 2 = 12) ∧
  (cyclical_sequence 4 = 24) ∧
  (cyclical_sequence 5 = 30) ∧
  (cyclical_sequence 6 = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_cyclical_sequence_value_of_3_cyclical_sequence_properties_l637_63762


namespace NUMINAMATH_CALUDE_additive_multiplicative_inverse_sum_l637_63767

theorem additive_multiplicative_inverse_sum (a b : ℝ) : 
  (a + a = 0) → (b * b = 1) → (a + b = 1 ∨ a + b = -1) := by sorry

end NUMINAMATH_CALUDE_additive_multiplicative_inverse_sum_l637_63767


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_shift_l637_63796

theorem quadratic_equation_solution_shift 
  (m h k : ℝ) 
  (hm : m ≠ 0) 
  (h1 : m * (2 - h)^2 - k = 0) 
  (h2 : m * (5 - h)^2 - k = 0) :
  m * (1 - h + 1)^2 = k ∧ m * (4 - h + 1)^2 = k := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_shift_l637_63796


namespace NUMINAMATH_CALUDE_fifth_over_eight_fourth_power_l637_63795

theorem fifth_over_eight_fourth_power : (5 / 8 : ℚ) ^ 4 = 625 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_fifth_over_eight_fourth_power_l637_63795


namespace NUMINAMATH_CALUDE_shirt_sale_discount_l637_63790

/-- Proves that applying a 20% discount to a price that is 80% of the original
    results in a final price that is 64% of the original. -/
theorem shirt_sale_discount (original_price : ℝ) (original_price_pos : 0 < original_price) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.8 * first_sale_price
  final_price / original_price = 0.64 := by sorry

end NUMINAMATH_CALUDE_shirt_sale_discount_l637_63790


namespace NUMINAMATH_CALUDE_megans_vacation_pictures_l637_63757

theorem megans_vacation_pictures (zoo_pics museum_pics deleted_pics : ℕ) 
  (h1 : zoo_pics = 15)
  (h2 : museum_pics = 18)
  (h3 : deleted_pics = 31) :
  zoo_pics + museum_pics - deleted_pics = 2 := by
  sorry

end NUMINAMATH_CALUDE_megans_vacation_pictures_l637_63757


namespace NUMINAMATH_CALUDE_sum_gcd_lcm_18_30_45_l637_63714

def A : ℕ := Nat.gcd 18 (Nat.gcd 30 45)
def B : ℕ := Nat.lcm 18 (Nat.lcm 30 45)

theorem sum_gcd_lcm_18_30_45 : A + B = 93 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcd_lcm_18_30_45_l637_63714


namespace NUMINAMATH_CALUDE_bench_seating_l637_63747

theorem bench_seating (N : ℕ) : (∃ x : ℕ, 7 * N = x ∧ 11 * N = x) ↔ N ≥ 77 :=
sorry

end NUMINAMATH_CALUDE_bench_seating_l637_63747


namespace NUMINAMATH_CALUDE_solution_problem_l637_63783

theorem solution_problem (a₁ a₂ a₃ a₄ a₅ b : ℤ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ 
                a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ 
                a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ 
                a₄ ≠ a₅)
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ = 9)
  (h_root : (b - a₁) * (b - a₂) * (b - a₃) * (b - a₄) * (b - a₅) = 2009) :
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_solution_problem_l637_63783


namespace NUMINAMATH_CALUDE_subtract_inequality_l637_63785

theorem subtract_inequality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a < b) : a - 3 < b - 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_inequality_l637_63785


namespace NUMINAMATH_CALUDE_x_value_l637_63733

/-- Given that 20% of x is 15 less than 15% of 1500, prove that x = 1050 -/
theorem x_value : ∃ x : ℝ, (0.2 * x = 0.15 * 1500 - 15) ∧ x = 1050 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l637_63733


namespace NUMINAMATH_CALUDE_inequality_proof_l637_63730

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / (x + 2*y + 3*z)) + (y / (y + 2*z + 3*x)) + (z / (z + 2*x + 3*y)) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l637_63730


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l637_63764

/-- The base length of an isosceles triangle with specific conditions -/
theorem isosceles_triangle_base_length 
  (equilateral_perimeter : ℝ) 
  (isosceles_perimeter : ℝ) 
  (h_equilateral : equilateral_perimeter = 60) 
  (h_isosceles : isosceles_perimeter = 45) 
  (h_shared_side : equilateral_perimeter / 3 = (isosceles_perimeter - isosceles_base) / 2) : 
  isosceles_base = 5 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l637_63764


namespace NUMINAMATH_CALUDE_wolves_games_count_l637_63791

theorem wolves_games_count : 
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →
    ∃ (total_games : ℕ),
      total_games = initial_games + 10 ∧
      (initial_wins + 9 : ℚ) / total_games = 3/5 ∧
      total_games = 25 :=
by sorry

end NUMINAMATH_CALUDE_wolves_games_count_l637_63791


namespace NUMINAMATH_CALUDE_trig_expression_equality_l637_63731

theorem trig_expression_equality : 
  (Real.cos (27 * π / 180) - Real.sqrt 2 * Real.sin (18 * π / 180)) / Real.cos (63 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l637_63731


namespace NUMINAMATH_CALUDE_S_infinite_l637_63788

/-- Sum of positive integer divisors of n -/
def d (n : ℕ) : ℕ := sorry

/-- Euler's totient function: count of integers in [0,n] coprime with n -/
def φ (n : ℕ) : ℕ := sorry

/-- The set of integers n for which d(n) * φ(n) is a perfect square -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, d n * φ n = k^2}

/-- The main theorem: S is infinite -/
theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_infinite_l637_63788


namespace NUMINAMATH_CALUDE_triangle_area_l637_63722

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C,
    if b = 1, c = √3, and angle C = 2π/3, then the area of the triangle is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 1 → c = Real.sqrt 3 → C = 2 * Real.pi / 3 →
  (1/2) * b * c * Real.sin C = Real.sqrt 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l637_63722


namespace NUMINAMATH_CALUDE_positive_x_y_l637_63704

theorem positive_x_y (x y : ℝ) (h1 : x - y < x) (h2 : x + y > y) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_x_y_l637_63704


namespace NUMINAMATH_CALUDE_product_of_coefficients_l637_63744

theorem product_of_coefficients (x y z w A B : ℝ) 
  (eq1 : 4 * x * z + y * w = 3)
  (eq2 : x * w + y * z = 6)
  (eq3 : (A * x + y) * (B * z + w) = 15) :
  A * B = 4 := by sorry

end NUMINAMATH_CALUDE_product_of_coefficients_l637_63744


namespace NUMINAMATH_CALUDE_book_cost_l637_63782

/-- If two identical books cost $36 in total, then eight of these books will cost $144. -/
theorem book_cost (two_books_cost : ℕ) (h : two_books_cost = 36) : 
  (8 * (two_books_cost / 2) = 144) :=
sorry

end NUMINAMATH_CALUDE_book_cost_l637_63782


namespace NUMINAMATH_CALUDE_cuboid_to_cube_surface_area_l637_63786

/-- Given a cuboid with a square base, if reducing its height by 4 cm results in a cube
    and decreases its volume by 64 cubic centimeters, then the surface area of the
    resulting cube is 96 square centimeters. -/
theorem cuboid_to_cube_surface_area (l w h : ℝ) : 
  l = w → -- The base is square
  (l * w * h) - (l * w * (h - 4)) = 64 → -- Volume decrease
  l * w * 4 = 64 → -- Volume decrease equals base area times height reduction
  6 * (l * l) = 96 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_to_cube_surface_area_l637_63786


namespace NUMINAMATH_CALUDE_circle_rectangles_l637_63707

/-- The number of points on the circle's circumference -/
def n : ℕ := 12

/-- The number of diameters in the circle -/
def num_diameters : ℕ := n / 2

/-- The number of rectangles that can be formed -/
def num_rectangles : ℕ := Nat.choose num_diameters 2

theorem circle_rectangles :
  num_rectangles = 15 :=
sorry

end NUMINAMATH_CALUDE_circle_rectangles_l637_63707


namespace NUMINAMATH_CALUDE_remainder_problem_l637_63756

theorem remainder_problem : (98 * 103 + 7) % 12 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l637_63756


namespace NUMINAMATH_CALUDE_sum_g_15_neg_15_l637_63748

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 - e * x^6 + f * x^2 + 5

-- Theorem statement
theorem sum_g_15_neg_15 (d e f : ℝ) (h : g d e f 15 = 7) :
  g d e f 15 + g d e f (-15) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_g_15_neg_15_l637_63748


namespace NUMINAMATH_CALUDE_sqrt_twelve_less_than_four_l637_63700

theorem sqrt_twelve_less_than_four : Real.sqrt 12 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_less_than_four_l637_63700


namespace NUMINAMATH_CALUDE_regression_slope_l637_63742

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 2 - x

theorem regression_slope :
  ∀ x : ℝ, linear_regression (x + 1) = linear_regression x - 1 := by
  sorry

end NUMINAMATH_CALUDE_regression_slope_l637_63742


namespace NUMINAMATH_CALUDE_system_solution_l637_63755

theorem system_solution : 
  ∀ x y : ℝ, 
  (x + y + Real.sqrt (x * y) = 28 ∧ x^2 + y^2 + x * y = 336) ↔ 
  ((x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4)) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l637_63755


namespace NUMINAMATH_CALUDE_intersection_distance_l637_63776

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- State the theorem
theorem intersection_distance :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l637_63776


namespace NUMINAMATH_CALUDE_luke_mowing_money_l637_63711

/-- The amount of money Luke made mowing lawns -/
def mowing_money : ℝ := sorry

/-- The amount of money Luke made weed eating -/
def weed_eating_money : ℝ := 18

/-- The amount Luke spends per week -/
def weekly_spending : ℝ := 3

/-- The number of weeks the money lasts -/
def weeks_lasted : ℝ := 9

/-- Theorem stating that Luke made $9 mowing lawns -/
theorem luke_mowing_money : mowing_money = 9 := by
  sorry

end NUMINAMATH_CALUDE_luke_mowing_money_l637_63711


namespace NUMINAMATH_CALUDE_max_planks_from_trunk_l637_63749

/-- Represents a cylindrical tree trunk -/
structure Trunk :=
  (diameter : ℝ)

/-- Represents a plank with thickness and width -/
structure Plank :=
  (thickness : ℝ)
  (width : ℝ)

/-- Calculates the maximum number of planks that can be cut from a trunk -/
def max_planks (t : Trunk) (p : Plank) : ℕ :=
  sorry

/-- Theorem stating the maximum number of planks that can be cut -/
theorem max_planks_from_trunk (t : Trunk) (p : Plank) :
  t.diameter = 46 → p.thickness = 4 → p.width = 12 → max_planks t p = 29 := by
  sorry

end NUMINAMATH_CALUDE_max_planks_from_trunk_l637_63749


namespace NUMINAMATH_CALUDE_cube_preserves_order_l637_63775

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l637_63775


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_six_l637_63724

theorem sqrt_sum_equals_six : 
  Real.sqrt (11 + 6 * Real.sqrt 2) + Real.sqrt (11 - 6 * Real.sqrt 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_six_l637_63724


namespace NUMINAMATH_CALUDE_stock_change_theorem_l637_63772

theorem stock_change_theorem (initial_value : ℝ) : 
  let day1_value := initial_value * (1 - 0.15)
  let day2_value := day1_value * (1 + 0.25)
  let percent_change := (day2_value - initial_value) / initial_value * 100
  percent_change = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_stock_change_theorem_l637_63772


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_11_l637_63793

theorem unique_number_with_three_prime_divisors_including_11 :
  ∀ (x n : ℕ), 
    x = 9^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
    11 ∣ x →
    x = 59048 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_divisors_including_11_l637_63793


namespace NUMINAMATH_CALUDE_tabitha_initial_colors_l637_63771

/-- Represents Tabitha's hair coloring tradition --/
structure HairColorTradition where
  current_age : ℕ
  start_age : ℕ
  colors_in_three_years : ℕ
  
/-- The number of colors Tabitha had when she started the tradition --/
def initial_colors (t : HairColorTradition) : ℕ :=
  t.colors_in_three_years - (t.current_age - t.start_age) - 3

/-- The theorem stating that Tabitha had 2 colors when she started the tradition --/
theorem tabitha_initial_colors (t : HairColorTradition) 
  (h1 : t.current_age = 18)
  (h2 : t.start_age = 15)
  (h3 : t.colors_in_three_years = 8) :
  initial_colors t = 2 := by
  sorry

#check tabitha_initial_colors

end NUMINAMATH_CALUDE_tabitha_initial_colors_l637_63771


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l637_63708

/-- Given a parabola y = ax² - a (a ≠ 0) intersecting a line y = kx at points
    with sum of x-coordinates less than 0, prove that the line y = ax + k
    passes through the first and fourth quadrants. -/
theorem parabola_line_intersection (a k : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 - a = k * x₁ ∧
               a * x₂^2 - a = k * x₂ ∧
               x₁ + x₂ < 0) →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x + k) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = a * x + k) :=
by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l637_63708


namespace NUMINAMATH_CALUDE_conditioner_shampoo_ratio_l637_63789

/-- Proves the ratio of daily conditioner use to daily shampoo use -/
theorem conditioner_shampoo_ratio 
  (daily_shampoo : ℝ) 
  (total_volume : ℝ) 
  (days : ℕ) 
  (h1 : daily_shampoo = 1)
  (h2 : total_volume = 21)
  (h3 : days = 14) :
  (total_volume - daily_shampoo * days) / days / daily_shampoo = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_conditioner_shampoo_ratio_l637_63789


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l637_63765

theorem simplify_trig_expression :
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.cos (10 * π / 180) = 4 := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l637_63765


namespace NUMINAMATH_CALUDE_linear_system_solution_l637_63727

/-- Solution to a system of linear equations -/
theorem linear_system_solution (a b c h : ℝ) :
  let x := (h - b) * (h - c) / ((a - b) * (a - c))
  let y := (h - a) * (h - c) / ((b - a) * (b - c))
  let z := (h - a) * (h - b) / ((c - a) * (c - b))
  x + y + z = 1 ∧
  a * x + b * y + c * z = h ∧
  a^2 * x + b^2 * y + c^2 * z = h^2 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l637_63727


namespace NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l637_63792

theorem mark_and_carolyn_money_sum : 
  (5 : ℚ) / 6 + (2 : ℚ) / 5 = (37 : ℚ) / 30 := by sorry

end NUMINAMATH_CALUDE_mark_and_carolyn_money_sum_l637_63792


namespace NUMINAMATH_CALUDE_m_three_sufficient_not_necessary_l637_63706

def a (m : ℝ) : ℝ × ℝ := (-9, m^2)
def b : ℝ × ℝ := (1, -1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem m_three_sufficient_not_necessary :
  (∃ (m : ℝ), m ≠ 3 ∧ parallel (a m) b) ∧
  (∀ (m : ℝ), m = 3 → parallel (a m) b) :=
sorry

end NUMINAMATH_CALUDE_m_three_sufficient_not_necessary_l637_63706


namespace NUMINAMATH_CALUDE_waiter_tables_l637_63794

theorem waiter_tables (total_customers : ℕ) (people_per_table : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) :
  total_customers = 90 →
  people_per_table = women_per_table + men_per_table →
  women_per_table = 7 →
  men_per_table = 3 →
  total_customers / people_per_table = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l637_63794


namespace NUMINAMATH_CALUDE_inequality_properties_l637_63779

theorem inequality_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l637_63779


namespace NUMINAMATH_CALUDE_day_of_week_proof_l637_63758

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date -/
structure Date where
  year : Nat
  month : Nat
  day : Nat

/-- Calculate the number of days between two dates -/
def daysBetween (d1 d2 : Date) : Int :=
  sorry

/-- Get the day of the week for a given date -/
def getDayOfWeek (d : Date) (knownDate : Date) (knownDay : DayOfWeek) : DayOfWeek :=
  sorry

theorem day_of_week_proof :
  let knownDate := Date.mk 1998 4 10
  let knownDay := DayOfWeek.Friday
  let date1 := Date.mk 1918 7 6
  let date2 := Date.mk 2018 6 6
  (getDayOfWeek date1 knownDate knownDay = DayOfWeek.Saturday) ∧
  (getDayOfWeek date2 knownDate knownDay = DayOfWeek.Tuesday) := by
  sorry

end NUMINAMATH_CALUDE_day_of_week_proof_l637_63758


namespace NUMINAMATH_CALUDE_visit_probability_l637_63797

/-- The probability of Jen visiting either Chile or Madagascar, but not both -/
theorem visit_probability (p_chile p_madagascar : ℝ) 
  (h_chile : p_chile = 0.30)
  (h_madagascar : p_madagascar = 0.50) : 
  p_chile + p_madagascar - p_chile * p_madagascar = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_visit_probability_l637_63797


namespace NUMINAMATH_CALUDE_line_intersections_l637_63780

theorem line_intersections : 
  let line1 : ℝ → ℝ := λ x => 5 * x - 20
  let line2 : ℝ → ℝ := λ x => 190 - 3 * x
  let line3 : ℝ → ℝ := λ x => 2 * x + 15
  ∃ (x1 x2 : ℝ), 
    (line1 x1 = line2 x1 ∧ x1 = 105 / 4) ∧
    (line1 x2 = line3 x2 ∧ x2 = 35 / 3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersections_l637_63780


namespace NUMINAMATH_CALUDE_number_division_problem_l637_63746

theorem number_division_problem :
  ∃! n : ℕ, 
    n / (555 + 445) = 2 * (555 - 445) ∧ 
    n % (555 + 445) = 70 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l637_63746


namespace NUMINAMATH_CALUDE_gumball_probability_l637_63736

/-- Given a jar with blue and pink gumballs, if the probability of drawing two blue
    gumballs in succession (with replacement) is 9/49, then the probability of
    drawing a pink gumball is 4/7. -/
theorem gumball_probability (B P : ℕ) (h : B > 0 ∧ P > 0) :
  (B : ℚ)^2 / (B + P : ℚ)^2 = 9/49 → (P : ℚ) / (B + P : ℚ) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_gumball_probability_l637_63736


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l637_63702

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > -a) ↔ a > -3/2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l637_63702


namespace NUMINAMATH_CALUDE_cosine_vertical_shift_l637_63784

theorem cosine_vertical_shift 
  (a b c d : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_oscillation : ∀ x : ℝ, 0 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 4) : 
  d = 2 := by
sorry

end NUMINAMATH_CALUDE_cosine_vertical_shift_l637_63784


namespace NUMINAMATH_CALUDE_betty_rice_purchase_l637_63745

theorem betty_rice_purchase (o r : ℝ) : 
  (o ≥ 8 + r / 3 ∧ o ≤ 3 * r) → r ≥ 3 := by sorry

end NUMINAMATH_CALUDE_betty_rice_purchase_l637_63745


namespace NUMINAMATH_CALUDE_simplify_expression_l637_63720

theorem simplify_expression (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a ≥ b) :
  (a - b) / (Real.sqrt a + Real.sqrt b) + (a * Real.sqrt a + b * Real.sqrt b) / (a - Real.sqrt (a * b) + b) = 2 * Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l637_63720


namespace NUMINAMATH_CALUDE_no_k_exists_for_prime_and_binomial_cong_l637_63716

theorem no_k_exists_for_prime_and_binomial_cong (k : ℕ+) (p : ℕ) : 
  p = 6 * k + 1 → 
  Nat.Prime p → 
  (Nat.choose (3 * k) k : ZMod p) = 1 → 
  False := by sorry

end NUMINAMATH_CALUDE_no_k_exists_for_prime_and_binomial_cong_l637_63716


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l637_63766

/-- The number of ways to distribute n indistinguishable objects into k distinct categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ice cream flavor combinations -/
def ice_cream_combinations : ℕ := distribute 4 4

theorem ice_cream_theorem : ice_cream_combinations = 35 := by sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l637_63766


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l637_63769

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (7 + 18 * i) / (3 - 4 * i) = -51/25 + 82/25 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l637_63769


namespace NUMINAMATH_CALUDE_croissant_mix_time_l637_63777

/-- The time it takes to make croissants -/
def croissant_making : Prop :=
  let fold_count : ℕ := 4
  let fold_time : ℕ := 5
  let rest_count : ℕ := 4
  let rest_time : ℕ := 75
  let bake_time : ℕ := 30
  let total_time : ℕ := 6 * 60

  let fold_total : ℕ := fold_count * fold_time
  let rest_total : ℕ := rest_count * rest_time
  let known_time : ℕ := fold_total + rest_total + bake_time

  let mix_time : ℕ := total_time - known_time

  mix_time = 10

theorem croissant_mix_time : croissant_making := by
  sorry

end NUMINAMATH_CALUDE_croissant_mix_time_l637_63777


namespace NUMINAMATH_CALUDE_cube_volume_l637_63739

/-- The volume of a cube with total edge length of 60 cm is 125 cubic centimeters. -/
theorem cube_volume (total_edge_length : ℝ) (h : total_edge_length = 60) : 
  (total_edge_length / 12)^3 = 125 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_l637_63739


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l637_63763

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 25 km/hr in still water, traveling downstream
    in a stream with a speed of 5 km/hr for 4 hours, travels a distance of 120 km. -/
theorem boat_downstream_distance :
  distance_downstream 25 5 4 = 120 := by
  sorry

#eval distance_downstream 25 5 4

end NUMINAMATH_CALUDE_boat_downstream_distance_l637_63763


namespace NUMINAMATH_CALUDE_money_distribution_l637_63738

/-- Given that A, B, and C have a total of 250 Rs., and A and C together have 200 Rs.,
    prove that B has 50 Rs. -/
theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 250)  -- Total money of A, B, and C
  (h2 : A + C = 200)      -- Money of A and C together
  : B = 50 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l637_63738


namespace NUMINAMATH_CALUDE_imaginary_product_implies_a_value_l637_63770

theorem imaginary_product_implies_a_value (a : ℝ) : 
  (∃ b : ℝ, (a - Complex.I) * (3 - 2 * Complex.I) = b * Complex.I) → a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_product_implies_a_value_l637_63770


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_range_l637_63718

theorem geometric_sequence_fourth_term_range 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a1 : 0 < a 1 ∧ a 1 < 1)
  (h_a2 : 1 < a 2 ∧ a 2 < 2)
  (h_a3 : 2 < a 3 ∧ a 3 < 4) :
  2 * Real.sqrt 2 < a 4 ∧ a 4 < 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_range_l637_63718


namespace NUMINAMATH_CALUDE_orthocenter_locus_l637_63773

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  (ha : a > 0)
  (hb : b > 0)
  (hba : b ≤ a)

/-- A triangle inscribed in an ellipse -/
structure InscribedTriangle (e : Ellipse a b) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  hA : A = (-a, 0)
  hB : B = (a, 0)
  hC : (C.1^2 / a^2) + (C.2^2 / b^2) = 1

/-- The orthocenter of a triangle -/
def orthocenter (t : InscribedTriangle e) : ℝ × ℝ :=
  sorry

/-- The locus of the orthocenter is an ellipse -/
theorem orthocenter_locus (e : Ellipse a b) :
  ∀ t : InscribedTriangle e,
  let M := orthocenter t
  ((M.1^2 / a^2) + (M.2^2 / (a^2/b)^2) = 1) :=
sorry

end NUMINAMATH_CALUDE_orthocenter_locus_l637_63773


namespace NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l637_63712

theorem max_value_of_sum_and_powers (a b c d : ℝ) :
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  a + b + c + d = 2 →
  ∃ (m : ℝ), m = 2 ∧ ∀ (x y z w : ℝ), 
    x ≥ 0 → y ≥ 0 → z ≥ 0 → w ≥ 0 →
    x + y + z + w = 2 →
    x + y^2 + z^3 + w^4 ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_and_powers_l637_63712


namespace NUMINAMATH_CALUDE_l_shaped_area_l637_63719

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (outer_side : ℝ) (inner_side1 inner_side2 inner_side3 : ℝ) :
  outer_side = 6 ∧ 
  inner_side1 = 1 ∧ 
  inner_side2 = 2 ∧ 
  inner_side3 = 3 →
  outer_side ^ 2 - (inner_side1 ^ 2 + inner_side2 ^ 2 + inner_side3 ^ 2) = 22 :=
by sorry

end NUMINAMATH_CALUDE_l_shaped_area_l637_63719


namespace NUMINAMATH_CALUDE_derek_walking_time_l637_63701

/-- The time it takes Derek to walk a mile with his brother -/
def time_with_brother : ℝ := 12

/-- The time it takes Derek to walk a mile without his brother -/
def time_without_brother : ℝ := 9

/-- The additional time it takes to walk 20 miles with his brother -/
def additional_time : ℝ := 60

theorem derek_walking_time :
  time_with_brother = 12 ∧
  time_without_brother * 20 + additional_time = time_with_brother * 20 :=
sorry

end NUMINAMATH_CALUDE_derek_walking_time_l637_63701


namespace NUMINAMATH_CALUDE_B_cannot_be_possible_l637_63799

-- Define the set A
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Define the set B (the one we want to prove cannot be possible)
def B : Set ℝ := {x : ℝ | x ≥ -1}

-- Theorem statement
theorem B_cannot_be_possible : A ∩ B = ∅ → False := by
  sorry

end NUMINAMATH_CALUDE_B_cannot_be_possible_l637_63799


namespace NUMINAMATH_CALUDE_blueberry_picking_difference_l637_63760

theorem blueberry_picking_difference (annie kathryn ben : ℕ) : 
  annie = 8 →
  kathryn = annie + 2 →
  ben < kathryn →
  annie + kathryn + ben = 25 →
  kathryn - ben = 3 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_picking_difference_l637_63760


namespace NUMINAMATH_CALUDE_no_solution_exists_l637_63798

/-- S(x) represents the sum of the digits of the natural number x -/
def S (x : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are no natural numbers x satisfying the equation -/
theorem no_solution_exists : ¬ ∃ x : ℕ, x + S x + S (S x) = 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l637_63798


namespace NUMINAMATH_CALUDE_no_integer_solution_for_175_l637_63723

theorem no_integer_solution_for_175 :
  ∀ x y : ℤ, x^2 + y^2 ≠ 175 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_175_l637_63723


namespace NUMINAMATH_CALUDE_sum_even_integers_200_to_400_l637_63734

def even_integers_between (a b : ℕ) : List ℕ :=
  (List.range (b - a + 1)).map (fun i => a + 2 * i)

theorem sum_even_integers_200_to_400 :
  (even_integers_between 200 400).sum = 30100 := by
  sorry

end NUMINAMATH_CALUDE_sum_even_integers_200_to_400_l637_63734


namespace NUMINAMATH_CALUDE_brownie_problem_l637_63725

def initial_brownies : ℕ := 16

theorem brownie_problem (B : ℕ) (h1 : B = initial_brownies) :
  let remaining_after_children : ℚ := 3/4 * B
  let remaining_after_family : ℚ := 1/2 * remaining_after_children
  let final_remaining : ℚ := remaining_after_family - 1
  final_remaining = 5 := by sorry

end NUMINAMATH_CALUDE_brownie_problem_l637_63725


namespace NUMINAMATH_CALUDE_subtracted_number_l637_63751

theorem subtracted_number (t k x : ℝ) : 
  t = 5 / 9 * (k - x) → 
  t = 50 → 
  k = 122 → 
  x = 32 := by
sorry

end NUMINAMATH_CALUDE_subtracted_number_l637_63751


namespace NUMINAMATH_CALUDE_expression_value_l637_63703

theorem expression_value (a b : ℤ) (h : a - b = 1) : 2*b - (2*a + 6) = -8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l637_63703


namespace NUMINAMATH_CALUDE_circle_center_l637_63768

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y = 0

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 5

/-- Theorem: The center of the circle defined by x^2 + y^2 - 2x - 4y = 0 is at (1, 2) -/
theorem circle_center : CircleCenter 1 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l637_63768


namespace NUMINAMATH_CALUDE_soccer_tournament_points_l637_63705

theorem soccer_tournament_points (n k : ℕ) (h1 : n ≥ 3) (h2 : 2 ≤ k) (h3 : k ≤ n - 1) :
  let min_points := 3 * n - (3 * k + 1) / 2 - 2
  ∀ (team_points : ℕ → ℕ),
    (∀ i, i < n → team_points i ≤ 3 * (n - 1)) →
    (∀ i j, i < n → j < n → i ≠ j → 
      team_points i + team_points j ≥ 1 ∧ team_points i + team_points j ≤ 4) →
    (∀ i, i < n → team_points i ≥ min_points) →
    ∃ (top_teams : Finset ℕ),
      top_teams.card ≤ k ∧
      ∀ j, j < n → j ∉ top_teams → team_points j < min_points :=
by sorry


end NUMINAMATH_CALUDE_soccer_tournament_points_l637_63705


namespace NUMINAMATH_CALUDE_parallelogram_count_is_392_l637_63761

/-- Represents a parallelogram PQRS with the given properties -/
structure Parallelogram where
  q : ℕ+  -- x-coordinate of Q (also y-coordinate since Q is on y = x)
  s : ℕ+  -- x-coordinate of S
  m : ℕ   -- slope of line y = mx where S lies
  h_m_gt_one : m > 1
  h_area : (m - 1) * q * s = 250000

/-- Counts the number of valid parallelograms -/
def count_parallelograms : ℕ := sorry

/-- The main theorem stating that the count of valid parallelograms is 392 -/
theorem parallelogram_count_is_392 : count_parallelograms = 392 := by sorry

end NUMINAMATH_CALUDE_parallelogram_count_is_392_l637_63761


namespace NUMINAMATH_CALUDE_badge_exchange_problem_l637_63715

theorem badge_exchange_problem (V T : ℕ) :
  V = T + 5 →
  (V - V * 24 / 100 + T * 20 / 100) = (T - T * 20 / 100 + V * 24 / 100 - 1) →
  V = 50 ∧ T = 45 := by
sorry

end NUMINAMATH_CALUDE_badge_exchange_problem_l637_63715


namespace NUMINAMATH_CALUDE_people_per_column_l637_63713

theorem people_per_column (total_people : ℕ) (people_per_column : ℕ) : 
  (total_people = 16 * people_per_column) ∧ 
  (total_people = 15 * 32) → 
  people_per_column = 30 := by
  sorry

end NUMINAMATH_CALUDE_people_per_column_l637_63713


namespace NUMINAMATH_CALUDE_embroidery_time_l637_63732

-- Define the stitches per minute
def stitches_per_minute : ℕ := 4

-- Define the number of stitches for each design
def flower_stitches : ℕ := 60
def unicorn_stitches : ℕ := 180
def godzilla_stitches : ℕ := 800

-- Define the number of each design
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzilla : ℕ := 1

-- Theorem to prove
theorem embroidery_time :
  (num_flowers * flower_stitches + num_unicorns * unicorn_stitches + num_godzilla * godzilla_stitches) / stitches_per_minute = 1085 := by
  sorry

end NUMINAMATH_CALUDE_embroidery_time_l637_63732


namespace NUMINAMATH_CALUDE_symmetrical_line_over_x_axis_l637_63753

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Reflects a line over the X axis --/
def reflect_over_x_axis (l : Line) : Line :=
  { a := l.a,
    b := -l.b,
    c := -l.c,
    eq := sorry }

theorem symmetrical_line_over_x_axis :
  let original_line : Line := { a := 1, b := -2, c := 3, eq := sorry }
  let reflected_line := reflect_over_x_axis original_line
  reflected_line.a = 1 ∧ reflected_line.b = 2 ∧ reflected_line.c = -3 :=
by sorry

end NUMINAMATH_CALUDE_symmetrical_line_over_x_axis_l637_63753
