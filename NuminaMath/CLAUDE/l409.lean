import Mathlib

namespace NUMINAMATH_CALUDE_increasing_function_property_l409_40972

-- Define an increasing function on the real line
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- State the theorem
theorem increasing_function_property (f : ℝ → ℝ) (h : IncreasingFunction f) :
  (∀ a b : ℝ, a + b ≥ 0 → f a + f b ≥ f (-a) + f (-b)) ∧
  (∀ a b : ℝ, f a + f b ≥ f (-a) + f (-b) → a + b ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_property_l409_40972


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l409_40983

/-- The eccentricity of a hyperbola that shares foci with a specific ellipse -/
theorem hyperbola_eccentricity (m : ℝ) : 
  ∃ (e : ℝ), e = 2 ∧ 
  (∀ (x y : ℝ), x^2 - y^2/m^2 = 1 → 
    ∃ (c : ℝ), c^2 = x^2 + y^2 ∧
    (∀ (x' y' : ℝ), x'^2/9 + y'^2/5 = 1 → 
      ∃ (c' : ℝ), c'^2 = x'^2 + y'^2 ∧ c = c') ∧
    e = c) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l409_40983


namespace NUMINAMATH_CALUDE_a_is_perfect_square_l409_40929

def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 := by
  sorry

end NUMINAMATH_CALUDE_a_is_perfect_square_l409_40929


namespace NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l409_40963

theorem at_least_three_positive_and_negative 
  (a : Fin 12 → ℝ) 
  (h : ∀ i ∈ Finset.range 10, a (i + 2) * (a (i + 1) - a (i + 2) + a (i + 3)) < 0) :
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i > 0 ∧ a j > 0 ∧ a k > 0) ∧
  (∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a i < 0 ∧ a j < 0 ∧ a k < 0) :=
sorry

end NUMINAMATH_CALUDE_at_least_three_positive_and_negative_l409_40963


namespace NUMINAMATH_CALUDE_volleyballs_count_l409_40967

/-- The number of volleyballs in Reynald's purchase --/
def volleyballs : ℕ :=
  let total_balls : ℕ := 145
  let soccer_balls : ℕ := 20
  let basketballs : ℕ := soccer_balls + 5
  let tennis_balls : ℕ := 2 * soccer_balls
  let baseballs : ℕ := soccer_balls + 10
  total_balls - (soccer_balls + basketballs + tennis_balls + baseballs)

/-- Theorem stating that the number of volleyballs is 30 --/
theorem volleyballs_count : volleyballs = 30 := by
  sorry

end NUMINAMATH_CALUDE_volleyballs_count_l409_40967


namespace NUMINAMATH_CALUDE_total_limes_is_57_l409_40948

/-- The number of limes Alyssa picked -/
def alyssa_limes : ℕ := 25

/-- The number of limes Mike picked -/
def mike_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := alyssa_limes + mike_limes

/-- Theorem stating that the total number of limes picked is 57 -/
theorem total_limes_is_57 : total_limes = 57 := by
  sorry

end NUMINAMATH_CALUDE_total_limes_is_57_l409_40948


namespace NUMINAMATH_CALUDE_garage_sale_pricing_l409_40973

theorem garage_sale_pricing (total_items : ℕ) (radio_highest_rank : ℕ) (h1 : total_items = 38) (h2 : radio_highest_rank = 16) :
  total_items + 1 - radio_highest_rank = 24 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_pricing_l409_40973


namespace NUMINAMATH_CALUDE_line_equation_a_value_l409_40920

/-- Given a line passing through points (4,3) and (12,-3), if its equation is in the form (x/a) + (y/b) = 1, then a = 8 -/
theorem line_equation_a_value (a b : ℝ) : 
  (∀ x y : ℝ, (x / a + y / b = 1) ↔ (3 * x + 4 * y = 24)) →
  ((4 : ℝ) / a + (3 : ℝ) / b = 1) →
  ((12 : ℝ) / a + (-3 : ℝ) / b = 1) →
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_line_equation_a_value_l409_40920


namespace NUMINAMATH_CALUDE_no_solutions_to_radical_equation_l409_40961

theorem no_solutions_to_radical_equation :
  ∀ x : ℝ, x ≥ 2 →
    ¬ (Real.sqrt (x + 7 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_to_radical_equation_l409_40961


namespace NUMINAMATH_CALUDE_problem_statement_l409_40957

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ x * y ≤ a * b) ∧
  (a^2 + b^2 ≥ 1/2) ∧
  (4/a + 1/b ≥ 9) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ Real.sqrt x + Real.sqrt y < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l409_40957


namespace NUMINAMATH_CALUDE_inverse_g_at_negative_one_l409_40912

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 - 5

-- State the theorem
theorem inverse_g_at_negative_one :
  Function.invFun g (-1) = 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_g_at_negative_one_l409_40912


namespace NUMINAMATH_CALUDE_inequality_solution_set_l409_40981

theorem inequality_solution_set :
  {x : ℝ | (|x| + x) * (Real.sin x - 2) < 0} = Set.Ioo 0 Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l409_40981


namespace NUMINAMATH_CALUDE_area_is_33_l409_40944

/-- A line with slope -3 intersecting positive x and y axes -/
structure Line1 where
  slope : ℝ
  x_intercept : ℝ
  y_intercept : ℝ

/-- A line intersecting x and y axes -/
structure Line2 where
  x_intercept : ℝ
  y_intercept : ℝ

/-- The intersection point of two lines -/
structure Intersection where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def problem_setup (l1 : Line1) (l2 : Line2) (e : Intersection) : Prop :=
  l1.slope = -3 ∧
  l1.x_intercept > 0 ∧
  l1.y_intercept > 0 ∧
  l2.x_intercept = 10 ∧
  e.x = 3 ∧
  e.y = 3

/-- The area of quadrilateral OBEC -/
def area_OBEC (l1 : Line1) (l2 : Line2) (e : Intersection) : ℝ := sorry

/-- Theorem stating the area of quadrilateral OBEC is 33 -/
theorem area_is_33 (l1 : Line1) (l2 : Line2) (e : Intersection) :
  problem_setup l1 l2 e → area_OBEC l1 l2 e = 33 := by sorry

end NUMINAMATH_CALUDE_area_is_33_l409_40944


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l409_40904

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^12 + i^17 + i^22 + i^27 + i^32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l409_40904


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l409_40923

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_to_use : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_use * 
  Nat.factorial rings_to_use * 
  Nat.choose (rings_to_use + fingers - 1) (fingers - 1)

/-- Theorem stating the number of ring arrangements for the given problem -/
theorem ring_arrangement_count : ring_arrangements 10 6 5 = 31752000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l409_40923


namespace NUMINAMATH_CALUDE_leap_years_in_200_years_l409_40901

/-- A calendrical system where leap years occur every four years without exception. -/
structure CalendarSystem where
  /-- The period in years -/
  period : ℕ
  /-- The frequency of leap years -/
  leap_year_frequency : ℕ
  /-- Assertion that leap years occur every four years -/
  leap_year_every_four : leap_year_frequency = 4

/-- The number of leap years in a given period for a calendar system -/
def num_leap_years (c : CalendarSystem) : ℕ :=
  c.period / c.leap_year_frequency

/-- Theorem stating that in a 200-year period with leap years every 4 years, there are 50 leap years -/
theorem leap_years_in_200_years (c : CalendarSystem) 
  (h_period : c.period = 200) : num_leap_years c = 50 := by
  sorry

end NUMINAMATH_CALUDE_leap_years_in_200_years_l409_40901


namespace NUMINAMATH_CALUDE_emmas_drive_speed_l409_40921

/-- Proves that given the conditions of Emma's drive, her average speed during the last 40 minutes was 75 mph -/
theorem emmas_drive_speed (total_distance : ℝ) (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 120)
  (h2 : total_time = 2)
  (h3 : speed1 = 50)
  (h4 : speed2 = 55) :
  let segment_time := total_time / 3
  let speed3 := (total_distance - (speed1 + speed2) * segment_time) / segment_time
  speed3 = 75 := by sorry

end NUMINAMATH_CALUDE_emmas_drive_speed_l409_40921


namespace NUMINAMATH_CALUDE_right_triangle_configurations_l409_40977

def points_on_line : ℕ := 58

theorem right_triangle_configurations :
  let total_points := 2 * points_on_line
  let ways_hypotenuse_on_line := points_on_line.choose 2 * points_on_line
  let ways_leg_on_line := points_on_line * points_on_line
  ways_hypotenuse_on_line * 2 + ways_leg_on_line * 2 = 6724 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_configurations_l409_40977


namespace NUMINAMATH_CALUDE_prime_squares_sum_theorem_l409_40962

theorem prime_squares_sum_theorem (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ (x y z : ℕ), p^(2*x) + q^(2*y) = z^2) ↔ ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_squares_sum_theorem_l409_40962


namespace NUMINAMATH_CALUDE_collectors_edition_dolls_l409_40988

/-- Prove that given the conditions, Ivy and Luna have 30 collectors edition dolls combined -/
theorem collectors_edition_dolls (dina ivy luna : ℕ) : 
  dina = 60 →
  dina = 2 * ivy →
  ivy = luna + 10 →
  dina + ivy + luna = 150 →
  (2 * ivy / 3 : ℚ) + (luna / 2 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_collectors_edition_dolls_l409_40988


namespace NUMINAMATH_CALUDE_andrews_grapes_l409_40913

/-- The amount of grapes Andrew purchased -/
def grapes : ℕ := sorry

/-- The price of grapes per kg -/
def grape_price : ℕ := 98

/-- The amount of mangoes Andrew purchased in kg -/
def mangoes : ℕ := 7

/-- The price of mangoes per kg -/
def mango_price : ℕ := 50

/-- The total amount Andrew paid -/
def total_paid : ℕ := 1428

theorem andrews_grapes : 
  grapes * grape_price + mangoes * mango_price = total_paid ∧ grapes = 11 := by sorry

end NUMINAMATH_CALUDE_andrews_grapes_l409_40913


namespace NUMINAMATH_CALUDE_original_price_after_percentage_changes_l409_40955

theorem original_price_after_percentage_changes
  (d r s : ℝ) 
  (h1 : 0 < r ∧ r < 100) 
  (h2 : 0 < s ∧ s < 100) 
  (h3 : s < r) :
  let x := (d * 10000) / (10000 + 100 * (r - s) - r * s)
  x * (1 + r / 100) * (1 - s / 100) = d :=
by sorry

end NUMINAMATH_CALUDE_original_price_after_percentage_changes_l409_40955


namespace NUMINAMATH_CALUDE_side_length_b_l409_40945

open Real

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isArithmeticSequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

def hasCorrectArea (t : Triangle) : Prop :=
  1/2 * t.a * t.c * sin t.B = 5 * sqrt 3

-- Main theorem
theorem side_length_b (t : Triangle) 
  (h1 : isArithmeticSequence t)
  (h2 : t.a = 4)
  (h3 : hasCorrectArea t) :
  t.b = sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_side_length_b_l409_40945


namespace NUMINAMATH_CALUDE_equation_solution_l409_40915

theorem equation_solution : ∀ x y : ℕ, 
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) → 
  x = 2 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l409_40915


namespace NUMINAMATH_CALUDE_three_roles_four_people_l409_40952

def number_of_assignments (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

theorem three_roles_four_people :
  number_of_assignments 4 3 = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_three_roles_four_people_l409_40952


namespace NUMINAMATH_CALUDE_photo_framing_yards_l409_40966

/-- Calculates the minimum number of linear yards of framing needed for an enlarged photo with border. -/
def min_framing_yards (original_width : ℕ) (original_height : ℕ) (enlarge_factor : ℕ) (border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let framed_width := enlarged_width + 2 * border_width
  let framed_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (framed_width + framed_height)
  let yards_needed := (perimeter_inches + 35) / 36  -- Ceiling division
  yards_needed

/-- Theorem stating that for a 5x7 inch photo enlarged 4 times with a 3-inch border,
    the minimum number of linear yards of framing needed is 4. -/
theorem photo_framing_yards :
  min_framing_yards 5 7 4 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_photo_framing_yards_l409_40966


namespace NUMINAMATH_CALUDE_inequality_proof_l409_40942

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : y * z + z * x + x * y = 1) : 
  x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ 4 / (9 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l409_40942


namespace NUMINAMATH_CALUDE_largest_integer_fraction_l409_40908

theorem largest_integer_fraction (n : ℤ) : (n / 11 : ℚ) < 2/3 ↔ n ≤ 7 :=
  sorry

end NUMINAMATH_CALUDE_largest_integer_fraction_l409_40908


namespace NUMINAMATH_CALUDE_solution_of_system_l409_40968

variable (a b c x y z : ℝ)

theorem solution_of_system :
  (a * x + b * y - c * z = 2 * a * b) →
  (a * x - b * y + c * z = 2 * a * c) →
  (-a * x + b * y - c * z = 2 * b * c) →
  (x = b + c ∧ y = a + c ∧ z = a + b) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l409_40968


namespace NUMINAMATH_CALUDE_first_system_solution_second_system_solution_l409_40936

-- First system of equations
theorem first_system_solution :
  ∃ (x y : ℝ), 3 * x + 2 * y = 5 ∧ y = 2 * x - 8 ∧ x = 3 ∧ y = -2 := by
sorry

-- Second system of equations
theorem second_system_solution :
  ∃ (x y : ℝ), 2 * x - y = 10 ∧ 2 * x + 3 * y = 2 ∧ x = 4 ∧ y = -2 := by
sorry

end NUMINAMATH_CALUDE_first_system_solution_second_system_solution_l409_40936


namespace NUMINAMATH_CALUDE_shorts_folded_l409_40930

/-- Given the following:
  * There are 20 shirts and 8 pairs of shorts in total
  * 12 shirts are folded
  * 11 pieces of clothing remain to be folded
  Prove that 5 pairs of shorts were folded -/
theorem shorts_folded (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (remaining_to_fold : ℕ) : ℕ :=
  by
  have h1 : total_shirts = 20 := by sorry
  have h2 : total_shorts = 8 := by sorry
  have h3 : folded_shirts = 12 := by sorry
  have h4 : remaining_to_fold = 11 := by sorry
  exact 5

end NUMINAMATH_CALUDE_shorts_folded_l409_40930


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l409_40979

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℚ) : 4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l409_40979


namespace NUMINAMATH_CALUDE_division_problem_l409_40914

theorem division_problem (A : ℕ) (h : 34 = A * 6 + 4) : A = 5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l409_40914


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l409_40954

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_34 : ℚ := 34 / 99

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_34 = 46 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l409_40954


namespace NUMINAMATH_CALUDE_crow_votes_l409_40985

/-- Represents the number of votes for each singer -/
structure Votes where
  rooster : ℕ
  crow : ℕ
  cuckoo : ℕ

/-- Represents the reported vote counts, which may be inaccurate -/
structure ReportedCounts where
  total : ℕ
  roosterCrow : ℕ
  crowCuckoo : ℕ
  cuckooRooster : ℕ

/-- Checks if a reported count is within the error margin of the actual count -/
def isWithinErrorMargin (reported actual : ℕ) : Prop :=
  (reported ≤ actual + 13) ∧ (actual ≤ reported + 13)

/-- The main theorem statement -/
theorem crow_votes (v : Votes) (r : ReportedCounts) : 
  (v.rooster + v.crow + v.cuckoo > 0) →
  isWithinErrorMargin r.total (v.rooster + v.crow + v.cuckoo) →
  isWithinErrorMargin r.roosterCrow (v.rooster + v.crow) →
  isWithinErrorMargin r.crowCuckoo (v.crow + v.cuckoo) →
  isWithinErrorMargin r.cuckooRooster (v.cuckoo + v.rooster) →
  r.total = 59 →
  r.roosterCrow = 15 →
  r.crowCuckoo = 18 →
  r.cuckooRooster = 20 →
  v.crow = 13 := by
  sorry

end NUMINAMATH_CALUDE_crow_votes_l409_40985


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l409_40938

/-- Given a geometric sequence {a_n} with positive terms where a_1, (1/2)a_3, and 2a_2 form an arithmetic sequence, 
    the ratio a_10 / a_8 is equal to 3 + 2√2. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
    (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
    (h_arithmetic : 2 * ((1/2) * a 3) = a 1 + 2 * a 2) :
    a 10 / a 8 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l409_40938


namespace NUMINAMATH_CALUDE_family_trip_eggs_l409_40998

/-- Calculates the total number of boiled eggs prepared for a family trip -/
def total_eggs (num_adults num_girls num_boys : ℕ) (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) : ℕ :=
  num_adults * eggs_per_adult + num_girls * eggs_per_girl + num_boys * (eggs_per_girl + 1)

/-- Theorem stating that the total number of boiled eggs for the given family trip is 36 -/
theorem family_trip_eggs :
  total_eggs 3 7 10 3 1 = 36 := by
  sorry

end NUMINAMATH_CALUDE_family_trip_eggs_l409_40998


namespace NUMINAMATH_CALUDE_parallel_lines_k_l409_40903

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x - 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k : ∃ k : ℝ, 
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * k) * x + 7) ↔ k = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_l409_40903


namespace NUMINAMATH_CALUDE_abc_sum_product_l409_40956

theorem abc_sum_product (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b * c > 0) :
  a * b + b * c + c * a < 0 := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_product_l409_40956


namespace NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l409_40969

theorem product_of_sums_equals_difference_of_powers : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_difference_of_powers_l409_40969


namespace NUMINAMATH_CALUDE_inscribed_square_area_l409_40906

/-- The parabola function -/
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- A point is on the parabola if its y-coordinate equals f(x) -/
def on_parabola (p : ℝ × ℝ) : Prop := p.2 = f p.1

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

/-- A square is inscribed if its top vertices are on the parabola and bottom vertices are on the x-axis -/
def is_inscribed_square (s : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a < b ∧
    s = {(a, 0), (b, 0), (a, b-a), (b, b-a)} ∧
    on_parabola (a, b-a) ∧ on_parabola (b, b-a)

/-- The area of a square with side length s -/
def square_area (s : ℝ) : ℝ := s^2

theorem inscribed_square_area :
  ∀ s : Set (ℝ × ℝ), is_inscribed_square s → ∃ a : ℝ, square_area a = (3 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l409_40906


namespace NUMINAMATH_CALUDE_ben_egg_count_l409_40939

/-- The number of trays Ben was given -/
def num_trays : ℕ := 7

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 10

/-- The total number of eggs Ben examined -/
def total_eggs : ℕ := num_trays * eggs_per_tray

theorem ben_egg_count : total_eggs = 70 := by
  sorry

end NUMINAMATH_CALUDE_ben_egg_count_l409_40939


namespace NUMINAMATH_CALUDE_area_of_ABCM_l409_40927

/-- A 12-sided polygon with specific properties -/
structure TwelveSidedPolygon where
  /-- The length of each side of the polygon -/
  side_length : ℝ
  /-- The property that each two consecutive sides form a right angle -/
  right_angles : Bool

/-- The intersection point of two diagonals in the polygon -/
def IntersectionPoint (p : TwelveSidedPolygon) := Unit

/-- A quadrilateral formed by three vertices of the polygon and the intersection point -/
def Quadrilateral (p : TwelveSidedPolygon) (m : IntersectionPoint p) := Unit

/-- The area of a quadrilateral -/
def area (q : Quadrilateral p m) : ℝ := sorry

/-- Theorem stating the area of quadrilateral ABCM in the given polygon -/
theorem area_of_ABCM (p : TwelveSidedPolygon) (m : IntersectionPoint p) 
  (q : Quadrilateral p m) (h1 : p.side_length = 4) (h2 : p.right_angles = true) : 
  area q = 88 / 5 := by sorry

end NUMINAMATH_CALUDE_area_of_ABCM_l409_40927


namespace NUMINAMATH_CALUDE_largest_number_l409_40931

theorem largest_number : ∀ (a b c d : ℝ), 
  a = -1 → b = 0 → c = 2 → d = Real.sqrt 3 →
  a < b ∧ b < d ∧ d < c :=
fun a b c d ha hb hc hd => by
  sorry

end NUMINAMATH_CALUDE_largest_number_l409_40931


namespace NUMINAMATH_CALUDE_c_alone_time_l409_40978

-- Define the work rates of A, B, and C
variable (A B C : ℚ)

-- Define the conditions from the problem
variable (h1 : A + B = 1 / 15)
variable (h2 : A + B + C = 1 / 12)

-- The theorem to prove
theorem c_alone_time : C = 1 / 60 :=
  sorry

end NUMINAMATH_CALUDE_c_alone_time_l409_40978


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l409_40934

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 5 scoops of 3 basic flavors -/
def ice_cream_flavors : ℕ := stars_and_bars 5 3

theorem ice_cream_flavors_count : ice_cream_flavors = 21 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l409_40934


namespace NUMINAMATH_CALUDE_inequality_reversal_l409_40994

theorem inequality_reversal (x y : ℝ) (h : x > y) : ¬(-x > -y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_reversal_l409_40994


namespace NUMINAMATH_CALUDE_point_on_line_expression_l409_40933

/-- For any point (a,b) on the line y=2x+1, the expression 1-4a+2b equals 3 -/
theorem point_on_line_expression (a b : ℝ) : b = 2 * a + 1 → 1 - 4 * a + 2 * b = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_expression_l409_40933


namespace NUMINAMATH_CALUDE_log_equation_solution_l409_40922

theorem log_equation_solution (x : ℝ) (h : Real.log 729 / Real.log (3 * x) = x) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l409_40922


namespace NUMINAMATH_CALUDE_grammar_club_committee_probability_l409_40982

/-- The number of boys in the Grammar club -/
def num_boys : ℕ := 15

/-- The number of girls in the Grammar club -/
def num_girls : ℕ := 15

/-- The size of the committee to be formed -/
def committee_size : ℕ := 5

/-- The minimum number of boys required in the committee -/
def min_boys : ℕ := 2

/-- The probability of forming a committee with at least 2 boys and at least 1 girl -/
def committee_probability : ℚ := 515 / 581

/-- Theorem stating the probability of forming a committee with the given conditions -/
theorem grammar_club_committee_probability :
  let total_members := num_boys + num_girls
  let valid_committees := (Finset.range (committee_size + 1)).filter (λ k => k ≥ min_boys ∧ k < committee_size)
    |>.sum (λ k => (Nat.choose num_boys k) * (Nat.choose num_girls (committee_size - k)))
  let total_committees := Nat.choose total_members committee_size
  (valid_committees : ℚ) / total_committees = committee_probability := by
  sorry

#check grammar_club_committee_probability

end NUMINAMATH_CALUDE_grammar_club_committee_probability_l409_40982


namespace NUMINAMATH_CALUDE_luther_pancakes_correct_l409_40991

/-- The number of people in Luther's family -/
def family_size : ℕ := 8

/-- The number of additional pancakes needed for everyone to have a second pancake -/
def additional_pancakes : ℕ := 4

/-- The number of pancakes Luther made initially -/
def initial_pancakes : ℕ := 12

/-- Theorem stating that the number of pancakes Luther made initially is correct -/
theorem luther_pancakes_correct :
  initial_pancakes = family_size * 2 - additional_pancakes :=
by sorry

end NUMINAMATH_CALUDE_luther_pancakes_correct_l409_40991


namespace NUMINAMATH_CALUDE_ellipse_point_inside_circle_l409_40919

theorem ellipse_point_inside_circle 
  (a b c : ℝ) 
  (h_ab : a > b) 
  (h_b_pos : b > 0) 
  (h_e : c / a = 1 / 2) 
  (x₁ x₂ : ℝ) 
  (h_roots : x₁ * x₂ = -c / a ∧ x₁ + x₂ = -b / a) : 
  x₁^2 + x₂^2 < 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_point_inside_circle_l409_40919


namespace NUMINAMATH_CALUDE_problem_statement_l409_40918

theorem problem_statement (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l409_40918


namespace NUMINAMATH_CALUDE_model_a_sample_size_l409_40946

/-- Represents the number of cars to be sampled for a given model -/
def sample_size (total_cars : ℕ) (model_cars : ℕ) (total_sample : ℕ) : ℕ :=
  (model_cars * total_sample) / total_cars

/-- Proves that the sample size for Model A is 6 -/
theorem model_a_sample_size :
  let total_cars := 1200 + 6000 + 2000
  let model_a_cars := 1200
  let total_sample := 46
  sample_size total_cars model_a_cars total_sample = 6 := by
  sorry

#eval sample_size (1200 + 6000 + 2000) 1200 46

end NUMINAMATH_CALUDE_model_a_sample_size_l409_40946


namespace NUMINAMATH_CALUDE_factorization_problems_l409_40950

theorem factorization_problems (x y : ℝ) : 
  (x^2 - 6*x + 9 = (x - 3)^2) ∧ 
  (x^2*(y - 2) - 4*(y - 2) = (y - 2)*(x + 2)*(x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l409_40950


namespace NUMINAMATH_CALUDE_samuel_spent_one_fifth_l409_40992

theorem samuel_spent_one_fifth (total : ℕ) (samuel_initial : ℚ) (samuel_left : ℕ) : 
  total = 240 →
  samuel_initial = 3/4 * total →
  samuel_left = 132 →
  (samuel_initial - samuel_left : ℚ) / total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_samuel_spent_one_fifth_l409_40992


namespace NUMINAMATH_CALUDE_line_through_points_2m_plus_3b_l409_40997

/-- Given a line passing through the points (-1, 1/2) and (2, -3/2), 
    prove that 2m+3b = -11/6 when the line is expressed as y = mx + b -/
theorem line_through_points_2m_plus_3b (m b : ℚ) : 
  (1/2 : ℚ) = m * (-1) + b →
  (-3/2 : ℚ) = m * 2 + b →
  2 * m + 3 * b = -11/6 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_2m_plus_3b_l409_40997


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l409_40907

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l409_40907


namespace NUMINAMATH_CALUDE_product_of_cosines_l409_40999

theorem product_of_cosines : 
  (1 + Real.cos (π / 12)) * (1 + Real.cos (5 * π / 12)) * 
  (1 + Real.cos (7 * π / 12)) * (1 + Real.cos (11 * π / 12)) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_product_of_cosines_l409_40999


namespace NUMINAMATH_CALUDE_twentieth_meeting_point_theorem_ant_meeting_theorem_l409_40974

/-- Represents the meeting point of two ants -/
structure MeetingPoint where
  distance : ℝ
  meeting_number : ℕ

/-- Calculates the meeting point of two ants -/
def calculate_meeting_point (total_distance : ℝ) (speed_ratio : ℝ) (meeting_number : ℕ) : MeetingPoint :=
  { distance := 2,  -- The actual calculation is omitted
    meeting_number := meeting_number }

/-- The theorem stating the 20th meeting point of the ants -/
theorem twentieth_meeting_point_theorem (total_distance : ℝ) (speed_ratio : ℝ) :
  (calculate_meeting_point total_distance speed_ratio 20).distance = 2 :=
by
  sorry

#check twentieth_meeting_point_theorem

/-- Main theorem about the ant problem -/
theorem ant_meeting_theorem :
  ∃ (total_distance : ℝ) (speed_ratio : ℝ),
    total_distance = 6 ∧ speed_ratio = 2.5 ∧
    (calculate_meeting_point total_distance speed_ratio 20).distance = 2 :=
by
  sorry

#check ant_meeting_theorem

end NUMINAMATH_CALUDE_twentieth_meeting_point_theorem_ant_meeting_theorem_l409_40974


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l409_40937

theorem prime_sum_theorem (a b c : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → 
  b + c = 13 → c^2 - a^2 = 72 → 
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l409_40937


namespace NUMINAMATH_CALUDE_find_N_l409_40995

theorem find_N : ∃ N : ℤ, (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l409_40995


namespace NUMINAMATH_CALUDE_exists_72_degree_angle_l409_40970

/-- Represents a hexagon with angles in arithmetic progression -/
structure ArithmeticHexagon where
  a : ℝ  -- First angle in the progression
  d : ℝ  -- Common difference

/-- The sum of angles in a hexagon is 720° -/
axiom hexagon_angle_sum (h : ArithmeticHexagon) : 
  h.a + (h.a + h.d) + (h.a + 2*h.d) + (h.a + 3*h.d) + (h.a + 4*h.d) + (h.a + 5*h.d) = 720

/-- Theorem: There exists a hexagon with angles in arithmetic progression that has a 72° angle -/
theorem exists_72_degree_angle : ∃ h : ArithmeticHexagon, 
  h.a = 72 ∨ (h.a + h.d) = 72 ∨ (h.a + 2*h.d) = 72 ∨ 
  (h.a + 3*h.d) = 72 ∨ (h.a + 4*h.d) = 72 ∨ (h.a + 5*h.d) = 72 :=
sorry

end NUMINAMATH_CALUDE_exists_72_degree_angle_l409_40970


namespace NUMINAMATH_CALUDE_consecutive_cube_divisible_l409_40940

theorem consecutive_cube_divisible (k : ℕ+) :
  ∃ n : ℤ, ∀ j : ℕ, j ∈ Finset.range k → ∃ p : ℕ, Nat.Prime p ∧ p > 1 ∧ (n + j + 1 : ℤ) % (p^3 : ℤ) = 0 :=
sorry

end NUMINAMATH_CALUDE_consecutive_cube_divisible_l409_40940


namespace NUMINAMATH_CALUDE_linda_furniture_spending_l409_40984

/-- 
Given that Linda's original savings were $800 and she spent the remaining amount 
after furniture purchase on a TV that cost $200, prove that the fraction of 
savings Linda spent on furniture is 3/4.
-/
theorem linda_furniture_spending (
  original_savings : ℚ) 
  (tv_cost : ℚ) 
  (h1 : original_savings = 800)
  (h2 : tv_cost = 200) : 
  (original_savings - tv_cost) / original_savings = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_linda_furniture_spending_l409_40984


namespace NUMINAMATH_CALUDE_polygon_has_13_sides_l409_40909

/-- A polygon has n sides. The number of diagonals is equal to 5 times the number of sides. -/
def polygon_diagonals (n : ℕ) : Prop :=
  n * (n - 3) = 5 * n

/-- The polygon satisfying the given condition has 13 sides. -/
theorem polygon_has_13_sides : 
  ∃ (n : ℕ), polygon_diagonals n ∧ n = 13 :=
sorry

end NUMINAMATH_CALUDE_polygon_has_13_sides_l409_40909


namespace NUMINAMATH_CALUDE_probability_of_correct_number_l409_40911

def first_three_options : ℕ := 3

def last_five_digits : ℕ := 5
def repeating_digits : ℕ := 2

def total_combinations : ℕ := first_three_options * (Nat.factorial last_five_digits / Nat.factorial repeating_digits)

theorem probability_of_correct_number :
  (1 : ℚ) / total_combinations = 1 / 180 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_correct_number_l409_40911


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_1_l409_40932

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * |x - 1| + |x - a|

-- Statement 1
theorem solution_set_when_a_is_2 :
  let f2 := f 2
  {x : ℝ | f2 x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 8/3} := by sorry

-- Statement 2
theorem range_of_a_when_f_geq_1 :
  {a : ℝ | a > 0 ∧ ∀ x, f a x ≥ 1} = {a : ℝ | a ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_1_l409_40932


namespace NUMINAMATH_CALUDE_infinite_special_numbers_l409_40949

theorem infinite_special_numbers (k : ℕ) :
  let n := 250 * 3^(6*k)
  ∃ (a b c d : ℕ), 
    n = a^2 + b^2 ∧ 
    n = c^3 + d^3 ∧ 
    ¬∃ (x y : ℕ), n = x^6 + y^6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_special_numbers_l409_40949


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l409_40910

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 7 + a 13 = 20 →
  a 9 + a 10 + a 11 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l409_40910


namespace NUMINAMATH_CALUDE_journey_fraction_is_one_fourth_l409_40987

/-- Represents the journey from Petya's home to school -/
structure Journey where
  totalTime : ℕ
  timeBeforeBell : ℕ
  timeLateIfReturn : ℕ

/-- Calculates the fraction of the journey completed when Petya remembered the pen -/
def fractionCompleted (j : Journey) : ℚ :=
  let detourTime := j.timeBeforeBell + j.timeLateIfReturn
  let timeToRememberedPoint := detourTime / 2
  timeToRememberedPoint / j.totalTime

/-- Theorem stating that the fraction of the journey completed when Petya remembered the pen is 1/4 -/
theorem journey_fraction_is_one_fourth (j : Journey) 
  (h1 : j.totalTime = 20)
  (h2 : j.timeBeforeBell = 3)
  (h3 : j.timeLateIfReturn = 7) : 
  fractionCompleted j = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_journey_fraction_is_one_fourth_l409_40987


namespace NUMINAMATH_CALUDE_factor_expression_l409_40900

theorem factor_expression (c : ℝ) : 180 * c^2 + 36 * c = 36 * c * (5 * c + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l409_40900


namespace NUMINAMATH_CALUDE_unique_right_triangle_l409_40980

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if a triangle is right-angled -/
def Triangle.isRight (t : Triangle) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2 ∨ t.b ^ 2 + t.c ^ 2 = t.a ^ 2 ∨ t.c ^ 2 + t.a ^ 2 = t.b ^ 2

/-- The main theorem -/
theorem unique_right_triangle :
  ∃! k : ℕ+, 
    (∃ t : Triangle, t.a = 8 ∧ t.b = 12 ∧ t.c = k) ∧ 
    (∃ t : Triangle, t.a + t.b + t.c = 30 ∧ t.a = 8 ∧ t.b = 12 ∧ t.c = k) ∧
    (∃ t : Triangle, t.a = 8 ∧ t.b = 12 ∧ t.c = k ∧ t.isRight) :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_l409_40980


namespace NUMINAMATH_CALUDE_angle_A_measure_l409_40947

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_of_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem statement
theorem angle_A_measure (t : Triangle) (h1 : t.C = 3 * t.B) (h2 : t.B = 15) : t.A = 120 := by
  sorry


end NUMINAMATH_CALUDE_angle_A_measure_l409_40947


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l409_40990

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : d ≠ 0
  h2 : ∀ n : ℕ, a (n + 1) = a n + d

/-- Condition for terms forming a geometric sequence -/
def isGeometric (s : ArithmeticSequence) : Prop :=
  (s.a 3)^2 = (s.a 1) * (s.a 9)

theorem arithmetic_geometric_ratio
  (s : ArithmeticSequence)
  (h : isGeometric s) :
  (s.a 1 + s.a 3 + s.a 9) / (s.a 2 + s.a 4 + s.a 10) = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l409_40990


namespace NUMINAMATH_CALUDE_power_of_power_l409_40975

theorem power_of_power (x : ℝ) : (x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l409_40975


namespace NUMINAMATH_CALUDE_vector_calculation_l409_40905

def v1 : Fin 2 → ℝ := ![3, -6]
def v2 : Fin 2 → ℝ := ![-1, 5]
def v3 : Fin 2 → ℝ := ![5, -20]

theorem vector_calculation :
  (2 • v1 + 4 • v2 - v3) = ![(-3 : ℝ), 28] := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l409_40905


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l409_40965

/-- Given a circular arrangement of students, if the 8th student is directly opposite the 33rd student, then the total number of students is 52. -/
theorem circular_arrangement_students (n : ℕ) : 
  (∃ (a b : ℕ), a = 8 ∧ b = 33 ∧ a < b ∧ b - a = n - (b - a)) → n = 52 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l409_40965


namespace NUMINAMATH_CALUDE_sequence_integer_count_l409_40993

def sequence_term (n : ℕ) : ℚ :=
  24300 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬ is_integer (sequence_term k)) →
  (∃! (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬ is_integer (sequence_term k)) ∧
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    ¬ is_integer (sequence_term k) ∧
    k = 3) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l409_40993


namespace NUMINAMATH_CALUDE_initial_cherries_l409_40943

theorem initial_cherries (eaten : ℕ) (left : ℕ) (h1 : eaten = 25) (h2 : left = 42) :
  eaten + left = 67 := by
  sorry

end NUMINAMATH_CALUDE_initial_cherries_l409_40943


namespace NUMINAMATH_CALUDE_alice_age_problem_l409_40926

theorem alice_age_problem :
  ∃! x : ℕ+, 
    (∃ n : ℕ+, (x : ℤ) - 4 = n^2) ∧ 
    (∃ m : ℕ+, (x : ℤ) + 2 = m^3) ∧ 
    x = 58 := by
  sorry

end NUMINAMATH_CALUDE_alice_age_problem_l409_40926


namespace NUMINAMATH_CALUDE_transport_speed_problem_l409_40976

/-- Proves that given two transports traveling in opposite directions, with one traveling at 60 mph,
    if they are 348 miles apart after 2.71875 hours, then the speed of the second transport is 68 mph. -/
theorem transport_speed_problem (speed_a speed_b : ℝ) (time : ℝ) (distance : ℝ) : 
  speed_a = 60 →
  time = 2.71875 →
  distance = 348 →
  (speed_a + speed_b) * time = distance →
  speed_b = 68 := by
  sorry

#check transport_speed_problem

end NUMINAMATH_CALUDE_transport_speed_problem_l409_40976


namespace NUMINAMATH_CALUDE_cricket_average_l409_40925

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 20 → 
  next_runs = 120 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 36 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l409_40925


namespace NUMINAMATH_CALUDE_patrol_impossibility_l409_40964

/-- Represents the number of people in the group -/
def n : ℕ := 100

/-- Represents the number of people on duty each evening -/
def k : ℕ := 3

/-- Represents the total number of possible pairs of people -/
def totalPairs : ℕ := n.choose 2

/-- Represents the number of pairs formed each evening -/
def pairsPerEvening : ℕ := k.choose 2

theorem patrol_impossibility : ¬ ∃ (m : ℕ), m * pairsPerEvening = totalPairs ∧ 
  ∃ (f : Fin n → Fin m → Bool), 
    (∀ i j, i ≠ j → (∃! t, f i t ∧ f j t)) ∧
    (∀ t, ∃! (s : Fin k → Fin n), (∀ i, f (s i) t)) :=
sorry

end NUMINAMATH_CALUDE_patrol_impossibility_l409_40964


namespace NUMINAMATH_CALUDE_apple_purchase_theorem_l409_40958

/-- Represents the cost in cents for a pack of apples --/
structure ApplePack where
  count : ℕ
  cost : ℕ

/-- Represents a purchase of apple packs --/
structure Purchase where
  pack : ApplePack
  quantity : ℕ

def total_apples (purchases : List Purchase) : ℕ :=
  purchases.foldl (fun acc p => acc + p.pack.count * p.quantity) 0

def total_cost (purchases : List Purchase) : ℕ :=
  purchases.foldl (fun acc p => acc + p.pack.cost * p.quantity) 0

def average_cost (purchases : List Purchase) : ℚ :=
  (total_cost purchases : ℚ) / (total_apples purchases : ℚ)

theorem apple_purchase_theorem (scheme1 scheme2 : ApplePack) 
  (purchase1 purchase2 : Purchase) : 
  scheme1.count = 4 → 
  scheme1.cost = 15 → 
  scheme2.count = 7 → 
  scheme2.cost = 28 → 
  purchase1.pack = scheme2 → 
  purchase1.quantity = 4 → 
  purchase2.pack = scheme1 → 
  purchase2.quantity = 2 → 
  total_cost [purchase1, purchase2] = 142 ∧ 
  average_cost [purchase1, purchase2] = 5.0714 := by
  sorry

end NUMINAMATH_CALUDE_apple_purchase_theorem_l409_40958


namespace NUMINAMATH_CALUDE_xiaohong_total_score_l409_40960

/-- Calculates the total score based on midterm and final exam scores -/
def total_score (midterm_weight : ℝ) (final_weight : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  midterm_weight * midterm_score + final_weight * final_score

theorem xiaohong_total_score :
  let midterm_weight : ℝ := 0.4
  let final_weight : ℝ := 0.6
  let midterm_score : ℝ := 80
  let final_score : ℝ := 90
  total_score midterm_weight final_weight midterm_score final_score = 86 := by
  sorry

#eval total_score 0.4 0.6 80 90

end NUMINAMATH_CALUDE_xiaohong_total_score_l409_40960


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l409_40924

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 * I - 5) / (2 + I)
  Complex.im z = 11 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l409_40924


namespace NUMINAMATH_CALUDE_open_box_volume_l409_40953

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48) 
  (h2 : sheet_width = 36) 
  (h3 : cut_length = 8) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5120 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l409_40953


namespace NUMINAMATH_CALUDE_triangle_properties_l409_40916

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.c^2 - t.b^2 = t.a * t.c) 
  (h2 : t.c = 3 * t.a) : 
  t.B = π/3 ∧ Real.sin t.A = Real.sqrt 21 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l409_40916


namespace NUMINAMATH_CALUDE_fruit_cost_calculation_l409_40935

/-- Calculates the total cost of fruits given the specified conditions --/
theorem fruit_cost_calculation (apple_price : ℚ) (orange_price : ℚ) (total_fruits : ℕ) :
  apple_price = 1 →
  orange_price = 1/2 →
  total_fruits = 36 →
  (∃ (n : ℕ), total_fruits = 3 * n) →
  (∃ (watermelon_price : ℚ), 4 * apple_price = watermelon_price) →
  ∃ (total_cost : ℚ), total_cost = 66 :=
by
  sorry

#check fruit_cost_calculation

end NUMINAMATH_CALUDE_fruit_cost_calculation_l409_40935


namespace NUMINAMATH_CALUDE_worker_travel_time_l409_40902

theorem worker_travel_time (T : ℝ) 
  (h1 : T > 0) 
  (h2 : (3/4 : ℝ) * T * (T + 12) = T * T) : 
  T = 36 := by
sorry

end NUMINAMATH_CALUDE_worker_travel_time_l409_40902


namespace NUMINAMATH_CALUDE_bob_dog_cost_l409_40941

/-- The cost of each show dog given the number of dogs, number of puppies,
    price per puppy, and total profit -/
def cost_per_dog (num_dogs : ℕ) (num_puppies : ℕ) (price_per_puppy : ℕ) (total_profit : ℕ) : ℕ :=
  (num_puppies * price_per_puppy - total_profit) / num_dogs

theorem bob_dog_cost :
  cost_per_dog 2 6 350 1600 = 250 := by
  sorry

end NUMINAMATH_CALUDE_bob_dog_cost_l409_40941


namespace NUMINAMATH_CALUDE_power_of_two_equation_l409_40917

theorem power_of_two_equation (N : ℕ) : (32^5 * 16^4) / 8^7 = 2^N → N = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l409_40917


namespace NUMINAMATH_CALUDE_vector_decomposition_l409_40989

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![(-15), 5, 6]
def p : Fin 3 → ℝ := ![0, 5, 1]
def q : Fin 3 → ℝ := ![3, 2, (-1)]
def r : Fin 3 → ℝ := ![(-1), 1, 0]

/-- The theorem to be proved -/
theorem vector_decomposition :
  x = (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l409_40989


namespace NUMINAMATH_CALUDE_equal_sum_sequence_sixth_term_l409_40996

/-- An Equal Sum Sequence is a sequence where the sum of each term and its next term is always the same constant. -/
def EqualSumSequence (a : ℕ → ℝ) (c : ℝ) :=
  ∀ n, a n + a (n + 1) = c

theorem equal_sum_sequence_sixth_term
  (a : ℕ → ℝ)
  (h1 : EqualSumSequence a 5)
  (h2 : a 1 = 2) :
  a 6 = 3 := by
sorry

end NUMINAMATH_CALUDE_equal_sum_sequence_sixth_term_l409_40996


namespace NUMINAMATH_CALUDE_square_sum_equation_l409_40971

theorem square_sum_equation (n m : ℕ) (h : n ^ 2 = (Finset.range (m - 99)).sum (λ i => i + 100)) : n + m = 497 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_l409_40971


namespace NUMINAMATH_CALUDE_x_value_proof_l409_40986

theorem x_value_proof : ∃ x : ℝ, 
  3.5 * ((3.6 * 0.48 * x) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l409_40986


namespace NUMINAMATH_CALUDE_vector_parallelism_l409_40928

theorem vector_parallelism (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![-1, m]
  let c : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • c) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_vector_parallelism_l409_40928


namespace NUMINAMATH_CALUDE_percent_of_125_l409_40959

theorem percent_of_125 : ∃ p : ℚ, p * 125 / 100 = 70 ∧ p = 56 := by sorry

end NUMINAMATH_CALUDE_percent_of_125_l409_40959


namespace NUMINAMATH_CALUDE_figure_50_squares_initial_values_correct_l409_40951

/-- Represents the number of nonoverlapping unit squares in the nth figure -/
def g (n : ℕ) : ℕ := 2 * n^2 + 5 * n + 2

/-- The theorem states that the 50th term of the sequence equals 5252 -/
theorem figure_50_squares : g 50 = 5252 := by
  sorry

/-- Verifies that the function g matches the given initial values -/
theorem initial_values_correct :
  g 0 = 2 ∧ g 1 = 9 ∧ g 2 = 20 ∧ g 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_figure_50_squares_initial_values_correct_l409_40951
