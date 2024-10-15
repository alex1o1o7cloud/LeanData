import Mathlib

namespace NUMINAMATH_CALUDE_minimal_coloring_exists_l894_89440

/-- Define the function f for a given set M and subset A -/
def f (M : Finset ℕ) (A : Finset ℕ) : Finset ℕ :=
  M.filter (fun x => (A.filter (fun a => x % a = 0)).card % 2 = 1)

/-- The main theorem -/
theorem minimal_coloring_exists :
  ∀ (M : Finset ℕ), M.card = 2017 →
  ∃ (c : Finset ℕ → Bool),
    ∀ (A : Finset ℕ), A ⊆ M →
      A ≠ f M A → c A ≠ c (f M A) :=
by sorry

end NUMINAMATH_CALUDE_minimal_coloring_exists_l894_89440


namespace NUMINAMATH_CALUDE_second_guide_children_l894_89485

/-- Given information about zoo guides and children -/
structure ZooTour where
  total_children : ℕ
  first_guide_children : ℕ

/-- Theorem: The second guide spoke to 25 children -/
theorem second_guide_children (tour : ZooTour) 
  (h1 : tour.total_children = 44)
  (h2 : tour.first_guide_children = 19) :
  tour.total_children - tour.first_guide_children = 25 := by
  sorry

#eval 44 - 19  -- Expected output: 25

end NUMINAMATH_CALUDE_second_guide_children_l894_89485


namespace NUMINAMATH_CALUDE_earrings_price_decrease_l894_89495

/-- Given a pair of earrings with the following properties:
  - Purchase price: $240
  - Original markup: 25% of the selling price
  - Gross profit after price decrease: $16
  Prove that the percentage decrease in the selling price is 5% -/
theorem earrings_price_decrease (purchase_price : ℝ) (markup_percentage : ℝ) (gross_profit : ℝ) :
  purchase_price = 240 →
  markup_percentage = 0.25 →
  gross_profit = 16 →
  let original_selling_price := purchase_price / (1 - markup_percentage)
  let new_selling_price := original_selling_price - gross_profit
  let price_decrease := original_selling_price - new_selling_price
  let percentage_decrease := price_decrease / original_selling_price * 100
  percentage_decrease = 5 := by
  sorry

end NUMINAMATH_CALUDE_earrings_price_decrease_l894_89495


namespace NUMINAMATH_CALUDE_unique_prime_generating_x_l894_89454

theorem unique_prime_generating_x (x : ℕ+) 
  (h : Nat.Prime (x^5 + x + 1)) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_generating_x_l894_89454


namespace NUMINAMATH_CALUDE_complex_coordinate_proof_l894_89429

theorem complex_coordinate_proof (z : ℂ) : (z - 2*I) * (1 + I) = I → z = 1/2 + 5/2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinate_proof_l894_89429


namespace NUMINAMATH_CALUDE_number_of_lineups_l894_89467

def team_size : ℕ := 15
def lineup_size : ℕ := 5

def cannot_play_together : Prop := true
def at_least_one_must_play : Prop := true

theorem number_of_lineups : 
  ∃ (n : ℕ), n = Nat.choose (team_size - 2) (lineup_size - 1) * 2 + 
             Nat.choose (team_size - 3) (lineup_size - 2) ∧
  n = 1210 := by
  sorry

end NUMINAMATH_CALUDE_number_of_lineups_l894_89467


namespace NUMINAMATH_CALUDE_symmetry_line_l894_89491

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 4 = 0

-- Define the line
def line_l (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem symmetry_line :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle1 x1 y1 → circle2 x2 y2 →
  ∃ (x y : ℝ), line_l x y ∧
  (x = (x1 + x2) / 2) ∧ (y = (y1 + y2) / 2) ∧
  ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2) :=
sorry

end NUMINAMATH_CALUDE_symmetry_line_l894_89491


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l894_89425

/-- Calculates the total revenue from selling various reading materials -/
theorem total_revenue_calculation (magazines newspapers books pamphlets : ℕ) 
  (magazine_price newspaper_price book_price pamphlet_price : ℚ) : 
  magazines = 425 → 
  newspapers = 275 → 
  books = 150 → 
  pamphlets = 75 → 
  magazine_price = 5/2 → 
  newspaper_price = 3/2 → 
  book_price = 5 → 
  pamphlet_price = 1/2 → 
  (magazines : ℚ) * magazine_price + 
  (newspapers : ℚ) * newspaper_price + 
  (books : ℚ) * book_price + 
  (pamphlets : ℚ) * pamphlet_price = 2262.5 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l894_89425


namespace NUMINAMATH_CALUDE_point_on_line_through_two_points_l894_89410

/-- A point lies on a line if it satisfies the line equation --/
def point_on_line (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

/-- The theorem statement --/
theorem point_on_line_through_two_points :
  point_on_line 1 2 5 10 3 6 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_through_two_points_l894_89410


namespace NUMINAMATH_CALUDE_mechanism_composition_l894_89479

/-- Represents a mechanism with small and large parts. -/
structure Mechanism where
  total_parts : ℕ
  small_parts : ℕ
  large_parts : ℕ
  total_eq : total_parts = small_parts + large_parts

/-- Property: Among any 12 parts, there is at least one small part. -/
def has_small_in_12 (m : Mechanism) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 12 → (∃ (x : ℕ), x ∈ subset ∧ x ≤ m.small_parts)

/-- Property: Among any 15 parts, there is at least one large part. -/
def has_large_in_15 (m : Mechanism) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 15 → (∃ (x : ℕ), x ∈ subset ∧ x > m.small_parts)

/-- Main theorem: If a mechanism satisfies the given conditions, it has 11 large parts and 14 small parts. -/
theorem mechanism_composition (m : Mechanism) 
    (h_total : m.total_parts = 25)
    (h_small : has_small_in_12 m)
    (h_large : has_large_in_15 m) : 
    m.large_parts = 11 ∧ m.small_parts = 14 := by
  sorry


end NUMINAMATH_CALUDE_mechanism_composition_l894_89479


namespace NUMINAMATH_CALUDE_arithmetic_sequence_count_l894_89452

theorem arithmetic_sequence_count :
  let a : ℤ := -5  -- First term
  let l : ℤ := 85  -- Last term
  let d : ℤ := 5   -- Common difference
  (l - a) / d + 1 = 19
  :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_count_l894_89452


namespace NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l894_89420

theorem binomial_coefficient_modulo_prime (p n q : ℕ) : 
  Prime p → 
  0 < n → 
  0 < q → 
  (n ≠ q * (p - 1) → Nat.choose n (p - 1) % p = 0) ∧
  (n = q * (p - 1) → Nat.choose n (p - 1) % p = 1) :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_modulo_prime_l894_89420


namespace NUMINAMATH_CALUDE_cookie_problem_l894_89412

/-- The number of guests who did not come to Brenda's mother's cookie event -/
def guests_not_came (total_guests : ℕ) (total_cookies : ℕ) (cookies_per_guest : ℕ) : ℕ :=
  total_guests - (total_cookies / cookies_per_guest)

theorem cookie_problem :
  guests_not_came 10 18 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l894_89412


namespace NUMINAMATH_CALUDE_equation_roots_exist_l894_89416

/-- Proves that the equation x|x| + px + q = 0 can have real roots even when p^2 - 4q < 0 -/
theorem equation_roots_exist (p q : ℝ) (h : p^2 - 4*q < 0) : 
  ∃ x : ℝ, x * |x| + p * x + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_exist_l894_89416


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l894_89403

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := 5*(x^5 - 3*x^4 + 2*x^3) - 6*(x^5 + x^3 + 1) + 2*(3*x^5 - x^4 + x^2)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = 5 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l894_89403


namespace NUMINAMATH_CALUDE_water_breadth_in_cistern_l894_89435

/-- Calculates the breadth of water in a cistern given its dimensions and wet surface area -/
theorem water_breadth_in_cistern (length width wet_surface_area : ℝ) :
  length = 9 →
  width = 6 →
  wet_surface_area = 121.5 →
  ∃ (breadth : ℝ),
    breadth = 2.25 ∧
    wet_surface_area = length * width + 2 * length * breadth + 2 * width * breadth :=
by sorry

end NUMINAMATH_CALUDE_water_breadth_in_cistern_l894_89435


namespace NUMINAMATH_CALUDE_triangle_ABC_proof_l894_89406

theorem triangle_ABC_proof (A B C : Real) (a b c : Real) :
  -- Conditions
  A + B + C = π →
  2 * Real.sin (B + C) ^ 2 - 3 * Real.cos A = 0 →
  B = π / 4 →
  a = 2 * Real.sqrt 3 →
  -- Conclusions
  A = π / 3 ∧ c = Real.sqrt 6 + Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ABC_proof_l894_89406


namespace NUMINAMATH_CALUDE_marcia_project_time_l894_89401

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours Marcia spent on her science project -/
def hours_spent : ℕ := 5

/-- The total number of minutes Marcia spent on her science project -/
def total_minutes : ℕ := hours_spent * minutes_per_hour

theorem marcia_project_time : total_minutes = 300 := by
  sorry

end NUMINAMATH_CALUDE_marcia_project_time_l894_89401


namespace NUMINAMATH_CALUDE_dinner_bill_friends_l894_89458

theorem dinner_bill_friends (total_bill : ℝ) (silas_payment : ℝ) (one_friend_payment : ℝ) : 
  total_bill = 150 →
  silas_payment = total_bill / 2 →
  one_friend_payment = 18 →
  ∃ (num_friends : ℕ),
    num_friends = 6 ∧
    (num_friends - 1) * one_friend_payment = (total_bill - silas_payment) * 1.1 :=
by sorry

end NUMINAMATH_CALUDE_dinner_bill_friends_l894_89458


namespace NUMINAMATH_CALUDE_sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three_l894_89422

theorem sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three :
  Real.sqrt 9 - 2^(0 : ℕ) + |(-1 : ℝ)| = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_nine_minus_two_power_zero_plus_abs_negative_one_equals_three_l894_89422


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_one_l894_89450

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_at_point_one :
  f 1 = 3 ∧ f' 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_one_l894_89450


namespace NUMINAMATH_CALUDE_jerry_collection_cost_l894_89477

/-- The amount of money Jerry needs to finish his action figure collection. -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem stating the amount Jerry needs to finish his collection. -/
theorem jerry_collection_cost :
  money_needed 7 25 12 = 216 := by
  sorry

end NUMINAMATH_CALUDE_jerry_collection_cost_l894_89477


namespace NUMINAMATH_CALUDE_select_cards_probability_l894_89476

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define the number of cards for Alex's name
def alex_cards : ℕ := 4

-- Define the number of cards for Jamie's name
def jamie_cards : ℕ := 8

-- Define the number of cards to be selected
def selected_cards : ℕ := 3

-- Define the probability of selecting 2 cards from Alex's name and 1 from Jamie's name
def probability : ℚ := 12 / 55

-- Theorem statement
theorem select_cards_probability :
  (Nat.choose alex_cards 2 * Nat.choose jamie_cards 1) / Nat.choose total_cards selected_cards = probability :=
sorry

end NUMINAMATH_CALUDE_select_cards_probability_l894_89476


namespace NUMINAMATH_CALUDE_solve_factorial_equation_l894_89475

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem solve_factorial_equation : ∃ n : ℕ, n * factorial n + factorial n = 720 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_factorial_equation_l894_89475


namespace NUMINAMATH_CALUDE_base4_21012_equals_582_l894_89432

/-- Converts a base 4 digit to its base 10 equivalent -/
def base4_digit_to_base10 (d : Nat) : Nat :=
  if d < 4 then d else 0

/-- Represents the base 4 number 21012 -/
def base4_number : List Nat := [2, 1, 0, 1, 2]

/-- Converts a list of base 4 digits to a base 10 number -/
def base4_to_base10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base4_digit_to_base10 d * (4 ^ (digits.length - 1 - i))) 0

theorem base4_21012_equals_582 :
  base4_to_base10 base4_number = 582 := by
  sorry

end NUMINAMATH_CALUDE_base4_21012_equals_582_l894_89432


namespace NUMINAMATH_CALUDE_fraction_product_equals_27_l894_89449

theorem fraction_product_equals_27 : 
  (1 : ℚ) / 3 * 9 / 1 * 1 / 27 * 81 / 1 * 1 / 243 * 729 / 1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equals_27_l894_89449


namespace NUMINAMATH_CALUDE_grunters_win_probability_l894_89493

/-- The probability of a team winning a single game -/
def win_probability : ℚ := 4/5

/-- The number of games in the series -/
def num_games : ℕ := 5

/-- The probability of winning all games in the series -/
def win_all_probability : ℚ := (4/5)^5

theorem grunters_win_probability :
  win_all_probability = 1024/3125 := by
  sorry

end NUMINAMATH_CALUDE_grunters_win_probability_l894_89493


namespace NUMINAMATH_CALUDE_time_to_write_117639_l894_89417

def digits_count (n : ℕ) : ℕ := 
  if n < 10 then 1
  else if n < 100 then 2
  else if n < 1000 then 3
  else if n < 10000 then 4
  else if n < 100000 then 5
  else 6

def total_digits (n : ℕ) : ℕ := 
  (List.range n).map digits_count |>.sum

def time_to_write (n : ℕ) (digits_per_minute : ℕ) : ℕ := 
  (total_digits n + digits_per_minute - 1) / digits_per_minute

theorem time_to_write_117639 : 
  time_to_write 117639 93 = 4 * 24 * 60 + 10 * 60 + 34 := by sorry

end NUMINAMATH_CALUDE_time_to_write_117639_l894_89417


namespace NUMINAMATH_CALUDE_grade_A_students_over_three_years_l894_89408

theorem grade_A_students_over_three_years 
  (total : ℕ) 
  (first_year : ℕ) 
  (growth_rate : ℝ) 
  (h1 : total = 728)
  (h2 : first_year = 200)
  (h3 : first_year + first_year * (1 + growth_rate) + first_year * (1 + growth_rate)^2 = total) :
  first_year + first_year * (1 + growth_rate) + first_year * (1 + growth_rate)^2 = 728 := by
sorry

end NUMINAMATH_CALUDE_grade_A_students_over_three_years_l894_89408


namespace NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l894_89402

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- A point (x, y) is on the parabola if it satisfies the equation -/
def on_parabola (x y : ℝ) : Prop := x = parabola_equation y

/-- An x-intercept is a point on the parabola where y = 0 -/
def is_x_intercept (x : ℝ) : Prop := on_parabola x 0

/-- The theorem stating that the parabola has exactly one x-intercept -/
theorem parabola_has_one_x_intercept : ∃! x : ℝ, is_x_intercept x := by sorry

end NUMINAMATH_CALUDE_parabola_has_one_x_intercept_l894_89402


namespace NUMINAMATH_CALUDE_semicircle_perimeter_equilateral_triangle_l894_89456

/-- The perimeter of a region formed by three semicircular arcs,
    each constructed on a side of an equilateral triangle with side length 1,
    is equal to 3π/2. -/
theorem semicircle_perimeter_equilateral_triangle :
  let triangle_side_length : ℝ := 1
  let semicircle_radius : ℝ := triangle_side_length / 2
  let num_sides : ℕ := 3
  let perimeter : ℝ := num_sides * (π * semicircle_radius)
  perimeter = 3 * π / 2 := by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_equilateral_triangle_l894_89456


namespace NUMINAMATH_CALUDE_percentage_problem_l894_89444

theorem percentage_problem (x : ℝ) (h : x = 300) : 
  ∃ P : ℝ, P * x / 100 = x / 3 + 110 := by sorry

end NUMINAMATH_CALUDE_percentage_problem_l894_89444


namespace NUMINAMATH_CALUDE_gcd_of_consecutive_odd_terms_l894_89407

theorem gcd_of_consecutive_odd_terms (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  Nat.gcd ((n + 1) * (n + 3) * (n + 7) * (n + 9)) 15 = 15 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_consecutive_odd_terms_l894_89407


namespace NUMINAMATH_CALUDE_max_point_difference_is_n_l894_89484

/-- Represents a hockey tournament with n teams -/
structure HockeyTournament where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- The maximum point difference between consecutively ranked teams -/
def maxPointDifference (t : HockeyTournament) : ℕ := t.n

/-- Theorem: The maximum point difference between consecutively ranked teams is n -/
theorem max_point_difference_is_n (t : HockeyTournament) : 
  maxPointDifference t = t.n := by sorry

end NUMINAMATH_CALUDE_max_point_difference_is_n_l894_89484


namespace NUMINAMATH_CALUDE_odds_against_third_horse_l894_89469

/-- Represents the probability of a horse winning a race -/
def probability (p q : ℚ) : ℚ := q / (p + q)

/-- Given three horses in a race with no ties, calculates the odds against the third horse winning -/
theorem odds_against_third_horse 
  (prob_x prob_y : ℚ) 
  (hx : prob_x = probability 3 1) 
  (hy : prob_y = probability 2 3) 
  (h_sum : prob_x + prob_y < 1) :
  ∃ (p q : ℚ), p / q = 17 / 3 ∧ probability p q = 1 - prob_x - prob_y := by
sorry


end NUMINAMATH_CALUDE_odds_against_third_horse_l894_89469


namespace NUMINAMATH_CALUDE_triangle_side_length_l894_89459

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  B = π / 6 →
  c = 2 * Real.sqrt 3 →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l894_89459


namespace NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l894_89498

/-- An acute-angled triangle with its perimeter and circumradius -/
structure AcuteTriangle where
  -- The perimeter of the triangle
  perimeter : ℝ
  -- The circumradius of the triangle
  circumradius : ℝ
  -- Condition ensuring the triangle is acute-angled (this is a simplification)
  is_acute : perimeter > 0 ∧ circumradius > 0

/-- Theorem stating that for any acute-angled triangle, its perimeter is greater than 4 times its circumradius -/
theorem perimeter_gt_four_times_circumradius (t : AcuteTriangle) : t.perimeter > 4 * t.circumradius := by
  sorry


end NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l894_89498


namespace NUMINAMATH_CALUDE_point_O_is_circumcenter_l894_89487

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the triangle ABC
def Triangle (A B C : Point3D) : Prop := True

-- Define the plane α containing the triangle
def PlaneContainsTriangle (α : Plane) (A B C : Point3D) : Prop := True

-- Define a point being outside a plane
def PointOutsidePlane (P : Point3D) (α : Plane) : Prop := True

-- Define perpendicularity between a line and a plane
def PerpendicularToPlane (P O : Point3D) (α : Plane) : Prop := True

-- Define the foot of a perpendicular
def FootOfPerpendicular (O : Point3D) (P : Point3D) (α : Plane) : Prop := True

-- Define equality of distances
def EqualDistances (P A B C : Point3D) : Prop := True

-- Define circumcenter
def Circumcenter (O : Point3D) (A B C : Point3D) : Prop := True

theorem point_O_is_circumcenter 
  (α : Plane) (A B C P O : Point3D)
  (h1 : Triangle A B C)
  (h2 : PlaneContainsTriangle α A B C)
  (h3 : PointOutsidePlane P α)
  (h4 : PerpendicularToPlane P O α)
  (h5 : FootOfPerpendicular O P α)
  (h6 : EqualDistances P A B C) :
  Circumcenter O A B C := by
  sorry


end NUMINAMATH_CALUDE_point_O_is_circumcenter_l894_89487


namespace NUMINAMATH_CALUDE_cost_increase_is_six_percent_l894_89482

/-- Represents the cost components of manufacturing a car -/
structure CarCost where
  rawMaterial : ℝ
  labor : ℝ
  overheads : ℝ

/-- Calculates the total cost of manufacturing a car -/
def totalCost (cost : CarCost) : ℝ :=
  cost.rawMaterial + cost.labor + cost.overheads

/-- Represents the cost ratio in the first year -/
def initialRatio : CarCost :=
  { rawMaterial := 4
    labor := 3
    overheads := 2 }

/-- Calculates the new cost after applying percentage changes -/
def newCost (cost : CarCost) : CarCost :=
  { rawMaterial := cost.rawMaterial * 1.1
    labor := cost.labor * 1.08
    overheads := cost.overheads * 0.95 }

/-- Theorem stating that the total cost increase is 6% -/
theorem cost_increase_is_six_percent :
  (totalCost (newCost initialRatio) - totalCost initialRatio) / totalCost initialRatio * 100 = 6 := by
  sorry


end NUMINAMATH_CALUDE_cost_increase_is_six_percent_l894_89482


namespace NUMINAMATH_CALUDE_nicky_received_card_value_l894_89448

/-- The value of a card Nicky received in a trade, given the value of cards he traded and his profit -/
def card_value (traded_card_value : ℕ) (num_traded_cards : ℕ) (profit : ℕ) : ℕ :=
  traded_card_value * num_traded_cards + profit

/-- Theorem stating the value of the card Nicky received from Jill -/
theorem nicky_received_card_value :
  card_value 8 2 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_nicky_received_card_value_l894_89448


namespace NUMINAMATH_CALUDE_perfect_squares_digits_parity_l894_89447

/-- A natural number is a perfect square if it is equal to the square of some natural number. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- The units digit of a natural number. -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The tens digit of a natural number. -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem perfect_squares_digits_parity (a b : ℕ) (x y : ℕ) :
  is_perfect_square a →
  is_perfect_square b →
  units_digit a = 1 →
  tens_digit a = x →
  units_digit b = 6 →
  tens_digit b = y →
  Even x ∧ Odd y :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_digits_parity_l894_89447


namespace NUMINAMATH_CALUDE_function_symmetry_implies_m_range_l894_89446

theorem function_symmetry_implies_m_range 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h_f : ∀ x, f x = m * 4^x - 2^x) 
  (h_symmetry : ∃ x_0 : ℝ, x_0 ≠ 0 ∧ f (-x_0) = f x_0) : 
  0 < m ∧ m < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_symmetry_implies_m_range_l894_89446


namespace NUMINAMATH_CALUDE_summer_jolly_degree_difference_l894_89462

theorem summer_jolly_degree_difference :
  ∀ (summer_degrees jolly_degrees : ℕ),
    summer_degrees = 150 →
    summer_degrees + jolly_degrees = 295 →
    summer_degrees - jolly_degrees = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_summer_jolly_degree_difference_l894_89462


namespace NUMINAMATH_CALUDE_chess_tournament_games_l894_89488

theorem chess_tournament_games (n : Nat) (games : Fin n → Nat) :
  n = 5 →
  (∀ i j : Fin n, i ≠ j → games i + games j ≤ n - 1) →
  (∃ p : Fin n, games p = 4) →
  (∃ p : Fin n, games p = 3) →
  (∃ p : Fin n, games p = 2) →
  (∃ p : Fin n, games p = 1) →
  (∃ p : Fin n, games p = 2) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l894_89488


namespace NUMINAMATH_CALUDE_roots_equation_value_l894_89405

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^4 + 3*β = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_value_l894_89405


namespace NUMINAMATH_CALUDE_tangent_line_passes_fixed_point_l894_89411

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 2

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line_l : line_l x y

-- Define the tangent condition
def is_tangent (P : Point_P) (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
  (∃ t : ℝ, A.1 = P.x + t * (P.y - 2) ∧ A.2 = P.y - t * P.x) ∧
  (∃ t : ℝ, B.1 = P.x + t * (P.y - 2) ∧ B.2 = P.y - t * P.x)

-- Theorem statement
theorem tangent_line_passes_fixed_point (P : Point_P) (A B : ℝ × ℝ) :
  is_tangent P A B →
  ∃ t : ℝ, t * A.1 + (1 - t) * B.1 = 1/2 ∧ t * A.2 + (1 - t) * B.2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_passes_fixed_point_l894_89411


namespace NUMINAMATH_CALUDE_parabola_reflection_sum_l894_89486

theorem parabola_reflection_sum (a b c : ℝ) :
  let f := fun x : ℝ => a * x^2 + b * x + c + 3
  let g := fun x : ℝ => -a * x^2 - b * x - c - 3
  ∀ x, f x + g x = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_reflection_sum_l894_89486


namespace NUMINAMATH_CALUDE_line_and_chord_problem_l894_89481

-- Define the circle M
def circle_M : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l
def line_l : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}

-- Define the midpoint P
def point_P : ℝ × ℝ := (1, 1)

-- Define the intersection points A and B
def point_A : ℝ × ℝ := sorry
def point_B : ℝ × ℝ := sorry

theorem line_and_chord_problem :
  point_P = ((point_A.1 + point_B.1) / 2, (point_A.2 + point_B.2) / 2) ∧
  point_A ∈ circle_M ∧ point_B ∈ circle_M ∧
  point_A ∈ line_l ∧ point_B ∈ line_l →
  (∀ p : ℝ × ℝ, p ∈ line_l ↔ p.1 + p.2 = 2) ∧
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_and_chord_problem_l894_89481


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l894_89439

/-- Given an arithmetic sequence {aₙ}, where Sₙ is the sum of the first n terms,
    prove that S₈ = 80 when a₃ = 20 - a₆ -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  a 3 = 20 - a 6 →
  S 8 = 80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l894_89439


namespace NUMINAMATH_CALUDE_divisible_by_3_4_5_count_l894_89426

theorem divisible_by_3_4_5_count : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 50 ∧ (3 ∣ n ∨ 4 ∣ n ∨ 5 ∣ n)) ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 50 ∧ (3 ∣ n ∨ 4 ∣ n ∨ 5 ∣ n) → n ∈ S) ∧ 
  Finset.card S = 29 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_3_4_5_count_l894_89426


namespace NUMINAMATH_CALUDE_best_meeting_days_l894_89494

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define the team members
inductive Member
| Alice
| Bob
| Cindy
| Dave
| Eve

-- Define the availability function
def isAvailable (m : Member) (d : Day) : Bool :=
  match m, d with
  | Member.Alice, Day.Monday => false
  | Member.Alice, Day.Thursday => false
  | Member.Alice, Day.Saturday => false
  | Member.Bob, Day.Tuesday => false
  | Member.Bob, Day.Wednesday => false
  | Member.Bob, Day.Friday => false
  | Member.Cindy, Day.Wednesday => false
  | Member.Cindy, Day.Saturday => false
  | Member.Dave, Day.Monday => false
  | Member.Dave, Day.Tuesday => false
  | Member.Dave, Day.Thursday => false
  | Member.Eve, Day.Thursday => false
  | Member.Eve, Day.Friday => false
  | Member.Eve, Day.Saturday => false
  | _, _ => true

-- Define the function to count available members on a given day
def availableCount (d : Day) : Nat :=
  (List.filter (fun m => isAvailable m d) [Member.Alice, Member.Bob, Member.Cindy, Member.Dave, Member.Eve]).length

-- Theorem statement
theorem best_meeting_days :
  (∀ d : Day, availableCount d ≤ 3) ∧
  (availableCount Day.Monday = 3) ∧
  (availableCount Day.Tuesday = 3) ∧
  (availableCount Day.Wednesday = 3) ∧
  (availableCount Day.Friday = 3) ∧
  (∀ d : Day, availableCount d = 3 → d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Friday) :=
by sorry


end NUMINAMATH_CALUDE_best_meeting_days_l894_89494


namespace NUMINAMATH_CALUDE_number_operations_l894_89434

theorem number_operations (x : ℚ) : x = 192 → 6 * (((x/8) + 8) - 30) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l894_89434


namespace NUMINAMATH_CALUDE_specific_wall_has_30_bricks_l894_89421

/-- Represents a brick wall with a specific pattern -/
structure BrickWall where
  num_rows : ℕ
  bottom_row_bricks : ℕ
  brick_decrease : ℕ

/-- Calculates the total number of bricks in the wall -/
def total_bricks (wall : BrickWall) : ℕ :=
  sorry

/-- Theorem stating that a specific brick wall configuration has 30 bricks in total -/
theorem specific_wall_has_30_bricks :
  let wall : BrickWall := {
    num_rows := 5,
    bottom_row_bricks := 8,
    brick_decrease := 1
  }
  total_bricks wall = 30 := by
  sorry

end NUMINAMATH_CALUDE_specific_wall_has_30_bricks_l894_89421


namespace NUMINAMATH_CALUDE_haman_initial_trays_l894_89463

/-- Represents the number of eggs in a standard tray -/
def eggs_per_tray : ℕ := 30

/-- Represents the number of trays Haman dropped -/
def dropped_trays : ℕ := 2

/-- Represents the number of trays added after the accident -/
def added_trays : ℕ := 7

/-- Represents the total number of eggs sold -/
def total_eggs_sold : ℕ := 540

/-- Theorem stating that Haman initially collected 13 trays of eggs -/
theorem haman_initial_trays :
  (total_eggs_sold / eggs_per_tray - added_trays + dropped_trays : ℕ) = 13 := by
  sorry

end NUMINAMATH_CALUDE_haman_initial_trays_l894_89463


namespace NUMINAMATH_CALUDE_number_pattern_equality_l894_89409

theorem number_pattern_equality (n : ℕ) (h : n > 1) :
  3 * (6 * (10^n - 1) / 9)^3 = 
    8 * ((10^n - 1) / 9) * 10^(2*n+1) + 
    6 * 10^(2*n) + 
    2 * ((10^n - 1) / 9) * 10^(n+1) + 
    4 * 10^n + 
    8 * ((10^n - 1) / 9) := by
  sorry

end NUMINAMATH_CALUDE_number_pattern_equality_l894_89409


namespace NUMINAMATH_CALUDE_school_supplies_problem_l894_89423

/-- Represents the school supplies problem --/
theorem school_supplies_problem 
  (num_students : ℕ) 
  (pens_per_student : ℕ) 
  (notebooks_per_student : ℕ) 
  (binders_per_student : ℕ) 
  (pen_cost : ℚ) 
  (notebook_cost : ℚ) 
  (binder_cost : ℚ) 
  (highlighter_cost : ℚ) 
  (teacher_discount : ℚ) 
  (total_spent : ℚ) 
  (h1 : num_students = 30)
  (h2 : pens_per_student = 5)
  (h3 : notebooks_per_student = 3)
  (h4 : binders_per_student = 1)
  (h5 : pen_cost = 1/2)
  (h6 : notebook_cost = 5/4)
  (h7 : binder_cost = 17/4)
  (h8 : highlighter_cost = 3/4)
  (h9 : teacher_discount = 100)
  (h10 : total_spent = 260) :
  (total_spent - (num_students * (pens_per_student * pen_cost + 
   notebooks_per_student * notebook_cost + 
   binders_per_student * binder_cost) - teacher_discount)) / 
  (num_students * highlighter_cost) = 2 := by
sorry


end NUMINAMATH_CALUDE_school_supplies_problem_l894_89423


namespace NUMINAMATH_CALUDE_shaded_area_is_108pi_l894_89436

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- Represents a semicircle -/
structure Semicircle :=
  (center : Point)
  (radius : ℝ)

/-- The configuration of points and semicircles -/
structure Configuration :=
  (A B C D E F : Point)
  (AF AB BC CD DE EF : Semicircle)

/-- The conditions of the problem -/
def problem_conditions (config : Configuration) : Prop :=
  let {A, B, C, D, E, F, AF, AB, BC, CD, DE, EF} := config
  (B.x - A.x = 6) ∧ 
  (C.x - B.x = 6) ∧ 
  (D.x - C.x = 6) ∧ 
  (E.x - D.x = 6) ∧ 
  (F.x - E.x = 6) ∧
  (AF.radius = 15) ∧
  (AB.radius = 3) ∧
  (BC.radius = 3) ∧
  (CD.radius = 3) ∧
  (DE.radius = 3) ∧
  (EF.radius = 3)

/-- The area of the shaded region -/
def shaded_area (config : Configuration) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The theorem stating that the shaded area is 108π -/
theorem shaded_area_is_108pi (config : Configuration) 
  (h : problem_conditions config) : shaded_area config = 108 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_108pi_l894_89436


namespace NUMINAMATH_CALUDE_decreasing_quadratic_function_l894_89400

theorem decreasing_quadratic_function (a : ℝ) :
  (∀ x < 4, (∀ y < x, x^2 + 2*(a-1)*x + 2 < y^2 + 2*(a-1)*y + 2)) →
  a ≤ -3 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_quadratic_function_l894_89400


namespace NUMINAMATH_CALUDE_initial_peaches_count_l894_89490

/-- Represents the state of the fruit bowl on a given day -/
structure FruitBowl :=
  (day : Nat)
  (ripe : Nat)
  (unripe : Nat)

/-- Updates the fruit bowl state for the next day -/
def nextDay (bowl : FruitBowl) : FruitBowl :=
  let newRipe := bowl.ripe + 2
  let newUnripe := bowl.unripe - 2
  { day := bowl.day + 1, ripe := newRipe, unripe := newUnripe }

/-- Represents eating 3 peaches on day 3 -/
def eatPeaches (bowl : FruitBowl) : FruitBowl :=
  { bowl with ripe := bowl.ripe - 3 }

/-- The initial state of the fruit bowl -/
def initialBowl : FruitBowl :=
  { day := 0, ripe := 4, unripe := 13 }

/-- The final state of the fruit bowl after 5 days -/
def finalBowl : FruitBowl :=
  (nextDay ∘ nextDay ∘ eatPeaches ∘ nextDay ∘ nextDay ∘ nextDay) initialBowl

/-- Theorem stating that the initial number of peaches was 17 -/
theorem initial_peaches_count :
  initialBowl.ripe + initialBowl.unripe = 17 ∧
  finalBowl.ripe = finalBowl.unripe + 7 :=
by sorry


end NUMINAMATH_CALUDE_initial_peaches_count_l894_89490


namespace NUMINAMATH_CALUDE_mask_price_problem_l894_89470

theorem mask_price_problem (first_total second_total : ℚ) 
  (price_increase : ℚ) (quantity_increase : ℕ) :
  first_total = 500000 →
  second_total = 770000 →
  price_increase = 1.4 →
  quantity_increase = 10000 →
  ∃ (first_price first_quantity : ℚ),
    first_price * first_quantity = first_total ∧
    price_increase * first_price * (first_quantity + quantity_increase) = second_total ∧
    first_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_mask_price_problem_l894_89470


namespace NUMINAMATH_CALUDE_right_triangle_area_l894_89465

theorem right_triangle_area (AB AC : ℝ) (h1 : AB = 12) (h2 : AC = 5) :
  let BC : ℝ := Real.sqrt (AB^2 - AC^2)
  (1 / 2) * AC * BC = (5 * Real.sqrt 119) / 2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l894_89465


namespace NUMINAMATH_CALUDE_square_cardinality_continuum_l894_89457

/-- A square in the 2D plane -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

/-- The unit interval [0, 1] -/
def UnitInterval : Set ℝ :=
  {x | 0 ≤ x ∧ x ≤ 1}

theorem square_cardinality_continuum :
  Cardinal.mk (Square) = Cardinal.mk (UnitInterval) :=
sorry

end NUMINAMATH_CALUDE_square_cardinality_continuum_l894_89457


namespace NUMINAMATH_CALUDE_house_transaction_net_worth_change_l894_89489

/-- Represents the financial state of a person -/
structure FinancialState where
  cash : Int
  houseValue : Int

/-- Calculates the net worth of a person -/
def netWorth (state : FinancialState) : Int :=
  state.cash + state.houseValue

/-- Represents a house transaction between two people -/
def houseTransaction (buyer seller : FinancialState) (price : Int) : (FinancialState × FinancialState) :=
  ({ cash := buyer.cash - price, houseValue := seller.houseValue },
   { cash := seller.cash + price, houseValue := 0 })

theorem house_transaction_net_worth_change 
  (initialA initialB : FinancialState)
  (houseValue firstPrice secondPrice : Int) :
  initialA.cash = 15000 →
  initialA.houseValue = 12000 →
  initialB.cash = 13000 →
  initialB.houseValue = 0 →
  houseValue = 12000 →
  firstPrice = 14000 →
  secondPrice = 10000 →
  let (afterFirstA, afterFirstB) := houseTransaction initialB initialA firstPrice
  let (finalB, finalA) := houseTransaction afterFirstA afterFirstB secondPrice
  netWorth finalA - netWorth initialA = 4000 ∧
  netWorth finalB - netWorth initialB = -4000 := by sorry


end NUMINAMATH_CALUDE_house_transaction_net_worth_change_l894_89489


namespace NUMINAMATH_CALUDE_original_price_calculation_l894_89433

theorem original_price_calculation (a b : ℝ) : 
  ∃ x : ℝ, (x - a) * (1 - 0.4) = b ∧ x = a + (5/3) * b := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l894_89433


namespace NUMINAMATH_CALUDE_expression_simplification_l894_89443

theorem expression_simplification :
  ∀ p : ℝ, ((7*p+3)-3*p*5)*(2)+(5-2/4)*(8*p-12) = 20*p - 48 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l894_89443


namespace NUMINAMATH_CALUDE_quadratic_inequality_l894_89415

/-- A quadratic function f(x) = x^2 + bx + c -/
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

/-- Theorem: If f(-3) = f(1) for a quadratic function f(x) = x^2 + bx + c, 
    then f(1) > c > f(-1) -/
theorem quadratic_inequality (b c : ℝ) : 
  f b c (-3) = f b c 1 → f b c 1 > c ∧ c > f b c (-1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l894_89415


namespace NUMINAMATH_CALUDE_g_extreme_points_l894_89492

noncomputable def f (x : ℝ) : ℝ := Real.log x - x - 1

noncomputable def g (x : ℝ) : ℝ := x * f x + (1/2) * x^2 + 2 * x

noncomputable def g' (x : ℝ) : ℝ := f x + 3

theorem g_extreme_points :
  ∃ (x₁ x₂ : ℝ), 
    0 < x₁ ∧ x₁ < 1 ∧
    3 < x₂ ∧ x₂ < 4 ∧
    g' x₁ = 0 ∧ g' x₂ = 0 ∧
    (∀ x ∈ Set.Ioo 0 1, x ≠ x₁ → g' x ≠ 0) ∧
    (∀ x ∈ Set.Ioo 3 4, x ≠ x₂ → g' x ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_g_extreme_points_l894_89492


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l894_89478

theorem triangle_abc_proof (a b c : ℝ) (A B : ℝ) :
  a = 2 * Real.sqrt 3 →
  c = Real.sqrt 6 + Real.sqrt 2 →
  B = 45 * (π / 180) →
  b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) →
  Real.cos A = (b^2 + c^2 - a^2) / (2*b*c) →
  b = 2 * Real.sqrt 2 ∧ A = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_proof_l894_89478


namespace NUMINAMATH_CALUDE_compute_expression_l894_89419

theorem compute_expression : 10 + 8 * (2 - 9)^2 = 402 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l894_89419


namespace NUMINAMATH_CALUDE_exists_winning_strategy_l894_89455

/-- Represents the color of a hat -/
inductive HatColor
| Black
| White

/-- Represents an agent's guess about their own hat color -/
def Guess := HatColor

/-- Represents a strategy function that takes the observed hat color and returns a guess -/
def Strategy := HatColor → Guess

/-- Represents the outcome of applying strategies to a pair of hat colors -/
def Outcome (c1 c2 : HatColor) (s1 s2 : Strategy) : Prop :=
  (s1 c2 = c1) ∨ (s2 c1 = c2)

/-- Theorem stating that there exists a pair of strategies that guarantees
    at least one correct guess for any combination of hat colors -/
theorem exists_winning_strategy :
  ∃ (s1 s2 : Strategy), ∀ (c1 c2 : HatColor), Outcome c1 c2 s1 s2 := by
  sorry

end NUMINAMATH_CALUDE_exists_winning_strategy_l894_89455


namespace NUMINAMATH_CALUDE_tangent_line_parallel_proof_l894_89442

/-- The parabola y = x^2 -/
def parabola (x : ℝ) : ℝ := x^2

/-- The slope of the tangent line to the parabola at point (a, a^2) -/
def tangent_slope (a : ℝ) : ℝ := 2 * a

/-- The slope of the line 2x - y + 4 = 0 -/
def given_line_slope : ℝ := 2

/-- The equation of the tangent line at point (a, a^2) -/
def tangent_line_eq (a : ℝ) (x y : ℝ) : Prop :=
  y - a^2 = tangent_slope a * (x - a)

theorem tangent_line_parallel_proof (a : ℝ) :
  tangent_slope a = given_line_slope →
  a = 1 ∧
  ∀ x y : ℝ, tangent_line_eq a x y ↔ 2*x - y - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_proof_l894_89442


namespace NUMINAMATH_CALUDE_liar_proportion_is_half_l894_89473

/-- Represents the proportion of liars in a village -/
def proportion_of_liars : ℝ := sorry

/-- The proportion of liars is between 0 and 1 -/
axiom proportion_bounds : 0 ≤ proportion_of_liars ∧ proportion_of_liars ≤ 1

/-- The proportion of liars is indistinguishable from the proportion of truth-tellers when roles are reversed -/
axiom indistinguishable_proportion : proportion_of_liars = 1 - proportion_of_liars

theorem liar_proportion_is_half : proportion_of_liars = 1/2 := by sorry

end NUMINAMATH_CALUDE_liar_proportion_is_half_l894_89473


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l894_89466

/-- Given a circle with equation x^2 + y^2 - 4x = 0, prove that its center is (2, 0) and its radius is 2 -/
theorem circle_center_and_radius :
  let equation := fun (x y : ℝ) => x^2 + y^2 - 4*x
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, equation x y = 0 ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (2, 0) ∧
    radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l894_89466


namespace NUMINAMATH_CALUDE_rectangle_area_change_l894_89438

theorem rectangle_area_change (L B x : ℝ) (h1 : L > 0) (h2 : B > 0) (h3 : x > 0) : 
  (L + x / 100 * L) * (B - x / 100 * B) = 99 / 100 * (L * B) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l894_89438


namespace NUMINAMATH_CALUDE_triangle_side_length_l894_89471

theorem triangle_side_length
  (a b c : ℝ)
  (A : ℝ)
  (area : ℝ)
  (h1 : a + b + c = 20)
  (h2 : area = 10 * Real.sqrt 3)
  (h3 : A = π / 3) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l894_89471


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l894_89464

theorem units_digit_of_expression : 
  (3 * 19 * 1981 - 3^4) % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l894_89464


namespace NUMINAMATH_CALUDE_quadrupled_base_and_exponent_l894_89414

theorem quadrupled_base_and_exponent (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (4 * a)^(4 * b) = a^b * x^(2 * b) → x = 16 * a^(3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadrupled_base_and_exponent_l894_89414


namespace NUMINAMATH_CALUDE_min_value_theorem_l894_89441

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 2 ∧
  ∀ x, x = (a^2 + b^2) / (a - b) → x ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l894_89441


namespace NUMINAMATH_CALUDE_no_solution_linear_system_l894_89483

theorem no_solution_linear_system :
  ¬ ∃ (x y z : ℝ),
    (3 * x - 4 * y + z = 10) ∧
    (6 * x - 8 * y + 2 * z = 5) ∧
    (2 * x - y - z = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_linear_system_l894_89483


namespace NUMINAMATH_CALUDE_translation_right_2_units_l894_89424

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_2_units :
  let A : Point := { x := 3, y := -2 }
  let A' : Point := translateRight A 2
  A'.x = 5 ∧ A'.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_2_units_l894_89424


namespace NUMINAMATH_CALUDE_problem_solution_l894_89418

/-- Given m ≥ 0 and f(x) = 2|x - 1| - |2x + m| with a maximum value of 3,
    prove that m = 1 and min(a² + b² + c²) = 1/6 where a - 2b + c = m -/
theorem problem_solution (m : ℝ) (h_m : m ≥ 0)
  (f : ℝ → ℝ) (h_f : ∀ x, f x = 2 * |x - 1| - |2*x + m|)
  (h_max : ∀ x, f x ≤ 3) (h_exists : ∃ x, f x = 3) :
  m = 1 ∧ (∃ a b c : ℝ, a - 2*b + c = m ∧
    a^2 + b^2 + c^2 = 1/6 ∧
    ∀ a' b' c' : ℝ, a' - 2*b' + c' = m → a'^2 + b'^2 + c'^2 ≥ 1/6) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l894_89418


namespace NUMINAMATH_CALUDE_range_and_minimum_l894_89497

theorem range_and_minimum (x y a : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : x^2 - y^2 = 2)
  (h_ineq : (1 / (2*x^2)) + (2*y/x) < a) :
  (0 < y/x ∧ y/x < 1) ∧ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_and_minimum_l894_89497


namespace NUMINAMATH_CALUDE_age_ratio_after_two_years_l894_89472

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given the specified conditions. -/
theorem age_ratio_after_two_years (son_age : ℕ) (man_age : ℕ) : 
  son_age = 27 → 
  man_age = son_age + 29 → 
  (man_age + 2) / (son_age + 2) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_after_two_years_l894_89472


namespace NUMINAMATH_CALUDE_probability_at_least_3_of_6_l894_89427

def probability_at_least_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  Finset.sum (Finset.range (n - k + 1))
    (λ i => Nat.choose n (k + i) * p ^ (k + i) * (1 - p) ^ (n - k - i))

theorem probability_at_least_3_of_6 :
  probability_at_least_k_successes 6 3 (2/3) = 656/729 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_3_of_6_l894_89427


namespace NUMINAMATH_CALUDE_police_coverage_l894_89431

-- Define the set of intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the streets as sets of intersections
def horizontal1 : Set Intersection := {Intersection.A, Intersection.B, Intersection.C, Intersection.D}
def horizontal2 : Set Intersection := {Intersection.E, Intersection.F, Intersection.G}
def horizontal3 : Set Intersection := {Intersection.H, Intersection.I, Intersection.J, Intersection.K}
def vertical1 : Set Intersection := {Intersection.A, Intersection.E, Intersection.H}
def vertical2 : Set Intersection := {Intersection.B, Intersection.F, Intersection.I}
def vertical3 : Set Intersection := {Intersection.D, Intersection.G, Intersection.J}
def diagonal1 : Set Intersection := {Intersection.H, Intersection.F, Intersection.C}
def diagonal2 : Set Intersection := {Intersection.C, Intersection.G, Intersection.K}

-- Define the set of all streets
def allStreets : Set (Set Intersection) :=
  {horizontal1, horizontal2, horizontal3, vertical1, vertical2, vertical3, diagonal1, diagonal2}

-- Define the set of intersections with police officers
def policeLocations : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}

-- Theorem statement
theorem police_coverage :
  ∀ street ∈ allStreets, ∃ intersection ∈ street, intersection ∈ policeLocations :=
by sorry

end NUMINAMATH_CALUDE_police_coverage_l894_89431


namespace NUMINAMATH_CALUDE_system_solution_l894_89437

theorem system_solution (x y : Real) (k₁ k₂ : Int) : 
  (Real.sqrt 2 * Real.sin x = Real.sin y) →
  (Real.sqrt 2 * Real.cos x = Real.sqrt 3 * Real.cos y) →
  (∃ n₁ n₂ : Int, x = n₁ * π / 6 + k₂ * π ∧ y = n₂ * π / 4 + k₁ * π ∧ 
   (n₁ = 1 ∨ n₁ = -1) ∧ (n₂ = 1 ∨ n₂ = -1) ∧ k₁ % 2 = k₂ % 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l894_89437


namespace NUMINAMATH_CALUDE_expected_male_athletes_expected_male_athletes_eq_twelve_l894_89451

/-- Given a team of athletes with a specific male-to-total ratio,
    calculate the expected number of male athletes in a stratified sample. -/
theorem expected_male_athletes 
  (total_athletes : ℕ) 
  (male_ratio : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = 84) 
  (h2 : male_ratio = 4/7) 
  (h3 : sample_size = 21) : 
  ℕ := by
  sorry

#check expected_male_athletes

theorem expected_male_athletes_eq_twelve 
  (total_athletes : ℕ) 
  (male_ratio : ℚ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = 84) 
  (h2 : male_ratio = 4/7) 
  (h3 : sample_size = 21) : 
  expected_male_athletes total_athletes male_ratio sample_size h1 h2 h3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expected_male_athletes_expected_male_athletes_eq_twelve_l894_89451


namespace NUMINAMATH_CALUDE_parallelepiped_inequality_l894_89468

theorem parallelepiped_inequality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diagonal : d^2 = a^2 + b^2 + c^2) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := by
sorry

end NUMINAMATH_CALUDE_parallelepiped_inequality_l894_89468


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l894_89445

theorem complex_fraction_equality (a b : ℂ) 
  (h : (a + b) / (a - b) + (a - b) / (a + b) = 2) :
  (a^4 + b^4) / (a^4 - b^4) + (a^4 - b^4) / (a^4 + b^4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l894_89445


namespace NUMINAMATH_CALUDE_original_price_calculation_l894_89453

/-- Given an article sold for $115 with a 15% gain, prove that the original price was $100 --/
theorem original_price_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 115 ∧ gain_percent = 15 → 
  ∃ (original_price : ℝ), 
    original_price = 100 ∧ 
    selling_price = original_price * (1 + gain_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_original_price_calculation_l894_89453


namespace NUMINAMATH_CALUDE_copper_needed_in_mixture_l894_89404

/-- Given a manufacturing mixture with specified percentages of materials,
    this theorem calculates the amount of copper needed when a certain amount of lead is used. -/
theorem copper_needed_in_mixture (total : ℝ) (cobalt_percent lead_percent copper_percent : ℝ) 
    (lead_amount : ℝ) (copper_amount : ℝ) : 
  cobalt_percent = 0.15 →
  lead_percent = 0.25 →
  copper_percent = 0.60 →
  cobalt_percent + lead_percent + copper_percent = 1 →
  lead_amount = 5 →
  total * lead_percent = lead_amount →
  copper_amount = total * copper_percent →
  copper_amount = 12 := by
sorry

end NUMINAMATH_CALUDE_copper_needed_in_mixture_l894_89404


namespace NUMINAMATH_CALUDE_n_equals_ten_l894_89461

/-- The number of sides in a regular polygon satisfying the given condition -/
def n : ℕ := sorry

/-- The measure of the internal angle in a regular polygon with k sides -/
def internal_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

/-- The condition that the internal angle of an n-sided polygon is 12° less
    than that of a polygon with n/4 fewer sides -/
axiom angle_condition : internal_angle n = internal_angle (3 * n / 4) - 12

/-- Theorem stating that n = 10 -/
theorem n_equals_ten : n = 10 := by sorry

end NUMINAMATH_CALUDE_n_equals_ten_l894_89461


namespace NUMINAMATH_CALUDE_solve_equation_for_k_l894_89428

theorem solve_equation_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4)) →
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_for_k_l894_89428


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l894_89499

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the theorem
theorem sufficient_but_not_necessary
  (a b : Line) (α β : Plane)
  (h_diff : a ≠ b)
  (h_parallel : parallel α β)
  (h_perp : perpendicular a α) :
  (∃ (c : Line), c ≠ b ∧ perpendicularLines a c ∧ ¬lineParallelPlane c β) ∧
  (lineParallelPlane b β → perpendicularLines a b) ∧
  ¬(perpendicularLines a b → lineParallelPlane b β) :=
sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l894_89499


namespace NUMINAMATH_CALUDE_square_difference_153_147_l894_89430

theorem square_difference_153_147 : 153^2 - 147^2 = 1800 := by sorry

end NUMINAMATH_CALUDE_square_difference_153_147_l894_89430


namespace NUMINAMATH_CALUDE_smallest_valid_number_l894_89413

def is_valid (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  4 * (n % 10 * 10 + n / 10) = 2 * n

theorem smallest_valid_number : 
  (∃ (n : ℕ), is_valid n) ∧ 
  (∀ (m : ℕ), is_valid m → m ≥ 52) ∧
  is_valid 52 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l894_89413


namespace NUMINAMATH_CALUDE_percentage_problem_l894_89496

theorem percentage_problem (P : ℝ) : 
  0 ≤ P ∧ P ≤ 100 → P * 700 = (60 / 100) * 150 + 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l894_89496


namespace NUMINAMATH_CALUDE_four_numbers_with_one_sixth_property_l894_89474

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

/-- The three-digit number obtained by removing the leftmost digit of a four-digit number -/
def RemoveLeftmostDigit (n : ℕ) : ℕ := n % 1000

/-- The property that the three-digit number obtained by removing the leftmost digit is one sixth of the original number -/
def HasOneSixthProperty (n : ℕ) : Prop :=
  FourDigitNumber n ∧ RemoveLeftmostDigit n = n / 6

/-- The theorem stating that there are exactly 4 numbers satisfying the property -/
theorem four_numbers_with_one_sixth_property :
  ∃! (s : Finset ℕ), s.card = 4 ∧ ∀ n, n ∈ s ↔ HasOneSixthProperty n :=
sorry

end NUMINAMATH_CALUDE_four_numbers_with_one_sixth_property_l894_89474


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_twelve_l894_89460

theorem sum_of_solutions_eq_twelve : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 6)^2 = 50 ∧ (x₂ - 6)^2 = 50 ∧ x₁ + x₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_twelve_l894_89460


namespace NUMINAMATH_CALUDE_square_ratios_area_ratio_diagonal_ratio_l894_89480

/-- Given two squares where the perimeter of one is 4 times the other, 
    prove the ratios of their areas and diagonals -/
theorem square_ratios (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a ^ 2 = 16 * b ^ 2 ∧ a * Real.sqrt 2 = 4 * (b * Real.sqrt 2) := by
  sorry

/-- The area of the larger square is 16 times the area of the smaller square -/
theorem area_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a ^ 2 = 16 * b ^ 2 := by
  sorry

/-- The diagonal of the larger square is 4 times the diagonal of the smaller square -/
theorem diagonal_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perimeter : 4 * a = 4 * (4 * b)) : 
  a * Real.sqrt 2 = 4 * (b * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_square_ratios_area_ratio_diagonal_ratio_l894_89480
