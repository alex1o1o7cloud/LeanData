import Mathlib

namespace NUMINAMATH_CALUDE_james_coins_value_l3036_303681

/-- Represents the value of James's coins in cents -/
def coin_value : ℕ := 38

/-- Represents the total number of coins James has -/
def total_coins : ℕ := 15

/-- Represents the number of nickels James has -/
def num_nickels : ℕ := 6

/-- Represents the number of pennies James has -/
def num_pennies : ℕ := 9

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

theorem james_coins_value :
  (num_nickels * nickel_value + num_pennies * penny_value = coin_value) ∧
  (num_nickels + num_pennies = total_coins) ∧
  (num_pennies = num_nickels + 2) := by
  sorry

end NUMINAMATH_CALUDE_james_coins_value_l3036_303681


namespace NUMINAMATH_CALUDE_sum_of_cubes_values_l3036_303660

open Complex Matrix

/-- A 3x3 circulant matrix with complex entries a, b, c -/
def M (a b c : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  !![a, b, c; b, c, a; c, a, b]

/-- The theorem statement -/
theorem sum_of_cubes_values (a b c : ℂ) : 
  M a b c ^ 2 = 1 → a * b * c = 1 → a^3 + b^3 + c^3 = 2 ∨ a^3 + b^3 + c^3 = 4 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_cubes_values_l3036_303660


namespace NUMINAMATH_CALUDE_correct_propositions_l3036_303604

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations and operations
def subset (l : Line) (p : Plane) : Prop := sorry
def parallel (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p₁ p₂ : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def perpendicular_planes (p₁ p₂ : Plane) : Prop := sorry
def intersection (p₁ p₂ : Plane) : Line := sorry

-- State the theorem
theorem correct_propositions 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  (∀ (m n : Line) (α β : Plane) (l : Line),
    subset m α → subset n β → perpendicular_planes α β → 
    intersection α β = l → perpendicular m l → perpendicular m n) ∧
  (∀ (m : Line) (α β : Plane),
    perpendicular m α → perpendicular m β → parallel_planes α β) := by
  sorry

end NUMINAMATH_CALUDE_correct_propositions_l3036_303604


namespace NUMINAMATH_CALUDE_solve_equation_l3036_303678

theorem solve_equation (x : ℝ) : (x^3).sqrt = 9 * (81^(1/9)) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3036_303678


namespace NUMINAMATH_CALUDE_distance_focus_to_asymptote_l3036_303668

/-- The distance from the right focus of the hyperbola x²/4 - y² = 1 to its asymptote x - 2y = 0 is 1 -/
theorem distance_focus_to_asymptote (x y : ℝ) : 
  let hyperbola := (x^2 / 4 - y^2 = 1)
  let right_focus := (x = Real.sqrt 5 ∧ y = 0)
  let asymptote := (x - 2*y = 0)
  let distance := |x - 2*y| / Real.sqrt 5
  (hyperbola ∧ right_focus ∧ asymptote) → distance = 1 := by
sorry


end NUMINAMATH_CALUDE_distance_focus_to_asymptote_l3036_303668


namespace NUMINAMATH_CALUDE_octagon_area_l3036_303683

/-- The area of a regular octagon inscribed in a square -/
theorem octagon_area (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) :
  let octagon_side := 2 * Real.sqrt 2
  let square_area := s^2
  let triangle_area := 2
  square_area - 4 * triangle_area = 16 + 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_octagon_area_l3036_303683


namespace NUMINAMATH_CALUDE_a_eq_2_sufficient_not_necessary_for_abs_a_eq_2_l3036_303663

theorem a_eq_2_sufficient_not_necessary_for_abs_a_eq_2 :
  (∃ a : ℝ, a = 2 → |a| = 2) ∧ 
  (∃ a : ℝ, |a| = 2 ∧ a ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_a_eq_2_sufficient_not_necessary_for_abs_a_eq_2_l3036_303663


namespace NUMINAMATH_CALUDE_domain_subset_theorem_l3036_303655

theorem domain_subset_theorem (a : ℝ) : 
  (Set.Ioo a (a + 1) ⊆ Set.Ioo (-1) 1) ↔ a ∈ Set.Icc (-1) 0 := by
sorry

end NUMINAMATH_CALUDE_domain_subset_theorem_l3036_303655


namespace NUMINAMATH_CALUDE_problem_1_l3036_303644

theorem problem_1 : (-3) + (-9) - 10 - (-18) = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3036_303644


namespace NUMINAMATH_CALUDE_max_overtakes_l3036_303641

/-- Represents a team in the relay race -/
structure Team :=
  (members : Nat)
  (segments : Nat)

/-- Represents the relay race setup -/
structure RelayRace :=
  (team1 : Team)
  (team2 : Team)
  (simultaneous_start : Bool)
  (instantaneous_exchange : Bool)

/-- Defines what constitutes an overtake in the race -/
def is_valid_overtake (race : RelayRace) (position : Nat) : Prop :=
  position > 0 ∧ position < race.team1.segments ∧ position < race.team2.segments

/-- The main theorem stating the maximum number of overtakes -/
theorem max_overtakes (race : RelayRace) : 
  race.team1.members = 20 →
  race.team2.members = 20 →
  race.team1.segments = 20 →
  race.team2.segments = 20 →
  race.simultaneous_start = true →
  race.instantaneous_exchange = true →
  ∃ (n : Nat), n = 38 ∧ 
    (∀ (m : Nat), (∃ (valid_overtakes : List Nat), 
      (∀ o ∈ valid_overtakes, is_valid_overtake race o) ∧ 
      valid_overtakes.length = m) → m ≤ n) :=
sorry


end NUMINAMATH_CALUDE_max_overtakes_l3036_303641


namespace NUMINAMATH_CALUDE_pie_cost_is_six_l3036_303612

/-- The cost of a pie given initial and remaining amounts -/
def pieCost (initialAmount remainingAmount : ℕ) : ℕ :=
  initialAmount - remainingAmount

/-- Theorem: The cost of the pie is $6 -/
theorem pie_cost_is_six :
  pieCost 63 57 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pie_cost_is_six_l3036_303612


namespace NUMINAMATH_CALUDE_complex_multiplication_simplification_l3036_303688

theorem complex_multiplication_simplification :
  ((-3 - 2 * Complex.I) - (1 + 4 * Complex.I)) * (2 - 3 * Complex.I) = 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_simplification_l3036_303688


namespace NUMINAMATH_CALUDE_overlap_implies_ratio_l3036_303658

/-- Two overlapping rectangles with dimensions p and q -/
def overlap_rectangles (p q : ℝ) : Prop :=
  ∃ (overlap_area total_area : ℝ),
    overlap_area = q^2 ∧
    total_area = 2*p*q - q^2 ∧
    overlap_area = (1/4) * total_area

/-- The ratio of p to q is 5:2 -/
def ratio_is_5_2 (p q : ℝ) : Prop :=
  p / q = 5/2

/-- Theorem: If two rectangles of dimensions p and q overlap such that
    the overlap area is one-quarter of the total area, then p:q = 5:2 -/
theorem overlap_implies_ratio (p q : ℝ) (h : q ≠ 0) :
  overlap_rectangles p q → ratio_is_5_2 p q :=
by
  sorry


end NUMINAMATH_CALUDE_overlap_implies_ratio_l3036_303658


namespace NUMINAMATH_CALUDE_unique_prime_sum_and_diff_l3036_303671

theorem unique_prime_sum_and_diff : 
  ∃! p : ℕ, Prime p ∧ 
  (∃ a b : ℕ, Prime a ∧ Prime b ∧ p = a + b) ∧ 
  (∃ c d : ℕ, Prime c ∧ Prime d ∧ p = c - d) ∧ 
  p = 5 := by sorry

end NUMINAMATH_CALUDE_unique_prime_sum_and_diff_l3036_303671


namespace NUMINAMATH_CALUDE_composition_value_l3036_303695

/-- Given two functions g and f, prove that f(g(3)) = 29 -/
theorem composition_value (g f : ℝ → ℝ) 
  (hg : ∀ x, g x = x^2) 
  (hf : ∀ x, f x = 3*x + 2) : 
  f (g 3) = 29 := by
  sorry

end NUMINAMATH_CALUDE_composition_value_l3036_303695


namespace NUMINAMATH_CALUDE_danny_carpooling_l3036_303615

/-- Given Danny's carpooling route, prove the distance to the first friend's house -/
theorem danny_carpooling (x : ℝ) :
  x > 0 ∧ 
  (x / 2 > 0) ∧ 
  (3 * (x + x / 2) = 36) →
  x = 8 :=
by sorry

end NUMINAMATH_CALUDE_danny_carpooling_l3036_303615


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l3036_303657

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I * ((2 : ℂ) + a * Complex.I) * ((1 : ℂ) - Complex.I)).re = 0 → 
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l3036_303657


namespace NUMINAMATH_CALUDE_keystone_arch_angle_l3036_303670

theorem keystone_arch_angle (n : ℕ) (angle : ℝ) : 
  n = 10 → -- There are 10 trapezoids
  angle = (180 : ℝ) - (360 / (2 * n)) → -- The larger interior angle
  angle = 99 := by
  sorry

end NUMINAMATH_CALUDE_keystone_arch_angle_l3036_303670


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3036_303669

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧ x = 1 ∧ y = 0) →
  (∃ (c : ℝ), c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 5) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3036_303669


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l3036_303645

theorem ceiling_product_equation :
  ∃ x : ℝ, ⌈x⌉ * x = 210 ∧ x = 14 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l3036_303645


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3036_303635

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ),
    (∀ x, 3 * x^2 - 1 = 5 * x ↔ a * x^2 + b * x + c = 0) →
    a = 3 ∧ b = -5 ∧ c = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3036_303635


namespace NUMINAMATH_CALUDE_travel_options_count_l3036_303605

/-- The number of travel options from A to C given the number of trains from A to B and ferries from B to C -/
def travelOptions (trains : ℕ) (ferries : ℕ) : ℕ :=
  trains * ferries

/-- Theorem stating that the number of travel options from A to C is 6 -/
theorem travel_options_count :
  travelOptions 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_travel_options_count_l3036_303605


namespace NUMINAMATH_CALUDE_exponential_inequality_l3036_303616

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2^a + 2*a = 2^b + 3*b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3036_303616


namespace NUMINAMATH_CALUDE_rain_probability_tel_aviv_l3036_303634

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : 
  binomial_probability 6 4 0.5 = 0.234375 := by sorry

end NUMINAMATH_CALUDE_rain_probability_tel_aviv_l3036_303634


namespace NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_gt_1_l3036_303648

theorem x_gt_2_sufficient_not_necessary_for_x_gt_1 :
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_gt_1_l3036_303648


namespace NUMINAMATH_CALUDE_circle_line_regions_l3036_303664

/-- Represents a configuration of concentric circles and intersecting lines. -/
structure CircleLineConfiguration where
  n : ℕ  -- number of concentric circles
  k : ℕ  -- number of lines through point A
  m : ℕ  -- number of lines through point B

/-- Calculates the maximum number of regions formed by the configuration. -/
def max_regions (config : CircleLineConfiguration) : ℕ :=
  (config.k + 1) * (config.m + 1) * config.n

/-- Calculates the minimum number of regions formed by the configuration. -/
def min_regions (config : CircleLineConfiguration) : ℕ :=
  (config.k + config.m + 1) + config.n - 1

/-- Theorem stating the maximum and minimum number of regions formed. -/
theorem circle_line_regions (config : CircleLineConfiguration) :
  (max_regions config = (config.k + 1) * (config.m + 1) * config.n) ∧
  (min_regions config = (config.k + config.m + 1) + config.n - 1) :=
sorry

end NUMINAMATH_CALUDE_circle_line_regions_l3036_303664


namespace NUMINAMATH_CALUDE_expand_and_compare_l3036_303600

theorem expand_and_compare (m n : ℝ) :
  (∀ x : ℝ, (x + 2) * (x + 3) = x^2 + m*x + n) → m = 5 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_compare_l3036_303600


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l3036_303622

/-- Two lines L₁ and L₂ are defined as follows:
    L₁: ax + 3y + 1 = 0
    L₂: 2x + (a+1)y + 1 = 0
    This theorem proves that if L₁ and L₂ are parallel, then a = -3. -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax + 3*y + 1 = 0 ↔ 2*x + (a+1)*y + 1 = 0) →
  a = -3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l3036_303622


namespace NUMINAMATH_CALUDE_find_number_l3036_303642

theorem find_number (x : ℝ) : (0.05 * x = 0.2 * 650 + 190) → x = 6400 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3036_303642


namespace NUMINAMATH_CALUDE_equal_diagonals_implies_quad_or_pent_l3036_303672

/-- A convex n-gon with n ≥ 4 -/
structure ConvexNGon where
  n : ℕ
  convex : n ≥ 4

/-- The property that all diagonals of a polygon are equal -/
def all_diagonals_equal (F : ConvexNGon) : Prop := sorry

/-- The property that a polygon is a quadrilateral -/
def is_quadrilateral (F : ConvexNGon) : Prop := F.n = 4

/-- The property that a polygon is a pentagon -/
def is_pentagon (F : ConvexNGon) : Prop := F.n = 5

/-- Theorem: If all diagonals of a convex n-gon (n ≥ 4) are equal, 
    then it is either a quadrilateral or a pentagon -/
theorem equal_diagonals_implies_quad_or_pent (F : ConvexNGon) :
  all_diagonals_equal F → is_quadrilateral F ∨ is_pentagon F := by sorry

end NUMINAMATH_CALUDE_equal_diagonals_implies_quad_or_pent_l3036_303672


namespace NUMINAMATH_CALUDE_intersection_point_l3036_303651

theorem intersection_point (x y : ℚ) :
  (8 * x - 5 * y = 10) ∧ (9 * x + 4 * y = 20) ↔ x = 140 / 77 ∧ y = 70 / 77 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3036_303651


namespace NUMINAMATH_CALUDE_school_distance_proof_l3036_303691

/-- Represents the time taken to drive to school -/
structure DriveTime where
  rushHour : ℝ  -- Time in hours during rush hour
  holiday : ℝ   -- Time in hours during holiday

/-- Represents the speed of driving to school -/
structure DriveSpeed where
  rushHour : ℝ  -- Speed in miles per hour during rush hour
  holiday : ℝ   -- Speed in miles per hour during holiday

/-- The distance to school in miles -/
def distanceToSchool : ℝ := 10

theorem school_distance_proof (t : DriveTime) (s : DriveSpeed) : distanceToSchool = 10 :=
  by
  have h1 : t.rushHour = 1/2 := by sorry
  have h2 : t.holiday = 1/4 := by sorry
  have h3 : s.holiday = s.rushHour + 20 := by sorry
  have h4 : distanceToSchool = s.rushHour * t.rushHour := by sorry
  have h5 : distanceToSchool = s.holiday * t.holiday := by sorry
  sorry

#check school_distance_proof

end NUMINAMATH_CALUDE_school_distance_proof_l3036_303691


namespace NUMINAMATH_CALUDE_chocolate_pieces_per_box_l3036_303603

theorem chocolate_pieces_per_box 
  (total_boxes : ℕ) 
  (given_boxes : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : total_boxes = 14)
  (h2 : given_boxes = 8)
  (h3 : remaining_pieces = 18)
  (h4 : total_boxes > given_boxes) :
  (remaining_pieces / (total_boxes - given_boxes) : ℕ) = 3 := by
sorry

end NUMINAMATH_CALUDE_chocolate_pieces_per_box_l3036_303603


namespace NUMINAMATH_CALUDE_andrews_age_l3036_303602

/-- Proves that Andrew's current age is 30, given the donation information -/
theorem andrews_age (donation_start_age : ℕ) (annual_donation : ℕ) (total_donation : ℕ) :
  donation_start_age = 11 →
  annual_donation = 7 →
  total_donation = 133 →
  donation_start_age + (total_donation / annual_donation) = 30 :=
by sorry

end NUMINAMATH_CALUDE_andrews_age_l3036_303602


namespace NUMINAMATH_CALUDE_basement_water_pump_time_l3036_303685

/-- Proves that it takes 450 minutes to pump out water from a basement given specific conditions -/
theorem basement_water_pump_time : 
  let basement_length : ℝ := 30
  let basement_width : ℝ := 40
  let water_depth_inches : ℝ := 24
  let num_pumps : ℕ := 4
  let pump_rate : ℝ := 10  -- gallons per minute
  let cubic_foot_to_gallon : ℝ := 7.5
  let inches_per_foot : ℝ := 12

  let water_depth_feet : ℝ := water_depth_inches / inches_per_foot
  let water_volume_cubic_feet : ℝ := basement_length * basement_width * water_depth_feet
  let water_volume_gallons : ℝ := water_volume_cubic_feet * cubic_foot_to_gallon
  let total_pump_rate : ℝ := pump_rate * num_pumps
  let pump_time_minutes : ℝ := water_volume_gallons / total_pump_rate

  pump_time_minutes = 450 := by sorry

end NUMINAMATH_CALUDE_basement_water_pump_time_l3036_303685


namespace NUMINAMATH_CALUDE_quadratic_comparison_l3036_303687

/-- Given two quadratic functions A and B, prove that B can be expressed in terms of x
    and that A is always greater than B for all real x. -/
theorem quadratic_comparison (x : ℝ) : 
  let A := 3 * x^2 - 2 * x + 1
  let B := 2 * x^2 - x - 3
  (A + B = 5 * x^2 - 4 * x - 2) ∧ (A > B) := by sorry

end NUMINAMATH_CALUDE_quadratic_comparison_l3036_303687


namespace NUMINAMATH_CALUDE_gumball_difference_l3036_303638

theorem gumball_difference (x : ℤ) : 
  (19 * 3 ≤ 16 + 12 + x ∧ 16 + 12 + x ≤ 25 * 3) →
  (∃ (max min : ℤ), 
    (∀ y : ℤ, 19 * 3 ≤ 16 + 12 + y ∧ 16 + 12 + y ≤ 25 * 3 → y ≤ max) ∧
    (∀ y : ℤ, 19 * 3 ≤ 16 + 12 + y ∧ 16 + 12 + y ≤ 25 * 3 → min ≤ y) ∧
    max - min = 18) :=
by sorry

end NUMINAMATH_CALUDE_gumball_difference_l3036_303638


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3036_303625

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3036_303625


namespace NUMINAMATH_CALUDE_pastry_combinations_l3036_303618

/-- The number of ways to select pastries -/
def select_pastries (total : ℕ) (types : ℕ) : ℕ :=
  if types > total then 0
  else
    let remaining := total - types
    -- Ways to distribute remaining pastries among types
    (types^remaining + types * (types - 1) * remaining + Nat.choose types remaining) / Nat.factorial remaining

/-- Theorem: Selecting 8 pastries from 5 types, with at least one of each type, results in 25 combinations -/
theorem pastry_combinations : select_pastries 8 5 = 25 := by
  sorry


end NUMINAMATH_CALUDE_pastry_combinations_l3036_303618


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3036_303647

theorem book_pages_calculation (total_pages : ℕ) : 
  (7 : ℚ) / 13 * total_pages + 
  (5 : ℚ) / 9 * ((6 : ℚ) / 13 * total_pages) + 
  96 = total_pages → 
  total_pages = 468 := by
sorry

end NUMINAMATH_CALUDE_book_pages_calculation_l3036_303647


namespace NUMINAMATH_CALUDE_population_trend_l3036_303693

theorem population_trend (P₀ k : ℝ) (h₁ : P₀ > 0) (h₂ : -1 < k) (h₃ : k < 0) :
  ∀ n : ℕ, P₀ * (1 + k) ^ (n + 1) < P₀ * (1 + k) ^ n :=
by sorry

end NUMINAMATH_CALUDE_population_trend_l3036_303693


namespace NUMINAMATH_CALUDE_juniors_score_l3036_303623

/-- Given a class with juniors and seniors, prove the juniors' score -/
theorem juniors_score (n : ℕ) (junior_score : ℝ) :
  n > 0 →
  (0.2 * n : ℝ) * junior_score + (0.8 * n : ℝ) * 80 = n * 82 →
  junior_score = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_juniors_score_l3036_303623


namespace NUMINAMATH_CALUDE_negation_abc_zero_l3036_303633

theorem negation_abc_zero (a b c : ℝ) : (a = 0 ∨ b = 0 ∨ c = 0) → a * b * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_abc_zero_l3036_303633


namespace NUMINAMATH_CALUDE_sequence_with_differences_two_or_five_l3036_303675

theorem sequence_with_differences_two_or_five :
  ∃ (p : Fin 101 → Fin 101), Function.Bijective p ∧
    (∀ i : Fin 100, (p (i + 1) : ℕ) - (p i : ℕ) = 2 ∨ (p (i + 1) : ℕ) - (p i : ℕ) = 5 ∨
                    (p i : ℕ) - (p (i + 1) : ℕ) = 2 ∨ (p i : ℕ) - (p (i + 1) : ℕ) = 5) :=
by sorry


end NUMINAMATH_CALUDE_sequence_with_differences_two_or_five_l3036_303675


namespace NUMINAMATH_CALUDE_vector_problem_l3036_303650

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
noncomputable def b (y : ℝ) : ℝ × ℝ := (Real.cos y, Real.sin y)
noncomputable def c (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def f (x : ℝ) : ℝ := dot_product (a x) (c x)

noncomputable def g (m x : ℝ) : ℝ := f (x + m)

theorem vector_problem (x y : ℝ) 
  (h : ‖a x - b y‖ = 2 * Real.sqrt 5 / 5) : 
  (Real.cos (x - y) = 3 / 5) ∧ 
  (∃ (m : ℝ), m > 0 ∧ m = Real.pi / 4 ∧ 
    ∀ (n : ℝ), n > 0 → (∀ (t : ℝ), g n t = g n (-t)) → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l3036_303650


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l3036_303677

theorem arithmetic_square_root_of_one_fourth (x : ℝ) : x = Real.sqrt (1/4) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l3036_303677


namespace NUMINAMATH_CALUDE_special_quadrilateral_area_l3036_303619

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : Point)

-- Define the properties of the quadrilateral
def has_inscribed_circle (Q : Quadrilateral) : Prop := sorry
def has_circumscribed_circle (Q : Quadrilateral) : Prop := sorry
def perpendicular_diagonals (Q : Quadrilateral) : Prop := sorry
def circumradius (Q : Quadrilateral) : ℝ := sorry
def side_length_relation (Q : Quadrilateral) : Prop := sorry

-- Define the area of the quadrilateral
def area (Q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem special_quadrilateral_area 
  (Q : Quadrilateral) 
  (h1 : has_inscribed_circle Q)
  (h2 : has_circumscribed_circle Q)
  (h3 : perpendicular_diagonals Q)
  (h4 : circumradius Q = R)
  (h5 : side_length_relation Q) :
  area Q = (8 * R^2) / 5 := by sorry

end NUMINAMATH_CALUDE_special_quadrilateral_area_l3036_303619


namespace NUMINAMATH_CALUDE_initial_friends_correct_l3036_303676

/-- The number of friends initially playing the game -/
def initial_friends : ℕ := 8

/-- The number of additional players who joined -/
def additional_players : ℕ := 2

/-- The number of lives each player has -/
def lives_per_player : ℕ := 6

/-- The total number of lives after new players joined -/
def total_lives : ℕ := 60

/-- Theorem stating that the initial number of friends is correct -/
theorem initial_friends_correct :
  initial_friends * lives_per_player + additional_players * lives_per_player = total_lives :=
by sorry

end NUMINAMATH_CALUDE_initial_friends_correct_l3036_303676


namespace NUMINAMATH_CALUDE_equation_solution_l3036_303667

theorem equation_solution (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3036_303667


namespace NUMINAMATH_CALUDE_one_minus_repeating_8_l3036_303606

/-- The value of the repeating decimal 0.888... -/
def repeating_decimal_8 : ℚ := 8/9

/-- Proof that 1 - 0.888... = 1/9 -/
theorem one_minus_repeating_8 : 1 - repeating_decimal_8 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_8_l3036_303606


namespace NUMINAMATH_CALUDE_pool_capacity_l3036_303661

theorem pool_capacity (fill_time_both : ℝ) (fill_time_first : ℝ) (additional_rate : ℝ) :
  fill_time_both = 48 →
  fill_time_first = 120 →
  additional_rate = 50 →
  ∃ (capacity : ℝ),
    capacity = 12000 ∧
    capacity / fill_time_both = capacity / fill_time_first + (capacity / fill_time_first + additional_rate) :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_l3036_303661


namespace NUMINAMATH_CALUDE_f_properties_l3036_303697

def f (x : ℝ) : ℝ := |2*x - 1| + 1

theorem f_properties :
  (∀ x, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m, (∃ n, f n ≤ m - f (-n)) → m ≥ 4) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3036_303697


namespace NUMINAMATH_CALUDE_fixed_point_of_line_l3036_303653

theorem fixed_point_of_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_line_l3036_303653


namespace NUMINAMATH_CALUDE_square_exterior_points_diagonal_l3036_303610

-- Define the square ABCD
def square_side_length : ℝ := 15

-- Define the lengths BG, DH, AG, and CH
def BG : ℝ := 7
def DH : ℝ := 7
def AG : ℝ := 17
def CH : ℝ := 17

-- Define the theorem
theorem square_exterior_points_diagonal (A B C D G H : ℝ × ℝ) :
  let AB := square_side_length
  let AD := square_side_length
  (B.1 - G.1)^2 + (B.2 - G.2)^2 = BG^2 →
  (D.1 - H.1)^2 + (D.2 - H.2)^2 = DH^2 →
  (A.1 - G.1)^2 + (A.2 - G.2)^2 = AG^2 →
  (C.1 - H.1)^2 + (C.2 - H.2)^2 = CH^2 →
  (G.1 - H.1)^2 + (G.2 - H.2)^2 = 98 :=
by sorry


end NUMINAMATH_CALUDE_square_exterior_points_diagonal_l3036_303610


namespace NUMINAMATH_CALUDE_remainder_theorem_l3036_303662

theorem remainder_theorem (n m p q r : ℤ)
  (hn : n % 18 = 10)
  (hm : m % 27 = 16)
  (hp : p % 6 = 4)
  (hq : q % 12 = 8)
  (hr : r % 3 = 2) :
  ((3*n + 2*m) - (p + q) / r) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3036_303662


namespace NUMINAMATH_CALUDE_square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero_l3036_303649

theorem square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero :
  ∀ x : ℝ, x = Real.sqrt 5 - 1 → x^2 + 2*x - 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_square_root_five_minus_one_squared_plus_two_times_minus_four_equals_zero_l3036_303649


namespace NUMINAMATH_CALUDE_only_B_is_random_l3036_303617

-- Define the type for events
inductive Event
| A  -- A coin thrown from the ground will fall down
| B  -- A shooter hits the target with 10 points in one shot
| C  -- The sun rises from the east
| D  -- A horse runs at a speed of 70 meters per second

-- Define what it means for an event to be random
def is_random (e : Event) : Prop :=
  match e with
  | Event.A => false
  | Event.B => true
  | Event.C => false
  | Event.D => false

-- Theorem stating that only event B is random
theorem only_B_is_random :
  ∀ e : Event, is_random e ↔ e = Event.B :=
by
  sorry


end NUMINAMATH_CALUDE_only_B_is_random_l3036_303617


namespace NUMINAMATH_CALUDE_adam_figurines_l3036_303636

/-- The number of figurines that can be made from one block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be made from one block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be made from one block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The number of blocks of basswood Adam owns -/
def basswood_blocks : ℕ := 15

/-- The number of blocks of butternut wood Adam owns -/
def butternut_blocks : ℕ := 20

/-- The number of blocks of Aspen wood Adam owns -/
def aspen_blocks : ℕ := 20

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines

theorem adam_figurines : total_figurines = 245 := by
  sorry

end NUMINAMATH_CALUDE_adam_figurines_l3036_303636


namespace NUMINAMATH_CALUDE_solutions_of_fourth_power_equation_l3036_303639

theorem solutions_of_fourth_power_equation :
  let S : Set ℂ := {x | x^4 - 16 = 0}
  S = {2, -2, Complex.I * 2, -Complex.I * 2} := by
  sorry

end NUMINAMATH_CALUDE_solutions_of_fourth_power_equation_l3036_303639


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt2_over_2_l3036_303692

theorem complex_modulus_sqrt2_over_2 (z : ℂ) (h : z * Complex.I / (z - Complex.I) = 1) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt2_over_2_l3036_303692


namespace NUMINAMATH_CALUDE_mary_snake_observation_l3036_303690

/-- Given the number of breeding balls, snakes per ball, and total snakes observed,
    calculate the number of additional pairs of snakes. -/
def additional_snake_pairs (breeding_balls : ℕ) (snakes_per_ball : ℕ) (total_snakes : ℕ) : ℕ :=
  ((total_snakes - breeding_balls * snakes_per_ball) / 2)

/-- Theorem stating that given 3 breeding balls with 8 snakes each,
    and a total of 36 snakes observed, the number of additional pairs of snakes is 6. -/
theorem mary_snake_observation :
  additional_snake_pairs 3 8 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mary_snake_observation_l3036_303690


namespace NUMINAMATH_CALUDE_three_eighths_decimal_l3036_303652

theorem three_eighths_decimal : (3 : ℚ) / 8 = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_three_eighths_decimal_l3036_303652


namespace NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l3036_303684

theorem sum_of_opposite_sign_integers (a b : ℤ) : 
  (abs a = 3) → (abs b = 5) → (a * b < 0) → (a + b = -2 ∨ a + b = 2) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_opposite_sign_integers_l3036_303684


namespace NUMINAMATH_CALUDE_shaded_area_circles_l3036_303643

/-- Given a larger circle of radius 8 and two smaller circles touching the larger circle
    and each other at the center of the larger circle, the area of the shaded region
    (the area of the larger circle minus the areas of the two smaller circles) is 32π. -/
theorem shaded_area_circles (r : ℝ) (h : r = 8) : 
  r^2 * π - 2 * (r/2)^2 * π = 32 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_circles_l3036_303643


namespace NUMINAMATH_CALUDE_su_buqing_star_distance_l3036_303621

theorem su_buqing_star_distance (distance : ℝ) : 
  distance = 218000000 → distance = 2.18 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_su_buqing_star_distance_l3036_303621


namespace NUMINAMATH_CALUDE_child_tickets_sold_l3036_303654

/-- Proves the number of child tickets sold given ticket prices and total sales information -/
theorem child_tickets_sold 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_sales : ℕ) 
  (total_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : total_tickets = 42) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_sales ∧
    child_tickets = 16 :=
by sorry

end NUMINAMATH_CALUDE_child_tickets_sold_l3036_303654


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3036_303609

theorem quadratic_equation_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h1 : c^2 + c*c + d = 0) (h2 : d^2 + c*d + d = 0) : c = 1 ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3036_303609


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3036_303629

theorem smallest_n_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 4 * x = z^3) → x ≥ n) ∧
  n = 1080 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l3036_303629


namespace NUMINAMATH_CALUDE_digit_sum_subtraction_l3036_303646

theorem digit_sum_subtraction (n : ℕ) : 
  2010 ≤ n ∧ n ≤ 2019 → n - (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) = 2007 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_subtraction_l3036_303646


namespace NUMINAMATH_CALUDE_sqrt_calculation_l3036_303628

theorem sqrt_calculation :
  (Real.sqrt 80 - Real.sqrt 20 + Real.sqrt 5 = 3 * Real.sqrt 5) ∧
  (2 * Real.sqrt 6 * 3 * Real.sqrt (1/2) / Real.sqrt 3 = 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l3036_303628


namespace NUMINAMATH_CALUDE_ceiling_of_negative_decimal_l3036_303682

theorem ceiling_of_negative_decimal : ⌈(-3.87 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_decimal_l3036_303682


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l3036_303699

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 44) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l3036_303699


namespace NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l3036_303666

/-- If the system of equations ax + y = 1 and x + y = 2 has no solution, then a = 1 -/
theorem no_solution_implies_a_equals_one (a : ℝ) : 
  (∀ x y : ℝ, ¬(ax + y = 1 ∧ x + y = 2)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_equals_one_l3036_303666


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_five_l3036_303632

theorem cubic_fraction_equals_five :
  let a : ℚ := 3
  let b : ℚ := 2
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 := by sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_five_l3036_303632


namespace NUMINAMATH_CALUDE_largest_sample_number_l3036_303665

def systematic_sampling (total : ℕ) (start : ℕ) (interval : ℕ) : ℕ :=
  let sample_size := total / interval
  start + interval * (sample_size - 1)

theorem largest_sample_number :
  systematic_sampling 500 7 25 = 482 := by
  sorry

end NUMINAMATH_CALUDE_largest_sample_number_l3036_303665


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3036_303673

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (x - a) * (x - 2*a) < 0}
  S = if a < 0 then Set.Ioo (2*a) a
      else if a = 0 then ∅
      else Set.Ioo a (2*a) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3036_303673


namespace NUMINAMATH_CALUDE_right_triangle_longer_leg_l3036_303626

theorem right_triangle_longer_leg (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- b is the longer leg
  b ≤ c →            -- Longer leg is shorter than hypotenuse
  b = 60 :=          -- Conclusion: longer leg is 60
by
  sorry

#check right_triangle_longer_leg

end NUMINAMATH_CALUDE_right_triangle_longer_leg_l3036_303626


namespace NUMINAMATH_CALUDE_final_marble_difference_l3036_303689

/- Define the initial difference in marbles between Ed and Doug -/
def initial_difference : ℕ := 30

/- Define the number of marbles Ed lost -/
def marbles_lost : ℕ := 21

/- Define Ed's final number of marbles -/
def ed_final_marbles : ℕ := 91

/- Define Doug's number of marbles (which remains constant) -/
def doug_marbles : ℕ := ed_final_marbles + marbles_lost - initial_difference

/- Theorem stating the final difference in marbles -/
theorem final_marble_difference :
  ed_final_marbles - doug_marbles = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_final_marble_difference_l3036_303689


namespace NUMINAMATH_CALUDE_bounds_on_ratio_of_squares_l3036_303627

theorem bounds_on_ratio_of_squares (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ), 
    (∀ a b : ℝ, a ≠ 0 → b ≠ 0 → m ≤ |a + b|^2 / (|a|^2 + |b|^2) ∧ |a + b|^2 / (|a|^2 + |b|^2) ≤ M) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ |c + d|^2 / (|c|^2 + |d|^2) = m) ∧
    (∃ e f : ℝ, e ≠ 0 ∧ f ≠ 0 ∧ |e + f|^2 / (|e|^2 + |f|^2) = M) ∧
    M - m = 2 :=
by sorry

end NUMINAMATH_CALUDE_bounds_on_ratio_of_squares_l3036_303627


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3036_303608

theorem inequality_system_solution (x : ℝ) :
  (4 * x + 6 > 1 - x) ∧ (3 * (x - 1) ≤ x + 5) → -1 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3036_303608


namespace NUMINAMATH_CALUDE_equation_solution_l3036_303611

theorem equation_solution : ∃ x : ℚ, (5/100 * x + 12/100 * (30 + x) = 144/10) ∧ x = 108/17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3036_303611


namespace NUMINAMATH_CALUDE_sum_in_M_alpha_sum_l3036_303613

/-- The set of functions f(x) that satisfy the condition:
    For all x₁, x₂ ∈ ℝ and x₂ > x₁, -α(x₂ - x₁) < f(x₂) - f(x₁) < α(x₂ - x₁) -/
def M_alpha (α : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₂ > x₁ → -α * (x₂ - x₁) < f x₂ - f x₁ ∧ f x₂ - f x₁ < α * (x₂ - x₁)

/-- Theorem: If f ∈ Mα₁ and g ∈ Mα₂, then f + g ∈ Mα₁+α₂ -/
theorem sum_in_M_alpha_sum (α₁ α₂ : ℝ) (f g : ℝ → ℝ) 
  (hα₁ : α₁ > 0) (hα₂ : α₂ > 0)
  (hf : M_alpha α₁ f) (hg : M_alpha α₂ g) : 
  M_alpha (α₁ + α₂) (fun x ↦ f x + g x) :=
by sorry

end NUMINAMATH_CALUDE_sum_in_M_alpha_sum_l3036_303613


namespace NUMINAMATH_CALUDE_min_valid_configuration_l3036_303680

/-- Represents a configuration of two piles of bricks -/
structure BrickPiles where
  first : ℕ
  second : ℕ

/-- Checks if moving 100 bricks from the first pile to the second makes the second pile twice as large as the first -/
def satisfiesFirstCondition (piles : BrickPiles) : Prop :=
  2 * (piles.first - 100) = piles.second + 100

/-- Checks if there exists a number of bricks that can be moved from the second pile to the first to make the first pile six times as large as the second -/
def satisfiesSecondCondition (piles : BrickPiles) : Prop :=
  ∃ z : ℕ, piles.first + z = 6 * (piles.second - z)

/-- Checks if a given configuration satisfies both conditions -/
def isValidConfiguration (piles : BrickPiles) : Prop :=
  satisfiesFirstCondition piles ∧ satisfiesSecondCondition piles

/-- The main theorem stating the minimum valid configuration -/
theorem min_valid_configuration :
  ∀ piles : BrickPiles, isValidConfiguration piles →
  piles.first ≥ 170 ∧
  (piles.first = 170 → piles.second = 40) :=
by sorry

#check min_valid_configuration

end NUMINAMATH_CALUDE_min_valid_configuration_l3036_303680


namespace NUMINAMATH_CALUDE_B_power_99_l3036_303659

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  !![0, 0, 0;
     0, 0, 1;
     0, -1, 0]

theorem B_power_99 : B^99 = B := by sorry

end NUMINAMATH_CALUDE_B_power_99_l3036_303659


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3036_303624

/-- Given a line L1 with equation 4x - 2y + 1 = 0, prove that the line L2 passing through
    the point (2, -3) and perpendicular to L1 has the equation x + 2y + 4 = 0. -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x - 2 * y + 1 = 0
  let P : ℝ × ℝ := (2, -3)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y + 4 = 0
  (∀ x y, L1 x y → L2 x y → (y - P.2) = -(x - P.1)) →
  (∀ x y, L1 x y → L2 x y → (y - P.2) * (x - P.1) = -1) →
  L2 P.1 P.2 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l3036_303624


namespace NUMINAMATH_CALUDE_num_terms_eq_508020_l3036_303601

/-- The number of terms in the simplified expression of (x+y+z+w)^2008 + (x-y-z-w)^2008 -/
def num_terms : ℕ :=
  let n := 2008
  let sum := (n / 2 + 1)^2 - (n / 2) * (n / 2 + 1) / 2
  sum

/-- Theorem stating that the number of terms in the simplified expression
    of (x+y+z+w)^2008 + (x-y-z-w)^2008 is equal to 508020 -/
theorem num_terms_eq_508020 : num_terms = 508020 := by
  sorry

end NUMINAMATH_CALUDE_num_terms_eq_508020_l3036_303601


namespace NUMINAMATH_CALUDE_second_month_sale_l3036_303694

/-- Represents the sales data for a grocer over six months -/
structure GrocerSales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ

/-- Theorem: Given the sales for five months and the average sale,
    prove that the sale in the second month was 7000 -/
theorem second_month_sale
  (sales : GrocerSales)
  (h1 : sales.month1 = 6400)
  (h3 : sales.month3 = 6800)
  (h4 : sales.month4 = 7200)
  (h5 : sales.month5 = 6500)
  (h6 : sales.month6 = 5100)
  (avg : (sales.month1 + sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6) / 6 = 6500) :
  sales.month2 = 7000 := by
  sorry

end NUMINAMATH_CALUDE_second_month_sale_l3036_303694


namespace NUMINAMATH_CALUDE_circle_center_l3036_303620

/-- The center of a circle described by the equation x^2 - 6x + y^2 + 2y = 20 is (3, -1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 - 6*x + y^2 + 2*y = 20) → 
  ∃ (h : ℝ) (k : ℝ) (r : ℝ), 
    h = 3 ∧ k = -1 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l3036_303620


namespace NUMINAMATH_CALUDE_probability_is_half_l3036_303607

/-- Represents a circular field with 6 equally spaced roads -/
structure CircularField :=
  (radius : ℝ)
  (num_roads : Nat)
  (road_angle : ℝ)

/-- Represents a geologist's position after traveling -/
structure GeologistPosition :=
  (road : Nat)
  (distance : ℝ)

/-- Calculates the distance between two geologists -/
def distance_between (field : CircularField) (pos1 pos2 : GeologistPosition) : ℝ :=
  sorry

/-- Determines if two roads are neighboring -/
def are_neighboring (field : CircularField) (road1 road2 : Nat) : Bool :=
  sorry

/-- Calculates the probability of two geologists being more than 8 km apart -/
def probability_more_than_8km (field : CircularField) (speed : ℝ) (time : ℝ) : ℝ :=
  sorry

/-- Main theorem: Probability of geologists being more than 8 km apart is 0.5 -/
theorem probability_is_half (field : CircularField) :
  probability_more_than_8km field 5 1 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l3036_303607


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l3036_303679

theorem sum_of_square_areas (square1_side : ℝ) (square2_side : ℝ) 
  (h1 : square1_side = 8) (h2 : square2_side = 10) : 
  square1_side^2 + square2_side^2 = 164 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l3036_303679


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l3036_303614

/-- Converts a number from base 5 to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- The number given in the problem in base 5 --/
def problemNumber : List Nat := [3, 3, 0, 4, 2, 0, 3, 1, 2]

/-- The decimal representation of the problem number --/
def decimalNumber : Nat := base5ToDecimal problemNumber

/-- Checks if a number is prime --/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Theorem stating the largest prime divisor of the problem number --/
theorem largest_prime_divisor :
  ∃ (p : Nat), p = 11019 ∧ 
    isPrime p ∧ 
    (decimalNumber % p = 0) ∧
    (∀ q : Nat, isPrime q → decimalNumber % q = 0 → q ≤ p) :=
by sorry


end NUMINAMATH_CALUDE_largest_prime_divisor_l3036_303614


namespace NUMINAMATH_CALUDE_rectangle_area_l3036_303631

/-- The area of a rectangle with perimeter 90 feet and length three times the width is 380.15625 square feet. -/
theorem rectangle_area (w : ℝ) (l : ℝ) (h1 : 2 * l + 2 * w = 90) (h2 : l = 3 * w) :
  l * w = 380.15625 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3036_303631


namespace NUMINAMATH_CALUDE_cos_three_pi_halves_l3036_303696

theorem cos_three_pi_halves : Real.cos (3 * π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_three_pi_halves_l3036_303696


namespace NUMINAMATH_CALUDE_pet_shop_total_cost_l3036_303640

/-- The cost of purchasing all pets in a pet shop given specific conditions. -/
theorem pet_shop_total_cost :
  let num_puppies : ℕ := 2
  let num_kittens : ℕ := 2
  let num_parakeets : ℕ := 3
  let parakeet_cost : ℕ := 10
  let puppy_cost : ℕ := 3 * parakeet_cost
  let kitten_cost : ℕ := 2 * parakeet_cost
  num_puppies * puppy_cost + num_kittens * kitten_cost + num_parakeets * parakeet_cost = 130 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_total_cost_l3036_303640


namespace NUMINAMATH_CALUDE_seventh_term_is_2187_l3036_303637

/-- A geometric sequence of positive integers -/
structure GeometricSequence where
  a : ℕ → ℕ  -- The sequence
  r : ℕ      -- The common ratio
  first_term : a 1 = 3
  ratio_def : ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_is_2187 (seq : GeometricSequence) (h : seq.a 6 = 972) :
  seq.a 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_2187_l3036_303637


namespace NUMINAMATH_CALUDE_candy_given_to_haley_l3036_303686

def initial_candy : ℕ := 15
def remaining_candy : ℕ := 9

theorem candy_given_to_haley : initial_candy - remaining_candy = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_given_to_haley_l3036_303686


namespace NUMINAMATH_CALUDE_comic_book_collections_l3036_303656

/-- Kymbrea's initial comic book collection -/
def kymbrea_initial : ℕ := 50

/-- Kymbrea's monthly comic book collection rate -/
def kymbrea_rate : ℕ := 3

/-- LaShawn's initial comic book collection -/
def lashawn_initial : ℕ := 20

/-- LaShawn's monthly comic book collection rate -/
def lashawn_rate : ℕ := 7

/-- The number of months after which LaShawn's collection is twice Kymbrea's -/
def months : ℕ := 80

theorem comic_book_collections : 
  lashawn_initial + lashawn_rate * months = 2 * (kymbrea_initial + kymbrea_rate * months) := by
  sorry

end NUMINAMATH_CALUDE_comic_book_collections_l3036_303656


namespace NUMINAMATH_CALUDE_sum_square_diagonals_formula_l3036_303698

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  R : ℝ  -- radius of the circumscribed circle
  OP : ℝ  -- length of segment OP
  h_R_pos : R > 0  -- radius is positive
  h_OP_pos : OP > 0  -- OP is positive
  h_OP_le_2R : OP ≤ 2 * R  -- OP cannot be longer than the diameter

/-- The sum of squares of diagonals of an inscribed quadrilateral -/
def sumSquareDiagonals (q : InscribedQuadrilateral) : ℝ :=
  8 * q.R^2 - 4 * q.OP^2

/-- Theorem: The sum of squares of diagonals of an inscribed quadrilateral
    is equal to 8R^2 - 4OP^2 -/
theorem sum_square_diagonals_formula (q : InscribedQuadrilateral) :
  sumSquareDiagonals q = 8 * q.R^2 - 4 * q.OP^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_diagonals_formula_l3036_303698


namespace NUMINAMATH_CALUDE_same_day_after_313_weeks_l3036_303674

/-- The day of the week is represented as an integer from 0 to 6 -/
def DayOfWeek := Fin 7

/-- The number of weeks that have passed -/
def weeks : ℕ := 313

/-- Given an initial day of the week, returns the day of the week after a specified number of weeks -/
def day_after_weeks (initial_day : DayOfWeek) (n : ℕ) : DayOfWeek :=
  ⟨(initial_day.val + 7 * n) % 7, by sorry⟩

/-- Theorem: The day of the week remains the same after exactly 313 weeks -/
theorem same_day_after_313_weeks (d : DayOfWeek) : 
  day_after_weeks d weeks = d := by sorry

end NUMINAMATH_CALUDE_same_day_after_313_weeks_l3036_303674


namespace NUMINAMATH_CALUDE_elliptical_cylinder_stability_l3036_303630

/-- A cylinder with an elliptical cross-section -/
structure EllipticalCylinder where
  a : ℝ
  b : ℝ
  h : a > b

/-- Stability condition for an elliptical cylinder -/
def is_stable (c : EllipticalCylinder) : Prop :=
  c.b / c.a < 1 / Real.sqrt 2

/-- Theorem: An elliptical cylinder is in stable equilibrium iff b/a < 1/√2 -/
theorem elliptical_cylinder_stability (c : EllipticalCylinder) :
  is_stable c ↔ c.b / c.a < 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_elliptical_cylinder_stability_l3036_303630
