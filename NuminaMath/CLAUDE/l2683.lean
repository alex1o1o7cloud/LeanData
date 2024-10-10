import Mathlib

namespace special_triangle_sum_l2683_268377

/-- A right triangle with a special inscribed circle -/
structure SpecialTriangle where
  /-- The perimeter of the triangle -/
  perimeter : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The distance from the center of the circle to one vertex -/
  center_to_vertex : ℝ
  /-- The numerator of the fraction representing center_to_vertex -/
  p : ℕ
  /-- The denominator of the fraction representing center_to_vertex -/
  q : ℕ
  /-- The perimeter is 180 -/
  perimeter_eq : perimeter = 180
  /-- The radius is 25 -/
  radius_eq : radius = 25
  /-- center_to_vertex is equal to p/q -/
  center_to_vertex_eq : center_to_vertex = p / q
  /-- p and q are coprime -/
  coprime : Nat.Coprime p q

/-- The main theorem -/
theorem special_triangle_sum (t : SpecialTriangle) : t.p + t.q = 145 := by
  sorry

end special_triangle_sum_l2683_268377


namespace triangular_pyramid_volume_l2683_268367

-- Define a triangular pyramid with edge length √2
def TriangularPyramid := {edge_length : ℝ // edge_length = Real.sqrt 2}

-- Define the volume of a triangular pyramid
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  -- The actual calculation of the volume is not implemented here
  -- We're just declaring that such a function exists
  sorry

-- Theorem statement
theorem triangular_pyramid_volume (p : TriangularPyramid) : volume p = 1/3 := by
  sorry

end triangular_pyramid_volume_l2683_268367


namespace sqrt_2y_lt_3y_iff_y_gt_2_div_9_l2683_268391

theorem sqrt_2y_lt_3y_iff_y_gt_2_div_9 :
  ∀ y : ℝ, y > 0 → (Real.sqrt (2 * y) < 3 * y ↔ y > 2 / 9) := by
  sorry

end sqrt_2y_lt_3y_iff_y_gt_2_div_9_l2683_268391


namespace geometric_sequence_decreasing_l2683_268322

/-- A geometric sequence with first term a₁ and common ratio q. -/
def GeometricSequence (a₁ q : ℝ) : ℕ → ℝ := fun n ↦ a₁ * q ^ (n - 1)

/-- A sequence is decreasing if each term is less than the previous term. -/
def IsDecreasing (s : ℕ → ℝ) : Prop := ∀ n : ℕ, s (n + 1) < s n

theorem geometric_sequence_decreasing (a₁ q : ℝ) (h1 : a₁ * (q - 1) < 0) (h2 : q > 0) :
  IsDecreasing (GeometricSequence a₁ q) := by
  sorry

end geometric_sequence_decreasing_l2683_268322


namespace prob_B_given_A_l2683_268315

/-- Represents the number of male students -/
def num_male : ℕ := 3

/-- Represents the number of female students -/
def num_female : ℕ := 2

/-- Represents the total number of students -/
def total_students : ℕ := num_male + num_female

/-- Event A: drawing two students of the same gender -/
def prob_A : ℚ := (num_male.choose 2 + num_female.choose 2) / total_students.choose 2

/-- Event B: drawing two female students -/
def prob_B : ℚ := num_female.choose 2 / total_students.choose 2

/-- Event AB: drawing two female students given that two students of the same gender were drawn -/
def prob_AB : ℚ := num_female.choose 2 / total_students.choose 2

/-- Theorem stating that the probability of drawing two female students given that 
    two students of the same gender were drawn is 1/4 -/
theorem prob_B_given_A : prob_AB / prob_A = 1 / 4 := by
  sorry

end prob_B_given_A_l2683_268315


namespace jack_sandwich_change_l2683_268314

/-- Calculates the change Jack receives after buying sandwiches -/
theorem jack_sandwich_change :
  let sandwich1_price : ℚ := 5
  let sandwich2_price : ℚ := 5
  let sandwich3_price : ℚ := 6
  let sandwich4_price : ℚ := 7
  let discount1 : ℚ := 0.1
  let discount2 : ℚ := 0.1
  let discount3 : ℚ := 0.15
  let discount4 : ℚ := 0
  let tax_rate : ℚ := 0.05
  let service_fee : ℚ := 2
  let payment : ℚ := 34

  let total_cost : ℚ :=
    (sandwich1_price * (1 - discount1) +
     sandwich2_price * (1 - discount2) +
     sandwich3_price * (1 - discount3) +
     sandwich4_price * (1 - discount4)) *
    (1 + tax_rate) + service_fee

  payment - total_cost = 9.84
:= by sorry

end jack_sandwich_change_l2683_268314


namespace h_of_3_eq_72_minus_18_sqrt_15_l2683_268344

/-- Given functions f, g, and h defined as:
  f(x) = 3x + 6
  g(x) = (√(f(x)) - 3)²
  h(x) = f(g(x))
  Prove that h(3) = 72 - 18√15 -/
theorem h_of_3_eq_72_minus_18_sqrt_15 :
  let f : ℝ → ℝ := λ x ↦ 3 * x + 6
  let g : ℝ → ℝ := λ x ↦ (Real.sqrt (f x) - 3)^2
  let h : ℝ → ℝ := λ x ↦ f (g x)
  h 3 = 72 - 18 * Real.sqrt 15 := by
  sorry

end h_of_3_eq_72_minus_18_sqrt_15_l2683_268344


namespace m_minus_n_value_l2683_268385

theorem m_minus_n_value (m n : ℤ) 
  (h1 : |m| = 4) 
  (h2 : |n| = 6) 
  (h3 : m + n = |m + n|) : 
  m - n = -2 ∨ m - n = -10 := by
  sorry

end m_minus_n_value_l2683_268385


namespace john_door_replacement_l2683_268337

def outside_door_cost : ℕ := 20
def bedroom_door_count : ℕ := 3
def total_cost : ℕ := 70

def outside_door_count : ℕ := 2

theorem john_door_replacement :
  ∃ (x : ℕ),
    x * outside_door_cost + 
    bedroom_door_count * (outside_door_cost / 2) = 
    total_cost ∧
    x = outside_door_count :=
by sorry

end john_door_replacement_l2683_268337


namespace least_n_satisfying_inequality_l2683_268335

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 4 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ 1 / 15) ∧ 
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < 1 / 15) := by
  sorry

end least_n_satisfying_inequality_l2683_268335


namespace symmetric_point_theorem_l2683_268363

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry of a point with respect to the origin -/
def symmetricPoint (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Theorem: The symmetric point of P(3, 2) with respect to the origin is (-3, -2) -/
theorem symmetric_point_theorem :
  let P : Point := { x := 3, y := 2 }
  let P' : Point := symmetricPoint P
  P'.x = -3 ∧ P'.y = -2 := by
  sorry

end symmetric_point_theorem_l2683_268363


namespace parabola_tangent_through_origin_l2683_268339

theorem parabola_tangent_through_origin (c : ℝ) : 
  (∃ y : ℝ, (y = (-2)^2 - (-2) + c) ∧ 
   (0 = y + 5 * 2)) → c = 4 := by
  sorry

end parabola_tangent_through_origin_l2683_268339


namespace rain_all_three_days_l2683_268381

def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.2

theorem rain_all_three_days :
  let prob_all_days := prob_rain_friday * prob_rain_saturday * prob_rain_sunday
  prob_all_days = 0.04 := by sorry

end rain_all_three_days_l2683_268381


namespace smallest_k_for_digit_sum_l2683_268375

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The product of 9 and (10^k - 1) -/
def special_product (k : ℕ) : ℕ := 9 * (10^k - 1)

/-- The statement to prove -/
theorem smallest_k_for_digit_sum : 
  (∀ k < 167, sum_of_digits (special_product k) < 1500) ∧ 
  sum_of_digits (special_product 167) ≥ 1500 := by sorry

end smallest_k_for_digit_sum_l2683_268375


namespace max_package_volume_l2683_268346

/-- Represents the dimensions of a rectangular package. -/
structure PackageDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular package. -/
def volume (d : PackageDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the rope length required for a package. -/
def ropeLength (d : PackageDimensions) : ℝ :=
  2 * d.length + 4 * d.width + 6 * d.height

/-- The total available rope length in centimeters. -/
def totalRopeLength : ℝ := 360

/-- Theorem stating that the maximum volume of a package is 36000 cubic centimeters
    given the rope length and wrapping constraints. -/
theorem max_package_volume :
  ∃ (d : PackageDimensions),
    ropeLength d = totalRopeLength ∧
    ∀ (d' : PackageDimensions),
      ropeLength d' = totalRopeLength →
      volume d' ≤ volume d ∧
      volume d = 36000 :=
sorry

end max_package_volume_l2683_268346


namespace increase_when_multiplied_l2683_268349

theorem increase_when_multiplied (n : ℕ) (m : ℕ) (increase : ℕ) : n = 25 → m = 16 → increase = m * n - n → increase = 375 := by
  sorry

end increase_when_multiplied_l2683_268349


namespace sin_390_degrees_l2683_268311

theorem sin_390_degrees : Real.sin (390 * π / 180) = 1 / 2 := by
  sorry

end sin_390_degrees_l2683_268311


namespace solve_for_x_l2683_268365

theorem solve_for_x (x y : ℚ) (h1 : x / y = 10 / 4) (h2 : y = 18) : x = 45 := by
  sorry

end solve_for_x_l2683_268365


namespace largest_circle_radius_is_two_l2683_268340

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b
  h_a_ge_b : a ≥ b

/-- Represents a circle with center (c, 0) and radius r -/
structure Circle where
  c : ℝ
  r : ℝ
  h_positive_r : 0 < r

/-- Returns true if the circle is entirely contained within the ellipse -/
def circleInEllipse (e : Ellipse) (c : Circle) : Prop :=
  ∀ x y : ℝ, (x - c.c)^2 + y^2 = c.r^2 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- Returns true if the circle is tangent to the ellipse -/
def circleTangentToEllipse (e : Ellipse) (c : Circle) : Prop :=
  ∃ x y : ℝ, (x - c.c)^2 + y^2 = c.r^2 ∧ x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating that the largest circle centered at a focus of the ellipse
    and entirely contained within it has radius 2 -/
theorem largest_circle_radius_is_two (e : Ellipse) (c : Circle) 
    (h_a : e.a = 7) (h_b : e.b = 5) (h_c : c.c = 2 * Real.sqrt 6) : 
    circleInEllipse e c ∧ circleTangentToEllipse e c → c.r = 2 := by
  sorry

end largest_circle_radius_is_two_l2683_268340


namespace quadratic_maximum_l2683_268330

theorem quadratic_maximum : 
  (∃ (s : ℝ), -3 * s^2 + 24 * s + 5 = 53) ∧ 
  (∀ (s : ℝ), -3 * s^2 + 24 * s + 5 ≤ 53) := by
sorry

end quadratic_maximum_l2683_268330


namespace percentage_men_science_majors_l2683_268313

/-- Represents the composition of a college class -/
structure ClassComposition where
  total : ℝ
  women : ℝ
  men : ℝ
  scienceMajors : ℝ
  womenScienceMajors : ℝ
  nonScienceMajors : ℝ

/-- Theorem stating the percentage of men who are science majors -/
theorem percentage_men_science_majors (c : ClassComposition) : 
  c.total > 0 ∧ 
  c.women = 0.6 * c.total ∧ 
  c.men = 0.4 * c.total ∧ 
  c.nonScienceMajors = 0.6 * c.total ∧
  c.womenScienceMajors = 0.2 * c.women →
  (c.scienceMajors - c.womenScienceMajors) / c.men = 0.7 := by
  sorry

#check percentage_men_science_majors

end percentage_men_science_majors_l2683_268313


namespace rectangular_plot_area_l2683_268353

/-- 
Given a rectangular plot where the length is thrice the breadth and the breadth is 17 meters,
prove that the area of the plot is 867 square meters.
-/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 17 →
  length = 3 * breadth →
  area = length * breadth →
  area = 867 := by
sorry

end rectangular_plot_area_l2683_268353


namespace line_properties_l2683_268307

/-- A line with slope -3 and x-intercept (8, 0) has y-intercept (0, 24) and the point 4 units
    to the left of the x-intercept has coordinates (4, 12) -/
theorem line_properties (f : ℝ → ℝ) (h_slope : ∀ x y, f y - f x = -3 * (y - x))
    (h_x_intercept : f 8 = 0) :
  f 0 = 24 ∧ f 4 = 12 := by
  sorry

end line_properties_l2683_268307


namespace inequality_proof_l2683_268384

theorem inequality_proof (x y z : ℝ) (h : 2 * x + y^2 + z^2 ≤ 2) : x + y + z ≤ 2 := by
  sorry

end inequality_proof_l2683_268384


namespace min_valid_subset_l2683_268329

def isValid (S : Finset ℕ) : Prop :=
  ∀ n : ℕ, n ≤ 20 → (n ∈ S ∨ ∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a + b = n)

theorem min_valid_subset :
  ∃ S : Finset ℕ,
    S ⊆ Finset.range 11 \ {0} ∧
    Finset.card S = 6 ∧
    isValid S ∧
    ∀ T : Finset ℕ, T ⊆ Finset.range 11 \ {0} → Finset.card T < 6 → ¬isValid T :=
  sorry

end min_valid_subset_l2683_268329


namespace solve_equation_l2683_268390

-- Define the original equation
def original_equation (x a : ℚ) : Prop :=
  (2 * x - 1) / 5 + 1 = (x + a) / 2

-- Define the incorrect equation after clearing denominators (with the mistake)
def incorrect_equation (x a : ℚ) : Prop :=
  2 * (2 * x - 1) + 1 = 5 * (x + a)

-- Theorem statement
theorem solve_equation :
  ∀ a : ℚ, 
    (∃ x : ℚ, incorrect_equation x a ∧ x = 4) →
    (a = -1 ∧ ∃ x : ℚ, original_equation x (-1) ∧ x = 13) :=
by sorry

end solve_equation_l2683_268390


namespace power_of_power_l2683_268312

theorem power_of_power (a : ℝ) : (a^2)^4 = a^8 := by sorry

end power_of_power_l2683_268312


namespace power_of_half_squared_times_32_l2683_268399

theorem power_of_half_squared_times_32 : ∃ x : ℝ, x * (1/2)^2 = 2^3 ∧ x = 32 := by
  sorry

end power_of_half_squared_times_32_l2683_268399


namespace find_divisor_l2683_268382

theorem find_divisor (dividend : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = 507 → quotient = 61 → remainder = 19 →
  ∃ (divisor : Nat), dividend = divisor * quotient + remainder ∧ divisor = 8 := by
  sorry

end find_divisor_l2683_268382


namespace triangle_side_length_l2683_268320

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, a + b = 6, and (a*cos B + b*cos A) / c = 2*cos C, then c = 2√3 -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3 →
  a + b = 6 →
  (a * Real.cos B + b * Real.cos A) / c = 2 * Real.cos C →
  c = 2 * Real.sqrt 3 := by
  sorry


end triangle_side_length_l2683_268320


namespace triangle_angle_measure_l2683_268376

theorem triangle_angle_measure (a b c A B C : Real) : 
  a = 4 →
  b = 4 * Real.sqrt 3 →
  A = π / 6 →
  a * Real.sin B = b * Real.sin A →
  (B = π / 3 ∨ B = 2 * π / 3) := by
  sorry

end triangle_angle_measure_l2683_268376


namespace frequency_distribution_required_l2683_268398

/-- Represents a sample of data -/
structure Sample (α : Type*) where
  data : List α

/-- Represents a frequency distribution of a sample -/
def FrequencyDistribution (α : Type*) := α → ℕ

/-- Represents a range of values -/
structure Range (α : Type*) where
  lower : α
  upper : α

/-- Function to determine if a value is within a range -/
def inRange {α : Type*} [PartialOrder α] (x : α) (r : Range α) : Prop :=
  r.lower ≤ x ∧ x ≤ r.upper

/-- Theorem stating that a frequency distribution is required to understand 
    the proportion of a sample within a certain range -/
theorem frequency_distribution_required 
  {α : Type*} [PartialOrder α] (s : Sample α) (r : Range α) :
  ∃ (fd : FrequencyDistribution α), 
    (∀ x, x ∈ s.data → inRange x r → fd x > 0) ∧
    (∀ x, x ∉ s.data ∨ ¬inRange x r → fd x = 0) :=
sorry

end frequency_distribution_required_l2683_268398


namespace lowest_discount_l2683_268334

theorem lowest_discount (cost_price marked_price : ℝ) (min_profit_margin : ℝ) : 
  cost_price = 100 → 
  marked_price = 150 → 
  min_profit_margin = 0.05 → 
  ∃ (discount : ℝ), 
    discount = 0.7 ∧ 
    marked_price * discount = cost_price * (1 + min_profit_margin) ∧
    ∀ (d : ℝ), d > discount → marked_price * d > cost_price * (1 + min_profit_margin) :=
by sorry


end lowest_discount_l2683_268334


namespace cubic_real_root_l2683_268301

/-- The cubic equation with real coefficients c and d, having -3 - 4i as a root, has -4 as its real root -/
theorem cubic_real_root (c d : ℝ) (h : c * (-3 - 4*I)^3 + 4 * (-3 - 4*I)^2 + d * (-3 - 4*I) - 100 = 0) :
  ∃ x : ℝ, c * x^3 + 4 * x^2 + d * x - 100 = 0 ∧ x = -4 := by
  sorry

end cubic_real_root_l2683_268301


namespace prob_no_absolute_winner_is_correct_l2683_268397

/-- Represents a player in the mini-tournament -/
inductive Player : Type
| Alyosha : Player
| Borya : Player
| Vasya : Player

/-- Represents the result of a match between two players -/
def MatchResult (p1 p2 : Player) : Type := Bool

/-- The probability that Alyosha wins against Borya -/
def prob_Alyosha_wins_Borya : ℝ := 0.6

/-- The probability that Borya wins against Vasya -/
def prob_Borya_wins_Vasya : ℝ := 0.4

/-- The score of a player in the mini-tournament -/
def score (p : Player) (results : Π p1 p2 : Player, MatchResult p1 p2) : ℕ :=
  sorry

/-- There is an absolute winner if one player has a score of 2 -/
def has_absolute_winner (results : Π p1 p2 : Player, MatchResult p1 p2) : Prop :=
  ∃ p : Player, score p results = 2

/-- The probability of no absolute winner in the mini-tournament -/
def prob_no_absolute_winner : ℝ :=
  sorry

theorem prob_no_absolute_winner_is_correct :
  prob_no_absolute_winner = 0.24 :=
sorry

end prob_no_absolute_winner_is_correct_l2683_268397


namespace range_of_m_minus_one_times_n_minus_one_l2683_268383

/-- The function f(x) = |x^2 - 2x - 1| -/
def f (x : ℝ) : ℝ := abs (x^2 - 2*x - 1)

/-- Theorem stating the range of (m-1)(n-1) given the conditions -/
theorem range_of_m_minus_one_times_n_minus_one 
  (m n : ℝ) 
  (h1 : m > n) 
  (h2 : n > 1) 
  (h3 : f m = f n) : 
  0 < (m - 1) * (n - 1) ∧ (m - 1) * (n - 1) < 2 := by
  sorry

end range_of_m_minus_one_times_n_minus_one_l2683_268383


namespace even_function_monotone_interval_l2683_268368

theorem even_function_monotone_interval
  (ω φ : ℝ) (x₁ x₂ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < π)
  (h_even : ∀ x, 2 * Real.sin (ω * x + φ) = 2 * Real.sin (ω * (-x) + φ))
  (h_intersect : 2 * Real.sin (ω * x₁ + φ) = 2 ∧ 2 * Real.sin (ω * x₂ + φ) = 2)
  (h_min_distance : ∀ x y, 2 * Real.sin (ω * x + φ) = 2 → 2 * Real.sin (ω * y + φ) = 2 → |x - y| ≥ π)
  (h_exists_min : ∃ x y, 2 * Real.sin (ω * x + φ) = 2 ∧ 2 * Real.sin (ω * y + φ) = 2 ∧ |x - y| = π) :
  ∃ a b, a = -π/2 ∧ b = -π/4 ∧
    ∀ x y, a < x ∧ x < y ∧ y < b →
      2 * Real.sin (ω * x + φ) < 2 * Real.sin (ω * y + φ) :=
by sorry

end even_function_monotone_interval_l2683_268368


namespace geometric_sequence_fourth_term_l2683_268324

theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 2 = 2 →
  a 6 = 32 →
  a 4 = 8 := by
sorry

end geometric_sequence_fourth_term_l2683_268324


namespace death_rate_per_two_seconds_prove_death_rate_l2683_268318

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate in people per two seconds -/
def birth_rate : ℕ := 4

/-- Represents the net population increase per day -/
def net_increase_per_day : ℕ := 86400

/-- Theorem stating the death rate in people per two seconds -/
theorem death_rate_per_two_seconds : ℕ :=
  2

/-- Proof of the death rate given the birth rate and net population increase -/
theorem prove_death_rate : death_rate_per_two_seconds = 2 := by
  sorry


end death_rate_per_two_seconds_prove_death_rate_l2683_268318


namespace percentage_increase_l2683_268357

theorem percentage_increase (x y p : ℝ) : 
  y = x * (1 + p / 100) →
  y = 150 →
  x = 120 →
  p = 25 := by
sorry

end percentage_increase_l2683_268357


namespace x_gt_one_sufficient_not_necessary_l2683_268348

theorem x_gt_one_sufficient_not_necessary :
  (∀ x : ℝ, x > 1 → x^2 > x) ∧
  (∃ x : ℝ, x^2 > x ∧ ¬(x > 1)) :=
by sorry

end x_gt_one_sufficient_not_necessary_l2683_268348


namespace reduce_to_less_than_100_l2683_268302

/-- Represents a digit from 4 to 9 -/
inductive ValidDigit
  | four
  | five
  | six
  | seven
  | eight
  | nine

/-- Represents a natural number composed of ValidDigits -/
def ValidNumber := List ValidDigit

/-- Represents an operation that can be performed on a ValidNumber -/
inductive Operation
  | deletePair (d : ValidDigit) : Operation
  | deleteDoublePair (d1 d2 : ValidDigit) : Operation
  | insertPair (d : ValidDigit) : Operation
  | insertDoublePair (d1 d2 : ValidDigit) : Operation

/-- Applies an operation to a ValidNumber -/
def applyOperation (n : ValidNumber) (op : Operation) : ValidNumber :=
  sorry

/-- Checks if a ValidNumber is less than 100 -/
def isLessThan100 (n : ValidNumber) : Prop :=
  sorry

theorem reduce_to_less_than_100 (n : ValidNumber) (h : n.length = 2019) :
  ∃ (ops : List Operation), isLessThan100 (ops.foldl applyOperation n) :=
  sorry

end reduce_to_less_than_100_l2683_268302


namespace zachary_pushups_l2683_268308

theorem zachary_pushups (david_pushups : ℕ) (difference : ℕ) (zachary_pushups : ℕ) 
  (h1 : david_pushups = 37)
  (h2 : david_pushups = zachary_pushups + difference)
  (h3 : difference = 30) : 
  zachary_pushups = 7 := by
  sorry

end zachary_pushups_l2683_268308


namespace triangle_similarity_theorem_l2683_268333

-- Define the triangles and their side lengths
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

def triangle_PQR : Triangle :=
  { side1 := 10,
    side2 := 12,
    side3 := 0 }  -- We don't know the length of PR

def triangle_STU : Triangle :=
  { side1 := 5,
    side2 := 0,  -- We need to prove this is 6
    side3 := 0 }  -- We need to prove this is 6

-- Define similarity of triangles
def similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.side1 = k * t1.side1 ∧
    t2.side2 = k * t1.side2 ∧
    t2.side3 = k * t1.side3

-- Define the theorem
theorem triangle_similarity_theorem :
  similar triangle_PQR triangle_STU →
  triangle_STU.side2 = 6 ∧
  triangle_STU.side3 = 6 ∧
  triangle_STU.side1 + triangle_STU.side2 + triangle_STU.side3 = 17 :=
by
  sorry

end triangle_similarity_theorem_l2683_268333


namespace intersection_of_A_and_B_l2683_268392

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2683_268392


namespace jimin_tangerines_l2683_268306

/-- Given an initial number of tangerines and a number of eaten tangerines,
    calculate the number of tangerines left. -/
def tangerines_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that given 12 initial tangerines and 7 eaten tangerines,
    the number of tangerines left is 5. -/
theorem jimin_tangerines :
  tangerines_left 12 7 = 5 := by
  sorry

end jimin_tangerines_l2683_268306


namespace compute_expression_l2683_268359

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end compute_expression_l2683_268359


namespace micro_lesson_production_properties_l2683_268303

/-- Represents the cost and profit structure of an online education micro-lesson production team -/
structure MicroLessonProduction where
  cost_2A_3B : ℕ  -- Cost of producing 2 A type and 3 B type micro-lessons
  cost_3A_4B : ℕ  -- Cost of producing 3 A type and 4 B type micro-lessons
  price_A : ℕ     -- Selling price of A type micro-lesson
  price_B : ℕ     -- Selling price of B type micro-lesson
  days_per_month : ℕ  -- Number of production days per month

/-- Theorem stating the properties of the micro-lesson production system -/
theorem micro_lesson_production_properties (p : MicroLessonProduction)
  (h1 : p.cost_2A_3B = 2900)
  (h2 : p.cost_3A_4B = 4100)
  (h3 : p.price_A = 1500)
  (h4 : p.price_B = 1000)
  (h5 : p.days_per_month = 22) :
  ∃ (cost_A cost_B : ℕ) (profit_function : ℕ → ℕ) (max_profit max_profit_days : ℕ),
    cost_A = 700 ∧
    cost_B = 500 ∧
    (∀ a : ℕ, 0 < a ∧ a ≤ 66 / 7 → profit_function a = 50 * a + 16500) ∧
    max_profit = 16900 ∧
    max_profit_days = 8 ∧
    (∀ a : ℕ, 0 < a ∧ a ≤ 66 / 7 → profit_function a ≤ max_profit) :=
by sorry


end micro_lesson_production_properties_l2683_268303


namespace fish_westward_l2683_268331

/-- The number of fish that swam westward -/
def W : ℕ := sorry

/-- The number of fish that swam eastward -/
def E : ℕ := 3200

/-- The number of fish that swam north -/
def N : ℕ := 500

/-- The fraction of eastward-swimming fish caught by fishers -/
def east_catch_ratio : ℚ := 2 / 5

/-- The fraction of westward-swimming fish caught by fishers -/
def west_catch_ratio : ℚ := 3 / 4

/-- The number of fish left in the sea after catching -/
def fish_left : ℕ := 2870

theorem fish_westward :
  W = 1800 ∧
  (W : ℚ) + E + N - (east_catch_ratio * E + west_catch_ratio * W) = fish_left :=
sorry

end fish_westward_l2683_268331


namespace initial_chairs_per_row_l2683_268300

theorem initial_chairs_per_row (rows : ℕ) (extra_chairs : ℕ) (total_chairs : ℕ) :
  rows = 7 →
  extra_chairs = 11 →
  total_chairs = 95 →
  ∃ (chairs_per_row : ℕ), chairs_per_row * rows + extra_chairs = total_chairs ∧ chairs_per_row = 12 := by
  sorry

end initial_chairs_per_row_l2683_268300


namespace gcd_294_84_l2683_268352

theorem gcd_294_84 : Nat.gcd 294 84 = 42 := by
  have h1 : 294 = 84 * 3 + 42 := by rfl
  have h2 : 84 = 42 * 2 + 0 := by rfl
  sorry

end gcd_294_84_l2683_268352


namespace tensor_inequality_implies_a_bound_l2683_268319

-- Define the ⊗ operation
def tensor (x y : ℝ) := x * (1 - y)

-- Define the main theorem
theorem tensor_inequality_implies_a_bound (a : ℝ) : 
  (∀ x > 2, tensor (x - a) x ≤ a + 2) → a ≤ 7 := by
  sorry


end tensor_inequality_implies_a_bound_l2683_268319


namespace eighteen_percent_of_42_equals_27_percent_of_x_l2683_268328

theorem eighteen_percent_of_42_equals_27_percent_of_x (x : ℝ) : 
  (18 / 100) * 42 = (27 / 100) * x → x = 28 := by
sorry

end eighteen_percent_of_42_equals_27_percent_of_x_l2683_268328


namespace bus_rows_l2683_268351

/-- Given a bus with a certain capacity and seats per row, calculate the number of rows. -/
def calculate_rows (total_capacity : ℕ) (children_per_row : ℕ) : ℕ :=
  total_capacity / children_per_row

/-- Theorem: A bus with 36 children capacity and 4 children per row has 9 rows of seats. -/
theorem bus_rows :
  calculate_rows 36 4 = 9 := by
  sorry

#eval calculate_rows 36 4

end bus_rows_l2683_268351


namespace odd_function_property_l2683_268395

-- Define an odd function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem odd_function_property :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is an odd function
  (∀ x > 0, f x = x - 1) →    -- f(x) = x - 1 for x > 0
  (∀ x < 0, f x * f (-x) ≤ 0) -- f(x)f(-x) ≤ 0 for x < 0
:= by sorry

end odd_function_property_l2683_268395


namespace boat_speed_is_48_l2683_268305

/-- The speed of the stream in kilometers per hour -/
def stream_speed : ℝ := 16

/-- The speed of the boat in still water in kilometers per hour -/
def boat_speed : ℝ := 48

/-- The time taken to row downstream -/
def time_downstream : ℝ := 1

/-- The time taken to row upstream -/
def time_upstream : ℝ := 2 * time_downstream

/-- The theorem stating that the boat's speed in still water is 48 kmph -/
theorem boat_speed_is_48 : 
  (boat_speed + stream_speed) * time_downstream = 
  (boat_speed - stream_speed) * time_upstream :=
by sorry

end boat_speed_is_48_l2683_268305


namespace sum_of_last_two_digits_of_factorial_series_l2683_268388

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Function to get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- The series in question -/
def series : List ℕ := [1, 2, 5, 13, 34]

/-- Theorem stating that the sum of the last two digits of the factorial series is 23 -/
theorem sum_of_last_two_digits_of_factorial_series : 
  (series.map (λ n => lastTwoDigits (factorial n))).sum = 23 := by sorry

end sum_of_last_two_digits_of_factorial_series_l2683_268388


namespace mean_d_formula_l2683_268325

/-- The set of all positive integers with n 1s, n 2s, n 3s, ..., n ms -/
def S (m n : ℕ) : Set ℕ := sorry

/-- The sum of absolute differences between all pairs of adjacent digits in N -/
def d (N : ℕ) : ℕ := sorry

/-- The mean value of d(N) for N in S(m, n) -/
def mean_d (m n : ℕ) : ℚ := sorry

theorem mean_d_formula {m n : ℕ} (hm : 0 < m ∧ m < 10) (hn : 0 < n) :
  mean_d m n = n * (m^2 - 1) / 3 := by sorry

end mean_d_formula_l2683_268325


namespace election_votes_l2683_268371

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (difference_percent : ℚ) : 
  total_votes = 9720 →
  invalid_percent = 1/5 →
  difference_percent = 3/20 →
  ∃ (a_votes b_votes : ℕ),
    b_votes = 3159 ∧
    a_votes + b_votes = total_votes * (1 - invalid_percent) ∧
    a_votes = b_votes + total_votes * difference_percent :=
by sorry

end election_votes_l2683_268371


namespace compute_expression_l2683_268373

theorem compute_expression : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end compute_expression_l2683_268373


namespace hexagonal_pattern_selections_l2683_268317

-- Define the structure of the hexagonal grid
def HexagonalGrid := Unit

-- Define the specific hexagonal pattern to be selected
def HexagonalPattern := Unit

-- Define the number of distinct positions without rotations
def distinctPositionsWithoutRotations : ℕ := 26

-- Define the number of distinct rotations for the hexagonal pattern
def distinctRotations : ℕ := 3

-- Theorem statement
theorem hexagonal_pattern_selections (grid : HexagonalGrid) (pattern : HexagonalPattern) :
  (distinctPositionsWithoutRotations * distinctRotations) = 78 := by
  sorry

end hexagonal_pattern_selections_l2683_268317


namespace expenditure_increase_l2683_268304

theorem expenditure_increase 
  (income : ℝ) 
  (expenditure : ℝ) 
  (savings : ℝ) 
  (new_income : ℝ) 
  (new_expenditure : ℝ) 
  (new_savings : ℝ) 
  (h1 : expenditure = 0.75 * income) 
  (h2 : savings = income - expenditure) 
  (h3 : new_income = 1.2 * income) 
  (h4 : new_savings = 1.4999999999999996 * savings) 
  (h5 : new_savings = new_income - new_expenditure) : 
  new_expenditure = 1.1 * expenditure := by
sorry

end expenditure_increase_l2683_268304


namespace polynomial_irreducibility_l2683_268372

/-- Given n ≥ 2 distinct integers, the polynomial f(x) = (x - a₁)(x - a₂) ... (x - aₙ) - 1 is irreducible over the integers. -/
theorem polynomial_irreducibility (n : ℕ) (a : Fin n → ℤ) (h1 : n ≥ 2) (h2 : Function.Injective a) :
  Irreducible (((Polynomial.X : Polynomial ℤ) - (Finset.univ.prod (fun i => Polynomial.X - Polynomial.C (a i)))) - 1) := by
  sorry

end polynomial_irreducibility_l2683_268372


namespace cos_negative_750_degrees_l2683_268393

theorem cos_negative_750_degrees : Real.cos ((-750 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_negative_750_degrees_l2683_268393


namespace arithmetic_calculation_l2683_268327

theorem arithmetic_calculation : 18 * 36 + 54 * 18 + 18 * 9 = 1782 := by
  sorry

end arithmetic_calculation_l2683_268327


namespace ship_river_flow_equation_l2683_268332

theorem ship_river_flow_equation (v : ℝ) : 
  (144 / (30 + v) = 96 / (30 - v)) ↔ 
  (144 / (30 + v) = 96 / (30 - v) ∧ 
   v > 0 ∧ v < 30 ∧
   144 / (30 + v) = 96 / (30 - v)) :=
by sorry

end ship_river_flow_equation_l2683_268332


namespace f_max_min_values_l2683_268396

noncomputable def f (x : ℝ) : ℝ := 5 * (Real.cos x)^2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem f_max_min_values :
  (∀ x, f x ≤ 16 + 10 * Real.sqrt 13) ∧
  (∀ x, f x ≥ 16 - 10 * Real.sqrt 13) ∧
  (∃ x, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end f_max_min_values_l2683_268396


namespace simplify_expression_l2683_268350

theorem simplify_expression (a b c d x y : ℝ) (h : cx + dy ≠ 0) :
  (c*x*(b^2*x^2 + 3*b^2*y^2 + a^2*y^2) + d*y*(b^2*x^2 + 3*a^2*x^2 + a^2*y^2)) / (c*x + d*y) =
  (b^2 + 3*a^2)*x^2 + (a^2 + 3*b^2)*y^2 := by sorry

end simplify_expression_l2683_268350


namespace photo_voting_total_l2683_268378

/-- Represents a photo voting system with applauds and boos -/
structure PhotoVoting where
  total_votes : ℕ
  applaud_ratio : ℚ
  score : ℤ

/-- Theorem: Given the conditions, the total votes cast is 300 -/
theorem photo_voting_total (pv : PhotoVoting) 
  (h1 : pv.applaud_ratio = 3/4)
  (h2 : pv.score = 150) :
  pv.total_votes = 300 := by
  sorry

end photo_voting_total_l2683_268378


namespace not_both_rising_left_l2683_268316

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := 2 * x^2
def parabola2 (x : ℝ) : ℝ := -2 * x^2

-- Define what it means for a function to be rising on an interval
def is_rising (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Theorem stating that it's not true that both parabolas are rising on the left side of the y-axis
theorem not_both_rising_left : ¬(∃ (a : ℝ), a < 0 ∧ 
  is_rising parabola1 a 0 ∧ is_rising parabola2 a 0) :=
sorry

end not_both_rising_left_l2683_268316


namespace james_cattle_profit_l2683_268394

/-- Represents the profit calculation for James' cattle business --/
theorem james_cattle_profit :
  let num_cattle : ℕ := 100
  let total_buying_cost : ℚ := 40000
  let buying_cost_per_cattle : ℚ := total_buying_cost / num_cattle
  let feeding_cost_per_cattle : ℚ := buying_cost_per_cattle * 1.2
  let total_feeding_cost_per_month : ℚ := feeding_cost_per_cattle * num_cattle
  let months_held : ℕ := 6
  let total_feeding_cost : ℚ := total_feeding_cost_per_month * months_held
  let total_cost : ℚ := total_buying_cost + total_feeding_cost
  let weight_per_cattle : ℕ := 1000
  let june_price_per_pound : ℚ := 2.2
  let total_selling_price : ℚ := num_cattle * weight_per_cattle * june_price_per_pound
  let profit : ℚ := total_selling_price - total_cost
  profit = -108000 := by sorry

end james_cattle_profit_l2683_268394


namespace duck_pricing_problem_l2683_268366

/-- A problem about duck pricing and profit -/
theorem duck_pricing_problem 
  (num_ducks : ℕ) 
  (weight_per_duck : ℝ) 
  (selling_price_per_pound : ℝ) 
  (total_profit : ℝ) 
  (h1 : num_ducks = 30)
  (h2 : weight_per_duck = 4)
  (h3 : selling_price_per_pound = 5)
  (h4 : total_profit = 300) :
  let total_revenue := num_ducks * weight_per_duck * selling_price_per_pound
  let price_per_duck := (total_revenue - total_profit) / num_ducks
  price_per_duck = 10 := by
sorry

end duck_pricing_problem_l2683_268366


namespace valuable_files_count_l2683_268358

def initial_download : ℕ := 800
def first_deletion_rate : ℚ := 70 / 100
def second_download : ℕ := 400
def second_deletion_rate : ℚ := 3 / 5

theorem valuable_files_count :
  (initial_download - (initial_download * first_deletion_rate).floor) +
  (second_download - (second_download * second_deletion_rate).floor) = 400 := by
  sorry

end valuable_files_count_l2683_268358


namespace contrapositive_real_roots_l2683_268338

theorem contrapositive_real_roots (m : ℝ) : 
  (¬(∃ x : ℝ, x^2 + 2*x - 3*m = 0) → m ≤ 0) ↔ (m > 0 → ∃ x : ℝ, x^2 + 2*x - 3*m = 0) :=
by sorry

end contrapositive_real_roots_l2683_268338


namespace tangent_line_implies_base_l2683_268341

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem tangent_line_implies_base (a : ℝ) :
  (∃ m : ℝ, f a m = (1/3) * m ∧ 
    (∀ x : ℝ, x > 0 → HasDerivAt (f a) ((1/3) : ℝ) x)) →
  a = Real.exp ((3:ℝ) / Real.exp 1) :=
sorry

end tangent_line_implies_base_l2683_268341


namespace blue_tshirts_per_pack_l2683_268374

/-- Given the following:
  * Dave bought 3 packs of white T-shirts and 2 packs of blue T-shirts
  * White T-shirts come in packs of 6
  * Dave bought 26 T-shirts in total
Prove that the number of blue T-shirts in each pack is 4 -/
theorem blue_tshirts_per_pack (white_packs : ℕ) (blue_packs : ℕ) (white_per_pack : ℕ) (total_tshirts : ℕ)
  (h1 : white_packs = 3)
  (h2 : blue_packs = 2)
  (h3 : white_per_pack = 6)
  (h4 : total_tshirts = 26)
  (h5 : white_packs * white_per_pack + blue_packs * (total_tshirts - white_packs * white_per_pack) / blue_packs = total_tshirts) :
  (total_tshirts - white_packs * white_per_pack) / blue_packs = 4 := by
  sorry

end blue_tshirts_per_pack_l2683_268374


namespace tangent_point_x_coordinate_l2683_268309

theorem tangent_point_x_coordinate 
  (f : ℝ → ℝ) 
  (h₁ : ∀ x, f x = x^2 + 1) 
  (h₂ : ∃ x, deriv f x = 4) : 
  ∃ x, deriv f x = 4 ∧ x = 2 := by
  sorry

end tangent_point_x_coordinate_l2683_268309


namespace largest_value_l2683_268347

theorem largest_value (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max 1 (2*a*b)) (a^2 + b^2)) a = a^2 + b^2 := by
  sorry

end largest_value_l2683_268347


namespace three_Z_five_equals_two_l2683_268342

/-- The Z operation defined on real numbers -/
def Z (a b : ℝ) : ℝ := b + 5*a - 2*a^2

/-- Theorem stating that 3Z5 equals 2 -/
theorem three_Z_five_equals_two : Z 3 5 = 2 := by
  sorry

end three_Z_five_equals_two_l2683_268342


namespace three_from_ten_combination_l2683_268326

theorem three_from_ten_combination : Nat.choose 10 3 = 120 := by
  sorry

end three_from_ten_combination_l2683_268326


namespace smallest_positive_root_floor_l2683_268387

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 * Real.tan x

theorem smallest_positive_root_floor :
  ∃ s : ℝ, s > 0 ∧ g s = 0 ∧ (∀ t, t > 0 ∧ g t = 0 → s ≤ t) ∧ 4 ≤ s ∧ s < 5 :=
sorry

end smallest_positive_root_floor_l2683_268387


namespace john_james_age_relation_james_brother_age_is_16_l2683_268362

-- Define the ages
def john_age : ℕ := 39
def james_age : ℕ := 12

-- Define the relationship between John and James' ages
theorem john_james_age_relation : john_age - 3 = 2 * (james_age + 6) := by sorry

-- Define James' older brother's age
def james_brother_age : ℕ := james_age + 4

-- Theorem to prove
theorem james_brother_age_is_16 : james_brother_age = 16 := by sorry

end john_james_age_relation_james_brother_age_is_16_l2683_268362


namespace bob_over_budget_l2683_268369

def budget : ℕ := 100
def necklaceA : ℕ := 34
def necklaceB : ℕ := 42
def necklaceC : ℕ := 50
def book1 : ℕ := necklaceA + 20
def book2 : ℕ := necklaceC - 10

theorem bob_over_budget : 
  necklaceA + necklaceB + necklaceC + book1 + book2 - budget = 120 := by
  sorry

end bob_over_budget_l2683_268369


namespace f_has_two_extreme_points_l2683_268361

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3 - 9

-- Define what an extreme point is
def is_extreme_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → f y ≠ f x

-- State the theorem
theorem f_has_two_extreme_points :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x ∈ s, is_extreme_point f x :=
sorry

end f_has_two_extreme_points_l2683_268361


namespace product_of_three_numbers_l2683_268380

theorem product_of_three_numbers (a b c : ℝ) : 
  (a + b + c = 44) → 
  (a^2 + b^2 + c^2 = 890) → 
  (a^2 + b^2 > 2*c^2) → 
  (b^2 + c^2 > 2*a^2) → 
  (c^2 + a^2 > 2*b^2) → 
  (a * b * c = 23012) := by
sorry

end product_of_three_numbers_l2683_268380


namespace li_age_l2683_268310

/-- Given the ages of Zhang, Jung, and Li, prove that Li is 12 years old. -/
theorem li_age (zhang_age li_age jung_age : ℕ) 
  (h1 : zhang_age = 2 * li_age)
  (h2 : jung_age = zhang_age + 2)
  (h3 : jung_age = 26) :
  li_age = 12 := by
  sorry

end li_age_l2683_268310


namespace equation_sum_equals_one_l2683_268336

theorem equation_sum_equals_one 
  (p q r u v w : ℝ) 
  (eq1 : 15 * u + q * v + r * w = 0)
  (eq2 : p * u + 25 * v + r * w = 0)
  (eq3 : p * u + q * v + 50 * w = 0)
  (hp : p ≠ 15)
  (hu : u ≠ 0) :
  p / (p - 15) + q / (q - 25) + r / (r - 50) = 1 := by
sorry

end equation_sum_equals_one_l2683_268336


namespace continuity_at_two_and_delta_l2683_268343

def f (x : ℝ) := -3 * x^2 - 5

theorem continuity_at_two_and_delta (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε ∧
  ∃ δ₀ > 0, δ₀ = ε/3 ∧ ∀ x, |x - 2| < δ₀ → |f x - f 2| < ε :=
by sorry

end continuity_at_two_and_delta_l2683_268343


namespace value_of_difference_product_l2683_268360

theorem value_of_difference_product (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (x + y) = 95 := by
  sorry

end value_of_difference_product_l2683_268360


namespace unique_solution_condition_l2683_268355

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 := by
  sorry

end unique_solution_condition_l2683_268355


namespace green_balls_count_l2683_268356

theorem green_balls_count (total : ℕ) (red : ℕ) (green : ℕ) (prob_red : ℚ) : 
  red = 8 →
  green = total - red →
  prob_red = 1/3 →
  prob_red = red / total →
  green = 16 := by sorry

end green_balls_count_l2683_268356


namespace remainder_sum_l2683_268354

theorem remainder_sum (a b : ℤ) (h1 : a % 84 = 78) (h2 : b % 120 = 114) :
  (a + b) % 42 = 24 := by
sorry

end remainder_sum_l2683_268354


namespace chloe_min_nickels_l2683_268323

/-- The minimum number of nickels Chloe needs to afford the book -/
def min_nickels : ℕ := 120

/-- The cost of the book in cents -/
def book_cost : ℕ := 4850

/-- The value of Chloe's $10 bills in cents -/
def ten_dollar_bills : ℕ := 4000

/-- The value of Chloe's quarters in cents -/
def quarters : ℕ := 250

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

theorem chloe_min_nickels :
  ∀ n : ℕ, n ≥ min_nickels →
  ten_dollar_bills + quarters + n * nickel_value ≥ book_cost :=
by sorry

end chloe_min_nickels_l2683_268323


namespace garden_ratio_l2683_268386

/-- Proves that a rectangular garden with area 432 square meters and width 12 meters has a length to width ratio of 3:1 -/
theorem garden_ratio :
  ∀ (length width : ℝ),
    width = 12 →
    length * width = 432 →
    length / width = 3 := by
  sorry

end garden_ratio_l2683_268386


namespace saturday_newspaper_delivery_l2683_268389

/-- Given that Peter delivers newspapers on weekends, prove that he delivers 45 papers on Saturday. -/
theorem saturday_newspaper_delivery :
  ∀ (saturday_delivery sunday_delivery : ℕ),
  saturday_delivery + sunday_delivery = 110 →
  sunday_delivery = saturday_delivery + 20 →
  saturday_delivery = 45 := by
sorry

end saturday_newspaper_delivery_l2683_268389


namespace equal_savings_after_820_weeks_l2683_268379

/-- Represents the number of weeks it takes for Jim and Sara to have saved the same amount -/
def weeks_to_equal_savings : ℕ :=
  820

/-- Sara's initial savings in dollars -/
def sara_initial_savings : ℕ :=
  4100

/-- Sara's weekly savings in dollars -/
def sara_weekly_savings : ℕ :=
  10

/-- Jim's weekly savings in dollars -/
def jim_weekly_savings : ℕ :=
  15

theorem equal_savings_after_820_weeks :
  sara_initial_savings + sara_weekly_savings * weeks_to_equal_savings =
  jim_weekly_savings * weeks_to_equal_savings :=
by
  sorry

#check equal_savings_after_820_weeks

end equal_savings_after_820_weeks_l2683_268379


namespace john_camera_rental_payment_l2683_268370

def camera_rental_problem (camera_value : ℝ) (rental_rate : ℝ) (rental_weeks : ℕ) (friend_contribution_rate : ℝ) : Prop :=
  let weekly_rental := camera_value * rental_rate
  let total_rental := weekly_rental * rental_weeks
  let friend_contribution := total_rental * friend_contribution_rate
  let john_payment := total_rental - friend_contribution
  john_payment = 1200

theorem john_camera_rental_payment :
  camera_rental_problem 5000 0.10 4 0.40 := by
  sorry

end john_camera_rental_payment_l2683_268370


namespace range_of_a_l2683_268364

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 - a ≥ 0) ∨ 
  (∃ x : ℝ, x^2 + 2*a*x + a + 2 = 0) ↔ 
  (-1 < a ∧ a ≤ 0) ∨ (a ≥ 2) :=
by sorry

end range_of_a_l2683_268364


namespace tan_beta_value_l2683_268321

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 2) 
  (h2 : Real.tan (α + β) = 1/5) : 
  Real.tan β = -9/7 := by
sorry

end tan_beta_value_l2683_268321


namespace sum_squares_five_consecutive_not_perfect_square_l2683_268345

theorem sum_squares_five_consecutive_not_perfect_square (n : ℤ) :
  ¬∃ m : ℤ, (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = m^2 :=
sorry


end sum_squares_five_consecutive_not_perfect_square_l2683_268345
