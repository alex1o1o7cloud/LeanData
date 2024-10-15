import Mathlib

namespace NUMINAMATH_CALUDE_expression_value_l4027_402787

theorem expression_value (x : ℝ) (h : x = 5) : 2 * x + 3 - 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4027_402787


namespace NUMINAMATH_CALUDE_jose_distance_l4027_402752

/-- Given a speed of 2 kilometers per hour and a time of 2 hours, 
    the distance traveled is equal to 4 kilometers. -/
theorem jose_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 2 → distance = speed * time → distance = 4 := by
  sorry


end NUMINAMATH_CALUDE_jose_distance_l4027_402752


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l4027_402765

theorem quadratic_real_roots (a b c : ℝ) :
  (∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2*(a + b + c) * x + 3 = 0) ↔
  (a = b ∧ b = c ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l4027_402765


namespace NUMINAMATH_CALUDE_triangle_side_length_l4027_402715

/-- A square with side length 10 cm is divided into two right trapezoids and a right triangle. -/
structure DividedSquare where
  /-- Side length of the square -/
  side_length : ℝ
  /-- Height of the trapezoids -/
  trapezoid_height : ℝ
  /-- Area difference between the trapezoids -/
  area_difference : ℝ
  /-- Length of one side of the right triangle -/
  triangle_side : ℝ
  /-- The side length is 10 cm -/
  side_length_eq : side_length = 10
  /-- The area difference between trapezoids is 10 cm² -/
  area_difference_eq : area_difference = 10
  /-- The trapezoids have equal height -/
  trapezoid_height_eq : trapezoid_height = side_length / 2

/-- The theorem to be proved -/
theorem triangle_side_length (s : DividedSquare) : s.triangle_side = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4027_402715


namespace NUMINAMATH_CALUDE_min_difference_is_one_l4027_402735

/-- Triangle with integer side lengths and specific conditions -/
structure Triangle where
  DE : ℕ
  EF : ℕ
  FD : ℕ
  perimeter_eq : DE + EF + FD = 3010
  side_order : DE < EF ∧ EF ≤ FD

/-- The smallest possible difference between EF and DE is 1 -/
theorem min_difference_is_one (t : Triangle) : 
  (∀ t' : Triangle, t'.EF - t'.DE ≥ 1) ∧ (∃ t' : Triangle, t'.EF - t'.DE = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_difference_is_one_l4027_402735


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l4027_402771

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x = 1 → x^2 = 1) ∧ 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l4027_402771


namespace NUMINAMATH_CALUDE_parabola_translation_l4027_402702

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola vertically -/
def translateVertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

/-- Translates a parabola horizontally -/
def translateHorizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := 2 * p.a * h + p.b, c := p.a * h^2 + p.b * h + p.c }

/-- The main theorem stating that translating y = 3x² upwards by 3 and left by 2 results in y = 3(x+2)² + 3 -/
theorem parabola_translation (original : Parabola) 
  (h : original = { a := 3, b := 0, c := 0 }) : 
  translateHorizontal (translateVertical original 3) 2 = { a := 3, b := 12, c := 15 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l4027_402702


namespace NUMINAMATH_CALUDE_circle_area_ratio_l4027_402750

theorem circle_area_ratio : 
  ∀ (r₁ r₂ : ℝ), r₁ > 0 → r₂ > 0 → r₂ = 3 * r₁ →
  (π * r₂^2 - π * r₁^2) / (π * r₁^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l4027_402750


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4027_402777

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : GeometricSequence a q)
  (h_arith : a 3 - 3 * a 1 = a 2 - a 3) :
  q = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4027_402777


namespace NUMINAMATH_CALUDE_right_triangle_area_perimeter_l4027_402794

/-- A right triangle with hypotenuse 13 and one leg 5 has area 30 and perimeter 30 -/
theorem right_triangle_area_perimeter :
  ∀ (a b c : ℝ),
  a = 5 →
  c = 13 →
  a^2 + b^2 = c^2 →
  (1/2 * a * b = 30) ∧ (a + b + c = 30) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_perimeter_l4027_402794


namespace NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive_l4027_402720

theorem x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive :
  (∃ x : ℝ, x > 0 ∧ x^2 + x > 0) ∧
  (∃ x : ℝ, x^2 + x > 0 ∧ ¬(x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_x_positive_sufficient_not_necessary_for_x_squared_plus_x_positive_l4027_402720


namespace NUMINAMATH_CALUDE_complement_of_A_in_B_l4027_402725

def A : Set ℕ := {2, 3}
def B : Set ℕ := {0, 1, 2, 3, 4}

theorem complement_of_A_in_B :
  (B \ A) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_B_l4027_402725


namespace NUMINAMATH_CALUDE_pi_comparison_l4027_402778

theorem pi_comparison : -Real.pi < -3.14 := by sorry

end NUMINAMATH_CALUDE_pi_comparison_l4027_402778


namespace NUMINAMATH_CALUDE_complement_intersect_equal_l4027_402733

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}

theorem complement_intersect_equal : (U \ B) ∩ A = {0, 2} := by sorry

end NUMINAMATH_CALUDE_complement_intersect_equal_l4027_402733


namespace NUMINAMATH_CALUDE_max_spheres_in_specific_cylinder_l4027_402701

/-- Represents a cylindrical container -/
structure Cylinder where
  diameter : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  diameter : ℝ

/-- Calculates the maximum number of spheres that can fit in a cylinder -/
def maxSpheresInCylinder (c : Cylinder) (s : Sphere) : ℕ :=
  sorry

theorem max_spheres_in_specific_cylinder :
  let c := Cylinder.mk 82 225
  let s := Sphere.mk 38
  maxSpheresInCylinder c s = 21 := by
  sorry

end NUMINAMATH_CALUDE_max_spheres_in_specific_cylinder_l4027_402701


namespace NUMINAMATH_CALUDE_math_books_same_box_probability_l4027_402793

/-- Represents a box with a given capacity -/
structure Box where
  capacity : ℕ

/-- Represents the collection of boxes -/
def boxes : List Box := [⟨4⟩, ⟨5⟩, ⟨6⟩]

/-- Total number of textbooks -/
def total_textbooks : ℕ := 15

/-- Number of mathematics textbooks -/
def math_textbooks : ℕ := 4

/-- Calculates the probability of all mathematics textbooks being in the same box -/
noncomputable def prob_math_books_same_box : ℚ := sorry

/-- Theorem stating the probability of all mathematics textbooks being in the same box -/
theorem math_books_same_box_probability :
  prob_math_books_same_box = 1 / 91 := by sorry

end NUMINAMATH_CALUDE_math_books_same_box_probability_l4027_402793


namespace NUMINAMATH_CALUDE_min_value_fraction_l4027_402732

theorem min_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y - 1 = 0) :
  (x + 2*y) / (x*y) ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2*x₀ + y₀ - 1 = 0 ∧ (x₀ + 2*y₀) / (x₀*y₀) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l4027_402732


namespace NUMINAMATH_CALUDE_root_difference_for_arithmetic_progression_cubic_l4027_402718

theorem root_difference_for_arithmetic_progression_cubic (a b c d : ℝ) :
  (∃ x y z : ℝ, 
    (49 * x^3 - 105 * x^2 + 63 * x - 10 = 0) ∧
    (49 * y^3 - 105 * y^2 + 63 * y - 10 = 0) ∧
    (49 * z^3 - 105 * z^2 + 63 * z - 10 = 0) ∧
    (y - x = z - y) ∧
    (x < y) ∧ (y < z)) →
  (z - x = 2 * Real.sqrt 11 / 7) :=
by sorry

end NUMINAMATH_CALUDE_root_difference_for_arithmetic_progression_cubic_l4027_402718


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l4027_402721

/-- Given a boat that travels 10 km/hr downstream and 4 km/hr upstream, 
    its speed in still water is 7 km/hr. -/
theorem boat_speed_in_still_water 
  (downstream_speed : ℝ) 
  (upstream_speed : ℝ) 
  (h_downstream : downstream_speed = 10) 
  (h_upstream : upstream_speed = 4) : 
  (downstream_speed + upstream_speed) / 2 = 7 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l4027_402721


namespace NUMINAMATH_CALUDE_combined_tax_rate_l4027_402724

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem combined_tax_rate (mork_rate mindy_rate : ℚ) (income_ratio : ℚ) :
  mork_rate = 2/5 →
  mindy_rate = 1/4 →
  income_ratio = 4 →
  (mork_rate + income_ratio * mindy_rate) / (1 + income_ratio) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l4027_402724


namespace NUMINAMATH_CALUDE_synesthesia_demonstrates_mutual_influence_and_restriction_l4027_402709

/-- Represents a sensory perception -/
inductive Sense
  | Sight
  | Hearing
  | Taste
  | Smell
  | Touch

/-- Represents the phenomenon of synesthesia -/
def Synesthesia := Set (Sense × Sense)

/-- Represents the property of mutual influence and restriction -/
def MutualInfluenceAndRestriction (s : Synesthesia) : Prop := sorry

/-- Represents a thing and its internal elements -/
structure Thing where
  elements : Set Sense

theorem synesthesia_demonstrates_mutual_influence_and_restriction 
  (s : Synesthesia) 
  (h : s.Nonempty) : 
  MutualInfluenceAndRestriction s := by
  sorry

#check synesthesia_demonstrates_mutual_influence_and_restriction

end NUMINAMATH_CALUDE_synesthesia_demonstrates_mutual_influence_and_restriction_l4027_402709


namespace NUMINAMATH_CALUDE_coordinates_are_precise_l4027_402760

-- Define a type for location descriptions
inductive LocationDescription
  | indoor : String → String → String → LocationDescription  -- Building, room, etc.
  | roadSection : String → LocationDescription  -- Road name
  | coordinates : Float → Float → LocationDescription  -- Longitude and Latitude
  | direction : Float → String → LocationDescription  -- Angle and cardinal direction

-- Function to check if a location description is precise
def isPreciseLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.coordinates _ _ => True
  | _ => False

-- Theorem statement
theorem coordinates_are_precise (locations : List LocationDescription) :
  ∃ (loc : LocationDescription), loc ∈ locations ∧ isPreciseLocation loc ↔
    ∃ (lon lat : Float), LocationDescription.coordinates lon lat ∈ locations :=
sorry

end NUMINAMATH_CALUDE_coordinates_are_precise_l4027_402760


namespace NUMINAMATH_CALUDE_power_of_square_l4027_402799

theorem power_of_square (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_square_l4027_402799


namespace NUMINAMATH_CALUDE_rent_reduction_percentage_l4027_402790

-- Define the room prices
def cheap_room_price : ℕ := 40
def expensive_room_price : ℕ := 60

-- Define the total rent
def total_rent : ℕ := 1000

-- Define the number of rooms to be moved
def rooms_to_move : ℕ := 10

-- Define the function to calculate the new total rent
def new_total_rent : ℕ := total_rent - rooms_to_move * (expensive_room_price - cheap_room_price)

-- Define the reduction percentage
def reduction_percentage : ℚ := (total_rent - new_total_rent : ℚ) / total_rent * 100

-- Theorem statement
theorem rent_reduction_percentage :
  reduction_percentage = 20 :=
sorry

end NUMINAMATH_CALUDE_rent_reduction_percentage_l4027_402790


namespace NUMINAMATH_CALUDE_circle_ratio_l4027_402761

theorem circle_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  π * b^2 - π * a^2 = 4 * (π * a^2) → a / b = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l4027_402761


namespace NUMINAMATH_CALUDE_transform_minus3_minus8i_l4027_402786

def rotate90 (z : ℂ) : ℂ := z * Complex.I

def dilate2 (z : ℂ) : ℂ := 2 * z

def transform (z : ℂ) : ℂ := dilate2 (rotate90 z)

theorem transform_minus3_minus8i :
  transform (-3 - 8 * Complex.I) = 16 - 6 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_transform_minus3_minus8i_l4027_402786


namespace NUMINAMATH_CALUDE_article_cost_l4027_402746

/-- The cost of an article satisfying given profit conditions -/
theorem article_cost : ∃ (C : ℝ), 
  (C = 70) ∧ 
  (∃ (S : ℝ), S = 1.25 * C) ∧ 
  (∃ (S_new : ℝ), S_new = 0.8 * C + 0.3 * (0.8 * C) ∧ S_new = 1.25 * C - 14.70) :=
sorry

end NUMINAMATH_CALUDE_article_cost_l4027_402746


namespace NUMINAMATH_CALUDE_star_three_neg_five_l4027_402716

-- Define the new operation "*"
def star (a b : ℚ) : ℚ := a * b + a - b

-- Theorem statement
theorem star_three_neg_five : star 3 (-5) = -7 := by sorry

end NUMINAMATH_CALUDE_star_three_neg_five_l4027_402716


namespace NUMINAMATH_CALUDE_alcohol_solution_concentration_l4027_402748

/-- Given a 6-liter solution that is 40% alcohol, prove that adding 1.2 liters
    of pure alcohol will result in a solution that is 50% alcohol. -/
theorem alcohol_solution_concentration (initial_volume : ℝ) (initial_concentration : ℝ)
    (added_alcohol : ℝ) (target_concentration : ℝ) :
  initial_volume = 6 →
  initial_concentration = 0.4 →
  added_alcohol = 1.2 →
  target_concentration = 0.5 →
  (initial_volume * initial_concentration + added_alcohol) /
    (initial_volume + added_alcohol) = target_concentration := by
  sorry

#check alcohol_solution_concentration

end NUMINAMATH_CALUDE_alcohol_solution_concentration_l4027_402748


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l4027_402726

universe u

theorem complement_intersection_problem :
  let U : Set ℕ := {1, 2, 3, 4, 5}
  let M : Set ℕ := {3, 4, 5}
  let N : Set ℕ := {2, 3}
  (U \ N) ∩ M = {4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l4027_402726


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l4027_402739

theorem min_value_expression (x : ℝ) :
  (15 - x) * (8 - x) * (15 + x) * (8 + x) ≥ -6480.25 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x : ℝ, (15 - x) * (8 - x) * (15 + x) * (8 + x) < -6480.25 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l4027_402739


namespace NUMINAMATH_CALUDE_lg_sum_equals_zero_l4027_402737

-- Define lg as the logarithm with base 10
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_zero : lg 2 + lg 0.5 = 0 := by sorry

end NUMINAMATH_CALUDE_lg_sum_equals_zero_l4027_402737


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l4027_402788

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_9 : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l4027_402788


namespace NUMINAMATH_CALUDE_tangent_slope_at_negative_two_l4027_402744

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Define the point of interest
def point : ℝ × ℝ := (-2, -8)

-- State the theorem
theorem tangent_slope_at_negative_two :
  (deriv f) point.1 = 12 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_negative_two_l4027_402744


namespace NUMINAMATH_CALUDE_leadership_team_selection_l4027_402729

theorem leadership_team_selection (n : ℕ) (h : n = 20) :
  (n.choose 2) * ((n - 2).choose 1) = 3420 := by
  sorry

end NUMINAMATH_CALUDE_leadership_team_selection_l4027_402729


namespace NUMINAMATH_CALUDE_total_earnings_is_228_l4027_402710

/-- Calculates Zainab's total earnings for 4 weeks of passing out flyers -/
def total_earnings : ℝ :=
  let monday_hours : ℝ := 3
  let monday_rate : ℝ := 2.5
  let wednesday_hours : ℝ := 4
  let wednesday_rate : ℝ := 3
  let saturday_hours : ℝ := 5
  let saturday_rate : ℝ := 3.5
  let saturday_flyers : ℝ := 200
  let flyer_commission : ℝ := 0.1
  let weeks : ℝ := 4

  let monday_earnings := monday_hours * monday_rate
  let wednesday_earnings := wednesday_hours * wednesday_rate
  let saturday_hourly_earnings := saturday_hours * saturday_rate
  let saturday_commission := saturday_flyers * flyer_commission
  let saturday_total_earnings := saturday_hourly_earnings + saturday_commission
  let weekly_earnings := monday_earnings + wednesday_earnings + saturday_total_earnings

  weeks * weekly_earnings

theorem total_earnings_is_228 : total_earnings = 228 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_is_228_l4027_402710


namespace NUMINAMATH_CALUDE_mean_temperature_l4027_402766

def temperatures : List ℝ := [-6.5, -2, -3.5, -1, 0.5, 4, 1.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l4027_402766


namespace NUMINAMATH_CALUDE_unique_five_digit_multiple_of_6_l4027_402742

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_five_digit_multiple_of_6 :
  ∃! d : ℕ, d < 10 ∧ is_divisible_by_6 (47360 + d) ∧ sum_of_digits (47360 + d) % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_five_digit_multiple_of_6_l4027_402742


namespace NUMINAMATH_CALUDE_gina_charity_fraction_l4027_402705

def initial_amount : ℚ := 400
def mom_fraction : ℚ := 1/4
def clothes_fraction : ℚ := 1/8
def kept_amount : ℚ := 170
def charity_fraction : ℚ := 1/5

theorem gina_charity_fraction :
  charity_fraction = (initial_amount - mom_fraction * initial_amount - clothes_fraction * initial_amount - kept_amount) / initial_amount := by
  sorry

end NUMINAMATH_CALUDE_gina_charity_fraction_l4027_402705


namespace NUMINAMATH_CALUDE_always_winnable_l4027_402773

/-- Represents a move in the card game -/
def move (deck : List ℕ) : List ℕ :=
  match deck with
  | [] => []
  | x :: xs => (xs.take x).reverse ++ [x] ++ xs.drop x

/-- Predicate to check if 1 is at the top of the deck -/
def hasOneOnTop (deck : List ℕ) : Prop :=
  match deck with
  | 1 :: _ => True
  | _ => False

/-- Theorem stating that the game is always winnable -/
theorem always_winnable (n : ℕ) (deck : List ℕ) :
  (deck.length = n) →
  (∀ i, i ∈ deck ↔ 1 ≤ i ∧ i ≤ n) →
  ∃ k, hasOneOnTop ((move^[k]) deck) :=
sorry


end NUMINAMATH_CALUDE_always_winnable_l4027_402773


namespace NUMINAMATH_CALUDE_problem_235_l4027_402734

theorem problem_235 (x y : ℝ) : 
  y + Real.sqrt (x^2 + y^2) = 16 ∧ x - y = 2 → x = 8 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_235_l4027_402734


namespace NUMINAMATH_CALUDE_log_inequality_l4027_402754

theorem log_inequality : (1 : ℝ) / 3 < Real.log 3 - Real.log 2 ∧ Real.log 3 - Real.log 2 < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l4027_402754


namespace NUMINAMATH_CALUDE_hyperbola_m_range_l4027_402730

-- Define the equation
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 / (|m| - 1) - y^2 / (m - 2) = 1

-- Define the condition for the equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_equation x y m

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  (-1 < m ∧ m < 1) ∨ m > 2

-- Theorem statement
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ m_range m := by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l4027_402730


namespace NUMINAMATH_CALUDE_election_winner_votes_l4027_402756

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) 
  (h1 : winner_percentage = 56 / 100) 
  (h2 : vote_difference = 288) 
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 1344 :=
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l4027_402756


namespace NUMINAMATH_CALUDE_real_part_of_z_is_negative_four_l4027_402797

theorem real_part_of_z_is_negative_four :
  let i : ℂ := Complex.I
  let z : ℂ := (3 + 4 * i) * i
  (z.re : ℝ) = -4 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_is_negative_four_l4027_402797


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4027_402759

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4027_402759


namespace NUMINAMATH_CALUDE_f_at_2_l4027_402774

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_at_2 (a b : ℝ) : f a b (-2) = 3 → f a b 2 = -19 := by sorry

end NUMINAMATH_CALUDE_f_at_2_l4027_402774


namespace NUMINAMATH_CALUDE_blue_paint_cans_l4027_402769

def paint_mixture (total_cans : ℕ) (blue_ratio green_ratio : ℕ) : ℕ := 
  (blue_ratio * total_cans) / (blue_ratio + green_ratio)

theorem blue_paint_cans : paint_mixture 45 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_cans_l4027_402769


namespace NUMINAMATH_CALUDE_sixth_operation_result_l4027_402758

def operation (a b : ℕ) : ℕ := (a + b) * a - a

theorem sixth_operation_result : operation 7 8 = 98 := by
  sorry

end NUMINAMATH_CALUDE_sixth_operation_result_l4027_402758


namespace NUMINAMATH_CALUDE_size_relationship_l4027_402762

theorem size_relationship : 5^30 < 3^50 ∧ 3^50 < 4^40 := by
  sorry

end NUMINAMATH_CALUDE_size_relationship_l4027_402762


namespace NUMINAMATH_CALUDE_sum_of_digits_double_permutation_l4027_402719

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Permutation relation between natural numbers -/
def isPermutationOf (a b : ℕ) : Prop := sorry

theorem sum_of_digits_double_permutation (A B : ℕ) 
  (h : isPermutationOf A B) : 
  sumOfDigits (2 * A) = sumOfDigits (2 * B) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_double_permutation_l4027_402719


namespace NUMINAMATH_CALUDE_cube_sum_odd_implies_product_odd_l4027_402776

theorem cube_sum_odd_implies_product_odd (n m : ℤ) : 
  Odd (n^3 + m^3) → Odd (n * m) := by
sorry

end NUMINAMATH_CALUDE_cube_sum_odd_implies_product_odd_l4027_402776


namespace NUMINAMATH_CALUDE_exists_x_where_inequality_fails_l4027_402751

theorem exists_x_where_inequality_fails : ∃ x : ℝ, x > 0 ∧ 2^x - x^2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_where_inequality_fails_l4027_402751


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l4027_402723

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : 3150 ∣ b) :
  Nat.gcd (Int.natAbs (b^2 + 9*b + 54)) (Int.natAbs (b + 4)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l4027_402723


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_meaningful_range_l4027_402736

-- Define the property of being meaningful for a square root
def is_meaningful (x : ℝ) : Prop := x ≥ 0

-- State the theorem
theorem sqrt_x_minus_3_meaningful_range (x : ℝ) :
  is_meaningful (x - 3) → x ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_meaningful_range_l4027_402736


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4027_402708

open Set

def U : Set Nat := {1,2,3,4,5,6}
def A : Set Nat := {2,3}
def B : Set Nat := {3,5}

theorem intersection_complement_equality : A ∩ (U \ B) = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4027_402708


namespace NUMINAMATH_CALUDE_modular_inverse_13_mod_1200_l4027_402703

theorem modular_inverse_13_mod_1200 : ∃ x : ℕ, x < 1200 ∧ (13 * x) % 1200 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_13_mod_1200_l4027_402703


namespace NUMINAMATH_CALUDE_certain_number_problem_l4027_402706

theorem certain_number_problem (N : ℚ) : 
  (5 / 6 : ℚ) * N = (5 / 16 : ℚ) * N + 200 → N = 384 := by
sorry

end NUMINAMATH_CALUDE_certain_number_problem_l4027_402706


namespace NUMINAMATH_CALUDE_square_side_length_l4027_402783

theorem square_side_length (area : ℝ) (side : ℝ) :
  area = 169 →
  side * side = area →
  side = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l4027_402783


namespace NUMINAMATH_CALUDE_subtract_negatives_l4027_402707

theorem subtract_negatives : -2 - 1 = -3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l4027_402707


namespace NUMINAMATH_CALUDE_mean_proportional_234_104_l4027_402714

theorem mean_proportional_234_104 : ∃ x : ℝ, x^2 = 234 * 104 ∧ x = 156 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_234_104_l4027_402714


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l4027_402722

theorem rectangle_perimeter (a b : ℕ) : 
  a ≠ b →  -- non-square condition
  a * b = 2 * (2 * a + 2 * b) - 8 →  -- area condition
  2 * (a + b) = 36 :=  -- perimeter conclusion
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l4027_402722


namespace NUMINAMATH_CALUDE_rhombus_area_in_rectangle_l4027_402791

/-- The area of a rhombus formed by intersecting equilateral triangles in a rectangle --/
theorem rhombus_area_in_rectangle (a b : ℝ) (h1 : a = 4 * Real.sqrt 3) (h2 : b = 3 * Real.sqrt 3) :
  let triangle_height := (Real.sqrt 3 / 2) * a
  let overlap := 2 * triangle_height - b
  let rhombus_area := (1 / 2) * overlap * a
  rhombus_area = 54 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_rectangle_l4027_402791


namespace NUMINAMATH_CALUDE_min_value_theorem_l4027_402731

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 4) :
  (∀ u v : ℝ, u > 0 → v > 0 → Real.log 2 * u + Real.log 8 * v = Real.log 4 → 
    1/x + 1/(3*y) ≤ 1/u + 1/(3*v)) ∧ 
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ Real.log 2 * x₀ + Real.log 8 * y₀ = Real.log 4 ∧ 
    1/x₀ + 1/(3*y₀) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4027_402731


namespace NUMINAMATH_CALUDE_sin_cos_relation_l4027_402711

theorem sin_cos_relation (θ : Real) (h1 : Real.sin θ + Real.cos θ = 1/2) 
  (h2 : π/2 < θ ∧ θ < π) : Real.cos θ - Real.sin θ = -Real.sqrt 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_relation_l4027_402711


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l4027_402745

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees and 20 meters between trees is 500 meters -/
theorem yard_length_26_trees : 
  yard_length 26 20 = 500 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l4027_402745


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l4027_402704

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) :
  1 / x + 1 / y = -3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l4027_402704


namespace NUMINAMATH_CALUDE_smallest_three_digit_number_with_sum_condition_l4027_402780

theorem smallest_three_digit_number_with_sum_condition 
  (x y z : Nat) 
  (h1 : x < 10 ∧ y < 10 ∧ z < 10) 
  (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (h3 : x + y + z = 10) 
  (h4 : x < y ∧ y < z) : 
  100 * x + 10 * y + z = 127 := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_number_with_sum_condition_l4027_402780


namespace NUMINAMATH_CALUDE_wheel_probability_l4027_402741

theorem wheel_probability (W X Y Z : ℝ) : 
  W = 3/8 → X = 1/4 → Y = 1/8 → W + X + Y + Z = 1 → Z = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l4027_402741


namespace NUMINAMATH_CALUDE_photo_frame_border_area_l4027_402785

/-- The area of the border surrounding a rectangular photograph -/
theorem photo_frame_border_area (photo_height photo_width border_width : ℕ) : 
  photo_height = 12 →
  photo_width = 15 →
  border_width = 3 →
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - photo_height * photo_width = 198 := by
  sorry

#check photo_frame_border_area

end NUMINAMATH_CALUDE_photo_frame_border_area_l4027_402785


namespace NUMINAMATH_CALUDE_flowers_planted_per_day_l4027_402795

theorem flowers_planted_per_day (total_people : ℕ) (total_days : ℕ) (total_flowers : ℕ) 
  (h1 : total_people = 5)
  (h2 : total_days = 2)
  (h3 : total_flowers = 200)
  (h4 : total_people > 0)
  (h5 : total_days > 0) :
  total_flowers / (total_people * total_days) = 20 := by
sorry

end NUMINAMATH_CALUDE_flowers_planted_per_day_l4027_402795


namespace NUMINAMATH_CALUDE_total_laundry_pieces_l4027_402798

def start_time : Nat := 8
def end_time : Nat := 12
def pieces_per_hour : Nat := 20

theorem total_laundry_pieces :
  (end_time - start_time) * pieces_per_hour = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_pieces_l4027_402798


namespace NUMINAMATH_CALUDE_parabola_equation_l4027_402740

/-- A parabola with vertex at the origin and directrix x = -1 has the equation y^2 = 4x -/
theorem parabola_equation (p : ℝ → ℝ → Prop) : 
  (∀ x y, p x y ↔ y^2 = 4*x) → 
  (∀ x, p x 0 ↔ x = 0) →  -- vertex at origin
  (∀ y, p (-1) y ↔ False) →  -- directrix at x = -1
  ∀ x y, p x y ↔ y^2 = 4*x := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l4027_402740


namespace NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l4027_402763

/-- The number of diagonals that can be drawn from one vertex of a polygon. -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: A polygon with 6 diagonals from one vertex has 9 vertices. -/
theorem polygon_vertices_from_diagonals :
  ∃ (n : ℕ), n > 2 ∧ diagonals_from_vertex n = 6 → n = 9 :=
by sorry

end NUMINAMATH_CALUDE_polygon_vertices_from_diagonals_l4027_402763


namespace NUMINAMATH_CALUDE_p_3_eq_10_p_condition_l4027_402796

/-- A polynomial function p: ℝ → ℝ satisfying specific conditions -/
def p : ℝ → ℝ := fun x ↦ x^2 + 1

/-- The first condition: p(3) = 10 -/
theorem p_3_eq_10 : p 3 = 10 := by sorry

/-- The second condition: p(x)p(y) = p(x) + p(y) + p(xy) - 2 for all real x and y -/
theorem p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2 := by sorry

end NUMINAMATH_CALUDE_p_3_eq_10_p_condition_l4027_402796


namespace NUMINAMATH_CALUDE_square_and_rectangles_problem_l4027_402747

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- Theorem statement for the given problem -/
theorem square_and_rectangles_problem
  (small_square : Square)
  (large_rectangle : Rectangle)
  (R : Rectangle)
  (large_square : Square)
  (h1 : small_square.side = 2)
  (h2 : large_rectangle.width = 2 ∧ large_rectangle.height = 4)
  (h3 : small_square.area + large_rectangle.area + R.area = large_square.area)
  : large_square.side = 4 ∧ R.area = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_and_rectangles_problem_l4027_402747


namespace NUMINAMATH_CALUDE_unique_arrangements_count_l4027_402784

/-- The number of letters in the word -/
def word_length : ℕ := 7

/-- The number of identical letters (B and S) -/
def identical_letters : ℕ := 2

/-- Calculates the number of unique arrangements for the given word -/
def unique_arrangements : ℕ := (Nat.factorial word_length) / (Nat.factorial identical_letters)

/-- Theorem stating that the number of unique arrangements is 2520 -/
theorem unique_arrangements_count : unique_arrangements = 2520 := by
  sorry

end NUMINAMATH_CALUDE_unique_arrangements_count_l4027_402784


namespace NUMINAMATH_CALUDE_morse_code_symbols_l4027_402764

/-- The number of possible symbols for a given sequence length -/
def symbolCount (n : ℕ) : ℕ := 2^n

/-- The total number of distinct Morse code symbols with lengths 1 to 4 -/
def totalSymbols : ℕ := symbolCount 1 + symbolCount 2 + symbolCount 3 + symbolCount 4

theorem morse_code_symbols : totalSymbols = 30 := by
  sorry

end NUMINAMATH_CALUDE_morse_code_symbols_l4027_402764


namespace NUMINAMATH_CALUDE_intersection_range_l4027_402779

-- Define the function f(x) = |x^2 - 4x + 3|
def f (x : ℝ) : ℝ := abs (x^2 - 4*x + 3)

-- Define the property of having at least three intersections
def has_at_least_three_intersections (b : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ f x₁ = b ∧ f x₂ = b ∧ f x₃ = b

-- State the theorem
theorem intersection_range :
  ∀ b : ℝ, has_at_least_three_intersections b ↔ 0 < b ∧ b ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_range_l4027_402779


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_18_l4027_402781

/-- Counts the number of ways to arrange n items with given multiplicities. -/
def multinomial_coefficient (n : ℕ) (multiplicities : List ℕ) : ℕ :=
  Nat.factorial n / (multiplicities.map Nat.factorial).prod

/-- The number of five-digit numbers composed of 2 zeros, 2 ones, and 1 two. -/
def count_special_numbers : ℕ :=
  -- Case 1: First digit is 2
  (multinomial_coefficient 4 [2, 2]) +
  -- Case 2: First digit is 1
  (multinomial_coefficient 4 [2, 1, 1])

theorem count_special_numbers_eq_18 :
  count_special_numbers = 18 := by
  sorry

#eval count_special_numbers

end NUMINAMATH_CALUDE_count_special_numbers_eq_18_l4027_402781


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l4027_402768

theorem solution_set_abs_inequality (x : ℝ) :
  (|1 - 2*x| < 3) ↔ (x ∈ Set.Ioo (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l4027_402768


namespace NUMINAMATH_CALUDE_image_of_two_is_five_l4027_402728

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem image_of_two_is_five : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_image_of_two_is_five_l4027_402728


namespace NUMINAMATH_CALUDE_max_brownies_is_294_l4027_402738

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  m : ℕ  -- length
  n : ℕ  -- width

/-- Calculates the number of interior pieces in a brownie pan -/
def interiorPieces (pan : BrowniePan) : ℕ :=
  (pan.m - 2) * (pan.n - 2)

/-- Calculates the number of perimeter pieces in a brownie pan -/
def perimeterPieces (pan : BrowniePan) : ℕ :=
  2 * pan.m + 2 * pan.n - 4

/-- Checks if the interior pieces are twice the perimeter pieces -/
def validCutting (pan : BrowniePan) : Prop :=
  interiorPieces pan = 2 * perimeterPieces pan

/-- Calculates the total number of brownies in a pan -/
def totalBrownies (pan : BrowniePan) : ℕ :=
  pan.m * pan.n

/-- Theorem: The maximum number of brownies is 294 given the conditions -/
theorem max_brownies_is_294 :
  ∃ (pan : BrowniePan), validCutting pan ∧
    (∀ (other : BrowniePan), validCutting other → totalBrownies other ≤ totalBrownies pan) ∧
    totalBrownies pan = 294 :=
  sorry

end NUMINAMATH_CALUDE_max_brownies_is_294_l4027_402738


namespace NUMINAMATH_CALUDE_symmetry_about_origin_l4027_402782

-- Define a function f and its inverse
variable (f : ℝ → ℝ)
variable (f_inv : ℝ → ℝ)

-- Define the property of f_inv being the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- Define the third function g
def g (x : ℝ) : ℝ := -f_inv (-x)

-- Theorem stating that g is symmetric to f_inv about the origin
theorem symmetry_about_origin :
  ∀ x, g (-x) = -g x :=
sorry

end NUMINAMATH_CALUDE_symmetry_about_origin_l4027_402782


namespace NUMINAMATH_CALUDE_perimeter_ABCD_l4027_402775

-- Define the points A, B, C, D, E
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the triangles
def is_right_angled (X Y Z : ℝ × ℝ) : Prop := sorry
def angle_equals_45_deg (X Y Z : ℝ × ℝ) : Prop := sorry
def is_45_45_90_triangle (X Y Z : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (X Y : ℝ × ℝ) : ℝ := sorry

-- Define the perimeter function for a quadrilateral
def perimeter_quadrilateral (W X Y Z : ℝ × ℝ) : ℝ :=
  distance W X + distance X Y + distance Y Z + distance Z W

-- State the theorem
theorem perimeter_ABCD (h1 : is_right_angled A B E)
                       (h2 : is_right_angled B C E)
                       (h3 : is_right_angled C D E)
                       (h4 : angle_equals_45_deg A E B)
                       (h5 : angle_equals_45_deg B E C)
                       (h6 : angle_equals_45_deg C E D)
                       (h7 : distance A E = 32)
                       (h8 : is_45_45_90_triangle A B E)
                       (h9 : is_45_45_90_triangle B C E)
                       (h10 : is_45_45_90_triangle C D E) :
  perimeter_quadrilateral A B C D = 32 + 32 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_perimeter_ABCD_l4027_402775


namespace NUMINAMATH_CALUDE_greatest_k_for_100_power_dividing_50_factorial_l4027_402712

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def highest_power_of_2 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n / 2) + highest_power_of_2 (n / 2)

def highest_power_of_5 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (n / 5) + highest_power_of_5 (n / 5)

theorem greatest_k_for_100_power_dividing_50_factorial :
  (∃ k : ℕ, k = 6 ∧
    ∀ m : ℕ, (100 ^ m : ℕ) ∣ factorial 50 → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_greatest_k_for_100_power_dividing_50_factorial_l4027_402712


namespace NUMINAMATH_CALUDE_a_2008_mod_4_l4027_402789

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => sequence_a n * sequence_a (n + 1) + 1

theorem a_2008_mod_4 : sequence_a 2008 % 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a_2008_mod_4_l4027_402789


namespace NUMINAMATH_CALUDE_distance_center_to_origin_l4027_402767

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define the center of the circle
def center_C : ℝ × ℝ := (1, 0)

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem distance_center_to_origin :
  Real.sqrt ((center_C.1 - origin.1)^2 + (center_C.2 - origin.2)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_distance_center_to_origin_l4027_402767


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l4027_402772

theorem quadratic_always_positive (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + 1 > 0) → -2 < b ∧ b < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l4027_402772


namespace NUMINAMATH_CALUDE_zinc_copper_ratio_in_mixture_l4027_402713

/-- Represents the ratio of two quantities -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Represents the composition of a metal mixture -/
structure MetalMixture where
  totalWeight : ℝ
  zincWeight : ℝ

/-- Calculates the ratio of zinc to copper in a metal mixture -/
def zincCopperRatio (mixture : MetalMixture) : Ratio :=
  sorry

theorem zinc_copper_ratio_in_mixture :
  let mixture : MetalMixture := { totalWeight := 70, zincWeight := 31.5 }
  (zincCopperRatio mixture).numerator = 9 ∧
  (zincCopperRatio mixture).denominator = 11 :=
by sorry

end NUMINAMATH_CALUDE_zinc_copper_ratio_in_mixture_l4027_402713


namespace NUMINAMATH_CALUDE_max_ab_value_l4027_402717

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  a * b ≤ (1 : ℝ) / 4 := by sorry

end NUMINAMATH_CALUDE_max_ab_value_l4027_402717


namespace NUMINAMATH_CALUDE_min_beta_value_l4027_402753

theorem min_beta_value (α β : ℕ+) 
  (h1 : (43 : ℚ) / 197 < α / β)
  (h2 : α / β < (17 : ℚ) / 77) :
  ∀ β' : ℕ+, ((43 : ℚ) / 197 < α / β' ∧ α / β' < (17 : ℚ) / 77) → β' ≥ 32 := by
  sorry

end NUMINAMATH_CALUDE_min_beta_value_l4027_402753


namespace NUMINAMATH_CALUDE_smallest_x_for_1260x_perfect_square_l4027_402755

theorem smallest_x_for_1260x_perfect_square : 
  ∃ (x : ℕ+), 
    (∀ (y : ℕ+), ∃ (N : ℤ), 1260 * y = N^2 → x ≤ y) ∧
    (∃ (N : ℤ), 1260 * x = N^2) ∧
    x = 35 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_1260x_perfect_square_l4027_402755


namespace NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l4027_402792

theorem sin_sum_of_complex_exponentials (α β : ℝ) :
  Complex.exp (Complex.I * α) = 3/5 + 4/5 * Complex.I ∧
  Complex.exp (Complex.I * β) = -12/13 + 5/13 * Complex.I →
  Real.sin (α + β) = -33/65 := by
sorry

end NUMINAMATH_CALUDE_sin_sum_of_complex_exponentials_l4027_402792


namespace NUMINAMATH_CALUDE_find_z2_l4027_402770

def complex_i : ℂ := Complex.I

theorem find_z2 (z1 z2 : ℂ) : 
  ((z1 - 2) * (1 + complex_i) = 1 - complex_i) →
  (z2.im = 2) →
  ((z1 * z2).im = 0) →
  z2 = 4 + 2 * complex_i :=
by sorry

end NUMINAMATH_CALUDE_find_z2_l4027_402770


namespace NUMINAMATH_CALUDE_fourth_sample_is_20_l4027_402757

def random_numbers : List ℕ := [71, 11, 5, 65, 9, 95, 86, 68, 76, 83, 20, 37, 90, 57, 16, 3, 11, 63, 14, 90]

def is_valid_sample (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 50

def get_fourth_sample (numbers : List ℕ) : ℕ :=
  (numbers.filter is_valid_sample).nthLe 3 sorry

theorem fourth_sample_is_20 :
  get_fourth_sample random_numbers = 20 := by sorry

end NUMINAMATH_CALUDE_fourth_sample_is_20_l4027_402757


namespace NUMINAMATH_CALUDE_n_fourth_plus_four_prime_iff_n_eq_one_l4027_402700

theorem n_fourth_plus_four_prime_iff_n_eq_one (n : ℕ+) :
  Nat.Prime (n^4 + 4) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_n_fourth_plus_four_prime_iff_n_eq_one_l4027_402700


namespace NUMINAMATH_CALUDE_line_intersection_triangle_l4027_402743

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Line type
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define a function to check if three points are collinear
def collinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

-- Define a function to check if a point lies on a line
def pointOnLine (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Define a function to check if a line intersects a segment
def lineIntersectsSegment (L : Line) (A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    pointOnLine (Point.mk (A.x + t * (B.x - A.x)) (A.y + t * (B.y - A.y))) L

-- Main theorem
theorem line_intersection_triangle (A B C : Point) (L : Line) :
  ¬collinear A B C →
  ¬pointOnLine A L →
  ¬pointOnLine B L →
  ¬pointOnLine C L →
  (¬lineIntersectsSegment L B C ∧ ¬lineIntersectsSegment L C A ∧ ¬lineIntersectsSegment L A B) ∨
  (lineIntersectsSegment L B C ∧ lineIntersectsSegment L C A ∧ ¬lineIntersectsSegment L A B) ∨
  (lineIntersectsSegment L B C ∧ ¬lineIntersectsSegment L C A ∧ lineIntersectsSegment L A B) ∨
  (¬lineIntersectsSegment L B C ∧ lineIntersectsSegment L C A ∧ lineIntersectsSegment L A B) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_triangle_l4027_402743


namespace NUMINAMATH_CALUDE_quadratic_general_form_l4027_402727

theorem quadratic_general_form :
  ∀ x : ℝ, x^2 = 3*x + 1 ↔ x^2 - 3*x - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_general_form_l4027_402727


namespace NUMINAMATH_CALUDE_variance_of_letters_l4027_402749

def letters : List ℕ := [10, 6, 8, 5, 6]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

def variance (l : List ℕ) : ℚ :=
  let μ := mean l
  (l.map (fun x => ((x : ℚ) - μ)^2)).sum / l.length

theorem variance_of_letters :
  variance letters = 16/5 := by sorry

end NUMINAMATH_CALUDE_variance_of_letters_l4027_402749
