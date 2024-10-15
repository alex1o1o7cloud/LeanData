import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_side_c_l3447_344783

theorem right_triangle_side_c (a b c : ℝ) : 
  a = 3 → b = 4 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 5 ∨ c = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_c_l3447_344783


namespace NUMINAMATH_CALUDE_log_equality_implies_y_value_l3447_344763

theorem log_equality_implies_y_value (m y : ℝ) (hm : m > 0) (hy : y > 0) :
  (Real.log y / Real.log m) * (Real.log m / Real.log 7) = 4 → y = 2401 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_y_value_l3447_344763


namespace NUMINAMATH_CALUDE_probability_two_green_marbles_l3447_344751

/-- The probability of drawing two green marbles without replacement from a jar -/
theorem probability_two_green_marbles (red green white blue : ℕ) 
  (h_red : red = 3)
  (h_green : green = 4)
  (h_white : white = 10)
  (h_blue : blue = 5) :
  let total := red + green + white + blue
  let prob_first := green / total
  let prob_second := (green - 1) / (total - 1)
  prob_first * prob_second = 2 / 77 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_green_marbles_l3447_344751


namespace NUMINAMATH_CALUDE_remainder_problem_l3447_344733

theorem remainder_problem : 2851 * 7347 * 419^2 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3447_344733


namespace NUMINAMATH_CALUDE_expression_percentage_of_y_l3447_344714

theorem expression_percentage_of_y (y : ℝ) (z : ℂ) (h : y > 0) :
  ((6 * y + 3 * z * Complex.I) / 20 + (3 * y + 4 * z * Complex.I) / 10) / y = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_expression_percentage_of_y_l3447_344714


namespace NUMINAMATH_CALUDE_power_division_equals_27_l3447_344797

theorem power_division_equals_27 : 3^12 / 27^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_27_l3447_344797


namespace NUMINAMATH_CALUDE_video_game_players_l3447_344758

theorem video_game_players (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : 
  players_quit = 8 →
  lives_per_player = 6 →
  total_lives = 30 →
  players_quit + (total_lives / lives_per_player) = 13 := by
sorry

end NUMINAMATH_CALUDE_video_game_players_l3447_344758


namespace NUMINAMATH_CALUDE_vote_difference_is_42_l3447_344752

/-- Proves that the difference in votes for the bill between re-vote and original vote is 42 -/
theorem vote_difference_is_42 
  (total_members : ℕ) 
  (original_for original_against : ℕ) 
  (revote_for revote_against : ℕ) :
  total_members = 400 →
  original_for + original_against = total_members →
  original_against > original_for →
  revote_for + revote_against = total_members →
  revote_for > revote_against →
  (revote_for - revote_against) = 3 * (original_against - original_for) →
  revote_for = (11 * original_against) / 10 →
  revote_for - original_for = 42 := by
sorry


end NUMINAMATH_CALUDE_vote_difference_is_42_l3447_344752


namespace NUMINAMATH_CALUDE_line_curve_intersection_l3447_344795

-- Define the line
def line (x : ℝ) : ℝ := x + 3

-- Define the curve
def curve (x y : ℝ) : Prop := y^2 / 9 - x * abs x / 4 = 1

-- State the theorem
theorem line_curve_intersection :
  ∃! (points : Finset (ℝ × ℝ)), 
    points.card = 3 ∧ 
    (∀ p ∈ points, curve p.1 p.2 ∧ p.2 = line p.1) ∧
    (∀ x y, curve x y ∧ y = line x → (x, y) ∈ points) :=
sorry

end NUMINAMATH_CALUDE_line_curve_intersection_l3447_344795


namespace NUMINAMATH_CALUDE_rosys_age_l3447_344730

/-- Proves Rosy's current age given the conditions of the problem -/
theorem rosys_age :
  ∀ (rosy_age : ℕ),
  (∃ (david_age : ℕ),
    david_age = rosy_age + 18 ∧
    david_age + 6 = 2 * (rosy_age + 6)) →
  rosy_age = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_rosys_age_l3447_344730


namespace NUMINAMATH_CALUDE_smallest_crate_dimension_l3447_344736

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.radius * 2 ≤ crate.height)

theorem smallest_crate_dimension (x : ℝ) :
  let crate := CrateDimensions.mk x 8 12
  let cylinder := Cylinder.mk 6 (max x (max 8 12))
  cylinderFitsInCrate crate cylinder →
  min x (min 8 12) = 8 := by
  sorry

#check smallest_crate_dimension

end NUMINAMATH_CALUDE_smallest_crate_dimension_l3447_344736


namespace NUMINAMATH_CALUDE_base_seven_234567_equals_41483_l3447_344713

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_234567_equals_41483 :
  base_seven_to_decimal [7, 6, 5, 4, 3, 2] = 41483 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_234567_equals_41483_l3447_344713


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3447_344787

theorem more_girls_than_boys (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 42 →
  3 * girls = 4 * boys →
  total_students = boys + girls →
  girls - boys = 6 := by
sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3447_344787


namespace NUMINAMATH_CALUDE_mayor_approval_probability_l3447_344743

/-- The probability of a voter approving the mayor's work -/
def p_approve : ℝ := 0.6

/-- The number of voters selected -/
def n_voters : ℕ := 4

/-- The number of approvals we're interested in -/
def k_approvals : ℕ := 2

/-- The probability of exactly k_approvals in n_voters independent trials -/
def prob_k_approvals (p : ℝ) (n k : ℕ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

theorem mayor_approval_probability :
  prob_k_approvals p_approve n_voters k_approvals = 0.3456 := by
  sorry

end NUMINAMATH_CALUDE_mayor_approval_probability_l3447_344743


namespace NUMINAMATH_CALUDE_notebook_cost_l3447_344746

theorem notebook_cost (total_students : ℕ) (total_cost : ℕ) 
  (h1 : total_students = 36)
  (h2 : ∃ (buyers : ℕ) (notebooks_per_student : ℕ) (cost_per_notebook : ℕ),
    buyers > total_students / 2 ∧
    notebooks_per_student > 1 ∧
    cost_per_notebook > notebooks_per_student ∧
    buyers * notebooks_per_student * cost_per_notebook = total_cost)
  (h3 : total_cost = 2310) :
  ∃ (cost_per_notebook : ℕ), cost_per_notebook = 11 ∧
    ∃ (buyers : ℕ) (notebooks_per_student : ℕ),
      buyers > total_students / 2 ∧
      notebooks_per_student > 1 ∧
      cost_per_notebook > notebooks_per_student ∧
      buyers * notebooks_per_student * cost_per_notebook = total_cost :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l3447_344746


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3447_344757

def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a₁ d : ℚ) (h₁ : a₁ = 1/2) (h₂ : d = 1/2) :
  arithmeticSequence a₁ d 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3447_344757


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l3447_344782

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 16

/-- Represents the number of people eating pizza -/
def num_people : ℕ := 4

/-- Represents the number of people eating both types of pizza -/
def num_people_both_types : ℕ := 3

/-- Represents the number of cheese slices left -/
def cheese_slices_left : ℕ := 7

/-- Represents the number of pepperoni slices left -/
def pepperoni_slices_left : ℕ := 1

/-- Represents the total number of slices each person eats -/
def slices_per_person : ℕ := 6

theorem pizza_slices_theorem :
  slices_per_person * num_people = 
    2 * slices_per_pizza - cheese_slices_left - pepperoni_slices_left :=
by sorry

end NUMINAMATH_CALUDE_pizza_slices_theorem_l3447_344782


namespace NUMINAMATH_CALUDE_rectangle_area_lower_bound_l3447_344717

theorem rectangle_area_lower_bound 
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : a * x = 1) 
  (eq2 : c * x = 3) 
  (eq3 : b * y = 10) 
  (eq4 : a * z = 9) : 
  (a + b + c) * (x + y + z) ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_lower_bound_l3447_344717


namespace NUMINAMATH_CALUDE_rhea_and_husband_eggs_per_night_l3447_344776

/-- The number of egg trays Rhea buys every week -/
def trays_per_week : ℕ := 2

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 24

/-- The number of eggs eaten by each child every morning -/
def eggs_per_child_per_morning : ℕ := 2

/-- The number of children -/
def number_of_children : ℕ := 2

/-- The number of eggs not eaten every week -/
def uneaten_eggs_per_week : ℕ := 6

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that Rhea and her husband eat 2 eggs every night -/
theorem rhea_and_husband_eggs_per_night :
  (trays_per_week * eggs_per_tray - 
   eggs_per_child_per_morning * number_of_children * days_per_week - 
   uneaten_eggs_per_week) / days_per_week = 2 := by
  sorry

end NUMINAMATH_CALUDE_rhea_and_husband_eggs_per_night_l3447_344776


namespace NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l3447_344766

/-- Given a triangle with area S, N is the area of the triangle formed by connecting
    the midpoints of its sides, and P is the area of the triangle formed by connecting
    the midpoints of the sides of the triangle with area N. -/
theorem midpoint_triangle_area_ratio (S N P : ℝ) (hS : S > 0) (hN : N > 0) (hP : P > 0)
  (hN_def : N = S / 4) (hP_def : P = N / 4) : P / S = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_triangle_area_ratio_l3447_344766


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l3447_344731

theorem quadratic_equation_proof (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 + (2*k - 1)*x + k^2 - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 + x₂^2 = 16 + x₁*x₂) →
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l3447_344731


namespace NUMINAMATH_CALUDE_regression_result_l3447_344739

/-- The regression equation -/
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 2.57

/-- Theorem: For the given regression equation, when x = 28, y = 135.57 -/
theorem regression_result : regression_equation 28 = 135.57 := by
  sorry

end NUMINAMATH_CALUDE_regression_result_l3447_344739


namespace NUMINAMATH_CALUDE_largest_prime_factor_l3447_344793

theorem largest_prime_factor (n : ℕ) : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (18^4 + 3 * 18^2 + 1 - 17^4) ∧ 
   ∀ q : ℕ, Nat.Prime q → q ∣ (18^4 + 3 * 18^2 + 1 - 17^4) → q ≤ p) →
  (∃ p : ℕ, p = 307 ∧ Nat.Prime p ∧ p ∣ (18^4 + 3 * 18^2 + 1 - 17^4) ∧ 
   ∀ q : ℕ, Nat.Prime q → q ∣ (18^4 + 3 * 18^2 + 1 - 17^4) → q ≤ p) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_l3447_344793


namespace NUMINAMATH_CALUDE_red_pigment_in_brown_l3447_344715

/-- Represents the composition of a paint mixture -/
structure PaintMixture where
  blue : Real
  red : Real
  yellow : Real
  weight : Real

/-- The sky blue paint composition -/
def skyBlue : PaintMixture := {
  blue := 0.1
  red := 0.9
  yellow := 0
  weight := 1
}

/-- The green paint composition -/
def green : PaintMixture := {
  blue := 0.7
  red := 0
  yellow := 0.3
  weight := 1
}

/-- The resulting brown paint composition -/
def brown : PaintMixture := {
  blue := 0.4
  red := 0
  yellow := 0
  weight := 10
}

/-- Theorem stating the amount of red pigment in the brown paint -/
theorem red_pigment_in_brown :
  ∃ (x y : Real),
    x + y = brown.weight ∧
    x * skyBlue.blue + y * green.blue = brown.blue * brown.weight ∧
    x * skyBlue.red = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_red_pigment_in_brown_l3447_344715


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l3447_344738

theorem mixed_number_calculation : 
  53 * ((3 + 1/5) - (4 + 1/2)) / ((2 + 3/4) + (1 + 2/3)) = -(15 + 3/5) := by sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l3447_344738


namespace NUMINAMATH_CALUDE_circle_segment_distance_squared_l3447_344702

theorem circle_segment_distance_squared (r AB BC : ℝ) (angle_ABC : ℝ) : 
  r = Real.sqrt 75 →
  AB = 7 →
  BC = 3 →
  angle_ABC = 2 * Real.pi / 3 →
  ∃ (O B : ℝ × ℝ), 
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = r^2 ∧
    (B.1 - O.1)^2 + (B.2 - O.2)^2 = 61 :=
by sorry

end NUMINAMATH_CALUDE_circle_segment_distance_squared_l3447_344702


namespace NUMINAMATH_CALUDE_circle_center_and_equation_l3447_344770

/-- A circle passing through two points with a given radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passesThrough : (ℝ × ℝ) → Prop

/-- The line passing through two points -/
def Line (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r : ℝ × ℝ | ∃ t : ℝ, r = (1 - t) • p + t • q}

theorem circle_center_and_equation 
  (C : Circle) 
  (h1 : C.passesThrough (1, 0)) 
  (h2 : C.passesThrough (0, 1)) 
  (h3 : C.radius = 1) : 
  (∃ t : ℝ, C.center = (t, t)) ∧ 
  (∀ x y : ℝ, C.passesThrough (x, y) ↔ x^2 + y^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_circle_center_and_equation_l3447_344770


namespace NUMINAMATH_CALUDE_shelf_filling_problem_l3447_344761

/-- Represents the shelf filling problem with biology and geography books -/
theorem shelf_filling_problem 
  (B G P Q K : ℕ) 
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ K ∧ 
                G ≠ P ∧ G ≠ Q ∧ G ≠ K ∧ 
                P ≠ Q ∧ P ≠ K ∧ 
                Q ≠ K)
  (h_positive : B > 0 ∧ G > 0 ∧ P > 0 ∧ Q > 0 ∧ K > 0)
  (h_fill1 : ∃ (a : ℚ), a > 0 ∧ B * a + G * (2 * a) = K * a)
  (h_fill2 : ∃ (a : ℚ), a > 0 ∧ P * a + Q * (2 * a) = K * a) :
  K = B + 2 * G :=
sorry

end NUMINAMATH_CALUDE_shelf_filling_problem_l3447_344761


namespace NUMINAMATH_CALUDE_line_translation_invariance_l3447_344759

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + dy - l.slope * dx }

theorem line_translation_invariance (l : Line) (dx dy : ℝ) :
  l.slope = -2 ∧ l.intercept = -2 ∧ dx = -1 ∧ dy = 2 →
  translate l dx dy = l :=
sorry

end NUMINAMATH_CALUDE_line_translation_invariance_l3447_344759


namespace NUMINAMATH_CALUDE_frank_lamp_purchase_l3447_344732

/-- Frank's lamp purchase problem -/
theorem frank_lamp_purchase (frank_money : ℕ) (cheapest_lamp : ℕ) (expensive_factor : ℕ) :
  frank_money = 90 →
  cheapest_lamp = 20 →
  expensive_factor = 3 →
  frank_money - (cheapest_lamp * expensive_factor) = 30 := by
  sorry

end NUMINAMATH_CALUDE_frank_lamp_purchase_l3447_344732


namespace NUMINAMATH_CALUDE_weight_difference_proof_l3447_344791

/-- Proves the difference between the average weight of two departing students and Joe's weight --/
theorem weight_difference_proof 
  (n : ℕ) -- number of students in the original group
  (initial_avg : ℝ) -- initial average weight
  (joe_weight : ℝ) -- Joe's weight
  (new_avg : ℝ) -- new average weight after Joe joins
  (final_avg : ℝ) -- final average weight after two students leave
  (h1 : initial_avg = 30)
  (h2 : joe_weight = 43)
  (h3 : new_avg = initial_avg + 1)
  (h4 : final_avg = initial_avg)
  (h5 : (n * initial_avg + joe_weight) / (n + 1) = new_avg)
  (h6 : ((n + 1) * new_avg - 2 * final_avg) / (n - 1) = final_avg) :
  (((n + 1) * new_avg - n * final_avg) / 2) - joe_weight = -6.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_proof_l3447_344791


namespace NUMINAMATH_CALUDE_complex_exponential_form_l3447_344747

/-- Given a complex number z = e^a(cos b + i sin b), its exponential form is e^(a + ib) -/
theorem complex_exponential_form (a b : ℝ) :
  let z : ℂ := Complex.exp a * (Complex.cos b + Complex.I * Complex.sin b)
  z = Complex.exp (a + Complex.I * b) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_form_l3447_344747


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3447_344794

/-- The perimeter of a regular polygon with side length 8 and exterior angle 72 degrees is 40 units. -/
theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) : 
  side_length = 8 → 
  exterior_angle = 72 → 
  (360 / exterior_angle) * side_length = 40 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3447_344794


namespace NUMINAMATH_CALUDE_line_intersects_circle_twice_min_chord_line_equation_l3447_344792

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem 1: L always intersects C at two points for any real m
theorem line_intersects_circle_twice :
  ∀ m : ℝ, ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ C p1.1 p1.2 ∧ C p2.1 p2.2 ∧ L m p1.1 p1.2 ∧ L m p2.1 p2.2 :=
sorry

-- Theorem 2: When chord length is minimum, L has equation y = 2x - 5
theorem min_chord_line_equation :
  ∃! m : ℝ, (∀ x y : ℝ, L m x y ↔ y = 2*x - 5) ∧
  (∀ m' : ℝ, m' ≠ m → 
    ∃ p1 p2 q1 q2 : ℝ × ℝ, p1 ≠ p2 ∧ q1 ≠ q2 ∧
    C p1.1 p1.2 ∧ C p2.1 p2.2 ∧ L m p1.1 p1.2 ∧ L m p2.1 p2.2 ∧
    C q1.1 q1.2 ∧ C q2.1 q2.2 ∧ L m' q1.1 q1.2 ∧ L m' q2.1 q2.2 ∧
    (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 < (q1.1 - q2.1)^2 + (q1.2 - q2.2)^2) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_twice_min_chord_line_equation_l3447_344792


namespace NUMINAMATH_CALUDE_min_value_of_f_inequality_abc_l3447_344790

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m :=
sorry

-- Theorem for the inequality
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_6 : a + b + c = 6) : 
  a^2 + b^2 + c^2 ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_inequality_abc_l3447_344790


namespace NUMINAMATH_CALUDE_line_equation_with_opposite_intercepts_l3447_344721

/-- A line passing through a given point with opposite intercepts on the coordinate axes -/
structure LineWithOppositeIntercepts where
  -- The x-coordinate of the point
  x : ℝ
  -- The y-coordinate of the point
  y : ℝ
  -- The equation of the line in the form ax + by + c = 0
  a : ℝ
  b : ℝ
  c : ℝ
  -- The line passes through the point (x, y)
  point_on_line : a * x + b * y + c = 0
  -- The intercepts are opposite in value
  opposite_intercepts : a * c = -b * c ∨ a = 0 ∧ b = 0 ∧ c = 0

/-- The equation of a line with opposite intercepts passing through (3, -2) -/
theorem line_equation_with_opposite_intercepts :
  ∀ (l : LineWithOppositeIntercepts),
  l.x = 3 ∧ l.y = -2 →
  (l.a = 2 ∧ l.b = 3 ∧ l.c = 0) ∨ (l.a = 1 ∧ l.b = -1 ∧ l.c = -5) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_with_opposite_intercepts_l3447_344721


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3447_344724

theorem min_value_of_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ m : ℝ, m = 1 - Real.sqrt 2 ∧ ∀ z : ℝ, z = x*y/(x+y-2) → m ≤ z :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3447_344724


namespace NUMINAMATH_CALUDE_subset_condition_l3447_344716

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- State the theorem
theorem subset_condition (a : ℝ) : A a ⊆ (A a ∩ B) ↔ 6 ≤ a ∧ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_subset_condition_l3447_344716


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l3447_344703

/-- The number of ways to choose 3 books from 5 different books for 3 students -/
def choose_books : ℕ := 60

/-- The number of ways to buy 3 books from 5 different books for 3 students -/
def buy_books : ℕ := 125

/-- The number of different books available -/
def num_books : ℕ := 5

/-- The number of students receiving books -/
def num_students : ℕ := 3

theorem book_distribution_theorem :
  (choose_books = num_books * (num_books - 1) * (num_books - 2)) ∧
  (buy_books = num_books * num_books * num_books) := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l3447_344703


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l3447_344762

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 12000) → 
  (round x : ℤ) = 33097 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l3447_344762


namespace NUMINAMATH_CALUDE_mi_gu_li_fen_problem_l3447_344768

/-- The "Mi-Gu-Li-Fen" problem from the "Mathematical Treatise in Nine Sections" -/
theorem mi_gu_li_fen_problem (total_mixture : ℚ) (sample_size : ℕ) (wheat_in_sample : ℕ) 
  (h1 : total_mixture = 1512)
  (h2 : sample_size = 216)
  (h3 : wheat_in_sample = 27) :
  (total_mixture * (wheat_in_sample : ℚ) / (sample_size : ℚ)) = 189 := by
  sorry

end NUMINAMATH_CALUDE_mi_gu_li_fen_problem_l3447_344768


namespace NUMINAMATH_CALUDE_min_value_theorem_l3447_344780

theorem min_value_theorem (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0)
  (h5 : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + 4 * y^2 - c = 0 →
    (2 * x + y)^2 ≤ (2 * a + b)^2) :
  ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + 4 * y^2 - z = 0 →
    3 / a - 4 / b + 5 / c ≤ 3 / x - 4 / y + 5 / z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3447_344780


namespace NUMINAMATH_CALUDE_x_plus_q_equals_five_l3447_344764

theorem x_plus_q_equals_five (x q : ℝ) (h1 : |x - 5| = q) (h2 : x < 5) : x + q = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_q_equals_five_l3447_344764


namespace NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3447_344726

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) (set2_count : ℕ) (set2_mean : ℚ) :
  set1_count = 7 →
  set1_mean = 16 →
  set2_count = 9 →
  set2_mean = 20 →
  let total_count := set1_count + set2_count
  let combined_sum := set1_count * set1_mean + set2_count * set2_mean
  combined_sum / total_count = 18.25 := by
  sorry

end NUMINAMATH_CALUDE_combined_mean_of_two_sets_l3447_344726


namespace NUMINAMATH_CALUDE_unique_matrix_solution_l3447_344754

open Matrix

theorem unique_matrix_solution {n : ℕ} (A : Matrix (Fin n) (Fin n) ℝ) 
  (h : A ^ 3 = 0) : 
  ∃! X : Matrix (Fin n) (Fin n) ℝ, X + A * X + X * A ^ 2 = A ∧ 
    X = A * (1 + A + A ^ 2)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_unique_matrix_solution_l3447_344754


namespace NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l3447_344799

theorem halfway_between_one_third_and_one_fifth :
  (1 / 3 : ℚ) + (1 / 5 : ℚ) = 2 * (4 / 15 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_halfway_between_one_third_and_one_fifth_l3447_344799


namespace NUMINAMATH_CALUDE_a_values_l3447_344704

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define the set of possible values for a
def possible_a : Set ℝ := {-1, 0, 2/3}

-- Statement to prove
theorem a_values (a : ℝ) : (N a ⊆ M) ↔ a ∈ possible_a := by sorry

end NUMINAMATH_CALUDE_a_values_l3447_344704


namespace NUMINAMATH_CALUDE_same_remainder_divisor_l3447_344774

theorem same_remainder_divisor : ∃ (N : ℕ), N > 1 ∧ 
  N = 23 ∧ 
  (1743 % N = 2019 % N) ∧ 
  (2019 % N = 3008 % N) ∧ 
  ∀ (M : ℕ), M > N → (1743 % M ≠ 2019 % M ∨ 2019 % M ≠ 3008 % M) := by
  sorry

end NUMINAMATH_CALUDE_same_remainder_divisor_l3447_344774


namespace NUMINAMATH_CALUDE_sum_remainder_mod_13_l3447_344734

theorem sum_remainder_mod_13 : (9123 + 9124 + 9125 + 9126) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_13_l3447_344734


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3447_344705

theorem square_plus_reciprocal_square (m : ℝ) (h : m + 1/m = 10) :
  m^2 + 1/m^2 + 6 = 104 := by
sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3447_344705


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3447_344760

theorem quadratic_expression_value
  (a b c x : ℝ)
  (h1 : (2 - a)^2 + Real.sqrt (a^2 + b + c) + |c + 8| = 0)
  (h2 : a * x^2 + b * x + c = 0) :
  3 * x^2 + 6 * x + 1 = 13 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3447_344760


namespace NUMINAMATH_CALUDE_system_no_solution_l3447_344777

theorem system_no_solution (n : ℝ) : 
  (∀ x y z : ℝ, n^2 * x + y ≠ 1 ∨ n * y + z ≠ 1 ∨ x + n^2 * z ≠ 1) ↔ n = -1 := by
  sorry

end NUMINAMATH_CALUDE_system_no_solution_l3447_344777


namespace NUMINAMATH_CALUDE_root_of_polynomial_l3447_344789

theorem root_of_polynomial (b : ℝ) (h : b^5 = 2 - Real.sqrt 3) :
  (b + (2 + Real.sqrt 3)^(1/5 : ℝ))^5 - 5*(b + (2 + Real.sqrt 3)^(1/5 : ℝ))^3 + 
  5*(b + (2 + Real.sqrt 3)^(1/5 : ℝ)) - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l3447_344789


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l3447_344745

theorem binomial_expansion_constant_term (n : ℕ) : 
  (∃ k : ℚ, 2 * (n.choose 2) = (n.choose 1) + k ∧ (n.choose 3) = (n.choose 2) + k) →
  (∃ r : ℕ, r ≤ n ∧ 21 = 7 * r ∧ n.choose r = 35) := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l3447_344745


namespace NUMINAMATH_CALUDE_bottles_lasted_74_days_l3447_344718

/-- The number of bottles Debby bought -/
def total_bottles : ℕ := 8066

/-- The number of bottles Debby drank per day -/
def bottles_per_day : ℕ := 109

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := total_bottles / bottles_per_day

theorem bottles_lasted_74_days : days_lasted = 74 := by
  sorry

end NUMINAMATH_CALUDE_bottles_lasted_74_days_l3447_344718


namespace NUMINAMATH_CALUDE_tan_alpha_three_implies_expression_equals_two_l3447_344781

theorem tan_alpha_three_implies_expression_equals_two (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin (α - π) + Real.cos (π - α)) / 
  (Real.sin (π / 2 - α) + Real.cos (π / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_three_implies_expression_equals_two_l3447_344781


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_12_l3447_344728

def is_valid_number (a b : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ a + b = 11 ∧ (a * 1000 + 520 + b) % 12 = 0

theorem four_digit_divisible_by_12 :
  ∀ a b : ℕ, is_valid_number a b → (a = 7 ∧ b = 4) ∨ (a = 3 ∧ b = 8) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_12_l3447_344728


namespace NUMINAMATH_CALUDE_city_mpg_is_14_l3447_344737

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway_miles_per_tankful : ℝ
  city_miles_per_tankful : ℝ
  city_mpg_difference : ℝ

/-- Calculates the city miles per gallon given the car's fuel efficiency data -/
def calculate_city_mpg (car : CarFuelEfficiency) : ℝ :=
  sorry

/-- Theorem stating that for a car with given fuel efficiency data, 
    the city miles per gallon is 14 -/
theorem city_mpg_is_14 (car : CarFuelEfficiency) 
  (h1 : car.highway_miles_per_tankful = 480)
  (h2 : car.city_miles_per_tankful = 336)
  (h3 : car.city_mpg_difference = 6) :
  calculate_city_mpg car = 14 := by
  sorry

end NUMINAMATH_CALUDE_city_mpg_is_14_l3447_344737


namespace NUMINAMATH_CALUDE_sports_lottery_winners_l3447_344711

theorem sports_lottery_winners
  (win : Prop → Prop)
  (A B C D : Prop)
  (h1 : A → B)
  (h2 : B → (C ∨ ¬A))
  (h3 : ¬D → (A ∧ ¬C))
  (h4 : D → A) :
  A ∧ B ∧ C ∧ D :=
by sorry

end NUMINAMATH_CALUDE_sports_lottery_winners_l3447_344711


namespace NUMINAMATH_CALUDE_ice_cream_theorem_l3447_344744

theorem ice_cream_theorem (n : ℕ) (h : n > 7) : ∃ x y : ℕ, 3 * x + 5 * y = n := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_theorem_l3447_344744


namespace NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half_l3447_344788

theorem tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half :
  (Real.tan (22.5 * π / 180)) / (1 - (Real.tan (22.5 * π / 180))^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_over_one_minus_tan_22_5_deg_squared_eq_half_l3447_344788


namespace NUMINAMATH_CALUDE_ratio_from_mean_ratio_l3447_344742

theorem ratio_from_mean_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (a + b) / 2 / Real.sqrt (a * b) = 25 / 24 →
  a / b = 16 / 9 ∨ a / b = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_from_mean_ratio_l3447_344742


namespace NUMINAMATH_CALUDE_find_divisor_l3447_344710

theorem find_divisor (x y n : ℕ+) : 
  x = n * y + 4 →
  2 * x = 14 * y + 1 →
  5 * y - x = 3 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l3447_344710


namespace NUMINAMATH_CALUDE_binomial_congruence_l3447_344778

theorem binomial_congruence (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 1 ≤ k ∧ k ≤ p - 1) :
  (Nat.choose (p - 1) k) ≡ ((-1 : ℤ) ^ k) [ZMOD p] := by
  sorry

end NUMINAMATH_CALUDE_binomial_congruence_l3447_344778


namespace NUMINAMATH_CALUDE_hannah_unique_number_l3447_344712

/-- Represents a student's counting sequence -/
structure StudentSequence where
  start : Nat
  step : Nat

/-- The set of all numbers from 1 to 1200 -/
def allNumbers : Set Nat := {n | 1 ≤ n ∧ n ≤ 1200}

/-- Generate a sequence for a student -/
def generateSequence (s : StudentSequence) : Set Nat :=
  {n ∈ allNumbers | ∃ k, n = s.start + k * s.step}

/-- Alice's sequence -/
def aliceSeq : Set Nat := allNumbers \ (generateSequence ⟨4, 4⟩)

/-- Barbara's sequence -/
def barbaraSeq : Set Nat := (allNumbers \ aliceSeq) \ (generateSequence ⟨5, 5⟩)

/-- Candice's sequence -/
def candiceSeq : Set Nat := (allNumbers \ (aliceSeq ∪ barbaraSeq)) \ (generateSequence ⟨6, 6⟩)

/-- Debbie, Eliza, and Fatima's combined sequence -/
def defSeq : Set Nat := 
  (allNumbers \ (aliceSeq ∪ barbaraSeq ∪ candiceSeq)) \ 
  (generateSequence ⟨7, 7⟩ ∪ generateSequence ⟨14, 7⟩ ∪ generateSequence ⟨21, 7⟩)

/-- George's sequence -/
def georgeSeq : Set Nat := allNumbers \ (aliceSeq ∪ barbaraSeq ∪ candiceSeq ∪ defSeq)

/-- Hannah's number -/
def hannahNumber : Nat := 1189

/-- Theorem: Hannah's number is the only number not spoken by any other student -/
theorem hannah_unique_number : 
  hannahNumber ∈ allNumbers ∧ 
  hannahNumber ∉ aliceSeq ∧ 
  hannahNumber ∉ barbaraSeq ∧ 
  hannahNumber ∉ candiceSeq ∧ 
  hannahNumber ∉ defSeq ∧ 
  hannahNumber ∉ georgeSeq ∧
  ∀ n ∈ allNumbers, n ≠ hannahNumber → 
    n ∈ aliceSeq ∨ n ∈ barbaraSeq ∨ n ∈ candiceSeq ∨ n ∈ defSeq ∨ n ∈ georgeSeq := by
  sorry

end NUMINAMATH_CALUDE_hannah_unique_number_l3447_344712


namespace NUMINAMATH_CALUDE_solution_set_real_iff_k_less_than_neg_three_l3447_344725

theorem solution_set_real_iff_k_less_than_neg_three (k : ℝ) :
  (∀ x : ℝ, |x + 1| - |x - 2| > k) ↔ k < -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_real_iff_k_less_than_neg_three_l3447_344725


namespace NUMINAMATH_CALUDE_units_digit_3_pow_20_l3447_344722

def units_digit_pattern : ℕ → ℕ
| 0 => 3
| 1 => 9
| 2 => 7
| 3 => 1
| n + 4 => units_digit_pattern n

theorem units_digit_3_pow_20 :
  units_digit_pattern 19 = 1 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_3_pow_20_l3447_344722


namespace NUMINAMATH_CALUDE_pipe_problem_l3447_344756

theorem pipe_problem (fill_time : ℕ → ℝ) (h1 : fill_time 2 = 18) (h2 : ∃ n : ℕ, fill_time n = 12) : 
  ∃ n : ℕ, n = 3 ∧ fill_time n = 12 := by
  sorry

end NUMINAMATH_CALUDE_pipe_problem_l3447_344756


namespace NUMINAMATH_CALUDE_class_average_after_exclusion_l3447_344750

theorem class_average_after_exclusion (total_students : ℕ) (initial_avg : ℚ) 
  (excluded_students : ℕ) (excluded_avg : ℚ) :
  total_students = 30 →
  initial_avg = 80 →
  excluded_students = 5 →
  excluded_avg = 30 →
  let remaining_students := total_students - excluded_students
  let total_marks := initial_avg * total_students
  let excluded_marks := excluded_avg * excluded_students
  let remaining_marks := total_marks - excluded_marks
  (remaining_marks / remaining_students) = 90 := by
  sorry

end NUMINAMATH_CALUDE_class_average_after_exclusion_l3447_344750


namespace NUMINAMATH_CALUDE_shoes_cost_proof_l3447_344700

def budget : ℕ := 200
def shirt_cost : ℕ := 30
def pants_cost : ℕ := 46
def coat_cost : ℕ := 38
def socks_cost : ℕ := 11
def belt_cost : ℕ := 18
def remaining : ℕ := 16

theorem shoes_cost_proof :
  budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost) - remaining = 41 := by
  sorry

end NUMINAMATH_CALUDE_shoes_cost_proof_l3447_344700


namespace NUMINAMATH_CALUDE_shaded_area_is_correct_l3447_344767

/-- A square and a right triangle with equal height -/
structure GeometricSetup where
  /-- Height of both the square and the triangle -/
  height : ℝ
  /-- Base length of both the square and the triangle -/
  base : ℝ
  /-- The lower right vertex of the square and lower left vertex of the triangle -/
  intersection : ℝ × ℝ
  /-- Assertion that the height equals the base -/
  height_eq_base : height = base
  /-- Assertion that the intersection point is at (15, 0) -/
  intersection_is_fifteen : intersection = (15, 0)
  /-- Assertion that the base length is 15 -/
  base_is_fifteen : base = 15

/-- The area of the shaded region -/
def shaded_area (setup : GeometricSetup) : ℝ := 168.75

/-- Theorem stating that the shaded area is 168.75 square units -/
theorem shaded_area_is_correct (setup : GeometricSetup) : 
  shaded_area setup = 168.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_correct_l3447_344767


namespace NUMINAMATH_CALUDE_max_n_for_L_perfect_square_l3447_344741

/-- Definition of L(n): the number of permutations of {1,2,...,n} with exactly one landmark point -/
def L (n : ℕ) : ℕ := 4 * (2^(n-2) - 1)

/-- Theorem stating that 3 is the maximum n ≥ 3 for which L(n) is a perfect square -/
theorem max_n_for_L_perfect_square :
  ∀ n : ℕ, n ≥ 3 → (∃ k : ℕ, L n = k^2) → n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_n_for_L_perfect_square_l3447_344741


namespace NUMINAMATH_CALUDE_complex_power_difference_l3447_344785

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference : (2 + i)^12 - (2 - i)^12 = 503 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l3447_344785


namespace NUMINAMATH_CALUDE_sqrt_expression_simplification_l3447_344771

theorem sqrt_expression_simplification :
  (Real.sqrt 48 + Real.sqrt 20) - (Real.sqrt 12 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_simplification_l3447_344771


namespace NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l3447_344755

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- Calculates the amount of an ingredient based on a given amount of another ingredient -/
def calculate_ingredient (ratio : RecipeRatio) (known_amount : ℚ) (known_part : ℚ) (target_part : ℚ) : ℚ :=
  (target_part / known_part) * known_amount

theorem sugar_amount_in_new_recipe 
  (original_ratio : RecipeRatio)
  (h_original : original_ratio = ⟨11, 5, 2⟩)
  (new_ratio : RecipeRatio)
  (h_double_flour_water : new_ratio.flour / new_ratio.water = 2 * (original_ratio.flour / original_ratio.water))
  (h_half_flour_sugar : new_ratio.flour / new_ratio.sugar = (1/2) * (original_ratio.flour / original_ratio.sugar))
  (h_water_amount : calculate_ingredient new_ratio 7.5 new_ratio.water new_ratio.sugar = 6) :
  calculate_ingredient new_ratio 7.5 new_ratio.water new_ratio.sugar = 6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_in_new_recipe_l3447_344755


namespace NUMINAMATH_CALUDE_profit_at_35_selling_price_for_600_profit_no_900_profit_l3447_344707

/-- Represents the daily sales and profit model for a product in a shopping mall. -/
structure SalesModel where
  purchase_price : ℝ
  min_selling_price : ℝ
  max_selling_price : ℝ
  sales_volume : ℝ → ℝ
  profit : ℝ → ℝ

/-- The specific sales model for the given problem. -/
def mall_model : SalesModel :=
  { purchase_price := 30
    min_selling_price := 30
    max_selling_price := 55
    sales_volume := fun x => -2 * x + 140
    profit := fun x => (x - 30) * (-2 * x + 140) }

/-- Theorem 1: The daily profit when the selling price is 35 yuan is 350 yuan. -/
theorem profit_at_35 (model : SalesModel := mall_model) :
    model.profit 35 = 350 := by sorry

/-- Theorem 2: The selling price that yields a daily profit of 600 yuan is 40 yuan. -/
theorem selling_price_for_600_profit (model : SalesModel := mall_model) :
    ∃ x, model.min_selling_price ≤ x ∧ x ≤ model.max_selling_price ∧ model.profit x = 600 ∧ x = 40 := by sorry

/-- Theorem 3: There is no selling price within the given range that can yield a daily profit of 900 yuan. -/
theorem no_900_profit (model : SalesModel := mall_model) :
    ¬∃ x, model.min_selling_price ≤ x ∧ x ≤ model.max_selling_price ∧ model.profit x = 900 := by sorry

end NUMINAMATH_CALUDE_profit_at_35_selling_price_for_600_profit_no_900_profit_l3447_344707


namespace NUMINAMATH_CALUDE_product_equals_48_l3447_344784

theorem product_equals_48 : 12 * (1 / 7) * 14 * 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_48_l3447_344784


namespace NUMINAMATH_CALUDE_square_area_error_l3447_344749

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x + 0.38 * x
  let actual_area := x^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let percentage_error := (area_error / actual_area) * 100
  percentage_error = 90.44 := by
sorry

end NUMINAMATH_CALUDE_square_area_error_l3447_344749


namespace NUMINAMATH_CALUDE_volunteer_members_count_l3447_344719

/-- The number of sheets of cookies baked by each member -/
def sheets_per_member : ℕ := 10

/-- The number of cookies on each sheet -/
def cookies_per_sheet : ℕ := 16

/-- The total number of cookies baked -/
def total_cookies : ℕ := 16000

/-- The number of members who volunteered to bake cookies -/
def num_members : ℕ := total_cookies / (sheets_per_member * cookies_per_sheet)

theorem volunteer_members_count : num_members = 100 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_members_count_l3447_344719


namespace NUMINAMATH_CALUDE_boat_upstream_time_l3447_344720

theorem boat_upstream_time (B C : ℝ) (h1 : B = 4 * C) (h2 : B > 0) (h3 : C > 0) : 
  (10 : ℝ) * (B + C) / (B - C) = 50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_boat_upstream_time_l3447_344720


namespace NUMINAMATH_CALUDE_cost_of_5_spoons_l3447_344769

-- Define the cost of a set of 7 spoons
def cost_7_spoons : ℝ := 21

-- Define the number of spoons in a set
def spoons_in_set : ℕ := 7

-- Define the number of spoons we want to buy
def spoons_to_buy : ℕ := 5

-- Theorem: The cost of 5 spoons is $15
theorem cost_of_5_spoons :
  (cost_7_spoons / spoons_in_set) * spoons_to_buy = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_5_spoons_l3447_344769


namespace NUMINAMATH_CALUDE_circular_window_panes_l3447_344727

theorem circular_window_panes (r : ℝ) (x : ℝ) : 
  r = 20 → 
  (9 : ℝ) * (π * r^2) = π * (r + x)^2 → 
  x = 40 :=
by sorry

end NUMINAMATH_CALUDE_circular_window_panes_l3447_344727


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3447_344729

-- Define the sets P and Q
def P : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x | 3*x - 7 ≥ 8 - 2*x}

-- Theorem statement
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3447_344729


namespace NUMINAMATH_CALUDE_vector_operation_l3447_344748

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![0, -1]

theorem vector_operation :
  (3 • b - a) = ![(-3 : ℝ), -5] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3447_344748


namespace NUMINAMATH_CALUDE_probability_sum_greater_than_9_l3447_344735

def number_set : Finset ℕ := {1, 3, 5, 7, 9}

def sum_greater_than_9 (a b : ℕ) : Prop := a + b > 9

def valid_pair (a b : ℕ) : Prop := a ∈ number_set ∧ b ∈ number_set ∧ a ≠ b

theorem probability_sum_greater_than_9 :
  Nat.card {p : ℕ × ℕ | p.1 < p.2 ∧ valid_pair p.1 p.2 ∧ sum_greater_than_9 p.1 p.2} /
  Nat.card {p : ℕ × ℕ | p.1 < p.2 ∧ valid_pair p.1 p.2} = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_greater_than_9_l3447_344735


namespace NUMINAMATH_CALUDE_egyptian_fraction_sum_l3447_344786

theorem egyptian_fraction_sum : ∃! (b₂ b₃ b₄ b₅ b₆ : ℕ),
  (11 : ℚ) / 13 = b₂ / 6 + b₃ / 24 + b₄ / 120 + b₅ / 720 + b₆ / 5040 ∧
  b₂ < 3 ∧ b₃ < 4 ∧ b₄ < 5 ∧ b₅ < 6 ∧ b₆ < 7 ∧
  b₂ + b₃ + b₄ + b₅ + b₆ = 1751 := by
  sorry

end NUMINAMATH_CALUDE_egyptian_fraction_sum_l3447_344786


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3447_344773

theorem min_value_sum_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c) * ((a + b)⁻¹ + (a + c)⁻¹ + (b + c)⁻¹) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3447_344773


namespace NUMINAMATH_CALUDE_magical_elixir_combinations_l3447_344772

/-- The number of magical herbs. -/
def num_herbs : ℕ := 4

/-- The number of enchanted crystals. -/
def num_crystals : ℕ := 6

/-- The number of incompatible herb-crystal pairs. -/
def num_incompatible : ℕ := 3

/-- The number of valid combinations for the magical elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible

theorem magical_elixir_combinations :
  valid_combinations = 21 :=
by sorry

end NUMINAMATH_CALUDE_magical_elixir_combinations_l3447_344772


namespace NUMINAMATH_CALUDE_intersection_A_complementB_l3447_344708

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Define the complement of B with respect to U
def complementB : Set ℝ := U \ B

-- Theorem statement
theorem intersection_A_complementB : A ∩ complementB = {x | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complementB_l3447_344708


namespace NUMINAMATH_CALUDE_broken_glass_problem_l3447_344723

/-- The number of broken glass pieces during transportation --/
def broken_glass (total : ℕ) (safe_fee : ℕ) (compensation : ℕ) (total_fee : ℕ) : ℕ :=
  total - (total_fee + total * safe_fee) / (safe_fee + compensation)

theorem broken_glass_problem :
  broken_glass 100 3 5 260 = 5 := by
  sorry

end NUMINAMATH_CALUDE_broken_glass_problem_l3447_344723


namespace NUMINAMATH_CALUDE_correct_substitution_proof_l3447_344796

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 2 * x - y = 5
def equation2 (x y : ℝ) : Prop := y = 1 + x

-- Define the correct substitution
def correct_substitution (x : ℝ) : Prop := 2 * x - 1 - x = 5

-- Theorem statement
theorem correct_substitution_proof :
  ∀ x y : ℝ, equation1 x y ∧ equation2 x y → correct_substitution x :=
by sorry

end NUMINAMATH_CALUDE_correct_substitution_proof_l3447_344796


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3447_344706

theorem smallest_n_congruence : ∃! n : ℕ+, 
  (∀ m : ℕ+, 13 * m ≡ 456 [ZMOD 5] → n ≤ m) ∧ 
  13 * n ≡ 456 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3447_344706


namespace NUMINAMATH_CALUDE_seating_theorem_l3447_344701

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange 10 people in a row with seating restrictions -/
def seating_arrangements : ℕ :=
  factorial 10 - 2 * (factorial 8 * factorial 3) + factorial 8 * factorial 3 * factorial 3

/-- Theorem stating the number of valid seating arrangements -/
theorem seating_theorem :
  seating_arrangements = 4596480 :=
sorry

end NUMINAMATH_CALUDE_seating_theorem_l3447_344701


namespace NUMINAMATH_CALUDE_paint_calculation_l3447_344740

/-- The amount of white paint needed in ounces -/
def white_paint : ℕ := 20

/-- The amount of green paint needed in ounces -/
def green_paint : ℕ := 15

/-- The amount of brown paint needed in ounces -/
def brown_paint : ℕ := 34

/-- The total amount of paint needed in ounces -/
def total_paint : ℕ := white_paint + green_paint + brown_paint

theorem paint_calculation : total_paint = 69 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3447_344740


namespace NUMINAMATH_CALUDE_boat_distance_along_stream_l3447_344798

/-- The distance traveled by a boat along a stream in one hour -/
def distance_along_stream (boat_speed : ℝ) (against_stream_distance : ℝ) : ℝ :=
  boat_speed + (boat_speed - against_stream_distance)

/-- Theorem: The distance traveled by the boat along the stream in one hour is 8 km -/
theorem boat_distance_along_stream :
  distance_along_stream 5 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_distance_along_stream_l3447_344798


namespace NUMINAMATH_CALUDE_polynomial_properties_l3447_344709

/-- Definition of the polynomial -/
def p (x y : ℝ) : ℝ := -5*x^2 - x*y^4 + 2^6*x*y + 3

/-- The number of terms in the polynomial -/
def num_terms : ℕ := 4

/-- The degree of the polynomial -/
def degree : ℕ := 5

/-- The coefficient of the highest degree term -/
def highest_coeff : ℝ := -1

/-- Theorem stating the properties of the polynomial -/
theorem polynomial_properties :
  (num_terms = 4) ∧ 
  (degree = 5) ∧ 
  (highest_coeff = -1) := by sorry

end NUMINAMATH_CALUDE_polynomial_properties_l3447_344709


namespace NUMINAMATH_CALUDE_matrix19_sum_nonzero_l3447_344779

def Matrix19 := Fin 19 → Fin 19 → Int

def isValidMatrix (A : Matrix19) : Prop :=
  ∀ i j, A i j = 1 ∨ A i j = -1

def rowProduct (A : Matrix19) (i : Fin 19) : Int :=
  (Finset.univ.prod fun j => A i j)

def colProduct (A : Matrix19) (j : Fin 19) : Int :=
  (Finset.univ.prod fun i => A i j)

theorem matrix19_sum_nonzero (A : Matrix19) (h : isValidMatrix A) :
  (Finset.univ.sum fun i => rowProduct A i) + (Finset.univ.sum fun j => colProduct A j) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix19_sum_nonzero_l3447_344779


namespace NUMINAMATH_CALUDE_valid_partition_exists_l3447_344753

/-- Represents a person with their country and position on the circle. -/
structure Person where
  country : Fin 50
  position : Fin 100

/-- Represents a partition of people into two groups. -/
def Partition := Fin 100 → Fin 2

/-- Checks if two people are from the same country. -/
def sameCountry (p1 p2 : Person) : Prop := p1.country = p2.country

/-- Checks if two people are consecutive on the circle. -/
def consecutive (p1 p2 : Person) : Prop :=
  (p1.position + 1) % 100 = p2.position ∨ (p2.position + 1) % 100 = p1.position

/-- The main theorem stating the existence of a valid partition. -/
theorem valid_partition_exists :
  ∃ (people : Fin 100 → Person) (partition : Partition),
    (∀ i : Fin 100, ∃! j : Fin 100, i ≠ j ∧ sameCountry (people i) (people j)) ∧
    (∀ i j : Fin 100, sameCountry (people i) (people j) → partition i ≠ partition j) ∧
    (∀ i j k : Fin 100, consecutive (people i) (people j) ∧ consecutive (people j) (people k) →
      ¬(partition i = partition j ∧ partition j = partition k)) :=
  sorry

end NUMINAMATH_CALUDE_valid_partition_exists_l3447_344753


namespace NUMINAMATH_CALUDE_population_increase_theorem_l3447_344765

/-- Calculates the average percent increase of population per year given initial and final populations over a specified number of years. -/
def avgPercentIncrease (initialPop finalPop : ℕ) (years : ℕ) : ℚ :=
  ((finalPop - initialPop) : ℚ) / (initialPop * years) * 100

/-- Theorem stating that the average percent increase of population per year is 5% given the specified conditions. -/
theorem population_increase_theorem :
  avgPercentIncrease 175000 262500 10 = 5 := by
  sorry

#eval avgPercentIncrease 175000 262500 10

end NUMINAMATH_CALUDE_population_increase_theorem_l3447_344765


namespace NUMINAMATH_CALUDE_bumper_car_cost_correct_l3447_344775

/-- The number of tickets required for one bumper car ride -/
def bumper_car_cost : ℕ := 5

/-- The number of times Paula rides the bumper cars -/
def bumper_car_rides : ℕ := 4

/-- The cost of riding go-karts once -/
def go_kart_cost : ℕ := 4

/-- The total number of tickets Paula needs -/
def total_tickets : ℕ := 24

/-- Theorem stating that the bumper car cost satisfies the given conditions -/
theorem bumper_car_cost_correct :
  bumper_car_cost * bumper_car_rides + go_kart_cost = total_tickets :=
by sorry

end NUMINAMATH_CALUDE_bumper_car_cost_correct_l3447_344775
