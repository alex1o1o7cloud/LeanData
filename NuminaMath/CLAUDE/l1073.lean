import Mathlib

namespace correct_calculation_l1073_107357

theorem correct_calculation : ∃ (x : ℝ), x * 5 = 40 ∧ x * 2 = 16 := by
  sorry

end correct_calculation_l1073_107357


namespace determinant_of_special_matrix_l1073_107335

theorem determinant_of_special_matrix (y : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := 
    ![![2*y + 1, 2*y, 2*y],
      ![2*y, 2*y + 1, 2*y],
      ![2*y, 2*y, 2*y + 1]]
  Matrix.det A = 6*y + 1 := by
sorry

end determinant_of_special_matrix_l1073_107335


namespace circle_equation_l1073_107387

/-- Given a circle with center (2, -3) and radius 4, its equation is (x-2)^2 + (y+3)^2 = 16 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (2, -3)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 := by sorry

end circle_equation_l1073_107387


namespace school_survey_probability_l1073_107380

theorem school_survey_probability (total_students : ℕ) (selected_students : ℕ) 
  (eliminated_students : ℕ) (h1 : total_students = 883) (h2 : selected_students = 80) 
  (h3 : eliminated_students = 3) :
  (selected_students : ℚ) / total_students = 80 / 883 := by
  sorry

end school_survey_probability_l1073_107380


namespace min_value_sum_reciprocals_l1073_107364

theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = 1 → 1/a + 1/b ≤ 1/x + 1/y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = 1 ∧ 1/x + 1/y = 3 + 2*Real.sqrt 2) :=
sorry

end min_value_sum_reciprocals_l1073_107364


namespace gcf_of_75_and_105_l1073_107369

theorem gcf_of_75_and_105 : Nat.gcd 75 105 = 15 := by
  sorry

end gcf_of_75_and_105_l1073_107369


namespace trigonometric_problem_l1073_107300

theorem trigonometric_problem (α : Real) 
  (h1 : α > π / 2 ∧ α < π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = 3 * Real.sqrt 5 / 5) : 
  Real.sin α = 4 / 5 ∧ 
  Real.cos (2 * α + π / 3) = (24 * Real.sqrt 3 - 7) / 50 := by
  sorry

end trigonometric_problem_l1073_107300


namespace min_value_xy_l1073_107382

theorem min_value_xy (x y : ℝ) (h1 : x * y + 9 = 6 * x + 2 * y) (h2 : x > 2) :
  ∃ (min_xy : ℝ), min_xy = 27 ∧ ∀ (x' y' : ℝ), x' * y' + 9 = 6 * x' + 2 * y' → x' > 2 → x' * y' ≥ min_xy := by
  sorry

end min_value_xy_l1073_107382


namespace least_candies_l1073_107384

theorem least_candies (c : ℕ) : 
  c < 150 ∧ 
  c % 5 = 4 ∧ 
  c % 6 = 3 ∧ 
  c % 8 = 5 ∧
  (∀ k : ℕ, k < c → ¬(k < 150 ∧ k % 5 = 4 ∧ k % 6 = 3 ∧ k % 8 = 5)) →
  c = 69 := by
sorry

end least_candies_l1073_107384


namespace rhombus_area_l1073_107315

/-- The area of a rhombus with side length √145 and diagonals differing by 10 units is 208 square units. -/
theorem rhombus_area (side_length : ℝ) (diagonal_difference : ℝ) (area : ℝ) : 
  side_length = Real.sqrt 145 →
  diagonal_difference = 10 →
  area = 208 :=
by sorry

end rhombus_area_l1073_107315


namespace trigonometric_equalities_l1073_107331

theorem trigonometric_equalities :
  (6 * (Real.tan (30 * π / 180))^2 - Real.sqrt 3 * Real.sin (60 * π / 180) - 2 * Real.sin (45 * π / 180) = 1/2 - Real.sqrt 2) ∧
  (Real.sqrt 2 / 2 * Real.cos (45 * π / 180) - (Real.tan (40 * π / 180) + 1)^0 + Real.sqrt (1/4) + Real.sin (30 * π / 180) = 1/2) := by
  sorry

end trigonometric_equalities_l1073_107331


namespace square_difference_divided_by_nine_l1073_107348

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by
  sorry

end square_difference_divided_by_nine_l1073_107348


namespace pythagorean_triple_with_8_and_17_l1073_107336

/-- A Pythagorean triple is a set of three positive integers (a, b, c) such that a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Given a set of Pythagorean triples containing 8 and 17, the third number is 15 -/
theorem pythagorean_triple_with_8_and_17 :
  ∃ (x : ℕ), (is_pythagorean_triple 8 15 17 ∨ is_pythagorean_triple 8 17 15 ∨ is_pythagorean_triple 15 8 17) ∧
  ¬∃ (y : ℕ), y ≠ 15 ∧ (is_pythagorean_triple 8 y 17 ∨ is_pythagorean_triple 8 17 y ∨ is_pythagorean_triple y 8 17) :=
sorry

end pythagorean_triple_with_8_and_17_l1073_107336


namespace max_value_of_f_l1073_107383

/-- The quadratic function f(z) = -9z^2 + 27z + 3 -/
def f (z : ℝ) : ℝ := -9 * z^2 + 27 * z + 3

theorem max_value_of_f :
  ∃ (max : ℝ), max = 117/4 ∧ ∀ (z : ℝ), f z ≤ max :=
sorry

end max_value_of_f_l1073_107383


namespace b_to_c_interest_rate_b_to_c_interest_rate_is_12_percent_l1073_107353

/-- The interest rate at which B lent money to C, given the following conditions:
  * A lends Rs. 3500 to B at 10% per annum
  * B lends the same sum to C
  * B's gain over 3 years is Rs. 210
-/
theorem b_to_c_interest_rate : ℝ :=
  let principal : ℝ := 3500
  let a_to_b_rate : ℝ := 0.1
  let time : ℝ := 3
  let b_gain : ℝ := 210
  let a_to_b_interest : ℝ := principal * a_to_b_rate * time
  let total_interest_from_c : ℝ := a_to_b_interest + b_gain
  total_interest_from_c / (principal * time)

/-- Proof that the interest rate at which B lent money to C is 12% per annum -/
theorem b_to_c_interest_rate_is_12_percent : b_to_c_interest_rate = 0.12 := by
  sorry

end b_to_c_interest_rate_b_to_c_interest_rate_is_12_percent_l1073_107353


namespace relationship_abc_l1073_107397

theorem relationship_abc (a b c : ℝ) : 
  a = 1 + Real.sqrt 7 → 
  b = Real.sqrt 3 + Real.sqrt 5 → 
  c = 4 → 
  c > b ∧ b > a := by
sorry

end relationship_abc_l1073_107397


namespace slope_determines_y_coordinate_l1073_107334

/-- Given two points R and S in a coordinate plane, if the slope of the line through R and S
    is -5/4, then the y-coordinate of S is -2. -/
theorem slope_determines_y_coordinate (x_R y_R x_S : ℚ) :
  let R : ℚ × ℚ := (x_R, y_R)
  let S : ℚ × ℚ := (x_S, y_S)
  x_R = -3 →
  y_R = 8 →
  x_S = 5 →
  (y_S - y_R) / (x_S - x_R) = -5/4 →
  y_S = -2 := by
sorry

end slope_determines_y_coordinate_l1073_107334


namespace f_at_neg_point_two_eq_approx_l1073_107350

/-- Horner's algorithm for polynomial evaluation -/
def horner (coeffs : List Float) (x : Float) : Float :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial function f(x) -/
def f (x : Float) : Float :=
  horner [1, 1, 0.5, 0.16667, 0.04167, 0.00833] x

/-- Theorem stating that f(-0.2) equals 0.81873 (approximately) -/
theorem f_at_neg_point_two_eq_approx :
  (f (-0.2) - 0.81873).abs < 1e-5 := by
  sorry

#eval f (-0.2)

end f_at_neg_point_two_eq_approx_l1073_107350


namespace wrapping_paper_division_l1073_107341

theorem wrapping_paper_division (total_used : ℚ) (num_presents : ℕ) (paper_per_present : ℚ) :
  total_used = 5 / 8 →
  num_presents = 5 →
  paper_per_present * num_presents = total_used →
  paper_per_present = 1 / 8 := by
  sorry

end wrapping_paper_division_l1073_107341


namespace cone_cylinder_theorem_l1073_107372

/-- Given a cone with base radius 2 and slant height 4, and a cylinder with height √3 inside the cone -/
def cone_cylinder_problem :=
  ∃ (cone_base_radius cone_slant_height cylinder_height : ℝ),
    cone_base_radius = 2 ∧
    cone_slant_height = 4 ∧
    cylinder_height = Real.sqrt 3

theorem cone_cylinder_theorem (h : cone_cylinder_problem) :
  ∃ (max_cylinder_area sphere_surface_area sphere_volume : ℝ),
    max_cylinder_area = 2 * (1 + Real.sqrt 3) * Real.pi ∧
    sphere_surface_area = 7 * Real.pi ∧
    sphere_volume = (7 * Real.sqrt 7 * Real.pi) / 6 :=
by
  sorry

end cone_cylinder_theorem_l1073_107372


namespace max_stores_visited_l1073_107318

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 7) (h2 : total_visits = 21) 
  (h3 : total_shoppers = 11) (h4 : double_visitors = 7) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : 2 * double_visitors ≤ total_visits) : 
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits := by
  sorry

end max_stores_visited_l1073_107318


namespace rearrangement_does_not_increase_length_l1073_107378

/-- A segment on a line --/
structure Segment where
  left : ℝ
  right : ℝ
  h : left ≤ right

/-- A finite set of segments on a line --/
def SegmentSystem := Finset Segment

/-- The total length of the union of segments in a system --/
def totalLength (S : SegmentSystem) : ℝ := sorry

/-- The distance between midpoints of two segments --/
def midpointDistance (s₁ s₂ : Segment) : ℝ := sorry

/-- A rearrangement of segments that minimizes midpoint distances --/
def rearrange (S : SegmentSystem) : SegmentSystem := sorry

/-- The theorem stating that rearrangement does not increase total length --/
theorem rearrangement_does_not_increase_length (S : SegmentSystem) :
  totalLength (rearrange S) ≤ totalLength S := by sorry

end rearrangement_does_not_increase_length_l1073_107378


namespace victor_games_ratio_l1073_107362

theorem victor_games_ratio : 
  let victor_wins : ℕ := 36
  let friend_wins : ℕ := 20
  let gcd := Nat.gcd victor_wins friend_wins
  (victor_wins / gcd) = 9 ∧ (friend_wins / gcd) = 5 := by
sorry

end victor_games_ratio_l1073_107362


namespace wickets_in_last_match_is_three_l1073_107366

/-- Represents a cricket bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  runsInLastMatch : ℕ
  averageDecrease : ℝ
  approximateWicketsBefore : ℕ

/-- Calculates the number of wickets taken in the last match -/
def wicketsInLastMatch (stats : BowlerStats) : ℕ :=
  -- The actual calculation would go here
  3 -- We're stating the result directly as per the problem

/-- Theorem stating that given the specific conditions, the number of wickets in the last match is 3 -/
theorem wickets_in_last_match_is_three (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.runsInLastMatch = 26)
  (h3 : stats.averageDecrease = 0.4)
  (h4 : stats.approximateWicketsBefore = 25) :
  wicketsInLastMatch stats = 3 := by
  sorry

#eval wicketsInLastMatch { 
  initialAverage := 12.4, 
  runsInLastMatch := 26, 
  averageDecrease := 0.4, 
  approximateWicketsBefore := 25 
}

end wickets_in_last_match_is_three_l1073_107366


namespace negation_of_universal_proposition_l1073_107368

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) := by
  sorry

end negation_of_universal_proposition_l1073_107368


namespace equation_linear_iff_a_eq_plus_minus_two_l1073_107345

-- Define the equation
def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3*a) * x + (a + 1) * y + 3*a

-- Define what it means for the equation to be linear in two variables
def is_linear_two_var (a : ℝ) : Prop :=
  (a^2 - 4 = 0) ∧ (2 - 3*a ≠ 0 ∨ a + 1 ≠ 0)

-- State the theorem
theorem equation_linear_iff_a_eq_plus_minus_two :
  ∀ a : ℝ, is_linear_two_var a ↔ (a = 2 ∨ a = -2) :=
by sorry

end equation_linear_iff_a_eq_plus_minus_two_l1073_107345


namespace division_simplification_l1073_107344

theorem division_simplification (m : ℝ) (h : m ≠ 0) :
  (4 * m^2 - 2 * m) / (2 * m) = 2 * m - 1 := by
  sorry

end division_simplification_l1073_107344


namespace first_division_divisor_l1073_107337

theorem first_division_divisor
  (x : ℕ+) -- x is a positive integer
  (y : ℕ) -- y is a natural number (quotient)
  (d : ℕ) -- d is the divisor we're looking for
  (h1 : ∃ q : ℕ, x = d * y + 3) -- x divided by d gives quotient y and remainder 3
  (h2 : ∃ q : ℕ, 2 * x = 7 * (3 * y) + 1) -- 2x divided by 7 gives quotient 3y and remainder 1
  (h3 : 11 * y - x = 2) -- Given equation
  : d = 10 := by
  sorry

end first_division_divisor_l1073_107337


namespace quadratic_equation_with_integer_roots_l1073_107376

theorem quadratic_equation_with_integer_roots (m : ℤ) 
  (h1 : ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
    a^2 + m*a - m + 1 = 0 ∧ b^2 + m*b - m + 1 = 0) : 
  m = -5 := by
sorry

end quadratic_equation_with_integer_roots_l1073_107376


namespace sector_max_area_l1073_107358

/-- Given a sector with perimeter 20 cm, its area is maximized when the central angle is 2 radians, 
    and the maximum area is 25 cm². -/
theorem sector_max_area (r : ℝ) (α : ℝ) (l : ℝ) (S : ℝ) :
  0 < r → r < 10 →
  l + 2 * r = 20 →
  l = r * α →
  S = 1/2 * r * l →
  (∀ r' α' l' S', 
    0 < r' → r' < 10 →
    l' + 2 * r' = 20 →
    l' = r' * α' →
    S' = 1/2 * r' * l' →
    S' ≤ S) →
  α = 2 ∧ S = 25 := by
sorry


end sector_max_area_l1073_107358


namespace larger_cube_volume_l1073_107347

/-- Proves that a cube containing 64 smaller cubes of 1 cubic inch each, with a surface area
    difference of 288 square inches between the sum of the smaller cubes' surface areas and
    the larger cube's surface area, has a volume of 64 cubic inches. -/
theorem larger_cube_volume (s : ℝ) (h1 : s > 0) :
  (s^3 : ℝ) = 64 ∧
  64 * (6 : ℝ) - 6 * s^2 = 288 →
  (s^3 : ℝ) = 64 := by sorry

end larger_cube_volume_l1073_107347


namespace functional_equation_solution_l1073_107394

-- Define a monotonic function f from real numbers to real numbers
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y ∨ (∀ z : ℝ, f z = f x)

-- State the theorem
theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h_monotonic : MonotonicFunction f)
  (h_equation : ∀ x y : ℝ, f x * f y = f (x + y)) :
  ∃! a : ℝ, a > 0 ∧ a ≠ 1 ∧ (∀ x : ℝ, f x = a^x) :=
sorry

end functional_equation_solution_l1073_107394


namespace chord_equation_l1073_107303

/-- The equation of a line containing a chord of the ellipse x^2/2 + y^2 = 1,
    passing through and bisected by the point (1/2, 1/2) -/
theorem chord_equation (x y : ℝ) : 
  (∃ (x1 y1 x2 y2 : ℝ),
    -- Ellipse equation
    x1^2 / 2 + y1^2 = 1 ∧ 
    x2^2 / 2 + y2^2 = 1 ∧
    -- Point P is on the ellipse
    (1/2)^2 / 2 + (1/2)^2 = 1 ∧
    -- P is the midpoint of the chord
    (x1 + x2) / 2 = 1/2 ∧
    (y1 + y2) / 2 = 1/2 ∧
    -- The line passes through P
    y - 1/2 = (y - 1/2) / (x - 1/2) * (x - 1/2)) →
  2*x + 4*y - 3 = 0 :=
sorry

end chord_equation_l1073_107303


namespace range_of_f_sum_of_endpoints_l1073_107307

open Set Real

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 9 * x^2)

theorem range_of_f :
  range f = Ioo 0 3 ∪ {3} :=
sorry

theorem sum_of_endpoints :
  ∃ c d : ℝ, range f = Ioc c d ∧ c + d = 3 :=
sorry

end range_of_f_sum_of_endpoints_l1073_107307


namespace sequence_existence_l1073_107349

theorem sequence_existence : ∃ (a : ℕ → ℕ+), 
  (∀ k : ℕ+, ∃ n : ℕ, a n = k) ∧ 
  (∀ k : ℕ+, (Finset.range k).sum (λ i => (a i.succ).val) % k = 0) :=
sorry

end sequence_existence_l1073_107349


namespace photos_per_album_l1073_107324

theorem photos_per_album (total_photos : ℕ) (num_albums : ℕ) (h1 : total_photos = 180) (h2 : num_albums = 9) :
  total_photos / num_albums = 20 := by
sorry

end photos_per_album_l1073_107324


namespace different_gender_choices_eq_450_l1073_107388

/-- The number of boys in the club -/
def num_boys : ℕ := 15

/-- The number of girls in the club -/
def num_girls : ℕ := 15

/-- The total number of members in the club -/
def total_members : ℕ := num_boys + num_girls

/-- The number of ways to choose a president and a vice-president of different genders -/
def different_gender_choices : ℕ := num_boys * num_girls * 2

theorem different_gender_choices_eq_450 : different_gender_choices = 450 := by
  sorry

end different_gender_choices_eq_450_l1073_107388


namespace original_rectangle_area_l1073_107310

/-- Given a rectangle whose dimensions are doubled to form a new rectangle with an area of 32 square meters, 
    the area of the original rectangle is 8 square meters. -/
theorem original_rectangle_area (original_length original_width : ℝ) 
  (new_length new_width : ℝ) (new_area : ℝ) :
  new_length = 2 * original_length →
  new_width = 2 * original_width →
  new_area = new_length * new_width →
  new_area = 32 →
  original_length * original_width = 8 :=
by sorry

end original_rectangle_area_l1073_107310


namespace part_one_part_two_l1073_107316

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}
def B : Set ℝ := {-1, 1}

-- Theorem for part I
theorem part_one (a b : ℝ) : B ⊆ A a b → a = -1 := by sorry

-- Theorem for part II
theorem part_two (a b : ℝ) : (A a b ∩ B).Nonempty → a^2 - b^2 + 2*a = -1 := by sorry

end part_one_part_two_l1073_107316


namespace tyler_scissors_purchase_l1073_107308

theorem tyler_scissors_purchase
  (initial_amount : ℕ)
  (scissors_cost : ℕ)
  (eraser_count : ℕ)
  (eraser_cost : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : scissors_cost = 5)
  (h3 : eraser_count = 10)
  (h4 : eraser_cost = 4)
  (h5 : remaining_amount = 20) :
  ∃ (scissors_count : ℕ), 
    scissors_count * scissors_cost + eraser_count * eraser_cost = initial_amount - remaining_amount ∧
    scissors_count = 8 :=
by
  sorry

end tyler_scissors_purchase_l1073_107308


namespace f_properties_l1073_107338

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x^2 + 1) + x - 1) / (Real.sqrt (x^2 + 1) + x + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) ≠ -f x) ∧
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ y : ℝ, -1 < y ∧ y < 1 ↔ ∃ x : ℝ, f x = y) :=
by sorry

end f_properties_l1073_107338


namespace exists_skew_line_l1073_107330

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line intersecting a plane
variable (intersects : Line → Plane → Prop)

-- Define the property of a line being in a plane
variable (inPlane : Line → Plane → Prop)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem exists_skew_line 
  (l : Line) (α : Plane) 
  (h : intersects l α) : 
  ∃ m : Line, inPlane m α ∧ skew l m :=
sorry

end exists_skew_line_l1073_107330


namespace stream_speed_calculation_l1073_107329

/-- The speed of the stream in mph -/
def stream_speed : ℝ := 3.5

/-- The speed of the boat in still water in mph -/
def boat_speed : ℝ := 15

/-- The distance traveled in miles -/
def distance : ℝ := 60

/-- The time difference between upstream and downstream trips in hours -/
def time_difference : ℝ := 2

theorem stream_speed_calculation :
  (distance / (boat_speed - stream_speed)) - (distance / (boat_speed + stream_speed)) = time_difference :=
by sorry

end stream_speed_calculation_l1073_107329


namespace chris_average_speed_l1073_107313

/-- Calculates the average speed given initial and final odometer readings and total time. -/
def average_speed (initial_reading : ℕ) (final_reading : ℕ) (total_time : ℕ) : ℚ :=
  (final_reading - initial_reading : ℚ) / total_time

/-- Proves that Chris's average speed is approximately 36.67 miles per hour. -/
theorem chris_average_speed :
  let initial_reading := 2332
  let final_reading := 2772
  let total_time := 12
  abs (average_speed initial_reading final_reading total_time - 36.67) < 0.01 := by
  sorry

#eval average_speed 2332 2772 12

end chris_average_speed_l1073_107313


namespace number_problem_l1073_107351

theorem number_problem (x : ℝ) : 0.75 * x = 0.45 * 1500 + 495 → x = 1560 := by
  sorry

end number_problem_l1073_107351


namespace no_consecutive_even_fibonacci_l1073_107333

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem no_consecutive_even_fibonacci :
  ∀ n : ℕ, ¬(Even (fibonacci n) ∧ Even (fibonacci (n + 1))) := by
  sorry

end no_consecutive_even_fibonacci_l1073_107333


namespace vector_simplification_l1073_107326

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_simplification (A B C D : V) :
  (B - A) + (C - B) - (D - A) = D - C := by sorry

end vector_simplification_l1073_107326


namespace sample_customers_l1073_107321

theorem sample_customers (samples_per_box : ℕ) (boxes_opened : ℕ) (samples_left : ℕ) : 
  samples_per_box = 20 →
  boxes_opened = 12 →
  samples_left = 5 →
  (samples_per_box * boxes_opened - samples_left) = 235 := by
  sorry

end sample_customers_l1073_107321


namespace investment_amount_l1073_107354

/-- Represents the investment scenario with simple and compound interest --/
structure Investment where
  P : ℝ  -- Principal amount invested
  y : ℝ  -- Interest rate (in percentage)
  simpleInterest : ℝ  -- Simple interest earned
  compoundInterest : ℝ  -- Compound interest earned

/-- The investment satisfies the given conditions --/
def validInvestment (inv : Investment) : Prop :=
  inv.simpleInterest = inv.P * inv.y * 2 / 100 ∧
  inv.compoundInterest = inv.P * ((1 + inv.y / 100)^2 - 1) ∧
  inv.simpleInterest = 500 ∧
  inv.compoundInterest = 512.50

/-- The theorem stating that the investment amount is 5000 --/
theorem investment_amount (inv : Investment) 
  (h : validInvestment inv) : inv.P = 5000 := by
  sorry

end investment_amount_l1073_107354


namespace conservation_center_count_l1073_107340

/-- The number of turtles in a conservation center -/
def total_turtles (green : ℕ) (hawksbill : ℕ) : ℕ := green + hawksbill

/-- The number of hawksbill turtles is twice more than the number of green turtles -/
def hawksbill_count (green : ℕ) : ℕ := green + 2 * green

theorem conservation_center_count :
  let green := 800
  let hawksbill := hawksbill_count green
  total_turtles green hawksbill = 3200 := by sorry

end conservation_center_count_l1073_107340


namespace circumradius_of_specific_isosceles_triangle_l1073_107320

/-- An isosceles triangle with base 6 and side length 5 -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  is_isosceles : base = 6 ∧ side = 5

/-- The radius of the circumcircle of a triangle -/
def circumradius (t : IsoscelesTriangle) : ℝ := sorry

/-- Theorem: The radius of the circumcircle of an isosceles triangle with base 6 and side length 5 is 25/8 -/
theorem circumradius_of_specific_isosceles_triangle (t : IsoscelesTriangle) : 
  circumradius t = 25/8 := by sorry

end circumradius_of_specific_isosceles_triangle_l1073_107320


namespace two_zeros_iff_a_in_set_l1073_107371

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2*x - |x^2 - a*x + 1|

/-- The set of a values that satisfy the condition -/
def A : Set ℝ := {a | a < 0 ∨ (0 < a ∧ a < 1) ∨ 1 < a}

theorem two_zeros_iff_a_in_set (a : ℝ) : 
  (∃! (x y : ℝ), x ≠ y ∧ f a x = 0 ∧ f a y = 0) ↔ a ∈ A := by sorry

end two_zeros_iff_a_in_set_l1073_107371


namespace intersection_of_A_and_B_l1073_107343

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | 4 * p.1 + p.2 = 6}
def B : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(2, -2)} := by
  sorry

end intersection_of_A_and_B_l1073_107343


namespace bus_ride_duration_l1073_107367

/-- Calculates the bus ride time given the total trip time and other component times -/
def bus_ride_time (total_trip_time walk_time train_ride_time : ℕ) : ℕ :=
  let waiting_time := 2 * walk_time
  let total_trip_minutes := total_trip_time * 60
  let train_ride_minutes := train_ride_time * 60
  total_trip_minutes - (walk_time + waiting_time + train_ride_minutes)

/-- Theorem stating that given the specific trip components, the bus ride time is 75 minutes -/
theorem bus_ride_duration :
  bus_ride_time 8 15 6 = 75 := by
  sorry

#eval bus_ride_time 8 15 6

end bus_ride_duration_l1073_107367


namespace larger_jar_initial_fill_fraction_l1073_107375

/-- Proves that under the given conditions, the larger jar was initially 1/3 full -/
theorem larger_jar_initial_fill_fraction 
  (small_capacity large_capacity : ℝ) 
  (water_amount : ℝ) 
  (h1 : small_capacity > 0)
  (h2 : large_capacity > 0)
  (h3 : water_amount > 0)
  (h4 : water_amount = 1/3 * small_capacity)
  (h5 : water_amount < large_capacity)
  (h6 : water_amount + water_amount = 2/3 * large_capacity) :
  water_amount = 1/3 * large_capacity := by
sorry

end larger_jar_initial_fill_fraction_l1073_107375


namespace other_root_of_quadratic_l1073_107305

theorem other_root_of_quadratic (m : ℝ) : 
  (2 : ℝ)^2 + m * 2 - 6 = 0 → (-3 : ℝ)^2 + m * (-3) - 6 = 0 := by
  sorry

end other_root_of_quadratic_l1073_107305


namespace exists_strictly_increasing_set_function_l1073_107352

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define the power set of positive integers
def PowerSetOfPositiveIntegers : Set (Set ℕ) :=
  {X : Set ℕ | X ⊆ PositiveIntegers}

-- State the theorem
theorem exists_strictly_increasing_set_function :
  ∃ (f : ℝ → Set ℕ),
    (∀ x, f x ∈ PowerSetOfPositiveIntegers) ∧
    (∀ a b, a < b → f a ⊂ f b ∧ f a ≠ f b) :=
sorry

end exists_strictly_increasing_set_function_l1073_107352


namespace factor_implies_a_value_l1073_107359

theorem factor_implies_a_value (a b : ℤ) (x : ℝ) :
  (∀ x, (x^2 - x - 1) ∣ (a*x^17 + b*x^16 + 1)) →
  a = 987 := by
sorry

end factor_implies_a_value_l1073_107359


namespace f_monotonicity_and_bound_l1073_107398

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a / x

theorem f_monotonicity_and_bound (a : ℝ) :
  (a > 0 → ∀ x y, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  ((∀ x, x > 1 → f a x < x^2) → a ≥ -1) :=
sorry

end f_monotonicity_and_bound_l1073_107398


namespace square_sum_zero_implies_both_zero_l1073_107390

theorem square_sum_zero_implies_both_zero (a b : ℝ) :
  a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l1073_107390


namespace quadratic_inequality_range_l1073_107314

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 > 0) → -1 < a ∧ a < 3 := by
  sorry

end quadratic_inequality_range_l1073_107314


namespace composition_ratio_l1073_107381

def f (x : ℝ) : ℝ := 3 * x + 1

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio : f (g (f 3)) / g (f (g 3)) = 112 / 109 := by
  sorry

end composition_ratio_l1073_107381


namespace six_circles_l1073_107346

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (a : Point) (b : Point) (c : Point)

/-- Two identical equilateral triangles sharing one vertex -/
structure TwoTriangles :=
  (t1 : EquilateralTriangle)
  (t2 : EquilateralTriangle)
  (shared_vertex : Point)
  (h1 : t1.c = shared_vertex)
  (h2 : t2.a = shared_vertex)

/-- A function that returns all circles satisfying the conditions -/
def circles_through_vertices (triangles : TwoTriangles) : Finset Circle := sorry

/-- The main theorem -/
theorem six_circles (triangles : TwoTriangles) :
  (circles_through_vertices triangles).card = 6 := by sorry

end six_circles_l1073_107346


namespace always_possible_to_reach_final_state_l1073_107386

/-- Represents the two types of operations that can be performed. -/
inductive Operation
  | RedToBlue
  | BlueToRed

/-- Represents the state of the slips for a single MOPper. -/
structure MOPperState where
  number : Nat
  redSlip : Nat
  blueSlip : Nat

/-- Represents the state of all MOPpers' slips. -/
def SystemState := List MOPperState

/-- Initializes the system state based on the given A and B values. -/
def initializeState (A B : Nat) : SystemState :=
  sorry

/-- Performs a single operation on the system state. -/
def performOperation (state : SystemState) (op : Operation) : SystemState :=
  sorry

/-- Checks if the system state is in the desired final configuration. -/
def isFinalState (state : SystemState) : Bool :=
  sorry

/-- The main theorem to be proved. -/
theorem always_possible_to_reach_final_state :
  ∀ (A B : Nat), A ≤ 2010 → B ≤ 2010 →
  ∃ (ops : List Operation),
    isFinalState (ops.foldl performOperation (initializeState A B)) = true :=
  sorry

end always_possible_to_reach_final_state_l1073_107386


namespace max_area_rectangle_l1073_107317

/-- Represents a rectangular enclosure --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of the rectangle is 400 feet --/
def isValidPerimeter (r : Rectangle) : Prop :=
  2 * r.length + 2 * r.width = 400

/-- The length is at least 90 feet --/
def hasValidLength (r : Rectangle) : Prop :=
  r.length ≥ 90

/-- The width is at least 50 feet --/
def hasValidWidth (r : Rectangle) : Prop :=
  r.width ≥ 50

/-- The area of the rectangle --/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 400 feet, 
    length ≥ 90 feet, and width ≥ 50 feet is 10,000 square feet --/
theorem max_area_rectangle :
  ∃ (r : Rectangle), isValidPerimeter r ∧ hasValidLength r ∧ hasValidWidth r ∧
    (∀ (s : Rectangle), isValidPerimeter s ∧ hasValidLength s ∧ hasValidWidth s →
      area s ≤ area r) ∧
    area r = 10000 := by
  sorry

end max_area_rectangle_l1073_107317


namespace tangent_line_and_monotonicity_l1073_107339

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / x + a) * Real.log (1 + x)

theorem tangent_line_and_monotonicity (a : ℝ) :
  (∀ x > 0, ∀ y > 0, x ≠ y → (f (-1) x - f (-1) y) / (x - y) = Real.log 2 * x + f (-1) x - Real.log 2) ∧
  (∀ x > 0, Monotone (f a) ↔ a ≥ (1 : ℝ) / 2) := by
  sorry

end tangent_line_and_monotonicity_l1073_107339


namespace Q_formula_l1073_107304

def T (n : ℕ) : ℕ := (n * (n + 1)) / 2

def Q (n : ℕ) : ℚ :=
  if n < 2 then 0
  else Finset.prod (Finset.range (n - 1)) (fun k => (T (k + 2) : ℚ) / ((T (k + 3) : ℚ) - 1))

theorem Q_formula (n : ℕ) (h : n ≥ 2) : Q n = 2 / (n + 3) := by
  sorry

end Q_formula_l1073_107304


namespace sqrt_two_four_three_two_five_two_l1073_107323

theorem sqrt_two_four_three_two_five_two : Real.sqrt (2^4 * 3^2 * 5^2) = 60 := by
  sorry

end sqrt_two_four_three_two_five_two_l1073_107323


namespace complete_square_d_value_l1073_107332

/-- Given a quadratic equation x^2 - 6x + 5 = 0, when transformed
    into the form (x + c)^2 = d, the value of d is 4. -/
theorem complete_square_d_value :
  ∃ c d : ℝ, (∀ x : ℝ, x^2 - 6*x + 5 = 0 ↔ (x + c)^2 = d) ∧ d = 4 := by
  sorry

end complete_square_d_value_l1073_107332


namespace train_length_l1073_107385

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 120 → time = 15 → ∃ (length : ℝ), abs (length - 500) < 1 :=
by
  sorry

#check train_length

end train_length_l1073_107385


namespace angle_with_special_supplement_complement_l1073_107365

theorem angle_with_special_supplement_complement (x : ℝ) : 
  (180 - x = 2 * (90 - x) + 10) → x = 10 := by
  sorry

end angle_with_special_supplement_complement_l1073_107365


namespace composite_triangle_perimeter_l1073_107395

/-- A triangle composed of four smaller equilateral triangles -/
structure CompositeTriangle where
  /-- The side length of the smaller equilateral triangles -/
  small_side : ℝ
  /-- The perimeter of each smaller equilateral triangle is 9 -/
  small_perimeter : small_side * 3 = 9

/-- The perimeter of the large equilateral triangle -/
def large_perimeter (t : CompositeTriangle) : ℝ :=
  3 * (2 * t.small_side)

/-- Theorem: The perimeter of the large equilateral triangle is 18 -/
theorem composite_triangle_perimeter (t : CompositeTriangle) :
  large_perimeter t = 18 := by
  sorry

end composite_triangle_perimeter_l1073_107395


namespace ones_digit_multiplication_l1073_107306

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_multiplication (n : ℕ) (h : ones_digit n = 2) :
  ones_digit (n * 3) = 6 := by
  sorry

end ones_digit_multiplication_l1073_107306


namespace system_solution_l1073_107328

theorem system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℚ),
    (x₁^2 - 9*y₁^2 = 36 ∧ 3*x₁ + y₁ = 6) ∧
    (x₂^2 - 9*y₂^2 = 36 ∧ 3*x₂ + y₂ = 6) ∧
    x₁ = 12/5 ∧ y₁ = -6/5 ∧ x₂ = 3 ∧ y₂ = -3 ∧
    ∀ (x y : ℚ), (x^2 - 9*y^2 = 36 ∧ 3*x + y = 6) → ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry


end system_solution_l1073_107328


namespace range_of_a_l1073_107325

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -2 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | x < 2*a ∨ x > -a}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (a : ℝ) (x : ℝ) : Prop := x ∈ B a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  a < 0 → 
  (∀ x, ¬(p x) → ¬(q a x)) ∧ 
  (∃ x, ¬(p x) ∧ q a x) → 
  a ≤ -3 :=
sorry

end range_of_a_l1073_107325


namespace semicircle_chord_product_l1073_107361

/-- The radius of the semicircle -/
def radius : ℝ := 3

/-- The number of equal parts the semicircle is divided into -/
def num_parts : ℕ := 8

/-- The number of chords -/
def num_chords : ℕ := 14

/-- The product of the lengths of the chords in a semicircle -/
def chord_product (r : ℝ) (n : ℕ) : ℝ :=
  (2 * r ^ (n - 1)) * (2 ^ n)

theorem semicircle_chord_product :
  chord_product radius num_chords = 196608 := by
  sorry

end semicircle_chord_product_l1073_107361


namespace log_50000_sum_consecutive_integers_l1073_107327

theorem log_50000_sum_consecutive_integers : ∃ (a b : ℕ), 
  (a + 1 = b) ∧ 
  (a : ℝ) < Real.log 50000 / Real.log 10 ∧ 
  Real.log 50000 / Real.log 10 < (b : ℝ) ∧ 
  a + b = 9 := by
sorry

end log_50000_sum_consecutive_integers_l1073_107327


namespace income_comparison_l1073_107396

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mary = 0.84 * juan) : 
  (mary - tim) / tim * 100 = 40 := by
sorry

end income_comparison_l1073_107396


namespace largest_prime_divisor_of_sum_of_squares_l1073_107392

theorem largest_prime_divisor_of_sum_of_squares : 
  (∃ p : Nat, Nat.Prime p ∧ p ∣ (17^2 + 60^2) ∧ ∀ q : Nat, Nat.Prime q → q ∣ (17^2 + 60^2) → q ≤ p) ∧ 
  (37 : Nat).Prime ∧ 
  37 ∣ (17^2 + 60^2) ∧ 
  ∀ q : Nat, Nat.Prime q → q ∣ (17^2 + 60^2) → q ≤ 37 :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l1073_107392


namespace inequality_range_of_p_l1073_107377

-- Define the inequality function
def inequality (a p : ℝ) : Prop :=
  Real.sqrt a - Real.sqrt (a - 1) > Real.sqrt (a - 2) - Real.sqrt (a - p)

-- State the theorem
theorem inequality_range_of_p :
  ∀ a p : ℝ, a ≥ 3 → p > 2 → 
  (∀ x : ℝ, x ≥ 3 → inequality x p) →
  2 < p ∧ p < 2 * Real.sqrt 6 + 2 * Real.sqrt 3 - 2 * Real.sqrt 2 :=
sorry

end inequality_range_of_p_l1073_107377


namespace division_of_fractions_l1073_107363

theorem division_of_fractions : (3 : ℚ) / 7 / 5 = 3 / 35 := by sorry

end division_of_fractions_l1073_107363


namespace lisa_children_count_l1073_107342

/-- The number of children Lisa has -/
def num_children : ℕ := sorry

/-- The number of eggs Lisa cooks for her family in a year -/
def total_eggs_per_year : ℕ := 3380

/-- The number of days Lisa cooks breakfast in a year -/
def days_per_year : ℕ := 5 * 52

/-- The number of eggs Lisa cooks each day -/
def eggs_per_day (c : ℕ) : ℕ := 2 * c + 3 + 2

theorem lisa_children_count : 
  num_children = 4 ∧ 
  total_eggs_per_year = days_per_year * eggs_per_day num_children :=
sorry

end lisa_children_count_l1073_107342


namespace carlos_won_one_game_l1073_107393

/-- Represents a chess player in the tournament -/
structure Player where
  wins : ℕ
  losses : ℕ

/-- Represents the chess tournament -/
structure Tournament where
  laura : Player
  mike : Player
  carlos : Player
  total_games : ℕ

/-- The number of games Carlos won in the tournament -/
def carlos_wins (t : Tournament) : ℕ :=
  t.total_games - (t.laura.wins + t.laura.losses + t.mike.wins + t.mike.losses + t.carlos.losses)

theorem carlos_won_one_game (t : Tournament) 
  (h1 : t.laura.wins = 5)
  (h2 : t.laura.losses = 4)
  (h3 : t.mike.wins = 7)
  (h4 : t.mike.losses = 2)
  (h5 : t.carlos.losses = 5)
  (h6 : t.total_games = (t.laura.wins + t.laura.losses + t.mike.wins + t.mike.losses + t.carlos.losses + carlos_wins t) / 2) :
  carlos_wins t = 1 := by
  sorry

end carlos_won_one_game_l1073_107393


namespace sum_between_14_and_14_half_l1073_107360

theorem sum_between_14_and_14_half :
  let sum := (3 + 3/8) + (4 + 3/4) + (6 + 2/23)
  14 < sum ∧ sum < 14.5 := by
sorry

end sum_between_14_and_14_half_l1073_107360


namespace shifted_sine_function_l1073_107312

/-- Given a function f and its right-shifted version g, prove that g has the correct form -/
theorem shifted_sine_function 
  (f g : ℝ → ℝ) 
  (h₁ : ∀ x, f x = 3 * Real.sin (2 * x))
  (h₂ : ∀ x, g x = f (x - π/8)) :
  ∀ x, g x = 3 * Real.sin (2 * x - π/4) := by
  sorry


end shifted_sine_function_l1073_107312


namespace hurricane_damage_conversion_l1073_107379

/-- Calculates the equivalent amount in Canadian dollars given an amount in American dollars and the exchange rate. -/
def convert_to_canadian_dollars (american_dollars : ℚ) (exchange_rate : ℚ) : ℚ :=
  american_dollars * exchange_rate

/-- Theorem stating the correct conversion of hurricane damage from American to Canadian dollars. -/
theorem hurricane_damage_conversion :
  let damage_usd : ℚ := 45000000
  let exchange_rate : ℚ := 3/2
  convert_to_canadian_dollars damage_usd exchange_rate = 67500000 := by
  sorry

#check hurricane_damage_conversion

end hurricane_damage_conversion_l1073_107379


namespace implicit_derivative_l1073_107391

-- Define the implicit function
def implicit_function (x y : ℝ) : Prop := x^2 - y^2 = 4

-- State the theorem
theorem implicit_derivative (x y : ℝ) (h : implicit_function x y) :
  ∃ (y' : ℝ), y' = x / y :=
sorry

end implicit_derivative_l1073_107391


namespace roses_sold_l1073_107399

theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) (sold : ℕ) : 
  initial = 5 → picked = 34 → final = 36 → 
  final = initial - sold + picked → sold = 3 := by
sorry

end roses_sold_l1073_107399


namespace digit_405_is_zero_l1073_107373

/-- The decimal representation of 18/47 -/
def decimal_rep : ℚ := 18 / 47

/-- The length of the repeating sequence in the decimal representation of 18/47 -/
def period : ℕ := 93

/-- The position of the target digit within the repeating sequence -/
def target_position : ℕ := 405 % period

/-- The digit at the specified position in the repeating sequence -/
def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_405_is_zero :
  digit_at_position target_position = 0 :=
sorry

end digit_405_is_zero_l1073_107373


namespace min_value_2x_plus_y_l1073_107356

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' + 6 = x' * y' → 2 * x + y ≤ 2 * x' + y') ∧
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ + 6 = x₀ * y₀ ∧ 2 * x₀ + y₀ = 12) :=
by sorry

end min_value_2x_plus_y_l1073_107356


namespace possible_values_of_a_minus_b_l1073_107322

theorem possible_values_of_a_minus_b (a b : ℝ) (ha : |a| = 7) (hb : |b| = 5) :
  {x | ∃ (a' b' : ℝ), |a'| = 7 ∧ |b'| = 5 ∧ x = a' - b'} = {2, 12, -12, -2} := by
sorry

end possible_values_of_a_minus_b_l1073_107322


namespace square_area_l1073_107374

theorem square_area (s : ℝ) (h : (2/5 * s) * 10 = 140) : s^2 = 1225 := by
  sorry

end square_area_l1073_107374


namespace stating_comprehensive_investigation_is_census_l1073_107370

/-- Represents a comprehensive investigation. -/
structure ComprehensiveInvestigation where
  subject : String
  purpose : String

/-- Defines what a census is. -/
def Census : Type := ComprehensiveInvestigation

/-- 
Theorem stating that a comprehensive investigation on the subject of examination 
for a specific purpose is equivalent to a census.
-/
theorem comprehensive_investigation_is_census 
  (investigation : ComprehensiveInvestigation) 
  (h1 : investigation.subject = "examination") 
  (h2 : investigation.purpose ≠ "") : 
  ∃ (c : Census), c = investigation :=
sorry

end stating_comprehensive_investigation_is_census_l1073_107370


namespace new_person_weight_l1073_107309

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 89 kg -/
theorem new_person_weight :
  weight_of_new_person 8 3 65 = 89 := by
  sorry

end new_person_weight_l1073_107309


namespace neha_removed_amount_l1073_107301

/-- The amount removed from Neha's share in a money division problem -/
theorem neha_removed_amount (total : ℝ) (mahi_share : ℝ) (sabi_removed : ℝ) (mahi_removed : ℝ) :
  total = 1100 →
  mahi_share = 102 →
  sabi_removed = 8 →
  mahi_removed = 4 →
  ∃ (neha_share sabi_share neha_removed : ℝ),
    neha_share + sabi_share + mahi_share = total ∧
    neha_share - neha_removed = 2 * ((sabi_share - sabi_removed) / 8) ∧
    mahi_share - mahi_removed = 6 * ((sabi_share - sabi_removed) / 8) ∧
    neha_removed = 826.70 := by
  sorry

#eval (826.70 : Float)

end neha_removed_amount_l1073_107301


namespace solve_exponential_equation_l1073_107302

theorem solve_exponential_equation :
  ∃ x : ℝ, (4 : ℝ) ^ x * (4 : ℝ) ^ x * (4 : ℝ) ^ x = 256 ^ 3 ∧ x = 4 := by
  sorry

end solve_exponential_equation_l1073_107302


namespace initial_amount_of_liquid_a_l1073_107319

/-- Given a mixture of liquids A and B, prove the initial amount of A. -/
theorem initial_amount_of_liquid_a (a b : ℝ) : 
  a > 0 → b > 0 →  -- Ensure positive quantities
  a / b = 4 / 1 →  -- Initial ratio
  (a - 24) / (b - 6 + 30) = 2 / 3 →  -- New ratio after replacement
  a = 48 := by
sorry

end initial_amount_of_liquid_a_l1073_107319


namespace no_real_solutions_log_equation_l1073_107311

theorem no_real_solutions_log_equation :
  ¬ ∃ (x : ℝ), Real.log (x^2 - 3*x + 9) = 1 := by sorry

end no_real_solutions_log_equation_l1073_107311


namespace polygon_sides_l1073_107355

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end polygon_sides_l1073_107355


namespace quadratic_roots_inversely_proportional_l1073_107389

/-- 
Given a quadratic equation x^2 + px + q = 0 where q is constant and p is variable,
prove that the roots x₁ and x₂ are inversely proportional to each other.
-/
theorem quadratic_roots_inversely_proportional 
  (p q : ℝ) (x₁ x₂ : ℝ) (h_const : q ≠ 0) :
  (x₁^2 + p*x₁ + q = 0) → (x₂^2 + p*x₂ + q = 0) → 
  ∃ (k : ℝ), k ≠ 0 ∧ x₁ * x₂ = k :=
by sorry


end quadratic_roots_inversely_proportional_l1073_107389
