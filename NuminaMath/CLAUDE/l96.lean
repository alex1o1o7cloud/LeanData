import Mathlib

namespace NUMINAMATH_CALUDE_song_listens_proof_l96_9674

/-- Given a song with an initial number of listens that doubles each month for 3 months,
    resulting in a total of 900,000 listens, prove that the initial number of listens is 60,000. -/
theorem song_listens_proof (L : ℕ) : 
  (L + 2*L + 4*L + 8*L = 900000) → L = 60000 := by
  sorry

end NUMINAMATH_CALUDE_song_listens_proof_l96_9674


namespace NUMINAMATH_CALUDE_function_non_negative_implies_a_range_l96_9693

theorem function_non_negative_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, x^2 - 4*x + a ≥ 0) → a ∈ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_function_non_negative_implies_a_range_l96_9693


namespace NUMINAMATH_CALUDE_percentage_problem_l96_9696

theorem percentage_problem : 
  ∃ (P : ℝ), (P / 100) * 40 = 0.25 * 16 + 2 ∧ P = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l96_9696


namespace NUMINAMATH_CALUDE_negative_cube_root_of_negative_eight_equals_two_l96_9616

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem negative_cube_root_of_negative_eight_equals_two :
  -cubeRoot (-8) = 2 := by sorry

end NUMINAMATH_CALUDE_negative_cube_root_of_negative_eight_equals_two_l96_9616


namespace NUMINAMATH_CALUDE_magic_square_sum_l96_9611

/-- Represents a 3x3 magic square --/
structure MagicSquare :=
  (a b c d e : ℕ)
  (row1_sum : 30 + d + 24 = 32 + e + b)
  (row2_sum : 20 + e + b = 32 + e + b)
  (row3_sum : c + 32 + a = 32 + e + b)
  (col1_sum : 30 + 20 + c = 32 + e + b)
  (col2_sum : d + e + 32 = 32 + e + b)
  (col3_sum : 24 + b + a = 32 + e + b)
  (diag1_sum : 30 + e + a = 32 + e + b)
  (diag2_sum : 24 + e + c = 32 + e + b)

/-- The sum of d and e in the magic square is 54 --/
theorem magic_square_sum (ms : MagicSquare) : ms.d + ms.e = 54 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_sum_l96_9611


namespace NUMINAMATH_CALUDE_disney_banquet_residents_l96_9645

theorem disney_banquet_residents (total_attendees : ℕ) (resident_price non_resident_price : ℚ) (total_revenue : ℚ) :
  total_attendees = 586 →
  resident_price = 12.95 →
  non_resident_price = 17.95 →
  total_revenue = 9423.70 →
  ∃ (residents non_residents : ℕ),
    residents + non_residents = total_attendees ∧
    residents * resident_price + non_residents * non_resident_price = total_revenue ∧
    residents = 220 :=
by sorry

end NUMINAMATH_CALUDE_disney_banquet_residents_l96_9645


namespace NUMINAMATH_CALUDE_divisible_by_24_l96_9690

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, (n + 7)^2 - (n - 5)^2 = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l96_9690


namespace NUMINAMATH_CALUDE_isosceles_triangles_equal_perimeter_area_l96_9654

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equal_side : ℝ
  base : ℝ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equal_side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  let h := Real.sqrt (t.equal_side^2 - (t.base/2)^2)
  (1/2) * t.base * h

/-- The theorem to be proved -/
theorem isosceles_triangles_equal_perimeter_area (t1 t2 : IsoscelesTriangle)
  (h1 : t1.equal_side = 6 ∧ t1.base = 10)
  (h2 : perimeter t1 = perimeter t2)
  (h3 : area t1 = area t2)
  (h4 : t2.base^2 / 2 = perimeter t2 / 2) :
  t2.base = Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangles_equal_perimeter_area_l96_9654


namespace NUMINAMATH_CALUDE_box_surface_areas_and_cost_l96_9670

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular box -/
def surfaceArea (box : BoxDimensions) : ℝ :=
  2 * (box.length * box.width + box.length * box.height + box.width * box.height)

/-- Theorem about the surface areas of two boxes and their cost -/
theorem box_surface_areas_and_cost 
  (a b c : ℝ) 
  (small_box : BoxDimensions := ⟨a, b, c⟩)
  (large_box : BoxDimensions := ⟨2*a, 2*b, 1.5*c⟩)
  (cardboard_cost_per_sqm : ℝ := 15) : 
  (surfaceArea small_box + surfaceArea large_box = 10*a*b + 8*b*c + 8*a*c) ∧ 
  (surfaceArea large_box - surfaceArea small_box = 6*a*b + 4*b*c + 4*a*c) ∧
  (a = 20 → b = 10 → c = 15 → 
    cardboard_cost_per_sqm * (surfaceArea small_box + surfaceArea large_box) / 10000 = 8.4) :=
by sorry


end NUMINAMATH_CALUDE_box_surface_areas_and_cost_l96_9670


namespace NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l96_9681

theorem negative_one_greater_than_negative_two : -1 > -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_greater_than_negative_two_l96_9681


namespace NUMINAMATH_CALUDE_cos_shift_right_l96_9689

theorem cos_shift_right (x : ℝ) :
  2 * Real.cos (2 * (x - π/8)) = 2 * Real.cos (2 * x - π/4) :=
by sorry

end NUMINAMATH_CALUDE_cos_shift_right_l96_9689


namespace NUMINAMATH_CALUDE_school_population_l96_9623

theorem school_population (total_students : ℕ) : 
  (5 : ℚ)/8 * total_students + (3 : ℚ)/8 * total_students = total_students →  -- Girls + Boys = Total
  ((3 : ℚ)/10 * (5 : ℚ)/8 * total_students : ℚ) + 
  ((3 : ℚ)/5 * (3 : ℚ)/8 * total_students : ℚ) = 330 →                       -- Middle schoolers
  total_students = 800 := by
sorry

end NUMINAMATH_CALUDE_school_population_l96_9623


namespace NUMINAMATH_CALUDE_player_a_advantage_l96_9646

/-- Represents the outcome of a roll of two dice -/
structure DiceRoll :=
  (sum : Nat)
  (probability : Rat)

/-- Calculates the expected value for a player given a list of dice rolls -/
def expectedValue (rolls : List DiceRoll) : Rat :=
  rolls.foldl (fun acc roll => acc + roll.sum * roll.probability) 0

/-- Represents the game rules -/
def gameRules (roll : DiceRoll) : Rat :=
  if roll.sum % 2 = 1 then roll.sum * roll.probability
  else if roll.sum = 2 then 0
  else -roll.sum * roll.probability

/-- The list of all possible dice rolls and their probabilities -/
def allRolls : List DiceRoll := [
  ⟨2, 1/36⟩, ⟨3, 1/18⟩, ⟨4, 1/12⟩, ⟨5, 1/9⟩, ⟨6, 5/36⟩, 
  ⟨7, 1/6⟩, ⟨8, 5/36⟩, ⟨9, 1/9⟩, ⟨10, 1/12⟩, ⟨11, 1/18⟩, ⟨12, 1/36⟩
]

/-- The expected value for player A per roll -/
def expectedValueA : Rat := allRolls.foldl (fun acc roll => acc + gameRules roll) 0

theorem player_a_advantage : 
  expectedValueA > 0 ∧ 36 * expectedValueA = 2 := by sorry


end NUMINAMATH_CALUDE_player_a_advantage_l96_9646


namespace NUMINAMATH_CALUDE_direction_vector_value_l96_9634

/-- A line with direction vector (a, 4) passing through points (-2, 3) and (3, 5) has a = 10 -/
theorem direction_vector_value (a : ℝ) : 
  let v : ℝ × ℝ := (a, 4)
  let p₁ : ℝ × ℝ := (-2, 3)
  let p₂ : ℝ × ℝ := (3, 5)
  (∃ (t : ℝ), p₂ = p₁ + t • v) → a = 10 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_value_l96_9634


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_l96_9691

/-- Given a regular tetrahedron with base area S and volume V,
    the radius R of its inscribed sphere is equal to 3V/(4S) -/
theorem inscribed_sphere_radius_regular_tetrahedron
  (S V : ℝ) (h_S : S > 0) (h_V : V > 0) :
  ∃ R : ℝ, R = (3 * V) / (4 * S) ∧ R > 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_regular_tetrahedron_l96_9691


namespace NUMINAMATH_CALUDE_stratified_sampling_science_students_l96_9651

theorem stratified_sampling_science_students 
  (total_students : ℕ) 
  (science_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 720) 
  (h2 : science_students = 480) 
  (h3 : sample_size = 90) :
  (science_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 60 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_science_students_l96_9651


namespace NUMINAMATH_CALUDE_number_ratio_l96_9668

theorem number_ratio (x : ℝ) (h : 3 * (2 * x + 9) = 69) : 2 * x / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l96_9668


namespace NUMINAMATH_CALUDE_largest_perimeter_l96_9683

/-- Represents a triangle with two fixed sides and one variable side --/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ

/-- Checks if the given lengths can form a valid triangle --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- Calculates the perimeter of a triangle --/
def perimeter (t : Triangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The largest possible perimeter of a triangle with sides 7, 9, and an integer y is 31 --/
theorem largest_perimeter :
  ∀ y : ℕ,
    is_valid_triangle ⟨7, 9, y⟩ →
    perimeter ⟨7, 9, y⟩ ≤ 31 ∧
    ∃ (y' : ℕ), is_valid_triangle ⟨7, 9, y'⟩ ∧ perimeter ⟨7, 9, y'⟩ = 31 :=
by sorry


end NUMINAMATH_CALUDE_largest_perimeter_l96_9683


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l96_9632

/-- The area of the triangle formed by the tangent line to y = x^3 at (3, 27) and the axes is 54 -/
theorem tangent_line_triangle_area : 
  let f : ℝ → ℝ := fun x ↦ x^3
  let point : ℝ × ℝ := (3, 27)
  let tangent_line : ℝ → ℝ := fun x ↦ 27 * x - 54
  let triangle_area := 
    let x_intercept := (tangent_line 0) / (-27)
    let y_intercept := tangent_line 0
    (1/2) * x_intercept * (-y_intercept)
  triangle_area = 54 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l96_9632


namespace NUMINAMATH_CALUDE_solve_for_b_l96_9601

theorem solve_for_b (m a k c d b : ℝ) (h : m = (k * c * a * b) / (k * a - d)) :
  b = (m * k * a - m * d) / (k * c * a) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l96_9601


namespace NUMINAMATH_CALUDE_smallest_possible_d_l96_9684

theorem smallest_possible_d (c d : ℝ) : 
  (1 < c) → 
  (c < d) → 
  (1 + c ≤ d) → 
  (1 / c + 1 / d ≤ 1) → 
  d ≥ (3 + Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_d_l96_9684


namespace NUMINAMATH_CALUDE_quadratic_one_root_l96_9617

theorem quadratic_one_root (m : ℝ) : 
  (∃! x : ℝ, x^2 + 6*m*x + 2*m = 0) → m = 2/9 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l96_9617


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l96_9641

theorem largest_multiple_of_nine_less_than_hundred : 
  ∀ n : ℕ, n % 9 = 0 ∧ n < 100 → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l96_9641


namespace NUMINAMATH_CALUDE_calculation_difference_l96_9672

def correct_calculation : ℤ := 12 - (3 * 4)

def incorrect_calculation : ℤ := 12 - 3 * 4

theorem calculation_difference :
  correct_calculation - incorrect_calculation = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_difference_l96_9672


namespace NUMINAMATH_CALUDE_range_of_a_l96_9610

/-- Given sets A and B, where "x ∈ B" is a sufficient but not necessary condition for "x ∈ A",
    this theorem proves that the range of values for a is [0, 1]. -/
theorem range_of_a (A B : Set ℝ) (a : ℝ) : 
  A = {x : ℝ | x^2 - x - 2 ≤ 0} →
  B = {x : ℝ | |x - a| ≤ 1} →
  (∀ x, x ∈ B → x ∈ A) →
  ¬(∀ x, x ∈ A → x ∈ B) →
  0 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l96_9610


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l96_9662

/-- The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches. -/
theorem cylinder_surface_area : 
  let h : ℝ := 8
  let r : ℝ := 3
  let lateral_area := 2 * π * r * h
  let base_area := π * r^2
  let total_surface_area := lateral_area + 2 * base_area
  total_surface_area = 66 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l96_9662


namespace NUMINAMATH_CALUDE_youngbin_shopping_combinations_l96_9640

def n : ℕ := 3
def k : ℕ := 2

theorem youngbin_shopping_combinations : Nat.choose n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_youngbin_shopping_combinations_l96_9640


namespace NUMINAMATH_CALUDE_adam_gave_seven_boxes_l96_9666

/-- The number of boxes Adam gave to his little brother -/
def boxes_given (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : ℕ :=
  (total_boxes * pieces_per_box - pieces_left) / pieces_per_box

/-- Proof that Adam gave 7 boxes to his little brother -/
theorem adam_gave_seven_boxes :
  boxes_given 13 6 36 = 7 := by
  sorry

end NUMINAMATH_CALUDE_adam_gave_seven_boxes_l96_9666


namespace NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seat_capacity_l96_9631

/-- Represents a Ferris wheel with a given number of seats and total capacity. -/
structure FerrisWheel where
  numSeats : ℕ
  totalCapacity : ℕ

/-- Calculates the capacity of each seat in a Ferris wheel. -/
def seatCapacity (wheel : FerrisWheel) : ℕ :=
  wheel.totalCapacity / wheel.numSeats

theorem paradise_park_ferris_wheel_seat_capacity :
  let wheel : FerrisWheel := { numSeats := 14, totalCapacity := 84 }
  seatCapacity wheel = 6 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_ferris_wheel_seat_capacity_l96_9631


namespace NUMINAMATH_CALUDE_alpha_beta_cosine_l96_9635

theorem alpha_beta_cosine (α β : Real)
  (h_α : α ∈ Set.Ioo 0 (π / 3))
  (h_β : β ∈ Set.Ioo (π / 6) (π / 2))
  (eq_α : 5 * Real.sqrt 3 * Real.sin α + 5 * Real.cos α = 8)
  (eq_β : Real.sqrt 2 * Real.sin β + Real.sqrt 6 * Real.cos β = 2) :
  Real.cos (α + π / 6) = 3 / 5 ∧ Real.cos (α + β) = - Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_cosine_l96_9635


namespace NUMINAMATH_CALUDE_power_calculation_l96_9639

theorem power_calculation (a : ℝ) (h : a ≠ 0) : (a^2)^3 / a^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l96_9639


namespace NUMINAMATH_CALUDE_avg_sq_feet_per_person_approx_l96_9656

/-- The population of the United States -/
def us_population : ℕ := 226504825

/-- The area of the United States in square miles -/
def us_area_sq_miles : ℕ := 3615122

/-- The number of square feet in a square mile -/
def sq_feet_per_sq_mile : ℕ := 5280 * 5280

/-- The average square feet per person in the United States -/
def avg_sq_feet_per_person : ℚ :=
  (us_area_sq_miles * sq_feet_per_sq_mile : ℚ) / us_population

/-- Theorem stating that the average square feet per person is approximately 500000 -/
theorem avg_sq_feet_per_person_approx :
  ∃ ε > 0, abs (avg_sq_feet_per_person - 500000) < ε := by
  sorry

end NUMINAMATH_CALUDE_avg_sq_feet_per_person_approx_l96_9656


namespace NUMINAMATH_CALUDE_players_who_quit_l96_9628

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives_after : ℕ) :
  initial_players = 16 →
  lives_per_player = 8 →
  total_lives_after = 72 →
  initial_players - (total_lives_after / lives_per_player) = 7 :=
by sorry

end NUMINAMATH_CALUDE_players_who_quit_l96_9628


namespace NUMINAMATH_CALUDE_racing_game_cost_l96_9608

/-- The cost of the racing game given the total spent and the cost of the basketball game -/
theorem racing_game_cost (total_spent basketball_cost : ℚ) 
  (h1 : total_spent = 9.43)
  (h2 : basketball_cost = 5.20) : 
  total_spent - basketball_cost = 4.23 := by
  sorry

end NUMINAMATH_CALUDE_racing_game_cost_l96_9608


namespace NUMINAMATH_CALUDE_even_function_sum_l96_9661

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

-- Define the property of being an even function
def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (-2*a) (3*a - 1), f b x = f b (-x)) →
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_even_function_sum_l96_9661


namespace NUMINAMATH_CALUDE_playstation_cost_proof_l96_9613

-- Define the given values
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5
def games_to_sell : ℕ := 20

-- Define the cost of the PlayStation
def playstation_cost : ℝ := 500

-- Theorem statement
theorem playstation_cost_proof :
  birthday_money + christmas_money + game_price * (games_to_sell : ℝ) = playstation_cost := by
  sorry

end NUMINAMATH_CALUDE_playstation_cost_proof_l96_9613


namespace NUMINAMATH_CALUDE_product_equals_square_l96_9630

theorem product_equals_square : 
  250 * 9.996 * 3.996 * 500 = (4998 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l96_9630


namespace NUMINAMATH_CALUDE_percentage_calculation_l96_9649

theorem percentage_calculation (x y : ℝ) (P : ℝ) 
  (h1 : x / y = 4)
  (h2 : 0.8 * x = P / 100 * y) :
  P = 320 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l96_9649


namespace NUMINAMATH_CALUDE_tangent_slope_values_l96_9694

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 2 * x

-- Define the tangent line equation
def tangent_line (t : ℝ) (x : ℝ) : ℝ := t^2 + 2*t*(x - t)

-- Theorem statement
theorem tangent_slope_values :
  ∃ (t : ℝ), (tangent_line t 1 = 0) ∧ 
  ((curve_derivative t = 0) ∨ (curve_derivative t = 4)) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_values_l96_9694


namespace NUMINAMATH_CALUDE_circles_intersect_iff_l96_9638

/-- Two circles C1 and C2 in the plane -/
structure TwoCircles where
  /-- The parameter a for the second circle -/
  a : ℝ
  /-- a is positive -/
  a_pos : a > 0

/-- The condition for the two circles to intersect -/
def intersect (c : TwoCircles) : Prop :=
  3 < c.a ∧ c.a < 5

/-- Theorem stating the necessary and sufficient condition for the circles to intersect -/
theorem circles_intersect_iff (c : TwoCircles) :
  (∃ (x y : ℝ), x^2 + (y-1)^2 = 1 ∧ (x-c.a)^2 + (y-1)^2 = 16) ↔ intersect c := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_iff_l96_9638


namespace NUMINAMATH_CALUDE_min_sum_distances_l96_9618

/-- An ellipse with equation x²/2 + y² = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The squared distance between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- The theorem stating the minimum value of |OP|² + |PF|² -/
theorem min_sum_distances (P : ℝ × ℝ) (h : P ∈ Ellipse) :
  ∃ (m : ℝ), m = 2 ∧ ∀ Q ∈ Ellipse, m ≤ dist_squared O Q + dist_squared Q F :=
sorry

end NUMINAMATH_CALUDE_min_sum_distances_l96_9618


namespace NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_range_l96_9665

/-- A sequence is a jump sequence if for any three consecutive terms,
    the product (a_i - a_i+2)(a_i+2 - a_i+1) is positive. -/
def is_jump_sequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, (a i - a (i + 2)) * (a (i + 2) - a (i + 1)) > 0

/-- A sequence is geometric with ratio q if each term is q times the previous term. -/
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_jump_sequence_ratio_range
  (a : ℕ → ℝ) (q : ℝ) (h_geom : is_geometric_sequence a q) (h_jump : is_jump_sequence a) :
  q ∈ Set.Ioo (-1 : ℝ) 0 :=
sorry

end NUMINAMATH_CALUDE_geometric_jump_sequence_ratio_range_l96_9665


namespace NUMINAMATH_CALUDE_center_in_triangle_probability_l96_9663

theorem center_in_triangle_probability (n : ℕ) (hn : n > 0) :
  let sides := 2 * n + 1
  (n + 1 : ℚ) / (4 * n - 2) =
    1 - (sides * (n.choose 2) : ℚ) / (sides.choose 3) :=
by sorry

end NUMINAMATH_CALUDE_center_in_triangle_probability_l96_9663


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l96_9604

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y - x*y = 0) :
  x + 2*y ≥ 8 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y - x*y = 0 ∧ x + 2*y = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l96_9604


namespace NUMINAMATH_CALUDE_playground_area_l96_9676

/-- A rectangular playground with perimeter 72 feet and length three times the width has an area of 243 square feet. -/
theorem playground_area : ∀ w l : ℝ,
  w > 0 →
  l > 0 →
  2 * (w + l) = 72 →
  l = 3 * w →
  w * l = 243 := by
sorry

end NUMINAMATH_CALUDE_playground_area_l96_9676


namespace NUMINAMATH_CALUDE_bike_riding_average_l96_9697

/-- Calculates the average miles ridden per day given total miles and years --/
def average_miles_per_day (total_miles : ℕ) (years : ℕ) : ℚ :=
  total_miles / (years * 365)

/-- Theorem stating that riding 3,285 miles over 3 years averages to 3 miles per day --/
theorem bike_riding_average :
  average_miles_per_day 3285 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_bike_riding_average_l96_9697


namespace NUMINAMATH_CALUDE_line_segments_in_proportion_l96_9609

theorem line_segments_in_proportion :
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 5
  let c : ℝ := 2 * Real.sqrt 3
  let d : ℝ := Real.sqrt 15
  a * d = b * c := by sorry

end NUMINAMATH_CALUDE_line_segments_in_proportion_l96_9609


namespace NUMINAMATH_CALUDE_nine_by_nine_min_unoccupied_l96_9607

/-- Represents a chessboard with grasshoppers -/
structure Chessboard :=
  (size : Nat)
  (initial_grasshoppers : Nat)
  (diagonal_jump : Bool)

/-- Calculates the minimum number of unoccupied squares after jumps -/
def min_unoccupied_squares (board : Chessboard) : Nat :=
  sorry

/-- Theorem stating the minimum number of unoccupied squares for a 9x9 board -/
theorem nine_by_nine_min_unoccupied (board : Chessboard) : 
  board.size = 9 ∧ 
  board.initial_grasshoppers = 9 * 9 ∧ 
  board.diagonal_jump = true →
  min_unoccupied_squares board = 9 :=
sorry

end NUMINAMATH_CALUDE_nine_by_nine_min_unoccupied_l96_9607


namespace NUMINAMATH_CALUDE_not_divisible_by_five_l96_9600

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_five_l96_9600


namespace NUMINAMATH_CALUDE_unique_root_implies_m_equals_one_l96_9615

def f (x : ℝ) := 2 * x^2 + x - 4

theorem unique_root_implies_m_equals_one (m n : ℤ) :
  (n = m + 1) →
  (∃! x : ℝ, m < x ∧ x < n ∧ f x = 0) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_root_implies_m_equals_one_l96_9615


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l96_9699

theorem quadratic_roots_condition (p q : ℝ) : 
  (q < 0) ↔ (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l96_9699


namespace NUMINAMATH_CALUDE_intermediate_value_theorem_l96_9624

theorem intermediate_value_theorem {f : ℝ → ℝ} {a b : ℝ} (h_cont : ContinuousOn f (Set.Icc a b)) 
  (h_ab : a ≤ b) (h_sign : f a * f b < 0) : 
  ∃ c ∈ Set.Icc a b, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_intermediate_value_theorem_l96_9624


namespace NUMINAMATH_CALUDE_parallelogram_double_reflection_l96_9620

-- Define the parallelogram vertices
def A : ℝ × ℝ := (3, 6)
def B : ℝ × ℝ := (5, 10)
def C : ℝ × ℝ := (7, 6)
def D : ℝ × ℝ := (5, 2)

-- Define the reflection functions
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)  -- Translate down by 2
  let p'' := (p'.2, p'.1)   -- Reflect over y = x
  (p''.1, p''.2 + 2)        -- Translate up by 2

-- Theorem statement
theorem parallelogram_double_reflection :
  reflect_y_eq_x_plus_2 (reflect_x_axis D) = (-4, 7) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_double_reflection_l96_9620


namespace NUMINAMATH_CALUDE_geometric_sequence_a7_l96_9647

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a7 (a : ℕ → ℝ) :
  geometric_sequence a → a 3 = 3 → a 11 = 27 → a 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a7_l96_9647


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l96_9653

theorem min_value_expression (x : ℝ) : 
  (13 - x) * (8 - x) * (13 + x) * (8 + x) ≥ -2746.25 :=
by
  sorry

theorem min_value_attained : 
  ∃ x : ℝ, (13 - x) * (8 - x) * (13 + x) * (8 + x) = -2746.25 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l96_9653


namespace NUMINAMATH_CALUDE_savings_theorem_l96_9667

/-- Represents the savings and interest calculations for Dick and Jane --/
structure Savings where
  dick_1989 : ℝ
  jane_1989 : ℝ
  dick_increase_rate : ℝ
  interest_rate : ℝ

/-- Calculates the total savings of Dick and Jane in 1990 --/
def total_savings_1990 (s : Savings) : ℝ :=
  (s.dick_1989 * (1 + s.dick_increase_rate) + s.dick_1989) * (1 + s.interest_rate) +
  s.jane_1989 * (1 + s.interest_rate)

/-- Calculates the percent change in Jane's savings from 1989 to 1990 --/
def jane_savings_percent_change (s : Savings) : ℝ := 0

/-- Theorem stating the total savings in 1990 and Jane's savings percent change --/
theorem savings_theorem (s : Savings) 
  (h1 : s.dick_1989 = 5000)
  (h2 : s.jane_1989 = 3000)
  (h3 : s.dick_increase_rate = 0.1)
  (h4 : s.interest_rate = 0.03) :
  total_savings_1990 s = 8740 ∧ jane_savings_percent_change s = 0 := by
  sorry

end NUMINAMATH_CALUDE_savings_theorem_l96_9667


namespace NUMINAMATH_CALUDE_leaves_collection_time_l96_9626

/-- The time taken to collect leaves under given conditions -/
def collect_leaves_time (total_leaves : ℕ) (collect_rate : ℕ) (scatter_rate : ℕ) (cycle_time : ℚ) : ℚ :=
  let net_increase := collect_rate - scatter_rate
  let full_cycles := (total_leaves - net_increase) / net_increase
  (full_cycles * cycle_time + cycle_time) / 60

/-- The problem statement -/
theorem leaves_collection_time :
  collect_leaves_time 50 5 3 (45 / 60) = 75 / 4 :=
sorry

end NUMINAMATH_CALUDE_leaves_collection_time_l96_9626


namespace NUMINAMATH_CALUDE_max_square_partitions_l96_9621

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFencing : ℕ := 2100

/-- Calculates the number of square partitions given the side length of each square -/
def numPartitions (field : FieldDimensions) (squareSide : ℕ) : ℕ :=
  (field.width / squareSide) * (field.length / squareSide)

/-- Calculates the required internal fencing for given partitions -/
def requiredFencing (field : FieldDimensions) (squareSide : ℕ) : ℕ :=
  (field.width / squareSide - 1) * field.length + 
  (field.length / squareSide - 1) * field.width

/-- Theorem stating the maximum number of square partitions -/
theorem max_square_partitions (field : FieldDimensions) 
  (h1 : field.width = 30) 
  (h2 : field.length = 45) : 
  (∃ (squareSide : ℕ), 
    numPartitions field squareSide = 75 ∧ 
    requiredFencing field squareSide ≤ availableFencing ∧
    ∀ (otherSide : ℕ), 
      requiredFencing field otherSide ≤ availableFencing → 
      numPartitions field otherSide ≤ 75) :=
  sorry

#check max_square_partitions

end NUMINAMATH_CALUDE_max_square_partitions_l96_9621


namespace NUMINAMATH_CALUDE_cubic_roots_of_unity_l96_9664

theorem cubic_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_of_unity_l96_9664


namespace NUMINAMATH_CALUDE_product_325_67_base_7_units_digit_l96_9692

theorem product_325_67_base_7_units_digit : 
  (325 * 67) % 7 = 5 := by
sorry

end NUMINAMATH_CALUDE_product_325_67_base_7_units_digit_l96_9692


namespace NUMINAMATH_CALUDE_square_area_specific_vertices_l96_9686

/-- The area of a square with vertices at (1, 1), (-4, 2), (-3, 7), and (2, 6) is 26 square units. -/
theorem square_area_specific_vertices : 
  let P : ℝ × ℝ := (1, 1)
  let Q : ℝ × ℝ := (-4, 2)
  let R : ℝ × ℝ := (-3, 7)
  let S : ℝ × ℝ := (2, 6)
  let side_length := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  side_length^2 = 26 := by sorry

end NUMINAMATH_CALUDE_square_area_specific_vertices_l96_9686


namespace NUMINAMATH_CALUDE_another_rational_right_triangle_with_same_area_l96_9688

/-- Given a right triangle with rational sides and area S, 
    there exists another right triangle with rational sides and area S -/
theorem another_rational_right_triangle_with_same_area 
  (a b c S : ℚ) : 
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem for right triangle
  (S = (1/2) * a * b) →  -- Area formula
  (∃ (a' b' c' : ℚ), 
    a'^2 + b'^2 = c'^2 ∧  -- New triangle is right-angled
    (1/2) * a' * b' = S ∧  -- New triangle has the same area
    (a' ≠ a ∨ b' ≠ b ∨ c' ≠ c))  -- New triangle is different from the original
  := by sorry

end NUMINAMATH_CALUDE_another_rational_right_triangle_with_same_area_l96_9688


namespace NUMINAMATH_CALUDE_min_value_theorem_l96_9614

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (2/a + 3/b) ≥ 8 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l96_9614


namespace NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l96_9648

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem inequality_solution (x : ℝ) : 
  f (x - 1) + f (1 - x) ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

-- Theorem for part II
theorem inequality_proof (x a : ℝ) (h : a < 0) : 
  f (a * x) - a * f x ≥ f a := by sorry

end NUMINAMATH_CALUDE_inequality_solution_inequality_proof_l96_9648


namespace NUMINAMATH_CALUDE_tangent_line_parallel_and_inequality_l96_9603

noncomputable def f (x : ℝ) := Real.log x

noncomputable def g (a : ℝ) (x : ℝ) := f x + a / x - 1

theorem tangent_line_parallel_and_inequality (a : ℝ) :
  (∃ (m : ℝ), m = (1 / 2 : ℝ) - a / 4 ∧ m = -(1 / 2 : ℝ)) ∧
  (∀ (m n : ℝ), m > n → n > 0 → (m - n) / (m + n) < (Real.log m - Real.log n) / 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_and_inequality_l96_9603


namespace NUMINAMATH_CALUDE_solve_for_a_l96_9682

theorem solve_for_a : ∃ a : ℝ, (2 - 3 * (a + 1) = 2 * 1) ∧ (a = -1) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l96_9682


namespace NUMINAMATH_CALUDE_a_5_plus_a_6_equals_152_l96_9627

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 3 * n + 1

-- Define the partial sum S_n
def S (n : ℕ) : ℤ := n^3

-- State the theorem
theorem a_5_plus_a_6_equals_152 : a 5 + a 6 = 152 := by
  sorry

end NUMINAMATH_CALUDE_a_5_plus_a_6_equals_152_l96_9627


namespace NUMINAMATH_CALUDE_expression_one_expression_two_expression_three_expression_four_l96_9660

-- 1. 75 + 7 × 5
theorem expression_one : 75 + 7 * 5 = 110 := by sorry

-- 2. 148 - 48 ÷ 2
theorem expression_two : 148 - 48 / 2 = 124 := by sorry

-- 3. (400 - 160) ÷ 8
theorem expression_three : (400 - 160) / 8 = 30 := by sorry

-- 4. 4 × 25 × 7
theorem expression_four : 4 * 25 * 7 = 700 := by sorry

end NUMINAMATH_CALUDE_expression_one_expression_two_expression_three_expression_four_l96_9660


namespace NUMINAMATH_CALUDE_intersection_product_l96_9695

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection of line l with y-axis
def point_M : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem intersection_product (A B : ℝ × ℝ) :
  curve_C A.1 A.2 →
  curve_C B.1 B.2 →
  line_l A.1 A.2 →
  line_l B.1 B.2 →
  A ≠ B →
  (∃ (t : ℝ), line_l t (point_M.2 + (t - point_M.1))) →
  |point_M.1 - A.1| * |point_M.1 - B.1| = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_product_l96_9695


namespace NUMINAMATH_CALUDE_unknown_percentage_of_250_l96_9643

/-- Given that 28% of 400 plus some percentage of 250 equals 224.5,
    prove that the unknown percentage of 250 is 45%. -/
theorem unknown_percentage_of_250 (p : ℝ) : 
  (0.28 * 400 + p / 100 * 250 = 224.5) → p = 45 :=
by sorry

end NUMINAMATH_CALUDE_unknown_percentage_of_250_l96_9643


namespace NUMINAMATH_CALUDE_xiaoming_math_score_l96_9606

theorem xiaoming_math_score (class_a_avg : ℝ) (class_b_avg : ℝ) 
  (xiaoming_score : ℝ) :
  class_a_avg = 87 →
  class_b_avg = 82 →
  xiaoming_score > class_b_avg →
  xiaoming_score < class_a_avg →
  xiaoming_score = 85 :=
by
  sorry

end NUMINAMATH_CALUDE_xiaoming_math_score_l96_9606


namespace NUMINAMATH_CALUDE_expected_outcome_is_correct_l96_9677

/-- Represents the possible outcomes of rolling a die -/
inductive DieOutcome
| One
| Two
| Three
| Four
| Five
| Six

/-- The probability of rolling a specific outcome -/
def probability (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One | DieOutcome.Two | DieOutcome.Three => 1/3
  | DieOutcome.Four | DieOutcome.Five | DieOutcome.Six => 1/6

/-- The monetary value associated with each outcome -/
def monetaryValue (outcome : DieOutcome) : ℚ :=
  match outcome with
  | DieOutcome.One | DieOutcome.Two | DieOutcome.Three => 4
  | DieOutcome.Four => -2
  | DieOutcome.Five => -5
  | DieOutcome.Six => -7

/-- The expected monetary outcome of a roll -/
def expectedMonetaryOutcome : ℚ :=
  (probability DieOutcome.One * monetaryValue DieOutcome.One) +
  (probability DieOutcome.Two * monetaryValue DieOutcome.Two) +
  (probability DieOutcome.Three * monetaryValue DieOutcome.Three) +
  (probability DieOutcome.Four * monetaryValue DieOutcome.Four) +
  (probability DieOutcome.Five * monetaryValue DieOutcome.Five) +
  (probability DieOutcome.Six * monetaryValue DieOutcome.Six)

theorem expected_outcome_is_correct :
  expectedMonetaryOutcome = 167/100 := by sorry

end NUMINAMATH_CALUDE_expected_outcome_is_correct_l96_9677


namespace NUMINAMATH_CALUDE_function_identity_l96_9642

theorem function_identity (f : ℕ+ → ℤ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ+, f (m * n) = f m * f n)
  (h3 : ∀ m n : ℕ+, m > n → f m > f n) :
  ∀ n : ℕ+, f n = n := by
sorry

end NUMINAMATH_CALUDE_function_identity_l96_9642


namespace NUMINAMATH_CALUDE_inner_circle_radius_l96_9657

/-- Given a circle of radius R and a point A on its diameter at distance a from the center,
    the radius of the circle that touches the diameter at A and is internally tangent to the given circle
    is (R^2 - a^2) / (2R). -/
theorem inner_circle_radius (R a : ℝ) (h₁ : R > 0) (h₂ : 0 < a ∧ a < R) :
  ∃ x : ℝ, x > 0 ∧ x = (R^2 - a^2) / (2*R) ∧
  x^2 + a^2 = (R - x)^2 :=
sorry

end NUMINAMATH_CALUDE_inner_circle_radius_l96_9657


namespace NUMINAMATH_CALUDE_sum_two_digit_integers_mod_1000_l96_9636

/-- The sum of all four-digit integers formed using exactly two different digits -/
def S : ℕ := sorry

/-- Theorem stating that S mod 1000 = 370 -/
theorem sum_two_digit_integers_mod_1000 : S % 1000 = 370 := by sorry

end NUMINAMATH_CALUDE_sum_two_digit_integers_mod_1000_l96_9636


namespace NUMINAMATH_CALUDE_sale_price_ratio_l96_9669

theorem sale_price_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) :
  y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_ratio_l96_9669


namespace NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l96_9644

theorem sum_of_first_six_primes_mod_seventh_prime : 
  let sum_first_six_primes := 41
  let seventh_prime := 17
  sum_first_six_primes % seventh_prime = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_six_primes_mod_seventh_prime_l96_9644


namespace NUMINAMATH_CALUDE_arithmetic_progression_11_arithmetic_progression_10000_no_infinite_arithmetic_progression_l96_9650

-- Define a function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Define what it means for a sequence to be an arithmetic progression
def isArithmeticProgression (seq : ℕ → ℕ) : Prop := sorry

-- Define what it means for a sequence to be increasing
def isIncreasing (seq : ℕ → ℕ) : Prop := sorry

-- Theorem for the case of 11 terms
theorem arithmetic_progression_11 : 
  ∃ (seq : Fin 11 → ℕ), 
    isArithmeticProgression (λ i => seq i) ∧ 
    isIncreasing (λ i => seq i) ∧
    isArithmeticProgression (λ i => sumOfDigits (seq i)) ∧
    isIncreasing (λ i => sumOfDigits (seq i)) := sorry

-- Theorem for the case of 10,000 terms
theorem arithmetic_progression_10000 : 
  ∃ (seq : Fin 10000 → ℕ), 
    isArithmeticProgression (λ i => seq i) ∧ 
    isIncreasing (λ i => seq i) ∧
    isArithmeticProgression (λ i => sumOfDigits (seq i)) ∧
    isIncreasing (λ i => sumOfDigits (seq i)) := sorry

-- Theorem for the case of infinite natural numbers
theorem no_infinite_arithmetic_progression :
  ¬∃ (seq : ℕ → ℕ), 
    isArithmeticProgression seq ∧ 
    isIncreasing seq ∧
    isArithmeticProgression (λ n => sumOfDigits (seq n)) ∧
    isIncreasing (λ n => sumOfDigits (seq n)) := sorry

end NUMINAMATH_CALUDE_arithmetic_progression_11_arithmetic_progression_10000_no_infinite_arithmetic_progression_l96_9650


namespace NUMINAMATH_CALUDE_x_equals_seven_l96_9602

theorem x_equals_seven (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_seven_l96_9602


namespace NUMINAMATH_CALUDE_weight_of_8_moles_AlI3_l96_9698

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of Aluminum atoms in AlI3 -/
def num_Al_atoms : ℕ := 1

/-- The number of Iodine atoms in AlI3 -/
def num_I_atoms : ℕ := 3

/-- The number of moles of AlI3 -/
def num_moles_AlI3 : ℝ := 8

/-- The molecular weight of AlI3 in g/mol -/
def molecular_weight_AlI3 : ℝ := 
  num_Al_atoms * atomic_weight_Al + num_I_atoms * atomic_weight_I

/-- The weight of a given number of moles of AlI3 in grams -/
def weight_AlI3 (moles : ℝ) : ℝ := moles * molecular_weight_AlI3

theorem weight_of_8_moles_AlI3 : 
  weight_AlI3 num_moles_AlI3 = 3261.44 := by
  sorry


end NUMINAMATH_CALUDE_weight_of_8_moles_AlI3_l96_9698


namespace NUMINAMATH_CALUDE_volume_T_coefficients_qr_ps_ratio_l96_9680

/-- A right rectangular prism with edge lengths 2, 4, and 6 units -/
structure RectangularPrism where
  length : ℝ := 2
  width : ℝ := 4
  height : ℝ := 6

/-- The set of points within distance r from any point in the prism -/
def T (C : RectangularPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume function of T(r) -/
def volume_T (C : RectangularPrism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function -/
structure VolumeCoefficients where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ

theorem volume_T_coefficients (C : RectangularPrism) :
  ∃ (coeff : VolumeCoefficients),
    ∀ r : ℝ, volume_T C r = coeff.P * r^3 + coeff.Q * r^2 + coeff.R * r + coeff.S :=
  sorry

theorem qr_ps_ratio (C : RectangularPrism) (coeff : VolumeCoefficients)
    (h : ∀ r : ℝ, volume_T C r = coeff.P * r^3 + coeff.Q * r^2 + coeff.R * r + coeff.S) :
    coeff.Q * coeff.R / (coeff.P * coeff.S) = 16.5 :=
  sorry

end NUMINAMATH_CALUDE_volume_T_coefficients_qr_ps_ratio_l96_9680


namespace NUMINAMATH_CALUDE_smallest_angle_in_special_trapezoid_l96_9687

/-- Represents the angles of a trapezoid --/
structure TrapezoidAngles where
  a : ℝ  -- First angle
  b : ℝ  -- Second angle
  c : ℝ  -- Third angle
  d : ℝ  -- Fourth angle

/-- Checks if the angles form a valid trapezoid configuration --/
def is_valid_trapezoid (angles : TrapezoidAngles) : Prop :=
  -- Sum of angles in a quadrilateral is 360°
  angles.a + angles.b + angles.c + angles.d = 360 ∧
  -- One pair of opposite angles are supplementary
  (angles.a + angles.c = 180 ∨ angles.b + angles.d = 180) ∧
  -- Consecutive angles on each side form arithmetic sequences
  (∃ x y : ℝ, (angles.a = x ∧ angles.b = x + y) ∨ (angles.c = x ∧ angles.d = x + y)) ∧
  (∃ p q : ℝ, (angles.b = p ∧ angles.c = p + q) ∨ (angles.d = p ∧ angles.a = p + q))

/-- The main theorem --/
theorem smallest_angle_in_special_trapezoid :
  ∀ angles : TrapezoidAngles,
  is_valid_trapezoid angles →
  (angles.a = 140 ∨ angles.b = 140 ∨ angles.c = 140 ∨ angles.d = 140) →
  (angles.a = 20 ∨ angles.b = 20 ∨ angles.c = 20 ∨ angles.d = 20) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_special_trapezoid_l96_9687


namespace NUMINAMATH_CALUDE_staircase_polygon_perimeter_staircase_polygon_area_l96_9671

/-- A polygonal region formed by removing a 3x4 rectangle from an 8x12 rectangle -/
structure StaircasePolygon where
  width : ℕ := 12
  height : ℕ := 8
  small_width : ℕ := 3
  small_height : ℕ := 4
  area : ℕ := 86
  stair_side_length : ℕ := 1
  stair_side_count : ℕ := 12

/-- The perimeter of a StaircasePolygon is 44 -/
theorem staircase_polygon_perimeter (p : StaircasePolygon) : 
  p.width + p.height + (p.width - p.small_width) + (p.height - p.small_height) + p.stair_side_count * p.stair_side_length = 44 := by
  sorry

/-- The area of a StaircasePolygon is consistent with its dimensions -/
theorem staircase_polygon_area (p : StaircasePolygon) :
  p.area = p.width * p.height - p.small_width * p.small_height := by
  sorry

end NUMINAMATH_CALUDE_staircase_polygon_perimeter_staircase_polygon_area_l96_9671


namespace NUMINAMATH_CALUDE_total_cost_proof_l96_9675

def hand_mitts_cost : ℚ := 14
def apron_cost : ℚ := 16
def utensils_cost : ℚ := 10
def knife_cost : ℚ := 2 * utensils_cost
def discount_rate : ℚ := 0.25
def tax_rate : ℚ := 0.08
def num_recipients : ℕ := 8

def total_cost : ℚ :=
  let set_cost := hand_mitts_cost + apron_cost + utensils_cost + knife_cost
  let total_before_discount := num_recipients * set_cost
  let discounted_total := total_before_discount * (1 - discount_rate)
  discounted_total * (1 + tax_rate)

theorem total_cost_proof : total_cost = 388.8 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_proof_l96_9675


namespace NUMINAMATH_CALUDE_remaining_payment_theorem_l96_9637

def calculate_remaining_payment (deposit : ℚ) (percentage : ℚ) : ℚ :=
  deposit / percentage - deposit

def total_remaining_payment (deposit1 deposit2 deposit3 : ℚ) (percentage1 percentage2 percentage3 : ℚ) : ℚ :=
  calculate_remaining_payment deposit1 percentage1 +
  calculate_remaining_payment deposit2 percentage2 +
  calculate_remaining_payment deposit3 percentage3

theorem remaining_payment_theorem (deposit1 deposit2 deposit3 : ℚ) (percentage1 percentage2 percentage3 : ℚ)
  (h1 : deposit1 = 105)
  (h2 : deposit2 = 180)
  (h3 : deposit3 = 300)
  (h4 : percentage1 = 1/10)
  (h5 : percentage2 = 15/100)
  (h6 : percentage3 = 1/5) :
  total_remaining_payment deposit1 deposit2 deposit3 percentage1 percentage2 percentage3 = 3165 := by
  sorry

#eval total_remaining_payment 105 180 300 (1/10) (15/100) (1/5)

end NUMINAMATH_CALUDE_remaining_payment_theorem_l96_9637


namespace NUMINAMATH_CALUDE_rectangle_placement_l96_9655

theorem rectangle_placement (a b c d : ℝ) 
  (h1 : a < c) (h2 : c < d) (h3 : d < b) (h4 : a * b < c * d) :
  (∃ (θ : ℝ), 0 < θ ∧ θ < π / 2 ∧ 
    b * Real.cos θ + a * Real.sin θ ≤ c ∧
    b * Real.sin θ + a * Real.cos θ ≤ d) ↔ 
  (b^2 - a^2)^2 ≤ (b*d - a*c)^2 + (b*c - a*d)^2 := by sorry

end NUMINAMATH_CALUDE_rectangle_placement_l96_9655


namespace NUMINAMATH_CALUDE_valid_lineup_count_l96_9673

/-- The total number of players in the basketball team -/
def total_players : ℕ := 16

/-- The number of quadruplets in the team -/
def quadruplets : ℕ := 4

/-- The size of the starting lineup -/
def lineup_size : ℕ := 6

/-- The maximum number of quadruplets allowed in the starting lineup -/
def max_quadruplets : ℕ := 2

/-- The number of ways to choose the starting lineup with the given restrictions -/
def valid_lineups : ℕ := 7062

theorem valid_lineup_count : 
  (Nat.choose total_players lineup_size) - 
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (lineup_size - 3) +
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (lineup_size - 4)) = 
  valid_lineups :=
sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l96_9673


namespace NUMINAMATH_CALUDE_car_average_speed_l96_9658

/-- Calculate the average speed of a car given its uphill and downhill speeds and distances --/
theorem car_average_speed (uphill_speed downhill_speed uphill_distance downhill_distance : ℝ) :
  uphill_speed = 30 →
  downhill_speed = 40 →
  uphill_distance = 100 →
  downhill_distance = 50 →
  let total_distance := uphill_distance + downhill_distance
  let uphill_time := uphill_distance / uphill_speed
  let downhill_time := downhill_distance / downhill_speed
  let total_time := uphill_time + downhill_time
  let average_speed := total_distance / total_time
  average_speed = 1800 / 55 := by
  sorry

#eval (1800 : ℚ) / 55

end NUMINAMATH_CALUDE_car_average_speed_l96_9658


namespace NUMINAMATH_CALUDE_base_9_addition_multiplication_l96_9629

/-- Converts a number from base 9 to base 10 --/
def base9ToBase10 (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 9^2 + tens * 9 + ones

/-- Converts a number from base 10 to base 9 --/
def base10ToBase9 (n : ℕ) : ℕ :=
  sorry -- Implementation not provided, as it's not required for the statement

theorem base_9_addition_multiplication :
  let a := base9ToBase10 436
  let b := base9ToBase10 782
  let c := base9ToBase10 204
  let d := base9ToBase10 12
  base10ToBase9 ((a + b + c) * d) = 18508 := by
  sorry

end NUMINAMATH_CALUDE_base_9_addition_multiplication_l96_9629


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l96_9685

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m - y^2 = 1 ∧ m > 0

-- Define the asymptote of C
def asymptote (m : ℝ) (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + m * y = 0

-- Theorem statement
theorem hyperbola_focal_length (m : ℝ) :
  (∀ x y, hyperbola m x y → asymptote m x y) →
  (∃ a b c : ℝ, a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l96_9685


namespace NUMINAMATH_CALUDE_simplify_polynomial_l96_9605

theorem simplify_polynomial (x : ℝ) : 
  4 - 6*x - 8*x^2 + 12 - 14*x + 16*x^2 - 18 + 20*x + 24*x^2 = 32*x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l96_9605


namespace NUMINAMATH_CALUDE_percentage_in_70_79_is_one_third_l96_9622

/-- Represents the frequency distribution of test scores -/
def score_distribution : List (String × ℕ) :=
  [("90% - 100%", 3),
   ("80% - 89%", 5),
   ("70% - 79%", 8),
   ("60% - 69%", 4),
   ("50% - 59%", 1),
   ("Below 50%", 3)]

/-- Total number of students in the class -/
def total_students : ℕ := (score_distribution.map (λ x => x.2)).sum

/-- Number of students who scored in the 70%-79% range -/
def students_in_70_79 : ℕ := 
  (score_distribution.filter (λ x => x.1 = "70% - 79%")).map (λ x => x.2) |>.sum

/-- Theorem stating that the percentage of students who scored in the 70%-79% range is 1/3 of the class -/
theorem percentage_in_70_79_is_one_third :
  (students_in_70_79 : ℚ) / (total_students : ℚ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percentage_in_70_79_is_one_third_l96_9622


namespace NUMINAMATH_CALUDE_concrete_mixture_cement_percentage_l96_9659

/-- Proves that given two types of concrete mixed in equal amounts to create a total mixture with a specific cement percentage, if one type has a known cement percentage, then the other type's cement percentage can be determined. -/
theorem concrete_mixture_cement_percentage 
  (total_weight : ℝ) 
  (final_cement_percentage : ℝ) 
  (weight_each_type : ℝ) 
  (cement_percentage_type1 : ℝ) :
  total_weight = 4500 →
  final_cement_percentage = 10.8 →
  weight_each_type = 1125 →
  cement_percentage_type1 = 10.8 →
  ∃ (cement_percentage_type2 : ℝ),
    cement_percentage_type2 = 32.4 ∧
    weight_each_type * cement_percentage_type1 / 100 + 
    weight_each_type * cement_percentage_type2 / 100 = 
    total_weight * final_cement_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_concrete_mixture_cement_percentage_l96_9659


namespace NUMINAMATH_CALUDE_jen_birds_count_l96_9678

/-- The number of birds Jen has given the conditions -/
def total_birds (chickens ducks geese : ℕ) : ℕ :=
  chickens + ducks + geese

/-- Theorem stating the total number of birds Jen has -/
theorem jen_birds_count :
  ∀ (chickens ducks geese : ℕ),
    ducks = 150 →
    ducks = 4 * chickens + 10 →
    geese = (ducks + chickens) / 2 →
    total_birds chickens ducks geese = 277 := by
  sorry

#check jen_birds_count

end NUMINAMATH_CALUDE_jen_birds_count_l96_9678


namespace NUMINAMATH_CALUDE_twenty_five_binary_l96_9612

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem twenty_five_binary :
  toBinary 25 = [true, false, false, true, true] :=
by sorry

end NUMINAMATH_CALUDE_twenty_five_binary_l96_9612


namespace NUMINAMATH_CALUDE_specific_right_triangle_l96_9633

/-- A right triangle with inscribed and circumscribed circles -/
structure RightTriangle where
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- The shortest side of the triangle -/
  a : ℝ
  /-- The middle-length side of the triangle -/
  b : ℝ
  /-- The hypotenuse of the triangle -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The inradius is correct for this triangle -/
  inradius_correct : inradius = (a + b - c) / 2
  /-- The circumradius is correct for this triangle -/
  circumradius_correct : circumradius = c / 2

/-- The main theorem about the specific right triangle -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.inradius = 8 ∧ t.circumradius = 41 ∧ t.a = 18 ∧ t.b = 80 ∧ t.c = 82 := by
  sorry

end NUMINAMATH_CALUDE_specific_right_triangle_l96_9633


namespace NUMINAMATH_CALUDE_sandy_comic_books_l96_9652

theorem sandy_comic_books (initial : ℕ) (final : ℕ) (bought : ℕ) : 
  initial = 14 →
  final = 13 →
  bought = final - (initial / 2) →
  bought = 6 := by
sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l96_9652


namespace NUMINAMATH_CALUDE_sum_first_last_number_l96_9679

theorem sum_first_last_number (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 5 →
  d = 4 →
  a + d = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_first_last_number_l96_9679


namespace NUMINAMATH_CALUDE_salary_increase_after_two_years_l96_9619

-- Define the raise percentage
def raise_percentage : ℝ := 0.05

-- Define the number of six-month periods in two years
def periods : ℕ := 4

-- Theorem stating the salary increase after two years
theorem salary_increase_after_two_years :
  let final_multiplier := (1 + raise_percentage) ^ periods
  abs (final_multiplier - 1 - 0.2155) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_after_two_years_l96_9619


namespace NUMINAMATH_CALUDE_ball_radius_under_shadow_l96_9625

/-- The radius of a ball under specific shadow conditions -/
theorem ball_radius_under_shadow (ball_shadow_length : ℝ) (ruler_shadow_length : ℝ) 
  (h1 : ball_shadow_length = 10)
  (h2 : ruler_shadow_length = 2) : 
  ∃ (r : ℝ), r = 10 * Real.sqrt 5 - 20 ∧ r > 0 := by
  sorry

end NUMINAMATH_CALUDE_ball_radius_under_shadow_l96_9625
