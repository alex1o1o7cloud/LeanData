import Mathlib

namespace NUMINAMATH_CALUDE_sine_tangent_sum_greater_than_2pi_l2074_207486

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

-- State the theorem
theorem sine_tangent_sum_greater_than_2pi (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B + Real.sin t.C +
  Real.tan t.A + Real.tan t.B + Real.tan t.C > 2 * π :=
by sorry

end NUMINAMATH_CALUDE_sine_tangent_sum_greater_than_2pi_l2074_207486


namespace NUMINAMATH_CALUDE_vector_v_satisfies_conditions_l2074_207467

/-- Parametric equation of a line in 2D space -/
structure ParamLine2D where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Definition of line l -/
def line_l : ParamLine2D :=
  { x := λ t => 2 + 5*t,
    y := λ t => 3 + 2*t }

/-- Definition of line m -/
def line_m : ParamLine2D :=
  { x := λ s => -7 + 5*s,
    y := λ s => 9 + 2*s }

/-- Point A on line l -/
def point_A : Vector2D :=
  { x := line_l.x 0,
    y := line_l.y 0 }

/-- Point B on line m -/
def point_B : Vector2D :=
  { x := line_m.x 0,
    y := line_m.y 0 }

/-- Vector v that PA is projected onto -/
def vector_v : Vector2D :=
  { x := -2,
    y := 5 }

/-- Theorem: The vector v satisfies the given conditions -/
theorem vector_v_satisfies_conditions :
  vector_v.y - vector_v.x = 7 ∧
  (∃ (P : Vector2D), P.x = point_A.x ∧ P.y = point_A.y) ∧
  (∀ (B : Vector2D), B.x = line_m.x 0 ∧ B.y = line_m.y 0 →
    ∃ (k : ℝ), vector_v.x * k = 0 ∧ vector_v.y * k = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_v_satisfies_conditions_l2074_207467


namespace NUMINAMATH_CALUDE_min_moves_for_equal_distribution_l2074_207450

/-- Represents the number of boxes -/
def N : ℕ := 2019

/-- Represents the number of boxes that receive 100 stones in each operation -/
def l : ℕ := 100

/-- Calculates the minimum number of moves required to distribute stones equally -/
def min_moves (N l : ℕ) : ℕ :=
  let d := Nat.gcd N l
  Nat.ceil ((N^2 : ℚ) / (d * l))

theorem min_moves_for_equal_distribution :
  min_moves N l = 40762 :=
sorry

end NUMINAMATH_CALUDE_min_moves_for_equal_distribution_l2074_207450


namespace NUMINAMATH_CALUDE_percentage_ratio_theorem_l2074_207474

theorem percentage_ratio_theorem (y : ℝ) : 
  let x := 7 * y
  let z := 3 * (x - y)
  let percentage := (x - y) / x * 100
  let ratio := z / (x + y)
  percentage / ratio = 800 / 21 := by
sorry

end NUMINAMATH_CALUDE_percentage_ratio_theorem_l2074_207474


namespace NUMINAMATH_CALUDE_gain_percent_is_one_percent_l2074_207425

-- Define the gain and cost price
def gain : ℚ := 70 / 100  -- 70 paise = 0.70 Rs
def cost_price : ℚ := 70  -- 70 Rs

-- Define the gain percent formula
def gain_percent (g c : ℚ) : ℚ := (g / c) * 100

-- Theorem statement
theorem gain_percent_is_one_percent :
  gain_percent gain cost_price = 1 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_is_one_percent_l2074_207425


namespace NUMINAMATH_CALUDE_knights_count_l2074_207477

/-- Represents an islander, who can be either a knight or a liar -/
inductive Islander
| Knight
| Liar

/-- The total number of islanders -/
def total_islanders : Nat := 6

/-- Determines if an islander's statement is true based on the actual number of liars -/
def statement_is_true (actual_liars : Nat) : Prop :=
  actual_liars = 4

/-- Determines if an islander's behavior is consistent with their type and statement -/
def is_consistent (islander : Islander) (actual_liars : Nat) : Prop :=
  match islander with
  | Islander.Knight => statement_is_true actual_liars
  | Islander.Liar => ¬statement_is_true actual_liars

/-- The main theorem to prove -/
theorem knights_count :
  ∀ (knights : Nat),
    (knights ≤ total_islanders) →
    (∀ i : Fin total_islanders,
      is_consistent
        (if i.val < knights then Islander.Knight else Islander.Liar)
        (total_islanders - knights - 1)) →
    (knights = 0 ∨ knights = 2) :=
by sorry


end NUMINAMATH_CALUDE_knights_count_l2074_207477


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2074_207402

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₃ = -39 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2074_207402


namespace NUMINAMATH_CALUDE_sheet_width_calculation_l2074_207466

theorem sheet_width_calculation (paper_length : Real) (margin : Real) (picture_area : Real) :
  paper_length = 10 ∧ margin = 1.5 ∧ picture_area = 38.5 →
  ∃ (paper_width : Real), 
    paper_width = 8.5 ∧
    (paper_width - 2 * margin) * (paper_length - 2 * margin) = picture_area :=
by sorry

end NUMINAMATH_CALUDE_sheet_width_calculation_l2074_207466


namespace NUMINAMATH_CALUDE_mothers_day_rose_ratio_l2074_207411

/-- The number of roses Kyle picked last year -/
def last_year_roses : ℕ := 12

/-- The cost of one rose at the grocery store in dollars -/
def rose_cost : ℕ := 3

/-- The total amount Kyle spent on roses at the grocery store in dollars -/
def total_spent : ℕ := 54

/-- The ratio of roses in this year's bouquet to roses picked last year -/
def rose_ratio : Rat := 3 / 2

theorem mothers_day_rose_ratio :
  (total_spent / rose_cost : ℚ) / last_year_roses = rose_ratio :=
sorry

end NUMINAMATH_CALUDE_mothers_day_rose_ratio_l2074_207411


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_675_49_l2074_207418

/-- Two circles in a 2D plane -/
structure TwoCircles where
  center1 : ℝ × ℝ
  radius1 : ℝ
  center2 : ℝ × ℝ
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionDistanceSquared (c : TwoCircles) : ℝ := sorry

/-- The specific configuration of circles from the problem -/
def problemCircles : TwoCircles :=
  { center1 := (3, -1)
  , radius1 := 5
  , center2 := (3, 6)
  , radius2 := 3 }

/-- Theorem stating that the square of the distance between intersection points
    of the given circles is 675/49 -/
theorem intersection_distance_squared_is_675_49 :
  intersectionDistanceSquared problemCircles = 675 / 49 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_is_675_49_l2074_207418


namespace NUMINAMATH_CALUDE_popsicle_sticks_given_away_l2074_207459

/-- Given that Gino initially had 63.0 popsicle sticks and now has 13 left,
    prove that he gave away 50 popsicle sticks. -/
theorem popsicle_sticks_given_away 
  (initial_sticks : ℝ) 
  (remaining_sticks : ℕ) 
  (h1 : initial_sticks = 63.0)
  (h2 : remaining_sticks = 13) :
  initial_sticks - remaining_sticks = 50 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_given_away_l2074_207459


namespace NUMINAMATH_CALUDE_binary_111_equals_7_l2074_207476

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Nat) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + b * 2^i) 0

/-- The binary representation of the number we want to convert -/
def binary_111 : List Nat := [1, 1, 1]

/-- Theorem stating that the binary number 111 is equal to the decimal number 7 -/
theorem binary_111_equals_7 : binary_to_decimal binary_111 = 7 := by
  sorry

end NUMINAMATH_CALUDE_binary_111_equals_7_l2074_207476


namespace NUMINAMATH_CALUDE_two_books_from_three_genres_l2074_207493

/-- The number of ways to select 2 books of different genres from 3 genres with 4 books each -/
def select_two_books (num_genres : ℕ) (books_per_genre : ℕ) : ℕ :=
  let total_books := num_genres * books_per_genre
  let books_in_other_genres := (num_genres - 1) * books_per_genre
  (total_books * books_in_other_genres) / 2

/-- Theorem stating that selecting 2 books of different genres from 3 genres with 4 books each results in 48 possibilities -/
theorem two_books_from_three_genres : 
  select_two_books 3 4 = 48 := by
  sorry

#eval select_two_books 3 4

end NUMINAMATH_CALUDE_two_books_from_three_genres_l2074_207493


namespace NUMINAMATH_CALUDE_kiwi_count_l2074_207424

theorem kiwi_count (initial_oranges : ℕ) (added_kiwis : ℕ) (orange_percentage : ℚ) : 
  initial_oranges = 24 →
  added_kiwis = 26 →
  orange_percentage = 30 / 100 →
  ∃ initial_kiwis : ℕ, 
    (initial_oranges : ℚ) = orange_percentage * ((initial_oranges : ℚ) + (initial_kiwis : ℚ) + (added_kiwis : ℚ)) →
    initial_kiwis = 30 :=
by sorry

end NUMINAMATH_CALUDE_kiwi_count_l2074_207424


namespace NUMINAMATH_CALUDE_quadratic_equation_results_l2074_207419

theorem quadratic_equation_results (y : ℝ) (h : 6 * y^2 + 7 = 5 * y + 12) : 
  ((12 * y - 5)^2 = 145) ∧ 
  ((5 * y + 2)^2 = (4801 + 490 * Real.sqrt 145 + 3625) / 144 ∨
   (5 * y + 2)^2 = (4801 - 490 * Real.sqrt 145 + 3625) / 144) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_results_l2074_207419


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l2074_207469

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) → n ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l2074_207469


namespace NUMINAMATH_CALUDE_f_is_odd_and_piecewise_l2074_207443

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f defined piecewise -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x * (x + 2) else -x^2 + 2*x

theorem f_is_odd_and_piecewise :
  OddFunction f ∧ (∀ x < 0, f x = x * (x + 2)) → ∀ x > 0, f x = -x^2 + 2*x := by
  sorry

end NUMINAMATH_CALUDE_f_is_odd_and_piecewise_l2074_207443


namespace NUMINAMATH_CALUDE_circle_points_count_l2074_207484

def number_of_triangles (n : ℕ) : ℕ := n.choose 4

theorem circle_points_count : ∃ (n : ℕ), n > 3 ∧ number_of_triangles n = 126 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_circle_points_count_l2074_207484


namespace NUMINAMATH_CALUDE_polar_curve_arc_length_l2074_207453

noncomputable def arcLength (ρ : ℝ → ℝ) (φ₀ φ₁ : ℝ) : ℝ :=
  ∫ x in φ₀..φ₁, Real.sqrt ((ρ x)^2 + ((deriv ρ) x)^2)

theorem polar_curve_arc_length :
  let ρ : ℝ → ℝ := fun φ ↦ 2 * φ
  arcLength ρ 0 (3/4) = 15/8 + 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_polar_curve_arc_length_l2074_207453


namespace NUMINAMATH_CALUDE_cube_tank_volume_l2074_207498

/-- Represents the number of metal sheets required to make the cube-shaped tank -/
def required_sheets : ℝ := 74.99999999999997

/-- Represents the length of a metal sheet in meters -/
def sheet_length : ℝ := 4

/-- Represents the width of a metal sheet in meters -/
def sheet_width : ℝ := 2

/-- Represents the number of faces in a cube -/
def cube_faces : ℕ := 6

/-- Represents the conversion factor from cubic meters to liters -/
def cubic_meter_to_liter : ℝ := 1000

/-- Theorem stating that the volume of the cube-shaped tank is 1,000,000 liters -/
theorem cube_tank_volume :
  let sheet_area := sheet_length * sheet_width
  let sheets_per_face := required_sheets / cube_faces
  let face_area := sheets_per_face * sheet_area
  let side_length := Real.sqrt face_area
  let volume_cubic_meters := side_length ^ 3
  let volume_liters := volume_cubic_meters * cubic_meter_to_liter
  volume_liters = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cube_tank_volume_l2074_207498


namespace NUMINAMATH_CALUDE_pigeonhole_birthday_l2074_207482

theorem pigeonhole_birthday (n : ℕ) :
  (∀ f : Fin n → Fin 366, ∃ i j, i ≠ j ∧ f i = f j) ↔ n ≥ 367 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_birthday_l2074_207482


namespace NUMINAMATH_CALUDE_k_travel_time_l2074_207434

theorem k_travel_time (x : ℝ) 
  (h1 : x > 0) -- K's speed is positive
  (h2 : x - 0.5 > 0) -- M's speed is positive
  (h3 : 45 / (x - 0.5) - 45 / x = 3/4) -- K takes 45 minutes (3/4 hour) less than M
  : 45 / x = 9 := by
  sorry

end NUMINAMATH_CALUDE_k_travel_time_l2074_207434


namespace NUMINAMATH_CALUDE_donny_piggy_bank_l2074_207495

theorem donny_piggy_bank (initial_amount kite_cost frisbee_cost : ℕ) 
  (h1 : initial_amount = 78)
  (h2 : kite_cost = 8)
  (h3 : frisbee_cost = 9) :
  initial_amount - kite_cost - frisbee_cost = 61 := by
  sorry

end NUMINAMATH_CALUDE_donny_piggy_bank_l2074_207495


namespace NUMINAMATH_CALUDE_number_of_big_boxes_l2074_207426

theorem number_of_big_boxes (soaps_per_package : ℕ) (packages_per_box : ℕ) (total_soaps : ℕ) : 
  soaps_per_package = 192 →
  packages_per_box = 6 →
  total_soaps = 2304 →
  total_soaps / (soaps_per_package * packages_per_box) = 2 :=
by
  sorry

#check number_of_big_boxes

end NUMINAMATH_CALUDE_number_of_big_boxes_l2074_207426


namespace NUMINAMATH_CALUDE_distance_p_to_y_axis_l2074_207405

/-- The distance from a point to the y-axis is the absolute value of its x-coordinate. -/
def distance_to_y_axis (x y : ℝ) : ℝ := |x|

/-- Given point P(-3, 5), prove that its distance to the y-axis is 3. -/
theorem distance_p_to_y_axis :
  let P : ℝ × ℝ := (-3, 5)
  distance_to_y_axis P.1 P.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_p_to_y_axis_l2074_207405


namespace NUMINAMATH_CALUDE_line_length_difference_l2074_207436

/-- Conversion rate from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Length of the white line in inches -/
def white_line_inch : ℝ := 7.666666666666667

/-- Length of the blue line in inches -/
def blue_line_inch : ℝ := 3.3333333333333335

/-- Converts a length from inches to centimeters -/
def to_cm (inches : ℝ) : ℝ := inches * inch_to_cm

/-- The difference in length between the white and blue lines in centimeters -/
theorem line_length_difference : 
  to_cm white_line_inch - to_cm blue_line_inch = 11.005555555555553 := by
  sorry

end NUMINAMATH_CALUDE_line_length_difference_l2074_207436


namespace NUMINAMATH_CALUDE_B_power_93_l2074_207488

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -1; 0, 1, 0]

theorem B_power_93 : B^93 = B := by sorry

end NUMINAMATH_CALUDE_B_power_93_l2074_207488


namespace NUMINAMATH_CALUDE_max_roses_for_680_l2074_207439

/-- Represents the pricing options for roses -/
structure RosePricing where
  individual : ℚ  -- Price of a single rose
  oneDozen : ℚ    -- Price of one dozen roses
  twoDozen : ℚ    -- Price of two dozen roses

/-- Calculates the maximum number of roses that can be purchased with a given amount -/
def maxRoses (pricing : RosePricing) (amount : ℚ) : ℕ :=
  sorry

/-- Theorem: Given the specific pricing, the maximum number of roses for $680 is 325 -/
theorem max_roses_for_680 (pricing : RosePricing) 
  (h1 : pricing.individual = 230 / 100)
  (h2 : pricing.oneDozen = 36)
  (h3 : pricing.twoDozen = 50) :
  maxRoses pricing 680 = 325 :=
sorry

end NUMINAMATH_CALUDE_max_roses_for_680_l2074_207439


namespace NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l2074_207409

/-- The circumference of a circular track given two people walking in opposite directions -/
theorem track_circumference (speed1 speed2 : ℝ) (meeting_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let time_in_hours := meeting_time / 60
  let circumference := relative_speed * time_in_hours
  circumference

/-- The actual problem statement -/
theorem jogging_track_circumference : 
  ∃ (c : ℝ), abs (c - track_circumference 20 17 37) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_track_circumference_jogging_track_circumference_l2074_207409


namespace NUMINAMATH_CALUDE_balloon_distribution_l2074_207452

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 400) 
  (h2 : num_friends = 10) : 
  (total_balloons / num_friends) - ((total_balloons / num_friends) * 3 / 5) = 16 := by
  sorry

end NUMINAMATH_CALUDE_balloon_distribution_l2074_207452


namespace NUMINAMATH_CALUDE_halfway_distance_theorem_l2074_207431

def errand_distances : List ℕ := [10, 15, 5]

theorem halfway_distance_theorem (distances : List ℕ) (h : distances = errand_distances) :
  (distances.sum / 2 : ℕ) = 15 := by sorry

end NUMINAMATH_CALUDE_halfway_distance_theorem_l2074_207431


namespace NUMINAMATH_CALUDE_shifted_quadratic_roots_l2074_207430

theorem shifted_quadratic_roots
  (b c : ℝ)
  (h1 : ∃ x1 x2 : ℝ, x1 = 2 ∧ x2 = -3 ∧ ∀ x, x^2 + b*x + c = 0 ↔ x = x1 ∨ x = x2) :
  ∃ y1 y2 : ℝ, y1 = 6 ∧ y2 = 1 ∧ ∀ x, (x-4)^2 + b*(x-4) + c = 0 ↔ x = y1 ∨ x = y2 :=
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_roots_l2074_207430


namespace NUMINAMATH_CALUDE_complex_real_condition_l2074_207462

theorem complex_real_condition (a : ℝ) : 
  (((a : ℂ) + Complex.I) / (3 + 4 * Complex.I)).im = 0 → a = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_condition_l2074_207462


namespace NUMINAMATH_CALUDE_weight_of_BaF2_l2074_207441

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of fluorine in g/mol -/
def atomic_weight_F : ℝ := 19.00

/-- The number of moles of BaF2 -/
def moles_BaF2 : ℝ := 6

/-- The molecular weight of BaF2 in g/mol -/
def molecular_weight_BaF2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_F

/-- The weight of BaF2 in grams -/
def weight_BaF2 : ℝ := moles_BaF2 * molecular_weight_BaF2

theorem weight_of_BaF2 : weight_BaF2 = 1051.98 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_BaF2_l2074_207441


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2074_207490

-- Define the sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -1 < x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2074_207490


namespace NUMINAMATH_CALUDE_windows_installed_proof_l2074_207448

/-- Calculates the number of windows already installed given the total number of windows,
    time to install each window, and remaining installation time. -/
def windows_installed (total_windows : ℕ) (install_time_per_window : ℕ) (remaining_time : ℕ) : ℕ :=
  total_windows - (remaining_time / install_time_per_window)

/-- Proves that given the specific conditions, the number of windows already installed is 8. -/
theorem windows_installed_proof :
  windows_installed 14 8 48 = 8 := by
  sorry

end NUMINAMATH_CALUDE_windows_installed_proof_l2074_207448


namespace NUMINAMATH_CALUDE_count_even_factors_l2074_207451

def n : ℕ := 2^3 * 3^2 * 5

/-- The number of even positive factors of n -/
def num_even_factors (n : ℕ) : ℕ := sorry

theorem count_even_factors :
  num_even_factors n = 18 :=
sorry

end NUMINAMATH_CALUDE_count_even_factors_l2074_207451


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2074_207472

theorem trigonometric_problem (α : Real) 
  (h1 : 3 * Real.pi / 4 < α ∧ α < Real.pi) 
  (h2 : Real.tan α + 1 / Real.tan α = -10/3) : 
  Real.tan α = -1/3 ∧ 
  (Real.sin (Real.pi + α))^2 + 2 * Real.sin α * Real.sin (Real.pi/2 + α) + 1 / 
  (3 * Real.sin α * Real.cos (Real.pi/2 - α) - 2 * Real.cos α * Real.cos (Real.pi - α)) = 5/21 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2074_207472


namespace NUMINAMATH_CALUDE_book_arrangement_problem_l2074_207489

/-- The number of ways to arrange books on a shelf -/
def arrange_books (total : ℕ) (identical : ℕ) (different : ℕ) (adjacent_pair : ℕ) : ℕ :=
  (Nat.factorial (total - identical + 1 - adjacent_pair + 1) * Nat.factorial adjacent_pair) / 
  Nat.factorial identical

/-- Theorem stating the correct number of arrangements for the given problem -/
theorem book_arrangement_problem : 
  arrange_books 7 3 4 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_problem_l2074_207489


namespace NUMINAMATH_CALUDE_a5_b5_ratio_l2074_207403

def S (n : ℕ+) : ℝ := sorry
def T (n : ℕ+) : ℝ := sorry
def a : ℕ+ → ℝ := sorry
def b : ℕ+ → ℝ := sorry

axiom arithmetic_sum_property (n : ℕ+) : S n / T n = (n + 1) / (2 * n - 1)

theorem a5_b5_ratio : a 5 / b 5 = 10 / 17 := by
  sorry

end NUMINAMATH_CALUDE_a5_b5_ratio_l2074_207403


namespace NUMINAMATH_CALUDE_total_tax_deduction_in_cents_l2074_207463

-- Define the hourly wage in dollars
def hourly_wage : ℝ := 25

-- Define the local tax rate as a percentage
def local_tax_rate : ℝ := 2

-- Define the state tax rate as a percentage
def state_tax_rate : ℝ := 0.5

-- Define the conversion rate from dollars to cents
def dollars_to_cents : ℝ := 100

-- Theorem statement
theorem total_tax_deduction_in_cents :
  (hourly_wage * dollars_to_cents) * (local_tax_rate / 100 + state_tax_rate / 100) = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_total_tax_deduction_in_cents_l2074_207463


namespace NUMINAMATH_CALUDE_range_of_function_l2074_207408

theorem range_of_function (k : ℝ) (h : k > 0) :
  let f : ℝ → ℝ := fun x ↦ 3 * x^k
  Set.range (fun x ↦ f x) = Set.Ici (3 * 2^k) := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l2074_207408


namespace NUMINAMATH_CALUDE_concert_tickets_sold_l2074_207475

theorem concert_tickets_sold (T : ℕ) : 
  (3 / 4 : ℚ) * T + (5 / 9 : ℚ) * (1 / 4 : ℚ) * T + 80 + 20 = T → T = 900 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_sold_l2074_207475


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2074_207464

theorem root_difference_implies_k_value (k : ℝ) : 
  (∀ r s : ℝ, r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0 → 
    (r+3)^2 - k*(r+3) + 12 = 0 ∧ (s+3)^2 - k*(s+3) + 12 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2074_207464


namespace NUMINAMATH_CALUDE_different_ball_counts_l2074_207480

/-- Represents a box in the game -/
structure Box :=
  (id : Nat)

/-- Represents a pair of boxes -/
structure BoxPair :=
  (box1 : Box)
  (box2 : Box)

/-- The game state -/
structure GameState :=
  (boxes : Finset Box)
  (pairs : Finset BoxPair)
  (ballCount : Box → Nat)

/-- The theorem statement -/
theorem different_ball_counts (n : Nat) (h : n = 2018) :
  ∃ (finalState : GameState),
    finalState.boxes.card = n ∧
    finalState.pairs.card = 2 * n - 2 ∧
    ∀ (b1 b2 : Box), b1 ∈ finalState.boxes → b2 ∈ finalState.boxes → b1 ≠ b2 →
      finalState.ballCount b1 ≠ finalState.ballCount b2 :=
by sorry

end NUMINAMATH_CALUDE_different_ball_counts_l2074_207480


namespace NUMINAMATH_CALUDE_second_grade_survey_size_l2074_207412

/-- Represents a school with three grades and a stratified sampling plan. -/
structure School where
  total_students : ℕ
  grade_ratio : Fin 3 → ℕ
  survey_size : ℕ

/-- Calculates the number of students to be surveyed from a specific grade. -/
def students_surveyed_in_grade (school : School) (grade : Fin 3) : ℕ :=
  (school.survey_size * school.grade_ratio grade) / (school.grade_ratio 0 + school.grade_ratio 1 + school.grade_ratio 2)

/-- The main theorem stating that 50 second-grade students should be surveyed. -/
theorem second_grade_survey_size (school : School) 
  (h1 : school.total_students = 1500)
  (h2 : school.grade_ratio 0 = 4)
  (h3 : school.grade_ratio 1 = 5)
  (h4 : school.grade_ratio 2 = 6)
  (h5 : school.survey_size = 150) :
  students_surveyed_in_grade school 1 = 50 := by
  sorry


end NUMINAMATH_CALUDE_second_grade_survey_size_l2074_207412


namespace NUMINAMATH_CALUDE_max_distance_to_point_l2074_207487

/-- The maximum distance from a point on the curve y = √(2 - x^2) to (0, -1) -/
theorem max_distance_to_point (x : ℝ) : 
  let y : ℝ := Real.sqrt (2 - x^2)
  let d : ℝ := Real.sqrt (x^2 + (y + 1)^2)
  d ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_point_l2074_207487


namespace NUMINAMATH_CALUDE_malingerers_exposed_l2074_207423

/-- Represents a five-digit number where each digit is represented by a letter --/
structure CryptarithmNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a_digit : a < 10
  h_b_digit : b < 10
  h_c_digit : c < 10
  h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

def draftees (n : CryptarithmNumber) : Nat :=
  10000 * n.a + 1000 * n.a + 100 * n.b + 10 * n.b + n.b

def malingerers (n : CryptarithmNumber) : Nat :=
  10000 * n.a + 1000 * n.b + 100 * n.c + 10 * n.c + n.c

theorem malingerers_exposed (n : CryptarithmNumber) :
  draftees n - 1 = malingerers n → malingerers n = 10999 := by
  sorry

#check malingerers_exposed

end NUMINAMATH_CALUDE_malingerers_exposed_l2074_207423


namespace NUMINAMATH_CALUDE_probability_of_specific_draw_l2074_207468

def total_silverware : ℕ := 24
def forks : ℕ := 8
def spoons : ℕ := 8
def knives : ℕ := 8
def pieces_drawn : ℕ := 4

def favorable_outcomes : ℕ := forks * spoons * knives * (forks - 1 + spoons - 1 + knives - 1)
def total_outcomes : ℕ := Nat.choose total_silverware pieces_drawn

theorem probability_of_specific_draw :
  (favorable_outcomes : ℚ) / total_outcomes = 214 / 253 := by sorry

end NUMINAMATH_CALUDE_probability_of_specific_draw_l2074_207468


namespace NUMINAMATH_CALUDE_hot_dog_buns_per_package_l2074_207438

/-- Proves that the number of hot dog buns in one package is 8, given the conditions of the problem -/
theorem hot_dog_buns_per_package 
  (total_packages : ℕ) 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (buns_per_student : ℕ) 
  (h1 : total_packages = 30)
  (h2 : num_classes = 4)
  (h3 : students_per_class = 30)
  (h4 : buns_per_student = 2) : 
  (num_classes * students_per_class * buns_per_student) / total_packages = 8 := by
  sorry

#check hot_dog_buns_per_package

end NUMINAMATH_CALUDE_hot_dog_buns_per_package_l2074_207438


namespace NUMINAMATH_CALUDE_distance_point_to_line_l2074_207435

/-- The distance from a point to a line in 3D space --/
def distancePointToLine (p : ℝ × ℝ × ℝ) (l1 l2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_point_to_line :
  let p := (2, -2, 3)
  let l1 := (1, 3, -1)
  let l2 := (0, 0, 2)
  distancePointToLine p l1 l2 = Real.sqrt 2750 / 19 := by
  sorry

end NUMINAMATH_CALUDE_distance_point_to_line_l2074_207435


namespace NUMINAMATH_CALUDE_sin_neg_pi_third_l2074_207483

theorem sin_neg_pi_third : Real.sin (-π / 3) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_neg_pi_third_l2074_207483


namespace NUMINAMATH_CALUDE_faye_halloween_candy_l2074_207497

/-- Represents the number of candy pieces Faye scored on Halloween. -/
def initial_candy : ℕ := 47

/-- Represents the number of candy pieces Faye ate on the first night. -/
def eaten_candy : ℕ := 25

/-- Represents the number of candy pieces Faye's sister gave her. -/
def received_candy : ℕ := 40

/-- Represents the number of candy pieces Faye has now. -/
def current_candy : ℕ := 62

theorem faye_halloween_candy : 
  initial_candy - eaten_candy + received_candy = current_candy := by
  sorry

end NUMINAMATH_CALUDE_faye_halloween_candy_l2074_207497


namespace NUMINAMATH_CALUDE_min_sin_cos_expression_l2074_207454

theorem min_sin_cos_expression (A : Real) : 
  let f := λ x : Real => Real.sin (x / 2) - Real.sqrt 3 * Real.cos (x / 2)
  ∃ m : Real, (∀ x, f x ≥ m) ∧ f (-π/3) = m :=
sorry

end NUMINAMATH_CALUDE_min_sin_cos_expression_l2074_207454


namespace NUMINAMATH_CALUDE_max_winning_pieces_l2074_207446

/-- Represents the game board -/
def Board := Fin 1000 → Option Nat

/-- The maximum number of pieces a player can place in one turn -/
def max_placement : Nat := 17

/-- Checks if a series of pieces is consecutive -/
def is_consecutive (b : Board) (start finish : Fin 1000) : Prop :=
  ∀ i : Fin 1000, start ≤ i ∧ i ≤ finish → b i.val ≠ none

/-- Represents a valid move by the first player -/
def valid_first_move (b1 b2 : Board) : Prop :=
  ∃ placed : Nat, placed ≤ max_placement ∧
    (∀ i : Fin 1000, b1 i = none → b2 i = none ∨ (∃ n : Nat, b2 i = some n)) ∧
    (∀ i : Fin 1000, b1 i ≠ none → b2 i = b1 i)

/-- Represents a valid move by the second player -/
def valid_second_move (b1 b2 : Board) : Prop :=
  ∃ start finish : Fin 1000, start ≤ finish ∧ is_consecutive b1 start finish ∧
    (∀ i : Fin 1000, (i < start ∨ finish < i) → b2 i = b1 i) ∧
    (∀ i : Fin 1000, start ≤ i ∧ i ≤ finish → b2 i = none)

/-- Checks if the first player has won -/
def first_player_wins (b : Board) (n : Nat) : Prop :=
  ∃ start finish : Fin 1000, start ≤ finish ∧ 
    is_consecutive b start finish ∧
    (∀ i : Fin 1000, i < start ∨ finish < i → b i = none) ∧
    (finish - start + 1 : Nat) = n

/-- The main theorem stating that 98 is the maximum number of pieces for which
    the first player can always win -/
theorem max_winning_pieces : 
  (∀ n : Nat, n ≤ 98 → 
    ∀ initial : Board, (∀ i : Fin 1000, initial i = none) → 
      ∃ strategy : Nat → Board → Board,
        ∀ opponent_strategy : Board → Board,
          ∃ final : Board, first_player_wins final n) ∧
  ¬(∀ n : Nat, n ≤ 99 → 
    ∀ initial : Board, (∀ i : Fin 1000, initial i = none) → 
      ∃ strategy : Nat → Board → Board,
        ∀ opponent_strategy : Board → Board,
          ∃ final : Board, first_player_wins final n) :=
sorry

end NUMINAMATH_CALUDE_max_winning_pieces_l2074_207446


namespace NUMINAMATH_CALUDE_rectangle_area_l2074_207465

theorem rectangle_area (x : ℝ) (h1 : (x + 5) * (2 * (x + 10)) = 3 * x * (x + 10)) (h2 : x > 0) :
  x * (x + 10) = 200 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2074_207465


namespace NUMINAMATH_CALUDE_new_rectangle_area_l2074_207400

theorem new_rectangle_area (a b : ℝ) (h : 0 < a ∧ a < b) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_base := 2 * (d + b)
  let new_height := (d - b) / 2
  new_base * new_height = a^2 := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_l2074_207400


namespace NUMINAMATH_CALUDE_board_tiling_condition_l2074_207421

/-- Represents a tile shape -/
inductive TileShape
  | L  -- L-shaped tile covering 3 squares
  | T  -- T-shaped tile covering 4 squares

/-- Represents a tiling of an n × n board -/
def Tiling (n : ℕ) := List (TileShape × Fin n × Fin n)

/-- Checks if a tiling is valid for an n × n board -/
def is_valid_tiling (n : ℕ) (tiling : Tiling n) : Prop :=
  -- Each square is covered exactly once
  -- No tile extends beyond the board
  -- The tiling uses only L and T shapes
  sorry

theorem board_tiling_condition (n : ℕ) :
  (∃ (tiling : Tiling n), is_valid_tiling n tiling) ↔ 
  (4 ∣ n ∧ n > 4) :=
sorry

end NUMINAMATH_CALUDE_board_tiling_condition_l2074_207421


namespace NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2074_207485

theorem rahul_deepak_age_ratio : 
  ∀ (rahul_age deepak_age : ℕ),
    rahul_age + 6 = 18 →
    deepak_age = 9 →
    (rahul_age : ℚ) / deepak_age = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rahul_deepak_age_ratio_l2074_207485


namespace NUMINAMATH_CALUDE_sanchez_rope_purchase_sanchez_rope_purchase_l2074_207440

theorem sanchez_rope_purchase (inches_per_foot : ℕ) (this_week_inches : ℕ) : ℕ :=
  let last_week_feet := (this_week_inches / inches_per_foot) + 4
  last_week_feet

#check sanchez_rope_purchase 12 96 = 12

/- Proof
theorem sanchez_rope_purchase (inches_per_foot : ℕ) (this_week_inches : ℕ) : ℕ :=
  let last_week_feet := (this_week_inches / inches_per_foot) + 4
  last_week_feet
sorry
-/

end NUMINAMATH_CALUDE_sanchez_rope_purchase_sanchez_rope_purchase_l2074_207440


namespace NUMINAMATH_CALUDE_no_real_solutions_ratio_equation_l2074_207447

theorem no_real_solutions_ratio_equation :
  ∀ x : ℝ, (x + 3) / (2 * x + 5) ≠ (5 * x + 4) / (8 * x + 6) :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_ratio_equation_l2074_207447


namespace NUMINAMATH_CALUDE_tangent_line_slope_intersecting_line_equation_l2074_207470

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 - 4*x + y^2 + 3 = 0

-- Define points P and Q
def P : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (0, -2)

-- Define the condition for the slopes of OA and OB
def slope_condition (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1/7

-- Statement for part (1)
theorem tangent_line_slope :
  ∃ m : ℝ, (m = 0 ∨ m = -4/3) ∧
  ∀ x y : ℝ, y = m * x + P.2 →
  (∃! t : ℝ, circle_C t (m * t + P.2)) :=
sorry

-- Statement for part (2)
theorem intersecting_line_equation :
  ∃ k : ℝ, (k = 1 ∨ k = 5/3) ∧
  ∀ x y : ℝ, y = k * x + Q.2 →
  (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    A.2 = k * A.1 + Q.2 ∧
    B.2 = k * B.1 + Q.2 ∧
    slope_condition (A.2 / A.1) (B.2 / B.1)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_intersecting_line_equation_l2074_207470


namespace NUMINAMATH_CALUDE_student_count_l2074_207442

theorem student_count (avg_age_students : ℝ) (teacher_age : ℕ) (new_avg_age : ℝ)
  (h1 : avg_age_students = 14)
  (h2 : teacher_age = 65)
  (h3 : new_avg_age = 15) :
  ∃ n : ℕ, n * avg_age_students + teacher_age = (n + 1) * new_avg_age ∧ n = 50 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l2074_207442


namespace NUMINAMATH_CALUDE_diagonal_intersections_12x17_l2074_207457

/-- Counts the number of intersection points between a diagonal and grid lines in an m × n grid -/
def countIntersections (m n : ℕ) : ℕ :=
  (n + 1) + (m + 1) - 2

/-- Theorem: In a 12 × 17 grid, the diagonal from A to B intersects the grid at 29 points -/
theorem diagonal_intersections_12x17 :
  countIntersections 12 17 = 29 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersections_12x17_l2074_207457


namespace NUMINAMATH_CALUDE_edith_book_ratio_l2074_207432

/-- Given that Edith has 80 novels on her schoolbook shelf and a total of 240 books (novels and writing books combined), 
    prove that the ratio of novels on the shelf to writing books in the suitcase is 1:2. -/
theorem edith_book_ratio :
  let novels_on_shelf : ℕ := 80
  let total_books : ℕ := 240
  let writing_books : ℕ := total_books - novels_on_shelf
  novels_on_shelf * 2 = writing_books := by
  sorry

end NUMINAMATH_CALUDE_edith_book_ratio_l2074_207432


namespace NUMINAMATH_CALUDE_fine_payment_l2074_207445

theorem fine_payment (F : ℚ) 
  (joe_payment : ℚ) (peter_payment : ℚ) (sam_payment : ℚ)
  (h1 : joe_payment = F / 4 + 7)
  (h2 : peter_payment = F / 3 - 7)
  (h3 : sam_payment = F / 2 - 12)
  (h4 : joe_payment + peter_payment + sam_payment = F) :
  sam_payment / F = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fine_payment_l2074_207445


namespace NUMINAMATH_CALUDE_ming_ladybugs_l2074_207481

/-- The number of spiders Sami found -/
def spiders : ℕ := 3

/-- The number of ants Hunter saw -/
def ants : ℕ := 12

/-- The number of ladybugs that flew away -/
def flown_ladybugs : ℕ := 2

/-- The number of insects remaining in the playground -/
def remaining_insects : ℕ := 21

/-- The number of ladybugs Ming discovered initially -/
def initial_ladybugs : ℕ := remaining_insects + flown_ladybugs - (spiders + ants)

theorem ming_ladybugs : initial_ladybugs = 8 := by
  sorry

end NUMINAMATH_CALUDE_ming_ladybugs_l2074_207481


namespace NUMINAMATH_CALUDE_quotient_problem_l2074_207471

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 158)
  (h2 : divisor = 17)
  (h3 : remainder = 5)
  (h4 : dividend = quotient * divisor + remainder) :
  quotient = 9 := by
  sorry

end NUMINAMATH_CALUDE_quotient_problem_l2074_207471


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l2074_207458

/-- The minimum distance from a point on the parabola y = x^2 to the line 2x - y - 11 = 0 is 2√5 -/
theorem min_distance_parabola_to_line : 
  let parabola := {(x, y) : ℝ × ℝ | y = x^2}
  let line := {(x, y) : ℝ × ℝ | 2*x - y - 11 = 0}
  ∃ d : ℝ, d = 2 * Real.sqrt 5 ∧ 
    (∀ p ∈ parabola, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ parabola, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l2074_207458


namespace NUMINAMATH_CALUDE_mean_temperature_l2074_207401

def temperatures : List ℤ := [-7, -4, -4, -5, 1, 3, 2]

theorem mean_temperature :
  (temperatures.sum : ℚ) / temperatures.length = -2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l2074_207401


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2074_207492

-- Define sets A and B
def A : Set ℝ := {x | x > 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Define the theorem
theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, x ∈ A → x ∈ B a) ∧ (∃ x, x ∈ B a ∧ x ∉ A) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2074_207492


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2074_207494

-- System (1)
theorem system_one_solution (x y : ℚ) : 
  (3 * x - 6 * y = 4 ∧ x + 5 * y = 6) ↔ (x = 8/3 ∧ y = 2/3) := by sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  (x/4 + y/3 = 3 ∧ 3*(x-4) - 2*(y-1) = -1) ↔ (x = 6 ∧ y = 9/2) := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l2074_207494


namespace NUMINAMATH_CALUDE_smaller_rectangle_area_percentage_l2074_207407

/-- A circle with a rectangle inscribed in it -/
structure InscribedRectangle where
  center : ℝ × ℝ
  radius : ℝ
  rect_length : ℝ
  rect_width : ℝ

/-- A smaller rectangle with one side coinciding with the larger rectangle and two vertices on the circle -/
structure SmallerRectangle where
  length : ℝ
  width : ℝ

/-- The configuration of the inscribed rectangle and the smaller rectangle -/
structure Configuration where
  inscribed : InscribedRectangle
  smaller : SmallerRectangle

/-- The theorem stating that the area of the smaller rectangle is 0% of the area of the larger rectangle -/
theorem smaller_rectangle_area_percentage (config : Configuration) : 
  (config.smaller.length * config.smaller.width) / (config.inscribed.rect_length * config.inscribed.rect_width) = 0 := by
  sorry

end NUMINAMATH_CALUDE_smaller_rectangle_area_percentage_l2074_207407


namespace NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l2074_207444

theorem cube_surface_area_equal_volume (a b c : ℝ) (h1 : a = 12) (h2 : b = 4) (h3 : c = 18) :
  let prism_volume := a * b * c
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = 864 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equal_volume_l2074_207444


namespace NUMINAMATH_CALUDE_hindi_speakers_count_l2074_207404

/-- Represents the number of children who can speak a given language or combination of languages -/
structure LanguageCount where
  total : ℕ
  onlyEnglish : ℕ
  onlyHindi : ℕ
  onlySpanish : ℕ
  englishAndHindi : ℕ
  englishAndSpanish : ℕ
  hindiAndSpanish : ℕ
  allThree : ℕ

/-- Calculates the number of children who can speak Hindi -/
def hindiSpeakers (c : LanguageCount) : ℕ :=
  c.onlyHindi + c.englishAndHindi + c.hindiAndSpanish + c.allThree

/-- Theorem stating that the number of Hindi speakers is 45 given the conditions -/
theorem hindi_speakers_count (c : LanguageCount)
  (h_total : c.total = 90)
  (h_onlyEnglish : c.onlyEnglish = 90 * 25 / 100)
  (h_onlyHindi : c.onlyHindi = 90 * 15 / 100)
  (h_onlySpanish : c.onlySpanish = 90 * 10 / 100)
  (h_englishAndHindi : c.englishAndHindi = 90 * 20 / 100)
  (h_englishAndSpanish : c.englishAndSpanish = 90 * 15 / 100)
  (h_hindiAndSpanish : c.hindiAndSpanish = 90 * 10 / 100)
  (h_allThree : c.allThree = 90 * 5 / 100) :
  hindiSpeakers c = 45 := by
  sorry


end NUMINAMATH_CALUDE_hindi_speakers_count_l2074_207404


namespace NUMINAMATH_CALUDE_circle_equation_and_k_value_l2074_207420

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (0, 2)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the line l: y = kx + 1
def line_l (k : ℝ) (p : ℝ × ℝ) : Prop := p.2 = k * p.1 + 1

-- Define the dot product of vectors OP and OQ
def dot_product_OP_OQ (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

theorem circle_equation_and_k_value :
  ∃ (center : ℝ × ℝ) (P Q : ℝ × ℝ) (k : ℝ),
    point_A ∈ circle_C ∧
    point_B ∈ circle_C ∧
    line_y_eq_x center ∧
    line_l k P ∧
    line_l k Q ∧
    P ∈ circle_C ∧
    Q ∈ circle_C ∧
    dot_product_OP_OQ P Q = -2 →
    (∀ p : ℝ × ℝ, p ∈ circle_C ↔ p.1^2 + p.2^2 = 4) ∧
    k = 0 := by sorry

end NUMINAMATH_CALUDE_circle_equation_and_k_value_l2074_207420


namespace NUMINAMATH_CALUDE_james_writing_time_l2074_207415

/-- James' writing scenario -/
structure WritingScenario where
  pages_per_hour : ℕ
  pages_per_day_per_person : ℕ
  people_per_day : ℕ

/-- Calculate the hours spent writing per week -/
def hours_per_week (s : WritingScenario) : ℕ :=
  let pages_per_day := s.pages_per_day_per_person * s.people_per_day
  let pages_per_week := pages_per_day * 7
  pages_per_week / s.pages_per_hour

/-- Theorem: James spends 7 hours a week writing -/
theorem james_writing_time :
  let james := WritingScenario.mk 10 5 2
  hours_per_week james = 7 := by
  sorry

end NUMINAMATH_CALUDE_james_writing_time_l2074_207415


namespace NUMINAMATH_CALUDE_smallest_candy_count_l2074_207416

theorem smallest_candy_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  n = 110 ∧
  ∀ (m : ℕ), (m ≥ 100 ∧ m ≤ 999) → 
    (m + 7) % 9 = 0 → (m - 9) % 7 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l2074_207416


namespace NUMINAMATH_CALUDE_equation_solution_l2074_207427

theorem equation_solution :
  ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ (x = -14) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2074_207427


namespace NUMINAMATH_CALUDE_option_a_correct_option_b_correct_option_c_correct_option_d_incorrect_l2074_207410

-- Define variables
variable (a b c : ℝ)

-- Theorem for Option A
theorem option_a_correct : a = b → a + 6 = b + 6 := by sorry

-- Theorem for Option B
theorem option_b_correct : a = b → a / 9 = b / 9 := by sorry

-- Theorem for Option C
theorem option_c_correct (h : c ≠ 0) : a / c = b / c → a = b := by sorry

-- Theorem for Option D (incorrect transformation)
theorem option_d_incorrect : ∃ a b : ℝ, -2 * a = -2 * b ∧ a ≠ -b := by sorry

end NUMINAMATH_CALUDE_option_a_correct_option_b_correct_option_c_correct_option_d_incorrect_l2074_207410


namespace NUMINAMATH_CALUDE_max_colors_theorem_l2074_207461

/-- Represents a color configuration of an n × n × n cube -/
structure ColorConfig (n : ℕ) where
  colors : Fin n → Fin n → Fin n → ℕ

/-- Represents a set of colors in an n × n × 1 box -/
def ColorSet (n : ℕ) := Set ℕ

/-- Returns the set of colors in an n × n × 1 box for a given configuration and orientation -/
def getColorSet (n : ℕ) (config : ColorConfig n) (orientation : Fin 3) (i : Fin n) : ColorSet n :=
  sorry

/-- Checks if the color configuration satisfies the problem conditions -/
def validConfig (n : ℕ) (config : ColorConfig n) : Prop :=
  ∀ (o1 o2 o3 : Fin 3) (i j : Fin n),
    o1 ≠ o2 ∧ o2 ≠ o3 ∧ o1 ≠ o3 →
    ∃ (k l : Fin n), 
      getColorSet n config o1 i = getColorSet n config o2 k ∧
      getColorSet n config o1 i = getColorSet n config o3 l

/-- The maximal number of colors in a valid configuration -/
def maxColors (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1) / 6

theorem max_colors_theorem (n : ℕ) (h : n > 1) :
  ∃ (config : ColorConfig n),
    validConfig n config ∧
    (∀ (config' : ColorConfig n), validConfig n config' →
      Finset.card (Finset.image (config.colors) Finset.univ) ≥
      Finset.card (Finset.image (config'.colors) Finset.univ)) ∧
    Finset.card (Finset.image (config.colors) Finset.univ) = maxColors n :=
  sorry

end NUMINAMATH_CALUDE_max_colors_theorem_l2074_207461


namespace NUMINAMATH_CALUDE_max_value_inequality_l2074_207437

theorem max_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^3 * (b + c)^3) ≤ 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2074_207437


namespace NUMINAMATH_CALUDE_circle_angle_measure_l2074_207428

noncomputable def Circle := ℝ × ℝ → Prop

def diameter (c : Circle) (A B : ℝ × ℝ) : Prop := sorry

def parallel (A B C D : ℝ × ℝ) : Prop := sorry

def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem circle_angle_measure 
  (c : Circle) (A B C D E : ℝ × ℝ) :
  diameter c E B →
  parallel E B D C →
  parallel A B E C →
  angle A E B = (3/7) * Real.pi →
  angle A B E = (4/7) * Real.pi →
  angle B D C = (900/7) * (Real.pi/180) :=
by sorry

end NUMINAMATH_CALUDE_circle_angle_measure_l2074_207428


namespace NUMINAMATH_CALUDE_unique_solution_for_2n_plus_m_l2074_207417

theorem unique_solution_for_2n_plus_m : 
  ∀ n m : ℤ, 
    (3 * n - m < 5) → 
    (n + m > 26) → 
    (3 * m - 2 * n < 46) → 
    (2 * n + m = 36) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_2n_plus_m_l2074_207417


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l2074_207455

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_17_l2074_207455


namespace NUMINAMATH_CALUDE_four_digit_permutations_l2074_207478

/-- The number of distinct permutations of a multiset with repeated elements -/
def multinomial_coefficient (n : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial n / (repetitions.map Nat.factorial).prod

/-- The multiset representation of the given digits -/
def digit_multiset : List ℕ := [3, 3, 3, 5]

/-- The total number of digits -/
def total_digits : ℕ := digit_multiset.length

/-- The list of repetitions for each unique digit -/
def repetitions : List ℕ := [3, 1]

theorem four_digit_permutations :
  multinomial_coefficient total_digits repetitions = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_permutations_l2074_207478


namespace NUMINAMATH_CALUDE_waiter_tip_problem_l2074_207473

theorem waiter_tip_problem (total_customers : ℕ) (tip_amount : ℕ) (total_tips : ℕ) 
  (h1 : total_customers = 10)
  (h2 : tip_amount = 3)
  (h3 : total_tips = 15) :
  total_customers - (total_tips / tip_amount) = 5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tip_problem_l2074_207473


namespace NUMINAMATH_CALUDE_prob_non_expired_single_draw_prob_expired_two_draws_l2074_207406

/-- Represents the total number of bottles --/
def total_bottles : ℕ := 6

/-- Represents the number of expired bottles --/
def expired_bottles : ℕ := 2

/-- Represents the number of non-expired bottles --/
def non_expired_bottles : ℕ := total_bottles - expired_bottles

/-- Theorem for the probability of drawing a non-expired bottle in a single draw --/
theorem prob_non_expired_single_draw : 
  (non_expired_bottles : ℚ) / total_bottles = 2 / 3 := by sorry

/-- Theorem for the probability of drawing at least one expired bottle in two draws --/
theorem prob_expired_two_draws : 
  1 - (non_expired_bottles * (non_expired_bottles - 1) : ℚ) / (total_bottles * (total_bottles - 1)) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_prob_non_expired_single_draw_prob_expired_two_draws_l2074_207406


namespace NUMINAMATH_CALUDE_lee_cookies_l2074_207460

/-- Given that Lee can make 24 cookies with 4 cups of flour, 
    this function calculates how many cookies he can make with any amount of flour. -/
def cookies_from_flour (flour : ℚ) : ℚ :=
  (24 / 4) * flour

/-- Theorem stating that Lee can make 36 cookies with 6 cups of flour. -/
theorem lee_cookies : cookies_from_flour 6 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_l2074_207460


namespace NUMINAMATH_CALUDE_product_of_integers_l2074_207456

theorem product_of_integers (x y : ℤ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 34) : x * y = 15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_integers_l2074_207456


namespace NUMINAMATH_CALUDE_fermat_like_theorem_l2074_207496

theorem fermat_like_theorem (k : ℕ) : ¬ ∃ (x y z : ℤ), 
  (x^k + y^k = z^k) ∧ (z > 0) ∧ (0 < x) ∧ (x < k) ∧ (0 < y) ∧ (y < k) := by
  sorry

end NUMINAMATH_CALUDE_fermat_like_theorem_l2074_207496


namespace NUMINAMATH_CALUDE_value_of_expression_constant_difference_implies_b_value_l2074_207499

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := 2*a^2 + 3*a*b - 2*a - 1

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := a^2 + a*b - 1

/-- Theorem 1: The value of 4A - (3A - 2B) -/
theorem value_of_expression (a b : ℝ) :
  4 * A a b - (3 * A a b - 2 * B a b) = 4*a^2 + 5*a*b - 2*a - 3 := by sorry

/-- Theorem 2: When A - 2B is constant for all a, b must equal 2 -/
theorem constant_difference_implies_b_value (b : ℝ) :
  (∀ a : ℝ, ∃ k : ℝ, A a b - 2 * B a b = k) → b = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_expression_constant_difference_implies_b_value_l2074_207499


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2074_207491

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l2074_207491


namespace NUMINAMATH_CALUDE_highest_points_fewer_wins_l2074_207422

/-- Represents a football team in the tournament -/
structure Team :=
  (id : Nat)
  (wins : Nat)
  (draws : Nat)
  (losses : Nat)

/-- Calculates the points for a team based on their wins and draws -/
def points (t : Team) : Nat :=
  3 * t.wins + t.draws

/-- Represents the tournament results -/
structure TournamentResult :=
  (teams : Finset Team)
  (team_count : Nat)
  (hteam_count : teams.card = team_count)

/-- Theorem stating that it's possible for a team to have the highest points but fewer wins -/
theorem highest_points_fewer_wins (tr : TournamentResult) 
  (h_six_teams : tr.team_count = 6) : 
  ∃ (t1 t2 : Team), t1 ∈ tr.teams ∧ t2 ∈ tr.teams ∧ 
    (∀ t ∈ tr.teams, points t1 ≥ points t) ∧
    t1.wins < t2.wins :=
  sorry

end NUMINAMATH_CALUDE_highest_points_fewer_wins_l2074_207422


namespace NUMINAMATH_CALUDE_matthew_water_bottle_fills_l2074_207414

/-- Represents the number of times Matthew needs to fill his water bottle per week -/
def fill_times_per_week (glasses_per_day : ℕ) (ounces_per_glass : ℕ) (bottle_size : ℕ) : ℕ :=
  (7 * glasses_per_day * ounces_per_glass) / bottle_size

/-- Proves that Matthew will fill his water bottle 4 times per week -/
theorem matthew_water_bottle_fills :
  fill_times_per_week 4 5 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_matthew_water_bottle_fills_l2074_207414


namespace NUMINAMATH_CALUDE_car_speed_equality_l2074_207433

/-- Prove that given the conditions of the car problem, the average speed of Car Y is equal to the average speed of Car X. -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_starts : ℝ)
  (h1 : speed_x = 35)
  (h2 : start_delay = 72 / 60)
  (h3 : distance_after_y_starts = 105) :
  ∃ (speed_y : ℝ), speed_y = speed_x := by
  sorry

end NUMINAMATH_CALUDE_car_speed_equality_l2074_207433


namespace NUMINAMATH_CALUDE_symmetry_implies_extremum_l2074_207449

/-- Given a function f(x) = 2sin(wx + φ) that is symmetric about x = π/6,
    prove that f(π/6) is either -2 or 2. -/
theorem symmetry_implies_extremum (w φ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (w * x + φ)
  (∀ x, f (π/6 + x) = f (π/6 - x)) →
  (f (π/6) = -2 ∨ f (π/6) = 2) :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_extremum_l2074_207449


namespace NUMINAMATH_CALUDE_division_fifteen_by_negative_five_l2074_207479

theorem division_fifteen_by_negative_five : (15 : ℤ) / (-5 : ℤ) = -3 := by sorry

end NUMINAMATH_CALUDE_division_fifteen_by_negative_five_l2074_207479


namespace NUMINAMATH_CALUDE_parabola_vertex_l2074_207413

/-- A parabola defined by y = x^2 - 2ax + b passing through (1, 1) and intersecting the x-axis at only one point -/
structure Parabola where
  a : ℝ
  b : ℝ
  point_condition : 1 = 1^2 - 2*a*1 + b
  single_intersection : ∃! x, x^2 - 2*a*x + b = 0

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.a, p.a^2 - p.b)

theorem parabola_vertex (p : Parabola) : vertex p = (0, 0) ∨ vertex p = (2, 0) := by
  sorry


end NUMINAMATH_CALUDE_parabola_vertex_l2074_207413


namespace NUMINAMATH_CALUDE_table_length_is_77_l2074_207429

/-- Represents the dimensions and placement of sheets on a table. -/
structure TableSetup where
  tableWidth : ℕ
  tableLength : ℕ
  sheetWidth : ℕ
  sheetHeight : ℕ
  sheetCount : ℕ

/-- Checks if the given setup satisfies the conditions of the problem. -/
def isValidSetup (setup : TableSetup) : Prop :=
  setup.tableWidth = 80 ∧
  setup.sheetWidth = 8 ∧
  setup.sheetHeight = 5 ∧
  setup.sheetWidth + setup.sheetCount = setup.tableWidth ∧
  setup.sheetHeight + setup.sheetCount = setup.tableLength

/-- The main theorem stating that if the setup is valid, the table length must be 77. -/
theorem table_length_is_77 (setup : TableSetup) :
  isValidSetup setup → setup.tableLength = 77 := by
  sorry

#check table_length_is_77

end NUMINAMATH_CALUDE_table_length_is_77_l2074_207429
