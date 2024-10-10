import Mathlib

namespace square_of_negative_sum_l3493_349315

theorem square_of_negative_sum (a b : ℝ) : (-a - b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end square_of_negative_sum_l3493_349315


namespace root_sum_reciprocal_shifted_l3493_349364

theorem root_sum_reciprocal_shifted (a b c : ℂ) : 
  (a^3 - 2*a - 5 = 0) → 
  (b^3 - 2*b - 5 = 0) → 
  (c^3 - 2*c - 5 = 0) → 
  (1/(a-2) + 1/(b-2) + 1/(c-2) = 10) := by
sorry

end root_sum_reciprocal_shifted_l3493_349364


namespace rain_amount_l3493_349361

theorem rain_amount (malina_initial : ℕ) (jahoda_initial : ℕ) (rain_amount : ℕ) : 
  malina_initial = 48 →
  malina_initial = jahoda_initial + 32 →
  (malina_initial + rain_amount) - (jahoda_initial + rain_amount) = 32 →
  malina_initial + rain_amount = 2 * (jahoda_initial + rain_amount) →
  rain_amount = 16 := by
sorry

end rain_amount_l3493_349361


namespace complex_number_quadrant_l3493_349343

theorem complex_number_quadrant : ∃ (a b : ℝ), (a > 0 ∧ b < 0) ∧ (Complex.mk a b = 5 / (Complex.mk 2 1)) := by
  sorry

end complex_number_quadrant_l3493_349343


namespace perpendicular_bisector_of_intersection_l3493_349349

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Define the intersection points
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the perpendicular bisector equation
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

theorem perpendicular_bisector_of_intersection :
  ∃ (A B : ℝ × ℝ),
    (C₁ A.1 A.2 ∧ C₂ A.1 A.2) ∧
    (C₁ B.1 B.2 ∧ C₂ B.1 B.2) ∧
    A ≠ B ∧
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2) :=
by sorry

end perpendicular_bisector_of_intersection_l3493_349349


namespace largest_divisor_of_difference_of_squares_l3493_349327

theorem largest_divisor_of_difference_of_squares (m n : ℕ) : 
  Odd m → Odd n → n < m → m - n > 2 → 
  (∀ k : ℕ, k > 4 → ∃ x y : ℕ, Odd x ∧ Odd y ∧ y < x ∧ x - y > 2 ∧ ¬(k ∣ x^2 - y^2)) ∧ 
  (∀ x y : ℕ, Odd x → Odd y → y < x → x - y > 2 → (4 ∣ x^2 - y^2)) := by
sorry

end largest_divisor_of_difference_of_squares_l3493_349327


namespace difference_of_reciprocals_l3493_349345

theorem difference_of_reciprocals (x y : ℝ) : 
  x = Real.sqrt 5 - 1 → y = Real.sqrt 5 + 1 → 1 / x - 1 / y = 1 / 2 := by
  sorry

end difference_of_reciprocals_l3493_349345


namespace conditional_probability_in_box_l3493_349350

/-- A box containing products of different classes -/
structure Box where
  total : ℕ
  firstClass : ℕ
  secondClass : ℕ

/-- The probability of drawing a first-class product followed by another first-class product -/
def probBothFirstClass (b : Box) : ℚ :=
  (b.firstClass : ℚ) * ((b.firstClass - 1) : ℚ) / ((b.total : ℚ) * ((b.total - 1) : ℚ))

/-- The probability of drawing a first-class product first -/
def probFirstClassFirst (b : Box) : ℚ :=
  (b.firstClass : ℚ) / (b.total : ℚ)

/-- The conditional probability of drawing a first-class product second, given that the first draw was a first-class product -/
def conditionalProbability (b : Box) : ℚ :=
  probBothFirstClass b / probFirstClassFirst b

theorem conditional_probability_in_box (b : Box) 
  (h1 : b.total = 4)
  (h2 : b.firstClass = 3)
  (h3 : b.secondClass = 1)
  (h4 : b.firstClass + b.secondClass = b.total) :
  conditionalProbability b = 2 / 3 := by
  sorry

end conditional_probability_in_box_l3493_349350


namespace second_concert_attendance_l3493_349371

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (additional_people : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : additional_people = 119) : 
  first_concert + additional_people = 66018 := by
sorry

end second_concert_attendance_l3493_349371


namespace tuesday_max_hours_l3493_349379

/-- Represents the days of the week from Monday to Friday -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

/-- Returns the number of hours Gabe spent riding his bike on a given day -/
def hours_spent (day : Weekday) : ℕ :=
  match day with
  | Weekday.Monday => 3
  | Weekday.Tuesday => 4
  | Weekday.Wednesday => 2
  | Weekday.Thursday => 3
  | Weekday.Friday => 1

/-- Theorem: Tuesday is the day when Gabe spent the greatest number of hours riding his bike -/
theorem tuesday_max_hours :
  ∀ (day : Weekday), hours_spent Weekday.Tuesday ≥ hours_spent day :=
by sorry

end tuesday_max_hours_l3493_349379


namespace three_collinear_points_same_color_l3493_349337

-- Define a color type
inductive Color
| Black
| White

-- Define a point as a pair of real number (position) and color
structure Point where
  position : ℝ
  color : Color

-- Define a function to check if three points are collinear with one in the middle
def areCollinearWithMiddle (p1 p2 p3 : Point) : Prop :=
  p2.position = (p1.position + p3.position) / 2

-- State the theorem
theorem three_collinear_points_same_color (points : Set Point) : 
  ∃ (p1 p2 p3 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
  p1.color = p2.color ∧ p2.color = p3.color ∧
  areCollinearWithMiddle p1 p2 p3 := by
  sorry

end three_collinear_points_same_color_l3493_349337


namespace seating_arrangements_l3493_349376

def dodgers : ℕ := 3
def astros : ℕ := 4
def mets : ℕ := 2
def marlins : ℕ := 1

def total_players : ℕ := dodgers + astros + mets + marlins

def number_of_teams : ℕ := 4

theorem seating_arrangements :
  (number_of_teams.factorial) * (dodgers.factorial) * (astros.factorial) * (mets.factorial) * (marlins.factorial) = 6912 :=
by sorry

end seating_arrangements_l3493_349376


namespace sliced_meat_cost_l3493_349384

/-- Given a 4 pack of sliced meat costing $40.00 with an additional 30% for rush delivery,
    the cost per type of meat is $13.00. -/
theorem sliced_meat_cost (pack_size : ℕ) (base_cost rush_percentage : ℚ) :
  pack_size = 4 →
  base_cost = 40 →
  rush_percentage = 0.3 →
  (base_cost + base_cost * rush_percentage) / pack_size = 13 :=
by sorry

end sliced_meat_cost_l3493_349384


namespace integral_x_zero_to_one_l3493_349338

theorem integral_x_zero_to_one :
  ∫ x in (0 : ℝ)..1, x = (1 : ℝ) / 2 := by sorry

end integral_x_zero_to_one_l3493_349338


namespace cubic_difference_999_l3493_349324

theorem cubic_difference_999 : 
  ∀ m n : ℕ+, m^3 - n^3 = 999 ↔ (m = 10 ∧ n = 1) ∨ (m = 12 ∧ n = 9) := by
sorry

end cubic_difference_999_l3493_349324


namespace floor_width_calculation_l3493_349356

def tile_length : ℝ := 65
def tile_width : ℝ := 25
def floor_length : ℝ := 150
def max_tiles : ℕ := 36

theorem floor_width_calculation (floor_width : ℝ) 
  (h1 : floor_length = 150)
  (h2 : tile_length = 65)
  (h3 : tile_width = 25)
  (h4 : max_tiles = 36)
  (h5 : 2 * tile_length ≤ floor_length)
  (h6 : floor_width = (max_tiles / 2 : ℝ) * tile_width) :
  floor_width = 450 := by
sorry

end floor_width_calculation_l3493_349356


namespace boxes_in_case_l3493_349375

/-- Given information about Maria's eggs and boxes -/
structure EggBoxes where
  num_boxes : ℕ
  eggs_per_box : ℕ
  total_eggs : ℕ

/-- Theorem stating that the number of boxes in a case is 3 -/
theorem boxes_in_case (maria : EggBoxes) 
  (h1 : maria.num_boxes = 3)
  (h2 : maria.eggs_per_box = 7)
  (h3 : maria.total_eggs = 21) :
  maria.num_boxes = 3 := by
  sorry

end boxes_in_case_l3493_349375


namespace smallest_perfect_square_divisible_by_2_3_5_l3493_349352

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k^2) → 2 ∣ n → 3 ∣ n → 5 ∣ n → n ≥ 900 :=
by sorry

end smallest_perfect_square_divisible_by_2_3_5_l3493_349352


namespace carla_initial_marbles_l3493_349358

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The total number of marbles Carla has after buying -/
def total_marbles : ℕ := 187

/-- The number of marbles Carla started with -/
def initial_marbles : ℕ := total_marbles - marbles_bought

theorem carla_initial_marbles :
  initial_marbles = 53 := by sorry

end carla_initial_marbles_l3493_349358


namespace hyperbola_foci_coordinates_l3493_349318

/-- Given a hyperbola with equation 5x^2 - 4y^2 + 60 = 0, its foci have coordinates (0, ±3√3) -/
theorem hyperbola_foci_coordinates :
  let hyperbola := fun (x y : ℝ) => 5 * x^2 - 4 * y^2 + 60
  ∃ (c : ℝ), c = 3 * Real.sqrt 3 ∧
    (∀ (x y : ℝ), hyperbola x y = 0 →
      (hyperbola 0 c = 0 ∧ hyperbola 0 (-c) = 0)) :=
by sorry

end hyperbola_foci_coordinates_l3493_349318


namespace expression_evaluation_l3493_349316

theorem expression_evaluation (a : ℚ) (h : a = 4/3) :
  (6 * a^2 - 8 * a + 3) * (3 * a - 4) = 0 := by
  sorry

end expression_evaluation_l3493_349316


namespace statement_1_statement_2_statement_3_l3493_349359

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- Statement 1
theorem statement_1 : ∃ (a b : Line) (α : Plane),
  parallel_line a b ∧ contained_in b α ∧ 
  ¬(parallel_line_plane a α) ∧ ¬(contained_in a α) := by sorry

-- Statement 2
theorem statement_2 : ∃ (a b : Line) (α : Plane),
  parallel_line_plane a α ∧ parallel_line_plane b α ∧ 
  ¬(parallel_line a b) := by sorry

-- Statement 3
theorem statement_3 : ¬(∀ (a : Line) (α β : Plane),
  parallel_line_plane a α → parallel_line_plane a β → 
  (α = β ∨ ∃ (l : Line), parallel_line_plane l α ∧ parallel_line_plane l β)) := by sorry

end statement_1_statement_2_statement_3_l3493_349359


namespace chorus_arrangement_l3493_349342

/-- The maximum number of chorus members that satisfies both arrangements -/
def max_chorus_members : ℕ := 300

/-- The number of columns in the rectangular formation -/
def n : ℕ := 15

/-- The side length of the square formation -/
def k : ℕ := 17

theorem chorus_arrangement :
  (∃ m : ℕ, m = max_chorus_members) ∧
  (∃ k : ℕ, max_chorus_members = k^2 + 11) ∧
  (max_chorus_members = n * (n + 5)) ∧
  (∀ m : ℕ, m > max_chorus_members →
    (¬∃ k : ℕ, m = k^2 + 11) ∨ (¬∃ n : ℕ, m = n * (n + 5))) :=
by sorry

#eval max_chorus_members
#eval n
#eval k

end chorus_arrangement_l3493_349342


namespace quadratic_solution_difference_squared_l3493_349367

theorem quadratic_solution_difference_squared :
  ∀ α β : ℝ,
  (α^2 - 5*α + 6 = 0) →
  (β^2 - 5*β + 6 = 0) →
  (α ≠ β) →
  (α - β)^2 = 1 :=
by
  sorry

end quadratic_solution_difference_squared_l3493_349367


namespace x_equals_plus_minus_fifteen_l3493_349389

theorem x_equals_plus_minus_fifteen (x : ℝ) :
  (x / 5) / 3 = 3 / (x / 5) → x = 15 ∨ x = -15 := by
  sorry

end x_equals_plus_minus_fifteen_l3493_349389


namespace orchard_expansion_l3493_349300

theorem orchard_expansion (n : ℕ) (h1 : n^2 + 146 = 7890) (h2 : (n + 1)^2 = n^2 + 31 + 146) : (n + 1)^2 = 7921 := by
  sorry

end orchard_expansion_l3493_349300


namespace circle_angle_theorem_l3493_349378

/-- The number of angles not greater than 120° in a circle with n points -/
def S (n : ℕ) : ℕ := sorry

/-- The minimum number of points required for a given S(n) -/
def n_min (s : ℕ) : ℕ := sorry

theorem circle_angle_theorem :
  (∀ n : ℕ, n ≥ 3 → 
    (∀ k : ℕ, k ≥ 1 →
      (2 * Nat.choose k 2 < S n ∧ S n ≤ Nat.choose k 2 + Nat.choose (k + 1) 2) →
        n_min (S n) = 2 * k + 1)) ∧
  (∀ n : ℕ, n ≥ 3 →
    (∀ k : ℕ, k ≥ 2 →
      (Nat.choose (k - 1) 2 + Nat.choose k 2 < S n ∧ S n ≤ 2 * Nat.choose k 2) →
        n_min (S n) = 2 * k)) :=
by sorry

end circle_angle_theorem_l3493_349378


namespace wall_width_is_100cm_l3493_349319

/-- Represents the dimensions of a brick in centimeters -/
structure BrickDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a wall in centimeters -/
structure WallDimensions where
  length : ℝ
  width : ℝ
  thickness : ℝ

/-- Calculates the volume of a brick given its dimensions -/
def brickVolume (d : BrickDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the volume of a wall given its dimensions -/
def wallVolume (d : WallDimensions) : ℝ :=
  d.length * d.width * d.thickness

/-- Theorem stating that the width of the wall is 100 cm -/
theorem wall_width_is_100cm
  (brick : BrickDimensions)
  (wall : WallDimensions)
  (h1 : brick.length = 25)
  (h2 : brick.width = 11)
  (h3 : brick.height = 6)
  (h4 : wall.length = 800)
  (h5 : wall.thickness = 5)
  (h6 : 242.42424242424244 * brickVolume brick = wallVolume wall) :
  wall.width = 100 := by
  sorry

end wall_width_is_100cm_l3493_349319


namespace exists_positive_x_hash_equals_63_l3493_349314

/-- Definition of the # operation -/
def hash (a b : ℝ) : ℝ := a * b - b + b^2

/-- Theorem stating the existence of a positive real number x such that 3 # x = 63 -/
theorem exists_positive_x_hash_equals_63 : ∃ x : ℝ, x > 0 ∧ hash 3 x = 63 := by
  sorry

end exists_positive_x_hash_equals_63_l3493_349314


namespace choose_two_cooks_from_eight_l3493_349368

theorem choose_two_cooks_from_eight (n : ℕ) (k : ℕ) :
  n = 8 ∧ k = 2 → Nat.choose n k = 28 := by
  sorry

end choose_two_cooks_from_eight_l3493_349368


namespace largest_lcm_with_18_l3493_349377

theorem largest_lcm_with_18 : 
  (Finset.max {lcm 18 3, lcm 18 5, lcm 18 9, lcm 18 12, lcm 18 15, lcm 18 18}) = 90 := by
  sorry

end largest_lcm_with_18_l3493_349377


namespace negation_of_cube_odd_is_odd_l3493_349334

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ Even (n^3)) :=
sorry

end negation_of_cube_odd_is_odd_l3493_349334


namespace range_of_m_for_nonempty_solution_l3493_349305

theorem range_of_m_for_nonempty_solution (m : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + m| ≤ 4) → m ∈ Set.Icc (-5) 3 :=
by sorry

end range_of_m_for_nonempty_solution_l3493_349305


namespace sine_range_theorem_l3493_349326

theorem sine_range_theorem (x : ℝ) :
  x ∈ Set.Icc (0 : ℝ) (2 * Real.pi) →
  (Set.Icc (0 : ℝ) (2 * Real.pi) ∩ {x | Real.sin x ≥ Real.sqrt 3 / 2}) =
  Set.Icc (Real.pi / 3) ((2 * Real.pi) / 3) :=
by sorry

end sine_range_theorem_l3493_349326


namespace angle_addition_l3493_349311

-- Define a structure for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  let remainingMinutes := totalMinutes % 60
  ⟨a.degrees + b.degrees + extraDegrees, remainingMinutes⟩

-- Theorem statement
theorem angle_addition :
  Angle.add ⟨36, 28⟩ ⟨25, 34⟩ = ⟨62, 2⟩ :=
by sorry

end angle_addition_l3493_349311


namespace no_extrema_in_interval_l3493_349302

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the open interval (-1, 1)
def openInterval : Set ℝ := {x | -1 < x ∧ x < 1}

-- Theorem statement
theorem no_extrema_in_interval :
  ¬∃ (max_val min_val : ℝ), 
    (∀ x ∈ openInterval, f x ≤ max_val) ∧
    (∃ x_max ∈ openInterval, f x_max = max_val) ∧
    (∀ x ∈ openInterval, min_val ≤ f x) ∧
    (∃ x_min ∈ openInterval, f x_min = min_val) :=
sorry

end no_extrema_in_interval_l3493_349302


namespace solve_equation_for_A_l3493_349354

theorem solve_equation_for_A : ∃ A : ℝ,
  (1 / ((5 / (1 + (24 / A))) - 5 / 9)) * (3 / (2 + (5 / 7))) / (2 / (3 + (3 / 4))) + 2.25 = 4 ∧ A = 2.25 := by
  sorry

end solve_equation_for_A_l3493_349354


namespace interesting_numbers_200_to_400_l3493_349360

/-- A natural number is interesting if there exists another natural number that satisfies certain conditions. -/
def IsInteresting (A : ℕ) : Prop :=
  ∃ B : ℕ, A > B ∧ Nat.Prime (A - B) ∧ ∃ n : ℕ, A * B = n * n

/-- The theorem stating the interesting numbers between 200 and 400. -/
theorem interesting_numbers_200_to_400 :
  ∀ A : ℕ, 200 < A → A < 400 → (IsInteresting A ↔ A = 225 ∨ A = 256 ∨ A = 361) := by
  sorry


end interesting_numbers_200_to_400_l3493_349360


namespace smallest_four_digit_divisible_by_34_l3493_349339

theorem smallest_four_digit_divisible_by_34 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 34 ∣ n → n ≥ 1020 :=
by
  sorry

end smallest_four_digit_divisible_by_34_l3493_349339


namespace sum_of_repeating_decimals_l3493_349320

-- Define the repeating decimals
def repeating_decimal_1 : ℚ := 2/9
def repeating_decimal_2 : ℚ := 2/99

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 = 8/33 := by
  sorry

end sum_of_repeating_decimals_l3493_349320


namespace edward_pen_expenses_l3493_349362

/-- Given Edward's initial money, book expenses, and remaining money, 
    calculate the amount spent on pens. -/
theorem edward_pen_expenses (initial_money : ℕ) (book_expenses : ℕ) (remaining_money : ℕ) 
    (h1 : initial_money = 41)
    (h2 : book_expenses = 6)
    (h3 : remaining_money = 19) :
  initial_money - book_expenses - remaining_money = 16 := by
  sorry

end edward_pen_expenses_l3493_349362


namespace vector_magnitude_problem_l3493_349374

theorem vector_magnitude_problem (a b : ℝ × ℝ) :
  let angle := Real.pi / 3
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product := a.1 * b.1 + a.2 * b.2
  angle = Real.arccos (dot_product / (norm_a * norm_b)) →
  norm_a = 1 →
  norm_b = 1 / 2 →
  Real.sqrt (((a.1 - 2 * b.1) ^ 2) + ((a.2 - 2 * b.2) ^ 2)) = 1 := by
sorry

end vector_magnitude_problem_l3493_349374


namespace cole_average_speed_back_home_l3493_349399

/-- Proves that Cole's average speed back home was 120 km/h given the conditions of his round trip. -/
theorem cole_average_speed_back_home 
  (speed_to_work : ℝ) 
  (total_time : ℝ) 
  (time_to_work : ℝ) 
  (h1 : speed_to_work = 80) 
  (h2 : total_time = 3) 
  (h3 : time_to_work = 108 / 60) : 
  (speed_to_work * time_to_work) / (total_time - time_to_work) = 120 := by
  sorry

#check cole_average_speed_back_home

end cole_average_speed_back_home_l3493_349399


namespace inverse_of_inverse_sixteen_l3493_349388

def f (x : ℝ) : ℝ := 5 * x + 6

theorem inverse_of_inverse_sixteen (hf : ∀ x, f x = 5 * x + 6) :
  (f ∘ f) (-4/5) = 16 :=
sorry

end inverse_of_inverse_sixteen_l3493_349388


namespace geometric_sequence_fifth_term_l3493_349328

theorem geometric_sequence_fifth_term 
  (t : ℕ → ℝ) 
  (h_positive : ∀ n, t n > 0) 
  (h_decreasing : t 1 > t 2) 
  (h_sum : t 1 + t 2 = 15/2) 
  (h_sum_squares : t 1^2 + t 2^2 = 153/4) 
  (h_geometric : ∃ r : ℝ, ∀ n, t (n+1) = t n * r) :
  t 5 = 3/128 := by
  sorry

end geometric_sequence_fifth_term_l3493_349328


namespace other_replaced_man_age_proof_l3493_349395

/-- The age of the other replaced man in a group of three men -/
def other_replaced_man_age : ℕ := 26

theorem other_replaced_man_age_proof 
  (initial_men : ℕ) 
  (replaced_men : ℕ) 
  (known_replaced_age : ℕ) 
  (new_men_avg_age : ℝ) 
  (h1 : initial_men = 3)
  (h2 : replaced_men = 2)
  (h3 : known_replaced_age = 23)
  (h4 : new_men_avg_age = 25)
  (h5 : ∀ (initial_avg new_avg : ℝ), new_avg > initial_avg) :
  other_replaced_man_age = 26 := by
  sorry

end other_replaced_man_age_proof_l3493_349395


namespace selection_options_count_l3493_349353

/-- Represents the number of people skilled in the first method -/
def skilled_in_first_method : ℕ := 5

/-- Represents the number of people skilled in the second method -/
def skilled_in_second_method : ℕ := 4

/-- Represents the total number of people -/
def total_people : ℕ := skilled_in_first_method + skilled_in_second_method

/-- Theorem: The number of ways to select one person from the group is equal to the total number of people -/
theorem selection_options_count : 
  (skilled_in_first_method + skilled_in_second_method) = total_people := by
  sorry

end selection_options_count_l3493_349353


namespace parabola_standard_form_l3493_349306

/-- A parabola with axis of symmetry x = 1 -/
structure Parabola where
  axis_of_symmetry : ℝ
  h_axis : axis_of_symmetry = 1

/-- The standard form of a parabola equation y^2 = ax -/
def standard_form (a : ℝ) (x y : ℝ) : Prop :=
  y^2 = a * x

/-- Theorem stating that the standard form of the parabola with axis of symmetry x = 1 is y^2 = -4x -/
theorem parabola_standard_form (p : Parabola) :
  ∃ a : ℝ, (∀ x y : ℝ, standard_form a x y) ∧ a = -4 :=
sorry

end parabola_standard_form_l3493_349306


namespace roberto_outfits_l3493_349301

/-- Calculates the number of possible outfits given the number of choices for each item -/
def number_of_outfits (trousers shirts jackets shoes : ℕ) : ℕ :=
  trousers * shirts * jackets * shoes

/-- Theorem stating that Roberto can create 240 different outfits -/
theorem roberto_outfits :
  let trousers : ℕ := 4
  let shirts : ℕ := 5
  let jackets : ℕ := 3
  let shoes : ℕ := 4
  number_of_outfits trousers shirts jackets shoes = 240 := by
  sorry

end roberto_outfits_l3493_349301


namespace exactly_two_false_l3493_349357

-- Define the types
def Quadrilateral : Type := sorry
def Square : Quadrilateral → Prop := sorry
def Rectangle : Quadrilateral → Prop := sorry

-- Define the propositions
def P1 : Prop := ∀ q : Quadrilateral, Square q → Rectangle q
def P2 : Prop := ∀ q : Quadrilateral, Rectangle q → Square q
def P3 : Prop := ∀ q : Quadrilateral, ¬(Square q) → ¬(Rectangle q)
def P4 : Prop := ∀ q : Quadrilateral, ¬(Rectangle q) → ¬(Square q)

-- The theorem to prove
theorem exactly_two_false : 
  (¬P1 ∧ ¬P2 ∧ P3 ∧ P4) ∨ 
  (¬P1 ∧ P2 ∧ ¬P3 ∧ P4) ∨ 
  (¬P1 ∧ P2 ∧ P3 ∧ ¬P4) ∨ 
  (P1 ∧ ¬P2 ∧ ¬P3 ∧ P4) ∨ 
  (P1 ∧ ¬P2 ∧ P3 ∧ ¬P4) ∨ 
  (P1 ∧ P2 ∧ ¬P3 ∧ ¬P4) :=
sorry

end exactly_two_false_l3493_349357


namespace sqrt_real_implies_x_leq_one_l3493_349385

theorem sqrt_real_implies_x_leq_one (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 1 - x) → x ≤ 1 := by sorry

end sqrt_real_implies_x_leq_one_l3493_349385


namespace fiveDigitNumbers_eq_ten_l3493_349333

/-- The number of five-digit natural numbers formed with digits 1 and 0, containing exactly three 1s -/
def fiveDigitNumbers : ℕ :=
  Nat.choose 5 2

/-- Theorem stating that the number of such five-digit numbers is 10 -/
theorem fiveDigitNumbers_eq_ten : fiveDigitNumbers = 10 := by
  sorry

end fiveDigitNumbers_eq_ten_l3493_349333


namespace new_student_weight_l3493_349330

/-- Given 5 students, if replacing a 92 kg student with a new student
    causes the average weight to decrease by 4 kg,
    then the new student's weight is 72 kg. -/
theorem new_student_weight
  (n : Nat)
  (old_weight : Nat)
  (weight_decrease : Nat)
  (h1 : n = 5)
  (h2 : old_weight = 92)
  (h3 : weight_decrease = 4)
  : n * weight_decrease = old_weight - (old_weight - n * weight_decrease) :=
by
  sorry

#check new_student_weight

end new_student_weight_l3493_349330


namespace complement_intersection_empty_l3493_349380

open Set

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3, 5}
def B : Set Nat := {2, 4, 5}

theorem complement_intersection_empty :
  (U \ A) ∩ (U \ B) = ∅ := by sorry

end complement_intersection_empty_l3493_349380


namespace recurrence_sequence_x7_l3493_349396

/-- A sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (x : ℕ → ℕ) : Prop :=
  (∀ n, x n > 0) ∧
  (∀ n ∈ ({1, 2, 3, 4} : Finset ℕ), x (n + 3) = x (n + 2) * (x (n + 1) + x n))

theorem recurrence_sequence_x7 (x : ℕ → ℕ) (h : RecurrenceSequence x) (h6 : x 6 = 144) :
  x 7 = 3456 := by
  sorry

#check recurrence_sequence_x7

end recurrence_sequence_x7_l3493_349396


namespace surface_area_unchanged_l3493_349381

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (d : Dimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Represents a cube -/
structure Cube where
  side : ℝ

/-- Theorem: The surface area of a rectangular solid remains unchanged
    when two unit cubes are removed from opposite corners -/
theorem surface_area_unchanged
  (solid : Dimensions)
  (cube : Cube)
  (h1 : solid.length = 2)
  (h2 : solid.width = 3)
  (h3 : solid.height = 4)
  (h4 : cube.side = 1) :
  surfaceArea solid = surfaceArea solid - 2 * (3 * cube.side^2) + 2 * (3 * cube.side^2) :=
by sorry

end surface_area_unchanged_l3493_349381


namespace termite_ridden_collapsing_fraction_value_l3493_349348

/-- The fraction of homes on Gotham Street that are termite-ridden -/
def termite_ridden_fraction : ℚ := 1/3

/-- The fraction of homes on Gotham Street that are termite-ridden but not collapsing -/
def termite_ridden_not_collapsing_fraction : ℚ := 1/10

/-- The fraction of termite-ridden homes that are collapsing -/
def termite_ridden_collapsing_fraction : ℚ := 
  (termite_ridden_fraction - termite_ridden_not_collapsing_fraction) / termite_ridden_fraction

theorem termite_ridden_collapsing_fraction_value : 
  termite_ridden_collapsing_fraction = 7/30 := by
  sorry

end termite_ridden_collapsing_fraction_value_l3493_349348


namespace intersection_k_values_eq_four_and_fourteen_l3493_349309

/-- The set of possible k values for which |z - 4| = 3|z + 4| and |z| = k intersect at exactly one point. -/
def intersection_k_values : Set ℝ :=
  {k : ℝ | ∃! (z : ℂ), Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k}

/-- Theorem stating that the intersection_k_values set contains only 4 and 14. -/
theorem intersection_k_values_eq_four_and_fourteen :
  intersection_k_values = {4, 14} := by
  sorry

end intersection_k_values_eq_four_and_fourteen_l3493_349309


namespace unique_parallel_line_l3493_349331

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (in_plane : Point → Plane → Prop)
variable (passes_through : Line → Point → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_parallel : Line → Line → Prop)

-- State the theorem
theorem unique_parallel_line 
  (α β : Plane) (a : Line) (B : Point)
  (h1 : parallel α β)
  (h2 : contains α a)
  (h3 : in_plane B β) :
  ∃! l : Line, line_in_plane l β ∧ passes_through l B ∧ line_parallel l a :=
sorry

end unique_parallel_line_l3493_349331


namespace divisibility_by_17_l3493_349391

theorem divisibility_by_17 (x y : ℤ) : 
  (∃ k : ℤ, 2*x + 3*y = 17*k) → (∃ m : ℤ, 9*x + 5*y = 17*m) :=
by sorry

end divisibility_by_17_l3493_349391


namespace largest_variable_l3493_349310

theorem largest_variable (a b c d : ℝ) (h : a - 2 = b + 3 ∧ a - 2 = c - 4 ∧ a - 2 = d + 5) :
  c ≥ a ∧ c ≥ b ∧ c ≥ d :=
by sorry

end largest_variable_l3493_349310


namespace quadratic_roots_sum_l3493_349313

theorem quadratic_roots_sum (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 503 := by
  sorry

end quadratic_roots_sum_l3493_349313


namespace chef_pies_l3493_349363

theorem chef_pies (apple_pies pecan_pies pumpkin_pies : ℕ) 
  (h1 : apple_pies = 2) 
  (h2 : pecan_pies = 4) 
  (h3 : pumpkin_pies = 7) : 
  apple_pies + pecan_pies + pumpkin_pies = 13 := by
  sorry

end chef_pies_l3493_349363


namespace quadratic_solution_l3493_349322

theorem quadratic_solution (b : ℝ) : 
  ((-9 : ℝ)^2 + b * (-9) - 36 = 0) → b = 5 := by
sorry

end quadratic_solution_l3493_349322


namespace interest_percentage_calculation_l3493_349347

/-- Calculates the interest percentage for a purchase with a payment plan -/
theorem interest_percentage_calculation (purchase_price : ℝ) (down_payment : ℝ) (monthly_payment : ℝ) (num_months : ℕ) :
  purchase_price = 110 →
  down_payment = 10 →
  monthly_payment = 10 →
  num_months = 12 →
  let total_paid := down_payment + (monthly_payment * num_months)
  let interest_paid := total_paid - purchase_price
  let interest_percentage := (interest_paid / purchase_price) * 100
  ∃ ε > 0, |interest_percentage - 18.2| < ε := by
  sorry

end interest_percentage_calculation_l3493_349347


namespace big_eighteen_game_count_l3493_349329

/-- Calculates the total number of games in a basketball conference. -/
def total_conference_games (num_divisions : ℕ) (teams_per_division : ℕ) 
  (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let total_teams := num_divisions * teams_per_division
  let intra_division_total := num_divisions * (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games
  let inter_division_total := (total_teams * (total_teams - teams_per_division) * inter_division_games) / 2
  intra_division_total + inter_division_total

/-- The Big Eighteen Basketball Conference game count theorem -/
theorem big_eighteen_game_count : 
  total_conference_games 3 6 3 2 = 486 := by
  sorry

end big_eighteen_game_count_l3493_349329


namespace initial_distance_between_cars_l3493_349390

theorem initial_distance_between_cars (speed_A speed_B time_to_overtake distance_ahead : ℝ) 
  (h1 : speed_A = 58)
  (h2 : speed_B = 50)
  (h3 : time_to_overtake = 4.75)
  (h4 : distance_ahead = 8) : 
  (speed_A - speed_B) * time_to_overtake = 30 + distance_ahead := by
  sorry

end initial_distance_between_cars_l3493_349390


namespace income_increase_percentage_l3493_349392

/-- Proves that given the ratio of expenditure to savings is 3:2, if savings increase by 6% and expenditure increases by 21%, then the income increases by 15% -/
theorem income_increase_percentage 
  (I : ℝ) -- Initial income
  (E : ℝ) -- Initial expenditure
  (S : ℝ) -- Initial savings
  (h1 : E / S = 3 / 2) -- Ratio of expenditure to savings is 3:2
  (h2 : I = E + S) -- Income is the sum of expenditure and savings
  (h3 : S * 1.06 + E * 1.21 = I * (1 + 15/100)) -- New savings + new expenditure = new income
  : ∃ (x : ℝ), x = 15 ∧ I * (1 + x/100) = S * 1.06 + E * 1.21 :=
by sorry

end income_increase_percentage_l3493_349392


namespace book_completion_time_l3493_349325

/-- Calculates the number of weeks needed to complete a book given the writing schedule and book length -/
theorem book_completion_time (writing_hours_per_day : ℕ) (pages_per_hour : ℕ) (total_pages : ℕ) :
  writing_hours_per_day = 3 →
  pages_per_hour = 5 →
  total_pages = 735 →
  (total_pages / (writing_hours_per_day * pages_per_hour) + 6) / 7 = 7 :=
by
  sorry

#check book_completion_time

end book_completion_time_l3493_349325


namespace fraction_equality_l3493_349341

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 37) = 925 / 1000 → a = 455 := by
  sorry

end fraction_equality_l3493_349341


namespace three_digit_numbers_count_l3493_349393

def given_numbers : List Nat := [0, 2, 3, 4, 6]

def is_valid_three_digit (n : Nat) : Bool :=
  n ≥ 100 ∧ n ≤ 999 ∧ (n / 100 ∈ given_numbers) ∧ ((n / 10) % 10 ∈ given_numbers) ∧ (n % 10 ∈ given_numbers)

def count_valid_three_digit : Nat :=
  (List.range 1000).filter is_valid_three_digit |>.length

def is_divisible_by_three (n : Nat) : Bool :=
  n % 3 = 0

def count_valid_three_digit_divisible_by_three : Nat :=
  (List.range 1000).filter (λ n => is_valid_three_digit n ∧ is_divisible_by_three n) |>.length

theorem three_digit_numbers_count :
  count_valid_three_digit = 48 ∧
  count_valid_three_digit_divisible_by_three = 20 := by
  sorry


end three_digit_numbers_count_l3493_349393


namespace inverse_of_3_mod_35_l3493_349340

theorem inverse_of_3_mod_35 : ∃ x : ℕ, x < 35 ∧ (3 * x) % 35 = 1 :=
by
  use 12
  sorry

end inverse_of_3_mod_35_l3493_349340


namespace speaking_sequences_count_l3493_349397

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def totalStudents : ℕ := 6

/-- The number of speakers to be selected -/
def speakersToSelect : ℕ := 4

/-- The number of specific students (A and B) -/
def specificStudents : ℕ := 2

theorem speaking_sequences_count :
  (choose specificStudents 1 * choose (totalStudents - specificStudents) (speakersToSelect - 1) * arrange speakersToSelect speakersToSelect) +
  (choose specificStudents 2 * choose (totalStudents - specificStudents) (speakersToSelect - 2) * arrange speakersToSelect speakersToSelect) = 336 :=
by sorry

end speaking_sequences_count_l3493_349397


namespace wrong_number_calculation_l3493_349394

theorem wrong_number_calculation (n : ℕ) (initial_avg correct_avg correct_num wrong_num : ℚ) : 
  n = 10 →
  initial_avg = 18 →
  correct_avg = 19 →
  correct_num = 36 →
  (n : ℚ) * initial_avg + (correct_num - wrong_num) = (n : ℚ) * correct_avg →
  wrong_num = 26 := by
sorry

end wrong_number_calculation_l3493_349394


namespace solution_exists_l3493_349321

theorem solution_exists : ∃ (v : ℝ), 4 * v^2 = 144 ∧ v = 6 := by
  sorry

end solution_exists_l3493_349321


namespace complex_sum_of_parts_l3493_349365

theorem complex_sum_of_parts (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) :
  (z.re : ℝ) + (z.im : ℝ) = -1 := by sorry

end complex_sum_of_parts_l3493_349365


namespace triangle_side_equality_l3493_349335

-- Define the triangle ABC
structure Triangle (α : Type) where
  A : α
  B : α
  C : α

-- Define the sides of the triangle
def side_AB (a b : ℤ) : ℤ := b^2 - 1
def side_BC (a b : ℤ) : ℤ := a^2
def side_CA (a b : ℤ) : ℤ := 2*a

-- State the theorem
theorem triangle_side_equality (a b : ℤ) (ABC : Triangle ℤ) :
  a > 1 ∧ b > 1 ∧ 
  side_AB a b = b^2 - 1 ∧
  side_BC a b = a^2 ∧
  side_CA a b = 2*a →
  b - a = 0 :=
sorry

end triangle_side_equality_l3493_349335


namespace number_problem_l3493_349346

theorem number_problem (x : ℝ) : (6 * x) / 1.5 = 3.8 → x = 0.95 := by
  sorry

end number_problem_l3493_349346


namespace power_of_negative_one_product_l3493_349332

theorem power_of_negative_one_product (n : ℕ) : 
  ((-1 : ℤ) ^ n) * ((-1 : ℤ) ^ (2 * n + 1)) * ((-1 : ℤ) ^ (n + 1)) = 1 := by
  sorry

end power_of_negative_one_product_l3493_349332


namespace min_value_of_complex_expression_l3493_349369

theorem min_value_of_complex_expression (z : ℂ) (h : Complex.abs (z - 3 + 3*I) = 3) :
  ∃ (min_val : ℝ), min_val = 19 - 6 * Real.sqrt 2 ∧
    ∀ (w : ℂ), Complex.abs (w - 3 + 3*I) = 3 →
      Complex.abs (w + 2 - I)^2 + Complex.abs (w - 4 + 2*I)^2 ≥ min_val :=
by
  sorry

end min_value_of_complex_expression_l3493_349369


namespace blueberry_picking_total_l3493_349366

theorem blueberry_picking_total (annie kathryn ben sam : ℕ) : 
  annie = 16 ∧ 
  kathryn = 2 * annie + 2 ∧ 
  ben = kathryn / 2 - 3 ∧ 
  sam = 2 * (ben + kathryn) / 3 → 
  annie + kathryn + ben + sam = 96 := by
sorry

end blueberry_picking_total_l3493_349366


namespace caps_lost_per_year_l3493_349308

def caps_first_year : ℕ := 3 * 12
def caps_subsequent_years (years : ℕ) : ℕ := 5 * 12 * years
def christmas_caps (years : ℕ) : ℕ := 40 * years
def total_collection_years : ℕ := 5
def current_cap_count : ℕ := 401

theorem caps_lost_per_year :
  let total_caps := caps_first_year + 
                    caps_subsequent_years (total_collection_years - 1) + 
                    christmas_caps total_collection_years
  let total_lost := total_caps - current_cap_count
  (total_lost / total_collection_years : ℚ) = 15 := by sorry

end caps_lost_per_year_l3493_349308


namespace triangle_properties_l3493_349323

open Real

theorem triangle_properties (a b c A B C : Real) :
  -- Given conditions
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (2 * Real.sqrt 3 * a * c * Real.sin B = a^2 + b^2 - c^2) →
  -- First part
  (C = π / 6) ∧
  -- Additional conditions for the second part
  (b * Real.sin (π - A) = a * Real.cos B) →
  (b = Real.sqrt 2) →
  -- Second part
  (1/2 * b * c * Real.sin A = (Real.sqrt 3 + 1) / 4) :=
by sorry

end triangle_properties_l3493_349323


namespace friends_total_score_l3493_349370

/-- Given three friends' scores in a table football game, prove their total score. -/
theorem friends_total_score (darius_score matt_score marius_score : ℕ) : 
  marius_score = darius_score + 3 →
  matt_score = darius_score + 5 →
  darius_score = 10 →
  darius_score + matt_score + marius_score = 38 := by
sorry


end friends_total_score_l3493_349370


namespace arithmetic_sequence_sum_l3493_349307

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 2 + a 8 = 180) :=
by sorry

end arithmetic_sequence_sum_l3493_349307


namespace intersection_of_P_and_Q_l3493_349373

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def Q : Set ℝ := {x : ℝ | x > 3}

-- State the theorem
theorem intersection_of_P_and_Q :
  P ∩ Q = {x : ℝ | 3 < x ∧ x < 4} := by sorry

end intersection_of_P_and_Q_l3493_349373


namespace population_growth_rate_l3493_349303

theorem population_growth_rate (initial_population : ℝ) (population_after_two_years : ℝ) 
  (h1 : initial_population = 12000)
  (h2 : population_after_two_years = 18451.2) : 
  ∃ (r : ℝ), r = 24 ∧ population_after_two_years = initial_population * (1 + r / 100)^2 :=
by
  sorry

end population_growth_rate_l3493_349303


namespace greatest_common_piece_length_l3493_349304

theorem greatest_common_piece_length : Nat.gcd 42 (Nat.gcd 63 84) = 21 := by
  sorry

end greatest_common_piece_length_l3493_349304


namespace find_divisor_l3493_349383

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 127)
  (h2 : quotient = 5)
  (h3 : remainder = 2)
  (h4 : dividend = quotient * (dividend / quotient) + remainder) :
  dividend / quotient = 25 := by
  sorry

end find_divisor_l3493_349383


namespace parabola_shift_l3493_349312

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := 2 * x^2

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 - 5

/-- The horizontal shift amount -/
def h_shift : ℝ := 1

/-- The vertical shift amount -/
def v_shift : ℝ := -5

theorem parabola_shift :
  ∀ x : ℝ, shifted_parabola x = original_parabola (x - h_shift) + v_shift :=
by sorry

end parabola_shift_l3493_349312


namespace decagon_diagonals_from_vertex_l3493_349351

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to define the specifics of a regular polygon for this statement
  -- Just the number of sides is sufficient

/-- The number of diagonals that can be drawn from a single vertex in a regular polygon -/
def diagonalsFromVertex (p : RegularPolygon n) : ℕ := n - 3

/-- Theorem: In a regular decagon, 7 diagonals can be drawn from any vertex -/
theorem decagon_diagonals_from_vertex :
  ∀ (p : RegularPolygon 10), diagonalsFromVertex p = 7 := by
  sorry

end decagon_diagonals_from_vertex_l3493_349351


namespace tv_screen_area_l3493_349386

theorem tv_screen_area (width height area : ℝ) : 
  width = 3 ∧ height = 7 ∧ area = width * height → area = 21 := by
  sorry

end tv_screen_area_l3493_349386


namespace triangle_problem_l3493_349336

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C)
  (h2 : t.a = 3)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 2) :
  t.B = π/3 ∧ t.a * t.c * Real.cos (π - t.A) = -1 := by
  sorry

end triangle_problem_l3493_349336


namespace percentage_b_of_d_l3493_349317

theorem percentage_b_of_d (A B C D : ℝ) 
  (hB : B = 1.71 * A) 
  (hC : C = 1.80 * A) 
  (hD : D = 1.90 * B) : 
  ∃ ε > 0, |100 * B / D - 52.63| < ε :=
sorry

end percentage_b_of_d_l3493_349317


namespace annas_weight_anna_weighs_80_l3493_349355

/-- The weight of Anna given Jack's weight and the balancing condition on a see-saw -/
theorem annas_weight (jack_weight : ℕ) (rock_weight : ℕ) (rock_count : ℕ) : ℕ :=
  jack_weight + rock_weight * rock_count

/-- Proof that Anna weighs 80 pounds given the conditions -/
theorem anna_weighs_80 :
  annas_weight 60 4 5 = 80 := by
  sorry

end annas_weight_anna_weighs_80_l3493_349355


namespace second_part_interest_rate_l3493_349372

-- Define the total sum and the two parts
def total_sum : ℚ := 2717
def second_part : ℚ := 1672
def first_part : ℚ := total_sum - second_part

-- Define the interest rates and time periods
def first_rate : ℚ := 3 / 100
def first_time : ℚ := 8
def second_time : ℚ := 3

-- Define the theorem
theorem second_part_interest_rate :
  ∃ (r : ℚ), 
    (first_part * first_rate * first_time = second_part * r * second_time) ∧
    (r = 5 / 100) := by
  sorry

end second_part_interest_rate_l3493_349372


namespace six_balls_three_boxes_l3493_349398

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 729 := by
  sorry

end six_balls_three_boxes_l3493_349398


namespace sequence_convergence_condition_l3493_349387

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The limit of a sequence -/
def HasLimit (s : Sequence) (l : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |s n - l| < ε

/-- The condition on the sequence -/
def SequenceCondition (a b : ℝ) (x : Sequence) : Prop :=
  HasLimit (fun n => a * x (n + 1) - b * x n) 0

/-- The main theorem -/
theorem sequence_convergence_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : Sequence, SequenceCondition a b x → HasLimit x 0) ↔
  (a = 0 ∧ b ≠ 0) ∨ (a ≠ 0 ∧ |b / a| < 1) :=
sorry

end sequence_convergence_condition_l3493_349387


namespace becky_lunch_days_proof_l3493_349344

/-- The number of school days in an academic year -/
def school_days : ℕ := 180

/-- The fraction of time Aliyah packs her lunch -/
def aliyah_lunch_fraction : ℚ := 1/2

/-- The fraction of Aliyah's lunch-packing frequency that Becky packs her lunch -/
def becky_lunch_fraction : ℚ := 1/2

/-- The number of days Becky packs her lunch in a school year -/
def becky_lunch_days : ℕ := 45

theorem becky_lunch_days_proof :
  (school_days : ℚ) * aliyah_lunch_fraction * becky_lunch_fraction = becky_lunch_days := by
  sorry

end becky_lunch_days_proof_l3493_349344


namespace max_value_and_right_triangle_l3493_349382

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^3 + x^2 else a * Real.log x

theorem max_value_and_right_triangle (a : ℝ) :
  (∃ (m : ℝ), ∀ x ∈ Set.Icc (-1 : ℝ) (Real.exp 1), f a x ≤ m ∧
    (m = max 2 a ∨ (a < 2 ∧ m = 2))) ∧
  (a > 0 → ∃ (P Q : ℝ × ℝ),
    (P.1 > 0 ∧ P.2 = f a P.1) ∧
    (Q.1 < 0 ∧ Q.2 = f a Q.1) ∧
    (P.1 * Q.1 + P.2 * Q.2 = 0) ∧
    ((P.1 + Q.1) / 2 = 0)) :=
by sorry

end max_value_and_right_triangle_l3493_349382
