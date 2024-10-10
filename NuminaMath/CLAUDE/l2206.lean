import Mathlib

namespace system_solution_l2206_220638

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -7) ∧ (5 * x + 6 * y = -20) ∧ 
  (x = -34/13) ∧ (y = -15/13) := by
  sorry

end system_solution_l2206_220638


namespace probability_of_passing_test_probability_is_two_thirds_l2206_220677

/-- Represents the probability of passing a test given specific conditions -/
theorem probability_of_passing_test
  (total_questions : ℕ)
  (selected_questions : ℕ)
  (correct_answers : ℕ)
  (passing_threshold : ℕ)
  (h1 : total_questions = 10)
  (h2 : selected_questions = 3)
  (h3 : correct_answers = 6)
  (h4 : passing_threshold = 2)
  (h5 : passing_threshold ≤ selected_questions)
  (h6 : correct_answers ≤ total_questions) :
  ℝ :=
2/3

/-- The main theorem stating that the probability of passing the test is 2/3 -/
theorem probability_is_two_thirds
  (total_questions : ℕ)
  (selected_questions : ℕ)
  (correct_answers : ℕ)
  (passing_threshold : ℕ)
  (h1 : total_questions = 10)
  (h2 : selected_questions = 3)
  (h3 : correct_answers = 6)
  (h4 : passing_threshold = 2)
  (h5 : passing_threshold ≤ selected_questions)
  (h6 : correct_answers ≤ total_questions) :
  probability_of_passing_test total_questions selected_questions correct_answers passing_threshold h1 h2 h3 h4 h5 h6 = 2/3 := by
  sorry

end probability_of_passing_test_probability_is_two_thirds_l2206_220677


namespace one_and_one_third_l2206_220660

theorem one_and_one_third : ∃ x : ℚ, (4 / 3) * x = 45 ∧ x = 135 / 4 := by
  sorry

end one_and_one_third_l2206_220660


namespace rhombus_q_value_l2206_220683

/-- A rhombus ABCD on a Cartesian plane -/
structure Rhombus where
  P : ℝ
  u : ℝ
  v : ℝ
  A : ℝ × ℝ := (0, 0)
  B : ℝ × ℝ := (P, 1)
  C : ℝ × ℝ := (u, v)
  D : ℝ × ℝ := (1, P)

/-- The sum of u and v coordinates of point C -/
def Q (r : Rhombus) : ℝ := r.u + r.v

/-- Theorem: For a rhombus ABCD with given coordinates, Q equals 2P + 2 -/
theorem rhombus_q_value (r : Rhombus) : Q r = 2 * r.P + 2 := by
  sorry

end rhombus_q_value_l2206_220683


namespace fraction_equality_l2206_220676

theorem fraction_equality (x : ℚ) (c : ℚ) (h1 : c ≠ 0) (h2 : c ≠ 3) :
  (4 + x) / (5 + x) = c / (3 * c) → x = -7/2 := by
  sorry

end fraction_equality_l2206_220676


namespace custom_op_negative_four_six_l2206_220641

/-- Custom binary operation "*" -/
def custom_op (a b : ℤ) : ℤ := a + 2 * b^2

/-- Theorem stating that (-4) * 6 = 68 under the custom operation -/
theorem custom_op_negative_four_six : custom_op (-4) 6 = 68 := by
  sorry

end custom_op_negative_four_six_l2206_220641


namespace external_polygon_sides_l2206_220621

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon :=
  (sides : ℕ)

/-- Represents the arrangement of polygons as described in the problem -/
structure PolygonArrangement :=
  (hexagon : RegularPolygon)
  (triangle : RegularPolygon)
  (square : RegularPolygon)
  (pentagon : RegularPolygon)
  (heptagon : RegularPolygon)
  (nonagon : RegularPolygon)

/-- Calculates the number of exposed sides in the resulting external polygon -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  arrangement.hexagon.sides +
  arrangement.triangle.sides +
  arrangement.square.sides +
  arrangement.pentagon.sides +
  arrangement.heptagon.sides +
  arrangement.nonagon.sides -
  10 -- Subtracting the sides that are shared between polygons

/-- The main theorem stating that the resulting external polygon has 20 sides -/
theorem external_polygon_sides (arrangement : PolygonArrangement)
  (h1 : arrangement.hexagon.sides = 6)
  (h2 : arrangement.triangle.sides = 3)
  (h3 : arrangement.square.sides = 4)
  (h4 : arrangement.pentagon.sides = 5)
  (h5 : arrangement.heptagon.sides = 7)
  (h6 : arrangement.nonagon.sides = 9) :
  exposedSides arrangement = 20 := by
  sorry

end external_polygon_sides_l2206_220621


namespace part_one_part_two_l2206_220648

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- Part I
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x + f a (x - 2) ≥ 1) → (a ≥ 1/2 ∨ a ≤ -1/2) :=
sorry

-- Part II
theorem part_two (a b c : ℝ) :
  f a ((a - 1) / a) + f a ((b - 1) / a) + f a ((c - 1) / a) = 4 →
  (f a ((a^2 - 1) / a) + f a ((b^2 - 1) / a) + f a ((c^2 - 1) / a) ≥ 16/3 ∧
   ∃ x y z : ℝ, f a ((x^2 - 1) / a) + f a ((y^2 - 1) / a) + f a ((z^2 - 1) / a) = 16/3) :=
sorry

end part_one_part_two_l2206_220648


namespace state_university_cost_l2206_220665

theorem state_university_cost (tuition room_and_board total_cost : ℕ) : 
  tuition = 1644 →
  tuition = room_and_board + 704 →
  total_cost = tuition + room_and_board →
  total_cost = 2584 := by
  sorry

end state_university_cost_l2206_220665


namespace lineup_combinations_count_l2206_220673

/-- The number of ways to choose 6 players from 15 players for 6 specific positions -/
def lineup_combinations : ℕ := 15 * 14 * 13 * 12 * 11 * 10

/-- Theorem stating that the number of ways to choose 6 players from 15 players for 6 specific positions is 3,603,600 -/
theorem lineup_combinations_count : lineup_combinations = 3603600 := by
  sorry

end lineup_combinations_count_l2206_220673


namespace geometric_sum_property_l2206_220699

-- Define a geometric sequence with positive terms and common ratio 2
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ a (n + 1) = 2 * a n

-- Theorem statement
theorem geometric_sum_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_sum : a 1 + a 2 + a 3 = 21) : 
  a 3 + a 4 + a 5 = 84 := by
  sorry

end geometric_sum_property_l2206_220699


namespace min_tiles_for_square_l2206_220670

def tile_length : ℕ := 6
def tile_width : ℕ := 4

def tile_area : ℕ := tile_length * tile_width

def square_side : ℕ := Nat.lcm tile_length tile_width

theorem min_tiles_for_square :
  (square_side * square_side) / tile_area = 6 := by
  sorry

end min_tiles_for_square_l2206_220670


namespace min_tangent_equals_radius_l2206_220629

/-- Circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- Line of symmetry with equation 2ax + by + 6 = 0 -/
def LineOfSymmetry (a b x y : ℝ) : Prop :=
  2*a*x + b*y + 6 = 0

/-- Point (a, b) -/
structure Point where
  a : ℝ
  b : ℝ

/-- Tangent from a point to the circle -/
def Tangent (p : Point) (x y : ℝ) : ℝ :=
  sorry

/-- The radius of the circle -/
def Radius : ℝ :=
  2

theorem min_tangent_equals_radius (a b : ℝ) :
  ∀ (p : Point), p.a = a ∧ p.b = b →
  (∀ (x y : ℝ), Circle x y → LineOfSymmetry a b x y) →
  (∃ (x y : ℝ), Tangent p x y = Radius) ∧
  (∀ (x y : ℝ), Tangent p x y ≥ Radius) :=
sorry

end min_tangent_equals_radius_l2206_220629


namespace steve_initial_boxes_l2206_220612

/-- The number of boxes Steve had initially -/
def initial_boxes (pencils_per_box : ℕ) (pencils_to_lauren : ℕ) (pencils_to_matt_diff : ℕ) (pencils_left : ℕ) : ℕ :=
  (pencils_to_lauren + (pencils_to_lauren + pencils_to_matt_diff) + pencils_left) / pencils_per_box

theorem steve_initial_boxes :
  initial_boxes 12 6 3 9 = 2 := by
  sorry

end steve_initial_boxes_l2206_220612


namespace faculty_marriage_percentage_l2206_220654

theorem faculty_marriage_percentage (total : ℕ) (total_pos : 0 < total) : 
  let women := (70 : ℚ) / 100 * total
  let men := total - women
  let single_men := (1 : ℚ) / 3 * men
  let married_men := (2 : ℚ) / 3 * men
  (married_men : ℚ) / total ≥ (20 : ℚ) / 100 :=
by
  sorry

end faculty_marriage_percentage_l2206_220654


namespace rectangle_y_value_l2206_220687

theorem rectangle_y_value (y : ℝ) : 
  y > 0 → -- y is positive
  (6 - (-2)) * (y - 2) = 64 → -- area of rectangle is 64
  y = 10 := by
sorry

end rectangle_y_value_l2206_220687


namespace initial_bananas_per_child_l2206_220616

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ)
  (h1 : total_children = 660)
  (h2 : absent_children = 330)
  (h3 : extra_bananas = 2) :
  ∃ (initial_bananas : ℕ),
    initial_bananas * total_children = (initial_bananas + extra_bananas) * (total_children - absent_children) ∧
    initial_bananas = 2 := by
  sorry

end initial_bananas_per_child_l2206_220616


namespace initial_water_percentage_l2206_220688

/-- Proves that the initial percentage of water in a 40-liter mixture is 10%
    given that adding 5 liters of water results in a 20% water mixture. -/
theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 40)
  (h2 : added_water = 5)
  (h3 : final_water_percentage = 20)
  (h4 : (initial_volume * x / 100 + added_water) / (initial_volume + added_water) * 100 = final_water_percentage) :
  x = 10 := by
  sorry

#check initial_water_percentage

end initial_water_percentage_l2206_220688


namespace smallest_divisor_after_429_l2206_220603

theorem smallest_divisor_after_429 (n : ℕ) : 
  10000 ≤ n ∧ n < 100000 →  -- n is a five-digit number
  429 ∣ n →                 -- 429 is a divisor of n
  ∃ d : ℕ, d ∣ n ∧ 429 < d ∧ d ≤ 858 ∧ 
    ∀ d' : ℕ, d' ∣ n → 429 < d' → d ≤ d' :=
by sorry

end smallest_divisor_after_429_l2206_220603


namespace tan_sum_pi_twelfths_l2206_220691

theorem tan_sum_pi_twelfths : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end tan_sum_pi_twelfths_l2206_220691


namespace expression_evaluation_l2206_220632

theorem expression_evaluation : 2 - (-3) - 4 + (-5) + 6 - (-7) - 8 = 1 := by
  sorry

end expression_evaluation_l2206_220632


namespace cuboid_max_volume_l2206_220606

theorem cuboid_max_volume (d : ℝ) (p : ℝ) (h1 : d = 10) (h2 : p = 8) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * b * c ≤ 192 ∧
  a^2 + b^2 + c^2 = d^2 ∧
  a^2 + b^2 = p^2 ∧
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    x^2 + y^2 + z^2 = d^2 → x^2 + y^2 = p^2 → x * y * z ≤ 192) :=
by
  sorry

end cuboid_max_volume_l2206_220606


namespace function_property_l2206_220622

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f)
    (h_sym : ∀ x, f (3/2 + x) = -f (3/2 - x))
    (h_f1 : f 1 = 2) : 
  f 2 + f 3 = -2 := by
  sorry

end function_property_l2206_220622


namespace cinema_hall_capacity_l2206_220639

/-- Represents a cinema hall with a given number of rows and seats per row -/
structure CinemaHall where
  rows : ℕ
  seatsPerRow : ℕ

/-- Calculates the approximate seating capacity of a cinema hall -/
def approximateCapacity (hall : CinemaHall) : ℕ :=
  900

/-- Calculates the actual seating capacity of a cinema hall -/
def actualCapacity (hall : CinemaHall) : ℕ :=
  hall.rows * hall.seatsPerRow

theorem cinema_hall_capacity (hall : CinemaHall) 
  (h1 : hall.rows = 28) 
  (h2 : hall.seatsPerRow = 31) : 
  approximateCapacity hall = 900 ∧ actualCapacity hall = 868 := by
  sorry

end cinema_hall_capacity_l2206_220639


namespace nori_initial_boxes_l2206_220645

/-- The number of crayons in each box -/
def crayons_per_box : ℕ := 8

/-- The number of crayons Nori gave to Mae -/
def crayons_to_mae : ℕ := 5

/-- The additional number of crayons Nori gave to Lea compared to Mae -/
def additional_crayons_to_lea : ℕ := 7

/-- The number of crayons Nori has left -/
def crayons_left : ℕ := 15

/-- The number of boxes Nori had initially -/
def initial_boxes : ℕ := 4

theorem nori_initial_boxes : 
  crayons_per_box * initial_boxes = 
    crayons_left + crayons_to_mae + (crayons_to_mae + additional_crayons_to_lea) :=
by sorry

end nori_initial_boxes_l2206_220645


namespace merill_marbles_vivian_marbles_l2206_220642

-- Define the number of marbles for each person
def Selma : ℕ := 50
def Elliot : ℕ := 15
def Merill : ℕ := 2 * Elliot
def Vivian : ℕ := 21

-- Theorem to prove Merill's marbles
theorem merill_marbles : Merill = 30 := by sorry

-- Theorem to prove Vivian's marbles
theorem vivian_marbles : Vivian = 21 ∧ Vivian > Elliot + 5 ∧ Vivian ≥ (135 * Elliot) / 100 := by sorry

end merill_marbles_vivian_marbles_l2206_220642


namespace matching_jelly_bean_probability_l2206_220633

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans a person has -/
def total_jelly_beans (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe_jelly_beans : JellyBeans :=
  { green := 1, red := 1, blue := 1, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob_jelly_beans : JellyBeans :=
  { green := 2, red := 3, blue := 0, yellow := 2 }

/-- Calculates the probability of two people showing the same color jelly bean -/
def matching_color_probability (person1 person2 : JellyBeans) : ℚ :=
  let total1 := total_jelly_beans person1
  let total2 := total_jelly_beans person2
  (person1.green * person2.green + person1.red * person2.red + person1.blue * person2.blue) / (total1 * total2)

theorem matching_jelly_bean_probability :
  matching_color_probability abe_jelly_beans bob_jelly_beans = 5 / 21 := by
  sorry

end matching_jelly_bean_probability_l2206_220633


namespace suitcase_lock_settings_l2206_220631

/-- Represents a lock with a specified number of dials and digits per dial -/
structure Lock :=
  (numDials : ℕ)
  (digitsPerDial : ℕ)

/-- Calculates the number of different settings for a lock with all digits different -/
def countDifferentSettings (lock : Lock) : ℕ :=
  sorry

/-- The specific lock in the problem -/
def suitcaseLock : Lock :=
  { numDials := 3,
    digitsPerDial := 10 }

/-- Theorem stating that the number of different settings for the suitcase lock is 720 -/
theorem suitcase_lock_settings :
  countDifferentSettings suitcaseLock = 720 :=
sorry

end suitcase_lock_settings_l2206_220631


namespace envelope_height_l2206_220684

theorem envelope_height (width : ℝ) (area : ℝ) (height : ℝ) : 
  width = 6 → area = 36 → area = width * height → height = 6 := by
  sorry

end envelope_height_l2206_220684


namespace arithmetic_sequence_sum_l2206_220604

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * a₁ + n * (n - 1) * d / 2

theorem arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) :
  (sum_arithmetic_sequence a₁ d 12 / 12 - sum_arithmetic_sequence a₁ d 10 / 10 = 2) →
  (sum_arithmetic_sequence (-2008) d 2008 = -2008) :=
by
  sorry

end arithmetic_sequence_sum_l2206_220604


namespace forum_posts_l2206_220675

/-- A forum with members posting questions and answers -/
structure Forum where
  members : ℕ
  questions_per_hour : ℕ
  answer_ratio : ℕ

/-- Calculate the total number of questions posted in a day -/
def total_questions_per_day (f : Forum) : ℕ :=
  f.members * (f.questions_per_hour * 24)

/-- Calculate the total number of answers posted in a day -/
def total_answers_per_day (f : Forum) : ℕ :=
  f.members * (f.questions_per_hour * 24 * f.answer_ratio)

/-- Theorem stating the number of questions and answers posted in a day -/
theorem forum_posts (f : Forum) 
  (h1 : f.members = 200)
  (h2 : f.questions_per_hour = 3)
  (h3 : f.answer_ratio = 3) :
  total_questions_per_day f = 14400 ∧ total_answers_per_day f = 43200 := by
  sorry

end forum_posts_l2206_220675


namespace complex_argument_cube_l2206_220653

theorem complex_argument_cube (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 3)
  (h2 : Complex.abs z₂ = 5)
  (h3 : Complex.abs (z₁ + z₂) = 7) :
  Complex.arg ((z₂ / z₁) ^ 3) = π := by sorry

end complex_argument_cube_l2206_220653


namespace friends_team_assignments_l2206_220674

/-- The number of ways to assign n distinguishable objects to k distinct categories -/
def assignments (n : ℕ) (k : ℕ+) : ℕ := k.val ^ n

/-- Proof that for 8 friends and 3 teams, the number of assignments is 3^8 -/
theorem friends_team_assignments :
  assignments 8 3 = 6561 := by
  sorry

end friends_team_assignments_l2206_220674


namespace knicks_knacks_knocks_equivalence_l2206_220634

theorem knicks_knacks_knocks_equivalence 
  (h1 : (8 : ℚ) * knicks = (3 : ℚ) * knacks)
  (h2 : (4 : ℚ) * knacks = (5 : ℚ) * knocks) :
  (64 : ℚ) * knicks = (30 : ℚ) * knocks := by
  sorry

end knicks_knacks_knocks_equivalence_l2206_220634


namespace matrix_inverse_problem_l2206_220630

open Matrix

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem matrix_inverse_problem (B : Matrix n n ℝ) (h_inv : Invertible B) 
  (h_eq : (B - 3 • 1) * (B - 5 • 1) = 0) :
  B + 10 • B⁻¹ = (160 / 15 : ℝ) • 1 := by sorry

end matrix_inverse_problem_l2206_220630


namespace distance_between_points_l2206_220637

/-- The distance between points (0,15,5) and (8,0,12) in 3D space is √338. -/
theorem distance_between_points : Real.sqrt 338 = Real.sqrt ((8 - 0)^2 + (0 - 15)^2 + (12 - 5)^2) := by
  sorry

end distance_between_points_l2206_220637


namespace second_apartment_rent_l2206_220611

/-- Calculates the total monthly cost for an apartment --/
def total_monthly_cost (rent : ℚ) (utilities : ℚ) (miles_per_day : ℚ) (work_days : ℚ) (cost_per_mile : ℚ) : ℚ :=
  rent + utilities + (miles_per_day * work_days * cost_per_mile)

/-- The problem statement --/
theorem second_apartment_rent :
  let first_rent : ℚ := 800
  let first_utilities : ℚ := 260
  let first_miles : ℚ := 31
  let second_utilities : ℚ := 200
  let second_miles : ℚ := 21
  let work_days : ℚ := 20
  let cost_per_mile : ℚ := 58 / 100
  let cost_difference : ℚ := 76
  ∃ second_rent : ℚ,
    second_rent = 900 ∧
    total_monthly_cost first_rent first_utilities first_miles work_days cost_per_mile -
    total_monthly_cost second_rent second_utilities second_miles work_days cost_per_mile = cost_difference :=
by
  sorry


end second_apartment_rent_l2206_220611


namespace partial_fraction_decomposition_l2206_220651

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ),
    (∀ x : ℚ, x ≠ 1 ∧ x ≠ 4 ∧ x ≠ 6 →
      (x^2 - 8) / ((x - 1) * (x - 4) * (x - 6)) =
      P / (x - 1) + Q / (x - 4) + R / (x - 6)) ∧
    P = 7/15 ∧ Q = -4/3 ∧ R = 14/5 := by
  sorry

end partial_fraction_decomposition_l2206_220651


namespace fourth_root_of_fourth_power_l2206_220607

theorem fourth_root_of_fourth_power (a : ℝ) (h : a < 2) :
  (((a - 2) ^ 4) ^ (1/4 : ℝ)) = 2 - a := by
  sorry

end fourth_root_of_fourth_power_l2206_220607


namespace daniels_horses_l2206_220640

/-- The number of horses Daniel has -/
def num_horses : ℕ := 2

/-- The number of dogs Daniel has -/
def num_dogs : ℕ := 5

/-- The number of cats Daniel has -/
def num_cats : ℕ := 7

/-- The number of turtles Daniel has -/
def num_turtles : ℕ := 3

/-- The number of goats Daniel has -/
def num_goats : ℕ := 1

/-- The number of legs each animal has -/
def legs_per_animal : ℕ := 4

/-- The total number of legs of all animals -/
def total_legs : ℕ := 72

theorem daniels_horses :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = total_legs :=
by sorry

end daniels_horses_l2206_220640


namespace sum_of_reciprocal_relations_l2206_220652

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (eq1 : 1 / x + 1 / y = 4) 
  (eq2 : 1 / x - 1 / y = -5) : 
  x + y = -16 / 9 := by
sorry

end sum_of_reciprocal_relations_l2206_220652


namespace a_10_value_l2206_220615

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem a_10_value (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q → a 7 = 10 → q = -2 → a 10 = -80 := by
  sorry

end a_10_value_l2206_220615


namespace fat_per_serving_perrys_recipe_fat_per_serving_l2206_220663

/-- Calculates the grams of fat per serving in a sauce recipe. -/
theorem fat_per_serving (servings : ℕ) (total_mixture : ℝ) 
  (cream_ratio cheese_ratio butter_ratio : ℕ) 
  (cream_fat cheese_fat butter_fat : ℝ) : ℝ :=
  let total_ratio := cream_ratio + cheese_ratio + butter_ratio
  let part_size := total_mixture / total_ratio
  let cream_amount := part_size * cream_ratio
  let cheese_amount := part_size * cheese_ratio
  let butter_amount := part_size * butter_ratio * 2 -- Convert half-cups to cups
  let total_fat := cream_amount * cream_fat + cheese_amount * cheese_fat + butter_amount * butter_fat
  total_fat / servings

/-- The amount of fat per serving in Perry's recipe is approximately 37.65 grams. -/
theorem perrys_recipe_fat_per_serving : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |fat_per_serving 6 1.5 5 3 2 88 110 184 - 37.65| < ε :=
sorry

end fat_per_serving_perrys_recipe_fat_per_serving_l2206_220663


namespace no_solution_fractional_equation_l2206_220624

theorem no_solution_fractional_equation :
  ¬ ∃ (x : ℝ), x ≠ 5 ∧ (3 * x / (x - 5) + 15 / (5 - x) = 1) := by
  sorry

end no_solution_fractional_equation_l2206_220624


namespace arithmetic_calculation_l2206_220667

theorem arithmetic_calculation : 1273 + 120 / 60 - 173 = 1102 := by
  sorry

end arithmetic_calculation_l2206_220667


namespace student_distribution_theorem_l2206_220613

/-- The number of ways to distribute students into classes -/
def distribute_students (total_students : ℕ) (num_classes : ℕ) (must_be_together : ℕ) : ℕ :=
  sorry

/-- The theorem to prove -/
theorem student_distribution_theorem :
  distribute_students 5 3 2 = 36 :=
sorry

end student_distribution_theorem_l2206_220613


namespace number_of_schnauzers_l2206_220671

theorem number_of_schnauzers : ℕ := by
  -- Define the number of Doberman puppies
  let doberman : ℕ := 20

  -- Define the equation from the problem
  let equation (s : ℕ) : Prop := 3 * doberman - 5 + (doberman - s) = 90

  -- Assert that the equation holds for s = 55
  have h : equation 55 := by sorry

  -- Prove that 55 is the unique solution
  have unique : ∀ s : ℕ, equation s → s = 55 := by sorry

  -- Conclude that the number of Schnauzers is 55
  exact 55

end number_of_schnauzers_l2206_220671


namespace degree_to_radian_conversion_l2206_220682

theorem degree_to_radian_conversion (π : Real) : 
  ((-300 : Real) * (π / 180)) = (-5 * π / 3) := by sorry

end degree_to_radian_conversion_l2206_220682


namespace women_average_age_l2206_220686

theorem women_average_age (n : ℕ) (A : ℝ) (W₁ W₂ : ℝ) :
  n = 7 ∧ 
  (n * A - 26 - 30 + W₁ + W₂) / n = A + 4 →
  (W₁ + W₂) / 2 = 42 := by
sorry

end women_average_age_l2206_220686


namespace polynomial_no_integral_roots_l2206_220625

/-- A polynomial with integral coefficients that has odd integer values at 0 and 1 has no integral roots. -/
theorem polynomial_no_integral_roots 
  (p : Polynomial ℤ) 
  (h0 : Odd (p.eval 0)) 
  (h1 : Odd (p.eval 1)) : 
  ∀ (x : ℤ), p.eval x ≠ 0 := by
sorry

end polynomial_no_integral_roots_l2206_220625


namespace coinciding_rest_days_count_l2206_220626

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 6

/-- Carol's schedule cycle length -/
def carol_cycle : ℕ := 6

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Al's rest days in a cycle -/
def al_rest_days : Finset ℕ := {5, 6}

/-- Carol's rest days in a cycle -/
def carol_rest_days : Finset ℕ := {6}

/-- The number of days both Al and Carol have rest-days on the same day -/
def coinciding_rest_days : ℕ := (al_rest_days ∩ carol_rest_days).card * (total_days / al_cycle)

theorem coinciding_rest_days_count : coinciding_rest_days = 166 := by
  sorry

end coinciding_rest_days_count_l2206_220626


namespace binomial_square_equivalence_l2206_220680

theorem binomial_square_equivalence (x y : ℝ) : 
  (-x - y) * (-x + y) = (-x - y)^2 := by sorry

end binomial_square_equivalence_l2206_220680


namespace quadratic_equation_roots_l2206_220614

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x + 3 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + m*y + 3 = 0 ∧ y = -3 ∧ m = 4) :=
by sorry

end quadratic_equation_roots_l2206_220614


namespace unique_prime_between_squares_l2206_220695

theorem unique_prime_between_squares : ∃! p : ℕ, 
  Prime p ∧ 
  ∃ n : ℕ, p = n^2 + 9 ∧ 
  ∃ m : ℕ, p + 2 = m^2 ∧
  m = n + 1 :=
by
  sorry

end unique_prime_between_squares_l2206_220695


namespace lcm_triple_count_l2206_220656

/-- The least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ+ := sorry

/-- The number of ordered triples (a, b, c) satisfying the LCM conditions -/
def count_triples : ℕ := sorry

/-- Main theorem: There are exactly 70 ordered triples satisfying the LCM conditions -/
theorem lcm_triple_count :
  count_triples = 70 :=
by sorry

end lcm_triple_count_l2206_220656


namespace root_sum_reciprocal_l2206_220619

-- Define the polynomial
def p (x : ℝ) : ℝ := 42 * x^3 - 35 * x^2 + 10 * x - 1

-- Define the roots
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem root_sum_reciprocal :
  p a = 0 ∧ p b = 0 ∧ p c = 0 ∧   -- a, b, c are roots of p
  0 < a ∧ a < 1 ∧                 -- a is between 0 and 1
  0 < b ∧ b < 1 ∧                 -- b is between 0 and 1
  0 < c ∧ c < 1 ∧                 -- c is between 0 and 1
  a ≠ b ∧ b ≠ c ∧ a ≠ c →         -- roots are distinct
  1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) = 2.875 := by
  sorry

end root_sum_reciprocal_l2206_220619


namespace a_2018_value_l2206_220643

def triangle_sequence (A : ℕ → ℝ) : ℕ → ℝ := λ n => A (n + 1) - A n

theorem a_2018_value (A : ℕ → ℝ) 
  (h1 : ∀ n, triangle_sequence (triangle_sequence A) n = 1)
  (h2 : A 18 = 0)
  (h3 : A 2017 = 0) :
  A 2018 = 1000 := by
  sorry

end a_2018_value_l2206_220643


namespace pencils_remaining_l2206_220694

/-- Given a box of pencils with an initial count and a number of pencils taken,
    prove that the remaining number of pencils is the difference between the initial count and the number taken. -/
theorem pencils_remaining (initial_count taken : ℕ) : 
  initial_count = 79 → taken = 4 → initial_count - taken = 75 := by
  sorry

end pencils_remaining_l2206_220694


namespace parabola_vertex_l2206_220668

/-- The parabola is defined by the equation y = 2(x-5)^2 + 3 -/
def parabola (x y : ℝ) : Prop := y = 2 * (x - 5)^2 + 3

/-- The vertex of a parabola is the point where it reaches its minimum or maximum -/
def is_vertex (x₀ y₀ : ℝ) : Prop :=
  ∀ x y, parabola x y → y ≥ y₀

/-- Theorem: The vertex of the parabola y = 2(x-5)^2 + 3 has coordinates (5, 3) -/
theorem parabola_vertex :
  is_vertex 5 3 :=
sorry

end parabola_vertex_l2206_220668


namespace forgotten_angles_sum_l2206_220601

/-- The sum of interior angles of a polygon with n sides --/
def polygon_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

/-- A convex polygon with n sides where n ≥ 3 --/
structure ConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

theorem forgotten_angles_sum (p : ConvexPolygon) 
  (partial_sum : ℝ) (h_partial_sum : partial_sum = 2345) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 175 ∧ 
  polygon_angle_sum p.n = partial_sum + a + b := by
  sorry

end forgotten_angles_sum_l2206_220601


namespace arithmetic_to_geometric_sequence_l2206_220620

theorem arithmetic_to_geometric_sequence (a d : ℝ) : 
  (2 * (a - d)) * ((a + d) + 7) = a^2 ∧ 
  (a - d) * a * (a + d) = 1000 →
  d = 8 ∨ d = -15 := by sorry

end arithmetic_to_geometric_sequence_l2206_220620


namespace product_mod_450_l2206_220678

theorem product_mod_450 : (2011 * 1537) % 450 = 307 := by
  sorry

end product_mod_450_l2206_220678


namespace number_difference_l2206_220650

theorem number_difference (S L : ℕ) (h1 : S = 270) (h2 : L = 6 * S + 15) :
  L - S = 1365 := by
  sorry

end number_difference_l2206_220650


namespace mod_inverse_sum_l2206_220644

theorem mod_inverse_sum : ∃ (a b : ℤ), 
  (5 * a) % 35 = 1 ∧ 
  (15 * b) % 35 = 1 ∧ 
  (a + b) % 35 = 21 := by
  sorry

end mod_inverse_sum_l2206_220644


namespace smallest_number_greater_than_0_4_l2206_220689

theorem smallest_number_greater_than_0_4 (S : Set ℝ) : 
  S = {0.8, 1/2, 0.3, 1/3} → 
  (∃ x ∈ S, x > 0.4 ∧ ∀ y ∈ S, y > 0.4 → x ≤ y) → 
  (1/2 ∈ S ∧ 1/2 > 0.4 ∧ ∀ y ∈ S, y > 0.4 → 1/2 ≤ y) :=
by sorry

end smallest_number_greater_than_0_4_l2206_220689


namespace days_to_watch_all_episodes_l2206_220628

-- Define the number of episodes for each season type
def regular_season_episodes : ℕ := 22
def third_season_episodes : ℕ := 24
def last_season_episodes : ℕ := regular_season_episodes + 4

-- Define the duration of episodes for different seasons
def early_episode_duration : ℚ := 1/2
def later_episode_duration : ℚ := 3/4

-- Define John's daily watching time
def daily_watching_time : ℚ := 2

-- Define the total number of seasons
def total_seasons : ℕ := 10

-- Define a function to calculate the total viewing time
def total_viewing_time : ℚ :=
  let early_seasons_episodes : ℕ := 2 * regular_season_episodes + third_season_episodes
  let early_seasons_time : ℚ := early_seasons_episodes * early_episode_duration
  let later_seasons_episodes : ℕ := (total_seasons - 3) * regular_season_episodes + last_season_episodes
  let later_seasons_time : ℚ := later_seasons_episodes * later_episode_duration
  early_seasons_time + later_seasons_time

-- Theorem statement
theorem days_to_watch_all_episodes :
  ⌈total_viewing_time / daily_watching_time⌉ = 77 := by sorry

end days_to_watch_all_episodes_l2206_220628


namespace gcd_problem_l2206_220672

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = (2 * k + 1) * 8531) :
  Int.gcd (8 * b^2 + 33 * b + 125) (4 * b + 15) = 5 := by
  sorry

end gcd_problem_l2206_220672


namespace fifth_house_gnomes_l2206_220657

/-- The number of houses on the street -/
def num_houses : Nat := 5

/-- The number of gnomes in each of the first four houses -/
def gnomes_per_house : Nat := 3

/-- The total number of gnomes on the street -/
def total_gnomes : Nat := 20

/-- The number of gnomes in the fifth house -/
def gnomes_in_fifth_house : Nat := total_gnomes - (4 * gnomes_per_house)

theorem fifth_house_gnomes :
  gnomes_in_fifth_house = 8 := by sorry

end fifth_house_gnomes_l2206_220657


namespace donghwan_candies_l2206_220659

theorem donghwan_candies (total_candies bag_size : ℕ) 
  (h1 : total_candies = 138)
  (h2 : bag_size = 18) :
  total_candies % bag_size = 12 := by
  sorry

end donghwan_candies_l2206_220659


namespace point_on_unit_circle_l2206_220618

theorem point_on_unit_circle (Q : ℝ × ℝ) : 
  (Q.1^2 + Q.2^2 = 1) →  -- Q is on the unit circle
  (Q.1 = -1/2 ∧ Q.2 = -Real.sqrt 3/2) ↔ 
  (∃ θ : ℝ, θ = -2*Real.pi/3 ∧ Q.1 = Real.cos θ ∧ Q.2 = Real.sin θ) :=
by sorry

end point_on_unit_circle_l2206_220618


namespace invisible_square_existence_l2206_220661

theorem invisible_square_existence (n : ℕ+) :
  ∃ (x y : ℤ), ∀ (i j : ℕ), 0 < i ∧ i ≤ n ∧ 0 < j ∧ j ≤ n →
    Nat.gcd (Int.natAbs (x + i)) (Int.natAbs (y + j)) > 1 := by
  sorry

end invisible_square_existence_l2206_220661


namespace solve_system_l2206_220666

theorem solve_system (x y z w : ℤ)
  (eq1 : x + y = 4)
  (eq2 : x - y = 36)
  (eq3 : x * z + y * w = 50)
  (eq4 : z - w = 5) :
  x = 20 := by
sorry

end solve_system_l2206_220666


namespace function_inequality_l2206_220610

theorem function_inequality (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ x, (x - 1) * (deriv f x) < 0) : 
  f 0 + f 2 < 2 * f 1 := by
  sorry

end function_inequality_l2206_220610


namespace product_difference_equals_2019_l2206_220692

theorem product_difference_equals_2019 : 672 * 673 * 674 - 671 * 673 * 675 = 2019 := by
  sorry

end product_difference_equals_2019_l2206_220692


namespace four_tangent_circles_l2206_220636

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum of their radii --/
def are_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

/-- A circle is tangent to two other circles --/
def is_tangent_to_both (c c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

theorem four_tangent_circles (c1 c2 : Circle)
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 2)
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), s.card = 4 ∧ ∀ c ∈ s, c.radius = 3 ∧ is_tangent_to_both c c1 c2 :=
sorry

end four_tangent_circles_l2206_220636


namespace roger_trays_capacity_l2206_220658

/-- The number of trays Roger can carry at a time -/
def trays_per_trip : ℕ := sorry

/-- The number of trips Roger made -/
def num_trips : ℕ := 3

/-- The total number of trays Roger picked up -/
def total_trays : ℕ := 12

theorem roger_trays_capacity :
  trays_per_trip = 4 ∧ 
  num_trips * trays_per_trip = total_trays :=
by sorry

end roger_trays_capacity_l2206_220658


namespace exponential_function_fixed_point_l2206_220681

theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 3) - 5
  f (3/2) = -4 := by
  sorry

end exponential_function_fixed_point_l2206_220681


namespace max_n_for_int_polynomial_l2206_220627

/-- A polynomial with integer coefficients -/
def IntPolynomial := Polynomial ℤ

/-- The property that P(aᵢ) = i for all 1 ≤ i ≤ n -/
def SatisfiesProperty (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (a : ℕ → ℤ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → P.eval (a i) = i

/-- The theorem stating the maximum n for which the property holds -/
theorem max_n_for_int_polynomial (P : IntPolynomial) (h : P.degree = 2022) :
    (∃ n : ℕ, SatisfiesProperty P n ∧ ∀ m : ℕ, SatisfiesProperty P m → m ≤ n) ∧
    (∃ n : ℕ, n = 2022 ∧ SatisfiesProperty P n) :=
  sorry

end max_n_for_int_polynomial_l2206_220627


namespace rd_scenario_theorem_l2206_220623

/-- Represents a firm in the R&D scenario -/
structure Firm where
  participates : Bool

/-- Represents the R&D scenario -/
structure RDScenario where
  V : ℝ  -- Revenue if successful
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  firms : Fin 2 → Firm

/-- Expected revenue for a firm when both participate -/
def expectedRevenueBoth (s : RDScenario) : ℝ :=
  s.α * (1 - s.α) * s.V + 0.5 * s.α^2 * s.V

/-- Expected revenue for a firm when only one participates -/
def expectedRevenueOne (s : RDScenario) : ℝ :=
  s.α * s.V

/-- Condition for both firms to participate -/
def bothParticipateCondition (s : RDScenario) : Prop :=
  s.V * s.α * (1 - 0.5 * s.α) ≥ s.IC

/-- Total profit when both firms participate -/
def totalProfitBoth (s : RDScenario) : ℝ :=
  2 * (expectedRevenueBoth s - s.IC)

/-- Total profit when only one firm participates -/
def totalProfitOne (s : RDScenario) : ℝ :=
  expectedRevenueOne s - s.IC

/-- The main theorem to prove -/
theorem rd_scenario_theorem (s : RDScenario) 
    (h1 : 0 < s.α ∧ s.α < 1) 
    (h2 : s.V > 0) 
    (h3 : s.IC > 0) : 
  (bothParticipateCondition s ↔ expectedRevenueBoth s ≥ s.IC) ∧
  (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → bothParticipateCondition s) ∧
  (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → totalProfitOne s > totalProfitBoth s) := by
  sorry

end rd_scenario_theorem_l2206_220623


namespace prime_power_square_sum_l2206_220664

theorem prime_power_square_sum (p n k : ℕ) : 
  p.Prime → p > 0 → n > 0 → k > 0 → 144 + p^n = k^2 →
  ((p = 5 ∧ n = 2 ∧ k = 13) ∨ (p = 2 ∧ n = 8 ∧ k = 20) ∨ (p = 3 ∧ n = 4 ∧ k = 15)) :=
by sorry

end prime_power_square_sum_l2206_220664


namespace sum_of_roots_equation_l2206_220609

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, (a + 2) * (a - 3) = 20 ∧ (b + 2) * (b - 3) = 20 ∧ a + b = 1) := by
  sorry

end sum_of_roots_equation_l2206_220609


namespace sin_cos_shift_l2206_220696

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x - π / 4) = Real.cos (2 * (x - 3 * π / 8)) := by
  sorry

end sin_cos_shift_l2206_220696


namespace mn_square_value_l2206_220655

theorem mn_square_value (m n : ℤ) 
  (h1 : |m - n| = n - m) 
  (h2 : |m| = 4) 
  (h3 : |n| = 3) : 
  (m + n)^2 = 1 ∨ (m + n)^2 = 49 := by
  sorry

end mn_square_value_l2206_220655


namespace symmetric_parabola_b_eq_six_l2206_220698

/-- A function f(x) = x^2 + (a+2)x + 3 with domain [a, b] that is symmetric about x = 1 -/
def symmetric_parabola (a b : ℝ) : Prop :=
  ∃ f : ℝ → ℝ, 
    (∀ x ∈ Set.Icc a b, f x = x^2 + (a+2)*x + 3) ∧ 
    (∀ x ∈ Set.Icc a b, f (2 - x) = f x)

/-- If a parabola is symmetric about x = 1, then b = 6 -/
theorem symmetric_parabola_b_eq_six (a b : ℝ) :
  symmetric_parabola a b → b = 6 :=
by sorry

end symmetric_parabola_b_eq_six_l2206_220698


namespace race_head_start_l2206_220617

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (20 / 14) * Vb) :
  ∃ H : ℝ, H = (3 / 10) * L ∧ L / Va = (L - H) / Vb :=
sorry

end race_head_start_l2206_220617


namespace min_translation_for_symmetry_l2206_220635

/-- The minimum translation that makes sin(3x + π/6) symmetric about y-axis -/
theorem min_translation_for_symmetry :
  let f (x : ℝ) := Real.sin (3 * x + π / 6)
  ∃ (m : ℝ), m > 0 ∧
    (∀ (x : ℝ), f (x - m) = f (-x - m) ∨ f (x + m) = f (-x + m)) ∧
    (∀ (m' : ℝ), m' > 0 → 
      (∀ (x : ℝ), f (x - m') = f (-x - m') ∨ f (x + m') = f (-x + m')) →
      m ≤ m') ∧
    m = π / 9 := by
  sorry

end min_translation_for_symmetry_l2206_220635


namespace ticket_difference_l2206_220608

theorem ticket_difference (fair_tickets : ℕ) (baseball_tickets : ℕ)
  (h1 : fair_tickets = 25)
  (h2 : baseball_tickets = 56) :
  2 * baseball_tickets - fair_tickets = 87 := by
  sorry

end ticket_difference_l2206_220608


namespace train_journey_theorem_l2206_220693

/-- Represents the train's journey with two potential accident scenarios -/
theorem train_journey_theorem (D v : ℝ) : 
  (D > 0) →  -- Distance is positive
  (v > 0) →  -- Speed is positive
  -- First accident scenario
  (2 + 1 + (3 * (D - 2*v)) / (2*v) = D/v + 4) → 
  -- Second accident scenario
  (2.5 + 120/v + (6 * (D - 2*v - 120)) / (5*v) = D/v + 3.5) → 
  -- The distance D is one of the given choices
  (D = 420 ∨ D = 480 ∨ D = 540 ∨ D = 600 ∨ D = 660) :=
by sorry


end train_journey_theorem_l2206_220693


namespace scientific_notation_equivalence_l2206_220685

-- Define the original number
def original_number : ℝ := 1300000

-- Define the scientific notation components
def coefficient : ℝ := 1.3
def exponent : ℕ := 6

-- Theorem statement
theorem scientific_notation_equivalence :
  original_number = coefficient * (10 : ℝ) ^ exponent := by
  sorry

end scientific_notation_equivalence_l2206_220685


namespace unique_prime_pair_l2206_220679

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Nat.Prime p ∧ 
  Nat.Prime q ∧ 
  Nat.Prime (p + q) ∧ 
  Nat.Prime (p^2 + q^2 - q) ∧ 
  p = 3 ∧ 
  q = 2 := by
sorry

end unique_prime_pair_l2206_220679


namespace rectangles_with_one_gray_count_l2206_220690

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents the count of different types of cells in the grid -/
structure CellCount :=
  (total_gray : ℕ)
  (interior_gray : ℕ)
  (edge_gray : ℕ)

/-- Calculates the number of rectangles containing exactly one gray cell -/
def count_rectangles_with_one_gray (g : Grid) (c : CellCount) : ℕ :=
  c.interior_gray * 4 + c.edge_gray * 8

/-- The main theorem stating the number of rectangles with one gray cell -/
theorem rectangles_with_one_gray_count 
  (g : Grid) 
  (c : CellCount) 
  (h1 : g.rows = 5) 
  (h2 : g.cols = 22) 
  (h3 : c.total_gray = 40) 
  (h4 : c.interior_gray = 36) 
  (h5 : c.edge_gray = 4) :
  count_rectangles_with_one_gray g c = 176 := by
  sorry

#check rectangles_with_one_gray_count

end rectangles_with_one_gray_count_l2206_220690


namespace complement_union_original_equals_universe_l2206_220649

-- Define the universe set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {1, 2, 4}

-- Define set C as the complement of M in U
def C : Finset Nat := U \ M

-- Theorem statement
theorem complement_union_original_equals_universe :
  C ∪ M = U := by sorry

end complement_union_original_equals_universe_l2206_220649


namespace cost_to_fly_AB_l2206_220646

/-- The cost of flying between two cities -/
def flying_cost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

/-- The distance from A to B in kilometers -/
def distance_AB : ℝ := 4500

theorem cost_to_fly_AB : flying_cost distance_AB = 660 := by
  sorry

end cost_to_fly_AB_l2206_220646


namespace z_squared_minus_four_z_is_real_l2206_220669

/-- Given a real number a and a complex number z = 2 + ai, 
    prove that z^2 - 4z is a real number. -/
theorem z_squared_minus_four_z_is_real (a : ℝ) : 
  let z : ℂ := 2 + a * Complex.I
  (z^2 - 4*z).im = 0 := by sorry

end z_squared_minus_four_z_is_real_l2206_220669


namespace infinite_set_sum_of_digits_squared_equal_l2206_220647

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Proposition: There exists an infinite set of natural numbers n, not ending in 0,
    such that the sum of digits of n^2 equals the sum of digits of n -/
theorem infinite_set_sum_of_digits_squared_equal :
  ∃ (S : Set ℕ), Set.Infinite S ∧ 
    (∀ n ∈ S, n % 10 ≠ 0 ∧ sum_of_digits (n^2) = sum_of_digits n) :=
sorry

end infinite_set_sum_of_digits_squared_equal_l2206_220647


namespace chord_length_on_xaxis_l2206_220600

/-- The length of the chord intercepted by the x-axis on the circle (x-1)^2+(y-1)^2=2 is 2 -/
theorem chord_length_on_xaxis (x y : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    ((x₁ - 1)^2 + (0 - 1)^2 = 2) ∧ 
    ((x₂ - 1)^2 + (0 - 1)^2 = 2) ∧ 
    (x₂ - x₁ = 2)) :=
by sorry

end chord_length_on_xaxis_l2206_220600


namespace acute_angles_sum_l2206_220605

theorem acute_angles_sum (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end acute_angles_sum_l2206_220605


namespace roots_relation_l2206_220697

theorem roots_relation (a b c d : ℝ) : 
  (∀ x, (x - a) * (x - b) - x = 0 ↔ x = c ∨ x = d) →
  (∀ x, (x - c) * (x - d) + x = 0 ↔ x = a ∨ x = b) :=
by sorry

end roots_relation_l2206_220697


namespace sum_plus_five_mod_seven_l2206_220662

/-- Sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The problem statement -/
theorem sum_plus_five_mod_seven :
  (sum_to_n 99 + 5) % 7 = 6 := by
  sorry

end sum_plus_five_mod_seven_l2206_220662


namespace days_per_month_is_30_l2206_220602

/-- Represents the number of trees a single logger can cut down in one day. -/
def trees_per_logger_per_day : ℕ := 6

/-- Represents the length of the forest in miles. -/
def forest_length : ℕ := 4

/-- Represents the width of the forest in miles. -/
def forest_width : ℕ := 6

/-- Represents the number of trees in each square mile of the forest. -/
def trees_per_square_mile : ℕ := 600

/-- Represents the number of loggers working on cutting down the trees. -/
def num_loggers : ℕ := 8

/-- Represents the number of months it takes to cut down all trees. -/
def num_months : ℕ := 10

/-- Theorem stating that the number of days in each month is 30. -/
theorem days_per_month_is_30 :
  ∃ (days_per_month : ℕ),
    days_per_month = 30 ∧
    (forest_length * forest_width * trees_per_square_mile =
     num_loggers * trees_per_logger_per_day * num_months * days_per_month) :=
by sorry

end days_per_month_is_30_l2206_220602
