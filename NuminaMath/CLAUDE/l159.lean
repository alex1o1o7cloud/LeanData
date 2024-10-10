import Mathlib

namespace vector_addition_path_l159_15956

-- Define a 2D vector
def Vector2D := ℝ × ℝ

-- Define vector addition
def vec_add (v w : Vector2D) : Vector2D :=
  (v.1 + w.1, v.2 + w.2)

-- Define vector from point to point
def vec_from_to (A B : Vector2D) : Vector2D :=
  (B.1 - A.1, B.2 - A.2)

-- Theorem statement
theorem vector_addition_path (A B C D : Vector2D) :
  vec_add (vec_add (vec_from_to A B) (vec_from_to B C)) (vec_from_to C D) =
  vec_from_to A D :=
by sorry

end vector_addition_path_l159_15956


namespace exists_ring_configuration_l159_15983

/-- A structure representing a configuration of connected rings -/
structure RingConfiguration (n : ℕ) where
  rings : Fin n → Bool
  connected : Bool
  
/-- A function that simulates cutting a ring from the configuration -/
def cut_ring (config : RingConfiguration n) (i : Fin n) : RingConfiguration n :=
  { rings := λ j => if j = i then false else config.rings j,
    connected := false }

/-- The property that a ring configuration satisfies the problem conditions -/
def satisfies_conditions (config : RingConfiguration n) : Prop :=
  (n ≥ 3) ∧
  config.connected ∧
  (∀ i : Fin n, ¬(cut_ring config i).connected)

/-- The main theorem stating that for any number of rings ≥ 3, 
    there exists a configuration satisfying the problem conditions -/
theorem exists_ring_configuration (n : ℕ) (h : n ≥ 3) :
  ∃ (config : RingConfiguration n), satisfies_conditions config :=
sorry

end exists_ring_configuration_l159_15983


namespace bee_colony_reduction_l159_15963

/-- Prove that a bee colony with given initial size and daily loss rate
    reaches 1/4 of its initial size in the calculated number of days. -/
theorem bee_colony_reduction (initial_size : ℕ) (daily_loss : ℕ) (days : ℕ) :
  initial_size = 80000 →
  daily_loss = 1200 →
  days = 50 →
  initial_size - days * daily_loss = initial_size / 4 :=
by sorry

end bee_colony_reduction_l159_15963


namespace carlo_practice_difference_l159_15973

/-- Represents Carlo's practice schedule for a week --/
structure PracticeSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Theorem about Carlo's practice schedule --/
theorem carlo_practice_difference (schedule : PracticeSchedule) :
  schedule.monday = 2 * schedule.tuesday ∧
  schedule.tuesday < schedule.wednesday ∧
  schedule.wednesday = schedule.thursday + 5 ∧
  schedule.thursday = 50 ∧
  schedule.friday = 60 ∧
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday = 300 →
  schedule.wednesday - schedule.tuesday = 10 := by
  sorry

end carlo_practice_difference_l159_15973


namespace vector_subtraction_magnitude_l159_15925

def vector_a : ℝ × ℝ := (-2, 2)

theorem vector_subtraction_magnitude (b : ℝ × ℝ) 
  (h1 : ‖b‖ = 1) 
  (h2 : vector_a • b = 2) : ‖vector_a - 2 • b‖ = 2 := by
  sorry

end vector_subtraction_magnitude_l159_15925


namespace gears_can_rotate_l159_15937

/-- A gear system with n identical gears arranged in a closed loop. -/
structure GearSystem where
  n : ℕ
  is_closed_loop : n ≥ 2

/-- Represents the rotation direction of a gear. -/
inductive RotationDirection
  | Clockwise
  | Counterclockwise

/-- Function to determine if adjacent gears have opposite rotation directions. -/
def opposite_rotation (d1 d2 : RotationDirection) : Prop :=
  (d1 = RotationDirection.Clockwise ∧ d2 = RotationDirection.Counterclockwise) ∨
  (d1 = RotationDirection.Counterclockwise ∧ d2 = RotationDirection.Clockwise)

/-- Theorem stating that the gears can rotate if and only if the number of gears is even. -/
theorem gears_can_rotate (system : GearSystem) :
  (∃ (rotation : ℕ → RotationDirection), 
    (∀ i : ℕ, i < system.n → opposite_rotation (rotation i) (rotation ((i + 1) % system.n))) ∧
    opposite_rotation (rotation 0) (rotation (system.n - 1)))
  ↔ 
  Even system.n :=
sorry

end gears_can_rotate_l159_15937


namespace clare_remaining_money_l159_15906

/-- Calculates the remaining money after Clare's purchases -/
def remaining_money (initial_amount bread_price milk_price cereal_price apple_price : ℕ) : ℕ :=
  let bread_cost := 4 * bread_price
  let milk_cost := 2 * milk_price
  let cereal_cost := 3 * cereal_price
  let apple_cost := apple_price
  let total_cost := bread_cost + milk_cost + cereal_cost + apple_cost
  initial_amount - total_cost

/-- Proves that Clare has $22 left after her purchases -/
theorem clare_remaining_money :
  remaining_money 47 2 2 3 4 = 22 := by
  sorry

end clare_remaining_money_l159_15906


namespace speaking_orders_eq_264_l159_15944

/-- The number of students in the group -/
def total_students : ℕ := 6

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

/-- Function to calculate the number of different speaking orders -/
def speaking_orders : ℕ :=
  -- Case 1: Either A or B participates
  (special_students * (total_students - special_students).choose (selected_students - 1) * selected_students.factorial) +
  -- Case 2: Both A and B participate
  ((total_students - special_students).choose (selected_students - special_students) * special_students.factorial * (selected_students - 1).factorial)

/-- Theorem stating that the number of different speaking orders is 264 -/
theorem speaking_orders_eq_264 : speaking_orders = 264 := by
  sorry

end speaking_orders_eq_264_l159_15944


namespace x_coordinate_difference_on_line_l159_15911

/-- Prove that for any two points on the line x = 4y + 5, where the y-coordinates differ by 0.5,
    the difference between their x-coordinates is 2. -/
theorem x_coordinate_difference_on_line (m n : ℝ) : 
  (m = 4 * n + 5) → 
  (∃ x, x = 4 * (n + 0.5) + 5) → 
  (∃ x, x - m = 2) :=
by sorry

end x_coordinate_difference_on_line_l159_15911


namespace least_prime_factor_of_9_4_minus_9_3_l159_15918

theorem least_prime_factor_of_9_4_minus_9_3 :
  Nat.minFac (9^4 - 9^3) = 2 := by sorry

end least_prime_factor_of_9_4_minus_9_3_l159_15918


namespace line_inclination_angle_l159_15968

theorem line_inclination_angle (a : ℝ) : 
  (∃ y : ℝ → ℝ, ∀ x, a * x - y x - 1 = 0) →  -- line equation
  (Real.tan (π / 3) = a) →                   -- angle of inclination
  a = Real.sqrt 3 :=                         -- conclusion
by sorry

end line_inclination_angle_l159_15968


namespace sqrt_x_div_sqrt_y_l159_15927

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/2)^2 + (1/3)^2 = ((1/3)^2 + (1/6)^2) * (13*x)/(47*y)) :
  Real.sqrt x / Real.sqrt y = Real.sqrt 47 / Real.sqrt 5 := by
  sorry

end sqrt_x_div_sqrt_y_l159_15927


namespace percent_of_x_is_z_l159_15957

theorem percent_of_x_is_z (x y z w : ℝ) 
  (h1 : 0.45 * z = 0.72 * y)
  (h2 : y = 0.75 * x)
  (h3 : w = 0.60 * z^2)
  (h4 : z = 0.30 * w^(1/3)) :
  z = 1.2 * x := by
sorry

end percent_of_x_is_z_l159_15957


namespace remaining_reading_time_l159_15940

/-- Calculates the remaining reading time for Sunday given the total assigned time and time spent reading on Friday and Saturday. -/
theorem remaining_reading_time 
  (total_assigned : ℕ) 
  (friday_reading : ℕ) 
  (saturday_reading : ℕ) 
  (h1 : total_assigned = 60)  -- 1 hour = 60 minutes
  (h2 : friday_reading = 16)
  (h3 : saturday_reading = 28) :
  total_assigned - (friday_reading + saturday_reading) = 16 :=
by sorry

#check remaining_reading_time

end remaining_reading_time_l159_15940


namespace friends_meeting_movie_and_games_l159_15990

theorem friends_meeting_movie_and_games 
  (total : ℕ) 
  (movie : ℕ) 
  (picnic : ℕ) 
  (games : ℕ) 
  (movie_and_picnic : ℕ) 
  (picnic_and_games : ℕ) 
  (all_three : ℕ) 
  (h1 : total = 31)
  (h2 : movie = 10)
  (h3 : picnic = 20)
  (h4 : games = 5)
  (h5 : movie_and_picnic = 4)
  (h6 : picnic_and_games = 0)
  (h7 : all_three = 2) : 
  ∃ (movie_and_games : ℕ), 
    total = movie + picnic + games - movie_and_picnic - movie_and_games - picnic_and_games + all_three ∧ 
    movie_and_games = 2 := by
  sorry

end friends_meeting_movie_and_games_l159_15990


namespace correct_card_order_l159_15967

/-- Represents a card in the arrangement --/
inductive Card : Type
  | A | B | C | D | E | F

/-- Represents the relative position of two cards --/
inductive Position
  | Above
  | Below
  | SameLevel

/-- Determines the relative position of two cards based on their overlaps --/
def relative_position (c1 c2 : Card) : Position := sorry

/-- Represents the final ordering of cards --/
def card_order : List Card := sorry

/-- Theorem stating the correct order of cards --/
theorem correct_card_order :
  card_order = [Card.F, Card.E, Card.A, Card.D, Card.C, Card.B] ∧
  relative_position Card.E Card.A = Position.SameLevel ∧
  relative_position Card.A Card.E = Position.SameLevel :=
sorry

end correct_card_order_l159_15967


namespace min_tiles_to_cover_classroom_l159_15924

/-- Represents the dimensions of a rectangular area in centimeters -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents a tile with its area in square centimeters -/
structure Tile where
  area : ℕ

def classroom : Dimensions := ⟨624, 432⟩
def rectangular_tile : Tile := ⟨60 * 80⟩
def triangular_tile : Tile := ⟨40 * 40 / 2⟩

def tiles_needed (room : Dimensions) (tile : Tile) : ℕ :=
  (room.length * room.width + tile.area - 1) / tile.area

theorem min_tiles_to_cover_classroom :
  min (tiles_needed classroom rectangular_tile) (tiles_needed classroom triangular_tile) = 57 := by
  sorry

end min_tiles_to_cover_classroom_l159_15924


namespace unique_solution_equation_l159_15965

theorem unique_solution_equation (x p q : ℕ) : 
  x ≥ 2 ∧ p ≥ 2 ∧ q ≥ 2 →
  ((x + 1) ^ p - x ^ q = 1) ↔ (x = 2 ∧ p = 2 ∧ q = 3) :=
by sorry

end unique_solution_equation_l159_15965


namespace circle_diameter_from_intersection_l159_15978

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the resulting circle
def result_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define a function to represent a point
def Point := ℝ × ℝ

-- Theorem statement
theorem circle_diameter_from_intersection :
  ∃ (P Q : Point),
    (given_circle P.1 P.2 ∧ given_line P.1 P.2) ∧
    (given_circle Q.1 Q.2 ∧ given_line Q.1 Q.2) ∧
    (∀ (x y : ℝ), result_circle x y ↔ 
      ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
        x = P.1 * (1 - t) + Q.1 * t ∧
        y = P.2 * (1 - t) + Q.2 * t) :=
sorry

end circle_diameter_from_intersection_l159_15978


namespace simplify_like_terms_l159_15952

theorem simplify_like_terms (x : ℝ) : 3*x + 5*x + 7*x = 15*x := by
  sorry

end simplify_like_terms_l159_15952


namespace ellipse_minor_axis_length_l159_15964

/-- An ellipse with a focus and distances to vertices -/
structure Ellipse :=
  (F : ℝ × ℝ)  -- Focus
  (d1 : ℝ)     -- Distance from focus to first vertex
  (d2 : ℝ)     -- Distance from focus to second vertex

/-- The length of the minor axis of the ellipse -/
def minorAxisLength (E : Ellipse) : ℝ := sorry

/-- Theorem: If the distances from the focus to the vertices are 1 and 9, 
    then the minor axis length is 6 -/
theorem ellipse_minor_axis_length 
  (E : Ellipse) 
  (h1 : E.d1 = 1) 
  (h2 : E.d2 = 9) : 
  minorAxisLength E = 6 := by sorry

end ellipse_minor_axis_length_l159_15964


namespace square_area_6cm_l159_15980

/-- The area of a square with side length 6 cm is 36 square centimeters. -/
theorem square_area_6cm : 
  let side_length : ℝ := 6
  let area : ℝ := side_length * side_length
  area = 36 := by sorry

end square_area_6cm_l159_15980


namespace min_cost_and_optimal_batch_funds_sufficient_l159_15953

/-- The total cost function for shipping and storage fees -/
def f (x : ℕ+) : ℚ := 144 / x.val + 4 * x.val

/-- The theorem stating the minimum cost and the optimal number of desks per batch -/
theorem min_cost_and_optimal_batch :
  (∀ x : ℕ+, x.val ≤ 36 → f x ≥ 48) ∧
  (∃ x : ℕ+, x.val ≤ 36 ∧ f x = 48 ∧ x.val = 6) := by
  sorry

/-- The theorem stating that the available funds are sufficient for the optimal arrangement -/
theorem funds_sufficient : 
  ∃ x : ℕ+, x.val ≤ 36 ∧ f x ≤ 480 := by
  sorry

end min_cost_and_optimal_batch_funds_sufficient_l159_15953


namespace performance_arrangement_count_l159_15943

/-- The number of ways to arrange n elements from a set of k elements --/
def A (k n : ℕ) : ℕ := sorry

/-- The number of ways to choose n elements from a set of k elements, where order matters --/
def P (k n : ℕ) : ℕ := sorry

/-- The number of ways to arrange 6 singing programs and 4 dance programs, 
    where no two dance programs can be adjacent --/
def arrangement_count : ℕ := P 7 4 * A 6 6

theorem performance_arrangement_count : 
  arrangement_count = P 7 4 * A 6 6 := by sorry

end performance_arrangement_count_l159_15943


namespace fraction_equality_l159_15935

theorem fraction_equality (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) := by
  sorry

#check fraction_equality

end fraction_equality_l159_15935


namespace imaginary_part_of_z_l159_15928

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z * (1 + i) = Complex.abs (i + 1) → Complex.im z = -Real.sqrt 2 / 2 := by
  sorry

end imaginary_part_of_z_l159_15928


namespace sixth_term_equals_two_l159_15922

/-- A geometric sequence with common ratio 2 and positive terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem sixth_term_equals_two
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_product : a 4 * a 10 = 16) :
  a 6 = 2 := by
sorry

end sixth_term_equals_two_l159_15922


namespace linear_case_quadratic_case_l159_15914

-- Define the coefficient of x^2
def coeff_x2 (k : ℝ) : ℝ := k^2 - 1

-- Define the coefficient of x
def coeff_x (k : ℝ) : ℝ := 2*(k + 1)

-- Define the constant term
def const_term (k : ℝ) : ℝ := 3*(k - 1)

-- Theorem for the linear case
theorem linear_case (k : ℝ) : 
  (coeff_x2 k = 0 ∧ coeff_x k ≠ 0) ↔ k = 1 := by sorry

-- Theorem for the quadratic case
theorem quadratic_case (k : ℝ) :
  coeff_x2 k ≠ 0 ↔ k ≠ 1 ∧ k ≠ -1 := by sorry

end linear_case_quadratic_case_l159_15914


namespace article_price_decrease_l159_15939

theorem article_price_decrease (decreased_price : ℝ) (decrease_percentage : ℝ) (original_price : ℝ) : 
  decreased_price = 836 →
  decrease_percentage = 24 →
  decreased_price = original_price * (1 - decrease_percentage / 100) →
  original_price = 1100 := by
  sorry

end article_price_decrease_l159_15939


namespace sqrt_sum_comparison_l159_15930

theorem sqrt_sum_comparison : Real.sqrt 2 + Real.sqrt 7 < Real.sqrt 3 + Real.sqrt 6 := by
  sorry

end sqrt_sum_comparison_l159_15930


namespace unique_number_l159_15904

theorem unique_number : ∃! x : ℝ, x / 4 = x - 6 := by sorry

end unique_number_l159_15904


namespace simplify_fraction_l159_15985

theorem simplify_fraction : (45 : ℚ) * (8 / 15) * (1 / 4) = 6 := by
  sorry

end simplify_fraction_l159_15985


namespace positive_numbers_inequality_l159_15910

theorem positive_numbers_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a^2 + b^2 - a*b = c^2) : 
  (a - c) * (b - c) ≤ 0 := by sorry

end positive_numbers_inequality_l159_15910


namespace volume_ratio_of_cubes_l159_15902

/-- The ratio of volumes of two cubes -/
theorem volume_ratio_of_cubes (inches_per_foot : ℚ) (small_edge : ℚ) (large_edge : ℚ) :
  inches_per_foot = 12 →
  small_edge = 3 →
  large_edge = 3/2 →
  (small_edge^3) / ((large_edge * inches_per_foot)^3) = 1/216 := by sorry

end volume_ratio_of_cubes_l159_15902


namespace football_banquet_min_guests_l159_15926

/-- The minimum number of guests at a banquet given the total food consumed and maximum food per guest -/
def min_guests (total_food : ℕ) (max_food_per_guest : ℕ) : ℕ :=
  (total_food + max_food_per_guest - 1) / max_food_per_guest

/-- Theorem stating the minimum number of guests at the football banquet -/
theorem football_banquet_min_guests :
  min_guests 319 2 = 160 := by
  sorry

end football_banquet_min_guests_l159_15926


namespace angle_E_is_180_l159_15931

/-- A quadrilateral with specific angle relationships -/
structure SpecialQuadrilateral where
  E : ℝ  -- Angle E in degrees
  F : ℝ  -- Angle F in degrees
  G : ℝ  -- Angle G in degrees
  H : ℝ  -- Angle H in degrees
  angle_sum : E + F + G + H = 360  -- Sum of angles in a quadrilateral
  E_F_relation : E = 3 * F  -- Relationship between E and F
  E_G_relation : E = 2 * G  -- Relationship between E and G
  E_H_relation : E = 6 * H  -- Relationship between E and H

/-- The measure of angle E in the special quadrilateral is 180 degrees -/
theorem angle_E_is_180 (q : SpecialQuadrilateral) : q.E = 180 := by
  sorry

end angle_E_is_180_l159_15931


namespace subcommittee_formation_count_l159_15901

theorem subcommittee_formation_count : 
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 7
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 3
  (Nat.choose total_republicans subcommittee_republicans) * 
  (Nat.choose total_democrats subcommittee_democrats) = 7350 := by
  sorry

end subcommittee_formation_count_l159_15901


namespace find_n_l159_15920

theorem find_n (a b c n : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_n : 0 < n)
  (eq1 : (a + b) / a = 3)
  (eq2 : (b + c) / b = 4)
  (eq3 : (c + a) / c = n) :
  n = 7 / 6 := by
sorry

end find_n_l159_15920


namespace jogger_train_distance_l159_15969

/-- Represents the problem of calculating the distance a jogger is ahead of a train. -/
theorem jogger_train_distance
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (passing_time : ℝ)
  (h1 : jogger_speed = 9 * (5 / 18))  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * (5 / 18))  -- 45 km/hr in m/s
  (h3 : train_length = 150)           -- 150 meters
  (h4 : passing_time = 39)            -- 39 seconds
  : (train_speed - jogger_speed) * passing_time = train_length + 240 :=
by sorry

end jogger_train_distance_l159_15969


namespace complement_of_union_equals_singleton_five_l159_15981

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_singleton_five :
  (U \ (M ∪ N)) = {5} := by
  sorry

end complement_of_union_equals_singleton_five_l159_15981


namespace expression_evaluation_l159_15945

theorem expression_evaluation (x : ℤ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 := by
  sorry

end expression_evaluation_l159_15945


namespace handmade_ornaments_fraction_l159_15998

theorem handmade_ornaments_fraction (total : ℕ) (handmade_fraction : ℚ) :
  total = 20 →
  (1 : ℚ) / 3 * total = (1 : ℚ) / 2 * (handmade_fraction * total) →
  handmade_fraction = 2 / 3 := by
  sorry

end handmade_ornaments_fraction_l159_15998


namespace arithmetic_sequence_sum_l159_15907

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a₁ : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a₁ + (n - 1 : ℚ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a₁ + (n - 1 : ℚ) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.nthTerm 5 = 1 → seq.nthTerm 17 = 18 → seq.sumFirstN 12 = 75/2 := by
  sorry

end arithmetic_sequence_sum_l159_15907


namespace sector_chord_length_l159_15971

/-- Given a circular sector with area 1 cm² and perimeter 4 cm, its chord length is 2sin(1) cm. -/
theorem sector_chord_length (r : ℝ) (α : ℝ) :
  (1/2 * α * r^2 = 1) →  -- Area of sector is 1 cm²
  (2 * r + α * r = 4) →  -- Perimeter of sector is 4 cm
  (2 * r * Real.sin (α/2) = 2 * Real.sin 1) := by
  sorry

end sector_chord_length_l159_15971


namespace ticket_price_uniqueness_l159_15921

theorem ticket_price_uniqueness :
  ∃! x : ℕ, x > 0 ∧ 
  ∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  a * x = 60 ∧ b * x = 90 ∧ c * x = 49 :=
by
  sorry

end ticket_price_uniqueness_l159_15921


namespace weight_replaced_person_correct_l159_15909

/-- Represents the weight change scenario of a group of people -/
structure WeightChangeScenario where
  initial_count : ℕ
  average_increase : ℝ
  new_person_weight : ℝ

/-- Calculates the weight of the replaced person given a WeightChangeScenario -/
def weight_of_replaced_person (scenario : WeightChangeScenario) : ℝ :=
  scenario.new_person_weight - scenario.initial_count * scenario.average_increase

theorem weight_replaced_person_correct (scenario : WeightChangeScenario) :
  scenario.initial_count = 6 →
  scenario.average_increase = 2.5 →
  scenario.new_person_weight = 80 →
  weight_of_replaced_person scenario = 65 := by
  sorry

end weight_replaced_person_correct_l159_15909


namespace total_potatoes_sold_l159_15979

/-- The number of bags of potatoes sold in the morning -/
def morning_bags : ℕ := 29

/-- The number of bags of potatoes sold in the afternoon -/
def afternoon_bags : ℕ := 17

/-- The weight of each bag of potatoes in kilograms -/
def bag_weight : ℕ := 7

/-- The total kilograms of potatoes sold for the whole day -/
def total_kg : ℕ := (morning_bags + afternoon_bags) * bag_weight

theorem total_potatoes_sold : total_kg = 322 := by
  sorry

end total_potatoes_sold_l159_15979


namespace longest_segment_squared_in_quarter_circle_l159_15995

-- Define the diameter of the circle
def circle_diameter : ℝ := 16

-- Define the number of equal sectors
def num_sectors : ℕ := 4

-- Define the longest line segment in a sector
def longest_segment (d : ℝ) (n : ℕ) : ℝ := d

-- Theorem statement
theorem longest_segment_squared_in_quarter_circle :
  (longest_segment circle_diameter num_sectors)^2 = 256 := by
  sorry

end longest_segment_squared_in_quarter_circle_l159_15995


namespace peter_walks_to_grocery_store_l159_15986

/-- The total distance Peter walks to the grocery store -/
def total_distance (walking_speed : ℝ) (distance_walked : ℝ) (remaining_time : ℝ) : ℝ :=
  distance_walked + walking_speed * remaining_time

/-- Theorem: Peter walks 2.5 miles to the grocery store -/
theorem peter_walks_to_grocery_store :
  let walking_speed : ℝ := 1 / 20 -- 1 mile per 20 minutes
  let distance_walked : ℝ := 1 -- 1 mile already walked
  let remaining_time : ℝ := 30 -- 30 more minutes to walk
  total_distance walking_speed distance_walked remaining_time = 2.5 := by
sorry

end peter_walks_to_grocery_store_l159_15986


namespace fraction_reduction_l159_15934

theorem fraction_reduction (a : ℚ) : 
  (4 - a) / (5 - a) = 16 / 25 → 9 * a = 20 := by
  sorry

end fraction_reduction_l159_15934


namespace inscribed_square_side_length_l159_15949

/-- Given a right triangle with legs of lengths 6 and 8, and an inscribed square
    sharing the right angle with the triangle, the side length of the square is 24/7. -/
theorem inscribed_square_side_length :
  ∀ (a b c : Real) (s : Real),
    a = 6 →
    b = 8 →
    c^2 = a^2 + b^2 →  -- Pythagorean theorem for right triangle
    s^2 + (a - s) * s + (b - s) * s = (a * b) / 2 →  -- Area equality
    s = 24 / 7 :=
by sorry

end inscribed_square_side_length_l159_15949


namespace digit_distribution_l159_15958

theorem digit_distribution (n : ℕ) 
  (h1 : n > 0)
  (h2 : (n / 2 : ℚ) = n * (1 / 2 : ℚ))
  (h3 : (n / 5 : ℚ) = n * (1 / 5 : ℚ))
  (h4 : (n / 10 : ℚ) = n * (1 / 10 : ℚ))
  (h5 : (1 / 2 : ℚ) + 2 * (1 / 5 : ℚ) + (1 / 10 : ℚ) = 1) :
  n = 10 := by
sorry

end digit_distribution_l159_15958


namespace triangle_area_l159_15988

/-- The area of a triangle with side lengths 10, 10, and 12 is 48 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 10) (h2 : b = 10) (h3 : c = 12) :
  (1 / 2 : ℝ) * c * Real.sqrt (4 * a^2 * b^2 - (a^2 + b^2 - c^2)^2) / (2 * a * b) = 48 := by
  sorry

end triangle_area_l159_15988


namespace essay_competition_probability_l159_15974

theorem essay_competition_probability (n : ℕ) (h : n = 6) :
  let total_outcomes := n * n
  let favorable_outcomes := n * (n - 1)
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 6 :=
by sorry

end essay_competition_probability_l159_15974


namespace prob_king_is_one_thirteenth_l159_15993

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (kings_per_suit : ℕ)
  (h_total : total_cards = num_ranks * num_suits)
  (h_kings : kings_per_suit = 1)

/-- The probability of drawing a King from a standard deck -/
def prob_draw_king (d : Deck) : ℚ :=
  (d.num_suits * d.kings_per_suit : ℚ) / d.total_cards

/-- Theorem: The probability of drawing a King from a standard deck is 1/13 -/
theorem prob_king_is_one_thirteenth (d : Deck) 
  (h_standard : d.total_cards = 52 ∧ d.num_ranks = 13 ∧ d.num_suits = 4) : 
  prob_draw_king d = 1 / 13 := by
  sorry

#check prob_king_is_one_thirteenth

end prob_king_is_one_thirteenth_l159_15993


namespace matrix_equality_l159_15997

theorem matrix_equality (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = ![![5, 2], ![-3, 6]]) : 
  D * C = ![![5, 2], ![-3, 6]] := by
sorry

end matrix_equality_l159_15997


namespace quadratic_no_real_roots_l159_15923

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0) (pos_q : q > 0) (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq : b + c = p + q) :
  ∀ x : ℝ, b * x^2 - 2 * a * x + c ≠ 0 := by
sorry

end quadratic_no_real_roots_l159_15923


namespace polynomial_equality_l159_15933

theorem polynomial_equality : 98^5 - 5 * 98^4 + 10 * 98^3 - 10 * 98^2 + 5 * 98 - 1 = 97^5 := by
  sorry

end polynomial_equality_l159_15933


namespace somu_age_problem_l159_15954

/-- Somu's age problem -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_back : ℕ) :
  somu_age = 18 →
  somu_age = father_age / 3 →
  somu_age - years_back = (father_age - years_back) / 5 →
  years_back = 9 := by
  sorry

end somu_age_problem_l159_15954


namespace spoons_to_knives_ratio_l159_15970

/-- Represents the initial number of each type of cutlery and the total after adding 2 of each -/
structure Cutlery where
  forks : ℕ
  knives : ℕ
  teaspoons : ℕ
  spoons : ℕ
  total_after_adding : ℕ

/-- The given conditions for the cutlery problem -/
def cutlery_conditions : Cutlery :=
  { forks := 6,
    knives := 6 + 9,
    teaspoons := 6 / 2,
    spoons := 28,  -- This is what we're proving
    total_after_adding := 62 }

/-- Theorem stating that the ratio of spoons to knives is 28:15 given the conditions -/
theorem spoons_to_knives_ratio (c : Cutlery) 
  (h1 : c.forks = 6)
  (h2 : c.knives = c.forks + 9)
  (h3 : c.teaspoons = c.forks / 2)
  (h4 : c.total_after_adding = c.forks + 2 + c.knives + 2 + c.teaspoons + 2 + c.spoons + 2) :
  c.spoons * 15 = 28 * c.knives := by
  sorry

#check spoons_to_knives_ratio

end spoons_to_knives_ratio_l159_15970


namespace simplify_sqrt_expression_l159_15959

theorem simplify_sqrt_expression :
  Real.sqrt 300 / Real.sqrt 75 - Real.sqrt 98 / Real.sqrt 49 = 2 - Real.sqrt 2 := by
  sorry

end simplify_sqrt_expression_l159_15959


namespace product_pure_imaginary_l159_15917

theorem product_pure_imaginary (x : ℝ) :
  (∃ y : ℝ, (x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + 2 * Complex.I) = Complex.I * y) ↔
  x^3 + 3*x^2 - 9*x - 7 = 0 :=
by sorry

end product_pure_imaginary_l159_15917


namespace cubic_is_constant_tangent_bounds_on_m_for_constant_tangent_function_l159_15946

-- Definition of a "constant tangent function"
def is_constant_tangent_function (f : ℝ → ℝ) : Prop :=
  ∀ k b : ℝ, ∃ x₀ : ℝ, f x₀ + k * x₀ + b = k * x₀ + b ∧ 
  (deriv f) x₀ + k = k

-- Part 1: Prove that x^3 is a constant tangent function
theorem cubic_is_constant_tangent : is_constant_tangent_function (λ x : ℝ => x^3) := by
  sorry

-- Part 2: Prove the bounds on m for the given function
theorem bounds_on_m_for_constant_tangent_function :
  is_constant_tangent_function (λ x : ℝ => 1/2 * (Real.exp x - x - 1) * Real.exp x + m) →
  -1/8 < m ∧ m ≤ 0 := by
  sorry

end cubic_is_constant_tangent_bounds_on_m_for_constant_tangent_function_l159_15946


namespace circle_E_radius_l159_15951

-- Define the equilateral triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 12 ∧ dist B C = 12 ∧ dist C A = 12

-- Define point D as the midpoint of BC
def Midpoint (D B C : ℝ × ℝ) : Prop :=
  D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define circles A and D
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {P : ℝ × ℝ | dist P center = radius}

-- Define the tangency condition for circle E
def Tangent (E A D : ℝ × ℝ) (r : ℝ) : Prop :=
  dist E A + r = 6 ∧ dist E D + r = 6

-- Main theorem
theorem circle_E_radius
  (A B C D E : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : Midpoint D B C)
  (h3 : ∃ M N, M ∈ Circle A 6 ∩ Circle D 6 ∧ N ∈ Circle A 6 ∩ Circle D 6 ∧
               Midpoint M A B ∧ Midpoint N A C)
  (h4 : ∃ t, E = (1 - t) • A + t • D ∧ 0 ≤ t ∧ t ≤ 1)
  (h5 : ∃ r, Tangent E A D r) :
  ∃ r, r = 3 * Real.sqrt 3 - 6 ∧ Tangent E A D r :=
sorry

end circle_E_radius_l159_15951


namespace valbonne_group_separation_l159_15903

-- Define a type for participants
variable {Participant : Type}

-- Define a friendship relation
variable (friends : Participant → Participant → Prop)

-- Define a property that each participant has at most three friends
variable (at_most_three_friends : ∀ p : Participant, ∃ (f₁ f₂ f₃ : Participant), 
  ∀ f : Participant, friends p f → (f = f₁ ∨ f = f₂ ∨ f = f₃))

-- Define a partition of participants into two groups
variable (group : Participant → Bool)

-- State the theorem
theorem valbonne_group_separation :
  ∃ group : Participant → Bool,
    ∀ p : Participant,
      (∃! f : Participant, friends p f ∧ group p = group f) ∨
      (∀ f : Participant, friends p f → group p ≠ group f) :=
sorry

end valbonne_group_separation_l159_15903


namespace a_eq_b_sufficient_not_necessary_l159_15961

/-- The line y = x + 2 is tangent to the circle (x - a)² + (y - b)² = 2 -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x + 2 ∧ (x - a)^2 + (y - b)^2 = 2 ∧
  ∀ (x' y' : ℝ), y' = x' + 2 → (x' - a)^2 + (y' - b)^2 ≥ 2

/-- The condition a = b is sufficient but not necessary for the tangency -/
theorem a_eq_b_sufficient_not_necessary :
  (∀ a b : ℝ, a = b → is_tangent a b) ∧
  ¬(∀ a b : ℝ, is_tangent a b → a = b) :=
sorry

end a_eq_b_sufficient_not_necessary_l159_15961


namespace hexagon_vertex_traces_line_hexagon_lines_intersect_common_point_l159_15982

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- A regular hexagon -/
structure RegularHexagon :=
  (center : Point)
  (vertices : Fin 6 → Point)

/-- The center of the hexagon moves along this line -/
def centerLine : Line := sorry

/-- The fixed vertex A of the hexagon -/
def fixedVertexA : Point := sorry

/-- Function to check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Function to check if two lines intersect -/
def linesIntersect (l1 l2 : Line) : Prop := sorry

/-- Theorem: Each vertex of a regular hexagon traces a straight line when the center moves along a line -/
theorem hexagon_vertex_traces_line (h : RegularHexagon) (i : Fin 6) :
  ∃ l : Line, ∀ t : ℝ, pointOnLine (h.vertices i) l :=
sorry

/-- Theorem: The lines traced by the five non-fixed vertices intersect at a common point -/
theorem hexagon_lines_intersect_common_point (h : RegularHexagon) :
  ∃ p : Point, ∀ i j : Fin 6, i ≠ j → i ≠ 0 → j ≠ 0 →
    ∃ l1 l2 : Line,
      (∀ t : ℝ, pointOnLine (h.vertices i) l1) ∧
      (∀ t : ℝ, pointOnLine (h.vertices j) l2) ∧
      linesIntersect l1 l2 ∧
      pointOnLine p l1 ∧ pointOnLine p l2 :=
sorry

end hexagon_vertex_traces_line_hexagon_lines_intersect_common_point_l159_15982


namespace inequality_solution_1_inequality_solution_2_l159_15975

/-- The solution set of the inequality 2 < |2x-5| ≤ 7 -/
def solution_set_1 : Set ℝ :=
  {x | -1 ≤ x ∧ x < 3/2 ∨ 7/2 < x ∧ x ≤ 6}

/-- The solution set of the inequality 1/(x-1) > x + 1 -/
def solution_set_2 : Set ℝ :=
  {x | x < -Real.sqrt 2 ∨ (1 < x ∧ x < Real.sqrt 2)}

theorem inequality_solution_1 :
  {x : ℝ | 2 < |2*x - 5| ∧ |2*x - 5| ≤ 7} = solution_set_1 := by sorry

theorem inequality_solution_2 :
  {x : ℝ | 1/(x-1) > x + 1} = solution_set_2 := by sorry

end inequality_solution_1_inequality_solution_2_l159_15975


namespace counterexample_exists_l159_15929

-- Define the types for planes and lines in space
variable (Plane Line : Type)

-- Define the intersection operation for planes
variable (intersect_planes : Plane → Plane → Line)

-- Define the perpendicular relation between lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem counterexample_exists :
  ∃ (α β γ : Plane) (l m n : Line),
    intersect_planes α β = m ∧
    intersect_planes β γ = l ∧
    intersect_planes γ α = n ∧
    perpendicular l m ∧
    perpendicular l n ∧
    ¬ perpendicular m n :=
sorry

end counterexample_exists_l159_15929


namespace sector_central_angle_l159_15999

theorem sector_central_angle (area : Real) (radius : Real) (central_angle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  area = 1 / 2 * central_angle * radius ^ 2 →
  central_angle = 3 * Real.pi / 4 := by
sorry

end sector_central_angle_l159_15999


namespace polynomial_factorization_sum_l159_15912

theorem polynomial_factorization_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃) * (x^2 - 1)) :
  a₁*d₁ + a₂*d₂ + a₃*d₃ = -1 := by
  sorry

end polynomial_factorization_sum_l159_15912


namespace tangent_sphere_radius_l159_15962

/-- Given a sphere of radius R and three mutually perpendicular planes drawn through its center,
    the radius x of a sphere that is tangent to all these planes and the given sphere
    is x = (√3 ± 1)R / 2. -/
theorem tangent_sphere_radius (R : ℝ) (R_pos : R > 0) :
  ∃ x : ℝ, x > 0 ∧
  (x = (Real.sqrt 3 + 1) * R / 2 ∨ x = (Real.sqrt 3 - 1) * R / 2) :=
by sorry

end tangent_sphere_radius_l159_15962


namespace rice_box_theorem_l159_15916

/-- Represents the number of grains in each box -/
def grains_in_box (first_grain_count : ℕ) (common_difference : ℕ) (box_number : ℕ) : ℕ :=
  first_grain_count + (box_number - 1) * common_difference

/-- The total number of grains in all boxes -/
def total_grains (first_grain_count : ℕ) (common_difference : ℕ) (num_boxes : ℕ) : ℕ :=
  (num_boxes * (2 * first_grain_count + (num_boxes - 1) * common_difference)) / 2

theorem rice_box_theorem :
  (∃ (d : ℕ), total_grains 11 d 9 = 351 ∧ d = 7) ∧
  (∃ (d : ℕ), grains_in_box (23 - 2 * d) d 3 = 23 ∧ total_grains (23 - 2 * d) d 9 = 351 ∧ d = 8) :=
by sorry

end rice_box_theorem_l159_15916


namespace gold_percentage_in_first_metal_l159_15908

theorem gold_percentage_in_first_metal
  (total_weight : Real)
  (desired_gold_percentage : Real)
  (first_metal_weight : Real)
  (second_metal_weight : Real)
  (second_metal_gold_percentage : Real)
  (h1 : total_weight = 12.4)
  (h2 : desired_gold_percentage = 0.5)
  (h3 : first_metal_weight = 6.2)
  (h4 : second_metal_weight = 6.2)
  (h5 : second_metal_gold_percentage = 0.4)
  (h6 : total_weight = first_metal_weight + second_metal_weight) :
  let total_gold := total_weight * desired_gold_percentage
  let second_metal_gold := second_metal_weight * second_metal_gold_percentage
  let first_metal_gold := total_gold - second_metal_gold
  let first_metal_gold_percentage := first_metal_gold / first_metal_weight
  first_metal_gold_percentage = 0.6 := by
  sorry

end gold_percentage_in_first_metal_l159_15908


namespace dads_borrowed_nickels_l159_15919

/-- The number of nickels Mike's dad borrowed -/
def nickels_borrowed (initial_nickels remaining_nickels : ℕ) : ℕ :=
  initial_nickels - remaining_nickels

theorem dads_borrowed_nickels :
  let initial_nickels : ℕ := 87
  let remaining_nickels : ℕ := 12
  nickels_borrowed initial_nickels remaining_nickels = 75 := by
sorry

end dads_borrowed_nickels_l159_15919


namespace chelsea_needs_52_bullseyes_l159_15941

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : Nat
  chelsea_lead : Nat
  bullseye_score : Nat
  chelsea_min_score : Nat
  opponent_max_score : Nat

/-- Calculates the minimum number of consecutive bullseyes needed for Chelsea to win -/
def min_bullseyes_needed (comp : ArcheryCompetition) : Nat :=
  let remaining_shots := comp.total_shots / 2
  let chelsea_score := remaining_shots * comp.chelsea_min_score + comp.chelsea_lead
  let opponent_max := remaining_shots * comp.opponent_max_score
  let score_diff := opponent_max - chelsea_score
  (score_diff + comp.bullseye_score - comp.chelsea_min_score - 1) / (comp.bullseye_score - comp.chelsea_min_score) + 1

/-- The main theorem stating that 52 consecutive bullseyes are needed for Chelsea to win -/
theorem chelsea_needs_52_bullseyes :
  let comp := ArcheryCompetition.mk 120 60 10 3 10
  min_bullseyes_needed comp = 52 := by
  sorry

end chelsea_needs_52_bullseyes_l159_15941


namespace quadratic_form_equivalence_l159_15913

theorem quadratic_form_equivalence :
  let f (x : ℝ) := 2 * x^2 - 8 * x + 3
  let g (x : ℝ) := 2 * (x - 2)^2 - 5
  ∀ x, f x = g x := by sorry

end quadratic_form_equivalence_l159_15913


namespace fraction_simplification_l159_15972

theorem fraction_simplification :
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end fraction_simplification_l159_15972


namespace absolute_difference_inequality_characterization_l159_15966

def absolute_difference_inequality (a : ℝ) : Set ℝ :=
  {x : ℝ | |x - 1| - |x - 2| < a}

theorem absolute_difference_inequality_characterization (a : ℝ) :
  (absolute_difference_inequality a = Set.univ ↔ a > 1) ∧
  (absolute_difference_inequality a ≠ ∅ ↔ a > -1) ∧
  (absolute_difference_inequality a = ∅ ↔ a ≤ -1) :=
sorry

end absolute_difference_inequality_characterization_l159_15966


namespace cubic_bijective_l159_15987

/-- The cubic function from reals to reals -/
def f (x : ℝ) : ℝ := x^3

/-- Theorem stating that the cubic function is bijective -/
theorem cubic_bijective : Function.Bijective f := by sorry

end cubic_bijective_l159_15987


namespace p_satisfies_equation_l159_15900

/-- The polynomial p(x) that satisfies the given equation -/
def p (x : ℝ) : ℝ := (x - 2) * (x - 4) * (x - 8) * (x - 16)

/-- Theorem stating that p(x) satisfies the given equation for all real x -/
theorem p_satisfies_equation (x : ℝ) : (x - 16) * p (2 * x) = (16 * x - 16) * p x := by
  sorry

end p_satisfies_equation_l159_15900


namespace probability_B3_l159_15992

structure Box where
  number : Nat
  balls : List Nat

def initial_boxes : List Box := [
  ⟨1, [1, 1, 2, 3]⟩,
  ⟨2, [1, 1, 3]⟩,
  ⟨3, [1, 1, 1, 2, 2]⟩
]

def draw_and_transfer (boxes : List Box) : List Box := sorry

def second_draw (boxes : List Box) : ℝ := sorry

theorem probability_B3 (boxes : List Box) :
  boxes = initial_boxes →
  second_draw (draw_and_transfer boxes) = 13/48 := by sorry

end probability_B3_l159_15992


namespace two_digit_perfect_square_divisible_by_seven_l159_15947

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

theorem two_digit_perfect_square_divisible_by_seven :
  ∃! n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ n % 7 = 0 :=
sorry

end two_digit_perfect_square_divisible_by_seven_l159_15947


namespace female_students_count_l159_15915

theorem female_students_count (total_students : ℕ) 
  (h1 : total_students = 63) 
  (h2 : ∀ (female_count : ℕ), 
    female_count ≤ total_students → 
    (female_count : ℚ) / total_students = 
    (10 : ℚ) / 11 * ((total_students - female_count) : ℚ) / total_students) : 
  ∃ (female_count : ℕ), female_count = 30 ∧ female_count ≤ total_students :=
sorry

end female_students_count_l159_15915


namespace range_of_a_l159_15955

/-- The piecewise function f(x) as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then |Real.log x| else -(x - 3*a + 1)^2 + (2*a - 1)^2 + a

/-- The function g(x) defined as f(x) - b -/
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x - b

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) :
  (∃ b : ℝ, b > 0 ∧ (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ g a b x₃ = 0 ∧ g a b x₄ = 0)) →
  0 < a ∧ a < 1/2 :=
sorry

end range_of_a_l159_15955


namespace subset_implies_M_M_implies_subset_M_iff_subset_l159_15938

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 6 = 0}

-- Define the set M
def M : Set ℝ := {0, 3, -3}

-- Theorem statement
theorem subset_implies_M (a : ℝ) : B a ⊆ A → a ∈ M := by sorry

-- Theorem for the converse
theorem M_implies_subset (a : ℝ) : a ∈ M → B a ⊆ A := by sorry

-- Theorem for the equivalence
theorem M_iff_subset (a : ℝ) : a ∈ M ↔ B a ⊆ A := by sorry

end subset_implies_M_M_implies_subset_M_iff_subset_l159_15938


namespace gcd_max_digits_l159_15976

theorem gcd_max_digits (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 := by
sorry

end gcd_max_digits_l159_15976


namespace max_sum_given_sum_squares_and_cubes_l159_15948

theorem max_sum_given_sum_squares_and_cubes :
  ∃ (max : ℝ), max = 4 ∧
  ∀ (x y : ℝ), x^2 + y^2 = 7 → x^3 + y^3 = 10 →
  x + y ≤ max :=
by sorry

end max_sum_given_sum_squares_and_cubes_l159_15948


namespace test_total_points_l159_15950

theorem test_total_points (total_questions : ℕ) (two_point_questions : ℕ) : 
  total_questions = 40 → 
  two_point_questions = 30 → 
  (total_questions - two_point_questions) * 4 + two_point_questions * 2 = 100 := by
  sorry

end test_total_points_l159_15950


namespace dozen_pens_cost_l159_15996

/-- The cost of a pen in rupees -/
def pen_cost : ℝ := sorry

/-- The cost of a pencil in rupees -/
def pencil_cost : ℝ := sorry

/-- The cost ratio of a pen to a pencil is 5:1 -/
axiom cost_ratio : pen_cost = 5 * pencil_cost

/-- The cost of 3 pens and 5 pencils is Rs. 200 -/
axiom total_cost : 3 * pen_cost + 5 * pencil_cost = 200

/-- The cost of one dozen pens is Rs. 600 -/
theorem dozen_pens_cost : 12 * pen_cost = 600 := by sorry

end dozen_pens_cost_l159_15996


namespace min_value_of_g_l159_15960

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^2 - 4*x + 9

-- State the theorem
theorem min_value_of_g :
  ∀ x ∈ Set.Icc (-2 : ℝ) 0, g x ≥ 9 ∧ ∃ y ∈ Set.Icc (-2 : ℝ) 0, g y = 9 :=
by sorry

end min_value_of_g_l159_15960


namespace series_sum_implies_k_l159_15991

theorem series_sum_implies_k (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 2) / k^n = 17/2) : k = 17/7 := by
  sorry

end series_sum_implies_k_l159_15991


namespace inverse_13_mod_1373_l159_15936

theorem inverse_13_mod_1373 : ∃ x : ℕ, 0 ≤ x ∧ x < 1373 ∧ (13 * x) % 1373 = 1 := by
  use 843
  sorry

end inverse_13_mod_1373_l159_15936


namespace smallest_k_for_inequality_l159_15994

theorem smallest_k_for_inequality : ∃ (k : ℕ), k = 4 ∧ 
  (∀ (a : ℝ) (n : ℕ), a ∈ Set.Icc 0 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (a : ℝ) (n : ℕ), a ∈ Set.Icc 0 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end smallest_k_for_inequality_l159_15994


namespace correct_operation_l159_15989

theorem correct_operation (x : ℝ) : 3 * x - 2 * x = x := by
  sorry

end correct_operation_l159_15989


namespace range_of_a_l159_15942

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.union (Set.Iic (-2)) {1} :=
by sorry

end range_of_a_l159_15942


namespace f_properties_l159_15932

noncomputable def f (x : ℝ) := Real.log (|x| + 1)

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (f 0 = 0 ∧ ∀ x : ℝ, f x ≥ 0) :=
by sorry

end f_properties_l159_15932


namespace investment_income_percentage_l159_15977

/-- Proves that the total annual income from two investments is equal to 6% of the total investment amount. -/
theorem investment_income_percentage (initial_investment : ℝ) (additional_investment : ℝ) 
  (initial_rate : ℝ) (additional_rate : ℝ) :
  initial_investment = 2400 →
  additional_investment = 599.9999999999999 →
  initial_rate = 0.05 →
  additional_rate = 0.10 →
  let total_investment := initial_investment + additional_investment
  let total_income := initial_investment * initial_rate + additional_investment * additional_rate
  (total_income / total_investment) * 100 = 6 := by
  sorry

end investment_income_percentage_l159_15977


namespace set_equivalence_l159_15905

theorem set_equivalence : {x : ℕ | x < 5} = {0, 1, 2, 3, 4} := by
  sorry

end set_equivalence_l159_15905


namespace first_digit_base7_528_l159_15984

/-- The first digit of the base 7 representation of a natural number -/
def firstDigitBase7 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 7).succ
    (n / 7^(k-1)) % 7

theorem first_digit_base7_528 :
  firstDigitBase7 528 = 1 := by sorry

end first_digit_base7_528_l159_15984
