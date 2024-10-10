import Mathlib

namespace julia_tag_players_l1829_182903

theorem julia_tag_players (monday_kids : ℕ) (difference : ℕ) (tuesday_kids : ℕ) 
  (h1 : monday_kids = 18)
  (h2 : difference = 8)
  (h3 : monday_kids = tuesday_kids + difference) :
  tuesday_kids = 10 := by
  sorry

end julia_tag_players_l1829_182903


namespace thomas_escalator_problem_l1829_182956

/-- Thomas's escalator problem -/
theorem thomas_escalator_problem 
  (l : ℝ) -- length of the escalator
  (v : ℝ) -- speed of the escalator when working
  (r : ℝ) -- Thomas's running speed
  (w : ℝ) -- Thomas's walking speed
  (h1 : l / (v + r) = 15) -- Thomas runs down moving escalator in 15 seconds
  (h2 : l / (v + w) = 30) -- Thomas walks down moving escalator in 30 seconds
  (h3 : l / r = 20) -- Thomas runs down broken escalator in 20 seconds
  : l / w = 60 := by
  sorry

end thomas_escalator_problem_l1829_182956


namespace voyage_year_difference_l1829_182931

def zheng_he_voyage_year : ℕ := 2005 - 600
def columbus_voyage_year : ℕ := 1492

theorem voyage_year_difference : columbus_voyage_year - zheng_he_voyage_year = 87 := by
  sorry

end voyage_year_difference_l1829_182931


namespace power_product_equality_l1829_182952

theorem power_product_equality : (15 : ℕ)^2 * 8^3 * 256 = 29491200 := by
  sorry

end power_product_equality_l1829_182952


namespace fraction_comparison_l1829_182979

theorem fraction_comparison (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a / b > (a + 1) / (b + 1) := by
  sorry

end fraction_comparison_l1829_182979


namespace probability_three_correct_out_of_seven_l1829_182981

/-- The number of derangements of n elements -/
def derangement (n : ℕ) : ℕ := sorry

/-- The probability of exactly k people receiving their correct letter when n letters are randomly distributed to n people -/
def probability_correct_letters (n k : ℕ) : ℚ :=
  (Nat.choose n k * derangement (n - k)) / n.factorial

theorem probability_three_correct_out_of_seven :
  probability_correct_letters 7 3 = 1 / 16 := by sorry

end probability_three_correct_out_of_seven_l1829_182981


namespace quadratic_y_values_order_l1829_182960

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  f (-2) = 1 ∧ (∀ x, f x ≤ f (-2))

/-- Theorem stating the relationship between y-values of specific points -/
theorem quadratic_y_values_order (f : ℝ → ℝ) (y₁ y₂ y₃ : ℝ)
  (hf : QuadraticFunction f)
  (h1 : f 1 = y₁)
  (h2 : f (-1) = y₂)
  (h3 : f (-4) = y₃) :
  y₁ < y₃ ∧ y₃ < y₂ := by
  sorry

end quadratic_y_values_order_l1829_182960


namespace find_a_l1829_182924

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x => 
  if x > 0 then a^x else -a^(-x)

-- State the theorem
theorem find_a : 
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ 
  (∀ x, f a (-x) = -(f a x)) ∧  -- odd function property
  (∀ x > 0, f a x = a^x) ∧      -- definition for x > 0
  f a (Real.log 2 / Real.log (1/2)) = -3 ∧ -- given condition
  a = Real.sqrt 3 := by
sorry

end find_a_l1829_182924


namespace cookie_box_count_l1829_182980

/-- The number of cookies in a bag -/
def cookies_per_bag : ℕ := 7

/-- The number of cookies in a box -/
def cookies_per_box : ℕ := 12

/-- The number of bags used for comparison -/
def num_bags : ℕ := 9

/-- The additional number of cookies in boxes compared to bags -/
def extra_cookies : ℕ := 33

/-- The number of boxes -/
def num_boxes : ℕ := 8

theorem cookie_box_count :
  num_boxes * cookies_per_box = num_bags * cookies_per_bag + extra_cookies :=
by sorry

end cookie_box_count_l1829_182980


namespace min_height_is_eleven_l1829_182936

/-- Represents the dimensions of a rectangular box with square bases -/
structure BoxDimensions where
  base : ℝ
  height : ℝ

/-- Calculates the surface area of the box -/
def surfaceArea (d : BoxDimensions) : ℝ :=
  2 * d.base^2 + 4 * d.base * d.height

/-- Checks if the box dimensions satisfy the given conditions -/
def isValidBox (d : BoxDimensions) : Prop :=
  d.height = d.base + 6 ∧ surfaceArea d ≥ 150 ∧ d.base > 0

theorem min_height_is_eleven :
  ∀ d : BoxDimensions, isValidBox d → d.height ≥ 11 :=
by sorry

end min_height_is_eleven_l1829_182936


namespace min_pool_cost_l1829_182946

/-- Represents the construction cost of a rectangular pool -/
def pool_cost (length width depth : ℝ) (wall_price : ℝ) : ℝ :=
  (2 * (length + width) * depth * wall_price) + (length * width * 1.5 * wall_price)

/-- Theorem stating the minimum cost for the pool construction -/
theorem min_pool_cost (a : ℝ) (h_a : a > 0) :
  let volume := 4800
  let depth := 3
  ∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    length * width * depth = volume ∧
    ∀ (l w : ℝ), l > 0 → w > 0 → l * w * depth = volume →
      pool_cost length width depth a ≤ pool_cost l w depth a ∧
      pool_cost length width depth a = 2880 * a :=
sorry

end min_pool_cost_l1829_182946


namespace factorial_trailing_zeros_l1829_182988

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_trailing_zeros : 
  trailing_zeros 238 = 57 ∧ trailing_zeros 238 - trailing_zeros 236 = 0 :=
by sorry

end factorial_trailing_zeros_l1829_182988


namespace election_votes_proof_l1829_182985

theorem election_votes_proof (total_votes : ℕ) 
  (h1 : (total_votes : ℝ) * 0.8 * 0.55 + 2520 = (total_votes : ℝ) * 0.8) : 
  total_votes = 7000 := by
  sorry

end election_votes_proof_l1829_182985


namespace minimum_radios_l1829_182954

theorem minimum_radios (n d : ℕ) (h1 : d > 0) : 
  (3 * (d / n / 3) + (n - 3) * (d / n + 12) - d = 108) →
  (∀ m : ℕ, m < n → ¬(3 * (d / m / 3) + (m - 3) * (d / m + 12) - d = 108)) →
  n = 12 := by
  sorry

end minimum_radios_l1829_182954


namespace expression_evaluation_l1829_182941

theorem expression_evaluation : (10 + 1/3) + (-11.5) + (-10 - 1/3) - 4.5 = -16 := by
  sorry

end expression_evaluation_l1829_182941


namespace ceiling_floor_product_l1829_182977

theorem ceiling_floor_product (y : ℝ) : 
  y > 0 → ⌈y⌉ * ⌊y⌋ = 72 → 8 < y ∧ y < 9 := by
sorry

end ceiling_floor_product_l1829_182977


namespace complex_simplification_l1829_182939

theorem complex_simplification :
  3 * (4 - 2 * Complex.I) - 2 * Complex.I * (3 - Complex.I) + 2 * (1 + 2 * Complex.I) = 10 - 8 * Complex.I :=
by sorry

end complex_simplification_l1829_182939


namespace max_midpoints_on_circle_l1829_182910

/-- A regular n-gon with n ≥ 3 -/
structure RegularNGon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- The set of midpoints of all sides and diagonals of a regular n-gon -/
def midpoints (ngon : RegularNGon) : Set (ℝ × ℝ) :=
  sorry

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The number of points from a set that lie on a given circle -/
def pointsOnCircle (S : Set (ℝ × ℝ)) (c : Circle) : ℕ :=
  sorry

/-- The maximum number of points from a set that lie on any circle -/
def maxPointsOnCircle (S : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem: The maximum number of marked midpoints that lie on the same circle is n -/
theorem max_midpoints_on_circle (ngon : RegularNGon) :
    maxPointsOnCircle (midpoints ngon) = ngon.n :=
  sorry

end max_midpoints_on_circle_l1829_182910


namespace jack_final_position_l1829_182929

/-- Represents the number of steps in each flight of stairs -/
def steps_per_flight : ℕ := 12

/-- Represents the height of each step in inches -/
def step_height : ℕ := 8

/-- Represents the number of flights Jack goes up -/
def flights_up : ℕ := 3

/-- Represents the number of flights Jack goes down -/
def flights_down : ℕ := 6

/-- Represents the number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Theorem stating that Jack ends up 24 feet further down than his starting point -/
theorem jack_final_position : 
  (flights_down - flights_up) * steps_per_flight * step_height / inches_per_foot = 24 := by
  sorry


end jack_final_position_l1829_182929


namespace stratified_sampling_theorem_l1829_182974

/-- Calculates the total number of students in three grades given stratified sampling information -/
def totalStudents (sampleSize : ℕ) (firstGradeSample : ℕ) (thirdGradeSample : ℕ) (secondGradeTotal : ℕ) : ℕ :=
  let secondGradeSample := sampleSize - firstGradeSample - thirdGradeSample
  sampleSize * (secondGradeTotal / secondGradeSample)

/-- The total number of students in three grades is 900 given the stratified sampling information -/
theorem stratified_sampling_theorem (sampleSize : ℕ) (firstGradeSample : ℕ) (thirdGradeSample : ℕ) (secondGradeTotal : ℕ)
  (h1 : sampleSize = 45)
  (h2 : firstGradeSample = 20)
  (h3 : thirdGradeSample = 10)
  (h4 : secondGradeTotal = 300) :
  totalStudents sampleSize firstGradeSample thirdGradeSample secondGradeTotal = 900 := by
  sorry

end stratified_sampling_theorem_l1829_182974


namespace uv_value_l1829_182938

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -2/3 * x + 6

-- Define points P and Q
def P : ℝ × ℝ := (9, 0)
def Q : ℝ × ℝ := (0, 6)

-- Define point R
def R (u v : ℝ) : ℝ × ℝ := (u, v)

-- Define that R is on the line segment PQ
def R_on_PQ (u v : ℝ) : Prop :=
  line_equation u v ∧ 0 ≤ u ∧ u ≤ 9

-- Define the area ratio condition
def area_condition (u v : ℝ) : Prop :=
  (1/2 * 9 * 6) = 2 * (1/2 * 9 * v)

-- Theorem statement
theorem uv_value (u v : ℝ) 
  (h1 : R_on_PQ u v) 
  (h2 : area_condition u v) : 
  u * v = 13.5 := by
  sorry

end uv_value_l1829_182938


namespace min_cards_for_even_product_l1829_182927

/-- Represents a card with an integer value -/
structure Card where
  value : Int
  even : Bool

/-- The set of cards in the box -/
def cards : Finset Card :=
  sorry

/-- A valid sequence of drawn cards according to the rules -/
def ValidSequence : List Card → Prop :=
  sorry

/-- The product of the values of a list of cards -/
def product : List Card → Int :=
  sorry

/-- Theorem: The minimum number of cards to ensure an even product is 3 -/
theorem min_cards_for_even_product :
  ∀ (s : List Card), ValidSequence s → product s % 2 = 0 → s.length ≥ 3 :=
sorry

end min_cards_for_even_product_l1829_182927


namespace laptop_sticker_price_l1829_182906

theorem laptop_sticker_price (sticker_price : ℝ) : 
  (0.8 * sticker_price - 50 = 0.7 * sticker_price + 30) → 
  sticker_price = 800 := by
sorry

end laptop_sticker_price_l1829_182906


namespace wonderland_roads_l1829_182986

/-- The number of vertices in the complete graph -/
def n : ℕ := 5

/-- The number of edges shown on Alice's map -/
def shown_edges : ℕ := 7

/-- The total number of edges in a complete graph with n vertices -/
def total_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of missing edges -/
def missing_edges : ℕ := total_edges n - shown_edges

theorem wonderland_roads :
  missing_edges = 3 := by sorry

end wonderland_roads_l1829_182986


namespace loss_60_l1829_182967

/-- Represents the financial recording of a transaction amount in dollars -/
def record_transaction (amount : Int) : Int := amount

/-- Records a profit of $370 as +370 dollars -/
axiom profit_370 : record_transaction 370 = 370

/-- Proves that a loss of $60 is recorded as -60 dollars -/
theorem loss_60 : record_transaction (-60) = -60 := by sorry

end loss_60_l1829_182967


namespace point_in_second_quadrant_implies_m_range_l1829_182994

/-- A point in the Cartesian plane is in the second quadrant if and only if its x-coordinate is negative and its y-coordinate is positive. -/
def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Given a real number m, prove that if the point P(m-3, m+1) is in the second quadrant,
    then -1 < m and m < 3. -/
theorem point_in_second_quadrant_implies_m_range (m : ℝ) :
  is_in_second_quadrant (m - 3) (m + 1) → -1 < m ∧ m < 3 := by
  sorry


end point_in_second_quadrant_implies_m_range_l1829_182994


namespace complex_number_location_l1829_182911

theorem complex_number_location :
  ∀ (z : ℂ), (2 + I) * z = -I →
  (z.re < 0 ∧ z.im < 0) :=
by sorry

end complex_number_location_l1829_182911


namespace bart_mixtape_second_side_l1829_182978

def mixtape (first_side_songs : ℕ) (song_length : ℕ) (total_length : ℕ) : ℕ :=
  (total_length - first_side_songs * song_length) / song_length

theorem bart_mixtape_second_side :
  mixtape 6 4 40 = 4 := by
  sorry

end bart_mixtape_second_side_l1829_182978


namespace not_p_sufficient_not_necessary_for_not_q_l1829_182928

theorem not_p_sufficient_not_necessary_for_not_q (p q : Prop) 
  (h1 : q → p)  -- p is necessary for q
  (h2 : ¬(p → q))  -- p is not sufficient for q
  : (¬p → ¬q) ∧ ¬(¬q → ¬p) := by
  sorry

end not_p_sufficient_not_necessary_for_not_q_l1829_182928


namespace weight_moved_is_540_l1829_182951

/-- Calculates the total weight moved in three triples given the initial back squat and increase -/
def weightMovedInThreeTriples (initialBackSquat : ℝ) (backSquatIncrease : ℝ) : ℝ :=
  let newBackSquat := initialBackSquat + backSquatIncrease
  let frontSquat := 0.8 * newBackSquat
  let tripleWeight := 0.9 * frontSquat
  3 * tripleWeight

/-- Theorem stating that given John's initial back squat of 200 kg and an increase of 50 kg,
    the total weight moved in three triples is 540 kg -/
theorem weight_moved_is_540 :
  weightMovedInThreeTriples 200 50 = 540 := by
  sorry

#eval weightMovedInThreeTriples 200 50

end weight_moved_is_540_l1829_182951


namespace fourth_term_is_one_tenth_l1829_182933

theorem fourth_term_is_one_tenth (a : ℕ → ℚ) :
  (∀ n : ℕ, a n = 2 / (n^2 + n)) →
  a 4 = 1 / 10 := by
sorry

end fourth_term_is_one_tenth_l1829_182933


namespace sum_of_three_numbers_sum_of_three_numbers_proof_l1829_182920

theorem sum_of_three_numbers : ℕ → ℕ → ℕ → Prop :=
  fun second first third =>
    first = 2 * second ∧
    third = first / 3 ∧
    second = 60 →
    first + second + third = 220

-- The proof is omitted
theorem sum_of_three_numbers_proof : sum_of_three_numbers 60 120 40 := by
  sorry

end sum_of_three_numbers_sum_of_three_numbers_proof_l1829_182920


namespace parallel_line_theorem_l1829_182922

/-- A line parallel to another line with a given y-intercept -/
def parallel_line_with_y_intercept (a b c : ℝ) (y_intercept : ℝ) : Prop :=
  ∃ k : ℝ, (k ≠ 0) ∧ 
  (∀ x y : ℝ, a * x + b * y + c = 0 ↔ a * x + b * y + k = 0) ∧
  (a * 0 + b * y_intercept + k = 0)

/-- The equation x + y + 1 = 0 represents the line parallel to x + y + 4 = 0 with y-intercept -1 -/
theorem parallel_line_theorem :
  parallel_line_with_y_intercept 1 1 4 (-1) →
  ∀ x y : ℝ, x + y + 1 = 0 ↔ parallel_line_with_y_intercept 1 1 4 (-1) :=
by sorry

end parallel_line_theorem_l1829_182922


namespace square_difference_divided_by_nine_l1829_182932

theorem square_difference_divided_by_nine : (110^2 - 95^2) / 9 = 3075 / 9 := by
  sorry

end square_difference_divided_by_nine_l1829_182932


namespace win_sectors_area_l1829_182973

theorem win_sectors_area (r : ℝ) (p : ℝ) : 
  r = 15 → p = 3/7 → (p * π * r^2) = 675*π/7 := by sorry

end win_sectors_area_l1829_182973


namespace yellow_pairs_count_l1829_182987

theorem yellow_pairs_count (total_students : ℕ) (blue_students : ℕ) (yellow_students : ℕ) 
  (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  total_students = 156 →
  blue_students = 68 →
  yellow_students = 88 →
  total_pairs = 78 →
  blue_blue_pairs = 31 →
  total_students = blue_students + yellow_students →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 41 ∧ 
    yellow_yellow_pairs + blue_blue_pairs + (blue_students - 2 * blue_blue_pairs) = total_pairs :=
by sorry

end yellow_pairs_count_l1829_182987


namespace int_part_one_plus_sqrt_seven_l1829_182907

theorem int_part_one_plus_sqrt_seven : ⌊1 + Real.sqrt 7⌋ = 3 := by
  sorry

end int_part_one_plus_sqrt_seven_l1829_182907


namespace topsoil_cost_l1829_182923

/-- The cost of topsoil in dollars per cubic foot -/
def cost_per_cubic_foot : ℝ := 8

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of topsoil in cubic yards -/
def volume_in_cubic_yards : ℝ := 8

/-- Theorem: The cost of 8 cubic yards of topsoil is $1728 -/
theorem topsoil_cost : 
  volume_in_cubic_yards * cubic_yards_to_cubic_feet * cost_per_cubic_foot = 1728 := by
  sorry

end topsoil_cost_l1829_182923


namespace min_distance_point_to_line_l1829_182900

/-- Circle O₁ with center (a, b) and radius √(b² + 1) -/
def circle_O₁ (a b x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = b^2 + 1

/-- Circle O₂ with center (c, d) and radius √(d² + 1) -/
def circle_O₂ (c d x y : ℝ) : Prop :=
  (x - c)^2 + (y - d)^2 = d^2 + 1

/-- Line l: 3x - 4y - 25 = 0 -/
def line_l (x y : ℝ) : Prop :=
  3*x - 4*y - 25 = 0

/-- The minimum distance between a point on the intersection of two circles and a line -/
theorem min_distance_point_to_line
  (a b c d : ℝ)
  (h1 : a * c = 8)
  (h2 : a / b = c / d)
  : ∃ (P : ℝ × ℝ),
    (circle_O₁ a b P.1 P.2 ∧ circle_O₂ c d P.1 P.2) →
    (∀ (M : ℝ × ℝ), line_l M.1 M.2 →
      ∃ (dist : ℝ),
        dist = Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) ∧
        dist ≥ 2 ∧
        (∃ (M₀ : ℝ × ℝ), line_l M₀.1 M₀.2 ∧
          Real.sqrt ((P.1 - M₀.1)^2 + (P.2 - M₀.2)^2) = 2)) :=
sorry

end min_distance_point_to_line_l1829_182900


namespace university_applications_l1829_182915

theorem university_applications (n m k : ℕ) (h1 : n = 7) (h2 : m = 2) (h3 : k = 4) : 
  (Nat.choose (n - m + 1) k) + (Nat.choose m 1 * Nat.choose (n - m) (k - 1)) = 25 := by
  sorry

end university_applications_l1829_182915


namespace find_number_l1829_182998

theorem find_number : ∃! x : ℕ+, 
  (172 / x.val : ℚ) = 172 / 4 - 28 ∧ 
  172 % x.val = 7 ∧ 
  x = 11 := by
  sorry

end find_number_l1829_182998


namespace mikes_video_games_l1829_182917

theorem mikes_video_games :
  ∀ (total_games working_games nonworking_games : ℕ) 
    (price_per_game total_earnings : ℕ),
  nonworking_games = 8 →
  price_per_game = 7 →
  total_earnings = 56 →
  working_games * price_per_game = total_earnings →
  total_games = working_games + nonworking_games →
  total_games = 16 := by
sorry

end mikes_video_games_l1829_182917


namespace tangent_plane_and_normal_line_at_point_A_l1829_182975

-- Define the elliptic paraboloid
def elliptic_paraboloid (x y z : ℝ) : Prop := z = 2 * x^2 + y^2

-- Define the point A
def point_A : ℝ × ℝ × ℝ := (1, -1, 3)

-- Define the tangent plane equation
def tangent_plane (x y z : ℝ) : Prop := 4 * x - 2 * y - z - 3 = 0

-- Define the normal line equations
def normal_line (x y z : ℝ) : Prop :=
  (x - 1) / 4 = (y + 1) / (-2) ∧ (y + 1) / (-2) = (z - 3) / (-1)

-- Theorem statement
theorem tangent_plane_and_normal_line_at_point_A :
  ∀ x y z : ℝ,
  elliptic_paraboloid x y z →
  (x, y, z) = point_A →
  tangent_plane x y z ∧ normal_line x y z :=
sorry

end tangent_plane_and_normal_line_at_point_A_l1829_182975


namespace min_sum_with_product_and_even_constraint_l1829_182909

theorem min_sum_with_product_and_even_constraint (a b : ℤ) : 
  a * b = 72 → Even a → (∀ (x y : ℤ), x * y = 72 → Even x → a + b ≤ x + y) → a + b = -38 := by
  sorry

end min_sum_with_product_and_even_constraint_l1829_182909


namespace clients_equal_cars_l1829_182976

theorem clients_equal_cars (num_cars : ℕ) (selections_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_cars = 18)
  (h2 : selections_per_client = 3)
  (h3 : selections_per_car = 3) :
  (num_cars * selections_per_car) / selections_per_client = num_cars :=
by sorry

end clients_equal_cars_l1829_182976


namespace max_visible_cubes_11_l1829_182947

/-- Represents a cube made of unit cubes --/
structure UnitCube where
  size : ℕ

/-- Calculates the maximum number of visible unit cubes from a single point --/
def max_visible_cubes (cube : UnitCube) : ℕ :=
  3 * cube.size^2 - 3 * (cube.size - 1) + 1

/-- Theorem stating that for an 11x11x11 cube, the maximum number of visible unit cubes is 331 --/
theorem max_visible_cubes_11 :
  max_visible_cubes ⟨11⟩ = 331 := by
  sorry

#eval max_visible_cubes ⟨11⟩

end max_visible_cubes_11_l1829_182947


namespace man_speed_in_still_water_l1829_182904

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 8 km/h. -/
theorem man_speed_in_still_water 
  (s : SwimmerSpeeds)
  (h1 : effectiveSpeed s true = 30 / 3)  -- Downstream condition
  (h2 : effectiveSpeed s false = 18 / 3) -- Upstream condition
  : s.manSpeed = 8 := by
  sorry

#check man_speed_in_still_water

end man_speed_in_still_water_l1829_182904


namespace largest_frog_weight_l1829_182902

theorem largest_frog_weight (S L : ℝ) 
  (h1 : L = 10 * S) 
  (h2 : L = S + 108) : 
  L = 120 := by
sorry

end largest_frog_weight_l1829_182902


namespace second_digit_is_seven_l1829_182982

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10

/-- The second digit of a three-digit number satisfying the given condition is 7 -/
theorem second_digit_is_seven (n : ThreeDigitNumber) :
  100 * n.a + 10 * n.b + n.c - (n.a + n.b + n.c) = 261 → n.b = 7 := by
  sorry

#check second_digit_is_seven

end second_digit_is_seven_l1829_182982


namespace beetle_projection_theorem_l1829_182993

/-- Represents a line in 2D space --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a beetle moving on a line --/
structure Beetle where
  line : Line
  speed : ℝ
  initialPosition : ℝ

/-- Theorem: If two beetles move on intersecting lines with constant speeds,
    and their projections on the OX axis never coincide,
    then their projections on the OY axis must either coincide or have coincided in the past --/
theorem beetle_projection_theorem (L1 L2 : Line) (b1 b2 : Beetle)
    (h_intersect : L1 ≠ L2)
    (h_b1_on_L1 : b1.line = L1)
    (h_b2_on_L2 : b2.line = L2)
    (h_constant_speed : b1.speed ≠ 0 ∧ b2.speed ≠ 0)
    (h_x_proj_never_coincide : ∀ t : ℝ, 
      b1.initialPosition + b1.speed * t ≠ b2.initialPosition + b2.speed * t) :
    ∃ t : ℝ, 
      L1.slope * (b1.initialPosition + b1.speed * t) + L1.intercept = 
      L2.slope * (b2.initialPosition + b2.speed * t) + L2.intercept :=
sorry

end beetle_projection_theorem_l1829_182993


namespace product_evaluation_l1829_182935

theorem product_evaluation : (7 - 5) * (7 - 4) * (7 - 3) * (7 - 2) * (7 - 1) * 7 = 5040 := by
  sorry

end product_evaluation_l1829_182935


namespace factorization_equality_l1829_182908

theorem factorization_equality (a b : ℝ) : (2*a - b)^2 + 8*a*b = (2*a + b)^2 := by
  sorry

end factorization_equality_l1829_182908


namespace nested_function_evaluation_l1829_182912

-- Define P and Q functions
def P (x : ℝ) : ℝ := 3 * (x ^ (1/3))
def Q (x : ℝ) : ℝ := x ^ 3

-- State the theorem
theorem nested_function_evaluation :
  P (Q (P (Q (P (Q 4))))) = 108 :=
sorry

end nested_function_evaluation_l1829_182912


namespace circle_area_l1829_182995

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4*y = 3

-- Define the center and radius of the circle
def circle_center : ℝ × ℝ := (-3, 2)
def circle_radius : ℝ := 4

-- Theorem statement
theorem circle_area :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    (center = circle_center) ∧
    (radius = circle_radius) ∧
    (Real.pi * radius^2 = 16 * Real.pi) :=
sorry

end circle_area_l1829_182995


namespace tennis_preference_theorem_l1829_182968

/-- The percentage of students preferring tennis when combining two schools -/
def combined_tennis_preference 
  (central_students : ℕ) 
  (central_tennis_percentage : ℚ)
  (north_students : ℕ)
  (north_tennis_percentage : ℚ) : ℚ :=
  ((central_students : ℚ) * central_tennis_percentage + 
   (north_students : ℚ) * north_tennis_percentage) / 
  ((central_students + north_students) : ℚ)

theorem tennis_preference_theorem : 
  combined_tennis_preference 1800 (25/100) 3000 (35/100) = 31/100 := by
  sorry

end tennis_preference_theorem_l1829_182968


namespace function_decomposition_symmetry_l1829_182966

theorem function_decomposition_symmetry (f : ℝ → ℝ) :
  ∃ (f₁ f₂ : ℝ → ℝ) (a : ℝ), a > 0 ∧
    (∀ x, f x = f₁ x + f₂ x) ∧
    (∀ x, f₁ (-x) = f₁ x) ∧
    (∀ x, f₂ (2 * a - x) = f₂ x) :=
by sorry

end function_decomposition_symmetry_l1829_182966


namespace tangent_line_at_zero_l1829_182997

noncomputable def f (x : ℝ) : ℝ := Real.cos x - x / 2

theorem tangent_line_at_zero (x y : ℝ) :
  (f 0 = 1) →
  (∀ x, HasDerivAt f (-Real.sin x - 1/2) x) →
  (y = -1/2 * x + 1) →
  (x + 2*y = 2) :=
sorry

end tangent_line_at_zero_l1829_182997


namespace simplify_expression_l1829_182925

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l1829_182925


namespace problem_1_problem_2_l1829_182989

-- Problem 1
theorem problem_1 (x y : ℝ) :
  (x + y) * (x - 2*y) + (x - y)^2 + 3*x * 2*y = 2*x^2 + 3*x*y - y^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx1 : x ≠ 0) (hx2 : x ≠ 1) :
  (x^2 - 4*x + 4) / (x^2 - x) / (x + 1 - 3 / (x - 1)) = (x - 2) / (x * (x + 2)) := by sorry

end problem_1_problem_2_l1829_182989


namespace smallest_square_containing_circle_l1829_182948

theorem smallest_square_containing_circle (r : ℝ) (h : r = 5) :
  (2 * r) ^ 2 = 100 := by
  sorry

end smallest_square_containing_circle_l1829_182948


namespace dessert_distribution_l1829_182984

theorem dessert_distribution (mini_cupcakes : ℕ) (donut_holes : ℕ) (students : ℕ) :
  mini_cupcakes = 14 →
  donut_holes = 12 →
  students = 13 →
  (mini_cupcakes + donut_holes) % students = 0 →
  (mini_cupcakes + donut_holes) / students = 2 :=
by sorry

end dessert_distribution_l1829_182984


namespace sum_of_distinct_words_l1829_182914

/-- Calculates the number of distinct permutations of a word with repeated letters -/
def distinctPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "САМСА" has 5 total letters and 2 letters that repeat twice each -/
def samsa : ℕ := distinctPermutations 5 [2, 2]

/-- The word "ПАСТА" has 5 total letters and 1 letter that repeats twice -/
def pasta : ℕ := distinctPermutations 5 [2]

theorem sum_of_distinct_words : samsa + pasta = 90 := by
  sorry

end sum_of_distinct_words_l1829_182914


namespace factor_expression_l1829_182942

theorem factor_expression (x : ℝ) : 5*x*(x-4) + 6*(x-4) = (x-4)*(5*x+6) := by
  sorry

end factor_expression_l1829_182942


namespace court_cases_dismissed_l1829_182940

theorem court_cases_dismissed (total_cases : ℕ) 
  (remaining_cases : ℕ) (innocent_cases : ℕ) (delayed_cases : ℕ) (guilty_cases : ℕ) :
  total_cases = 17 →
  remaining_cases = innocent_cases + delayed_cases + guilty_cases →
  innocent_cases = 2 * (remaining_cases / 3) →
  delayed_cases = 1 →
  guilty_cases = 4 →
  total_cases - remaining_cases = 2 := by
sorry

end court_cases_dismissed_l1829_182940


namespace sum_equation_l1829_182937

theorem sum_equation (x y z : ℝ) (h1 : x + y = 4) (h2 : x * y = z^2 + 4) : 
  x + 2*y + 3*z = 6 := by
sorry

end sum_equation_l1829_182937


namespace fraction_equality_l1829_182961

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 2 * b) (h2 : b ≠ 0) :
  let x := a / b
  (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) := by
  sorry

end fraction_equality_l1829_182961


namespace four_solutions_l1829_182962

/-- The system of equations has exactly 4 distinct real solutions -/
theorem four_solutions (x y z w : ℝ) : 
  (x = z - w + x * z ∧
   y = w - x + y * w ∧
   z = x - y + x * z ∧
   w = y - z + y * w) →
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 4 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔
    (a = c - d + a * c ∧
     b = d - a + b * d ∧
     c = a - b + a * c ∧
     d = b - c + b * d) :=
by sorry

end four_solutions_l1829_182962


namespace mike_toys_count_l1829_182957

/-- Proves that Mike has 6 toys given the conditions of the problem -/
theorem mike_toys_count :
  ∀ (mike annie tom : ℕ),
  annie = 3 * mike →
  tom = annie + 2 →
  mike + annie + tom = 56 →
  mike = 6 :=
by
  sorry

end mike_toys_count_l1829_182957


namespace square_area_with_corner_circles_l1829_182983

theorem square_area_with_corner_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by sorry

end square_area_with_corner_circles_l1829_182983


namespace ellipse_conditions_l1829_182919

/-- A curve defined by the equation ax^2 + by^2 = 1 -/
structure Curve where
  a : ℝ
  b : ℝ

/-- Predicate for a curve being an ellipse -/
def is_ellipse (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0 ∧ c.a ≠ c.b

/-- The conditions a > 0 and b > 0 -/
def positive_conditions (c : Curve) : Prop :=
  c.a > 0 ∧ c.b > 0

theorem ellipse_conditions (c : Curve) :
  (positive_conditions c → is_ellipse c) ∧
  ¬(is_ellipse c → positive_conditions c) :=
sorry

end ellipse_conditions_l1829_182919


namespace stars_per_student_l1829_182953

theorem stars_per_student (total_students : ℕ) (total_stars : ℕ) 
  (h1 : total_students = 124) 
  (h2 : total_stars = 372) : 
  total_stars / total_students = 3 := by
  sorry

end stars_per_student_l1829_182953


namespace rectangle_area_l1829_182905

theorem rectangle_area : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧
  x * y = (x + 3) * (y - 1) ∧
  x * y = (x - 4) * (y + 1.5) ∧
  x * y = 108 := by
  sorry

end rectangle_area_l1829_182905


namespace expression_value_l1829_182913

theorem expression_value : 
  (2 * 6) / (12 * 14) * (3 * 12 * 14) / (2 * 6 * 3) * 2 = 2 := by sorry

end expression_value_l1829_182913


namespace total_groom_time_l1829_182972

def poodle_groom_time : ℕ := 30
def terrier_groom_time : ℕ := poodle_groom_time / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

theorem total_groom_time :
  num_poodles * poodle_groom_time + num_terriers * terrier_groom_time = 210 := by
  sorry

end total_groom_time_l1829_182972


namespace xiao_ping_weighing_l1829_182964

/-- Represents the weights of 8 items -/
structure EightWeights where
  weights : Fin 8 → ℕ
  distinct : ∀ i j, i ≠ j → weights i ≠ weights j
  bounded : ∀ i, 1 ≤ weights i ∧ weights i ≤ 15

/-- The weighing inequalities -/
def weighing_inequalities (w : EightWeights) : Prop :=
  w.weights 0 + w.weights 4 + w.weights 5 + w.weights 6 >
    w.weights 1 + w.weights 2 + w.weights 3 + w.weights 7 ∧
  w.weights 4 + w.weights 5 > w.weights 0 + w.weights 6 ∧
  w.weights 4 > w.weights 5

theorem xiao_ping_weighing (w : EightWeights) :
  weighing_inequalities w →
  (∀ i, i ≠ 4 → w.weights 4 ≤ w.weights i) →
  (∀ i, i ≠ 3 → w.weights i ≤ w.weights 3) →
  w.weights 4 = 11 ∧ w.weights 6 = 5 := by
  sorry

end xiao_ping_weighing_l1829_182964


namespace rectangle_triangle_count_l1829_182950

/-- A point in a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A rectangle defined by 6 points -/
structure Rectangle :=
  (A B C D E F : Point)

/-- A triangle defined by 3 points -/
structure Triangle :=
  (p1 p2 p3 : Point)

/-- Count the number of triangles with one vertex at a given point -/
def countTriangles (R : Rectangle) (p : Point) : ℕ :=
  sorry

theorem rectangle_triangle_count (R : Rectangle) :
  (countTriangles R R.A = 9) ∧ (countTriangles R R.F = 9) :=
sorry

end rectangle_triangle_count_l1829_182950


namespace existence_condition_l1829_182955

variable {M : Type u}
variable (A B C : Set M)

theorem existence_condition :
  (∃ X : Set M, (X ∪ A) \ (X ∩ B) = C) ↔ 
  ((A ∩ Bᶜ ∩ Cᶜ = ∅) ∧ (Aᶜ ∩ B ∩ C = ∅)) := by sorry

end existence_condition_l1829_182955


namespace perfect_square_condition_l1829_182965

/-- A quadratic trinomial ax^2 + bx + c is a perfect square if there exists a real number r such that ax^2 + bx + c = (√a * x + r)^2 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (Real.sqrt a * x + r)^2

/-- The main theorem: if x^2 - mx + 16 is a perfect square trinomial, then m = 8 or m = -8 -/
theorem perfect_square_condition (m : ℝ) :
  is_perfect_square_trinomial 1 (-m) 16 → m = 8 ∨ m = -8 := by
  sorry

end perfect_square_condition_l1829_182965


namespace rumor_spread_l1829_182958

theorem rumor_spread (n : ℕ) : (∃ m : ℕ, (3^(m+1) - 1) / 2 ≥ 1000 ∧ ∀ k < m, (3^(k+1) - 1) / 2 < 1000) → m = 5 := by
  sorry

end rumor_spread_l1829_182958


namespace instruction_set_exists_l1829_182971

/-- Represents a box that may contain a ball or be empty. -/
inductive Box
| withBall : Box
| empty : Box

/-- Represents an instruction to swap the contents of two boxes. -/
structure SwapInstruction where
  i : Nat
  j : Nat

/-- Represents a configuration of N boxes. -/
def BoxConfiguration (N : Nat) := Fin N → Box

/-- Represents an instruction set. -/
def InstructionSet := List SwapInstruction

/-- Checks if a configuration is sorted (balls to the left of empty boxes). -/
def isSorted (config : BoxConfiguration N) : Prop :=
  ∀ i j, i < j → config i = Box.empty → config j = Box.empty

/-- Applies an instruction set to a configuration. -/
def applyInstructions (config : BoxConfiguration N) (instructions : InstructionSet) : BoxConfiguration N :=
  sorry

/-- The main theorem to be proved. -/
theorem instruction_set_exists (N : Nat) :
  ∃ (instructions : InstructionSet),
    instructions.length ≤ 100 * N ∧
    ∀ (config : BoxConfiguration N),
      ∃ (subset : InstructionSet),
        subset.length ≤ instructions.length ∧
        isSorted (applyInstructions config subset) :=
  sorry

end instruction_set_exists_l1829_182971


namespace three_minus_one_point_two_repeating_l1829_182949

/-- The decimal representation of 1.2 repeating -/
def one_point_two_repeating : ℚ := 11 / 9

/-- Proof that 3 - 1.2 repeating equals 16/9 -/
theorem three_minus_one_point_two_repeating :
  3 - one_point_two_repeating = 16 / 9 := by
  sorry

end three_minus_one_point_two_repeating_l1829_182949


namespace surface_area_of_cut_solid_l1829_182992

/-- A right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  baseSideLength : ℝ

/-- Midpoints of edges in the prism -/
structure Midpoints where
  L : ℝ × ℝ × ℝ
  M : ℝ × ℝ × ℝ
  N : ℝ × ℝ × ℝ

/-- The solid formed by cutting the prism through midpoints -/
def CutSolid (p : RightPrism) (m : Midpoints) : ℝ × ℝ × ℝ × ℝ := sorry

/-- Calculate the surface area of the cut solid -/
def surfaceArea (solid : ℝ × ℝ × ℝ × ℝ) : ℝ := sorry

/-- Main theorem: The surface area of the cut solid -/
theorem surface_area_of_cut_solid (p : RightPrism) (m : Midpoints) :
  p.height = 20 ∧ p.baseSideLength = 10 →
  surfaceArea (CutSolid p m) = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 := by
  sorry

end surface_area_of_cut_solid_l1829_182992


namespace total_pennies_thrown_l1829_182934

/-- The number of pennies thrown by Rachelle, Gretchen, and Rocky -/
def pennies_thrown (rachelle gretchen rocky : ℕ) : ℕ := rachelle + gretchen + rocky

/-- Theorem: The total number of pennies thrown is 300 -/
theorem total_pennies_thrown : 
  ∀ (rachelle gretchen rocky : ℕ),
  rachelle = 180 →
  gretchen = rachelle / 2 →
  rocky = gretchen / 3 →
  pennies_thrown rachelle gretchen rocky = 300 := by
sorry

end total_pennies_thrown_l1829_182934


namespace jonas_bookshelves_l1829_182926

/-- Calculates the maximum number of bookshelves that can fit in a room -/
def max_bookshelves (total_space desk_space shelf_space : ℕ) : ℕ :=
  (total_space - desk_space) / shelf_space

/-- Proves that the maximum number of bookshelves in Jonas' room is 3 -/
theorem jonas_bookshelves :
  max_bookshelves 400 160 80 = 3 := by
  sorry

#eval max_bookshelves 400 160 80

end jonas_bookshelves_l1829_182926


namespace line_circle_intersection_l1829_182916

/-- The line kx - 2y + 1 = 0 always intersects the circle x^2 + (y-1)^2 = 1 for any real k -/
theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), k * x - 2 * y + 1 = 0 ∧ x^2 + (y - 1)^2 = 1 := by sorry

end line_circle_intersection_l1829_182916


namespace planes_parallel_if_perpendicular_to_same_line_l1829_182921

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- Define the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l1829_182921


namespace geometric_sequence_problem_l1829_182969

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a → a 6 = 6 → a 9 = 9 → a 3 = 4 := by
  sorry

end geometric_sequence_problem_l1829_182969


namespace divisibility_equivalence_l1829_182991

theorem divisibility_equivalence (r : ℕ) (k : ℕ) :
  (∃ (m n : ℕ), m > 1 ∧ m % 2 = 1 ∧ k ∣ m^(2^r) - 1 ∧ m ∣ n^((m^(2^r) - 1)/k) + 1) ↔
  (2^(r+1) ∣ k) := by
sorry

end divisibility_equivalence_l1829_182991


namespace pattern_steps_l1829_182963

/-- The number of sticks used in the kth step of the pattern -/
def sticks_in_step (k : ℕ) : ℕ := 2 * k + 1

/-- The total number of sticks used in a pattern with n steps -/
def total_sticks (n : ℕ) : ℕ := n^2

/-- Theorem: If a stair-like pattern is constructed where the kth step uses 2k + 1 sticks,
    and the total number of sticks used is 169, then the number of steps in the pattern is 13 -/
theorem pattern_steps :
  ∀ n : ℕ, (∀ k : ℕ, k ≤ n → sticks_in_step k = 2 * k + 1) →
  total_sticks n = 169 → n = 13 := by sorry

end pattern_steps_l1829_182963


namespace arithmetic_sequence_sum_l1829_182944

/-- An arithmetic sequence with first term 1 and sum of first n terms S_n -/
structure ArithSeq where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 1
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum (seq : ArithSeq) 
  (h : seq.S 19 / 19 - seq.S 17 / 17 = 6) : 
  seq.S 10 = 280 := by
  sorry

end arithmetic_sequence_sum_l1829_182944


namespace integral_proof_l1829_182999

open Real

noncomputable def f (x : ℝ) := (1/2) * log (abs (x^2 + 2 * sin x))

theorem integral_proof (x : ℝ) :
  deriv f x = (x + cos x) / (x^2 + 2 * sin x) :=
by sorry

end integral_proof_l1829_182999


namespace dependent_variable_influence_l1829_182918

/-- Linear regression model -/
structure LinearRegressionModel where
  y : ℝ → ℝ  -- Dependent variable
  x : ℝ      -- Independent variable
  b : ℝ      -- Slope
  a : ℝ      -- Intercept
  e : ℝ → ℝ  -- Random error term

/-- The dependent variable is influenced by both the independent variable and other factors -/
theorem dependent_variable_influence (model : LinearRegressionModel) :
  ∃ (x₁ x₂ : ℝ), model.y x₁ ≠ model.y x₂ ∧ model.x = model.x :=
by sorry

end dependent_variable_influence_l1829_182918


namespace tan_squared_sum_l1829_182945

theorem tan_squared_sum (a b : ℝ) 
  (h1 : (Real.sin a)^2 / (Real.cos b)^2 + (Real.sin b)^2 / (Real.cos a)^2 = 2)
  (h2 : (Real.cos a)^3 / (Real.sin b)^3 + (Real.cos b)^3 / (Real.sin a)^3 = 4) :
  (Real.tan a)^2 / (Real.tan b)^2 + (Real.tan b)^2 / (Real.tan a)^2 = 30/13 := by
  sorry

end tan_squared_sum_l1829_182945


namespace strawberries_eaten_l1829_182943

theorem strawberries_eaten (initial : Float) (remaining : Nat) (eaten : Nat) : 
  initial = 78.0 → remaining = 36 → eaten = 42 → initial - remaining.toFloat = eaten.toFloat := by
  sorry

end strawberries_eaten_l1829_182943


namespace markers_final_count_l1829_182970

def markers_problem (initial : ℕ) (robert_gave : ℕ) (sarah_took : ℕ) (teacher_multiplier : ℕ) : ℕ :=
  let after_robert := initial + robert_gave
  let after_sarah := after_robert - sarah_took
  let after_teacher := after_sarah + teacher_multiplier * after_sarah
  (after_teacher) / 2

theorem markers_final_count : 
  markers_problem 217 109 35 3 = 582 := by sorry

end markers_final_count_l1829_182970


namespace percentage_difference_l1829_182901

theorem percentage_difference (n : ℝ) (x y : ℝ) 
  (h1 : n = 160) 
  (h2 : x > y) 
  (h3 : (x / 100) * n - (y / 100) * n = 24) : 
  x - y = 15 := by
sorry

end percentage_difference_l1829_182901


namespace simplify_expression_1_simplify_expression_2_l1829_182959

-- First expression
theorem simplify_expression_1 (a b : ℝ) :
  4 * (a + b) + 2 * (a + b) - (a + b) = 5 * a + 5 * b := by sorry

-- Second expression
theorem simplify_expression_2 (m : ℝ) :
  3 * m / 2 - (5 * m / 2 - 1) + 3 * (4 - m) = -4 * m + 13 := by sorry

end simplify_expression_1_simplify_expression_2_l1829_182959


namespace custom_op_difference_l1829_182930

/-- Custom operation @ defined as x@y = xy - 3x -/
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

/-- Theorem stating that (6@2)-(2@6) = -12 -/
theorem custom_op_difference : at_op 6 2 - at_op 2 6 = -12 := by
  sorry

end custom_op_difference_l1829_182930


namespace max_value_sum_l1829_182990

/-- Given positive real numbers x, y, and z satisfying 4x^2 + 9y^2 + 16z^2 = 144,
    the maximum value N of the expression 3xz + 5yz + 8xy plus the sum of x, y, and z
    that produce this maximum is equal to 319. -/
theorem max_value_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 4*x^2 + 9*y^2 + 16*z^2 = 144) :
  ∃ (N x_N y_N z_N : ℝ),
    (∀ x' y' z' : ℝ, x' > 0 → y' > 0 → z' > 0 → 4*x'^2 + 9*y'^2 + 16*z'^2 = 144 →
      3*x'*z' + 5*y'*z' + 8*x'*y' ≤ N) ∧
    3*x_N*z_N + 5*y_N*z_N + 8*x_N*y_N = N ∧
    4*x_N^2 + 9*y_N^2 + 16*z_N^2 = 144 ∧
    N + x_N + y_N + z_N = 319 :=
by sorry

end max_value_sum_l1829_182990


namespace candy_bar_difference_l1829_182996

theorem candy_bar_difference (lena kevin nicole : ℕ) : 
  lena = 16 →
  lena + 5 = 3 * kevin →
  kevin + 4 = nicole →
  lena - nicole = 5 := by
sorry

end candy_bar_difference_l1829_182996
