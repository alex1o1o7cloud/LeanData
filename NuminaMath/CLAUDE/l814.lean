import Mathlib

namespace classmates_lateness_l814_81465

theorem classmates_lateness 
  (charlize_lateness : ℕ) 
  (total_lateness : ℕ) 
  (num_classmates : ℕ) 
  (h1 : charlize_lateness = 20)
  (h2 : total_lateness = 140)
  (h3 : num_classmates = 4) :
  (total_lateness - charlize_lateness) / num_classmates = 30 :=
by sorry

end classmates_lateness_l814_81465


namespace tangent_line_of_cubic_l814_81428

/-- Given a cubic function f(x) with specific derivative conditions, 
    prove that its tangent line at x = 1 has a specific equation. -/
theorem tangent_line_of_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 + 2*a*x + b
  (f' 1 = 2*a) → (f' 2 = -b) → 
  ∃ m c : ℝ, m = -3 ∧ c = -5/2 ∧ 
    (∀ x y : ℝ, y - c = m * (x - 1) ↔ 6*x + 2*y - 1 = 0) :=
by sorry

end tangent_line_of_cubic_l814_81428


namespace direct_proportion_shift_right_l814_81416

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shift a linear function horizontally -/
def shift_right (f : LinearFunction) (shift : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.slope * shift + f.intercept }

theorem direct_proportion_shift_right :
  let f : LinearFunction := { slope := -2, intercept := 0 }
  let shifted_f := shift_right f 3
  shifted_f.slope = -2 ∧ shifted_f.intercept = 6 := by sorry

end direct_proportion_shift_right_l814_81416


namespace diameter_of_circumscribing_circle_l814_81481

/-- The diameter of a circle circumscribing six smaller tangent circles -/
theorem diameter_of_circumscribing_circle (r : ℝ) (h : r = 5) :
  let small_circle_radius : ℝ := r
  let small_circle_count : ℕ := 6
  let inner_hexagon_side : ℝ := 2 * small_circle_radius
  let inner_hexagon_radius : ℝ := inner_hexagon_side
  let large_circle_radius : ℝ := inner_hexagon_radius + small_circle_radius
  large_circle_radius * 2 = 30 := by sorry

end diameter_of_circumscribing_circle_l814_81481


namespace smallest_addend_proof_l814_81496

/-- The smallest non-negative integer that, when added to 27452, makes the sum divisible by 9 -/
def smallest_addend : ℕ := 7

/-- The original number we're working with -/
def original_number : ℕ := 27452

theorem smallest_addend_proof :
  (∀ k : ℕ, k < smallest_addend → ¬((original_number + k) % 9 = 0)) ∧
  ((original_number + smallest_addend) % 9 = 0) := by
  sorry

end smallest_addend_proof_l814_81496


namespace unique_three_digit_number_l814_81499

theorem unique_three_digit_number : ∃! (m g u : ℕ),
  m ≠ g ∧ m ≠ u ∧ g ≠ u ∧
  m < 10 ∧ g < 10 ∧ u < 10 ∧
  m ≥ 1 ∧
  100 * m + 10 * g + u = (m + g + u) * (m + g + u - 2) ∧
  100 * m + 10 * g + u = 195 := by
sorry

end unique_three_digit_number_l814_81499


namespace right_triangle_trig_identity_l814_81464

/-- Given a right triangle PQR with hypotenuse PQ = 15 and PR = 9, prove that sin Q = 4/5 and the trigonometric identity sin² Q + cos² Q = 1 holds. -/
theorem right_triangle_trig_identity (PQ PR : ℝ) (hPQ : PQ = 15) (hPR : PR = 9) :
  let sinQ := Real.sqrt (PQ^2 - PR^2) / PQ
  let cosQ := PR / PQ
  sinQ = 4/5 ∧ sinQ^2 + cosQ^2 = 1 := by
  sorry

end right_triangle_trig_identity_l814_81464


namespace bowtie_equation_solution_l814_81401

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 7 y = 14 ∧ y = 42 :=
by sorry

end bowtie_equation_solution_l814_81401


namespace stratified_sampling_sample_size_l814_81462

theorem stratified_sampling_sample_size 
  (total_teachers : ℕ) 
  (total_male_students : ℕ) 
  (total_female_students : ℕ) 
  (sampled_female_students : ℕ) 
  (h1 : total_teachers = 100) 
  (h2 : total_male_students = 600) 
  (h3 : total_female_students = 500) 
  (h4 : sampled_female_students = 40) : 
  (sampled_female_students : ℚ) / total_female_students = 
  96 / (total_teachers + total_male_students + total_female_students) := by
sorry

end stratified_sampling_sample_size_l814_81462


namespace concrete_volume_l814_81405

-- Define constants
def sidewalk_width : ℚ := 4/3  -- in yards
def sidewalk_length : ℚ := 80/3  -- in yards
def sidewalk_thickness : ℚ := 1/9  -- in yards
def border_width : ℚ := 2/3  -- in yards (1 foot on each side)
def border_thickness : ℚ := 1/18  -- in yards

-- Define the theorem
theorem concrete_volume : 
  let sidewalk_volume := sidewalk_width * sidewalk_length * sidewalk_thickness
  let border_volume := border_width * sidewalk_length * border_thickness
  let total_volume := sidewalk_volume + border_volume
  ⌈total_volume⌉ = 6 := by
sorry


end concrete_volume_l814_81405


namespace amanda_almonds_l814_81448

theorem amanda_almonds (total_almonds : ℚ) (num_piles : ℕ) (amanda_piles : ℕ) : 
  total_almonds = 66 / 7 →
  num_piles = 6 →
  amanda_piles = 3 →
  amanda_piles * (total_almonds / num_piles) = 33 / 7 := by
  sorry

end amanda_almonds_l814_81448


namespace max_S_n_is_three_halves_l814_81434

/-- Given a geometric sequence {a_n} with first term 3/2 and common ratio -1/2,
    S_n is the sum of the first n terms. -/
def S_n (n : ℕ) : ℚ :=
  (3/2) * (1 - (-1/2)^n) / (1 - (-1/2))

/-- The maximum value of S_n is 3/2. -/
theorem max_S_n_is_three_halves :
  ∃ (M : ℚ), M = 3/2 ∧ ∀ (n : ℕ), S_n n ≤ M :=
sorry

end max_S_n_is_three_halves_l814_81434


namespace square_difference_equality_l814_81404

theorem square_difference_equality : (25 + 15)^2 - (25 - 15)^2 = 1500 := by
  sorry

end square_difference_equality_l814_81404


namespace cricket_average_increase_l814_81435

def increase_average (current_innings : ℕ) (current_average : ℚ) (next_innings_runs : ℕ) : ℚ :=
  let total_runs := current_innings * current_average
  let new_total_runs := total_runs + next_innings_runs
  let new_average := new_total_runs / (current_innings + 1)
  new_average - current_average

theorem cricket_average_increase :
  increase_average 12 48 178 = 10 := by sorry

end cricket_average_increase_l814_81435


namespace bales_in_barn_l814_81488

/-- The number of bales in the barn after Tim's addition -/
def total_bales (initial_bales added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- Theorem stating that the total number of bales after Tim's addition is 54 -/
theorem bales_in_barn (initial_bales added_bales : ℕ) 
  (h1 : initial_bales = 28)
  (h2 : added_bales = 26) :
  total_bales initial_bales added_bales = 54 := by
  sorry

end bales_in_barn_l814_81488


namespace billboard_perimeter_l814_81449

/-- Represents a rectangular billboard --/
structure Billboard where
  length : ℝ
  width : ℝ

/-- The area of a billboard --/
def area (b : Billboard) : ℝ := b.length * b.width

/-- The perimeter of a billboard --/
def perimeter (b : Billboard) : ℝ := 2 * (b.length + b.width)

theorem billboard_perimeter :
  ∀ b : Billboard,
    area b = 91 ∧
    b.width = 7 →
    perimeter b = 40 := by
  sorry

end billboard_perimeter_l814_81449


namespace incorrect_statement_l814_81445

theorem incorrect_statement : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (p ∨ q)) := by
  sorry

end incorrect_statement_l814_81445


namespace line_bisects_and_perpendicular_l814_81491

/-- The circle C in the xy-plane -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y + 1 = 0

/-- The line perpendicular to l -/
def PerpendicularLine (x y : ℝ) : Prop := x + 2*y + 3 = 0

/-- The line l -/
def Line_l (x y : ℝ) : Prop := 2*x - y + 2 = 0

/-- Theorem stating that line l bisects circle C and is perpendicular to the given line -/
theorem line_bisects_and_perpendicular :
  (∀ x y : ℝ, Line_l x y → (∃ x' y' : ℝ, Circle x' y' ∧ x = (x' + (-1/2))/2 ∧ y = (y' + 1)/2)) ∧ 
  (∀ x y : ℝ, Line_l x y → PerpendicularLine x y → x * 2 + y * 1 = 0) :=
sorry

end line_bisects_and_perpendicular_l814_81491


namespace regular_polygon_center_containment_l814_81444

/-- A regular polygon with 2n sides -/
structure RegularPolygon (n : ℕ) where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Predicate to check if one polygon is inside another -/
def isInside (p1 p2 : RegularPolygon n) : Prop := sorry

/-- Predicate to check if a point is inside a polygon -/
def containsPoint (p : RegularPolygon n) (point : ℝ × ℝ) : Prop := sorry

theorem regular_polygon_center_containment (n : ℕ) (a : ℝ) 
  (M₁ : RegularPolygon n) (M₂ : RegularPolygon n) 
  (h1 : M₁.sideLength = a) 
  (h2 : M₂.sideLength = 2 * a) 
  (h3 : isInside M₁ M₂) :
  containsPoint M₁ M₂.center := by
  sorry

end regular_polygon_center_containment_l814_81444


namespace bananas_arrangements_eq_240_l814_81403

/-- The number of letters in the word BANANAS -/
def total_letters : ℕ := 7

/-- The number of 'B's in BANANAS -/
def count_B : ℕ := 1

/-- The number of 'A's in BANANAS -/
def count_A : ℕ := 3

/-- The number of 'N's in BANANAS -/
def count_N : ℕ := 1

/-- The number of 'S's in BANANAS -/
def count_S : ℕ := 2

/-- The function to calculate the number of arrangements of BANANAS with no 'A' at the first position -/
def bananas_arrangements : ℕ := sorry

/-- Theorem stating that the number of arrangements of BANANAS with no 'A' at the first position is 240 -/
theorem bananas_arrangements_eq_240 : bananas_arrangements = 240 := by sorry

end bananas_arrangements_eq_240_l814_81403


namespace magic_square_y_value_l814_81424

/-- Represents a 3x3 magic square -/
def MagicSquare (a b c d e f g h i : ℚ) : Prop :=
  a + b + c = d + e + f ∧
  d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧
  b + e + h = c + f + i ∧
  a + e + i = c + e + g

theorem magic_square_y_value :
  ∀ (y a b c d e : ℚ),
  MagicSquare y 7 24 8 a b c d e →
  y = 39.5 := by
  sorry

end magic_square_y_value_l814_81424


namespace carol_peanuts_l814_81422

/-- The total number of peanuts Carol has -/
def total_peanuts (tree_peanuts ground_peanuts bags bag_capacity : ℕ) : ℕ :=
  tree_peanuts + ground_peanuts + bags * bag_capacity

/-- Theorem: Carol has 976 peanuts in total -/
theorem carol_peanuts :
  total_peanuts 48 178 3 250 = 976 := by
  sorry

end carol_peanuts_l814_81422


namespace final_score_l814_81455

def bullseye_points : ℕ := 50

def dart_throws (bullseye half_bullseye miss : ℕ) : Prop :=
  bullseye = bullseye_points ∧
  half_bullseye = bullseye_points / 2 ∧
  miss = 0

theorem final_score (bullseye half_bullseye miss : ℕ) 
  (h : dart_throws bullseye half_bullseye miss) : 
  bullseye + half_bullseye + miss = 75 := by
  sorry

end final_score_l814_81455


namespace simplify_sqrt_sum_l814_81458

theorem simplify_sqrt_sum : 2 * Real.sqrt 8 + 3 * Real.sqrt 32 = 16 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_sum_l814_81458


namespace equivalent_statements_l814_81495

theorem equivalent_statements :
  (∀ x : ℝ, x ≥ 0 → x^2 ≤ 0) ↔ (∀ x : ℝ, x^2 > 0 → x < 0) :=
by sorry

end equivalent_statements_l814_81495


namespace integral_x_cubed_minus_reciprocal_x_fourth_l814_81406

theorem integral_x_cubed_minus_reciprocal_x_fourth (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - 1/x^4) →
  ∫ x in (-1)..1, f x = 2/3 := by sorry

end integral_x_cubed_minus_reciprocal_x_fourth_l814_81406


namespace mirasol_account_balance_l814_81447

def remaining_balance (initial_balance : ℕ) (expense1 : ℕ) (expense2 : ℕ) : ℕ :=
  initial_balance - (expense1 + expense2)

theorem mirasol_account_balance : remaining_balance 50 10 30 = 10 := by
  sorry

end mirasol_account_balance_l814_81447


namespace article_price_proof_l814_81492

-- Define the selling price and loss percentage
def selling_price : ℝ := 800
def loss_percentage : ℝ := 33.33333333333333

-- Define the original price
def original_price : ℝ := 1200

-- Theorem statement
theorem article_price_proof :
  (selling_price = (1 - loss_percentage / 100) * original_price) → 
  (original_price = 1200) := by
  sorry

end article_price_proof_l814_81492


namespace longest_segment_in_cylinder_l814_81437

/-- The longest segment in a cylinder with radius 5 and height 12 is 2√61 -/
theorem longest_segment_in_cylinder (r h : ℝ) (hr : r = 5) (hh : h = 12) :
  Real.sqrt ((2 * r)^2 + h^2) = 2 * Real.sqrt 61 := by
  sorry

end longest_segment_in_cylinder_l814_81437


namespace equal_roots_cubic_polynomial_l814_81407

theorem equal_roots_cubic_polynomial (m : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 135 * a + m = 0) ∧
              (3 * b^3 + 9 * b^2 - 135 * b + m = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 135 * x + m = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 135 * y + m = 0 ∧
                      (∀ z : ℝ, 3 * z^3 + 9 * z^2 - 135 * z + m = 0 → z = x ∨ z = y))) ∧
  m > 0 →
  m = 22275 :=
by sorry

end equal_roots_cubic_polynomial_l814_81407


namespace least_addition_for_divisibility_l814_81454

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x ≤ 24 ∧ (1015 + x) % 25 = 0 ∧ ∀ y : ℕ, y < x → (1015 + y) % 25 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l814_81454


namespace team_b_score_l814_81433

/-- Given a trivia game where:
  * Team A scored 2 points
  * Team C scored 4 points
  * The total points scored by all teams is 15
  Prove that Team B scored 9 points -/
theorem team_b_score (team_a_score team_c_score total_score : ℕ)
  (h1 : team_a_score = 2)
  (h2 : team_c_score = 4)
  (h3 : total_score = 15) :
  total_score - (team_a_score + team_c_score) = 9 := by
  sorry

end team_b_score_l814_81433


namespace max_y_coordinate_value_l814_81473

noncomputable def max_y_coordinate (θ : ℝ) : ℝ :=
  let r := Real.sin (3 * θ)
  r * Real.sin θ

theorem max_y_coordinate_value :
  ∃ (θ : ℝ), ∀ (φ : ℝ), max_y_coordinate θ ≥ max_y_coordinate φ ∧
  max_y_coordinate θ = 3 * (3 / 16) ^ (1 / 3) - 4 * 3 ^ (4 / 3) / 16 ^ (4 / 3) :=
sorry

end max_y_coordinate_value_l814_81473


namespace total_cost_approx_636_38_l814_81470

def membership_fee (initial_fee : ℝ) (increase_rates : List ℝ) (discount_rates : List ℝ) : ℝ :=
  let fees := List.scanl (λ acc rate => acc * (1 + rate)) initial_fee increase_rates
  let discounted_fees := List.zipWith (λ fee discount => fee * (1 - discount)) fees discount_rates
  discounted_fees.sum

def total_cost : ℝ :=
  membership_fee 80 [0.1, 0.12, 0.14, 0.15, 0.15, 0.15] [0, 0, 0, 0, 0.1, 0.05]

theorem total_cost_approx_636_38 : 
  ∃ ε > 0, abs (total_cost - 636.38) < ε :=
sorry

end total_cost_approx_636_38_l814_81470


namespace inequality_of_product_one_l814_81415

theorem inequality_of_product_one (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end inequality_of_product_one_l814_81415


namespace expression_equals_one_tenth_l814_81483

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the expression
def expression : ℚ :=
  (ceiling ((21 : ℚ) / 8 - ceiling ((35 : ℚ) / 21))) /
  (ceiling ((35 : ℚ) / 8 + ceiling ((8 * 21 : ℚ) / 35)))

-- Theorem statement
theorem expression_equals_one_tenth : expression = 1 / 10 := by
  sorry

end expression_equals_one_tenth_l814_81483


namespace line_sum_m_b_l814_81409

/-- A line passing through points (-2, 0) and (0, 2) can be represented by y = mx + b -/
def line_equation (x y m b : ℝ) : Prop := y = m * x + b

/-- The line passes through (-2, 0) -/
def point1_condition (m b : ℝ) : Prop := line_equation (-2) 0 m b

/-- The line passes through (0, 2) -/
def point2_condition (m b : ℝ) : Prop := line_equation 0 2 m b

/-- Theorem: For a line passing through (-2, 0) and (0, 2), represented by y = mx + b, m + b = 3 -/
theorem line_sum_m_b : ∀ m b : ℝ, point1_condition m b → point2_condition m b → m + b = 3 := by
  sorry

end line_sum_m_b_l814_81409


namespace perspective_right_angle_l814_81439

-- Define the types for points and triangles
def Point : Type := ℝ × ℝ
def Triangle : Type := Point × Point × Point

-- Define the perspective transformation
def perspective_transform : Triangle → Triangle := sorry

-- Define the property of being horizontally placed
def is_horizontal (t : Triangle) : Prop := sorry

-- Define the property of a line being parallel to y' axis
def parallel_to_y_axis (p q : Point) : Prop := sorry

-- Define the property of a line being on x' axis
def on_x_axis (p q : Point) : Prop := sorry

-- Define the property of the angle formed by x'o'y' being 45°
def x_o_y_angle_45 (t : Triangle) : Prop := sorry

-- Define a right-angled triangle
def is_right_angled (t : Triangle) : Prop := sorry

-- The main theorem
theorem perspective_right_angle 
  (abc : Triangle) 
  (a'b'c' : Triangle) 
  (h1 : is_horizontal abc)
  (h2 : a'b'c' = perspective_transform abc)
  (h3 : parallel_to_y_axis a'b'c'.1 a'b'c'.2.1)
  (h4 : on_x_axis a'b'c'.2.1 a'b'c'.2.2)
  (h5 : x_o_y_angle_45 a'b'c') :
  is_right_angled abc :=
sorry

end perspective_right_angle_l814_81439


namespace edward_lawn_problem_l814_81466

/-- The number of dollars Edward earns per lawn -/
def dollars_per_lawn : ℕ := 4

/-- The number of lawns Edward forgot to mow -/
def forgotten_lawns : ℕ := 9

/-- The amount of money Edward actually earned -/
def actual_earnings : ℕ := 32

/-- The original number of lawns Edward had to mow -/
def original_lawns : ℕ := 17

theorem edward_lawn_problem :
  dollars_per_lawn * (original_lawns - forgotten_lawns) = actual_earnings :=
by sorry

end edward_lawn_problem_l814_81466


namespace coord_sum_of_point_on_line_l814_81487

/-- Given two points A and B in a 2D plane, where A is at the origin and B is on the line y = 5,
    if the slope of segment AB is 3/4, then the sum of the x- and y-coordinates of B is 35/3. -/
theorem coord_sum_of_point_on_line (B : ℝ × ℝ) : 
  B.2 = 5 →  -- B is on the line y = 5
  (B.2 - 0) / (B.1 - 0) = 3/4 →  -- slope of AB is 3/4
  B.1 + B.2 = 35/3 := by
sorry

end coord_sum_of_point_on_line_l814_81487


namespace family_boys_count_l814_81429

/-- Represents a family with boys and girls -/
structure Family where
  boys : ℕ
  girls : ℕ

/-- A child in the family -/
structure Child where
  brothers : ℕ
  sisters : ℕ

/-- Defines a valid family based on the problem conditions -/
def isValidFamily (f : Family) : Prop :=
  ∃ (c1 c2 : Child),
    c1.brothers = 3 ∧ c1.sisters = 6 ∧
    c2.brothers = 4 ∧ c2.sisters = 5 ∧
    f.boys = c1.brothers + 1 ∧
    f.girls = c1.sisters + 1

theorem family_boys_count (f : Family) :
  isValidFamily f → f.boys = 4 :=
by
  sorry

end family_boys_count_l814_81429


namespace magic_king_episodes_l814_81486

/-- Calculates the total number of episodes for a TV show with a given number of seasons and episodes per season for each half. -/
def totalEpisodes (totalSeasons : ℕ) (episodesFirstHalf : ℕ) (episodesSecondHalf : ℕ) : ℕ :=
  let halfSeasons := totalSeasons / 2
  halfSeasons * episodesFirstHalf + halfSeasons * episodesSecondHalf

/-- Proves that a show with 10 seasons, 20 episodes per season for the first half, and 25 episodes per season for the second half has 225 total episodes. -/
theorem magic_king_episodes : totalEpisodes 10 20 25 = 225 := by
  sorry

end magic_king_episodes_l814_81486


namespace twentieth_fisherman_catch_l814_81456

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) 
  (fishermen_with_400 : ℕ) (fish_per_fisherman : ℕ) :
  total_fishermen = 20 →
  total_fish = 10000 →
  fishermen_with_400 = 19 →
  fish_per_fisherman = 400 →
  total_fish - (fishermen_with_400 * fish_per_fisherman) = 2400 := by
sorry

end twentieth_fisherman_catch_l814_81456


namespace consecutive_numbers_with_lcm_168_l814_81475

theorem consecutive_numbers_with_lcm_168 (a b c : ℕ) : 
  (b = a + 1) → (c = b + 1) → Nat.lcm (Nat.lcm a b) c = 168 → a + b + c = 21 := by
  sorry

end consecutive_numbers_with_lcm_168_l814_81475


namespace perfect_square_sum_l814_81453

theorem perfect_square_sum (a b : ℕ) :
  (∃ k : ℕ, 2^(2*a) + 2^b + 5 = k^2) → (a + b = 4 ∨ a + b = 5) := by
  sorry

end perfect_square_sum_l814_81453


namespace product_digits_sum_l814_81426

def A : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def B : ℕ := 909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909090909

def C : ℕ := (A * B) / 10000 % 10
def D : ℕ := A * B % 10

theorem product_digits_sum : C + D = 6 := by
  sorry

end product_digits_sum_l814_81426


namespace land_plot_area_land_plot_area_is_1267200_l814_81484

/-- Calculates the total area of a land plot in acres given the dimensions in cm and conversion factors. -/
theorem land_plot_area 
  (triangle_base : ℝ) 
  (triangle_height : ℝ) 
  (rect_length : ℝ) 
  (rect_width : ℝ) 
  (scale_cm_to_miles : ℝ) 
  (acres_per_sq_mile : ℝ) : ℝ :=
  let triangle_area := (1/2) * triangle_base * triangle_height
  let rect_area := rect_length * rect_width
  let total_area_cm2 := triangle_area + rect_area
  let total_area_miles2 := total_area_cm2 * (scale_cm_to_miles^2)
  let total_area_acres := total_area_miles2 * acres_per_sq_mile
  total_area_acres

/-- Proves that the total area of the given land plot is 1267200 acres. -/
theorem land_plot_area_is_1267200 : 
  land_plot_area 20 12 20 5 3 640 = 1267200 := by
  sorry

end land_plot_area_land_plot_area_is_1267200_l814_81484


namespace intersection_complement_theorem_l814_81421

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {2, 3, 4}

theorem intersection_complement_theorem : A ∩ (U \ B) = {1, 6} := by
  sorry

end intersection_complement_theorem_l814_81421


namespace mistaken_division_l814_81482

theorem mistaken_division (n : ℕ) (h : 2 * n = 622) : 
  (n / 12) + (n % 12) = 36 := by
sorry

end mistaken_division_l814_81482


namespace intersection_M_N_l814_81418

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {-1, 0, 1}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end intersection_M_N_l814_81418


namespace k_value_l814_81438

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer. -/
def length (k : ℕ) : ℕ := sorry

/-- k is an integer greater than 1 with a length of 4 and prime factors 2, 2, 2, and 3 -/
def k : ℕ := sorry

theorem k_value : k = 24 := by sorry

end k_value_l814_81438


namespace top_card_is_eleven_l814_81480

/-- Represents a card in the array -/
structure Card where
  row : Fin 3
  col : Fin 6
  number : Fin 18

/-- Represents the initial 3x6 array of cards -/
def initial_array : Array (Array Card) := sorry

/-- Folds the left third over the middle third -/
def fold_left_third (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Folds the right third over the overlapped left and middle thirds -/
def fold_right_third (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Folds the bottom half over the top half -/
def fold_bottom_half (arr : Array (Array Card)) : Array (Array Card) := sorry

/-- Performs all folding operations -/
def perform_all_folds (arr : Array (Array Card)) : Array (Array Card) :=
  arr |> fold_left_third |> fold_right_third |> fold_bottom_half

/-- The top card after all folds -/
def top_card (arr : Array (Array Card)) : Card := sorry

theorem top_card_is_eleven :
  (top_card (perform_all_folds initial_array)).number = 11 := by
  sorry

end top_card_is_eleven_l814_81480


namespace girls_pass_percentage_l814_81419

theorem girls_pass_percentage 
  (total_boys : ℕ) 
  (total_girls : ℕ) 
  (boys_pass_rate : ℚ) 
  (total_fail_rate : ℚ) :
  total_boys = 50 →
  total_girls = 100 →
  boys_pass_rate = 1/2 →
  total_fail_rate = 5667/10000 →
  (total_girls - (((total_boys + total_girls) * total_fail_rate).floor - (total_boys * (1 - boys_pass_rate)).floor)) / total_girls = 2/5 := by
  sorry

end girls_pass_percentage_l814_81419


namespace fifth_month_sale_l814_81432

theorem fifth_month_sale
  (target_average : ℕ)
  (num_months : ℕ)
  (sales : Fin 4 → ℕ)
  (sixth_month_sale : ℕ)
  (h1 : target_average = 6000)
  (h2 : num_months = 5)
  (h3 : sales 0 = 5420)
  (h4 : sales 1 = 5660)
  (h5 : sales 2 = 6200)
  (h6 : sales 3 = 6350)
  (h7 : sixth_month_sale = 5870) :
  ∃ (fifth_month_sale : ℕ),
    fifth_month_sale = target_average * num_months - (sales 0 + sales 1 + sales 2 + sales 3) :=
by sorry

end fifth_month_sale_l814_81432


namespace division_problem_l814_81430

theorem division_problem (x y : ℤ) (k : ℕ) (h1 : x > 0) 
  (h2 : x = 11 * y + 4) 
  (h3 : 2 * x = k * (3 * y) + 1) 
  (h4 : 7 * y - x = 3) : 
  k = 6 := by sorry

end division_problem_l814_81430


namespace most_precise_announcement_l814_81463

def K_approx : ℝ := 5.72788
def error_margin : ℝ := 0.00625

def is_valid_announcement (x : ℝ) : Prop :=
  ∀ y : ℝ, |y - K_approx| ≤ error_margin → |x - y| < 0.05

theorem most_precise_announcement :
  is_valid_announcement 5.7 ∧
  ∀ z : ℝ, is_valid_announcement z → |z - 5.7| < 0.05 :=
sorry

end most_precise_announcement_l814_81463


namespace cos_eighteen_degrees_l814_81459

theorem cos_eighteen_degrees : Real.cos (18 * π / 180) = (5 + Real.sqrt 5) / 8 := by
  sorry

end cos_eighteen_degrees_l814_81459


namespace arithmetic_progression_formula_geometric_progression_formula_l814_81400

-- Arithmetic Progression
def arithmeticProgression (u₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := u₁ + (n - 1 : ℝ) * d

-- Geometric Progression
def geometricProgression (u₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := u₁ * q ^ (n - 1)

theorem arithmetic_progression_formula (u₁ d : ℝ) (n : ℕ) :
  ∀ k : ℕ, k ≤ n → arithmeticProgression u₁ d k = u₁ + (k - 1 : ℝ) * d :=
by sorry

theorem geometric_progression_formula (u₁ q : ℝ) (n : ℕ) :
  ∀ k : ℕ, k ≤ n → geometricProgression u₁ q k = u₁ * q ^ (k - 1) :=
by sorry

end arithmetic_progression_formula_geometric_progression_formula_l814_81400


namespace expression_evaluation_l814_81414

theorem expression_evaluation : 
  let x : ℝ := -2
  3 * (-2 * x^2 + 5 + 4 * x) - (5 * x - 4 - 7 * x^2) = 9 := by
sorry

end expression_evaluation_l814_81414


namespace rectangle_square_overlap_l814_81477

theorem rectangle_square_overlap (s w h : ℝ) 
  (h1 : 0.4 * s^2 = 0.25 * w * h) 
  (h2 : w = 4 * h) : 
  w / h = 4 := by
sorry

end rectangle_square_overlap_l814_81477


namespace greatest_n_value_l814_81493

theorem greatest_n_value (n : ℤ) (h : 102 * n^2 ≤ 8100) : n ≤ 8 ∧ ∃ (m : ℤ), m = 8 ∧ 102 * m^2 ≤ 8100 := by
  sorry

end greatest_n_value_l814_81493


namespace initial_roses_l814_81423

theorem initial_roses (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 10 → total = 16 → total = initial + added → initial = 6 := by
  sorry

end initial_roses_l814_81423


namespace growth_rate_equation_l814_81441

/-- Represents the average annual growth rate of a company's capital -/
def x : ℝ := sorry

/-- The initial capital of the company in millions of yuan -/
def initial_capital : ℝ := 10

/-- The final capital of the company after two years in millions of yuan -/
def final_capital : ℝ := 14.4

/-- The number of years over which the growth occurred -/
def years : ℕ := 2

/-- Theorem stating that the equation 1000(1+x)^2 = 1440 correctly represents 
    the average annual growth rate of the company's capital -/
theorem growth_rate_equation : 1000 * (1 + x)^years = 1440 := by sorry

end growth_rate_equation_l814_81441


namespace money_division_l814_81417

theorem money_division (alice bond charlie : ℕ) 
  (h1 : charlie = 495)
  (h2 : (alice - 10) * 18 * 24 = (bond - 20) * 11 * 24)
  (h3 : (alice - 10) * 24 * 18 = (charlie - 15) * 11 * 18)
  (h4 : (bond - 20) * 24 * 11 = (charlie - 15) * 18 * 11) :
  alice + bond + charlie = 1105 := by
  sorry

end money_division_l814_81417


namespace similar_triangles_side_length_l814_81452

/-- Represents a triangle with an area and a side length -/
structure Triangle where
  area : ℝ
  side : ℝ

/-- Proves that given two similar triangles with specific properties, 
    the corresponding side of the larger triangle is 12 feet -/
theorem similar_triangles_side_length 
  (t1 t2 : Triangle) 
  (h_area_diff : t1.area - t2.area = 32)
  (h_area_ratio : t1.area / t2.area = 9)
  (h_smaller_area_int : ∃ n : ℕ, t2.area = n)
  (h_smaller_side : t2.side = 4)
  (h_similar : ∃ k : ℝ, t1.side = k * t2.side ∧ t1.area = k^2 * t2.area) :
  t1.side = 12 := by
  sorry

#check similar_triangles_side_length

end similar_triangles_side_length_l814_81452


namespace pizza_cost_three_pizzas_cost_l814_81431

/-- The cost of all pizzas given the number of pizzas, slices per pizza, and the cost of a subset of slices. -/
theorem pizza_cost (num_pizzas : ℕ) (slices_per_pizza : ℕ) (subset_slices : ℕ) (subset_cost : ℚ) : ℚ :=
  let total_slices := num_pizzas * slices_per_pizza
  let cost_per_slice := subset_cost / subset_slices
  total_slices * cost_per_slice

/-- Proof that 3 pizzas with 12 slices each cost $72, given that 5 slices cost $10. -/
theorem three_pizzas_cost : pizza_cost 3 12 5 10 = 72 := by
  sorry

end pizza_cost_three_pizzas_cost_l814_81431


namespace negation_of_all_geq_two_l814_81408

theorem negation_of_all_geq_two :
  (¬ (∀ x : ℝ, x ≥ 2)) ↔ (∃ x₀ : ℝ, x₀ < 2) :=
by sorry

end negation_of_all_geq_two_l814_81408


namespace polynomial_value_at_3_l814_81427

def is_valid_coeff (b : ℤ) : Prop := 0 ≤ b ∧ b < 5

def P (b : Fin 6 → ℤ) (x : ℝ) : ℝ :=
  (b 0) + (b 1) * x + (b 2) * x^2 + (b 3) * x^3 + (b 4) * x^4 + (b 5) * x^5

theorem polynomial_value_at_3 (b : Fin 6 → ℤ) :
  (∀ i : Fin 6, is_valid_coeff (b i)) →
  P b (Real.sqrt 5) = 40 + 31 * Real.sqrt 5 →
  P b 3 = 381 := by
  sorry

end polynomial_value_at_3_l814_81427


namespace tangent_and_unique_zero_l814_81425

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (a * Real.log x) / x

def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := f a (f a x) - t

theorem tangent_and_unique_zero (a : ℝ) (h1 : a > 0) :
  (∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ = f a x₀ ∧ x₀ - 2 * y₀ = 0 ∧ 
    (∀ x : ℝ, x > 0 → x - 2 * f a x ≥ 0) ∧
    (∀ x : ℝ, x > 0 → x - 2 * f a x = 0 → x = x₀)) →
  (∃! t : ℝ, ∃! x : ℝ, x > 0 ∧ g a t x = 0) →
  (∀ t : ℝ, (∃! x : ℝ, x > 0 ∧ g a t x = 0) → t = 0) :=
sorry

end tangent_and_unique_zero_l814_81425


namespace equality_of_fractions_l814_81411

theorem equality_of_fractions (x y z k : ℝ) :
  (5 / (x + y) = k / (x - z)) ∧ (k / (x - z) = 9 / (z + y)) → k = 14 :=
by
  sorry

end equality_of_fractions_l814_81411


namespace hans_deposit_is_101_l814_81497

/-- Calculates the deposit for a restaurant reservation --/
def calculate_deposit (num_adults num_children num_seniors : ℕ) 
  (flat_deposit adult_charge child_charge senior_charge service_charge : ℕ) 
  (split_bill : Bool) : ℕ :=
  flat_deposit + 
  num_adults * adult_charge + 
  num_children * child_charge + 
  num_seniors * senior_charge +
  (if split_bill then service_charge else 0)

/-- Theorem: The deposit for Hans' reservation is $101 --/
theorem hans_deposit_is_101 : 
  calculate_deposit 10 2 3 25 5 2 4 10 true = 101 := by
  sorry

end hans_deposit_is_101_l814_81497


namespace rectangular_grid_toothpicks_l814_81446

/-- Calculates the number of toothpicks in a rectangular grid. -/
def toothpick_count (height : ℕ) (width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height

/-- Theorem: A rectangular grid of toothpicks that is 20 high and 10 wide uses 430 toothpicks. -/
theorem rectangular_grid_toothpicks : toothpick_count 20 10 = 430 := by
  sorry

end rectangular_grid_toothpicks_l814_81446


namespace base6_addition_problem_l814_81413

-- Define a function to convert a base 6 number to base 10
def base6ToBase10 (d₂ d₁ d₀ : Nat) : Nat :=
  d₂ * 6^2 + d₁ * 6^1 + d₀ * 6^0

-- Define a function to convert a base 10 number to base 6
def base10ToBase6 (n : Nat) : Nat × Nat × Nat :=
  let d₂ := n / 36
  let r₂ := n % 36
  let d₁ := r₂ / 6
  let d₀ := r₂ % 6
  (d₂, d₁, d₀)

theorem base6_addition_problem :
  ∀ S H E : Nat,
    S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 →
    S < 6 ∧ H < 6 ∧ E < 6 →
    S ≠ H ∧ S ≠ E ∧ H ≠ E →
    base6ToBase10 S H E + base6ToBase10 0 H E = base6ToBase10 E S H →
    S = 5 ∧ H = 4 ∧ E = 5 ∧ base10ToBase6 (S + H + E) = (2, 2, 0) := by
  sorry

end base6_addition_problem_l814_81413


namespace even_odd_sum_difference_l814_81410

/-- Sum of first n positive even integers -/
def sumEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

/-- Positive difference between two natural numbers -/
def positiveDifference (a b : ℕ) : ℕ := max a b - min a b

theorem even_odd_sum_difference :
  positiveDifference (sumEvenIntegers 25) (3 * sumOddIntegers 20) = 550 := by
  sorry

end even_odd_sum_difference_l814_81410


namespace correct_probability_order_l814_81490

/-- Enum representing the five types of phenomena -/
inductive Phenomenon
  | CertainToHappen
  | VeryLikelyToHappen
  | PossibleToHappen
  | ImpossibleToHappen
  | UnlikelyToHappen

/-- Function to compare the probability of two phenomena -/
def probabilityLessThan (a b : Phenomenon) : Prop :=
  match a, b with
  | Phenomenon.ImpossibleToHappen, _ => a ≠ b
  | Phenomenon.UnlikelyToHappen, Phenomenon.ImpossibleToHappen => False
  | Phenomenon.UnlikelyToHappen, _ => a ≠ b
  | Phenomenon.PossibleToHappen, Phenomenon.ImpossibleToHappen => False
  | Phenomenon.PossibleToHappen, Phenomenon.UnlikelyToHappen => False
  | Phenomenon.PossibleToHappen, _ => a ≠ b
  | Phenomenon.VeryLikelyToHappen, Phenomenon.CertainToHappen => True
  | Phenomenon.VeryLikelyToHappen, _ => False
  | Phenomenon.CertainToHappen, _ => False

/-- Theorem stating the correct order of phenomena by probability -/
theorem correct_probability_order :
  probabilityLessThan Phenomenon.ImpossibleToHappen Phenomenon.UnlikelyToHappen ∧
  probabilityLessThan Phenomenon.UnlikelyToHappen Phenomenon.PossibleToHappen ∧
  probabilityLessThan Phenomenon.PossibleToHappen Phenomenon.VeryLikelyToHappen ∧
  probabilityLessThan Phenomenon.VeryLikelyToHappen Phenomenon.CertainToHappen :=
sorry

end correct_probability_order_l814_81490


namespace house_width_calculation_l814_81420

/-- Given a house with length 20.5 feet, a porch measuring 6 feet by 4.5 feet,
    and a total shingle area of 232 square feet, the width of the house is 10 feet. -/
theorem house_width_calculation (house_length porch_length porch_width total_shingle_area : ℝ)
    (h1 : house_length = 20.5)
    (h2 : porch_length = 6)
    (h3 : porch_width = 4.5)
    (h4 : total_shingle_area = 232) :
    (total_shingle_area - porch_length * porch_width) / house_length = 10 := by
  sorry

#check house_width_calculation

end house_width_calculation_l814_81420


namespace range_of_a_l814_81467

theorem range_of_a (a b c : ℝ) 
  (eq1 : a^2 - b*c - 8*a + 7 = 0)
  (eq2 : b^2 + c^2 + b*c - 6*a + 6 = 0) :
  1 ≤ a ∧ a ≤ 9 :=
sorry

end range_of_a_l814_81467


namespace is_stratified_sampling_l814_81468

/-- Represents a sampling method -/
structure SamplingMethod where
  name : String
  dividePopulation : Bool
  sampleFromParts : Bool
  proportionalSampling : Bool
  combineSamples : Bool

/-- Definition of stratified sampling -/
def stratifiedSampling : SamplingMethod :=
  { name := "Stratified Sampling",
    dividePopulation := true,
    sampleFromParts := true,
    proportionalSampling := true,
    combineSamples := true }

/-- Theorem stating that a sampling method with specific characteristics is stratified sampling -/
theorem is_stratified_sampling
  (method : SamplingMethod)
  (h1 : method.dividePopulation = true)
  (h2 : method.sampleFromParts = true)
  (h3 : method.proportionalSampling = true)
  (h4 : method.combineSamples = true) :
  method = stratifiedSampling := by
  sorry

#check is_stratified_sampling

end is_stratified_sampling_l814_81468


namespace fraction_product_l814_81474

theorem fraction_product (a b c d e f : ℝ) 
  (h1 : a / b = 5 / 2)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 1)
  (h4 : d / e = 3 / 2)
  (h5 : e / f = 4 / 3) :
  a * b * c / (d * e * f) = 2.5 := by
  sorry

end fraction_product_l814_81474


namespace square_area_ratio_l814_81494

/-- Given three square regions A, B, and C, where the perimeter of A is 20 units and
    the perimeter of B is 40 units, and assuming the side length of C increases
    proportionally from B as B did from A, the ratio of the area of A to the area of C is 1/16. -/
theorem square_area_ratio (A B C : ℝ) : 
  (A * 4 = 20) →  -- Perimeter of A is 20 units
  (B * 4 = 40) →  -- Perimeter of B is 40 units
  (C = 2 * B) →   -- Side length of C increases proportionally
  (A^2 / C^2 = 1/16) :=
by sorry

end square_area_ratio_l814_81494


namespace solution_to_equation_l814_81471

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (9*x)^18 - (18*x)^9 = 0 ∧ x = 2 := by
  sorry

end solution_to_equation_l814_81471


namespace inequality_proof_l814_81412

theorem inequality_proof (x : ℝ) (h : x ≠ 2) : x^2 / (x - 2)^2 ≥ 0 := by
  sorry

end inequality_proof_l814_81412


namespace intersection_at_one_point_l814_81478

theorem intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 2 * x + 3 = 3 * x + 4) ↔ b = -1/4 := by
sorry

end intersection_at_one_point_l814_81478


namespace satisfactory_grades_fraction_l814_81461

/-- Represents the grades in a class -/
structure ClassGrades where
  total_students : Nat
  grade_a : Nat
  grade_b : Nat
  grade_c : Nat
  grade_d : Nat
  grade_f : Nat

/-- Calculates the fraction of satisfactory grades -/
def satisfactory_fraction (grades : ClassGrades) : Rat :=
  (grades.grade_a + grades.grade_b + grades.grade_c : Rat) / grades.total_students

/-- The main theorem about the fraction of satisfactory grades -/
theorem satisfactory_grades_fraction :
  let grades : ClassGrades := {
    total_students := 30,
    grade_a := 8,
    grade_b := 7,
    grade_c := 6,
    grade_d := 5,
    grade_f := 4
  }
  satisfactory_fraction grades = 7 / 10 := by sorry

end satisfactory_grades_fraction_l814_81461


namespace polynomial_divisibility_l814_81443

theorem polynomial_divisibility (a b c d : ℤ) : 
  (∀ x : ℤ, ∃ k : ℤ, a * x^3 + b * x^2 + c * x + d = 5 * k) →
  (∃ (ka kb kc kd : ℤ), a = 5 * ka ∧ b = 5 * kb ∧ c = 5 * kc ∧ d = 5 * kd) :=
by sorry

end polynomial_divisibility_l814_81443


namespace pad_usage_duration_l814_81440

/-- Represents the number of sheets in a pad of paper -/
def sheets_per_pad : ℕ := 60

/-- Represents the number of working days per week -/
def working_days_per_week : ℕ := 3

/-- Represents the number of sheets used per working day -/
def sheets_per_day : ℕ := 12

/-- Calculates the number of weeks it takes to use a full pad of paper -/
def weeks_per_pad : ℚ :=
  sheets_per_pad / (working_days_per_week * sheets_per_day)

/-- Theorem stating that the rounded-up number of weeks to use a pad is 2 -/
theorem pad_usage_duration :
  ⌈weeks_per_pad⌉ = 2 := by sorry

end pad_usage_duration_l814_81440


namespace polar_to_cartesian_line_l814_81476

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (2 * Real.sin θ - Real.cos θ)

-- Define the Cartesian equation of a line
def line_equation (x y : ℝ) : Prop :=
  2 * y - x = 1

-- Theorem statement
theorem polar_to_cartesian_line :
  ∀ (r θ x y : ℝ), 
    polar_equation r θ → 
    x = r * Real.cos θ →
    y = r * Real.sin θ →
    line_equation x y :=
by
  sorry


end polar_to_cartesian_line_l814_81476


namespace least_positive_integer_with_remainder_one_l814_81489

theorem least_positive_integer_with_remainder_one (n : ℕ) : n = 9241 ↔ 
  n > 1 ∧ 
  (∀ d ∈ ({5, 7, 8, 10, 11, 12} : Set ℕ), n % d = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ d ∈ ({5, 7, 8, 10, 11, 12} : Set ℕ), m % d = 1) → m ≥ n) :=
by sorry

end least_positive_integer_with_remainder_one_l814_81489


namespace large_box_chocolate_count_l814_81450

/-- The number of chocolate bars in a large box -/
def total_chocolate_bars (num_small_boxes : ℕ) (bars_per_small_box : ℕ) : ℕ :=
  num_small_boxes * bars_per_small_box

/-- Theorem: The large box contains 475 chocolate bars -/
theorem large_box_chocolate_count :
  total_chocolate_bars 19 25 = 475 := by
  sorry

end large_box_chocolate_count_l814_81450


namespace book_store_problem_l814_81472

/-- Represents the purchase and sale of books in a stationery store -/
structure BookStore where
  costA : ℝ  -- Cost price of type A book
  costB : ℝ  -- Cost price of type B book
  sellA : ℝ  -- Selling price of type A book
  sellB : ℝ  -- Selling price of type B book
  totalCost : ℝ  -- Total purchase cost
  profit : ℝ  -- Total profit from first sale

/-- Represents the second purchase and sale scenario -/
structure SecondPurchase where
  minSellA : ℝ  -- Minimum selling price for type A in second purchase
  minProfit : ℝ  -- Minimum required profit for second sale

/-- Theorem stating the solution to the book store problem -/
theorem book_store_problem (store : BookStore) (second : SecondPurchase) 
  (h1 : store.costA = 12)
  (h2 : store.costB = 10)
  (h3 : store.sellA = 15)
  (h4 : store.sellB = 12)
  (h5 : store.totalCost = 1200)
  (h6 : store.profit = 270)
  (h7 : second.minProfit = 340) :
  ∃ (x y : ℕ), 
    (x : ℝ) * store.costA + (y : ℝ) * store.costB = store.totalCost ∧ 
    (x : ℝ) * (store.sellA - store.costA) + (y : ℝ) * (store.sellB - store.costB) = store.profit ∧
    x = 50 ∧ 
    y = 60 ∧
    second.minSellA = 14 ∧
    (50 : ℝ) * (second.minSellA - store.costA) + 2 * (60 : ℝ) * (store.sellB - store.costB) ≥ second.minProfit := by
  sorry


end book_store_problem_l814_81472


namespace dress_discount_percentage_l814_81460

theorem dress_discount_percentage (d : ℝ) (x : ℝ) 
  (h1 : d > 0) 
  (h2 : 0.6 * d = d * (1 - x / 100) * 0.8) : 
  x = 25 := by sorry

end dress_discount_percentage_l814_81460


namespace polynomial_inequality_l814_81498

theorem polynomial_inequality (a b c d : ℝ) :
  (∀ x : ℝ, |x| ≤ 1 → |a * x^3 + b * x^2 + c * x + d| ≤ 1) →
  (∀ x : ℝ, |x| ≤ 1 → |3 * a * x^2 + 2 * b * x + c| ≤ 9) :=
by sorry

end polynomial_inequality_l814_81498


namespace orange_boxes_l814_81485

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 35) (h2 : oranges_per_box = 5) :
  total_oranges / oranges_per_box = 7 :=
by
  sorry

end orange_boxes_l814_81485


namespace equation_solution_l814_81457

theorem equation_solution :
  ∀ x : ℝ, (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - 4.5 = 0) ↔ x = 3/2 := by
  sorry

end equation_solution_l814_81457


namespace max_distance_OM_l814_81402

/-- The ellipse C -/
def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The circle O -/
def circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The tangent line l -/
def tangent_line_l (x y m t : ℝ) : Prop :=
  x = m * y + t

theorem max_distance_OM (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : (a^2 - b^2).sqrt / a = Real.sqrt 3 / 2) (h4 : 2 * b = 2)
  (A B M : ℝ × ℝ) (m t : ℝ)
  (hA : ellipse_C A.1 A.2 a b)
  (hB : ellipse_C B.1 B.2 a b)
  (hM : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
  (hl : tangent_line_l A.1 A.2 m t ∧ tangent_line_l B.1 B.2 m t)
  (htangent : t^2 = m^2 + 1) :
  (∀ P : ℝ × ℝ, ellipse_C P.1 P.2 a b → (P.1^2 + P.2^2).sqrt ≤ 5/4) ∧
  (∃ Q : ℝ × ℝ, ellipse_C Q.1 Q.2 a b ∧ (Q.1^2 + Q.2^2).sqrt = 5/4) :=
sorry

end max_distance_OM_l814_81402


namespace factorization_1_factorization_2_l814_81451

-- Factorization of 4x^2 - 16
theorem factorization_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x - 2) * (x + 2) := by sorry

-- Factorization of a^2b - 4ab + 4b
theorem factorization_2 (a b : ℝ) : a^2 * b - 4 * a * b + 4 * b = b * (a - 2)^2 := by sorry

end factorization_1_factorization_2_l814_81451


namespace second_catch_size_l814_81442

theorem second_catch_size (tagged_initial : ℕ) (tagged_second : ℕ) (total_fish : ℕ) :
  tagged_initial = 50 →
  tagged_second = 2 →
  total_fish = 1250 →
  (tagged_second : ℚ) / (tagged_initial : ℚ) = (tagged_second : ℚ) / x →
  x = 50 :=
by sorry

end second_catch_size_l814_81442


namespace max_M_value_l814_81479

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (hzy : z ≥ y) :
  ∃ (M : ℝ), M > 0 ∧ M ≤ z/y ∧ ∀ (N : ℝ), (N > 0 ∧ N ≤ z/y) → N ≤ 6 + 4*Real.sqrt 2 :=
sorry

end max_M_value_l814_81479


namespace jamal_cart_books_l814_81469

def books_in_cart (history : ℕ) (fiction : ℕ) (children : ℕ) (children_misplaced : ℕ) 
  (science : ℕ) (science_misplaced : ℕ) (biography : ℕ) (remaining : ℕ) : ℕ :=
  history + fiction + (children - children_misplaced) + (science - science_misplaced) + biography + remaining

theorem jamal_cart_books :
  books_in_cart 15 22 10 5 8 3 12 20 = 79 := by
  sorry

end jamal_cart_books_l814_81469


namespace log_equation_solution_l814_81436

theorem log_equation_solution (x : ℝ) (h : Real.log 125 / Real.log (3 * x) = x) :
  (∃ (a b : ℤ), x = a / b ∧ a ≠ 0 ∧ b > 0 ∧ (∀ n : ℕ, n > 1 → (a : ℝ) / b ≠ n^2 ∧ (a : ℝ) / b ≠ n^3) ∧ ¬∃ (n : ℤ), (a : ℝ) / b = n) := by
  sorry

end log_equation_solution_l814_81436
