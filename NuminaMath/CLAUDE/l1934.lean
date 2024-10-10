import Mathlib

namespace zoo_theorem_l1934_193477

def zoo_problem (tiger_enclosures : ℕ) (zebra_enclosures_per_tiger : ℕ) 
  (giraffe_enclosures_multiplier : ℕ) (tigers_per_enclosure : ℕ) 
  (zebras_per_enclosure : ℕ) (giraffes_per_enclosure : ℕ) : Prop :=
  let total_zebra_enclosures := tiger_enclosures * zebra_enclosures_per_tiger
  let total_giraffe_enclosures := total_zebra_enclosures * giraffe_enclosures_multiplier
  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let total_zebras := total_zebra_enclosures * zebras_per_enclosure
  let total_giraffes := total_giraffe_enclosures * giraffes_per_enclosure
  let total_animals := total_tigers + total_zebras + total_giraffes
  total_animals = 144

theorem zoo_theorem : 
  zoo_problem 4 2 3 4 10 2 := by
  sorry

end zoo_theorem_l1934_193477


namespace correct_atomic_symbol_proof_l1934_193403

/-- Represents an element X in an ionic compound XCl_n -/
structure ElementX where
  m : ℕ  -- number of neutrons
  y : ℕ  -- number of electrons outside the nucleus
  n : ℕ  -- number of chlorine atoms in the compound

/-- Represents the atomic symbol of an isotope -/
structure AtomicSymbol where
  subscript : ℕ
  superscript : ℕ

/-- Returns the correct atomic symbol for an element X -/
def correct_atomic_symbol (x : ElementX) : AtomicSymbol :=
  { subscript := x.y + x.n
  , superscript := x.m + x.y + x.n }

/-- Theorem stating that the correct atomic symbol for element X is _{y+n}^{m+y+n}X -/
theorem correct_atomic_symbol_proof (x : ElementX) :
  correct_atomic_symbol x = { subscript := x.y + x.n, superscript := x.m + x.y + x.n } :=
by sorry

end correct_atomic_symbol_proof_l1934_193403


namespace intersection_points_parallel_lines_l1934_193433

/-- Given two parallel lines with m and n points respectively, 
    this theorem states the number of intersection points formed by 
    segments connecting these points. -/
theorem intersection_points_parallel_lines 
  (m n : ℕ) : ℕ := by
  sorry

end intersection_points_parallel_lines_l1934_193433


namespace set_operations_l1934_193490

-- Define the sets A and B
def A : Set ℝ := {x | x - 1 ≥ 0}
def B : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) ∧
  ((Aᶜ ∩ Bᶜ) = {x : ℝ | x < -1}) := by
  sorry

end set_operations_l1934_193490


namespace intersection_A_B_l1934_193405

-- Define set A
def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 ≤ 3}

-- Define set B
def B : Set ℝ := {2, 3, 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {3, 4} := by
  sorry

end intersection_A_B_l1934_193405


namespace triangle_tangent_inequality_l1934_193450

theorem triangle_tangent_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.tan A ^ 2 + Real.tan B ^ 2 + Real.tan C ^ 2 ≥ 
  Real.tan A * Real.tan B + Real.tan B * Real.tan C + Real.tan C * Real.tan A :=
by sorry

end triangle_tangent_inequality_l1934_193450


namespace squared_ratios_sum_ge_sum_l1934_193426

theorem squared_ratios_sum_ge_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c := by sorry

end squared_ratios_sum_ge_sum_l1934_193426


namespace ellipse_proof_l1934_193474

-- Define the given ellipse
def given_ellipse (x y : ℝ) : Prop := 3 * x^2 + 8 * y^2 = 24

-- Define the point that the new ellipse should pass through
def point : ℝ × ℝ := (3, -2)

-- Define the new ellipse
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

-- Theorem statement
theorem ellipse_proof :
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), given_ellipse x y ↔ x^2 / (c^2 + 5) + y^2 / c^2 = 1)) →
  (new_ellipse point.1 point.2) ∧
  (∃ (c : ℝ), c > 0 ∧
    (∀ (x y : ℝ), new_ellipse x y ↔ x^2 / (c^2 + 5) + y^2 / c^2 = 1)) :=
sorry

end ellipse_proof_l1934_193474


namespace sin_alpha_plus_beta_l1934_193475

theorem sin_alpha_plus_beta (α β : Real) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = 0) : 
  Real.sin (α + β) = -1/2 := by
sorry

end sin_alpha_plus_beta_l1934_193475


namespace unique_solution_exists_l1934_193467

theorem unique_solution_exists : 
  ∃! (a b c : ℕ+), 
    (a.val * b.val + 3 * b.val * c.val = 63) ∧ 
    (a.val * c.val + 3 * b.val * c.val = 39) := by
  sorry

end unique_solution_exists_l1934_193467


namespace inequality_proof_l1934_193416

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 :=
by sorry

end inequality_proof_l1934_193416


namespace additional_money_needed_l1934_193421

/-- The cost of the dictionary -/
def dictionary_cost : ℚ := 5.50

/-- The cost of the dinosaur book -/
def dinosaur_book_cost : ℚ := 11.25

/-- The cost of the children's cookbook -/
def cookbook_cost : ℚ := 5.75

/-- The cost of the science experiment kit -/
def science_kit_cost : ℚ := 8.40

/-- The cost of the set of colored pencils -/
def pencils_cost : ℚ := 3.60

/-- The amount Emir has saved -/
def saved_amount : ℚ := 24.50

/-- The theorem stating how much more money Emir needs -/
theorem additional_money_needed :
  dictionary_cost + dinosaur_book_cost + cookbook_cost + science_kit_cost + pencils_cost - saved_amount = 10 := by
  sorry

end additional_money_needed_l1934_193421


namespace odd_numbers_pascal_triangle_l1934_193465

/-- 
Given a non-negative integer n, count_ones n returns the number of 1's 
in the binary representation of n.
-/
def count_ones (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 2) + count_ones (n / 2)

/-- 
Given a non-negative integer n, odd_numbers_in_pascal_row n returns the 
number of odd numbers in the n-th row of Pascal's triangle.
-/
def odd_numbers_in_pascal_row (n : ℕ) : ℕ :=
  2^(count_ones n)

/-- 
Theorem: The number of odd numbers in the n-th row of Pascal's triangle 
is equal to 2^k, where k is the number of 1's in the binary representation of n.
-/
theorem odd_numbers_pascal_triangle (n : ℕ) : 
  odd_numbers_in_pascal_row n = 2^(count_ones n) := by
  sorry


end odd_numbers_pascal_triangle_l1934_193465


namespace egyptian_fraction_proof_l1934_193463

theorem egyptian_fraction_proof :
  ∃! (b₂ b₃ b₅ b₆ b₇ b₈ : ℕ),
    (3 : ℚ) / 8 = b₂ / 2 + b₃ / 6 + b₅ / 120 + b₆ / 720 + b₇ / 5040 + b₈ / 40320 ∧
    b₂ < 2 ∧ b₃ < 3 ∧ b₅ < 5 ∧ b₆ < 6 ∧ b₇ < 7 ∧ b₈ < 8 ∧
    b₂ + b₃ + b₅ + b₆ + b₇ + b₈ = 12 :=
by sorry

end egyptian_fraction_proof_l1934_193463


namespace sum_of_integers_l1934_193468

theorem sum_of_integers (a b c d e : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + e = 9)
  (eq4 : d - e + a = 4)
  (eq5 : e - a + b = 3) : 
  a + b + c + d + e = 31 := by
  sorry

end sum_of_integers_l1934_193468


namespace lcd_of_fractions_l1934_193406

theorem lcd_of_fractions (a b c d : ℕ) (ha : a = 2) (hb : b = 4) (hc : c = 5) (hd : d = 6) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 60 := by
  sorry

end lcd_of_fractions_l1934_193406


namespace initial_number_of_boys_l1934_193401

theorem initial_number_of_boys (B : ℝ) : 
  (1.2 * B) + B + (2.4 * B) = 51 → B = 15 := by
  sorry

end initial_number_of_boys_l1934_193401


namespace factor_polynomial_l1934_193498

theorem factor_polynomial (x : ℝ) : 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) := by
  sorry

end factor_polynomial_l1934_193498


namespace lap_length_l1934_193437

/-- Proves that the length of one lap is 1/4 mile, given the total distance and number of laps. -/
theorem lap_length (total_distance : ℚ) (num_laps : ℕ) :
  total_distance = 13/4 ∧ num_laps = 13 →
  total_distance / num_laps = 1/4 := by
sorry

end lap_length_l1934_193437


namespace order_xyz_l1934_193430

theorem order_xyz (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d)
  (x : ℝ) (hx : x = (a+b)*(c+d))
  (y : ℝ) (hy : y = (a+c)*(b+d))
  (z : ℝ) (hz : z = (a+d)*(b+c)) :
  x < y ∧ y < z :=
by sorry

end order_xyz_l1934_193430


namespace library_book_return_percentage_l1934_193407

theorem library_book_return_percentage 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (loaned_books : ℕ) 
  (h1 : initial_books = 300) 
  (h2 : final_books = 244) 
  (h3 : loaned_books = 160) : 
  (((loaned_books - (initial_books - final_books)) / loaned_books) * 100 : ℚ) = 65 := by
  sorry

end library_book_return_percentage_l1934_193407


namespace am_gm_inequality_application_l1934_193411

theorem am_gm_inequality_application (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 ∧
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2 ↔ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) :=
by sorry

end am_gm_inequality_application_l1934_193411


namespace sandy_shopping_money_l1934_193410

theorem sandy_shopping_money (remaining_amount : ℝ) (spent_percentage : ℝ) (initial_amount : ℝ) :
  remaining_amount = 224 ∧
  spent_percentage = 0.3 ∧
  remaining_amount = initial_amount * (1 - spent_percentage) →
  initial_amount = 320 :=
by sorry

end sandy_shopping_money_l1934_193410


namespace trajectory_is_ellipse_l1934_193419

/-- The set of points (x, y) satisfying the given equation forms an ellipse -/
theorem trajectory_is_ellipse :
  ∀ x y : ℝ, 
  Real.sqrt (x^2 + (y + 3)^2) + Real.sqrt (x^2 + (y - 3)^2) = 10 →
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
  x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end trajectory_is_ellipse_l1934_193419


namespace tournament_probability_l1934_193443

/-- The number of teams in the tournament -/
def num_teams : ℕ := 30

/-- The total number of games played in the tournament -/
def total_games : ℕ := num_teams.choose 2

/-- The probability of a team winning any given game -/
def win_probability : ℚ := 1/2

/-- The probability that no two teams win the same number of games -/
noncomputable def unique_wins_probability : ℚ := (num_teams.factorial : ℚ) / 2^total_games

theorem tournament_probability :
  ∃ (m : ℕ), Odd m ∧ unique_wins_probability = 1 / (2^409 * m) :=
sorry

end tournament_probability_l1934_193443


namespace angle_XYZ_measure_l1934_193427

-- Define the regular octagon
def RegularOctagon : Type := Unit

-- Define the square inside the octagon
def Square : Type := Unit

-- Define the vertices
def X : RegularOctagon := Unit.unit
def Y : Square := Unit.unit
def Z : Square := Unit.unit

-- Define the angle measure function
def angle_measure : RegularOctagon → Square → Square → ℝ := sorry

-- State the theorem
theorem angle_XYZ_measure (o : RegularOctagon) (s : Square) :
  angle_measure X Y Z = 90 := by sorry

end angle_XYZ_measure_l1934_193427


namespace square_area_from_string_length_l1934_193436

theorem square_area_from_string_length (string_length : ℝ) (h : string_length = 32) :
  let side_length := string_length / 4
  side_length * side_length = 64 := by sorry

end square_area_from_string_length_l1934_193436


namespace science_club_problem_l1934_193478

theorem science_club_problem (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : math = 85)
  (h3 : physics = 60)
  (h4 : both = 20) :
  total - (math + physics - both) = 25 := by
sorry

end science_club_problem_l1934_193478


namespace parabola_vertex_l1934_193458

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := (x - 2)^2 - 3

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (2, -3)

/-- Theorem: The vertex of the parabola y = (x-2)^2 - 3 is (2, -3) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 :=
sorry

end parabola_vertex_l1934_193458


namespace complementary_events_l1934_193472

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure Draw where
  first : Color
  second : Color

/-- The set of all possible draws -/
def all_draws : Finset Draw :=
  sorry

/-- The event "Exactly no red ball" -/
def exactly_no_red (draw : Draw) : Prop :=
  draw.first = Color.White ∧ draw.second = Color.White

/-- The event "At most 1 white ball" -/
def at_most_one_white (draw : Draw) : Prop :=
  draw.first = Color.Red ∨ draw.second = Color.Red

/-- Theorem stating that "Exactly no red ball" and "At most 1 white ball" are complementary events -/
theorem complementary_events :
  ∀ (draw : Draw), draw ∈ all_draws → (exactly_no_red draw ↔ ¬at_most_one_white draw) :=
sorry

end complementary_events_l1934_193472


namespace slower_whale_length_is_45_l1934_193445

/-- The length of a slower whale given the speeds of two whales and the time for the faster to cross the slower -/
def slower_whale_length (faster_speed slower_speed crossing_time : ℝ) : ℝ :=
  (faster_speed - slower_speed) * crossing_time

/-- Theorem stating that the length of the slower whale is 45 meters given the problem conditions -/
theorem slower_whale_length_is_45 :
  slower_whale_length 18 15 15 = 45 := by
  sorry

end slower_whale_length_is_45_l1934_193445


namespace consecutive_integers_square_difference_l1934_193471

theorem consecutive_integers_square_difference (n : ℕ) : 
  n > 0 ∧ n + (n + 1) < 100 → (n + 1)^2 - n^2 = 79 := by
  sorry

end consecutive_integers_square_difference_l1934_193471


namespace planes_through_three_points_l1934_193439

/-- Three points in 3D space -/
structure ThreePoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ

/-- Possible number of planes through three points -/
inductive NumPlanes
  | one
  | infinite

/-- The number of planes that can be constructed through three points in 3D space 
    is either one or infinite -/
theorem planes_through_three_points (points : ThreePoints) : 
  ∃ (n : NumPlanes), n = NumPlanes.one ∨ n = NumPlanes.infinite :=
sorry

end planes_through_three_points_l1934_193439


namespace cricket_team_ratio_proof_l1934_193464

def cricket_team_ratio (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) : Prop :=
  let non_throwers := total_players - throwers
  let right_handed_non_throwers := right_handed - throwers
  let left_handed_non_throwers := non_throwers - right_handed_non_throwers
  2 * left_handed_non_throwers = right_handed_non_throwers

theorem cricket_team_ratio_proof :
  cricket_team_ratio 64 37 55 := by
  sorry

end cricket_team_ratio_proof_l1934_193464


namespace no_rain_probability_l1934_193404

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^4 = 1/81 := by sorry

end no_rain_probability_l1934_193404


namespace min_value_theorem_l1934_193493

theorem min_value_theorem (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0)
  (h_x : ∀ x, x^2 - 4*a*x + 3*a^2 < 0 ↔ x ∈ (Set.Ioo x₁ x₂)) :
  ∃ (min_val : ℝ), 
    (∀ y, x₁ + x₂ + a / (x₁ * x₂) ≥ y) ∧ 
    (x₁ + x₂ + a / (x₁ * x₂) = y ↔ y = 4 * Real.sqrt 3 / 3) :=
sorry

end min_value_theorem_l1934_193493


namespace ellipse_equal_angles_point_l1934_193460

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines the equation of the ellipse -/
def onEllipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines a chord passing through a given point -/
def isChord (e : Ellipse) (a b f : Point) : Prop :=
  onEllipse e a ∧ onEllipse e b ∧ ∃ t : ℝ, f = Point.mk (t * a.x + (1 - t) * b.x) (t * a.y + (1 - t) * b.y)

/-- Defines the property of equal angles -/
def equalAngles (p f a b : Point) : Prop :=
  (a.y - p.y) * (b.x - p.x) = (b.y - p.y) * (a.x - p.x)

/-- Main theorem statement -/
theorem ellipse_equal_angles_point :
  ∀ (e : Ellipse),
    e.a = 2 ∧ e.b = 1 →
    ∀ (f : Point),
      f.x = Real.sqrt 3 ∧ f.y = 0 →
      ∃! (p : Point),
        p.x > 0 ∧ p.y = 0 ∧
        (∀ (a b : Point), isChord e a b f → equalAngles p f a b) ∧
        p.x = 2 * Real.sqrt 3 :=
sorry

end ellipse_equal_angles_point_l1934_193460


namespace email_count_correct_l1934_193440

/-- Calculates the number of emails in Jackson's inbox after deletion and reception process -/
def final_email_count (deleted1 deleted2 received1 received2 received_after : ℕ) : ℕ :=
  received1 + received2 + received_after

/-- Theorem stating that the final email count is correct given the problem conditions -/
theorem email_count_correct :
  let deleted1 := 50
  let deleted2 := 20
  let received1 := 15
  let received2 := 5
  let received_after := 10
  final_email_count deleted1 deleted2 received1 received2 received_after = 30 := by sorry

end email_count_correct_l1934_193440


namespace inequality_proof_l1934_193409

theorem inequality_proof (k n : ℕ) (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 1) :
  (1 - (1 - x)^n)^k ≥ 1 - (1 - x^k)^n := by
  sorry

end inequality_proof_l1934_193409


namespace cat_video_length_is_correct_l1934_193451

/-- The length of the cat video in minutes -/
def cat_video_length : ℝ := 4

/-- The total time spent watching videos in minutes -/
def total_watching_time : ℝ := 36

/-- Theorem stating that the cat video length is correct given the conditions -/
theorem cat_video_length_is_correct :
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  cat_video_length + dog_video_length + gorilla_video_length = total_watching_time :=
by sorry

end cat_video_length_is_correct_l1934_193451


namespace no_integer_roots_l1934_193489

theorem no_integer_roots : ¬ ∃ (x : ℤ), x^3 - 4*x^2 - 14*x + 28 = 0 := by
  sorry

end no_integer_roots_l1934_193489


namespace average_marker_cost_correct_l1934_193470

def average_marker_cost (num_markers : ℕ) (marker_price : ℚ) (handling_fee : ℚ) (shipping_cost : ℚ) : ℕ :=
  let total_cost := marker_price + (num_markers : ℚ) * handling_fee + shipping_cost
  let total_cents := (total_cost * 100).floor
  let average_cents := (total_cents + (num_markers / 2)) / num_markers
  average_cents.toNat

theorem average_marker_cost_correct :
  average_marker_cost 300 45 0.1 8.5 = 28 := by
  sorry

end average_marker_cost_correct_l1934_193470


namespace base_conversion_problem_l1934_193497

theorem base_conversion_problem :
  ∃! (x y z b : ℕ),
    x * b^2 + y * b + z = 1987 ∧
    x + y + z = 25 ∧
    x < b ∧ y < b ∧ z < b ∧
    b > 10 ∧
    x = 5 ∧ y = 9 ∧ z = 11 ∧ b = 19 :=
by sorry

end base_conversion_problem_l1934_193497


namespace trigonometric_identities_l1934_193422

theorem trigonometric_identities :
  (Real.sin (20 * π / 180))^2 + (Real.cos (80 * π / 180))^2 + Real.sqrt 3 * Real.sin (20 * π / 180) * Real.cos (80 * π / 180) = 1/4 ∧
  (Real.sin (20 * π / 180))^2 + (Real.cos (50 * π / 180))^2 + Real.sin (20 * π / 180) * Real.cos (50 * π / 180) = 3/4 := by
  sorry

end trigonometric_identities_l1934_193422


namespace contact_box_price_l1934_193424

/-- The price of a box of contacts given the number of contacts and cost per contact -/
def box_price (num_contacts : ℕ) (cost_per_contact : ℚ) : ℚ :=
  num_contacts * cost_per_contact

/-- The cost per contact for a box given its total price and number of contacts -/
def cost_per_contact (total_price : ℚ) (num_contacts : ℕ) : ℚ :=
  total_price / num_contacts

theorem contact_box_price :
  let box1_contacts : ℕ := 50
  let box2_contacts : ℕ := 99
  let box2_price : ℚ := 33

  let box2_cost_per_contact := cost_per_contact box2_price box2_contacts
  let chosen_cost_per_contact : ℚ := 1 / 3

  box_price box1_contacts chosen_cost_per_contact = 50 * (1 / 3) := by
  sorry

end contact_box_price_l1934_193424


namespace range_of_expression_l1934_193476

theorem range_of_expression (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < 1/2 * α - β ∧ 1/2 * α - β < 11/2 := by
  sorry

end range_of_expression_l1934_193476


namespace puppies_count_l1934_193485

/-- Calculates the number of puppies given the total food needed, mom's food consumption, and puppies' food consumption. -/
def number_of_puppies (total_food : ℚ) (mom_meal : ℚ) (mom_meals_per_day : ℕ) (puppy_meal : ℚ) (puppy_meals_per_day : ℕ) (days : ℕ) : ℕ :=
  let mom_food := mom_meal * mom_meals_per_day * days
  let puppy_food := total_food - mom_food
  let puppy_food_per_puppy := puppy_meal * puppy_meals_per_day * days
  (puppy_food / puppy_food_per_puppy).num.toNat

/-- Theorem stating that the number of puppies is 5 given the specified conditions. -/
theorem puppies_count : number_of_puppies 57 (3/2) 3 (1/2) 2 6 = 5 := by
  sorry

end puppies_count_l1934_193485


namespace wheat_packets_fill_gunny_bag_l1934_193461

/-- The number of pounds in one ton -/
def pounds_per_ton : ℕ := 2300

/-- The number of packets of wheat -/
def num_packets : ℕ := 1840

/-- The weight of each packet in pounds -/
def packet_weight_pounds : ℕ := 16

/-- The additional weight of each packet in ounces -/
def packet_weight_ounces : ℕ := 4

/-- The capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℕ := 13

/-- The number of ounces in one pound -/
def ounces_per_pound : ℕ := 16

theorem wheat_packets_fill_gunny_bag :
  ounces_per_pound = 16 :=
sorry

end wheat_packets_fill_gunny_bag_l1934_193461


namespace A_intersect_C_U_B_eq_open_zero_closed_two_l1934_193429

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | x > 2}

-- Define the complement of B in the universal set ℝ
def C_U_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem A_intersect_C_U_B_eq_open_zero_closed_two : 
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end A_intersect_C_U_B_eq_open_zero_closed_two_l1934_193429


namespace hash_two_three_l1934_193428

-- Define the operation #
def hash (a b : ℕ) : ℕ := a * b - b + b^2

-- Theorem statement
theorem hash_two_three : hash 2 3 = 12 := by sorry

end hash_two_three_l1934_193428


namespace min_square_area_is_121_l1934_193408

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The set of rectangles given in the problem -/
def problem_rectangles : List Rectangle := [
  { width := 2, height := 3 },
  { width := 3, height := 4 },
  { width := 1, height := 4 }
]

/-- 
  Given a list of rectangles, computes the smallest possible side length of a square 
  that can contain all rectangles without overlapping
-/
def min_square_side (rectangles : List Rectangle) : ℕ :=
  sorry

/-- 
  Theorem: The smallest possible area of a square containing the given rectangles 
  without overlapping is 121
-/
theorem min_square_area_is_121 : 
  (min_square_side problem_rectangles) ^ 2 = 121 := by
  sorry

end min_square_area_is_121_l1934_193408


namespace billion_to_scientific_notation_l1934_193438

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to its scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (463.4 * 10^9) = ScientificNotation.mk 4.634 11 sorry :=
sorry

end billion_to_scientific_notation_l1934_193438


namespace big_boxes_count_l1934_193492

theorem big_boxes_count (dolls_per_big_box : ℕ) (dolls_per_small_box : ℕ) 
  (small_box_count : ℕ) (total_dolls : ℕ) (h1 : dolls_per_big_box = 7) 
  (h2 : dolls_per_small_box = 4) (h3 : small_box_count = 9) (h4 : total_dolls = 71) :
  ∃ (big_box_count : ℕ), big_box_count * dolls_per_big_box + 
    small_box_count * dolls_per_small_box = total_dolls ∧ big_box_count = 5 :=
by sorry

end big_boxes_count_l1934_193492


namespace polynomial_value_at_n_plus_one_l1934_193479

/-- Given an n-th degree polynomial P(x) such that P(k) = 1 / C_n^k for k = 0, 1, 2, ..., n,
    prove that P(n+1) = 0 if n is odd and P(n+1) = 1 if n is even. -/
theorem polynomial_value_at_n_plus_one (n : ℕ) (P : ℝ → ℝ) :
  (∀ k : ℕ, k ≤ n → P k = 1 / (n.choose k)) →
  P (n + 1) = if n % 2 = 1 then 0 else 1 := by
  sorry

end polynomial_value_at_n_plus_one_l1934_193479


namespace triangle_isosceles_condition_l1934_193448

theorem triangle_isosceles_condition (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- Positive side lengths
  0 < A ∧ 0 < B ∧ 0 < C →  -- Positive angles
  A + B + C = π →          -- Sum of angles in a triangle
  a * Real.cos B + b * Real.cos C + c * Real.cos A = (a + b + c) / 2 →
  (a = b ∨ b = c ∨ c = a) :=
by sorry

end triangle_isosceles_condition_l1934_193448


namespace solution_set_for_f_l1934_193414

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem solution_set_for_f
  (a : ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f a (2/a) > f a (3/a)) :
  ∀ x, f a (1 - 1/x) > 1 ↔ 1 < x ∧ x < 1/(1-a) :=
by sorry

end solution_set_for_f_l1934_193414


namespace remainder_problem_l1934_193418

theorem remainder_problem (x : ℕ+) : 
  203 % x.val = 13 ∧ 298 % x.val = 13 → x.val = 19 ∨ x.val = 95 :=
by sorry

end remainder_problem_l1934_193418


namespace max_value_of_trig_function_l1934_193459

theorem max_value_of_trig_function :
  let f : ℝ → ℝ := λ x ↦ 3 * Real.sin x + 4 * Real.cos x
  ∃ M : ℝ, (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 5 := by
  sorry

end max_value_of_trig_function_l1934_193459


namespace abs_diff_geq_sum_abs_iff_product_nonpositive_l1934_193446

theorem abs_diff_geq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  a * b ≤ 0 ↔ |a - b| ≥ |a| + |b| := by
  sorry

end abs_diff_geq_sum_abs_iff_product_nonpositive_l1934_193446


namespace cube_volume_doubled_edges_l1934_193442

/-- Given a cube, doubling each edge results in a volume 8 times larger than the original. -/
theorem cube_volume_doubled_edges (a : ℝ) (ha : a > 0) :
  (2 * a)^3 = 8 * a^3 := by sorry

end cube_volume_doubled_edges_l1934_193442


namespace simplify_expression_l1934_193431

theorem simplify_expression : 
  2 * Real.sqrt 12 + 3 * Real.sqrt (4/3) - Real.sqrt (16/3) - 2/3 * Real.sqrt 48 = 2 * Real.sqrt 3 := by
  sorry

end simplify_expression_l1934_193431


namespace circle_area_ratio_l1934_193435

theorem circle_area_ratio (A B : Real) (rA rB : ℝ) (hA : A = 2 * π * rA) (hB : B = 2 * π * rB)
  (h_arc : (60 / 360) * A = (40 / 360) * B) :
  π * rA^2 / (π * rB^2) = 4 / 9 := by
sorry

end circle_area_ratio_l1934_193435


namespace function_symmetry_l1934_193486

/-- Given a function f(x) = ax^4 - bx^2 + c - 1 where a, b, and c are real numbers,
    if f(2) = -1, then f(-2) = -1 -/
theorem function_symmetry (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^4 - b * x^2 + c - 1
  f 2 = -1 → f (-2) = -1 := by
  sorry

end function_symmetry_l1934_193486


namespace cookies_given_to_cousin_l1934_193487

theorem cookies_given_to_cousin (initial_boxes : ℕ) (brother_boxes : ℕ) (sister_boxes : ℕ) (self_boxes : ℕ) :
  initial_boxes = 45 →
  brother_boxes = 12 →
  sister_boxes = 9 →
  self_boxes = 17 →
  initial_boxes - brother_boxes - sister_boxes - self_boxes = 7 :=
by sorry

end cookies_given_to_cousin_l1934_193487


namespace pelicans_in_shark_bite_cove_l1934_193444

/-- The number of Pelicans remaining in Shark Bite Cove after some have moved to Pelican Bay -/
def remaining_pelicans (initial_pelicans : ℕ) : ℕ :=
  initial_pelicans - initial_pelicans / 3

/-- The theorem stating the number of remaining Pelicans in Shark Bite Cove -/
theorem pelicans_in_shark_bite_cove :
  ∃ (initial_pelicans : ℕ),
    (2 * initial_pelicans = 60) ∧
    (remaining_pelicans initial_pelicans = 20) := by
  sorry

end pelicans_in_shark_bite_cove_l1934_193444


namespace smallest_n_satisfying_condition_l1934_193452

def sum_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def satisfies_condition (n : ℕ) : Prop :=
  is_perfect_square (sum_squares n * (sum_squares (3 * n) - sum_squares n))

theorem smallest_n_satisfying_condition :
  (∀ m : ℕ, 10 ≤ m ∧ m < 71 → ¬ satisfies_condition m) ∧
  satisfies_condition 71 := by sorry

end smallest_n_satisfying_condition_l1934_193452


namespace equilateral_triangle_sum_product_l1934_193481

/-- Given complex numbers p, q, and r forming an equilateral triangle with side length 24,
    if |p + q + r| = 48, then |pq + pr + qr| = 768 -/
theorem equilateral_triangle_sum_product (p q r : ℂ) :
  (Complex.abs (p - q) = 24) →
  (Complex.abs (q - r) = 24) →
  (Complex.abs (r - p) = 24) →
  (Complex.abs (p + q + r) = 48) →
  Complex.abs (p*q + q*r + r*p) = 768 := by
  sorry

end equilateral_triangle_sum_product_l1934_193481


namespace cube_sum_and_reciprocal_l1934_193402

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := by
  sorry

end cube_sum_and_reciprocal_l1934_193402


namespace sum_of_squares_l1934_193457

theorem sum_of_squares (a b c : ℝ) (h : a + 19 = b + 9 ∧ b + 9 = c + 8) :
  (a - b)^2 + (b - c)^2 + (c - a)^2 = 222 := by
  sorry

end sum_of_squares_l1934_193457


namespace largest_divisor_of_expression_l1934_193425

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  ∃ (k : ℤ), (15 * x + 3) * (15 * x + 9) * (5 * x + 10) = 90 * k ∧
  ∀ (m : ℤ), m > 90 → ¬(∀ (y : ℤ), Even y →
    ∃ (l : ℤ), (15 * y + 3) * (15 * y + 9) * (5 * y + 10) = m * l) :=
sorry

end largest_divisor_of_expression_l1934_193425


namespace a_composition_zero_l1934_193423

def a (k : ℕ) : ℕ := (2 * k + 1) ^ k

theorem a_composition_zero : a (a (a 0)) = 343 := by sorry

end a_composition_zero_l1934_193423


namespace prime_factor_count_l1934_193415

theorem prime_factor_count (p : ℕ) : 
  (26 : ℕ) + p + (2 : ℕ) = (33 : ℕ) → p = (5 : ℕ) := by
  sorry

#check prime_factor_count

end prime_factor_count_l1934_193415


namespace biased_coin_theorem_l1934_193420

/-- The probability of getting heads in one flip of a biased coin -/
def h : ℚ := 3/7

/-- The probability of getting exactly k heads in n flips -/
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1-p)^(n-k)

theorem biased_coin_theorem :
  (prob_k_heads 6 2 h ≠ 0) ∧
  (prob_k_heads 6 2 h = prob_k_heads 6 3 h) ∧
  (h = 3/7) ∧
  (prob_k_heads 6 4 h = 240/1453) := by
  sorry

#eval Nat.gcd 240 1453 -- To verify that 240/1453 is in lowest terms

#eval 240 + 1453 -- To verify the final answer

end biased_coin_theorem_l1934_193420


namespace sirokas_guests_l1934_193417

/-- The number of guests Mrs. Široká was expecting -/
def num_guests : ℕ := 11

/-- The number of sandwiches in the first scenario -/
def sandwiches1 : ℕ := 25

/-- The number of sandwiches in the second scenario -/
def sandwiches2 : ℕ := 35

/-- The number of sandwiches in the final scenario -/
def sandwiches3 : ℕ := 52

theorem sirokas_guests :
  (sandwiches1 < 2 * num_guests + 3) ∧
  (sandwiches1 ≥ 2 * num_guests) ∧
  (sandwiches2 < 3 * num_guests + 4) ∧
  (sandwiches2 ≥ 3 * num_guests) ∧
  (sandwiches3 ≥ 4 * num_guests) ∧
  (sandwiches3 < 5 * num_guests) :=
by sorry

end sirokas_guests_l1934_193417


namespace value_subtracted_l1934_193483

theorem value_subtracted (n : ℝ) (x : ℝ) : 
  (2 * n + 20 = 8 * n - x) → 
  (n = 4) → 
  x = 4 := by
sorry

end value_subtracted_l1934_193483


namespace correct_calculation_l1934_193488

theorem correct_calculation (x : ℝ) (h : 7 * x = 70) : 36 - x = 26 := by
  sorry

end correct_calculation_l1934_193488


namespace infinite_series_sum_l1934_193496

theorem infinite_series_sum : 
  let r := (1 : ℝ) / 1950
  let S := ∑' n, n * r^(n-1)
  S = 3802500 / 3802601 := by sorry

end infinite_series_sum_l1934_193496


namespace sufficient_not_necessary_l1934_193434

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x = 3 → x^2 = 9) ∧ 
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) :=
sorry

end sufficient_not_necessary_l1934_193434


namespace complement_intersection_l1934_193482

theorem complement_intersection (I A B : Set ℕ) : 
  I = {1, 2, 3, 4, 5} →
  A = {1, 2} →
  B = {1, 3, 5} →
  (I \ A) ∩ B = {3, 5} := by
sorry

end complement_intersection_l1934_193482


namespace age_difference_decade_difference_l1934_193499

/-- Given that the sum of ages of x and y is 10 years greater than the sum of ages of y and z,
    prove that x is 1 decade older than z. -/
theorem age_difference (x y z : ℕ) (h : x + y = y + z + 10) : x = z + 10 := by
  sorry

/-- A decade is defined as 10 years. -/
def decade : ℕ := 10

/-- Given that x is 10 years older than z, prove that x is 1 decade older than z. -/
theorem decade_difference (x z : ℕ) (h : x = z + 10) : x = z + decade := by
  sorry

end age_difference_decade_difference_l1934_193499


namespace pi_over_two_not_fraction_l1934_193413

-- Define what a fraction is
def is_fraction (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- State the theorem
theorem pi_over_two_not_fraction : ¬ is_fraction (π / 2) := by
  sorry

end pi_over_two_not_fraction_l1934_193413


namespace balance_after_transfer_l1934_193469

def initial_balance : ℝ := 400
def transfer_amount : ℝ := 90
def service_charge_rate : ℝ := 0.02

def final_balance : ℝ := initial_balance - (transfer_amount * (1 + service_charge_rate))

theorem balance_after_transfer :
  final_balance = 308.2 := by sorry

end balance_after_transfer_l1934_193469


namespace molecular_weight_of_one_mole_l1934_193456

/-- The molecular weight of aluminum sulfide for a given number of moles -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- The number of moles used in the given condition -/
def given_moles : ℝ := 4

/-- The molecular weight for the given number of moles -/
def given_weight : ℝ := 600

/-- Theorem: The molecular weight of one mole of aluminum sulfide is 150 g/mol -/
theorem molecular_weight_of_one_mole : 
  molecular_weight 1 = 150 := by sorry

end molecular_weight_of_one_mole_l1934_193456


namespace problem_1_problem_2_problem_3_l1934_193455

def balanced_about_2 (a b : ℝ) : Prop := a + b = 2

theorem problem_1 : balanced_about_2 3 (-1) := by sorry

theorem problem_2 (x : ℝ) : balanced_about_2 (x - 3) (5 - x) := by sorry

def a (x : ℝ) : ℝ := 2 * x^2 - 3 * (x^2 + x) + 4
def b (x : ℝ) : ℝ := 2 * x - (3 * x - (4 * x + x^2) - 2)

theorem problem_3 : ∀ x : ℝ, a x + b x ≠ 2 := by sorry

end problem_1_problem_2_problem_3_l1934_193455


namespace trig_equation_solution_l1934_193491

noncomputable def solve_trig_equation (x : ℝ) : Prop :=
  (1 - Real.sin (2 * x) ≠ 0) ∧ 
  (1 - Real.tan x ≠ 0) ∧ 
  (Real.cos x ≠ 0) ∧
  ((1 + Real.sin (2 * x)) / (1 - Real.sin (2 * x)) + 
   2 * ((1 + Real.tan x) / (1 - Real.tan x)) - 3 = 0)

theorem trig_equation_solution :
  ∀ x : ℝ, solve_trig_equation x ↔ 
    (∃ k : ℤ, x = k * Real.pi) ∨
    (∃ n : ℤ, x = Real.arctan 2 + n * Real.pi) :=
by sorry

end trig_equation_solution_l1934_193491


namespace player_a_winning_strategy_l1934_193473

/-- Represents a point on the chessboard -/
structure Point where
  x : Int
  y : Int

/-- Defines the chessboard -/
def is_on_board (p : Point) : Prop :=
  abs p.x ≤ 2019 ∧ abs p.y ≤ 2019 ∧ abs p.x + abs p.y < 4038

/-- Defines a boundary point -/
def is_boundary_point (p : Point) : Prop :=
  abs p.x = 2019 ∨ abs p.y = 2019

/-- Defines adjacent points -/
def are_adjacent (p1 p2 : Point) : Prop :=
  abs (p1.x - p2.x) + abs (p1.y - p2.y) = 1

/-- Represents the state of the game -/
structure GameState where
  piece_position : Point
  removed_points : Set Point

/-- Player A's move -/
def player_a_move (state : GameState) : GameState :=
  sorry

/-- Player B's move -/
def player_b_move (state : GameState) : GameState :=
  sorry

/-- Theorem stating that Player A has a winning strategy -/
theorem player_a_winning_strategy :
  ∃ (strategy : GameState → GameState),
    ∀ (initial_state : GameState),
      initial_state.piece_position = ⟨0, 0⟩ →
      ∀ (n : ℕ),
        let final_state := (strategy ∘ player_b_move)^[n] initial_state
        ∀ (p : Point), is_boundary_point p → p ∈ final_state.removed_points :=
  sorry

end player_a_winning_strategy_l1934_193473


namespace triangle_angle_ratio_l1934_193412

theorem triangle_angle_ratio (a b c : ℝ) (h_sum : a + b + c = 180)
  (h_ratio : ∃ (x : ℝ), a = 4*x ∧ b = 5*x ∧ c = 9*x) (h_smallest : min a (min b c) > 40) :
  max a (max b c) = 90 := by
  sorry

end triangle_angle_ratio_l1934_193412


namespace min_value_theorem_l1934_193495

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 3) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → 1/x + 4/(5+y) ≥ 9/8) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 3 ∧ 1/x + 4/(5+y) = 9/8) :=
by sorry

end min_value_theorem_l1934_193495


namespace total_salaries_proof_l1934_193454

/-- Proves that given the conditions of A and B's salaries and spending,
    their total salaries amount to $5000 -/
theorem total_salaries_proof (A_salary B_salary : ℝ) : 
  A_salary = 3750 →
  A_salary * 0.05 = B_salary * 0.15 →
  A_salary + B_salary = 5000 := by
  sorry

end total_salaries_proof_l1934_193454


namespace range_of_a_l1934_193432

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- State the theorem
theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  C a ⊆ (A ∩ (Set.univ \ B)) →
  (0 < a ∧ a ≤ 2/3) ∨ (-2/3 ≤ a ∧ a < 0) :=
sorry

end range_of_a_l1934_193432


namespace max_students_distribution_l1934_193441

theorem max_students_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1230) (h_pencils : pencils = 920) : 
  (Nat.gcd pens pencils) = 10 := by
  sorry

end max_students_distribution_l1934_193441


namespace class_composition_l1934_193462

theorem class_composition (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 56)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  (boy_ratio : ℚ) / (boy_ratio + girl_ratio) * 100 = 42.86 ∧ 
  (girl_ratio * total_students) / (boy_ratio + girl_ratio) = 32 := by
sorry

end class_composition_l1934_193462


namespace expand_expression_l1934_193480

theorem expand_expression (x : ℝ) : (15 * x^2 + 5 - 3 * x) * 3 * x^3 = 45 * x^5 - 9 * x^4 + 15 * x^3 := by
  sorry

end expand_expression_l1934_193480


namespace sum_of_abs_roots_l1934_193484

theorem sum_of_abs_roots : ∃ (r₁ r₂ r₃ r₄ : ℝ),
  (∀ x : ℝ, x^4 - 4*x^3 - 4*x^2 + 16*x - 8 = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄)) ∧
  |r₁| + |r₂| + |r₃| + |r₄| = 2 + 2 * Real.sqrt 2 + 2 * Real.sqrt 3 := by
  sorry

end sum_of_abs_roots_l1934_193484


namespace planes_parallel_to_same_plane_are_parallel_l1934_193494

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define parallelism between planes
def parallel (p1 p2 : Plane3D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

-- Theorem statement
theorem planes_parallel_to_same_plane_are_parallel (p1 p2 p3 : Plane3D) :
  parallel p1 p3 → parallel p2 p3 → parallel p1 p2 := by
  sorry


end planes_parallel_to_same_plane_are_parallel_l1934_193494


namespace minimum_bottles_needed_l1934_193447

/-- The capacity of a medium-sized bottle in milliliters -/
def medium_bottle_capacity : ℕ := 80

/-- The capacity of a very large bottle in milliliters -/
def large_bottle_capacity : ℕ := 1200

/-- The maximum number of additional bottles allowed -/
def max_additional_bottles : ℕ := 5

/-- The minimum number of medium-sized bottles needed -/
def min_bottles_needed : ℕ := 15

theorem minimum_bottles_needed :
  (large_bottle_capacity / medium_bottle_capacity = min_bottles_needed) ∧
  (min_bottles_needed + max_additional_bottles ≥ 
   (large_bottle_capacity + medium_bottle_capacity - 1) / medium_bottle_capacity) :=
sorry

end minimum_bottles_needed_l1934_193447


namespace rulers_in_drawer_l1934_193449

/-- The number of rulers remaining in a drawer after some are removed -/
def rulers_remaining (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem: Given 46 initial rulers and 25 removed, 21 rulers remain -/
theorem rulers_in_drawer : rulers_remaining 46 25 = 21 := by
  sorry

end rulers_in_drawer_l1934_193449


namespace bob_oyster_shucking_l1934_193400

/-- The number of oysters Bob can shuck in 2 hours -/
def oysters_in_two_hours : ℕ :=
  let oysters_per_five_minutes : ℕ := 10
  let minutes_per_hour : ℕ := 60
  let hours : ℕ := 2
  let total_minutes : ℕ := hours * minutes_per_hour
  (oysters_per_five_minutes * total_minutes) / 5

theorem bob_oyster_shucking :
  oysters_in_two_hours = 240 :=
by sorry

end bob_oyster_shucking_l1934_193400


namespace find_a_value_l1934_193453

theorem find_a_value (x a : ℝ) (h : x = -1 ∧ -2 * (x - a) = 4) : a = 1 := by
  sorry

end find_a_value_l1934_193453


namespace cristina_catches_nicky_l1934_193466

/-- Proves that Cristina catches up to Nicky in 27 seconds --/
theorem cristina_catches_nicky (cristina_speed nicky_speed : ℝ) (head_start : ℝ) 
  (h1 : cristina_speed > nicky_speed)
  (h2 : cristina_speed = 5)
  (h3 : nicky_speed = 3)
  (h4 : head_start = 54) :
  (head_start / (cristina_speed - nicky_speed) = 27) := by
  sorry

end cristina_catches_nicky_l1934_193466
