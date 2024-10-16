import Mathlib

namespace NUMINAMATH_CALUDE_article_large_font_pages_l1282_128216

/-- Represents the number of pages in large font for an article -/
def large_font_pages (total_words : ℕ) (words_per_large_page : ℕ) (words_per_small_page : ℕ) (total_pages : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of large font pages in the article -/
theorem article_large_font_pages :
  large_font_pages 48000 1800 2400 21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_article_large_font_pages_l1282_128216


namespace NUMINAMATH_CALUDE_sarah_walk_probability_l1282_128272

/-- The number of gates at the airport -/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet -/
def gate_distance : ℕ := 80

/-- The maximum distance Sarah is willing to walk in feet -/
def max_walk_distance : ℕ := 320

/-- The probability that Sarah walks 320 feet or less to her new gate -/
theorem sarah_walk_probability : 
  (num_gates : ℚ) * (max_walk_distance / gate_distance * 2 : ℚ) / 
  ((num_gates : ℚ) * (num_gates - 1 : ℚ)) = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_sarah_walk_probability_l1282_128272


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1282_128243

/-- The line (k+1)x-(2k-1)y+3k=0 always passes through the point (-1, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k + 1) * (-1) - (2 * k - 1) * 1 + 3 * k = 0 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1282_128243


namespace NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_implies_squares_l1282_128259

theorem arithmetic_progression_reciprocals_implies_squares
  (a b c : ℝ)
  (h : ∃ (k : ℝ), (1 / (a + c)) - (1 / (a + b)) = k ∧ (1 / (b + c)) - (1 / (a + c)) = k) :
  ∃ (r : ℝ), b^2 - a^2 = r ∧ c^2 - b^2 = r :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_reciprocals_implies_squares_l1282_128259


namespace NUMINAMATH_CALUDE_binomial_10_choose_2_l1282_128278

theorem binomial_10_choose_2 : Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_2_l1282_128278


namespace NUMINAMATH_CALUDE_nell_cards_to_john_l1282_128257

def cards_problem (initial_cards : ℕ) (cards_to_jeff : ℕ) (cards_left : ℕ) : Prop :=
  let total_given_away := initial_cards - cards_left
  let cards_to_john := total_given_away - cards_to_jeff
  cards_to_john = 195

theorem nell_cards_to_john :
  cards_problem 573 168 210 :=
by
  sorry

end NUMINAMATH_CALUDE_nell_cards_to_john_l1282_128257


namespace NUMINAMATH_CALUDE_third_angle_is_75_l1282_128264

/-- A triangle formed by folding a square piece of paper -/
structure FoldedTriangle where
  /-- Angle formed by splitting a right angle in half -/
  angle_mna : ℝ
  /-- Angle formed by three equal angles adding up to 180° -/
  angle_amn : ℝ
  /-- The third angle of the triangle -/
  angle_anm : ℝ
  /-- Proof that angle_mna is 45° -/
  h_mna : angle_mna = 45
  /-- Proof that angle_amn is 60° -/
  h_amn : angle_amn = 60
  /-- Proof that the sum of all angles is 180° -/
  h_sum : angle_mna + angle_amn + angle_anm = 180

/-- Theorem stating that the third angle is 75° -/
theorem third_angle_is_75 (t : FoldedTriangle) : t.angle_anm = 75 := by
  sorry

end NUMINAMATH_CALUDE_third_angle_is_75_l1282_128264


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l1282_128293

theorem inequality_system_solution_set :
  let S := {x : ℝ | (1 + x > -1) ∧ (4 - 2*x ≥ 0)}
  S = {x : ℝ | -2 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l1282_128293


namespace NUMINAMATH_CALUDE_inequality_solution_l1282_128267

theorem inequality_solution (x : ℝ) :
  x ≠ -4 ∧ x ≠ -10/3 →
  ((2*x + 3) / (x + 4) > (4*x + 5) / (3*x + 10) ↔ 
   x < -5/2 ∨ x > -2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1282_128267


namespace NUMINAMATH_CALUDE_camping_bowls_l1282_128227

theorem camping_bowls (total_bowls : ℕ) (rice_per_person : ℚ) (dish_per_person : ℚ) (soup_per_person : ℚ) :
  total_bowls = 55 ∧ 
  rice_per_person = 1 ∧ 
  dish_per_person = 1/2 ∧ 
  soup_per_person = 1/3 →
  (total_bowls : ℚ) / (rice_per_person + dish_per_person + soup_per_person) = 30 := by
sorry

end NUMINAMATH_CALUDE_camping_bowls_l1282_128227


namespace NUMINAMATH_CALUDE_points_needed_for_average_l1282_128240

/-- 
Given a basketball player who has scored 333 points in 10 games, 
this theorem proves that the player needs to score 41 points in the 11th game 
to achieve an average of 34 points over 11 games.
-/
theorem points_needed_for_average (total_points : ℕ) (num_games : ℕ) (target_average : ℕ) :
  total_points = 333 →
  num_games = 10 →
  target_average = 34 →
  (total_points + 41) / (num_games + 1) = target_average := by
  sorry

end NUMINAMATH_CALUDE_points_needed_for_average_l1282_128240


namespace NUMINAMATH_CALUDE_complex_simplification_l1282_128225

/-- Given that i^2 = -1, prove that 3(4-2i) - 2i(3-i) + i(1+2i) = 8 - 11i -/
theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  3 * (4 - 2*i) - 2*i*(3 - i) + i*(1 + 2*i) = 8 - 11*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1282_128225


namespace NUMINAMATH_CALUDE_f_max_min_l1282_128236

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- State the theorem
theorem f_max_min :
  (∀ x, f x ≤ 15) ∧ (∃ x, f x = 15) ∧
  (∀ x, f x ≥ -1) ∧ (∃ x, f x = -1) :=
sorry

end NUMINAMATH_CALUDE_f_max_min_l1282_128236


namespace NUMINAMATH_CALUDE_max_value_on_circle_l1282_128255

/-- Given a complex number z = x + yi where x and y are real numbers,
    and |z - 3| = 1, the maximum value of x^2 + y^2 + 4x + 1 is 33. -/
theorem max_value_on_circle (x y : ℝ) (z : ℂ) (h : z = x + y * Complex.I) 
    (h_circle : Complex.abs (z - 3) = 1) : 
    (∀ a b : ℝ, Complex.abs ((a + b * Complex.I) - 3) = 1 → 
      x^2 + y^2 + 4*x + 1 ≥ a^2 + b^2 + 4*a + 1) ∧
    (∃ u v : ℝ, Complex.abs ((u + v * Complex.I) - 3) = 1 ∧ 
      u^2 + v^2 + 4*u + 1 = 33) :=
by sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l1282_128255


namespace NUMINAMATH_CALUDE_min_distance_after_11_hours_l1282_128260

/-- Represents the turtle's movement on a 2D plane -/
structure TurtleMovement where
  speed : ℝ
  duration : ℕ

/-- Calculates the minimum possible distance from the starting point -/
def minDistanceFromStart (movement : TurtleMovement) : ℝ :=
  sorry

/-- Theorem stating the minimum distance for the given conditions -/
theorem min_distance_after_11_hours :
  let movement : TurtleMovement := ⟨5, 11⟩
  minDistanceFromStart movement = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_after_11_hours_l1282_128260


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1282_128279

def M : Set ℝ := {x | |x| = 1}
def N : Set ℝ := {x | x^2 ≠ x}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1282_128279


namespace NUMINAMATH_CALUDE_no_prime_arithmetic_progression_l1282_128287

def arithmeticProgression (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem no_prime_arithmetic_progression :
  ∀ (a₁ d : ℕ), Prime a₁ → d ≠ 0 →
  ∃ (n : ℕ), ¬ Prime (arithmeticProgression a₁ d n) :=
by sorry

end NUMINAMATH_CALUDE_no_prime_arithmetic_progression_l1282_128287


namespace NUMINAMATH_CALUDE_white_washing_cost_l1282_128273

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the window dimensions
def window_height : ℝ := 4
def window_width : ℝ := 3

-- Define the number of windows
def num_windows : ℕ := 3

-- Define the cost per square foot
def cost_per_sqft : ℝ := 7

-- Theorem statement
theorem white_washing_cost :
  let total_wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := window_height * window_width * num_windows
  let adjusted_wall_area := total_wall_area - door_area - window_area
  adjusted_wall_area * cost_per_sqft = 6342 := by
  sorry


end NUMINAMATH_CALUDE_white_washing_cost_l1282_128273


namespace NUMINAMATH_CALUDE_election_result_l1282_128275

theorem election_result (r d : ℝ) (x_rep x_dem : ℝ) :
  r / d = 3 / 2 →
  x_dem = 15 →
  (r * x_rep / 100 + d * x_dem / 100) / (r + d) = 60.000000000000002 / 100 →
  x_rep = 90.00000000000003 :=
by sorry

end NUMINAMATH_CALUDE_election_result_l1282_128275


namespace NUMINAMATH_CALUDE_equation_solution_l1282_128266

theorem equation_solution (y z w : ℝ) :
  let f : ℝ → ℝ := λ x => (x + y) / (y + z) - (z + w) / (w + x)
  let sol₁ := (-(w + y) + Real.sqrt ((w + y)^2 + 4*(z - w)*(z - y))) / 2
  let sol₂ := (-(w + y) - Real.sqrt ((w + y)^2 + 4*(z - w)*(z - y))) / 2
  (∀ x, f x = 0 ↔ x = sol₁ ∨ x = sol₂) ∧ (f sol₁ = 0 ∧ f sol₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1282_128266


namespace NUMINAMATH_CALUDE_sin_690_degrees_l1282_128295

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l1282_128295


namespace NUMINAMATH_CALUDE_hash_difference_eight_five_l1282_128292

-- Define the # operation
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

-- Theorem statement
theorem hash_difference_eight_five : hash 8 5 - hash 5 8 = -12 := by sorry

end NUMINAMATH_CALUDE_hash_difference_eight_five_l1282_128292


namespace NUMINAMATH_CALUDE_mandy_reading_progression_l1282_128229

/-- Calculates the present book length given Mandy's reading progression --/
def present_book_length (starting_age : ℕ) (starting_length : ℕ) 
  (double_age_multiplier : ℕ) (eight_years_later_multiplier : ℕ) 
  (present_multiplier : ℕ) : ℕ :=
  let double_age_length := starting_length * double_age_multiplier
  let eight_years_later_length := double_age_length * eight_years_later_multiplier
  eight_years_later_length * present_multiplier

/-- Theorem stating that the present book length is 480 pages --/
theorem mandy_reading_progression : 
  present_book_length 6 8 5 3 4 = 480 := by
  sorry

#eval present_book_length 6 8 5 3 4

end NUMINAMATH_CALUDE_mandy_reading_progression_l1282_128229


namespace NUMINAMATH_CALUDE_equation_solution_l1282_128285

theorem equation_solution (a : ℚ) : 
  (∀ x, a * x - 4 * (x - a) = 1) → (a * 2 - 4 * (2 - a) = 1) → a = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1282_128285


namespace NUMINAMATH_CALUDE_checker_rearrangement_impossible_l1282_128219

/-- Represents a 5x5 chessboard -/
def Chessboard := Fin 5 → Fin 5 → Bool

/-- A function that determines if two positions are adjacent (horizontally or vertically) -/
def adjacent (p1 p2 : Fin 5 × Fin 5) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- A function representing the initial placement of checkers -/
def initial_placement : Fin 5 × Fin 5 → Fin 5 × Fin 5 :=
  fun p => p

/-- A function representing the final placement of checkers -/
def final_placement : Fin 5 × Fin 5 → Fin 5 × Fin 5 :=
  sorry

/-- Theorem stating that it's impossible to rearrange the checkers as required -/
theorem checker_rearrangement_impossible :
  ¬ (∀ p : Fin 5 × Fin 5, adjacent (initial_placement p) (final_placement p)) ∧
    (∀ p : Fin 5 × Fin 5, ∃ q, final_placement q = p) :=
sorry

end NUMINAMATH_CALUDE_checker_rearrangement_impossible_l1282_128219


namespace NUMINAMATH_CALUDE_platform_length_l1282_128253

/-- The length of a platform given train specifications -/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmph = 72 →
  crossing_time = 20 →
  (train_speed_kmph * (1000 / 3600) * crossing_time) - train_length = 150 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l1282_128253


namespace NUMINAMATH_CALUDE_f_value_at_3_l1282_128208

theorem f_value_at_3 (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 + b * x + 2) →
  f 1 = 3 →
  f 2 = 12 →
  f 3 = 29 := by
sorry

end NUMINAMATH_CALUDE_f_value_at_3_l1282_128208


namespace NUMINAMATH_CALUDE_soda_cans_problem_l1282_128294

/-- The number of cans Tim initially had -/
def initial_cans : ℕ := 22

/-- The number of cans Jeff took -/
def cans_taken : ℕ := 6

/-- The number of cans Tim had after Jeff took some -/
def cans_after_taken : ℕ := initial_cans - cans_taken

/-- The number of cans Tim bought -/
def cans_bought : ℕ := cans_after_taken / 2

/-- The final number of cans Tim had -/
def final_cans : ℕ := 24

theorem soda_cans_problem :
  cans_after_taken + cans_bought = final_cans :=
by sorry

end NUMINAMATH_CALUDE_soda_cans_problem_l1282_128294


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_l1282_128228

theorem square_area_from_rectangle (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) (rectangle_area : ℝ) :
  rectangle_length = (2 / 5) * circle_radius →
  rectangle_breadth = 10 →
  rectangle_area = 160 →
  rectangle_area = rectangle_length * rectangle_breadth →
  (circle_radius ^ 2 : ℝ) = 1600 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_l1282_128228


namespace NUMINAMATH_CALUDE_art_cost_theorem_l1282_128211

def art_cost_problem (cost_A : ℝ) (cost_B : ℝ) (cost_C : ℝ) (cost_D : ℝ) : Prop :=
  let pieces_A := 3
  let pieces_B := 2
  let pieces_C := 3
  let pieces_D := 1
  let total_cost_A := cost_A * pieces_A
  let total_cost_B := cost_B * pieces_B
  let total_cost_C := cost_C * pieces_C
  let total_cost_D := cost_D * pieces_D
  let total_cost := total_cost_A + total_cost_B + total_cost_C + total_cost_D

  (total_cost_A = 45000) ∧
  (cost_B = cost_A * 1.25) ∧
  (cost_C = cost_A * 1.5) ∧
  (cost_D = total_cost_C * 2) ∧
  (total_cost = 285000)

theorem art_cost_theorem : ∃ cost_A cost_B cost_C cost_D, art_cost_problem cost_A cost_B cost_C cost_D :=
  sorry

end NUMINAMATH_CALUDE_art_cost_theorem_l1282_128211


namespace NUMINAMATH_CALUDE_movie_theater_seating_l1282_128232

/-- The number of ways to seat people in a row of seats with constraints -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  -- Define the function to calculate the number of seating arrangements
  sorry

/-- Theorem stating the number of seating arrangements for the specific problem -/
theorem movie_theater_seating : seating_arrangements 9 3 = 42 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_seating_l1282_128232


namespace NUMINAMATH_CALUDE_junk_mail_per_block_l1282_128284

/-- The number of houses in a block -/
def houses_per_block : ℕ := 20

/-- The number of junk mail pieces given to each house -/
def junk_mail_per_house : ℕ := 32

/-- Theorem: The total number of junk mail pieces per block -/
theorem junk_mail_per_block :
  houses_per_block * junk_mail_per_house = 640 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_per_block_l1282_128284


namespace NUMINAMATH_CALUDE_velvet_fabric_cost_l1282_128281

/-- Proves that the cost of velvet fabric per yard is $24 -/
theorem velvet_fabric_cost
  (total_spent : ℝ)
  (pattern_cost : ℝ)
  (thread_cost_per_spool : ℝ)
  (num_thread_spools : ℕ)
  (fabric_yards : ℝ)
  (h1 : total_spent = 141)
  (h2 : pattern_cost = 15)
  (h3 : thread_cost_per_spool = 3)
  (h4 : num_thread_spools = 2)
  (h5 : fabric_yards = 5)
  : (total_spent - pattern_cost - thread_cost_per_spool * num_thread_spools) / fabric_yards = 24 := by
  sorry

end NUMINAMATH_CALUDE_velvet_fabric_cost_l1282_128281


namespace NUMINAMATH_CALUDE_hyperbola_center_l1282_128204

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 + 54 * x - 16 * y^2 - 128 * y - 200 = 0

/-- The center of a hyperbola -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (-3, -4) -/
theorem hyperbola_center : is_center (-3) (-4) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l1282_128204


namespace NUMINAMATH_CALUDE_lumberjack_problem_l1282_128207

theorem lumberjack_problem (logs_per_tree : ℕ) (firewood_per_log : ℕ) (total_firewood : ℕ) :
  logs_per_tree = 4 →
  firewood_per_log = 5 →
  total_firewood = 500 →
  (total_firewood / firewood_per_log) / logs_per_tree = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_lumberjack_problem_l1282_128207


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1282_128226

theorem reciprocal_of_negative_two :
  ∃ x : ℚ, x * (-2) = 1 ∧ x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1282_128226


namespace NUMINAMATH_CALUDE_fifteen_students_prefer_dogs_l1282_128270

/-- Represents the preferences of students in a class survey -/
structure ClassPreferences where
  total_students : ℕ
  dogs_videogames_chocolate : Rat
  dogs_videogames_vanilla : Rat
  dogs_movies_chocolate : Rat
  dogs_movies_vanilla : Rat
  cats_movies_chocolate : Rat
  cats_movies_vanilla : Rat
  cats_videogames_chocolate : Rat
  cats_videogames_vanilla : Rat

/-- Theorem stating that 15 students prefer dogs given the survey results -/
theorem fifteen_students_prefer_dogs (prefs : ClassPreferences) : 
  prefs.total_students = 30 ∧
  prefs.dogs_videogames_chocolate = 25/100 ∧
  prefs.dogs_videogames_vanilla = 5/100 ∧
  prefs.dogs_movies_chocolate = 10/100 ∧
  prefs.dogs_movies_vanilla = 10/100 ∧
  prefs.cats_movies_chocolate = 15/100 ∧
  prefs.cats_movies_vanilla = 10/100 ∧
  prefs.cats_videogames_chocolate = 5/100 ∧
  prefs.cats_videogames_vanilla = 10/100 →
  (prefs.dogs_videogames_chocolate + prefs.dogs_videogames_vanilla + 
   prefs.dogs_movies_chocolate + prefs.dogs_movies_vanilla) * prefs.total_students = 15 := by
  sorry


end NUMINAMATH_CALUDE_fifteen_students_prefer_dogs_l1282_128270


namespace NUMINAMATH_CALUDE_right_triangle_ab_length_l1282_128218

/-- Given a right triangle ABC in the x-y plane where:
  - ∠B = 90°
  - The length of AC is 100
  - The slope of line segment AC is 4/3
  Prove that the length of AB is 80 -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 100)
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_ab_length_l1282_128218


namespace NUMINAMATH_CALUDE_sum_of_cubes_equation_l1282_128274

theorem sum_of_cubes_equation (x y : ℝ) :
  x^3 + 21*x*y + y^3 = 343 → (x + y = 7 ∨ x + y = -14) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equation_l1282_128274


namespace NUMINAMATH_CALUDE_expression_evaluation_l1282_128291

theorem expression_evaluation (x : ℝ) (h : 2*x - 9 ≥ 0) :
  (3 + Real.sqrt (2*x - 9))^2 - 3*x = -x + 6*Real.sqrt (2*x - 9) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1282_128291


namespace NUMINAMATH_CALUDE_expression_value_l1282_128262

theorem expression_value : (85 + 32 / 113) * 113 = 9637 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1282_128262


namespace NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1282_128201

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  (x % 9 = 8) ∧
  (∀ y : ℕ, y > 0 → y % 5 = 4 → y % 7 = 6 → y % 9 = 8 → x ≤ y) ∧
  (x = 314) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_with_remainders_l1282_128201


namespace NUMINAMATH_CALUDE_passengers_at_terminal_l1282_128209

/-- Represents the number of stations on the bus route. -/
def num_stations : ℕ := 8

/-- Represents the number of people who boarded the bus at the first 6 stations. -/
def passengers_boarded : ℕ := 100

/-- Represents the number of people who got off at all stations except the terminal station. -/
def passengers_got_off : ℕ := 80

/-- Theorem stating that the number of passengers who boarded at the first 6 stations
    and got off at the terminal station is 20. -/
theorem passengers_at_terminal : ℕ := by
  sorry

#check passengers_at_terminal

end NUMINAMATH_CALUDE_passengers_at_terminal_l1282_128209


namespace NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l1282_128220

/-- A polygon with interior angles summing to 540° has 5 sides -/
theorem polygon_with_540_degree_sum_is_pentagon (n : ℕ) : 
  (n - 2) * 180 = 540 → n = 5 := by sorry

end NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l1282_128220


namespace NUMINAMATH_CALUDE_dodecagon_enclosure_l1282_128283

theorem dodecagon_enclosure (n : ℕ) : n > 2 → (
  let dodecagon_exterior_angle : ℝ := 360 / 12
  let ngon_exterior_angle : ℝ := 360 / n
  3 * ngon_exterior_angle = dodecagon_exterior_angle
) → n = 12 := by sorry

end NUMINAMATH_CALUDE_dodecagon_enclosure_l1282_128283


namespace NUMINAMATH_CALUDE_equation_equivalence_l1282_128213

theorem equation_equivalence (a b c : ℕ) 
  (ha : 0 < a ∧ a < 12) 
  (hb : 0 < b ∧ b < 12) 
  (hc : 0 < c ∧ c < 12) : 
  ((12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) ↔ (b + c = 12) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1282_128213


namespace NUMINAMATH_CALUDE_lemon_juice_per_lemon_l1282_128215

/-- The amount of lemon juice needed for one dozen cupcakes, in tablespoons -/
def juice_per_dozen : ℚ := 12

/-- The number of dozens of cupcakes to be made -/
def dozens_to_make : ℚ := 3

/-- The number of lemons needed for the total amount of cupcakes -/
def lemons_needed : ℚ := 9

/-- Proves that each lemon provides 4 tablespoons of juice -/
theorem lemon_juice_per_lemon : 
  (juice_per_dozen * dozens_to_make) / lemons_needed = 4 := by
  sorry

end NUMINAMATH_CALUDE_lemon_juice_per_lemon_l1282_128215


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1282_128289

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {y | ∃ x, y = 2^x}

def N : Set Int := {x | x^2 - x - 2 = 0}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1282_128289


namespace NUMINAMATH_CALUDE_water_remaining_l1282_128256

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 11/4 → remaining = initial - used → remaining = 1/4 := by
sorry

end NUMINAMATH_CALUDE_water_remaining_l1282_128256


namespace NUMINAMATH_CALUDE_log_stack_sum_l1282_128276

/-- 
Given a stack of logs where:
- The bottom row has 15 logs
- Each successive row has one less log
- The top row has 5 logs
Prove that the total number of logs in the stack is 110.
-/
theorem log_stack_sum : 
  ∀ (a l n : ℕ), 
    a = 15 → 
    l = 5 → 
    n = a - l + 1 → 
    (n : ℚ) / 2 * (a + l) = 110 := by
  sorry

end NUMINAMATH_CALUDE_log_stack_sum_l1282_128276


namespace NUMINAMATH_CALUDE_sum_of_quadratic_roots_l1282_128299

theorem sum_of_quadratic_roots (y : ℝ) : 
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ y₁^2 - 6*y₁ + 8 = 0 ∧ y₂^2 - 6*y₂ + 8 = 0 ∧ y₁ + y₂ = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_quadratic_roots_l1282_128299


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1282_128205

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : π * R^2 = 4 * π * r^2) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l1282_128205


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_9_mod_18_l1282_128251

theorem least_five_digit_congruent_to_9_mod_18 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 18 = 9 → n ≥ 10008 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_9_mod_18_l1282_128251


namespace NUMINAMATH_CALUDE_special_sequence_is_arithmetic_l1282_128296

/-- A sequence satisfying specific conditions -/
def SpecialSequence (a : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, a (n + 1) > a n ∧ a n > 0) ∧
  (∀ n : ℕ+, a n - 1 / a n < n ∧ n < a n + 1 / a n) ∧
  (∃ m : ℕ+, m ≥ 2 ∧ ∀ n : ℕ+, a (m * n) = m * a n)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Theorem: A special sequence is an arithmetic sequence -/
theorem special_sequence_is_arithmetic (a : ℕ+ → ℝ) 
    (h : SpecialSequence a) : ArithmeticSequence a := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_is_arithmetic_l1282_128296


namespace NUMINAMATH_CALUDE_choose_three_from_eleven_l1282_128271

theorem choose_three_from_eleven (n : ℕ) (k : ℕ) : n = 11 → k = 3 → Nat.choose n k = 165 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_from_eleven_l1282_128271


namespace NUMINAMATH_CALUDE_set_A_properties_l1282_128263

def A : Set ℝ := {x | x^2 - 1 = 0}

theorem set_A_properties : 
  (1 ∈ A) ∧ (∅ ⊆ A) ∧ ({1, -1} ⊆ A) := by sorry

end NUMINAMATH_CALUDE_set_A_properties_l1282_128263


namespace NUMINAMATH_CALUDE_shawn_pebbles_count_l1282_128248

/-- The number of pebbles Shawn collected initially -/
def total_pebbles : ℕ := 40

/-- The number of red pebbles -/
def red_pebbles : ℕ := 9

/-- The number of blue pebbles -/
def blue_pebbles : ℕ := 13

/-- The difference between blue and yellow pebbles -/
def blue_yellow_diff : ℕ := 7

/-- The number of groups the remaining pebbles are divided into -/
def remaining_groups : ℕ := 3

theorem shawn_pebbles_count :
  total_pebbles = red_pebbles + blue_pebbles + remaining_groups * ((blue_pebbles - blue_yellow_diff)) :=
by sorry

end NUMINAMATH_CALUDE_shawn_pebbles_count_l1282_128248


namespace NUMINAMATH_CALUDE_teacher_wang_travel_time_l1282_128286

theorem teacher_wang_travel_time (bicycle_speed : ℝ) (bicycle_time : ℝ) (walking_speed : ℝ) (max_walking_time : ℝ)
  (h1 : bicycle_speed = 15)
  (h2 : bicycle_time = 0.2)
  (h3 : walking_speed = 5)
  (h4 : max_walking_time = 0.7) :
  (bicycle_speed * bicycle_time) / walking_speed < max_walking_time :=
by sorry

end NUMINAMATH_CALUDE_teacher_wang_travel_time_l1282_128286


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l1282_128288

theorem smallest_perfect_square_divisible_by_5_and_6 :
  ∀ n : ℕ, n > 0 → n * n < 900 → ¬(5 ∣ (n * n) ∧ 6 ∣ (n * n)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_5_and_6_l1282_128288


namespace NUMINAMATH_CALUDE_soccer_team_subjects_l1282_128210

theorem soccer_team_subjects (total : ℕ) (physics : ℕ) (both : ℕ) (math : ℕ) :
  total = 20 →
  physics = 12 →
  both = 6 →
  total = physics + math - both →
  math = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_team_subjects_l1282_128210


namespace NUMINAMATH_CALUDE_vector_sum_problem_l1282_128231

theorem vector_sum_problem (x y : ℝ) : 
  let a : ℝ × ℝ := (x, 1)
  let b : ℝ × ℝ := (2, y)
  (a.1 + b.1, a.2 + b.2) = (1, -1) → x + y = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_problem_l1282_128231


namespace NUMINAMATH_CALUDE_stream_speed_l1282_128269

/-- Given a canoe's upstream and downstream speeds, calculate the stream speed -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 3)
  (h_downstream : downstream_speed = 12) :
  ∃ (stream_speed : ℝ), stream_speed = 4.5 ∧
    upstream_speed = (downstream_speed - upstream_speed) / 2 - stream_speed ∧
    downstream_speed = (downstream_speed - upstream_speed) / 2 + stream_speed :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l1282_128269


namespace NUMINAMATH_CALUDE_fishing_ratio_proof_l1282_128221

/-- The ratio of Brian's fishing trips to Chris's fishing trips -/
def fishing_ratio : ℚ := 26/15

theorem fishing_ratio_proof (brian_catch : ℕ) (total_catch : ℕ) (chris_trips : ℕ) :
  brian_catch = 400 →
  total_catch = 13600 →
  chris_trips = 10 →
  (∃ (brian_trips : ℚ),
    brian_trips * brian_catch + chris_trips * (brian_catch * 5/3) = total_catch ∧
    brian_trips / chris_trips = fishing_ratio) :=
by sorry

end NUMINAMATH_CALUDE_fishing_ratio_proof_l1282_128221


namespace NUMINAMATH_CALUDE_factor_expression_l1282_128212

theorem factor_expression (x : ℝ) : x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1282_128212


namespace NUMINAMATH_CALUDE_perpendicular_and_tangent_l1282_128252

-- Define the given curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 5

-- Define the given line
def l₁ (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the line we want to prove
def l₂ (x y : ℝ) : Prop := 3*x + y + 6 = 0

-- Theorem statement
theorem perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- l₂ is perpendicular to l₁
    (∀ (x₁ y₁ x₂ y₂ : ℝ), l₁ x₁ y₁ → l₁ x₂ y₂ → l₂ x₁ y₁ → l₂ x₂ y₂ →
      (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
      ((x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁)) *
      ((x₂ - x₁) * (3) + (y₂ - y₁) * (1)) = 0) ∧
    -- l₂ is tangent to f at (x₀, y₀)
    (l₂ x₀ y₀ ∧ f x₀ = y₀ ∧
      ∀ (x : ℝ), x ≠ x₀ → l₂ x (f x) → False) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_and_tangent_l1282_128252


namespace NUMINAMATH_CALUDE_smaller_bill_denomination_l1282_128241

/-- Given a cashier with bills of two denominations, prove the value of the smaller denomination. -/
theorem smaller_bill_denomination
  (total_bills : ℕ)
  (total_value : ℕ)
  (smaller_bills : ℕ)
  (twenty_bills : ℕ)
  (h_total_bills : total_bills = smaller_bills + twenty_bills)
  (h_total_bills_value : total_bills = 30)
  (h_total_value : total_value = 330)
  (h_smaller_bills : smaller_bills = 27)
  (h_twenty_bills : twenty_bills = 3) :
  ∃ (x : ℕ), x * smaller_bills + 20 * twenty_bills = total_value ∧ x = 10 := by
sorry


end NUMINAMATH_CALUDE_smaller_bill_denomination_l1282_128241


namespace NUMINAMATH_CALUDE_quadratic_transformation_l1282_128280

-- Define the quadratic function
def quadratic (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

-- Define the transformed quadratic function
def transformed_quadratic (m h k : ℝ) (x : ℝ) : ℝ := m * (x - h)^2 + k

-- State the theorem
theorem quadratic_transformation (p q r : ℝ) :
  (∃ m k : ℝ, ∀ x : ℝ, quadratic p q r x = transformed_quadratic 5 3 15 x) →
  (∃ m k : ℝ, ∀ x : ℝ, quadratic (4*p) (4*q) (4*r) x = transformed_quadratic m 3 k x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l1282_128280


namespace NUMINAMATH_CALUDE_total_crayons_l1282_128247

/-- Given that each child has 5 crayons and there are 10 children, 
    prove that the total number of crayons is 50. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 5) 
  (h2 : num_children = 10) : 
  crayons_per_child * num_children = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1282_128247


namespace NUMINAMATH_CALUDE_inequality_relationship_l1282_128217

theorem inequality_relationship (a b : ℝ) : 
  ¬(((2 : ℝ)^a > (2 : ℝ)^b → (1 : ℝ)/a < (1 : ℝ)/b) ∧ 
    ((1 : ℝ)/a < (1 : ℝ)/b → (2 : ℝ)^a > (2 : ℝ)^b)) :=
sorry

end NUMINAMATH_CALUDE_inequality_relationship_l1282_128217


namespace NUMINAMATH_CALUDE_hall_tiling_proof_l1282_128265

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℚ
  width : ℚ

/-- Converts inches to feet -/
def inchesToFeet (inches : ℚ) : ℚ := inches / 12

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℚ := d.length * d.width

/-- Calculates the number of smaller rectangles needed to cover a larger rectangle -/
def tilesRequired (hall : Dimensions) (tile : Dimensions) : ℕ :=
  (area hall / area tile).ceil.toNat

theorem hall_tiling_proof :
  let hall : Dimensions := { length := 15, width := 18 }
  let tile : Dimensions := { length := inchesToFeet 3, width := inchesToFeet 9 }
  tilesRequired hall tile = 1440 := by
  sorry

end NUMINAMATH_CALUDE_hall_tiling_proof_l1282_128265


namespace NUMINAMATH_CALUDE_new_cube_edge_l1282_128245

/-- Given three cubes with edges 6 cm, 8 cm, and 10 cm, prove that when melted and formed into a new cube, the edge of the new cube is 12 cm. -/
theorem new_cube_edge (cube1 cube2 cube3 new_cube : ℝ) : 
  cube1 = 6 → cube2 = 8 → cube3 = 10 → 
  (cube1^3 + cube2^3 + cube3^3)^(1/3) = new_cube → 
  new_cube = 12 := by
sorry

#eval (6^3 + 8^3 + 10^3)^(1/3) -- This should evaluate to 12

end NUMINAMATH_CALUDE_new_cube_edge_l1282_128245


namespace NUMINAMATH_CALUDE_battle_station_staffing_l1282_128235

theorem battle_station_staffing (total_resumes : ℕ) (suitable_fraction : ℚ) 
  (job_openings : ℕ) (h1 : total_resumes = 30) (h2 : suitable_fraction = 2/3) 
  (h3 : job_openings = 5) :
  (total_resumes : ℚ) * suitable_fraction * 
  (total_resumes : ℚ) * suitable_fraction - 1 * 
  (total_resumes : ℚ) * suitable_fraction - 2 * 
  (total_resumes : ℚ) * suitable_fraction - 3 * 
  (total_resumes : ℚ) * suitable_fraction - 4 = 930240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l1282_128235


namespace NUMINAMATH_CALUDE_four_digit_perfect_square_same_digits_l1282_128223

theorem four_digit_perfect_square_same_digits : ∃ n : ℕ,
  (1000 ≤ n) ∧ (n < 10000) ∧  -- four-digit number
  (∃ m : ℕ, n = m^2) ∧  -- perfect square
  (∃ a b : ℕ, n = 1100 * a + 11 * b) ∧  -- first two digits same, last two digits same
  (n = 7744) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_perfect_square_same_digits_l1282_128223


namespace NUMINAMATH_CALUDE_distinct_z_values_exist_l1282_128222

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def reverse_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  1000 * d4 + 100 * d3 + 10 * d2 + d1

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values_exist :
  ∃ (x y : ℕ), is_four_digit x ∧ is_four_digit y ∧ 
  y = reverse_digits x ∧ z x ≠ z y :=
sorry

end NUMINAMATH_CALUDE_distinct_z_values_exist_l1282_128222


namespace NUMINAMATH_CALUDE_tangency_lines_through_diagonal_intersection_l1282_128298

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  vertices : Vector Point 4

-- Function to check if a quadrilateral is circumscribed around a circle
def is_circumscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Function to get tangency points
def tangency_points (q : Quadrilateral) (c : Circle) : Vector Point 4 := sorry

-- Function to get lines connecting opposite tangency points
def opposite_tangency_lines (q : Quadrilateral) (c : Circle) : Vector Line 2 := sorry

-- Function to get diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : Vector Line 2 := sorry

-- Function to check if two lines intersect
def lines_intersect (l1 : Line) (l2 : Line) : Prop := sorry

-- Function to get intersection point of two lines
def intersection_point (l1 : Line) (l2 : Line) : Point := sorry

-- Theorem statement
theorem tangency_lines_through_diagonal_intersection 
  (q : Quadrilateral) (c : Circle) : 
  is_circumscribed q c → 
  let tl := opposite_tangency_lines q c
  let d := diagonals q
  lines_intersect tl[0] tl[1] ∧ 
  lines_intersect d[0] d[1] ∧
  intersection_point tl[0] tl[1] = intersection_point d[0] d[1] := by
  sorry

end NUMINAMATH_CALUDE_tangency_lines_through_diagonal_intersection_l1282_128298


namespace NUMINAMATH_CALUDE_toy_car_gift_difference_l1282_128202

/-- Proves the difference between Mum's and Dad's toy car gifts --/
theorem toy_car_gift_difference :
  ∀ (initial final dad uncle auntie grandpa : ℕ),
  initial = 150 →
  final = 196 →
  dad = 10 →
  auntie = uncle + 1 →
  auntie = 6 →
  grandpa = 2 * uncle →
  ∃ (mum : ℕ),
    final = initial + dad + uncle + auntie + grandpa + mum ∧
    mum - dad = 5 := by
  sorry

end NUMINAMATH_CALUDE_toy_car_gift_difference_l1282_128202


namespace NUMINAMATH_CALUDE_yeonseo_skirt_count_l1282_128234

/-- Given that Yeonseo has more than two types of skirts and pants each,
    there are 4 types of pants, and 7 ways to choose pants or skirts,
    prove that the number of types of skirts is 3. -/
theorem yeonseo_skirt_count :
  ∀ (S P : ℕ),
  S > 2 →
  P > 2 →
  P = 4 →
  S + P = 7 →
  S = 3 := by
sorry

end NUMINAMATH_CALUDE_yeonseo_skirt_count_l1282_128234


namespace NUMINAMATH_CALUDE_simplify_expression_l1282_128258

theorem simplify_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) - ((x^3 - 2) / y) * ((y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1282_128258


namespace NUMINAMATH_CALUDE_bobs_favorite_number_l1282_128246

def is_bobs_favorite (n : ℕ) : Prop :=
  50 < n ∧ n < 100 ∧
  n % 11 = 0 ∧
  n % 2 ≠ 0 ∧
  (n / 10 + n % 10) % 3 = 0

theorem bobs_favorite_number :
  ∃! n : ℕ, is_bobs_favorite n ∧ n = 99 :=
sorry

end NUMINAMATH_CALUDE_bobs_favorite_number_l1282_128246


namespace NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l1282_128237

theorem sqrt_eighteen_div_sqrt_two_equals_three :
  Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eighteen_div_sqrt_two_equals_three_l1282_128237


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1282_128242

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x ∨ f x = 2 - x) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1282_128242


namespace NUMINAMATH_CALUDE_sum_of_digits_of_multiple_of_five_l1282_128233

/-- Given two natural numbers, returns true if they have the same digits in any order -/
def sameDigits (a b : ℕ) : Prop := sorry

/-- Returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_multiple_of_five (a b : ℕ) :
  sameDigits a b → sumOfDigits (5 * a) = sumOfDigits (5 * b) := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_multiple_of_five_l1282_128233


namespace NUMINAMATH_CALUDE_prob_two_sunny_days_value_l1282_128224

/-- The probability of exactly 2 sunny days out of 5 days, where each day has a 75% chance of rain -/
def prob_two_sunny_days : ℚ :=
  (Nat.choose 5 2 : ℚ) * (1/4)^2 * (3/4)^3

/-- The main theorem stating that the probability is equal to 135/512 -/
theorem prob_two_sunny_days_value : prob_two_sunny_days = 135/512 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_sunny_days_value_l1282_128224


namespace NUMINAMATH_CALUDE_cake_payment_dimes_l1282_128250

/-- Represents the number of each type of coin used in the payment -/
structure CoinCount where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- The value of the payment in cents -/
def payment_value (c : CoinCount) : ℕ :=
  c.pennies + 5 * c.nickels + 10 * c.dimes

theorem cake_payment_dimes :
  ∃ (c : CoinCount),
    c.pennies + c.nickels + c.dimes = 50 ∧
    payment_value c = 200 ∧
    c.dimes = 14 := by
  sorry

end NUMINAMATH_CALUDE_cake_payment_dimes_l1282_128250


namespace NUMINAMATH_CALUDE_cos_315_degrees_l1282_128277

theorem cos_315_degrees : Real.cos (315 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_315_degrees_l1282_128277


namespace NUMINAMATH_CALUDE_triangle_side_length_l1282_128261

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A + C = 2 * B → 
  A + B + C = π → 
  a = 1 → 
  b = Real.sqrt 3 → 
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos B) →
  c = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1282_128261


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_count_l1282_128290

/-- Calculates the number of right-handed players in a cricket team -/
def right_handed_players (total_players throwers : ℕ) : ℕ :=
  let non_throwers := total_players - throwers
  let left_handed_non_throwers := non_throwers / 3
  let right_handed_non_throwers := non_throwers - left_handed_non_throwers
  throwers + right_handed_non_throwers

theorem cricket_team_right_handed_count :
  right_handed_players 70 37 = 59 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_count_l1282_128290


namespace NUMINAMATH_CALUDE_shooting_outcomes_l1282_128254

/-- Represents the number of shots -/
def num_shots : ℕ := 6

/-- Represents the number of hits we're interested in -/
def num_hits : ℕ := 3

/-- Calculates the total number of possible outcomes for n shots -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Calculates the number of outcomes with exactly k hits out of n shots -/
def outcomes_with_k_hits (n k : ℕ) : ℕ := choose n k

/-- Calculates the number of outcomes with exactly k hits and exactly 2 consecutive hits out of n shots -/
def outcomes_with_k_hits_and_2_consecutive (n k : ℕ) : ℕ := choose (n - k + 1) 2

theorem shooting_outcomes :
  (total_outcomes num_shots = 64) ∧
  (outcomes_with_k_hits num_shots num_hits = 20) ∧
  (outcomes_with_k_hits_and_2_consecutive num_shots num_hits = 6) := by
  sorry

end NUMINAMATH_CALUDE_shooting_outcomes_l1282_128254


namespace NUMINAMATH_CALUDE_probability_is_half_l1282_128238

/-- An isosceles triangle with 45-degree base angles -/
structure IsoscelesTriangle45 where
  -- We don't need to define the specific geometry, just that it exists
  exists_triangle : True

/-- The triangle is divided into six equal areas -/
def divided_into_six_areas (t : IsoscelesTriangle45) : Prop :=
  ∃ (areas : Finset ℝ), areas.card = 6 ∧ ∀ a ∈ areas, a > 0 ∧ (∀ b ∈ areas, a = b)

/-- Three areas are selected -/
def three_areas_selected (t : IsoscelesTriangle45) (areas : Finset ℝ) : Prop :=
  ∃ (selected : Finset ℝ), selected ⊆ areas ∧ selected.card = 3

/-- The probability of a point falling in the selected areas -/
def probability_in_selected (t : IsoscelesTriangle45) (areas selected : Finset ℝ) : ℚ :=
  (selected.card : ℚ) / (areas.card : ℚ)

/-- The main theorem -/
theorem probability_is_half (t : IsoscelesTriangle45) 
  (h1 : divided_into_six_areas t)
  (h2 : ∃ areas selected, three_areas_selected t areas ∧ selected ⊆ areas) :
  ∃ areas selected, three_areas_selected t areas ∧ 
    probability_in_selected t areas selected = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_half_l1282_128238


namespace NUMINAMATH_CALUDE_total_school_supplies_cost_l1282_128206

-- Define the quantities and prices
def haley_paper_reams : ℕ := 2
def haley_paper_price : ℚ := 3.5
def sister_paper_reams : ℕ := 3
def sister_paper_price : ℚ := 4.25
def haley_pens : ℕ := 5
def haley_pen_price : ℚ := 1.25
def sister_pens : ℕ := 8
def sister_pen_price : ℚ := 1.5

-- Define the theorem
theorem total_school_supplies_cost :
  (haley_paper_reams : ℚ) * haley_paper_price +
  (sister_paper_reams : ℚ) * sister_paper_price +
  (haley_pens : ℚ) * haley_pen_price +
  (sister_pens : ℚ) * sister_pen_price = 38 :=
by sorry

end NUMINAMATH_CALUDE_total_school_supplies_cost_l1282_128206


namespace NUMINAMATH_CALUDE_semicircle_area_l1282_128203

theorem semicircle_area (rectangle_width : Real) (rectangle_length : Real)
  (triangle_leg1 : Real) (triangle_leg2 : Real) :
  rectangle_width = 1 →
  rectangle_length = 3 →
  triangle_leg1 = 1 →
  triangle_leg2 = 2 →
  triangle_leg1^2 + triangle_leg2^2 = rectangle_length^2 →
  (π * rectangle_length^2) / 8 = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_l1282_128203


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l1282_128249

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Given that 2 - a/(2-i) is a pure imaginary number, prove that a = 5 -/
theorem complex_pure_imaginary (a : ℝ) : 
  (∃ b : ℝ, 2 - a / (2 - i) = b * i) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l1282_128249


namespace NUMINAMATH_CALUDE_value_of_E_l1282_128282

/-- Given random integer values for letters of the alphabet, prove that E = 25 -/
theorem value_of_E (Z Q U I T E : ℤ) : Z = 15 → Q + U + I + Z = 60 → Q + U + I + E + T = 75 → Q + U + I + T = 50 → E = 25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_E_l1282_128282


namespace NUMINAMATH_CALUDE_sum_of_powers_of_five_squares_l1282_128268

theorem sum_of_powers_of_five_squares (m n : ℕ+) :
  (∃ a b : ℤ, (5 : ℤ)^(n : ℕ) + (5 : ℤ)^(m : ℕ) = a^2 + b^2) ↔ Even (n - m) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_five_squares_l1282_128268


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1282_128214

-- Define a geometric sequence with positive common ratio
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_cond : a 4 * a 8 = 2 * (a 5)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1282_128214


namespace NUMINAMATH_CALUDE_binary_subtraction_example_l1282_128230

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNum := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : ℕ) : BinaryNum :=
  sorry

/-- Converts a binary number to its natural number representation -/
def fromBinary (b : BinaryNum) : ℕ :=
  sorry

/-- Performs binary subtraction -/
def binarySubtract (a b : BinaryNum) : BinaryNum :=
  sorry

theorem binary_subtraction_example :
  binarySubtract (toBinary 27) (toBinary 5) = toBinary 22 :=
sorry

end NUMINAMATH_CALUDE_binary_subtraction_example_l1282_128230


namespace NUMINAMATH_CALUDE_lcm_12_18_l1282_128239

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l1282_128239


namespace NUMINAMATH_CALUDE_factorial_equality_l1282_128244

theorem factorial_equality : 5 * 8 * 2 * 63 = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equality_l1282_128244


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1282_128297

-- Define the polynomial
def polynomial (p : ℝ) (x : ℝ) : ℝ := 4 * x^3 - 12 * x^2 + p * x - 16

-- Define divisibility condition
def is_divisible_by (f : ℝ → ℝ) (a : ℝ) : Prop := f a = 0

-- Theorem statement
theorem polynomial_divisibility (p : ℝ) :
  (is_divisible_by (polynomial p) 2) →
  (is_divisible_by (polynomial p) 4 ↔ p = 16) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1282_128297


namespace NUMINAMATH_CALUDE_elephant_count_l1282_128200

/-- The number of elephants at We Preserve For Future park -/
def W : ℕ := 70

/-- The number of elephants at Gestures For Good park -/
def G : ℕ := 3 * W

/-- The total number of elephants in both parks -/
def total_elephants : ℕ := W + G

theorem elephant_count : total_elephants = 280 := by
  sorry

end NUMINAMATH_CALUDE_elephant_count_l1282_128200
