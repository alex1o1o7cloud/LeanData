import Mathlib

namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l2189_218949

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l2189_218949


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l2189_218929

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

/-- Circle C₂ with equation x² + y² - 6x - 8y + m = 0 -/
def C₂ (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + m = 0}

/-- Two circles are tangent if they intersect at exactly one point -/
def AreTangent (A B : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ A ∧ p ∈ B

/-- The main theorem: If C₁ and C₂ are tangent, then m = 9 -/
theorem tangent_circles_m_value :
  AreTangent C₁ (C₂ 9) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l2189_218929


namespace NUMINAMATH_CALUDE_inverse_sum_equals_negative_twelve_l2189_218924

-- Define the function f
def f (x : ℝ) : ℝ := x * |x| + 3 * x

-- State the theorem
theorem inverse_sum_equals_negative_twelve :
  ∃ (a b : ℝ), f a = 9 ∧ f b = -121 ∧ a + b = -12 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_negative_twelve_l2189_218924


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2189_218917

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 4

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2189_218917


namespace NUMINAMATH_CALUDE_jasons_military_career_l2189_218942

theorem jasons_military_career (join_age retire_age : ℕ) 
  (chief_to_master_chief_factor : ℚ) (additional_years : ℕ) :
  join_age = 18 →
  retire_age = 46 →
  chief_to_master_chief_factor = 1.25 →
  additional_years = 10 →
  ∃ (years_to_chief : ℕ),
    years_to_chief + (chief_to_master_chief_factor * years_to_chief) + additional_years = retire_age - join_age ∧
    years_to_chief = 8 := by
  sorry

end NUMINAMATH_CALUDE_jasons_military_career_l2189_218942


namespace NUMINAMATH_CALUDE_group_collection_proof_l2189_218990

/-- Calculates the total amount collected by a group of students, where each student
    contributes as many paise as there are members in the group. -/
def total_amount_collected (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Proves that a group of 85 students, each contributing as many paise as there are members,
    will collect a total of 72.25 rupees. -/
theorem group_collection_proof :
  total_amount_collected 85 = 72.25 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_proof_l2189_218990


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l2189_218902

/-- The number of players on the basketball team -/
def total_players : ℕ := 15

/-- The number of players in the starting lineup -/
def starting_lineup_size : ℕ := 6

/-- The number of predetermined players in the starting lineup -/
def predetermined_players : ℕ := 3

/-- The number of different possible starting lineups -/
def different_lineups : ℕ := 220

theorem basketball_lineup_combinations :
  Nat.choose (total_players - predetermined_players) (starting_lineup_size - predetermined_players) = different_lineups := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l2189_218902


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l2189_218951

theorem sqrt_fourth_power_equals_256 (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l2189_218951


namespace NUMINAMATH_CALUDE_cube_color_probability_l2189_218995

-- Define the colors
inductive Color
| Black
| White
| Gray

-- Define a cube as a list of 6 colors
def Cube := List Color

-- Function to check if a cube meets the conditions
def meetsConditions (cube : Cube) : Bool :=
  sorry

-- Probability of a specific color
def colorProb : ℚ := 1 / 3

-- Total number of possible cube colorings
def totalColorings : ℕ := 729

-- Number of colorings that meet the conditions
def validColorings : ℕ := 39

theorem cube_color_probability :
  (validColorings : ℚ) / totalColorings = 13 / 243 := by
  sorry

end NUMINAMATH_CALUDE_cube_color_probability_l2189_218995


namespace NUMINAMATH_CALUDE_probability_first_box_given_defective_l2189_218912

def box1_total : ℕ := 5
def box1_defective : ℕ := 2
def box2_total : ℕ := 10
def box2_defective : ℕ := 3

def prob_select_box1 : ℚ := 1/2
def prob_select_box2 : ℚ := 1/2

def prob_defective_given_box1 : ℚ := box1_defective / box1_total
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

theorem probability_first_box_given_defective :
  (prob_select_box1 * prob_defective_given_box1) /
  (prob_select_box1 * prob_defective_given_box1 + prob_select_box2 * prob_defective_given_box2) = 4/7 :=
by sorry

end NUMINAMATH_CALUDE_probability_first_box_given_defective_l2189_218912


namespace NUMINAMATH_CALUDE_vertex_locus_is_parabola_l2189_218906

/-- The locus of vertices of a family of parabolas forms another parabola -/
theorem vertex_locus_is_parabola (a c : ℝ) (ha : a > 0) (hc : c > 0) :
  ∃ (A B C : ℝ), A ≠ 0 ∧
    ∀ (x y : ℝ), (∃ t : ℝ, x = -t / (2 * a) ∧ y = c - t^2 / (4 * a)) ↔
      y = A * x^2 + B * x + C :=
by sorry

end NUMINAMATH_CALUDE_vertex_locus_is_parabola_l2189_218906


namespace NUMINAMATH_CALUDE_max_value_expression_l2189_218959

theorem max_value_expression (a b c d : ℝ) 
  (ha : -10.5 ≤ a ∧ a ≤ 10.5)
  (hb : -10.5 ≤ b ∧ b ≤ 10.5)
  (hc : -10.5 ≤ c ∧ c ≤ 10.5)
  (hd : -10.5 ≤ d ∧ d ≤ 10.5) :
  (∀ x y z w, -10.5 ≤ x ∧ x ≤ 10.5 → -10.5 ≤ y ∧ y ≤ 10.5 → 
              -10.5 ≤ z ∧ z ≤ 10.5 → -10.5 ≤ w ∧ w ≤ 10.5 →
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 462) ∧
  (∃ x y z w, -10.5 ≤ x ∧ x ≤ 10.5 ∧ -10.5 ≤ y ∧ y ≤ 10.5 ∧
              -10.5 ≤ z ∧ z ≤ 10.5 ∧ -10.5 ≤ w ∧ w ≤ 10.5 ∧
              x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 462) :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2189_218959


namespace NUMINAMATH_CALUDE_shelbys_scooter_problem_l2189_218927

/-- Shelby's scooter problem -/
theorem shelbys_scooter_problem 
  (speed_no_rain : ℝ) 
  (speed_rain : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : speed_no_rain = 25)
  (h2 : speed_rain = 15)
  (h3 : total_distance = 18)
  (h4 : total_time = 36)
  : ∃ (time_no_rain : ℝ), 
    time_no_rain = 6 ∧ 
    speed_no_rain * (time_no_rain / 60) + speed_rain * ((total_time - time_no_rain) / 60) = total_distance :=
by
  sorry


end NUMINAMATH_CALUDE_shelbys_scooter_problem_l2189_218927


namespace NUMINAMATH_CALUDE_basketball_score_l2189_218970

theorem basketball_score (three_pointers two_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * two_pointers) →
  (two_pointers = 2 * free_throws) →
  (3 * three_pointers + 2 * two_pointers + free_throws = 73) →
  free_throws = 8 := by
sorry

end NUMINAMATH_CALUDE_basketball_score_l2189_218970


namespace NUMINAMATH_CALUDE_expression_bounds_l2189_218991

theorem expression_bounds (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) (hd : 0 ≤ d ∧ d ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ∧
  Real.sqrt (a^2 + (2-b)^2) + Real.sqrt (b^2 + (2-c)^2) + 
    Real.sqrt (c^2 + (2-d)^2) + Real.sqrt (d^2 + (2-a)^2) ≤ 8 := by
  sorry


end NUMINAMATH_CALUDE_expression_bounds_l2189_218991


namespace NUMINAMATH_CALUDE_two_numbers_problem_l2189_218999

theorem two_numbers_problem (x y : ℝ) 
  (sum_condition : x + y = 15)
  (relation_condition : 3 * x = 5 * y - 11)
  (smaller_number : x = 7) :
  y = 8 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l2189_218999


namespace NUMINAMATH_CALUDE_evaluate_expression_l2189_218921

theorem evaluate_expression : 5^3 * 5^4 * 2 = 156250 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2189_218921


namespace NUMINAMATH_CALUDE_mathematics_competition_is_good_l2189_218964

theorem mathematics_competition_is_good :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    1000 * x₁ + y₁ = 2 * x₁ * y₁ ∧
    1000 * x₂ + y₂ = 2 * x₂ * y₂ ∧
    1000 * x₁ + y₁ = 13520 ∧
    1000 * x₂ + y₂ = 63504 :=
by sorry

end NUMINAMATH_CALUDE_mathematics_competition_is_good_l2189_218964


namespace NUMINAMATH_CALUDE_power_equation_solution_l2189_218925

theorem power_equation_solution (m : ℤ) : (7 : ℝ) ^ (2 * m) = (1 / 7 : ℝ) ^ (m - 30) → m = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2189_218925


namespace NUMINAMATH_CALUDE_basketball_handshakes_l2189_218996

/-- Represents the number of handshakes in a basketball game --/
def total_handshakes (players_per_team : ℕ) (num_teams : ℕ) (num_referees : ℕ) : ℕ :=
  let total_players := players_per_team * num_teams
  let player_handshakes := players_per_team * players_per_team
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem stating the total number of handshakes in the given scenario --/
theorem basketball_handshakes :
  total_handshakes 6 2 3 = 72 := by
  sorry

#eval total_handshakes 6 2 3

end NUMINAMATH_CALUDE_basketball_handshakes_l2189_218996


namespace NUMINAMATH_CALUDE_a_7_equals_63_l2189_218941

def sequence_a : ℕ → ℚ
  | 0 => 0  -- We define a₀ and a₁ arbitrarily as they are not used
  | 1 => 0
  | 2 => 1
  | 3 => 3
  | (n + 1) => (sequence_a n ^ 2 - sequence_a (n - 1) + 2 * sequence_a n) / (sequence_a (n - 1) + 1)

theorem a_7_equals_63 : sequence_a 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_a_7_equals_63_l2189_218941


namespace NUMINAMATH_CALUDE_rachels_mystery_book_shelves_l2189_218943

theorem rachels_mystery_book_shelves 
  (books_per_shelf : ℕ) 
  (picture_book_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : books_per_shelf = 9)
  (h2 : picture_book_shelves = 2)
  (h3 : total_books = 72) :
  (total_books - picture_book_shelves * books_per_shelf) / books_per_shelf = 6 := by
  sorry

end NUMINAMATH_CALUDE_rachels_mystery_book_shelves_l2189_218943


namespace NUMINAMATH_CALUDE_kenny_lawn_mowing_l2189_218984

theorem kenny_lawn_mowing (cost_per_lawn : ℕ) (cost_per_game : ℕ) (cost_per_book : ℕ)
  (num_games : ℕ) (num_books : ℕ) 
  (h1 : cost_per_lawn = 15)
  (h2 : cost_per_game = 45)
  (h3 : cost_per_book = 5)
  (h4 : num_games = 5)
  (h5 : num_books = 60) :
  cost_per_lawn * 35 = cost_per_game * num_games + cost_per_book * num_books :=
by sorry

end NUMINAMATH_CALUDE_kenny_lawn_mowing_l2189_218984


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l2189_218904

theorem unique_solution_modular_equation :
  ∃! n : ℕ, n < 103 ∧ (100 * n) % 103 = 65 % 103 ∧ n = 68 := by sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l2189_218904


namespace NUMINAMATH_CALUDE_jeans_sold_proof_l2189_218983

/-- The number of pairs of jeans sold by a clothing store -/
def num_jeans : ℕ := 10

theorem jeans_sold_proof (shirts : ℕ) (shirt_price : ℕ) (jeans_price : ℕ) (total_revenue : ℕ) :
  shirts = 20 →
  shirt_price = 10 →
  jeans_price = 2 * shirt_price →
  total_revenue = 400 →
  shirts * shirt_price + num_jeans * jeans_price = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_jeans_sold_proof_l2189_218983


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2189_218903

theorem pure_imaginary_complex_number (x : ℝ) :
  (((x^2 - 2*x - 3) : ℂ) + (x + 1)*I).re = 0 ∧ (((x^2 - 2*x - 3) : ℂ) + (x + 1)*I).im ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2189_218903


namespace NUMINAMATH_CALUDE_seed_selection_correct_l2189_218937

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is a valid seed number --/
def isValidSeed (n : Nat) : Bool :=
  0 < n && n ≤ 500

/-- Extracts the next three-digit number from the random number table --/
def nextThreeDigitNumber (table : RandomNumberTable) (row : Nat) (col : Nat) : Option Nat :=
  sorry

/-- Selects the first n valid seeds from the random number table --/
def selectValidSeeds (table : RandomNumberTable) (startRow : Nat) (startCol : Nat) (n : Nat) : List Nat :=
  sorry

/-- The given random number table --/
def givenTable : RandomNumberTable := [
  [84, 42, 17, 53, 31, 57, 24, 55, 06, 88, 77, 04, 74, 47, 67],
  [21, 76, 33, 50, 25, 83, 92, 12, 06, 76],
  [63, 01, 63, 78, 59, 16, 95, 55, 67, 19, 98, 10, 50, 71, 75],
  [12, 86, 73, 58, 07, 44, 39, 52, 38, 79],
  [33, 21, 12, 34, 29, 78, 64, 56, 07, 82, 52, 42, 07, 44, 38],
  [15, 51, 00, 13, 42, 99, 66, 02, 79, 54]
]

theorem seed_selection_correct :
  selectValidSeeds givenTable 7 8 5 = [331, 455, 068, 047, 447] :=
sorry

end NUMINAMATH_CALUDE_seed_selection_correct_l2189_218937


namespace NUMINAMATH_CALUDE_investment_problem_l2189_218947

/-- Investment problem -/
theorem investment_problem (x y : ℝ) 
  (h1 : 0.06 * x = 0.05 * y + 160)  -- Income difference condition
  (h2 : 0.05 * y = 6000)            -- Income from 5% part
  : x + y = 222666.67 := by          -- Total investment
sorry

end NUMINAMATH_CALUDE_investment_problem_l2189_218947


namespace NUMINAMATH_CALUDE_beyonce_song_count_l2189_218998

/-- The number of singles released by Beyonce -/
def singles : Nat := 5

/-- The number of albums with 15 songs -/
def albums_15 : Nat := 2

/-- The number of songs in each of the albums_15 -/
def songs_per_album_15 : Nat := 15

/-- The number of albums with 20 songs -/
def albums_20 : Nat := 1

/-- The number of songs in each of the albums_20 -/
def songs_per_album_20 : Nat := 20

/-- The total number of songs released by Beyonce -/
def total_songs : Nat := singles + albums_15 * songs_per_album_15 + albums_20 * songs_per_album_20

theorem beyonce_song_count : total_songs = 55 := by
  sorry

end NUMINAMATH_CALUDE_beyonce_song_count_l2189_218998


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2189_218966

theorem inequality_solution_set (x : ℝ) : 
  (|x + 1| - 2 > 0) ↔ (x < -3 ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2189_218966


namespace NUMINAMATH_CALUDE_eleven_students_in_line_l2189_218948

/-- The number of students in a line, given Yoonjung's position -/
def total_students (students_in_front : ℕ) (position_from_back : ℕ) : ℕ :=
  students_in_front + 1 + (position_from_back - 1)

/-- Theorem: There are 11 students in the line -/
theorem eleven_students_in_line : 
  total_students 6 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_students_in_line_l2189_218948


namespace NUMINAMATH_CALUDE_f_satisfies_equation_l2189_218908

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x

-- State the theorem
theorem f_satisfies_equation : ∀ x : ℝ, 2 * (f x) + f (-x) = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_equation_l2189_218908


namespace NUMINAMATH_CALUDE_difference_of_a_and_reciprocal_l2189_218972

theorem difference_of_a_and_reciprocal (a : ℝ) (h : a + 1/a = Real.sqrt 13) :
  a - 1/a = 3 ∨ a - 1/a = -3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_a_and_reciprocal_l2189_218972


namespace NUMINAMATH_CALUDE_parabola_equation_l2189_218920

/-- A parabola with the same shape as y = -5x^2 + 2 and vertex at (4, -2) -/
structure Parabola where
  /-- The coefficient of x^2 in the parabola equation -/
  a : ℝ
  /-- The x-coordinate of the vertex -/
  h : ℝ
  /-- The y-coordinate of the vertex -/
  k : ℝ
  /-- The parabola has the same shape as y = -5x^2 + 2 -/
  shape_cond : a = -5
  /-- The vertex is at (4, -2) -/
  vertex_cond : h = 4 ∧ k = -2

/-- The analytical expression of the parabola -/
def parabola_expression (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - p.h)^2 + p.k

theorem parabola_equation (p : Parabola) :
  ∀ x, parabola_expression p x = -5 * (x - 4)^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_equation_l2189_218920


namespace NUMINAMATH_CALUDE_complement_A_U_equality_l2189_218977

-- Define the universal set U
def U : Set ℝ := {x | -3 < x ∧ x < 3}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}

-- Define the complement of A with respect to U
def complement_A_U : Set ℝ := U \ A

-- Theorem statement
theorem complement_A_U_equality :
  complement_A_U = {x | (-3 < x ∧ x ≤ -2) ∨ (1 < x ∧ x < 3)} := by
  sorry

end NUMINAMATH_CALUDE_complement_A_U_equality_l2189_218977


namespace NUMINAMATH_CALUDE_sara_movie_tickets_l2189_218974

/-- The number of movie theater tickets Sara bought -/
def num_tickets : ℕ := 2

/-- The cost of each movie theater ticket in cents -/
def ticket_cost : ℕ := 1062

/-- The cost of renting a movie in cents -/
def rental_cost : ℕ := 159

/-- The cost of buying a movie in cents -/
def purchase_cost : ℕ := 1395

/-- The total amount Sara spent in cents -/
def total_spent : ℕ := 3678

/-- Theorem stating that the number of tickets Sara bought is correct -/
theorem sara_movie_tickets : 
  num_tickets * ticket_cost + rental_cost + purchase_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_sara_movie_tickets_l2189_218974


namespace NUMINAMATH_CALUDE_alberto_bjorn_bike_distance_l2189_218939

/-- The problem of comparing distances biked by Alberto and Bjorn -/
theorem alberto_bjorn_bike_distance :
  let alberto_rate : ℝ := 80 / 4  -- Alberto's constant rate in miles per hour
  let alberto_time : ℝ := 4  -- Alberto's total time in hours
  let bjorn_rate1 : ℝ := 20  -- Bjorn's first rate in miles per hour
  let bjorn_rate2 : ℝ := 25  -- Bjorn's second rate in miles per hour
  let bjorn_time1 : ℝ := 2  -- Bjorn's time at first rate in hours
  let bjorn_time2 : ℝ := 2  -- Bjorn's time at second rate in hours
  
  let alberto_distance : ℝ := alberto_rate * alberto_time
  let bjorn_distance : ℝ := bjorn_rate1 * bjorn_time1 + bjorn_rate2 * bjorn_time2
  
  alberto_distance - bjorn_distance = -10
  := by sorry

end NUMINAMATH_CALUDE_alberto_bjorn_bike_distance_l2189_218939


namespace NUMINAMATH_CALUDE_pirate_treasure_division_l2189_218985

theorem pirate_treasure_division (N : ℕ) : 
  220 ≤ N ∧ N ≤ 300 →
  let first_take := 2 + (N - 2) / 3
  let remain_after_first := N - first_take
  let second_take := 2 + (remain_after_first - 2) / 3
  let remain_after_second := remain_after_first - second_take
  let third_take := 2 + (remain_after_second - 2) / 3
  let final_remain := remain_after_second - third_take
  final_remain % 3 = 0 →
  first_take = 84 ∧ 
  second_take = 54 ∧ 
  third_take = 54 ∧
  final_remain / 3 = 54 := by
sorry


end NUMINAMATH_CALUDE_pirate_treasure_division_l2189_218985


namespace NUMINAMATH_CALUDE_reciprocal_magnitude_of_one_minus_i_l2189_218971

open Complex

theorem reciprocal_magnitude_of_one_minus_i :
  let z : ℂ := 1 - I
  abs (z⁻¹) = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_magnitude_of_one_minus_i_l2189_218971


namespace NUMINAMATH_CALUDE_determinant_equals_negative_two_l2189_218930

-- Define the polynomial and its roots
def polynomial (p q : ℝ) (x : ℝ) : ℝ := x^3 - 3*p*x^2 + q*x - 2

-- Define the roots of the polynomial
def roots (p q : ℝ) : Set ℝ := {x | polynomial p q x = 0}

-- Assume the polynomial has exactly three roots
axiom three_roots (p q : ℝ) : ∃ (a b c : ℝ), roots p q = {a, b, c}

-- Define the determinant
def determinant (r a b c : ℝ) : ℝ :=
  (r + a) * ((r + b) * (r + c) - r^2) -
  r * (r * (r + c) - r^2) +
  r * (r * (r + b) - r^2)

-- State the theorem
theorem determinant_equals_negative_two (p q r : ℝ) :
  ∃ (a b c : ℝ), roots p q = {a, b, c} ∧ determinant r a b c = -2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equals_negative_two_l2189_218930


namespace NUMINAMATH_CALUDE_overlapping_sticks_length_l2189_218907

/-- The total length of overlapping wooden sticks -/
def total_length (n : ℕ) (stick_length overlap : ℝ) : ℝ :=
  stick_length + (n - 1) * (stick_length - overlap)

/-- Theorem: The total length of 30 wooden sticks, each 25 cm long, 
    when overlapped by 6 cm, is equal to 576 cm -/
theorem overlapping_sticks_length :
  total_length 30 25 6 = 576 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_sticks_length_l2189_218907


namespace NUMINAMATH_CALUDE_circle_triangle_problem_l2189_218938

/-- Given that a triangle equals three circles and a triangle plus a circle equals 40,
    prove that the circle equals 10 and the triangle equals 30. -/
theorem circle_triangle_problem (circle triangle : ℕ) 
    (h1 : triangle = 3 * circle)
    (h2 : triangle + circle = 40) :
    circle = 10 ∧ triangle = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_problem_l2189_218938


namespace NUMINAMATH_CALUDE_product_not_exceeding_sum_l2189_218997

theorem product_not_exceeding_sum (x y : ℕ) (h : x * y ≤ x + y) :
  (x = 1 ∧ y ≥ 1) ∨ (x = 2 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_product_not_exceeding_sum_l2189_218997


namespace NUMINAMATH_CALUDE_fraction_chain_l2189_218944

theorem fraction_chain (a b c d e : ℝ) 
  (h1 : a/b = 5)
  (h2 : b/c = 1/2)
  (h3 : c/d = 4)
  (h4 : d/e = 1/3)
  : e/a = 3/10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_chain_l2189_218944


namespace NUMINAMATH_CALUDE_square_condition_l2189_218905

theorem square_condition (n : ℕ) : 
  (∃ k : ℕ, (n^3 + 39*n - 2)*n.factorial + 17*21^n + 5 = k^2) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_square_condition_l2189_218905


namespace NUMINAMATH_CALUDE_prob_three_red_is_one_fifty_fifth_l2189_218963

/-- The number of red balls in the bag -/
def red_balls : ℕ := 4

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 5

/-- The number of green balls in the bag -/
def green_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + blue_balls + green_balls

/-- The number of balls to be picked -/
def picked_balls : ℕ := 3

/-- The probability of picking 3 red balls when randomly selecting 3 balls without replacement -/
def prob_three_red : ℚ := (red_balls * (red_balls - 1) * (red_balls - 2)) / 
  (total_balls * (total_balls - 1) * (total_balls - 2))

theorem prob_three_red_is_one_fifty_fifth : prob_three_red = 1 / 55 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_red_is_one_fifty_fifth_l2189_218963


namespace NUMINAMATH_CALUDE_sniper_B_wins_l2189_218926

/-- Represents the probabilities of scoring 1, 2, and 3 points for a sniper -/
structure SniperProbabilities where
  one : Real
  two : Real
  three : Real
  sum_to_one : one + two + three = 1
  non_negative : one ≥ 0 ∧ two ≥ 0 ∧ three ≥ 0

/-- Calculates the expected score for a sniper given their probabilities -/
def expectedScore (p : SniperProbabilities) : Real :=
  1 * p.one + 2 * p.two + 3 * p.three

/-- Sniper A's probabilities -/
def sniperA : SniperProbabilities where
  one := 0.4
  two := 0.1
  three := 0.5
  sum_to_one := by sorry
  non_negative := by sorry

/-- Sniper B's probabilities -/
def sniperB : SniperProbabilities where
  one := 0.1
  two := 0.6
  three := 0.3
  sum_to_one := by sorry
  non_negative := by sorry

/-- Theorem stating that Sniper B has a higher expected score than Sniper A -/
theorem sniper_B_wins : expectedScore sniperB > expectedScore sniperA := by
  sorry

end NUMINAMATH_CALUDE_sniper_B_wins_l2189_218926


namespace NUMINAMATH_CALUDE_window_width_is_30_l2189_218936

/-- Represents the width of a pane of glass in the window -/
def pane_width : ℝ := 6

/-- Represents the height of a pane of glass in the window -/
def pane_height : ℝ := 3 * pane_width

/-- Represents the width of the borders around and between panes -/
def border_width : ℝ := 3

/-- Represents the number of columns of panes in the window -/
def num_columns : ℕ := 3

/-- Represents the number of rows of panes in the window -/
def num_rows : ℕ := 2

/-- Calculates the total width of the window -/
def window_width : ℝ := num_columns * pane_width + (num_columns + 1) * border_width

/-- Theorem stating that the width of the rectangular window is 30 inches -/
theorem window_width_is_30 : window_width = 30 := by sorry

end NUMINAMATH_CALUDE_window_width_is_30_l2189_218936


namespace NUMINAMATH_CALUDE_harry_stamps_l2189_218958

theorem harry_stamps (total : ℕ) (harry_ratio : ℕ) (harry_stamps : ℕ) : 
  total = 240 →
  harry_ratio = 3 →
  harry_stamps = total * harry_ratio / (harry_ratio + 1) →
  harry_stamps = 180 := by
sorry

end NUMINAMATH_CALUDE_harry_stamps_l2189_218958


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2189_218968

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
    a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10 + a₁₁*(x-1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2189_218968


namespace NUMINAMATH_CALUDE_prob_three_same_color_l2189_218934

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 4

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

/-- The probability of drawing three marbles of the same color without replacement -/
theorem prob_three_same_color : 
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) + 
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) / 
  (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l2189_218934


namespace NUMINAMATH_CALUDE_problem_statement_l2189_218918

theorem problem_statement (x y : ℝ) (h : x + 2*y - 1 = 0) : 3 + 2*x + 4*y = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2189_218918


namespace NUMINAMATH_CALUDE_decimal_point_problem_l2189_218935

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_decimal_point_problem_l2189_218935


namespace NUMINAMATH_CALUDE_balloon_difference_l2189_218988

def your_balloons : ℕ := 7
def friend_balloons : ℕ := 5

theorem balloon_difference : your_balloons - friend_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_balloon_difference_l2189_218988


namespace NUMINAMATH_CALUDE_function_derivative_problem_l2189_218900

theorem function_derivative_problem (a : ℝ) :
  (∀ x, f x = (2 * x + a) ^ 2) →
  (deriv f) 2 = 20 →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_derivative_problem_l2189_218900


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_max_area_is_5_exists_triangle_with_max_area_l2189_218915

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a = t.b + t.c - 2

theorem angle_A_is_60_degrees (t : Triangle) 
  (h : triangle_condition t) : t.A = 60 := by sorry

theorem max_area_is_5 (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2) : 
  ∀ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A → s ≤ 5 := by sorry

theorem exists_triangle_with_max_area (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2) : 
  ∃ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A ∧ s = 5 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_max_area_is_5_exists_triangle_with_max_area_l2189_218915


namespace NUMINAMATH_CALUDE_least_addition_to_perfect_square_l2189_218909

theorem least_addition_to_perfect_square : ∃ (x : ℝ), 
  (x ≥ 0) ∧ 
  (∃ (n : ℕ), (0.0320 + x) = n^2) ∧
  (∀ (y : ℝ), y ≥ 0 → (∃ (m : ℕ), (0.0320 + y) = m^2) → y ≥ x) ∧
  (x = 0.9680) := by
sorry

end NUMINAMATH_CALUDE_least_addition_to_perfect_square_l2189_218909


namespace NUMINAMATH_CALUDE_joe_taller_than_roy_l2189_218989

/-- Given the heights of Sara and Roy, and the difference between Sara and Joe's heights,
    prove that Joe is 3 inches taller than Roy. -/
theorem joe_taller_than_roy (sara_height joe_height roy_height : ℕ)
  (h1 : sara_height = 45)
  (h2 : sara_height = joe_height + 6)
  (h3 : roy_height = 36) :
  joe_height - roy_height = 3 :=
by sorry

end NUMINAMATH_CALUDE_joe_taller_than_roy_l2189_218989


namespace NUMINAMATH_CALUDE_bird_count_theorem_l2189_218913

theorem bird_count_theorem (initial_parrots : ℕ) (remaining_parrots : ℕ) 
  (remaining_crows : ℕ) (remaining_pigeons : ℕ) : 
  initial_parrots = 15 →
  remaining_parrots = 5 →
  remaining_crows = 3 →
  remaining_pigeons = 2 →
  ∃ (flew_away : ℕ), 
    flew_away = initial_parrots - remaining_parrots ∧
    initial_parrots + (flew_away + remaining_crows) + (flew_away + remaining_pigeons) = 40 :=
by sorry

end NUMINAMATH_CALUDE_bird_count_theorem_l2189_218913


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2189_218962

/-- The lateral surface area of a cylinder with base radius 2 and generatrix length 3 is 12π. -/
theorem cylinder_lateral_surface_area :
  ∀ (r g : ℝ), r = 2 → g = 3 → 2 * π * r * g = 12 * π :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l2189_218962


namespace NUMINAMATH_CALUDE_quadratic_roots_l2189_218914

/-- A quadratic function f(x) = x^2 - px + q -/
def f (p q x : ℝ) : ℝ := x^2 - p*x + q

/-- Theorem: If f(p + q) = 0 and f(p - q) = 0, then either q = 0 (and p can be any value) or (p, q) = (0, -1) -/
theorem quadratic_roots (p q : ℝ) : 
  f p q (p + q) = 0 ∧ f p q (p - q) = 0 → 
  (q = 0 ∨ (p = 0 ∧ q = -1)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l2189_218914


namespace NUMINAMATH_CALUDE_exactly_five_cheaper_points_l2189_218916

-- Define the cost function
def C (n : ℕ) : ℕ :=
  if n ≥ 50 then 13 * n
  else if n ≥ 20 then 14 * n
  else 15 * n

-- Define the property we want to prove
def cheaper_to_buy_more (n : ℕ) : Prop :=
  C (n + 1) < C n

-- Theorem statement
theorem exactly_five_cheaper_points :
  ∃ (S : Finset ℕ), S.card = 5 ∧ 
  (∀ n, n ∈ S ↔ cheaper_to_buy_more n) :=
sorry

end NUMINAMATH_CALUDE_exactly_five_cheaper_points_l2189_218916


namespace NUMINAMATH_CALUDE_truth_teller_liar_arrangement_l2189_218965

def is_valid_arrangement (n k : ℕ) : Prop :=
  n > 0 ∧ k > 0 ∧ k < n ∧ 
  ∃ (m : ℕ), 2^m * k < n ∧ n ≤ 2^(m+1) * k

theorem truth_teller_liar_arrangement (n k : ℕ) :
  is_valid_arrangement n k →
  ∃ (m : ℕ), n = 2^m * (n.gcd k) ∧ 2^m > (k / (n.gcd k)) :=
sorry

end NUMINAMATH_CALUDE_truth_teller_liar_arrangement_l2189_218965


namespace NUMINAMATH_CALUDE_tan_inequality_equiv_l2189_218946

theorem tan_inequality_equiv (x : ℝ) : 
  Real.tan (2 * x - π / 4) ≤ 1 ↔ 
  ∃ k : ℤ, k * π / 2 - π / 8 < x ∧ x ≤ k * π / 2 + π / 4 := by
sorry

end NUMINAMATH_CALUDE_tan_inequality_equiv_l2189_218946


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2189_218981

/-- A function to check if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- Theorem stating that among the given sets, only (2, 5, 6) cannot form a right triangle --/
theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  is_right_triangle 2 2 (2 * Real.sqrt 2) ∧
  ¬is_right_triangle 2 5 6 ∧
  is_right_triangle 5 12 13 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2189_218981


namespace NUMINAMATH_CALUDE_dons_walking_speed_l2189_218932

/-- Proof of Don's walking speed given the conditions of Cara and Don's walk --/
theorem dons_walking_speed
  (total_distance : ℝ)
  (caras_speed : ℝ)
  (caras_distance : ℝ)
  (don_delay : ℝ)
  (h1 : total_distance = 45)
  (h2 : caras_speed = 6)
  (h3 : caras_distance = 30)
  (h4 : don_delay = 2) :
  ∃ (dons_speed : ℝ), dons_speed = 5 := by
  sorry

end NUMINAMATH_CALUDE_dons_walking_speed_l2189_218932


namespace NUMINAMATH_CALUDE_tan_beta_value_l2189_218969

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end NUMINAMATH_CALUDE_tan_beta_value_l2189_218969


namespace NUMINAMATH_CALUDE_determine_hidden_numbers_l2189_218955

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Given two sums S1 and S2, it is possible to determine the original numbers a, b, and c -/
theorem determine_hidden_numbers (a b c : ℕ) :
  let k := num_digits (a + b + c)
  let S1 := a + b + c
  let S2 := a + b * 10^k + c * 10^(2*k)
  ∃! (a' b' c' : ℕ), S1 = a' + b' + c' ∧ S2 = a' + b' * 10^k + c' * 10^(2*k) ∧ a' = a ∧ b' = b ∧ c' = c :=
by sorry

end NUMINAMATH_CALUDE_determine_hidden_numbers_l2189_218955


namespace NUMINAMATH_CALUDE_fiftyEighthDigitOfOneSeventh_l2189_218986

/-- The decimal representation of 1/7 as a sequence of digits -/
def oneSeventhDecimal : ℕ → ℕ
| 0 => 1
| 1 => 4
| 2 => 2
| 3 => 8
| 4 => 5
| 5 => 7
| n + 6 => oneSeventhDecimal n

/-- The 58th digit after the decimal point in the decimal representation of 1/7 is 8 -/
theorem fiftyEighthDigitOfOneSeventh : oneSeventhDecimal 57 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fiftyEighthDigitOfOneSeventh_l2189_218986


namespace NUMINAMATH_CALUDE_marbles_distribution_l2189_218987

theorem marbles_distribution (marbles_per_class : ℕ) (num_classes : ℕ) (leftover_marbles : ℕ)
  (h1 : marbles_per_class = 37)
  (h2 : num_classes = 23)
  (h3 : leftover_marbles = 16) :
  marbles_per_class * num_classes + leftover_marbles = 867 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l2189_218987


namespace NUMINAMATH_CALUDE_total_amount_proof_l2189_218994

def coffee_maker_price : ℝ := 70
def blender_price : ℝ := 100
def coffee_maker_discount : ℝ := 0.2
def blender_discount : ℝ := 0.15
def num_coffee_makers : ℕ := 2

def total_price : ℝ :=
  (num_coffee_makers : ℝ) * coffee_maker_price * (1 - coffee_maker_discount) +
  blender_price * (1 - blender_discount)

theorem total_amount_proof :
  total_price = 197 := by sorry

end NUMINAMATH_CALUDE_total_amount_proof_l2189_218994


namespace NUMINAMATH_CALUDE_vertex_of_parabola_l2189_218922

/-- The function f(x) = 3(x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := 3 * (x - 1)^2 + 2

/-- The vertex of the parabola defined by f -/
def vertex : ℝ × ℝ := (1, 2)

theorem vertex_of_parabola :
  ∀ x : ℝ, f x ≥ f (vertex.1) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_parabola_l2189_218922


namespace NUMINAMATH_CALUDE_range_of_m_l2189_218973

theorem range_of_m (x y m : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h_eq : 2/x + 1/y = 1) 
  (h_ineq : x + 2*y > m^2 - 2*m) : 
  -2 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l2189_218973


namespace NUMINAMATH_CALUDE_stripe_ratio_l2189_218953

/-- Given the conditions about stripes on tennis shoes, prove the ratio of Hortense's to Olga's stripes -/
theorem stripe_ratio (olga_stripes_per_shoe : ℕ) (rick_stripes_per_shoe : ℕ) (total_stripes : ℕ)
  (h1 : olga_stripes_per_shoe = 3)
  (h2 : rick_stripes_per_shoe = olga_stripes_per_shoe - 1)
  (h3 : total_stripes = 22)
  (h4 : total_stripes = 2 * olga_stripes_per_shoe + 2 * rick_stripes_per_shoe + hortense_stripes) :
  hortense_stripes / (2 * olga_stripes_per_shoe) = 2 :=
by sorry

end NUMINAMATH_CALUDE_stripe_ratio_l2189_218953


namespace NUMINAMATH_CALUDE_symmetric_point_of_2_5_l2189_218923

/-- Given a point P(a,b) and a line with equation x+y=0, 
    the symmetric point Q(x,y) satisfies:
    1. x + y = 0 (lies on the line)
    2. The midpoint of PQ lies on the line
    3. PQ is perpendicular to the line -/
def is_symmetric_point (a b x y : ℝ) : Prop :=
  x + y = 0 ∧
  (a + x) / 2 + (b + y) / 2 = 0 ∧
  (x - a) = (b - y)

/-- The point symmetric to P(2,5) with respect to the line x+y=0 
    has coordinates (-5,-2) -/
theorem symmetric_point_of_2_5 : 
  is_symmetric_point 2 5 (-5) (-2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_of_2_5_l2189_218923


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2189_218940

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- General term of the sequence
  S : ℕ → ℝ  -- Sum of the first n terms
  b : ℕ → ℝ  -- Related sequence
  h1 : a 3 = 10
  h2 : S 6 = 72
  h3 : ∀ n, b n = (1/2) * a n - 30

/-- The minimum value of the sum of the first n terms of b_n -/
def T_min (seq : ArithmeticSequence) : ℝ :=
  Finset.sum (Finset.range 15) (λ i => seq.b (i + 1))

/-- Main theorem about the arithmetic sequence and its properties -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 4 * n - 2) ∧ T_min seq = -225 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2189_218940


namespace NUMINAMATH_CALUDE_ellipse_focal_length_2_implies_a_5_l2189_218956

/-- Represents an ellipse with equation x^2/a + y^2 = 1 -/
structure Ellipse where
  a : ℝ
  h_a_gt_one : a > 1

/-- The focal length of an ellipse -/
def focal_length (e : Ellipse) : ℝ := sorry

/-- Theorem: If the focal length of the ellipse is 2, then a = 5 -/
theorem ellipse_focal_length_2_implies_a_5 (e : Ellipse) 
  (h : focal_length e = 2) : e.a = 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_2_implies_a_5_l2189_218956


namespace NUMINAMATH_CALUDE_intersection_theorem_l2189_218960

/-- A permutation of {1, ..., n} is a bijective function from {1, ..., n} to itself. -/
def Permutation (n : ℕ) := {f : Fin n → Fin n // Function.Bijective f}

/-- Two permutations intersect if they have the same value at some position. -/
def intersect {n : ℕ} (p q : Permutation n) : Prop :=
  ∃ k : Fin n, p.val k = q.val k

/-- There exists a set of 1006 permutations of {1, ..., 2010} such that 
    any permutation of {1, ..., 2010} intersects with at least one of them. -/
theorem intersection_theorem : 
  ∃ (S : Finset (Permutation 2010)), S.card = 1006 ∧ 
    ∀ p : Permutation 2010, ∃ q ∈ S, intersect p q := by
  sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2189_218960


namespace NUMINAMATH_CALUDE_units_digit_periodicity_l2189_218910

theorem units_digit_periodicity (k : ℕ) : 
  (k * (k + 1) * (k + 2)) % 10 = ((k + 10) * (k + 11) * (k + 12)) % 10 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_periodicity_l2189_218910


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2189_218979

theorem rationalize_denominator : 
  (Real.sqrt 18 + Real.sqrt 8) / (Real.sqrt 12 + Real.sqrt 8) = 2.5 * Real.sqrt 6 - 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2189_218979


namespace NUMINAMATH_CALUDE_jamie_calculation_l2189_218957

theorem jamie_calculation (x : ℝ) : (x / 8 + 20 = 28) → (x * 8 - 20 = 492) := by
  sorry

end NUMINAMATH_CALUDE_jamie_calculation_l2189_218957


namespace NUMINAMATH_CALUDE_penny_count_l2189_218933

/-- Proves that given 4 nickels, 3 dimes, and a total value of $0.59, the number of pennies is 9 -/
theorem penny_count (nickels : ℕ) (dimes : ℕ) (total_cents : ℕ) (pennies : ℕ) : 
  nickels = 4 → 
  dimes = 3 → 
  total_cents = 59 → 
  5 * nickels + 10 * dimes + pennies = total_cents → 
  pennies = 9 := by
sorry

end NUMINAMATH_CALUDE_penny_count_l2189_218933


namespace NUMINAMATH_CALUDE_price_increase_problem_l2189_218978

theorem price_increase_problem (candy_new : ℝ) (soda_new : ℝ) (chips_new : ℝ) (chocolate_new : ℝ)
  (candy_increase : ℝ) (soda_increase : ℝ) (chips_increase : ℝ) (chocolate_increase : ℝ)
  (h_candy : candy_new = 10) (h_soda : soda_new = 6) (h_chips : chips_new = 4) (h_chocolate : chocolate_new = 2)
  (h_candy_inc : candy_increase = 0.25) (h_soda_inc : soda_increase = 0.5)
  (h_chips_inc : chips_increase = 0.4) (h_chocolate_inc : chocolate_increase = 0.75) :
  (candy_new / (1 + candy_increase)) + (soda_new / (1 + soda_increase)) +
  (chips_new / (1 + chips_increase)) + (chocolate_new / (1 + chocolate_increase)) = 16 :=
by sorry

end NUMINAMATH_CALUDE_price_increase_problem_l2189_218978


namespace NUMINAMATH_CALUDE_decomposition_fifth_power_fourth_l2189_218993

-- Define the function that gives the starting odd number for m^n
def startOdd (m n : ℕ) : ℕ := 
  2 * (m - 1) * (n - 1) + 1

-- Define the function that gives the k-th odd number in the sequence
def kthOdd (start k : ℕ) : ℕ := 
  start + 2 * (k - 1)

-- Theorem statement
theorem decomposition_fifth_power_fourth (m : ℕ) (h : m = 5) : 
  kthOdd (startOdd m 4) 3 = 125 := by
sorry

end NUMINAMATH_CALUDE_decomposition_fifth_power_fourth_l2189_218993


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2189_218928

theorem absolute_value_inequality (x : ℝ) : 
  |x - 1| + |x - 2| > 5 ↔ x < -1 ∨ x > 4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2189_218928


namespace NUMINAMATH_CALUDE_optimal_transportation_plan_l2189_218954

structure VehicleType where
  capacity : ℕ
  cost : ℕ

def total_supply : ℕ := 120

def vehicle_a : VehicleType := ⟨5, 300⟩
def vehicle_b : VehicleType := ⟨8, 400⟩
def vehicle_c : VehicleType := ⟨10, 500⟩

def total_vehicles : ℕ := 18

def transportation_plan (a b c : ℕ) : Prop :=
  a + b + c = total_vehicles ∧
  a * vehicle_a.capacity + b * vehicle_b.capacity + c * vehicle_c.capacity ≥ total_supply

def transportation_cost (a b c : ℕ) : ℕ :=
  a * vehicle_a.cost + b * vehicle_b.cost + c * vehicle_c.cost

theorem optimal_transportation_plan :
  ∀ (a b c : ℕ),
    transportation_plan a b c →
    transportation_cost a b c ≥ transportation_cost 8 10 0 :=
by sorry

end NUMINAMATH_CALUDE_optimal_transportation_plan_l2189_218954


namespace NUMINAMATH_CALUDE_team_selection_count_l2189_218919

def total_students : ℕ := 11
def num_girls : ℕ := 3
def num_boys : ℕ := 8
def team_size : ℕ := 5

theorem team_selection_count :
  (Nat.choose total_students team_size) - (Nat.choose num_boys team_size) = 406 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l2189_218919


namespace NUMINAMATH_CALUDE_max_salary_basketball_team_l2189_218961

/-- Represents the maximum possible salary for a single player in a basketball team. -/
def maxSalary (numPlayers : ℕ) (minSalary : ℕ) (totalSalaryCap : ℕ) : ℕ :=
  totalSalaryCap - (numPlayers - 1) * minSalary

/-- Theorem stating the maximum possible salary for a single player
    given the team composition and salary constraints. -/
theorem max_salary_basketball_team :
  maxSalary 12 20000 500000 = 280000 := by
  sorry

#eval maxSalary 12 20000 500000

end NUMINAMATH_CALUDE_max_salary_basketball_team_l2189_218961


namespace NUMINAMATH_CALUDE_root_bounds_l2189_218967

theorem root_bounds (b c x : ℝ) (hb : 5.025 ≤ b ∧ b ≤ 5.035) (hc : 1.745 ≤ c ∧ c ≤ 1.755)
  (hx : (3 * x + b) / 4 = (2 * x - 3) / c) :
  7.512 ≤ x ∧ x ≤ 7.618 :=
by sorry

end NUMINAMATH_CALUDE_root_bounds_l2189_218967


namespace NUMINAMATH_CALUDE_olivias_paper_count_l2189_218952

/-- Calculates the total remaining pieces of paper given initial amounts and usage --/
def totalRemainingPieces (initialFolder1 initialFolder2 usedFolder1 usedFolder2 : ℕ) : ℕ :=
  (initialFolder1 - usedFolder1) + (initialFolder2 - usedFolder2)

/-- Theorem stating that given the initial conditions and usage, the total remaining pieces of paper is 130 --/
theorem olivias_paper_count :
  totalRemainingPieces 152 98 78 42 = 130 := by
  sorry

end NUMINAMATH_CALUDE_olivias_paper_count_l2189_218952


namespace NUMINAMATH_CALUDE_symmetry_line_of_circles_l2189_218931

/-- Given two circles O and C that are symmetric with respect to a line l, 
    prove that l has the equation x - y + 2 = 0 -/
theorem symmetry_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 4) →  -- equation of circle O
  (x^2 + y^2 + 4*x - 4*y + 4 = 0) →  -- equation of circle C
  ∃ (l : Set (ℝ × ℝ)), 
    (∀ (p : ℝ × ℝ), p ∈ l ↔ p.1 - p.2 + 2 = 0) ∧  -- equation of line l
    (∀ (p : ℝ × ℝ), p ∈ l ↔ 
      ∃ (q r : ℝ × ℝ), 
        (q.1^2 + q.2^2 = 4) ∧  -- q is on circle O
        (r.1^2 + r.2^2 + 4*r.1 - 4*r.2 + 4 = 0) ∧  -- r is on circle C
        (p = ((q.1 + r.1)/2, (q.2 + r.2)/2)) ∧  -- p is midpoint of qr
        ((r.1 - q.1) * (p.1 - q.1) + (r.2 - q.2) * (p.2 - q.2) = 0))  -- qr ⊥ l
  := by sorry

end NUMINAMATH_CALUDE_symmetry_line_of_circles_l2189_218931


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l2189_218945

/-- Given a square and a circle that intersect such that each side of the square
    contains a chord of the circle with length equal to half the radius of the circle,
    the ratio of the area of the square to the area of the circle is 15/(4π). -/
theorem square_circle_area_ratio :
  ∀ (r : ℝ) (square_side : ℝ) (square_area circle_area : ℝ),
  r > 0 →
  square_side = (r * Real.sqrt 15) / 2 →
  square_area = square_side ^ 2 →
  circle_area = π * r ^ 2 →
  square_area / circle_area = 15 / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l2189_218945


namespace NUMINAMATH_CALUDE_banana_cost_l2189_218901

theorem banana_cost (num_bananas : ℕ) (num_oranges : ℕ) (orange_cost : ℚ) (total_cost : ℚ) :
  num_bananas = 5 →
  num_oranges = 10 →
  orange_cost = 3/2 →
  total_cost = 25 →
  (total_cost - num_oranges * orange_cost) / num_bananas = 2 :=
by sorry

end NUMINAMATH_CALUDE_banana_cost_l2189_218901


namespace NUMINAMATH_CALUDE_fraction_equality_l2189_218982

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2189_218982


namespace NUMINAMATH_CALUDE_kenya_peanuts_count_l2189_218992

/-- The number of peanuts Jose has -/
def jose_peanuts : ℕ := 85

/-- The additional number of peanuts Kenya has compared to Jose -/
def kenya_extra_peanuts : ℕ := 48

/-- The number of peanuts Kenya has -/
def kenya_peanuts : ℕ := jose_peanuts + kenya_extra_peanuts

theorem kenya_peanuts_count : kenya_peanuts = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_peanuts_count_l2189_218992


namespace NUMINAMATH_CALUDE_total_shared_amount_l2189_218976

def ken_share : ℕ := 1750
def tony_share : ℕ := 2 * ken_share

theorem total_shared_amount : ken_share + tony_share = 5250 := by
  sorry

end NUMINAMATH_CALUDE_total_shared_amount_l2189_218976


namespace NUMINAMATH_CALUDE_endocrine_cells_synthesize_both_l2189_218911

structure Cell :=
  (canSynthesizeEnzymes : Bool)
  (canSynthesizeHormones : Bool)

structure Hormone :=
  (producedByEndocrine : Bool)
  (directlyParticipateInCells : Bool)

structure Enzyme :=
  (producedByLivingCells : Bool)

def EndocrineCell := {c : Cell // c.canSynthesizeHormones = true}

theorem endocrine_cells_synthesize_both :
  ∀ (h : Hormone) (e : Enzyme) (ec : EndocrineCell),
    h.directlyParticipateInCells = false →
    e.producedByLivingCells = true →
    h.producedByEndocrine = true →
    ec.val.canSynthesizeEnzymes = true ∧ ec.val.canSynthesizeHormones = true :=
by sorry

end NUMINAMATH_CALUDE_endocrine_cells_synthesize_both_l2189_218911


namespace NUMINAMATH_CALUDE_f_negative_before_root_l2189_218950

-- Define the function f(x) = 2^x + log_2(x)
noncomputable def f (x : ℝ) : ℝ := 2^x + Real.log x / Real.log 2

-- State the theorem
theorem f_negative_before_root (a : ℝ) (h1 : f a = 0) (x : ℝ) (h2 : 0 < x) (h3 : x < a) :
  f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_before_root_l2189_218950


namespace NUMINAMATH_CALUDE_coffee_table_price_correct_l2189_218980

/-- Represents the price of the coffee table -/
def coffee_table_price : ℝ := 429.24

/-- Calculates the total cost before discount and tax -/
def total_before_discount_and_tax (coffee_table_price : ℝ) : ℝ :=
  1250 + 2 * 425 + 350 + 200 + coffee_table_price

/-- Calculates the discounted total -/
def discounted_total (coffee_table_price : ℝ) : ℝ :=
  0.9 * total_before_discount_and_tax coffee_table_price

/-- Calculates the final invoice amount after tax -/
def final_invoice_amount (coffee_table_price : ℝ) : ℝ :=
  1.06 * discounted_total coffee_table_price

/-- Theorem stating that the calculated coffee table price results in the given final invoice amount -/
theorem coffee_table_price_correct :
  final_invoice_amount coffee_table_price = 2937.60 := by
  sorry


end NUMINAMATH_CALUDE_coffee_table_price_correct_l2189_218980


namespace NUMINAMATH_CALUDE_sixteenth_number_with_digit_sum_13_l2189_218975

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ := sorry

/-- Theorem stating that the 16th number with digit sum 13 is 247 -/
theorem sixteenth_number_with_digit_sum_13 : 
  nth_number_with_digit_sum_13 16 = 247 := by sorry

end NUMINAMATH_CALUDE_sixteenth_number_with_digit_sum_13_l2189_218975
