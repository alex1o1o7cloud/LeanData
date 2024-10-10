import Mathlib

namespace intersection_sum_problem_l3693_369303

theorem intersection_sum_problem (digits : Finset ℕ) 
  (h_digits : digits.card = 6 ∧ digits ⊆ Finset.range 10 ∧ 1 ∈ digits)
  (vertical : Finset ℕ) (horizontal : Finset ℕ)
  (h_vert : vertical.card = 3 ∧ vertical ⊆ digits)
  (h_horiz : horizontal.card = 4 ∧ horizontal ⊆ digits)
  (h_intersect : (vertical ∩ horizontal).card = 1)
  (h_vert_sum : vertical.sum id = 25)
  (h_horiz_sum : horizontal.sum id = 14) :
  digits.sum id = 31 := by
sorry

end intersection_sum_problem_l3693_369303


namespace black_chess_pieces_count_l3693_369388

theorem black_chess_pieces_count 
  (white_pieces : ℕ) 
  (white_probability : ℚ) 
  (h1 : white_pieces = 9)
  (h2 : white_probability = 3/10) : 
  ∃ (black_pieces : ℕ), 
    (white_pieces : ℚ) / (white_pieces + black_pieces) = white_probability ∧ 
    black_pieces = 21 := by
  sorry

end black_chess_pieces_count_l3693_369388


namespace cylinder_lateral_surface_area_l3693_369365

/-- The area of the unfolded lateral surface of a cylinder with base radius 2 and height 2 is 8π. -/
theorem cylinder_lateral_surface_area : 
  ∀ (r h : ℝ), r = 2 → h = 2 → 2 * π * r * h = 8 * π :=
by
  sorry

end cylinder_lateral_surface_area_l3693_369365


namespace min_students_all_correct_l3693_369338

theorem min_students_all_correct (total_students : ℕ) 
  (q1_correct q2_correct q3_correct q4_correct : ℕ) 
  (h1 : total_students = 45)
  (h2 : q1_correct = 35)
  (h3 : q2_correct = 27)
  (h4 : q3_correct = 41)
  (h5 : q4_correct = 38) :
  total_students - (total_students - q1_correct) - 
  (total_students - q2_correct) - (total_students - q3_correct) - 
  (total_students - q4_correct) = 6 := by
  sorry

end min_students_all_correct_l3693_369338


namespace solve_for_x_l3693_369353

theorem solve_for_x (x y : ℝ) : 3 * x - 4 * y = 6 → x = (6 + 4 * y) / 3 := by
  sorry

end solve_for_x_l3693_369353


namespace hyperbola_eccentricity_l3693_369366

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes y = ±2x is √5 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = 2) : 
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l3693_369366


namespace emerald_count_l3693_369348

/-- Represents a box of gemstones -/
structure GemBox where
  count : ℕ

/-- Represents the collection of gem boxes -/
structure GemCollection where
  diamonds : Array GemBox
  rubies : Array GemBox
  emeralds : Array GemBox

/-- The theorem to be proved -/
theorem emerald_count (collection : GemCollection) : 
  collection.diamonds.size = 2 ∧ 
  collection.rubies.size = 2 ∧ 
  collection.emeralds.size = 2 ∧ 
  (collection.rubies.foldl (λ acc box => acc + box.count) 0 = 
   collection.diamonds.foldl (λ acc box => acc + box.count) 0 + 15) →
  collection.emeralds.foldl (λ acc box => acc + box.count) 0 = 12 := by
  sorry

end emerald_count_l3693_369348


namespace abs_lt_one_sufficient_not_necessary_for_cube_lt_one_l3693_369344

theorem abs_lt_one_sufficient_not_necessary_for_cube_lt_one :
  (∃ x : ℝ, (|x| < 1 → x^3 < 1) ∧ ¬(x^3 < 1 → |x| < 1)) := by
  sorry

end abs_lt_one_sufficient_not_necessary_for_cube_lt_one_l3693_369344


namespace circle_area_increase_l3693_369309

theorem circle_area_increase (r : ℝ) : 
  let initial_area := π * r^2
  let final_area := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π :=
by sorry

end circle_area_increase_l3693_369309


namespace candy_mixture_cost_l3693_369331

/-- Proves that the cost of the first candy is $8.00 per pound given the conditions of the candy mixture problem. -/
theorem candy_mixture_cost (first_candy_weight : ℝ) (second_candy_weight : ℝ) (second_candy_cost : ℝ) (mixture_cost : ℝ) 
  (h1 : first_candy_weight = 25)
  (h2 : second_candy_weight = 50)
  (h3 : second_candy_cost = 5)
  (h4 : mixture_cost = 6)
  (h5 : first_candy_weight + second_candy_weight = 75) :
  ∃ (C : ℝ), C = 8 ∧ 
  C * first_candy_weight + second_candy_cost * second_candy_weight = 
  mixture_cost * (first_candy_weight + second_candy_weight) :=
by sorry

end candy_mixture_cost_l3693_369331


namespace night_games_count_l3693_369383

theorem night_games_count (total_games : ℕ) (h1 : total_games = 864) 
  (h2 : ∃ (night_games day_games : ℕ), night_games + day_games = total_games ∧ night_games = day_games) : 
  ∃ (night_games : ℕ), night_games = 432 := by
sorry

end night_games_count_l3693_369383


namespace quadratic_functions_property_l3693_369390

/-- Two quadratic functions with specific properties -/
theorem quadratic_functions_property (h j k : ℝ) : 
  (∃ (a b c d : ℕ), a ≠ b ∧ c ≠ d ∧ 
    3 * (a - h)^2 + j = 0 ∧ 
    3 * (b - h)^2 + j = 0 ∧
    2 * (c - h)^2 + k = 0 ∧ 
    2 * (d - h)^2 + k = 0) →
  (3 * h^2 + j = 2013 ∧ 2 * h^2 + k = 2014) →
  h = 36 := by
sorry

end quadratic_functions_property_l3693_369390


namespace problem_solution_l3693_369391

theorem problem_solution (a b : ℚ) 
  (eq1 : 3020 * a + 3026 * b = 3030)
  (eq2 : 3024 * a + 3028 * b = 3034) :
  a - 2 * b = -1509 / 1516 := by
sorry

end problem_solution_l3693_369391


namespace quadratic_always_positive_iff_a_in_range_l3693_369308

theorem quadratic_always_positive_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 - a*x + 1 > 0) ↔ -2 < a ∧ a < 2 := by
  sorry

end quadratic_always_positive_iff_a_in_range_l3693_369308


namespace smallest_solution_biquadratic_l3693_369359

theorem smallest_solution_biquadratic (x : ℝ) :
  x^4 - 26*x^2 + 169 = 0 → x ≥ -Real.sqrt 13 :=
by
  sorry

end smallest_solution_biquadratic_l3693_369359


namespace cube_side_area_l3693_369330

/-- Given a cube with a total surface area of 54.3 square centimeters,
    the area of one side is 9.05 square centimeters. -/
theorem cube_side_area (total_area : ℝ) (h1 : total_area = 54.3) : ∃ (side_area : ℝ), 
  side_area = 9.05 ∧ 6 * side_area = total_area := by
  sorry

end cube_side_area_l3693_369330


namespace author_average_earnings_l3693_369382

theorem author_average_earnings 
  (months_per_book : ℕ) 
  (years_writing : ℕ) 
  (total_earnings : ℕ) : 
  months_per_book = 2 → 
  years_writing = 20 → 
  total_earnings = 3600000 → 
  (total_earnings : ℚ) / ((12 / months_per_book) * years_writing) = 30000 :=
by sorry

end author_average_earnings_l3693_369382


namespace height_difference_in_inches_l3693_369377

-- Define conversion factors
def meters_to_feet : ℝ := 3.28084
def inches_per_foot : ℕ := 12

-- Define heights in meters
def mark_height : ℝ := 1.60
def mike_height : ℝ := 1.85

-- Function to convert meters to inches
def meters_to_inches (m : ℝ) : ℝ := m * meters_to_feet * inches_per_foot

-- Theorem statement
theorem height_difference_in_inches :
  ⌊meters_to_inches mike_height - meters_to_inches mark_height⌋ = 10 := by
  sorry

end height_difference_in_inches_l3693_369377


namespace roots_sum_squares_l3693_369347

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x - 2

-- Define the theorem
theorem roots_sum_squares (p q r : ℝ) : 
  f p = 0 → f q = 0 → f r = 0 → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = 12 := by
  sorry

end roots_sum_squares_l3693_369347


namespace sin_x_equals_x_unique_root_l3693_369387

theorem sin_x_equals_x_unique_root :
  ∃! x : ℝ, x ∈ Set.Icc (-π) π ∧ x = Real.sin x := by
  sorry

end sin_x_equals_x_unique_root_l3693_369387


namespace equal_segments_imply_equal_x_y_l3693_369346

/-- Given two pairs of equal lengths (a₁, a₂) and (b₁, b₂), prove that x = y. -/
theorem equal_segments_imply_equal_x_y (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h1 : a₁ = a₂) (h2 : b₁ = b₂) : x = y := by
  sorry

end equal_segments_imply_equal_x_y_l3693_369346


namespace complex_number_line_l3693_369343

theorem complex_number_line (z : ℂ) (h : z * (1 + Complex.I)^2 = 1 - Complex.I) :
  z.im = -1/2 := by sorry

end complex_number_line_l3693_369343


namespace zack_andrew_same_team_probability_l3693_369372

-- Define the total number of players
def total_players : ℕ := 27

-- Define the number of teams
def num_teams : ℕ := 3

-- Define the number of players per team
def players_per_team : ℕ := 9

-- Define the set of players
def Player : Type := Fin total_players

-- Define the function that assigns players to teams
def team_assignment : Player → Fin num_teams := sorry

-- Define Zack, Mihir, and Andrew as specific players
def Zack : Player := sorry
def Mihir : Player := sorry
def Andrew : Player := sorry

-- State that Zack and Mihir are on different teams
axiom zack_mihir_different : team_assignment Zack ≠ team_assignment Mihir

-- State that Mihir and Andrew are on different teams
axiom mihir_andrew_different : team_assignment Mihir ≠ team_assignment Andrew

-- Define the probability function
def probability_same_team (p1 p2 : Player) : ℚ := sorry

-- State the theorem to be proved
theorem zack_andrew_same_team_probability :
  probability_same_team Zack Andrew = 8 / 17 := sorry

end zack_andrew_same_team_probability_l3693_369372


namespace avocado_count_is_two_l3693_369326

/-- Represents the contents and cost of a fruit basket -/
structure FruitBasket where
  banana_count : ℕ
  apple_count : ℕ
  strawberry_count : ℕ
  avocado_count : ℕ
  grapes_count : ℕ
  banana_price : ℚ
  apple_price : ℚ
  strawberry_price : ℚ
  avocado_price : ℚ
  grapes_price : ℚ
  total_cost : ℚ

/-- The fruit basket problem -/
def fruit_basket_problem : FruitBasket :=
  { banana_count := 4
  , apple_count := 3
  , strawberry_count := 24
  , avocado_count := 0  -- This is what we need to prove
  , grapes_count := 1
  , banana_price := 1
  , apple_price := 2
  , strawberry_price := 1/3  -- $4 for 12 strawberries
  , avocado_price := 3
  , grapes_price := 4  -- $2 for half a bunch, so $4 for a full bunch
  , total_cost := 28 }

/-- Theorem stating that the number of avocados in the fruit basket is 2 -/
theorem avocado_count_is_two (fb : FruitBasket) 
  (h1 : fb = fruit_basket_problem) :
  fb.avocado_count = 2 := by
  sorry


end avocado_count_is_two_l3693_369326


namespace green_paint_calculation_l3693_369381

/-- Given a paint mixture ratio and the amount of white paint, 
    calculate the amount of green paint needed. -/
theorem green_paint_calculation 
  (blue green white : ℚ) 
  (ratio : blue / green = 5 / 3 ∧ green / white = 3 / 7) 
  (white_amount : white = 21) : 
  green = 9 := by
sorry

end green_paint_calculation_l3693_369381


namespace hcd_7560_180_minus_12_l3693_369334

theorem hcd_7560_180_minus_12 : Nat.gcd 7560 180 - 12 = 168 := by sorry

end hcd_7560_180_minus_12_l3693_369334


namespace suki_coffee_bags_suki_coffee_bags_proof_l3693_369341

theorem suki_coffee_bags (suki_bag_weight jimmy_bag_weight container_weight : ℕ)
                         (jimmy_bags : ℚ)
                         (num_containers : ℕ)
                         (suki_bags : ℕ) : Prop :=
  suki_bag_weight = 22 →
  jimmy_bag_weight = 18 →
  jimmy_bags = 4.5 →
  container_weight = 8 →
  num_containers = 28 →
  (↑suki_bags * suki_bag_weight + jimmy_bags * jimmy_bag_weight : ℚ) = ↑(num_containers * container_weight) →
  suki_bags = 6

theorem suki_coffee_bags_proof : suki_coffee_bags 22 18 8 (4.5 : ℚ) 28 6 := by
  sorry

end suki_coffee_bags_suki_coffee_bags_proof_l3693_369341


namespace percentage_difference_l3693_369325

theorem percentage_difference : 
  (45 / 100 * 60) - (35 / 100 * 40) = 13 := by
  sorry

end percentage_difference_l3693_369325


namespace sqrt_13_plus_1_parts_l3693_369329

theorem sqrt_13_plus_1_parts : ∃ (a : ℤ) (b : ℝ),
  (a : ℝ) + b = Real.sqrt 13 + 1 ∧ 
  a = 4 ∧
  b = Real.sqrt 13 - 3 ∧
  0 ≤ b ∧ 
  b < 1 := by
sorry

end sqrt_13_plus_1_parts_l3693_369329


namespace unknown_number_value_l3693_369336

theorem unknown_number_value : 
  ∃ (unknown_number : ℝ), 
    (∀ x : ℝ, (3 + 2 * x)^5 = (unknown_number + 3 * x)^4) ∧
    ((3 + 2 * 1.5)^5 = (unknown_number + 3 * 1.5)^4) →
    unknown_number = 1.5 :=
by sorry

end unknown_number_value_l3693_369336


namespace sqrt_six_div_sqrt_two_eq_sqrt_three_l3693_369397

theorem sqrt_six_div_sqrt_two_eq_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_div_sqrt_two_eq_sqrt_three_l3693_369397


namespace work_completion_time_l3693_369313

/-- If a group of people can complete a work in 8 days, then twice the number of people can complete half the work in 2 days. -/
theorem work_completion_time 
  (P : ℕ) -- Number of people
  (W : ℝ) -- Amount of work
  (h : P > 0) -- Assumption that there is at least one person
  (completion_time : ℝ) -- Time to complete the work
  (h_completion : completion_time = 8) -- Given that the work is completed in 8 days
  : (2 * P) * (W / 2) / W * completion_time = 2 :=
by sorry

end work_completion_time_l3693_369313


namespace quadratic_equation_solution_l3693_369327

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = -3 ∧ x₂ = 4) ∧ 
  (x₁^2 - x₁ - 12 = 0) ∧ 
  (x₂^2 - x₂ - 12 = 0) ∧
  (∀ x : ℝ, x^2 - x - 12 = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_equation_solution_l3693_369327


namespace sector_central_angle_l3693_369392

/-- Given a sector with radius 1 and arc length 2, its central angle is 2 radians -/
theorem sector_central_angle (radius : ℝ) (arc_length : ℝ) (h1 : radius = 1) (h2 : arc_length = 2) :
  arc_length / radius = 2 := by sorry

end sector_central_angle_l3693_369392


namespace cara_seating_arrangements_l3693_369307

theorem cara_seating_arrangements (n : ℕ) (h : n = 6) : (n.choose 2) = 15 := by
  sorry

end cara_seating_arrangements_l3693_369307


namespace triangle_point_coordinates_l3693_369373

/-- Given a triangle ABC with median CM and angle bisector BL, prove that the coordinates of C are (14, 2) -/
theorem triangle_point_coordinates (A M L : ℝ × ℝ) : 
  A = (2, 8) → M = (4, 11) → L = (6, 6) → 
  ∃ (B C : ℝ × ℝ), 
    (M.1 = (A.1 + C.1) / 2 ∧ M.2 = (A.2 + C.2) / 2) ∧  -- M is midpoint of AC
    (∃ (t : ℝ), L = B + t • (A - C)) ∧                 -- L is on angle bisector BL
    C = (14, 2) := by
sorry

end triangle_point_coordinates_l3693_369373


namespace sqrt_combinable_with_sqrt_two_l3693_369335

theorem sqrt_combinable_with_sqrt_two : ∃! x : ℝ, 
  (x = Real.sqrt 10 ∨ x = Real.sqrt 12 ∨ x = Real.sqrt (1/2) ∨ x = 1 / Real.sqrt 6) ∧
  ∃ (a : ℝ), x = a * Real.sqrt 2 := by
  sorry

end sqrt_combinable_with_sqrt_two_l3693_369335


namespace complex_sum_equals_two_l3693_369324

theorem complex_sum_equals_two (z : ℂ) (h : z = Complex.exp (2 * Real.pi * I / 5)) :
  z / (1 + z^2) + z^2 / (1 + z^4) + z^3 / (1 + z^6) = 2 := by
  sorry

end complex_sum_equals_two_l3693_369324


namespace rainfall_2011_l3693_369319

/-- The total rainfall in Rainville for 2011, given the average monthly rainfall in 2010 and the increase in 2011. -/
def total_rainfall_2011 (avg_2010 : ℝ) (increase : ℝ) : ℝ :=
  (avg_2010 + increase) * 12

/-- Theorem stating that the total rainfall in Rainville for 2011 was 483.6 mm. -/
theorem rainfall_2011 : total_rainfall_2011 36.8 3.5 = 483.6 := by
  sorry

end rainfall_2011_l3693_369319


namespace sophie_shopping_budget_l3693_369305

def initial_budget : ℚ := 260
def shirt_cost : ℚ := 18.5
def num_shirts : ℕ := 2
def trouser_cost : ℚ := 63
def num_additional_items : ℕ := 4

theorem sophie_shopping_budget :
  let total_spent := shirt_cost * num_shirts + trouser_cost
  let remaining_budget := initial_budget - total_spent
  remaining_budget / num_additional_items = 40 := by
  sorry

end sophie_shopping_budget_l3693_369305


namespace sum_of_square_roots_geq_one_l3693_369323

theorem sum_of_square_roots_geq_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a / (a + 3 * b)) + Real.sqrt (b / (b + 3 * a)) ≥ 1 := by
sorry

end sum_of_square_roots_geq_one_l3693_369323


namespace positive_integer_solutions_count_l3693_369356

theorem positive_integer_solutions_count : 
  (Finset.filter (fun (xyz : ℕ × ℕ × ℕ) => 
    xyz.1 + xyz.2.1 + xyz.2.2 = 12 ∧ 
    xyz.1 > 0 ∧ xyz.2.1 > 0 ∧ xyz.2.2 > 0) 
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 12) (Finset.range 12)))).card = 55 :=
by sorry

end positive_integer_solutions_count_l3693_369356


namespace eight_book_distribution_l3693_369349

/-- The number of ways to distribute n identical books between two locations,
    with at least one book in each location. -/
def distribution_ways (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

/-- Theorem stating that there are 7 ways to distribute 8 identical books
    between storage and students, with at least one book in each location. -/
theorem eight_book_distribution :
  distribution_ways 8 = 7 := by
  sorry

end eight_book_distribution_l3693_369349


namespace circle_equation_l3693_369393

/-- The equation of a circle with center (-1, 2) and radius 4 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-1, 2)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ (x + 1)^2 + (y - 2)^2 = 16 :=
by sorry

end circle_equation_l3693_369393


namespace total_books_calculation_l3693_369368

/-- The total number of books assigned to Mcgregor and Floyd -/
def total_books : ℕ := 89

/-- The number of books Mcgregor finished -/
def mcgregor_books : ℕ := 34

/-- The number of books Floyd finished -/
def floyd_books : ℕ := 32

/-- The number of books remaining to be read -/
def remaining_books : ℕ := 23

/-- Theorem stating that the total number of books is the sum of the books finished by Mcgregor and Floyd, plus the remaining books -/
theorem total_books_calculation : 
  total_books = mcgregor_books + floyd_books + remaining_books :=
by sorry

end total_books_calculation_l3693_369368


namespace karen_savings_l3693_369350

/-- The sum of a geometric series with initial term 2, common ratio 3, and 7 terms -/
def geometric_sum : ℕ → ℚ
| 0 => 0
| n + 1 => 2 * (3^(n+1) - 1) / (3 - 1)

/-- The theorem stating that the sum of the geometric series after 7 days is 2186 -/
theorem karen_savings : geometric_sum 7 = 2186 := by
  sorry

end karen_savings_l3693_369350


namespace sum_of_coefficients_l3693_369306

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^10 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 20 :=
by sorry

end sum_of_coefficients_l3693_369306


namespace snyder_cookies_l3693_369396

/-- Mrs. Snyder's cookie problem -/
theorem snyder_cookies (red_cookies pink_cookies : ℕ) 
  (h1 : red_cookies = 36)
  (h2 : pink_cookies = 50) :
  red_cookies + pink_cookies = 86 := by
  sorry

end snyder_cookies_l3693_369396


namespace female_employees_count_l3693_369367

/-- Proves that the total number of female employees in a company is 500 under given conditions -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) (female_managers : ℕ) :
  female_managers = 200 →
  (2 : ℚ) / 5 * total_employees = female_managers + (2 : ℚ) / 5 * male_employees →
  total_employees = male_employees + 500 :=
by sorry

end female_employees_count_l3693_369367


namespace largest_circle_radius_l3693_369357

/-- Represents a chessboard square --/
structure Square where
  x : Nat
  y :Nat
  isWhite : Bool

/-- Represents a chessboard --/
def Chessboard := List Square

/-- Creates an 8x8 chessboard with alternating white and black squares --/
def createChessboard : Chessboard :=
  sorry

/-- Checks if a given point (x, y) is on a white square or corner --/
def isOnWhiteSquareOrCorner (board : Chessboard) (x : ℝ) (y : ℝ) : Prop :=
  sorry

/-- Represents a circle on the chessboard --/
structure Circle where
  centerX : ℝ
  centerY : ℝ
  radius : ℝ

/-- Checks if a circle's circumference is entirely on white squares or corners --/
def isValidCircle (board : Chessboard) (circle : Circle) : Prop :=
  sorry

/-- The theorem to be proved --/
theorem largest_circle_radius (board : Chessboard := createChessboard) :
  ∃ (c : Circle), isValidCircle board c ∧
    ∀ (c' : Circle), isValidCircle board c' → c'.radius ≤ c.radius ∧
    c.radius = Real.sqrt 10 / 2 :=
  sorry

end largest_circle_radius_l3693_369357


namespace quadratic_minimum_l3693_369375

/-- Given two quadratic functions f and g, if the sum of roots of f equals the product of roots of g,
    and the product of roots of f equals the sum of roots of g, then f attains its minimum at x = 3 -/
theorem quadratic_minimum (r s : ℝ) : 
  let f (x : ℝ) := x^2 + r*x + s
  let g (x : ℝ) := x^2 - 9*x + 6
  let sum_roots_f := -r
  let prod_roots_f := s
  let sum_roots_g := 9
  let prod_roots_g := 6
  (sum_roots_f = prod_roots_g) → (prod_roots_f = sum_roots_g) →
  ∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), f x ≥ f a :=
by sorry

end quadratic_minimum_l3693_369375


namespace Q_3_volume_l3693_369300

/-- Recursive definition of the volume of Qᵢ -/
def Q_volume : ℕ → ℚ
  | 0 => 1
  | n + 1 => Q_volume n + 4 * 4^n * (1 / 27)^(n + 1)

/-- The volume of Q₃ is 73/81 -/
theorem Q_3_volume : Q_volume 3 = 73 / 81 := by
  sorry

end Q_3_volume_l3693_369300


namespace simplify_square_roots_l3693_369311

theorem simplify_square_roots : 
  (Real.sqrt 800 / Real.sqrt 100) - (Real.sqrt 288 / Real.sqrt 72) = 2 * Real.sqrt 2 - 2 := by
sorry

end simplify_square_roots_l3693_369311


namespace tobys_sharing_l3693_369322

theorem tobys_sharing (initial_amount : ℚ) (remaining_amount : ℚ) (num_brothers : ℕ) :
  initial_amount = 343 →
  remaining_amount = 245 →
  num_brothers = 2 →
  (initial_amount - remaining_amount) / (num_brothers * initial_amount) = 1 / 7 := by
  sorry

end tobys_sharing_l3693_369322


namespace simplify_trig_expression_l3693_369333

theorem simplify_trig_expression :
  (Real.sin (10 * π / 180) + Real.sin (20 * π / 180)) /
  (Real.cos (10 * π / 180) + Real.cos (20 * π / 180)) =
  Real.tan (15 * π / 180) :=
by sorry

end simplify_trig_expression_l3693_369333


namespace sum_of_squares_problem_l3693_369304

theorem sum_of_squares_problem (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (eq1 : a^2 + a*b + b^2 = 1)
  (eq2 : b^2 + b*c + c^2 = 3)
  (eq3 : c^2 + c*a + a^2 = 4) :
  a + b + c = Real.sqrt 7 := by
  sorry

end sum_of_squares_problem_l3693_369304


namespace smallest_lcm_with_gcd_five_l3693_369385

theorem smallest_lcm_with_gcd_five (a b : ℕ) : 
  1000 ≤ a ∧ a < 10000 ∧ 
  1000 ≤ b ∧ b < 10000 ∧ 
  Nat.gcd a b = 5 →
  201000 ≤ Nat.lcm a b ∧ 
  ∃ (x y : ℕ), 1000 ≤ x ∧ x < 10000 ∧ 
               1000 ≤ y ∧ y < 10000 ∧ 
               Nat.gcd x y = 5 ∧ 
               Nat.lcm x y = 201000 :=
by sorry

end smallest_lcm_with_gcd_five_l3693_369385


namespace each_girl_receives_two_dollars_l3693_369360

def debt : ℕ := 40

def lulu_savings : ℕ := 6

def nora_savings : ℕ := 5 * lulu_savings

def tamara_savings : ℕ := nora_savings / 3

def total_savings : ℕ := tamara_savings + nora_savings + lulu_savings

def remaining_money : ℕ := total_savings - debt

theorem each_girl_receives_two_dollars : 
  remaining_money / 3 = 2 := by sorry

end each_girl_receives_two_dollars_l3693_369360


namespace problem_solving_probability_l3693_369320

theorem problem_solving_probability (p_A p_B : ℝ) (h_A : p_A = 1/5) (h_B : p_B = 1/3) :
  1 - (1 - p_A) * (1 - p_B) = 7/15 :=
by sorry

end problem_solving_probability_l3693_369320


namespace min_distance_intersection_l3693_369321

/-- The minimum distance between intersection points --/
theorem min_distance_intersection (m : ℝ) : 
  let f (x : ℝ) := |x - (x + Real.exp x + 3) / 2|
  ∃ (x : ℝ), f x = 2 ∧ ∀ (y : ℝ), f y ≥ 2 := by
  sorry

end min_distance_intersection_l3693_369321


namespace prob_red_card_standard_deck_l3693_369351

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (red_suits : ℕ)
  (cards_per_suit : ℕ)

/-- A standard deck of 52 cards -/
def standard_deck : Deck :=
  { total_cards := 52,
    ranks := 13,
    suits := 4,
    red_suits := 2,
    cards_per_suit := 13 }

/-- The probability of drawing a red suit card from the top of a randomly shuffled deck -/
def prob_red_card (d : Deck) : ℚ :=
  (d.red_suits * d.cards_per_suit) / d.total_cards

/-- Theorem stating that the probability of drawing a red suit card from a standard deck is 1/2 -/
theorem prob_red_card_standard_deck :
  prob_red_card standard_deck = 1/2 := by
  sorry

end prob_red_card_standard_deck_l3693_369351


namespace sum_of_ages_l3693_369358

/-- Given the ages and relationships of Beckett, Olaf, Shannen, and Jack, prove that the sum of their ages is 71. -/
theorem sum_of_ages (beckett_age olaf_age shannen_age jack_age : ℕ) : 
  beckett_age = 12 →
  olaf_age = beckett_age + 3 →
  shannen_age = olaf_age - 2 →
  jack_age = 2 * shannen_age + 5 →
  beckett_age + olaf_age + shannen_age + jack_age = 71 := by
  sorry


end sum_of_ages_l3693_369358


namespace greatest_y_value_l3693_369371

theorem greatest_y_value (y : ℝ) : 
  (3 * y^2 + 5 * y + 2 = 6) → 
  y ≤ (-5 + Real.sqrt 73) / 6 :=
by sorry

end greatest_y_value_l3693_369371


namespace youtube_dislikes_difference_l3693_369389

theorem youtube_dislikes_difference (D : ℕ) : 
  D + 1000 = 2600 → D - D / 2 = 800 := by
  sorry

end youtube_dislikes_difference_l3693_369389


namespace expression_simplification_l3693_369301

theorem expression_simplification (x y : ℝ) :
  (2 * x - (3 * y - (2 * x + 1))) - ((3 * y - (2 * x + 1)) - 2 * x) = 8 * x - 6 * y + 2 := by
  sorry

end expression_simplification_l3693_369301


namespace ice_cream_flavors_l3693_369312

/-- The number of flavors in the ice cream shop -/
def F : ℕ := sorry

/-- The number of flavors Gretchen tried two years ago -/
def tried_two_years_ago : ℚ := F / 4

/-- The number of flavors Gretchen tried last year -/
def tried_last_year : ℚ := 2 * tried_two_years_ago

/-- The number of flavors Gretchen still needs to try this year -/
def flavors_left : ℕ := 25

theorem ice_cream_flavors :
  F = 100 ∧
  tried_two_years_ago = F / 4 ∧
  tried_last_year = 2 * tried_two_years_ago ∧
  flavors_left = 25 ∧
  F = (tried_two_years_ago + tried_last_year + flavors_left) :=
sorry

end ice_cream_flavors_l3693_369312


namespace iesha_school_books_l3693_369378

/-- The number of books Iesha has about school -/
def books_about_school (total_books sports_books : ℕ) : ℕ :=
  total_books - sports_books

/-- Theorem stating that Iesha has 136 books about school -/
theorem iesha_school_books :
  books_about_school 344 208 = 136 := by
  sorry

end iesha_school_books_l3693_369378


namespace irrationality_of_pi_l3693_369315

-- Define rational numbers
def isRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrational numbers as the complement of rational numbers
def isIrrational (x : ℝ) : Prop := ¬(isRational x)

-- Theorem statement
theorem irrationality_of_pi :
  isIrrational π ∧ isRational 0 ∧ isRational (22/7) ∧ isRational (Real.rpow 8 (1/3)) := by
  sorry


end irrationality_of_pi_l3693_369315


namespace sqrt_three_minus_sqrt_two_plus_sqrt_six_simplify_sqrt_expression_l3693_369395

-- Problem 1
theorem sqrt_three_minus_sqrt_two_plus_sqrt_six : 
  Real.sqrt 3 * (Real.sqrt 3 - Real.sqrt 2) + Real.sqrt 6 = 3 := by sorry

-- Problem 2
theorem simplify_sqrt_expression (a : ℝ) (ha : a > 0) : 
  2 * Real.sqrt (12 * a) + Real.sqrt (6 * a^2) + Real.sqrt (2 * a) = 
  4 * Real.sqrt (3 * a) + Real.sqrt 6 * a + Real.sqrt (2 * a) := by sorry

end sqrt_three_minus_sqrt_two_plus_sqrt_six_simplify_sqrt_expression_l3693_369395


namespace constant_b_equals_negative_two_l3693_369362

/-- Given a polynomial equation, prove that the constant b must equal -2. -/
theorem constant_b_equals_negative_two :
  ∀ (a c : ℝ) (b : ℝ),
  (fun x : ℝ => (4 * x^3 - 2 * x + 5/2) * (a * x^3 + b * x^2 + c)) =
  (fun x : ℝ => 20 * x^6 - 8 * x^4 + 15 * x^3 - 5 * x^2 + 5) →
  b = -2 := by
sorry

end constant_b_equals_negative_two_l3693_369362


namespace log_base_value_l3693_369316

theorem log_base_value (f : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x = Real.log x / Real.log a) →  -- Definition of f as logarithm base a
  a > 0 →                                     -- Condition: a > 0
  a ≠ 1 →                                     -- Condition: a ≠ 1
  f 9 = 2 →                                   -- Condition: f(9) = 2
  a = 3 :=                                    -- Conclusion: a = 3
by sorry

end log_base_value_l3693_369316


namespace min_x_plus_y_l3693_369384

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) :
  ∀ z w : ℝ, z > 0 → w > 0 → 2*z + 8*w - z*w = 0 → x + y ≤ z + w ∧ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 2*a + 8*b - a*b = 0 ∧ a + b = 18 :=
by sorry

end min_x_plus_y_l3693_369384


namespace jamie_alex_payment_difference_l3693_369345

-- Define the problem parameters
def total_slices : ℕ := 10
def plain_pizza_cost : ℚ := 10
def spicy_topping_cost : ℚ := 3
def spicy_fraction : ℚ := 1/3

-- Define the number of slices each person ate
def jamie_spicy_slices : ℕ := (spicy_fraction * total_slices).num.toNat
def jamie_plain_slices : ℕ := 2
def alex_plain_slices : ℕ := total_slices - jamie_spicy_slices - jamie_plain_slices

-- Define the theorem
theorem jamie_alex_payment_difference :
  let total_cost : ℚ := plain_pizza_cost + spicy_topping_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let jamie_payment : ℚ := cost_per_slice * (jamie_spicy_slices + jamie_plain_slices)
  let alex_payment : ℚ := cost_per_slice * alex_plain_slices
  jamie_payment - alex_payment = 0 :=
sorry

end jamie_alex_payment_difference_l3693_369345


namespace discount_difference_l3693_369364

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  bill = 15000 ∧ 
  single_discount = 0.3 ∧ 
  first_discount = 0.25 ∧ 
  second_discount = 0.05 →
  bill * (1 - first_discount) * (1 - second_discount) - bill * (1 - single_discount) = 187.5 := by
sorry

end discount_difference_l3693_369364


namespace max_value_x_minus_x_squared_l3693_369376

theorem max_value_x_minus_x_squared (f : ℝ → ℝ) (h : ∀ x, 0 < x → x < 1 → f x = x * (1 - x)) :
  ∃ m : ℝ, m = 1/4 ∧ ∀ x, 0 < x → x < 1 → f x ≤ m :=
sorry

end max_value_x_minus_x_squared_l3693_369376


namespace order_of_magnitudes_l3693_369352

-- Define the function f(x) = ln(x) - x
noncomputable def f (x : ℝ) : ℝ := Real.log x - x

-- Define a, b, and c
noncomputable def a : ℝ := f (3/2)
noncomputable def b : ℝ := f Real.pi
noncomputable def c : ℝ := f 3

-- State the theorem
theorem order_of_magnitudes (h1 : 3/2 < 3) (h2 : 3 < Real.pi) : a > c ∧ c > b := by
  sorry

end order_of_magnitudes_l3693_369352


namespace greatest_area_difference_l3693_369370

/-- A rectangle with integer dimensions and perimeter 160 cm -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_condition : length * 2 + width * 2 = 160

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the greatest possible difference between areas of two such rectangles -/
theorem greatest_area_difference :
  ∃ (r1 r2 : Rectangle), ∀ (s1 s2 : Rectangle),
    (area r1 - area r2 : ℤ) ≥ (area s1 - area s2 : ℤ) ∧
    (area r1 - area r2 : ℕ) = 1521 := by
  sorry

end greatest_area_difference_l3693_369370


namespace scientific_notation_570_million_l3693_369337

theorem scientific_notation_570_million :
  (570000000 : ℝ) = 5.7 * (10 : ℝ) ^ 8 := by
  sorry

end scientific_notation_570_million_l3693_369337


namespace min_abs_ab_for_perpendicular_lines_l3693_369317

theorem min_abs_ab_for_perpendicular_lines (a b : ℝ) : 
  (∀ x y : ℝ, x + a^2 * y + 1 = 0 ∧ (a^2 + 1) * x - b * y + 3 = 0 → 
    (1 : ℝ) + a^2 * (-b) = 0) → 
  ∃ m : ℝ, m = 2 ∧ ∀ k : ℝ, k = |a * b| → k ≥ m :=
by sorry

end min_abs_ab_for_perpendicular_lines_l3693_369317


namespace car_travel_time_ratio_l3693_369310

theorem car_travel_time_ratio : 
  let distance : ℝ := 540
  let original_time : ℝ := 8
  let new_speed : ℝ := 45
  let new_time : ℝ := distance / new_speed
  new_time / original_time = 1.5 := by
sorry

end car_travel_time_ratio_l3693_369310


namespace trees_planted_specific_plot_l3693_369318

/-- Calculates the number of trees planted around a rectangular plot -/
def trees_planted (length width spacing : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let total_intervals := perimeter / spacing
  total_intervals - 4

/-- Theorem stating the number of trees planted around the specific rectangular plot -/
theorem trees_planted_specific_plot :
  trees_planted 60 30 6 = 26 :=
by
  sorry

end trees_planted_specific_plot_l3693_369318


namespace parking_cost_savings_l3693_369379

-- Define the cost per week
def cost_per_week : ℕ := 10

-- Define the cost per month
def cost_per_month : ℕ := 35

-- Define the number of weeks in a year
def weeks_per_year : ℕ := 52

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem parking_cost_savings : 
  (weeks_per_year * cost_per_week) - (months_per_year * cost_per_month) = 100 := by
  sorry


end parking_cost_savings_l3693_369379


namespace motorcycle_vs_car_profit_difference_l3693_369340

/-- Represents the production and sales data for a vehicle type -/
structure VehicleData where
  materialCost : ℕ
  quantity : ℕ
  price : ℕ

/-- Calculates the profit for a given vehicle type -/
def profit (data : VehicleData) : ℤ :=
  (data.quantity * data.price) - data.materialCost

/-- Theorem: The difference in profit between motorcycle and car production is $50 -/
theorem motorcycle_vs_car_profit_difference :
  let carData : VehicleData := ⟨100, 4, 50⟩
  let motorcycleData : VehicleData := ⟨250, 8, 50⟩
  profit motorcycleData - profit carData = 50 := by
  sorry

#eval profit ⟨250, 8, 50⟩ - profit ⟨100, 4, 50⟩

end motorcycle_vs_car_profit_difference_l3693_369340


namespace perpendicular_vectors_imply_k_l3693_369399

/-- Given vectors a, b, and c in R², prove that if (a - 2b) is perpendicular to c, then k = -3 -/
theorem perpendicular_vectors_imply_k (a b c : ℝ × ℝ) (h1 : a = (Real.sqrt 3, 1))
    (h2 : b = (0, -1)) (h3 : c = (k, Real.sqrt 3)) 
    (h4 : (a - 2 • b) • c = 0) : k = -3 := by
  sorry

end perpendicular_vectors_imply_k_l3693_369399


namespace containers_used_l3693_369339

def initial_balls : ℕ := 100
def balls_per_container : ℕ := 10

theorem containers_used :
  let remaining_balls := initial_balls / 2
  remaining_balls / balls_per_container = 5 := by
  sorry

end containers_used_l3693_369339


namespace subtract_three_numbers_l3693_369394

theorem subtract_three_numbers : 15 - 3 - 15 = -3 := by
  sorry

end subtract_three_numbers_l3693_369394


namespace locus_of_centers_l3693_369355

-- Define the circles C1 and C3
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C3 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the property of being externally tangent to C1 and internally tangent to C3
def is_tangent_to_C1_C3 (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (3 - r)^2)

-- State the theorem
theorem locus_of_centers (a b : ℝ) :
  (∃ r, is_tangent_to_C1_C3 a b r) → a^2 - 12*a + 4*b^2 = 0 :=
by sorry

end locus_of_centers_l3693_369355


namespace square_area_l3693_369342

/-- A square with a circle tangent to three sides and passing through the diagonal midpoint -/
structure SquareWithCircle where
  s : ℝ  -- side length of the square
  r : ℝ  -- radius of the circle
  s_pos : 0 < s  -- side length is positive
  r_pos : 0 < r  -- radius is positive
  tangent_condition : s = 4 * r  -- derived from the tangent and midpoint conditions

/-- The area of the square is 16r^2 -/
theorem square_area (config : SquareWithCircle) : config.s^2 = 16 * config.r^2 := by
  sorry

#check square_area

end square_area_l3693_369342


namespace volleyball_tournament_games_l3693_369302

/-- The number of games played in a volleyball tournament -/
def tournament_games (n : ℕ) (g : ℕ) : ℕ :=
  (n * (n - 1) * g) / 2

/-- Theorem: A volleyball tournament with 10 teams, where each team plays 4 games
    with every other team, has a total of 180 games. -/
theorem volleyball_tournament_games :
  tournament_games 10 4 = 180 := by
  sorry

end volleyball_tournament_games_l3693_369302


namespace phoenix_airport_on_time_rate_l3693_369386

def late_flights : ℕ := 1
def initial_on_time_flights : ℕ := 3

def on_time_rate (additional_on_time : ℕ) : ℚ :=
  (initial_on_time_flights + additional_on_time) / (late_flights + initial_on_time_flights + additional_on_time)

theorem phoenix_airport_on_time_rate :
  ∃ n : ℕ, n > 0 ∧ on_time_rate n > (2 : ℚ) / 5 :=
sorry

end phoenix_airport_on_time_rate_l3693_369386


namespace truck_distance_l3693_369363

theorem truck_distance (distance : ℝ) (time_minutes : ℝ) (travel_time_hours : ℝ) : 
  distance = 2 ∧ time_minutes = 2.5 ∧ travel_time_hours = 3 →
  (distance / time_minutes) * (travel_time_hours * 60) = 144 := by
sorry

end truck_distance_l3693_369363


namespace stating_snail_reaches_top_l3693_369398

/-- Represents the height of the tree in meters -/
def tree_height : ℕ := 10

/-- Represents the distance the snail climbs during the day in meters -/
def day_climb : ℕ := 4

/-- Represents the distance the snail slips at night in meters -/
def night_slip : ℕ := 3

/-- Calculates the net distance the snail moves in one day -/
def net_daily_progress : ℤ := day_climb - night_slip

/-- Represents the number of days it takes for the snail to reach the top -/
def days_to_reach_top : ℕ := 7

/-- 
Theorem stating that the snail reaches the top of the tree in 7 days
given the defined tree height, day climb, and night slip distances.
-/
theorem snail_reaches_top : 
  (days_to_reach_top - 1) * net_daily_progress + day_climb ≥ tree_height :=
sorry

end stating_snail_reaches_top_l3693_369398


namespace exponential_equation_solution_l3693_369380

theorem exponential_equation_solution (x : ℝ) :
  3^(3*x + 2) = (1 : ℝ) / 27 → x = -(5 : ℝ) / 3 := by
  sorry

end exponential_equation_solution_l3693_369380


namespace farmer_rabbit_problem_l3693_369314

theorem farmer_rabbit_problem :
  ∀ (initial_rabbits : ℕ),
    (∃ (rabbits_per_cage : ℕ),
      initial_rabbits + 6 = 17 * rabbits_per_cage) →
    initial_rabbits = 28 := by
  sorry

end farmer_rabbit_problem_l3693_369314


namespace farm_bird_difference_l3693_369332

/-- Given a farm with chickens, ducks, and geese, calculate the difference between
    the combined number of chickens and geese and the number of ducks. -/
theorem farm_bird_difference (chickens ducks geese : ℕ) : 
  chickens = 42 →
  ducks = 48 →
  geese = chickens →
  chickens + geese - ducks = 36 := by
sorry

end farm_bird_difference_l3693_369332


namespace average_increase_is_four_l3693_369328

/-- Represents a cricket player's statistics -/
structure CricketPlayer where
  innings : ℕ
  totalRuns : ℕ
  nextInningsRuns : ℕ

/-- Calculates the increase in average runs after the next innings -/
def averageIncrease (player : CricketPlayer) : ℚ :=
  let currentAverage : ℚ := player.totalRuns / player.innings
  let newTotalRuns : ℕ := player.totalRuns + player.nextInningsRuns
  let newAverage : ℚ := newTotalRuns / (player.innings + 1)
  newAverage - currentAverage

/-- Theorem: The increase in average runs is 4 for the given conditions -/
theorem average_increase_is_four :
  ∀ (player : CricketPlayer),
    player.innings = 10 →
    player.totalRuns = 400 →
    player.nextInningsRuns = 84 →
    averageIncrease player = 4 := by
  sorry

end average_increase_is_four_l3693_369328


namespace frogs_eaten_by_fish_l3693_369354

/-- The number of flies eaten by each frog per day -/
def flies_per_frog : ℕ := 30

/-- The number of fish eaten by each gharial per day -/
def fish_per_gharial : ℕ := 15

/-- The number of gharials in the swamp -/
def num_gharials : ℕ := 9

/-- The total number of flies eaten per day -/
def total_flies_eaten : ℕ := 32400

/-- The number of frogs each fish needs to eat per day -/
def frogs_per_fish : ℕ := 8

theorem frogs_eaten_by_fish :
  frogs_per_fish = 
    total_flies_eaten / (flies_per_frog * (num_gharials * fish_per_gharial)) :=
by sorry

end frogs_eaten_by_fish_l3693_369354


namespace profit_percentage_calculation_l3693_369361

theorem profit_percentage_calculation (selling_price profit : ℝ) 
  (h1 : selling_price = 900)
  (h2 : profit = 150) :
  (profit / (selling_price - profit)) * 100 = 20 := by
sorry

end profit_percentage_calculation_l3693_369361


namespace second_plant_production_l3693_369369

/-- Represents the production of tomatoes from three plants -/
structure TomatoProduction where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of the tomato production problem -/
def TomatoProblem (p : TomatoProduction) : Prop :=
  p.first = 24 ∧
  p.third = p.second + 2 ∧
  p.first + p.second + p.third = 60

theorem second_plant_production (p : TomatoProduction) 
  (h : TomatoProblem p) : p.first - p.second = 7 :=
by
  sorry

#check second_plant_production

end second_plant_production_l3693_369369


namespace odd_divisors_of_power_minus_one_smallest_odd_divisors_second_smallest_odd_divisor_three_divides_nine_divides_infinitely_many_divisors_l3693_369374

theorem odd_divisors_of_power_minus_one (n : ℕ) :
  Odd n → n ∣ 2023^n - 1 → n ≥ 3 :=
sorry

theorem smallest_odd_divisors :
  (∃ (n : ℕ), Odd n ∧ n ∣ 2023^n - 1 ∧ n < 3) → False :=
sorry

theorem second_smallest_odd_divisor :
  (∃ (n : ℕ), Odd n ∧ n ∣ 2023^n - 1 ∧ 3 < n ∧ n < 9) → False :=
sorry

theorem three_divides : 3 ∣ 2023^3 - 1 :=
sorry

theorem nine_divides : 9 ∣ 2023^9 - 1 :=
sorry

theorem infinitely_many_divisors (k : ℕ) :
  k ≥ 1 → 3^k ∣ 2023^(3^k) - 1 :=
sorry

end odd_divisors_of_power_minus_one_smallest_odd_divisors_second_smallest_odd_divisor_three_divides_nine_divides_infinitely_many_divisors_l3693_369374
