import Mathlib

namespace glass_volume_proof_l832_83292

theorem glass_volume_proof (V : ℝ) 
  (h1 : 0.4 * V = V - (0.6 * V))  -- pessimist's glass is 60% empty
  (h2 : 0.6 * V - 0.4 * V = 46)   -- difference in water volume
  : V = 230 := by
sorry

end glass_volume_proof_l832_83292


namespace expected_cards_theorem_l832_83241

/-- A deck of cards with N cards, of which three are Aces -/
structure Deck :=
  (N : ℕ)
  (ace_count : Fin 3)

/-- The expected number of cards turned over until the second Ace appears -/
def expected_cards_until_second_ace (d : Deck) : ℚ :=
  (d.N + 1) / 2

/-- Theorem stating that the expected number of cards turned over until the second Ace appears is (N+1)/2 -/
theorem expected_cards_theorem (d : Deck) :
  expected_cards_until_second_ace d = (d.N + 1) / 2 := by
  sorry

#check expected_cards_theorem

end expected_cards_theorem_l832_83241


namespace males_count_l832_83283

/-- Represents the population of a village -/
structure Village where
  total_population : ℕ
  num_groups : ℕ
  males_in_one_group : Bool

/-- Theorem: In a village with 520 people divided into 4 equal groups,
    if one group represents all males, then the number of males is 130 -/
theorem males_count (v : Village)
  (h1 : v.total_population = 520)
  (h2 : v.num_groups = 4)
  (h3 : v.males_in_one_group = true) :
  v.total_population / v.num_groups = 130 := by
  sorry

#check males_count

end males_count_l832_83283


namespace expand_and_simplify_expression_l832_83277

theorem expand_and_simplify_expression (x : ℝ) :
  2*x*(3*x^2 - 4*x + 5) - (x^2 - 3*x)*(4*x + 5) = 2*x^3 - x^2 + 25*x := by
  sorry

end expand_and_simplify_expression_l832_83277


namespace certain_number_proof_l832_83264

def w : ℕ := 132

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem certain_number_proof :
  ∃ (n : ℕ), 
    (is_factor (2^5) (n * w)) ∧ 
    (is_factor (3^3) (n * w)) ∧ 
    (is_factor (11^2) (n * w)) ∧
    (∀ (m : ℕ), m < w → ¬(is_factor (2^5) (n * m) ∧ is_factor (3^3) (n * m) ∧ is_factor (11^2) (n * m))) →
  n = 792 :=
sorry

end certain_number_proof_l832_83264


namespace pumpkin_patch_problem_l832_83297

def pumpkin_pie_filling_cans (total_pumpkins : ℕ) (price_per_pumpkin : ℕ) (total_earnings : ℕ) (pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_earnings / price_per_pumpkin
  let remaining_pumpkins := total_pumpkins - pumpkins_sold
  remaining_pumpkins / pumpkins_per_can

theorem pumpkin_patch_problem :
  pumpkin_pie_filling_cans 83 3 96 3 = 17 := by
  sorry

end pumpkin_patch_problem_l832_83297


namespace longer_diagonal_squared_is_80_l832_83230

/-- Represents a parallelogram LMNO with specific properties -/
structure Parallelogram where
  area : ℝ
  xy : ℝ
  zw : ℝ
  (area_positive : area > 0)
  (xy_positive : xy > 0)
  (zw_positive : zw > 0)

/-- The square of the longer diagonal of the parallelogram -/
def longer_diagonal_squared (p : Parallelogram) : ℝ := sorry

/-- Theorem stating the square of the longer diagonal equals 80 -/
theorem longer_diagonal_squared_is_80 (p : Parallelogram) 
  (h1 : p.area = 24) 
  (h2 : p.xy = 8) 
  (h3 : p.zw = 10) : 
  longer_diagonal_squared p = 80 := by sorry

end longer_diagonal_squared_is_80_l832_83230


namespace vertex_of_quadratic_l832_83298

/-- The quadratic function f(x) = 2(x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -3

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-1)^2 - 3 is (1, -3) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end vertex_of_quadratic_l832_83298


namespace initial_candies_are_52_or_56_l832_83275

def initial_candies : Set ℕ :=
  {x : ℕ | 
    -- The number of candies after Tracy ate 1/4
    ∃ (a : ℕ), 3 * x = 4 * a ∧
    -- The number of candies after giving 1/3 to Sam
    ∃ (b : ℕ), 2 * a = 3 * b ∧
    -- The number of candies after Tracy and her dad ate 20
    b ≥ 20 ∧
    -- The number of candies after Tracy's sister took 2 to 6
    ∃ (c : ℕ), b - 20 - c = 4 ∧ 2 ≤ c ∧ c ≤ 6
  }

theorem initial_candies_are_52_or_56 : initial_candies = {52, 56} := by sorry

end initial_candies_are_52_or_56_l832_83275


namespace num_spiders_is_one_l832_83236

/-- The number of spiders in a pet shop. -/
def num_spiders : ℕ :=
  let num_birds : ℕ := 3
  let num_dogs : ℕ := 5
  let num_snakes : ℕ := 4
  let total_legs : ℕ := 34
  let bird_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let snake_legs : ℕ := 0
  let spider_legs : ℕ := 8
  (total_legs - (num_birds * bird_legs + num_dogs * dog_legs + num_snakes * snake_legs)) / spider_legs

theorem num_spiders_is_one : num_spiders = 1 := by
  sorry

end num_spiders_is_one_l832_83236


namespace workshop_salary_problem_l832_83207

theorem workshop_salary_problem (total_workers : Nat) (avg_salary : ℝ) 
  (num_technicians : Nat) (avg_salary_technicians : ℝ) :
  total_workers = 28 →
  avg_salary = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := total_workers * avg_salary
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining := total_salary - total_salary_technicians
  let avg_salary_remaining := total_salary_remaining / remaining_workers
  avg_salary_remaining = 6000 := by
sorry

end workshop_salary_problem_l832_83207


namespace digit_values_divisible_by_99_l832_83210

theorem digit_values_divisible_by_99 (x y : Nat) : 
  (0 ≤ x ∧ x ≤ 9) → 
  (0 ≤ y ∧ y ≤ 9) → 
  (99 ∣ (141000 + 10000*x + 280 + 10*y + 3)) → 
  (x = 4 ∧ y = 4) := by
sorry

end digit_values_divisible_by_99_l832_83210


namespace blocks_per_group_l832_83286

theorem blocks_per_group (total_blocks : ℕ) (num_groups : ℕ) (blocks_per_group : ℕ) :
  total_blocks = 820 →
  num_groups = 82 →
  total_blocks = num_groups * blocks_per_group →
  blocks_per_group = 10 := by
  sorry

end blocks_per_group_l832_83286


namespace bhupathi_abhinav_fraction_l832_83220

theorem bhupathi_abhinav_fraction : 
  ∀ (abhinav bhupathi : ℚ),
  abhinav + bhupathi = 1210 →
  bhupathi = 484 →
  ∃ (x : ℚ), (4 / 15) * abhinav = x * bhupathi ∧ x = 2 / 5 :=
by
  sorry

end bhupathi_abhinav_fraction_l832_83220


namespace opposite_of_negative_one_fourth_l832_83294

theorem opposite_of_negative_one_fourth : 
  (-(-(1/4 : ℚ))) = (1/4 : ℚ) := by sorry

end opposite_of_negative_one_fourth_l832_83294


namespace point_on_line_l832_83212

theorem point_on_line (x₁ n : ℝ) : 
  (x₁ = n / 5 - 2 / 5 ∧ x₁ + 3 = (n + 15) / 5 - 2 / 5) → 
  x₁ = n / 5 - 2 / 5 := by
  sorry

end point_on_line_l832_83212


namespace slide_ratio_problem_l832_83280

/-- Given that x boys initially went down a slide, y more boys joined them later,
    and the ratio of boys who went down the slide to boys who watched (z) is 5:3,
    prove that z = 21 when x = 22 and y = 13. -/
theorem slide_ratio_problem (x y : ℕ) (z : ℚ) 
    (h1 : x = 22)
    (h2 : y = 13)
    (h3 : (5 : ℚ) / 3 = (x + y : ℚ) / z) : 
  z = 21 := by
  sorry

end slide_ratio_problem_l832_83280


namespace blue_tiles_in_45th_row_l832_83209

/-- Calculates the total number of tiles in a row given the row number. -/
def totalTiles (n : ℕ) : ℕ := 2 * n - 1

/-- Calculates the number of blue tiles in a row given the total number of tiles. -/
def blueTiles (total : ℕ) : ℕ := (total - 1) / 2

theorem blue_tiles_in_45th_row :
  blueTiles (totalTiles 45) = 44 := by
  sorry

end blue_tiles_in_45th_row_l832_83209


namespace animal_lifespans_l832_83231

theorem animal_lifespans (bat hamster frog tortoise : ℝ) : 
  hamster = bat - 6 →
  frog = 4 * hamster →
  tortoise = 2 * bat →
  bat + hamster + frog + tortoise = 62 →
  bat = 11.5 := by
sorry

end animal_lifespans_l832_83231


namespace unique_configuration_l832_83253

-- Define the type for statements
inductive Statement
| one_false : Statement
| two_false : Statement
| three_false : Statement
| four_false : Statement
| one_true : Statement

-- Define a function to evaluate the truth value of a statement
def evaluate (s : Statement) (true_count : Nat) : Prop :=
  match s with
  | Statement.one_false => true_count = 4
  | Statement.two_false => true_count = 3
  | Statement.three_false => true_count = 2
  | Statement.four_false => true_count = 1
  | Statement.one_true => true_count = 1

-- Define the card as a list of statements
def card : List Statement := [
  Statement.one_false,
  Statement.two_false,
  Statement.three_false,
  Statement.four_false,
  Statement.one_true
]

-- Theorem: There exists a unique configuration with exactly one true statement
theorem unique_configuration :
  ∃! true_count : Nat,
    true_count ≤ 5 ∧
    true_count > 0 ∧
    (∀ s ∈ card, evaluate s true_count ↔ s = Statement.one_true) :=
by sorry

end unique_configuration_l832_83253


namespace units_digit_of_n_l832_83240

/-- Given two natural numbers m and n, returns true if m has a units digit of 3 -/
def has_units_digit_3 (m : ℕ) : Prop :=
  m % 10 = 3

/-- Given a natural number n, returns its units digit -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : has_units_digit_3 m) :
  units_digit n = 2 := by
sorry

end units_digit_of_n_l832_83240


namespace back_seat_ticket_cost_l832_83219

/-- Proves that the cost of back seat tickets is $45 given the concert conditions -/
theorem back_seat_ticket_cost
  (total_seats : ℕ)
  (main_seat_cost : ℕ)
  (total_revenue : ℕ)
  (back_seat_sold : ℕ)
  (h_total_seats : total_seats = 20000)
  (h_main_seat_cost : main_seat_cost = 55)
  (h_total_revenue : total_revenue = 955000)
  (h_back_seat_sold : back_seat_sold = 14500) :
  (total_revenue - (total_seats - back_seat_sold) * main_seat_cost) / back_seat_sold = 45 :=
by sorry

end back_seat_ticket_cost_l832_83219


namespace b_spending_percentage_l832_83257

/-- Proves that B spends 85% of her salary given the conditions of the problem -/
theorem b_spending_percentage (total_salary : ℕ) (a_spending_rate : ℚ) (b_salary : ℕ) :
  total_salary = 14000 →
  a_spending_rate = 4/5 →
  b_salary = 8000 →
  let a_salary := total_salary - b_salary
  let a_savings := a_salary * (1 - a_spending_rate)
  let b_savings := a_savings
  let b_spending_rate := 1 - (b_savings / b_salary)
  b_spending_rate = 17/20 := by
sorry

#eval (17 : ℚ) / 20  -- Should output 0.85

end b_spending_percentage_l832_83257


namespace regular_polygon_with_150_degree_angles_has_12_sides_l832_83267

/-- A regular polygon with interior angles of 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150) →
    (180 * (n - 2) : ℝ) = (n * 150 : ℝ) →
    n = 12 :=
by
  sorry

end regular_polygon_with_150_degree_angles_has_12_sides_l832_83267


namespace optimal_meeting_time_l832_83272

/-- The optimal meeting time for a pedestrian and cyclist on a circular path -/
theorem optimal_meeting_time 
  (pedestrian_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (path_length : ℝ) 
  (walked_distance : ℝ) 
  (remaining_distance : ℝ) 
  (h1 : pedestrian_speed = 6.5)
  (h2 : cyclist_speed = 20)
  (h3 : path_length = 4 * Real.pi)
  (h4 : walked_distance = 6.5)
  (h5 : remaining_distance = 4 * Real.pi - 6.5)
  (h6 : walked_distance = pedestrian_speed * 1) -- 1 hour of walking
  : ∃ (t : ℝ), t = (155 - 28 * Real.pi) / 172 ∧ 
    t = min (remaining_distance / (pedestrian_speed + cyclist_speed))
            ((path_length - walked_distance) / (pedestrian_speed + cyclist_speed)) := by
  sorry

end optimal_meeting_time_l832_83272


namespace complex_power_sum_l832_83259

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^98 + z^99 + z^100 + z^101 + z^102 = -z := by
  sorry

end complex_power_sum_l832_83259


namespace parabola_symmetric_axis_given_parabola_symmetric_axis_l832_83291

/-- The symmetric axis of a parabola y = (x - h)^2 + k is x = h -/
theorem parabola_symmetric_axis (h k : ℝ) :
  let f : ℝ → ℝ := λ x => (x - h)^2 + k
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → |x - a| = |y - a| :=
by sorry

/-- The symmetric axis of the parabola y = (x - 2)^2 + 1 is x = 2 -/
theorem given_parabola_symmetric_axis :
  let f : ℝ → ℝ := λ x => (x - 2)^2 + 1
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → |x - a| = |y - a| ∧ a = 2 :=
by sorry

end parabola_symmetric_axis_given_parabola_symmetric_axis_l832_83291


namespace convex_polygon_diagonal_triangles_l832_83239

/-- Represents a convex polygon with diagonals drawn to create triangles -/
structure ConvexPolygonWithDiagonals where
  sides : ℕ
  triangles : ℕ
  diagonalTriangles : ℕ

/-- The property that needs to be proven -/
def impossibleHalfDiagonalTriangles (p : ConvexPolygonWithDiagonals) : Prop :=
  p.sides = 2016 ∧ p.triangles = 2014 → p.diagonalTriangles ≠ 1007

theorem convex_polygon_diagonal_triangles :
  ∀ p : ConvexPolygonWithDiagonals, impossibleHalfDiagonalTriangles p :=
sorry

end convex_polygon_diagonal_triangles_l832_83239


namespace distinct_roots_quadratic_l832_83281

theorem distinct_roots_quadratic (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 - m*x₁ - 2 = 0) ∧ 
  (x₂^2 - m*x₂ - 2 = 0) := by
sorry

end distinct_roots_quadratic_l832_83281


namespace natural_number_representation_l832_83203

theorem natural_number_representation (k : ℕ) : 
  ∃ n : ℕ, k = 3*n ∨ k = 3*n + 1 ∨ k = 3*n + 2 :=
sorry

end natural_number_representation_l832_83203


namespace wall_width_is_0_05_meters_l832_83242

-- Define the brick dimensions in meters
def brick_length : Real := 0.21
def brick_width : Real := 0.10
def brick_height : Real := 0.08

-- Define the wall dimensions
def wall_length : Real := 9
def wall_height : Real := 18.5

-- Define the number of bricks
def num_bricks : Real := 4955.357142857142

-- Theorem to prove
theorem wall_width_is_0_05_meters :
  let brick_volume := brick_length * brick_width * brick_height
  let total_brick_volume := brick_volume * num_bricks
  let wall_width := total_brick_volume / (wall_length * wall_height)
  wall_width = 0.05 := by sorry

end wall_width_is_0_05_meters_l832_83242


namespace rabbit_cleaner_amount_l832_83250

/-- The amount of cleaner used for a dog stain in ounces -/
def dog_cleaner : ℝ := 6

/-- The amount of cleaner used for a cat stain in ounces -/
def cat_cleaner : ℝ := 4

/-- The total amount of cleaner used in ounces -/
def total_cleaner : ℝ := 49

/-- The number of dogs -/
def num_dogs : ℕ := 6

/-- The number of cats -/
def num_cats : ℕ := 3

/-- The number of rabbits -/
def num_rabbits : ℕ := 1

/-- The amount of cleaner used for a rabbit stain in ounces -/
def rabbit_cleaner : ℝ := 1

theorem rabbit_cleaner_amount :
  dog_cleaner * num_dogs + cat_cleaner * num_cats + rabbit_cleaner * num_rabbits = total_cleaner := by
  sorry

end rabbit_cleaner_amount_l832_83250


namespace right_triangle_PR_length_l832_83226

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  sinR : ℝ
  angle_Q_is_right : True  -- Represents ∠Q = 90°

-- State the theorem
theorem right_triangle_PR_length 
  (triangle : RightTriangle) 
  (h1 : triangle.PQ = 9) 
  (h2 : triangle.sinR = 3/5) : 
  triangle.PR = 15 := by
sorry

end right_triangle_PR_length_l832_83226


namespace simplify_expression_l832_83227

theorem simplify_expression (r : ℝ) : 120 * r - 68 * r + 15 * r = 67 * r := by
  sorry

end simplify_expression_l832_83227


namespace more_girls_than_boys_l832_83229

theorem more_girls_than_boys 
  (total_pupils : ℕ) 
  (girls : ℕ) 
  (h1 : total_pupils = 1455)
  (h2 : girls = 868)
  (h3 : girls > total_pupils - girls) : 
  girls - (total_pupils - girls) = 281 :=
by
  sorry

end more_girls_than_boys_l832_83229


namespace parabola_sum_l832_83228

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-6) = 8 → p.x_coord (-4) = 10 → p.a + p.b + p.c = 32.5 := by
  sorry

end parabola_sum_l832_83228


namespace cube_frame_wire_ratio_l832_83295

/-- The ratio of wire lengths used by two people constructing cube frames -/
theorem cube_frame_wire_ratio : 
  ∀ (wire_a wire_b : ℕ) (pieces_a : ℕ) (volume : ℕ),
  wire_a = 8 →
  pieces_a = 12 →
  wire_b = 2 →
  volume = wire_a^3 →
  (wire_a * pieces_a) / (wire_b * 12 * volume) = 1 / 128 :=
by sorry

end cube_frame_wire_ratio_l832_83295


namespace sticks_per_pack_l832_83261

/-- Represents the number of packs in a carton -/
def packs_per_carton : ℕ := 5

/-- Represents the number of cartons in a brown box -/
def cartons_per_box : ℕ := 4

/-- Represents the total number of sticks in all brown boxes -/
def total_sticks : ℕ := 480

/-- Represents the total number of brown boxes -/
def total_boxes : ℕ := 8

/-- Theorem stating that the number of sticks in each pack is 3 -/
theorem sticks_per_pack : 
  total_sticks / (total_boxes * cartons_per_box * packs_per_carton) = 3 := by
  sorry


end sticks_per_pack_l832_83261


namespace computer_table_markup_l832_83262

/-- Calculate the percentage markup given the selling price and cost price -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem: The percentage markup for a computer table with selling price 3000 and cost price 2500 is 20% -/
theorem computer_table_markup :
  percentage_markup 3000 2500 = 20 := by
  sorry

end computer_table_markup_l832_83262


namespace pool_filling_rate_l832_83247

/-- Given a pool with the following properties:
  * Capacity: 60 gallons
  * Leak rate: 0.1 gallons per minute
  * Filling time: 40 minutes
  Prove that the rate at which water is provided to fill the pool is 1.6 gallons per minute. -/
theorem pool_filling_rate 
  (capacity : ℝ) 
  (leak_rate : ℝ) 
  (filling_time : ℝ) 
  (h1 : capacity = 60) 
  (h2 : leak_rate = 0.1) 
  (h3 : filling_time = 40) : 
  ∃ (fill_rate : ℝ), 
    fill_rate = 1.6 ∧ 
    (fill_rate - leak_rate) * filling_time = capacity :=
by sorry

end pool_filling_rate_l832_83247


namespace cannot_tile_8x9_with_6x1_l832_83225

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a rectangular tile -/
structure Tile :=
  (length : ℕ)
  (width : ℕ)

/-- Defines what it means for a board to be tileable by a given tile -/
def is_tileable (b : Board) (t : Tile) : Prop :=
  ∃ (n : ℕ), n * (t.length * t.width) = b.rows * b.cols ∧
  (t.length ∣ b.rows ∨ t.length ∣ b.cols) ∧
  (t.width ∣ b.rows ∨ t.width ∣ b.cols)

/-- The main theorem stating that an 8x9 board cannot be tiled with 6x1 tiles -/
theorem cannot_tile_8x9_with_6x1 :
  ¬ is_tileable (Board.mk 8 9) (Tile.mk 6 1) :=
sorry

end cannot_tile_8x9_with_6x1_l832_83225


namespace train_cars_problem_l832_83289

theorem train_cars_problem (passenger_cars cargo_cars : ℕ) : 
  cargo_cars = passenger_cars / 2 + 3 →
  passenger_cars + cargo_cars + 2 = 71 →
  passenger_cars = 44 := by
  sorry

end train_cars_problem_l832_83289


namespace douyin_sales_and_profit_l832_83205

/-- Represents an e-commerce platform selling a small commodity. -/
structure ECommercePlatform where
  cost_price : ℕ
  initial_price : ℕ
  initial_volume : ℕ
  price_decrease : ℕ
  volume_increase : ℕ

/-- Calculates the daily sales volume for a given selling price. -/
def daily_sales_volume (platform : ECommercePlatform) (selling_price : ℕ) : ℕ :=
  platform.initial_volume + 
    (platform.initial_price - selling_price) / platform.price_decrease * platform.volume_increase

/-- Calculates the daily profit for a given selling price. -/
def daily_profit (platform : ECommercePlatform) (selling_price : ℕ) : ℕ :=
  (selling_price - platform.cost_price) * daily_sales_volume platform selling_price

/-- The e-commerce platform with given conditions. -/
def douyin_platform : ECommercePlatform := {
  cost_price := 40
  initial_price := 60
  initial_volume := 20
  price_decrease := 5
  volume_increase := 10
}

theorem douyin_sales_and_profit :
  (daily_sales_volume douyin_platform 50 = 40) ∧
  (∃ (price : ℕ), daily_profit douyin_platform price = 448 ∧
    ∀ (p : ℕ), daily_profit douyin_platform p = 448 → p ≥ price) :=
by sorry

end douyin_sales_and_profit_l832_83205


namespace quadratic_inequality_l832_83273

theorem quadratic_inequality (a b c d : ℝ) 
  (h1 : b > d) 
  (h2 : b > 0) 
  (h3 : d > 0) 
  (h4 : Real.sqrt (a^2 - 4*b) > Real.sqrt (c^2 - 4*d)) : 
  a^2 - c^2 > b - d := by sorry

end quadratic_inequality_l832_83273


namespace expression_equality_l832_83208

theorem expression_equality : 3 * 2020 + 2 * 2020 - 4 * 2020 = 2020 := by
  sorry

end expression_equality_l832_83208


namespace continuous_function_satisfying_integral_equation_is_constant_l832_83278

/-- A continuous function satisfying the given integral equation is constant -/
theorem continuous_function_satisfying_integral_equation_is_constant 
  (f : ℝ → ℝ) (hf : Continuous f) 
  (h : ∀ a b : ℝ, (a^2 + a*b + b^2) * ∫ x in a..b, f x = 3 * ∫ x in a..b, x^2 * f x) : 
  ∃ C : ℝ, ∀ x : ℝ, f x = C := by
sorry

end continuous_function_satisfying_integral_equation_is_constant_l832_83278


namespace age_difference_l832_83222

/-- The difference in ages between (x + y) and (y + z) is 12 years, given that z is 12 years younger than x -/
theorem age_difference (x y z : ℕ) (h : z = x - 12) :
  (x + y) - (y + z) = 12 := by
  sorry

end age_difference_l832_83222


namespace ring_toss_total_l832_83243

/-- Calculates the total number of rings used in a ring toss game -/
def total_rings (rings_per_game : ℕ) (games_played : ℕ) : ℕ :=
  rings_per_game * games_played

/-- Theorem: Given 6 rings per game and 8 games played, the total rings used is 48 -/
theorem ring_toss_total :
  total_rings 6 8 = 48 := by
  sorry

end ring_toss_total_l832_83243


namespace backpack_profit_analysis_l832_83237

/-- Represents the daily profit function for backpack sales -/
def daily_profit (x : ℝ) : ℝ := -x^2 + 90*x - 1800

/-- Represents the daily sales quantity function -/
def sales_quantity (x : ℝ) : ℝ := -x + 60

theorem backpack_profit_analysis 
  (cost_price : ℝ) 
  (price_range : Set ℝ) 
  (max_price : ℝ) 
  (target_profit : ℝ) :
  cost_price = 30 →
  price_range = {x : ℝ | 30 ≤ x ∧ x ≤ 60} →
  max_price = 48 →
  target_profit = 200 →
  (∀ x ∈ price_range, daily_profit x = (x - cost_price) * sales_quantity x) ∧
  (∃ x ∈ price_range, x ≤ max_price ∧ daily_profit x = target_profit ∧ x = 40) ∧
  (∃ x ∈ price_range, ∀ y ∈ price_range, daily_profit x ≥ daily_profit y ∧ 
    x = 45 ∧ daily_profit x = 225) :=
by sorry

end backpack_profit_analysis_l832_83237


namespace black_cells_remain_even_one_black_cell_impossible_l832_83284

/-- Represents a chessboard -/
structure Chessboard :=
  (black_cells : ℕ)

/-- Represents a repainting operation on a 2x2 square -/
def repaint (board : Chessboard) : Chessboard :=
  { black_cells := board.black_cells + (4 - 2 * (board.black_cells % 4)) }

/-- Initial chessboard state -/
def initial_board : Chessboard :=
  { black_cells := 32 }

/-- Theorem stating that the number of black cells remains even after any number of repainting operations -/
theorem black_cells_remain_even (n : ℕ) :
  ∀ (board : Chessboard),
  (board.black_cells % 2 = 0) →
  ((repaint^[n] board).black_cells % 2 = 0) :=
sorry

/-- Main theorem: It's impossible to have exactly one black cell after repainting operations -/
theorem one_black_cell_impossible :
  ¬ ∃ (n : ℕ), (repaint^[n] initial_board).black_cells = 1 :=
sorry

end black_cells_remain_even_one_black_cell_impossible_l832_83284


namespace total_students_presentation_l832_83255

/-- The total number of students presenting given Eunjeong's position and students after her -/
def total_students (eunjeong_position : Nat) (students_after : Nat) : Nat :=
  (eunjeong_position - 1) + 1 + students_after

/-- Theorem stating the total number of students presenting -/
theorem total_students_presentation : total_students 6 7 = 13 := by
  sorry

end total_students_presentation_l832_83255


namespace f_negative_one_equals_negative_twelve_l832_83224

def f (x : ℝ) : ℝ := sorry

theorem f_negative_one_equals_negative_twelve
  (h_odd : ∀ x, f x = -f (-x))
  (h_nonneg : ∀ x ≥ 0, ∃ a : ℝ, f x = a^(x+1) - 4) :
  f (-1) = -12 := by sorry

end f_negative_one_equals_negative_twelve_l832_83224


namespace sydney_texts_total_l832_83221

/-- The number of texts Sydney sends to each person on Monday -/
def monday_texts : ℕ := 5

/-- The number of texts Sydney sends to each person on Tuesday -/
def tuesday_texts : ℕ := 15

/-- The number of people Sydney sends texts to -/
def num_recipients : ℕ := 2

/-- The total number of texts Sydney sent over both days -/
def total_texts : ℕ := (monday_texts * num_recipients) + (tuesday_texts * num_recipients)

theorem sydney_texts_total : total_texts = 40 := by
  sorry

end sydney_texts_total_l832_83221


namespace min_value_expression_l832_83213

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 18) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 22 ∧
  ∃ x₀ > 4, (x₀ + 18) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 22 := by
  sorry

end min_value_expression_l832_83213


namespace power_of_625_four_fifths_l832_83206

theorem power_of_625_four_fifths :
  (625 : ℝ) ^ (4/5 : ℝ) = 125 * (5 : ℝ) ^ (1/5 : ℝ) :=
by
  sorry

end power_of_625_four_fifths_l832_83206


namespace derivative_of_f_l832_83218

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^2 - 4

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 10 * x^4 - 6 * x := by
  sorry

end derivative_of_f_l832_83218


namespace pre_bought_ticket_price_l832_83282

/-- The price of pre-bought plane tickets is $155 -/
theorem pre_bought_ticket_price :
  ∀ (pre_bought_price : ℕ) (pre_bought_quantity : ℕ) (gate_price : ℕ) (gate_quantity : ℕ) (price_difference : ℕ),
  pre_bought_quantity = 20 →
  gate_quantity = 30 →
  gate_price = 200 →
  gate_quantity * gate_price = pre_bought_quantity * pre_bought_price + price_difference →
  price_difference = 2900 →
  pre_bought_price = 155 :=
by sorry

end pre_bought_ticket_price_l832_83282


namespace number_problem_l832_83293

theorem number_problem (x : ℚ) :
  (35 / 100) * x = (25 / 100) * 40 → x = 200 / 7 := by
  sorry

end number_problem_l832_83293


namespace gain_percent_calculation_l832_83246

theorem gain_percent_calculation (C S : ℝ) (h : C > 0) :
  50 * C = 45 * S → (S - C) / C * 100 = 100 / 9 := by
  sorry

end gain_percent_calculation_l832_83246


namespace triangle_properties_l832_83276

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.a + t.c = t.b * (Real.cos t.C + Real.sqrt 3 * Real.sin t.C))
  (h2 : t.b = 2) : 
  t.B = π / 3 ∧ 
  ∀ (s : Triangle), s.b = 2 → 
    Real.sqrt 3 / 4 * s.a * s.c * Real.sin s.B ≤ Real.sqrt 3 := by
  sorry

end triangle_properties_l832_83276


namespace alice_and_dave_weight_l832_83274

theorem alice_and_dave_weight
  (alice_bob : ℝ)
  (bob_charlie : ℝ)
  (charlie_dave : ℝ)
  (h1 : alice_bob = 230)
  (h2 : bob_charlie = 220)
  (h3 : charlie_dave = 250) :
  ∃ (alice dave : ℝ), alice + dave = 260 :=
by
  sorry

end alice_and_dave_weight_l832_83274


namespace census_suitable_for_electricity_usage_l832_83235

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents a survey population -/
structure Population where
  size : ℕ
  is_small : Bool
  is_manageable : Bool

/-- Represents a survey -/
structure Survey where
  population : Population
  method : SurveyMethod
  is_practical : Bool

/-- Theorem: A census method is most suitable for investigating the monthly average 
    electricity usage of 10 households in a residential building -/
theorem census_suitable_for_electricity_usage : 
  ∀ (p : Population) (s : Survey),
  p.size = 10 → 
  p.is_small = true → 
  p.is_manageable = true → 
  s.population = p → 
  s.is_practical = true → 
  s.method = SurveyMethod.Census :=
by sorry

end census_suitable_for_electricity_usage_l832_83235


namespace inscribed_square_area_l832_83223

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 16

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square inscribed in the region bound by the parabola and x-axis -/
structure InscribedSquare where
  E : Point
  F : Point
  G : Point
  H : Point
  -- E and F are on x-axis
  h1 : E.y = 0
  h2 : F.y = 0
  -- G is on the parabola
  h3 : G.y = parabola G.x
  -- EFGH forms a square
  h4 : (F.x - E.x)^2 + (G.y - F.y)^2 = (G.x - F.x)^2 + (G.y - F.y)^2

/-- The theorem stating that the area of the inscribed square is 16 -/
theorem inscribed_square_area (s : InscribedSquare) : (s.F.x - s.E.x)^2 = 16 := by
  sorry

end inscribed_square_area_l832_83223


namespace inequality_proof_l832_83254

theorem inequality_proof (a b x : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hn : 2 ≤ n) 
  (h : x^n ≤ a*x + b) : 
  x < (2*a)^(1/(n-1 : ℝ)) + (2*b)^(1/n) := by
  sorry

end inequality_proof_l832_83254


namespace mn_positive_necessary_not_sufficient_l832_83204

/-- A curve represented by the equation mx^2 + ny^2 = 1 -/
structure Curve (m n : ℝ) where
  equation : ∀ (x y : ℝ), m * x^2 + n * y^2 = 1

/-- Predicate to check if a curve is an ellipse -/
def IsEllipse (c : Curve m n) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is a necessary but not sufficient condition for the curve to be an ellipse -/
theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (∀ (c : Curve m n), IsEllipse c → m * n > 0) ∧
  ¬(∀ (c : Curve m n), m * n > 0 → IsEllipse c) :=
sorry

end mn_positive_necessary_not_sufficient_l832_83204


namespace triangle_side_b_value_l832_83211

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.sin t.B = Real.sin t.A + Real.sin t.C ∧
  Real.cos t.B = 3/5 ∧
  1/2 * t.a * t.c * Real.sin t.B = 4

-- Theorem statement
theorem triangle_side_b_value (t : Triangle) 
  (h : triangle_conditions t) : t.b = 4 * Real.sqrt 6 / 3 := by
  sorry

end triangle_side_b_value_l832_83211


namespace unique_solution_condition_l832_83256

theorem unique_solution_condition (A B : ℝ) :
  (∀ x y : ℝ, A * x + B * ⌊x⌋ = A * y + B * ⌊y⌋ → x = y) ↔ 
  (A = 0 ∨ -2 < B / A ∧ B / A < 0) :=
sorry

end unique_solution_condition_l832_83256


namespace solutions_of_quadratic_l832_83268

theorem solutions_of_quadratic (x : ℝ) : x^2 = 16*x ↔ x = 0 ∨ x = 16 := by
  sorry

end solutions_of_quadratic_l832_83268


namespace pet_store_cats_l832_83248

theorem pet_store_cats (initial_birds initial_puppies initial_spiders : ℕ)
  (sold_birds adopted_puppies loose_spiders : ℕ)
  (total_left : ℕ) :
  initial_birds = 12 →
  initial_puppies = 9 →
  initial_spiders = 15 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  total_left = 25 →
  ∃ initial_cats : ℕ,
    initial_cats = 5 ∧
    total_left = initial_birds - sold_birds +
                 initial_puppies - adopted_puppies +
                 initial_cats +
                 initial_spiders - loose_spiders :=
by sorry

end pet_store_cats_l832_83248


namespace triangle_third_side_length_l832_83296

theorem triangle_third_side_length 
  (a b : ℝ) 
  (cos_C : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 3) 
  (h3 : cos_C = -3/5) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*cos_C ∧ c = 2 * Real.sqrt 13 :=
sorry

end triangle_third_side_length_l832_83296


namespace circle_and_reflection_theorem_l832_83279

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a b r : ℝ), (x - a)^2 + (y - b)^2 = r^2 ∧ 2*a - b - 4 = 0

-- Define the points A and B
def point_A : ℝ × ℝ := (3, 0)
def point_B : ℝ × ℝ := (5, 2)

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the point M
def point_M : ℝ × ℝ := (-4, -3)

-- Define the theorem
theorem circle_and_reflection_theorem :
  (circle_C point_A.1 point_A.2) ∧
  (circle_C point_B.1 point_B.2) ∧
  (∃ (x y : ℝ), reflection_line x y ∧ 
    ∃ (t : ℝ), (1 - t) * point_M.1 + t * x = -4 ∧ (1 - t) * point_M.2 + t * y = -3 ∧
    circle_C x y) →
  (∀ (x y : ℝ), circle_C x y ↔ (x - 3)^2 + (y - 2)^2 = 4) ∧
  (∃ (k : ℝ), ∀ (x y : ℝ), (x = 1 ∨ 12*x - 5*y - 52 = 0) ↔ 
    (∃ (t : ℝ), x = (1 - t) * 1 + t * point_M.1 ∧ y = (1 - t) * (-8) + t * point_M.2)) :=
by sorry

end circle_and_reflection_theorem_l832_83279


namespace greatest_prime_factor_f_36_l832_83202

def f (m : ℕ) : ℕ := Finset.prod (Finset.filter (λ x => Even x) (Finset.range (m + 1))) id

theorem greatest_prime_factor_f_36 :
  ∃ (p : ℕ), Prime p ∧ p ∣ f 36 ∧ ∀ (q : ℕ), Prime q → q ∣ f 36 → q ≤ p :=
by sorry

end greatest_prime_factor_f_36_l832_83202


namespace unique_real_solution_and_two_imaginary_l832_83285

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x - 1)^2
def equation2 (x y : ℂ) : Prop := x * y + y = 2

-- Define a solution pair
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

-- Define the set of all solution pairs
def solution_set : Set (ℂ × ℂ) := {p | is_solution p.1 p.2}

-- State the theorem
theorem unique_real_solution_and_two_imaginary :
  ∃! (x y : ℝ), is_solution x y ∧
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
    is_solution (x + a * I) (y + b * I) ∧
    is_solution (x + c * I) (y + d * I) ∧
    (x + a * I ≠ x + c * I) ∧
    (∀ (u v : ℂ), is_solution u v → (u = x ∧ v = y) ∨ 
                                    (u = x + a * I ∧ v = y + b * I) ∨ 
                                    (u = x + c * I ∧ v = y + d * I)) :=
by sorry

end unique_real_solution_and_two_imaginary_l832_83285


namespace remainder_theorem_l832_83249

theorem remainder_theorem (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) (h4 : u + v < y) :
  (x + 3 * u * y + u) % y = u + v :=
sorry

end remainder_theorem_l832_83249


namespace acid_percentage_proof_l832_83270

/-- Given a solution with 1.4 litres of pure acid in 4 litres total volume,
    prove that the percentage of pure acid is 35%. -/
theorem acid_percentage_proof : 
  let pure_acid_volume : ℝ := 1.4
  let total_solution_volume : ℝ := 4
  let percentage_pure_acid : ℝ := (pure_acid_volume / total_solution_volume) * 100
  percentage_pure_acid = 35 := by
  sorry

end acid_percentage_proof_l832_83270


namespace problem_statement_l832_83200

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 := by
  sorry

end problem_statement_l832_83200


namespace product_of_sines_l832_83232

theorem product_of_sines (π : Real) : 
  (1 + Real.sin (π / 12)) * (1 + Real.sin (5 * π / 12)) * 
  (1 + Real.sin (7 * π / 12)) * (1 + Real.sin (11 * π / 12)) = 
  (17 / 16 + 2 * Real.sin (π / 12)) * (17 / 16 + 2 * Real.sin (5 * π / 12)) := by
  sorry

end product_of_sines_l832_83232


namespace kim_money_amount_l832_83258

theorem kim_money_amount (sal phil : ℝ) (h1 : sal = 0.8 * phil) (h2 : sal + phil = 1.8) : 
  1.4 * sal = 1.12 := by
  sorry

end kim_money_amount_l832_83258


namespace jerry_zinc_consumption_l832_83299

/-- Calculates the total milligrams of zinc consumed from antacids -/
def total_zinc_mg (large_antacid_count : ℕ) (large_antacid_weight : ℝ) (large_antacid_zinc_percent : ℝ)
                  (small_antacid_count : ℕ) (small_antacid_weight : ℝ) (small_antacid_zinc_percent : ℝ) : ℝ :=
  ((large_antacid_count : ℝ) * large_antacid_weight * large_antacid_zinc_percent +
   (small_antacid_count : ℝ) * small_antacid_weight * small_antacid_zinc_percent) * 1000

/-- Theorem stating the total zinc consumed by Jerry -/
theorem jerry_zinc_consumption :
  total_zinc_mg 2 2 0.05 3 1 0.15 = 650 := by
  sorry

end jerry_zinc_consumption_l832_83299


namespace sum_always_positive_l832_83238

-- Define a monotonically increasing odd function
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end sum_always_positive_l832_83238


namespace problem_solution_l832_83260

theorem problem_solution (x y : ℝ) (some_number : ℝ) 
  (h1 : x + 3 * y = some_number) 
  (h2 : y = 10) 
  (h3 : x = 3) : 
  some_number = 33 := by
  sorry

end problem_solution_l832_83260


namespace value_of_X_l832_83290

theorem value_of_X : ∃ X : ℚ, (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ X = 60 := by
  sorry

end value_of_X_l832_83290


namespace ice_cream_theorem_ice_cream_distribution_count_l832_83252

def ice_cream_distribution (n : ℕ) : ℕ :=
  (Nat.choose (n + 2) 2)

theorem ice_cream_theorem :
  ice_cream_distribution 62 = 2016 :=
by sorry

/-- Given:
    - 62 trainees choose from 5 ice cream flavors
    - Bubblegum flavor (r) at least as popular as tabasco (t)
    - Number of students choosing cactus flavor (a) is a multiple of 6
    - At most 5 students chose lemon basil flavor (b)
    - At most 1 student chose foie gras flavor (c)
    Prove: The number of possible distributions is 2016 -/
theorem ice_cream_distribution_count :
  ∃ (r t a b c : ℕ),
    r + t + a + b + c = 62 ∧
    r ≥ t ∧
    a % 6 = 0 ∧
    b ≤ 5 ∧
    c ≤ 1 ∧
    ice_cream_distribution 62 = 2016 :=
by sorry

end ice_cream_theorem_ice_cream_distribution_count_l832_83252


namespace max_area_central_angle_l832_83244

/-- The circumference of the sector -/
def circumference : ℝ := 40

/-- The radius of the sector -/
noncomputable def radius : ℝ := sorry

/-- The arc length of the sector -/
noncomputable def arc_length : ℝ := sorry

/-- The area of the sector -/
noncomputable def area (r : ℝ) : ℝ := 20 * r - r^2

/-- The central angle of the sector -/
noncomputable def central_angle : ℝ := sorry

/-- Theorem: The central angle that maximizes the area of a sector with circumference 40 is 2 radians -/
theorem max_area_central_angle :
  circumference = 2 * radius + arc_length →
  arc_length = central_angle * radius →
  central_angle = 2 ∧ IsLocalMax area radius :=
sorry

end max_area_central_angle_l832_83244


namespace caitlin_age_l832_83233

/-- Prove that Caitlin's age is 29 years -/
theorem caitlin_age :
  let aunt_anna_age : ℕ := 54
  let brianna_age : ℕ := (2 * aunt_anna_age) / 3
  let caitlin_age : ℕ := brianna_age - 7
  caitlin_age = 29 := by
  sorry

end caitlin_age_l832_83233


namespace quadratic_equations_solutions_l832_83251

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℚ, (3 * x₁^2 - 5 * x₁ - 2 = 0 ∧ x₁ = 2) ∧
                (3 * x₂^2 - 5 * x₂ - 2 = 0 ∧ x₂ = -1/3)) ∧
  (∃ y₁ y₂ : ℚ, (3 * y₁ * (y₁ - 1) = 2 - 2 * y₁ ∧ y₁ = 1) ∧
                (3 * y₂ * (y₂ - 1) = 2 - 2 * y₂ ∧ y₂ = -2/3)) :=
by
  sorry

end quadratic_equations_solutions_l832_83251


namespace roller_plate_acceleration_l832_83234

noncomputable def plate_acceleration (R r : ℝ) (m : ℝ) (α : ℝ) (g : ℝ) : ℝ :=
  g * Real.sqrt ((1 - Real.cos α) / 2)

noncomputable def plate_direction (α : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 - Real.cos α) / 2))

theorem roller_plate_acceleration 
  (R : ℝ) 
  (r : ℝ) 
  (m : ℝ) 
  (α : ℝ) 
  (g : ℝ) 
  (h_R : R = 1) 
  (h_r : r = 0.4) 
  (h_m : m = 150) 
  (h_α : α = Real.arccos 0.68) 
  (h_g : g = 10) :
  plate_acceleration R r m α g = 4 ∧ 
  plate_direction α = Real.arcsin 0.4 ∧
  plate_acceleration R r m α g = g * Real.sin (α / 2) :=
by
  sorry

#check roller_plate_acceleration

end roller_plate_acceleration_l832_83234


namespace intersection_counts_theorem_l832_83266

/-- Represents a line in 2D space -/
structure Line :=
  (a b c : ℝ)

/-- Represents a circle in 2D space -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- Represents the number of intersection points -/
inductive IntersectionCount : Type
  | Zero
  | One
  | Two
  | Three
  | Four

/-- Given two intersecting lines and a circle, this function returns the possible numbers of intersection points -/
def possibleIntersectionCounts (l1 l2 : Line) (c : Circle) : Set IntersectionCount :=
  sorry

/-- Theorem stating that the possible numbers of intersection points are 0, 1, 2, 3, and 4 -/
theorem intersection_counts_theorem (l1 l2 : Line) (c : Circle) :
  possibleIntersectionCounts l1 l2 c = {IntersectionCount.Zero, IntersectionCount.One, IntersectionCount.Two, IntersectionCount.Three, IntersectionCount.Four} :=
by sorry

end intersection_counts_theorem_l832_83266


namespace brother_grade_is_two_l832_83214

structure Brother where
  grade : ℕ

structure Grandmother where
  sneeze : Bool

def tells_truth (b : Brother) (statement : ℕ) : Prop :=
  b.grade = statement

def grandmother_sneezes (g : Grandmother) (b : Brother) (statement : ℕ) : Prop :=
  tells_truth b statement → g.sneeze = true

theorem brother_grade_is_two (b : Brother) (g : Grandmother) :
  grandmother_sneezes g b 5 ∧ g.sneeze = false →
  grandmother_sneezes g b 4 ∧ g.sneeze = true →
  grandmother_sneezes g b 3 ∧ g.sneeze = false →
  b.grade = 2 := by
  sorry

end brother_grade_is_two_l832_83214


namespace gcd_78_36_l832_83217

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end gcd_78_36_l832_83217


namespace not_not_or_implies_or_at_least_one_true_l832_83245

theorem not_not_or_implies_or (p q : Prop) : ¬¬(p ∨ q) → (p ∨ q) := by
  sorry

theorem at_least_one_true (p q : Prop) : ¬¬(p ∨ q) → (p ∨ q) := by
  sorry

end not_not_or_implies_or_at_least_one_true_l832_83245


namespace expression_equals_five_halves_l832_83271

theorem expression_equals_five_halves :
  Real.sqrt 12 - 2 * Real.cos (π / 6) + |Real.sqrt 3 - 2| + 2^(-1 : ℤ) = 5 / 2 := by
  sorry

end expression_equals_five_halves_l832_83271


namespace largest_five_digit_code_l832_83269

def is_power_of_5 (n : Nat) : Prop :=
  ∃ k : Nat, n = 5^k

def is_power_of_2 (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

def is_multiple_of_3 (n : Nat) : Prop :=
  ∃ k : Nat, n = 3 * k

def digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def has_unique_digits (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem largest_five_digit_code : 
  ∀ n : Nat,
  n ≤ 99999 ∧
  n ≥ 10000 ∧
  (∀ d : Nat, d ∈ n.digits 10 → d ≠ 0) ∧
  is_power_of_5 (n / 1000) ∧
  is_power_of_2 (n % 100) ∧
  is_multiple_of_3 ((n / 100) % 10) ∧
  Odd (digits_sum n) ∧
  has_unique_digits n
  →
  n ≤ 25916 :=
by sorry

end largest_five_digit_code_l832_83269


namespace last_digit_of_product_l832_83263

theorem last_digit_of_product (n : ℕ) : 
  (3^2001 * 7^2002 * 13^2003) % 10 = 9 := by sorry

end last_digit_of_product_l832_83263


namespace number_times_99_equals_2376_l832_83201

theorem number_times_99_equals_2376 : ∃ x : ℕ, x * 99 = 2376 ∧ x = 24 := by
  sorry

end number_times_99_equals_2376_l832_83201


namespace complex_equation_solution_l832_83288

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + Complex.I → z = 1 - 2 * Complex.I := by
  sorry

end complex_equation_solution_l832_83288


namespace boat_downstream_distance_l832_83287

/-- Proves that a boat with given characteristics travels 500 km downstream in 5 hours -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (upstream_distance : ℝ)
  (upstream_time : ℝ)
  (downstream_time : ℝ)
  (h_boat_speed : boat_speed = 70)
  (h_upstream_distance : upstream_distance = 240)
  (h_upstream_time : upstream_time = 6)
  (h_downstream_time : downstream_time = 5)
  : ∃ (stream_speed : ℝ),
    stream_speed > 0 ∧
    upstream_distance / upstream_time = boat_speed - stream_speed ∧
    downstream_time * (boat_speed + stream_speed) = 500 :=
by sorry

end boat_downstream_distance_l832_83287


namespace constant_age_difference_l832_83216

/-- The age difference between two brothers remains constant over time -/
theorem constant_age_difference (a b x : ℕ) : (a + x) - (b + x) = a - b := by
  sorry

end constant_age_difference_l832_83216


namespace factor_tree_problem_l832_83265

theorem factor_tree_problem (X Y Z W : ℕ) : 
  X = Y * Z ∧ 
  Y = 7 * 11 ∧ 
  Z = 2 * W ∧ 
  W = 3 * 2 → 
  X = 924 := by sorry

end factor_tree_problem_l832_83265


namespace smoothie_cost_l832_83215

def burger_cost : ℝ := 5
def sandwich_cost : ℝ := 4
def total_order_cost : ℝ := 17
def num_smoothies : ℕ := 2

theorem smoothie_cost :
  let non_smoothie_cost := burger_cost + sandwich_cost
  let smoothie_total_cost := total_order_cost - non_smoothie_cost
  let smoothie_cost := smoothie_total_cost / num_smoothies
  smoothie_cost = 4 := by sorry

end smoothie_cost_l832_83215
