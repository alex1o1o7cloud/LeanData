import Mathlib

namespace least_number_to_add_l2006_200600

theorem least_number_to_add (n : ℕ) (h : (1052 + n) % 37 = 0) : n = 19 := by
  sorry

end least_number_to_add_l2006_200600


namespace correct_statement_d_l2006_200640

theorem correct_statement_d : 
  (∃ x : ℝ, 2^x < x^2) ↔ ¬(∀ x : ℝ, 2^x ≥ x^2) :=
by
  sorry

end correct_statement_d_l2006_200640


namespace sandbox_perimeter_l2006_200661

def sandbox_width : ℝ := 5
def sandbox_length := 2 * sandbox_width
def perimeter (length width : ℝ) := 2 * (length + width)

theorem sandbox_perimeter : perimeter sandbox_length sandbox_width = 30 := 
by
  sorry

end sandbox_perimeter_l2006_200661


namespace total_people_present_l2006_200681

def number_of_parents : ℕ := 105
def number_of_pupils : ℕ := 698
def total_people : ℕ := number_of_parents + number_of_pupils

theorem total_people_present : total_people = 803 :=
by
  sorry

end total_people_present_l2006_200681


namespace horse_food_per_day_l2006_200686

-- Given conditions
def sheep_count : ℕ := 48
def horse_food_total : ℕ := 12880
def sheep_horse_ratio : ℚ := 6 / 7

-- Definition of the number of horses based on the ratio
def horse_count : ℕ := (sheep_count * 7) / 6

-- Statement to prove: each horse needs 230 ounces of food per day
theorem horse_food_per_day : horse_food_total / horse_count = 230 := by
  -- proof here
  sorry

end horse_food_per_day_l2006_200686


namespace inequality_proof_l2006_200639

theorem inequality_proof
  (a b c d : ℝ)
  (a_nonneg : 0 ≤ a)
  (b_nonneg : 0 ≤ b)
  (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) :
  abc + bcd + cda + dab ≤ (1 / 27) + (176 * abcd / 27) :=
sorry

end inequality_proof_l2006_200639


namespace number_of_odd_blue_faces_cubes_l2006_200650

/-
A wooden block is 5 inches long, 5 inches wide, and 1 inch high.
The block is painted blue on all six sides and then cut into twenty-five 1 inch cubes.
Prove that the number of cubes each have a total number of blue faces that is an odd number is 9.
-/

def cubes_with_odd_blue_faces : ℕ :=
  let corner_cubes := 4
  let edge_cubes_not_corners := 16
  let center_cubes := 5
  corner_cubes + center_cubes

theorem number_of_odd_blue_faces_cubes : cubes_with_odd_blue_faces = 9 := by
  have h1 : cubes_with_odd_blue_faces = 4 + 5 := sorry
  have h2 : 4 + 5 = 9 := by norm_num
  exact Eq.trans h1 h2

end number_of_odd_blue_faces_cubes_l2006_200650


namespace range_of_m_l2006_200613

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*x + m

theorem range_of_m {m : ℝ} :
  (∀ x : ℝ, (f x m = 0) → (∃ y z : ℝ, y ≠ z ∧ f y m = 0 ∧ f z m = 0)) ∧
  (∀ x : ℝ, f (1 - x) m ≥ -1)
  → (0 ≤ m ∧ m < 1) := 
sorry

end range_of_m_l2006_200613


namespace jacoby_needs_l2006_200697

-- Given conditions
def total_goal : ℤ := 5000
def job_earnings_per_hour : ℤ := 20
def total_job_hours : ℤ := 10
def cookie_price_each : ℤ := 4
def total_cookies_sold : ℤ := 24
def lottery_ticket_cost : ℤ := 10
def lottery_winning : ℤ := 500
def gift_from_sister_one : ℤ := 500
def gift_from_sister_two : ℤ := 500

-- Total money Jacoby has so far
def current_total_money : ℤ := 
  job_earnings_per_hour * total_job_hours +
  cookie_price_each * total_cookies_sold +
  lottery_winning +
  gift_from_sister_one + gift_from_sister_two -
  lottery_ticket_cost

-- The amount Jacoby needs to reach his goal
def amount_needed : ℤ := total_goal - current_total_money

-- The main statement to be proved
theorem jacoby_needs : amount_needed = 3214 := by
  -- The proof is skipped
  sorry

end jacoby_needs_l2006_200697


namespace diagonals_in_15_sided_polygon_l2006_200610

def numberOfDiagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_15_sided_polygon : numberOfDiagonals 15 = 90 := by
  sorry

end diagonals_in_15_sided_polygon_l2006_200610


namespace amanda_earnings_l2006_200641

def hourly_rate : ℝ := 20.00

def hours_monday : ℝ := 5 * 1.5

def hours_tuesday : ℝ := 3

def hours_thursday : ℝ := 2 * 2

def hours_saturday : ℝ := 6

def total_hours : ℝ := hours_monday + hours_tuesday + hours_thursday + hours_saturday

def total_earnings : ℝ := hourly_rate * total_hours

theorem amanda_earnings : total_earnings = 410.00 :=
by
  -- Proof steps can be filled here
  sorry

end amanda_earnings_l2006_200641


namespace paul_sold_11_books_l2006_200647

variable (initial_books : ℕ) (books_given : ℕ) (books_left : ℕ) (books_sold : ℕ)

def number_of_books_sold (initial_books books_given books_left books_sold : ℕ) : Prop :=
  initial_books - books_given - books_left = books_sold

theorem paul_sold_11_books : number_of_books_sold 108 35 62 11 :=
by
  sorry

end paul_sold_11_books_l2006_200647


namespace solve_equation_l2006_200690

-- Definitions based on the conditions
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 10) + 1 / (x^2 + 4*x - 5) + 1 / (x^2 - 17*x - 10) = 0

-- Theorem stating that the solutions of the given equation are the expected values
theorem solve_equation :
  {x : ℝ | equation x} = {-2 + 2 * Real.sqrt 14, -2 - 2 * Real.sqrt 14, (7 + Real.sqrt 89) / 2, (7 - Real.sqrt 89) / 2} :=
by
  sorry

end solve_equation_l2006_200690


namespace value_of_a_plus_c_l2006_200606

theorem value_of_a_plus_c (a b c r : ℝ)
  (h1 : a + b + c = 114)
  (h2 : a * b * c = 46656)
  (h3 : b = a * r)
  (h4 : c = a * r^2) :
  a + c = 78 :=
sorry

end value_of_a_plus_c_l2006_200606


namespace solve_Mary_height_l2006_200687

theorem solve_Mary_height :
  ∃ (m s : ℝ), 
  s = 150 ∧ 
  s * 1.2 = 180 ∧ 
  m = s + (180 - s) / 2 ∧ 
  m = 165 :=
by
  sorry

end solve_Mary_height_l2006_200687


namespace total_pieces_correct_l2006_200611

-- Definitions based on conditions
def rods_in_row (n : ℕ) : ℕ := 3 * n
def connectors_in_row (n : ℕ) : ℕ := n

-- Sum of natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Total rods in ten rows
def total_rods : ℕ := 3 * sum_first_n 10

-- Total connectors in eleven rows
def total_connectors : ℕ := sum_first_n 11

-- Total pieces
def total_pieces : ℕ := total_rods + total_connectors

-- Theorem to prove
theorem total_pieces_correct : total_pieces = 231 :=
by
  sorry

end total_pieces_correct_l2006_200611


namespace andre_max_points_visited_l2006_200679
noncomputable def largest_points_to_visit_in_alphabetical_order : ℕ :=
  10

theorem andre_max_points_visited : largest_points_to_visit_in_alphabetical_order = 10 := 
by
  sorry

end andre_max_points_visited_l2006_200679


namespace find_y_find_x_l2006_200642

-- Define vectors as per the conditions
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Define the condition for perpendicular vectors
def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

-- Define the condition for parallel vectors
def is_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

-- Question 1 Proof Statement
theorem find_y : ∀ (y : ℝ), is_perpendicular a (b y) → y = 3 / 2 :=
by
  intros y h
  unfold is_perpendicular at h
  unfold dot_product at h
  sorry

-- Question 2 Proof Statement
theorem find_x : ∀ (x : ℝ), is_parallel a (c x) → x = 15 / 2 :=
by
  intros x h
  unfold is_parallel at h
  sorry

end find_y_find_x_l2006_200642


namespace apples_count_l2006_200616

variable (A : ℕ)

axiom h1 : 134 = 80 + 54
axiom h2 : A + 98 = 134

theorem apples_count : A = 36 :=
by
  sorry

end apples_count_l2006_200616


namespace find_b_value_l2006_200605

/-- Given a line segment from point (0, b) to (8, 0) with a slope of -3/2, 
    prove that the value of b is 12. -/
theorem find_b_value (b : ℝ) : (8 - 0) ≠ 0 ∧ ((0 - b) / (8 - 0) = -3/2) → b = 12 := 
by
  intro h
  sorry

end find_b_value_l2006_200605


namespace sheila_earning_per_hour_l2006_200694

theorem sheila_earning_per_hour :
  (252 / ((8 * 3) + (6 * 2)) = 7) := 
by
  -- Prove that sheila earns $7 per hour
  
  sorry

end sheila_earning_per_hour_l2006_200694


namespace number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l2006_200664

-- Definitions for the problem
def n : Nat := 12

-- Statement 1: Number of diagonals in a dodecagon
theorem number_of_diagonals_dodecagon (n : Nat) (h : n = 12) : (n * (n - 3)) / 2 = 54 := by
  sorry

-- Statement 2: Sum of interior angles in a dodecagon
theorem sum_of_interior_angles_dodecagon (n : Nat) (h : n = 12) : 180 * (n - 2) = 1800 := by
  sorry

end number_of_diagonals_dodecagon_sum_of_interior_angles_dodecagon_l2006_200664


namespace half_vector_AB_l2006_200633

-- Define vectors MA and MB
def MA : ℝ × ℝ := (-2, 4)
def MB : ℝ × ℝ := (2, 6)

-- Define the proof statement 
theorem half_vector_AB : (1 / 2 : ℝ) • (MB - MA) = (2, 1) :=
by sorry

end half_vector_AB_l2006_200633


namespace rational_smaller_than_neg_half_l2006_200651

theorem rational_smaller_than_neg_half : ∃ q : ℚ, q < -1/2 := by
  use (-1 : ℚ)
  sorry

end rational_smaller_than_neg_half_l2006_200651


namespace stock_price_at_end_of_second_year_l2006_200655

def stock_price_first_year (initial_price : ℝ) : ℝ :=
  initial_price * 2

def stock_price_second_year (price_after_first_year : ℝ) : ℝ :=
  price_after_first_year * 0.75

theorem stock_price_at_end_of_second_year : 
  (stock_price_second_year (stock_price_first_year 100) = 150) :=
by
  sorry

end stock_price_at_end_of_second_year_l2006_200655


namespace carlos_paid_l2006_200670

theorem carlos_paid (a b c : ℝ) 
  (h1 : a = (1 / 3) * (b + c))
  (h2 : b = (1 / 4) * (a + c))
  (h3 : a + b + c = 120) :
  c = 72 :=
by
-- Proof omitted
sorry

end carlos_paid_l2006_200670


namespace indeterminate_original_value_percentage_l2006_200635

-- Lets define the problem as a structure with the given conditions
structure StockData where
  yield_percent : ℚ
  market_value : ℚ

-- We need to prove this condition
theorem indeterminate_original_value_percentage (d : StockData) :
  d.yield_percent = 8 ∧ d.market_value = 125 → false :=
by
  sorry

end indeterminate_original_value_percentage_l2006_200635


namespace min_value_dot_product_l2006_200620

-- Side length of the square
def side_length: ℝ := 1

-- Definition of points in vector space
variables {A B C D O M N P: Type}

-- Definitions assuming standard Euclidean geometry
variables (O P : ℝ) (a b c : ℝ)

-- Points M and N on the edges AD and BC respectively, line MN passes through O
-- Point P satisfies 2 * vector OP = l * vector OA + (1-l) * vector OB
theorem min_value_dot_product (l : ℝ) (O P M N : ℝ) :
  (2 * (O + P)) = l * (O - a) + (1 - l) * (b + c) ∧
  ((O - P) * (O + P) - ((l^2 - l + 1/2) / 4) = -7/16) :=
by
  sorry

end min_value_dot_product_l2006_200620


namespace pond_length_l2006_200637

theorem pond_length (
    W L P : ℝ) 
    (h1 : L = 2 * W) 
    (h2 : L = 32) 
    (h3 : (L * W) / 8 = P^2) : 
  P = 8 := 
by 
  sorry

end pond_length_l2006_200637


namespace arrangements_count_l2006_200675

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of positions
def num_positions : ℕ := 3

-- Define a type for the students
inductive Student
| A | B | C | D | E

-- Define the positions
inductive Position
| athletics | swimming | ball_games

-- Constraint: student A cannot be the swimming volunteer
def cannot_be_swimming_volunteer (s : Student) (p : Position) : Prop :=
  (s = Student.A → p ≠ Position.swimming)

-- Define the function to count the arrangements given the constraints
noncomputable def count_arrangements : ℕ :=
  (num_students.choose num_positions) - 1 -- Placeholder for the actual count based on given conditions

-- The theorem statement
theorem arrangements_count : count_arrangements = 16 :=
by
  sorry

end arrangements_count_l2006_200675


namespace king_chessboard_strategy_king_chessboard_strategy_odd_l2006_200692

theorem king_chessboard_strategy (m n : ℕ) : 
  (m * n) % 2 = 0 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) := 
sorry

theorem king_chessboard_strategy_odd (m n : ℕ) : 
  (m * n) % 2 = 1 → (∀ p, p < m * n → ∃ p', p' < m * n ∧ p' ≠ p) :=
sorry

end king_chessboard_strategy_king_chessboard_strategy_odd_l2006_200692


namespace famous_quote_author_l2006_200698

-- conditions
def statement_date := "July 20, 1969"
def mission := "Apollo 11"
def astronauts := ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]
def first_to_moon := "Neil Armstrong"

-- goal
theorem famous_quote_author : (statement_date = "July 20, 1969") ∧ (mission = "Apollo 11") ∧ (astronauts = ["Neil Armstrong", "Buzz Aldrin", "Michael Collins"]) ∧ (first_to_moon = "Neil Armstrong") → "Neil Armstrong" = "Neil Armstrong" :=
by 
  intros _; 
  exact rfl

end famous_quote_author_l2006_200698


namespace minimum_shoeing_time_l2006_200601

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) (horses : ℕ) (hooves_per_horse : ℕ) (time_per_hoof : ℕ) 
  (total_hooves : ℕ := horses * hooves_per_horse) 
  (time_for_one_blacksmith : ℕ := total_hooves * time_per_hoof) 
  (total_parallel_time : ℕ := time_for_one_blacksmith / blacksmiths)
  (h : blacksmiths = 48)
  (h' : horses = 60)
  (h'' : hooves_per_horse = 4)
  (h''' : time_per_hoof = 5) : 
  total_parallel_time = 25 :=
by
  sorry

end minimum_shoeing_time_l2006_200601


namespace seconds_in_part_of_day_l2006_200685

theorem seconds_in_part_of_day : (1 / 4) * (1 / 6) * (1 / 8) * 24 * 60 * 60 = 450 := by
  sorry

end seconds_in_part_of_day_l2006_200685


namespace intersection_complement_l2006_200615

-- Definitions based on the conditions in the problem
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

-- Definition of complement of set M in the universe U
def complement_U (M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

-- The proof statement
theorem intersection_complement :
  N ∩ (complement_U M) = {3, 5} :=
by
  sorry

end intersection_complement_l2006_200615


namespace fold_hexagon_possible_l2006_200624

theorem fold_hexagon_possible (a b : ℝ) :
  (∃ x : ℝ, (a - x)^2 + (b - x)^2 = x^2) ↔ (1 / 2 < b / a ∧ b / a < 2) :=
by
  sorry

end fold_hexagon_possible_l2006_200624


namespace cooking_time_at_least_l2006_200618

-- Definitions based on conditions
def total_potatoes : ℕ := 35
def cooked_potatoes : ℕ := 11
def time_per_potato : ℕ := 7 -- in minutes
def salad_time : ℕ := 15 -- in minutes

-- The statement to prove
theorem cooking_time_at_least (oven_capacity : ℕ) :
  ∃ t : ℕ, t ≥ salad_time :=
by
  sorry

end cooking_time_at_least_l2006_200618


namespace max_xy_l2006_200614

open Real

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eqn : x + 4 * y = 4) :
  ∃ y : ℝ, (x = 4 - 4 * y) → y = 1 / 2 → x * y = 1 :=
by
  sorry

end max_xy_l2006_200614


namespace min_direction_changes_l2006_200689

theorem min_direction_changes (n : ℕ) : 
  ∀ (path : Finset (ℕ × ℕ)), 
    (path.card = (n + 1) * (n + 2) / 2) → 
    (∀ (v : ℕ × ℕ), v ∈ path) →
    ∃ changes, (changes ≥ n) :=
by sorry

end min_direction_changes_l2006_200689


namespace min_a_for_50_pow_2023_div_17_l2006_200665

theorem min_a_for_50_pow_2023_div_17 (a : ℕ) (h : 17 ∣ (50 ^ 2023 + a)) : a = 18 :=
sorry

end min_a_for_50_pow_2023_div_17_l2006_200665


namespace num_articles_produced_l2006_200625

-- Conditions
def production_rate (x : ℕ) : ℕ := 2 * x^3 / (x * x * 2 * x)
def articles_produced (y : ℕ) : ℕ := y * 2 * y * y * production_rate y

-- Proof: Given the production rate, prove the number of articles produced.
theorem num_articles_produced (y : ℕ) : articles_produced y = 2 * y^3 := by sorry

end num_articles_produced_l2006_200625


namespace solve_for_n_l2006_200663

theorem solve_for_n (n : ℕ) (h : 3^n * 9^n = 81^(n - 12)) : n = 48 :=
sorry

end solve_for_n_l2006_200663


namespace perfect_square_2n_plus_65_l2006_200623

theorem perfect_square_2n_plus_65 (n : ℕ) (h : n > 0) : 
  (∃ m : ℕ, m * m = 2^n + 65) → n = 4 ∨ n = 10 :=
by 
  sorry

end perfect_square_2n_plus_65_l2006_200623


namespace quilt_square_side_length_l2006_200688

theorem quilt_square_side_length (length width : ℝ) (h1 : length = 6) (h2 : width = 24) :
  ∃ s : ℝ, (length * width = s * s) ∧ s = 12 :=
by {
  sorry
}

end quilt_square_side_length_l2006_200688


namespace find_range_of_a_l2006_200667

theorem find_range_of_a (a : ℝ) (x : ℝ) (y : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (hy : 2 ≤ y ∧ y ≤ 3) 
    (hineq : x * y ≤ a * x^2 + 2 * y^2) : 
    -1 ≤ a := sorry

end find_range_of_a_l2006_200667


namespace richmond_tigers_tickets_l2006_200668

theorem richmond_tigers_tickets (total_tickets first_half_tickets : ℕ) 
  (h1 : total_tickets = 9570)
  (h2 : first_half_tickets = 3867) : 
  total_tickets - first_half_tickets = 5703 :=
by
  -- Proof steps would go here
  sorry

end richmond_tigers_tickets_l2006_200668


namespace car_speed_l2006_200648

variable (D : ℝ) (V : ℝ)

theorem car_speed
  (h1 : 1 / ((D / 3) / 80) + (D / 3) / 15 + (D / 3) / V = D / 30) :
  V = 35.625 :=
by 
  sorry

end car_speed_l2006_200648


namespace find_angle4_l2006_200644

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 + angle2 = 180) 
  (h2 : angle3 = angle4) 
  (h3 : angle3 + angle4 = 70) :
  angle4 = 35 := 
by 
  sorry

end find_angle4_l2006_200644


namespace hyperbola_symmetric_slopes_l2006_200649

/-- 
Let \(M(x_0, y_0)\) and \(N(-x_0, -y_0)\) be points symmetric about the origin on the hyperbola 
\(\frac{x^2}{16} - \frac{y^2}{4} = 1\). Let \(P(x, y)\) be any point on the hyperbola. 
When the slopes \(k_{PM}\) and \(k_{PN}\) both exist, then \(k_{PM} \cdot k_{PN} = \frac{1}{4}\),
independent of the position of \(P\).
-/
theorem hyperbola_symmetric_slopes (x x0 y y0: ℝ) 
  (hP: x^2 / 16 - y^2 / 4 = 1)
  (hM: x0^2 / 16 - y0^2 / 4 = 1)
  (h_slop_M : x ≠ x0)
  (h_slop_N : x ≠ x0):
  ((y - y0) / (x - x0)) * ((y + y0) / (x + x0)) = 1 / 4 := 
sorry

end hyperbola_symmetric_slopes_l2006_200649


namespace proposition_truth_count_l2006_200612

namespace Geometry

def is_obtuse_angle (A : Type) : Prop := sorry
def is_obtuse_triangle (ABC : Type) : Prop := sorry

def original_proposition (A : Type) (ABC : Type) : Prop :=
is_obtuse_angle A → is_obtuse_triangle ABC

def contrapositive_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_triangle ABC) → ¬ (is_obtuse_angle A)

def converse_proposition (ABC : Type) (A : Type) : Prop :=
is_obtuse_triangle ABC → is_obtuse_angle A

def inverse_proposition (A : Type) (ABC : Type) : Prop :=
¬ (is_obtuse_angle A) → ¬ (is_obtuse_triangle ABC)

theorem proposition_truth_count (A : Type) (ABC : Type) :
  (original_proposition A ABC ∧ contrapositive_proposition A ABC ∧
  ¬ (converse_proposition ABC A) ∧ ¬ (inverse_proposition A ABC)) →
  ∃ n : ℕ, n = 2 :=
sorry

end Geometry

end proposition_truth_count_l2006_200612


namespace abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l2006_200672

theorem abs_x_minus_1_lt_2_is_necessary_but_not_sufficient (x : ℝ) :
  (-1 < x ∧ x < 3) ↔ (0 < x ∧ x < 3) :=
sorry

end abs_x_minus_1_lt_2_is_necessary_but_not_sufficient_l2006_200672


namespace circle_parabola_intersections_l2006_200699

theorem circle_parabola_intersections : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, (p.1 ^ 2 + p.2 ^ 2 = 16) ∧ (p.2 = p.1 ^ 2 - 4)) ∧
  points.card = 3 := 
sorry

end circle_parabola_intersections_l2006_200699


namespace probability_of_specific_cards_l2006_200662

noncomputable def probability_top_heart_second_spade_third_king 
  (deck_size : ℕ) (ranks_per_suit : ℕ) (suits : ℕ) (hearts : ℕ) (spades : ℕ) (kings : ℕ) : ℚ :=
  (hearts * spades * kings) / (deck_size * (deck_size - 1) * (deck_size - 2))

theorem probability_of_specific_cards :
  probability_top_heart_second_spade_third_king 104 26 4 26 26 8 = 169 / 34102 :=
by {
  sorry
}

end probability_of_specific_cards_l2006_200662


namespace find_range_of_a_l2006_200617

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 4 - 2 * a > 0 ∧ 4 - 2 * a < 1

noncomputable def problem_statement (a : ℝ) : Prop :=
  let p := proposition_p a
  let q := proposition_q a
  (p ∨ q) ∧ ¬(p ∧ q)

theorem find_range_of_a (a : ℝ) :
  problem_statement a → -2 < a ∧ a ≤ 3/2 :=
sorry

end find_range_of_a_l2006_200617


namespace find_k_l2006_200607

variable (m n k : ℝ)

-- Conditions from the problem
def quadratic_roots : Prop := (m + n = -2) ∧ (m * n = k) ∧ (1/m + 1/n = 6)

-- Theorem statement
theorem find_k (h : quadratic_roots m n k) : k = -1/3 :=
sorry

end find_k_l2006_200607


namespace marbles_left_l2006_200677

def initial_marbles : ℝ := 9.0
def given_marbles : ℝ := 3.0

theorem marbles_left : initial_marbles - given_marbles = 6.0 := 
by
  sorry

end marbles_left_l2006_200677


namespace find_gamma_k_l2006_200693

noncomputable def alpha (n d : ℕ) : ℕ := 1 + (n - 1) * d
noncomputable def beta (n r : ℕ) : ℕ := r^(n - 1)
noncomputable def gamma (n d r : ℕ) : ℕ := alpha n d + beta n r

theorem find_gamma_k (k d r : ℕ) (hk1 : gamma (k-1) d r = 200) (hk2 : gamma (k+1) d r = 2000) :
    gamma k d r = 387 :=
sorry

end find_gamma_k_l2006_200693


namespace train_crossing_time_l2006_200666

-- Definitions of the given problem conditions
def train_length : ℕ := 120  -- in meters.
def speed_kmph : ℕ := 144   -- in km/h.

-- Conversion factor
def km_per_hr_to_m_per_s (speed : ℕ) : ℚ :=
  speed * (1000 / 3600 : ℚ)

-- Speed in m/s
def train_speed : ℚ := km_per_hr_to_m_per_s speed_kmph

-- Time calculation
def time_to_cross_pole (length : ℕ) (speed : ℚ) : ℚ :=
  length / speed

-- The theorem we want to prove.
theorem train_crossing_time :
  time_to_cross_pole train_length train_speed = 3 := by 
  sorry

end train_crossing_time_l2006_200666


namespace tile_count_difference_l2006_200695

theorem tile_count_difference (W : ℕ) (B : ℕ) (B' : ℕ) (added_black_tiles : ℕ)
  (hW : W = 16) (hB : B = 9) (h_add : added_black_tiles = 8) (hB' : B' = B + added_black_tiles) :
  B' - W = 1 :=
by
  sorry

end tile_count_difference_l2006_200695


namespace binomial_sum_to_220_l2006_200603

open Nat

def binomial_coeff (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binomial_sum_to_220 :
  binomial_coeff 2 2 + binomial_coeff 3 2 + binomial_coeff 4 2 + binomial_coeff 5 2 +
  binomial_coeff 6 2 + binomial_coeff 7 2 + binomial_coeff 8 2 + binomial_coeff 9 2 +
  binomial_coeff 10 2 + binomial_coeff 11 2 = 220 :=
by
  /- Proof goes here, use the computed value of combinations -/
  sorry

end binomial_sum_to_220_l2006_200603


namespace total_cases_l2006_200691

def NY : ℕ := 2000
def CA : ℕ := NY / 2
def TX : ℕ := CA - 400

theorem total_cases : NY + CA + TX = 3600 :=
by
  -- use sorry placeholder to indicate the solution is omitted
  sorry

end total_cases_l2006_200691


namespace sandy_correct_sums_l2006_200660

-- Definitions based on the conditions
variables (c i : ℕ)

-- Conditions as Lean statements
axiom h1 : 3 * c - 2 * i = 65
axiom h2 : c + i = 30

-- Proof goal
theorem sandy_correct_sums : c = 25 := 
by
  sorry

end sandy_correct_sums_l2006_200660


namespace incorrect_statement_d_l2006_200631

-- Definitions from the problem:
variables (x y : ℝ)
variables (b a : ℝ)
variables (x_bar y_bar : ℝ)

-- Linear regression equation:
def linear_regression (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Properties given in the problem:
axiom pass_through_point : ∀ (x_bar y_bar : ℝ), ∃ b a, y_bar = b * x_bar + a
axiom avg_increase : ∀ (b a : ℝ), y = b * (x + 1) + a → y = b * x + a + b
axiom possible_at_origin : ∀ (b a : ℝ), ∃ y, y = a

-- The statement D which is incorrect:
theorem incorrect_statement_d : ¬ (∀ (b a : ℝ), ∀ y, x = 0 → y = a) :=
sorry

end incorrect_statement_d_l2006_200631


namespace minimum_value_of_f_on_interval_l2006_200638

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x^2 + 9 * x + a

theorem minimum_value_of_f_on_interval (a : ℝ) (h : ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x a ≤ 20) :
  a = -2 → ∃ min_val, min_val = -7 :=
by
  sorry

end minimum_value_of_f_on_interval_l2006_200638


namespace rational_square_of_one_minus_product_l2006_200629

theorem rational_square_of_one_minus_product (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : 
  ∃ (q : ℚ), 1 - x * y = q^2 := 
by 
  sorry

end rational_square_of_one_minus_product_l2006_200629


namespace num_integer_pairs_l2006_200676

theorem num_integer_pairs (m n : ℤ) :
  0 < m ∧ m < n ∧ n < 53 ∧ 53^2 + m^2 = 52^2 + n^2 →
  ∃ k, k = 3 := 
sorry

end num_integer_pairs_l2006_200676


namespace second_train_cross_time_l2006_200674

noncomputable def time_to_cross_second_train : ℝ :=
  let length := 120
  let t1 := 10
  let t_cross := 13.333333333333334
  let v1 := length / t1
  let v_combined := 240 / t_cross
  let v2 := v_combined - v1
  length / v2

theorem second_train_cross_time :
  let t2 := time_to_cross_second_train
  t2 = 20 :=
by
  sorry

end second_train_cross_time_l2006_200674


namespace prime_dates_in_2008_l2006_200652

noncomputable def num_prime_dates_2008 : Nat := 52

theorem prime_dates_in_2008 : 
  let prime_days := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let prime_months_days := [(2, 29), (3, 31), (5, 31), (7, 31), (11, 30)]
  -- Count the prime days for each month considering the list
  let prime_day_count (days : Nat) := (prime_days.filter (λ d => d <= days)).length
  -- Sum the counts for each prime month
  (prime_months_days.map (λ (m, days) => prime_day_count days)).sum = num_prime_dates_2008 :=
by
  sorry

end prime_dates_in_2008_l2006_200652


namespace smallest_integer_k_l2006_200682

theorem smallest_integer_k (k : ℤ) : k > 2 ∧ k % 19 = 2 ∧ k % 7 = 2 ∧ k % 4 = 2 ↔ k = 534 :=
by
  sorry

end smallest_integer_k_l2006_200682


namespace math_students_but_not_science_l2006_200683

theorem math_students_but_not_science (total_students : ℕ) (students_math : ℕ) (students_science : ℕ)
  (students_both : ℕ) (students_math_three_times : ℕ) :
  total_students = 30 ∧ students_both = 2 ∧ students_math = 3 * students_science ∧ 
  students_math = students_both + (22 - 2) → (students_math - students_both = 20) :=
by
  sorry

end math_students_but_not_science_l2006_200683


namespace x_minus_y_eq_14_l2006_200646

theorem x_minus_y_eq_14 (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 100) : x - y = 14 :=
sorry

end x_minus_y_eq_14_l2006_200646


namespace stock_price_percentage_increase_l2006_200669

theorem stock_price_percentage_increase :
  ∀ (total higher lower : ℕ), 
    total = 1980 →
    higher = 1080 →
    higher > lower →
    lower = total - higher →
  ((higher - lower) / lower : ℚ) * 100 = 20 :=
by
  intros total higher lower total_eq higher_eq higher_gt lower_eq
  sorry

end stock_price_percentage_increase_l2006_200669


namespace price_reduction_required_l2006_200673

variable (x : ℝ)
variable (profit_per_piece : ℝ := 40)
variable (initial_sales : ℝ := 20)
variable (additional_sales_per_unit_reduction : ℝ := 2)
variable (desired_profit : ℝ := 1200)

theorem price_reduction_required :
  (profit_per_piece - x) * (initial_sales + additional_sales_per_unit_reduction * x) = desired_profit → x = 20 :=
sorry

end price_reduction_required_l2006_200673


namespace students_neither_l2006_200622

def total_students : ℕ := 150
def students_math : ℕ := 85
def students_physics : ℕ := 63
def students_chemistry : ℕ := 40
def students_math_physics : ℕ := 20
def students_physics_chemistry : ℕ := 15
def students_math_chemistry : ℕ := 10
def students_all_three : ℕ := 5

theorem students_neither:
  total_students - 
  (students_math + students_physics + students_chemistry 
  - students_math_physics - students_physics_chemistry 
  - students_math_chemistry + students_all_three) = 2 := 
by sorry

end students_neither_l2006_200622


namespace absolute_value_equation_solution_l2006_200632

-- mathematical problem representation in Lean
theorem absolute_value_equation_solution (y : ℝ) (h : |y + 2| = |y - 3|) : y = 1 / 2 :=
sorry

end absolute_value_equation_solution_l2006_200632


namespace jane_total_investment_in_stocks_l2006_200627

-- Definitions
def total_investment := 220000
def bonds_investment := 13750
def stocks_investment := 5 * bonds_investment
def mutual_funds_investment := 2 * stocks_investment

-- Condition: The total amount invested
def total_investment_condition : Prop := 
  bonds_investment + stocks_investment + mutual_funds_investment = total_investment

-- Theorem: Jane's total investment in stocks
theorem jane_total_investment_in_stocks :
  total_investment_condition →
  stocks_investment = 68750 :=
by sorry

end jane_total_investment_in_stocks_l2006_200627


namespace div_by_5_implication_l2006_200659

theorem div_by_5_implication (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
    (h3 : ∃ k : ℕ, ab = 5 * k) : (∃ k : ℕ, a = 5 * k) ∨ (∃ k : ℕ, b = 5 * k) := 
by
  sorry

end div_by_5_implication_l2006_200659


namespace unit_digit_product_l2006_200657

-- Definition of unit digit function
def unit_digit (n : Nat) : Nat := n % 10

-- Conditions about unit digits of given powers
lemma unit_digit_3_pow_68 : unit_digit (3 ^ 68) = 1 := by sorry
lemma unit_digit_6_pow_59 : unit_digit (6 ^ 59) = 6 := by sorry
lemma unit_digit_7_pow_71 : unit_digit (7 ^ 71) = 3 := by sorry

-- Main statement
theorem unit_digit_product : unit_digit (3 ^ 68 * 6 ^ 59 * 7 ^ 71) = 8 := by
  have h3 := unit_digit_3_pow_68
  have h6 := unit_digit_6_pow_59
  have h7 := unit_digit_7_pow_71
  sorry

end unit_digit_product_l2006_200657


namespace students_with_uncool_parents_l2006_200654

def total_students : ℕ := 40
def cool_dads_count : ℕ := 18
def cool_moms_count : ℕ := 20
def both_cool_count : ℕ := 10

theorem students_with_uncool_parents :
  total_students - (cool_dads_count + cool_moms_count - both_cool_count) = 12 :=
by sorry

end students_with_uncool_parents_l2006_200654


namespace a2022_value_l2006_200626

theorem a2022_value 
  (a : Fin 2022 → ℤ)
  (h : ∀ n k : Fin 2022, a n - a k ≥ n.1^3 - k.1^3)
  (a1011 : a 1010 = 0) :
  a 2021 = 2022^3 - 1011^3 :=
by
  sorry

end a2022_value_l2006_200626


namespace second_investment_rate_l2006_200608

theorem second_investment_rate (P : ℝ) (r₁ t : ℝ) (I_diff : ℝ) (P900 : P = 900) (r1_4_percent : r₁ = 0.04) (t7 : t = 7) (I_years : I_diff = 31.50) :
∃ r₂ : ℝ, 900 * (r₂ / 100) * 7 - 900 * 0.04 * 7 = 31.50 → r₂ = 4.5 := 
by
  sorry

end second_investment_rate_l2006_200608


namespace shortest_ribbon_length_is_10_l2006_200680

noncomputable def shortest_ribbon_length (L : ℕ) : Prop :=
  (∃ k1 : ℕ, L = 2 * k1) ∧ (∃ k2 : ℕ, L = 5 * k2)

theorem shortest_ribbon_length_is_10 : shortest_ribbon_length 10 :=
by
  sorry

end shortest_ribbon_length_is_10_l2006_200680


namespace find_x_l2006_200645

-- Define the vectors a and b
def a : ℝ × ℝ := (4, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Condition for perpendicular vectors
def perpendicular (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem find_x (x : ℝ) (h : perpendicular a (b x)) : x = -3 / 2 :=
by
  sorry

end find_x_l2006_200645


namespace remainder_x_squared_div_25_l2006_200634

theorem remainder_x_squared_div_25 (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 20 [ZMOD 25]) :
  x^2 ≡ 0 [ZMOD 25] :=
sorry

end remainder_x_squared_div_25_l2006_200634


namespace perimeter_of_new_figure_is_correct_l2006_200684

-- Define the given conditions
def original_horizontal_segments := 16
def original_vertical_segments := 10
def original_side_length := 1
def new_side_length := 2

-- Define total lengths calculations
def total_horizontal_length (new_side_length original_horizontal_segments : ℕ) : ℕ :=
  original_horizontal_segments * new_side_length

def total_vertical_length (new_side_length original_vertical_segments : ℕ) : ℕ :=
  original_vertical_segments * new_side_length

-- Formulate the main theorem
theorem perimeter_of_new_figure_is_correct :
  total_horizontal_length new_side_length original_horizontal_segments + 
  total_vertical_length new_side_length original_vertical_segments = 52 := by
  sorry

end perimeter_of_new_figure_is_correct_l2006_200684


namespace cost_reduction_l2006_200621

variable (a : ℝ) -- original cost
variable (p : ℝ) -- percentage reduction (in decimal form)
variable (m : ℕ) -- number of years

def cost_after_years (a p : ℝ) (m : ℕ) : ℝ :=
  a * (1 - p) ^ m

theorem cost_reduction (a p : ℝ) (m : ℕ) :
  m > 0 → cost_after_years a p m = a * (1 - p) ^ m :=
sorry

end cost_reduction_l2006_200621


namespace jennifer_book_fraction_l2006_200602

theorem jennifer_book_fraction :
  (120 - (1/5 * 120 + 1/6 * 120 + 16)) / 120 = 1/2 :=
by
  sorry

end jennifer_book_fraction_l2006_200602


namespace line_passes_through_fixed_point_l2006_200636

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, (m - 1) * (-2) - 3 + 2 * m + 1 = 0 :=
by
  intros m
  sorry

end line_passes_through_fixed_point_l2006_200636


namespace sum_of_roots_unique_solution_l2006_200619

open Real

def operation (a b : ℝ) : ℝ := a^2 + 2 * a * b - b^2

def f (x : ℝ) : ℝ := operation x 2

theorem sum_of_roots_unique_solution
  (x1 x2 x3 x4 : ℝ)
  (h1 : ∀ x, f x = log (abs (x + 2)) → x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)
  (h2 : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_unique_solution_l2006_200619


namespace compare_2_5_sqrt_6_l2006_200630

theorem compare_2_5_sqrt_6 : 2.5 > Real.sqrt 6 := by
  sorry

end compare_2_5_sqrt_6_l2006_200630


namespace time_for_B_to_complete_work_l2006_200643

theorem time_for_B_to_complete_work 
  (A B C : ℝ)
  (h1 : A = 1 / 4) 
  (h2 : B + C = 1 / 3) 
  (h3 : A + C = 1 / 2) :
  1 / B = 12 :=
by
  -- Proof is omitted, as per instruction.
  sorry

end time_for_B_to_complete_work_l2006_200643


namespace cube_side_length_l2006_200671

-- Given definitions and conditions
variables (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

-- Statement of the theorem
theorem cube_side_length (x : ℝ) : 
  ( ∃ (y z : ℝ), 
      y + x + z = c ∧ 
      x + z = c * a / b ∧
      y = c * x / b ∧
      z = c * x / a 
  ) → x = a * b * c / (a * b + b * c + c * a) :=
sorry

end cube_side_length_l2006_200671


namespace neg_existential_proposition_l2006_200609

open Nat

theorem neg_existential_proposition :
  (¬ (∃ n : ℕ, n + 10 / n < 4)) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) :=
by
  sorry

end neg_existential_proposition_l2006_200609


namespace log_product_identity_l2006_200656

noncomputable def log {a b : ℝ} (ha : 1 < a) (hb : 0 < b) : ℝ := Real.log b / Real.log a

theorem log_product_identity : 
  log (by norm_num : (1 : ℝ) < 2) (by norm_num : (0 : ℝ) < 9) * 
  log (by norm_num : (1 : ℝ) < 3) (by norm_num : (0 : ℝ) < 8) = 6 :=
sorry

end log_product_identity_l2006_200656


namespace largest_integer_satisfying_sin_cos_condition_proof_l2006_200604

noncomputable def largest_integer_satisfying_sin_cos_condition :=
  ∀ (x : ℝ) (n : ℕ), (∀ (n' : ℕ), (∀ x : ℝ, (Real.sin x ^ n' + Real.cos x ^ n' ≥ 2 / n') → n ≤ n')) → n = 4

theorem largest_integer_satisfying_sin_cos_condition_proof :
  largest_integer_satisfying_sin_cos_condition :=
by
  sorry

end largest_integer_satisfying_sin_cos_condition_proof_l2006_200604


namespace distinct_values_for_D_l2006_200696

-- Define distinct digits
def distinct_digits (a b c d e : ℕ) :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
  c ≠ d ∧ c ≠ e ∧ 
  d ≠ e ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10

-- Declare the problem statement
theorem distinct_values_for_D : 
  ∃ D_values : Finset ℕ, 
    (∀ (A B C D E : ℕ), 
      distinct_digits A B C D E → 
      E + C = D ∧
      B + C = E ∧
      B + D = E) →
    D_values.card = 7 := 
by 
  sorry

end distinct_values_for_D_l2006_200696


namespace solution_inequality_l2006_200653

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

axiom odd_function (x : ℝ) : f (-x) = -f (x)
axiom increasing_function (x y : ℝ) : x < y → f x < f y

theorem solution_inequality (x : ℝ) : f (2 * x + 1) + f (x - 2) > 0 ↔ x > 1 / 3 := sorry

end solution_inequality_l2006_200653


namespace solution_l2006_200628

noncomputable def problem_statement : ℝ :=
  let a := 6
  let b := 5
  let x := 10 * a + b
  let y := 10 * b + a
  let m := 16.5
  x + y + m

theorem solution : problem_statement = 137.5 :=
by
  sorry

end solution_l2006_200628


namespace solve_system_of_equations_l2006_200678

theorem solve_system_of_equations :
  ∃ x y : ℝ, (3 * x - 5 * y = -1.5) ∧ (7 * x + 2 * y = 4.7) ∧ x = 0.5 ∧ y = 0.6 :=
by
  sorry -- Proof to be completed

end solve_system_of_equations_l2006_200678


namespace jessa_gave_3_bills_l2006_200658

variable (J G K : ℕ)
variable (billsGiven : ℕ)

/-- Initial conditions and question for the problem -/
def initial_conditions :=
  G = 16 ∧
  K = J - 2 ∧
  G = 2 * K ∧
  (J - billsGiven = 7)

/-- The theorem to prove: Jessa gave 3 bills to Geric -/
theorem jessa_gave_3_bills (h : initial_conditions J G K billsGiven) : billsGiven = 3 := 
sorry

end jessa_gave_3_bills_l2006_200658
