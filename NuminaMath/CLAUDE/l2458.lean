import Mathlib

namespace complement_intersection_theorem_l2458_245829

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {-1, 0, 2}
def B : Finset Int := {0, 1}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {3} := by sorry

end complement_intersection_theorem_l2458_245829


namespace thursday_withdrawal_l2458_245830

/-- Calculates the number of books withdrawn on Thursday given the initial number of books,
    the number of books taken out on Tuesday, the number of books returned on Wednesday,
    and the final number of books in the library. -/
def books_withdrawn_thursday (initial : ℕ) (taken_tuesday : ℕ) (returned_wednesday : ℕ) (final : ℕ) : ℕ :=
  initial - taken_tuesday + returned_wednesday - final

/-- Proves that the number of books withdrawn on Thursday is 15, given the specific values
    from the problem. -/
theorem thursday_withdrawal : books_withdrawn_thursday 250 120 35 150 = 15 := by
  sorry

end thursday_withdrawal_l2458_245830


namespace cube_surface_area_increase_l2458_245849

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge_length := 1.6 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 156 := by
  sorry

end cube_surface_area_increase_l2458_245849


namespace show_charge_day3_l2458_245833

/-- The charge per person on the first day in rupees -/
def charge_day1 : ℚ := 15

/-- The charge per person on the second day in rupees -/
def charge_day2 : ℚ := 15/2

/-- The ratio of attendance on the first day -/
def ratio_day1 : ℕ := 2

/-- The ratio of attendance on the second day -/
def ratio_day2 : ℕ := 5

/-- The ratio of attendance on the third day -/
def ratio_day3 : ℕ := 13

/-- The average charge per person for the whole show in rupees -/
def average_charge : ℚ := 5

/-- The charge per person on the third day in rupees -/
def charge_day3 : ℚ := 5/2

theorem show_charge_day3 :
  let total_ratio := ratio_day1 + ratio_day2 + ratio_day3
  let total_charge := ratio_day1 * charge_day1 + ratio_day2 * charge_day2 + ratio_day3 * charge_day3
  average_charge = total_charge / total_ratio := by
  sorry

end show_charge_day3_l2458_245833


namespace balance_point_specific_rod_l2458_245852

/-- Represents the rod with attached weights -/
structure WeightedRod where
  length : Real
  weights : List (Real × Real)  -- List of (position, weight) pairs

/-- Calculates the balance point of a weighted rod -/
def balancePoint (rod : WeightedRod) : Real :=
  sorry

/-- Theorem stating the balance point for the specific rod configuration -/
theorem balance_point_specific_rod :
  let rod : WeightedRod := {
    length := 4,
    weights := [(0, 20), (1, 30), (2, 40), (3, 50), (4, 60)]
  }
  balancePoint rod = 2.5 := by sorry

end balance_point_specific_rod_l2458_245852


namespace parking_problem_l2458_245826

/-- Calculates the number of vehicles that can still park in a lot -/
def vehiclesCanPark (totalSpaces : ℕ) (caravanSpaces : ℕ) (caravansParked : ℕ) : ℕ :=
  totalSpaces - (caravanSpaces * caravansParked)

/-- Theorem: Given the problem conditions, 24 vehicles can still park -/
theorem parking_problem :
  vehiclesCanPark 30 2 3 = 24 := by
  sorry

end parking_problem_l2458_245826


namespace parallelogram_dimensions_l2458_245861

/-- Proves the side lengths and perimeter of a parallelogram given its area, side ratio, and one angle -/
theorem parallelogram_dimensions (area : ℝ) (angle : ℝ) (h_area : area = 972) (h_angle : angle = 45 * π / 180) :
  ∃ (side1 side2 perimeter : ℝ),
    side1 / side2 = 4 / 3 ∧
    area = side1 * side2 * Real.sin angle ∧
    side1 = 36 * 2^(3/4) ∧
    side2 = 27 * 2^(3/4) ∧
    perimeter = 126 * 2^(3/4) := by
  sorry

end parallelogram_dimensions_l2458_245861


namespace bills_toilet_paper_duration_l2458_245869

/-- The number of days Bill's toilet paper supply will last -/
def toilet_paper_duration (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ) 
  (total_rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  (total_rolls * squares_per_roll) / (bathroom_visits_per_day * squares_per_visit)

/-- Theorem stating that Bill's toilet paper supply will last 20,000 days -/
theorem bills_toilet_paper_duration :
  toilet_paper_duration 3 5 1000 300 = 20000 := by
  sorry

#eval toilet_paper_duration 3 5 1000 300

end bills_toilet_paper_duration_l2458_245869


namespace min_value_theorem_l2458_245897

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (4 * x^2) / (y + 1) + y^2 / (2 * x + 2) ≥ 4/5 := by
sorry

end min_value_theorem_l2458_245897


namespace solution_exists_in_interval_l2458_245877

def f (x : ℝ) := x^2 + 12*x - 15

theorem solution_exists_in_interval :
  ∃ x ∈ Set.Ioo 1.1 1.2, f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

end solution_exists_in_interval_l2458_245877


namespace police_coverage_l2458_245893

-- Define the type for intersections
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

-- Define the type for streets
inductive Street : Type
| ABCD | EFG | HIJK    -- Horizontal streets
| AEH | BFI | DGJ      -- Vertical streets
| HFC | CGK           -- Diagonal streets

-- Define a function to check if an intersection is on a street
def isOnStreet (i : Intersection) (s : Street) : Prop :=
  match s with
  | Street.ABCD => i = Intersection.A ∨ i = Intersection.B ∨ i = Intersection.C ∨ i = Intersection.D
  | Street.EFG => i = Intersection.E ∨ i = Intersection.F ∨ i = Intersection.G
  | Street.HIJK => i = Intersection.H ∨ i = Intersection.I ∨ i = Intersection.J ∨ i = Intersection.K
  | Street.AEH => i = Intersection.A ∨ i = Intersection.E ∨ i = Intersection.H
  | Street.BFI => i = Intersection.B ∨ i = Intersection.F ∨ i = Intersection.I
  | Street.DGJ => i = Intersection.D ∨ i = Intersection.G ∨ i = Intersection.J
  | Street.HFC => i = Intersection.H ∨ i = Intersection.F ∨ i = Intersection.C
  | Street.CGK => i = Intersection.C ∨ i = Intersection.G ∨ i = Intersection.K

-- Define a function to check if a street is covered by a set of intersections
def isCovered (s : Street) (intersections : Set Intersection) : Prop :=
  ∃ i ∈ intersections, isOnStreet i s

-- Theorem statement
theorem police_coverage :
  let policemen : Set Intersection := {Intersection.B, Intersection.G, Intersection.H}
  ∀ s : Street, isCovered s policemen :=
by sorry

end police_coverage_l2458_245893


namespace triangle_area_implies_p_value_l2458_245823

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    if the area of the triangle is 36, then p = 12.75 -/
theorem triangle_area_implies_p_value :
  ∀ (p : ℝ),
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  triangle_area = 36 → p = 12.75 := by
sorry


end triangle_area_implies_p_value_l2458_245823


namespace fruit_salad_oranges_l2458_245846

theorem fruit_salad_oranges :
  ∀ (s k a o : ℕ),
    s + k + a + o = 360 →
    s = k / 2 →
    a = 2 * o →
    o = 3 * s →
    o = 60 :=
by
  sorry

end fruit_salad_oranges_l2458_245846


namespace expression_evaluation_l2458_245898

theorem expression_evaluation : 16^3 + 3*(16^2) + 3*16 + 1 = 4913 := by
  sorry

end expression_evaluation_l2458_245898


namespace sequence_modulo_eight_property_l2458_245821

theorem sequence_modulo_eight_property (s : ℕ → ℕ) 
  (h : ∀ n : ℕ, s (n + 2) = s (n + 1) + s n) : 
  ∃ r : ℤ, ∀ n : ℕ, ¬ (8 ∣ (s n - r)) :=
sorry

end sequence_modulo_eight_property_l2458_245821


namespace square_of_95_l2458_245891

theorem square_of_95 : (95 : ℤ)^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end square_of_95_l2458_245891


namespace movie_ticket_difference_l2458_245867

theorem movie_ticket_difference (romance_tickets horror_tickets : ℕ) : 
  romance_tickets = 25 → 
  horror_tickets = 93 → 
  horror_tickets - 3 * romance_tickets = 18 := by
sorry

end movie_ticket_difference_l2458_245867


namespace jiangsu_population_scientific_notation_l2458_245810

/-- The population of Jiangsu Province in 2021 -/
def jiangsu_population : ℕ := 85000000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Theorem: The population of Jiangsu Province expressed in scientific notation -/
theorem jiangsu_population_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.significand * (10 : ℝ) ^ sn.exponent) = jiangsu_population := by
  sorry

end jiangsu_population_scientific_notation_l2458_245810


namespace function_difference_l2458_245804

theorem function_difference (f : ℝ → ℝ) (h : ∀ x, f x = 9^x) :
  ∀ x, f (x + 1) - f x = 8 * f x := by
  sorry

end function_difference_l2458_245804


namespace commodity_price_problem_l2458_245834

theorem commodity_price_problem (total_cost first_price second_price : ℕ) :
  total_cost = 827 →
  first_price = second_price + 127 →
  total_cost = first_price + second_price →
  first_price = 477 := by
  sorry

end commodity_price_problem_l2458_245834


namespace circle_tangency_theorem_l2458_245841

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Represents the length of a chord as m√n/p -/
structure ChordLength where
  m : ℕ
  n : ℕ
  p : ℕ

/-- Checks if two numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

theorem circle_tangency_theorem (C1 C2 C3 : Circle) (chord : ChordLength) :
  are_externally_tangent C1 C2 ∧
  is_internally_tangent C1 C3 ∧
  is_internally_tangent C2 C3 ∧
  are_collinear C1.center C2.center C3.center ∧
  C1.radius = 5 ∧
  C2.radius = 13 ∧
  are_relatively_prime chord.m chord.p ∧
  not_divisible_by_prime_square chord.n →
  chord.m + chord.n + chord.p = 455 := by
  sorry


end circle_tangency_theorem_l2458_245841


namespace unique_solution_2a3b_7c_l2458_245863

theorem unique_solution_2a3b_7c : ∃! (a b c : ℕ+), 2^(a:ℕ) * 3^(b:ℕ) = 7^(c:ℕ) - 1 := by
  sorry

end unique_solution_2a3b_7c_l2458_245863


namespace multiply_powers_of_y_l2458_245847

theorem multiply_powers_of_y (y : ℝ) : 5 * y^3 * (3 * y^2) = 15 * y^5 := by
  sorry

end multiply_powers_of_y_l2458_245847


namespace perpendicular_vectors_t_value_l2458_245851

def a : Fin 2 → ℝ := ![3, 1]
def b : Fin 2 → ℝ := ![1, 3]
def c (t : ℝ) : Fin 2 → ℝ := ![t, 2]

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ, (∀ i : Fin 2, (a i - c t i) * b i = 0) → t = 0 := by
  sorry

end perpendicular_vectors_t_value_l2458_245851


namespace junk_mail_distribution_l2458_245855

theorem junk_mail_distribution (blocks : ℕ) (houses_per_block : ℕ) (total_mail : ℕ) 
  (h1 : blocks = 16) 
  (h2 : houses_per_block = 17) 
  (h3 : total_mail = 1088) : 
  total_mail / (blocks * houses_per_block) = 4 := by
  sorry

end junk_mail_distribution_l2458_245855


namespace pushup_comparison_l2458_245803

theorem pushup_comparison (zachary david emily : ℕ) 
  (h1 : zachary = 51)
  (h2 : david = 44)
  (h3 : emily = 37) :
  zachary = (david + emily) - 30 :=
by sorry

end pushup_comparison_l2458_245803


namespace stratified_sampling_problem_l2458_245884

theorem stratified_sampling_problem (total_students : ℕ) (sample_size : ℕ) (major_c_students : ℕ) :
  total_students = 1000 →
  sample_size = 40 →
  major_c_students = 400 →
  (major_c_students * sample_size) / total_students = 16 := by
sorry

end stratified_sampling_problem_l2458_245884


namespace brocard_angle_inequalities_l2458_245890

theorem brocard_angle_inequalities (α β γ φ : Real) 
  (triangle : α + β + γ = Real.pi)
  (brocard_condition : φ ≤ Real.pi / 6)
  (sin_relation : Real.sin (α - φ) * Real.sin (β - φ) * Real.sin (γ - φ) = Real.sin φ ^ 3) :
  φ ^ 3 ≤ (α - φ) * (β - φ) * (γ - φ) ∧ 8 * φ ^ 3 ≤ α * β * γ := by
  sorry

end brocard_angle_inequalities_l2458_245890


namespace search_plans_count_l2458_245825

/-- Represents the number of children in the group -/
def total_children : ℕ := 8

/-- Represents whether Grace participates in the task -/
inductive GraceParticipation
| Participates
| DoesNotParticipate

/-- Calculates the number of ways to distribute children for the search task -/
def count_search_plans : ℕ :=
  let grace_participates := Nat.choose 7 3  -- Choose 3 out of 7 to go with Grace
  let grace_not_participates := 7 * Nat.choose 6 3  -- Choose 1 to stay, then distribute 6
  grace_participates + grace_not_participates

/-- Theorem stating that the number of different search plans is 175 -/
theorem search_plans_count :
  count_search_plans = 175 := by sorry

end search_plans_count_l2458_245825


namespace soccer_team_wins_l2458_245899

/-- Given a soccer team that played 140 games and won 50 percent of them, 
    prove that the number of games won is 70. -/
theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) : 
  total_games = 140 → 
  win_percentage = 1/2 → 
  games_won = (total_games : ℚ) * win_percentage → 
  games_won = 70 := by
sorry

end soccer_team_wins_l2458_245899


namespace hyperbola_properties_l2458_245882

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (∀ f ∈ foci, (x - f.1)^2 + y^2 > 0)) ∧
  (∀ x y, hyperbola x y → 
    let a := 2  -- sqrt(4)
    let c := 4  -- distance from center to focus
    c / a = eccentricity) :=
sorry

end hyperbola_properties_l2458_245882


namespace cake_eating_ratio_l2458_245896

theorem cake_eating_ratio (cake_weight : ℝ) (parts : ℕ) (pierre_ate : ℝ) : 
  cake_weight = 400 →
  parts = 8 →
  pierre_ate = 100 →
  (pierre_ate / (cake_weight / parts.cast)) = 2 := by
  sorry

end cake_eating_ratio_l2458_245896


namespace wedding_tables_l2458_245870

theorem wedding_tables (total_fish : ℕ) (fish_per_regular_table : ℕ) (fish_at_special_table : ℕ) :
  total_fish = 65 →
  fish_per_regular_table = 2 →
  fish_at_special_table = 3 →
  ∃ (num_tables : ℕ), num_tables * fish_per_regular_table + (fish_at_special_table - fish_per_regular_table) = total_fish ∧
                       num_tables = 32 := by
  sorry

end wedding_tables_l2458_245870


namespace four_valid_dimensions_l2458_245817

/-- The number of valid floor dimensions -/
def valid_floor_dimensions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    a ≥ 5 ∧ b > a ∧ (a - 6) * (b - 6) = 36
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The theorem stating that there are exactly 4 valid floor dimensions -/
theorem four_valid_dimensions : valid_floor_dimensions = 4 := by
  sorry

end four_valid_dimensions_l2458_245817


namespace sum_of_specific_numbers_l2458_245816

theorem sum_of_specific_numbers : 3 + 33 + 333 + 33.3 = 402.3 := by
  sorry

end sum_of_specific_numbers_l2458_245816


namespace sqrt_511100_approx_l2458_245806

-- Define the approximation relation
def approx (x y : ℝ) : Prop := ∃ ε > 0, |x - y| < ε

-- State the theorem
theorem sqrt_511100_approx :
  approx (Real.sqrt 51.11) 7.149 →
  approx (Real.sqrt 511100) 714.9 :=
by
  sorry

end sqrt_511100_approx_l2458_245806


namespace taller_tree_height_l2458_245865

/-- Given two trees with specific height relationships, prove the height of the taller tree -/
theorem taller_tree_height (h_shorter h_taller : ℝ) : 
  h_taller = h_shorter + 20 →  -- The top of one tree is 20 feet higher
  h_shorter / h_taller = 2 / 3 →  -- The heights are in the ratio 2:3
  h_shorter = 40 →  -- The shorter tree is 40 feet tall
  h_taller = 60 := by sorry

end taller_tree_height_l2458_245865


namespace homework_difference_is_two_l2458_245824

/-- The number of pages of reading homework Rachel had to complete -/
def reading_pages : ℕ := 2

/-- The number of pages of math homework Rachel had to complete -/
def math_pages : ℕ := 4

/-- The difference between math homework pages and reading homework pages -/
def homework_difference : ℕ := math_pages - reading_pages

theorem homework_difference_is_two : homework_difference = 2 := by
  sorry

end homework_difference_is_two_l2458_245824


namespace emily_team_size_l2458_245894

/-- The number of players on Emily's team -/
def num_players : ℕ := 9

/-- The total points scored by the team -/
def total_points : ℕ := 39

/-- The points scored by Emily -/
def emily_points : ℕ := 23

/-- The points scored by each other player -/
def other_player_points : ℕ := 2

/-- Theorem stating that the number of players on Emily's team is correct -/
theorem emily_team_size :
  num_players = (total_points - emily_points) / other_player_points + 1 := by
  sorry


end emily_team_size_l2458_245894


namespace cubic_factorization_l2458_245856

theorem cubic_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end cubic_factorization_l2458_245856


namespace arithmetic_sequence_general_term_l2458_245811

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n + 1) - a n = d) 
  (h2 : a 1 = f (d - 1)) 
  (h3 : a 3 = f (d + 1)) :
  ∀ n, a n = 2 * n + 1 := by
sorry

end arithmetic_sequence_general_term_l2458_245811


namespace max_value_P_l2458_245818

theorem max_value_P (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  let P := (Real.sqrt (b^2 + c^2)) / (3 - a) + (Real.sqrt (c^2 + a^2)) / (3 - b) + a + b - 2022 * c
  P ≤ 3 ∧ (P = 3 ↔ a = 1 ∧ b = 1 ∧ c = 0) :=
by sorry

end max_value_P_l2458_245818


namespace cube_root_and_square_root_l2458_245813

theorem cube_root_and_square_root (a b : ℝ) : 
  (b - 4)^(1/3) = -2 → 
  b = -4 ∧ 
  Real.sqrt (5 * a - b) = 3 :=
by sorry

end cube_root_and_square_root_l2458_245813


namespace range_of_f_l2458_245864

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x - 9

-- Define the open interval (1, 4)
def open_interval : Set ℝ := {x | 1 < x ∧ x < 4}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ open_interval, f x = y} = {y | -18 ≤ y ∧ y < -14} := by sorry

end range_of_f_l2458_245864


namespace q_invest_time_is_20_l2458_245800

/-- Represents a business partnership between two partners -/
structure Partnership where
  investment_ratio : ℚ × ℚ
  profit_ratio : ℚ × ℚ
  p_invest_time : ℕ

/-- Calculates the investment time for partner q given a Partnership -/
def q_invest_time (p : Partnership) : ℚ :=
  (p.profit_ratio.2 * p.investment_ratio.1 * p.p_invest_time : ℚ) / (p.profit_ratio.1 * p.investment_ratio.2)

theorem q_invest_time_is_20 (p : Partnership) 
  (h1 : p.investment_ratio = (7, 5))
  (h2 : p.profit_ratio = (7, 10))
  (h3 : p.p_invest_time = 10) :
  q_invest_time p = 20 := by
  sorry

end q_invest_time_is_20_l2458_245800


namespace area_increase_bound_l2458_245857

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  perimeter : ℝ
  area : ℝ

/-- The result of moving all sides of a polygon outward by distance h -/
def moveOutward (poly : ConvexPolygon) (h : ℝ) : ConvexPolygon := sorry

theorem area_increase_bound (poly : ConvexPolygon) (h : ℝ) (h_pos : h > 0) :
  (moveOutward poly h).area - poly.area > poly.perimeter * h + π * h^2 := by
  sorry

end area_increase_bound_l2458_245857


namespace goods_transportable_l2458_245883

-- Define the problem parameters
def total_weight : ℝ := 13.5
def max_package_weight : ℝ := 0.35
def num_trucks : ℕ := 11
def truck_capacity : ℝ := 1.5

-- Theorem statement
theorem goods_transportable :
  total_weight ≤ (num_trucks : ℝ) * truck_capacity ∧
  ∃ (num_packages : ℕ), (num_packages : ℝ) * max_package_weight ≥ total_weight :=
by sorry

end goods_transportable_l2458_245883


namespace shortest_chord_equation_l2458_245808

-- Define the line l
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 3 + t ∧ p.2 = 1 + a * t}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the condition for shortest chord
def shortest_chord (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
  ∀ X Y : ℝ × ℝ, X ∈ l ∧ X ∈ C ∧ Y ∈ l ∧ Y ∈ C →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Theorem statement
theorem shortest_chord_equation :
  ∃ a : ℝ, shortest_chord (line_l a) circle_C →
  ∀ p : ℝ × ℝ, p ∈ line_l a ↔ p.1 + p.2 = 4 :=
sorry

end shortest_chord_equation_l2458_245808


namespace advanced_vowel_soup_sequences_l2458_245812

/-- The number of vowels in the alphabet soup -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet soup -/
def num_consonants : ℕ := 2

/-- The number of times each vowel appears -/
def vowel_occurrences : ℕ := 7

/-- The number of times each consonant appears -/
def consonant_occurrences : ℕ := 3

/-- The length of each sequence -/
def sequence_length : ℕ := 7

/-- The number of valid sequences in the Advanced Vowel Soup -/
theorem advanced_vowel_soup_sequences : 
  (num_vowels + num_consonants)^sequence_length - 
  num_vowels^sequence_length - 
  num_consonants^sequence_length = 745290 := by
  sorry

end advanced_vowel_soup_sequences_l2458_245812


namespace percentage_of_women_in_survey_l2458_245880

theorem percentage_of_women_in_survey (w : ℝ) (m : ℝ) : 
  w + m = 100 →
  (3/4 : ℝ) * w + (9/10 : ℝ) * m = 84 →
  w = 40 := by
sorry

end percentage_of_women_in_survey_l2458_245880


namespace A_inter_complement_B_l2458_245827

def U : Set Int := Set.univ

def A : Set Int := {-2, -1, 0, 1, 2}

def B : Set Int := {x | x^2 + 2*x = 0}

theorem A_inter_complement_B : A ∩ (U \ B) = {-1, 1, 2} := by sorry

end A_inter_complement_B_l2458_245827


namespace no_alpha_exists_for_all_x_l2458_245850

theorem no_alpha_exists_for_all_x (α : ℝ) (h : α > 0) : 
  ∃ x : ℝ, |Real.cos x| + |Real.cos (α * x)| ≤ Real.sin x + Real.sin (α * x) := by
sorry

end no_alpha_exists_for_all_x_l2458_245850


namespace inequality_solution_l2458_245872

theorem inequality_solution (x : ℝ) : 2 * (x - 3) < 8 ↔ x < 7 := by
  sorry

end inequality_solution_l2458_245872


namespace pedestrian_meets_sixteen_buses_l2458_245887

/-- Represents the problem of a pedestrian meeting buses on a road --/
structure BusMeetingProblem where
  road_length : ℝ
  bus_speed : ℝ
  bus_interval : ℝ
  pedestrian_start_time : ℝ
  pedestrian_speed : ℝ

/-- Calculates the number of buses the pedestrian meets --/
def count_bus_meetings (problem : BusMeetingProblem) : ℕ :=
  sorry

/-- The main theorem stating that the pedestrian meets 16 buses --/
theorem pedestrian_meets_sixteen_buses :
  let problem : BusMeetingProblem := {
    road_length := 8,
    bus_speed := 12,
    bus_interval := 1/6,  -- 10 minutes in hours
    pedestrian_start_time := 81/4,  -- 8:15 AM in hours since midnight
    pedestrian_speed := 4
  }
  count_bus_meetings problem = 16 := by
  sorry

end pedestrian_meets_sixteen_buses_l2458_245887


namespace ellipse_and_circle_theorem_l2458_245889

/-- Definition of the ellipse E -/
def is_ellipse (E : Set (ℝ × ℝ)) (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ ∀ x y, (x, y) ∈ E ↔ x^2 / a^2 + y^2 / b^2 = 1

/-- E passes through the points (2, √2) and (√6, 1) -/
def passes_through_points (E : Set (ℝ × ℝ)) : Prop :=
  (2, Real.sqrt 2) ∈ E ∧ (Real.sqrt 6, 1) ∈ E

/-- Definition of perpendicular vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem ellipse_and_circle_theorem (E : Set (ℝ × ℝ)) (a b : ℝ) 
  (h_ellipse : is_ellipse E a b) (h_points : passes_through_points E) :
  (∃ r : ℝ, r > 0 ∧
    (∀ x y, (x, y) ∈ E ↔ x^2 / 8 + y^2 / 4 = 1) ∧
    (∀ k m : ℝ,
      (∃ A B : ℝ × ℝ,
        A ∈ E ∧ B ∈ E ∧
        A.2 = k * A.1 + m ∧
        B.2 = k * B.1 + m ∧
        perpendicular A B ∧
        A.1^2 + A.2^2 = r^2 ∧
        B.1^2 + B.2^2 = r^2) ↔
      k^2 + 1 = (8 / 3) / m^2)) :=
sorry

end ellipse_and_circle_theorem_l2458_245889


namespace z_value_for_given_w_and_v_l2458_245885

/-- Given a relationship between z, w, and v, prove that z equals 7.5 when w = 4 and v = 8 -/
theorem z_value_for_given_w_and_v (k : ℝ) :
  (3 * 15 = k * 4 / 2^2) →  -- Initial condition
  (∀ z w v : ℝ, 3 * z = k * v / w^2) →  -- General relationship
  ∃ z : ℝ, (3 * z = k * 8 / 4^2) ∧ z = 7.5 :=
by sorry

end z_value_for_given_w_and_v_l2458_245885


namespace man_speed_man_speed_approx_6kmh_l2458_245862

/-- Calculates the speed of a man given the parameters of a train passing him --/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man is approximately 6 km/h --/
theorem man_speed_approx_6kmh :
  ∃ ε > 0, |man_speed 160 90 6 - 6| < ε :=
sorry

end man_speed_man_speed_approx_6kmh_l2458_245862


namespace min_value_theorem_l2458_245868

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 16 / (x + 1) ≥ 7 ∧ ∃ y > 0, y + 16 / (y + 1) = 7 :=
  sorry

end min_value_theorem_l2458_245868


namespace line_translation_slope_l2458_245831

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + dy - l.slope * dx }

theorem line_translation_slope (l : Line) :
  translate (translate l 3 0) 0 1 = l →
  l.slope = -1/3 := by
  sorry

end line_translation_slope_l2458_245831


namespace revenue_decrease_l2458_245807

theorem revenue_decrease (projected_increase : ℝ) (actual_vs_projected : ℝ) : 
  projected_increase = 0.30 →
  actual_vs_projected = 0.57692307692307686 →
  1 - actual_vs_projected * (1 + projected_increase) = 0.25 := by
sorry

end revenue_decrease_l2458_245807


namespace parallel_line_equation_l2458_245881

/-- Given a triangle ABC with vertices A(4,0), B(8,10), and C(0,6),
    the equation of the line passing through A and parallel to BC is x - 2y - 4 = 0 -/
theorem parallel_line_equation (A B C : ℝ × ℝ) : 
  A = (4, 0) → B = (8, 10) → C = (0, 6) → 
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    (x = 4 ∧ y = 0) ∨ (y - 0 = m * (x - 4)) ↔ x - 2*y - 4 = 0 :=
by sorry

end parallel_line_equation_l2458_245881


namespace simplify_expression_l2458_245837

theorem simplify_expression (x : ℝ) : (3*x)^5 + (5*x)*(x^4) - 7*x^5 = 241*x^5 := by
  sorry

end simplify_expression_l2458_245837


namespace total_birds_l2458_245844

/-- Given 3 pairs of birds, prove that the total number of birds is 6. -/
theorem total_birds (pairs : ℕ) (h : pairs = 3) : pairs * 2 = 6 := by
  sorry

end total_birds_l2458_245844


namespace factor_expression_l2458_245895

theorem factor_expression (b : ℝ) : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) := by
  sorry

end factor_expression_l2458_245895


namespace chess_game_probability_l2458_245859

theorem chess_game_probability (p_draw p_B_win p_A_win : ℚ) :
  p_draw = 1/2 →
  p_B_win = 1/3 →
  p_draw + p_B_win + p_A_win = 1 →
  p_A_win = 1/6 := by
sorry

end chess_game_probability_l2458_245859


namespace solution_inequality1_solution_system_inequalities_l2458_245875

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x + 1 > 2*x - 3

def inequality2 (x : ℝ) : Prop := 2*x - 1 > x

def inequality3 (x : ℝ) : Prop := (x + 5) / 2 - x ≥ 1

-- Theorem for the first inequality
theorem solution_inequality1 : 
  {x : ℝ | inequality1 x} = {x : ℝ | x < 4} :=
sorry

-- Theorem for the system of inequalities
theorem solution_system_inequalities :
  {x : ℝ | inequality2 x ∧ inequality3 x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
sorry

end solution_inequality1_solution_system_inequalities_l2458_245875


namespace decimal_addition_subtraction_l2458_245828

theorem decimal_addition_subtraction :
  (0.513 : ℝ) + (0.0067 : ℝ) - (0.048 : ℝ) = (0.4717 : ℝ) := by
  sorry

end decimal_addition_subtraction_l2458_245828


namespace water_for_bathing_is_two_l2458_245876

/-- Calculates the water needed for bathing per horse per day -/
def water_for_bathing (initial_horses : ℕ) (added_horses : ℕ) (drinking_water_per_horse : ℕ) (total_days : ℕ) (total_water : ℕ) : ℚ :=
  let total_horses := initial_horses + added_horses
  let total_drinking_water := total_horses * drinking_water_per_horse * total_days
  let total_bathing_water := total_water - total_drinking_water
  (total_bathing_water : ℚ) / (total_horses * total_days : ℚ)

/-- Theorem: Given the conditions, each horse needs 2 liters of water for bathing per day -/
theorem water_for_bathing_is_two :
  water_for_bathing 3 5 5 28 1568 = 2 := by
  sorry

end water_for_bathing_is_two_l2458_245876


namespace chef_nuts_total_weight_l2458_245879

theorem chef_nuts_total_weight (almond_weight pecan_weight : Real) 
  (h1 : almond_weight = 0.14)
  (h2 : pecan_weight = 0.38) :
  almond_weight + pecan_weight = 0.52 := by
sorry

end chef_nuts_total_weight_l2458_245879


namespace expansion_max_coefficient_l2458_245805

/-- The coefficient of x^3 in the expansion of (x - a/x)^5 is -5 -/
def coefficient_condition (a : ℝ) : Prop :=
  (5 : ℝ) * a = 5

/-- The maximum coefficient in the expansion of (x - a/x)^5 -/
def max_coefficient (a : ℝ) : ℕ :=
  Nat.max (Nat.choose 5 0)
    (Nat.max (Nat.choose 5 1)
      (Nat.max (Nat.choose 5 2)
        (Nat.max (Nat.choose 5 3)
          (Nat.max (Nat.choose 5 4)
            (Nat.choose 5 5)))))

theorem expansion_max_coefficient :
  ∀ a : ℝ, coefficient_condition a → max_coefficient a = 10 := by
  sorry

end expansion_max_coefficient_l2458_245805


namespace alex_jogging_speed_l2458_245842

/-- Given the jogging speeds of Eugene, Brianna, Katie, and Alex, prove that Alex jogs at 2.4 miles per hour. -/
theorem alex_jogging_speed 
  (eugene_speed : ℝ) 
  (brianna_speed : ℝ) 
  (katie_speed : ℝ) 
  (alex_speed : ℝ) 
  (h1 : eugene_speed = 5)
  (h2 : brianna_speed = 4/5 * eugene_speed)
  (h3 : katie_speed = 6/5 * brianna_speed)
  (h4 : alex_speed = 1/2 * katie_speed) :
  alex_speed = 2.4 := by
  sorry

end alex_jogging_speed_l2458_245842


namespace ab_value_l2458_245802

theorem ab_value (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 := by
  sorry

end ab_value_l2458_245802


namespace intersection_of_P_and_Q_l2458_245839

def P : Set (ℝ × ℝ) := {(x, y) | x + y = 0}
def Q : Set (ℝ × ℝ) := {(x, y) | x - y = 2}

theorem intersection_of_P_and_Q : P ∩ Q = {(1, -1)} := by
  sorry

end intersection_of_P_and_Q_l2458_245839


namespace almost_perfect_is_odd_square_l2458_245815

/-- Sum of divisors function -/
def sigma (N : ℕ+) : ℕ := sorry

/-- Definition of almost perfect number -/
def is_almost_perfect (N : ℕ+) : Prop :=
  sigma N = 2 * N.val + 1

/-- Main theorem: Every almost perfect number is a square of an odd number -/
theorem almost_perfect_is_odd_square (N : ℕ+) (h : is_almost_perfect N) :
  ∃ M : ℕ, N.val = M^2 ∧ Odd M := by sorry

end almost_perfect_is_odd_square_l2458_245815


namespace min_xy_value_l2458_245860

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x*y - x - 2*y = 4) :
  ∀ z, z = x*y → z ≥ 2 :=
by sorry

end min_xy_value_l2458_245860


namespace coordinate_system_change_l2458_245814

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a coordinate system in a 2D plane -/
structure CoordinateSystem where
  origin : Point2D

/-- Returns the coordinates of a point in a given coordinate system -/
def getCoordinates (p : Point2D) (cs : CoordinateSystem) : Point2D :=
  { x := p.x - cs.origin.x, y := p.y - cs.origin.y }

theorem coordinate_system_change 
  (A B : Point2D) 
  (csA csB : CoordinateSystem) 
  (h1 : csA.origin = A) 
  (h2 : csB.origin = B) 
  (h3 : getCoordinates B csA = { x := a, y := b }) :
  getCoordinates A csB = { x := -a, y := -b } := by
  sorry


end coordinate_system_change_l2458_245814


namespace square_properties_l2458_245854

theorem square_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ a^2 - 10 = 0 ∧ 3 < a ∧ a < 4 := by
  sorry

end square_properties_l2458_245854


namespace girls_combined_average_l2458_245809

structure School where
  boys_score : ℝ
  girls_score : ℝ
  combined_score : ℝ

def central : School := { boys_score := 68, girls_score := 72, combined_score := 70 }
def delta : School := { boys_score := 78, girls_score := 85, combined_score := 80 }

def combined_boys_score : ℝ := 74

theorem girls_combined_average (c d : ℝ) 
  (hc : c > 0) (hd : d > 0)
  (h_central : c * central.boys_score + c * central.girls_score = (c + c) * central.combined_score)
  (h_delta : d * delta.boys_score + d * delta.girls_score = (d + d) * delta.combined_score)
  (h_boys : (c * central.boys_score + d * delta.boys_score) / (c + d) = combined_boys_score) :
  (c * central.girls_score + d * delta.girls_score) / (c + d) = 79 := by
  sorry

end girls_combined_average_l2458_245809


namespace cube_pyramid_plane_pairs_l2458_245892

/-- A solid formed by a cube and a pyramid --/
structure CubePyramidSolid where
  cube_edges : Finset (Fin 12)
  pyramid_edges : Finset (Fin 5)

/-- Function to count pairs of edges that determine a plane --/
def count_plane_determining_pairs (solid : CubePyramidSolid) : ℕ :=
  sorry

/-- Theorem stating the number of edge pairs determining a plane --/
theorem cube_pyramid_plane_pairs :
  ∀ (solid : CubePyramidSolid),
  count_plane_determining_pairs solid = 82 :=
sorry

end cube_pyramid_plane_pairs_l2458_245892


namespace decagon_adjacent_vertices_probability_l2458_245873

/-- A decagon is a polygon with 10 sides -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices : Nat := 10

/-- The number of ways to choose 3 distinct vertices from a decagon -/
def total_choices : Nat := Nat.choose num_vertices 3

/-- The number of ways to choose 3 adjacent vertices in a decagon -/
def adjacent_choices : Nat := num_vertices

/-- The probability of choosing 3 adjacent vertices in a decagon -/
def prob_adjacent_vertices : Rat := adjacent_choices / total_choices

theorem decagon_adjacent_vertices_probability :
  prob_adjacent_vertices = 1 / 12 := by
  sorry

end decagon_adjacent_vertices_probability_l2458_245873


namespace sector_arc_length_l2458_245801

/-- Given a circular sector with circumference 4 and central angle 2 radians, 
    the arc length of the sector is 2. -/
theorem sector_arc_length (r : ℝ) (l : ℝ) : 
  l + 2 * r = 4 →  -- circumference of the sector
  l = 2 * r →      -- relationship between arc length and radius
  l = 2 :=         -- arc length is 2
by sorry

end sector_arc_length_l2458_245801


namespace det_matrix_eq_one_l2458_245820

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 2, 3]

theorem det_matrix_eq_one : Matrix.det matrix = 1 := by sorry

end det_matrix_eq_one_l2458_245820


namespace radhika_total_games_l2458_245871

def christmas_games : ℕ := 12
def birthday_games : ℕ := 8
def original_games_ratio : ℚ := 1 / 2

theorem radhika_total_games :
  let total_gift_games := christmas_games + birthday_games
  let original_games := (total_gift_games : ℚ) * original_games_ratio
  (original_games + total_gift_games : ℚ) = 30 := by
  sorry

end radhika_total_games_l2458_245871


namespace dans_purchases_cost_l2458_245866

/-- The total cost of Dan's purchases, given the cost of a snake toy, a cage, and finding a dollar bill. -/
theorem dans_purchases_cost (snake_toy_cost cage_cost found_money : ℚ) : 
  snake_toy_cost = 11.76 →
  cage_cost = 14.54 →
  found_money = 1 →
  snake_toy_cost + cage_cost - found_money = 25.30 := by
  sorry

end dans_purchases_cost_l2458_245866


namespace soccer_team_lineup_count_l2458_245838

theorem soccer_team_lineup_count :
  let total_players : ℕ := 15
  let roles : ℕ := 7
  total_players.factorial / (total_players - roles).factorial = 2541600 :=
by sorry

end soccer_team_lineup_count_l2458_245838


namespace initial_white_lights_correct_l2458_245843

/-- The number of white lights Malcolm had initially -/
def initial_white_lights : ℕ := 59

/-- The number of red lights Malcolm bought -/
def red_lights : ℕ := 12

/-- The number of blue lights Malcolm bought -/
def blue_lights : ℕ := 3 * red_lights

/-- The number of green lights Malcolm bought -/
def green_lights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy -/
def remaining_lights : ℕ := 5

/-- Theorem stating that the initial number of white lights is correct -/
theorem initial_white_lights_correct : 
  initial_white_lights = red_lights + blue_lights + green_lights + remaining_lights :=
sorry

end initial_white_lights_correct_l2458_245843


namespace range_of_x_l2458_245819

theorem range_of_x (M : Set ℝ) (h : M = {x ^ 2 | x : ℝ} ∪ {1}) :
  {x : ℝ | x ≠ 1 ∧ x ≠ -1} = {x : ℝ | ∃ y ∈ M, y = x ^ 2} := by
  sorry

end range_of_x_l2458_245819


namespace triangle_cosine_sum_l2458_245845

/-- For a triangle ABC with angles A, B, C satisfying A/B = B/C = 1/3, 
    the sum of cosines of these angles is (1 + √13) / 4 -/
theorem triangle_cosine_sum (A B C : Real) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  A / B = 1 / 3 →
  B / C = 1 / 3 →
  Real.cos A + Real.cos B + Real.cos C = (1 + Real.sqrt 13) / 4 := by
  sorry

end triangle_cosine_sum_l2458_245845


namespace new_range_theorem_l2458_245853

/-- Represents the number of mutual funds -/
def num_funds : ℕ := 150

/-- Represents the range of annual yield last year -/
def last_year_range : ℝ := 12500

/-- Represents the percentage increase for the first group of funds -/
def increase_group1 : ℝ := 0.12

/-- Represents the percentage increase for the second group of funds -/
def increase_group2 : ℝ := 0.17

/-- Represents the percentage increase for the third group of funds -/
def increase_group3 : ℝ := 0.22

/-- Represents the size of each group of funds -/
def group_size : ℕ := 50

/-- Theorem stating that the range of annual yield this year is $27,750 -/
theorem new_range_theorem : 
  ∃ (L H : ℝ), 
    H - L = last_year_range ∧ 
    (H * (1 + increase_group3)) - (L * (1 + increase_group1)) = 27750 :=
sorry

end new_range_theorem_l2458_245853


namespace bowling_ball_surface_area_l2458_245832

theorem bowling_ball_surface_area :
  ∀ d r A : ℝ,
  d = 9 →
  r = d / 2 →
  A = 4 * Real.pi * r^2 →
  A = 81 * Real.pi :=
by
  sorry

end bowling_ball_surface_area_l2458_245832


namespace solve_for_e_l2458_245836

theorem solve_for_e (x e : ℝ) (h1 : (10 * x + 2) / 4 - (3 * x - e) / 18 = (2 * x + 4) / 3)
                     (h2 : x = 0.3) : e = 6 := by
  sorry

end solve_for_e_l2458_245836


namespace rectangle_measurement_error_l2458_245840

theorem rectangle_measurement_error (L W : ℝ) (p : ℝ) (h_positive : L > 0 ∧ W > 0) :
  let measured_area := (1.05 * L) * (W * (1 - p))
  let actual_area := L * W
  let error_percent := |measured_area - actual_area| / actual_area
  error_percent = 0.008 → p = 0.04 := by
  sorry

end rectangle_measurement_error_l2458_245840


namespace circle_diameter_l2458_245874

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end circle_diameter_l2458_245874


namespace unique_three_digit_pair_l2458_245888

theorem unique_three_digit_pair : 
  ∃! (a b : ℕ), 100 ≤ a ∧ a < 1000 ∧ 100 ≤ b ∧ b < 1000 ∧ 1000 * a + b = 7 * a * b :=
by
  sorry

end unique_three_digit_pair_l2458_245888


namespace problem_statement_l2458_245878

theorem problem_statement (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * (Real.cos (π / 6 + α / 2))^2 + 1 = 7 / 3 := by
  sorry

end problem_statement_l2458_245878


namespace students_in_both_competitions_l2458_245886

theorem students_in_both_competitions 
  (total_students : ℕ) 
  (math_students : ℕ) 
  (physics_students : ℕ) 
  (no_competition_students : ℕ) 
  (h1 : total_students = 45) 
  (h2 : math_students = 32) 
  (h3 : physics_students = 28) 
  (h4 : no_competition_students = 5) :
  total_students - no_competition_students - 
  (math_students + physics_students - total_students + no_competition_students) = 20 :=
by sorry

end students_in_both_competitions_l2458_245886


namespace average_book_price_l2458_245835

/-- The average price of books Sandy bought, given the number of books and total cost from two shops. -/
theorem average_book_price 
  (shop1_books : ℕ) 
  (shop1_cost : ℕ) 
  (shop2_books : ℕ) 
  (shop2_cost : ℕ) 
  (h1 : shop1_books = 65) 
  (h2 : shop1_cost = 1480) 
  (h3 : shop2_books = 55) 
  (h4 : shop2_cost = 920) : 
  (shop1_cost + shop2_cost) / (shop1_books + shop2_books) = 20 := by
sorry

end average_book_price_l2458_245835


namespace smallest_period_five_cycles_l2458_245822

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def completes_n_cycles (f : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ T > 0, is_periodic f T ∧ n * T = b - a

theorem smallest_period_five_cycles (f : ℝ → ℝ) 
  (h : completes_n_cycles f 5 0 (2 * Real.pi)) :
  ∃ T > 0, is_periodic f T ∧ 
    (∀ T' > 0, is_periodic f T' → T ≤ T') ∧
    T = 2 * Real.pi / 5 := by
  sorry

end smallest_period_five_cycles_l2458_245822


namespace different_color_probability_l2458_245858

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 3
def green_chips : ℕ := 2

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_blue : ℚ := blue_chips / total_chips
  let p_red : ℚ := red_chips / total_chips
  let p_yellow : ℚ := yellow_chips / total_chips
  let p_green : ℚ := green_chips / total_chips
  
  p_blue * (1 - p_blue) + p_red * (1 - p_red) + 
  p_yellow * (1 - p_yellow) + p_green * (1 - p_green) = 91 / 128 := by
  sorry

end different_color_probability_l2458_245858


namespace sara_meets_bus_probability_l2458_245848

/-- Represents the time in minutes after 3:30 pm -/
def TimeAfter330 := { t : ℝ // 0 ≤ t ∧ t ≤ 60 }

/-- The bus arrives at a random time between 3:30 pm and 4:30 pm -/
def bus_arrival : TimeAfter330 := sorry

/-- Sara arrives at a random time between 3:30 pm and 4:30 pm -/
def sara_arrival : TimeAfter330 := sorry

/-- The bus waits for 40 minutes after arrival -/
def bus_wait_time : ℝ := 40

/-- The probability that Sara arrives while the bus is still waiting -/
def probability_sara_meets_bus : ℝ := sorry

theorem sara_meets_bus_probability :
  probability_sara_meets_bus = 2/3 := by sorry

end sara_meets_bus_probability_l2458_245848
