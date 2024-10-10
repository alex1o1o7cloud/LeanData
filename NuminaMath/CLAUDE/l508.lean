import Mathlib

namespace not_prime_expression_l508_50865

theorem not_prime_expression (n : ℕ) (h : n > 2) :
  ¬ Nat.Prime (n^(n^n) - 6*n^n + 5) := by
  sorry

end not_prime_expression_l508_50865


namespace emilys_speed_l508_50890

/-- Given a distance of 10 miles traveled in 2 hours, prove the speed is 5 miles per hour -/
theorem emilys_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 10)
  (h2 : time = 2)
  (h3 : speed = distance / time) :
  speed = 5 := by
  sorry

end emilys_speed_l508_50890


namespace min_coach_handshakes_l508_50825

/-- Represents the number of handshakes in a gymnastics championship. -/
def total_handshakes : ℕ := 456

/-- Calculates the number of handshakes between gymnasts given the total number of gymnasts. -/
def gymnast_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the total number of gymnasts. -/
def total_gymnasts : ℕ := 30

/-- Represents the number of handshakes involving coaches. -/
def coach_handshakes : ℕ := total_handshakes - gymnast_handshakes total_gymnasts

/-- Theorem stating the minimum number of handshakes involving at least one coach. -/
theorem min_coach_handshakes : ∃ (k₁ k₂ : ℕ), k₁ + k₂ = coach_handshakes ∧ min k₁ k₂ = 1 :=
sorry

end min_coach_handshakes_l508_50825


namespace perpendicular_line_coordinates_l508_50830

/-- Given two points P and Q in a 2D plane, where Q has fixed coordinates
    and P's coordinates depend on a parameter 'a', prove that if the line PQ
    is perpendicular to the y-axis, then P has specific coordinates. -/
theorem perpendicular_line_coordinates 
  (Q : ℝ × ℝ) 
  (P : ℝ → ℝ × ℝ) 
  (h1 : Q = (2, -3))
  (h2 : ∀ a, P a = (2*a + 2, a - 5))
  (h3 : ∀ a, (P a).1 = Q.1) :
  ∃ a, P a = (6, -3) := by
  sorry

end perpendicular_line_coordinates_l508_50830


namespace solve_cubic_equation_l508_50889

theorem solve_cubic_equation (y : ℝ) :
  5 * y^(1/3) + 3 * (y / y^(2/3)) = 10 - y^(1/3) ↔ y = (10/9)^3 := by
  sorry

end solve_cubic_equation_l508_50889


namespace correct_ranking_l508_50823

/-- Represents a contestant's score -/
structure Score where
  value : ℝ
  positive : value > 0

/-- Represents the scores of the four contestants -/
structure ContestScores where
  ann : Score
  bill : Score
  carol : Score
  dick : Score
  sum_equality : bill.value + dick.value = ann.value + carol.value
  interchange_inequality : carol.value + bill.value > dick.value + ann.value
  carol_exceeds_sum : carol.value > ann.value + bill.value

/-- Represents the ranking of contestants -/
inductive Ranking
  | CDBA : Ranking  -- Carol, Dick, Bill, Ann
  | CDAB : Ranking  -- Carol, Dick, Ann, Bill
  | DCBA : Ranking  -- Dick, Carol, Bill, Ann
  | ACDB : Ranking  -- Ann, Carol, Dick, Bill
  | DCAB : Ranking  -- Dick, Carol, Ann, Bill

/-- The theorem stating that given the contest conditions, the correct ranking is CDBA -/
theorem correct_ranking (scores : ContestScores) : Ranking.CDBA = 
  (match scores with
  | ⟨ann, bill, carol, dick, _, _, _⟩ => 
      if carol.value > dick.value ∧ dick.value > bill.value ∧ bill.value > ann.value
      then Ranking.CDBA
      else Ranking.CDBA) := by
  sorry

end correct_ranking_l508_50823


namespace quadratic_function_coefficients_l508_50851

/-- Given a quadratic function f(x) = 2(x-3)^2 + 2, prove that it can be expressed
    as ax^2 + bx + c where a = 2, b = -12, and c = 20 -/
theorem quadratic_function_coefficients :
  ∃ (f : ℝ → ℝ), 
    (∀ x, f x = 2 * (x - 3)^2 + 2) ∧
    (∃ (a b c : ℝ), a = 2 ∧ b = -12 ∧ c = 20 ∧ 
      ∀ x, f x = a * x^2 + b * x + c) :=
by sorry

end quadratic_function_coefficients_l508_50851


namespace cosine_inequality_l508_50820

theorem cosine_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : 0 < x^2 + y^2 ∧ x^2 + y^2 ≤ π) :
  1 + Real.cos (x * y) ≥ Real.cos x + Real.cos y := by
  sorry

end cosine_inequality_l508_50820


namespace y_value_l508_50822

theorem y_value : ∃ y : ℝ, (3 * y) / 7 = 15 ∧ y = 35 := by
  sorry

end y_value_l508_50822


namespace total_spending_equals_49_l508_50821

/-- Represents the total amount spent by Paula and Olive at the kiddy gift shop -/
def total_spent (bracelet_price keychain_price coloring_book_price sticker_price toy_car_price : ℕ)
  (paula_bracelets paula_keychains paula_coloring_books paula_stickers : ℕ)
  (olive_coloring_books olive_bracelets olive_toy_cars olive_stickers : ℕ) : ℕ :=
  (bracelet_price * (paula_bracelets + olive_bracelets)) +
  (keychain_price * paula_keychains) +
  (coloring_book_price * (paula_coloring_books + olive_coloring_books)) +
  (sticker_price * (paula_stickers + olive_stickers)) +
  (toy_car_price * olive_toy_cars)

/-- Theorem stating that Paula and Olive's total spending equals $49 -/
theorem total_spending_equals_49 :
  total_spent 4 5 3 1 6 3 2 1 4 1 2 1 3 = 49 := by
  sorry

end total_spending_equals_49_l508_50821


namespace marble_probability_l508_50881

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) : 
  total = 84 →
  p_white = 1/4 →
  p_green = 1/7 →
  (total : ℚ) * (1 - p_white - p_green) / total = 17/28 := by
sorry

end marble_probability_l508_50881


namespace store_refusal_illegal_l508_50867

/-- Represents a banknote --/
structure Banknote where
  damaged : Bool
  torn : Bool

/-- Represents the store's action --/
inductive StoreAction
  | Accept
  | Refuse

/-- Defines what constitutes legal tender in Russia --/
def is_legal_tender (b : Banknote) : Bool :=
  b.damaged && b.torn

/-- Determines if a store's action is legal based on the banknote --/
def is_legal_action (b : Banknote) (a : StoreAction) : Prop :=
  is_legal_tender b → a = StoreAction.Accept

/-- The main theorem stating that refusing a torn banknote is illegal --/
theorem store_refusal_illegal (b : Banknote) (h1 : b.damaged) (h2 : b.torn) :
  ¬(is_legal_action b StoreAction.Refuse) := by
  sorry


end store_refusal_illegal_l508_50867


namespace binomial_2586_1_l508_50892

theorem binomial_2586_1 : Nat.choose 2586 1 = 2586 := by sorry

end binomial_2586_1_l508_50892


namespace road_length_theorem_l508_50852

/-- Represents the distance between two markers on the road. -/
structure MarkerDistance where
  fromA : ℕ
  fromB : ℕ

/-- The road between cities A and B -/
structure Road where
  length : ℕ
  marker1 : MarkerDistance
  marker2 : MarkerDistance

/-- Conditions for a valid road configuration -/
def isValidRoad (r : Road) : Prop :=
  (r.marker1.fromA + r.marker1.fromB = r.length) ∧
  (r.marker2.fromA + r.marker2.fromB = r.length) ∧
  (r.marker2.fromA = r.marker1.fromA + 10) ∧
  ((r.marker1.fromA = 2 * r.marker1.fromB ∨ r.marker1.fromB = 2 * r.marker1.fromA) ∧
   (r.marker2.fromA = 3 * r.marker2.fromB ∨ r.marker2.fromB = 3 * r.marker2.fromA))

theorem road_length_theorem :
  ∀ r : Road, isValidRoad r → (r.length = 120 ∨ r.length = 24) ∧
  (∀ d : ℕ, d ≠ 120 ∧ d ≠ 24 → ¬∃ r' : Road, r'.length = d ∧ isValidRoad r') :=
by sorry

end road_length_theorem_l508_50852


namespace lao_you_fen_max_profit_l508_50878

/-- Represents the cost and quantity information for Lao You Fen brands -/
structure LaoYouFen where
  cost_a : ℝ
  cost_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- Calculates the profit given the quantities of each brand -/
def profit (l : LaoYouFen) (qa qb : ℝ) : ℝ :=
  (13 - l.cost_a) * qa + (13 - l.cost_b) * qb

/-- Theorem stating the maximum profit for Lao You Fen sales -/
theorem lao_you_fen_max_profit (l : LaoYouFen) :
  l.cost_b = l.cost_a + 2 →
  2700 / l.cost_a = 3300 / l.cost_b →
  l.quantity_a + l.quantity_b = 800 →
  l.quantity_a ≤ 3 * l.quantity_b →
  (∀ qa qb : ℝ, qa + qb = 800 → qa ≤ 3 * qb → profit l qa qb ≤ 2800) ∧
  profit l 600 200 = 2800 :=
sorry

end lao_you_fen_max_profit_l508_50878


namespace max_a4_in_geometric_sequence_l508_50808

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- The theorem statement -/
theorem max_a4_in_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : IsPositiveGeometricSequence a)
  (h_sum : a 3 + a 5 = 4) :
  ∀ b : ℝ, a 4 ≤ b → b ≤ 2 :=
sorry

end max_a4_in_geometric_sequence_l508_50808


namespace ellipse_eccentricity_l508_50866

/-- The eccentricity of an ellipse with major axis length three times its minor axis length -/
theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a = 3 * b) (h5 : a^2 = b^2 + c^2) : c / a = 2 * Real.sqrt 2 / 3 := by
  sorry

end ellipse_eccentricity_l508_50866


namespace power_multiplication_l508_50876

theorem power_multiplication (a : ℝ) : a^2 * a = a^3 := by
  sorry

end power_multiplication_l508_50876


namespace cube_pyramid_sum_l508_50868

/-- A solid figure formed by constructing a pyramid on one face of a cube -/
structure CubePyramid where
  cube_faces : ℕ := 6
  cube_edges : ℕ := 12
  cube_vertices : ℕ := 8
  pyramid_new_faces : ℕ := 4
  pyramid_new_edges : ℕ := 4
  pyramid_new_vertex : ℕ := 1

/-- The total number of exterior faces in the CubePyramid -/
def total_faces (cp : CubePyramid) : ℕ := cp.cube_faces - 1 + cp.pyramid_new_faces

/-- The total number of edges in the CubePyramid -/
def total_edges (cp : CubePyramid) : ℕ := cp.cube_edges + cp.pyramid_new_edges

/-- The total number of vertices in the CubePyramid -/
def total_vertices (cp : CubePyramid) : ℕ := cp.cube_vertices + cp.pyramid_new_vertex

theorem cube_pyramid_sum (cp : CubePyramid) : 
  total_faces cp + total_edges cp + total_vertices cp = 34 := by
  sorry

end cube_pyramid_sum_l508_50868


namespace cloth_selling_price_l508_50847

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  quantity * (cost_price_per_meter + profit_per_meter)

/-- Proves that the total selling price of 85 meters of cloth with a profit of 20 Rs per meter and a cost price of 85 Rs per meter is 8925 Rs. -/
theorem cloth_selling_price :
  total_selling_price 85 20 85 = 8925 := by
  sorry

end cloth_selling_price_l508_50847


namespace arcade_spending_fraction_l508_50812

theorem arcade_spending_fraction (allowance : ℚ) (arcade_fraction : ℚ) : 
  allowance = 3/2 →
  (2/3 * (1 - arcade_fraction) * allowance = 2/5) →
  arcade_fraction = 3/5 := by
sorry

end arcade_spending_fraction_l508_50812


namespace license_plate_count_l508_50886

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of odd digits -/
def num_odd_digits : ℕ := 5

/-- The number of even digits -/
def num_even_digits : ℕ := 5

/-- The number of license plates with the given conditions -/
def num_license_plates : ℕ := num_letters^3 * num_odd_digits^2 * num_even_digits

theorem license_plate_count : num_license_plates = 2197000 := by
  sorry

end license_plate_count_l508_50886


namespace pencil_pen_difference_l508_50898

/-- Given a ratio of pens to pencils and the number of pencils, 
    calculate the difference between pencils and pens. -/
theorem pencil_pen_difference 
  (ratio_pens ratio_pencils num_pencils : ℕ) 
  (h_ratio : ratio_pens < ratio_pencils)
  (h_pencils : num_pencils = 42)
  (h_prop : ratio_pens * num_pencils = ratio_pencils * (num_pencils - 7)) :
  num_pencils - (num_pencils - 7) = 7 := by
  sorry

#check pencil_pen_difference 5 6 42

end pencil_pen_difference_l508_50898


namespace gcd_765432_654321_l508_50811

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end gcd_765432_654321_l508_50811


namespace root_sum_theorem_l508_50863

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := m * (x^2 - 2*x) + 2*x + 3

-- Define the condition for m1 and m2
def condition (m : ℝ) : Prop :=
  ∃ (a b : ℝ), quadratic_equation m a = 0 ∧ quadratic_equation m b = 0 ∧ a/b + b/a = 3/2

-- Theorem statement
theorem root_sum_theorem (m1 m2 : ℝ) :
  condition m1 ∧ condition m2 → m1/m2 + m2/m1 = 833/64 :=
by sorry

end root_sum_theorem_l508_50863


namespace intersection_implies_p_value_l508_50833

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (p : ℝ) (t : ℝ) : ℝ × ℝ := (2 * p * t, 2 * p * Real.sqrt t)
def C₂ (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the distance between two points
def distance (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- State the theorem
theorem intersection_implies_p_value (p : ℝ) :
  p > 0 →
  ∃ (A B : ℝ × ℝ) (t₁ t₂ θ₁ θ₂ : ℝ),
    C₁ p t₁ = A ∧
    C₁ p t₂ = B ∧
    C₂ θ₁ = Real.sqrt (A.1^2 + A.2^2) ∧
    C₂ θ₂ = Real.sqrt (B.1^2 + B.2^2) ∧
    distance A B = 2 * Real.sqrt 3 →
    p = 3 * Real.sqrt 3 / 2 := by
  sorry

end

end intersection_implies_p_value_l508_50833


namespace greatest_k_value_l508_50802

theorem greatest_k_value (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + k*x₁ + 5 = 0 ∧ 
    x₂^2 + k*x₂ + 5 = 0 ∧ 
    |x₁ - x₂| = Real.sqrt 61) →
  k ≤ 9 :=
sorry

end greatest_k_value_l508_50802


namespace brothers_puzzle_l508_50831

-- Define the possible identities
inductive Identity : Type
| Tweedledee : Identity
| Tweedledum : Identity

-- Define the days of the week
inductive DayOfWeek : Type
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

-- Define the brothers
structure Brother :=
(identity : Identity)

-- Define the scenario
structure Scenario :=
(brother1 : Brother)
(brother2 : Brother)
(day : DayOfWeek)

-- Define the statements of the brothers
def statement1 (s : Scenario) : Prop :=
  s.brother1.identity = Identity.Tweedledee → s.brother2.identity = Identity.Tweedledum

def statement2 (s : Scenario) : Prop :=
  s.brother2.identity = Identity.Tweedledum → s.brother1.identity = Identity.Tweedledee

-- Theorem: The scenario must be on Sunday and identities cannot be determined
theorem brothers_puzzle (s : Scenario) :
  (statement1 s ∧ statement2 s) →
  (s.day = DayOfWeek.Sunday ∧
   ¬(s.brother1.identity ≠ s.brother2.identity)) :=
by sorry

end brothers_puzzle_l508_50831


namespace last_three_digits_sum_sum_of_last_three_digits_l508_50895

theorem last_three_digits_sum (C : ℕ) : ∃ (k : ℕ), 7^(4+C) = 1000 * k + 601 := by sorry

theorem sum_of_last_three_digits (C : ℕ) : (6 + 0 + 1 : ℕ) = 7 := by sorry

end last_three_digits_sum_sum_of_last_three_digits_l508_50895


namespace factorial_fraction_simplification_l508_50891

theorem factorial_fraction_simplification :
  (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end factorial_fraction_simplification_l508_50891


namespace sum_of_first_10_terms_l508_50845

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2 - 4*n + 1

-- Theorem statement
theorem sum_of_first_10_terms : S 10 = 61 := by
  sorry

end sum_of_first_10_terms_l508_50845


namespace chef_michel_pies_l508_50840

/-- Represents the number of pies sold given the number of pieces and customers -/
def pies_sold (pieces : ℕ) (customers : ℕ) : ℕ :=
  (customers + pieces - 1) / pieces

/-- The total number of pies sold by Chef Michel -/
def total_pies : ℕ :=
  pies_sold 4 52 + pies_sold 8 76 + pies_sold 5 80 + pies_sold 10 130

/-- Theorem stating that Chef Michel sold 52 pies in total -/
theorem chef_michel_pies :
  total_pies = 52 := by
  sorry

#eval total_pies

end chef_michel_pies_l508_50840


namespace olga_sons_daughters_l508_50832

/-- Represents the family structure of Grandma Olga -/
structure OlgaFamily where
  daughters : Nat
  sons : Nat
  grandchildren : Nat
  daughters_sons : Nat
  sons_daughters : Nat

/-- The theorem stating the number of daughters each of Grandma Olga's sons has -/
theorem olga_sons_daughters (family : OlgaFamily) :
  family.daughters = 3 →
  family.sons = 3 →
  family.daughters_sons = 6 →
  family.grandchildren = 33 →
  family.sons_daughters = 5 := by
  sorry

end olga_sons_daughters_l508_50832


namespace unknown_number_proof_l508_50837

theorem unknown_number_proof (y : ℝ) : (12^2 : ℝ) * y^3 / 432 = 72 → y = 6 := by
  sorry

end unknown_number_proof_l508_50837


namespace sum_of_cubes_l508_50885

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by sorry

end sum_of_cubes_l508_50885


namespace max_rabbits_l508_50861

theorem max_rabbits (N : ℕ) 
  (long_ears : ℕ) (jump_far : ℕ) (both : ℕ) :
  long_ears = 13 →
  jump_far = 17 →
  both ≥ 3 →
  long_ears + jump_far - both ≤ N →
  N ≤ 27 :=
by sorry

end max_rabbits_l508_50861


namespace nathans_blanket_temp_l508_50877

def initial_temp : ℝ := 50
def type_a_effect : ℝ := 2
def type_b_effect : ℝ := 3
def total_type_a : ℕ := 8
def used_type_a : ℕ := total_type_a / 2
def total_type_b : ℕ := 6

theorem nathans_blanket_temp :
  initial_temp + (used_type_a : ℝ) * type_a_effect + (total_type_b : ℝ) * type_b_effect = 76 := by
  sorry

end nathans_blanket_temp_l508_50877


namespace roberto_chicken_price_l508_50869

/-- Represents the scenario of Roberto's chicken and egg expenses --/
structure ChickenEggScenario where
  num_chickens : ℕ
  weekly_feed_cost : ℚ
  eggs_per_chicken_per_week : ℕ
  previous_weekly_egg_cost : ℚ
  break_even_weeks : ℕ

/-- Calculates the price per chicken that makes raising chickens cheaper than buying eggs after a given number of weeks --/
def price_per_chicken (scenario : ChickenEggScenario) : ℚ :=
  (scenario.previous_weekly_egg_cost * scenario.break_even_weeks - scenario.weekly_feed_cost * scenario.break_even_weeks) / scenario.num_chickens

/-- The theorem states that given Roberto's specific scenario, the price per chicken is $20.25 --/
theorem roberto_chicken_price : 
  let scenario : ChickenEggScenario := {
    num_chickens := 4,
    weekly_feed_cost := 1,
    eggs_per_chicken_per_week := 3,
    previous_weekly_egg_cost := 2,
    break_even_weeks := 81
  }
  price_per_chicken scenario = 81/4 := by sorry

end roberto_chicken_price_l508_50869


namespace exactly_three_solutions_l508_50871

-- Define the system of equations
def satisfies_system (a b c : ℤ) : Prop :=
  a * b + c = 17 ∧ a + b * c = 19

-- Theorem statement
theorem exactly_three_solutions :
  ∃! (s : Finset (ℤ × ℤ × ℤ)), 
    (∀ (x : ℤ × ℤ × ℤ), x ∈ s ↔ satisfies_system x.1 x.2.1 x.2.2) ∧
    s.card = 3 := by
  sorry

end exactly_three_solutions_l508_50871


namespace painting_area_l508_50806

/-- The area of a rectangular painting inside a border -/
theorem painting_area (outer_width outer_height border_width : ℕ) : 
  outer_width = 100 ∧ outer_height = 150 ∧ border_width = 15 →
  (outer_width - 2 * border_width) * (outer_height - 2 * border_width) = 8400 :=
by sorry

end painting_area_l508_50806


namespace geometric_series_r_value_l508_50872

theorem geometric_series_r_value (a r : ℝ) (h1 : |r| < 1) : 
  (∑' n, a * r^n) = 15 ∧ (∑' n, a * r^(2*n)) = 6 → r = 2/3 := by
  sorry

end geometric_series_r_value_l508_50872


namespace tournament_handshakes_count_l508_50826

/-- Calculates the total number of handshakes in a basketball tournament -/
def tournament_handshakes (num_teams : Nat) (players_per_team : Nat) (num_referees : Nat) : Nat :=
  let total_players := num_teams * players_per_team
  let player_handshakes := (total_players * (total_players - players_per_team)) / 2
  let referee_handshakes := total_players * num_referees
  player_handshakes + referee_handshakes

/-- Theorem: In a tournament with 3 teams of 7 players each and 3 referees, 
    there are 210 handshakes in total -/
theorem tournament_handshakes_count :
  tournament_handshakes 3 7 3 = 210 := by
  sorry

end tournament_handshakes_count_l508_50826


namespace andrews_age_proof_l508_50857

/-- Andrew's age in years -/
def andrew_age : ℝ := 7.875

/-- Andrew's grandfather's age in years -/
def grandfather_age : ℝ := 9 * andrew_age

/-- Age difference between Andrew and his grandfather at Andrew's birth -/
def age_difference : ℝ := 63

theorem andrews_age_proof : 
  grandfather_age - andrew_age = age_difference ∧ 
  grandfather_age = 9 * andrew_age ∧ 
  andrew_age = 7.875 := by sorry

end andrews_age_proof_l508_50857


namespace triangle_existence_l508_50816

theorem triangle_existence (a b c A B C : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : A > 0 ∧ B > 0 ∧ C > 0)
  (h3 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h4 : A + B > C ∧ B + C > A ∧ C + A > B) :
  ∃ (x y z : ℝ), 
    x = Real.sqrt (a^2 + A^2) ∧
    y = Real.sqrt (b^2 + B^2) ∧
    z = Real.sqrt (c^2 + C^2) ∧
    x + y > z ∧ y + z > x ∧ z + x > y :=
sorry

end triangle_existence_l508_50816


namespace solution_set_f_plus_x_positive_range_of_a_for_full_solution_set_l508_50815

def f (x : ℝ) := |x - 2| - |x + 1|

theorem solution_set_f_plus_x_positive :
  {x : ℝ | f x + x > 0} = Set.union (Set.union (Set.Ioo (-3) (-1)) (Set.Ico (-1) 1)) (Set.Ioi 3) :=
sorry

theorem range_of_a_for_full_solution_set :
  {a : ℝ | ∀ x, f x ≤ a^2 - 2*a} = Set.union (Set.Iic (-1)) (Set.Ici 3) :=
sorry

end solution_set_f_plus_x_positive_range_of_a_for_full_solution_set_l508_50815


namespace sum_of_squares_l508_50841

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = 0) (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) :
  a^2 + b^2 + c^2 = 6/5 := by
  sorry

end sum_of_squares_l508_50841


namespace total_defective_rate_proof_l508_50813

/-- The fraction of products checked by worker y -/
def worker_y_fraction : ℝ := 0.1666666666666668

/-- The defective rate for products checked by worker x -/
def worker_x_defective_rate : ℝ := 0.005

/-- The defective rate for products checked by worker y -/
def worker_y_defective_rate : ℝ := 0.008

/-- The total defective rate for all products -/
def total_defective_rate : ℝ := 0.0055

theorem total_defective_rate_proof :
  (1 - worker_y_fraction) * worker_x_defective_rate +
  worker_y_fraction * worker_y_defective_rate = total_defective_rate := by
  sorry

end total_defective_rate_proof_l508_50813


namespace kalebs_savings_l508_50807

/-- The amount of money Kaleb needs to buy the toys -/
def total_cost (num_toys : ℕ) (price_per_toy : ℕ) : ℕ := num_toys * price_per_toy

/-- The amount of money Kaleb has saved initially -/
def initial_savings (total_cost additional_money : ℕ) : ℕ := total_cost - additional_money

/-- Theorem stating Kaleb's initial savings -/
theorem kalebs_savings (num_toys price_per_toy additional_money : ℕ) 
  (h1 : num_toys = 6)
  (h2 : price_per_toy = 6)
  (h3 : additional_money = 15) :
  initial_savings (total_cost num_toys price_per_toy) additional_money = 21 := by
  sorry

#check kalebs_savings

end kalebs_savings_l508_50807


namespace equation_solution_l508_50853

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ * (x₁ + 2) = 3 * x₁ + 6) ∧ 
  (x₂ * (x₂ + 2) = 3 * x₂ + 6) ∧ 
  x₁ = -2 ∧ x₂ = 3 ∧ 
  (∀ x : ℝ, x * (x + 2) = 3 * x + 6 → x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l508_50853


namespace miles_guitars_l508_50801

/-- Represents the number of musical instruments Miles owns. -/
structure MilesInstruments where
  guitars : ℕ
  trumpets : ℕ
  trombones : ℕ
  frenchHorns : ℕ

/-- The total number of Miles' fingers. -/
def numFingers : ℕ := 10

/-- The number of Miles' hands. -/
def numHands : ℕ := 2

/-- The number of Miles' heads. -/
def numHeads : ℕ := 1

/-- The total number of musical instruments Miles owns. -/
def totalInstruments : ℕ := 17

/-- Theorem stating the number of guitars Miles owns. -/
theorem miles_guitars :
  ∃ (m : MilesInstruments),
    m.trumpets = numFingers - 3
    ∧ m.trombones = numHeads + 2
    ∧ m.frenchHorns = m.guitars - 1
    ∧ m.guitars = numHands + 2
    ∧ m.trumpets + m.trombones + m.guitars + m.frenchHorns = totalInstruments
    ∧ m.guitars = 4 := by
  sorry

end miles_guitars_l508_50801


namespace symmetric_points_sum_l508_50880

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

theorem symmetric_points_sum (x y : ℝ) :
  symmetric_wrt_origin (x, -2) (3, y) → x + y = -1 := by
  sorry

end symmetric_points_sum_l508_50880


namespace sarahs_bowling_score_l508_50893

theorem sarahs_bowling_score (jessica greg sarah : ℕ) 
  (h1 : sarah = greg + 50)
  (h2 : greg = 2 * jessica)
  (h3 : (sarah + greg + jessica) / 3 = 110) :
  sarah = 162 := by
  sorry

end sarahs_bowling_score_l508_50893


namespace percent_profit_calculation_l508_50829

theorem percent_profit_calculation (C S : ℝ) (h : 60 * C = 50 * S) :
  (S - C) / C * 100 = 20 := by
  sorry

end percent_profit_calculation_l508_50829


namespace jordan_oreos_l508_50882

theorem jordan_oreos (jordan : ℕ) (james : ℕ) : 
  james = 2 * jordan + 3 → 
  jordan + james = 36 → 
  jordan = 11 := by
sorry

end jordan_oreos_l508_50882


namespace unique_integer_square_Q_l508_50855

/-- Q is a function that maps an integer to an integer -/
def Q (x : ℤ) : ℤ := x^4 + 4*x^3 + 6*x^2 - x + 41

/-- There exists exactly one integer x such that Q(x) is a perfect square -/
theorem unique_integer_square_Q : ∃! x : ℤ, ∃ y : ℤ, Q x = y^2 := by sorry

end unique_integer_square_Q_l508_50855


namespace pq_relation_l508_50818

theorem pq_relation (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 ∨ q = 4 - 2 * Real.sqrt 2 := by
  sorry

end pq_relation_l508_50818


namespace job_completion_proof_l508_50883

/-- The number of days it takes for A to complete the job alone -/
def days_A : ℝ := 10

/-- The number of days A and B work together -/
def days_together : ℝ := 4

/-- The fraction of the job completed after A and B work together -/
def fraction_completed : ℝ := 0.6

/-- The number of days it takes for B to complete the job alone -/
def days_B : ℝ := 20

theorem job_completion_proof :
  (days_together * (1 / days_A + 1 / days_B) = fraction_completed) ∧
  (days_B = 20) := by sorry

end job_completion_proof_l508_50883


namespace recurring_decimal_to_fraction_l508_50849

theorem recurring_decimal_to_fraction : (6 / 10 : ℚ) + (23 / 99 : ℚ) = 412 / 495 := by sorry

end recurring_decimal_to_fraction_l508_50849


namespace tan_315_degrees_l508_50854

theorem tan_315_degrees : Real.tan (315 * Real.pi / 180) = -1 := by
  sorry

end tan_315_degrees_l508_50854


namespace negation_of_universal_proposition_l508_50860

theorem negation_of_universal_proposition (m : ℝ) :
  (¬ ∀ x : ℝ, x^2 + 2*x + m ≤ 0) ↔ (∃ x : ℝ, x^2 + 2*x + m > 0) :=
by sorry

end negation_of_universal_proposition_l508_50860


namespace smallest_next_divisor_after_221_l508_50810

theorem smallest_next_divisor_after_221 (n : ℕ) (h1 : 1000 ≤ n ∧ n ≤ 9999) 
  (h2 : Even n) (h3 : 221 ∣ n) : 
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ d ≥ 238 ∧ ∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d :=
sorry

end smallest_next_divisor_after_221_l508_50810


namespace function_inequality_l508_50824

/-- Given a differentiable function f: ℝ → ℝ such that f'(x) + f(x) < 0 for all x in ℝ,
    prove that f(m-m^2) / e^(m^2-m+1) > f(1) for all m in ℝ. -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : ∀ x, deriv f x + f x < 0) :
    ∀ m, f (m - m^2) / Real.exp (m^2 - m + 1) > f 1 := by
  sorry

end function_inequality_l508_50824


namespace sqrt_n_factorial_inequality_l508_50856

theorem sqrt_n_factorial_inequality (n : ℕ) (hn : n > 0) :
  Real.sqrt n < (n.factorial : ℝ) ^ (1 / n : ℝ) ∧ (n.factorial : ℝ) ^ (1 / n : ℝ) < (n + 1 : ℝ) / 2 := by
  sorry

#check sqrt_n_factorial_inequality

end sqrt_n_factorial_inequality_l508_50856


namespace distance_minus_one_to_2023_l508_50887

/-- The distance between two points on a number line -/
def distance (a b : ℝ) : ℝ := |b - a|

/-- Theorem: The distance between points representing -1 and 2023 on a number line is 2024 -/
theorem distance_minus_one_to_2023 : distance (-1) 2023 = 2024 := by
  sorry

end distance_minus_one_to_2023_l508_50887


namespace coprime_product_and_sum_l508_50836

theorem coprime_product_and_sum (a b : ℤ) (h : Nat.Coprime a.natAbs b.natAbs) :
  Nat.Coprime (a * b).natAbs (a + b).natAbs := by
  sorry

end coprime_product_and_sum_l508_50836


namespace correct_factorization_l508_50839

theorem correct_factorization (a : ℝ) : -1 + 4 * a^2 = (2*a + 1) * (2*a - 1) := by
  sorry

end correct_factorization_l508_50839


namespace share_difference_l508_50859

/-- Represents the share of money for each person -/
structure Share where
  amount : ℕ

/-- Represents the distribution of money -/
structure Distribution where
  a : Share
  b : Share
  c : Share
  d : Share

/-- The proposition that a distribution follows the given proportion -/
def follows_proportion (dist : Distribution) : Prop :=
  6 * dist.b.amount = 3 * dist.a.amount ∧
  5 * dist.b.amount = 3 * dist.c.amount ∧
  4 * dist.b.amount = 3 * dist.d.amount

/-- The theorem to be proved -/
theorem share_difference (dist : Distribution) 
  (h1 : follows_proportion dist) 
  (h2 : dist.b.amount = 3000) : 
  dist.c.amount - dist.d.amount = 1000 := by
  sorry


end share_difference_l508_50859


namespace sandbox_sand_weight_l508_50896

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ := r.length * r.width

/-- Calculates the total area of two rectangles -/
def totalArea (r1 r2 : Rectangle) : ℕ := rectangleArea r1 + rectangleArea r2

/-- Calculates the number of bags needed to fill an area -/
def bagsNeeded (area : ℕ) (areaPerBag : ℕ) : ℕ := (area + areaPerBag - 1) / areaPerBag

/-- Theorem: The total weight of sand needed to fill the sandbox -/
theorem sandbox_sand_weight :
  let rectangle1 : Rectangle := ⟨50, 30⟩
  let rectangle2 : Rectangle := ⟨20, 15⟩
  let areaPerBag : ℕ := 80
  let weightPerBag : ℕ := 30
  let totalSandboxArea : ℕ := totalArea rectangle1 rectangle2
  let bags : ℕ := bagsNeeded totalSandboxArea areaPerBag
  bags * weightPerBag = 690 := by
  sorry

end sandbox_sand_weight_l508_50896


namespace distance_negative_five_to_negative_fourteen_l508_50814

/-- The distance between two points on a number line -/
def numberLineDistance (a b : ℝ) : ℝ := |a - b|

/-- Theorem: The distance between -5 and -14 on a number line is 9 -/
theorem distance_negative_five_to_negative_fourteen :
  numberLineDistance (-5) (-14) = 9 := by
  sorry

end distance_negative_five_to_negative_fourteen_l508_50814


namespace chip_notes_theorem_l508_50835

/-- Represents the number of pages Chip takes for each class every day -/
def pages_per_class : ℕ :=
  let days_per_week : ℕ := 5
  let classes_per_day : ℕ := 5
  let weeks : ℕ := 6
  let packs_used : ℕ := 3
  let sheets_per_pack : ℕ := 100
  let total_sheets : ℕ := packs_used * sheets_per_pack
  let total_days : ℕ := weeks * days_per_week
  let total_classes : ℕ := total_days * classes_per_day
  total_sheets / total_classes

theorem chip_notes_theorem : pages_per_class = 2 := by
  sorry

end chip_notes_theorem_l508_50835


namespace tank_capacity_l508_50819

theorem tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (used_gallons : ℕ) 
  (h1 : initial_fraction = 3/4)
  (h2 : final_fraction = 1/3)
  (h3 : used_gallons = 18) :
  ∃ (capacity : ℕ), 
    capacity * initial_fraction - capacity * final_fraction = used_gallons ∧ 
    capacity = 43 := by
sorry


end tank_capacity_l508_50819


namespace train_bridge_time_l508_50817

/-- Given a train of length 18 meters that passes a pole in 9 seconds,
    prove that it takes 27 seconds to pass a bridge of length 36 meters. -/
theorem train_bridge_time (train_length : ℝ) (pole_pass_time : ℝ) (bridge_length : ℝ) :
  train_length = 18 →
  pole_pass_time = 9 →
  bridge_length = 36 →
  (train_length + bridge_length) / (train_length / pole_pass_time) = 27 := by
  sorry

end train_bridge_time_l508_50817


namespace cary_earns_five_per_lawn_l508_50848

/-- The amount earned per lawn mowed --/
def amount_per_lawn (cost_of_shoes amount_saved lawns_per_weekend num_weekends : ℚ) : ℚ :=
  (cost_of_shoes - amount_saved) / (lawns_per_weekend * num_weekends)

/-- Theorem: Cary earns $5 per lawn mowed --/
theorem cary_earns_five_per_lawn :
  amount_per_lawn 120 30 3 6 = 5 := by
  sorry

end cary_earns_five_per_lawn_l508_50848


namespace locus_of_symmetric_point_l508_50842

/-- Given a parabola y = x^2, a fixed point A(a, 0) where a ≠ 0, and a moving point P on the parabola,
    the point Q symmetric to A with respect to P has the locus y = (1/2)(x + a)^2 -/
theorem locus_of_symmetric_point (a : ℝ) (ha : a ≠ 0) :
  ∀ x₁ y₁ x y : ℝ,
  y₁ = x₁^2 →                        -- P(x₁, y₁) is on the parabola y = x^2
  x = 2*a - x₁ →                     -- x-coordinate of Q
  y = -y₁ →                          -- y-coordinate of Q
  y = (1/2) * (x + a)^2 := by sorry

end locus_of_symmetric_point_l508_50842


namespace shaded_to_white_ratio_l508_50899

/-- A square divided into smaller squares where the vertices of inner squares 
    are at the midpoints of the sides of the outer squares -/
structure NestedSquares :=
  (side : ℝ)
  (is_positive : side > 0)

/-- The area of the shaded part in a NestedSquares structure -/
def shaded_area (s : NestedSquares) : ℝ := sorry

/-- The area of the white part in a NestedSquares structure -/
def white_area (s : NestedSquares) : ℝ := sorry

/-- Theorem stating that the ratio of shaded area to white area is 5:3 -/
theorem shaded_to_white_ratio (s : NestedSquares) : 
  shaded_area s / white_area s = 5 / 3 := by sorry

end shaded_to_white_ratio_l508_50899


namespace find_a_and_b_l508_50858

theorem find_a_and_b (a b d : ℤ) : 
  (∃ x : ℝ, Real.sqrt (x - a) + Real.sqrt (x + b) = 7 ∧ x = 12) →
  (∃ x : ℝ, Real.sqrt (x + a) + Real.sqrt (x + d) = 7 ∧ x = 13) →
  a = 3 ∧ b = 4 := by
sorry

end find_a_and_b_l508_50858


namespace systematic_sampling_smallest_number_l508_50834

theorem systematic_sampling_smallest_number 
  (total_classes : ℕ) 
  (selected_classes : ℕ) 
  (sum_of_selected : ℕ) 
  (h1 : total_classes = 24)
  (h2 : selected_classes = 4)
  (h3 : sum_of_selected = 48) :
  let interval := total_classes / selected_classes
  let smallest := (sum_of_selected - (selected_classes - 1) * selected_classes * interval / 2) / selected_classes
  smallest = 3 := by
sorry

end systematic_sampling_smallest_number_l508_50834


namespace triangle_construction_exists_l508_50888

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given points and line
variable (A B : Point)
variable (bisector : Line)

-- Define the reflection of a point over a line
def reflect (p : Point) (l : Line) : Point :=
  sorry

-- Define a function to check if three points are collinear
def collinear (p q r : Point) : Prop :=
  sorry

-- Define a function to check if a point lies on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

-- Define a function to calculate the distance between two points
def distance (p q : Point) : ℝ :=
  sorry

-- Theorem statement
theorem triangle_construction_exists :
  ∃ C : Point,
    point_on_line C bisector ∧
    distance A C = distance (reflect A bisector) C ∧
    ¬ collinear C A B :=
  sorry

end triangle_construction_exists_l508_50888


namespace eight_books_three_piles_l508_50864

/-- The number of ways to divide n identical objects into k non-empty groups -/
def divide_objects (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 5 ways to divide 8 identical books into 3 piles -/
theorem eight_books_three_piles : divide_objects 8 3 = 5 := by sorry

end eight_books_three_piles_l508_50864


namespace middle_dimension_at_least_six_l508_50875

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : Real
  width : Real
  height : Real

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : Real
  height : Real

/-- Checks if a cylinder fits upright in a crate -/
def cylinderFitsUpright (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.length)

theorem middle_dimension_at_least_six 
  (crate : CrateDimensions)
  (h1 : crate.length = 3)
  (h2 : crate.height = 12)
  (h3 : cylinderFitsUpright crate { radius := 3, height := 12 }) :
  crate.width ≥ 6 := by
  sorry

end middle_dimension_at_least_six_l508_50875


namespace small_cube_edge_length_l508_50884

theorem small_cube_edge_length 
  (initial_volume : ℝ) 
  (remaining_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (h1 : initial_volume = 1000) 
  (h2 : remaining_volume = 488) 
  (h3 : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), 
    edge_length = 4 ∧ 
    initial_volume - num_small_cubes * edge_length ^ 3 = remaining_volume :=
by sorry

end small_cube_edge_length_l508_50884


namespace smallest_beneficial_discount_l508_50894

theorem smallest_beneficial_discount : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → 
    (1 - m / 100) > (1 - 20 / 100) * (1 - 20 / 100) ∨
    (1 - m / 100) > (1 - 10 / 100) * (1 - 15 / 100) ∨
    (1 - m / 100) > (1 - 8 / 100) * (1 - 8 / 100) * (1 - 8 / 100)) ∧
  (1 - n / 100) ≤ (1 - 20 / 100) * (1 - 20 / 100) ∧
  (1 - n / 100) ≤ (1 - 10 / 100) * (1 - 15 / 100) ∧
  (1 - n / 100) ≤ (1 - 8 / 100) * (1 - 8 / 100) * (1 - 8 / 100) ∧
  n = 37 :=
by sorry

end smallest_beneficial_discount_l508_50894


namespace a_squared_minus_b_squared_eq_zero_l508_50805

def first_seven_multiples_of_seven : List ℕ := [7, 14, 21, 28, 35, 42, 49]

def first_three_multiples_of_fourteen : List ℕ := [14, 28, 42]

def a : ℚ := (first_seven_multiples_of_seven.sum : ℚ) / 7

def b : ℕ := first_three_multiples_of_fourteen[1]

theorem a_squared_minus_b_squared_eq_zero : a^2 - (b^2 : ℚ) = 0 := by
  sorry

end a_squared_minus_b_squared_eq_zero_l508_50805


namespace walnut_trees_in_park_l508_50897

theorem walnut_trees_in_park (current : ℕ) (planted : ℕ) (total : ℕ) :
  current + planted = total →
  planted = 55 →
  total = 77 →
  current = 22 := by
sorry

end walnut_trees_in_park_l508_50897


namespace discount_calculation_l508_50828

theorem discount_calculation (marked_price : ℝ) (discount_rate : ℝ) (num_articles : ℕ) 
  (h1 : marked_price = 15)
  (h2 : discount_rate = 0.4)
  (h3 : num_articles = 2) :
  marked_price * num_articles * (1 - discount_rate) = 18 :=
by sorry

end discount_calculation_l508_50828


namespace keyboard_warrior_disapproval_l508_50862

theorem keyboard_warrior_disapproval 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (sample_approving : ℕ) 
  (h1 : total_population = 9600) 
  (h2 : sample_size = 50) 
  (h3 : sample_approving = 14) :
  ⌊(total_population : ℚ) * ((sample_size - sample_approving) : ℚ) / (sample_size : ℚ)⌋ = 6912 := by
  sorry

#check keyboard_warrior_disapproval

end keyboard_warrior_disapproval_l508_50862


namespace chess_team_selection_l508_50809

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_team_selection :
  let total_boys : ℕ := 8
  let total_girls : ℕ := 10
  let boys_to_select : ℕ := 5
  let girls_to_select : ℕ := 3
  (choose total_boys boys_to_select) * (choose total_girls girls_to_select) = 6720 := by
sorry

end chess_team_selection_l508_50809


namespace arithmetic_geometric_sequence_property_problem_solution_l508_50804

/-- An arithmetic sequence with the property that the sequence of products of consecutive terms
    forms a geometric progression, and the first term is 1, is constant with all terms equal to 1. -/
theorem arithmetic_geometric_sequence_property (a : ℕ → ℝ) : 
  (∀ n, ∃ d, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∀ n, ∃ r, (a (n + 1) * a (n + 2)) / (a n * a (n + 1)) = r) →  -- geometric progression of products
  a 1 = 1 →  -- first term is 1
  ∀ n, a n = 1 :=  -- all terms are 1
by sorry

/-- The 2017th term of the sequence described in the problem is 1. -/
theorem problem_solution (a : ℕ → ℝ) :
  (∀ n, ∃ d, a (n + 1) - a n = d) →  -- arithmetic sequence
  (∀ n, ∃ r, (a (n + 1) * a (n + 2)) / (a n * a (n + 1)) = r) →  -- geometric progression of products
  a 1 = 1 →  -- first term is 1
  a 2017 = 1 :=  -- 2017th term is 1
by sorry

end arithmetic_geometric_sequence_property_problem_solution_l508_50804


namespace common_difference_is_neg_four_general_term_l508_50870

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  a_1 : a 1 = 23
  d : ℤ
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d
  a_6_positive : a 6 > 0
  a_7_negative : a 7 < 0

/-- The common difference of the arithmetic sequence is -4 -/
theorem common_difference_is_neg_four (seq : ArithmeticSequence) : seq.d = -4 := by
  sorry

/-- The general term of the arithmetic sequence is -4n + 27 -/
theorem general_term (seq : ArithmeticSequence) (n : ℕ) : seq.a n = -4 * n + 27 := by
  sorry

end common_difference_is_neg_four_general_term_l508_50870


namespace divisibility_equivalence_l508_50879

/-- Definition of the sequence a_n -/
def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 2 * a (n + 1) + a n

/-- Main theorem: 2^k divides a_n if and only if 2^k divides n -/
theorem divisibility_equivalence (k n : ℕ) :
  (2^k : ℤ) ∣ a n ↔ 2^k ∣ n :=
by sorry

end divisibility_equivalence_l508_50879


namespace fraction_value_l508_50850

theorem fraction_value : (2 * 3 + 4) / (2 + 3) = 2 := by
  sorry

end fraction_value_l508_50850


namespace largest_sum_proof_l508_50800

theorem largest_sum_proof : 
  let sums : List ℚ := [1/4 + 1/2, 1/4 + 1/9, 1/4 + 1/3, 1/4 + 1/10, 1/4 + 1/6]
  (∀ x ∈ sums, x ≤ 1/4 + 1/2) ∧ (1/4 + 1/2 = 3/4) := by
  sorry

end largest_sum_proof_l508_50800


namespace b_plus_c_equals_seven_l508_50846

theorem b_plus_c_equals_seven (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : c + d = 5) 
  (h3 : a + d = 2) : 
  b + c = 7 := by sorry

end b_plus_c_equals_seven_l508_50846


namespace residue_mod_17_l508_50827

theorem residue_mod_17 : (195 * 15 - 18 * 8 + 4) % 17 = 7 := by
  sorry

end residue_mod_17_l508_50827


namespace root_less_than_one_l508_50838

theorem root_less_than_one (p q x₁ x₂ : ℝ) : 
  x₁^2 + p*x₁ - q = 0 →
  x₂^2 + p*x₂ - q = 0 →
  x₁ > 1 →
  p + q + 3 > 0 →
  x₂ < 1 :=
by sorry

end root_less_than_one_l508_50838


namespace kennel_long_furred_dogs_l508_50874

/-- The number of long-furred dogs in a kennel --/
def long_furred_dogs (total : ℕ) (brown : ℕ) (neither : ℕ) (long_furred_brown : ℕ) : ℕ :=
  total - neither - brown + long_furred_brown

/-- Theorem stating the number of long-furred dogs in the kennel --/
theorem kennel_long_furred_dogs :
  long_furred_dogs 45 17 8 9 = 29 := by
  sorry

#eval long_furred_dogs 45 17 8 9

end kennel_long_furred_dogs_l508_50874


namespace z_equation_solution_l508_50803

theorem z_equation_solution :
  let z : ℝ := Real.sqrt ((Real.sqrt 29) / 2 + 7 / 2)
  ∃! (d e f : ℕ+),
    z^100 = 2*z^98 + 14*z^96 + 11*z^94 - z^50 + (d : ℝ)*z^46 + (e : ℝ)*z^44 + (f : ℝ)*z^40 ∧
    d + e + f = 205 := by
  sorry

end z_equation_solution_l508_50803


namespace range_of_m_l508_50844

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 :=
by
  sorry

-- Define the final result
def result : Set ℝ := {m | m ≥ 9}

end range_of_m_l508_50844


namespace electronics_not_all_on_sale_l508_50843

-- Define the universe of discourse
variable (E : Type) [Nonempty E]

-- Define the predicate for "on sale"
variable (on_sale : E → Prop)

-- Define the store
variable (store : Set E)

-- Assume the store is not empty
variable (h_store_nonempty : store.Nonempty)

-- The main theorem
theorem electronics_not_all_on_sale
  (h : ¬∀ (e : E), e ∈ store → on_sale e) :
  (∃ (e : E), e ∈ store ∧ ¬on_sale e) ∧
  (¬∀ (e : E), e ∈ store → on_sale e) :=
by sorry


end electronics_not_all_on_sale_l508_50843


namespace sine_of_sum_inverse_sine_tangent_l508_50873

theorem sine_of_sum_inverse_sine_tangent : 
  Real.sin (Real.arcsin (3/5) + Real.arctan 2) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sine_of_sum_inverse_sine_tangent_l508_50873
