import Mathlib

namespace f_monotonicity_and_zero_point_l2786_278609

def f (t : ℝ) (x : ℝ) : ℝ := 4 * x^3 + 3 * t * x^2 - 6 * t^2 * x + t - 1

theorem f_monotonicity_and_zero_point :
  ∀ t : ℝ,
  (t > 0 →
    (∀ x y : ℝ, ((x < y ∧ y < -t) ∨ (t/2 < x ∧ x < y)) → f t x < f t y) ∧
    (∀ x y : ℝ, -t < x ∧ x < y ∧ y < t/2 → f t x > f t y)) ∧
  (t < 0 →
    (∀ x y : ℝ, ((x < y ∧ y < t/2) ∨ (-t < x ∧ x < y)) → f t x < f t y) ∧
    (∀ x y : ℝ, t/2 < x ∧ x < y ∧ y < -t → f t x > f t y)) ∧
  (t > 0 → ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f t x = 0) :=
by sorry

end f_monotonicity_and_zero_point_l2786_278609


namespace team_size_l2786_278630

/-- A soccer team with goalies, defenders, midfielders, and strikers -/
structure SoccerTeam where
  goalies : ℕ
  defenders : ℕ
  midfielders : ℕ
  strikers : ℕ

/-- The total number of players in a soccer team -/
def totalPlayers (team : SoccerTeam) : ℕ :=
  team.goalies + team.defenders + team.midfielders + team.strikers

/-- Theorem stating the total number of players in the given team -/
theorem team_size (team : SoccerTeam) 
  (h1 : team.goalies = 3)
  (h2 : team.defenders = 10)
  (h3 : team.midfielders = 2 * team.defenders)
  (h4 : team.strikers = 7) :
  totalPlayers team = 40 := by
  sorry

#eval totalPlayers { goalies := 3, defenders := 10, midfielders := 20, strikers := 7 }

end team_size_l2786_278630


namespace johns_final_weight_l2786_278632

/-- Calculates the final weight after a series of weight changes --/
def final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 0.9  -- 10% loss
  let weight2 := weight1 + 5           -- 5 pounds gain
  let weight3 := weight2 * 0.85        -- 15% loss
  let weight4 := weight3 + 8           -- 8 pounds gain
  weight4 * 0.8                        -- 20% loss

/-- Theorem stating that John's final weight is approximately 144.44 pounds --/
theorem johns_final_weight :
  ∃ ε > 0, |final_weight 220 - 144.44| < ε :=
sorry

end johns_final_weight_l2786_278632


namespace actual_distance_traveled_l2786_278679

/-- Proves that the actual distance traveled is 10 km given the conditions of the problem -/
theorem actual_distance_traveled (slow_speed fast_speed : ℝ) (extra_distance : ℝ) 
  (h1 : slow_speed = 5)
  (h2 : fast_speed = 15)
  (h3 : extra_distance = 20)
  (h4 : ∀ t, fast_speed * t = slow_speed * t + extra_distance) : 
  ∃ d, d = 10 ∧ slow_speed * (d / slow_speed) = d ∧ fast_speed * (d / slow_speed) = d + extra_distance :=
by
  sorry

end actual_distance_traveled_l2786_278679


namespace house_cost_proof_l2786_278660

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def num_cows : ℕ := 20
def cost_per_cow : ℕ := 1000
def num_chickens : ℕ := 100
def cost_per_chicken : ℕ := 5
def solar_install_hours : ℕ := 6
def solar_install_cost_per_hour : ℕ := 100
def solar_equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

def land_cost : ℕ := land_acres * land_cost_per_acre
def cows_cost : ℕ := num_cows * cost_per_cow
def chickens_cost : ℕ := num_chickens * cost_per_chicken
def solar_install_cost : ℕ := solar_install_hours * solar_install_cost_per_hour
def total_solar_cost : ℕ := solar_install_cost + solar_equipment_cost

theorem house_cost_proof :
  total_cost - (land_cost + cows_cost + chickens_cost + total_solar_cost) = 120000 := by
  sorry

end house_cost_proof_l2786_278660


namespace missing_fraction_proof_l2786_278672

theorem missing_fraction_proof :
  let given_fractions : List ℚ := [1/3, 1/2, -5/6, 1/4, -9/20, -5/6]
  let missing_fraction : ℚ := 56/30
  let total_sum : ℚ := 5/6
  (given_fractions.sum + missing_fraction = total_sum) := by
  sorry

end missing_fraction_proof_l2786_278672


namespace cube_edge_is_nine_l2786_278621

-- Define the dimensions of the cuboid
def cuboid_base : Real := 10
def cuboid_height : Real := 73

-- Define the volume difference between the cuboid and the cube
def volume_difference : Real := 1

-- Define the function to calculate the edge length of the cube
def cube_edge_length : Real :=
  (cuboid_base * cuboid_height - volume_difference) ^ (1/3)

-- Theorem statement
theorem cube_edge_is_nine :
  cube_edge_length = 9 := by
  sorry

end cube_edge_is_nine_l2786_278621


namespace impossible_tiling_l2786_278668

/-- Represents an L-tromino -/
structure LTromino :=
  (cells : Fin 3 → (Fin 5 × Fin 7))

/-- Represents a tiling of a 5x7 rectangle with L-trominos -/
structure Tiling :=
  (trominos : List LTromino)
  (coverage : Fin 5 → Fin 7 → ℕ)

/-- Theorem stating the impossibility of tiling a 5x7 rectangle with L-trominos 
    such that each cell is covered by the same number of trominos -/
theorem impossible_tiling : 
  ∀ (t : Tiling), ¬(∀ (i : Fin 5) (j : Fin 7), ∃ (k : ℕ), t.coverage i j = k) :=
sorry

end impossible_tiling_l2786_278668


namespace fruit_store_total_weight_l2786_278634

theorem fruit_store_total_weight 
  (boxes_sold : ℕ) 
  (weight_per_box : ℕ) 
  (remaining_weight : ℕ) 
  (h1 : boxes_sold = 14)
  (h2 : weight_per_box = 30)
  (h3 : remaining_weight = 80) :
  boxes_sold * weight_per_box + remaining_weight = 500 := by
sorry

end fruit_store_total_weight_l2786_278634


namespace katie_soccer_granola_l2786_278627

/-- The number of boxes of granola bars needed for a soccer game --/
def granola_boxes_needed (num_kids : ℕ) (bars_per_kid : ℕ) (bars_per_box : ℕ) : ℕ :=
  (num_kids * bars_per_kid + bars_per_box - 1) / bars_per_box

theorem katie_soccer_granola : granola_boxes_needed 30 2 12 = 5 := by
  sorry

end katie_soccer_granola_l2786_278627


namespace intersection_A_complement_B_l2786_278602

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define set A
def A : Set Nat := {2, 3, 5, 6}

-- Define set B
def B : Set Nat := {1, 3, 4, 6, 7}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 5} := by
  sorry

end intersection_A_complement_B_l2786_278602


namespace onion_sale_earnings_is_66_l2786_278603

/-- Calculates the money earned from selling onions given the initial quantities and conditions --/
def onion_sale_earnings (sally_onions fred_onions : ℕ) 
  (sara_plant_multiplier sara_harvest_multiplier : ℕ) 
  (onions_given_to_sara total_after_giving remaining_onions price_per_onion : ℕ) : ℕ :=
  let sara_planted := sara_plant_multiplier * sally_onions
  let sara_harvested := sara_harvest_multiplier * fred_onions
  let total_before_giving := sally_onions + fred_onions + sara_harvested
  let total_after_giving := total_before_giving - onions_given_to_sara
  let onions_sold := total_after_giving - remaining_onions
  onions_sold * price_per_onion

/-- Theorem stating that given the problem conditions, the earnings from selling onions is $66 --/
theorem onion_sale_earnings_is_66 : 
  onion_sale_earnings 5 9 3 2 4 24 6 3 = 66 := by
  sorry

end onion_sale_earnings_is_66_l2786_278603


namespace digit_sum_puzzle_l2786_278604

theorem digit_sum_puzzle : ∀ (a b c d e f g : ℕ),
  a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  b ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  c ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  d ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  e ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  f ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  g ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g →
  a + b + c = 24 →
  d + e + f + g = 14 →
  (b = e ∨ a = e ∨ c = e) →
  a + b + c + d + f + g = 30 :=
by sorry

end digit_sum_puzzle_l2786_278604


namespace anna_phone_chargers_l2786_278685

/-- The number of phone chargers Anna has -/
def phone_chargers : ℕ := sorry

/-- The number of laptop chargers Anna has -/
def laptop_chargers : ℕ := sorry

/-- The total number of chargers Anna has -/
def total_chargers : ℕ := 24

theorem anna_phone_chargers :
  (laptop_chargers = 5 * phone_chargers) →
  (phone_chargers + laptop_chargers = total_chargers) →
  phone_chargers = 4 := by
  sorry

end anna_phone_chargers_l2786_278685


namespace karen_start_time_l2786_278694

/-- Proves that Karen starts 4 minutes late in the car race with Tom -/
theorem karen_start_time (karen_speed tom_speed tom_distance karen_win_margin : ℝ) 
  (h1 : karen_speed = 60) 
  (h2 : tom_speed = 45)
  (h3 : tom_distance = 24)
  (h4 : karen_win_margin = 4) : 
  (tom_distance / tom_speed - (tom_distance + karen_win_margin) / karen_speed) * 60 = 4 := by
  sorry


end karen_start_time_l2786_278694


namespace expression_value_l2786_278663

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) : 3 * x + 5 - 4 * y = 6 := by
  sorry

end expression_value_l2786_278663


namespace mod_thirteen_problem_l2786_278635

theorem mod_thirteen_problem (a : ℤ) 
  (h1 : 0 < a) (h2 : a < 13) 
  (h3 : (53^2017 + a) % 13 = 0) : 
  a = 12 := by
  sorry

end mod_thirteen_problem_l2786_278635


namespace triangle_geometric_sequence_l2786_278658

theorem triangle_geometric_sequence (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  -- a, b, c form a geometric sequence
  (b ^ 2 = a * c) ∧
  -- Given trigonometric ratios
  (Real.sin B = 5 / 13) ∧
  (Real.cos B = 12 / (a * c)) →
  -- Conclusion
  a + c = 3 * Real.sqrt 7 := by
sorry

end triangle_geometric_sequence_l2786_278658


namespace sequence_problem_l2786_278645

theorem sequence_problem (a : Fin 100 → ℝ) 
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by
sorry

end sequence_problem_l2786_278645


namespace max_trees_on_road_l2786_278614

theorem max_trees_on_road (road_length : ℕ) (interval : ℕ) (h1 : road_length = 28) (h2 : interval = 4) :
  (road_length / interval) + 1 = 8 := by
  sorry

end max_trees_on_road_l2786_278614


namespace subset_relation_l2786_278662

def A : Set ℝ := {x : ℝ | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x : ℝ | a*x - 1 = 0}

theorem subset_relation :
  (¬(B (1/5) ⊆ A)) ∧
  (∀ a : ℝ, (B a ⊆ A) ↔ (a = 0 ∨ a = 1/3 ∨ a = 1/5)) := by
  sorry

end subset_relation_l2786_278662


namespace units_digit_of_F_F7_l2786_278628

-- Define the modified Fibonacci sequence
def modifiedFib : ℕ → ℕ
  | 0 => 3
  | 1 => 5
  | (n + 2) => modifiedFib (n + 1) + modifiedFib n

-- Function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_F_F7 :
  unitsDigit (modifiedFib (modifiedFib 7)) = 4 := by
  sorry

end units_digit_of_F_F7_l2786_278628


namespace drink_equality_l2786_278648

theorem drink_equality (x : ℝ) : 
  let eric_initial := x
  let sara_initial := 1.4 * x
  let eric_consumed := (2/3) * eric_initial
  let sara_consumed := (2/3) * sara_initial
  let eric_remaining := eric_initial - eric_consumed
  let sara_remaining := sara_initial - sara_consumed
  let transfer := (1/2) * sara_remaining + 3
  let eric_final := eric_consumed + transfer
  let sara_final := sara_consumed + (sara_remaining - transfer)
  eric_final = sara_final ∧ eric_final = 23 ∧ sara_final = 23 :=
by sorry

#check drink_equality

end drink_equality_l2786_278648


namespace salary_change_l2786_278656

theorem salary_change (S : ℝ) : 
  let increase := S * 1.2
  let decrease := increase * 0.8
  decrease = S * 0.96 := by sorry

end salary_change_l2786_278656


namespace max_value_of_expression_l2786_278643

theorem max_value_of_expression (a b c d : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c) (nonneg_d : 0 ≤ d)
  (sum_condition : a + b + c + d = 200)
  (a_condition : a = 2 * d) :
  a * b + b * c + c * d ≤ 42500 / 3 := by
sorry

end max_value_of_expression_l2786_278643


namespace complement_A_intersect_B_l2786_278678

open Set

-- Define the universal set U as ℝ
def U := ℝ

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {1, 2, 3, 4, 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {1, 4, 5} := by sorry

end complement_A_intersect_B_l2786_278678


namespace president_and_committee_count_l2786_278611

/-- The number of ways to choose a president and a 2-person committee -/
def choose_president_and_committee (total_people : ℕ) (people_over_30 : ℕ) : ℕ :=
  total_people * (people_over_30 * (people_over_30 - 1) / 2 + 
  (total_people - people_over_30) * people_over_30 * (people_over_30 - 1) / 2)

/-- Theorem stating the number of ways to choose a president and committee -/
theorem president_and_committee_count :
  choose_president_and_committee 10 6 = 120 := by sorry

end president_and_committee_count_l2786_278611


namespace games_missed_l2786_278605

theorem games_missed (total_games attended_games : ℕ) 
  (h1 : total_games = 89) 
  (h2 : attended_games = 47) : 
  total_games - attended_games = 42 := by
  sorry

end games_missed_l2786_278605


namespace multiples_of_four_between_100_and_350_l2786_278647

theorem multiples_of_four_between_100_and_350 : 
  (Finset.filter (fun n => n % 4 = 0 ∧ 100 < n ∧ n < 350) (Finset.range 350)).card = 62 := by
  sorry

end multiples_of_four_between_100_and_350_l2786_278647


namespace plane_parallel_from_skew_lines_l2786_278677

-- Define the types for planes and lines
variable (α β : Plane) (L m : Line)

-- Define the parallel relation between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the parallel relation between planes
def parallel_plane (p1 p2 : Plane) : Prop := sorry

-- Define skew lines
def skew_lines (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem plane_parallel_from_skew_lines 
  (h_skew : skew_lines L m) 
  (h_L_alpha : parallel_line_plane L α)
  (h_m_alpha : parallel_line_plane m α)
  (h_L_beta : parallel_line_plane L β)
  (h_m_beta : parallel_line_plane m β) :
  parallel_plane α β := sorry

end plane_parallel_from_skew_lines_l2786_278677


namespace compound_formula_l2786_278616

-- Define atomic weights
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00

-- Define the number of oxygen atoms
def num_O : ℕ := 3

-- Define the total molecular weight
def total_molecular_weight : ℝ := 102

-- Define the molecular formula
structure MolecularFormula where
  num_Al : ℕ
  num_O : ℕ

-- Theorem to prove
theorem compound_formula :
  ∃ (formula : MolecularFormula),
    formula.num_O = num_O ∧
    formula.num_Al * atomic_weight_Al + formula.num_O * atomic_weight_O = total_molecular_weight ∧
    formula = MolecularFormula.mk 2 3 := by
  sorry


end compound_formula_l2786_278616


namespace range_of_a_for_two_roots_l2786_278664

theorem range_of_a_for_two_roots (a : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∃ x y : ℝ, x ≠ y ∧ a^x = x ∧ a^y = y) → 
  1 < a ∧ a < Real.exp (1 / Real.exp 1) :=
sorry

end range_of_a_for_two_roots_l2786_278664


namespace imaginary_part_of_z_l2786_278626

theorem imaginary_part_of_z (z : ℂ) (h : Complex.I * (z - 4) = 3 + 2 * Complex.I) : 
  z.im = 3 := by sorry

end imaginary_part_of_z_l2786_278626


namespace shirt_cost_l2786_278697

theorem shirt_cost (total_money : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (money_left : ℕ) :
  total_money = 109 →
  num_shirts = 2 →
  pants_cost = 13 →
  money_left = 74 →
  ∃ shirt_cost : ℕ, shirt_cost * num_shirts + pants_cost = total_money - money_left ∧ shirt_cost = 11 :=
by sorry

end shirt_cost_l2786_278697


namespace hidden_primes_average_l2786_278633

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (a b : ℕ) : ℕ := a + b

theorem hidden_primes_average (p₁ p₂ p₃ : ℕ) 
  (h₁ : is_prime p₁) (h₂ : is_prime p₂) (h₃ : is_prime p₃)
  (h₄ : card_sum p₁ 51 = card_sum p₂ 72)
  (h₅ : card_sum p₂ 72 = card_sum p₃ 43)
  (h₆ : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h₇ : p₁ ≠ 51 ∧ p₂ ≠ 72 ∧ p₃ ≠ 43) :
  (p₁ + p₂ + p₃) / 3 = 56 / 3 :=
by sorry

end hidden_primes_average_l2786_278633


namespace right_triangle_area_l2786_278680

theorem right_triangle_area (a b c : ℝ) (ha : a^2 = 64) (hb : b^2 = 36) (hc : c^2 = 121) :
  (1/2) * a * b = 24 := by
sorry

end right_triangle_area_l2786_278680


namespace min_xy_value_l2786_278646

theorem min_xy_value (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) :
  ∀ z, x * y ≥ z → z ≤ 16 :=
by sorry

end min_xy_value_l2786_278646


namespace sum_first_15_odd_integers_l2786_278606

theorem sum_first_15_odd_integers : 
  (Finset.range 15).sum (fun n => 2 * n + 1) = 225 := by sorry

end sum_first_15_odd_integers_l2786_278606


namespace gcd_count_for_product_360_l2786_278637

theorem gcd_count_for_product_360 : 
  ∃! (s : Finset ℕ), 
    (∀ d ∈ s, d > 0 ∧ ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 360) ∧
    s.card = 12 := by
  sorry

end gcd_count_for_product_360_l2786_278637


namespace tangent_circle_height_l2786_278690

/-- A circle tangent to y = x^3 at two points -/
structure TangentCircle where
  a : ℝ  -- x-coordinate of the tangent point
  b : ℝ  -- y-coordinate of the circle's center
  r : ℝ  -- radius of the circle

/-- The circle is tangent to y = x^3 at (a, a^3) and (-a, a^3) -/
def is_tangent (c : TangentCircle) : Prop :=
  c.a^2 + (c.a^3 - c.b)^2 = c.r^2 ∧
  c.a^6 + (1 - 2*c.b)*c.a^3 + c.b^2 - c.r^2 = 0

/-- The center of the circle is higher than the tangent points by 1/2 -/
theorem tangent_circle_height (c : TangentCircle) (h : is_tangent c) : 
  c.b - c.a^3 = 1/2 :=
sorry

end tangent_circle_height_l2786_278690


namespace ln_exp_relationship_l2786_278698

theorem ln_exp_relationship :
  (∀ x : ℝ, (Real.log x > 0) → (Real.exp x > 1)) ∧
  (∃ x : ℝ, Real.exp x > 1 ∧ Real.log x ≤ 0) :=
by sorry

end ln_exp_relationship_l2786_278698


namespace collinear_points_unique_k_l2786_278666

/-- Three points are collinear if they lie on the same straight line -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- The theorem states that k = -33 is the unique value for which
    the points (1,4), (3,-2), and (6, k/3) are collinear -/
theorem collinear_points_unique_k :
  ∃! k : ℝ, collinear (1, 4) (3, -2) (6, k/3) ∧ k = -33 := by
  sorry

end collinear_points_unique_k_l2786_278666


namespace complex_square_root_of_negative_four_l2786_278657

theorem complex_square_root_of_negative_four :
  ∀ z : ℂ, z^2 = -4 ↔ z = 2*I ∨ z = -2*I :=
by sorry

end complex_square_root_of_negative_four_l2786_278657


namespace tiles_count_theorem_l2786_278619

/-- Represents a square floor tiled with congruent square tiles -/
structure TiledSquare where
  side_length : ℕ

/-- The number of tiles along the diagonals and central line of a tiled square -/
def diagonal_and_central_count (s : TiledSquare) : ℕ :=
  3 * s.side_length - 2

/-- The total number of tiles covering the floor -/
def total_tiles (s : TiledSquare) : ℕ :=
  s.side_length ^ 2

/-- Theorem stating that if the diagonal and central count is 55, 
    then the total number of tiles is 361 -/
theorem tiles_count_theorem (s : TiledSquare) :
  diagonal_and_central_count s = 55 → total_tiles s = 361 := by
  sorry

end tiles_count_theorem_l2786_278619


namespace min_a_for_inequality_l2786_278649

/-- The minimum value of a for which x^2 + ax + 1 ≥ 0 holds for all x ∈ (0, 1] is -2 -/
theorem min_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → x^2 + a*x + 1 ≥ 0) ↔ a ≥ -2 :=
sorry

end min_a_for_inequality_l2786_278649


namespace computer_operations_per_hour_l2786_278613

theorem computer_operations_per_hour :
  let additions_per_second : ℕ := 12000
  let multiplications_per_second : ℕ := 8000
  let seconds_per_hour : ℕ := 3600
  let total_operations_per_second : ℕ := additions_per_second + multiplications_per_second
  let operations_per_hour : ℕ := total_operations_per_second * seconds_per_hour
  operations_per_hour = 72000000 := by
sorry

end computer_operations_per_hour_l2786_278613


namespace binomial_probabilities_l2786_278650

/-- The probability of success in a single trial -/
def p : ℝ := 0.7

/-- The number of trials -/
def n : ℕ := 5

/-- The probability of failure in a single trial -/
def q : ℝ := 1 - p

/-- Binomial probability mass function -/
def binomialPMF (k : ℕ) : ℝ :=
  (n.choose k) * p^k * q^(n-k)

/-- The probability of at most 3 successes in 5 trials -/
def probAtMost3 : ℝ :=
  binomialPMF 0 + binomialPMF 1 + binomialPMF 2 + binomialPMF 3

/-- The probability of at least 4 successes in 5 trials -/
def probAtLeast4 : ℝ :=
  binomialPMF 4 + binomialPMF 5

theorem binomial_probabilities :
  probAtMost3 = 0.4718 ∧ probAtLeast4 = 0.5282 := by
  sorry

#eval probAtMost3
#eval probAtLeast4

end binomial_probabilities_l2786_278650


namespace construct_equilateral_triangle_l2786_278622

/-- A triangle with two 70° angles and one 40° angle -/
structure WoodenTriangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  two_70 : (angle1 = 70 ∧ angle2 = 70) ∨ (angle1 = 70 ∧ angle3 = 70) ∨ (angle2 = 70 ∧ angle3 = 70)
  one_40 : angle1 = 40 ∨ angle2 = 40 ∨ angle3 = 40

/-- An equilateral triangle has three 60° angles -/
def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = 60 ∧ b = 60 ∧ c = 60

/-- The theorem stating that an equilateral triangle can be constructed using only the wooden triangle -/
theorem construct_equilateral_triangle (wt : WoodenTriangle) :
  ∃ a b c : ℝ, is_equilateral_triangle a b c ∧
  (∃ (n : ℕ), n > 0 ∧ a + b + c = n * (wt.angle1 + wt.angle2 + wt.angle3)) :=
sorry

end construct_equilateral_triangle_l2786_278622


namespace no_equal_xyz_l2786_278670

theorem no_equal_xyz : ¬∃ t : ℝ, (1 - 3*t = 2*t - 3) ∧ (1 - 3*t = 4*t^2 - 5*t + 1) := by
  sorry

end no_equal_xyz_l2786_278670


namespace equality_of_sets_l2786_278636

theorem equality_of_sets (x y a : ℝ) : 
  (3 * x^2 = x^2 + x^2 + x^2) ∧ 
  ((x - y)^2 = (y - x)^2) ∧ 
  ((a^2)^3 = (a^3)^2) := by
sorry

end equality_of_sets_l2786_278636


namespace third_artist_set_duration_l2786_278618

/-- The duration of the music festival in minutes -/
def festival_duration : ℕ := 6 * 60

/-- The duration of the first artist's set in minutes -/
def first_artist_set : ℕ := 70 + 5

/-- The duration of the second artist's set in minutes -/
def second_artist_set : ℕ := 15 * 4 + 6 * 7 + 15 + 2 * 10

/-- The duration of the third artist's set in minutes -/
def third_artist_set : ℕ := festival_duration - first_artist_set - second_artist_set

theorem third_artist_set_duration : third_artist_set = 148 := by
  sorry

end third_artist_set_duration_l2786_278618


namespace f_properties_l2786_278683

def f (x : ℕ) : ℕ := x % 2

def g (x : ℕ) : ℕ := x % 3

theorem f_properties :
  (∀ x : ℕ, f (2 * x) = 0) ∧
  (∀ x : ℕ, f x + f (x + 3) = 1) := by
  sorry

end f_properties_l2786_278683


namespace total_animals_seen_l2786_278631

theorem total_animals_seen (initial_beavers initial_chipmunks : ℕ) : 
  initial_beavers = 35 →
  initial_chipmunks = 60 →
  (initial_beavers + initial_chipmunks) + (3 * initial_beavers + (initial_chipmunks - 15)) = 245 := by
  sorry

end total_animals_seen_l2786_278631


namespace z_in_first_quadrant_l2786_278610

/-- Given that i is the imaginary unit and zi = 2i - z, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (i : ℂ) (z : ℂ) 
  (h_i : i * i = -1) 
  (h_z : z * i = 2 * i - z) : 
  Real.sqrt 2 / 2 < z.re ∧ 0 < z.im :=
by sorry

end z_in_first_quadrant_l2786_278610


namespace total_sales_is_28_l2786_278617

/-- The number of crates of eggs Gabrielle sells on Monday -/
def monday_sales : ℕ := 5

/-- The number of crates of eggs Gabrielle sells on Tuesday -/
def tuesday_sales : ℕ := 2 * monday_sales

/-- The number of crates of eggs Gabrielle sells on Wednesday -/
def wednesday_sales : ℕ := tuesday_sales - 2

/-- The number of crates of eggs Gabrielle sells on Thursday -/
def thursday_sales : ℕ := tuesday_sales / 2

/-- The total number of crates of eggs Gabrielle sells over 4 days -/
def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales

theorem total_sales_is_28 : total_sales = 28 := by
  sorry

end total_sales_is_28_l2786_278617


namespace john_non_rent_expenses_l2786_278691

/-- Represents John's computer business finances --/
structure ComputerBusiness where
  parts_cost : ℝ
  selling_price_multiplier : ℝ
  computers_per_month : ℕ
  monthly_rent : ℝ
  monthly_profit : ℝ

/-- Calculates the non-rent extra expenses for John's computer business --/
def non_rent_extra_expenses (business : ComputerBusiness) : ℝ :=
  let selling_price := business.parts_cost * business.selling_price_multiplier
  let total_revenue := selling_price * business.computers_per_month
  let total_cost_components := business.parts_cost * business.computers_per_month
  let total_expenses := total_revenue - business.monthly_profit
  total_expenses - business.monthly_rent - total_cost_components

/-- Theorem stating that John's non-rent extra expenses are $3000 per month --/
theorem john_non_rent_expenses :
  let john_business : ComputerBusiness := {
    parts_cost := 800,
    selling_price_multiplier := 1.4,
    computers_per_month := 60,
    monthly_rent := 5000,
    monthly_profit := 11200
  }
  non_rent_extra_expenses john_business = 3000 := by
  sorry

end john_non_rent_expenses_l2786_278691


namespace roque_bike_time_l2786_278625

/-- Represents the time in hours for Roque's commute -/
structure CommuteTime where
  walk_one_way : ℝ
  bike_one_way : ℝ
  walk_trips_per_week : ℕ
  bike_trips_per_week : ℕ
  total_time_per_week : ℝ

/-- Theorem stating that given the conditions, Roque's bike ride to work takes 1 hour -/
theorem roque_bike_time (c : CommuteTime)
  (h1 : c.walk_one_way = 2)
  (h2 : c.walk_trips_per_week = 3)
  (h3 : c.bike_trips_per_week = 2)
  (h4 : c.total_time_per_week = 16)
  (h5 : c.total_time_per_week = 2 * c.walk_one_way * c.walk_trips_per_week + 2 * c.bike_one_way * c.bike_trips_per_week) :
  c.bike_one_way = 1 := by
  sorry

end roque_bike_time_l2786_278625


namespace smallest_angle_is_90_l2786_278692

/-- Represents a trapezoid with angles in arithmetic sequence -/
structure ArithmeticTrapezoid where
  -- The smallest angle
  a : ℝ
  -- The common difference in the arithmetic sequence
  d : ℝ
  -- Assertion that angles are in arithmetic sequence
  angle_sequence : List ℝ := [a, a + d, a + 2*d, a + 3*d]
  -- Assertion that the sum of any two consecutive angles is 180°
  consecutive_sum : a + (a + d) = 180 ∧ (a + d) + (a + 2*d) = 180 ∧ (a + 2*d) + (a + 3*d) = 180
  -- Assertion that the second largest angle is 150°
  second_largest : a + 2*d = 150

/-- 
Theorem: In a trapezoid where the angles form an arithmetic sequence 
and the second largest angle is 150°, the smallest angle measures 90°.
-/
theorem smallest_angle_is_90 (t : ArithmeticTrapezoid) : t.a = 90 := by
  sorry

end smallest_angle_is_90_l2786_278692


namespace xy_sum_when_equation_zero_l2786_278669

theorem xy_sum_when_equation_zero (x y : ℝ) :
  (3 * x - y + 5)^2 + |2 * x - y + 3| = 0 → x + y = -3 := by
  sorry

end xy_sum_when_equation_zero_l2786_278669


namespace initial_capital_calculation_l2786_278607

def profit_distribution_ratio : ℚ := 2/3
def income_increase : ℕ := 200
def initial_profit_rate : ℚ := 5/100
def final_profit_rate : ℚ := 7/100

theorem initial_capital_calculation (P : ℚ) : 
  P * final_profit_rate * profit_distribution_ratio - 
  P * initial_profit_rate * profit_distribution_ratio = income_increase →
  P = 15000 := by
sorry

end initial_capital_calculation_l2786_278607


namespace line_points_comparison_l2786_278652

theorem line_points_comparison (m n b : ℝ) : 
  (m = -3 * (-2) + b) → (n = -3 * 3 + b) → m > n := by
  sorry

end line_points_comparison_l2786_278652


namespace symmetric_function_is_odd_and_periodic_l2786_278639

/-- A function satisfying specific symmetry properties -/
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (10 + x) = f (10 - x)) ∧ 
  (∀ x, f (20 - x) = -f (20 + x))

/-- A function is odd -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function is periodic with period T -/
def PeriodicFunction (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

/-- Theorem: A function satisfying specific symmetry properties is odd and periodic with period 40 -/
theorem symmetric_function_is_odd_and_periodic (f : ℝ → ℝ) 
  (h : SymmetricFunction f) : 
  OddFunction f ∧ PeriodicFunction f 40 := by
  sorry

end symmetric_function_is_odd_and_periodic_l2786_278639


namespace first_number_problem_l2786_278682

theorem first_number_problem (x y : ℤ) : y = 11 → x + (y + 3) = 19 → x = 5 := by
  sorry

end first_number_problem_l2786_278682


namespace f_intersects_axes_l2786_278640

-- Define the function
def f (x : ℝ) : ℝ := x + 1

-- Theorem statement
theorem f_intersects_axes : 
  (∃ x : ℝ, x < 0 ∧ f x = 0) ∧ 
  (∃ y : ℝ, y > 0 ∧ f 0 = y) := by
sorry

end f_intersects_axes_l2786_278640


namespace smallest_odd_divisor_of_difference_of_squares_l2786_278620

theorem smallest_odd_divisor_of_difference_of_squares (m n : ℕ) : 
  Odd m → Odd n → n < m → 
  (∃ (k : ℕ), ∀ (a b : ℕ), Odd a → Odd b → b < a → k ∣ (a^2 - b^2)) → 
  (∃ (d : ℕ), Odd d ∧ d ∣ (m^2 - n^2) ∧ 
    ∀ (e : ℕ), Odd e → e ∣ (m^2 - n^2) → d ≤ e) → 
  ∃ (d : ℕ), d = 1 ∧ Odd d ∧ d ∣ (m^2 - n^2) ∧ 
    ∀ (e : ℕ), Odd e → e ∣ (m^2 - n^2) → d ≤ e :=
sorry

end smallest_odd_divisor_of_difference_of_squares_l2786_278620


namespace line_equation_through_points_l2786_278608

/-- The line passing through points A(0, -5) and B(1, 0) has the equation y = 5x - 5 -/
theorem line_equation_through_points (x y : ℝ) : 
  (x = 0 ∧ y = -5) ∨ (x = 1 ∧ y = 0) → y = 5*x - 5 :=
by sorry

end line_equation_through_points_l2786_278608


namespace multiplication_factor_exists_l2786_278673

theorem multiplication_factor_exists (x : ℝ) (hx : x = 2.6666666666666665) :
  ∃ y : ℝ, Real.sqrt ((x * y) / 3) = x ∧ abs (y - 8) < 0.0000001 := by
  sorry

end multiplication_factor_exists_l2786_278673


namespace star_power_equality_l2786_278623

/-- The k-th smallest positive integer not in X -/
def f_X (X : Finset ℕ+) (k : ℕ+) : ℕ+ := sorry

/-- The * operation on finite sets of positive integers -/
def star (X Y : Finset ℕ+) : Finset ℕ+ :=
  X ∪ (Y.image (f_X X))

/-- Repeated application of star operation n times -/
def star_power (X : Finset ℕ+) : ℕ → Finset ℕ+
  | 0 => X
  | n + 1 => star X (star_power X n)

theorem star_power_equality {A B : Finset ℕ+} (hA : A.Nonempty) (hB : B.Nonempty)
    (h : star A B = star B A) :
    star_power A B.card = star_power B A.card := by sorry

end star_power_equality_l2786_278623


namespace solve_system_l2786_278655

theorem solve_system (x y : ℝ) (h1 : x - 2*y = 10) (h2 : x * y = 40) : y = 2.5 := by
  sorry

end solve_system_l2786_278655


namespace inequality_proof_equality_condition_l2786_278687

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * (a^3 + b^3 + c^3 + 3) ≥ 3 * (a + 1) * (b + 1) * (c + 1) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * (a^3 + b^3 + c^3 + 3) = 3 * (a + 1) * (b + 1) * (c + 1) ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end inequality_proof_equality_condition_l2786_278687


namespace water_segment_length_l2786_278684

/-- Represents the problem of finding the length of a water segment in a journey --/
theorem water_segment_length 
  (total_distance : ℝ) 
  (find_probability : ℝ) 
  (h1 : total_distance = 2500) 
  (h2 : find_probability = 7/10) : 
  ∃ water_length : ℝ, 
    water_length / total_distance = 1 - find_probability ∧ 
    water_length = 750 := by
  sorry

end water_segment_length_l2786_278684


namespace quadratic_function_coefficient_l2786_278642

theorem quadratic_function_coefficient (a b c : ℤ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c) →
  (2, -3) = (-(b / (2 * a)), -(b^2 - 4 * a * c) / (4 * a)) →
  1 = a * 0^2 + b * 0 + c →
  6 = a * 5^2 + b * 5 + c →
  a = 1 := by sorry

end quadratic_function_coefficient_l2786_278642


namespace range_of_f_l2786_278612

/-- The function f defined on real numbers. -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The range of f is [1, +∞) -/
theorem range_of_f : Set.range f = {y : ℝ | y ≥ 1} := by
  sorry

end range_of_f_l2786_278612


namespace triangle_side_length_l2786_278653

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively

-- State the theorem
theorem triangle_side_length 
  (ABC : Triangle) 
  (h1 : 2 * (ABC.b * Real.cos ABC.A + ABC.a * Real.cos ABC.B) = ABC.c ^ 2)
  (h2 : ABC.b = 3)
  (h3 : 3 * Real.cos ABC.A = 1) :
  ABC.a = 3 := by
  sorry


end triangle_side_length_l2786_278653


namespace barn_size_calculation_barn_size_is_1000_l2786_278600

/-- Given a property with a house and a barn, calculate the size of the barn. -/
theorem barn_size_calculation (price_per_sqft : ℝ) (house_size : ℝ) (total_value : ℝ) : ℝ :=
  let house_value := price_per_sqft * house_size
  let barn_value := total_value - house_value
  barn_value / price_per_sqft

/-- The size of the barn is 1000 square feet. -/
theorem barn_size_is_1000 :
  barn_size_calculation 98 2400 333200 = 1000 := by
  sorry

end barn_size_calculation_barn_size_is_1000_l2786_278600


namespace beths_sister_age_l2786_278695

theorem beths_sister_age (beth_age : ℕ) (future_years : ℕ) (sister_age : ℕ) : 
  beth_age = 18 → 
  future_years = 8 → 
  beth_age + future_years = 2 * (sister_age + future_years) → 
  sister_age = 5 := by
sorry

end beths_sister_age_l2786_278695


namespace binomial_expansion_sum_abs_coeff_l2786_278693

theorem binomial_expansion_sum_abs_coeff :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 32 := by
sorry

end binomial_expansion_sum_abs_coeff_l2786_278693


namespace absolute_value_equality_l2786_278696

theorem absolute_value_equality (x : ℝ) :
  |x - 3| = |x - 5| → x = 4 := by
sorry

end absolute_value_equality_l2786_278696


namespace equality_of_fractions_l2786_278681

theorem equality_of_fractions (a b : ℝ) 
  (h1 : a^2 + b^2 = a^2 * b^2) 
  (h2 : |a| ≠ 1) 
  (h3 : |b| ≠ 1) : 
  a^7 / (1 - a)^2 - a^7 / (1 + a)^2 = b^7 / (1 - b)^2 - b^7 / (1 + b)^2 := by
  sorry

end equality_of_fractions_l2786_278681


namespace all_propositions_true_l2786_278675

def S (m n : ℝ) := {x : ℝ | m ≤ x ∧ x ≤ n}

theorem all_propositions_true 
  (m n : ℝ) 
  (h_nonempty : (S m n).Nonempty) 
  (h_closure : ∀ x ∈ S m n, x^2 ∈ S m n) :
  (m = 1 → S m n = {1}) ∧ 
  (m = -1/2 → 1/4 ≤ n ∧ n ≤ 1) ∧ 
  (n = 1/2 → -Real.sqrt 2/2 ≤ m ∧ m ≤ 0) :=
by sorry

end all_propositions_true_l2786_278675


namespace min_ratio_of_valid_partition_l2786_278644

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_valid_partition (partition : List ℕ × List ℕ) : Prop :=
  let (group1, group2) := partition
  (group1 ++ group2).toFinset = Finset.range 30
  ∧ (group1.prod % group2.prod = 0)

def ratio (partition : List ℕ × List ℕ) : ℚ :=
  let (group1, group2) := partition
  (group1.prod : ℚ) / (group2.prod : ℚ)

theorem min_ratio_of_valid_partition :
  ∀ partition : List ℕ × List ℕ,
    is_valid_partition partition →
    ratio partition ≥ 1077205 :=
sorry

end min_ratio_of_valid_partition_l2786_278644


namespace min_sum_squares_l2786_278654

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), m = 4 ∧ (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
  (∃ p q r : ℝ, p^3 + q^3 + r^3 - 3*p*q*r = 8 ∧ p^2 + q^2 + r^2 = m) :=
sorry

end min_sum_squares_l2786_278654


namespace extra_bananas_l2786_278688

theorem extra_bananas (total_children : ℕ) (original_bananas_per_child : ℕ) (absent_children : ℕ) : 
  total_children = 740 →
  original_bananas_per_child = 2 →
  absent_children = 370 →
  (total_children * original_bananas_per_child) / (total_children - absent_children) - original_bananas_per_child = 2 := by
sorry

end extra_bananas_l2786_278688


namespace min_value_theorem_l2786_278661

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 100/x^3 ≥ 3 * 50^(2/5) + 6 * 50^(1/5) ∧
  ∃ y > 0, y^2 + 6*y + 100/y^3 = 3 * 50^(2/5) + 6 * 50^(1/5) :=
by sorry

end min_value_theorem_l2786_278661


namespace biology_books_count_l2786_278671

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of different chemistry books -/
def chem_books : ℕ := 8

/-- The total number of ways to choose 2 books of each type -/
def total_ways : ℕ := 1260

/-- The number of different biology books -/
def bio_books : ℕ := 10

theorem biology_books_count :
  choose_two bio_books * choose_two chem_books = total_ways :=
sorry

#check biology_books_count

end biology_books_count_l2786_278671


namespace saree_stripes_l2786_278638

theorem saree_stripes (brown gold blue : ℕ) : 
  gold = 3 * brown → 
  blue = 5 * gold → 
  brown = 4 → 
  blue = 60 := by
  sorry

end saree_stripes_l2786_278638


namespace digit_count_proof_l2786_278651

theorem digit_count_proof (total_count : ℕ) (available_digits : ℕ) 
  (h1 : total_count = 28672) 
  (h2 : available_digits = 8) : 
  ∃ n : ℕ, available_digits ^ n = total_count ∧ n = 5 := by
  sorry

end digit_count_proof_l2786_278651


namespace clock_hands_coincide_l2786_278601

/-- The rate at which the hour hand moves, in degrees per minute -/
def hour_hand_rate : ℝ := 0.5

/-- The rate at which the minute hand moves, in degrees per minute -/
def minute_hand_rate : ℝ := 6

/-- The position of the hour hand at 7:00, in degrees -/
def initial_hour_hand_position : ℝ := 210

/-- The time interval in which we're checking for coincidence -/
def time_interval : Set ℝ := {t | 30 ≤ t ∧ t ≤ 45}

/-- The theorem stating that the clock hands coincide once in the given interval -/
theorem clock_hands_coincide : ∃ t ∈ time_interval, 
  initial_hour_hand_position + hour_hand_rate * t = minute_hand_rate * t :=
sorry

end clock_hands_coincide_l2786_278601


namespace internally_tangent_circles_l2786_278624

/-- Given two circles, where one is internally tangent to the other, prove the possible values of m -/
theorem internally_tangent_circles (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 + 6*x - 8*y - 11 = 0) ∧ 
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6*x - 8*y - 11 = 0) →
  m = 1 ∨ m = 121 :=
by sorry

end internally_tangent_circles_l2786_278624


namespace liar_count_theorem_l2786_278689

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  size : Nat
  statement : Nat

/-- The problem setup -/
def islanderProblem : List IslanderGroup :=
  [⟨2, 2⟩, ⟨4, 4⟩, ⟨8, 8⟩, ⟨14, 14⟩]

/-- The total number of islanders -/
def totalIslanders : Nat := 28

/-- Function to determine if a statement is true given the actual number of liars -/
def isStatementTrue (group : IslanderGroup) (actualLiars : Nat) : Bool :=
  group.statement == actualLiars

/-- Function to determine the type of an islander based on their statement and the actual number of liars -/
def determineType (group : IslanderGroup) (actualLiars : Nat) : IslanderType :=
  if isStatementTrue group actualLiars then IslanderType.Knight else IslanderType.Liar

/-- Theorem stating that the number of liars is either 14 or 28 -/
theorem liar_count_theorem :
  ∃ (liarCount : Nat), (liarCount = 14 ∨ liarCount = 28) ∧
  (∀ (group : IslanderGroup), group ∈ islanderProblem →
    (determineType group liarCount = IslanderType.Liar) = (group.size ≤ liarCount)) ∧
  (liarCount ≤ totalIslanders) := by
  sorry

end liar_count_theorem_l2786_278689


namespace intersection_count_l2786_278674

/-- Represents a line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The five lines given in the problem --/
def line1 : Line := { a := 3, b := -2, c := 9 }
def line2 : Line := { a := 6, b := 4, c := -12 }
def line3 : Line := { a := 1, b := 0, c := 3 }
def line4 : Line := { a := 0, b := 1, c := 1 }
def line5 : Line := { a := 2, b := 1, c := -1 }

/-- Determines if two lines intersect --/
def intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- Counts the number of unique intersection points --/
def countIntersections (lines : List Line) : ℕ :=
  sorry

theorem intersection_count :
  countIntersections [line1, line2, line3, line4, line5] = 6 := by
  sorry

end intersection_count_l2786_278674


namespace williams_hot_dogs_left_l2786_278665

/-- Calculates the number of hot dogs left after selling in two periods -/
def hot_dogs_left (initial : ℕ) (sold_first : ℕ) (sold_second : ℕ) : ℕ :=
  initial - (sold_first + sold_second)

/-- Theorem stating that for William's hot dog sales, 45 hot dogs were left -/
theorem williams_hot_dogs_left : hot_dogs_left 91 19 27 = 45 := by
  sorry

end williams_hot_dogs_left_l2786_278665


namespace solution_triplets_l2786_278659

theorem solution_triplets (x y z : ℝ) : 
  x^2 * y + y^2 * z = 1040 ∧
  x^2 * z + z^2 * y = 260 ∧
  (x - y) * (y - z) * (z - x) = -540 →
  (x = 16 ∧ y = 4 ∧ z = 1) ∨ (x = 1 ∧ y = 16 ∧ z = 4) :=
by sorry

end solution_triplets_l2786_278659


namespace roots_equation_sum_l2786_278615

theorem roots_equation_sum (α β : ℝ) : 
  α^2 - 3*α + 1 = 0 → β^2 - 3*β + 1 = 0 → 3*α^5 + 7*β^4 = 817 := by
  sorry

end roots_equation_sum_l2786_278615


namespace equation_solution_l2786_278629

theorem equation_solution : 
  ∃! x : ℚ, (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 2)) ∧ x = -13/2 := by
  sorry

end equation_solution_l2786_278629


namespace fewer_noodles_than_pirates_l2786_278699

theorem fewer_noodles_than_pirates (noodles pirates : ℕ) : 
  noodles < pirates →
  pirates = 45 →
  noodles + pirates = 83 →
  pirates - noodles = 7 := by
sorry

end fewer_noodles_than_pirates_l2786_278699


namespace tylers_age_l2786_278676

/-- Given the ages of Tyler (T), his brother (B), and their sister (S),
    prove that Tyler's age is 5 years old. -/
theorem tylers_age (T B S : ℕ) : 
  T = B - 3 → 
  S = B + 2 → 
  S = 2 * T → 
  T + B + S = 30 → 
  T = 5 := by
  sorry

end tylers_age_l2786_278676


namespace set_relationships_l2786_278667

-- Define the sets M, N, and P
def M : Set ℚ := {x | ∃ m : ℤ, x = m + 1/6}
def N : Set ℚ := {x | ∃ n : ℤ, x = n/2 - 1/3}
def P : Set ℚ := {x | ∃ p : ℤ, x = p/2 + 1/6}

-- State the theorem
theorem set_relationships : M ⊆ N ∧ N = P := by sorry

end set_relationships_l2786_278667


namespace quadratic_distinct_roots_l2786_278641

theorem quadratic_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end quadratic_distinct_roots_l2786_278641


namespace abs_gt_iff_square_gt_l2786_278686

theorem abs_gt_iff_square_gt (x y : ℝ) : |x| > |y| ↔ x^2 > y^2 := by
  sorry

end abs_gt_iff_square_gt_l2786_278686
