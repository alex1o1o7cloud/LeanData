import Mathlib

namespace quadratic_equation_from_root_properties_l1499_149901

theorem quadratic_equation_from_root_properties (a b c : ℝ) (h_sum : b / a = -4) (h_product : c / a = 3) :
  ∃ (k : ℝ), k ≠ 0 ∧ k * (a * X^2 + b * X + c) = X^2 - 4*X + 3 :=
sorry

end quadratic_equation_from_root_properties_l1499_149901


namespace can_determine_native_types_l1499_149988

/-- Represents the type of native: Knight or Liar -/
inductive NativeType
| Knight
| Liar

/-- Represents a native on the island -/
structure Native where
  type : NativeType
  leftNeighborAge : ℕ
  rightNeighborAge : ℕ

/-- The circle of natives -/
def NativeCircle := Vector Native 50

/-- Represents the statements made by a native -/
structure Statement where
  declaredLeftAge : ℕ
  declaredRightAge : ℕ

/-- Function to get the statements of all natives -/
def getAllStatements (circle : NativeCircle) : Vector Statement 50 := sorry

/-- Predicate to check if a native's statement is consistent with their type -/
def isConsistentStatement (native : Native) (statement : Statement) : Prop :=
  match native.type with
  | NativeType.Knight => 
      statement.declaredLeftAge = native.leftNeighborAge ∧ 
      statement.declaredRightAge = native.rightNeighborAge
  | NativeType.Liar => 
      (statement.declaredLeftAge = native.leftNeighborAge + 1 ∧ 
       statement.declaredRightAge = native.rightNeighborAge - 1) ∨
      (statement.declaredLeftAge = native.leftNeighborAge - 1 ∧ 
       statement.declaredRightAge = native.rightNeighborAge + 1)

/-- Main theorem: It's always possible to determine the identity of each native -/
theorem can_determine_native_types (circle : NativeCircle) :
  ∃ (determinedTypes : Vector NativeType 50),
    ∀ (i : Fin 50), 
      (circle.get i).type = determinedTypes.get i ∧
      isConsistentStatement (circle.get i) ((getAllStatements circle).get i) :=
sorry

end can_determine_native_types_l1499_149988


namespace sugar_left_l1499_149970

/-- Given a recipe requiring 2 cups of sugar, if you can make 0.165 of the recipe,
    then you have 0.33 cups of sugar left. -/
theorem sugar_left (full_recipe : ℝ) (fraction_possible : ℝ) (sugar_left : ℝ) :
  full_recipe = 2 →
  fraction_possible = 0.165 →
  sugar_left = full_recipe * fraction_possible →
  sugar_left = 0.33 := by
sorry

end sugar_left_l1499_149970


namespace tree_planting_multiple_l1499_149939

/-- The number of trees planted by 4th graders -/
def trees_4th : ℕ := 30

/-- The number of trees planted by 5th graders -/
def trees_5th : ℕ := 2 * trees_4th

/-- The total number of trees planted by all grades -/
def total_trees : ℕ := 240

/-- The multiple of 5th graders' trees compared to 6th graders' trees -/
def m : ℕ := 3

/-- Theorem stating that m is the correct multiple -/
theorem tree_planting_multiple :
  m * trees_5th - 30 = total_trees - trees_4th - trees_5th := by
  sorry

#check tree_planting_multiple

end tree_planting_multiple_l1499_149939


namespace representatives_selection_theorem_l1499_149982

/-- The number of ways to select 3 representatives from 3 different companies -/
def selectRepresentatives (totalCompanies : ℕ) (companiesWithOneRep : ℕ) (repsFromSpecialCompany : ℕ) : ℕ :=
  Nat.choose repsFromSpecialCompany 1 * Nat.choose companiesWithOneRep 2 +
  Nat.choose companiesWithOneRep 3

/-- Theorem stating that the number of ways to select 3 representatives from 3 different companies
    out of 5 companies (where one company has 2 representatives and the others have 1 each) is 16 -/
theorem representatives_selection_theorem :
  selectRepresentatives 5 4 2 = 16 := by
  sorry

end representatives_selection_theorem_l1499_149982


namespace victoria_beacon_ratio_l1499_149902

/-- The population of Richmond -/
def richmond_population : ℕ := 3000

/-- The population of Beacon -/
def beacon_population : ℕ := 500

/-- The difference between Richmond's and Victoria's populations -/
def richmond_victoria_diff : ℕ := 1000

/-- The population of Victoria -/
def victoria_population : ℕ := richmond_population - richmond_victoria_diff

theorem victoria_beacon_ratio : 
  (victoria_population : ℚ) / (beacon_population : ℚ) = 4 := by
  sorry

end victoria_beacon_ratio_l1499_149902


namespace hyperbola_intersection_line_l1499_149945

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the midpoint condition
def is_midpoint (x₁ y₁ x₂ y₂ xₘ yₘ : ℝ) : Prop :=
  xₘ = (x₁ + x₂) / 2 ∧ yₘ = (y₁ + y₂) / 2

-- Main theorem
theorem hyperbola_intersection_line :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    hyperbola x₁ y₁ →
    hyperbola x₂ y₂ →
    is_midpoint x₁ y₁ x₂ y₂ 1 3 →
    (∀ x y, line x y ↔ (y - 3 = x - 1)) :=
by sorry

end hyperbola_intersection_line_l1499_149945


namespace range_of_a_l1499_149979

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | (4*x - 3)^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 1}

-- Define the property that ¬p is a necessary but not sufficient condition for ¬q
def necessary_not_sufficient (a : ℝ) : Prop :=
  (∀ x, x ∉ B a → x ∉ A) ∧ ¬(∀ x, x ∉ A → x ∉ B a)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l1499_149979


namespace sam_puppies_l1499_149978

theorem sam_puppies (initial_puppies : Float) (given_away : Float) :
  initial_puppies = 6.0 → given_away = 2.0 → initial_puppies - given_away = 4.0 := by
  sorry

end sam_puppies_l1499_149978


namespace rally_ticket_cost_l1499_149947

theorem rally_ticket_cost 
  (total_attendance : ℕ)
  (door_ticket_price : ℚ)
  (total_receipts : ℚ)
  (pre_rally_tickets : ℕ)
  (h1 : total_attendance = 750)
  (h2 : door_ticket_price = 2.75)
  (h3 : total_receipts = 1706.25)
  (h4 : pre_rally_tickets = 475) :
  ∃ (pre_rally_price : ℚ), 
    pre_rally_price * pre_rally_tickets + 
    door_ticket_price * (total_attendance - pre_rally_tickets) = total_receipts ∧
    pre_rally_price = 2 :=
by sorry

end rally_ticket_cost_l1499_149947


namespace popping_corn_probability_l1499_149950

theorem popping_corn_probability (white yellow blue : ℝ)
  (white_pop yellow_pop blue_pop : ℝ) :
  white = 1/2 →
  yellow = 1/3 →
  blue = 1/6 →
  white_pop = 3/4 →
  yellow_pop = 1/2 →
  blue_pop = 1/3 →
  (white * white_pop) / (white * white_pop + yellow * yellow_pop + blue * blue_pop) = 27/43 := by
  sorry

end popping_corn_probability_l1499_149950


namespace largest_n_is_max_factorization_exists_l1499_149918

/-- The largest value of n for which 4x^2 + nx + 96 can be factored as two linear factors with integer coefficients -/
def largest_n : ℕ := 385

/-- A structure representing the factorization of 4x^2 + nx + 96 -/
structure Factorization where
  a : ℤ
  b : ℤ
  h1 : (4 * X + a) * (X + b) = 4 * X^2 + largest_n * X + 96

/-- Theorem stating that largest_n is indeed the largest value for which the factorization exists -/
theorem largest_n_is_max :
  ∀ n : ℕ, n > largest_n →
    ¬∃ (f : Factorization), (4 * X + f.a) * (X + f.b) = 4 * X^2 + n * X + 96 :=
by sorry

/-- Theorem stating that a factorization exists for largest_n -/
theorem factorization_exists : ∃ (f : Factorization), True :=
by sorry

end largest_n_is_max_factorization_exists_l1499_149918


namespace min_difference_in_sample_l1499_149959

theorem min_difference_in_sample (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e →
  c = 12 →
  (a + b + c + d + e) / 5 = 10 →
  e - a ≥ 5 :=
by sorry

end min_difference_in_sample_l1499_149959


namespace quadruple_solutions_l1499_149925

def is_solution (a b c k : ℕ) : Prop :=
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ k > 0 ∧
  a^2 + b^2 + 16*c^2 = 9*k^2 + 1

theorem quadruple_solutions :
  ∀ a b c k : ℕ,
    is_solution a b c k ↔
      ((a, b, c, k) = (3, 3, 2, 3) ∨
       (a, b, c, k) = (3, 17, 3, 7) ∨
       (a, b, c, k) = (17, 3, 3, 7) ∨
       (a, b, c, k) = (3, 37, 3, 13) ∨
       (a, b, c, k) = (37, 3, 3, 13)) :=
by sorry

end quadruple_solutions_l1499_149925


namespace magnitude_z_squared_l1499_149907

-- Define the complex number z
def z : ℂ := 1 + Complex.I^5

-- Theorem statement
theorem magnitude_z_squared : Complex.abs (z^2) = 2 := by
  sorry

end magnitude_z_squared_l1499_149907


namespace largest_integer_with_remainder_l1499_149911

theorem largest_integer_with_remainder (n : ℕ) : 
  (∀ m : ℕ, m < 100 ∧ m % 7 = 4 → m ≤ n) ∧ 
  n < 100 ∧ 
  n % 7 = 4 → 
  n = 95 := by
sorry

end largest_integer_with_remainder_l1499_149911


namespace solve_for_m_l1499_149968

theorem solve_for_m : ∃ m : ℚ, 
  (∃ x y : ℚ, 3 * x - 4 * (m - 1) * y + 30 = 0 ∧ x = 2 ∧ y = -3) → 
  m = -2 := by
  sorry

end solve_for_m_l1499_149968


namespace polynomial_has_real_root_l1499_149921

theorem polynomial_has_real_root (b : ℝ) : 
  ∃ x : ℝ, x^4 + b*x^3 + 2*x^2 + b*x - 2 = 0 := by
  sorry

end polynomial_has_real_root_l1499_149921


namespace coefficient_proof_l1499_149933

theorem coefficient_proof (n : ℤ) :
  (∃! (count : ℕ), count = 25 ∧
    count = (Finset.filter (fun i => 1 < 4 * i + 7 ∧ 4 * i + 7 < 100) (Finset.range 200)).card) →
  ∃ (a : ℤ), ∀ (x : ℤ), (a * x + 7 = 4 * x + 7) :=
by sorry

end coefficient_proof_l1499_149933


namespace number_of_players_is_64_l1499_149992

/-- The cost of a pair of shoes in dollars -/
def shoe_cost : ℕ := 12

/-- The cost of a jersey in dollars -/
def jersey_cost : ℕ := shoe_cost + 8

/-- The cost of a cap in dollars -/
def cap_cost : ℕ := jersey_cost / 2

/-- The total cost for one player's equipment in dollars -/
def player_cost : ℕ := 2 * (shoe_cost + jersey_cost) + cap_cost

/-- The total expenses for all players' equipment in dollars -/
def total_expenses : ℕ := 4760

theorem number_of_players_is_64 : 
  ∃ n : ℕ, n * player_cost = total_expenses ∧ n = 64 := by
  sorry

end number_of_players_is_64_l1499_149992


namespace triangle_properties_l1499_149958

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : (Real.cos t.C) / (Real.sin t.C) = (Real.cos t.A + Real.cos t.B) / (Real.sin t.A + Real.sin t.B)) :
  t.C = π / 3 ∧ 
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
sorry

end triangle_properties_l1499_149958


namespace min_edges_theorem_l1499_149927

/-- A simple graph with 19998 vertices -/
structure Graph :=
  (vertices : Finset Nat)
  (edges : Finset (Nat × Nat))
  (simple : ∀ e ∈ edges, e.1 ≠ e.2)
  (vertex_count : vertices.card = 19998)

/-- A subgraph of G with 9999 vertices -/
def Subgraph (G : Graph) :=
  {G' : Graph | G'.vertices ⊆ G.vertices ∧ G'.edges ⊆ G.edges ∧ G'.vertices.card = 9999}

/-- The condition that any subgraph with 9999 vertices has at least 9999 edges -/
def SubgraphEdgeCondition (G : Graph) :=
  ∀ G' ∈ Subgraph G, G'.edges.card ≥ 9999

/-- The theorem stating that G has at least 49995 edges -/
theorem min_edges_theorem (G : Graph) (h : SubgraphEdgeCondition G) :
  G.edges.card ≥ 49995 := by
  sorry

end min_edges_theorem_l1499_149927


namespace line_parameterization_l1499_149909

def is_valid_parameterization (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), p = (a, 3 * a - 4, b) ∧ v = (1/3, 1, 1)

theorem line_parameterization 
  (p : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  is_valid_parameterization p v ↔
    (∃ (t : ℝ), 
      let (x, y, z) := p + t • v
      y = 3 * x - 4 ∧ z = t) :=
sorry

end line_parameterization_l1499_149909


namespace edges_ge_twice_faces_l1499_149969

/-- A bipartite planar graph. -/
structure BipartitePlanarGraph where
  V : Type* -- Vertices
  E : Type* -- Edges
  F : Type* -- Faces
  edge_count : ℕ
  face_count : ℕ
  is_bipartite : Prop
  is_planar : Prop
  edge_count_ge_two : edge_count ≥ 2

/-- Theorem: In a bipartite planar graph with at least 2 edges, 
    the number of edges is at least twice the number of faces. -/
theorem edges_ge_twice_faces (G : BipartitePlanarGraph) : 
  G.edge_count ≥ 2 * G.face_count := by
  sorry

end edges_ge_twice_faces_l1499_149969


namespace rectangular_park_diagonal_l1499_149917

theorem rectangular_park_diagonal (x y : ℝ) (h_positive : x > 0 ∧ y > 0) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end rectangular_park_diagonal_l1499_149917


namespace city_female_population_l1499_149924

/-- Calculates the female population of a city given specific demographic information. -/
theorem city_female_population
  (total_population : ℕ)
  (migrant_percentage : ℚ)
  (rural_migrant_percentage : ℚ)
  (local_female_percentage : ℚ)
  (rural_migrant_female_percentage : ℚ)
  (urban_migrant_female_percentage : ℚ)
  (h_total : total_population = 728400)
  (h_migrant : migrant_percentage = 35 / 100)
  (h_rural : rural_migrant_percentage = 20 / 100)
  (h_local_female : local_female_percentage = 48 / 100)
  (h_rural_female : rural_migrant_female_percentage = 30 / 100)
  (h_urban_female : urban_migrant_female_percentage = 40 / 100) :
  ∃ (female_population : ℕ), female_population = 324128 :=
by
  sorry


end city_female_population_l1499_149924


namespace round_trip_average_speed_l1499_149976

/-- Calculates the average speed of a round trip given the following conditions:
  * Distance traveled one way is 5280 feet (1 mile)
  * Speed northward is 3 minutes per mile
  * Rest time is 10 minutes
  * Speed southward is 3 miles per minute
-/
theorem round_trip_average_speed :
  let distance_feet : ℝ := 5280
  let distance_miles : ℝ := distance_feet / 5280
  let speed_north : ℝ := 1 / 3  -- miles per minute
  let speed_south : ℝ := 3  -- miles per minute
  let rest_time : ℝ := 10  -- minutes
  let time_north : ℝ := distance_miles / speed_north
  let time_south : ℝ := distance_miles / speed_south
  let total_time : ℝ := time_north + time_south + rest_time
  let total_distance : ℝ := 2 * distance_miles
  let avg_speed : ℝ := total_distance / (total_time / 60)
  avg_speed = 9 := by
  sorry


end round_trip_average_speed_l1499_149976


namespace least_subtraction_for_divisibility_l1499_149906

theorem least_subtraction_for_divisibility : 
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), m < n → ¬(15 ∣ (427398 - m))) ∧ 
  (15 ∣ (427398 - n)) := by
  sorry

end least_subtraction_for_divisibility_l1499_149906


namespace cost_price_proof_l1499_149998

/-- The cost price of a ball in rupees -/
def cost_price : ℝ := 90

/-- The number of balls sold -/
def balls_sold : ℕ := 13

/-- The selling price of all balls in rupees -/
def selling_price : ℝ := 720

/-- The number of balls whose cost price equals the loss -/
def loss_balls : ℕ := 5

theorem cost_price_proof :
  cost_price * balls_sold = selling_price + cost_price * loss_balls :=
sorry

end cost_price_proof_l1499_149998


namespace quadratic_two_distinct_roots_l1499_149989

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + 2*x + k = 0 ∧ y^2 + 2*y + k = 0) ↔ k < 1 := by
  sorry

end quadratic_two_distinct_roots_l1499_149989


namespace function_machine_output_l1499_149977

/-- Function machine operation -/
def function_machine (input : ℕ) : ℕ :=
  let doubled := input * 2
  if doubled ≤ 15 then
    doubled * 3
  else
    doubled * 3

/-- Theorem: The function machine outputs 90 for an input of 15 -/
theorem function_machine_output : function_machine 15 = 90 := by
  sorry

end function_machine_output_l1499_149977


namespace overall_gain_percentage_l1499_149975

/-- Calculate the overall gain percentage for three items --/
theorem overall_gain_percentage
  (cycle_cp cycle_sp scooter_cp scooter_sp skateboard_cp skateboard_sp : ℚ)
  (h_cycle_cp : cycle_cp = 900)
  (h_cycle_sp : cycle_sp = 1170)
  (h_scooter_cp : scooter_cp = 15000)
  (h_scooter_sp : scooter_sp = 18000)
  (h_skateboard_cp : skateboard_cp = 2000)
  (h_skateboard_sp : skateboard_sp = 2400) :
  let total_cp := cycle_cp + scooter_cp + skateboard_cp
  let total_sp := cycle_sp + scooter_sp + skateboard_sp
  let gain_percentage := (total_sp - total_cp) / total_cp * 100
  ∃ (ε : ℚ), abs (gain_percentage - 20.50) < ε ∧ ε > 0 ∧ ε < 0.01 :=
by sorry

end overall_gain_percentage_l1499_149975


namespace ball_probabilities_l1499_149987

structure BallBag where
  red_balls : ℕ
  white_balls : ℕ

def initial_bag : BallBag := ⟨3, 2⟩

def total_balls (bag : BallBag) : ℕ := bag.red_balls + bag.white_balls

def P_A1 (bag : BallBag) : ℚ := bag.red_balls / total_balls bag
def P_A2 (bag : BallBag) : ℚ := bag.white_balls / total_balls bag

def P_B (bag : BallBag) : ℚ :=
  (P_A1 bag * (bag.red_balls - 1) / (total_balls bag - 1)) +
  (P_A2 bag * (bag.white_balls - 1) / (total_balls bag - 1))

def P_C_given_A2 (bag : BallBag) : ℚ := bag.red_balls / (total_balls bag - 1)

theorem ball_probabilities (bag : BallBag) :
  P_A1 bag + P_A2 bag = 1 ∧
  P_B initial_bag = 2/5 ∧
  P_C_given_A2 initial_bag = 3/4 := by
  sorry

end ball_probabilities_l1499_149987


namespace x_value_in_sequence_l1499_149905

def fibonacci_like_sequence (a : ℤ → ℤ) : Prop :=
  ∀ n, a (n + 2) = a (n + 1) + a n

theorem x_value_in_sequence (a : ℤ → ℤ) :
  fibonacci_like_sequence a →
  a 3 = 10 →
  a 4 = 5 →
  a 5 = 15 →
  a 6 = 20 →
  a 7 = 35 →
  a 8 = 55 →
  a 9 = 90 →
  a 0 = -20 :=
by
  sorry

end x_value_in_sequence_l1499_149905


namespace jill_draws_spade_prob_l1499_149965

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Represents the probability of drawing a spade from a standard deck -/
def ProbSpade : ℚ := NumSpades / StandardDeck

/-- Represents the probability of not drawing a spade from a standard deck -/
def ProbNotSpade : ℚ := 1 - ProbSpade

/-- Represents the probability that Jack draws a spade -/
def ProbJackSpade : ℚ := ProbSpade

/-- Represents the probability that Jill draws a spade -/
def ProbJillSpade : ℚ := ProbNotSpade * ProbSpade

/-- Represents the probability that John draws a spade -/
def ProbJohnSpade : ℚ := ProbNotSpade * ProbNotSpade * ProbSpade

/-- Represents the probability that a spade is drawn in one cycle -/
def ProbSpadeInCycle : ℚ := ProbJackSpade + ProbJillSpade + ProbJohnSpade

theorem jill_draws_spade_prob : 
  ProbJillSpade / ProbSpadeInCycle = 12 / 37 := by sorry

end jill_draws_spade_prob_l1499_149965


namespace cricket_team_right_handed_players_l1499_149999

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (h1 : total_players = 120)
  (h2 : throwers = 55)
  (h3 : 2 * (total_players - throwers) = 5 * (total_players - throwers - (total_players - throwers - throwers)))
  (h4 : throwers ≤ total_players) :
  throwers + (total_players - throwers - (2 * (total_players - throwers) / 5)) = 94 :=
by sorry

end cricket_team_right_handed_players_l1499_149999


namespace largest_n_divisible_equality_l1499_149980

def divisibleCount (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

def divisibleBy5or7 (n : ℕ) : ℕ :=
  divisibleCount n 5 + divisibleCount n 7 - divisibleCount n 35

theorem largest_n_divisible_equality : ∀ n : ℕ, n > 65 →
  (divisibleCount n 3 ≠ divisibleBy5or7 n) ∧
  (divisibleCount 65 3 = divisibleBy5or7 65) := by
  sorry

#eval divisibleCount 65 3  -- Expected: 21
#eval divisibleBy5or7 65   -- Expected: 21

end largest_n_divisible_equality_l1499_149980


namespace no_nonnegative_solutions_quadratic_l1499_149956

theorem no_nonnegative_solutions_quadratic :
  ∀ x : ℝ, x ≥ 0 → x^2 + 6*x + 9 ≠ 0 := by
sorry

end no_nonnegative_solutions_quadratic_l1499_149956


namespace sum_of_solutions_quadratic_l1499_149930

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 5*x - 26 = 4*x + 21) → 
  (∃ x₁ x₂ : ℝ, (x₁ + x₂ = 9) ∧ (x₁^2 - 5*x₁ - 26 = 4*x₁ + 21) ∧ (x₂^2 - 5*x₂ - 26 = 4*x₂ + 21)) :=
by sorry

end sum_of_solutions_quadratic_l1499_149930


namespace sarah_bowled_160_l1499_149951

def sarahs_score (gregs_score : ℕ) : ℕ := gregs_score + 60

theorem sarah_bowled_160 (gregs_score : ℕ) :
  sarahs_score gregs_score = 160 ∧ 
  (sarahs_score gregs_score + gregs_score) / 2 = 130 :=
by sorry

end sarah_bowled_160_l1499_149951


namespace min_value_and_inequality_l1499_149936

theorem min_value_and_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a + Real.sqrt 2 * b + Real.sqrt 3 * c = 2 * Real.sqrt 3) :
  (∃ m : ℝ, 
    (∀ a' b' c' : ℝ, a' + Real.sqrt 2 * b' + Real.sqrt 3 * c' = 2 * Real.sqrt 3 → 
      a'^2 + b'^2 + c'^2 ≥ m) ∧ 
    (a^2 + b^2 + c^2 = m) ∧
    m = 2) ∧
  (∃ p q : ℝ, ∀ x : ℝ, (|x - 3| ≥ 2 ↔ x^2 + p*x + q ≥ 0) ∧ p = -6) :=
sorry

end min_value_and_inequality_l1499_149936


namespace cycle_selling_price_l1499_149948

/-- Calculate the selling price of a cycle given its cost price and gain percent. -/
theorem cycle_selling_price (cost_price : ℝ) (gain_percent : ℝ) (selling_price : ℝ) :
  cost_price = 450 →
  gain_percent = 15.56 →
  selling_price = cost_price * (1 + gain_percent / 100) →
  selling_price = 520.02 := by
  sorry


end cycle_selling_price_l1499_149948


namespace complement_B_intersect_A_range_of_a_l1499_149928

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x - 18 ≥ 0}
def B : Set ℝ := {x | (x+5)/(x-14) ≤ 0}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a+1}

-- Theorem for part (1)
theorem complement_B_intersect_A : 
  (Set.univ \ B) ∩ A = Set.Iic (-5) ∪ Set.Ici 14 := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) : 
  (B ∩ C a = C a) ↔ a ≥ -5/2 := by sorry

end complement_B_intersect_A_range_of_a_l1499_149928


namespace passing_percentage_problem_l1499_149966

/-- The passing percentage problem -/
theorem passing_percentage_problem (mike_score : ℕ) (shortfall : ℕ) (max_marks : ℕ) 
  (h1 : mike_score = 212)
  (h2 : shortfall = 25)
  (h3 : max_marks = 790) :
  let passing_marks : ℕ := mike_score + shortfall
  let passing_percentage : ℚ := (passing_marks : ℚ) / max_marks * 100
  ∃ ε > 0, abs (passing_percentage - 30) < ε := by
  sorry

end passing_percentage_problem_l1499_149966


namespace series_sum_is_zero_l1499_149920

/-- The sum of the series -1 + 0 + 1 - 2 + 0 + 2 - 3 + 0 + 3 - ... + (-4001) + 0 + 4001 -/
def seriesSum : ℤ := sorry

/-- The number of terms in the series -/
def numTerms : ℕ := 12003

/-- The series ends at 4001 -/
def lastTerm : ℕ := 4001

theorem series_sum_is_zero :
  seriesSum = 0 :=
by sorry

end series_sum_is_zero_l1499_149920


namespace min_point_sum_l1499_149973

-- Define the function f(x) = 3x - x³
def f (x : ℝ) : ℝ := 3 * x - x^3

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3 - 3 * x^2

-- Theorem statement
theorem min_point_sum :
  ∃ (a b : ℝ), (∀ x, f x ≥ f a) ∧ (f a = b) ∧ (a + b = -3) := by
  sorry

end min_point_sum_l1499_149973


namespace min_sum_of_radii_l1499_149914

/-
  Define a regular tetrahedron with edge length 1
-/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_unit : edge_length = 1)

/-
  Define a sphere inside the tetrahedron
-/
structure Sphere :=
  (center : ℝ × ℝ × ℝ)
  (radius : ℝ)

/-
  Define the property of a sphere being tangent to three faces of the tetrahedron
-/
def is_tangent_to_three_faces (s : Sphere) (t : RegularTetrahedron) (vertex : ℝ × ℝ × ℝ) : Prop :=
  sorry  -- This would involve complex geometric conditions

/-
  State the theorem
-/
theorem min_sum_of_radii (t : RegularTetrahedron) 
  (s1 s2 : Sphere) 
  (h1 : is_tangent_to_three_faces s1 t (0, 0, 0))  -- Assume A is at (0,0,0)
  (h2 : is_tangent_to_three_faces s2 t (1, 0, 0))  -- Assume B is at (1,0,0)
  : 
  s1.radius + s2.radius ≥ (Real.sqrt 6 - 1) / 5 := by
  sorry


end min_sum_of_radii_l1499_149914


namespace continuous_stripe_probability_l1499_149963

/-- Represents a cube with painted diagonal stripes on each face -/
structure StripedCube where
  /-- The number of faces on the cube -/
  num_faces : Nat
  /-- The number of possible stripe orientations per face -/
  orientations_per_face : Nat
  /-- The total number of possible stripe combinations -/
  total_combinations : Nat
  /-- The number of favorable outcomes (continuous stripes) -/
  favorable_outcomes : Nat

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe (cube : StripedCube) : Rat :=
  cube.favorable_outcomes / cube.total_combinations

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  ∃ (cube : StripedCube),
    cube.num_faces = 6 ∧
    cube.orientations_per_face = 2 ∧
    cube.total_combinations = 2^6 ∧
    cube.favorable_outcomes = 6 ∧
    probability_continuous_stripe cube = 3/32 := by
  sorry

end continuous_stripe_probability_l1499_149963


namespace calculation_proof_l1499_149904

theorem calculation_proof :
  (1) * (Real.sqrt 2 + 2)^2 = 6 + 4 * Real.sqrt 2 ∧
  (2) * (Real.sqrt 3 - Real.sqrt 8) - (1/2) * (Real.sqrt 18 + Real.sqrt 12) = -(7/2) * Real.sqrt 2 := by
  sorry

end calculation_proof_l1499_149904


namespace union_of_M_and_N_l1499_149908

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = Set.Ici 0 := by sorry

end union_of_M_and_N_l1499_149908


namespace plane_equation_satisfies_conditions_l1499_149960

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if two planes are parallel -/
def planesParallel (plane1 plane2 : Plane) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ plane1.a = k * plane2.a ∧ plane1.b = k * plane2.b ∧ plane1.c = k * plane2.c

theorem plane_equation_satisfies_conditions : 
  let plane := Plane.mk 24 20 (-16) (-20)
  let point1 := Point3D.mk 2 (-1) 3
  let point2 := Point3D.mk 1 3 (-4)
  let parallelPlane := Plane.mk 3 (-4) 1 (-5)
  pointOnPlane plane point1 ∧ 
  pointOnPlane plane point2 ∧ 
  planesParallel plane parallelPlane :=
by sorry

end plane_equation_satisfies_conditions_l1499_149960


namespace correct_number_of_hens_l1499_149912

/-- Given a total number of animals and feet, calculate the number of hens -/
def number_of_hens (total_animals : ℕ) (total_feet : ℕ) : ℕ :=
  2 * total_animals - total_feet / 2

theorem correct_number_of_hens :
  let total_animals := 46
  let total_feet := 140
  number_of_hens total_animals total_feet = 22 := by
  sorry

#eval number_of_hens 46 140

end correct_number_of_hens_l1499_149912


namespace area_between_circles_first_quadrant_l1499_149941

/-- The area of the region between two concentric circles with radii 15 and 9,
    extending only within the first quadrant, is equal to 36π. -/
theorem area_between_circles_first_quadrant :
  let r₁ : ℝ := 15
  let r₂ : ℝ := 9
  let full_area := π * (r₁^2 - r₂^2)
  let quadrant_area := full_area / 4
  quadrant_area = 36 * π :=
by sorry

end area_between_circles_first_quadrant_l1499_149941


namespace negation_equivalence_l1499_149910

def exactly_one_even (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0)

def at_least_two_even_or_all_odd (a b c : ℕ) : Prop :=
  (a % 2 = 0 ∧ b % 2 = 0) ∨
  (a % 2 = 0 ∧ c % 2 = 0) ∨
  (b % 2 = 0 ∧ c % 2 = 0) ∨
  (a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ at_least_two_even_or_all_odd a b c :=
by sorry

end negation_equivalence_l1499_149910


namespace angleBMeasureApprox_l1499_149985

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  A : ℝ  -- Measure of angle A in degrees
  B : ℝ  -- Measure of angle B in degrees
  C : ℝ  -- Measure of angle C in degrees
  isIsosceles : B = C
  angleRelation : C = 3 * A + 10
  angleSum : A + B + C = 180

/-- The measure of angle B in the isosceles triangle -/
def angleBMeasure (triangle : IsoscelesTriangle) : ℝ := triangle.B

/-- Theorem stating the measure of angle B -/
theorem angleBMeasureApprox (triangle : IsoscelesTriangle) : 
  ∃ ε > 0, |angleBMeasure triangle - 550/7| < ε :=
sorry

end angleBMeasureApprox_l1499_149985


namespace no_real_roots_implies_a_greater_than_one_l1499_149972

/-- A quadratic function f(x) = x^2 + 2x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + a

/-- The discriminant of the quadratic function f -/
def discriminant (a : ℝ) : ℝ := 4 - 4*a

theorem no_real_roots_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, f a x ≠ 0) → a > 1 := by
  sorry

#check no_real_roots_implies_a_greater_than_one

end no_real_roots_implies_a_greater_than_one_l1499_149972


namespace riding_to_total_ratio_l1499_149942

/-- Represents the number of horses and owners -/
def total_count : ℕ := 18

/-- Represents the number of legs walking on the ground -/
def legs_on_ground : ℕ := 90

/-- Represents the number of owners riding their horses -/
def riding_owners : ℕ := total_count - (legs_on_ground - 4 * total_count) / 2

/-- Theorem stating the ratio of riding owners to total owners -/
theorem riding_to_total_ratio :
  (riding_owners : ℚ) / total_count = 1 / 2 := by sorry

end riding_to_total_ratio_l1499_149942


namespace tax_base_amount_theorem_l1499_149931

/-- Calculates the base amount given the tax rate and tax amount -/
def calculate_base_amount (tax_rate : ℚ) (tax_amount : ℚ) : ℚ :=
  tax_amount / (tax_rate / 100)

/-- Theorem: Given a tax rate of 65% and a tax amount of $65, the base amount is $100 -/
theorem tax_base_amount_theorem :
  let tax_rate : ℚ := 65
  let tax_amount : ℚ := 65
  calculate_base_amount tax_rate tax_amount = 100 := by
  sorry

#eval calculate_base_amount 65 65

end tax_base_amount_theorem_l1499_149931


namespace solution_absolute_value_equation_l1499_149986

theorem solution_absolute_value_equation (x : ℝ) : 5 * x + 2 * |x| = 3 * x → x ≤ 0 := by
  sorry

end solution_absolute_value_equation_l1499_149986


namespace periodic_function_l1499_149995

/-- A function f is periodic if there exists a non-zero real number p such that
    f(x + p) = f(x) for all x in the domain of f. -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

/-- The given conditions on function f -/
structure FunctionConditions (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  (sum_neq : a₁ + b₁ ≠ a₂ + b₂)
  (cond : ∀ x : ℝ, (f (a₁ + x) = f (b₁ - x) ∧ f (a₂ + x) = f (b₂ - x)) ∨
                   (f (a₁ + x) = -f (b₁ - x) ∧ f (a₂ + x) = -f (b₂ - x)))

/-- The main theorem stating that f is periodic with the given period -/
theorem periodic_function (f : ℝ → ℝ) (a₁ b₁ a₂ b₂ : ℝ)
    (h : FunctionConditions f a₁ b₁ a₂ b₂) :
    IsPeriodic f ∧ ∃ p : ℝ, p = |((a₂ + b₂) - (a₁ + b₁))| ∧
    ∀ x : ℝ, f (x + p) = f x :=
  sorry


end periodic_function_l1499_149995


namespace same_distance_different_time_l1499_149900

/-- Calculates the required average speed for a rider to cover the same distance as another rider in a different time. -/
theorem same_distance_different_time 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4) 
  (h3 : fran_time = 5) : 
  (joann_speed * joann_time) / fran_time = 12 := by
  sorry

#check same_distance_different_time

end same_distance_different_time_l1499_149900


namespace power_equality_no_quadratic_term_l1499_149964

-- Define the variables
variable (x y a b : ℝ)

-- Theorem 1
theorem power_equality (h1 : 4^x = a) (h2 : 8^y = b) : 2^(2*x - 3*y) = a / b := by sorry

-- Theorem 2
theorem no_quadratic_term (h : ∀ x, (x - 1) * (x^2 + a*x + 1) = x^3 + c*x + d) : a = 1 := by sorry

end power_equality_no_quadratic_term_l1499_149964


namespace ratio_abc_l1499_149974

theorem ratio_abc (a b c : ℝ) (ha : a ≠ 0) 
  (h : 14 * (a^2 + b^2 + c^2) = (a + 2*b + 3*c)^2) : 
  ∃ (k : ℝ), k ≠ 0 ∧ a = k ∧ b = 2*k ∧ c = 3*k := by
  sorry

end ratio_abc_l1499_149974


namespace sausage_problem_l1499_149932

theorem sausage_problem (initial_sausages : ℕ) (remaining_sausages : ℕ) 
  (h1 : initial_sausages = 600)
  (h2 : remaining_sausages = 45) :
  ∃ (x : ℚ), 
    0 < x ∧ x < 1 ∧
    remaining_sausages = (1/4 : ℚ) * (1/2 : ℚ) * (1 - x) * initial_sausages ∧
    x = (2/5 : ℚ) := by
  sorry

end sausage_problem_l1499_149932


namespace unique_solution_ceiling_equation_l1499_149981

theorem unique_solution_ceiling_equation :
  ∃! b : ℝ, b + ⌈b⌉ = 19.6 := by sorry

end unique_solution_ceiling_equation_l1499_149981


namespace num_small_orders_l1499_149997

def large_order_weight : ℕ := 200
def small_order_weight : ℕ := 50
def total_weight_used : ℕ := 800
def num_large_orders : ℕ := 3

theorem num_small_orders : 
  (total_weight_used - num_large_orders * large_order_weight) / small_order_weight = 4 := by
  sorry

end num_small_orders_l1499_149997


namespace octagonal_cube_removed_volume_l1499_149929

/-- The volume of tetrahedra removed from a cube of side length 2 to make octagonal faces -/
theorem octagonal_cube_removed_volume :
  let cube_side : ℝ := 2
  let octagon_side : ℝ := 2 * (Real.sqrt 2 - 1)
  let tetrahedron_height : ℝ := 2 / Real.sqrt 2
  let tetrahedron_base_area : ℝ := 2 * (3 - 2 * Real.sqrt 2)
  let single_tetrahedron_volume : ℝ := (1 / 3) * tetrahedron_base_area * tetrahedron_height
  let total_removed_volume : ℝ := 8 * single_tetrahedron_volume
  total_removed_volume = (80 - 56 * Real.sqrt 2) / 3 :=
by sorry

end octagonal_cube_removed_volume_l1499_149929


namespace smallest_number_in_special_set_l1499_149944

theorem smallest_number_in_special_set (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 26 →
  b = 27 →
  c = b + 5 →
  a < b ∧ b < c →
  a = 19 := by
sorry

end smallest_number_in_special_set_l1499_149944


namespace yunas_average_score_l1499_149938

/-- Given Yuna's average score for May and June and her July score, 
    calculate her average score over the three months. -/
theorem yunas_average_score 
  (may_june_avg : ℝ) 
  (july_score : ℝ) 
  (h1 : may_june_avg = 84) 
  (h2 : july_score = 96) : 
  (2 * may_june_avg + july_score) / 3 = 88 := by
  sorry

#eval (2 * 84 + 96) / 3  -- This should evaluate to 88

end yunas_average_score_l1499_149938


namespace sum_of_complex_sequence_l1499_149937

theorem sum_of_complex_sequence : 
  let n : ℕ := 150
  let a₀ : ℤ := -74
  let b₀ : ℤ := 30
  let d : ℤ := 1
  let sum : ℂ := (↑n / 2 : ℚ) * ↑(2 * a₀ + (n - 1) * d) + 
                 (↑n / 2 : ℚ) * ↑(2 * b₀ + (n - 1) * d) * Complex.I
  sum = 75 + 15675 * Complex.I :=
by sorry

end sum_of_complex_sequence_l1499_149937


namespace fraction_equality_l1499_149993

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 40)
  (h2 : p / n = 5)
  (h3 : p / q = 1 / 8) :
  m / q = 1 := by
sorry

end fraction_equality_l1499_149993


namespace coefficient_x3_in_expansion_l1499_149983

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the sum of coefficients
def sumCoefficients (n : ℕ) : ℕ := (3 : ℕ) ^ n

-- Define the function to calculate the coefficient of x³
def coefficientX3 (n : ℕ) : ℕ := 8 * binomial n 3

-- Theorem statement
theorem coefficient_x3_in_expansion :
  ∃ n : ℕ, sumCoefficients n = 243 ∧ coefficientX3 n = 80 := by sorry

end coefficient_x3_in_expansion_l1499_149983


namespace problem_solution_l1499_149922

theorem problem_solution : 
  (∀ π : ℝ, (π - 2)^0 + (-1)^3 = 0) ∧ 
  (∀ m n : ℝ, (3*m + n) * (m - 2*n) = 3*m^2 - 5*m*n - 2*n^2) := by
  sorry

end problem_solution_l1499_149922


namespace fifteenth_term_of_ap_l1499_149926

/-- The nth term of an arithmetic progression -/
def arithmeticProgressionTerm (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1 : ℝ) * d

/-- Theorem: The 15th term of an arithmetic progression with first term 2 and common difference 3 is 44 -/
theorem fifteenth_term_of_ap : arithmeticProgressionTerm 2 3 15 = 44 := by
  sorry

end fifteenth_term_of_ap_l1499_149926


namespace one_bedroom_apartment_fraction_l1499_149949

theorem one_bedroom_apartment_fraction :
  let two_bedroom_fraction : ℝ := 0.33
  let total_fraction : ℝ := 0.5
  let one_bedroom_fraction : ℝ := total_fraction - two_bedroom_fraction
  one_bedroom_fraction = 0.17 := by
sorry

end one_bedroom_apartment_fraction_l1499_149949


namespace books_count_l1499_149903

/-- The total number of books owned by six friends -/
def total_books (sandy benny tim rachel alex jordan : ℕ) : ℕ :=
  sandy + benny + tim + rachel + alex + jordan

/-- Theorem stating the total number of books owned by the six friends -/
theorem books_count :
  ∃ (sandy benny tim rachel alex jordan : ℕ),
    sandy = 10 ∧
    benny = 24 ∧
    tim = 33 ∧
    rachel = 2 * benny ∧
    alex = tim / 2 - 3 ∧
    jordan = sandy + benny ∧
    total_books sandy benny tim rachel alex jordan = 162 :=
by
  sorry

end books_count_l1499_149903


namespace min_cans_is_281_l1499_149916

/-- The number of liters of Maaza --/
def maaza : ℕ := 50

/-- The number of liters of Pepsi --/
def pepsi : ℕ := 144

/-- The number of liters of Sprite --/
def sprite : ℕ := 368

/-- The function to calculate the minimum number of cans required --/
def min_cans (m p s : ℕ) : ℕ :=
  (m / Nat.gcd m (Nat.gcd p s)) + (p / Nat.gcd m (Nat.gcd p s)) + (s / Nat.gcd m (Nat.gcd p s))

/-- Theorem stating that the minimum number of cans required is 281 --/
theorem min_cans_is_281 : min_cans maaza pepsi sprite = 281 := by
  sorry

end min_cans_is_281_l1499_149916


namespace all_four_digit_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_main_theorem_l1499_149935

/-- A four-digit palindrome between 1000 and 10000 -/
def FourDigitPalindrome : Type := { n : ℕ // 1000 ≤ n ∧ n < 10000 ∧ ∃ a b : ℕ, n = 1000 * a + 100 * b + 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 }

/-- The theorem stating that all four-digit palindromes are divisible by 11 -/
theorem all_four_digit_palindromes_divisible_by_11 (n : FourDigitPalindrome) : 11 ∣ n.val := by
  sorry

/-- The probability that a randomly chosen four-digit palindrome is divisible by 11 -/
theorem probability_palindrome_divisible_by_11 : ℚ :=
  1

/-- The main theorem proving that the probability is 1 -/
theorem main_theorem : probability_palindrome_divisible_by_11 = 1 := by
  sorry

end all_four_digit_palindromes_divisible_by_11_probability_palindrome_divisible_by_11_main_theorem_l1499_149935


namespace point_not_in_region_l1499_149934

theorem point_not_in_region (m : ℝ) : 
  (1 : ℝ) - (m^2 - 2*m + 4)*(1 : ℝ) + 6 ≤ 0 ↔ m ≤ -1 ∨ m ≥ 3 :=
by sorry

end point_not_in_region_l1499_149934


namespace pants_cost_is_correct_l1499_149940

/-- The cost of one pair of pants in dollars -/
def pants_cost : ℝ := 80

/-- The cost of one T-shirt in dollars -/
def tshirt_cost : ℝ := 20

/-- The cost of one pair of shoes in dollars -/
def shoes_cost : ℝ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℝ := 0.1

/-- The total cost after discount for Eugene's purchase -/
def total_cost_after_discount : ℝ := 558

theorem pants_cost_is_correct : 
  (4 * tshirt_cost + 3 * pants_cost + 2 * shoes_cost) * (1 - discount_rate) = total_cost_after_discount :=
by sorry

end pants_cost_is_correct_l1499_149940


namespace cookies_needed_l1499_149953

/-- Given 6.0 people who each should receive 24.0 cookies, prove that the total number of cookies needed is 144.0. -/
theorem cookies_needed (people : Float) (cookies_per_person : Float) (h1 : people = 6.0) (h2 : cookies_per_person = 24.0) :
  people * cookies_per_person = 144.0 := by
  sorry

end cookies_needed_l1499_149953


namespace coating_time_for_given_problem_l1499_149967

/-- Represents the properties of the sphere coating problem -/
structure SphereCoating where
  copper_sphere_diameter : ℝ
  silver_layer_thickness : ℝ
  hydrogen_production : ℝ
  hydrogen_silver_ratio : ℝ
  silver_density : ℝ

/-- Calculates the time required for coating the sphere -/
noncomputable def coating_time (sc : SphereCoating) : ℝ :=
  sorry

/-- Theorem stating the coating time for the given problem -/
theorem coating_time_for_given_problem :
  let sc : SphereCoating := {
    copper_sphere_diameter := 3,
    silver_layer_thickness := 0.05,
    hydrogen_production := 11.11,
    hydrogen_silver_ratio := 1 / 108,
    silver_density := 10.5
  }
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |coating_time sc - 987| < ε :=
sorry

end coating_time_for_given_problem_l1499_149967


namespace expand_expression_l1499_149990

theorem expand_expression (x : ℝ) : (7 * x^2 - 3) * 5 * x^3 = 35 * x^5 - 15 * x^3 := by
  sorry

end expand_expression_l1499_149990


namespace expression_equals_two_fifths_l1499_149984

theorem expression_equals_two_fifths :
  (((3^1 : ℚ) - 6 + 4^2 - 3)⁻¹ * 4) = 2/5 := by
  sorry

end expression_equals_two_fifths_l1499_149984


namespace cost_price_is_118_l1499_149955

/-- Calculates the cost price per meter of cloth -/
def cost_price_per_meter (total_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_length) / total_length

/-- Theorem: The cost price of one meter of cloth is 118 Rs -/
theorem cost_price_is_118 (total_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
    (h1 : total_length = 80)
    (h2 : selling_price = 10000)
    (h3 : profit_per_meter = 7) :
  cost_price_per_meter total_length selling_price profit_per_meter = 118 := by
  sorry

end cost_price_is_118_l1499_149955


namespace polynomial_ratio_theorem_l1499_149943

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^2009 - 19*x^2008 + 1

-- Define the set of distinct zeros of f
def zeros (f : ℝ → ℝ) : Set ℝ := {x | f x = 0}

-- Define the polynomial P
def P (z : ℝ) : ℝ := sorry

-- Theorem statement
theorem polynomial_ratio_theorem 
  (h1 : ∀ r ∈ zeros f, P (r - 1/r) = 0) 
  (h2 : Fintype (zeros f)) 
  (h3 : Fintype.card (zeros f) = 2009) :
  P 2 / P (-2) = 36 / 49 := by sorry

end polynomial_ratio_theorem_l1499_149943


namespace geometric_mean_of_sqrt2_plus_minus_one_l1499_149961

theorem geometric_mean_of_sqrt2_plus_minus_one :
  let a := Real.sqrt 2 + 1
  let b := Real.sqrt 2 - 1
  ∃ x : ℝ, x^2 = a * b ∧ (x = 1 ∨ x = -1) :=
by sorry

end geometric_mean_of_sqrt2_plus_minus_one_l1499_149961


namespace inequality_solution_set_l1499_149954

theorem inequality_solution_set (a : ℝ) : 
  (∀ x, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end inequality_solution_set_l1499_149954


namespace decimal_point_problem_l1499_149971

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
sorry

end decimal_point_problem_l1499_149971


namespace billion_to_scientific_notation_l1499_149994

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem billion_to_scientific_notation :
  toScientificNotation (87.65 * 10^9) = ScientificNotation.mk 8.765 10 sorry := by
  sorry

end billion_to_scientific_notation_l1499_149994


namespace power_function_through_point_l1499_149915

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 / 2 → f 9 = 1 / 3 := by
  sorry

end power_function_through_point_l1499_149915


namespace additional_people_needed_l1499_149991

/-- The number of person-hours required to mow the lawn -/
def personHours : ℕ := 32

/-- The initial number of people who can mow the lawn in 8 hours -/
def initialPeople : ℕ := 4

/-- The desired time to mow the lawn -/
def desiredTime : ℕ := 3

/-- The total number of people needed to mow the lawn in the desired time -/
def totalPeopleNeeded : ℕ := (personHours + desiredTime - 1) / desiredTime

theorem additional_people_needed :
  totalPeopleNeeded - initialPeople = 7 :=
by sorry

end additional_people_needed_l1499_149991


namespace arrangements_count_is_correct_l1499_149952

/-- The number of arrangements of 4 boys and 3 girls in a row,
    where exactly two girls are standing next to each other. -/
def arrangements_count : ℕ := 2880

/-- The number of boys -/
def num_boys : ℕ := 4

/-- The number of girls -/
def num_girls : ℕ := 3

/-- Theorem stating that the number of arrangements of 4 boys and 3 girls in a row,
    where exactly two girls are standing next to each other, is equal to 2880. -/
theorem arrangements_count_is_correct :
  arrangements_count = num_girls * (num_girls - 1) / 2 * 
    (num_boys * (num_boys - 1) * (num_boys - 2) * (num_boys - 3)) *
    ((num_boys + 1) * num_boys) :=
by sorry

end arrangements_count_is_correct_l1499_149952


namespace equation_system_solution_l1499_149996

theorem equation_system_solution (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 5)
  (h2 : a / x + b / y + c / z = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := by
  sorry

end equation_system_solution_l1499_149996


namespace first_obtuse_triangle_l1499_149919

/-- Represents a triangle with three angles -/
structure Triangle where
  angle1 : Real
  angle2 : Real
  angle3 : Real

/-- Constructs the pedal triangle of a given triangle -/
def pedal_triangle (t : Triangle) : Triangle :=
  { angle1 := 180 - 2 * t.angle1,
    angle2 := 180 - 2 * t.angle2,
    angle3 := 180 - 2 * t.angle3 }

/-- Checks if a triangle is obtuse -/
def is_obtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

/-- Generates the nth pedal triangle in the sequence -/
def nth_pedal_triangle (n : Nat) : Triangle :=
  match n with
  | 0 => { angle1 := 59.5, angle2 := 60, angle3 := 60.5 }
  | n + 1 => pedal_triangle (nth_pedal_triangle n)

theorem first_obtuse_triangle :
  ∀ n : Nat, n < 6 → ¬(is_obtuse (nth_pedal_triangle n)) ∧
  is_obtuse (nth_pedal_triangle 6) :=
by sorry

end first_obtuse_triangle_l1499_149919


namespace hyperbola_eccentricity_l1499_149957

/-- The eccentricity of a hyperbola x^2 - y^2/4 = 1 is √5 -/
theorem hyperbola_eccentricity :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := Real.sqrt 5
  let e : ℝ := c / a
  x^2 - y^2 / 4 = 1 → e = Real.sqrt 5 := by
  sorry

end hyperbola_eccentricity_l1499_149957


namespace cube_coloring_count_l1499_149923

/-- The number of ways to color a cube with two colors -/
def cube_colorings : ℕ :=
  let faces := 6  -- number of faces on a cube
  let colors := 2  -- number of colors (red and blue)
  -- The actual calculation is not provided, as per the instructions
  20  -- The result we want to prove

/-- Theorem stating that the number of valid cube colorings is 20 -/
theorem cube_coloring_count : cube_colorings = 20 := by
  sorry

end cube_coloring_count_l1499_149923


namespace solve_equation_for_x_l1499_149913

theorem solve_equation_for_x : ∃ x : ℚ, (3 * x / 7) - 2 = 12 ∧ x = 98 / 3 := by
  sorry

end solve_equation_for_x_l1499_149913


namespace optimal_range_golden_section_l1499_149962

theorem optimal_range_golden_section (m : ℝ) : 
  (1000 ≤ m) →  -- The optimal range starts at 1000
  (1000 + (m - 1000) * 0.618 = 1618) →  -- The good point is determined by the golden ratio
  (m = 2000) :=  -- We want to prove that m = 2000
by
  sorry

end optimal_range_golden_section_l1499_149962


namespace tesseract_sum_l1499_149946

/-- A tesseract is a 4-dimensional hypercube -/
structure Tesseract where

/-- The number of edges in a tesseract -/
def Tesseract.edges (t : Tesseract) : ℕ := 32

/-- The number of vertices in a tesseract -/
def Tesseract.vertices (t : Tesseract) : ℕ := 16

/-- The number of faces in a tesseract -/
def Tesseract.faces (t : Tesseract) : ℕ := 24

/-- The sum of edges, vertices, and faces in a tesseract is 72 -/
theorem tesseract_sum (t : Tesseract) : 
  t.edges + t.vertices + t.faces = 72 := by sorry

end tesseract_sum_l1499_149946
