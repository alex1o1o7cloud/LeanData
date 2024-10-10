import Mathlib

namespace sum_of_roots_equals_one_l909_90948

theorem sum_of_roots_equals_one (x : ℝ) :
  (x + 3) * (x - 4) = 24 → ∃ y z : ℝ, y + z = 1 ∧ (y + 3) * (y - 4) = 24 ∧ (z + 3) * (z - 4) = 24 :=
by sorry

end sum_of_roots_equals_one_l909_90948


namespace business_school_size_l909_90909

/-- The number of students in the law school -/
def law_students : ℕ := 800

/-- The number of sibling pairs -/
def sibling_pairs : ℕ := 30

/-- The probability of selecting a sibling pair -/
def sibling_pair_probability : ℚ := 75 / 1000000

/-- The number of students in the business school -/
def business_students : ℕ := 5000

theorem business_school_size :
  (sibling_pairs : ℚ) / (business_students * law_students) = sibling_pair_probability :=
by sorry

end business_school_size_l909_90909


namespace ceiling_floor_difference_l909_90990

theorem ceiling_floor_difference : 
  ⌈(18 : ℚ) / 11 * (-33 : ℚ) / 4⌉ - ⌊(18 : ℚ) / 11 * ⌊(-33 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end ceiling_floor_difference_l909_90990


namespace mark_and_carolyn_money_l909_90936

theorem mark_and_carolyn_money : 
  (5 : ℚ) / 8 + (2 : ℚ) / 5 = (41 : ℚ) / 40 := by sorry

end mark_and_carolyn_money_l909_90936


namespace correct_card_to_disprove_jane_l909_90958

-- Define the type for card sides
inductive CardSide
| Letter (c : Char)
| Number (n : Nat)

-- Define the structure for a card
structure Card where
  side1 : CardSide
  side2 : CardSide

-- Define the function to check if a number is odd
def isOdd (n : Nat) : Bool :=
  n % 2 = 1

-- Define the function to check if a character is a vowel
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

-- Define Jane's statement as a function
def janesStatement (card : Card) : Bool :=
  match card.side1, card.side2 with
  | CardSide.Number n, CardSide.Letter c => 
      ¬(isOdd n ∧ isVowel c)
  | CardSide.Letter c, CardSide.Number n => 
      ¬(isOdd n ∧ isVowel c)
  | _, _ => true

-- Define the theorem
theorem correct_card_to_disprove_jane : 
  ∀ (cards : List Card),
  cards = [
    Card.mk (CardSide.Letter 'A') (CardSide.Number 0),
    Card.mk (CardSide.Letter 'S') (CardSide.Number 0),
    Card.mk (CardSide.Number 5) (CardSide.Letter ' '),
    Card.mk (CardSide.Number 8) (CardSide.Letter ' '),
    Card.mk (CardSide.Number 7) (CardSide.Letter ' ')
  ] →
  ∃ (card : Card),
  card ∈ cards ∧ 
  card.side1 = CardSide.Letter 'A' ∧
  (∃ (n : Nat), card.side2 = CardSide.Number n ∧ isOdd n) ∧
  (∀ (c : Card), c ∈ cards ∧ c ≠ card → janesStatement c) :=
by sorry


end correct_card_to_disprove_jane_l909_90958


namespace tax_calculation_correct_l909_90906

/-- Calculates the personal income tax based on the given salary and tax brackets. -/
def calculate_tax (salary : ℕ) : ℕ :=
  let taxable_income := salary - 5000
  let first_bracket := min taxable_income 3000
  let second_bracket := min (taxable_income - 3000) 9000
  let third_bracket := max (taxable_income - 12000) 0
  (first_bracket * 3 + second_bracket * 10 + third_bracket * 20) / 100

/-- Theorem stating that the calculated tax for a salary of 20000 yuan is 1590 yuan. -/
theorem tax_calculation_correct :
  calculate_tax 20000 = 1590 := by sorry

end tax_calculation_correct_l909_90906


namespace acid_solution_mixture_l909_90969

/-- Proves that adding 40 ounces of pure water and 200/9 ounces of 10% acid solution
    to 40 ounces of 25% acid solution results in a 15% acid solution. -/
theorem acid_solution_mixture : 
  let initial_volume : ℝ := 40
  let initial_concentration : ℝ := 0.25
  let water_added : ℝ := 40
  let dilute_solution_added : ℝ := 200 / 9
  let dilute_concentration : ℝ := 0.1
  let final_concentration : ℝ := 0.15
  let final_volume : ℝ := initial_volume + water_added + dilute_solution_added
  let final_acid_amount : ℝ := initial_volume * initial_concentration + 
                                dilute_solution_added * dilute_concentration
  final_acid_amount / final_volume = final_concentration :=
by
  sorry


end acid_solution_mixture_l909_90969


namespace max_determinable_elements_l909_90967

open Finset

theorem max_determinable_elements : ∀ (a : Fin 11 → ℕ) (b : Fin 9 → ℕ),
  (∀ i : Fin 11, a i ∈ range 12 \ {0}) →
  (∀ i j : Fin 11, i ≠ j → a i ≠ a j) →
  (∀ i : Fin 9, b i = a i + a (i + 2)) →
  (∃ (S : Finset (Fin 11)), S.card = 5 ∧ 
    (∀ (a' : Fin 11 → ℕ),
      (∀ i : Fin 11, a' i ∈ range 12 \ {0}) →
      (∀ i j : Fin 11, i ≠ j → a' i ≠ a' j) →
      (∀ i : Fin 9, b i = a' i + a' (i + 2)) →
      (∀ i ∈ S, a i = a' i))) ∧
  ¬(∃ (S : Finset (Fin 11)), S.card > 5 ∧ 
    (∀ (a' : Fin 11 → ℕ),
      (∀ i : Fin 11, a' i ∈ range 12 \ {0}) →
      (∀ i j : Fin 11, i ≠ j → a' i ≠ a' j) →
      (∀ i : Fin 9, b i = a' i + a' (i + 2)) →
      (∀ i ∈ S, a i = a' i))) := by
  sorry

end max_determinable_elements_l909_90967


namespace det_dilation_matrix_5_l909_90923

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- Theorem: The determinant of a 3x3 dilation matrix with scale factor 5 is 125 -/
theorem det_dilation_matrix_5 :
  let D := dilationMatrix 5
  Matrix.det D = 125 := by sorry

end det_dilation_matrix_5_l909_90923


namespace return_speed_calculation_l909_90952

/-- Given a round trip where the return speed is twice the outbound speed,
    prove that the return speed is 15 km/h when the total distance is 60 km
    and the total travel time is 6 hours. -/
theorem return_speed_calculation (distance : ℝ) (total_time : ℝ) (outbound_speed : ℝ) :
  distance = 60 →
  total_time = 6 →
  outbound_speed > 0 →
  distance / (2 * outbound_speed) + distance / (2 * (2 * outbound_speed)) = total_time →
  2 * outbound_speed = 15 := by
  sorry

end return_speed_calculation_l909_90952


namespace kenzo_office_chairs_l909_90957

theorem kenzo_office_chairs :
  ∀ (initial_chairs : ℕ),
    (∃ (chairs_legs tables_legs remaining_chairs_legs : ℕ),
      chairs_legs = 5 * initial_chairs ∧
      tables_legs = 20 * 3 ∧
      remaining_chairs_legs = (6 * chairs_legs) / 10 ∧
      remaining_chairs_legs + tables_legs = 300) →
    initial_chairs = 80 := by
  sorry

end kenzo_office_chairs_l909_90957


namespace age_difference_proof_l909_90945

theorem age_difference_proof (younger_age elder_age : ℕ) 
  (h1 : younger_age = 30)
  (h2 : elder_age = 50)
  (h3 : elder_age - 5 = 5 * (younger_age - 5)) :
  elder_age - younger_age = 20 := by
  sorry

end age_difference_proof_l909_90945


namespace four_solutions_implies_a_greater_than_two_l909_90935

-- Define the equation
def equation (a x : ℝ) : Prop := |x^3 - a*x^2| = x

-- Theorem statement
theorem four_solutions_implies_a_greater_than_two (a : ℝ) :
  (∃ w x y z : ℝ, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    equation a w ∧ equation a x ∧ equation a y ∧ equation a z) →
  a > 2 := by
  sorry

end four_solutions_implies_a_greater_than_two_l909_90935


namespace prism_to_spheres_waste_l909_90960

/-- The volume of waste when polishing a regular triangular prism into spheres -/
theorem prism_to_spheres_waste (base_side : ℝ) (height : ℝ) (sphere_radius : ℝ) :
  base_side = 6 →
  height = 8 * Real.sqrt 3 →
  sphere_radius = Real.sqrt 3 →
  ((Real.sqrt 3 / 4) * base_side^2 * height) - (4 * (4 / 3) * Real.pi * sphere_radius^3) =
    216 - 16 * Real.sqrt 3 * Real.pi :=
by sorry

end prism_to_spheres_waste_l909_90960


namespace round_robin_equation_l909_90913

/-- Represents a round-robin tournament -/
structure RoundRobinTournament where
  teams : ℕ
  total_games : ℕ
  games_formula : total_games = teams * (teams - 1) / 2

/-- Theorem: In a round-robin tournament with 45 total games, the equation x(x-1) = 2 * 45 holds true -/
theorem round_robin_equation (t : RoundRobinTournament) (h : t.total_games = 45) :
  t.teams * (t.teams - 1) = 2 * 45 := by
  sorry


end round_robin_equation_l909_90913


namespace scout_weekend_earnings_l909_90921

def base_pay : ℝ := 10.00
def tip_per_customer : ℝ := 5.00
def saturday_hours : ℝ := 4
def saturday_customers : ℕ := 5
def sunday_hours : ℝ := 5
def sunday_customers : ℕ := 8

theorem scout_weekend_earnings :
  let saturday_earnings := base_pay * saturday_hours + tip_per_customer * saturday_customers
  let sunday_earnings := base_pay * sunday_hours + tip_per_customer * sunday_customers
  saturday_earnings + sunday_earnings = 155.00 := by
  sorry

end scout_weekend_earnings_l909_90921


namespace painted_cube_problem_l909_90947

/-- Represents a painted cube cut into smaller cubes -/
structure PaintedCube where
  /-- The number of small cubes along each edge of the large cube -/
  edge_count : ℕ
  /-- The number of small cubes with both brown and orange colors -/
  dual_color_count : ℕ

/-- Theorem stating the properties of the painted cube problem -/
theorem painted_cube_problem (cube : PaintedCube) 
  (h1 : cube.dual_color_count = 16) : 
  cube.edge_count = 4 ∧ cube.edge_count ^ 3 = 64 := by
  sorry

#check painted_cube_problem

end painted_cube_problem_l909_90947


namespace min_value_expression_l909_90999

theorem min_value_expression (x y : ℝ) : x^2 + x*y + y^2 - 3*y ≥ -3 := by
  sorry

end min_value_expression_l909_90999


namespace lunch_break_duration_l909_90988

/-- Represents the painting scenario with Paula and her helpers --/
structure PaintingScenario where
  paula_rate : ℝ
  helpers_rate : ℝ
  lunch_break : ℝ

/-- Conditions of the painting scenario --/
def painting_conditions (s : PaintingScenario) : Prop :=
  -- Monday's work
  (9 - s.lunch_break) * (s.paula_rate + s.helpers_rate) = 0.4 ∧
  -- Tuesday's work
  (8 - s.lunch_break) * s.helpers_rate = 0.33 ∧
  -- Wednesday's work
  (12 - s.lunch_break) * s.paula_rate = 0.27

/-- The main theorem: lunch break duration is 420 minutes --/
theorem lunch_break_duration (s : PaintingScenario) :
  painting_conditions s → s.lunch_break * 60 = 420 := by
  sorry

end lunch_break_duration_l909_90988


namespace constant_term_proof_l909_90919

theorem constant_term_proof (a k n : ℤ) :
  (∀ x, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) →
  a - n + k = 7 →
  (3 * 0 + 2) * (2 * 0 - 3) = -6 :=
by
  sorry

end constant_term_proof_l909_90919


namespace seventh_root_unity_product_l909_90932

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 := by
  sorry

end seventh_root_unity_product_l909_90932


namespace complex_equation_solution_l909_90946

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l909_90946


namespace cos_sin_power_eight_identity_l909_90986

theorem cos_sin_power_eight_identity (α : ℝ) : 
  Real.cos α ^ 8 - Real.sin α ^ 8 = Real.cos (2 * α) * ((3 + Real.cos (4 * α)) / 4) := by
  sorry

end cos_sin_power_eight_identity_l909_90986


namespace hexagon_problem_l909_90954

/-- Regular hexagon with side length 3 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (regular : side_length = 3)

/-- L is the intersection point of diagonals CE and DF -/
def L (h : RegularHexagon) : ℝ × ℝ := sorry

/-- K is defined such that LK = 3AB - AC -/
def K (h : RegularHexagon) : ℝ × ℝ := sorry

/-- Determine if a point is outside the hexagon -/
def is_outside (h : RegularHexagon) (p : ℝ × ℝ) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hexagon_problem (h : RegularHexagon) :
  is_outside h (K h) ∧ distance (K h) h.C = 3 * Real.sqrt 7 := by sorry

end hexagon_problem_l909_90954


namespace total_flight_distance_l909_90980

/-- Given the distances between Spain, Russia, and a stopover country, 
    calculate the total distance to fly from the stopover to Russia and back to Spain. -/
theorem total_flight_distance 
  (spain_russia : ℕ) 
  (spain_stopover : ℕ) 
  (h1 : spain_russia = 7019)
  (h2 : spain_stopover = 1615) : 
  spain_stopover + (spain_russia - spain_stopover) + spain_russia = 12423 :=
by sorry

#check total_flight_distance

end total_flight_distance_l909_90980


namespace two_lines_properties_l909_90900

/-- Two lines l₁ and l₂ in the xy-plane -/
structure TwoLines (m n : ℝ) :=
  (l₁ : ℝ → ℝ → Prop)
  (l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ m * x + 8 * y + n = 0)
  (h₂ : ∀ x y, l₂ x y ↔ 2 * x + m * y - 1 = 0)

/-- The lines intersect at point P(m, -1) -/
def intersect_at_P (l : TwoLines m n) : Prop :=
  l.l₁ m (-1) ∧ l.l₂ m (-1)

/-- The lines are parallel -/
def parallel (l : TwoLines m n) : Prop :=
  m / 2 = 8 / m ∧ m / 2 ≠ n / (-1)

/-- The lines are perpendicular -/
def perpendicular (l : TwoLines m n) : Prop :=
  m = 0 ∨ (m ≠ 0 ∧ (-m / 8) * (1 / m) = -1)

/-- Main theorem about the properties of the two lines -/
theorem two_lines_properties (m n : ℝ) (l : TwoLines m n) :
  (intersect_at_P l → m = 1 ∧ n = 7) ∧
  (parallel l → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)) ∧
  (perpendicular l → m = 0) :=
sorry

end two_lines_properties_l909_90900


namespace purple_chips_count_l909_90973

/-- Represents the number of chips of each color selected -/
structure ChipSelection where
  blue : ℕ
  green : ℕ
  purple : ℕ
  red : ℕ

/-- The theorem stating the number of purple chips selected -/
theorem purple_chips_count 
  (x : ℕ) 
  (h1 : 5 < x) 
  (h2 : x < 11) 
  (selection : ChipSelection) 
  (h3 : 1^selection.blue * 5^selection.green * x^selection.purple * 11^selection.red = 28160) :
  selection.purple = 2 :=
sorry

end purple_chips_count_l909_90973


namespace expand_binomial_product_l909_90996

theorem expand_binomial_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_binomial_product_l909_90996


namespace coloring_book_shelves_l909_90971

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 435 → books_sold = 218 → books_per_shelf = 17 →
  (initial_stock - books_sold + books_per_shelf - 1) / books_per_shelf = 13 := by
sorry


end coloring_book_shelves_l909_90971


namespace fuel_cost_calculation_l909_90992

theorem fuel_cost_calculation (original_cost : ℝ) (capacity_increase : ℝ) (price_increase : ℝ) : 
  original_cost = 200 → 
  capacity_increase = 2 → 
  price_increase = 1.2 → 
  original_cost * capacity_increase * price_increase = 480 := by
sorry

end fuel_cost_calculation_l909_90992


namespace golden_state_team_points_l909_90981

/-- The number of points earned by Draymond in the Golden State Team -/
def draymondPoints : ℕ := 12

/-- The total points earned by the Golden State Team -/
def totalTeamPoints : ℕ := 69

/-- The number of points earned by Kelly -/
def kellyPoints : ℕ := 9

theorem golden_state_team_points :
  ∃ (D : ℕ), 
    D = draymondPoints ∧
    D + 2*D + kellyPoints + 2*kellyPoints + D/2 = totalTeamPoints :=
by sorry

end golden_state_team_points_l909_90981


namespace g_range_values_l909_90975

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - 3*x) / (2 + 3*x))

theorem g_range_values :
  {y | ∃ x, g x = y} = {-π/2, π/4} := by sorry

end g_range_values_l909_90975


namespace farm_animals_difference_l909_90987

theorem farm_animals_difference (initial_horses : ℕ) (initial_cows : ℕ) : 
  initial_horses = 4 * initial_cows →  -- Initial ratio condition
  (initial_horses - 15) / (initial_cows + 15) = 13 / 7 →  -- New ratio condition
  initial_horses - 15 - (initial_cows + 15) = 30 :=  -- Difference after transaction
by
  sorry

end farm_animals_difference_l909_90987


namespace arithmetic_sequence_sum_l909_90938

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 5 + a 6 + a 7 + a 8 = 20) →
  (a 1 + a 12 = 10) :=
by
  sorry

end arithmetic_sequence_sum_l909_90938


namespace determinant_special_matrix_l909_90976

/-- The determinant of the matrix [[1, x, z], [1, x+z, z], [1, x, x+z]] is equal to xz - z^2 -/
theorem determinant_special_matrix (x z : ℝ) :
  Matrix.det !![1, x, z; 1, x + z, z; 1, x, x + z] = x * z - z^2 := by
  sorry

end determinant_special_matrix_l909_90976


namespace marnie_bracelets_l909_90901

/-- The number of bracelets that can be made from a given number of beads -/
def bracelets_from_beads (total_beads : ℕ) (beads_per_bracelet : ℕ) : ℕ :=
  total_beads / beads_per_bracelet

/-- The total number of beads from multiple bags -/
def total_beads_from_bags (bags_of_50 : ℕ) (bags_of_100 : ℕ) : ℕ :=
  bags_of_50 * 50 + bags_of_100 * 100

theorem marnie_bracelets : 
  let bags_of_50 : ℕ := 5
  let bags_of_100 : ℕ := 2
  let beads_per_bracelet : ℕ := 50
  let total_beads := total_beads_from_bags bags_of_50 bags_of_100
  bracelets_from_beads total_beads beads_per_bracelet = 9 := by
  sorry

end marnie_bracelets_l909_90901


namespace hurdle_race_calculations_l909_90944

/-- Calculates the distance between adjacent hurdles and the theoretical best time for a 110m hurdle race --/
theorem hurdle_race_calculations 
  (total_distance : ℝ) 
  (num_hurdles : ℕ) 
  (start_to_first : ℝ) 
  (last_to_finish : ℝ) 
  (time_to_first : ℝ) 
  (time_after_last : ℝ) 
  (fastest_cycle : ℝ) 
  (h1 : total_distance = 110) 
  (h2 : num_hurdles = 10) 
  (h3 : start_to_first = 13.72) 
  (h4 : last_to_finish = 14.02) 
  (h5 : time_to_first = 2.5) 
  (h6 : time_after_last = 1.4) 
  (h7 : fastest_cycle = 0.96) :
  let inter_hurdle_distance := (total_distance - start_to_first - last_to_finish) / num_hurdles
  let theoretical_best_time := time_to_first + (num_hurdles : ℝ) * fastest_cycle + time_after_last
  inter_hurdle_distance = 8.28 ∧ theoretical_best_time = 12.1 := by
  sorry


end hurdle_race_calculations_l909_90944


namespace farm_feet_count_l909_90972

/-- A farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of heads in the farm -/
def total_heads (f : Farm) : ℕ := f.hens + f.cows

/-- The total number of feet in the farm -/
def total_feet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- Theorem: Given a farm with 48 heads and 28 hens, the total number of feet is 136 -/
theorem farm_feet_count :
  ∀ f : Farm, total_heads f = 48 → f.hens = 28 → total_feet f = 136 :=
by
  sorry

end farm_feet_count_l909_90972


namespace solve_percentage_equation_l909_90977

theorem solve_percentage_equation (x : ℝ) : 
  (70 / 100) * 600 = (40 / 100) * x → x = 1050 := by
  sorry

end solve_percentage_equation_l909_90977


namespace point_positions_l909_90974

/-- Circle C is defined by the equation x^2 + y^2 - 2x + 4y - 4 = 0 --/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- Point M has coordinates (2, -4) --/
def point_M : ℝ × ℝ := (2, -4)

/-- Point N has coordinates (-2, 1) --/
def point_N : ℝ × ℝ := (-2, 1)

/-- A point (x, y) is inside the circle if x^2 + y^2 - 2x + 4y - 4 < 0 --/
def inside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 - 2*x + 4*y - 4 < 0

/-- A point (x, y) is outside the circle if x^2 + y^2 - 2x + 4y - 4 > 0 --/
def outside_circle (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 - 2*x + 4*y - 4 > 0

theorem point_positions :
  inside_circle point_M ∧ outside_circle point_N :=
sorry

end point_positions_l909_90974


namespace solution_set_equivalence_a_range_when_f_nonnegative_l909_90902

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Part 1
theorem solution_set_equivalence (a b : ℝ) :
  (∀ x, f a b x < 0 ↔ 2 < x ∧ x < 3) →
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Part 2
theorem a_range_when_f_nonnegative :
  ∀ a, (∀ x, f a (3-a) x ≥ 0) → a ∈ Set.Icc (-6) 2 :=
sorry

end solution_set_equivalence_a_range_when_f_nonnegative_l909_90902


namespace power_calculation_l909_90939

theorem power_calculation : (8^3 / 8^2) * 2^10 = 8192 := by
  sorry

end power_calculation_l909_90939


namespace roots_sum_and_product_l909_90918

theorem roots_sum_and_product (p q : ℝ) : 
  p^2 - 5*p + 6 = 0 → q^2 - 5*q + 6 = 0 → p^3 + p^4*q^2 + p^2*q^4 + q^3 = 503 := by
  sorry

end roots_sum_and_product_l909_90918


namespace prob_green_or_yellow_l909_90994

/-- A cube with colored faces -/
structure ColoredCube where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of an event -/
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  favorable_outcomes / total_outcomes

/-- Theorem: Probability of rolling a green or yellow face -/
theorem prob_green_or_yellow (cube : ColoredCube) 
  (h1 : cube.green_faces = 3)
  (h2 : cube.yellow_faces = 2)
  (h3 : cube.blue_faces = 1) :
  probability (cube.green_faces + cube.yellow_faces) 
    (cube.green_faces + cube.yellow_faces + cube.blue_faces) = 5 / 6 := by
  sorry

end prob_green_or_yellow_l909_90994


namespace partner_c_investment_l909_90908

/-- Represents the investment and profit structure of a business partnership --/
structure BusinessPartnership where
  capital_a : ℝ
  capital_b : ℝ
  capital_c : ℝ
  profit_b : ℝ
  profit_diff_ac : ℝ

/-- Theorem stating that given the conditions of the business partnership,
    the investment of partner c is 40000 --/
theorem partner_c_investment (bp : BusinessPartnership)
  (h1 : bp.capital_a = 8000)
  (h2 : bp.capital_b = 10000)
  (h3 : bp.profit_b = 3500)
  (h4 : bp.profit_diff_ac = 1399.9999999999998)
  : bp.capital_c = 40000 := by
  sorry


end partner_c_investment_l909_90908


namespace sector_perimeter_l909_90965

theorem sector_perimeter (r c θ : ℝ) (hr : r = 10) (hc : c = 10) (hθ : θ = 120 * π / 180) :
  r * θ + c = 20 * π / 3 + 10 := by
  sorry

end sector_perimeter_l909_90965


namespace total_cookies_count_l909_90912

def cookies_eaten : ℕ := 4
def cookies_to_brother : ℕ := 6
def friends_count : ℕ := 3
def cookies_per_friend : ℕ := 2
def team_members : ℕ := 10
def first_team_member_cookies : ℕ := 2
def team_cookie_difference : ℕ := 2

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem total_cookies_count :
  cookies_eaten +
  cookies_to_brother +
  (friends_count * cookies_per_friend) +
  arithmetic_sum first_team_member_cookies team_cookie_difference team_members =
  126 := by sorry

end total_cookies_count_l909_90912


namespace jump_rope_time_difference_l909_90920

-- Define the jump rope times for each person
def cindy_time : ℕ := 12
def betsy_time : ℕ := cindy_time / 2
def tina_time : ℕ := betsy_time * 3

-- Theorem to prove
theorem jump_rope_time_difference : tina_time - cindy_time = 6 := by
  sorry

end jump_rope_time_difference_l909_90920


namespace ratio_problem_l909_90941

theorem ratio_problem (a b x m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.25 * a) (h5 : m = b - 0.6 * b) : m / x = 2 / 5 := by
  sorry

end ratio_problem_l909_90941


namespace fractional_equation_solution_l909_90953

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ 2 ∧ (2 / x - 1 / (x - 2) = 0) ∧ x = 4 :=
by sorry

end fractional_equation_solution_l909_90953


namespace unique_friendly_pair_l909_90916

/-- Euler's totient function -/
def φ (n : ℕ) : ℕ := sorry

/-- The smallest positive integer greater than n that is not coprime to n -/
def f (n : ℕ) : ℕ := sorry

/-- A friendly pair is a pair of positive integers (n, m) where f(n) = m and φ(m) = n -/
def is_friendly_pair (n m : ℕ) : Prop :=
  f n = m ∧ φ m = n

theorem unique_friendly_pair : 
  ∀ n m : ℕ, is_friendly_pair n m → n = 2 ∧ m = 4 :=
sorry

end unique_friendly_pair_l909_90916


namespace triple_cheese_ratio_undetermined_l909_90910

/-- Represents the types of pizzas available --/
inductive PizzaType
| TripleCheese
| MeatLovers

/-- Represents the pricing structure for pizzas --/
structure PizzaPricing where
  standardPrice : ℕ
  meatLoversOffer : ℕ → ℕ  -- Function that takes number of pizzas and returns number to pay for
  tripleCheesePricing : ℕ → ℕ  -- Function for triple cheese pizzas (unknown specifics)

/-- Represents the order details --/
structure Order where
  tripleCheeseCount : ℕ
  meatLoversCount : ℕ

/-- Calculates the total cost of an order --/
def calculateTotalCost (pricing : PizzaPricing) (order : Order) : ℕ :=
  pricing.tripleCheesePricing order.tripleCheeseCount * pricing.standardPrice +
  pricing.meatLoversOffer order.meatLoversCount * pricing.standardPrice

/-- Theorem stating that the ratio for triple cheese pizzas cannot be determined --/
theorem triple_cheese_ratio_undetermined (pricing : PizzaPricing) (order : Order) :
  pricing.standardPrice = 5 ∧
  order.meatLoversCount = 9 ∧
  calculateTotalCost pricing order = 55 →
  ¬ ∃ (r : ℚ), r > 0 ∧ r < 1 ∧ ∀ (n : ℕ), pricing.tripleCheesePricing n = n - ⌊n * r⌋ :=
by sorry

end triple_cheese_ratio_undetermined_l909_90910


namespace binary_representation_253_l909_90983

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def count_ones (binary : List Bool) : ℕ :=
  sorry

def count_zeros (binary : List Bool) : ℕ :=
  sorry

theorem binary_representation_253 :
  let binary := decimal_to_binary 253
  let y := count_ones binary
  let x := count_zeros binary
  y - x = 6 := by sorry

end binary_representation_253_l909_90983


namespace parallel_and_perpendicular_lines_l909_90933

-- Define a line in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a point in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the concept of a point not being on a line
def PointNotOnLine (p : Point) (l : Line) : Prop :=
  p.y ≠ l.slope * p.x + l.intercept

-- Define the concept of parallel lines
def Parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope ∧ l1.intercept ≠ l2.intercept

-- Define the concept of perpendicular lines
def Perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

theorem parallel_and_perpendicular_lines
  (L : Line) (P : Point) (h : PointNotOnLine P L) :
  (∃! l : Line, Parallel l L ∧ l.slope * P.x + l.intercept = P.y) ∧
  (∃ f : ℝ → Line, ∀ t : ℝ, Perpendicular (f t) L ∧ (f t).slope * P.x + (f t).intercept = P.y) :=
sorry

end parallel_and_perpendicular_lines_l909_90933


namespace complex_arithmetic_expression_equals_zero_l909_90925

theorem complex_arithmetic_expression_equals_zero :
  -6 * (1/3 - 1/2) - 3^2 / (-12) - |-7/4| = 0 := by
  sorry

end complex_arithmetic_expression_equals_zero_l909_90925


namespace chicken_wings_distribution_l909_90989

theorem chicken_wings_distribution (friends : ℕ) (pre_cooked : ℕ) (additional : ℕ) :
  friends = 3 →
  pre_cooked = 8 →
  additional = 10 →
  (pre_cooked + additional) / friends = 6 :=
by sorry

end chicken_wings_distribution_l909_90989


namespace sum_of_reciprocals_l909_90993

theorem sum_of_reciprocals (x y : ℚ) 
  (h1 : x⁻¹ + y⁻¹ = 4)
  (h2 : x⁻¹ - y⁻¹ = 8) : 
  x + y = -1/3 := by
  sorry

end sum_of_reciprocals_l909_90993


namespace intersection_of_M_and_N_l909_90966

def M : Set ℝ := {x | x ≥ 2}
def N : Set ℝ := {x | x^2 - 25 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x | 2 ≤ x ∧ x < 5} := by sorry

end intersection_of_M_and_N_l909_90966


namespace area_of_RQST_l909_90942

/-- Square with side length 3 -/
def Square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}

/-- Points on the square -/
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (3, 0)
def C : ℝ × ℝ := (3, 3)
def D : ℝ × ℝ := (0, 3)

/-- Points E and F divide AB into three segments with ratios 1:2 -/
def E : ℝ × ℝ := (1, 0)
def F : ℝ × ℝ := (2, 0)

/-- Points G and H divide CD similarly -/
def G : ℝ × ℝ := (3, 1)
def H : ℝ × ℝ := (3, 2)

/-- S is the midpoint of AB -/
def S : ℝ × ℝ := (1.5, 0)

/-- Q is the midpoint of CD -/
def Q : ℝ × ℝ := (3, 1.5)

/-- R and T divide the square into two equal areas -/
def R : ℝ × ℝ := (0, 1.5)
def T : ℝ × ℝ := (3, 1.5)

/-- Area of a quadrilateral given its vertices -/
def quadrilateralArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  0.5 * abs (p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2
           - (p1.2 * p2.1 + p2.2 * p3.1 + p3.2 * p4.1 + p4.2 * p1.1))

theorem area_of_RQST :
  quadrilateralArea R Q S T = 1.125 := by
  sorry

end area_of_RQST_l909_90942


namespace network_connections_l909_90940

theorem network_connections (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (n * k) / 2 = 60 := by
  sorry

end network_connections_l909_90940


namespace binomial_probability_three_out_of_six_l909_90929

/-- The probability mass function for a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The problem statement -/
theorem binomial_probability_three_out_of_six :
  binomial_pmf 6 (1/2) 3 = 5/16 := by
  sorry

end binomial_probability_three_out_of_six_l909_90929


namespace projection_obtuse_implies_obtuse_projection_acute_inconclusive_l909_90907

/-- Represents an angle --/
structure Angle where
  measure : ℝ
  is_positive : 0 < measure

/-- Represents the rectangular projection of an angle onto a plane --/
def rectangular_projection (α : Angle) : Angle :=
  sorry

/-- An angle is obtuse if its measure is greater than π/2 --/
def is_obtuse (α : Angle) : Prop :=
  α.measure > Real.pi / 2

/-- An angle is acute if its measure is less than π/2 --/
def is_acute (α : Angle) : Prop :=
  α.measure < Real.pi / 2

theorem projection_obtuse_implies_obtuse (α : Angle) :
  is_obtuse (rectangular_projection α) → is_obtuse α :=
sorry

theorem projection_acute_inconclusive (α : Angle) :
  is_acute (rectangular_projection α) → 
  (is_acute α ∨ is_obtuse α) :=
sorry

end projection_obtuse_implies_obtuse_projection_acute_inconclusive_l909_90907


namespace fraction_simplification_l909_90995

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hxy : x^2 - 1/y ≠ 0) 
  (hyx : y^2 - 1/x ≠ 0) : 
  (x^2 - 1/y) / (y^2 - 1/x) = x * (x^2*y - 1) / (y * (y^2*x - 1)) := by
  sorry

end fraction_simplification_l909_90995


namespace soccer_ball_donation_l909_90915

theorem soccer_ball_donation :
  let num_schools : ℕ := 2
  let elementary_classes_per_school : ℕ := 4
  let middle_classes_per_school : ℕ := 5
  let balls_per_class : ℕ := 5
  let total_classes : ℕ := num_schools * (elementary_classes_per_school + middle_classes_per_school)
  let total_balls : ℕ := total_classes * balls_per_class
  total_balls = 90 := by
  sorry

end soccer_ball_donation_l909_90915


namespace equation_one_solution_l909_90984

-- Define the equation
def equation (x p : ℝ) : Prop :=
  2 * |x - p| + |x - 2| = 1

-- Define the property of having exactly one solution
def has_exactly_one_solution (p : ℝ) : Prop :=
  ∃! x, equation x p

-- Theorem statement
theorem equation_one_solution :
  ∀ p : ℝ, has_exactly_one_solution p ↔ (p = 1 ∨ p = 3) :=
sorry

end equation_one_solution_l909_90984


namespace jack_socks_problem_l909_90950

theorem jack_socks_problem :
  ∀ (x y z : ℕ),
    x + y + z = 15 →
    2 * x + 4 * y + 5 * z = 36 →
    x ≥ 1 →
    y ≥ 1 →
    z ≥ 1 →
    x = 4 :=
by sorry

end jack_socks_problem_l909_90950


namespace kikis_money_l909_90914

theorem kikis_money (num_scarves : ℕ) (scarf_price : ℚ) (hat_ratio : ℚ) (hat_percentage : ℚ) :
  num_scarves = 18 →
  scarf_price = 2 →
  hat_ratio = 2 →
  hat_percentage = 60 / 100 →
  ∃ (total_money : ℚ), 
    total_money = 90 ∧
    (num_scarves : ℚ) * scarf_price = (1 - hat_percentage) * total_money ∧
    (hat_ratio * num_scarves : ℚ) * (hat_percentage * total_money / (hat_ratio * num_scarves)) = hat_percentage * total_money :=
by sorry

end kikis_money_l909_90914


namespace least_k_for_inequality_l909_90904

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.00010101 * (10 : ℝ)^m ≤ 100)) ∧ 
  (0.00010101 * (10 : ℝ)^k > 100) → 
  k = 6 := by
sorry

end least_k_for_inequality_l909_90904


namespace overlap_percentage_l909_90926

theorem overlap_percentage (square_side : ℝ) (rect_length rect_width : ℝ) :
  square_side = 18 →
  rect_length = 20 →
  rect_width = 18 →
  (rect_length * rect_width - 2 * square_side * square_side + (2 * square_side - rect_length) * rect_width) / (rect_length * rect_width) = 4/5 := by
  sorry

end overlap_percentage_l909_90926


namespace quadratic_rational_solutions_l909_90937

/-- The quadratic equation kx^2 + 18x + 2k = 0 has rational solutions if and only if k = 4, where k is a positive integer. -/
theorem quadratic_rational_solutions (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 18 * x + 2 * k = 0) ↔ k = 4 :=
by sorry

end quadratic_rational_solutions_l909_90937


namespace lacsap_hospital_staff_product_l909_90903

/-- Represents the Lacsap Hospital staff composition -/
structure HospitalStaff where
  doctors_excluding_emily : ℕ
  nurses_excluding_robert : ℕ
  emily_is_doctor : Bool
  robert_is_nurse : Bool

/-- Calculates the total number of doctors -/
def total_doctors (staff : HospitalStaff) : ℕ :=
  staff.doctors_excluding_emily + (if staff.emily_is_doctor then 1 else 0)

/-- Calculates the total number of nurses -/
def total_nurses (staff : HospitalStaff) : ℕ :=
  staff.nurses_excluding_robert + (if staff.robert_is_nurse then 1 else 0)

/-- Calculates the number of doctors excluding Robert -/
def doctors_excluding_robert (staff : HospitalStaff) : ℕ :=
  total_doctors staff

/-- Calculates the number of nurses excluding Robert -/
def nurses_excluding_robert (staff : HospitalStaff) : ℕ :=
  staff.nurses_excluding_robert

theorem lacsap_hospital_staff_product :
  ∀ (staff : HospitalStaff),
    staff.doctors_excluding_emily = 5 →
    staff.nurses_excluding_robert = 3 →
    staff.emily_is_doctor = true →
    staff.robert_is_nurse = true →
    (doctors_excluding_robert staff) * (nurses_excluding_robert staff) = 12 := by
  sorry

end lacsap_hospital_staff_product_l909_90903


namespace inverse_g_equals_five_l909_90927

-- Define the function g
def g (x : ℝ) : ℝ := 4 * x^3 + 3

-- State the theorem
theorem inverse_g_equals_five (x : ℝ) : g (g⁻¹ x) = x → g⁻¹ x = 5 → x = 503 := by
  sorry

end inverse_g_equals_five_l909_90927


namespace probability_four_ones_in_six_rolls_l909_90911

theorem probability_four_ones_in_six_rolls (n : ℕ) (p : ℚ) : 
  n = 10 → p = 1 / n → 
  (Nat.choose 6 4 : ℚ) * p^4 * (1 - p)^2 = 243 / 200000 := by
  sorry

end probability_four_ones_in_six_rolls_l909_90911


namespace ancient_chinese_math_problem_l909_90998

theorem ancient_chinese_math_problem (people : ℕ) (price : ℕ) : 
  (8 * people - price = 3) →
  (price - 7 * people = 4) →
  people = 7 := by
  sorry

end ancient_chinese_math_problem_l909_90998


namespace point_equal_distance_to_axes_l909_90931

/-- A point P with coordinates (m-4, 2m+7) has equal distance from both coordinate axes if and only if m = -11 or m = -1 -/
theorem point_equal_distance_to_axes (m : ℝ) : 
  |m - 4| = |2*m + 7| ↔ m = -11 ∨ m = -1 := by
sorry

end point_equal_distance_to_axes_l909_90931


namespace trigonometric_identity_proof_l909_90928

theorem trigonometric_identity_proof (x : ℝ) : 
  Real.sin (x + Real.pi / 3) + 2 * Real.sin (x - Real.pi / 3) - Real.sqrt 3 * Real.cos (2 * Real.pi / 3 - x) = 0 := by
  sorry

end trigonometric_identity_proof_l909_90928


namespace train_length_l909_90930

/-- Calculates the length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : 
  speed_kmh = 50.4 → time_sec = 20 → speed_kmh * (1000 / 3600) * time_sec = 280 := by
  sorry

#check train_length

end train_length_l909_90930


namespace polygon_interior_angles_l909_90949

theorem polygon_interior_angles (n : ℕ) (sum_angles : ℝ) : 
  sum_angles = 720 → (180 * (n - 2) : ℝ) = sum_angles → n = 6 := by
  sorry

end polygon_interior_angles_l909_90949


namespace sqrt_relation_l909_90991

theorem sqrt_relation (h : Real.sqrt 262.44 = 16.2) : Real.sqrt 2.6244 = 1.62 := by
  sorry

end sqrt_relation_l909_90991


namespace absolute_value_inequality_solution_set_l909_90961

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| < 2} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end absolute_value_inequality_solution_set_l909_90961


namespace not_true_from_false_premises_l909_90979

theorem not_true_from_false_premises (p q : Prop) : 
  ¬ (∀ (p q : Prop), (p → q) → (¬p → q)) :=
sorry

end not_true_from_false_premises_l909_90979


namespace min_records_theorem_l909_90917

/-- The number of different labels -/
def n : ℕ := 50

/-- The total number of records -/
def total_records : ℕ := n * (n + 1) / 2

/-- The number of records we want to ensure have the same label -/
def target : ℕ := 10

/-- The function that calculates the minimum number of records to draw -/
def min_records_to_draw : ℕ := 
  (target - 1) * (n - (target - 1)) + (target - 1) * target / 2

/-- Theorem stating the minimum number of records to draw -/
theorem min_records_theorem : 
  min_records_to_draw = 415 := by sorry

end min_records_theorem_l909_90917


namespace dan_remaining_marbles_l909_90964

def initial_marbles : ℕ := 64
def marbles_given_away : ℕ := 14

theorem dan_remaining_marbles :
  initial_marbles - marbles_given_away = 50 := by
  sorry

end dan_remaining_marbles_l909_90964


namespace max_value_of_sum_and_powers_l909_90997

theorem max_value_of_sum_and_powers (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 2 → 
  ∃ (max : ℝ), max = 2 ∧ ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → a + b + c = 2 → 
  a + b^3 + c^4 ≤ max := by
sorry

end max_value_of_sum_and_powers_l909_90997


namespace ice_skating_falls_ratio_l909_90924

/-- Given the number of falls for Steven, Stephanie, and Sonya while ice skating,
    prove that the ratio of Sonya's falls to half of Stephanie's falls is 3:4. -/
theorem ice_skating_falls_ratio 
  (steven_falls : ℕ) 
  (stephanie_falls : ℕ) 
  (sonya_falls : ℕ) 
  (h1 : steven_falls = 3)
  (h2 : stephanie_falls = steven_falls + 13)
  (h3 : sonya_falls = 6) :
  (sonya_falls : ℚ) / ((stephanie_falls : ℚ) / 2) = 3 / 4 := by
sorry

end ice_skating_falls_ratio_l909_90924


namespace tree_planting_equation_system_l909_90959

theorem tree_planting_equation_system :
  ∀ (x y : ℕ),
  (x + y = 20) →
  (3 * x + 2 * y = 52) →
  (∀ (total_pioneers total_trees boys_trees girls_trees : ℕ),
    total_pioneers = 20 →
    total_trees = 52 →
    boys_trees = 3 →
    girls_trees = 2 →
    x + y = total_pioneers ∧
    3 * x + 2 * y = total_trees) :=
by sorry

end tree_planting_equation_system_l909_90959


namespace circle_arrangement_exists_l909_90985

theorem circle_arrangement_exists : ∃ (a : Fin 12 → Fin 12), Function.Bijective a ∧
  ∀ (i j : Fin 12), i < j → |a i - a j| ≠ |i - j| := by
  sorry

end circle_arrangement_exists_l909_90985


namespace sum_of_roots_cubic_equation_l909_90905

theorem sum_of_roots_cubic_equation : 
  let f : ℝ → ℝ := fun x ↦ 3 * x^3 - 9 * x^2 - 72 * x + 6
  ∃ r p q : ℝ, (∀ x : ℝ, f x = 0 ↔ x = r ∨ x = p ∨ x = q) ∧ r + p + q = 3 := by
  sorry

end sum_of_roots_cubic_equation_l909_90905


namespace die_throws_for_most_likely_two_l909_90956

theorem die_throws_for_most_likely_two (n : ℕ) : 
  let p : ℚ := 1/6  -- probability of rolling a two
  let q : ℚ := 5/6  -- probability of not rolling a two
  let k₀ : ℕ := 32  -- most likely number of times a two is rolled
  (n * p - q ≤ k₀ ∧ k₀ ≤ n * p + p) → (191 ≤ n ∧ n ≤ 197) :=
by sorry

end die_throws_for_most_likely_two_l909_90956


namespace division_problem_l909_90962

theorem division_problem (d q : ℚ) : 
  (100 / d = q) → 
  (d * ⌊q⌋ ≤ 100) → 
  (100 - d * ⌊q⌋ = 4) → 
  (d = 16 ∧ q = 6.65) := by
  sorry

end division_problem_l909_90962


namespace power_sum_equality_l909_90970

theorem power_sum_equality : (-2)^2005 + (-2)^2006 = 2^2005 := by sorry

end power_sum_equality_l909_90970


namespace system_solution_unique_l909_90922

theorem system_solution_unique :
  ∃! (x y : ℚ), x = 2 * y ∧ 2 * x - y = 5 :=
by
  -- The unique solution is x = 10/3 and y = 5/3
  sorry

end system_solution_unique_l909_90922


namespace max_value_of_vector_difference_l909_90963

/-- Given plane vectors a and b satisfying |b| = 2|a| = 2, 
    the maximum value of |a - 2b| is 5. -/
theorem max_value_of_vector_difference (a b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2 * ‖a‖) (h2 : ‖b‖ = 2) : 
  ∃ (max : ℝ), max = 5 ∧ ∀ (x : ℝ × ℝ), x = a - 2 • b → ‖x‖ ≤ max :=
by sorry

end max_value_of_vector_difference_l909_90963


namespace water_from_river_calculation_l909_90982

/-- The amount of water Jacob collects from the river daily -/
def water_from_river : ℕ := 1700

/-- Jacob's water tank capacity in milliliters -/
def tank_capacity : ℕ := 50000

/-- Water collected from rain daily in milliliters -/
def water_from_rain : ℕ := 800

/-- Number of days to fill the tank -/
def days_to_fill : ℕ := 20

/-- Theorem stating that the amount of water Jacob collects from the river daily is 1700 milliliters -/
theorem water_from_river_calculation :
  water_from_river = (tank_capacity - water_from_rain * days_to_fill) / days_to_fill :=
by sorry

end water_from_river_calculation_l909_90982


namespace transformed_line_equation_l909_90968

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2 * y = 2

-- Define the stretch transformation
def stretch_transform (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 2 * y

-- Theorem: The equation of line l after transformation is x - y - 2 = 0
theorem transformed_line_equation (x' y' : ℝ) :
  (∃ x y, original_line x y ∧ stretch_transform x y x' y') →
  x' - y' - 2 = 0 :=
by sorry

end transformed_line_equation_l909_90968


namespace range_of_a_l909_90943

-- Define the circle M
def circle_M (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 2*a*x - 2*a*y = 0

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition that A is outside the circle M
def A_outside_M (a : ℝ) : Prop :=
  ∀ x y, circle_M a x y → (x - 0)^2 + (y - 2)^2 > (x - a)^2 + (y - a)^2

-- Define the existence of point T
def exists_T (a : ℝ) : Prop :=
  ∃ x y, circle_M a x y ∧ 
    Real.cos (Real.pi/4) * (x - 0) + Real.sin (Real.pi/4) * (y - 2) = 
    Real.sqrt ((x - 0)^2 + (y - 2)^2) * Real.sqrt ((x - a)^2 + (y - a)^2)

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, a > 0 → A_outside_M a → exists_T a → Real.sqrt 3 - 1 ≤ a ∧ a < 1 :=
sorry

end range_of_a_l909_90943


namespace leading_zeros_count_l909_90955

theorem leading_zeros_count (n : ℕ) (h : n = 20^22) :
  (∃ k : ℕ, (1 : ℚ) / n = k / 10^28 ∧ k ≥ 10^27 ∧ k < 10^28) :=
sorry

end leading_zeros_count_l909_90955


namespace Z_is_real_Z_is_pure_imaginary_Z_in_fourth_quadrant_l909_90951

-- Define the complex number Z as a function of real number m
def Z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

-- Theorem 1: Z is real iff m = -3 or m = 5
theorem Z_is_real (m : ℝ) : (Z m).im = 0 ↔ m = -3 ∨ m = 5 := by sorry

-- Theorem 2: Z is pure imaginary iff m = -2
theorem Z_is_pure_imaginary (m : ℝ) : (Z m).re = 0 ↔ m = -2 := by sorry

-- Theorem 3: Z is in the fourth quadrant iff -2 < m < 5
theorem Z_in_fourth_quadrant (m : ℝ) : 
  ((Z m).re > 0 ∧ (Z m).im < 0) ↔ -2 < m ∧ m < 5 := by sorry

end Z_is_real_Z_is_pure_imaginary_Z_in_fourth_quadrant_l909_90951


namespace smallest_n_digits_l909_90978

/-- Sum of digits function -/
def sum_of_digits (k : ℕ) : ℕ := sorry

/-- Theorem stating the number of digits in the smallest n satisfying the condition -/
theorem smallest_n_digits :
  ∃ n : ℕ,
    (∀ m : ℕ, m < n → sum_of_digits m - sum_of_digits (5 * m) ≠ 2013) ∧
    (sum_of_digits n - sum_of_digits (5 * n) = 2013) ∧
    (Nat.digits 10 n).length = 224 :=
sorry

end smallest_n_digits_l909_90978


namespace slope_intercept_sum_l909_90934

/-- Given a line passing through points (1,3) and (3,7), 
    the sum of its slope and y-intercept is equal to 3. -/
theorem slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) →   -- Line passes through (1,3)
  (7 = m * 3 + b) →   -- Line passes through (3,7)
  m + b = 3 :=
by sorry

end slope_intercept_sum_l909_90934
