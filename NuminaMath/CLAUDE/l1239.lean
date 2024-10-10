import Mathlib

namespace gcd_lcm_product_75_125_l1239_123921

theorem gcd_lcm_product_75_125 : 
  (Nat.gcd 75 125) * (Nat.lcm 75 125) = 9375 := by
  sorry

end gcd_lcm_product_75_125_l1239_123921


namespace distance_between_intersection_points_l1239_123949

-- Define the curve
def curve (t : ℝ) : ℝ × ℝ := (t^2, t^3)

-- Define the line
def line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 4}

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {p | p ∈ line ∧ ∃ t, curve t = p}

-- Theorem statement
theorem distance_between_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 := by
  sorry

end distance_between_intersection_points_l1239_123949


namespace intersection_A_B_l1239_123994

def set_A : Set ℝ := {x | ∃ t : ℝ, x = t^2 + 1}
def set_B : Set ℝ := {x | x * (x - 1) = 0}

theorem intersection_A_B :
  set_A ∩ set_B = {1} := by sorry

end intersection_A_B_l1239_123994


namespace trail_mix_weight_l1239_123950

def peanuts_weight : Float := 0.16666666666666666
def chocolate_chips_weight : Float := 0.16666666666666666
def raisins_weight : Float := 0.08333333333333333

theorem trail_mix_weight :
  peanuts_weight + chocolate_chips_weight + raisins_weight = 0.41666666666666663 := by
  sorry

end trail_mix_weight_l1239_123950


namespace chord_length_theorem_l1239_123920

/-- Representation of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Check if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Check if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem chord_length_theorem (C1 C2 C3 : Circle) : 
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  C1.radius = 3 →
  C2.radius = 9 →
  are_collinear C1.center C2.center C3.center →
  ∃ (chord : ℝ), chord = 6 * Real.sqrt 15 := by
  sorry

end chord_length_theorem_l1239_123920


namespace reservoir_capacity_problem_l1239_123956

/-- Theorem about a reservoir's capacity and water levels -/
theorem reservoir_capacity_problem (current_amount : ℝ) 
  (h1 : current_amount = 6)
  (h2 : current_amount = 2 * normal_level)
  (h3 : current_amount = 0.6 * total_capacity) :
  total_capacity - normal_level = 7 := by
  sorry

end reservoir_capacity_problem_l1239_123956


namespace flour_in_mixing_bowl_l1239_123940

theorem flour_in_mixing_bowl (total_sugar : ℚ) (total_flour : ℚ) 
  (h1 : total_sugar = 5)
  (h2 : total_flour = 18)
  (h3 : total_flour - total_sugar = 5) :
  total_flour - (total_sugar + 5) = 8 := by
  sorry

end flour_in_mixing_bowl_l1239_123940


namespace smallest_n_for_divisible_by_20_l1239_123959

theorem smallest_n_for_divisible_by_20 :
  ∃ (n : ℕ), n = 7 ∧ n ≥ 4 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 7 → m ≥ 4 →
    ∃ (T : Finset ℤ), T.card = m ∧
      ∀ (a b c d : ℤ), a ∈ T → b ∈ T → c ∈ T → d ∈ T →
      a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
      ¬(20 ∣ (a + b - c - d))) :=
by sorry

end smallest_n_for_divisible_by_20_l1239_123959


namespace prob_not_beside_partner_is_four_fifths_l1239_123968

/-- The number of people to be seated -/
def total_people : ℕ := 5

/-- The number of couples -/
def num_couples : ℕ := 2

/-- The number of single people -/
def num_singles : ℕ := total_people - 2 * num_couples

/-- The total number of seating arrangements -/
def total_arrangements : ℕ := Nat.factorial total_people

/-- The number of arrangements where all couples are seated together -/
def couples_together_arrangements : ℕ := 
  (Nat.factorial (num_couples + num_singles)) * (2 ^ num_couples)

/-- The probability that at least one person is not beside their partner -/
def prob_not_beside_partner : ℚ := 
  1 - (couples_together_arrangements : ℚ) / (total_arrangements : ℚ)

theorem prob_not_beside_partner_is_four_fifths : 
  prob_not_beside_partner = 4 / 5 := by sorry

end prob_not_beside_partner_is_four_fifths_l1239_123968


namespace haley_balls_count_l1239_123971

/-- Given that each bag can contain 4 balls and 9 bags will be used,
    prove that the number of balls Haley has is equal to 36. -/
theorem haley_balls_count (balls_per_bag : ℕ) (num_bags : ℕ) (h1 : balls_per_bag = 4) (h2 : num_bags = 9) :
  balls_per_bag * num_bags = 36 := by
  sorry

end haley_balls_count_l1239_123971


namespace correct_tense_for_ongoing_past_to_present_action_l1239_123970

/-- Represents different verb tenses -/
inductive VerbTense
  | simple_past
  | past_continuous
  | present_perfect_continuous
  | future_continuous

/-- Represents the characteristics of an action -/
structure ActionCharacteristics where
  ongoing : Bool
  started_in_past : Bool
  continues_to_present : Bool

/-- Theorem stating that for an action that is ongoing, started in the past, 
    and continues to the present, the correct tense is present perfect continuous -/
theorem correct_tense_for_ongoing_past_to_present_action 
  (action : ActionCharacteristics) 
  (h1 : action.ongoing = true) 
  (h2 : action.started_in_past = true) 
  (h3 : action.continues_to_present = true) : 
  VerbTense.present_perfect_continuous = 
    (match action with
      | ⟨true, true, true⟩ => VerbTense.present_perfect_continuous
      | _ => VerbTense.simple_past) :=
by sorry


end correct_tense_for_ongoing_past_to_present_action_l1239_123970


namespace lemon_production_increase_l1239_123919

/-- Represents the lemon production data for normal and engineered trees -/
structure LemonProduction where
  normal_lemons_per_year : ℕ
  grove_size : ℕ
  total_lemons : ℕ
  years : ℕ

/-- Calculates the percentage increase in lemon production -/
def percentage_increase (data : LemonProduction) : ℚ :=
  let normal_total := data.normal_lemons_per_year * data.years
  let engineered_per_tree := data.total_lemons / data.grove_size
  ((engineered_per_tree - normal_total) / normal_total) * 100

/-- Theorem stating the percentage increase in lemon production -/
theorem lemon_production_increase (data : LemonProduction) 
  (h1 : data.normal_lemons_per_year = 60)
  (h2 : data.grove_size = 1500)
  (h3 : data.total_lemons = 675000)
  (h4 : data.years = 5) :
  percentage_increase data = 50 := by
  sorry

end lemon_production_increase_l1239_123919


namespace train_speed_equation_l1239_123982

theorem train_speed_equation (x : ℝ) (h1 : x > 0) (h2 : x + 20 > 0) : 
  (400 / x) - (400 / (x + 20)) = 0.5 ↔ 
  (400 / x) - (400 / (x + 20)) = (30 : ℝ) / 60 ∧
  (400 / x) > (400 / (x + 20)) ∧
  (400 / x) - (400 / (x + 20)) = (400 / x - 400 / (x + 20)) :=
by sorry

end train_speed_equation_l1239_123982


namespace function_derivative_implies_coefficients_l1239_123946

/-- Given a function f(x) = x^m + ax with derivative f'(x) = 2x + 1, prove that m = 3 and a = 1 -/
theorem function_derivative_implies_coefficients 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^m + a*x) 
  (h2 : ∀ x, deriv f x = 2*x + 1) : 
  m = 3 ∧ a = 1 := by
  sorry

end function_derivative_implies_coefficients_l1239_123946


namespace inscribed_square_area_l1239_123916

/-- The area of a square inscribed in the ellipse x²/4 + y²/9 = 1, with sides parallel to the coordinate axes. -/
theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, x^2 / 4 + y^2 / 9 = 1 ∧ x = s ∧ y = s) →
  (4 * s^2 = 144 / 13) :=
by sorry

end inscribed_square_area_l1239_123916


namespace panthers_scored_17_points_l1239_123961

-- Define the points scored by the Wildcats
def wildcats_points : ℕ := 36

-- Define the difference in points between Wildcats and Panthers
def point_difference : ℕ := 19

-- Define the points scored by the Panthers
def panthers_points : ℕ := wildcats_points - point_difference

-- Theorem to prove
theorem panthers_scored_17_points : panthers_points = 17 := by
  sorry

end panthers_scored_17_points_l1239_123961


namespace quadratic_equation_root_l1239_123978

theorem quadratic_equation_root (a b : ℝ) (h : a ≠ 0) :
  (a * 2019^2 + b * 2019 + 2 = 0) →
  (a * (2019 - 1)^2 + b * (2019 - 1) = -2) :=
by sorry

end quadratic_equation_root_l1239_123978


namespace selection_methods_count_l1239_123989

def num_students : ℕ := 5
def num_selected : ℕ := 4
def num_days : ℕ := 3
def num_friday : ℕ := 2
def num_saturday : ℕ := 1
def num_sunday : ℕ := 1

theorem selection_methods_count :
  (num_students.choose num_friday) *
  ((num_students - num_friday).choose num_saturday) *
  ((num_students - num_friday - num_saturday).choose num_sunday) = 60 := by
  sorry

end selection_methods_count_l1239_123989


namespace min_value_of_quadratic_l1239_123929

theorem min_value_of_quadratic (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x^2 + 4*y^2 ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = x^2 + 4*y^2 → w ≥ z := by
sorry

end min_value_of_quadratic_l1239_123929


namespace cylinder_volume_equality_l1239_123988

theorem cylinder_volume_equality (y : ℝ) : y > 0 →
  (π * (7 + 4)^2 * 5 = π * 7^2 * (5 + y)) → y = 360 / 49 := by
  sorry

end cylinder_volume_equality_l1239_123988


namespace max_odd_integers_with_even_product_l1239_123998

theorem max_odd_integers_with_even_product (integers : Finset ℕ) 
  (h1 : integers.card = 7)
  (h2 : ∀ n ∈ integers, n > 0)
  (h3 : Even (integers.prod id)) :
  { odd_count : ℕ // odd_count ≤ 6 ∧ 
    ∃ (odd_subset : Finset ℕ), 
      odd_subset ⊆ integers ∧ 
      odd_subset.card = odd_count ∧ 
      ∀ n ∈ odd_subset, Odd n } :=
by sorry

end max_odd_integers_with_even_product_l1239_123998


namespace complex_equation_solution_l1239_123974

theorem complex_equation_solution (z : ℂ) :
  z / (1 - Complex.I) = Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l1239_123974


namespace vegetable_factory_profit_profit_function_correct_l1239_123922

/-- Represents the net profit function for a vegetable processing factory -/
def net_profit (n : ℕ) : ℚ :=
  -4 * n^2 + 80 * n - 144

/-- Represents the year when the business starts making a net profit -/
def profit_start_year : ℕ := 3

theorem vegetable_factory_profit :
  (∀ n : ℕ, n < profit_start_year → net_profit n ≤ 0) ∧
  (∀ n : ℕ, n ≥ profit_start_year → net_profit n > 0) :=
sorry

theorem profit_function_correct (n : ℕ) :
  net_profit n = n * 1 - (0.24 * n + n * (n - 1) / 2 * 0.08) - 1.44 :=
sorry

end vegetable_factory_profit_profit_function_correct_l1239_123922


namespace average_temperature_l1239_123996

def temperature_problem (new_york miami san_diego phoenix denver : ℝ) : Prop :=
  miami = new_york + 10 ∧
  san_diego = miami + 25 ∧
  phoenix = san_diego * 1.15 ∧
  denver = (new_york + miami + san_diego) / 3 - 5 ∧
  new_york = 80

theorem average_temperature 
  (new_york miami san_diego phoenix denver : ℝ) 
  (h : temperature_problem new_york miami san_diego phoenix denver) : 
  (new_york + miami + san_diego + phoenix + denver) / 5 = 101.45 := by
  sorry

#check average_temperature

end average_temperature_l1239_123996


namespace integral_tangent_fraction_l1239_123925

theorem integral_tangent_fraction :
  ∫ x in -Real.arccos (1 / Real.sqrt 5)..0, (11 - 3 * Real.tan x) / (Real.tan x + 3) = Real.log 45 - 3 * Real.arctan 2 := by
  sorry

end integral_tangent_fraction_l1239_123925


namespace subtraction_result_l1239_123967

theorem subtraction_result : 500000000000 - 3 * 111111111111 = 166666666667 := by
  sorry

end subtraction_result_l1239_123967


namespace arithmetic_sequence_property_l1239_123969

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 5 + a 7 + a 9 + a 11 = 200) :
  4 * a 5 - 2 * a 3 = 80 :=
sorry

end arithmetic_sequence_property_l1239_123969


namespace number_of_paths_equals_combination_l1239_123962

def grid_width : ℕ := 7
def grid_height : ℕ := 4

def total_steps : ℕ := grid_width + grid_height - 2
def up_steps : ℕ := grid_height - 1

theorem number_of_paths_equals_combination :
  (Nat.choose total_steps up_steps) = 84 := by
  sorry

end number_of_paths_equals_combination_l1239_123962


namespace parabola_c_value_l1239_123914

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

/-- The vertex of a parabola -/
def Parabola.vertex (p : Parabola) : ℝ × ℝ :=
  (2, 3)

theorem parabola_c_value (p : Parabola) :
  p.vertex = (2, 3) →
  p.x_coord 2 = 0 →
  p.c = -16 := by
  sorry

end parabola_c_value_l1239_123914


namespace value_of_3x2_minus_3y2_l1239_123945

theorem value_of_3x2_minus_3y2 (x y : ℝ) 
  (h1 : x + y = 20) 
  (h2 : x - y = 4) : 
  3 * (x^2 - y^2) = 240 := by
sorry

end value_of_3x2_minus_3y2_l1239_123945


namespace cupcake_net_profit_l1239_123933

/-- Calculates the net profit from selling cupcakes given the specified conditions. -/
theorem cupcake_net_profit : 
  let cost_per_cupcake : ℚ := 0.75
  let selling_price : ℚ := 2.00
  let burnt_cupcakes : ℕ := 24
  let eaten_cupcakes : ℕ := 9
  let total_cupcakes : ℕ := 72
  let sellable_cupcakes : ℕ := total_cupcakes - (burnt_cupcakes + eaten_cupcakes)
  let total_cost : ℚ := cost_per_cupcake * total_cupcakes
  let total_revenue : ℚ := selling_price * sellable_cupcakes
  total_revenue - total_cost = 24.00 := by
sorry


end cupcake_net_profit_l1239_123933


namespace sum_of_15th_set_l1239_123930

/-- Define the first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + (n - 1) * n / 2

/-- Define the last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- Define the sum of elements in the nth set -/
def sum_of_set (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem: The sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : sum_of_set 15 = 1695 := by
  sorry

end sum_of_15th_set_l1239_123930


namespace tan_alpha_sqrt_three_l1239_123911

theorem tan_alpha_sqrt_three (α : Real) (h : ∃ (x y : Real), x = 1 ∧ y = Real.sqrt 3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) : 
  Real.tan α = Real.sqrt 3 := by
sorry

end tan_alpha_sqrt_three_l1239_123911


namespace original_number_proof_l1239_123926

theorem original_number_proof (x : ℤ) : x = 16 ↔ 
  (∃ k : ℤ, x + 10 = 26 * k) ∧ 
  (∀ y : ℤ, y < 10 → ∀ m : ℤ, x + y ≠ 26 * m) :=
sorry

end original_number_proof_l1239_123926


namespace power_multiplication_l1239_123939

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by sorry

end power_multiplication_l1239_123939


namespace dans_initial_money_l1239_123984

/-- Calculates the initial amount of money given the number of items bought,
    the cost per item, and the amount left after purchase. -/
def initialMoney (itemsBought : ℕ) (costPerItem : ℕ) (amountLeft : ℕ) : ℕ :=
  itemsBought * costPerItem + amountLeft

/-- Theorem stating that given the specific conditions of Dan's purchase,
    his initial amount of money was $298. -/
theorem dans_initial_money :
  initialMoney 99 3 1 = 298 := by
  sorry

end dans_initial_money_l1239_123984


namespace parallelograms_from_congruent_triangles_l1239_123966

/-- Represents a triangle -/
structure Triangle :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Checks if two triangles are congruent -/
def are_congruent (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a quadrilateral is a parallelogram -/
def is_parallelogram (q : Quadrilateral) : Prop := sorry

/-- Forms quadrilaterals from two triangles -/
def form_quadrilaterals (t1 t2 : Triangle) : Set Quadrilateral := sorry

/-- Counts the number of parallelograms in a set of quadrilaterals -/
def count_parallelograms (qs : Set Quadrilateral) : ℕ := sorry

theorem parallelograms_from_congruent_triangles 
  (t1 t2 : Triangle) 
  (h : are_congruent t1 t2) : 
  count_parallelograms (form_quadrilaterals t1 t2) = 3 := sorry

end parallelograms_from_congruent_triangles_l1239_123966


namespace final_order_exact_points_total_games_l1239_123986

-- Define the structure for a team's game outcomes
structure TeamOutcome where
  name : String
  wins : Nat
  losses : Nat
  draws : Nat
  bonusWins : Nat
  extraBonus : Nat

-- Define the point system
def regularWinPoints : Nat := 3
def regularLossPoints : Nat := 0
def regularDrawPoints : Nat := 1
def bonusWinPoints : Nat := 2
def extraBonusPoints : Nat := 1

-- Calculate total points for a team
def calculatePoints (team : TeamOutcome) : Nat :=
  team.wins * regularWinPoints +
  team.losses * regularLossPoints +
  team.draws * regularDrawPoints +
  team.bonusWins * bonusWinPoints +
  team.extraBonus * extraBonusPoints

-- Define the teams
def soccerStars : TeamOutcome := ⟨"Team Soccer Stars", 18, 5, 7, 6, 4⟩
def lightningStrikers : TeamOutcome := ⟨"Lightning Strikers", 15, 8, 7, 5, 3⟩
def goalGrabbers : TeamOutcome := ⟨"Goal Grabbers", 21, 5, 4, 4, 9⟩
def cleverKickers : TeamOutcome := ⟨"Clever Kickers", 11, 10, 9, 2, 1⟩

-- Theorem to prove the final order of teams
theorem final_order :
  calculatePoints goalGrabbers > calculatePoints soccerStars ∧
  calculatePoints soccerStars > calculatePoints lightningStrikers ∧
  calculatePoints lightningStrikers > calculatePoints cleverKickers :=
by sorry

-- Theorem to prove the exact points for each team
theorem exact_points :
  calculatePoints goalGrabbers = 84 ∧
  calculatePoints soccerStars = 77 ∧
  calculatePoints lightningStrikers = 65 ∧
  calculatePoints cleverKickers = 47 :=
by sorry

-- Theorem to prove that each team played exactly 30 games
theorem total_games (team : TeamOutcome) :
  team.wins + team.losses + team.draws = 30 :=
by sorry

end final_order_exact_points_total_games_l1239_123986


namespace product_354_78_base7_units_digit_l1239_123901

-- Define the multiplication of two numbers in base 10
def base10Multiply (a b : ℕ) : ℕ := a * b

-- Define the conversion of a number from base 10 to base 7
def toBase7 (n : ℕ) : ℕ := n

-- Define the units digit of a number in base 7
def unitsDigitBase7 (n : ℕ) : ℕ := n % 7

-- Theorem statement
theorem product_354_78_base7_units_digit :
  unitsDigitBase7 (toBase7 (base10Multiply 354 78)) = 4 := by sorry

end product_354_78_base7_units_digit_l1239_123901


namespace star_calculation_star_equation_solutions_l1239_123997

-- Define the ☆ operation
noncomputable def star (x y : ℤ) : ℤ :=
  if x = 0 then |y|
  else if y = 0 then |x|
  else if (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) then |x| + |y|
  else -(|x| + |y|)

-- Theorem for the first part of the problem
theorem star_calculation : star 11 (star 0 (-12)) = 23 := by sorry

-- Theorem for the second part of the problem
theorem star_equation_solutions :
  {a : ℤ | 2 * (star 2 a) - 1 = 3 * a} = {3, -5} := by sorry

end star_calculation_star_equation_solutions_l1239_123997


namespace quadratic_roots_property_l1239_123995

theorem quadratic_roots_property (m n : ℝ) : 
  m^2 + 2*m - 8 = 0 → n^2 + 2*n - 8 = 0 → m^2 + 3*m + n = 6 := by
  sorry

end quadratic_roots_property_l1239_123995


namespace min_perimeter_isosceles_triangles_l1239_123999

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (Real.sqrt ((t.leg : ℝ)^2 - ((t.base : ℝ)/2)^2)) / 2

/-- Theorem: The minimum perimeter of two noncongruent integer-sided isosceles triangles
    with the same perimeter, same area, and bases in the ratio 8:7 is 676 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t₁ t₂ : IsoscelesTriangle),
    t₁ ≠ t₂ ∧
    perimeter t₁ = perimeter t₂ ∧
    area t₁ = area t₂ ∧
    8 * t₁.base = 7 * t₂.base ∧
    (∀ (s₁ s₂ : IsoscelesTriangle),
      s₁ ≠ s₂ →
      perimeter s₁ = perimeter s₂ →
      area s₁ = area s₂ →
      8 * s₁.base = 7 * s₂.base →
      perimeter t₁ ≤ perimeter s₁) ∧
    perimeter t₁ = 676 :=
sorry

end min_perimeter_isosceles_triangles_l1239_123999


namespace granola_cost_per_bag_l1239_123981

theorem granola_cost_per_bag 
  (total_bags : ℕ) 
  (full_price_bags : ℕ) 
  (full_price : ℚ) 
  (discounted_bags : ℕ) 
  (discounted_price : ℚ) 
  (net_profit : ℚ) 
  (h1 : total_bags = 20)
  (h2 : full_price_bags = 15)
  (h3 : full_price = 6)
  (h4 : discounted_bags = 5)
  (h5 : discounted_price = 4)
  (h6 : net_profit = 50)
  (h7 : total_bags = full_price_bags + discounted_bags) :
  (full_price_bags * full_price + discounted_bags * discounted_price - net_profit) / total_bags = 3 := by
  sorry

end granola_cost_per_bag_l1239_123981


namespace parabola_max_q_y_l1239_123913

/-- Represents a parabola of the form y = -x^2 + bx + c -/
structure Parabola where
  b : ℝ
  c : ℝ

/-- The y-coordinate of point Q where the parabola intersects x = -5 -/
def q_y_coord (p : Parabola) : ℝ :=
  25 - 5 * p.b + p.c

/-- Condition that the vertex of the parabola lies on the line y = 3x + 1 -/
def vertex_on_line (p : Parabola) : Prop :=
  (4 * p.c + p.b^2) / 4 = 3 * (p.b / 2) + 1

theorem parabola_max_q_y :
  ∃ (max_y : ℝ), max_y = -47/4 ∧
  ∀ (p : Parabola), vertex_on_line p →
  q_y_coord p ≤ max_y :=
sorry

end parabola_max_q_y_l1239_123913


namespace soccer_substitutions_remainder_l1239_123965

/-- Number of players in a soccer team -/
def total_players : ℕ := 24

/-- Number of starting players -/
def starting_players : ℕ := 12

/-- Number of substitute players -/
def substitute_players : ℕ := 12

/-- Maximum number of substitutions allowed -/
def max_substitutions : ℕ := 4

/-- 
Calculate the number of ways to make substitutions in a soccer game
n: current number of substitutions made
-/
def substitution_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 12 * (13 - n) * substitution_ways (n - 1)

/-- 
The total number of ways to make substitutions is the sum of ways
to make 0, 1, 2, 3, and 4 substitutions
-/
def total_ways : ℕ := 
  (List.range 5).map substitution_ways |>.sum

/-- Main theorem to prove -/
theorem soccer_substitutions_remainder :
  total_ways % 1000 = 573 := by
  sorry

end soccer_substitutions_remainder_l1239_123965


namespace max_value_is_72_l1239_123935

/-- Represents a type of stone with its weight and value -/
structure Stone where
  weight : ℕ
  value : ℕ

/-- The problem setup -/
def stones : List Stone := [
  { weight := 3, value := 9 },
  { weight := 6, value := 15 },
  { weight := 1, value := 1 }
]

/-- The maximum weight Tanya can carry -/
def maxWeight : ℕ := 24

/-- The minimum number of each type of stone available -/
def minStoneCount : ℕ := 10

/-- Calculates the maximum value of stones that can be carried given the constraints -/
def maxValue : ℕ :=
  sorry -- Proof goes here

/-- Theorem stating that the maximum value is 72 -/
theorem max_value_is_72 : maxValue = 72 := by
  sorry -- Proof goes here

end max_value_is_72_l1239_123935


namespace quadratic_equations_solutions_l1239_123977

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 8*x + 1 = 0) ∧
  (∃ x : ℝ, x*(x-2) - x + 2 = 0) ∧
  (∀ x : ℝ, x^2 - 8*x + 1 = 0 ↔ x = 4 + Real.sqrt 15 ∨ x = 4 - Real.sqrt 15) ∧
  (∀ x : ℝ, x*(x-2) - x + 2 = 0 ↔ x = 2 ∨ x = 1) :=
by sorry

end quadratic_equations_solutions_l1239_123977


namespace min_seats_occupied_l1239_123963

theorem min_seats_occupied (total_seats : ℕ) (initial_occupied : ℕ) : 
  total_seats = 150 → initial_occupied = 2 → 
  (∃ (additional_seats : ℕ), 
    additional_seats = 49 ∧ 
    ∀ (x : ℕ), x < additional_seats → 
      ∃ (y : ℕ), y ≤ total_seats - initial_occupied - x ∧ 
      y ≥ 2 ∧ 
      ∀ (z : ℕ), z < y → (z = 1 ∨ z = y)) :=
by sorry

end min_seats_occupied_l1239_123963


namespace solutions_of_equation_1_sum_of_reciprocals_squared_difference_of_solutions_l1239_123923

-- Question 1
theorem solutions_of_equation_1 (x : ℝ) :
  (x + 5 / x = -6) ↔ (x = -1 ∨ x = -5) :=
sorry

-- Question 2
theorem sum_of_reciprocals (m n : ℝ) :
  (m - 3 / m = 4) ∧ (n - 3 / n = 4) → 1 / m + 1 / n = -4 / 3 :=
sorry

-- Question 3
theorem squared_difference_of_solutions (a : ℝ) (x₁ x₂ : ℝ) :
  a ≠ 0 →
  (x₁ + (a^2 + 2*a) / (x₁ + 1) = 2*a + 1) →
  (x₂ + (a^2 + 2*a) / (x₂ + 1) = 2*a + 1) →
  (x₁ - x₂)^2 = 4 :=
sorry

end solutions_of_equation_1_sum_of_reciprocals_squared_difference_of_solutions_l1239_123923


namespace meal_cost_is_27_l1239_123947

/-- Represents the cost of a meal with tax and tip. -/
structure MealCost where
  pretax : ℝ
  tax_rate : ℝ
  tip_rate : ℝ
  total : ℝ

/-- Calculates the total cost of a meal including tax and tip. -/
def total_cost (m : MealCost) : ℝ :=
  m.pretax * (1 + m.tax_rate + m.tip_rate)

/-- Theorem stating that given the conditions, the pre-tax meal cost is $27. -/
theorem meal_cost_is_27 :
  ∃ (m : MealCost),
    m.tax_rate = 0.08 ∧
    m.tip_rate = 0.18 ∧
    m.total = 33.60 ∧
    total_cost m = m.total ∧
    m.pretax = 27 := by
  sorry

end meal_cost_is_27_l1239_123947


namespace unique_solution_exists_l1239_123991

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 4 * x - 5 * y + 2 * x * y

-- Theorem statement
theorem unique_solution_exists :
  ∃! y : ℝ, star 2 y = 5 := by sorry

end unique_solution_exists_l1239_123991


namespace water_depth_is_208_l1239_123903

/-- The depth of water given Ron's height -/
def water_depth (ron_height : ℝ) : ℝ := 16 * ron_height

/-- Ron's height in feet -/
def ron_height : ℝ := 13

/-- Theorem stating that the water depth is 208 feet -/
theorem water_depth_is_208 : water_depth ron_height = 208 := by
  sorry

end water_depth_is_208_l1239_123903


namespace cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2_l1239_123987

theorem cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2 :
  (Real.cos (10 * π / 180) - Real.sqrt 3 * Real.cos (-100 * π / 180)) /
  Real.sqrt (1 - Real.sin (10 * π / 180)) = Real.sqrt 2 := by
  sorry

end cos_sqrt3_cos_over_sqrt_sin_equals_sqrt2_l1239_123987


namespace medicine_supply_duration_l1239_123973

/-- Represents the duration in days that a supply of pills will last -/
def duration_in_days (num_pills : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) : ℚ :=
  (num_pills : ℚ) * (days_between_doses : ℚ) / pill_fraction

/-- Converts days to months, assuming 30 days per month -/
def days_to_months (days : ℚ) : ℚ :=
  days / 30

theorem medicine_supply_duration :
  let num_pills : ℕ := 60
  let pill_fraction : ℚ := 3/4
  let days_between_doses : ℕ := 3
  let duration_days := duration_in_days num_pills pill_fraction days_between_doses
  let duration_months := days_to_months duration_days
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/3 ∧ |duration_months - 3| < ε :=
by sorry

end medicine_supply_duration_l1239_123973


namespace binomial_inequality_l1239_123932

theorem binomial_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : x ≠ 0) (h3 : n ≥ 2) :
  (1 + x)^n > 1 + n * x := by
  sorry

end binomial_inequality_l1239_123932


namespace g_8_equals_1036_l1239_123937

def g (x : ℝ) : ℝ := 3*x^4 - 22*x^3 + 37*x^2 - 28*x - 84

theorem g_8_equals_1036 : g 8 = 1036 := by
  sorry

end g_8_equals_1036_l1239_123937


namespace combined_shape_area_l1239_123948

/-- The area of a shape formed by attaching a square to a rectangle -/
theorem combined_shape_area (rectangle_length rectangle_width square_side : Real) :
  rectangle_length = 0.45 →
  rectangle_width = 0.25 →
  square_side = 0.15 →
  rectangle_length * rectangle_width + square_side * square_side = 0.135 := by
  sorry

end combined_shape_area_l1239_123948


namespace profit_percent_l1239_123955

theorem profit_percent (P : ℝ) (h : P > 0) : 
  (2 / 3 * P) * (1 + (-0.2)) = 0.8 * ((5 / 6) * P) → 
  (P - (5 / 6 * P)) / (5 / 6 * P) = 0.2 := by
sorry

end profit_percent_l1239_123955


namespace only_M_eq_neg_M_is_valid_assignment_l1239_123976

/-- Represents a simple programming language expression -/
inductive Expr
  | num (n : Int)
  | var (name : String)
  | assign (lhs : String) (rhs : Expr)
  | add (e1 e2 : Expr)

/-- Checks if an expression is a valid assignment statement -/
def isValidAssignment : Expr → Bool
  | Expr.assign _ _ => true
  | _ => false

/-- The given statements from the problem -/
def statements : List Expr := [
  Expr.assign "A" (Expr.num 3),
  Expr.assign "M" (Expr.var "M"),
  Expr.assign "B" (Expr.assign "A" (Expr.num 2)),
  Expr.add (Expr.var "x") (Expr.var "y")
]

theorem only_M_eq_neg_M_is_valid_assignment :
  statements.filter isValidAssignment = [Expr.assign "M" (Expr.var "M")] := by sorry

end only_M_eq_neg_M_is_valid_assignment_l1239_123976


namespace liam_activity_balance_l1239_123975

/-- Utility function for Liam's activities -/
def utility (reading : ℝ) (basketball : ℝ) : ℝ := reading * basketball

/-- Wednesday's utility calculation -/
def wednesday_utility (t : ℝ) : ℝ := utility (10 - t) t

/-- Thursday's utility calculation -/
def thursday_utility (t : ℝ) : ℝ := utility (t + 4) (3 - t)

/-- The theorem stating that t = 3 is the only valid solution -/
theorem liam_activity_balance :
  ∃! t : ℝ, t > 0 ∧ t < 10 ∧ wednesday_utility t = thursday_utility t ∧ t = 3 := by sorry

end liam_activity_balance_l1239_123975


namespace solution_set_of_f_greater_than_4_range_of_a_l1239_123904

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

-- Statement 1
theorem solution_set_of_f_greater_than_4 :
  {x : ℝ | f x > 4} = Set.Ioi (-2) ∪ Set.Ioi 0 :=
sorry

-- Statement 2
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-3/2) 1, a + 1 > f x) → a ∈ Set.Ioi (3/2) :=
sorry

end solution_set_of_f_greater_than_4_range_of_a_l1239_123904


namespace roots_pure_imaginary_for_pure_imaginary_k_l1239_123917

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z k : ℂ) : Prop :=
  8 * z^2 + 6 * i * z - k = 0

-- Define a pure imaginary number
def is_pure_imaginary (x : ℂ) : Prop :=
  x.re = 0 ∧ x.im ≠ 0

-- Define the nature of roots
def roots_are_pure_imaginary (k : ℂ) : Prop :=
  ∀ z : ℂ, quadratic_equation z k → is_pure_imaginary z

-- Theorem statement
theorem roots_pure_imaginary_for_pure_imaginary_k :
  ∀ k : ℂ, is_pure_imaginary k → roots_are_pure_imaginary k :=
by sorry

end roots_pure_imaginary_for_pure_imaginary_k_l1239_123917


namespace agricultural_profit_optimization_l1239_123944

/-- Represents the profit optimization problem for an agricultural product company -/
theorem agricultural_profit_optimization
  (retail_profit : ℝ) -- Profit from retailing one box
  (wholesale_profit : ℝ) -- Profit from wholesaling one box
  (total_boxes : ℕ) -- Total number of boxes to be sold
  (retail_limit : ℝ) -- Maximum percentage of boxes that can be sold through retail
  (h1 : retail_profit = 70)
  (h2 : wholesale_profit = 40)
  (h3 : total_boxes = 1000)
  (h4 : retail_limit = 0.3) :
  ∃ (retail_boxes wholesale_boxes : ℕ) (max_profit : ℝ),
    retail_boxes + wholesale_boxes = total_boxes ∧
    retail_boxes ≤ (retail_limit * total_boxes) ∧
    max_profit = retail_profit * retail_boxes + wholesale_profit * wholesale_boxes ∧
    retail_boxes = 300 ∧
    wholesale_boxes = 700 ∧
    max_profit = 49000 ∧
    ∀ (r w : ℕ),
      r + w = total_boxes →
      r ≤ (retail_limit * total_boxes) →
      retail_profit * r + wholesale_profit * w ≤ max_profit :=
by sorry

end agricultural_profit_optimization_l1239_123944


namespace function_properties_l1239_123938

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x - 1

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a ≠ 0) :
  -- f(x) has an extremum at x = -1
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -1 ∧ |x + 1| < ε → f a x ≤ f a (-1)) →
  -- The line y = m intersects the graph of y = f(x) at three distinct points
  (∃ (m : ℝ), ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) →
  -- 1. When a < 0, f(x) is increasing on (-∞, +∞)
  (a < 0 → ∀ (x y : ℝ), x < y → f a x < f a y) ∧
  -- 2. When a > 0, f(x) is increasing on (-∞, -√a) ∪ (√a, +∞) and decreasing on (-√a, √a)
  (a > 0 → (∀ (x y : ℝ), (x < y ∧ y < -Real.sqrt a) ∨ (x > Real.sqrt a ∧ y > x) → f a x < f a y) ∧
           (∀ (x y : ℝ), -Real.sqrt a < x ∧ x < y ∧ y < Real.sqrt a → f a x > f a y)) ∧
  -- 3. The range of values for m is (-3, 1)
  (∃ (m : ℝ), -3 < m ∧ m < 1 ∧
    ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) ∧
  (∀ (m : ℝ), (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = m ∧ f a x₂ = m ∧ f a x₃ = m) → -3 < m ∧ m < 1) :=
by sorry

end function_properties_l1239_123938


namespace price_change_l1239_123980

theorem price_change (P : ℝ) (h : P > 0) : 
  let price_after_decrease := P * (1 - 0.2)
  let price_after_increase := price_after_decrease * (1 + 0.5)
  price_after_increase = P * 1.2 := by
sorry

end price_change_l1239_123980


namespace fraction_equality_l1239_123918

theorem fraction_equality (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end fraction_equality_l1239_123918


namespace circle_equation_l1239_123906

theorem circle_equation (A B : ℝ × ℝ) (h_A : A = (4, 2)) (h_B : B = (-1, 3)) :
  ∃ (D E F : ℝ),
    (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 ↔ 
      ((x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) ∨ 
       ∃ (x1 x2 y1 y2 : ℝ), 
         x1 + x2 + y1 + y2 = 2 ∧
         x1^2 + D*x1 + F = 0 ∧
         x2^2 + D*x2 + F = 0 ∧
         y1^2 + E*y1 + F = 0 ∧
         y2^2 + E*y2 + F = 0)) →
    D = -2 ∧ E = 0 ∧ F = -12 :=
by sorry

end circle_equation_l1239_123906


namespace repacking_books_leftover_l1239_123915

/-- The number of books left over when repacking from boxes of 42 to boxes of 45 -/
def books_left_over (initial_boxes : ℕ) (books_per_initial_box : ℕ) (books_per_new_box : ℕ) : ℕ :=
  (initial_boxes * books_per_initial_box) % books_per_new_box

/-- Theorem stating that repacking 1573 boxes of 42 books into boxes of 45 books leaves 6 books over -/
theorem repacking_books_leftover :
  books_left_over 1573 42 45 = 6 := by
  sorry

#eval books_left_over 1573 42 45

end repacking_books_leftover_l1239_123915


namespace even_function_implies_a_equals_negative_one_l1239_123927

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x + a)

theorem even_function_implies_a_equals_negative_one :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end even_function_implies_a_equals_negative_one_l1239_123927


namespace ellipse_and_line_intersection_l1239_123952

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * Real.sqrt 2 * x

-- Define the line l
def line (k m x y : ℝ) : Prop :=
  y = k * x + m

-- Define the isosceles triangle condition
def isosceles_triangle (A M N : ℝ × ℝ) : Prop :=
  (A.1 - M.1)^2 + (A.2 - M.2)^2 = (A.1 - N.1)^2 + (A.2 - N.2)^2

theorem ellipse_and_line_intersection
  (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ (x y : ℝ), ellipse a b x y ∧ parabola x y)
  (h4 : ∃ (x1 x2 y1 y2 : ℝ), 
    x1^2 + x2^2 + y1^2 + y2^2 = 2 * a * b * Real.sqrt 3)
  (k m : ℝ) (h5 : k ≠ 0)
  (h6 : ∃ (M N : ℝ × ℝ), 
    ellipse a b M.1 M.2 ∧ 
    ellipse a b N.1 N.2 ∧ 
    line k m M.1 M.2 ∧ 
    line k m N.1 N.2 ∧ 
    M ≠ N)
  (h7 : ∃ (A : ℝ × ℝ), ellipse a b A.1 A.2 ∧ A.2 < 0)
  (h8 : ∀ (A M N : ℝ × ℝ), 
    ellipse a b A.1 A.2 ∧ A.2 < 0 ∧
    ellipse a b M.1 M.2 ∧ 
    ellipse a b N.1 N.2 ∧ 
    line k m M.1 M.2 ∧ 
    line k m N.1 N.2 →
    isosceles_triangle A M N) :
  a = Real.sqrt 3 ∧ b = 1 ∧ 1/2 < m ∧ m < 2 := by
  sorry

end ellipse_and_line_intersection_l1239_123952


namespace function_domain_l1239_123960

/-- The function f(x) = √(2-2^x) + 1/ln(x) is defined if and only if x ∈ (0,1) -/
theorem function_domain (x : ℝ) : 
  (∃ (y : ℝ), y = Real.sqrt (2 - 2^x) + 1 / Real.log x) ↔ 0 < x ∧ x < 1 :=
by sorry

end function_domain_l1239_123960


namespace machine_production_time_l1239_123942

theorem machine_production_time (x : ℝ) (T : ℝ) : T = 10 :=
  let machine_B_rate := 2 * x / 5
  let combined_rate := x / 2
  have h1 : x / T + machine_B_rate = combined_rate := by sorry
  sorry

end machine_production_time_l1239_123942


namespace parallel_plane_through_point_l1239_123931

def plane_equation (x y z : ℝ) : ℝ := 3*x + 2*y - 4*z - 16

theorem parallel_plane_through_point :
  let given_plane (x y z : ℝ) := 3*x + 2*y - 4*z - 5
  (∀ (x y z : ℝ), plane_equation x y z = 0 ↔ given_plane x y z = k) ∧
  plane_equation 2 3 (-1) = 0 ∧
  (∃ (A B C D : ℤ), 
    (∀ (x y z : ℝ), plane_equation x y z = A*x + B*y + C*z + D) ∧
    A > 0 ∧
    Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1) :=
by sorry

end parallel_plane_through_point_l1239_123931


namespace complex_absolute_value_l1239_123928

def i : ℂ := Complex.I

theorem complex_absolute_value : 
  Complex.abs ((1 : ℂ) / (1 - i) - i) = Real.sqrt 2 / 2 := by sorry

end complex_absolute_value_l1239_123928


namespace m_range_l1239_123924

theorem m_range (m : ℝ) : 
  ¬((m + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m*x + 1 > 0)) → 
  m ≤ -2 ∨ m > -1 := by
sorry

end m_range_l1239_123924


namespace deer_families_moved_out_l1239_123979

theorem deer_families_moved_out (total : ℕ) (stayed : ℕ) (moved_out : ℕ) : 
  total = 79 → stayed = 45 → moved_out = total - stayed → moved_out = 34 := by
  sorry

end deer_families_moved_out_l1239_123979


namespace rectangle_diagonal_l1239_123953

/-- A rectangle with a perimeter of 72 meters and a length-to-width ratio of 5:2 has a diagonal of 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
  sorry

end rectangle_diagonal_l1239_123953


namespace julia_baking_days_l1239_123908

/-- The number of cakes Julia bakes per day -/
def cakes_per_day : ℕ := 4

/-- The number of cakes eaten every two days -/
def cakes_eaten_per_two_days : ℕ := 1

/-- The final number of cakes remaining -/
def final_cakes : ℕ := 21

/-- The number of days Julia baked cakes -/
def baking_days : ℕ := 6

/-- Proves that the number of days Julia baked cakes is 6 -/
theorem julia_baking_days :
  baking_days * cakes_per_day - (baking_days / 2) * cakes_eaten_per_two_days = final_cakes := by
  sorry


end julia_baking_days_l1239_123908


namespace students_on_field_trip_l1239_123907

/-- The number of students going on a field trip --/
def students_on_trip (seats_per_bus : ℕ) (num_buses : ℕ) : ℕ :=
  seats_per_bus * num_buses

/-- Theorem: The number of students on the trip is 28 given 7 seats per bus and 4 buses --/
theorem students_on_field_trip :
  students_on_trip 7 4 = 28 := by
  sorry

end students_on_field_trip_l1239_123907


namespace group_size_calculation_l1239_123934

/-- Proves that the number of people in a group is 5, given the average weight increase and weight difference of replaced individuals. -/
theorem group_size_calculation (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 1.5 → weight_difference = 7.5 → 
  (weight_difference / average_increase : ℝ) = 5 := by
  sorry

end group_size_calculation_l1239_123934


namespace intersection_of_M_and_N_l1239_123954

-- Define the sets M and N
def M : Set ℝ := {x | -2 < x ∧ x < 2}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 2} := by
  sorry

end intersection_of_M_and_N_l1239_123954


namespace power_function_positive_l1239_123990

theorem power_function_positive (α : ℝ) (x : ℝ) (h : x > 0) : x^α > 0 := by
  sorry

end power_function_positive_l1239_123990


namespace hexagon_coloring_count_l1239_123905

/-- A regular hexagon with 6 regions -/
inductive HexagonRegion
| A | B | C | D | E | F

/-- The available colors for planting -/
inductive PlantColor
| Color1 | Color2 | Color3 | Color4

/-- A coloring of the hexagon -/
def HexagonColoring := HexagonRegion → PlantColor

/-- Check if two regions are adjacent -/
def isAdjacent (r1 r2 : HexagonRegion) : Bool :=
  match r1, r2 with
  | HexagonRegion.A, HexagonRegion.B => true
  | HexagonRegion.A, HexagonRegion.F => true
  | HexagonRegion.B, HexagonRegion.C => true
  | HexagonRegion.C, HexagonRegion.D => true
  | HexagonRegion.D, HexagonRegion.E => true
  | HexagonRegion.E, HexagonRegion.F => true
  | _, _ => false

/-- Check if a coloring is valid (adjacent regions have different colors) -/
def isValidColoring (c : HexagonColoring) : Prop :=
  ∀ r1 r2 : HexagonRegion, isAdjacent r1 r2 → c r1 ≠ c r2

/-- The number of valid colorings -/
def numValidColorings : ℕ := 732

/-- The main theorem -/
theorem hexagon_coloring_count :
  (c : HexagonColoring) → (isValidColoring c) → numValidColorings = 732 := by
  sorry

end hexagon_coloring_count_l1239_123905


namespace remainder_equality_l1239_123964

theorem remainder_equality (a b c : ℕ) :
  (2 * a + b) % 10 = (2 * b + c) % 10 ∧
  (2 * b + c) % 10 = (2 * c + a) % 10 →
  a % 10 = b % 10 ∧ b % 10 = c % 10 := by
  sorry

end remainder_equality_l1239_123964


namespace max_cans_consumed_correct_verify_100_cans_l1239_123941

def exchange_rate : ℕ := 3

def max_cans_consumed (n : ℕ) : ℕ :=
  n + (n - 1) / 2

theorem max_cans_consumed_correct (n : ℕ) (h : n > 0) :
  ∃ (k : ℕ), k * exchange_rate ≤ max_cans_consumed n ∧
             max_cans_consumed n < (k + 1) * exchange_rate :=
by sorry

-- Verify the specific case for 100 cans
theorem verify_100_cans :
  max_cans_consumed 67 ≥ 100 ∧ max_cans_consumed 66 < 100 :=
by sorry

end max_cans_consumed_correct_verify_100_cans_l1239_123941


namespace solve_for_y_l1239_123909

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : y = -1 := by
  sorry

end solve_for_y_l1239_123909


namespace problem_solution_l1239_123912

def f (x : ℝ) := |x - 3| - 2
def g (x : ℝ) := -|x + 1| + 4

theorem problem_solution :
  (∀ x, f x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 6) ∧
  (∀ x, f x - g x ≥ -2) ∧
  (∀ m, (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3) := by
  sorry

end problem_solution_l1239_123912


namespace water_volume_calculation_l1239_123943

/-- Represents a cylindrical tank with an internal obstruction --/
structure Tank where
  radius : ℝ
  height : ℝ
  obstruction_radius : ℝ

/-- Calculates the volume of water in the tank --/
def water_volume (tank : Tank) (depth : ℝ) : ℝ :=
  sorry

/-- The specific tank in the problem --/
def problem_tank : Tank :=
  { radius := 5
  , height := 12
  , obstruction_radius := 2 }

theorem water_volume_calculation :
  water_volume problem_tank 3 = 110 * Real.pi - 96 := by
  sorry

end water_volume_calculation_l1239_123943


namespace compare_fractions_l1239_123985

theorem compare_fractions : -3/8 > -4/9 := by sorry

end compare_fractions_l1239_123985


namespace work_completion_time_l1239_123972

/-- The time taken for three workers to complete a work together, given their individual completion times -/
theorem work_completion_time (tx ty tz : ℝ) (htx : tx = 20) (hty : ty = 40) (htz : tz = 30) :
  (1 / tx + 1 / ty + 1 / tz)⁻¹ = 120 / 13 := by
  sorry

end work_completion_time_l1239_123972


namespace height_weight_correlation_l1239_123992

-- Define the relationship types
inductive Relationship
| Functional
| Correlated
| Unrelated

-- Define the variables
structure Square where
  side : ℝ

structure Vehicle where
  speed : ℝ

structure Person where
  height : ℝ
  weight : ℝ
  eyesight : ℝ

-- Define the relationships between variables
def square_area_perimeter_relation (s : Square) : Relationship :=
  Relationship.Functional

def vehicle_distance_time_relation (v : Vehicle) : Relationship :=
  Relationship.Functional

def person_height_weight_relation (p : Person) : Relationship :=
  Relationship.Correlated

def person_height_eyesight_relation (p : Person) : Relationship :=
  Relationship.Unrelated

-- Theorem statement
theorem height_weight_correlation :
  ∃ (p : Person), person_height_weight_relation p = Relationship.Correlated ∧
    (∀ (s : Square), square_area_perimeter_relation s ≠ Relationship.Correlated) ∧
    (∀ (v : Vehicle), vehicle_distance_time_relation v ≠ Relationship.Correlated) ∧
    (person_height_eyesight_relation p ≠ Relationship.Correlated) :=
  sorry

end height_weight_correlation_l1239_123992


namespace article_cost_price_l1239_123983

theorem article_cost_price : ∃ (C : ℝ), 
  (C = 600) ∧ 
  (∃ (SP : ℝ), SP = 1.05 * C) ∧ 
  (∃ (SP_new C_new : ℝ), 
    C_new = 0.95 * C ∧ 
    SP_new = 1.05 * C - 3 ∧ 
    SP_new = 1.045 * C_new) :=
by sorry

end article_cost_price_l1239_123983


namespace systematic_sampling_result_l1239_123936

def population_size : ℕ := 50
def sample_size : ℕ := 5
def starting_point : ℕ := 5
def step_size : ℕ := population_size / sample_size

def systematic_sample (start : ℕ) (step : ℕ) (n : ℕ) : List ℕ :=
  List.range n |> List.map (fun i => start + i * step)

theorem systematic_sampling_result :
  systematic_sample starting_point step_size sample_size = [5, 15, 25, 35, 45] := by
  sorry

end systematic_sampling_result_l1239_123936


namespace imaginary_unit_problem_l1239_123902

theorem imaginary_unit_problem : Complex.I * (1 + Complex.I)^2 = -2 := by sorry

end imaginary_unit_problem_l1239_123902


namespace empty_set_proof_l1239_123951

theorem empty_set_proof : {x : ℝ | x > 9 ∧ x < 3} = ∅ := by
  sorry

end empty_set_proof_l1239_123951


namespace range_of_k_l1239_123957

-- Define the propositions p and q
def p (x k : ℝ) : Prop := x ≥ k
def q (x : ℝ) : Prop := x^2 - x - 2 > 0

-- Define the property that p is sufficient but not necessary for q
def sufficient_not_necessary (k : ℝ) : Prop :=
  (∀ x, p x k → q x) ∧ (∃ x, q x ∧ ¬(p x k))

-- Theorem statement
theorem range_of_k :
  ∀ k, sufficient_not_necessary k ↔ k > 2 :=
sorry

end range_of_k_l1239_123957


namespace factorial_units_digit_zero_sum_factorials_units_digit_l1239_123910

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorialsUnitsDigit (n : ℕ) : ℕ :=
  unitsDigit ((List.range n).map factorial).sum

theorem factorial_units_digit_zero (n : ℕ) (h : n ≥ 5) :
  unitsDigit (factorial n) = 0 := by sorry

theorem sum_factorials_units_digit :
  sumFactorialsUnitsDigit 2010 = 3 := by sorry

end factorial_units_digit_zero_sum_factorials_units_digit_l1239_123910


namespace opposite_and_abs_of_2_minus_sqrt_3_l1239_123993

theorem opposite_and_abs_of_2_minus_sqrt_3 :
  let x : ℝ := 2 - Real.sqrt 3
  (- x = Real.sqrt 3 - 2) ∧ (abs x = 2 - Real.sqrt 3) := by sorry

end opposite_and_abs_of_2_minus_sqrt_3_l1239_123993


namespace journey_distance_l1239_123900

/-- Represents the journey of Jack and Peter -/
structure Journey where
  speed : ℝ
  distHomeToStore : ℝ
  distStoreToPeter : ℝ
  distPeterToStore : ℝ

/-- The total distance of the journey -/
def Journey.totalDistance (j : Journey) : ℝ :=
  j.distHomeToStore + j.distStoreToPeter + j.distPeterToStore

/-- Theorem stating the total distance of the journey -/
theorem journey_distance (j : Journey) 
  (h1 : j.speed > 0)
  (h2 : j.distStoreToPeter = 50)
  (h3 : j.distPeterToStore = 50)
  (h4 : j.distHomeToStore / j.speed = 2 * (j.distStoreToPeter / j.speed)) :
  j.totalDistance = 150 := by
  sorry

#check journey_distance

end journey_distance_l1239_123900


namespace total_dress_designs_l1239_123958

/-- The number of fabric colors available. -/
def num_colors : ℕ := 4

/-- The number of patterns available. -/
def num_patterns : ℕ := 5

/-- Each dress design requires exactly one color and one pattern. -/
axiom dress_design_requirement : True

/-- The total number of different dress designs. -/
def total_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the total number of different dress designs is 20. -/
theorem total_dress_designs : total_designs = 20 := by
  sorry

end total_dress_designs_l1239_123958
