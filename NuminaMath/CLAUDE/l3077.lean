import Mathlib

namespace NUMINAMATH_CALUDE_cylinder_radius_determination_l3077_307737

/-- Given a cylinder with height 4 units, if increasing its radius by 3 units
    and increasing its height by 3 units both result in the same volume increase,
    then the original radius of the cylinder is 12 units. -/
theorem cylinder_radius_determination (r : ℝ) (y : ℝ) : 
  (4 * π * ((r + 3)^2 - r^2) = y) →
  (3 * π * r^2 = y) →
  r = 12 := by sorry

end NUMINAMATH_CALUDE_cylinder_radius_determination_l3077_307737


namespace NUMINAMATH_CALUDE_correct_number_of_children_l3077_307784

/-- The number of crayons each child has -/
def crayons_per_child : ℕ := 5

/-- The total number of crayons -/
def total_crayons : ℕ := 50

/-- The number of children -/
def number_of_children : ℕ := total_crayons / crayons_per_child

theorem correct_number_of_children : number_of_children = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_children_l3077_307784


namespace NUMINAMATH_CALUDE_negation_equivalence_l3077_307743

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 2*x + 4 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3077_307743


namespace NUMINAMATH_CALUDE_equal_probability_l3077_307764

def num_dice : ℕ := 8
def min_value : ℕ := 2
def max_value : ℕ := 7

def sum_probability (sum : ℕ) : ℝ :=
  sorry

theorem equal_probability : sum_probability 20 = sum_probability 52 := by
  sorry

end NUMINAMATH_CALUDE_equal_probability_l3077_307764


namespace NUMINAMATH_CALUDE_expression_equivalence_l3077_307719

theorem expression_equivalence (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2 * y) :
  (x - 2 / x) * (y + 2 / y) = (1 / 2) * (x^2 - 2*x + 8 - 16 / x) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l3077_307719


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3077_307778

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (1 + 3*I) / (3 + I)
  0 < z.re ∧ 0 < z.im :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3077_307778


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l3077_307753

/-- The probability of drawing two black balls from a box containing white and black balls. -/
theorem probability_two_black_balls (white_balls black_balls : ℕ) 
  (h_white : white_balls = 7) (h_black : black_balls = 8) : 
  (black_balls.choose 2 : ℚ) / ((white_balls + black_balls).choose 2) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_l3077_307753


namespace NUMINAMATH_CALUDE_decimal_difference_equals_fraction_l3077_307785

/-- The repeating decimal 0.2̅3̅ -/
def repeating_decimal : ℚ := 23 / 99

/-- The terminating decimal 0.23 -/
def terminating_decimal : ℚ := 23 / 100

/-- The difference between the repeating decimal 0.2̅3̅ and the terminating decimal 0.23 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_equals_fraction : decimal_difference = 23 / 9900 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_equals_fraction_l3077_307785


namespace NUMINAMATH_CALUDE_river_crossing_theorem_l3077_307702

/-- Calculates the time required for all explorers to cross a river --/
def river_crossing_time (num_explorers : ℕ) (boat_capacity : ℕ) (crossing_time : ℕ) : ℕ :=
  let first_trip := boat_capacity
  let remaining_explorers := num_explorers - first_trip
  let subsequent_trips := (remaining_explorers + 4) / 5  -- Ceiling division
  let total_crossings := 2 * subsequent_trips + 1
  total_crossings * crossing_time

theorem river_crossing_theorem :
  river_crossing_time 60 6 3 = 69 := by
  sorry

end NUMINAMATH_CALUDE_river_crossing_theorem_l3077_307702


namespace NUMINAMATH_CALUDE_expression_evaluation_l3077_307763

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- The expression to be evaluated -/
def expression : ℂ := 2 * i^13 - 3 * i^18 + 4 * i^23 - 5 * i^28 + 6 * i^33

/-- The theorem stating the equality of the expression and its simplified form -/
theorem expression_evaluation : expression = 4 * i - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3077_307763


namespace NUMINAMATH_CALUDE_stratified_sample_composition_l3077_307742

def total_students : ℕ := 2700
def freshmen : ℕ := 900
def sophomores : ℕ := 1200
def juniors : ℕ := 600
def sample_size : ℕ := 135

theorem stratified_sample_composition :
  let freshmen_sample := (freshmen * sample_size) / total_students
  let sophomores_sample := (sophomores * sample_size) / total_students
  let juniors_sample := (juniors * sample_size) / total_students
  freshmen_sample = 45 ∧ sophomores_sample = 60 ∧ juniors_sample = 30 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_composition_l3077_307742


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l3077_307795

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

theorem tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨1, -1, 2⟩
  let a₂ : Point3D := ⟨2, 1, 2⟩
  let a₃ : Point3D := ⟨1, 1, 4⟩
  let a₄ : Point3D := ⟨6, -3, 8⟩
  (tetrahedronVolume a₁ a₂ a₃ a₄ = 6) ∧
  (tetrahedronHeight a₄ a₁ a₂ a₃ = 3 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l3077_307795


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_quadratic_function_solution_l3077_307744

-- Equation 1
theorem equation_one_solution (x : ℝ) : 
  x^2 - 6*x + 3 = 0 ↔ x = 3 + Real.sqrt 6 ∨ x = 3 - Real.sqrt 6 :=
sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) :
  x*(x+2) = 3*(x+2) ↔ x = -2 ∨ x = 3 :=
sorry

-- Equation 3
def quadratic_function (x : ℝ) : ℝ := 4*x^2 + 5*x

theorem quadratic_function_solution :
  (quadratic_function 0 = 0) ∧ 
  (quadratic_function (-1) = -1) ∧ 
  (quadratic_function 1 = 9) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_quadratic_function_solution_l3077_307744


namespace NUMINAMATH_CALUDE_power_of_two_equation_l3077_307748

theorem power_of_two_equation (m : ℤ) : 
  2^2000 - 2^1999 - 2^1998 + 2^1997 = m * 2^1997 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_l3077_307748


namespace NUMINAMATH_CALUDE_afternoon_emails_l3077_307799

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- The difference between morning and evening emails -/
def email_difference : ℕ := 2

/-- Theorem stating that Jack received 7 emails in the afternoon -/
theorem afternoon_emails : ℕ := by
  sorry

end NUMINAMATH_CALUDE_afternoon_emails_l3077_307799


namespace NUMINAMATH_CALUDE_intersections_of_related_functions_l3077_307701

/-- Given a quadratic function that intersects (0, 2) and (1, 1), 
    prove that the related linear function intersects the axes at (1/2, 0) and (0, -1) -/
theorem intersections_of_related_functions 
  (a c : ℝ) 
  (h1 : c = 2) 
  (h2 : a + c = 1) : 
  let f (x : ℝ) := c * x + a
  (f (1/2) = 0 ∧ f 0 = -1) := by
sorry

end NUMINAMATH_CALUDE_intersections_of_related_functions_l3077_307701


namespace NUMINAMATH_CALUDE_tournament_max_wins_l3077_307786

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Minimum number of participants required for n wins -/
def f (n : ℕ) : ℕ := fib (n + 2)

/-- Tournament properties -/
structure Tournament :=
  (participants : ℕ)
  (one_match_at_a_time : Bool)
  (loser_drops_out : Bool)
  (max_win_diff : ℕ)

/-- Main theorem -/
theorem tournament_max_wins (t : Tournament) (h1 : t.participants = 55) 
  (h2 : t.one_match_at_a_time = true) (h3 : t.loser_drops_out = true) 
  (h4 : t.max_win_diff = 1) : 
  (∃ (n : ℕ), f n ≤ t.participants ∧ f (n + 1) > t.participants ∧ n = 8) :=
sorry

end NUMINAMATH_CALUDE_tournament_max_wins_l3077_307786


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_integers_l3077_307724

theorem largest_divisor_of_consecutive_even_integers (n : ℕ) : 
  ∃ k : ℕ, (2*n) * (2*n + 2) * (2*n + 4) = 48 * k :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_even_integers_l3077_307724


namespace NUMINAMATH_CALUDE_bird_stork_difference_l3077_307779

theorem bird_stork_difference : 
  ∀ (initial_storks initial_birds joining_birds : ℕ),
    initial_storks = 5 →
    initial_birds = 3 →
    joining_birds = 4 →
    (initial_birds + joining_birds) - initial_storks = 2 := by
  sorry

end NUMINAMATH_CALUDE_bird_stork_difference_l3077_307779


namespace NUMINAMATH_CALUDE_joan_seashells_l3077_307781

/-- The number of seashells Joan gave to Sam -/
def seashells_given : ℕ := 43

/-- The number of seashells Joan has left -/
def seashells_left : ℕ := 27

/-- The total number of seashells Joan found originally -/
def total_seashells : ℕ := seashells_given + seashells_left

theorem joan_seashells : total_seashells = 70 := by
  sorry

end NUMINAMATH_CALUDE_joan_seashells_l3077_307781


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3077_307704

theorem quadratic_is_square_of_binomial (x k : ℝ) : 
  (∃ a b : ℝ, x^2 - 20*x + k = (a*x + b)^2) ↔ k = 100 := by
sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3077_307704


namespace NUMINAMATH_CALUDE_volume_ratio_l3077_307754

-- Define the vertices of the larger pyramid
def large_pyramid_vertices : List (Fin 4 → ℚ) := [
  (λ i => if i = 0 then 1 else 0),
  (λ i => if i = 1 then 1 else 0),
  (λ i => if i = 2 then 1 else 0),
  (λ i => if i = 3 then 1 else 0),
  (λ _ => 0)
]

-- Define the center of the base of the larger pyramid
def base_center : Fin 4 → ℚ := λ _ => 1/4

-- Define the vertices of the smaller pyramid
def small_pyramid_vertices : List (Fin 4 → ℚ) := 
  base_center :: (List.range 4).map (λ i => λ j => if i = j then 1/2 else 0)

-- Define a function to calculate the volume of a pyramid
def pyramid_volume (vertices : List (Fin 4 → ℚ)) : ℚ := sorry

-- Theorem stating the volume ratio
theorem volume_ratio : 
  (pyramid_volume small_pyramid_vertices) / (pyramid_volume large_pyramid_vertices) = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_l3077_307754


namespace NUMINAMATH_CALUDE_polynomial_coefficient_G_l3077_307726

-- Define the polynomial p(z)
def p (z E F G H I : ℤ) : ℤ := z^7 - 13*z^6 + E*z^5 + F*z^4 + G*z^3 + H*z^2 + I*z + 36

-- Define the property that all roots are positive integers
def all_roots_positive_integers (p : ℤ → ℤ) : Prop :=
  ∀ z : ℤ, p z = 0 → z > 0

-- Theorem statement
theorem polynomial_coefficient_G (E F G H I : ℤ) :
  all_roots_positive_integers (p · E F G H I) →
  G = -82 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_coefficient_G_l3077_307726


namespace NUMINAMATH_CALUDE_harveys_steak_sales_l3077_307712

/-- Represents the number of steaks Harvey sold after having 12 steaks left -/
def steaks_sold_after_12_left (initial_steaks : ℕ) (steaks_left : ℕ) (total_sold : ℕ) : ℕ :=
  total_sold - (initial_steaks - steaks_left)

/-- Theorem stating that Harvey sold 4 steaks after having 12 steaks left -/
theorem harveys_steak_sales : steaks_sold_after_12_left 25 12 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_harveys_steak_sales_l3077_307712


namespace NUMINAMATH_CALUDE_y_greater_than_x_l3077_307761

theorem y_greater_than_x (x y : ℝ) (h1 : x + y > 2*x) (h2 : x - y < 2*y) : y > x := by
  sorry

end NUMINAMATH_CALUDE_y_greater_than_x_l3077_307761


namespace NUMINAMATH_CALUDE_valid_colorings_3x10_l3077_307710

/-- Represents the number of ways to color a 3 × 2n grid with black and white,
    such that no five squares in an 'X' configuration are all the same color. -/
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 8
  | n+2 => 7 * a n + 4 * a (n-1)

/-- The number of valid colorings for a 3 × 10 grid -/
def N : ℕ := (a 5)^2

/-- Theorem stating that the number of valid colorings for a 3 × 10 grid
    is equal to 25636^2 -/
theorem valid_colorings_3x10 : N = 25636^2 := by
  sorry

end NUMINAMATH_CALUDE_valid_colorings_3x10_l3077_307710


namespace NUMINAMATH_CALUDE_investment_income_l3077_307762

/-- Proves that an investment of $6800 in a 60% stock at a price of 136 yields an annual income of $3000 -/
theorem investment_income (investment : ℝ) (stock_percentage : ℝ) (stock_price : ℝ) (annual_income : ℝ) : 
  investment = 6800 ∧ 
  stock_percentage = 0.60 ∧ 
  stock_price = 136 ∧ 
  annual_income = 3000 → 
  investment * (stock_percentage / stock_price) = annual_income :=
by sorry

end NUMINAMATH_CALUDE_investment_income_l3077_307762


namespace NUMINAMATH_CALUDE_solution_set_implies_k_inequality_implies_k_range_l3077_307700

/-- The quadratic function f(x) = kx^2 - 2x + 6k --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

/-- Theorem 1: If f(x) < 0 has solution set (2,3), then k = 2/5 --/
theorem solution_set_implies_k (k : ℝ) :
  (∀ x, f k x < 0 ↔ 2 < x ∧ x < 3) → k = 2/5 := by sorry

/-- Theorem 2: If k > 0 and f(x) < 0 for all 2 < x < 3, then 0 < k ≤ 2/5 --/
theorem inequality_implies_k_range (k : ℝ) :
  k > 0 → (∀ x, 2 < x → x < 3 → f k x < 0) → 0 < k ∧ k ≤ 2/5 := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_k_inequality_implies_k_range_l3077_307700


namespace NUMINAMATH_CALUDE_sqrt_calculation_l3077_307777

theorem sqrt_calculation : 
  Real.sqrt 3 * Real.sqrt 12 - 2 * Real.sqrt 6 / Real.sqrt 3 + Real.sqrt 32 + (Real.sqrt 2)^2 = 8 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l3077_307777


namespace NUMINAMATH_CALUDE_speed_range_correct_l3077_307790

/-- Represents a road roller with its properties and the road to be compressed -/
structure RoadRoller where
  strip_width : Real
  overlap_ratio : Real
  road_length : Real
  road_width : Real
  compression_count : Nat
  min_time : Real
  max_time : Real

/-- Calculates the range of speeds for the road roller to complete the task -/
def calculate_speed_range (roller : RoadRoller) : Set Real :=
  let effective_width := roller.strip_width * (1 - roller.overlap_ratio)
  let passes := Nat.ceil (roller.road_width / effective_width)
  let total_distance := passes * roller.compression_count * 2 * roller.road_length
  let min_speed := total_distance / (roller.max_time * 1000)
  let max_speed := total_distance / (roller.min_time * 1000)
  {x | min_speed ≤ x ∧ x ≤ max_speed}

/-- Theorem stating that the calculated speed range is correct -/
theorem speed_range_correct (roller : RoadRoller) :
  roller.strip_width = 0.85 ∧
  roller.overlap_ratio = 1/4 ∧
  roller.road_length = 750 ∧
  roller.road_width = 6.5 ∧
  roller.compression_count = 2 ∧
  roller.min_time = 5 ∧
  roller.max_time = 6 →
  ∀ x ∈ calculate_speed_range roller, 2.75 ≤ x ∧ x ≤ 3.3 :=
by sorry

end NUMINAMATH_CALUDE_speed_range_correct_l3077_307790


namespace NUMINAMATH_CALUDE_games_within_division_is_48_l3077_307706

/-- Represents a basketball league with specific game scheduling rules -/
structure BasketballLeague where
  N : ℕ  -- Number of games against each team in own division
  M : ℕ  -- Number of games against each team in other division
  h1 : N > 3 * M
  h2 : M > 5
  h3 : 3 * N + 4 * M = 88

/-- The number of games a team plays within its own division -/
def gamesWithinDivision (league : BasketballLeague) : ℕ := 3 * league.N

/-- Theorem stating the number of games played within a team's own division -/
theorem games_within_division_is_48 (league : BasketballLeague) :
  gamesWithinDivision league = 48 := by
  sorry

#check games_within_division_is_48

end NUMINAMATH_CALUDE_games_within_division_is_48_l3077_307706


namespace NUMINAMATH_CALUDE_eating_contest_l3077_307728

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (noah_burgers jacob_pies mason_hotdog_weight : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  noah_burgers = 8 →
  mason_hotdog_weight = 30 →
  mason_hotdog_weight / hot_dog_weight = 15 :=
by sorry

end NUMINAMATH_CALUDE_eating_contest_l3077_307728


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l3077_307783

/-- The cost price of a bicycle for seller A, given the following conditions:
  - A sells the bicycle to B at a profit of 20%
  - B sells it to C at a profit of 25%
  - C pays Rs. 225 for the bicycle
-/
theorem bicycle_cost_price (profit_A_to_B : ℝ) (profit_B_to_C : ℝ) (price_C : ℝ) :
  profit_A_to_B = 0.20 →
  profit_B_to_C = 0.25 →
  price_C = 225 →
  ∃ (cost_price_A : ℝ), cost_price_A = 150 ∧
    price_C = cost_price_A * (1 + profit_A_to_B) * (1 + profit_B_to_C) :=
by sorry

end NUMINAMATH_CALUDE_bicycle_cost_price_l3077_307783


namespace NUMINAMATH_CALUDE_another_beast_holds_all_candy_l3077_307729

/-- Represents the state of candy distribution among beasts -/
inductive CandyDistribution
  | initial (n : ℕ)  -- Initial distribution with Grogg having n candies
  | distribute (d : List ℕ)  -- List representing candy counts for each beast

/-- Represents a single step in the candy distribution process -/
def distributeStep (d : CandyDistribution) : CandyDistribution :=
  match d with
  | CandyDistribution.initial n => CandyDistribution.distribute (List.replicate n 1)
  | CandyDistribution.distribute (k :: rest) => 
      CandyDistribution.distribute (List.map (· + 1) (List.take k rest) ++ List.drop k rest)
  | _ => d

/-- Checks if all candy is held by a single beast (except Grogg) -/
def allCandyHeldBySingleBeast (d : CandyDistribution) : Bool :=
  match d with
  | CandyDistribution.distribute [n] => true
  | _ => false

/-- Main theorem: Another beast holds all candy iff n = 1 or n = 2 -/
theorem another_beast_holds_all_candy (n : ℕ) (h : n ≥ 1) :
  (∃ d : CandyDistribution, d = distributeStep (CandyDistribution.initial n) ∧ 
    allCandyHeldBySingleBeast d) ↔ n = 1 ∨ n = 2 :=
  sorry

end NUMINAMATH_CALUDE_another_beast_holds_all_candy_l3077_307729


namespace NUMINAMATH_CALUDE_circle_properties_l3077_307780

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 6 * y - 10 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 3 * y = 0

-- Define the given conditions
axiom circle_intersects_line1 : ∃ (c : Circle), line1 4 (-1)
axiom center_on_line2 : ∀ (c : Circle), line2 c.center.1 c.center.2

-- Define the theorem to prove
theorem circle_properties (c : Circle) :
  (∀ (x y : ℝ), (x - 3)^2 + (y - 5)^2 = 37 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ (chord : ℝ), chord = 2 * Real.sqrt 3 ∧
    ∀ (l : ℝ → ℝ → Prop),
      (∀ x y, l x y → x = 0 ∨ y = 0) →
      (∃ x₁ y₁ x₂ y₂, l x₁ y₁ ∧ l x₂ y₂ ∧
        (x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
        (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
        (x₂ - x₁)^2 + (y₂ - y₁)^2 ≤ chord^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_circle_properties_l3077_307780


namespace NUMINAMATH_CALUDE_min_value_3x_4y_l3077_307796

theorem min_value_3x_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 5 * x₀ * y₀ ∧ 3 * x₀ + 4 * y₀ = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_3x_4y_l3077_307796


namespace NUMINAMATH_CALUDE_equation_solutions_l3077_307738

theorem equation_solutions : 
  {x : ℝ | (x - 1) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 1) / 
           ((x - 3) * (x - 6) * (x - 3)) = 2 ∧ 
           x ≠ 3 ∧ x ≠ 6} = 
  {2 + Real.sqrt 2, 2 - Real.sqrt 2} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3077_307738


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3077_307791

-- Define the conditions
def p (a : ℝ) : Prop := a ≤ 2
def q (a : ℝ) : Prop := a * (a - 2) ≤ 0

-- Theorem statement
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ a : ℝ, ¬(p a) → ¬(q a)) ∧
  ¬(∀ a : ℝ, ¬(q a) → ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3077_307791


namespace NUMINAMATH_CALUDE_range_of_a_l3077_307782

/-- A decreasing function defined on (-∞, 3] -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x > f y) ∧ (∀ x, f x ≤ 3)

theorem range_of_a (f : ℝ → ℝ) (h_f : DecreasingFunction f)
    (h_ineq : ∀ x a : ℝ, f (a^2 - Real.sin x) ≤ f (a + 1 + Real.cos x ^ 2)) :
    ∀ a : ℝ, a ∈ Set.Icc (-Real.sqrt 2) ((1 - Real.sqrt 10) / 2) :=
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3077_307782


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3077_307727

theorem complex_fraction_simplification :
  (2 - Complex.I) / (1 + 2 * Complex.I) = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3077_307727


namespace NUMINAMATH_CALUDE_expression_simplification_l3077_307745

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x + 2) / (x^2 - 2*x) / ((8*x) / (x - 2) + x - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3077_307745


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3077_307757

-- Problem 1
theorem problem_1 (x : ℝ) (h : x = 2 - Real.sqrt 7) : x^2 - 4*x + 5 = 8 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : 2*x = Real.sqrt 5 + 1) : x^3 - 2*x^2 = -1 := by
  sorry

-- Problem 3
theorem problem_3 (a : ℝ) (h : a^2 = Real.sqrt (a^2 + 10) + 3) : a^2 + 1/a^2 = Real.sqrt 53 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l3077_307757


namespace NUMINAMATH_CALUDE_y_derivative_l3077_307718

noncomputable def y (x : ℝ) : ℝ := 
  (3 * x^2 - 4 * x + 2) * Real.sqrt (9 * x^2 - 12 * x + 3) + 
  (3 * x - 2)^4 * Real.arcsin (1 / (3 * x - 2))

theorem y_derivative (x : ℝ) (h : 3 * x - 2 > 0) : 
  deriv y x = 12 * (3 * x - 2)^3 * Real.arcsin (1 / (3 * x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_y_derivative_l3077_307718


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3077_307774

theorem fourth_root_equation_solutions : 
  {x : ℝ | x > 0 ∧ x^(1/4) = 15 / (8 - x^(1/4))} = {81, 625} := by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3077_307774


namespace NUMINAMATH_CALUDE_gold_coins_percentage_l3077_307731

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  total : ℝ
  beads : ℝ
  coins : ℝ
  silver_coins : ℝ
  gold_coins : ℝ

/-- The conditions of the urn as given in the problem -/
def urn_conditions (u : UrnComposition) : Prop :=
  u.total > 0 ∧
  u.beads + u.coins = u.total ∧
  u.silver_coins + u.gold_coins = u.coins ∧
  u.beads = 0.3 * u.total ∧
  u.silver_coins = 0.3 * u.coins

/-- The theorem stating that 49% of the objects in the urn are gold coins -/
theorem gold_coins_percentage (u : UrnComposition) 
  (h : urn_conditions u) : u.gold_coins / u.total = 0.49 := by
  sorry


end NUMINAMATH_CALUDE_gold_coins_percentage_l3077_307731


namespace NUMINAMATH_CALUDE_prob_not_six_four_dice_value_l3077_307705

/-- The probability that (a-6)(b-6)(c-6)(d-6) ≠ 0 when four standard dice are tossed -/
def prob_not_six_four_dice : ℚ :=
  625 / 1296

/-- Theorem stating that the probability of (a-6)(b-6)(c-6)(d-6) ≠ 0 
    when four standard dice are tossed is equal to 625/1296 -/
theorem prob_not_six_four_dice_value : 
  prob_not_six_four_dice = 625 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_prob_not_six_four_dice_value_l3077_307705


namespace NUMINAMATH_CALUDE_roses_cut_is_difference_jessica_roses_problem_l3077_307733

/-- The number of roses Jessica cut from her flower garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that the number of roses Jessica cut is the difference between the final and initial number of roses -/
theorem roses_cut_is_difference (initial_roses final_roses : ℕ) 
  (h : final_roses ≥ initial_roses) : 
  roses_cut initial_roses final_roses = final_roses - initial_roses :=
by
  sorry

/-- The specific problem instance -/
theorem jessica_roses_problem :
  roses_cut 10 18 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_roses_cut_is_difference_jessica_roses_problem_l3077_307733


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_on_hyperbola_l3077_307709

/-- Given a hyperbola and a point M, prove that a line passing through M and intersecting the hyperbola at two points with M as their midpoint has a specific equation. -/
theorem line_equation_through_midpoint_on_hyperbola (x y : ℝ → ℝ) (A B M : ℝ × ℝ) :
  (∀ t : ℝ, (x t)^2 - (y t)^2 / 2 = 1) →  -- Hyperbola equation
  M = (2, 1) →  -- Coordinates of point M
  (∃ t₁ t₂ : ℝ, A = (x t₁, y t₁) ∧ B = (x t₂, y t₂)) →  -- A and B are on the hyperbola
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  ∃ k b : ℝ, k = 4 ∧ b = -7 ∧ ∀ x y : ℝ, y = k * x + b ↔ 4 * x - y - 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_on_hyperbola_l3077_307709


namespace NUMINAMATH_CALUDE_f_shifted_up_is_g_l3077_307792

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the shifted function g
def g : ℝ → ℝ := sorry

-- Theorem stating that g is f shifted up by 1
theorem f_shifted_up_is_g : ∀ x : ℝ, g x = f x + 1 := by sorry

end NUMINAMATH_CALUDE_f_shifted_up_is_g_l3077_307792


namespace NUMINAMATH_CALUDE_equation_has_solution_equation_has_unique_solution_l3077_307755

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (Real.log x / Real.log a) / (Real.log 2 / Real.log a) +
  (Real.log (2*a - x) / Real.log x) / (Real.log 2 / Real.log x) =
  1 / (Real.log 2 / Real.log (a^2 - 1))

-- Theorem for the first question
theorem equation_has_solution (a : ℝ) :
  (∃ x, equation a x) ↔ (a > 1 ∧ a ≠ Real.sqrt 2) :=
sorry

-- Theorem for the second question
theorem equation_has_unique_solution (a : ℝ) :
  (∃! x, equation a x) ↔ a = 2 :=
sorry

end NUMINAMATH_CALUDE_equation_has_solution_equation_has_unique_solution_l3077_307755


namespace NUMINAMATH_CALUDE_range_of_t_l3077_307767

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  let t := a*b - a^2 - b^2
  ∀ x, (∃ a b : ℝ, a^2 + a*b + b^2 = 1 ∧ t = a*b - a^2 - b^2) → -3 ≤ x ∧ x ≤ -1/3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_t_l3077_307767


namespace NUMINAMATH_CALUDE_clock_rings_seven_times_l3077_307715

/-- Calculates the number of rings for a clock with given interval and day length -/
def number_of_rings (interval : ℕ) (day_length : ℕ) : ℕ :=
  (day_length / interval) + 1

/-- Theorem: A clock ringing every 4 hours in a 24-hour day rings 7 times -/
theorem clock_rings_seven_times : number_of_rings 4 24 = 7 := by
  sorry

end NUMINAMATH_CALUDE_clock_rings_seven_times_l3077_307715


namespace NUMINAMATH_CALUDE_color_guard_row_length_l3077_307775

theorem color_guard_row_length 
  (num_students : ℕ) 
  (student_space : ℝ) 
  (gap_space : ℝ) 
  (h1 : num_students = 40)
  (h2 : student_space = 0.4)
  (h3 : gap_space = 0.5) : 
  (num_students : ℝ) * student_space + (num_students - 1 : ℝ) * gap_space = 35.5 :=
by sorry

end NUMINAMATH_CALUDE_color_guard_row_length_l3077_307775


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3077_307768

open Real

noncomputable def x (t : ℝ) : ℝ := exp (2 * t) * (-2 * cos t + sin t) + 2

noncomputable def y (t : ℝ) : ℝ := exp (2 * t) * (-cos t + 3 * sin t) + 3

theorem solution_satisfies_system :
  (∀ t, deriv x t = x t + y t - 3) ∧
  (∀ t, deriv y t = -2 * x t + 3 * y t + 1) ∧
  x 0 = 0 ∧
  y 0 = 0 :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3077_307768


namespace NUMINAMATH_CALUDE_equal_digit_prob_is_three_eighths_l3077_307736

/-- Represents a die with a given number of sides -/
structure Die :=
  (sides : ℕ)

/-- Probability of rolling a one-digit number on a given die -/
def prob_one_digit (d : Die) : ℚ :=
  if d.sides ≤ 9 then 1 else (9 : ℚ) / d.sides

/-- Probability of rolling a two-digit number on a given die -/
def prob_two_digit (d : Die) : ℚ :=
  1 - prob_one_digit d

/-- The set of dice used in the game -/
def game_dice : List Die :=
  [⟨6⟩, ⟨6⟩, ⟨6⟩, ⟨12⟩, ⟨12⟩]

/-- The probability of having an equal number of dice showing two-digit and one-digit numbers -/
def equal_digit_prob : ℚ :=
  2 * (prob_two_digit ⟨12⟩ * prob_one_digit ⟨12⟩)

theorem equal_digit_prob_is_three_eighths :
  equal_digit_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_prob_is_three_eighths_l3077_307736


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l3077_307721

/-- The radius of the inscribed circle in an isosceles triangle --/
theorem inscribed_circle_radius_isosceles_triangle 
  (A B C : EuclideanSpace ℝ (Fin 2)) 
  (h_isosceles : dist A B = dist A C) 
  (h_AB : dist A B = 7)
  (h_BC : dist B C = 6) :
  let s := (dist A B + dist A C + dist B C) / 2
  let area := Real.sqrt (s * (s - dist A B) * (s - dist A C) * (s - dist B C))
  area / s = (3 * Real.sqrt 10) / 5 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_isosceles_triangle_l3077_307721


namespace NUMINAMATH_CALUDE_parallelogram_contains_two_points_from_L_l3077_307713

/-- The set L of points in the coordinate plane -/
def L : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (41*x + 2*y, 59*x + 15*y)}

/-- A parallelogram centered at the origin -/
structure Parallelogram :=
  (a b c d : ℝ × ℝ)
  (center_origin : a + c = (0, 0) ∧ b + d = (0, 0))
  (area : ℝ)

/-- The theorem statement -/
theorem parallelogram_contains_two_points_from_L :
  ∀ P : Parallelogram, P.area = 1990 →
  ∃ p q : ℤ × ℤ, p ∈ L ∧ q ∈ L ∧ p ≠ q ∧ 
  (↑p.1, ↑p.2) ∈ {x : ℝ × ℝ | ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧ x = t • P.a + s • P.b} ∧
  (↑q.1, ↑q.2) ∈ {x : ℝ × ℝ | ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧ x = t • P.a + s • P.b} :=
sorry

end NUMINAMATH_CALUDE_parallelogram_contains_two_points_from_L_l3077_307713


namespace NUMINAMATH_CALUDE_tenth_number_value_l3077_307793

def known_numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

theorem tenth_number_value (x : ℕ) :
  (known_numbers.sum + x) / 10 = 750 →
  x = 1555 := by
  sorry

end NUMINAMATH_CALUDE_tenth_number_value_l3077_307793


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3077_307734

theorem trig_expression_equals_one : 
  let tan_30 : ℝ := 1 / Real.sqrt 3
  let sin_30 : ℝ := 1 / 2
  (tan_30^2 - sin_30^2) / (tan_30^2 * sin_30^2) = 1 := by sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3077_307734


namespace NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two_l3077_307750

theorem cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two :
  Real.cos (-17/4 * Real.pi) - Real.sin (-17/4 * Real.pi) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_equals_sqrt_two_l3077_307750


namespace NUMINAMATH_CALUDE_seashells_given_to_joan_l3077_307797

/-- Given that Sam initially found 35 seashells and now has 17 seashells,
    prove that the number of seashells he gave to Joan is 18. -/
theorem seashells_given_to_joan 
  (initial_seashells : ℕ) 
  (current_seashells : ℕ) 
  (h1 : initial_seashells = 35) 
  (h2 : current_seashells = 17) : 
  initial_seashells - current_seashells = 18 := by
sorry

end NUMINAMATH_CALUDE_seashells_given_to_joan_l3077_307797


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l3077_307776

theorem smallest_x_for_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
  (⌊x^2⌋ : ℝ) - x * (⌊x⌋ : ℝ) = 10 ∧ 
  (∀ y : ℝ, y > 0 ∧ (⌊y^2⌋ : ℝ) - y * (⌊y⌋ : ℝ) = 10 → y ≥ x) ∧
  x = 131 / 11 := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l3077_307776


namespace NUMINAMATH_CALUDE_intersection_in_second_quadrant_l3077_307751

theorem intersection_in_second_quadrant (k : ℝ) :
  (∃ x y : ℝ, k * x - y = k - 1 ∧ k * y = x + 2 * k ∧ x < 0 ∧ y > 0) ↔ 0 < k ∧ k < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_in_second_quadrant_l3077_307751


namespace NUMINAMATH_CALUDE_star_commutative_iff_three_lines_l3077_307756

/-- The ⋆ operation -/
def star (a b : ℝ) : ℝ := a^2 * b - 2 * a * b^2

/-- The set of points (x, y) where x ⋆ y = y ⋆ x -/
def star_commutative_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x = y -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2}

theorem star_commutative_iff_three_lines :
  star_commutative_set = three_lines := by sorry

end NUMINAMATH_CALUDE_star_commutative_iff_three_lines_l3077_307756


namespace NUMINAMATH_CALUDE_smallest_with_12_divisors_l3077_307725

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has_12_divisors (n : ℕ+) : Prop := num_divisors n = 12

theorem smallest_with_12_divisors :
  ∃ (n : ℕ+), has_12_divisors n ∧ ∀ (m : ℕ+), has_12_divisors m → n ≤ m := by
  use 288
  sorry

end NUMINAMATH_CALUDE_smallest_with_12_divisors_l3077_307725


namespace NUMINAMATH_CALUDE_chain_store_max_profit_l3077_307752

/-- Annual profit function for a chain store -/
def L (x a : ℝ) : ℝ := (x - 4 - a) * (10 - x)^2

/-- Maximum annual profit for the chain store -/
theorem chain_store_max_profit (a : ℝ) (ha : 1 ≤ a ∧ a ≤ 3) :
  ∃ (L_max : ℝ),
    (∀ x, 7 ≤ x → x ≤ 9 → L x a ≤ L_max) ∧
    ((1 ≤ a ∧ a ≤ 3/2 → L_max = 27 - 9*a) ∧
     (3/2 < a ∧ a ≤ 3 → L_max = 4*(2 - a/3)^3)) :=
sorry

end NUMINAMATH_CALUDE_chain_store_max_profit_l3077_307752


namespace NUMINAMATH_CALUDE_max_black_pens_l3077_307735

/-- The maximum number of pens in the basket -/
def max_pens : ℕ := 2500

/-- The probability of selecting two pens of the same color -/
def same_color_prob : ℚ := 1 / 3

/-- The function that calculates the probability of selecting two pens of the same color
    given the number of black pens and total pens -/
def calc_prob (black_pens total_pens : ℕ) : ℚ :=
  let red_pens := total_pens - black_pens
  (black_pens * (black_pens - 1) + red_pens * (red_pens - 1)) / (total_pens * (total_pens - 1))

theorem max_black_pens :
  ∃ (total_pens : ℕ) (black_pens : ℕ),
    total_pens ≤ max_pens ∧
    calc_prob black_pens total_pens = same_color_prob ∧
    black_pens = 1275 ∧
    ∀ (t : ℕ) (b : ℕ),
      t ≤ max_pens →
      calc_prob b t = same_color_prob →
      b ≤ 1275 :=
by sorry

end NUMINAMATH_CALUDE_max_black_pens_l3077_307735


namespace NUMINAMATH_CALUDE_spring_sports_event_probabilities_l3077_307714

def male_volunteers : ℕ := 4
def female_volunteers : ℕ := 3
def team_size : ℕ := 3

def total_volunteers : ℕ := male_volunteers + female_volunteers

theorem spring_sports_event_probabilities :
  let p_at_least_one_female := 1 - (Nat.choose male_volunteers team_size : ℚ) / (Nat.choose total_volunteers team_size : ℚ)
  let p_all_male_given_at_least_one_male := 
    (Nat.choose male_volunteers team_size : ℚ) / 
    ((Nat.choose total_volunteers team_size : ℚ) - (Nat.choose female_volunteers team_size : ℚ))
  p_at_least_one_female = 31 / 35 ∧ 
  p_all_male_given_at_least_one_male = 2 / 17 := by
  sorry

end NUMINAMATH_CALUDE_spring_sports_event_probabilities_l3077_307714


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3077_307759

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 1/2 ∧ x₂ = 1 ∧ 
  2 * x₁^2 - 3 * x₁ + 1 = 0 ∧ 
  2 * x₂^2 - 3 * x₂ + 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3077_307759


namespace NUMINAMATH_CALUDE_water_added_calculation_l3077_307711

def initial_volume : ℝ := 340
def initial_water_percentage : ℝ := 0.80
def initial_kola_percentage : ℝ := 0.06
def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percentage : ℝ := 0.14111111111111112

theorem water_added_calculation (water_added : ℝ) : 
  let initial_sugar_percentage := 1 - initial_water_percentage - initial_kola_percentage
  let initial_sugar := initial_sugar_percentage * initial_volume
  let total_sugar := initial_sugar + added_sugar
  let final_volume := initial_volume + water_added + added_sugar + added_kola
  final_sugar_percentage * final_volume = total_sugar →
  water_added = 10 := by sorry

end NUMINAMATH_CALUDE_water_added_calculation_l3077_307711


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l3077_307716

theorem sum_reciprocals_bound (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (sum : a + b = 1) :
  1/a + 1/b > 4 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l3077_307716


namespace NUMINAMATH_CALUDE_ratio_difference_theorem_l3077_307708

theorem ratio_difference_theorem (x : ℝ) (h1 : x > 0) :
  (2 * x) / (3 * x) = 2 / 3 ∧
  (2 * x + 4) / (3 * x + 4) = 5 / 7 →
  3 * x - 2 * x = 8 :=
by sorry

end NUMINAMATH_CALUDE_ratio_difference_theorem_l3077_307708


namespace NUMINAMATH_CALUDE_noelle_homework_assignments_l3077_307732

/-- The number of homework points Noelle needs to earn -/
def total_points : ℕ := 30

/-- The number of points for which one assignment is required per point -/
def first_tier_points : ℕ := 5

/-- The number of points for which two assignments are required per point -/
def second_tier_points : ℕ := 10

/-- The number of assignments required for each point in the first tier -/
def first_tier_assignments_per_point : ℕ := 1

/-- The number of assignments required for each point in the second tier -/
def second_tier_assignments_per_point : ℕ := 2

/-- The number of assignments required for each point after the first and second tiers -/
def third_tier_assignments_per_point : ℕ := 3

/-- The total number of assignments Noelle needs to complete -/
def total_assignments : ℕ := 
  first_tier_points * first_tier_assignments_per_point +
  second_tier_points * second_tier_assignments_per_point +
  (total_points - first_tier_points - second_tier_points) * third_tier_assignments_per_point

theorem noelle_homework_assignments : total_assignments = 70 := by
  sorry

end NUMINAMATH_CALUDE_noelle_homework_assignments_l3077_307732


namespace NUMINAMATH_CALUDE_certain_number_problem_l3077_307765

theorem certain_number_problem (n x : ℝ) (h1 : 4 / (n + 3 / x) = 1) (h2 : x = 1) : n = 1 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3077_307765


namespace NUMINAMATH_CALUDE_empty_proper_subset_singleton_zero_l3077_307730

theorem empty_proper_subset_singleton_zero :
  ∅ ⊂ ({0} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_empty_proper_subset_singleton_zero_l3077_307730


namespace NUMINAMATH_CALUDE_jerry_age_l3077_307722

/-- Given that Mickey's age is 18 and Mickey's age is 2 years less than 400% of Jerry's age,
    prove that Jerry's age is 5. -/
theorem jerry_age (mickey_age jerry_age : ℕ) 
  (h1 : mickey_age = 18)
  (h2 : mickey_age = 4 * jerry_age - 2) : 
  jerry_age = 5 := by
sorry

end NUMINAMATH_CALUDE_jerry_age_l3077_307722


namespace NUMINAMATH_CALUDE_eggs_taken_away_l3077_307723

/-- Proof that the number of eggs Amy took away is the difference between Virginia's initial and final number of eggs -/
theorem eggs_taken_away (initial_eggs final_eggs : ℕ) (h1 : initial_eggs = 96) (h2 : final_eggs = 93) :
  initial_eggs - final_eggs = 3 := by
  sorry

end NUMINAMATH_CALUDE_eggs_taken_away_l3077_307723


namespace NUMINAMATH_CALUDE_replaced_girl_weight_l3077_307769

/-- Given a group of girls where replacing one with a heavier girl increases the average weight, 
    this theorem proves the weight of the replaced girl. -/
theorem replaced_girl_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (new_girl_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : n = 10)
  (h2 : new_girl_weight = 100)
  (h3 : average_increase = 5) :
  initial_average * n + new_girl_weight - (initial_average * n + n * average_increase) = 50 :=
by
  sorry

#check replaced_girl_weight

end NUMINAMATH_CALUDE_replaced_girl_weight_l3077_307769


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_29_4_l3077_307703

theorem floor_plus_self_eq_29_4 (x : ℚ) :
  (⌊x⌋ : ℚ) + x = 29/4 → x = 29/4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_29_4_l3077_307703


namespace NUMINAMATH_CALUDE_cigar_purchase_problem_l3077_307717

theorem cigar_purchase_problem :
  ∃ (x y z : ℕ),
    x + y + z = 100 ∧
    (1/2 : ℚ) * x + 3 * y + 10 * z = 100 ∧
    x = 94 ∧ y = 1 ∧ z = 5 := by
  sorry

end NUMINAMATH_CALUDE_cigar_purchase_problem_l3077_307717


namespace NUMINAMATH_CALUDE_opposite_reciprocal_problem_l3077_307720

theorem opposite_reciprocal_problem (a b c d m : ℤ) : 
  (a = -b) →  -- a and b are opposite numbers
  (c * d = 1) →  -- c and d are reciprocals
  (m = -1) →  -- m is the largest negative integer
  c * d - a - b + m ^ 2022 = 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_problem_l3077_307720


namespace NUMINAMATH_CALUDE_ones_digit_of_3_to_53_l3077_307741

theorem ones_digit_of_3_to_53 : (3^53 : ℕ) % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_3_to_53_l3077_307741


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3077_307707

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3077_307707


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3077_307773

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3077_307773


namespace NUMINAMATH_CALUDE_playground_total_l3077_307798

/-- The number of children on the playground at recess -/
def total_children (soccer_boys soccer_girls swings_boys swings_girls snacks_boys snacks_girls : ℕ) : ℕ :=
  soccer_boys + soccer_girls + swings_boys + swings_girls + snacks_boys + snacks_girls

/-- Theorem stating the total number of children on the playground -/
theorem playground_total :
  total_children 27 35 15 20 10 5 = 112 := by
  sorry

end NUMINAMATH_CALUDE_playground_total_l3077_307798


namespace NUMINAMATH_CALUDE_number_of_valid_paths_l3077_307766

-- Define the grid dimensions
def columns : ℕ := 10
def rows : ℕ := 4

-- Define the forbidden segment
def forbidden_column : ℕ := 6
def forbidden_row_start : ℕ := 2
def forbidden_row_end : ℕ := 3

-- Define the total number of steps
def total_steps : ℕ := columns + rows

-- Function to calculate binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Function to calculate the number of paths between two points
def paths_between (col_diff row_diff : ℕ) : ℕ := 
  binomial (col_diff + row_diff) row_diff

-- Theorem statement
theorem number_of_valid_paths : 
  paths_between columns rows - 
  (paths_between forbidden_column (rows - forbidden_row_end) * 
   paths_between (columns - forbidden_column) (forbidden_row_end)) = 861 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_paths_l3077_307766


namespace NUMINAMATH_CALUDE_solution_set_implies_a_eq_one_solution_set_varies_with_a_l3077_307740

/-- The quadratic function f(x) = ax^2 + (1-2a)x - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (1 - 2*a) * x - 2

/-- The solution set of f(x) > 0 when a = 1 -/
def solution_set_a1 : Set ℝ := {x | x < -1 ∨ x > 2}

/-- Theorem: When the solution set of f(x) > 0 is {x | x < -1 or x > 2}, a = 1 -/
theorem solution_set_implies_a_eq_one :
  (∀ x, f 1 x > 0 ↔ x ∈ solution_set_a1) → 1 = 1 := by sorry

/-- The solution set of f(x) > 0 for a > 0 -/
def solution_set_a_pos (a : ℝ) : Set ℝ := {x | x < -1/a ∨ x > 2}

/-- The solution set of f(x) > 0 for a = 0 -/
def solution_set_a_zero : Set ℝ := {x | x > 2}

/-- The solution set of f(x) > 0 for -1/2 < a < 0 -/
def solution_set_a_neg_small (a : ℝ) : Set ℝ := {x | 2 < x ∧ x < -1/a}

/-- The solution set of f(x) > 0 for a = -1/2 -/
def solution_set_a_neg_half : Set ℝ := ∅

/-- The solution set of f(x) > 0 for a < -1/2 -/
def solution_set_a_neg_large (a : ℝ) : Set ℝ := {x | -1/a < x ∧ x < 2}

/-- Theorem: The solution set of f(x) > 0 varies for different ranges of a ∈ ℝ -/
theorem solution_set_varies_with_a (a : ℝ) :
  (∀ x, f a x > 0 ↔ 
    (a > 0 ∧ x ∈ solution_set_a_pos a) ∨
    (a = 0 ∧ x ∈ solution_set_a_zero) ∨
    (-1/2 < a ∧ a < 0 ∧ x ∈ solution_set_a_neg_small a) ∨
    (a = -1/2 ∧ x ∈ solution_set_a_neg_half) ∨
    (a < -1/2 ∧ x ∈ solution_set_a_neg_large a)) := by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_eq_one_solution_set_varies_with_a_l3077_307740


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3077_307758

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -4) (h2 : b = 1/2) :
  b * (a + b) + (-a + b) * (-a - b) - a^2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3077_307758


namespace NUMINAMATH_CALUDE_equation_solution_range_l3077_307788

theorem equation_solution_range (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k / (2 * x - 4) - 1 = x / (x - 2)) → 
  (k > -4 ∧ k ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l3077_307788


namespace NUMINAMATH_CALUDE_expression_evaluation_l3077_307772

theorem expression_evaluation : (1/8)^(1/3) - Real.log 2 / Real.log 3 * Real.log 27 / Real.log 4 + 2018^0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3077_307772


namespace NUMINAMATH_CALUDE_area_bounded_by_curves_l3077_307739

/-- The area of the region bounded by x = √(e^y - 1), x = 0, and y = ln 2 -/
theorem area_bounded_by_curves : ∃ (S : ℝ),
  (∀ x y : ℝ, x = Real.sqrt (Real.exp y - 1) → 
    0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ Real.log 2) →
  S = ∫ x in (0)..(1), (Real.log 2 - Real.log (x^2 + 1)) →
  S = 2 - π / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_curves_l3077_307739


namespace NUMINAMATH_CALUDE_circle_equation_radius_five_l3077_307794

/-- A circle equation in the form x^2 + 8x + y^2 + 4y - k = 0 -/
def CircleEquation (x y k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

/-- The standard form of a circle equation with center (h, j) and radius r -/
def StandardCircleEquation (x y h j r : ℝ) : Prop :=
  (x - h)^2 + (y - j)^2 = r^2

theorem circle_equation_radius_five (k : ℝ) :
  (∀ x y, CircleEquation x y k ↔ StandardCircleEquation x y (-4) (-2) 5) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_radius_five_l3077_307794


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3077_307770

theorem quadratic_equation_properties (m : ℝ) :
  let f := fun x => m * x^2 - 4 * x + 1
  (∃ x : ℝ, f x = 0) →
  (f 1 = 0 → m = 3) ∧
  (m ≠ 0 → m ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3077_307770


namespace NUMINAMATH_CALUDE_percentage_of_women_in_survey_l3077_307789

theorem percentage_of_women_in_survey (mothers_full_time : Real) 
  (fathers_full_time : Real) (total_not_full_time : Real) :
  mothers_full_time = 5/6 →
  fathers_full_time = 3/4 →
  total_not_full_time = 1/5 →
  ∃ (w : Real), w = 3/5 ∧ 
    w * (1 - mothers_full_time) + (1 - w) * (1 - fathers_full_time) = total_not_full_time :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_survey_l3077_307789


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l3077_307787

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  2 * x + 3 * y = 8 ∧ x = y - 1

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  2 * x - y = -1 ∧ x + 3 * y = 17

-- Theorem for the first system
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 1 ∧ y = 2 := by
  sorry

-- Theorem for the second system
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l3077_307787


namespace NUMINAMATH_CALUDE_min_value_theorem_l3077_307771

/-- The function f(x) = |x + a| + |x - b| -/
def f (a b x : ℝ) : ℝ := |x + a| + |x - b|

/-- The theorem statement -/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f a b x ≥ 4) (hmin_exists : ∃ x, f a b x = 4) :
  (a + b = 4) ∧ 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 4 → 1/4 * x^2 + 1/9 * y^2 ≥ 16/13) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 4 ∧ 1/4 * x^2 + 1/9 * y^2 = 16/13) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3077_307771


namespace NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_and_am_gm_inequality_l3077_307760

theorem xy_positive_iff_fraction_positive_and_am_gm_inequality :
  (∀ x y : ℝ, x * y > 0 ↔ x / y > 0) ∧
  (∀ a b : ℝ, a * b ≤ ((a + b) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_xy_positive_iff_fraction_positive_and_am_gm_inequality_l3077_307760


namespace NUMINAMATH_CALUDE_binary_sum_equals_669_l3077_307749

/-- Represents a binary number as a list of booleans, where true represents 1 and false represents 0 -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binary_to_decimal (b : BinaryNumber) : ℕ :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- The binary number 111111111₂ -/
def b1 : BinaryNumber := [true, true, true, true, true, true, true, true, true]

/-- The binary number 1111111₂ -/
def b2 : BinaryNumber := [true, true, true, true, true, true, true]

/-- The binary number 11111₂ -/
def b3 : BinaryNumber := [true, true, true, true, true]

theorem binary_sum_equals_669 :
  binary_to_decimal b1 + binary_to_decimal b2 + binary_to_decimal b3 = 669 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_669_l3077_307749


namespace NUMINAMATH_CALUDE_fishing_loss_fraction_l3077_307747

theorem fishing_loss_fraction (jordan_catch : ℕ) (perry_catch : ℕ) (remaining : ℕ) : 
  jordan_catch = 4 →
  perry_catch = 2 * jordan_catch →
  remaining = 9 →
  (jordan_catch + perry_catch - remaining : ℚ) / (jordan_catch + perry_catch) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_fishing_loss_fraction_l3077_307747


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l3077_307746

def chris_current_money : ℕ := 279
def grandmother_gift : ℕ := 25
def aunt_uncle_gift : ℕ := 20
def parents_gift : ℕ := 75

def total_birthday_gifts : ℕ := grandmother_gift + aunt_uncle_gift + parents_gift

theorem chris_money_before_birthday :
  chris_current_money - total_birthday_gifts = 159 := by
  sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l3077_307746
