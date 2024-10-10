import Mathlib

namespace intersection_of_logarithmic_functions_l793_79381

theorem intersection_of_logarithmic_functions :
  ∃! x : ℝ, x > 0 ∧ 3 * Real.log x = Real.log (3 * (x - 1)) :=
by sorry

end intersection_of_logarithmic_functions_l793_79381


namespace binomial_expansion_property_l793_79341

/-- Given a positive integer n, if the sum of the binomial coefficients of the first three terms
    in the expansion of (1/2 + 2x)^n equals 79, then n = 12 and the 11th term has the largest coefficient -/
theorem binomial_expansion_property (n : ℕ) (hn : n > 0) 
  (h_sum : Nat.choose n 0 + Nat.choose n 1 + Nat.choose n 2 = 79) : 
  n = 12 ∧ ∀ k, 0 ≤ k ∧ k ≤ 12 → 
    Nat.choose 12 10 * 4^10 ≥ Nat.choose 12 k * 4^k := by
  sorry


end binomial_expansion_property_l793_79341


namespace glenn_total_spent_l793_79342

/-- The cost of a movie ticket on Monday -/
def monday_price : ℚ := 5

/-- The cost of a movie ticket on Wednesday -/
def wednesday_price : ℚ := 2 * monday_price

/-- The cost of a movie ticket on Saturday -/
def saturday_price : ℚ := 5 * monday_price

/-- The discount rate for Wednesday -/
def discount_rate : ℚ := 1 / 10

/-- The cost of popcorn and drink on Saturday -/
def popcorn_drink_cost : ℚ := 7

/-- The total amount Glenn spends -/
def total_spent : ℚ := wednesday_price * (1 - discount_rate) + saturday_price + popcorn_drink_cost

theorem glenn_total_spent : total_spent = 41 := by
  sorry

end glenn_total_spent_l793_79342


namespace kalebs_savings_l793_79378

/-- Kaleb's initial savings problem -/
theorem kalebs_savings : ∀ (x : ℕ), 
  (x + 25 = 8 * 8) → x = 39 := by sorry

end kalebs_savings_l793_79378


namespace hypotenuse_increase_bound_l793_79337

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  let c := Real.sqrt (x^2 + y^2)
  let c' := Real.sqrt ((x+1)^2 + (y+1)^2)
  c' - c ≤ Real.sqrt 2 := by
sorry

end hypotenuse_increase_bound_l793_79337


namespace reciprocal_of_one_twentieth_l793_79359

theorem reciprocal_of_one_twentieth (x : ℚ) : x = 1 / 20 → 1 / x = 20 := by
  sorry

end reciprocal_of_one_twentieth_l793_79359


namespace water_percentage_fresh_is_75_percent_l793_79372

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 75

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 25

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 200

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 66.67

/-- Theorem stating that the percentage of water in fresh grapes is 75% -/
theorem water_percentage_fresh_is_75_percent :
  water_percentage_fresh = 75 := by sorry

end water_percentage_fresh_is_75_percent_l793_79372


namespace total_passengers_four_trips_l793_79357

/-- Calculates the total number of passengers transported in multiple round trips -/
def total_passengers (passengers_one_way : ℕ) (passengers_return : ℕ) (num_round_trips : ℕ) : ℕ :=
  (passengers_one_way + passengers_return) * num_round_trips

/-- Theorem stating that the total number of passengers transported in 4 round trips is 640 -/
theorem total_passengers_four_trips :
  total_passengers 100 60 4 = 640 := by
  sorry

end total_passengers_four_trips_l793_79357


namespace election_expectation_l793_79306

/-- The number of voters and candidates in the election -/
def n : ℕ := 5

/-- The probability of a candidate receiving no votes -/
def p_no_votes : ℚ := (4/5)^n

/-- The probability of a candidate receiving at least one vote -/
def p_at_least_one_vote : ℚ := 1 - p_no_votes

/-- The expected number of candidates receiving at least one vote -/
def expected_candidates_with_votes : ℚ := n * p_at_least_one_vote

theorem election_expectation :
  expected_candidates_with_votes = 2101/625 :=
by sorry

end election_expectation_l793_79306


namespace max_profit_at_upper_bound_l793_79376

/-- Represents the profit function for a product given its cost price, initial selling price,
    initial daily sales, and the rate of sales decrease per yuan increase in price. -/
def profit_function (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) (sales_decrease_rate : ℝ) (x : ℝ) : ℝ :=
  (x - cost_price) * (initial_sales - (x - initial_price) * sales_decrease_rate)

/-- Theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_at_upper_bound (cost_price : ℝ) (initial_price : ℝ) (initial_sales : ℝ) 
    (sales_decrease_rate : ℝ) (lower_bound : ℝ) (upper_bound : ℝ) :
    cost_price = 30 →
    initial_price = 40 →
    initial_sales = 600 →
    sales_decrease_rate = 10 →
    lower_bound = 40 →
    upper_bound = 60 →
    (∀ x, lower_bound ≤ x ∧ x ≤ upper_bound →
      profit_function cost_price initial_price initial_sales sales_decrease_rate x ≤
      profit_function cost_price initial_price initial_sales sales_decrease_rate upper_bound) ∧
    profit_function cost_price initial_price initial_sales sales_decrease_rate upper_bound = 12000 :=
  sorry

#check max_profit_at_upper_bound

end max_profit_at_upper_bound_l793_79376


namespace number_puzzle_l793_79393

theorem number_puzzle : ∃ x : ℝ, 22 * (x - 36) = 748 ∧ x = 70 := by
  sorry

end number_puzzle_l793_79393


namespace round_trip_average_speed_l793_79377

/-- Calculates the average speed of a round trip given the specified conditions -/
theorem round_trip_average_speed
  (outbound_distance : ℝ)
  (outbound_time : ℝ)
  (return_distance : ℝ)
  (return_speed : ℝ)
  (h1 : outbound_distance = 5)
  (h2 : outbound_time = 1)
  (h3 : return_distance = outbound_distance)
  (h4 : return_speed = 20)
  : (outbound_distance + return_distance) / (outbound_time + return_distance / return_speed) = 8 := by
  sorry

#check round_trip_average_speed

end round_trip_average_speed_l793_79377


namespace quadratic_roots_expression_l793_79375

theorem quadratic_roots_expression (x₁ x₂ : ℝ) : 
  x₁^2 + 5*x₁ + 1 = 0 →
  x₂^2 + 5*x₂ + 1 = 0 →
  (x₁*Real.sqrt 6 / (1 + x₂))^2 + (x₂*Real.sqrt 6 / (1 + x₁))^2 = 220 := by
sorry

end quadratic_roots_expression_l793_79375


namespace rectangle_C_in_position_I_l793_79388

-- Define the Rectangle type
structure Rectangle where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the five rectangles
def A : Rectangle := ⟨1, 4, 3, 1⟩
def B : Rectangle := ⟨4, 3, 1, 2⟩
def C : Rectangle := ⟨1, 1, 2, 4⟩
def D : Rectangle := ⟨3, 2, 2, 3⟩
def E : Rectangle := ⟨4, 4, 1, 1⟩

-- Define a function to check if two rectangles can be placed side by side
def canPlaceSideBySide (r1 r2 : Rectangle) : Bool :=
  r1.right = r2.left

-- Define a function to check if two rectangles can be placed top to bottom
def canPlaceTopToBottom (r1 r2 : Rectangle) : Bool :=
  r1.bottom = r2.top

-- Theorem: Rectangle C is the only one that can be placed in position I
theorem rectangle_C_in_position_I :
  ∃! r : Rectangle, r = C ∧ 
  (∃ r2 r3 : Rectangle, r2 ≠ r ∧ r3 ≠ r ∧ r2 ≠ r3 ∧
   canPlaceSideBySide r r2 ∧ canPlaceSideBySide r2 r3 ∧
   (∃ r4 : Rectangle, r4 ≠ r ∧ r4 ≠ r2 ∧ r4 ≠ r3 ∧
    canPlaceTopToBottom r r4 ∧
    (∃ r5 : Rectangle, r5 ≠ r ∧ r5 ≠ r2 ∧ r5 ≠ r3 ∧ r5 ≠ r4 ∧
     canPlaceTopToBottom r2 r5 ∧ canPlaceSideBySide r4 r5))) :=
sorry

end rectangle_C_in_position_I_l793_79388


namespace parabola_focus_directrix_distance_l793_79323

/-- For a parabola with equation y = 4x^2, the distance from the focus to the directrix is 1/8 -/
theorem parabola_focus_directrix_distance :
  let parabola := fun (x : ℝ) => 4 * x^2
  ∃ (focus : ℝ × ℝ) (directrix : ℝ → ℝ),
    (∀ x, parabola x = (x - focus.1)^2 / (4 * (focus.2 - directrix 0))) ∧
    (focus.2 - directrix 0 = 1/8) :=
by sorry

end parabola_focus_directrix_distance_l793_79323


namespace contrapositive_of_p_is_true_l793_79395

theorem contrapositive_of_p_is_true :
  (∀ x : ℝ, x^2 - 2*x - 8 ≤ 0 → x ≥ -3) := by sorry

end contrapositive_of_p_is_true_l793_79395


namespace possible_teams_count_l793_79398

/-- Represents the number of players in each position in the squad --/
structure SquadComposition :=
  (goalkeepers : Nat)
  (defenders : Nat)
  (midfielders : Nat)
  (strikers : Nat)

/-- Represents the required composition of a team --/
structure TeamComposition :=
  (goalkeepers : Nat)
  (defenders : Nat)
  (midfielders : Nat)
  (strikers : Nat)

/-- Function to calculate the number of possible teams --/
def calculatePossibleTeams (squad : SquadComposition) (team : TeamComposition) : Nat :=
  let goalkeeperChoices := Nat.choose squad.goalkeepers team.goalkeepers
  let strikerChoices := Nat.choose squad.strikers team.strikers
  let midfielderChoices := Nat.choose squad.midfielders team.midfielders
  let defenderChoices := Nat.choose (squad.defenders + (squad.midfielders - team.midfielders)) team.defenders
  goalkeeperChoices * strikerChoices * midfielderChoices * defenderChoices

/-- Theorem stating the number of possible teams --/
theorem possible_teams_count (squad : SquadComposition) (team : TeamComposition) :
  squad.goalkeepers = 3 →
  squad.defenders = 5 →
  squad.midfielders = 5 →
  squad.strikers = 5 →
  team.goalkeepers = 1 →
  team.defenders = 4 →
  team.midfielders = 4 →
  team.strikers = 2 →
  calculatePossibleTeams squad team = 2250 := by
  sorry

end possible_teams_count_l793_79398


namespace smallest_positive_integer_congruence_l793_79310

theorem smallest_positive_integer_congruence :
  ∃ (y : ℕ), y > 0 ∧ y + 3721 ≡ 803 [ZMOD 17] ∧
  ∀ (z : ℕ), z > 0 ∧ z + 3721 ≡ 803 [ZMOD 17] → y ≤ z ∧ y = 14 :=
by sorry

end smallest_positive_integer_congruence_l793_79310


namespace area_between_curves_l793_79379

-- Define the two functions
def f (x : ℝ) := 3 * x
def g (x : ℝ) := x^2

-- Define the intersection points
def x₁ : ℝ := 0
def x₂ : ℝ := 3

-- State the theorem
theorem area_between_curves :
  ∫ x in x₁..x₂, (f x - g x) = 9/2 := by sorry

end area_between_curves_l793_79379


namespace point_on_line_l793_79336

/-- Given two points (m, n) and (m + p, n + 18) on the line x = (y / 6) - (2 / 5), prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 6 - 2 / 5) → 
  (m + p = (n + 18) / 6 - 2 / 5) → 
  p = 3 := by
  sorry

end point_on_line_l793_79336


namespace partial_fraction_decomposition_product_l793_79363

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - 25) / ((x - 2) * (x + 3) * (x - 4)) = 
    A / (x - 2) + B / (x + 3) + C / (x - 4) → 
  A * B * C = 1 := by
sorry

end partial_fraction_decomposition_product_l793_79363


namespace cube_edge_sum_l793_79322

/-- The number of edges in a cube -/
def cube_edges : ℕ := 12

/-- The length of one edge of the cube in centimeters -/
def edge_length : ℝ := 15

/-- The sum of the lengths of all edges of the cube in centimeters -/
def sum_of_edges : ℝ := cube_edges * edge_length

theorem cube_edge_sum :
  sum_of_edges = 180 := by sorry

end cube_edge_sum_l793_79322


namespace zach_rental_cost_l793_79339

/-- Calculates the total cost of renting a car given the base cost, per-mile cost, and miles driven -/
def total_rental_cost (base_cost : ℝ) (per_mile_cost : ℝ) (miles_monday : ℝ) (miles_thursday : ℝ) : ℝ :=
  base_cost + per_mile_cost * (miles_monday + miles_thursday)

/-- Theorem: Given the rental conditions, Zach's total cost is $832 -/
theorem zach_rental_cost :
  total_rental_cost 150 0.5 620 744 = 832 := by
  sorry

#eval total_rental_cost 150 0.5 620 744

end zach_rental_cost_l793_79339


namespace negation_of_forall_not_prime_l793_79361

theorem negation_of_forall_not_prime :
  (¬ (∀ n : ℕ, ¬ (Nat.Prime (2^n - 2)))) ↔ (∃ n : ℕ, Nat.Prime (2^n - 2)) := by
  sorry

end negation_of_forall_not_prime_l793_79361


namespace volume_specific_pyramid_l793_79360

/-- A triangular pyramid with specific edge lengths -/
structure TriangularPyramid where
  edge_opposite1 : ℝ
  edge_opposite2 : ℝ
  edge_other : ℝ

/-- Volume of a triangular pyramid -/
def volume (p : TriangularPyramid) : ℝ := sorry

/-- Theorem: The volume of the specific triangular pyramid is 24 cm³ -/
theorem volume_specific_pyramid :
  let p : TriangularPyramid := {
    edge_opposite1 := 4,
    edge_opposite2 := 12,
    edge_other := 7
  }
  volume p = 24 := by sorry

end volume_specific_pyramid_l793_79360


namespace simplify_expression_l793_79329

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) (hx2 : x ≠ 2) :
  (x - 2) / (x^2) / (1 - 2/x) = 1/x := by
  sorry

end simplify_expression_l793_79329


namespace range_of_a_l793_79328

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |x + a| ≤ 2) → a ∈ Set.Icc (-3) 0 := by
  sorry

end range_of_a_l793_79328


namespace intersecting_chords_theorem_l793_79324

/-- A type representing a point on a circle -/
def CirclePoint : Type := Nat

/-- The total number of points on the circle -/
def total_points : Nat := 2023

/-- A type representing a selection of 6 distinct points -/
def Sextuple : Type := Fin 6 → CirclePoint

/-- Predicate to check if two chords intersect -/
def chords_intersect (a b c d : CirclePoint) : Prop := sorry

/-- The probability of selecting a sextuple where AB intersects both CD and EF -/
def intersecting_chords_probability : ℚ := 1 / 72

theorem intersecting_chords_theorem (s : Sextuple) :
  (∀ i j : Fin 6, i ≠ j → s i ≠ s j) →  -- all points are distinct
  (∀ s : Sextuple, (∀ i j : Fin 6, i ≠ j → s i ≠ s j) → 
    chords_intersect (s 0) (s 1) (s 2) (s 3) ∧ 
    chords_intersect (s 2) (s 3) (s 4) (s 5)) →  -- definition of intersecting chords
  intersecting_chords_probability = 1 / 72 := by sorry

end intersecting_chords_theorem_l793_79324


namespace problem_solution_l793_79338

theorem problem_solution : (-1)^2023 + |Real.sqrt 3 - 3| + Real.sqrt 9 - (-4) * (1/2) = 7 - Real.sqrt 3 := by
  sorry

end problem_solution_l793_79338


namespace swimmers_passing_count_l793_79340

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  turnTime : ℝ

/-- Calculates the number of times two swimmers pass each other --/
def calculatePassings (poolLength : ℝ) (totalTime : ℝ) (swimmerA : Swimmer) (swimmerB : Swimmer) : ℕ :=
  sorry

/-- The main theorem stating the number of times the swimmers pass each other --/
theorem swimmers_passing_count :
  let poolLength : ℝ := 120
  let totalTime : ℝ := 15 * 60  -- 15 minutes in seconds
  let swimmerA : Swimmer := { speed := 4, turnTime := 0 }
  let swimmerB : Swimmer := { speed := 3, turnTime := 2 }
  calculatePassings poolLength totalTime swimmerA swimmerB = 51 :=
by sorry

end swimmers_passing_count_l793_79340


namespace inequality_solution_l793_79307

theorem inequality_solution (x : ℝ) : (3 * x - 9) / ((x - 3)^2) < 0 ↔ x < 3 :=
sorry

end inequality_solution_l793_79307


namespace ryan_english_hours_l793_79353

/-- Ryan's daily study schedule -/
structure StudySchedule where
  english_hours : ℕ
  chinese_hours : ℕ
  chinese_more_than_english : chinese_hours = english_hours + 1

/-- Ryan's actual study schedule -/
def ryans_schedule : StudySchedule where
  english_hours := 6
  chinese_hours := 7
  chinese_more_than_english := by rfl

theorem ryan_english_hours :
  ryans_schedule.english_hours = 6 :=
by sorry

end ryan_english_hours_l793_79353


namespace fraction_sum_equals_2315_over_1200_l793_79304

theorem fraction_sum_equals_2315_over_1200 :
  (1/2 : ℚ) * (3/4) + (5/6) * (7/8) + (9/10) * (11/12) = 2315/1200 := by
  sorry

end fraction_sum_equals_2315_over_1200_l793_79304


namespace quadratic_two_distinct_roots_l793_79335

/-- For a quadratic equation x^2 + mx + k = 0 with real coefficients,
    this function determines if it has two distinct real roots. -/
def has_two_distinct_real_roots (m k : ℝ) : Prop :=
  k > 0 ∧ (m < -2 * Real.sqrt k ∨ m > 2 * Real.sqrt k)

/-- Theorem stating the conditions for a quadratic equation to have two distinct real roots. -/
theorem quadratic_two_distinct_roots (m k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + k = 0 ∧ y^2 + m*y + k = 0) ↔ has_two_distinct_real_roots m k :=
sorry

end quadratic_two_distinct_roots_l793_79335


namespace complex_fraction_evaluation_l793_79334

/-- Prove that the given expression evaluates to 1/5 -/
theorem complex_fraction_evaluation :
  (⌈(19 / 6 : ℚ) - ⌈(34 / 21 : ℚ)⌉⌉ : ℚ) / (⌈(34 / 6 : ℚ) + ⌈(6 * 19 / 34 : ℚ)⌉⌉ : ℚ) = 1 / 5 := by
  sorry

end complex_fraction_evaluation_l793_79334


namespace min_distance_ab_value_l793_79333

theorem min_distance_ab_value (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a^2 - a*b + 1 = 0) (hcd : c^2 + d^2 = 1) :
  let f := fun (x y : ℝ) => (a - x)^2 + (b - y)^2
  ∃ (m : ℝ), (∀ x y, c^2 + d^2 = 1 → f x y ≥ m) ∧ 
             (a * b = Real.sqrt 2 / 2 + 1) := by
  sorry

end min_distance_ab_value_l793_79333


namespace total_collection_is_32_49_l793_79367

/-- Represents the number of members in the group -/
def group_size : ℕ := 57

/-- Represents the contribution of each member in paise -/
def contribution_per_member : ℕ := group_size

/-- Converts paise to rupees -/
def paise_to_rupees (paise : ℕ) : ℚ :=
  (paise : ℚ) / 100

/-- Calculates the total collection amount in rupees -/
def total_collection : ℚ :=
  paise_to_rupees (group_size * contribution_per_member)

/-- Theorem stating that the total collection amount is 32.49 rupees -/
theorem total_collection_is_32_49 :
  total_collection = 32.49 := by sorry

end total_collection_is_32_49_l793_79367


namespace positive_integer_solution_is_perfect_square_l793_79371

theorem positive_integer_solution_is_perfect_square (t : ℤ) (n : ℕ+) 
  (h : n^2 + (4*t - 1)*n + 4*t^2 = 0) : 
  ∃ (k : ℕ), n = k^2 := by
  sorry

end positive_integer_solution_is_perfect_square_l793_79371


namespace distance_ratio_is_two_thirds_l793_79348

/-- Yan's position between home and stadium -/
structure Position where
  distanceToHome : ℝ
  distanceToStadium : ℝ
  isBetween : 0 < distanceToHome ∧ 0 < distanceToStadium

/-- Yan's travel speeds -/
structure Speeds where
  walkingSpeed : ℝ
  bicycleSpeed : ℝ
  bicycleFaster : bicycleSpeed = 5 * walkingSpeed

/-- The theorem stating the ratio of distances -/
theorem distance_ratio_is_two_thirds 
  (pos : Position) (speeds : Speeds) 
  (equalTime : pos.distanceToStadium / speeds.walkingSpeed = 
               pos.distanceToHome / speeds.walkingSpeed + 
               (pos.distanceToHome + pos.distanceToStadium) / speeds.bicycleSpeed) :
  pos.distanceToHome / pos.distanceToStadium = 2 / 3 := by
  sorry


end distance_ratio_is_two_thirds_l793_79348


namespace banquet_guests_l793_79392

theorem banquet_guests (x : ℚ) : 
  (1/3 * x + 3/5 * (2/3 * x) + 4 = x) → x = 15 := by
  sorry

end banquet_guests_l793_79392


namespace tshirt_company_profit_l793_79346

/-- Calculates the daily profit of a t-shirt company given specific conditions -/
theorem tshirt_company_profit 
  (num_employees : ℕ) 
  (shirts_per_employee : ℕ) 
  (hours_per_shift : ℕ) 
  (hourly_wage : ℚ) 
  (per_shirt_bonus : ℚ) 
  (shirt_price : ℚ) 
  (nonemployee_expenses : ℚ) 
  (h1 : num_employees = 20)
  (h2 : shirts_per_employee = 20)
  (h3 : hours_per_shift = 8)
  (h4 : hourly_wage = 12)
  (h5 : per_shirt_bonus = 5)
  (h6 : shirt_price = 35)
  (h7 : nonemployee_expenses = 1000) :
  (num_employees * shirts_per_employee * shirt_price) - 
  (num_employees * (hours_per_shift * hourly_wage + shirts_per_employee * per_shirt_bonus) + nonemployee_expenses) = 9080 := by
  sorry

end tshirt_company_profit_l793_79346


namespace sum_first_ten_primes_with_units_digit_3_l793_79351

def is_prime (n : ℕ) : Prop := sorry

def has_units_digit_3 (n : ℕ) : Prop := n % 10 = 3

def first_ten_primes_with_units_digit_3 : List ℕ := sorry

theorem sum_first_ten_primes_with_units_digit_3 :
  (first_ten_primes_with_units_digit_3.foldl (· + ·) 0) = 793 := by sorry

end sum_first_ten_primes_with_units_digit_3_l793_79351


namespace stock_price_increase_l793_79383

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (percent_increase : ℝ) (h1 : closing_price = 9) 
  (h2 : percent_increase = 12.5) :
  closing_price = opening_price * (1 + percent_increase / 100) →
  opening_price = 8 := by
  sorry

end stock_price_increase_l793_79383


namespace zero_not_in_range_of_g_l793_79368

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then
    ⌈(2 : ℝ) / (x + 3)⌉
  else if x < -3 then
    ⌊(2 : ℝ) / (x + 3)⌋
  else
    0  -- Arbitrary value for x = -3, as g is not defined there

theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
by sorry

end zero_not_in_range_of_g_l793_79368


namespace inscribed_squares_ratio_l793_79349

/-- Given two right triangles and inscribed squares, proves the ratio of their side lengths -/
theorem inscribed_squares_ratio : 
  ∀ (x y : ℝ),
  (∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 ∧ 
    x * (a + b - x) = a * b) →
  (∃ (p q r : ℝ), p = 5 ∧ q = 12 ∧ r = 13 ∧ p^2 + q^2 = r^2 ∧ 
    y * (r - y) = p * q) →
  x / y = 444 / 1183 := by
sorry

end inscribed_squares_ratio_l793_79349


namespace inequality_solution_set_l793_79394

theorem inequality_solution_set (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) ≤ 1) ↔ (x < 1 ∨ x ≥ 2) :=
by sorry

end inequality_solution_set_l793_79394


namespace andrews_hotdogs_l793_79345

theorem andrews_hotdogs (total : ℕ) (cheese_pops : ℕ) (chicken_nuggets : ℕ) 
  (h1 : total = 90)
  (h2 : cheese_pops = 20)
  (h3 : chicken_nuggets = 40)
  (h4 : ∃ hotdogs : ℕ, total = hotdogs + cheese_pops + chicken_nuggets) :
  ∃ hotdogs : ℕ, hotdogs = 30 ∧ total = hotdogs + cheese_pops + chicken_nuggets :=
by
  sorry

end andrews_hotdogs_l793_79345


namespace fraction_arithmetic_l793_79397

theorem fraction_arithmetic : (1/2 - 1/6) / (1/6009 : ℚ) = 2003 := by
  sorry

end fraction_arithmetic_l793_79397


namespace intersection_range_l793_79386

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The theorem stating that if 2x³ - 3x² + (1 + b) = 0 has three distinct real roots, then -1 < b < 0 -/
theorem intersection_range (b : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = -b ∧ f y = -b ∧ f z = -b) →
  -1 < b ∧ b < 0 :=
by sorry

end intersection_range_l793_79386


namespace solution_implies_c_value_l793_79373

-- Define the function f
def f (x b : ℝ) : ℝ := x^2 + x + b

-- State the theorem
theorem solution_implies_c_value
  (b : ℝ)  -- b is a real number
  (h1 : ∀ x, f x b ≥ 0)  -- Value range of f is [0, +∞)
  (h2 : ∃ m, ∀ x, f x b < 16 ↔ x < m + 8)  -- Solution to f(x) < c is m + 8
  : 16 = 16 :=  -- We want to prove c = 16
by
  sorry

end solution_implies_c_value_l793_79373


namespace outfits_count_l793_79391

/-- Represents the number of shirts available -/
def num_shirts : Nat := 7

/-- Represents the number of pants available -/
def num_pants : Nat := 5

/-- Represents the number of ties available -/
def num_ties : Nat := 4

/-- Represents the number of jackets available -/
def num_jackets : Nat := 2

/-- Represents the number of tie options (wearing a tie or not) -/
def tie_options : Nat := num_ties + 1

/-- Represents the number of jacket options (wearing a jacket or not) -/
def jacket_options : Nat := num_jackets + 1

/-- Calculates the total number of possible outfits -/
def total_outfits : Nat := num_shirts * num_pants * tie_options * jacket_options

/-- Proves that the total number of possible outfits is 525 -/
theorem outfits_count : total_outfits = 525 := by
  sorry

end outfits_count_l793_79391


namespace max_product_sum_2024_l793_79313

theorem max_product_sum_2024 :
  (∃ (a b : ℤ), a + b = 2024 ∧ a * b = 1024144) ∧
  (∀ (x y : ℤ), x + y = 2024 → x * y ≤ 1024144) := by
  sorry

end max_product_sum_2024_l793_79313


namespace range_of_m_l793_79354

theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b : ℝ, a > 0 → b > 0 → (1/a + 1/b) * Real.sqrt (a^2 + b^2) ≥ 2*m - 4) :
  m ≤ 2 + Real.sqrt 2 := by
sorry

end range_of_m_l793_79354


namespace clara_dina_age_difference_l793_79331

theorem clara_dina_age_difference : ∃! n : ℕ+, ∃ C D : ℕ+,
  C = D + n ∧
  C - 1 = 3 * (D - 1) ∧
  C = D^3 + 1 ∧
  n = 7 := by
  sorry

end clara_dina_age_difference_l793_79331


namespace crowdfunding_highest_level_l793_79314

theorem crowdfunding_highest_level (x : ℝ) : 
  x > 0 ∧ 
  7298 * x = 200000 → 
  ⌊1296 * x⌋ = 35534 :=
by
  sorry

end crowdfunding_highest_level_l793_79314


namespace price_restoration_l793_79316

theorem price_restoration (original_price : ℝ) (reduced_price : ℝ) (restored_price : ℝ) : 
  reduced_price = original_price * (1 - 0.2) →
  restored_price = reduced_price * (1 + 0.25) →
  restored_price = original_price :=
by sorry

end price_restoration_l793_79316


namespace book_selection_probability_book_selection_proof_l793_79370

theorem book_selection_probability : ℕ → ℝ
  | 12 => 55 / 209
  | _ => 0

theorem book_selection_proof (n : ℕ) :
  n = 12 →
  (book_selection_probability n) = 
    (Nat.choose n 3 * Nat.choose (n - 3) 2 * Nat.choose (n - 5) 2 : ℝ) / 
    ((Nat.choose n 5 : ℝ) ^ 2) :=
by sorry

end book_selection_probability_book_selection_proof_l793_79370


namespace jerry_walk_time_approx_l793_79350

/-- Calculates the time it takes Jerry to walk each way to the sink and recycling bin -/
def jerryWalkTime (totalCans : ℕ) (cansPerTrip : ℕ) (drainTime : ℕ) (totalTime : ℕ) : ℚ :=
  let trips := totalCans / cansPerTrip
  let totalDrainTime := drainTime * trips
  let totalWalkTime := totalTime - totalDrainTime
  let walkTimePerTrip := totalWalkTime / trips
  walkTimePerTrip / 3

/-- Theorem stating that Jerry's walk time is approximately 6.67 seconds -/
theorem jerry_walk_time_approx :
  let walkTime := jerryWalkTime 28 4 30 350
  (walkTime > 6.66) ∧ (walkTime < 6.68) := by
  sorry

end jerry_walk_time_approx_l793_79350


namespace nine_point_circle_chords_l793_79325

/-- The number of chords that can be drawn in a circle with n points on its circumference -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords that can be drawn in a circle with 9 points on its circumference is 36 -/
theorem nine_point_circle_chords : num_chords 9 = 36 := by
  sorry

end nine_point_circle_chords_l793_79325


namespace min_abs_z_is_zero_l793_79390

theorem min_abs_z_is_zero (z : ℂ) (h : Complex.abs (z + 2 - 3*I) + Complex.abs (z - 2*I) = 7) :
  ∃ (w : ℂ), ∀ (z : ℂ), Complex.abs (z + 2 - 3*I) + Complex.abs (z - 2*I) = 7 → Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 0 :=
sorry

end min_abs_z_is_zero_l793_79390


namespace one_third_between_one_fourth_one_sixth_l793_79309

/-- The fraction one-third of the way from a to b -/
def one_third_between (a b : ℚ) : ℚ := (2 * a + b) / 3

/-- Prove that the fraction one-third of the way from 1/4 to 1/6 is equal to 2/9 -/
theorem one_third_between_one_fourth_one_sixth :
  one_third_between (1/4) (1/6) = 2/9 := by sorry

end one_third_between_one_fourth_one_sixth_l793_79309


namespace nail_trimming_customers_l793_79301

/-- The number of nails per customer -/
def nails_per_customer : ℕ := 20

/-- The total number of sounds produced by the nail cutter -/
def total_sounds : ℕ := 120

/-- The number of customers -/
def num_customers : ℕ := total_sounds / nails_per_customer

theorem nail_trimming_customers :
  num_customers = 6 :=
by sorry

end nail_trimming_customers_l793_79301


namespace ellipse_foci_distance_l793_79352

/-- The distance between the foci of the ellipse 9x^2 + 16y^2 = 144 is 2√7 -/
theorem ellipse_foci_distance :
  let a : ℝ := 4
  let b : ℝ := 3
  let c : ℝ := Real.sqrt (a^2 - b^2)
  (∀ x y : ℝ, 9 * x^2 + 16 * y^2 = 144) →
  2 * c = 2 * Real.sqrt 7 := by
sorry

end ellipse_foci_distance_l793_79352


namespace A_inter_B_l793_79317

def A : Set ℝ := {-2, -1, 0, 1, 2}

def B : Set ℝ := {x | -Real.sqrt 3 ≤ x ∧ x ≤ Real.sqrt 3}

theorem A_inter_B : A ∩ B = {-1, 0, 1} := by sorry

end A_inter_B_l793_79317


namespace road_renovation_rates_l793_79308

-- Define the daily renovation rates for Team A and Team B
def daily_rate_A (x : ℝ) : ℝ := x + 20
def daily_rate_B (x : ℝ) : ℝ := x

-- Define the condition that the time to renovate 200m for Team A equals the time to renovate 150m for Team B
def time_equality (x : ℝ) : Prop := 200 / (daily_rate_A x) = 150 / (daily_rate_B x)

-- Theorem stating the solution
theorem road_renovation_rates :
  ∃ x : ℝ, time_equality x ∧ daily_rate_A x = 80 ∧ daily_rate_B x = 60 := by
  sorry

end road_renovation_rates_l793_79308


namespace equilateral_triangle_division_l793_79387

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  center : Point2D
  sideLength : ℝ

/-- Represents a division of an equilateral triangle -/
def TriangleDivision (t : EquilateralTriangle) (n : ℕ) :=
  { subdivisions : List EquilateralTriangle // 
    subdivisions.length = n * n ∧
    ∀ sub ∈ subdivisions, sub.sideLength = t.sideLength / n }

/-- Theorem: An equilateral triangle can be divided into 9 smaller congruent equilateral triangles -/
theorem equilateral_triangle_division (t : EquilateralTriangle) : 
  ∃ (div : TriangleDivision t 3), True := by
  sorry

end equilateral_triangle_division_l793_79387


namespace a_value_m_minimum_l793_79362

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}

-- Theorem 1: Prove that a = 1
theorem a_value : 
  ∀ x ∈ solution_set 1, f 1 x ≤ 6 ∧
  ∀ a : ℝ, (∀ x ∈ solution_set a, f a x ≤ 6) → a = 1 :=
sorry

-- Define the function g (which is f with a = 1)
def g (x : ℝ) : ℝ := |2*x - 1| + 1

-- Theorem 2: Prove that the minimum value of m is 3.5
theorem m_minimum :
  (∃ m : ℝ, ∀ t : ℝ, g (t/2) ≤ m - g (-t)) ∧
  (∀ m : ℝ, (∀ t : ℝ, g (t/2) ≤ m - g (-t)) → m ≥ 3.5) :=
sorry

end a_value_m_minimum_l793_79362


namespace complex_magnitude_one_l793_79389

/-- Given a complex number z satisfying z⋅2i = |z|² + 1, prove that |z| = 1 -/
theorem complex_magnitude_one (z : ℂ) (h : z * (2 * Complex.I) = Complex.abs z ^ 2 + 1) :
  Complex.abs z = 1 := by sorry

end complex_magnitude_one_l793_79389


namespace cookies_per_tray_l793_79385

/-- Given that Marian baked 276 oatmeal cookies and used 23 trays,
    prove that she can place 12 cookies on a tray at a time. -/
theorem cookies_per_tray (total_cookies : ℕ) (num_trays : ℕ) 
  (h1 : total_cookies = 276) (h2 : num_trays = 23) :
  total_cookies / num_trays = 12 := by
  sorry

end cookies_per_tray_l793_79385


namespace rex_cards_left_l793_79365

def nicole_cards : ℕ := 500

def cindy_cards : ℕ := (2 * nicole_cards) + (2 * nicole_cards * 25 / 100)

def total_cards : ℕ := nicole_cards + cindy_cards

def rex_cards : ℕ := (2 * total_cards) / 3

def num_siblings : ℕ := 5

def cards_per_person : ℕ := rex_cards / (num_siblings + 1)

def cards_given_away : ℕ := cards_per_person * num_siblings

theorem rex_cards_left : rex_cards - cards_given_away = 196 := by
  sorry

end rex_cards_left_l793_79365


namespace grape_purchases_l793_79305

theorem grape_purchases (lena_shown ira_shown combined_shown : ℝ)
  (h1 : lena_shown = 2)
  (h2 : ira_shown = 3)
  (h3 : combined_shown = 4.5) :
  ∃ (lena_actual ira_actual offset : ℝ),
    lena_actual + offset = lena_shown ∧
    ira_actual + offset = ira_shown ∧
    lena_actual + ira_actual = combined_shown ∧
    lena_actual = 1.5 ∧
    ira_actual = 2.5 := by
  sorry

end grape_purchases_l793_79305


namespace isosceles_triangle_a_values_l793_79366

/-- An isosceles triangle with sides 10-a, 7, and 6 -/
structure IsoscelesTriangle (a : ℝ) :=
  (side1 : ℝ := 10 - a)
  (side2 : ℝ := 7)
  (side3 : ℝ := 6)
  (isIsosceles : (side1 = side2 ∧ side3 ≠ side1) ∨ 
                 (side1 = side3 ∧ side2 ≠ side1) ∨ 
                 (side2 = side3 ∧ side1 ≠ side2))

/-- The theorem stating that a is either 3 or 4 for an isosceles triangle with sides 10-a, 7, and 6 -/
theorem isosceles_triangle_a_values :
  ∀ a : ℝ, IsoscelesTriangle a → a = 3 ∨ a = 4 := by
  sorry

end isosceles_triangle_a_values_l793_79366


namespace lcm_gcd_problem_l793_79302

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 22 → 
  a = 220 → 
  b = 462 := by
sorry

end lcm_gcd_problem_l793_79302


namespace z_in_fourth_quadrant_l793_79300

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + Complex.I) :
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end z_in_fourth_quadrant_l793_79300


namespace fifth_bounce_height_l793_79326

/-- Calculates the height of a bouncing ball after a given number of bounces. -/
def bounceHeight (initialHeight : ℝ) (initialEfficiency : ℝ) (efficiencyDecrease : ℝ) (airResistanceLoss : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The height of the ball after the fifth bounce is approximately 0.82 feet. -/
theorem fifth_bounce_height :
  let initialHeight : ℝ := 96
  let initialEfficiency : ℝ := 0.5
  let efficiencyDecrease : ℝ := 0.05
  let airResistanceLoss : ℝ := 0.02
  let bounces : ℕ := 5
  abs (bounceHeight initialHeight initialEfficiency efficiencyDecrease airResistanceLoss bounces - 0.82) < 0.01 := by
  sorry

end fifth_bounce_height_l793_79326


namespace factorization_3x_squared_minus_12_l793_79330

theorem factorization_3x_squared_minus_12 (x : ℝ) : 3 * x^2 - 12 = 3 * (x + 2) * (x - 2) := by
  sorry

end factorization_3x_squared_minus_12_l793_79330


namespace complement_intersection_theorem_l793_79344

open Set

universe u

theorem complement_intersection_theorem (U M N : Set ℕ) :
  U = {0, 1, 2, 3, 4} →
  M = {0, 1, 2} →
  N = {2, 3} →
  (U \ M) ∩ N = {3} := by
  sorry

end complement_intersection_theorem_l793_79344


namespace min_value_expression_l793_79319

theorem min_value_expression (α β : ℝ) (h1 : α ≠ 0) (h2 : |β| = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), x = |((β + α) / (1 + α * β))| → x ≥ m :=
by sorry

end min_value_expression_l793_79319


namespace largest_prime_factor_of_sum_of_divisors_180_l793_79384

/-- The sum of divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number n -/
def largest_prime_factor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_180 :
  largest_prime_factor (sum_of_divisors 180) = 13 := by sorry

end largest_prime_factor_of_sum_of_divisors_180_l793_79384


namespace unique_three_digit_number_l793_79374

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  ∃ k : ℕ, n + 1 = 4 * k ∧
  ∃ l : ℕ, n + 1 = 5 * l ∧
  ∃ m : ℕ, n + 1 = 6 * m ∧
  ∃ p : ℕ, n + 1 = 8 * p :=
by
  -- The proof goes here
  sorry

end unique_three_digit_number_l793_79374


namespace movie_of_the_year_requirement_l793_79327

/-- The number of members in the cinematic academy -/
def academy_members : ℕ := 795

/-- The fraction of lists a film must appear on to be considered for "movie of the year" -/
def required_fraction : ℚ := 1 / 4

/-- The smallest number of lists a film can appear on to be considered for "movie of the year" -/
def min_lists : ℕ := 199

/-- Theorem stating the smallest number of lists a film must appear on -/
theorem movie_of_the_year_requirement :
  min_lists = ⌈(required_fraction * academy_members : ℚ)⌉ :=
sorry

end movie_of_the_year_requirement_l793_79327


namespace almost_square_quotient_l793_79311

/-- Definition of an almost square -/
def AlmostSquare (k : ℕ) : Prop := ∃ n : ℕ, k = n * (n + 1)

/-- Theorem: Every almost square can be expressed as a quotient of two almost squares -/
theorem almost_square_quotient (n : ℕ) : 
  ∃ a b : ℕ, AlmostSquare a ∧ AlmostSquare b ∧ n * (n + 1) = a / b := by
  sorry

end almost_square_quotient_l793_79311


namespace square_area_is_26_l793_79364

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square defined by its four vertices -/
structure Square where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculate the area of a square given its vertices -/
def squareArea (sq : Square) : ℝ := 
  let dx := sq.p.x - sq.q.x
  let dy := sq.p.y - sq.q.y
  (dx * dx + dy * dy)

/-- The theorem stating that the area of the given square is 26 -/
theorem square_area_is_26 : 
  let sq := Square.mk 
    (Point.mk 2 3) 
    (Point.mk (-3) 4) 
    (Point.mk (-2) 9) 
    (Point.mk 3 8)
  squareArea sq = 26 := by
  sorry

end square_area_is_26_l793_79364


namespace subset_implies_a_equals_one_l793_79343

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l793_79343


namespace ketchup_tomatoes_ratio_l793_79321

/-- Given that 3 liters of ketchup require 69 kg of tomatoes, 
    prove that 5 liters of ketchup require 115 kg of tomatoes. -/
theorem ketchup_tomatoes_ratio (tomatoes_for_three : ℝ) (ketchup_liters : ℝ) : 
  tomatoes_for_three = 69 → ketchup_liters = 5 → 
  (tomatoes_for_three / 3) * ketchup_liters = 115 := by
sorry

end ketchup_tomatoes_ratio_l793_79321


namespace height_of_smaller_cone_is_9_l793_79399

/-- The height of the smaller cone removed from a right circular cone to create a frustum -/
def height_of_smaller_cone (frustum_height : ℝ) (larger_base_area : ℝ) (smaller_base_area : ℝ) : ℝ :=
  sorry

theorem height_of_smaller_cone_is_9 :
  height_of_smaller_cone 18 (324 * Real.pi) (36 * Real.pi) = 9 := by
  sorry

end height_of_smaller_cone_is_9_l793_79399


namespace line_perp_parallel_implies_planes_perp_l793_79356

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (distinct : Line → Line → Prop)
variable (nonCoincident : Plane → Plane → Prop)
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (m : Line) (α β : Plane)
  (h1 : nonCoincident α β)
  (h2 : perpendicular m α)
  (h3 : parallel m β) :
  planePerp α β :=
sorry

end line_perp_parallel_implies_planes_perp_l793_79356


namespace largest_prime_factor_of_1729_l793_79318

theorem largest_prime_factor_of_1729 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 1729 → q ≤ p :=
by sorry

end largest_prime_factor_of_1729_l793_79318


namespace largest_base7_to_base3_l793_79369

/-- Converts a number from base 7 to base 10 -/
def base7ToDecimal (n : Nat) : Nat :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7 + (n % 10)

/-- Converts a number from base 10 to base 3 -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

/-- The largest three-digit number in base 7 -/
def largestBase7 : Nat := 666

theorem largest_base7_to_base3 :
  decimalToBase3 (base7ToDecimal largestBase7) = [1, 1, 0, 2, 0, 0] := by
  sorry

#eval decimalToBase3 (base7ToDecimal largestBase7)

end largest_base7_to_base3_l793_79369


namespace expression_eval_zero_l793_79320

theorem expression_eval_zero (a : ℚ) (h : a = 3/2) : 
  (5 * a^2 - 13 * a + 4) * (2 * a - 3) = 0 := by
  sorry

end expression_eval_zero_l793_79320


namespace first_triangle_is_isosceles_l793_79303

-- Define a triangle structure
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_eq_180 : α + β + γ = 180

-- Define the theorem
theorem first_triangle_is_isosceles 
  (t1 t2 : Triangle) 
  (h1 : ∃ (θ : Real), t1.α + t1.β = θ ∧ θ ≤ 180) 
  (h2 : ∃ (φ : Real), t1.β + t1.γ = φ ∧ φ ≤ 180) : 
  t1.α = t1.γ ∨ t1.α = t1.β ∨ t1.β = t1.γ :=
sorry

end first_triangle_is_isosceles_l793_79303


namespace quadratic_root_expression_l793_79358

theorem quadratic_root_expression (x : ℝ) : 
  x > 0 ∧ x^2 - 10*x - 10 = 0 → 1/20 * x^4 - 6*x^2 - 45 = -50 := by
  sorry

end quadratic_root_expression_l793_79358


namespace triangle_properties_l793_79315

-- Define the triangle
def Triangle (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let DC := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  AB = 25 ∧
  (B.1 - D.1) * (A.1 - D.1) + (B.2 - D.2) * (A.2 - D.2) = 0 ∧ -- Right angle at D
  AD / AB = 4 / 5 ∧ -- sin A = 4/5
  BD / BC = 1 / 5   -- sin C = 1/5

-- Theorem statement
theorem triangle_properties (A B C D : ℝ × ℝ) (h : Triangle A B C D) :
  let DC := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let BD := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  DC = 40 * Real.sqrt 6 ∧
  1/2 * AD * BD = 150 := by
sorry

end triangle_properties_l793_79315


namespace log_inequality_implies_x_geq_125_l793_79382

theorem log_inequality_implies_x_geq_125 (x : ℝ) (h1 : x > 0) 
  (h2 : Real.log x / Real.log 3 ≥ Real.log 5 / Real.log 3 + (2/3) * (Real.log x / Real.log 3)) :
  x ≥ 125 := by
  sorry

end log_inequality_implies_x_geq_125_l793_79382


namespace card_probability_l793_79355

theorem card_probability : 
  let total_cards : ℕ := 52
  let cards_per_suit : ℕ := 13
  let top_cards : ℕ := 4
  let favorable_suits : ℕ := 2  -- spades and clubs

  let favorable_outcomes : ℕ := favorable_suits * (cards_per_suit.descFactorial top_cards)
  let total_outcomes : ℕ := total_cards.descFactorial top_cards

  (favorable_outcomes : ℚ) / total_outcomes = 286 / 54145 := by sorry

end card_probability_l793_79355


namespace square_root_and_cube_root_l793_79380

theorem square_root_and_cube_root : 
  (∃ x : ℝ, x^2 = 16 ∧ (x = 4 ∨ x = -4)) ∧ 
  (∃ y : ℝ, y^3 = -2 ∧ y = -Real.rpow 2 (1/3)) := by
  sorry

end square_root_and_cube_root_l793_79380


namespace container_filling_l793_79312

theorem container_filling (capacity : ℝ) (initial_fraction : ℝ) (added_water : ℝ) :
  capacity = 80 →
  initial_fraction = 1/2 →
  added_water = 20 →
  (initial_fraction * capacity + added_water) / capacity = 3/4 := by
sorry

end container_filling_l793_79312


namespace soccer_league_games_l793_79332

theorem soccer_league_games (n : ℕ) (total_games : ℕ) (h1 : n = 10) (h2 : total_games = 45) :
  (n * (n - 1)) / 2 = total_games → ∃ k : ℕ, k = 1 ∧ k * (n * (n - 1)) / 2 = total_games :=
by sorry

end soccer_league_games_l793_79332


namespace green_pill_cost_proof_l793_79396

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℝ := 21.5

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℝ := green_pill_cost - 2

/-- The number of days for which pills are taken -/
def days : ℕ := 18

/-- The total cost of all pills over the given period -/
def total_cost : ℝ := 738

theorem green_pill_cost_proof :
  (green_pill_cost + pink_pill_cost) * days = total_cost ∧
  green_pill_cost = pink_pill_cost + 2 ∧
  green_pill_cost = 21.5 :=
sorry

end green_pill_cost_proof_l793_79396


namespace walts_investment_rate_l793_79347

/-- Proves that given the conditions of Walt's investment, the unknown interest rate is 8% -/
theorem walts_investment_rate : ∀ (total_money : ℝ) (known_rate : ℝ) (unknown_amount : ℝ) (total_interest : ℝ),
  total_money = 9000 →
  known_rate = 0.09 →
  unknown_amount = 4000 →
  total_interest = 770 →
  ∃ (unknown_rate : ℝ),
    unknown_rate * unknown_amount + known_rate * (total_money - unknown_amount) = total_interest ∧
    unknown_rate = 0.08 :=
by sorry

end walts_investment_rate_l793_79347
