import Mathlib

namespace water_difference_before_exchange_l2361_236149

/-- The difference in water amounts before the exchange, given the conditions of the problem -/
theorem water_difference_before_exchange 
  (S H : ℝ) -- S and H represent the initial amounts of water for Seungmin and Hyoju
  (h1 : S > H) -- Seungmin has more water than Hyoju
  (h2 : S - 0.43 - (H + 0.43) = 0.88) -- Difference after exchange
  : S - H = 1.74 := by sorry

end water_difference_before_exchange_l2361_236149


namespace arithmetic_sequence_max_sum_l2361_236141

/-- Given an arithmetic sequence, prove that under certain conditions, 
    the maximum sum occurs at the 8th term -/
theorem arithmetic_sequence_max_sum 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2) 
  (h_15 : S 15 > 0) 
  (h_16 : S 16 < 0) : 
  ∃ (n : ℕ), ∀ (m : ℕ), S m ≤ S n ∧ n = 8 :=
sorry

end arithmetic_sequence_max_sum_l2361_236141


namespace unique_toy_value_l2361_236155

theorem unique_toy_value (total_toys : ℕ) (total_worth : ℕ) (common_value : ℕ) (common_count : ℕ) :
  total_toys = common_count + 1 →
  total_worth = common_value * common_count + (total_worth - common_value * common_count) →
  common_count = 8 →
  total_toys = 9 →
  total_worth = 52 →
  common_value = 5 →
  total_worth - common_value * common_count = 12 :=
by sorry

end unique_toy_value_l2361_236155


namespace driving_time_to_airport_l2361_236128

-- Define time in minutes since midnight
def flight_time : ℕ := 20 * 60
def check_in_buffer : ℕ := 2 * 60
def house_departure_time : ℕ := 17 * 60
def parking_and_terminal_time : ℕ := 15

-- Theorem statement
theorem driving_time_to_airport :
  let check_in_time := flight_time - check_in_buffer
  let airport_arrival_time := check_in_time - parking_and_terminal_time
  airport_arrival_time - house_departure_time = 45 := by
sorry

end driving_time_to_airport_l2361_236128


namespace partnership_profit_l2361_236165

/-- Calculates the total profit of a partnership given investments and one partner's share -/
def calculate_total_profit (investment_a investment_b investment_c c_share : ℕ) : ℕ :=
  let total_parts := investment_a / investment_c + investment_b / investment_c + 1
  total_parts * c_share

/-- Proves that given the investments and C's share, the total profit is 252000 -/
theorem partnership_profit (investment_a investment_b investment_c c_share : ℕ) 
  (h1 : investment_a = 8000)
  (h2 : investment_b = 4000)
  (h3 : investment_c = 2000)
  (h4 : c_share = 36000) :
  calculate_total_profit investment_a investment_b investment_c c_share = 252000 := by
  sorry

#eval calculate_total_profit 8000 4000 2000 36000

end partnership_profit_l2361_236165


namespace journey_equation_correct_l2361_236140

/-- Represents a car journey with a stop -/
structure Journey where
  initial_speed : ℝ
  final_speed : ℝ
  total_distance : ℝ
  total_time : ℝ
  stop_duration : ℝ

/-- Theorem stating that the given equation correctly represents the total distance traveled -/
theorem journey_equation_correct (j : Journey) 
  (h1 : j.initial_speed = 90)
  (h2 : j.final_speed = 110)
  (h3 : j.total_distance = 300)
  (h4 : j.total_time = 3.5)
  (h5 : j.stop_duration = 0.5) :
  ∃ t : ℝ, j.initial_speed * t + j.final_speed * (j.total_time - j.stop_duration - t) = j.total_distance :=
sorry

end journey_equation_correct_l2361_236140


namespace min_value_condition_l2361_236148

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then |x + a| + |x - 2|
  else x^2 - a*x + (1/2)*a + 1

theorem min_value_condition (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 2*a) ∧ (∃ x : ℝ, f a x = 2*a) ↔ a = -Real.sqrt 13 - 3 := by
  sorry

end min_value_condition_l2361_236148


namespace min_value_expression_l2361_236154

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + b = a*b) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2*x + y = x*y → 1/(a-1) + 2/(b-2) ≤ 1/(x-1) + 2/(y-2) :=
sorry

end min_value_expression_l2361_236154


namespace growth_rate_is_ten_percent_l2361_236118

def turnover_may : ℝ := 1
def turnover_july : ℝ := 1.21

def growth_rate (r : ℝ) : Prop :=
  turnover_may * (1 + r)^2 = turnover_july

theorem growth_rate_is_ten_percent :
  ∃ (r : ℝ), growth_rate r ∧ r = 0.1 :=
sorry

end growth_rate_is_ten_percent_l2361_236118


namespace car_distance_ratio_l2361_236193

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- Theorem stating the ratio of distances covered by Car A and Car B -/
theorem car_distance_ratio (carA carB : Car)
    (hA : carA = { speed := 80, time := 5 })
    (hB : carB = { speed := 100, time := 2 }) :
    distance carA / distance carB = 2 := by
  sorry

end car_distance_ratio_l2361_236193


namespace linda_earnings_l2361_236183

/-- Calculates the total money earned from selling jeans and tees -/
def total_money_earned (jeans_price : ℕ) (tees_price : ℕ) (jeans_sold : ℕ) (tees_sold : ℕ) : ℕ :=
  jeans_price * jeans_sold + tees_price * tees_sold

/-- Proves that Linda earned $100 from selling jeans and tees -/
theorem linda_earnings : total_money_earned 11 8 4 7 = 100 := by
  sorry

end linda_earnings_l2361_236183


namespace even_function_implies_a_equals_two_l2361_236126

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (a+1)x^2 + (a-2)x + a^2 - a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a+1)*x^2 + (a-2)*x + a^2 - a - 2

theorem even_function_implies_a_equals_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by sorry

end even_function_implies_a_equals_two_l2361_236126


namespace min_distance_to_line_l2361_236122

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

def line_equation (x y : ℝ) : Prop := x + y + 1 = 0

theorem min_distance_to_line (m n : ℝ) 
  (h : (a.1 - m) * (b.1 - m) + (a.2 - n) * (b.2 - n) = 0) : 
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
  ∀ (x y : ℝ), line_equation x y → 
    d ≤ Real.sqrt ((x - m)^2 + (y - n)^2) :=
sorry

end min_distance_to_line_l2361_236122


namespace mocktail_lime_cost_l2361_236150

/-- Represents the cost of limes in dollars for a given number of limes -/
def lime_cost (num_limes : ℕ) : ℚ :=
  (num_limes : ℚ) / 3

/-- Calculates the number of limes needed for a given number of days -/
def limes_needed (days : ℕ) : ℕ :=
  (days + 1) / 2

theorem mocktail_lime_cost : lime_cost (limes_needed 30) = 5 := by
  sorry

end mocktail_lime_cost_l2361_236150


namespace sum_of_squares_difference_l2361_236131

theorem sum_of_squares_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end sum_of_squares_difference_l2361_236131


namespace prob_club_then_heart_l2361_236191

/-- The number of cards in a standard deck --/
def standard_deck_size : ℕ := 52

/-- The number of clubs in a standard deck --/
def num_clubs : ℕ := 13

/-- The number of hearts in a standard deck --/
def num_hearts : ℕ := 13

/-- Probability of drawing a club first and then a heart from a standard 52-card deck --/
theorem prob_club_then_heart : 
  (num_clubs : ℚ) / standard_deck_size * num_hearts / (standard_deck_size - 1) = 13 / 204 := by
  sorry

end prob_club_then_heart_l2361_236191


namespace nick_money_value_l2361_236144

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of nickels Nick has -/
def num_nickels : ℕ := 6

/-- The number of dimes Nick has -/
def num_dimes : ℕ := 2

/-- The number of quarters Nick has -/
def num_quarters : ℕ := 1

/-- The total value of Nick's coins in cents -/
def total_value : ℕ := num_nickels * nickel_value + num_dimes * dime_value + num_quarters * quarter_value

theorem nick_money_value : total_value = 75 := by
  sorry

end nick_money_value_l2361_236144


namespace unique_solution_system_l2361_236196

theorem unique_solution_system (x y : ℝ) : 
  (x^3 + y^3 + 3*x*y = 1 ∧ x^2 - y^2 = 1) →
  ((x ≥ 0 ∧ y ≥ 0) ∨ (x + y > 0)) →
  x = 1 ∧ y = 0 := by sorry

end unique_solution_system_l2361_236196


namespace cat_food_percentage_l2361_236110

/-- Proves that given 7 dogs and 4 cats, where all dogs receive equal amounts of food,
    all cats receive equal amounts of food, and the total food for all cats equals
    the food for one dog, the percentage of total food that one cat receives is 1/32. -/
theorem cat_food_percentage :
  ∀ (dog_food cat_food : ℚ),
  dog_food > 0 →
  cat_food > 0 →
  4 * cat_food = dog_food →
  (cat_food / (7 * dog_food + 4 * cat_food)) = 1 / 32 :=
by
  sorry

end cat_food_percentage_l2361_236110


namespace larger_number_proof_l2361_236103

theorem larger_number_proof (x y : ℤ) (h1 : x - y = 5) (h2 : x + y = 37) :
  max x y = 21 := by
  sorry

end larger_number_proof_l2361_236103


namespace classroom_difference_maple_leaf_elementary_l2361_236102

theorem classroom_difference : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_classrooms, students_per_class, rabbits_per_class, guinea_pigs_per_class =>
  let total_students := num_classrooms * students_per_class
  let total_pets := num_classrooms * (rabbits_per_class + guinea_pigs_per_class)
  total_students - total_pets

theorem maple_leaf_elementary :
  classroom_difference 6 15 1 3 = 66 := by
  sorry

end classroom_difference_maple_leaf_elementary_l2361_236102


namespace inequality_theorem_l2361_236189

theorem inequality_theorem (x y a : ℝ) (h1 : x < y) (h2 : a < 1) : x + a < y + 1 := by
  sorry

end inequality_theorem_l2361_236189


namespace perpendicular_lines_from_perpendicular_planes_l2361_236117

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (m n : Line)
  (h1 : perp_plane_line α m)
  (h2 : perp_plane_line β n)
  (h3 : perp_plane α β) :
  perp_line m n :=
sorry

end perpendicular_lines_from_perpendicular_planes_l2361_236117


namespace diamond_properties_l2361_236173

def diamond (x y : ℝ) : ℝ := x^2 - 2*x*y + y^2

theorem diamond_properties :
  (∀ x y : ℝ, diamond x y = diamond y x) ∧
  (∃ x y : ℝ, 2 * (diamond x y) ≠ diamond (2*x) (2*y)) ∧
  (∀ x : ℝ, diamond x 0 = x^2) ∧
  (∀ x : ℝ, diamond x x = 0) ∧
  (∀ x y : ℝ, x = y → diamond x y = 0) :=
by sorry

end diamond_properties_l2361_236173


namespace find_number_l2361_236169

theorem find_number : ∃ x : ℝ, 0.123 + 0.321 + x = 1.794 ∧ x = 1.350 := by sorry

end find_number_l2361_236169


namespace four_number_sequence_l2361_236123

theorem four_number_sequence : ∃ (a₁ a₂ a₃ a₄ : ℝ),
  (a₂^2 = a₁ * a₃) ∧
  (2 * a₃ = a₂ + a₄) ∧
  (a₁ + a₄ = 21) ∧
  (a₂ + a₃ = 18) ∧
  ((a₁ = 3 ∧ a₂ = 6 ∧ a₃ = 12 ∧ a₄ = 18) ∨
   (a₁ = 18.75 ∧ a₂ = 11.25 ∧ a₃ = 6.75 ∧ a₄ = 2.25)) :=
by
  sorry


end four_number_sequence_l2361_236123


namespace brand_preference_ratio_l2361_236100

theorem brand_preference_ratio (total_respondents : ℕ) (brand_x_preference : ℕ) 
  (h1 : total_respondents = 80)
  (h2 : brand_x_preference = 60)
  (h3 : brand_x_preference < total_respondents) :
  (brand_x_preference : ℚ) / (total_respondents - brand_x_preference : ℚ) = 3 / 1 := by
  sorry

end brand_preference_ratio_l2361_236100


namespace no_integer_solution_a_squared_minus_3b_squared_equals_8_l2361_236187

theorem no_integer_solution_a_squared_minus_3b_squared_equals_8 :
  ¬ ∃ (a b : ℤ), a^2 - 3*b^2 = 8 := by
sorry

end no_integer_solution_a_squared_minus_3b_squared_equals_8_l2361_236187


namespace veg_eaters_count_l2361_236185

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  only_veg : ℕ
  only_nonveg : ℕ
  both_veg_and_nonveg : ℕ

/-- Calculates the total number of people who eat veg in the family -/
def total_veg_eaters (fd : FamilyDiet) : ℕ :=
  fd.only_veg + fd.both_veg_and_nonveg

/-- Theorem: The number of people who eat veg in the given family is 26 -/
theorem veg_eaters_count (fd : FamilyDiet) 
  (h1 : fd.only_veg = 15)
  (h2 : fd.only_nonveg = 8)
  (h3 : fd.both_veg_and_nonveg = 11) : 
  total_veg_eaters fd = 26 := by
  sorry

end veg_eaters_count_l2361_236185


namespace vector_sum_zero_l2361_236190

variable {V : Type*} [AddCommGroup V]
variable (A C D E : V)

theorem vector_sum_zero :
  (E - C) + (C - A) - (E - D) - (D - A) = (0 : V) := by sorry

end vector_sum_zero_l2361_236190


namespace root_location_l2361_236147

theorem root_location (a b : ℝ) (n : ℤ) : 
  (2 : ℝ)^a = 3 → 
  (3 : ℝ)^b = 2 → 
  (∃ x_b : ℝ, x_b ∈ Set.Ioo (n : ℝ) (n + 1) ∧ a^x_b + x_b - b = 0) → 
  n = -1 := by
sorry

end root_location_l2361_236147


namespace union_of_M_and_N_l2361_236161

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end union_of_M_and_N_l2361_236161


namespace smallest_n_divisible_by_primes_l2361_236153

def is_divisible_by_primes (n : ℕ) : Prop :=
  ∃ k : ℕ, k * (2213 * 3323 * 6121) = (n / 2).factorial * 2^(n / 2)

theorem smallest_n_divisible_by_primes :
  (∀ m : ℕ, m < 12242 → ¬(is_divisible_by_primes m)) ∧
  (is_divisible_by_primes 12242) :=
sorry

end smallest_n_divisible_by_primes_l2361_236153


namespace unique_a_for_equal_roots_l2361_236159

theorem unique_a_for_equal_roots :
  ∃! a : ℝ, ∀ x : ℝ, x^2 - (a + 1) * x + a = 0 → (∃! y : ℝ, y^2 - (a + 1) * y + a = 0) := by
  sorry

end unique_a_for_equal_roots_l2361_236159


namespace seven_balls_three_boxes_l2361_236112

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 7 distinguishable balls into 3 indistinguishable boxes is 301 -/
theorem seven_balls_three_boxes : distribute_balls 7 3 = 301 := by sorry

end seven_balls_three_boxes_l2361_236112


namespace john_finish_time_l2361_236104

/-- The time it takes for John to finish the job by himself -/
def john_time : ℝ := 1.5

/-- The time it takes for David to finish the job by himself -/
def david_time : ℝ := 2 * john_time

/-- The time it takes for John and David to finish the job together -/
def combined_time : ℝ := 1

theorem john_finish_time :
  (1 / john_time + 1 / david_time) * combined_time = 1 ∧ david_time = 2 * john_time → john_time = 1.5 := by
  sorry

end john_finish_time_l2361_236104


namespace existence_of_odd_powers_representation_l2361_236116

theorem existence_of_odd_powers_representation (m : ℤ) :
  ∃ (a b k : ℤ), 
    Odd a ∧ 
    Odd b ∧ 
    k ≥ 0 ∧ 
    2 * m = a^19 + b^99 + k * 2^1999 := by
  sorry

end existence_of_odd_powers_representation_l2361_236116


namespace solve_equation_l2361_236113

theorem solve_equation (x : ℚ) : (3 * x + 4) / 7 = 15 → x = 101 / 3 := by
  sorry

end solve_equation_l2361_236113


namespace value_of_b_l2361_236133

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 1) : b = 1 := by
  sorry

end value_of_b_l2361_236133


namespace inequality_solution_set_l2361_236171

-- Define the inequality function
def f (x : ℝ) := (3*x + 1) * (2*x - 1)

-- Define the solution set
def solution_set := {x : ℝ | x < -1/3 ∨ x > 1/2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | f x > 0} = solution_set := by sorry

end inequality_solution_set_l2361_236171


namespace infinitely_many_primes_of_form_l2361_236199

theorem infinitely_many_primes_of_form (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ (S : Set Nat), (∀ n ∈ S, Nat.Prime n ∧ ∃ x, n = 2 * p * x + 1) ∧ Set.Infinite S :=
sorry

end infinitely_many_primes_of_form_l2361_236199


namespace min_n_value_l2361_236178

theorem min_n_value (m : ℝ) :
  (∀ x : ℝ, |x - m| ≤ 2 → -1 ≤ x ∧ x ≤ 3) ∧
  ¬(∀ x : ℝ, |x - m| ≤ 2 → -1 ≤ x ∧ x < 3) :=
by sorry

end min_n_value_l2361_236178


namespace smiles_cookies_leftover_l2361_236136

theorem smiles_cookies_leftover (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 := by
  sorry

end smiles_cookies_leftover_l2361_236136


namespace triangle_properties_l2361_236160

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def satisfies_conditions (t : Triangle) : Prop :=
  t.c = 2 * Real.sqrt 3 ∧
  t.a * Real.sin t.A - t.c * Real.sin t.C = (t.a - t.b) * Real.sin t.B ∧
  t.c + t.b * Real.cos t.A = t.a * (4 * Real.cos t.A + Real.cos t.B)

/-- Theorem stating the conclusions -/
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = Real.pi / 3 ∧ t.a * t.b * Real.sin t.C / 2 = 2 * Real.sqrt 3 := by
  sorry

end triangle_properties_l2361_236160


namespace pascal_triangle_value_l2361_236139

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of elements in the row of Pascal's triangle we're considering -/
def row_length : ℕ := 51

/-- The position of the number we're looking for in the row (1-indexed) -/
def position : ℕ := 43

/-- The value we want to prove is correct -/
def target_value : ℕ := 10272278170

theorem pascal_triangle_value :
  binomial (row_length - 1) (position - 1) = target_value := by
  sorry

end pascal_triangle_value_l2361_236139


namespace robin_cupcakes_sold_l2361_236174

/-- Represents the number of cupcakes Robin initially made -/
def initial_cupcakes : ℕ := 42

/-- Represents the number of additional cupcakes Robin made -/
def additional_cupcakes : ℕ := 39

/-- Represents the final number of cupcakes Robin had -/
def final_cupcakes : ℕ := 59

/-- Represents the number of cupcakes Robin sold -/
def sold_cupcakes : ℕ := 22

theorem robin_cupcakes_sold :
  initial_cupcakes - sold_cupcakes + additional_cupcakes = final_cupcakes :=
by sorry

end robin_cupcakes_sold_l2361_236174


namespace mixed_strategy_optimal_mixed_strategy_optimal_at_60_l2361_236156

/-- Represents the cost function for purchasing heaters from a store -/
structure StoreCost where
  typeA : ℝ  -- Cost per unit of Type A heater (including shipping)
  typeB : ℝ  -- Cost per unit of Type B heater (including shipping)

/-- Calculates the total cost for a store given the number of Type A heaters -/
def totalCost (store : StoreCost) (x : ℝ) : ℝ :=
  store.typeA * x + store.typeB * (100 - x)

/-- Store A's cost structure -/
def storeA : StoreCost := { typeA := 110, typeB := 210 }

/-- Store B's cost structure -/
def storeB : StoreCost := { typeA := 120, typeB := 202 }

/-- Cost function for buying Type A from Store A and Type B from Store B -/
def mixedCost (x : ℝ) : ℝ := storeA.typeA * x + storeB.typeB * (100 - x)

/-- Theorem: The mixed purchasing strategy is always the most cost-effective -/
theorem mixed_strategy_optimal (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) : 
  mixedCost x ≤ min (totalCost storeA x) (totalCost storeB x) := by
  sorry

/-- Corollary: When x = 60, the mixed strategy is more cost-effective than buying from a single store -/
theorem mixed_strategy_optimal_at_60 : 
  mixedCost 60 < min (totalCost storeA 60) (totalCost storeB 60) := by
  sorry

end mixed_strategy_optimal_mixed_strategy_optimal_at_60_l2361_236156


namespace probability_three_students_same_canteen_l2361_236114

/-- The probability of all three students going to the same canteen -/
def probability_same_canteen (num_canteens : ℕ) (num_students : ℕ) : ℚ :=
  if num_canteens = 2 ∧ num_students = 3 then
    1 / 4
  else
    0

/-- Theorem: The probability of all three students going to the same canteen is 1/4 -/
theorem probability_three_students_same_canteen :
  probability_same_canteen 2 3 = 1 / 4 := by
  sorry

end probability_three_students_same_canteen_l2361_236114


namespace starting_lineup_combinations_l2361_236166

def total_players : ℕ := 18
def lineup_size : ℕ := 8
def triplets : ℕ := 3
def twins : ℕ := 2

def remaining_players : ℕ := total_players - (triplets + twins)
def players_to_choose : ℕ := lineup_size - (triplets + twins)

theorem starting_lineup_combinations : 
  Nat.choose remaining_players players_to_choose = 286 := by
  sorry

end starting_lineup_combinations_l2361_236166


namespace number_puzzle_2016_l2361_236132

theorem number_puzzle_2016 : ∃ (x y : ℕ), ∃ (z : ℕ), 
  x + y = 2016 ∧ 
  x = 10 * y + z ∧ 
  z < 10 ∧
  x = 1833 ∧ 
  y = 183 := by
  sorry

end number_puzzle_2016_l2361_236132


namespace possible_values_of_a_minus_b_l2361_236145

theorem possible_values_of_a_minus_b (a b : ℝ) 
  (ha : |a| = 8) 
  (hb : |b| = 6) 
  (hab : |a + b| = a + b) : 
  a - b = 2 ∨ a - b = 14 := by
sorry

end possible_values_of_a_minus_b_l2361_236145


namespace min_value_expression_min_value_attained_l2361_236162

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  8 * a^4 + 16 * b^4 + 27 * c^4 + 1 / (6 * a * b * c) ≥ 12 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  8 * a^4 + 16 * b^4 + 27 * c^4 + 1 / (6 * a * b * c) < 12 + ε :=
by sorry

end min_value_expression_min_value_attained_l2361_236162


namespace slope_product_for_30_degree_angle_l2361_236180

theorem slope_product_for_30_degree_angle (m₁ m₂ : ℝ) :
  m₁ ≠ 0 →
  m₂ = 4 * m₁ →
  |((m₂ - m₁) / (1 + m₁ * m₂))| = 1 / Real.sqrt 3 →
  m₁ * m₂ = (38 - 6 * Real.sqrt 33) / 16 :=
by sorry

end slope_product_for_30_degree_angle_l2361_236180


namespace construction_team_equation_l2361_236101

/-- Represents the equation for a construction team's road-laying project -/
theorem construction_team_equation (x : ℝ) (h : x > 0) :
  let total_length : ℝ := 480
  let efficiency_increase : ℝ := 0.5
  let days_ahead : ℝ := 4
  (total_length / x) - (total_length / ((1 + efficiency_increase) * x)) = days_ahead :=
by sorry

end construction_team_equation_l2361_236101


namespace find_number_l2361_236195

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 7) = 99 :=
  sorry

end find_number_l2361_236195


namespace perpendicular_parallel_implies_perpendicular_parallel_lines_implies_perpendicular_planes_l2361_236198

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the relation of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel_planes α β) :
  perpendicular_lines l m :=
sorry

-- Theorem 2
theorem parallel_lines_implies_perpendicular_planes
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel_lines l m) :
  perpendicular_planes α β :=
sorry

end perpendicular_parallel_implies_perpendicular_parallel_lines_implies_perpendicular_planes_l2361_236198


namespace tire_change_problem_l2361_236152

theorem tire_change_problem (total_cars : ℕ) (tires_per_car : ℕ) (half_change_cars : ℕ) (tires_left : ℕ) : 
  total_cars = 10 →
  tires_per_car = 4 →
  half_change_cars = 2 →
  tires_left = 20 →
  ∃ (no_change_cars : ℕ), 
    no_change_cars = total_cars - (half_change_cars + (total_cars * tires_per_car - tires_left - half_change_cars * (tires_per_car / 2)) / tires_per_car) ∧
    no_change_cars = 4 :=
by sorry

end tire_change_problem_l2361_236152


namespace sum_of_radii_tangent_circles_l2361_236184

/-- The sum of all possible radii of a circle tangent to both axes and externally tangent to another circle -/
theorem sum_of_radii_tangent_circles : ∃ (r₁ r₂ : ℝ),
  let c₁ : ℝ × ℝ := (r₁, r₁)  -- Center of the first circle
  let c₂ : ℝ × ℝ := (5, 0)    -- Center of the second circle
  let r₃ : ℝ := 3             -- Radius of the second circle
  (0 < r₁ ∧ 0 < r₂) ∧         -- Radii are positive
  (c₁.1 - c₂.1)^2 + (c₁.2 - c₂.2)^2 = (r₁ + r₃)^2 ∧  -- Circles are externally tangent
  r₁ + r₂ = 16 :=             -- Sum of radii is 16
by sorry

end sum_of_radii_tangent_circles_l2361_236184


namespace inscribed_circle_radius_external_tangents_l2361_236164

/-- Given two externally tangent circles, this theorem proves the radius of the circle
    tangent to their common external tangents and the line segment connecting the
    external points of tangency on the larger circle. -/
theorem inscribed_circle_radius_external_tangents
  (R : ℝ) (r : ℝ) (h_R : R = 4) (h_r : r = 3) (h_touch : R > r) :
  let d := R + r  -- Distance between circle centers
  let inscribed_radius := (R * r) / d
  inscribed_radius = 12 / 7 :=
by sorry

end inscribed_circle_radius_external_tangents_l2361_236164


namespace appropriate_sampling_methods_l2361_236170

/-- Represents different sampling methods -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents a survey with its characteristics -/
structure Survey where
  totalUnits : ℕ
  sampleSize : ℕ
  hasSignificantDifferences : Bool

/-- Determines the most appropriate sampling method for a given survey -/
def mostAppropriateSamplingMethod (s : Survey) : SamplingMethod :=
  if s.hasSignificantDifferences then
    SamplingMethod.Stratified
  else
    SamplingMethod.SimpleRandom

/-- The first survey of high school classes -/
def survey1 : Survey :=
  { totalUnits := 15
  , sampleSize := 2
  , hasSignificantDifferences := false }

/-- The second survey of stores in the city -/
def survey2 : Survey :=
  { totalUnits := 1500
  , sampleSize := 15
  , hasSignificantDifferences := true }

theorem appropriate_sampling_methods :
  (mostAppropriateSamplingMethod survey1 = SamplingMethod.SimpleRandom) ∧
  (mostAppropriateSamplingMethod survey2 = SamplingMethod.Stratified) := by
  sorry

end appropriate_sampling_methods_l2361_236170


namespace circle_area_with_diameter_10_l2361_236151

theorem circle_area_with_diameter_10 (π : ℝ) :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end circle_area_with_diameter_10_l2361_236151


namespace extrema_not_necessarily_unique_l2361_236158

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define what it means for a point to be an extremum
def IsExtremum (f : RealFunction) (x : ℝ) (a b : ℝ) : Prop :=
  ∀ y ∈ Set.Icc a b, f x ≥ f y ∨ f x ≤ f y

-- Theorem statement
theorem extrema_not_necessarily_unique :
  ∃ (f : RealFunction) (a b x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧ a < x₁ ∧ x₁ < b ∧ a < x₂ ∧ x₂ < b ∧
    IsExtremum f x₁ a b ∧ IsExtremum f x₂ a b :=
sorry

end extrema_not_necessarily_unique_l2361_236158


namespace expand_and_simplify_simplify_complex_fraction_l2361_236138

-- Problem 1
theorem expand_and_simplify (x : ℝ) :
  (2*x - 1)*(2*x - 3) - (1 - 2*x)*(2 - x) = 2*x^2 - 3*x + 1 := by sorry

-- Problem 2
theorem simplify_complex_fraction (a : ℝ) (ha : a ≠ 0) (ha1 : a ≠ 1) :
  (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) := by sorry

end expand_and_simplify_simplify_complex_fraction_l2361_236138


namespace quadratic_no_real_roots_probability_l2361_236167

/-- The type of integers with absolute value less than or equal to 5 -/
def IntWithinFive : Type := {n : ℤ // n.natAbs ≤ 5}

/-- The sample space of ordered pairs (b, c) -/
def SampleSpace : Type := IntWithinFive × IntWithinFive

/-- Predicate for when a quadratic equation has no real roots -/
def NoRealRoots (p : SampleSpace) : Prop :=
  let b := p.1.val
  let c := p.2.val
  b^2 < 4*c

/-- The number of elements in the sample space -/
def TotalCount : ℕ := 121

/-- The count of pairs (b, c) where the quadratic has no real roots -/
def FavorableCount : ℕ := 70

/-- The probability of the quadratic having no real roots -/
def Probability : ℚ := FavorableCount / TotalCount

theorem quadratic_no_real_roots_probability :
  Probability = 70 / 121 := by sorry

end quadratic_no_real_roots_probability_l2361_236167


namespace special_triangle_perimeter_l2361_236108

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a = t.b + 1 ∧ t.c = t.b - 1 ∧ t.A = 2 * t.C

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

/-- The main theorem to prove -/
theorem special_triangle_perimeter :
  ∀ t : Triangle, SpecialTriangle t → perimeter t = 15 := by
  sorry

end special_triangle_perimeter_l2361_236108


namespace remainder_sum_mod15_l2361_236179

theorem remainder_sum_mod15 (p q : ℤ) 
  (hp : p % 60 = 53) 
  (hq : q % 75 = 24) : 
  (p + q) % 15 = 2 := by sorry

end remainder_sum_mod15_l2361_236179


namespace missing_number_solution_l2361_236106

theorem missing_number_solution : ∃ x : ℤ, 10111 - 10 * x * 5 = 10011 ∧ x = 2 := by sorry

end missing_number_solution_l2361_236106


namespace total_money_found_l2361_236129

-- Define the value of each coin type in cents
def quarter_value : ℕ := 25
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def penny_value : ℕ := 1

-- Define the number of each coin type found
def quarters_found : ℕ := 10
def dimes_found : ℕ := 3
def nickels_found : ℕ := 3
def pennies_found : ℕ := 5

-- Theorem to prove
theorem total_money_found :
  (quarters_found * quarter_value +
   dimes_found * dime_value +
   nickels_found * nickel_value +
   pennies_found * penny_value) = 300 := by
  sorry

end total_money_found_l2361_236129


namespace johns_donation_l2361_236130

/-- Given 6 initial contributions and a new contribution that increases the average by 50% to $75, prove that the new contribution is $225. -/
theorem johns_donation (initial_contributions : ℕ) (new_average : ℚ) : 
  initial_contributions = 6 ∧ 
  new_average = 75 ∧ 
  new_average = (3/2) * (300 / initial_contributions) →
  ∃ (johns_contribution : ℚ), 
    johns_contribution = 225 ∧
    new_average = (300 + johns_contribution) / (initial_contributions + 1) :=
by sorry

end johns_donation_l2361_236130


namespace club_size_after_four_years_l2361_236135

/-- Represents the number of people in the club after k years -/
def club_size (k : ℕ) : ℕ :=
  match k with
  | 0 => 8
  | n + 1 => 2 * club_size n - 2

/-- Theorem stating that the club size after 4 years is 98 -/
theorem club_size_after_four_years :
  club_size 4 = 98 := by
  sorry

end club_size_after_four_years_l2361_236135


namespace division_problem_l2361_236146

theorem division_problem (a b q : ℕ) 
  (h1 : a - b = 1390) 
  (h2 : a = 1650) 
  (h3 : a = b * q + 15) : q = 6 := by
  sorry

end division_problem_l2361_236146


namespace quadratic_equation_distinct_roots_l2361_236134

theorem quadratic_equation_distinct_roots (k : ℝ) :
  k = 1 → ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 2 * x₁^2 - k = 0 ∧ 2 * x₂^2 - k = 0 :=
by sorry

end quadratic_equation_distinct_roots_l2361_236134


namespace krishans_money_l2361_236181

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount,
    prove that Krishan has Rs. 3468. -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 588 →
  krishan = 3468 := by
sorry

end krishans_money_l2361_236181


namespace A_initial_investment_l2361_236186

/-- Represents the initial investment of A in rupees -/
def A_investment : ℝ := sorry

/-- Represents B's investment in rupees -/
def B_investment : ℝ := 21000

/-- Represents the number of months A invested -/
def A_months : ℝ := 12

/-- Represents the number of months B invested -/
def B_months : ℝ := 3

/-- Represents A's share in the profit ratio -/
def A_share : ℝ := 2

/-- Represents B's share in the profit ratio -/
def B_share : ℝ := 3

/-- Theorem stating that A's initial investment is 3500 rupees -/
theorem A_initial_investment : 
  (A_investment * A_months) / (B_investment * B_months) = A_share / B_share → 
  A_investment = 3500 := by sorry

end A_initial_investment_l2361_236186


namespace rahul_deepak_age_ratio_l2361_236168

/-- Proves that the ratio of Rahul's current age to Deepak's current age is 4:3 -/
theorem rahul_deepak_age_ratio :
  let rahul_future_age : ℕ := 42
  let years_until_future : ℕ := 6
  let deepak_current_age : ℕ := 27
  let rahul_current_age : ℕ := rahul_future_age - years_until_future
  (rahul_current_age : ℚ) / deepak_current_age = 4 / 3 := by
  sorry

end rahul_deepak_age_ratio_l2361_236168


namespace smaller_number_in_ratio_l2361_236111

theorem smaller_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 3 / 5 → x + y + 10 = 50 → min x y = 15 := by
  sorry

end smaller_number_in_ratio_l2361_236111


namespace no_real_solutions_l2361_236125

theorem no_real_solutions : ∀ x : ℝ, x^2 ≠ 4 → x ≠ 2 → x ≠ -2 → 
  (8*x)/(x^2 - 4) ≠ (3*x)/(x - 2) - 4/(x + 2) := by
  sorry

end no_real_solutions_l2361_236125


namespace largest_difference_l2361_236105

def Digits : Finset ℕ := {1, 3, 7, 8, 9}

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 1000 ∧ a < 10000 ∧ b ≥ 100 ∧ b < 1000 ∧
  (Finset.card (Finset.filter (λ d => d ∈ Digits) (Finset.range 10)) = 5) ∧
  (∀ d ∈ Digits, (d ∈ Finset.filter (λ x => x ∈ Digits) (Finset.range 10)))

theorem largest_difference :
  ∃ (a b : ℕ), is_valid_pair a b ∧
    ∀ (x y : ℕ), is_valid_pair x y → (a - b ≥ x - y) ∧ (a - b = 9868) :=
by sorry

end largest_difference_l2361_236105


namespace reading_time_difference_l2361_236115

/-- The difference in reading time between two people reading the same book -/
theorem reading_time_difference (xanthia_rate molly_rate book_pages : ℕ) : 
  xanthia_rate = 150 → 
  molly_rate = 75 → 
  book_pages = 300 → 
  (book_pages / molly_rate - book_pages / xanthia_rate) * 60 = 120 := by
  sorry

end reading_time_difference_l2361_236115


namespace reciprocal_opposite_equation_l2361_236121

theorem reciprocal_opposite_equation (a b c d : ℝ) 
  (h1 : a * b = 1)  -- a and b are reciprocals
  (h2 : c + d = 0)  -- c and d are opposites
  : (a * b) ^ 4 - 3 * (c + d) ^ 3 = 1 := by
  sorry

end reciprocal_opposite_equation_l2361_236121


namespace power_multiplication_l2361_236142

theorem power_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end power_multiplication_l2361_236142


namespace inequality_proof_l2361_236143

theorem inequality_proof (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end inequality_proof_l2361_236143


namespace circle_properties_l2361_236119

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y + 4 = 0) :
  (∃ (k : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → y'/x' ≤ k ∧ k = 4/3) ∧
  (∃ (m : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → y'/x' ≥ m ∧ m = 0) ∧
  (∃ (M : ℝ), ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 2*y' + 4 = 0 → x' + y' ≤ M ∧ M = 3 + Real.sqrt 2) :=
by sorry

end circle_properties_l2361_236119


namespace z_max_min_l2361_236137

def z (x y : ℝ) : ℝ := 2 * x + y

theorem z_max_min (x y : ℝ) (h1 : x + y ≤ 2) (h2 : x ≥ 1) (h3 : y ≥ 0) :
  (∀ a b : ℝ, a + b ≤ 2 → a ≥ 1 → b ≥ 0 → z a b ≤ 4) ∧
  (∀ a b : ℝ, a + b ≤ 2 → a ≥ 1 → b ≥ 0 → z a b ≥ 2) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ a ≥ 1 ∧ b ≥ 0 ∧ z a b = 4) ∧
  (∃ a b : ℝ, a + b ≤ 2 ∧ a ≥ 1 ∧ b ≥ 0 ∧ z a b = 2) :=
by sorry

end z_max_min_l2361_236137


namespace ruler_cost_l2361_236175

theorem ruler_cost (total_students : ℕ) (buyers : ℕ) (rulers_per_student : ℕ) (ruler_cost : ℕ) :
  total_students = 36 →
  buyers > total_students / 2 →
  rulers_per_student > 1 →
  ruler_cost > rulers_per_student →
  buyers * rulers_per_student * ruler_cost = 1729 →
  ruler_cost = 13 :=
by sorry

end ruler_cost_l2361_236175


namespace exists_disjoint_graphs_l2361_236157

open Set

/-- The graph of a function f: [0, 1] → ℝ -/
def Graph (f : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ Icc 0 1 ∧ p.2 = f p.1}

/-- The graph of the translated function f(x-a) -/
def GraphTranslated (f : ℝ → ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ∈ Icc a (a+1) ∧ p.2 = f (p.1 - a)}

theorem exists_disjoint_graphs :
  ∀ a ∈ Ioo 0 1, ∃ f : ℝ → ℝ,
    Continuous f ∧
    f 0 = 0 ∧ f 1 = 0 ∧
    (Graph f) ∩ (GraphTranslated f a) = ∅ :=
sorry

end exists_disjoint_graphs_l2361_236157


namespace quadrilateral_perimeter_quadrilateral_perimeter_proof_l2361_236107

/-- The perimeter of a quadrilateral with vertices A(0,0), B(0,10), C(8,10), and D(8,0) is 36 -/
theorem quadrilateral_perimeter : ℝ → Prop :=
  fun perimeter =>
    let A : ℝ × ℝ := (0, 0)
    let B : ℝ × ℝ := (0, 10)
    let C : ℝ × ℝ := (8, 10)
    let D : ℝ × ℝ := (8, 0)
    let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
    let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
    let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
    let DA := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
    perimeter = AB + BC + CD + DA ∧ perimeter = 36

/-- Proof of the theorem -/
theorem quadrilateral_perimeter_proof : quadrilateral_perimeter 36 := by
  sorry

end quadrilateral_perimeter_quadrilateral_perimeter_proof_l2361_236107


namespace greg_needs_61_60_l2361_236109

/-- Calculates the additional amount Greg needs to buy a scooter, helmet, and lock -/
def additional_amount_needed (scooter_price helmet_price lock_price discount_rate tax_rate gift_card savings : ℚ) : ℚ :=
  let discounted_scooter := scooter_price * (1 - discount_rate)
  let subtotal := discounted_scooter + helmet_price + lock_price
  let total_with_tax := subtotal * (1 + tax_rate)
  let final_price := total_with_tax - gift_card
  final_price - savings

/-- Theorem stating that Greg needs $61.60 more -/
theorem greg_needs_61_60 :
  additional_amount_needed 90 30 15 0.1 0.1 20 57 = 61.6 := by
  sorry

end greg_needs_61_60_l2361_236109


namespace base6_division_theorem_l2361_236176

/-- Convert a number from base 6 to base 10 -/
def base6ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Convert a number from base 10 to base 6 -/
def base10ToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 6) ((m % 6) :: acc)
  aux n []

/-- Perform division in base 6 -/
def divBase6 (a b : List Nat) : List Nat × Nat :=
  let a10 := base6ToBase10 a
  let b10 := base6ToBase10 b
  let q := a10 / b10
  let r := a10 % b10
  (base10ToBase6 q, r)

theorem base6_division_theorem :
  let a := [3, 2, 1, 2]  -- 2123 in base 6
  let b := [3, 2]        -- 23 in base 6
  let (q, r) := divBase6 a b
  q = [2, 5] ∧ r = 3 := by
  sorry

end base6_division_theorem_l2361_236176


namespace condition_D_iff_right_triangle_l2361_236120

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

/-- The condition a² = b² - c² -/
def condition_D (t : Triangle) : Prop :=
  t.a^2 = t.b^2 - t.c^2

/-- Theorem stating that condition D is equivalent to the triangle being a right triangle -/
theorem condition_D_iff_right_triangle (t : Triangle) :
  condition_D t ↔ is_right_triangle t :=
sorry

end condition_D_iff_right_triangle_l2361_236120


namespace cube_volume_from_surface_area_l2361_236192

-- Define the surface area of the cube
def surface_area : ℝ := 864

-- Theorem stating the relationship between surface area and volume
theorem cube_volume_from_surface_area :
  ∃ (side_length : ℝ), 
    side_length > 0 ∧ 
    6 * side_length^2 = surface_area ∧ 
    side_length^3 = 1728 := by
  sorry

end cube_volume_from_surface_area_l2361_236192


namespace carla_restock_theorem_l2361_236163

/-- Represents the food bank inventory and distribution problem -/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day2_people : ℕ
  day2_cans_per_person : ℕ
  final_restock : ℕ
  total_given_away : ℕ

/-- Calculates the number of cans restocked after the first day -/
def cans_restocked_after_day1 (fb : FoodBank) : ℕ :=
  fb.total_given_away - (fb.initial_stock - fb.day1_people * fb.day1_cans_per_person) +
  (fb.final_restock - fb.day2_people * fb.day2_cans_per_person)

/-- Theorem stating that Carla restocked 2000 cans after the first day -/
theorem carla_restock_theorem (fb : FoodBank)
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day2_people = 1000)
  (h5 : fb.day2_cans_per_person = 2)
  (h6 : fb.final_restock = 3000)
  (h7 : fb.total_given_away = 2500) :
  cans_restocked_after_day1 fb = 2000 := by
  sorry

end carla_restock_theorem_l2361_236163


namespace largest_minus_smallest_l2361_236182

def problem (A B C : ℤ) : Prop :=
  A = 10 * 2 + 9 ∧
  A = B + 16 ∧
  C = B * 3

theorem largest_minus_smallest (A B C : ℤ) 
  (h : problem A B C) : 
  max A (max B C) - min A (min B C) = 26 := by
  sorry

end largest_minus_smallest_l2361_236182


namespace half_angle_quadrant_l2361_236124

def is_in_second_quadrant (α : Real) : Prop :=
  ∃ k : Int, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

def is_in_first_or_third_quadrant (α : Real) : Prop :=
  ∃ n : Int, (2 * n * Real.pi + Real.pi / 4 < α ∧ α < 2 * n * Real.pi + Real.pi / 2) ∨
             ((2 * n + 1) * Real.pi + Real.pi / 4 < α ∧ α < (2 * n + 1) * Real.pi + Real.pi / 2)

theorem half_angle_quadrant (α : Real) :
  is_in_second_quadrant α → is_in_first_or_third_quadrant (α / 2) := by
  sorry

end half_angle_quadrant_l2361_236124


namespace tourist_ratio_l2361_236197

theorem tourist_ratio (initial_tourists : ℕ) (eaten_by_anaconda : ℕ) (final_tourists : ℕ) :
  initial_tourists = 30 →
  eaten_by_anaconda = 2 →
  final_tourists = 16 →
  ∃ (poisoned_tourists : ℕ),
    poisoned_tourists * 1 = (initial_tourists - eaten_by_anaconda - final_tourists) * 2 :=
by sorry

end tourist_ratio_l2361_236197


namespace tickets_left_l2361_236172

/-- The number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- The number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := 25

/-- The number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- Theorem: Given the conditions, Tom has 50 tickets left -/
theorem tickets_left : 
  whack_a_mole_tickets + skee_ball_tickets - spent_tickets = 50 := by
  sorry

end tickets_left_l2361_236172


namespace volleyball_tournament_wins_l2361_236188

theorem volleyball_tournament_wins (n : ℕ) (h_n : n = 73) :
  ∀ (p m : ℕ) (x : ℕ) (h_x : 0 < x ∧ x < n),
  x * p + (n - x) * m = n * (n - 1) / 2 →
  p = m :=
by sorry

end volleyball_tournament_wins_l2361_236188


namespace arithmetic_sequence_problem_l2361_236177

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_4 + a_6 + a_8 = 12,
    prove that a_8 - (1/2)a_10 = 2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 4 + a 6 + a 8 = 12) :
  a 8 - (1/2) * a 10 = 2 := by
  sorry


end arithmetic_sequence_problem_l2361_236177


namespace smallest_n_for_negative_sum_l2361_236194

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℤ := 7 - 2 * (n - 1)
def S (n : ℕ) : ℤ := n * (2 * 7 + (n - 1) * (-2)) / 2

-- State the theorem
theorem smallest_n_for_negative_sum :
  (∀ k < 9, S k ≥ 0) ∧ (S 9 < 0) := by sorry

end smallest_n_for_negative_sum_l2361_236194


namespace min_sum_sequence_l2361_236127

theorem min_sum_sequence (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 → D > 0 →
  (∃ r : ℚ, C - B = B - A ∧ C / B = r ∧ D / C = r) →
  C / B = 7 / 3 →
  A + B + C + D ≥ 76 :=
by sorry

end min_sum_sequence_l2361_236127
