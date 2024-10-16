import Mathlib

namespace NUMINAMATH_CALUDE_mile_to_rod_l3161_316143

-- Define the conversion factors
def mile_to_furlong : ℝ := 8
def furlong_to_pace : ℝ := 220
def pace_to_rod : ℝ := 0.2

-- Theorem statement
theorem mile_to_rod : 
  1 * mile_to_furlong * furlong_to_pace * pace_to_rod = 352 := by
  sorry

end NUMINAMATH_CALUDE_mile_to_rod_l3161_316143


namespace NUMINAMATH_CALUDE_circle_properties_l3161_316135

theorem circle_properties (A : ℝ) (h : A = 4 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ A = Real.pi * r^2 ∧ 2 * r = 4 ∧ 2 * Real.pi * r = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3161_316135


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l3161_316187

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 9 = 7 ∧ ∀ m : ℕ, m < 100 ∧ m % 9 = 7 → m ≤ n ↔ n = 97 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l3161_316187


namespace NUMINAMATH_CALUDE_complex_fourth_power_integer_count_l3161_316131

theorem complex_fourth_power_integer_count : 
  ∃! (n : ℤ), ∃ (m : ℤ), (n + 2 * Complex.I) ^ 4 = m := by sorry

end NUMINAMATH_CALUDE_complex_fourth_power_integer_count_l3161_316131


namespace NUMINAMATH_CALUDE_production_average_proof_l3161_316175

/-- Calculates the new average daily production after adding a new day's production -/
def newAverageProduction (n : ℕ) (oldAverage : ℚ) (newProduction : ℚ) : ℚ :=
  ((n : ℚ) * oldAverage + newProduction) / ((n : ℚ) + 1)

theorem production_average_proof :
  let n : ℕ := 4
  let oldAverage : ℚ := 50
  let newProduction : ℚ := 90
  newAverageProduction n oldAverage newProduction = 58 := by
sorry

end NUMINAMATH_CALUDE_production_average_proof_l3161_316175


namespace NUMINAMATH_CALUDE_davids_english_marks_l3161_316179

def davidsMathMarks : ℕ := 65
def davidsPhysicsMarks : ℕ := 82
def davidsChemistryMarks : ℕ := 67
def davidsBiologyMarks : ℕ := 85
def davidsAverageMarks : ℕ := 76
def totalSubjects : ℕ := 5

theorem davids_english_marks :
  ∃ (englishMarks : ℕ), 
    (englishMarks + davidsMathMarks + davidsPhysicsMarks + davidsChemistryMarks + davidsBiologyMarks) / totalSubjects = davidsAverageMarks ∧
    englishMarks = 81 :=
by sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3161_316179


namespace NUMINAMATH_CALUDE_simple_interest_rate_l3161_316105

/-- Simple interest rate calculation -/
theorem simple_interest_rate 
  (principal : ℝ) 
  (final_amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 800)
  (h2 : final_amount = 950)
  (h3 : time = 5)
  : (final_amount - principal) * 100 / (principal * time) = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l3161_316105


namespace NUMINAMATH_CALUDE_function_root_implies_a_range_l3161_316108

theorem function_root_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, -1 < x ∧ x < 1 ∧ a * x + 1 = 0) → (a < -1 ∨ a > 1) := by
  sorry

end NUMINAMATH_CALUDE_function_root_implies_a_range_l3161_316108


namespace NUMINAMATH_CALUDE_annas_meal_cost_difference_l3161_316186

/-- Represents the cost of Anna's meals -/
def annas_meals (bagel_price cream_cheese_price orange_juice_price orange_juice_discount
                 sandwich_price avocado_price milk_price milk_discount : ℚ) : ℚ :=
  let breakfast_cost := bagel_price + cream_cheese_price + orange_juice_price * (1 - orange_juice_discount)
  let lunch_cost := sandwich_price + avocado_price + milk_price * (1 - milk_discount)
  lunch_cost - breakfast_cost

/-- The difference between Anna's lunch and breakfast costs is $4.14 -/
theorem annas_meal_cost_difference :
  annas_meals 0.95 0.50 1.25 0.32 4.65 0.75 1.15 0.10 = 4.14 := by
  sorry

end NUMINAMATH_CALUDE_annas_meal_cost_difference_l3161_316186


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3161_316140

theorem completing_square_equivalence (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ (x + 2)^2 = 5 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3161_316140


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l3161_316173

theorem quadratic_root_transformation (p q r : ℝ) (u v : ℝ) :
  (p * u^2 + q * u + r = 0) ∧ (p * v^2 + q * v + r = 0) →
  ((q * u + p)^2 - p * (q * u + p) + q * r = 0) ∧ ((q * v + p)^2 - p * (q * v + p) + q * r = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l3161_316173


namespace NUMINAMATH_CALUDE_select_perfect_square_l3161_316153

theorem select_perfect_square (nums : Finset ℕ) (h_card : nums.card = 48) 
  (h_prime_factors : (nums.prod id).factors.toFinset.card = 10) :
  ∃ (a b c d : ℕ), a ∈ nums ∧ b ∈ nums ∧ c ∈ nums ∧ d ∈ nums ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ (m : ℕ), a * b * c * d = m ^ 2 := by
sorry


end NUMINAMATH_CALUDE_select_perfect_square_l3161_316153


namespace NUMINAMATH_CALUDE_elmer_eats_more_l3161_316165

/-- The amount of food each animal eats per day in pounds -/
structure AnimalFood where
  penelope : ℝ
  greta : ℝ
  milton : ℝ
  elmer : ℝ
  rosie : ℝ
  carl : ℝ

/-- The conditions given in the problem -/
def satisfiesConditions (food : AnimalFood) : Prop :=
  food.penelope = 20 ∧
  food.penelope = 10 * food.greta ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.rosie = 3 * food.greta ∧
  food.carl = food.penelope / 2 ∧
  food.carl = 5 * food.greta

/-- The theorem to prove -/
theorem elmer_eats_more (food : AnimalFood) (h : satisfiesConditions food) :
    food.elmer - (food.penelope + food.greta + food.milton + food.rosie + food.carl) = 41.98 := by
  sorry

end NUMINAMATH_CALUDE_elmer_eats_more_l3161_316165


namespace NUMINAMATH_CALUDE_complex_number_equality_l3161_316176

theorem complex_number_equality : Complex.I * (1 - Complex.I)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3161_316176


namespace NUMINAMATH_CALUDE_marcus_car_mileage_l3161_316144

/-- Calculates the final mileage of a car after a road trip --/
def final_mileage (initial_mileage : ℕ) (tank_capacity : ℕ) (fuel_efficiency : ℕ) (refills : ℕ) : ℕ :=
  initial_mileage + tank_capacity * refills * fuel_efficiency

/-- Theorem stating the final mileage of Marcus' car after the road trip --/
theorem marcus_car_mileage :
  final_mileage 1728 20 30 2 = 2928 := by
  sorry

#eval final_mileage 1728 20 30 2

end NUMINAMATH_CALUDE_marcus_car_mileage_l3161_316144


namespace NUMINAMATH_CALUDE_expand_expression_l3161_316162

theorem expand_expression (y z : ℝ) : 
  -2 * (5 * y^3 - 3 * y^2 * z + 4 * y * z^2 - z^3) = 
  -10 * y^3 + 6 * y^2 * z - 8 * y * z^2 + 2 * z^3 := by
sorry

end NUMINAMATH_CALUDE_expand_expression_l3161_316162


namespace NUMINAMATH_CALUDE_race_finish_times_l3161_316102

/-- Race problem statement -/
theorem race_finish_times 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (lila_speed : ℝ) 
  (race_distance : ℝ) 
  (h1 : malcolm_speed = 6)
  (h2 : joshua_speed = 8)
  (h3 : lila_speed = 7)
  (h4 : race_distance = 12) :
  let malcolm_time := malcolm_speed * race_distance
  let joshua_time := joshua_speed * race_distance
  let lila_time := lila_speed * race_distance
  (joshua_time - malcolm_time = 24 ∧ lila_time - malcolm_time = 12) := by
  sorry


end NUMINAMATH_CALUDE_race_finish_times_l3161_316102


namespace NUMINAMATH_CALUDE_score_mode_is_85_l3161_316196

/-- Represents a score in the stem-and-leaf plot -/
structure Score where
  stem : Nat
  leaf : Nat

/-- The set of scores from the stem-and-leaf plot -/
def scores : List Score := [
  ⟨6, 1⟩, ⟨6, 1⟩, ⟨6, 1⟩,
  ⟨7, 2⟩, ⟨7, 5⟩,
  ⟨8, 3⟩, ⟨8, 5⟩, ⟨8, 5⟩, ⟨8, 5⟩, ⟨8, 7⟩, ⟨8, 7⟩,
  ⟨9, 0⟩, ⟨9, 2⟩, ⟨9, 2⟩, ⟨9, 4⟩, ⟨9, 6⟩, ⟨9, 6⟩,
  ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 1⟩, ⟨10, 3⟩,
  ⟨11, 0⟩, ⟨11, 0⟩
]

/-- Converts a Score to its numerical value -/
def scoreValue (s : Score) : Nat := s.stem * 10 + s.leaf

/-- Finds the mode of a list of natural numbers -/
def mode (l : List Nat) : Nat := sorry

/-- The theorem stating that the mode of the scores is 85 -/
theorem score_mode_is_85 : 
  mode (scores.map scoreValue) = 85 := by sorry

end NUMINAMATH_CALUDE_score_mode_is_85_l3161_316196


namespace NUMINAMATH_CALUDE_monotonicity_and_inequality_min_m_value_l3161_316116

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

theorem monotonicity_and_inequality (a : ℝ) (h_a : a > 0) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3 → f a x₁ ≥ f a x₂) ∧
  (∀ x₁ x₂ : ℝ, 3 ≤ x₁ ∧ x₁ < x₂ → f a x₁ ≤ f a x₂) ↔
  a = 9 :=
sorry

theorem min_m_value :
  (∀ x : ℝ, x ≥ 1 ∧ x ≤ 4 → x + 9 / x - 6.25 ≤ 0) ∧
  ∀ m : ℝ, m < 6.25 →
    ∃ x : ℝ, x ≥ 1 ∧ x ≤ 4 ∧ x + 9 / x - m > 0 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_inequality_min_m_value_l3161_316116


namespace NUMINAMATH_CALUDE_order_of_numbers_l3161_316139

theorem order_of_numbers (x y z : ℝ) (h1 : 0.9 < x) (h2 : x < 1) 
  (h3 : y = x^x) (h4 : z = x^(x^x)) : x < z ∧ z < y := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3161_316139


namespace NUMINAMATH_CALUDE_cupcakes_left_l3161_316123

/-- The number of cupcakes in a dozen -/
def dozen : ℕ := 12

/-- The number of cupcakes Dani brings -/
def cupcakes_brought : ℕ := 2 * dozen + dozen / 2

/-- The total number of students -/
def total_students : ℕ := 27

/-- The number of teachers -/
def teachers : ℕ := 1

/-- The number of teacher's aids -/
def teacher_aids : ℕ := 1

/-- The number of students who called in sick -/
def sick_students : ℕ := 3

/-- Theorem: The number of cupcakes left after distribution -/
theorem cupcakes_left : 
  cupcakes_brought - (total_students - sick_students + teachers + teacher_aids) = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_l3161_316123


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l3161_316118

theorem soccer_penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) 
  (h1 : total_players = 24) 
  (h2 : goalkeepers = 4) 
  (h3 : goalkeepers ≤ total_players) : 
  (total_players - 1) * goalkeepers = 92 := by
  sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l3161_316118


namespace NUMINAMATH_CALUDE_nick_quarters_count_l3161_316181

-- Define the total number of quarters
def total_quarters : ℕ := 35

-- Define the fraction of state quarters
def state_quarter_fraction : ℚ := 2 / 5

-- Define the fraction of Pennsylvania quarters among state quarters
def pennsylvania_quarter_fraction : ℚ := 1 / 2

-- Define the number of Pennsylvania quarters
def pennsylvania_quarters : ℕ := 7

-- Theorem statement
theorem nick_quarters_count :
  (pennsylvania_quarter_fraction * state_quarter_fraction * total_quarters : ℚ) = pennsylvania_quarters :=
by sorry

end NUMINAMATH_CALUDE_nick_quarters_count_l3161_316181


namespace NUMINAMATH_CALUDE_jeans_and_shirts_cost_l3161_316199

/-- The cost of one pair of jeans -/
def jean_cost : ℝ := 11

/-- The cost of one shirt -/
def shirt_cost : ℝ := 18

/-- The cost of 2 pairs of jeans and 3 shirts -/
def cost_2j_3s : ℝ := 76

/-- The cost of 3 pairs of jeans and 2 shirts -/
def cost_3j_2s : ℝ := 3 * jean_cost + 2 * shirt_cost

theorem jeans_and_shirts_cost : cost_3j_2s = 69 := by
  sorry

end NUMINAMATH_CALUDE_jeans_and_shirts_cost_l3161_316199


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3161_316183

def A : Set ℝ := {3, 5}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem subset_implies_a_values (a : ℝ) (h : B a ⊆ A) : a = 0 ∨ a = 1/3 ∨ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3161_316183


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3161_316100

theorem complex_equation_sum (a b : ℝ) (h : (3 * b : ℂ) + (2 * a - 2) * Complex.I = 1 - Complex.I) : 
  a + b = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3161_316100


namespace NUMINAMATH_CALUDE_distance_between_homes_l3161_316164

/-- Proves the distance between Maxwell's and Brad's homes given their speeds and meeting point -/
theorem distance_between_homes
  (maxwell_speed : ℝ)
  (brad_speed : ℝ)
  (maxwell_distance : ℝ)
  (h1 : maxwell_speed = 2)
  (h2 : brad_speed = 3)
  (h3 : maxwell_distance = 26)
  (h4 : maxwell_distance / maxwell_speed = (total_distance - maxwell_distance) / brad_speed) :
  total_distance = 65 :=
by
  sorry

#check distance_between_homes

end NUMINAMATH_CALUDE_distance_between_homes_l3161_316164


namespace NUMINAMATH_CALUDE_third_year_percentage_l3161_316189

theorem third_year_percentage
  (total : ℝ)
  (third_year : ℝ)
  (second_year : ℝ)
  (h1 : second_year = 0.1 * total)
  (h2 : second_year / (total - third_year) = 1 / 7)
  : third_year = 0.3 * total :=
by sorry

end NUMINAMATH_CALUDE_third_year_percentage_l3161_316189


namespace NUMINAMATH_CALUDE_oil_price_reduction_l3161_316174

/-- Proves that given a 35% reduction in oil price allowing 5 kg more for Rs. 800, the reduced price is Rs. 36.4 per kg -/
theorem oil_price_reduction (original_price : ℝ) : 
  (800 / (0.65 * original_price) - 800 / original_price = 5) →
  (0.65 * original_price = 36.4) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l3161_316174


namespace NUMINAMATH_CALUDE_train_length_problem_l3161_316178

/-- Given a train traveling at constant speed through a tunnel and over a bridge,
    prove that the length of the train is 200m. -/
theorem train_length_problem (tunnel_length : ℝ) (tunnel_time : ℝ) (bridge_length : ℝ) (bridge_time : ℝ)
    (h1 : tunnel_length = 860)
    (h2 : tunnel_time = 22)
    (h3 : bridge_length = 790)
    (h4 : bridge_time = 33)
    (h5 : (bridge_length + x) / bridge_time = (tunnel_length - x) / tunnel_time) :
    x = 200 := by
  sorry

#check train_length_problem

end NUMINAMATH_CALUDE_train_length_problem_l3161_316178


namespace NUMINAMATH_CALUDE_smallest_marble_count_l3161_316190

theorem smallest_marble_count : ∃ (M : ℕ), 
  M > 1 ∧
  M % 5 = 1 ∧
  M % 7 = 1 ∧
  M % 11 = 1 ∧
  M % 4 = 2 ∧
  (∀ (N : ℕ), N > 1 ∧ N % 5 = 1 ∧ N % 7 = 1 ∧ N % 11 = 1 ∧ N % 4 = 2 → M ≤ N) ∧
  M = 386 := by
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l3161_316190


namespace NUMINAMATH_CALUDE_class_project_total_l3161_316177

/-- Calculates the total amount gathered for a class project with discounts and fees -/
theorem class_project_total (total_students : ℕ) (full_price : ℚ) 
  (full_paying : ℕ) (high_merit : ℕ) (financial_needs : ℕ) (special_discount : ℕ)
  (high_merit_discount : ℚ) (financial_needs_discount : ℚ) (special_discount_rate : ℚ)
  (admin_fee : ℚ) :
  total_students = 35 →
  full_price = 50 →
  full_paying = 20 →
  high_merit = 5 →
  financial_needs = 7 →
  special_discount = 3 →
  high_merit_discount = 25 / 100 →
  financial_needs_discount = 1 / 2 →
  special_discount_rate = 10 / 100 →
  admin_fee = 100 →
  (full_paying * full_price + 
   high_merit * (full_price * (1 - high_merit_discount)) +
   financial_needs * (full_price * financial_needs_discount) +
   special_discount * (full_price * (1 - special_discount_rate))) - admin_fee = 1397.5 := by
  sorry


end NUMINAMATH_CALUDE_class_project_total_l3161_316177


namespace NUMINAMATH_CALUDE_tank_filling_time_l3161_316133

theorem tank_filling_time (fill_rate : ℝ → ℝ → ℝ) :
  (∀ (n : ℝ), fill_rate n (8 * n / 3) = 1) →
  fill_rate 2 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3161_316133


namespace NUMINAMATH_CALUDE_decimal_8543_to_base7_l3161_316155

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: decimal_to_base7 (n / 7)

def base7_to_decimal (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 7 * acc) 0

theorem decimal_8543_to_base7 :
  decimal_to_base7 8543 = [3, 2, 6, 3, 3] ∧
  base7_to_decimal [3, 2, 6, 3, 3] = 8543 := by
  sorry

end NUMINAMATH_CALUDE_decimal_8543_to_base7_l3161_316155


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3161_316132

/-- Given that the solution set of x^2 + ax + b < 0 is (1, 2), 
    prove that the solution set of bx^2 + ax + 1 > 0 is (-∞, 1/2) ∪ (1, +∞) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, b*x^2 + a*x + 1 > 0 ↔ x < (1/2) ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3161_316132


namespace NUMINAMATH_CALUDE_neglart_hands_count_l3161_316163

/-- Represents a race on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of toes on each hand for a given race -/
def toes_per_hand (race : Race) : ℕ :=
  match race with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of hands for Hoopits -/
def hoopit_hands : ℕ := 4

/-- Number of Hoopit students on the bus -/
def hoopit_students : ℕ := 7

/-- Number of Neglart students on the bus -/
def neglart_students : ℕ := 8

/-- Total number of toes on the bus -/
def total_toes : ℕ := 164

/-- Theorem stating the number of hands each Neglart has -/
theorem neglart_hands_count :
  ∃ (neglart_hands : ℕ),
    neglart_hands * neglart_students * toes_per_hand Race.Neglart +
    hoopit_hands * hoopit_students * toes_per_hand Race.Hoopit = total_toes ∧
    neglart_hands = 5 := by
  sorry

end NUMINAMATH_CALUDE_neglart_hands_count_l3161_316163


namespace NUMINAMATH_CALUDE_tetrahedron_sum_is_15_l3161_316161

/-- Represents a tetrahedron -/
structure Tetrahedron where
  edges : ℕ
  vertices : ℕ
  faces : ℕ

/-- The properties of a tetrahedron -/
def is_tetrahedron (t : Tetrahedron) : Prop :=
  t.edges = 6 ∧ t.vertices = 4 ∧ t.faces = 4

/-- The sum calculation with one vertex counted twice -/
def sum_with_extra_vertex (t : Tetrahedron) : ℕ :=
  t.edges + (t.vertices + 1) + t.faces

/-- Theorem: The sum of edges, faces, and vertices (with one counted twice) of a tetrahedron is 15 -/
theorem tetrahedron_sum_is_15 (t : Tetrahedron) (h : is_tetrahedron t) :
  sum_with_extra_vertex t = 15 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sum_is_15_l3161_316161


namespace NUMINAMATH_CALUDE_complex_number_properties_l3161_316170

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := m^2 - 2*m + m*Complex.I

/-- The line x - y + 2 = 0 -/
def line (z : ℂ) : Prop := z.re - z.im + 2 = 0

/-- z is in the second quadrant -/
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

theorem complex_number_properties :
  ∀ m : ℝ,
  (second_quadrant (z m) ∧ line (z m)) ↔ (m = 1 ∨ m = 2) ∧
  (m = 1 → Complex.abs (z m) = Real.sqrt 2) ∧
  (m = 2 → Complex.abs (z m) = 2) := by sorry

end NUMINAMATH_CALUDE_complex_number_properties_l3161_316170


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_54000_perfect_cube_l3161_316184

/-- 
A number is a perfect cube if it can be expressed as the cube of an integer.
-/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

/--
The smallest positive integer that, when multiplied by 54000, results in a perfect cube is 1.
-/
theorem smallest_multiplier_for_54000_perfect_cube :
  ∀ n : ℕ+, is_perfect_cube (54000 * n) → 1 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_54000_perfect_cube_l3161_316184


namespace NUMINAMATH_CALUDE_chord_intersection_triangle_area_l3161_316160

/-- Given two chords of a circle intersecting at a point, this theorem
    calculates the area of one triangle formed by the chords, given the
    area of the other triangle and the lengths of two segments. -/
theorem chord_intersection_triangle_area
  (PO SO : ℝ) (area_POR : ℝ) (h1 : PO = 3) (h2 : SO = 4) (h3 : area_POR = 7) :
  let area_QOS := (16 * area_POR) / (9 : ℝ)
  area_QOS = 112 / 9 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_triangle_area_l3161_316160


namespace NUMINAMATH_CALUDE_basketball_tournament_l3161_316110

theorem basketball_tournament (n : ℕ) : n * (n - 1) / 2 = 10 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tournament_l3161_316110


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3161_316195

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3161_316195


namespace NUMINAMATH_CALUDE_add_9876_seconds_to_8_45_am_l3161_316111

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : ℕ) : Time :=
  sorry

/-- The initial time (8:45:00 a.m.) -/
def initialTime : Time :=
  { hours := 8, minutes := 45, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : ℕ := 9876

/-- The expected final time (11:29:36 a.m.) -/
def expectedFinalTime : Time :=
  { hours := 11, minutes := 29, seconds := 36 }

theorem add_9876_seconds_to_8_45_am :
  addSeconds initialTime secondsToAdd = expectedFinalTime :=
sorry

end NUMINAMATH_CALUDE_add_9876_seconds_to_8_45_am_l3161_316111


namespace NUMINAMATH_CALUDE_fraction_reduced_to_lowest_terms_l3161_316130

theorem fraction_reduced_to_lowest_terms : 
  (4128 : ℚ) / 4386 = 295 / 313 := by sorry

end NUMINAMATH_CALUDE_fraction_reduced_to_lowest_terms_l3161_316130


namespace NUMINAMATH_CALUDE_log_base_three_seven_l3161_316106

theorem log_base_three_seven (a b : ℝ) (h1 : Real.log 2 / Real.log 3 = a) (h2 : Real.log 7 / Real.log 2 = b) :
  Real.log 7 / Real.log 3 = a * b := by
  sorry

end NUMINAMATH_CALUDE_log_base_three_seven_l3161_316106


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3161_316182

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E : ℝ) : 
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5) →
  A + B + C + D + E = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l3161_316182


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3161_316193

/-- Represents a parabola with equation y² = px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line with equation y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Calculates the chord length of intersection between a parabola and a line -/
def chordLength (parabola : Parabola) (line : Line) : ℝ := 
  sorry

/-- Theorem stating that if a parabola with equation y² = px intersects 
    the line y = x - 1 with a chord length of √10, then p = 1 -/
theorem parabola_intersection_theorem (parabola : Parabola) (line : Line) :
  line.m = 1 ∧ line.b = -1 → chordLength parabola line = Real.sqrt 10 → parabola.p = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3161_316193


namespace NUMINAMATH_CALUDE_jason_lost_cards_l3161_316194

/-- The number of Pokemon cards Jason lost at a tournament -/
def cards_lost (initial_cards bought_cards final_cards : ℕ) : ℕ :=
  initial_cards + bought_cards - final_cards

/-- Theorem stating that Jason lost 188 Pokemon cards at the tournament -/
theorem jason_lost_cards : cards_lost 676 224 712 = 188 := by
  sorry

end NUMINAMATH_CALUDE_jason_lost_cards_l3161_316194


namespace NUMINAMATH_CALUDE_evaluate_P_at_negative_two_l3161_316109

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x + 4

theorem evaluate_P_at_negative_two : P (-2) = -18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_P_at_negative_two_l3161_316109


namespace NUMINAMATH_CALUDE_factorization_x4_minus_81_l3161_316172

theorem factorization_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end NUMINAMATH_CALUDE_factorization_x4_minus_81_l3161_316172


namespace NUMINAMATH_CALUDE_simplify_expression_l3161_316157

theorem simplify_expression :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3161_316157


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_l3161_316159

theorem sqrt_12_minus_sqrt_3 : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_l3161_316159


namespace NUMINAMATH_CALUDE_arrangement_theorem_l3161_316166

def arrangement_count (n : ℕ) (zeros : ℕ) : ℕ :=
  if n = 27 ∧ zeros = 13 then 14
  else if n = 26 ∧ zeros = 13 then 105
  else 0

theorem arrangement_theorem (n : ℕ) (zeros : ℕ) :
  (n = 27 ∨ n = 26) ∧ zeros = 13 →
  arrangement_count n zeros = 
    (if n = 27 then 14 else 105) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_theorem_l3161_316166


namespace NUMINAMATH_CALUDE_sequence_general_term_l3161_316124

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := 2^(n-1)

theorem sequence_general_term (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n-1) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3161_316124


namespace NUMINAMATH_CALUDE_monday_sales_calculation_l3161_316180

def total_stock : ℕ := 1300
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135
def unsold_percentage : ℚ := 69.07692307692308

theorem monday_sales_calculation :
  ∃ (monday_sales : ℕ),
    monday_sales = total_stock - tuesday_sales - wednesday_sales - thursday_sales - friday_sales -
      (unsold_percentage / 100 * total_stock).floor ∧
    monday_sales = 75 := by
  sorry

end NUMINAMATH_CALUDE_monday_sales_calculation_l3161_316180


namespace NUMINAMATH_CALUDE_wrappers_collection_proof_l3161_316151

/-- The number of wrappers collected by Andy -/
def andy_wrappers : ℕ := 34

/-- The number of wrappers collected by Max -/
def max_wrappers : ℕ := 15

/-- The number of wrappers collected by Zoe -/
def zoe_wrappers : ℕ := 25

/-- The total number of wrappers collected by all three friends -/
def total_wrappers : ℕ := andy_wrappers + max_wrappers + zoe_wrappers

theorem wrappers_collection_proof : total_wrappers = 74 := by
  sorry

end NUMINAMATH_CALUDE_wrappers_collection_proof_l3161_316151


namespace NUMINAMATH_CALUDE_lemon_orange_ratio_decrease_l3161_316113

/-- Calculates the percentage decrease in the ratio of lemons to oranges --/
theorem lemon_orange_ratio_decrease 
  (initial_lemons : ℕ) 
  (initial_oranges : ℕ) 
  (final_lemons : ℕ) 
  (final_oranges : ℕ) 
  (h1 : initial_lemons = 50) 
  (h2 : initial_oranges = 60) 
  (h3 : final_lemons = 20) 
  (h4 : final_oranges = 40) :
  (1 - (final_lemons * initial_oranges) / (initial_lemons * final_oranges : ℚ)) * 100 = 40 := by
  sorry


end NUMINAMATH_CALUDE_lemon_orange_ratio_decrease_l3161_316113


namespace NUMINAMATH_CALUDE_x_less_than_one_implications_l3161_316185

theorem x_less_than_one_implications (x : ℝ) (h : x < 1) : x^3 < 1 ∧ |x| < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_one_implications_l3161_316185


namespace NUMINAMATH_CALUDE_choose_one_book_result_l3161_316136

/-- The number of ways to choose one book from a collection of Chinese, English, and Math books -/
def choose_one_book (chinese : ℕ) (english : ℕ) (math : ℕ) : ℕ :=
  chinese + english + math

/-- Theorem: Choosing one book from 10 Chinese, 7 English, and 5 Math books has 22 possibilities -/
theorem choose_one_book_result : choose_one_book 10 7 5 = 22 := by
  sorry

end NUMINAMATH_CALUDE_choose_one_book_result_l3161_316136


namespace NUMINAMATH_CALUDE_gcd_sum_fraction_eq_half_iff_special_triples_l3161_316120

/-- Given positive integers a, b, c satisfying a < b < c, prove that
    (a.gcd b + b.gcd c + c.gcd a) / (a + b + c) = 1/2
    if and only if there exists a positive integer d such that
    (a, b, c) = (d, 2*d, 3*d) or (a, b, c) = (d, 3*d, 6*d) -/
theorem gcd_sum_fraction_eq_half_iff_special_triples
  (a b c : ℕ+) (h1 : a < b) (h2 : b < c) :
  (a.gcd b + b.gcd c + c.gcd a : ℚ) / (a + b + c) = 1/2 ↔
  (∃ d : ℕ+, (a, b, c) = (d, 2*d, 3*d) ∨ (a, b, c) = (d, 3*d, 6*d)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_sum_fraction_eq_half_iff_special_triples_l3161_316120


namespace NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3161_316168

theorem solve_equation_and_evaluate (x : ℝ) : 
  (5 * x - 8 = 15 * x + 4) → (3 * (x + 10) = 26.4) := by sorry

end NUMINAMATH_CALUDE_solve_equation_and_evaluate_l3161_316168


namespace NUMINAMATH_CALUDE_sqrt_n_plus_9_equals_25_l3161_316122

theorem sqrt_n_plus_9_equals_25 (n : ℝ) : Real.sqrt (n + 9) = 25 → n = 616 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_n_plus_9_equals_25_l3161_316122


namespace NUMINAMATH_CALUDE_sarah_cans_yesterday_l3161_316125

theorem sarah_cans_yesterday (sarah_yesterday : ℕ) 
  (h1 : sarah_yesterday + (sarah_yesterday + 30) = 40 + 70 + 20) : 
  sarah_yesterday = 50 := by
  sorry

end NUMINAMATH_CALUDE_sarah_cans_yesterday_l3161_316125


namespace NUMINAMATH_CALUDE_hospital_age_l3161_316128

/-- Proves that the hospital's current age is 40 years, given Grant's current age and the relationship between their ages in 5 years. -/
theorem hospital_age (grant_current_age : ℕ) (hospital_age : ℕ) : 
  grant_current_age = 25 →
  grant_current_age + 5 = 2 / 3 * (hospital_age + 5) →
  hospital_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_hospital_age_l3161_316128


namespace NUMINAMATH_CALUDE_cos_150_deg_l3161_316126

theorem cos_150_deg : Real.cos (150 * π / 180) = - (Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_150_deg_l3161_316126


namespace NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l3161_316117

/-- Represents a mathematical problem that may or may not require conditional statements in its algorithm. -/
inductive Problem
| CommonLogarithm
| SquarePerimeter
| MaximumOfThree
| PiecewiseFunction

/-- Determines if a problem requires conditional statements in its algorithm. -/
def requiresConditionalStatements (p : Problem) : Bool :=
  match p with
  | Problem.CommonLogarithm => false
  | Problem.SquarePerimeter => false
  | Problem.MaximumOfThree => true
  | Problem.PiecewiseFunction => true

/-- The list of all problems given in the question. -/
def allProblems : List Problem :=
  [Problem.CommonLogarithm, Problem.SquarePerimeter, Problem.MaximumOfThree, Problem.PiecewiseFunction]

/-- Theorem stating that the number of problems not requiring conditional statements is 2. -/
theorem two_problems_without_conditional_statements :
  (allProblems.filter (fun p => ¬requiresConditionalStatements p)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_problems_without_conditional_statements_l3161_316117


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l3161_316145

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The equation of our specific circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

theorem circle_passes_through_points :
  ∃ (c : Circle),
    (∀ (x y : ℝ), circle_equation x y ↔ c.contains (x, y)) ∧
    c.contains (0, 0) ∧
    c.contains (4, 0) ∧
    c.contains (-1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l3161_316145


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3161_316141

theorem polynomial_factorization (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x*y + x*z + y*z) := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3161_316141


namespace NUMINAMATH_CALUDE_hardware_store_earnings_l3161_316188

/-- Represents the sales data for a single item -/
structure ItemSales where
  quantity : Nat
  price : Nat
  discount_percent : Nat
  returns : Nat

/-- Calculates the total earnings from the hardware store sales -/
def calculate_earnings (sales_data : List ItemSales) : Nat :=
  sales_data.foldl (fun acc item =>
    let gross_sales := item.quantity * item.price
    let discount := gross_sales * item.discount_percent / 100
    let returns := item.returns * item.price
    acc + gross_sales - discount - returns
  ) 0

/-- Theorem stating that the total earnings of the hardware store are $11740 -/
theorem hardware_store_earnings : 
  let sales_data : List ItemSales := [
    { quantity := 10, price := 600, discount_percent := 10, returns := 0 },  -- Graphics cards
    { quantity := 14, price := 80,  discount_percent := 0,  returns := 0 },  -- Hard drives
    { quantity := 8,  price := 200, discount_percent := 0,  returns := 2 },  -- CPUs
    { quantity := 4,  price := 60,  discount_percent := 0,  returns := 0 },  -- RAM
    { quantity := 12, price := 90,  discount_percent := 0,  returns := 0 },  -- Power supply units
    { quantity := 6,  price := 250, discount_percent := 0,  returns := 0 },  -- Monitors
    { quantity := 18, price := 40,  discount_percent := 0,  returns := 0 },  -- Keyboards
    { quantity := 24, price := 20,  discount_percent := 0,  returns := 0 }   -- Mice
  ]
  calculate_earnings sales_data = 11740 := by
  sorry


end NUMINAMATH_CALUDE_hardware_store_earnings_l3161_316188


namespace NUMINAMATH_CALUDE_pta_fundraising_savings_l3161_316127

theorem pta_fundraising_savings (initial_amount : ℚ) : 
  (3/4 : ℚ) * initial_amount - (1/2 : ℚ) * ((3/4 : ℚ) * initial_amount) = 150 →
  initial_amount = 400 := by
  sorry

end NUMINAMATH_CALUDE_pta_fundraising_savings_l3161_316127


namespace NUMINAMATH_CALUDE_square_area_problem_l3161_316156

theorem square_area_problem (s : ℝ) : 
  (2 / 5 : ℝ) * s * 10 = 140 → s^2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_problem_l3161_316156


namespace NUMINAMATH_CALUDE_min_sum_dimensions_l3161_316101

theorem min_sum_dimensions (l w h : ℕ+) : 
  l * w * h = 2310 → 
  ∀ a b c : ℕ+, a * b * c = 2310 → l + w + h ≤ a + b + c → 
  l + w + h = 52 := by
sorry

end NUMINAMATH_CALUDE_min_sum_dimensions_l3161_316101


namespace NUMINAMATH_CALUDE_solution_equivalence_l3161_316169

theorem solution_equivalence (x : ℝ) : 
  (3/10 : ℝ) + |x - 7/20| < 4/15 ↔ x ∈ Set.Ioo (19/60 : ℝ) (23/60 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l3161_316169


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l3161_316103

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 3 = 0 → 
  x₂^2 - 2*x₂ - 3 = 0 → 
  x₁ * x₂ = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l3161_316103


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3161_316114

theorem solution_set_abs_inequality (x : ℝ) :
  (|1 - 2*x| < 3) ↔ (x ∈ Set.Ioo (-1) 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3161_316114


namespace NUMINAMATH_CALUDE_m_range_theorem_l3161_316147

def prop_p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def prop_q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

def m_range (m : ℝ) : Prop :=
  (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

theorem m_range_theorem :
  ∀ m : ℝ, (prop_p m ∨ prop_q m) ∧ ¬(prop_p m ∧ prop_q m) → m_range m :=
by sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3161_316147


namespace NUMINAMATH_CALUDE_sum_of_roots_l3161_316107

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = -13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3161_316107


namespace NUMINAMATH_CALUDE_triangle_side_length_l3161_316148

theorem triangle_side_length (a c : ℝ) (A : ℝ) (h1 : a = 2) (h2 : c = 2) (h3 : A = π/6) :
  let b := Real.sqrt ((a^2 + c^2 - 2*a*c*(Real.cos A)) / (Real.sin A)^2)
  b = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3161_316148


namespace NUMINAMATH_CALUDE_square_fraction_implies_properties_l3161_316134

theorem square_fraction_implies_properties (n : ℕ+) 
  (h : ∃ m : ℕ, (n * (n + 1)) / 3 = m ^ 2) :
  (∃ k : ℕ, n = 3 * k) ∧ 
  (∃ b : ℕ, n + 1 = b ^ 2) ∧ 
  (∃ a : ℕ, n / 3 = a ^ 2) := by
sorry

end NUMINAMATH_CALUDE_square_fraction_implies_properties_l3161_316134


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3161_316154

theorem unique_four_digit_number : ∃! n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ 10 + (n / 100 % 10) = x * y) ∧
    (10 + (n / 100 % 10) - (n / d₂ % 10) = 1)) ∧
  n = 1014 :=
sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3161_316154


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3161_316152

/-- Given a cone with slant height 4 and angle between the slant height and axis of rotation 30°,
    the lateral surface area of the cone is 8π. -/
theorem cone_lateral_surface_area (l : ℝ) (θ : ℝ) (h1 : l = 4) (h2 : θ = 30 * π / 180) :
  π * l * (l * Real.sin θ) = 8 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3161_316152


namespace NUMINAMATH_CALUDE_triangle_properties_l3161_316198

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A + t.B + t.C = Real.pi ∧
  t.a = 2 * t.b * Real.sin t.A ∧
  t.a = 3 * Real.sqrt 3 ∧
  t.c = 5

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.B = Real.pi / 6 ∧
  t.b = Real.sqrt 7 ∧
  (1/2 * t.a * t.c * Real.sin t.B) = (15 * Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3161_316198


namespace NUMINAMATH_CALUDE_green_toad_count_l3161_316104

/-- Represents the number of toads per acre -/
structure ToadPopulation where
  green : ℕ
  brown : ℕ
  spotted_brown : ℕ

/-- The conditions of the toad population -/
def valid_population (p : ToadPopulation) : Prop :=
  p.brown = 25 * p.green ∧
  p.spotted_brown = p.brown / 4 ∧
  p.spotted_brown = 50

/-- Theorem stating that in a valid toad population, there are 8 green toads per acre -/
theorem green_toad_count (p : ToadPopulation) (h : valid_population p) : p.green = 8 := by
  sorry


end NUMINAMATH_CALUDE_green_toad_count_l3161_316104


namespace NUMINAMATH_CALUDE_triangle_ratio_l3161_316149

/-- Given an acute triangle ABC and a point D inside it, 
    if ∠ADB = ∠ACB + 90° and AC · BD = AD · BC, 
    then (AB · CD) / (AC · BD) = √2 -/
theorem triangle_ratio (A B C D : ℝ × ℝ) : 
  let triangle_is_acute : Bool := sorry
  let D_inside_triangle : Bool := sorry
  let angle_ADB : ℝ := sorry
  let angle_ACB : ℝ := sorry
  let AC : ℝ := sorry
  let BD : ℝ := sorry
  let AD : ℝ := sorry
  let BC : ℝ := sorry
  let AB : ℝ := sorry
  let CD : ℝ := sorry
  triangle_is_acute ∧ 
  D_inside_triangle ∧
  angle_ADB = angle_ACB + π/2 ∧ 
  AC * BD = AD * BC →
  (AB * CD) / (AC * BD) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l3161_316149


namespace NUMINAMATH_CALUDE_same_route_probability_l3161_316119

theorem same_route_probability (num_routes : ℕ) (num_students : ℕ) : 
  num_routes = 3 → num_students = 2 → 
  (num_routes : ℝ) / (num_routes * num_routes : ℝ) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_same_route_probability_l3161_316119


namespace NUMINAMATH_CALUDE_contract_schemes_count_l3161_316197

def projects : ℕ := 6
def company_a_projects : ℕ := 3
def company_b_projects : ℕ := 2
def company_c_projects : ℕ := 1

theorem contract_schemes_count :
  (Nat.choose projects company_a_projects) *
  (Nat.choose (projects - company_a_projects) company_b_projects) *
  (Nat.choose (projects - company_a_projects - company_b_projects) company_c_projects) = 60 := by
  sorry

end NUMINAMATH_CALUDE_contract_schemes_count_l3161_316197


namespace NUMINAMATH_CALUDE_horner_method_f_3_l3161_316112

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x -/
def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem horner_method_f_3 :
  f 3 = horner_eval [5, 4, 3, 2, 1, 0] 3 ∧ horner_eval [5, 4, 3, 2, 1, 0] 3 = 1641 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_f_3_l3161_316112


namespace NUMINAMATH_CALUDE_ben_time_to_school_l3161_316171

/-- Represents the walking parameters of a person -/
structure WalkingParams where
  steps_per_minute : ℕ
  step_length : ℕ
  time_to_school : ℕ

/-- Calculates the time it takes for a person to walk to school given their walking parameters and the distance to school -/
def time_to_school (params : WalkingParams) (distance : ℕ) : ℚ :=
  distance / (params.steps_per_minute * params.step_length)

theorem ben_time_to_school 
  (amy : WalkingParams)
  (ben : WalkingParams)
  (h1 : amy.steps_per_minute = 80)
  (h2 : amy.step_length = 70)
  (h3 : amy.time_to_school = 20)
  (h4 : ben.steps_per_minute = 120)
  (h5 : ben.step_length = 50) :
  time_to_school ben (amy.steps_per_minute * amy.step_length * amy.time_to_school) = 56/3 := by
  sorry

end NUMINAMATH_CALUDE_ben_time_to_school_l3161_316171


namespace NUMINAMATH_CALUDE_min_distance_curve_to_line_l3161_316138

noncomputable def f (x : ℝ) := x^2 - Real.log x

def line (x : ℝ) := x - 2

theorem min_distance_curve_to_line :
  ∀ x > 0, ∃ d : ℝ,
    d = Real.sqrt 2 ∧
    ∀ y > 0, 
      let p₁ := (x, f x)
      let p₂ := (y, line y)
      d ≤ Real.sqrt ((x - y)^2 + (f x - line y)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_to_line_l3161_316138


namespace NUMINAMATH_CALUDE_erased_numbers_l3161_316192

def numbers_with_one : ℕ := 20
def numbers_with_two : ℕ := 19
def numbers_without_one_or_two : ℕ := 30
def total_numbers : ℕ := 100

theorem erased_numbers :
  numbers_with_one + numbers_with_two + numbers_without_one_or_two ≤ total_numbers ∧
  total_numbers - (numbers_with_one + numbers_with_two + numbers_without_one_or_two - 2) = 33 :=
sorry

end NUMINAMATH_CALUDE_erased_numbers_l3161_316192


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3161_316129

theorem simplify_trig_expression (θ : ℝ) :
  (Real.sin (π - 2*θ) / (1 - Real.sin (π/2 + 2*θ))) * Real.tan (π + θ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3161_316129


namespace NUMINAMATH_CALUDE_smallest_d_for_injective_g_l3161_316146

def g (x : ℝ) : ℝ := (x - 3)^2 - 7

theorem smallest_d_for_injective_g :
  ∀ d : ℝ, (∀ x y, x ≥ d → y ≥ d → g x = g y → x = y) ↔ d ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_d_for_injective_g_l3161_316146


namespace NUMINAMATH_CALUDE_nine_points_chords_l3161_316115

/-- The number of different chords that can be drawn by connecting two of n points on a circle. -/
def number_of_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting two of nine points
    on the circumference of a circle is equal to 36. -/
theorem nine_points_chords :
  number_of_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_points_chords_l3161_316115


namespace NUMINAMATH_CALUDE_elvis_album_songs_l3161_316167

/-- Calculates the number of songs on Elvis' new album given the studio time constraints. -/
theorem elvis_album_songs (
  total_studio_time : ℕ
  ) (record_time : ℕ) (edit_time : ℕ) (write_time : ℕ) 
  (h1 : total_studio_time = 5 * 60)  -- 5 hours in minutes
  (h2 : record_time = 12)            -- 12 minutes to record each song
  (h3 : edit_time = 30)              -- 30 minutes to edit all songs
  (h4 : write_time = 15)             -- 15 minutes to write each song
  : ℕ := by
  
  -- The number of songs is equal to the available time for writing and recording
  -- divided by the time needed for writing and recording one song
  have num_songs : ℕ := (total_studio_time - edit_time) / (write_time + record_time)
  
  -- Prove that num_songs equals 10
  sorry

#eval (5 * 60 - 30) / (15 + 12)  -- Should evaluate to 10

end NUMINAMATH_CALUDE_elvis_album_songs_l3161_316167


namespace NUMINAMATH_CALUDE_right_triangle_area_l3161_316158

theorem right_triangle_area (h : ℝ) (h_positive : h > 0) :
  let angle_30 : ℝ := 30 * π / 180
  let angle_60 : ℝ := 60 * π / 180
  let angle_90 : ℝ := 90 * π / 180
  h = 4 →
  (1/2) * (h * (2 * h / Real.sqrt 3)) * (h * Real.sqrt 3) = (16 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3161_316158


namespace NUMINAMATH_CALUDE_inequality_solution_l3161_316150

/-- Given constants p, q, and r satisfying the conditions, prove that p + 2q + 3r = 32 -/
theorem inequality_solution (p q r : ℝ) (h1 : p < q)
  (h2 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x > 5 ∨ (3 ≤ x ∧ x ≤ 7)) :
  p + 2*q + 3*r = 32 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3161_316150


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3161_316142

theorem triangle_angle_measure (A B C : Real) : 
  A + B + C = 180 →
  B = A + 20 →
  C = 50 →
  B = 75 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3161_316142


namespace NUMINAMATH_CALUDE_circle_center_and_sum_l3161_316137

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x - 6*y + 9

/-- The center of a circle given by its equation -/
def CircleCenter (eq : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

/-- Theorem stating that the center of the given circle is (2, -3) 
    and the sum of its coordinates is -1 -/
theorem circle_center_and_sum :
  let center := CircleCenter CircleEquation
  center = (2, -3) ∧ center.1 + center.2 = -1 := by sorry

end NUMINAMATH_CALUDE_circle_center_and_sum_l3161_316137


namespace NUMINAMATH_CALUDE_smallest_y_divisible_by_11_l3161_316191

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def number_with_y (y : ℕ) : ℕ :=
  7000000 + y * 100000 + 86038

theorem smallest_y_divisible_by_11 :
  ∀ y : ℕ, y < 14 → ¬(is_divisible_by_11 (number_with_y y)) ∧
  is_divisible_by_11 (number_with_y 14) := by
  sorry

end NUMINAMATH_CALUDE_smallest_y_divisible_by_11_l3161_316191


namespace NUMINAMATH_CALUDE_odd_periodic_symmetry_ln_quotient_odd_main_theorem_l3161_316121

-- Definition of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Definition of a periodic function
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- Theorem 1: Odd function with period 4 is symmetric about (2,0)
theorem odd_periodic_symmetry (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : IsPeriodic f 4) :
  ∀ x, f (2 + x) = f (2 - x) :=
sorry

-- Theorem 2: ln((1+x)/(1-x)) is an odd function on (-1,1)
theorem ln_quotient_odd :
  IsOdd (fun x => Real.log ((1 + x) / (1 - x))) :=
sorry

-- Main theorem combining both results
theorem main_theorem :
  (∃ f : ℝ → ℝ, IsOdd f ∧ IsPeriodic f 4 ∧ (∀ x, f (2 + x) = f (2 - x))) ∧
  IsOdd (fun x => Real.log ((1 + x) / (1 - x))) :=
sorry

end NUMINAMATH_CALUDE_odd_periodic_symmetry_ln_quotient_odd_main_theorem_l3161_316121
