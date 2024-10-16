import Mathlib

namespace NUMINAMATH_CALUDE_elevator_weight_problem_l2291_229136

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (h1 : initial_people = 6) 
  (h2 : initial_avg_weight = 152) (h3 : new_avg_weight = 151) :
  let total_initial_weight := initial_people * initial_avg_weight
  let total_new_weight := (initial_people + 1) * new_avg_weight
  let seventh_person_weight := total_new_weight - total_initial_weight
  seventh_person_weight = 145 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l2291_229136


namespace NUMINAMATH_CALUDE_joes_initial_weight_l2291_229144

/-- Represents Joe's weight loss over time --/
structure WeightLoss where
  initial_weight : ℝ
  current_weight : ℝ
  final_weight : ℝ
  months_so_far : ℝ
  months_to_goal : ℝ
  weight_loss_rate : ℝ

/-- Theorem stating that Joe's initial weight was 222 pounds --/
theorem joes_initial_weight (w : WeightLoss) 
  (h1 : w.months_so_far = 3)
  (h2 : w.current_weight = 198)
  (h3 : w.months_to_goal = 3.5)
  (h4 : w.final_weight = 170)
  (h5 : w.weight_loss_rate = (w.current_weight - w.final_weight) / w.months_to_goal)
  (h6 : w.initial_weight = w.current_weight + w.weight_loss_rate * w.months_so_far) :
  w.initial_weight = 222 := by
  sorry

end NUMINAMATH_CALUDE_joes_initial_weight_l2291_229144


namespace NUMINAMATH_CALUDE_alex_overall_score_l2291_229180

def quiz_problems : ℕ := 30
def test_problems : ℕ := 50
def exam_problems : ℕ := 20

def quiz_score : ℚ := 75 / 100
def test_score : ℚ := 85 / 100
def exam_score : ℚ := 80 / 100

def total_problems : ℕ := quiz_problems + test_problems + exam_problems

def correct_problems : ℚ := 
  quiz_score * quiz_problems + test_score * test_problems + exam_score * exam_problems

theorem alex_overall_score : correct_problems / total_problems = 81 / 100 := by
  sorry

end NUMINAMATH_CALUDE_alex_overall_score_l2291_229180


namespace NUMINAMATH_CALUDE_visitors_count_l2291_229147

/-- Represents the cost per person based on the number of visitors -/
def cost_per_person (n : ℕ) : ℚ :=
  if n ≤ 30 then 100
  else max 72 (100 - 2 * (n - 30))

/-- The total cost for n visitors -/
def total_cost (n : ℕ) : ℚ := n * cost_per_person n

/-- Theorem stating that 35 is the number of visitors given the conditions -/
theorem visitors_count : ∃ (n : ℕ), n > 30 ∧ total_cost n = 3150 ∧ n = 35 := by
  sorry


end NUMINAMATH_CALUDE_visitors_count_l2291_229147


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l2291_229175

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 = 25) 
  (h_a2 : a 2 = 3) :
  a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l2291_229175


namespace NUMINAMATH_CALUDE_friend_jogging_time_l2291_229103

/-- Proves that if a person completes a route in 3 hours, and another person travels at twice the speed of the first person, then the second person will complete the same route in 90 minutes. -/
theorem friend_jogging_time (my_time : ℝ) (friend_speed : ℝ) (my_speed : ℝ) :
  my_time = 3 →
  friend_speed = 2 * my_speed →
  friend_speed * (90 / 60) = my_speed * my_time :=
by
  sorry

#check friend_jogging_time

end NUMINAMATH_CALUDE_friend_jogging_time_l2291_229103


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2291_229127

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_prod_eq : a * b + a * c + b * c = 3)
  (prod_eq : a * b * c = 5) :
  a^3 + b^3 + c^3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2291_229127


namespace NUMINAMATH_CALUDE_oil_depth_in_cylindrical_tank_l2291_229153

/-- Represents a horizontal cylindrical tank --/
structure HorizontalCylindricalTank where
  length : Real
  diameter : Real

/-- Represents the oil in the tank --/
structure Oil where
  depth : Real
  surface_area : Real

theorem oil_depth_in_cylindrical_tank
  (tank : HorizontalCylindricalTank)
  (oil : Oil)
  (h_length : tank.length = 12)
  (h_diameter : tank.diameter = 4)
  (h_surface_area : oil.surface_area = 24) :
  oil.depth = 2 - Real.sqrt 3 ∨ oil.depth = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_in_cylindrical_tank_l2291_229153


namespace NUMINAMATH_CALUDE_total_paid_is_705_l2291_229130

/-- Calculates the total amount paid for fruits given their quantities and rates -/
def total_amount_paid (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem: The total amount paid for the given quantities and rates of grapes and mangoes is 705 -/
theorem total_paid_is_705 :
  total_amount_paid 3 70 9 55 = 705 := by
  sorry

end NUMINAMATH_CALUDE_total_paid_is_705_l2291_229130


namespace NUMINAMATH_CALUDE_bound_cyclic_fraction_l2291_229101

theorem bound_cyclic_fraction (a b x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : 0 < a) (h₂ : a < b)
  (h₃ : a ≤ x₁ ∧ x₁ ≤ b) (h₄ : a ≤ x₂ ∧ x₂ ≤ b)
  (h₅ : a ≤ x₃ ∧ x₃ ≤ b) (h₆ : a ≤ x₄ ∧ x₄ ≤ b) :
  1/b ≤ (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ∧
  (x₁^2/x₂ + x₂^2/x₃ + x₃^2/x₄ + x₄^2/x₁) / (x₁ + x₂ + x₃ + x₄) ≤ 1/a :=
by sorry

end NUMINAMATH_CALUDE_bound_cyclic_fraction_l2291_229101


namespace NUMINAMATH_CALUDE_bird_count_l2291_229157

theorem bird_count (swallows bluebirds cardinals : ℕ) : 
  swallows = 2 →
  swallows * 2 = bluebirds →
  cardinals = bluebirds * 3 →
  swallows + bluebirds + cardinals = 18 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l2291_229157


namespace NUMINAMATH_CALUDE_complex_modulus_l2291_229107

theorem complex_modulus (z : ℂ) (h : z * (3 - 4*I) = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l2291_229107


namespace NUMINAMATH_CALUDE_quadratic_function_property_l2291_229118

/-- Given a quadratic function f(x) = ax² + bx + c, 
    if f(0) = f(4) > f(1), then a > 0 and 4a + b = 0 -/
theorem quadratic_function_property (a b c : ℝ) :
  let f := λ x : ℝ => a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l2291_229118


namespace NUMINAMATH_CALUDE_solution_set_equality_l2291_229163

/-- A function f: ℝ → ℝ that is odd, monotonically increasing on (0, +∞), and f(-1) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → 0 < y → x < y → f x < f y) ∧
  (f (-1) = 2)

/-- The theorem statement -/
theorem solution_set_equality (f : ℝ → ℝ) (h : special_function f) :
  {x : ℝ | x > 0 ∧ f (x - 1) + 2 ≤ 0} = Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2291_229163


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2291_229172

theorem complex_expression_equals_negative_two :
  (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2291_229172


namespace NUMINAMATH_CALUDE_polynomial_roots_l2291_229121

def f (x : ℝ) : ℝ := 2*x^4 - 5*x^3 - 7*x^2 + 34*x - 24

theorem polynomial_roots :
  (f 1 = 0) ∧
  (∀ x : ℝ, f x = 0 ∧ x ≠ 1 → 2*x^3 - 3*x^2 - 12*x + 10 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2291_229121


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l2291_229125

theorem fruit_arrangement_count : 
  let total_fruits : ℕ := 10
  let apple_count : ℕ := 4
  let orange_count : ℕ := 3
  let banana_count : ℕ := 2
  let grape_count : ℕ := 1
  apple_count + orange_count + banana_count + grape_count = total_fruits →
  (Nat.factorial total_fruits) / 
  ((Nat.factorial apple_count) * (Nat.factorial orange_count) * 
   (Nat.factorial banana_count) * (Nat.factorial grape_count)) = 12600 := by
sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l2291_229125


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l2291_229137

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l2291_229137


namespace NUMINAMATH_CALUDE_total_frogs_in_pond_l2291_229177

def frogs_on_lilypads : ℕ := 5
def frogs_on_logs : ℕ := 3
def dozen : ℕ := 12
def baby_frogs_dozens : ℕ := 2

theorem total_frogs_in_pond : 
  frogs_on_lilypads + frogs_on_logs + baby_frogs_dozens * dozen = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_frogs_in_pond_l2291_229177


namespace NUMINAMATH_CALUDE_probability_two_red_two_blue_eq_l2291_229151

def total_marbles : ℕ := 27
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 12
def marbles_selected : ℕ := 4

def probability_two_red_two_blue : ℚ :=
  6 * (red_marbles.choose 2 * blue_marbles.choose 2) / total_marbles.choose marbles_selected

theorem probability_two_red_two_blue_eq :
  probability_two_red_two_blue = 154 / 225 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_two_blue_eq_l2291_229151


namespace NUMINAMATH_CALUDE_symmetry_of_point_l2291_229178

def point_symmetric_to_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), -(p.2))

theorem symmetry_of_point :
  let A : ℝ × ℝ := (-1, 2)
  let A' : ℝ × ℝ := point_symmetric_to_origin A
  A' = (1, -2) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_point_l2291_229178


namespace NUMINAMATH_CALUDE_hockey_tournament_games_l2291_229150

/-- The number of teams in the hockey league --/
def num_teams : ℕ := 7

/-- The number of times each team plays against every other team --/
def games_per_matchup : ℕ := 4

/-- The total number of games played in the tournament --/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_matchup

theorem hockey_tournament_games :
  total_games = 84 :=
by sorry

end NUMINAMATH_CALUDE_hockey_tournament_games_l2291_229150


namespace NUMINAMATH_CALUDE_value_range_of_f_l2291_229128

-- Define the function f(x) = x^2 - 2x + 2
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

-- Define the interval (0, 4]
def interval : Set ℝ := {x | 0 < x ∧ x ≤ 4}

-- Theorem statement
theorem value_range_of_f : 
  (∀ x ∈ interval, 1 ≤ f x) ∧ 
  (∀ x ∈ interval, f x ≤ 10) ∧ 
  (∃ x ∈ interval, f x = 1) ∧ 
  (∃ x ∈ interval, f x = 10) :=
sorry

end NUMINAMATH_CALUDE_value_range_of_f_l2291_229128


namespace NUMINAMATH_CALUDE_lcm_of_1716_924_1260_l2291_229133

theorem lcm_of_1716_924_1260 : Nat.lcm (Nat.lcm 1716 924) 1260 = 13860 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_1716_924_1260_l2291_229133


namespace NUMINAMATH_CALUDE_fraction_domain_l2291_229197

theorem fraction_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) → x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_domain_l2291_229197


namespace NUMINAMATH_CALUDE_inequality_two_integer_solutions_l2291_229179

theorem inequality_two_integer_solutions (k : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    (k * (x : ℝ)^2 ≤ Real.log x + 1) ∧ 
    (k * (y : ℝ)^2 ≤ Real.log y + 1) ∧
    (∀ (z : ℕ), z ≠ x ∧ z ≠ y → k * (z : ℝ)^2 > Real.log z + 1)) →
  ((Real.log 3 + 1) / 9 < k ∧ k ≤ (Real.log 2 + 1) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_two_integer_solutions_l2291_229179


namespace NUMINAMATH_CALUDE_complex_sum_modulus_l2291_229176

theorem complex_sum_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = 1) : 
  Complex.abs (z₁ + z₂) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_modulus_l2291_229176


namespace NUMINAMATH_CALUDE_area_of_M_l2291_229187

-- Define the set M
def M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (abs x + abs (4 - x) ≤ 4) ∧
               ((x^2 - 4*x - 2*y + 2) / (y - x + 3) ≥ 0) ∧
               (0 ≤ x ∧ x ≤ 4)}

-- Define the area function for sets in ℝ²
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_M : area M = 4 := by sorry

end NUMINAMATH_CALUDE_area_of_M_l2291_229187


namespace NUMINAMATH_CALUDE_garden_operations_result_l2291_229131

/-- Represents the quantities of vegetables in the garden -/
structure VegetableQuantities where
  tomatoes : ℕ
  potatoes : ℕ
  cucumbers : ℕ
  cabbages : ℕ

/-- Calculates the final quantities of vegetables after operations -/
def final_quantities (initial : VegetableQuantities) 
  (picked_tomatoes picked_potatoes picked_cabbages : ℕ)
  (new_cucumber_plants new_cabbage_plants : ℕ)
  (cucumber_yield cabbage_yield : ℕ) : VegetableQuantities :=
  { tomatoes := initial.tomatoes - picked_tomatoes,
    potatoes := initial.potatoes - picked_potatoes,
    cucumbers := initial.cucumbers + new_cucumber_plants * cucumber_yield,
    cabbages := initial.cabbages - picked_cabbages + new_cabbage_plants * cabbage_yield }

theorem garden_operations_result :
  let initial := VegetableQuantities.mk 500 400 300 100
  let final := final_quantities initial 325 270 50 200 80 2 3
  final.tomatoes = 175 ∧ 
  final.potatoes = 130 ∧ 
  final.cucumbers = 700 ∧ 
  final.cabbages = 290 := by
  sorry

end NUMINAMATH_CALUDE_garden_operations_result_l2291_229131


namespace NUMINAMATH_CALUDE_no_positive_solutions_l2291_229135

theorem no_positive_solutions : ¬∃ (a b c d : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a * d^2 + b * d - c = 0 ∧
  Real.sqrt a * d + Real.sqrt b * Real.sqrt d - Real.sqrt c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_solutions_l2291_229135


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2291_229123

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  n > 2 → 
  exterior_angle = 20 → 
  (n : ℝ) * exterior_angle = 360 →
  n = 18 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2291_229123


namespace NUMINAMATH_CALUDE_heejin_is_oldest_l2291_229198

-- Define the ages of the three friends
def yoona_age : ℕ := 23
def miyoung_age : ℕ := 22
def heejin_age : ℕ := 24

-- Theorem stating that Heejin is the oldest
theorem heejin_is_oldest : 
  heejin_age ≥ yoona_age ∧ heejin_age ≥ miyoung_age := by
  sorry

end NUMINAMATH_CALUDE_heejin_is_oldest_l2291_229198


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l2291_229115

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (x + 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  (a₁ + 3*a₃ + 5*a₅ + 7*a₇ + 9*a₉)^2 - (2*a₂ + 4*a₄ + 6*a₆ + 8*a₈)^2 = 3^12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l2291_229115


namespace NUMINAMATH_CALUDE_value_added_to_half_l2291_229112

theorem value_added_to_half : ∃ (v : ℝ), (20 / 2) + v = 17 ∧ v = 7 := by sorry

end NUMINAMATH_CALUDE_value_added_to_half_l2291_229112


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2291_229105

/-- A geometric sequence of positive integers -/
def GeometricSequence (a : ℕ → ℕ) : Prop :=
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem seventh_term_of_geometric_sequence
  (a : ℕ → ℕ)
  (h_geom : GeometricSequence a)
  (h_first : a 1 = 3)
  (h_sixth : a 6 = 972) :
  a 7 = 2187 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l2291_229105


namespace NUMINAMATH_CALUDE_min_value_expression_l2291_229167

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x - y^2 = 1 → x^2 + 2*y^2 + 4*x - 1 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2291_229167


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2291_229148

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2291_229148


namespace NUMINAMATH_CALUDE_smallest_w_l2291_229162

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_w : ∃ (w : ℕ), 
  w > 0 ∧
  is_factor (2^4) (1452 * w) ∧
  is_factor (3^3) (1452 * w) ∧
  is_factor (13^3) (1452 * w) ∧
  ∀ (x : ℕ), x > 0 ∧ 
    is_factor (2^4) (1452 * x) ∧
    is_factor (3^3) (1452 * x) ∧
    is_factor (13^3) (1452 * x) →
    w ≤ x :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_w_l2291_229162


namespace NUMINAMATH_CALUDE_unique_number_with_special_divisors_l2291_229124

def has_twelve_divisors (N : ℕ) : Prop :=
  ∃ (d : Fin 12 → ℕ), 
    (∀ i j, i < j → d i < d j) ∧
    (∀ i, d i ∣ N) ∧
    (∀ m, m ∣ N → ∃ i, d i = m) ∧
    d 0 = 1 ∧ d 11 = N

theorem unique_number_with_special_divisors :
  ∃! N : ℕ, has_twelve_divisors N ∧
    ∃ (d : Fin 12 → ℕ), 
      (∀ i j, i < j → d i < d j) ∧
      (∀ i, d i ∣ N) ∧
      (d 0 = 1) ∧
      (d (d 3 - 2) = (d 0 + d 1 + d 3) * d 7) ∧
      N = 1989 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_special_divisors_l2291_229124


namespace NUMINAMATH_CALUDE_complex_number_problem_l2291_229109

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  z₁ * (2 + Complex.I) = 5 * Complex.I →
  (∃ (r : ℝ), z₁ + z₂ = r) →
  (∃ (y : ℝ), y ≠ 0 ∧ z₁ * z₂ = y * Complex.I) →
  z₂ = -4 - 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2291_229109


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2291_229184

theorem arithmetic_expression_equality : 1000 + 200 - 10 + 1 = 1191 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2291_229184


namespace NUMINAMATH_CALUDE_parabola_function_expression_l2291_229159

-- Define the parabola function
def parabola (a : ℝ) (x : ℝ) : ℝ := a * (x + 3)^2 + 2

-- State the theorem
theorem parabola_function_expression :
  ∃ a : ℝ, 
    (parabola a (-3) = 2) ∧ 
    (parabola a 1 = -14) ∧
    (∀ x : ℝ, parabola a x = -(x + 3)^2 + 2) := by
  sorry


end NUMINAMATH_CALUDE_parabola_function_expression_l2291_229159


namespace NUMINAMATH_CALUDE_initial_points_count_l2291_229169

/-- Represents the number of points after performing the point-adding operation n times -/
def pointsAfterOperations (k : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => 2 * (pointsAfterOperations k n) - 1

/-- The theorem stating that if 101 points result after two operations, then 26 points were initially marked -/
theorem initial_points_count : 
  ∀ k : ℕ, pointsAfterOperations k 2 = 101 → k = 26 := by
  sorry

end NUMINAMATH_CALUDE_initial_points_count_l2291_229169


namespace NUMINAMATH_CALUDE_max_cookies_per_student_l2291_229160

/-- Proves the maximum number of cookies a single student can take in a class -/
theorem max_cookies_per_student
  (num_students : ℕ)
  (mean_cookies : ℕ)
  (h_num_students : num_students = 25)
  (h_mean_cookies : mean_cookies = 4)
  (h_min_cookie : ∀ student, student ≥ 1) :
  (num_students * mean_cookies) - (num_students - 1) = 76 := by
sorry

end NUMINAMATH_CALUDE_max_cookies_per_student_l2291_229160


namespace NUMINAMATH_CALUDE_sum_of_terms_3_to_6_l2291_229141

/-- Given a sequence {aₙ} where the sum of the first n terms is Sₙ = n² + 2n + 5,
    prove that a₃ + a₄ + a₅ + a₆ = 40 -/
theorem sum_of_terms_3_to_6 (a : ℕ → ℤ) (S : ℕ → ℤ) 
    (h : ∀ n : ℕ, S n = n^2 + 2*n + 5) : 
    a 3 + a 4 + a 5 + a 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_terms_3_to_6_l2291_229141


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_in_open_unit_interval_l2291_229170

theorem quadratic_always_positive_implies_a_in_open_unit_interval (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_in_open_unit_interval_l2291_229170


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l2291_229139

theorem smaller_number_in_ratio (a b : ℝ) : 
  a / b = 3 / 4 → a + b = 420 → a = 180 := by sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l2291_229139


namespace NUMINAMATH_CALUDE_sixty_four_to_five_sixths_l2291_229190

theorem sixty_four_to_five_sixths (h : 64 = 2^6) : 64^(5/6) = 32 := by
  sorry

end NUMINAMATH_CALUDE_sixty_four_to_five_sixths_l2291_229190


namespace NUMINAMATH_CALUDE_exists_abs_less_than_one_l2291_229138

def sequence_property (a : ℕ → ℝ) : Prop :=
  (a 1 * a 2 < 0) ∧
  (∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧
    a n = a i + a j ∧
    ∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)

theorem exists_abs_less_than_one (a : ℕ → ℝ) (h : sequence_property a) :
  ∃ i : ℕ, |a i| < 1 := by sorry

end NUMINAMATH_CALUDE_exists_abs_less_than_one_l2291_229138


namespace NUMINAMATH_CALUDE_problem_statement_l2291_229168

theorem problem_statement (t : ℝ) :
  let x := 3 - 1.5 * t
  let y := 3 * t + 4
  x = 6 → y = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2291_229168


namespace NUMINAMATH_CALUDE_lottery_probability_l2291_229114

theorem lottery_probability (p : ℝ) (n : ℕ) (h1 : p = 1 / 10000000) (h2 : n = 5) :
  n * p = 5 / 10000000 := by sorry

end NUMINAMATH_CALUDE_lottery_probability_l2291_229114


namespace NUMINAMATH_CALUDE_onion_weight_proof_l2291_229155

/-- Proves that the total weight of onions on a scale is 7.68 kg given specific conditions --/
theorem onion_weight_proof (total_weight : ℝ) (remaining_onions : ℕ) (removed_onions : ℕ) 
  (avg_weight_remaining : ℝ) (avg_weight_removed : ℝ) : 
  total_weight = 7.68 ∧ 
  remaining_onions = 35 ∧ 
  removed_onions = 5 ∧ 
  avg_weight_remaining = 0.190 ∧ 
  avg_weight_removed = 0.206 → 
  total_weight = (remaining_onions : ℝ) * avg_weight_remaining + 
                 (removed_onions : ℝ) * avg_weight_removed :=
by
  sorry

#check onion_weight_proof

end NUMINAMATH_CALUDE_onion_weight_proof_l2291_229155


namespace NUMINAMATH_CALUDE_trees_in_yard_l2291_229100

/-- The number of trees in a yard with given length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem: There are 31 trees in a 360-meter yard with 12-meter spacing -/
theorem trees_in_yard :
  num_trees 360 12 = 31 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l2291_229100


namespace NUMINAMATH_CALUDE_no_solution_exists_l2291_229119

/-- A polynomial with roots -p, -p-1, -p-2, -p-3 -/
def g (p : ℕ+) (x : ℝ) : ℝ :=
  (x + p) * (x + p + 1) * (x + p + 2) * (x + p + 3)

/-- Coefficients of the expanded polynomial g -/
def a (p : ℕ+) : ℝ := 4 * p + 6
def b (p : ℕ+) : ℝ := 10 * p^2 + 15 * p + 11
def c (p : ℕ+) : ℝ := 12 * p^3 + 18 * p^2 + 22 * p + 6
def d (p : ℕ+) : ℝ := 6 * p^4 + 9 * p^3 + 20 * p^2 + 15 * p + 6

/-- Theorem stating that there is no positive integer p satisfying the given condition -/
theorem no_solution_exists : ¬ ∃ (p : ℕ+), a p + b p + c p + d p = 2056 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2291_229119


namespace NUMINAMATH_CALUDE_number_of_white_balls_l2291_229194

/-- Given the number of red and blue balls, and the relationship between red balls and the sum of blue and white balls, prove the number of white balls. -/
theorem number_of_white_balls (red blue : ℕ) (h1 : red = 60) (h2 : blue = 30) 
  (h3 : red = blue + white + 5) : white = 25 :=
by
  sorry

#check number_of_white_balls

end NUMINAMATH_CALUDE_number_of_white_balls_l2291_229194


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_half_l2291_229104

noncomputable def f (x : ℝ) : ℝ := Real.sin x / (Real.sin x + Real.cos x)

theorem derivative_f_at_pi_half :
  deriv f (π / 2) = 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_half_l2291_229104


namespace NUMINAMATH_CALUDE_vector_equality_l2291_229117

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, x)

theorem vector_equality (x : ℝ) :
  let a := vector_a x
  let b := vector_b x
  (a.1 + b.1)^2 + (a.2 + b.2)^2 = (a.1 - b.1)^2 + (a.2 - b.2)^2 →
  x = 1 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_equality_l2291_229117


namespace NUMINAMATH_CALUDE_cycle_selling_price_l2291_229142

/-- Given a cycle with a cost price of 1400 and sold at a loss of 25%, 
    prove that the selling price is 1050. -/
theorem cycle_selling_price 
  (cost_price : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : cost_price = 1400) 
  (h2 : loss_percentage = 25) : 
  cost_price * (1 - loss_percentage / 100) = 1050 := by
sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l2291_229142


namespace NUMINAMATH_CALUDE_male_animals_count_l2291_229171

/-- Represents the number of male animals in Fred's barn after a series of events -/
def male_animals_in_barn : ℕ :=
  let initial_horses : ℕ := 100
  let initial_sheep : ℕ := 29
  let initial_chickens : ℕ := 9
  let initial_total : ℕ := initial_horses + initial_sheep + initial_chickens
  let brians_purchase : ℕ := initial_total / 2
  let remaining_after_purchase : ℕ := initial_total - brians_purchase
  let jeremys_gift : ℕ := 37
  let final_total : ℕ := remaining_after_purchase + jeremys_gift
  final_total / 2

theorem male_animals_count : male_animals_in_barn = 53 := by
  sorry

end NUMINAMATH_CALUDE_male_animals_count_l2291_229171


namespace NUMINAMATH_CALUDE_art_arrangement_probability_l2291_229132

/-- The total number of art pieces --/
def total_pieces : ℕ := 12

/-- The number of Escher prints --/
def escher_prints : ℕ := 4

/-- The number of Picasso prints --/
def picasso_prints : ℕ := 2

/-- The probability of the desired arrangement --/
def arrangement_probability : ℚ := 912 / 479001600

theorem art_arrangement_probability :
  let remaining_pieces := total_pieces - escher_prints
  let escher_block_positions := remaining_pieces + 1
  let escher_internal_arrangements := Nat.factorial escher_prints
  let picasso_positions := total_pieces - escher_prints + 1
  let valid_picasso_arrangements := 38
  (escher_block_positions * escher_internal_arrangements * valid_picasso_arrangements : ℚ) /
    Nat.factorial total_pieces = arrangement_probability := by
  sorry

end NUMINAMATH_CALUDE_art_arrangement_probability_l2291_229132


namespace NUMINAMATH_CALUDE_tip_calculation_l2291_229166

/-- Calculates the tip amount given the meal cost, tax rate, and payment amount. -/
def calculate_tip (meal_cost : ℝ) (tax_rate : ℝ) (payment : ℝ) : ℝ :=
  payment - (meal_cost * (1 + tax_rate))

/-- Proves that given a meal cost of $15.00, a tax rate of 20%, and a payment of $20.00, the tip amount is $2.00. -/
theorem tip_calculation :
  calculate_tip 15 0.2 20 = 2 := by
  sorry

#eval calculate_tip 15 0.2 20

end NUMINAMATH_CALUDE_tip_calculation_l2291_229166


namespace NUMINAMATH_CALUDE_function_square_evaluation_l2291_229140

theorem function_square_evaluation (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2
  f (a + 1) = a^2 + 2*a + 1 := by sorry

end NUMINAMATH_CALUDE_function_square_evaluation_l2291_229140


namespace NUMINAMATH_CALUDE_exponent_equality_l2291_229143

theorem exponent_equality (n : ℕ) : 2^3 * 8^3 = 2^(2*n) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2291_229143


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2291_229185

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 10) / (factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2291_229185


namespace NUMINAMATH_CALUDE_B_power_87_l2291_229102

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 0]]

theorem B_power_87 : B ^ 87 = ![![0,  1, 0],
                                 ![-1, 0, 0],
                                 ![0,  0, 0]] := by
  sorry

end NUMINAMATH_CALUDE_B_power_87_l2291_229102


namespace NUMINAMATH_CALUDE_sin_721_degrees_equals_sin_1_degree_l2291_229145

theorem sin_721_degrees_equals_sin_1_degree :
  Real.sin (721 * π / 180) = Real.sin (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_721_degrees_equals_sin_1_degree_l2291_229145


namespace NUMINAMATH_CALUDE_remainder_3_102_mod_101_l2291_229196

theorem remainder_3_102_mod_101 (h : Nat.Prime 101) : 3^102 ≡ 9 [MOD 101] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_102_mod_101_l2291_229196


namespace NUMINAMATH_CALUDE_line_perpendicular_planes_parallel_l2291_229199

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicularToPlane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicularLines : Line → Line → Prop)

-- Define the "contained in" relation for a line in a plane
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_planes_parallel
  (l m : Line) (α β : Plane)
  (h1 : perpendicularToPlane l α)
  (h2 : containedIn m β) :
  (∀ x y, parallelPlanes x y → perpendicularLines l m) ∧
  ∃ x y, perpendicularLines x y ∧ ¬parallelPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_planes_parallel_l2291_229199


namespace NUMINAMATH_CALUDE_no_integer_with_five_divisors_sum_square_l2291_229111

theorem no_integer_with_five_divisors_sum_square : ¬ ∃ (n : ℕ+), 
  (∃ (d₁ d₂ d₃ d₄ d₅ : ℕ+), 
    (d₁ < d₂) ∧ (d₂ < d₃) ∧ (d₃ < d₄) ∧ (d₄ < d₅) ∧
    (d₁ ∣ n) ∧ (d₂ ∣ n) ∧ (d₃ ∣ n) ∧ (d₄ ∣ n) ∧ (d₅ ∣ n) ∧
    (∀ (d : ℕ+), d ∣ n → d ≥ d₅ ∨ d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄)) ∧
  (∃ (x : ℕ), (d₁ : ℕ)^2 + (d₂ : ℕ)^2 + (d₃ : ℕ)^2 + (d₄ : ℕ)^2 + (d₅ : ℕ)^2 = x^2) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_with_five_divisors_sum_square_l2291_229111


namespace NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l2291_229156

theorem x_power_2187_minus_reciprocal (x : ℂ) (h : x - 1/x = 2*I*Real.sqrt 2) : 
  x^2187 - 1/(x^2187) = -22*I*Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_x_power_2187_minus_reciprocal_l2291_229156


namespace NUMINAMATH_CALUDE_remainder_theorem_l2291_229108

-- Define the polynomial p(x) = x^4 - 2x^2 + 4x - 5
def p (x : ℝ) : ℝ := x^4 - 2*x^2 + 4*x - 5

-- State the theorem
theorem remainder_theorem : 
  ∃ (q : ℝ → ℝ), ∀ (x : ℝ), p x = (x - 1) * q x + (-2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2291_229108


namespace NUMINAMATH_CALUDE_larger_circle_radius_l2291_229106

/-- A system of two circles with specific properties -/
structure CircleSystem where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  longest_chord : ℝ  -- Length of the longest chord in the larger circle

/-- Properties of the circle system -/
def circle_system_properties (cs : CircleSystem) : Prop :=
  cs.longest_chord = 24 ∧  -- The longest chord of the larger circle is 24
  cs.r = cs.R / 2 ∧  -- The radius of the smaller circle is half the radius of the larger circle
  cs.R > 0 ∧  -- The radius of the larger circle is positive
  cs.r > 0  -- The radius of the smaller circle is positive

/-- Theorem stating that the radius of the larger circle is 12 -/
theorem larger_circle_radius (cs : CircleSystem) 
  (h : circle_system_properties cs) : cs.R = 12 := by
  sorry

end NUMINAMATH_CALUDE_larger_circle_radius_l2291_229106


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2291_229149

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_first : a 1 = 6) 
  (h_third : a 3 = 2) : 
  a 5 = -2 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l2291_229149


namespace NUMINAMATH_CALUDE_no_tetrahedron_with_given_edges_l2291_229158

/-- Represents a tetrahedron with three pairs of opposite edges --/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ

/-- Checks if a tetrahedron with given edge lengths can exist --/
def tetrahedronExists (t : Tetrahedron) : Prop :=
  t.edge1 > 0 ∧ t.edge2 > 0 ∧ t.edge3 > 0 ∧
  t.edge1^2 + t.edge2^2 > t.edge3^2 ∧
  t.edge1^2 + t.edge3^2 > t.edge2^2 ∧
  t.edge2^2 + t.edge3^2 > t.edge1^2

/-- Theorem stating that a tetrahedron with the given edge lengths does not exist --/
theorem no_tetrahedron_with_given_edges :
  ¬ ∃ (t : Tetrahedron), t.edge1 = 12 ∧ t.edge2 = 12.5 ∧ t.edge3 = 13 ∧ tetrahedronExists t :=
by sorry


end NUMINAMATH_CALUDE_no_tetrahedron_with_given_edges_l2291_229158


namespace NUMINAMATH_CALUDE_cost_price_articles_l2291_229120

/-- Given that the cost price of N articles equals the selling price of 50 articles,
    and the profit percentage is 10.000000000000004%, prove that N = 55. -/
theorem cost_price_articles (N : ℕ) (C S : ℝ) : 
  N * C = 50 * S →
  (S - C) / C * 100 = 10.000000000000004 →
  N = 55 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_articles_l2291_229120


namespace NUMINAMATH_CALUDE_zeros_sum_greater_than_four_l2291_229129

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x - k * x + k

theorem zeros_sum_greater_than_four (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : k > Real.exp 2)
  (h₂ : f k x₁ = 0)
  (h₃ : f k x₂ = 0)
  (h₄ : x₁ ≠ x₂) :
  x₁ + x₂ > 4 := by
  sorry

end NUMINAMATH_CALUDE_zeros_sum_greater_than_four_l2291_229129


namespace NUMINAMATH_CALUDE_greater_fraction_l2291_229165

theorem greater_fraction (x y : ℚ) (h_sum : x + y = 5/6) (h_prod : x * y = 1/8) :
  max x y = (5 + Real.sqrt 7) / 12 := by
  sorry

end NUMINAMATH_CALUDE_greater_fraction_l2291_229165


namespace NUMINAMATH_CALUDE_root_sum_inverse_squares_l2291_229182

theorem root_sum_inverse_squares (a b c : ℝ) : 
  a^3 - 12*a^2 + 20*a - 3 = 0 →
  b^3 - 12*b^2 + 20*b - 3 = 0 →
  c^3 - 12*c^2 + 20*c - 3 = 0 →
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 328/9 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_inverse_squares_l2291_229182


namespace NUMINAMATH_CALUDE_urn_probability_l2291_229164

/-- Represents the color of a ball -/
inductive Color
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a sequence of ball draws -/
def DrawSequence := List Color

/-- The initial state of the urn -/
def initial_state : UrnState := ⟨2, 1⟩

/-- The number of operations performed -/
def num_operations : ℕ := 5

/-- The final number of balls in the urn -/
def final_total_balls : ℕ := 8

/-- The desired final state of the urn -/
def target_state : UrnState := ⟨3, 3⟩

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of valid sequences that result in the target state -/
def num_valid_sequences : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability : 
  (num_valid_sequences * sequence_probability (List.replicate num_operations Color.Red)) = 8 / 105 :=
sorry

end NUMINAMATH_CALUDE_urn_probability_l2291_229164


namespace NUMINAMATH_CALUDE_dog_tail_length_is_ten_inches_l2291_229192

/-- Represents the length of a dog's body parts in inches -/
structure DogMeasurements where
  overall_length : ℝ
  body_length : ℝ
  tail_length : ℝ

/-- Calculates the tail length of a dog given its measurements -/
def calculate_tail_length (dog : DogMeasurements) : ℝ :=
  dog.tail_length

/-- Theorem stating that the tail length of a dog with given measurements is 10 inches -/
theorem dog_tail_length_is_ten_inches (dog : DogMeasurements) 
  (h1 : dog.overall_length = 30)
  (h2 : dog.tail_length = dog.body_length / 2)
  (h3 : dog.overall_length = dog.body_length + dog.tail_length) :
  calculate_tail_length dog = 10 := by
  sorry

#check dog_tail_length_is_ten_inches

end NUMINAMATH_CALUDE_dog_tail_length_is_ten_inches_l2291_229192


namespace NUMINAMATH_CALUDE_probability_of_point_in_region_l2291_229116

-- Define the lines
def line1 (x : ℝ) : ℝ := -2 * x + 8
def line2 (x : ℝ) : ℝ := -3 * x + 9

-- Define the region of interest
def region_of_interest (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ y ≤ line1 x ∧ y ≥ line2 x

-- Define the total area under line1 in the first quadrant
def total_area : ℝ := 16

-- Define the area of the region of interest
def area_of_interest : ℝ := 14.5

-- Theorem statement
theorem probability_of_point_in_region :
  (area_of_interest / total_area) = 0.90625 :=
sorry

end NUMINAMATH_CALUDE_probability_of_point_in_region_l2291_229116


namespace NUMINAMATH_CALUDE_peters_horses_feeding_days_l2291_229193

theorem peters_horses_feeding_days :
  let num_horses : ℕ := 4
  let oats_per_meal : ℕ := 4
  let oats_meals_per_day : ℕ := 2
  let grain_per_day : ℕ := 3
  let total_food : ℕ := 132
  
  let food_per_horse_per_day : ℕ := oats_per_meal * oats_meals_per_day + grain_per_day
  let total_food_per_day : ℕ := num_horses * food_per_horse_per_day
  
  total_food / total_food_per_day = 3 :=
by sorry

end NUMINAMATH_CALUDE_peters_horses_feeding_days_l2291_229193


namespace NUMINAMATH_CALUDE_carnival_tickets_l2291_229195

/-- Calculates the total number of tickets used at a carnival -/
theorem carnival_tickets (ferris_wheel_rides bumper_car_rides roller_coaster_rides teacup_rides : ℕ)
                         (ferris_wheel_cost bumper_car_cost roller_coaster_cost teacup_cost : ℕ) :
  ferris_wheel_rides * ferris_wheel_cost +
  bumper_car_rides * bumper_car_cost +
  roller_coaster_rides * roller_coaster_cost +
  teacup_rides * teacup_cost = 105 :=
by
  -- Assuming ferris_wheel_rides = 7, bumper_car_rides = 3, roller_coaster_rides = 4, teacup_rides = 5
  -- and ferris_wheel_cost = 5, bumper_car_cost = 6, roller_coaster_cost = 8, teacup_cost = 4
  sorry


end NUMINAMATH_CALUDE_carnival_tickets_l2291_229195


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2291_229186

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (a^2 - 1) * x^2 + (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2291_229186


namespace NUMINAMATH_CALUDE_shoe_company_earnings_l2291_229181

/-- Proves that the current monthly earnings of a shoe company are $4000,
    given their annual goal and required monthly increase. -/
theorem shoe_company_earnings (annual_goal : ℕ) (monthly_increase : ℕ) (months_per_year : ℕ) :
  annual_goal = 60000 →
  monthly_increase = 1000 →
  months_per_year = 12 →
  (annual_goal / months_per_year - monthly_increase : ℕ) = 4000 := by
  sorry

end NUMINAMATH_CALUDE_shoe_company_earnings_l2291_229181


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2291_229126

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^2 + y^2 + 3^3 = 456 * (x - y).sqrt →
    ((x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2291_229126


namespace NUMINAMATH_CALUDE_rudy_total_running_time_l2291_229152

/-- Calculates the total running time for Rudy given his running segments -/
theorem rudy_total_running_time :
  let segment1 : ℝ := 5 * 10  -- 5 miles at 10 minutes per mile
  let segment2 : ℝ := 4 * 9.5 -- 4 miles at 9.5 minutes per mile
  let segment3 : ℝ := 3 * 8.5 -- 3 miles at 8.5 minutes per mile
  let segment4 : ℝ := 2 * 12  -- 2 miles at 12 minutes per mile
  segment1 + segment2 + segment3 + segment4 = 137.5 := by
sorry

end NUMINAMATH_CALUDE_rudy_total_running_time_l2291_229152


namespace NUMINAMATH_CALUDE_quadratic_completion_l2291_229188

theorem quadratic_completion (y : ℝ) : ∃ b : ℝ, y^2 + 14*y + 60 = (y + b)^2 + 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l2291_229188


namespace NUMINAMATH_CALUDE_picnic_blankets_theorem_l2291_229189

/-- The area of a blanket after a given number of folds -/
def folded_area (initial_area : ℕ) (num_folds : ℕ) : ℕ :=
  initial_area / 2^num_folds

/-- The total area of multiple blankets after folding -/
def total_folded_area (num_blankets : ℕ) (initial_area : ℕ) (num_folds : ℕ) : ℕ :=
  num_blankets * folded_area initial_area num_folds

theorem picnic_blankets_theorem :
  total_folded_area 3 64 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_picnic_blankets_theorem_l2291_229189


namespace NUMINAMATH_CALUDE_andrea_rhinestones_l2291_229173

theorem andrea_rhinestones (total : ℕ) (bought : ℚ) (found : ℚ) : 
  total = 120 → 
  bought = 2 / 5 → 
  found = 1 / 6 → 
  total - (total * bought + total * found) = 52 := by
sorry

end NUMINAMATH_CALUDE_andrea_rhinestones_l2291_229173


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_30_l2291_229161

/-- The maximum area of a rectangle with perimeter 30 meters is 225/4 square meters. -/
theorem max_area_rectangle_with_perimeter_30 :
  let perimeter : ℝ := 30
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 * (x + y) = perimeter ∧
    ∀ (a b : ℝ), a > 0 → b > 0 → 2 * (a + b) = perimeter →
      x * y ≥ a * b ∧ x * y = 225 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_30_l2291_229161


namespace NUMINAMATH_CALUDE_sneakers_cost_l2291_229154

/-- The cost of sneakers calculated from lawn mowing charges -/
theorem sneakers_cost (charge_per_yard : ℝ) (yards_to_cut : ℕ) : 
  charge_per_yard * (yards_to_cut : ℝ) = 12.90 :=
by
  sorry

#check sneakers_cost 2.15 6

end NUMINAMATH_CALUDE_sneakers_cost_l2291_229154


namespace NUMINAMATH_CALUDE_angelina_speed_l2291_229174

/-- Angelina's walk from home to grocery to gym -/
def angelina_walk (v : ℝ) : Prop :=
  let home_to_grocery_distance : ℝ := 180
  let grocery_to_gym_distance : ℝ := 240
  let home_to_grocery_time : ℝ := home_to_grocery_distance / v
  let grocery_to_gym_time : ℝ := grocery_to_gym_distance / (2 * v)
  home_to_grocery_time = grocery_to_gym_time + 40

theorem angelina_speed : ∃ v : ℝ, angelina_walk v ∧ 2 * v = 3 := by sorry

end NUMINAMATH_CALUDE_angelina_speed_l2291_229174


namespace NUMINAMATH_CALUDE_seven_numbers_even_sum_after_removal_l2291_229122

theorem seven_numbers_even_sum_after_removal (S : Finset ℕ) (h : S.card = 7) :
  ∃ x ∈ S, Even (S.sum id - x) := by
  sorry

end NUMINAMATH_CALUDE_seven_numbers_even_sum_after_removal_l2291_229122


namespace NUMINAMATH_CALUDE_soccer_game_scoring_l2291_229134

/-- Soccer game scoring theorem -/
theorem soccer_game_scoring
  (team_a_first_half : ℕ)
  (team_b_first_half : ℕ)
  (team_a_second_half : ℕ)
  (team_b_second_half : ℕ)
  (h1 : team_a_first_half = 8)
  (h2 : team_b_second_half = team_a_first_half)
  (h3 : team_a_second_half = team_b_second_half - 2)
  (h4 : team_a_first_half + team_b_first_half + team_a_second_half + team_b_second_half = 26) :
  team_b_first_half / team_a_first_half = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_soccer_game_scoring_l2291_229134


namespace NUMINAMATH_CALUDE_quadratic_trinomial_transformation_root_l2291_229113

/-- Given a quadratic trinomial ax^2 + bx + c, if we swap b and c, 
    add the result to the original trinomial, and the resulting 
    trinomial has a single root, then that root must be either 0 or -2. -/
theorem quadratic_trinomial_transformation_root (a b c : ℝ) :
  let original := fun x => a * x^2 + b * x + c
  let swapped := fun x => a * x^2 + c * x + b
  let result := fun x => original x + swapped x
  (∃! r, result r = 0) → (result 0 = 0 ∨ result (-2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_transformation_root_l2291_229113


namespace NUMINAMATH_CALUDE_tournament_games_l2291_229110

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 30 players, 435 games are played -/
theorem tournament_games :
  num_games 30 = 435 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_l2291_229110


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_l2291_229183

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for (ᶜB) ∪ P
theorem union_complement_B_P : (Bᶜ : Set ℝ) ∪ P = {x : ℝ | x ≤ 0 ∨ x ≥ 5/2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_l2291_229183


namespace NUMINAMATH_CALUDE_cube_root_8000_l2291_229146

theorem cube_root_8000 (c d : ℕ+) (h1 : (8000 : ℝ)^(1/3) = c * d^(1/3)) 
  (h2 : ∀ (k : ℕ+), (8000 : ℝ)^(1/3) = c * k^(1/3) → d ≤ k) : 
  c + d = 21 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_l2291_229146


namespace NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l2291_229191

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem two_ninths_of_three_fourths :
  (2 : ℚ) / 9 / ((3 : ℚ) / 4) = 8 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l2291_229191
