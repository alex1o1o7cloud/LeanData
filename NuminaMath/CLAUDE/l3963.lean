import Mathlib

namespace salary_calculation_l3963_396333

def initial_salary : ℝ := 5000

def final_salary (s : ℝ) : ℝ :=
  let s1 := s * 1.3
  let s2 := s1 * 0.93
  let s3 := s2 * 0.8
  let s4 := s3 - 100
  let s5 := s4 * 1.1
  let s6 := s5 * 0.9
  s6 * 0.75

theorem salary_calculation :
  final_salary initial_salary = 3516.48 := by sorry

end salary_calculation_l3963_396333


namespace flooring_rate_calculation_l3963_396399

/-- Given a rectangular room with length 5.5 meters and width 3.75 meters,
    and a total flooring cost of 20625 rupees, the rate per square meter is 1000 rupees. -/
theorem flooring_rate_calculation (length : ℝ) (width : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  width = 3.75 →
  total_cost = 20625 →
  total_cost / (length * width) = 1000 := by
  sorry

#check flooring_rate_calculation

end flooring_rate_calculation_l3963_396399


namespace quadratic_inequality_solution_set_l3963_396373

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 3) < 0} = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end quadratic_inequality_solution_set_l3963_396373


namespace steves_take_home_pay_l3963_396344

/-- Calculates the take-home pay given salary and deduction rates -/
def takeHomePay (salary : ℝ) (taxRate : ℝ) (healthcareRate : ℝ) (unionDues : ℝ) : ℝ :=
  salary - (salary * taxRate) - (salary * healthcareRate) - unionDues

/-- Theorem: Steve's take-home pay is $27,200 -/
theorem steves_take_home_pay :
  takeHomePay 40000 0.20 0.10 800 = 27200 := by
  sorry

#eval takeHomePay 40000 0.20 0.10 800

end steves_take_home_pay_l3963_396344


namespace rhombus_acute_angle_l3963_396315

-- Define a rhombus
structure Rhombus where
  -- We don't need to define all properties of a rhombus, just what we need
  acute_angle : ℝ

-- Define the plane passing through a side
structure Plane where
  -- The angles it forms with the diagonals
  angle1 : ℝ
  angle2 : ℝ

-- The main theorem
theorem rhombus_acute_angle (r : Rhombus) (p : Plane) 
  (h1 : p.angle1 = α)
  (h2 : p.angle2 = 2 * α)
  : r.acute_angle = 2 * Real.arctan (1 / (2 * Real.cos α)) := by
  sorry

end rhombus_acute_angle_l3963_396315


namespace male_alligators_mating_season_l3963_396339

/-- Represents the alligator population on Lagoon Island -/
structure AlligatorPopulation where
  males : ℕ
  adultFemales : ℕ
  juvenileFemales : ℕ

/-- Calculates the total number of alligators -/
def totalAlligators (pop : AlligatorPopulation) : ℕ :=
  pop.males + pop.adultFemales + pop.juvenileFemales

/-- Represents the population ratio of males:adult females:juvenile females -/
structure PopulationRatio where
  maleRatio : ℕ
  adultFemaleRatio : ℕ
  juvenileFemaleRatio : ℕ

/-- Theorem: Given the conditions, the number of male alligators during mating season is 10 -/
theorem male_alligators_mating_season
  (ratio : PopulationRatio)
  (nonMatingAdultFemales : ℕ)
  (resourceLimit : ℕ)
  (turtleRatio : ℕ)
  (h1 : ratio.maleRatio = 2 ∧ ratio.adultFemaleRatio = 3 ∧ ratio.juvenileFemaleRatio = 5)
  (h2 : nonMatingAdultFemales = 15)
  (h3 : resourceLimit = 200)
  (h4 : turtleRatio = 3)
  : ∃ (pop : AlligatorPopulation),
    pop.males = 10 ∧
    pop.adultFemales = 2 * nonMatingAdultFemales ∧
    totalAlligators pop ≤ resourceLimit ∧
    turtleRatio * (totalAlligators pop) ≤ 3 * resourceLimit :=
by sorry


end male_alligators_mating_season_l3963_396339


namespace positive_plus_negative_implies_negative_l3963_396391

theorem positive_plus_negative_implies_negative (a b : ℝ) :
  a > 0 → a + b < 0 → b < 0 := by sorry

end positive_plus_negative_implies_negative_l3963_396391


namespace arithmetic_mean_problem_l3963_396332

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 17 + 23 + 7 + y) / 5 = 15 → y = 20 := by
sorry

end arithmetic_mean_problem_l3963_396332


namespace expand_and_simplify_l3963_396351

theorem expand_and_simplify (x : ℝ) : (x^2 + 4) * (x - 5) = x^3 - 5*x^2 + 4*x - 20 := by
  sorry

end expand_and_simplify_l3963_396351


namespace expression_simplification_l3963_396367

theorem expression_simplification (x : ℝ) (hx : x ≠ 0) :
  (x - 2)^2 - x*(x - 1) + (x^3 - 4*x^2) / x^2 = -2*x := by
  sorry

end expression_simplification_l3963_396367


namespace james_pizza_toppings_cost_l3963_396371

/-- Calculates the cost of pizza toppings eaten by James -/
theorem james_pizza_toppings_cost :
  let num_pizzas : ℕ := 2
  let slices_per_pizza : ℕ := 6
  let topping_costs : List ℚ := [3/2, 2, 5/4]
  let james_portion : ℚ := 2/3

  let total_slices : ℕ := num_pizzas * slices_per_pizza
  let total_topping_cost : ℚ := (num_pizzas : ℚ) * (topping_costs.sum)
  let james_topping_cost : ℚ := james_portion * total_topping_cost

  james_topping_cost = 633/100 :=
by
  sorry

end james_pizza_toppings_cost_l3963_396371


namespace lines_properties_l3963_396316

/-- Two lines in 2D space -/
structure TwoLines where
  m : ℝ
  l1 : ℝ → ℝ → Prop := λ x y ↦ x + m * y - 1 = 0
  l2 : ℝ → ℝ → Prop := λ x y ↦ m * x + y - 1 = 0

/-- The distance between two parallel lines -/
def distance_between_parallel_lines (lines : TwoLines) : ℝ :=
  sorry

/-- Predicate for perpendicular lines -/
def are_perpendicular (lines : TwoLines) : Prop :=
  sorry

theorem lines_properties (lines : TwoLines) :
  (lines.l1 = lines.l2 → distance_between_parallel_lines lines = Real.sqrt 2) ∧
  (are_perpendicular lines → lines.m = 0) ∧
  lines.l2 0 1 := by
  sorry

end lines_properties_l3963_396316


namespace geometric_sequence_third_term_l3963_396328

/-- Given a geometric sequence {a_n} where a₁ = -2 and a₅ = -4, prove that a₃ = -2√2 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a1 : a 1 = -2) 
  (h_a5 : a 5 = -4) : 
  a 3 = -2 * Real.sqrt 2 := by
sorry

end geometric_sequence_third_term_l3963_396328


namespace expected_string_length_l3963_396302

/-- Represents the states of Clayton's progress -/
inductive State
  | S0  -- No letters written
  | S1  -- M written
  | S2  -- M and A written
  | S3  -- M, A, and T written
  | S4  -- M, A, T, and H written (final state)

/-- The hexagon with vertices M, M, A, T, H, S -/
def Hexagon : Type := Unit

/-- Clayton's starting position (M adjacent to M and A) -/
def start_position : Hexagon := Unit.unit

/-- Probability of moving to an adjacent vertex -/
def move_probability : ℚ := 1/2

/-- Expected number of steps to reach the final state from a given state -/
noncomputable def expected_steps : State → ℚ
  | State.S0 => 5
  | State.S1 => 4
  | State.S2 => 3
  | State.S3 => 2
  | State.S4 => 0

/-- The main theorem: Expected length of Clayton's string is 6 -/
theorem expected_string_length :
  expected_steps State.S0 + 1 = 6 := by sorry

end expected_string_length_l3963_396302


namespace gumball_machine_problem_l3963_396345

theorem gumball_machine_problem (red blue green : ℕ) : 
  blue = red / 2 →
  green = 4 * blue →
  red + blue + green = 56 →
  red = 16 := by
  sorry

end gumball_machine_problem_l3963_396345


namespace total_trip_time_l3963_396342

/-- Given that Tim drove for 5 hours and was stuck in traffic for twice as long as he was driving,
    prove that the total trip time is 15 hours. -/
theorem total_trip_time (driving_time : ℕ) (traffic_time : ℕ) : 
  driving_time = 5 →
  traffic_time = 2 * driving_time →
  driving_time + traffic_time = 15 :=
by sorry

end total_trip_time_l3963_396342


namespace six_friends_assignment_l3963_396362

/-- The number of ways to assign friends to rooms -/
def assignment_ways (n : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose n 3 * 1 * Nat.factorial 3

/-- Theorem stating the number of ways to assign 6 friends to 6 rooms -/
theorem six_friends_assignment :
  assignment_ways 6 = 1800 := by
  sorry

end six_friends_assignment_l3963_396362


namespace third_pedal_similar_l3963_396368

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a point in a 2D plane -/
def Point := ℝ × ℝ

/-- Generates the pedal triangle of a point P with respect to a given triangle -/
def pedalTriangle (P : Point) (T : Triangle) : Triangle :=
  sorry

/-- Checks if two triangles are similar -/
def areSimilar (T1 T2 : Triangle) : Prop :=
  sorry

/-- Theorem stating that the third pedal triangle is similar to the original triangle -/
theorem third_pedal_similar (P : Point) (H₀ : Triangle) :
  let H₁ := pedalTriangle P H₀
  let H₂ := pedalTriangle P H₁
  let H₃ := pedalTriangle P H₂
  areSimilar H₃ H₀ :=
by
  sorry

end third_pedal_similar_l3963_396368


namespace unspent_portion_after_transfer_l3963_396387

/-- Represents a credit card with a spending limit and balance. -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Calculates the unspent portion of a credit card's limit after a balance transfer. -/
def unspentPortionAfterTransfer (gold : CreditCard) (platinum : CreditCard) : ℝ :=
  platinum.limit - (platinum.balance + gold.balance)

/-- Theorem stating the unspent portion of the platinum card's limit after transferring the gold card's balance. -/
theorem unspent_portion_after_transfer 
  (gold : CreditCard) 
  (platinum : CreditCard) 
  (h1 : gold.limit > 0)
  (h2 : platinum.limit = 2 * gold.limit)
  (h3 : gold.balance = (1/3) * gold.limit)
  (h4 : platinum.balance = (1/4) * platinum.limit) :
  unspentPortionAfterTransfer gold platinum = (7/6) * gold.limit :=
by
  sorry

#check unspent_portion_after_transfer

end unspent_portion_after_transfer_l3963_396387


namespace function_forms_theorem_l3963_396375

/-- The set of all non-negative integers -/
def S : Set ℕ := Set.univ

/-- The condition that must be satisfied by f, g, and h -/
def satisfies_condition (f g h : ℕ → ℕ) : Prop :=
  ∀ m n, f (m + n) = g m + h n + 2 * m * n

/-- The theorem stating the only possible forms of f, g, and h -/
theorem function_forms_theorem (f g h : ℕ → ℕ) 
  (h1 : satisfies_condition f g h) (h2 : g 1 = 1) (h3 : h 1 = 1) :
  ∃ a : ℕ, a ≤ 4 ∧ 
    (∀ n, f n = n^2 - a*n + 2*a) ∧
    (∀ n, g n = n^2 - a*n + a) ∧
    (∀ n, h n = n^2 - a*n + a) :=
sorry


end function_forms_theorem_l3963_396375


namespace customers_after_family_l3963_396312

/-- Represents the taco truck's sales during lunch rush -/
def taco_truck_sales (soft_taco_price hard_taco_price : ℕ) 
  (family_hard_tacos family_soft_tacos : ℕ)
  (other_customers : ℕ) (total_revenue : ℕ) : Prop :=
  let family_revenue := family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price
  let other_revenue := other_customers * 2 * soft_taco_price
  family_revenue + other_revenue = total_revenue

/-- Theorem stating the number of customers after the family -/
theorem customers_after_family : 
  taco_truck_sales 2 5 4 3 10 66 := by sorry

end customers_after_family_l3963_396312


namespace probability_red_ball_two_fifths_l3963_396360

/-- Represents a bag of colored balls -/
structure BallBag where
  red : ℕ
  black : ℕ

/-- Calculates the probability of drawing a red ball from the bag -/
def probabilityRedBall (bag : BallBag) : ℚ :=
  bag.red / (bag.red + bag.black)

/-- Theorem: The probability of drawing a red ball from a bag with 2 red balls and 3 black balls is 2/5 -/
theorem probability_red_ball_two_fifths :
  let bag : BallBag := { red := 2, black := 3 }
  probabilityRedBall bag = 2 / 5 := by
  sorry

end probability_red_ball_two_fifths_l3963_396360


namespace set_operation_equality_l3963_396304

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, 1, 2}

theorem set_operation_equality : A ∪ (U \ B) = {-1, 0, 1, 2} := by sorry

end set_operation_equality_l3963_396304


namespace number_properties_l3963_396377

theorem number_properties :
  (∃! x : ℤ, ¬(x > 0) ∧ ¬(x < 0) ∧ x = 0) ∧
  (∃ x : ℤ, x < 0 ∧ ∀ y : ℤ, y < 0 → y ≤ x ∧ x = -1) ∧
  (∃ x : ℤ, x > 0 ∧ ∀ y : ℤ, y > 0 → x ≤ y ∧ x = 1) ∧
  (∃! x : ℤ, ∀ y : ℤ, |x| ≤ |y| ∧ x = 0) :=
by sorry

end number_properties_l3963_396377


namespace valid_words_length_10_l3963_396327

/-- Represents the number of valid words of length n -/
def validWords : ℕ → ℕ
  | 0 => 1  -- Base case: empty word
  | 1 => 2  -- Base case: "a" and "b"
  | (n+2) => validWords (n+1) + validWords n

/-- The problem statement -/
theorem valid_words_length_10 : validWords 10 = 144 := by
  sorry

end valid_words_length_10_l3963_396327


namespace equation_solution_l3963_396394

theorem equation_solution (a : ℤ) : 
  (∃ x : ℕ+, (x - 4) / 6 - (a * x - 1) / 3 = 1 / 3) ↔ a = 0 :=
by sorry

end equation_solution_l3963_396394


namespace min_value_fraction_sum_l3963_396341

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  2/a + 3/b ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end min_value_fraction_sum_l3963_396341


namespace fencing_calculation_l3963_396346

/-- Represents a rectangular field with given dimensions and fencing requirements -/
structure RectangularField where
  length : ℝ
  width : ℝ
  uncoveredSide : ℝ
  area : ℝ

/-- Calculates the required fencing for a rectangular field -/
def requiredFencing (field : RectangularField) : ℝ :=
  2 * field.width + field.length

/-- Theorem stating the required fencing for the given field specifications -/
theorem fencing_calculation (field : RectangularField) 
  (h1 : field.length = 20)
  (h2 : field.area = 390)
  (h3 : field.area = field.length * field.width)
  (h4 : field.uncoveredSide = field.length) :
  requiredFencing field = 59 := by
  sorry

end fencing_calculation_l3963_396346


namespace polar_equation_is_line_and_circle_l3963_396323

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2 * Real.sin (2 * θ)

-- Define what it means for a curve to be a line in polar coordinates
def is_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ θ₀ : ℝ, ∀ ρ θ : ℝ, f ρ θ → θ = θ₀

-- Define what it means for a curve to be a circle in polar coordinates
def is_circle (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b r : ℝ, ∀ ρ θ : ℝ, f ρ θ → (ρ * Real.cos θ - a)^2 + (ρ * Real.sin θ - b)^2 = r^2

-- Theorem statement
theorem polar_equation_is_line_and_circle :
  is_line polar_equation ∧ is_circle polar_equation := by sorry

end polar_equation_is_line_and_circle_l3963_396323


namespace quadratic_one_solution_l3963_396305

theorem quadratic_one_solution (m : ℚ) :
  (∃! x, 3 * x^2 - 6 * x + m = 0) → m = 3 := by
  sorry

end quadratic_one_solution_l3963_396305


namespace outfits_count_l3963_396355

/-- The number of outfits with different colored shirts and hats -/
def num_outfits (blue_shirts yellow_shirts pants blue_hats yellow_hats : ℕ) : ℕ :=
  blue_shirts * pants * yellow_hats + yellow_shirts * pants * blue_hats

/-- Theorem: The number of outfits is 756 given the specified numbers of clothing items -/
theorem outfits_count :
  num_outfits 6 6 7 9 9 = 756 :=
by sorry

end outfits_count_l3963_396355


namespace optimal_price_l3963_396347

def sales_volume (x : ℝ) : ℝ := -10 * x + 800

theorem optimal_price (production_cost : ℝ) (max_price : ℝ) (target_profit : ℝ) :
  production_cost = 20 →
  max_price = 45 →
  target_profit = 8000 →
  sales_volume 30 = 500 →
  sales_volume 40 = 400 →
  ∃ (price : ℝ), price ≤ max_price ∧
                 (price - production_cost) * sales_volume price = target_profit ∧
                 price = 40 := by
  sorry

end optimal_price_l3963_396347


namespace twenty_seven_thousand_six_hundred_scientific_notation_l3963_396348

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem twenty_seven_thousand_six_hundred_scientific_notation :
  toScientificNotation 27600 = ScientificNotation.mk 2.76 4 (by norm_num) :=
sorry

end twenty_seven_thousand_six_hundred_scientific_notation_l3963_396348


namespace silverware_probability_l3963_396376

theorem silverware_probability (forks spoons knives : ℕ) 
  (h1 : forks = 4) (h2 : spoons = 8) (h3 : knives = 6) : 
  let total := forks + spoons + knives
  let ways_to_choose_3 := Nat.choose total 3
  let ways_to_choose_2_spoons := Nat.choose spoons 2
  let ways_to_choose_1_knife := Nat.choose knives 1
  let favorable_outcomes := ways_to_choose_2_spoons * ways_to_choose_1_knife
  (favorable_outcomes : ℚ) / ways_to_choose_3 = 7 / 34 := by
sorry

end silverware_probability_l3963_396376


namespace remainder_problem_l3963_396303

theorem remainder_problem (n : ℤ) : (3 * n) % 7 = 3 → n % 7 = 1 := by
  sorry

end remainder_problem_l3963_396303


namespace min_value_of_expression_l3963_396320

theorem min_value_of_expression (x y : ℝ) 
  (h1 : x^2 + y^2 = 2) 
  (h2 : |x| ≠ |y|) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), a^2 + b^2 = 2 → |a| ≠ |b| → 
    (1 / (a + b)^2 + 1 / (a - b)^2) ≥ m :=
sorry

end min_value_of_expression_l3963_396320


namespace parabola_directrix_l3963_396383

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (2 * x^2 - 8 * x + 6) / 16

/-- The directrix equation -/
def directrix (y : ℝ) : Prop :=
  y = -3/2

/-- Theorem stating that the given directrix is correct for the parabola -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola x y → ∃ y_d : ℝ, directrix y_d ∧ 
  (∀ p q : ℝ × ℝ, parabola p.1 p.2 → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.2 - y_d)^2) :=
sorry

end parabola_directrix_l3963_396383


namespace cable_cost_per_person_l3963_396334

/-- Represents the cable program tiers and discount rates --/
structure CableProgram where
  tier1_channels : ℕ := 100
  tier1_cost : ℚ := 100
  tier2_channels : ℕ := 150
  tier2_cost : ℚ := 75
  tier3_channels : ℕ := 200
  tier4_channels : ℕ := 250
  discount_200 : ℚ := 0.1
  discount_300 : ℚ := 0.15
  discount_500 : ℚ := 0.2

/-- Calculates the cost for a given number of channels --/
def calculateCost (program : CableProgram) (channels : ℕ) : ℚ :=
  sorry

/-- Applies the appropriate discount based on the number of channels --/
def applyDiscount (program : CableProgram) (cost : ℚ) (channels : ℕ) : ℚ :=
  sorry

/-- Theorem: The cost per person for 375 channels split among 4 people is $57.11 --/
theorem cable_cost_per_person (program : CableProgram) :
  let total_cost := calculateCost program 375
  let discounted_cost := applyDiscount program total_cost 375
  let cost_per_person := discounted_cost / 4
  cost_per_person = 57.11 := by
  sorry

end cable_cost_per_person_l3963_396334


namespace range_of_f_l3963_396307

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x - 2

-- State the theorem
theorem range_of_f :
  ∀ x ∈ Set.Icc 0 (Real.sqrt 3 / 2),
  ∃ y ∈ Set.Icc (-3) (-2),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-3) (-2) :=
by sorry

-- Define the trigonometric identity
axiom cos_triple_angle (θ : ℝ) :
  Real.cos (3 * θ) = 4 * (Real.cos θ)^3 - 3 * (Real.cos θ)

end range_of_f_l3963_396307


namespace odd_even_function_sum_l3963_396378

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- State the theorem
theorem odd_even_function_sum (f g : ℝ → ℝ) (h : ℝ → ℝ) 
  (hf : IsOdd f) (hg : IsEven g) 
  (sum_eq : ∀ x ≠ 1, f x + g x = 1 / (x - 1)) :
  ∀ x ≠ 1, f x = x / (x^2 - 1) := by
  sorry

end odd_even_function_sum_l3963_396378


namespace textbook_profit_example_l3963_396325

/-- The profit of a textbook sale given its cost and selling prices -/
def textbook_profit (cost_price selling_price : ℝ) : ℝ :=
  selling_price - cost_price

/-- Theorem: The profit of a textbook sold by a bookstore is $11,
    given that the cost price is $44 and the selling price is $55. -/
theorem textbook_profit_example : textbook_profit 44 55 = 11 := by
  sorry

end textbook_profit_example_l3963_396325


namespace towel_price_calculation_l3963_396386

theorem towel_price_calculation (price1 price2 avg_price : ℕ) 
  (h1 : price1 = 100)
  (h2 : price2 = 150)
  (h3 : avg_price = 145) : 
  ∃ (unknown_price : ℕ), 
    (3 * price1 + 5 * price2 + 2 * unknown_price) / 10 = avg_price ∧ 
    unknown_price = 200 := by
sorry

end towel_price_calculation_l3963_396386


namespace fifth_element_row_20_l3963_396321

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- The element at position k in row n of Pascal's triangle -/
def pascal_element (n k : ℕ) : ℕ :=
  binomial n (k - 1)

/-- The fifth element in row 20 of Pascal's triangle is 4845 -/
theorem fifth_element_row_20 : pascal_element 20 5 = 4845 := by
  sorry

end fifth_element_row_20_l3963_396321


namespace geometric_sequence_common_ratio_l3963_396354

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (m : ℕ) 
  (h1 : m > 0)
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q))
  (h3 : ∀ n, a (n + 1) = q * a n)
  (h4 : S (2 * m) / S m = 9)
  (h5 : a (2 * m) / a m = (5 * m + 1) / (m - 1)) :
  q = 2 := by sorry

end geometric_sequence_common_ratio_l3963_396354


namespace quadratic_equal_roots_l3963_396396

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0 ∧ 
   ∀ y : ℝ, k * y^2 + 2 * y + 1 = 0 → y = x) ↔ 
  k = 1 := by
  sorry

end quadratic_equal_roots_l3963_396396


namespace fraction_equality_l3963_396318

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (2 * a - b) / (a + 4 * b) = 3) : 
  (a - 4 * b) / (2 * a + b) = 17 / 25 := by
  sorry

end fraction_equality_l3963_396318


namespace odd_product_over_sum_equals_fifteen_fourths_l3963_396319

theorem odd_product_over_sum_equals_fifteen_fourths : 
  (1 * 3 * 5 * 7) / (1 + 2 + 3 + 4 + 5 + 6 + 7) = 15 / 4 := by
sorry

end odd_product_over_sum_equals_fifteen_fourths_l3963_396319


namespace ladybug_dots_average_l3963_396374

/-- The number of ladybugs caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The total number of dots on all ladybugs -/
def total_dots : ℕ := 78

/-- The average number of dots per ladybug -/
def average_dots : ℚ := total_dots / (monday_ladybugs + tuesday_ladybugs)

theorem ladybug_dots_average :
  average_dots = 6 := by sorry

end ladybug_dots_average_l3963_396374


namespace necessary_condition_implies_m_range_l3963_396338

-- Define the sets A and B
def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
def B (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ [-1, 1] ∧ y = 1/3 * x + m}

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

-- State the theorem
theorem necessary_condition_implies_m_range :
  ∀ m : ℝ, (∀ x : ℝ, q m x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q m x) →
  m > 1/3 ∧ m < 2/3 := by
  sorry

end necessary_condition_implies_m_range_l3963_396338


namespace arccos_negative_half_l3963_396311

theorem arccos_negative_half : Real.arccos (-1/2) = 2*π/3 := by sorry

end arccos_negative_half_l3963_396311


namespace solution_difference_l3963_396310

-- Define the equation
def equation (x : ℝ) : Prop :=
  (6 * x - 18) / (x^2 + 3 * x - 18) = x + 3

-- Define the theorem
theorem solution_difference (r s : ℝ) : 
  equation r ∧ equation s ∧ r ≠ s ∧ r > s → r - s = 3 := by
  sorry


end solution_difference_l3963_396310


namespace zoo_animal_count_l3963_396366

/-- Calculates the total number of animals in a zoo with specific enclosure arrangements. -/
def total_animals_in_zoo : ℕ :=
  let tiger_enclosures : ℕ := 4
  let zebra_enclosures : ℕ := tiger_enclosures * 2
  let elephant_enclosures : ℕ := zebra_enclosures + 1
  let giraffe_enclosures : ℕ := elephant_enclosures * 3
  let rhino_enclosures : ℕ := 4

  let tigers : ℕ := tiger_enclosures * 4
  let zebras : ℕ := zebra_enclosures * 10
  let elephants : ℕ := elephant_enclosures * 3
  let giraffes : ℕ := giraffe_enclosures * 2
  let rhinos : ℕ := rhino_enclosures * 1

  tigers + zebras + elephants + giraffes + rhinos

/-- Theorem stating that the total number of animals in the zoo is 181. -/
theorem zoo_animal_count : total_animals_in_zoo = 181 := by
  sorry

end zoo_animal_count_l3963_396366


namespace point_difference_l3963_396372

def zachScore : ℕ := 42
def benScore : ℕ := 21

theorem point_difference : zachScore - benScore = 21 := by
  sorry

end point_difference_l3963_396372


namespace triangle_problem_l3963_396388

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  (1 + Real.sin t.B + Real.cos t.B) * (Real.cos (t.B / 2) - Real.sin (t.B / 2)) = 
    7 / 12 * Real.sqrt (2 + 2 * Real.cos t.B) ∧
  t.c / t.a = 2 / 3

-- Define point D on side AC such that BD = AC
def point_D (t : Triangle) (D : ℝ) : Prop :=
  0 < D ∧ D < t.c ∧ Real.sqrt (t.a^2 + D^2 - 2 * t.a * D * Real.cos t.A) = t.c

-- State the theorem
theorem triangle_problem (t : Triangle) (D : ℝ) :
  given_conditions t → point_D t D →
  Real.cos t.B = 7 / 12 ∧ D / (t.c - D) = 2 :=
by sorry

end triangle_problem_l3963_396388


namespace geometric_arithmetic_sequence_exists_l3963_396364

theorem geometric_arithmetic_sequence_exists : ∃ (a b : ℝ),
  1 < a ∧ a < b ∧ b < 16 ∧
  (∃ (r : ℝ), a = 1 * r ∧ b = 1 * r^2) ∧
  (∃ (d : ℝ), b = a + d ∧ 16 = b + d) ∧
  a + b = 12.64 := by
  sorry

end geometric_arithmetic_sequence_exists_l3963_396364


namespace P_sufficient_not_necessary_for_Q_l3963_396306

-- Define P and Q as propositions depending on a real number x
def P (x : ℝ) : Prop := (2*x - 3)^2 < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- State the theorem
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) := by
sorry

end P_sufficient_not_necessary_for_Q_l3963_396306


namespace lemonade_percentage_in_second_solution_l3963_396337

/-- Represents a solution mixture --/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)

/-- Represents the mixture of two solutions --/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)
  (proportion2 : ℝ)
  (total_carbonated_water : ℝ)

/-- The theorem to be proved --/
theorem lemonade_percentage_in_second_solution 
  (mix : Mixture) 
  (h1 : mix.solution1.lemonade = 0.2)
  (h2 : mix.solution1.carbonated_water = 0.8)
  (h3 : mix.solution2.lemonade + mix.solution2.carbonated_water = 1)
  (h4 : mix.proportion1 = 0.4)
  (h5 : mix.proportion2 = 0.6)
  (h6 : mix.total_carbonated_water = 0.65) :
  mix.solution2.lemonade = 0.9945 :=
sorry

end lemonade_percentage_in_second_solution_l3963_396337


namespace ages_solution_l3963_396313

/-- Represents the ages of Henry, Jill, and Alex -/
structure Ages where
  henry : ℕ
  jill : ℕ
  alex : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.henry + ages.jill + ages.alex = 90 ∧
  ages.henry - 5 = 2 * (ages.jill - 5) ∧
  ages.henry + ages.jill - 10 = ages.alex

/-- The theorem to prove -/
theorem ages_solution :
  ∃ (ages : Ages), satisfies_conditions ages ∧ 
    ages.henry = 32 ∧ ages.jill = 18 ∧ ages.alex = 40 := by
  sorry

end ages_solution_l3963_396313


namespace meeting_distance_calculation_l3963_396380

/-- Represents the problem of calculating the distance to a meeting location --/
theorem meeting_distance_calculation (initial_speed : ℝ) (speed_increase : ℝ) 
  (late_time : ℝ) (early_time : ℝ) :
  initial_speed = 40 →
  speed_increase = 20 →
  late_time = 1.5 →
  early_time = 1 →
  ∃ (distance : ℝ) (total_time : ℝ),
    distance = initial_speed * (total_time + late_time) ∧
    distance = initial_speed + (initial_speed + speed_increase) * (total_time - early_time - 1) ∧
    distance = 420 := by
  sorry


end meeting_distance_calculation_l3963_396380


namespace baker_remaining_pastries_l3963_396309

def remaining_pastries (initial : ℕ) (sold : ℕ) : ℕ :=
  initial - sold

theorem baker_remaining_pastries :
  remaining_pastries 56 29 = 27 := by
  sorry

end baker_remaining_pastries_l3963_396309


namespace probability_of_selecting_specific_car_type_l3963_396370

theorem probability_of_selecting_specific_car_type 
  (total_car_types : ℕ) 
  (cars_selected : ℕ) 
  (h1 : total_car_types = 5) 
  (h2 : cars_selected = 2) :
  (cars_selected : ℚ) / (total_car_types.choose cars_selected) = 2/5 := by
sorry

end probability_of_selecting_specific_car_type_l3963_396370


namespace g_of_5_equals_22_l3963_396335

/-- Given that g(x) = 4x + 2 for all x, prove that g(5) = 22 -/
theorem g_of_5_equals_22 (g : ℝ → ℝ) (h : ∀ x, g x = 4 * x + 2) : g 5 = 22 := by
  sorry

end g_of_5_equals_22_l3963_396335


namespace keystone_arch_angle_l3963_396349

/-- Represents a keystone arch configuration -/
structure KeystoneArch where
  num_trapezoids : ℕ
  trapezoids_are_congruent : Bool
  trapezoids_are_isosceles : Bool
  sides_meet_at_center : Bool

/-- Calculates the larger interior angle of a trapezoid in the keystone arch -/
def larger_interior_angle (arch : KeystoneArch) : ℝ :=
  sorry

/-- Theorem stating that the larger interior angle of each trapezoid in a 12-piece keystone arch is 97.5° -/
theorem keystone_arch_angle (arch : KeystoneArch) :
  arch.num_trapezoids = 12 ∧ 
  arch.trapezoids_are_congruent ∧ 
  arch.trapezoids_are_isosceles ∧ 
  arch.sides_meet_at_center →
  larger_interior_angle arch = 97.5 :=
sorry

end keystone_arch_angle_l3963_396349


namespace olaf_game_ratio_l3963_396392

theorem olaf_game_ratio : 
  ∀ (father_points son_points : ℕ),
  father_points = 7 →
  ∃ (x : ℕ), son_points = x * father_points →
  father_points + son_points = 28 →
  son_points / father_points = 3 := by
sorry

end olaf_game_ratio_l3963_396392


namespace fly_distance_from_floor_l3963_396329

theorem fly_distance_from_floor (x y z h : ℝ) :
  x = 2 →
  y = 5 →
  h - z = 7 →
  x^2 + y^2 + z^2 = 11^2 →
  h = Real.sqrt 92 + 7 := by
sorry

end fly_distance_from_floor_l3963_396329


namespace set_inclusion_implies_a_value_l3963_396340

theorem set_inclusion_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {1, 2, a}
  let B : Set ℝ := {1, a^2 - a}
  A ⊇ B → a = -1 ∨ a = 0 := by
sorry

end set_inclusion_implies_a_value_l3963_396340


namespace function_floor_property_l3963_396385

theorem function_floor_property (f : ℝ → ℝ) :
  (∃ g : ℝ → ℝ, ∀ x y : ℝ, f x + f y = ⌊g (x + y)⌋) →
  ∃ n : ℤ, ∀ x : ℝ, f x = n / 2 := by
  sorry

end function_floor_property_l3963_396385


namespace problem_sculpture_area_l3963_396353

/-- Represents a pyramid-like sculpture made of unit cubes -/
structure PyramidSculpture where
  total_cubes : ℕ
  num_layers : ℕ
  layer_sizes : List ℕ
  (total_cubes_sum : total_cubes = layer_sizes.sum)
  (layer_count : num_layers = layer_sizes.length)

/-- Calculates the exposed surface area of a pyramid sculpture -/
def exposed_surface_area (p : PyramidSculpture) : ℕ :=
  sorry

/-- The specific pyramid sculpture described in the problem -/
def problem_sculpture : PyramidSculpture :=
  { total_cubes := 19
  , num_layers := 4
  , layer_sizes := [1, 3, 5, 10]
  , total_cubes_sum := by sorry
  , layer_count := by sorry
  }

/-- Theorem stating that the exposed surface area of the problem sculpture is 43 square meters -/
theorem problem_sculpture_area : exposed_surface_area problem_sculpture = 43 := by
  sorry

end problem_sculpture_area_l3963_396353


namespace three_digit_number_divisibility_l3963_396369

theorem three_digit_number_divisibility : ∃! x : ℕ, 
  100 ≤ x ∧ x ≤ 999 ∧ 
  (x - 6) % 7 = 0 ∧ 
  (x - 7) % 8 = 0 ∧ 
  (x - 8) % 9 = 0 ∧ 
  x = 503 := by sorry

end three_digit_number_divisibility_l3963_396369


namespace no_solutions_for_equation_l3963_396326

theorem no_solutions_for_equation : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (2 / x + 3 / y = 1 / (x + y)) := by
  sorry

end no_solutions_for_equation_l3963_396326


namespace mango_price_reduction_mango_price_reduction_result_l3963_396357

/-- Calculates the percentage reduction in mango prices --/
theorem mango_price_reduction (original_cost : ℝ) (original_quantity : ℕ) 
  (reduced_cost : ℝ) (original_purchase : ℕ) (additional_mangoes : ℕ) : ℝ :=
  let original_price_per_mango := original_cost / original_quantity
  let original_purchase_quantity := reduced_cost / original_price_per_mango
  let new_purchase_quantity := original_purchase_quantity + additional_mangoes
  let new_price_per_mango := reduced_cost / new_purchase_quantity
  let price_reduction_percentage := (original_price_per_mango - new_price_per_mango) / original_price_per_mango * 100
  price_reduction_percentage

/-- The percentage reduction in mango prices is approximately 9.91% --/
theorem mango_price_reduction_result : 
  abs (mango_price_reduction 450 135 360 108 12 - 9.91) < 0.01 := by
  sorry

end mango_price_reduction_mango_price_reduction_result_l3963_396357


namespace scientific_notation_of_120_million_l3963_396314

theorem scientific_notation_of_120_million : 
  ∃ (a : ℝ) (n : ℤ), 120000000 = a * (10 : ℝ)^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.2 ∧ n = 7 := by
  sorry

end scientific_notation_of_120_million_l3963_396314


namespace reaction_properties_l3963_396384

-- Define the reaction components
structure Reaction where
  k2cr2o7 : ℕ
  hcl : ℕ
  kcl : ℕ
  crcl3 : ℕ
  cl2 : ℕ
  h2o : ℕ

-- Define oxidation states
def oxidation_state_cr_initial : Int := 6
def oxidation_state_cr_final : Int := 3
def oxidation_state_cl_initial : Int := -1
def oxidation_state_cl_final : Int := 0

-- Define the balanced equation
def balanced_reaction : Reaction := {
  k2cr2o7 := 2,
  hcl := 14,
  kcl := 2,
  crcl3 := 2,
  cl2 := 3,
  h2o := 7
}

-- Define the number of electrons transferred
def electrons_transferred : ℕ := 6

-- Define the oxidizing agent
def oxidizing_agent : String := "K2Cr2O7"

-- Define the element being oxidized
def element_oxidized : String := "Cl in HCl"

-- Define the oxidation product
def oxidation_product : String := "Cl2"

-- Define the mass ratio of oxidized to unoxidized HCl
def mass_ratio_oxidized_unoxidized : Rat := 3 / 4

-- Define the number of electrons transferred for 0.1 mol of Cl2
def electrons_transferred_for_0_1_mol_cl2 : ℕ := 120400000000000000000000

theorem reaction_properties :
  -- (1) Verify the oxidizing agent, element oxidized, and oxidation product
  (oxidizing_agent = "K2Cr2O7") ∧
  (element_oxidized = "Cl in HCl") ∧
  (oxidation_product = "Cl2") ∧
  -- (2) Verify the mass ratio of oxidized to unoxidized HCl
  (mass_ratio_oxidized_unoxidized = 3 / 4) ∧
  -- (3) Verify the number of electrons transferred for 0.1 mol of Cl2
  (electrons_transferred_for_0_1_mol_cl2 = 120400000000000000000000) := by
  sorry

#check reaction_properties

end reaction_properties_l3963_396384


namespace complex_number_existence_l3963_396352

theorem complex_number_existence : ∃! (z₁ z₂ : ℂ),
  (z₁ + 10 / z₁).im = 0 ∧
  (z₂ + 10 / z₂).im = 0 ∧
  (z₁ + 4).re = -(z₁ + 4).im ∧
  (z₂ + 4).re = -(z₂ + 4).im ∧
  z₁ = -1 - 3*I ∧
  z₂ = -3 - I :=
by sorry

end complex_number_existence_l3963_396352


namespace four_integer_sum_problem_l3963_396395

theorem four_integer_sum_problem (a b c d : ℕ+) 
  (h_order : a < b ∧ b < c ∧ c < d)
  (h_sums_different : a + b ≠ a + c ∧ a + b ≠ a + d ∧ a + b ≠ b + c ∧ 
                      a + b ≠ b + d ∧ a + b ≠ c + d ∧ a + c ≠ a + d ∧ 
                      a + c ≠ b + c ∧ a + c ≠ b + d ∧ a + c ≠ c + d ∧ 
                      a + d ≠ b + c ∧ a + d ≠ b + d ∧ a + d ≠ c + d ∧ 
                      b + c ≠ b + d ∧ b + c ≠ c + d ∧ b + d ≠ c + d)
  (h_smallest_sums : min (a + b) (min (a + c) (min (a + d) (min (b + c) (min (b + d) (c + d))))) = 6 ∧
                     min (a + c) (min (a + d) (min (b + c) (min (b + d) (c + d)))) = 8 ∧
                     min (a + d) (min (b + c) (min (b + d) (c + d))) = 12 ∧
                     min (b + c) (min (b + d) (c + d)) = 21) : 
  d = 20 := by
sorry

end four_integer_sum_problem_l3963_396395


namespace animal_costs_l3963_396382

theorem animal_costs (dog_cost cow_cost horse_cost : ℚ) : 
  cow_cost = 4 * dog_cost →
  horse_cost = 4 * cow_cost →
  dog_cost + 2 * cow_cost + horse_cost = 200 →
  dog_cost = 8 ∧ cow_cost = 32 ∧ horse_cost = 128 := by
sorry

end animal_costs_l3963_396382


namespace min_value_2x_l3963_396358

theorem min_value_2x (x y z : ℕ+) (h1 : 2 * x = 6 * z) (h2 : x + y + z = 26) : 2 * x = 6 := by
  sorry

end min_value_2x_l3963_396358


namespace blake_change_l3963_396331

def oranges_cost : ℝ := 40
def apples_cost : ℝ := 50
def mangoes_cost : ℝ := 60
def strawberries_cost : ℝ := 30
def bananas_cost : ℝ := 20
def strawberries_discount : ℝ := 0.10
def bananas_discount : ℝ := 0.05
def blake_money : ℝ := 300

theorem blake_change :
  let discounted_strawberries := strawberries_cost * (1 - strawberries_discount)
  let discounted_bananas := bananas_cost * (1 - bananas_discount)
  let total_cost := oranges_cost + apples_cost + mangoes_cost + discounted_strawberries + discounted_bananas
  blake_money - total_cost = 104 := by sorry

end blake_change_l3963_396331


namespace fishing_ratio_l3963_396361

/-- Given the conditions of the fishing problem, prove that the ratio of trout to bass is 1:4 -/
theorem fishing_ratio : 
  ∀ (trout bass bluegill : ℕ),
  bass = 32 →
  bluegill = 2 * bass →
  trout + bass + bluegill = 104 →
  trout.gcd bass = 8 →
  (trout / 8 : ℚ) = 1 ∧ (bass / 8 : ℚ) = 4 := by
  sorry

end fishing_ratio_l3963_396361


namespace range_of_x_for_positive_f_l3963_396317

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a-4)*x + 4-2*a

-- State the theorem
theorem range_of_x_for_positive_f :
  ∀ a ∈ Set.Icc (-1 : ℝ) 1,
    (∀ x, f a x > 0) ↔ (∀ x, x < 1 ∨ x > 3) := by sorry

end range_of_x_for_positive_f_l3963_396317


namespace greatest_integer_pi_minus_five_l3963_396308

theorem greatest_integer_pi_minus_five :
  ⌊Real.pi - 5⌋ = -2 := by
  sorry

end greatest_integer_pi_minus_five_l3963_396308


namespace monkey_banana_distribution_l3963_396389

/-- Represents the number of bananas each monkey receives when dividing the total equally -/
def bananas_per_monkey (num_monkeys : ℕ) (num_piles_type1 num_piles_type2 : ℕ) 
  (hands_per_pile_type1 hands_per_pile_type2 : ℕ) 
  (bananas_per_hand_type1 bananas_per_hand_type2 : ℕ) : ℕ :=
  let total_bananas := 
    num_piles_type1 * hands_per_pile_type1 * bananas_per_hand_type1 +
    num_piles_type2 * hands_per_pile_type2 * bananas_per_hand_type2
  total_bananas / num_monkeys

/-- Theorem stating that given the problem conditions, each monkey receives 99 bananas -/
theorem monkey_banana_distribution :
  bananas_per_monkey 12 6 4 9 12 14 9 = 99 := by
  sorry

end monkey_banana_distribution_l3963_396389


namespace milk_butter_revenue_l3963_396336

/-- Calculates the total revenue from selling milk and butter --/
def total_revenue (num_cows : ℕ) (milk_per_cow : ℕ) (milk_price : ℚ) (butter_sticks_per_gallon : ℕ) (butter_price : ℚ) : ℚ :=
  let total_milk := num_cows * milk_per_cow
  let milk_revenue := total_milk * milk_price
  milk_revenue

theorem milk_butter_revenue :
  let num_cows : ℕ := 12
  let milk_per_cow : ℕ := 4
  let milk_price : ℚ := 3
  let butter_sticks_per_gallon : ℕ := 2
  let butter_price : ℚ := 3/2
  total_revenue num_cows milk_per_cow milk_price butter_sticks_per_gallon butter_price = 144 := by
  sorry

end milk_butter_revenue_l3963_396336


namespace jokes_theorem_l3963_396301

def calculate_jokes (initial : ℕ) : ℕ :=
  initial + 2 * initial + 4 * initial + 8 * initial + 16 * initial

def total_jokes : ℕ :=
  calculate_jokes 11 + calculate_jokes 7 + calculate_jokes 5 + calculate_jokes 3

theorem jokes_theorem : total_jokes = 806 := by
  sorry

end jokes_theorem_l3963_396301


namespace total_material_bought_l3963_396330

/-- The total amount of material bought by a construction company -/
theorem total_material_bought (gravel sand : ℝ) (h1 : gravel = 5.91) (h2 : sand = 8.11) :
  gravel + sand = 14.02 := by
  sorry

end total_material_bought_l3963_396330


namespace chocolate_cost_l3963_396398

theorem chocolate_cost (box_size : ℕ) (box_cost : ℚ) (total_candies : ℕ) : 
  box_size = 30 → 
  box_cost = 9 → 
  total_candies = 450 → 
  (total_candies / box_size : ℚ) * box_cost = 135 := by
sorry

end chocolate_cost_l3963_396398


namespace prob_same_color_is_17_35_l3963_396397

/-- A box containing chess pieces -/
structure ChessBox where
  total_pieces : ℕ
  black_pieces : ℕ
  white_pieces : ℕ
  prob_two_black : ℚ
  prob_two_white : ℚ

/-- The probability of drawing two pieces of the same color -/
def prob_same_color (box : ChessBox) : ℚ :=
  box.prob_two_black + box.prob_two_white

/-- Theorem stating the probability of drawing two pieces of the same color -/
theorem prob_same_color_is_17_35 (box : ChessBox)
  (h1 : box.total_pieces = 15)
  (h2 : box.black_pieces = 6)
  (h3 : box.white_pieces = 9)
  (h4 : box.prob_two_black = 1 / 7)
  (h5 : box.prob_two_white = 12 / 35) :
  prob_same_color box = 17 / 35 := by
  sorry

end prob_same_color_is_17_35_l3963_396397


namespace largest_odd_digit_multiple_of_5_is_correct_l3963_396379

/-- A function that checks if a positive integer has only odd digits -/
def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 1

/-- The largest positive integer less than 10000 with only odd digits that is a multiple of 5 -/
def largest_odd_digit_multiple_of_5 : ℕ := 9995

theorem largest_odd_digit_multiple_of_5_is_correct :
  (largest_odd_digit_multiple_of_5 < 10000) ∧
  (has_only_odd_digits largest_odd_digit_multiple_of_5) ∧
  (largest_odd_digit_multiple_of_5 % 5 = 0) ∧
  (∀ n : ℕ, n < 10000 → has_only_odd_digits n → n % 5 = 0 → n ≤ largest_odd_digit_multiple_of_5) :=
by sorry

#eval largest_odd_digit_multiple_of_5

end largest_odd_digit_multiple_of_5_is_correct_l3963_396379


namespace wrong_to_right_ratio_l3963_396381

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) 
  (h1 : total = 54) (h2 : correct = 18) :
  (total - correct) / correct = 2 := by
  sorry

end wrong_to_right_ratio_l3963_396381


namespace arithmetic_sequence_10th_term_l3963_396356

/-- An arithmetic sequence with given second and third terms -/
def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ a 2 = 2 ∧ a 3 = 4

/-- The 10th term of the arithmetic sequence is 18 -/
theorem arithmetic_sequence_10th_term (a : ℕ → ℝ) (h : arithmeticSequence a) : 
  a 10 = 18 := by
  sorry

end arithmetic_sequence_10th_term_l3963_396356


namespace profit_comparison_l3963_396390

/-- The profit function for Product A before upgrade -/
def profit_A_before (raw_material : ℝ) : ℝ := 120000 * raw_material

/-- The profit function for Product A after upgrade -/
def profit_A_after (x : ℝ) : ℝ := 12 * (500 - x) * (1 + 0.005 * x)

/-- The profit function for Product B -/
def profit_B (x a : ℝ) : ℝ := 12 * (a - 0.013 * x) * x

theorem profit_comparison (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, 0 < x ∧ x ≤ 300 ∧ profit_A_after x ≥ profit_A_before 500) ∧
  (∀ x : ℝ, 0 < x → x ≤ 300 → profit_B x a ≤ profit_A_after x) →
  a ≤ 5.5 :=
sorry

end profit_comparison_l3963_396390


namespace storks_joining_fence_l3963_396363

theorem storks_joining_fence (initial_birds initial_storks : ℕ) 
  (h1 : initial_birds = 6)
  (h2 : initial_storks = 3)
  (joined_storks : ℕ)
  (h3 : initial_birds = initial_storks + joined_storks + 1) :
  joined_storks = 2 := by
sorry

end storks_joining_fence_l3963_396363


namespace smallest_c_for_g_range_contains_one_l3963_396365

/-- The function g(x) defined as x^2 - 2x + c -/
def g (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

/-- Theorem stating that 2 is the smallest value of c such that 1 is in the range of g(x) -/
theorem smallest_c_for_g_range_contains_one :
  ∀ c : ℝ, (∃ x : ℝ, g c x = 1) ↔ c ≥ 2 :=
sorry

end smallest_c_for_g_range_contains_one_l3963_396365


namespace production_increase_l3963_396359

theorem production_increase (original_hours original_output : ℝ) 
  (h_positive_hours : original_hours > 0)
  (h_positive_output : original_output > 0) :
  let new_hours := 0.9 * original_hours
  let new_rate := 2 * (original_output / original_hours)
  let new_output := new_hours * new_rate
  (new_output - original_output) / original_output = 0.8 := by
sorry

end production_increase_l3963_396359


namespace complex_cube_root_l3963_396322

theorem complex_cube_root (a b : ℕ+) :
  (a + b * Complex.I) ^ 3 = 2 + 11 * Complex.I →
  a + b * Complex.I = 2 + Complex.I :=
by sorry

end complex_cube_root_l3963_396322


namespace lcm_of_40_and_14_l3963_396350

theorem lcm_of_40_and_14 :
  let n : ℕ := 40
  let m : ℕ := 14
  let gcf : ℕ := 10
  Nat.gcd n m = gcf →
  Nat.lcm n m = 56 := by
sorry

end lcm_of_40_and_14_l3963_396350


namespace total_weight_is_350_l3963_396343

/-- Represents the weight of a single box in kilograms -/
def box_weight : ℕ := 25

/-- Represents the number of columns with 3 boxes -/
def columns_with_3 : ℕ := 1

/-- Represents the number of columns with 2 boxes -/
def columns_with_2 : ℕ := 4

/-- Represents the number of columns with 1 box -/
def columns_with_1 : ℕ := 3

/-- Calculates the total number of boxes in the stack -/
def total_boxes : ℕ := columns_with_3 * 3 + columns_with_2 * 2 + columns_with_1 * 1

/-- Calculates the total weight of all boxes in kilograms -/
def total_weight : ℕ := total_boxes * box_weight

/-- Theorem stating that the total weight of all boxes is 350 kg -/
theorem total_weight_is_350 : total_weight = 350 := by sorry

end total_weight_is_350_l3963_396343


namespace tan_difference_equals_one_eighth_l3963_396324

theorem tan_difference_equals_one_eighth 
  (α β : ℝ) 
  (h1 : Real.tan (α - β) = 2/3) 
  (h2 : Real.tan (π/6 - β) = 1/2) : 
  Real.tan (α - π/6) = 1/8 := by
  sorry

end tan_difference_equals_one_eighth_l3963_396324


namespace ab_less_than_a_plus_b_l3963_396300

theorem ab_less_than_a_plus_b (a b : ℝ) (ha : a < 1) (hb : b > 1) : a * b < a + b := by
  sorry

end ab_less_than_a_plus_b_l3963_396300


namespace rectangle_burn_time_l3963_396393

/-- Represents a rectangle made of wooden toothpicks -/
structure ToothpickRectangle where
  rows : Nat
  cols : Nat
  burnTime : Nat  -- Time for one toothpick to burn in seconds

/-- Calculates the time for the entire structure to burn -/
def burnTime (rect : ToothpickRectangle) : Nat :=
  let maxPath := rect.rows + rect.cols - 2  -- Longest path from corner to middle
  (maxPath * rect.burnTime) + (rect.burnTime / 2)

theorem rectangle_burn_time :
  let rect := ToothpickRectangle.mk 3 5 10
  burnTime rect = 65 := by
  sorry

#eval burnTime (ToothpickRectangle.mk 3 5 10)

end rectangle_burn_time_l3963_396393
