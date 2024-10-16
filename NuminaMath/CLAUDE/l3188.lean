import Mathlib

namespace NUMINAMATH_CALUDE_angle_DEB_is_165_l3188_318810

-- Define the geometric configuration
structure GeometricConfiguration where
  -- Triangle ABC
  angleACB : ℝ
  angleABC : ℝ
  -- Other angles
  angleADE : ℝ
  angleCDE : ℝ
  -- AEB is a straight angle
  angleAEB : ℝ

-- Define the theorem
theorem angle_DEB_is_165 (config : GeometricConfiguration) 
  (h1 : config.angleACB = 90)
  (h2 : config.angleABC = 55)
  (h3 : config.angleADE = 130)
  (h4 : config.angleCDE = 50)
  (h5 : config.angleAEB = 180) :
  ∃ (angleDEB : ℝ), angleDEB = 165 := by
    sorry

end NUMINAMATH_CALUDE_angle_DEB_is_165_l3188_318810


namespace NUMINAMATH_CALUDE_total_is_600_l3188_318897

/-- Represents the shares of money for three individuals -/
structure Shares :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The conditions of the money division problem -/
def SatisfiesConditions (s : Shares) : Prop :=
  s.a = (2/3) * (s.b + s.c) ∧
  s.b = (6/9) * (s.a + s.c) ∧
  s.a = 240

/-- The theorem stating that the total amount is 600 given the conditions -/
theorem total_is_600 (s : Shares) (h : SatisfiesConditions s) :
  s.a + s.b + s.c = 600 := by
  sorry

#check total_is_600

end NUMINAMATH_CALUDE_total_is_600_l3188_318897


namespace NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l3188_318890

theorem remainder_of_product (a b m : ℕ) : (a * b) % m = ((a % m) * (b % m)) % m := by sorry

theorem product_remainder : (2002 * 1493) % 300 = 86 := by
  -- The proof would go here, but we're omitting it as per instructions
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_product_remainder_l3188_318890


namespace NUMINAMATH_CALUDE_lei_lei_sheep_count_l3188_318805

/-- The number of sheep Lei Lei bought -/
def num_sheep : ℕ := 10

/-- The initial average price per sheep in yuan -/
def initial_avg_price : ℚ := sorry

/-- The total price of all sheep and goats -/
def total_price : ℚ := sorry

/-- The number of goats Lei Lei bought -/
def num_goats : ℕ := sorry

theorem lei_lei_sheep_count :
  (total_price + 2 * (initial_avg_price + 60) = (num_sheep + 2) * (initial_avg_price + 60)) ∧
  (total_price - 2 * (initial_avg_price - 90) = (num_sheep - 2) * (initial_avg_price - 90)) →
  num_sheep = 10 := by sorry

end NUMINAMATH_CALUDE_lei_lei_sheep_count_l3188_318805


namespace NUMINAMATH_CALUDE_max_y_coordinate_l3188_318802

theorem max_y_coordinate (x y : ℝ) :
  (x^2 / 49) + ((y - 3)^2 / 25) + y = 0 →
  y ≤ (-19 + Real.sqrt 325) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_l3188_318802


namespace NUMINAMATH_CALUDE_exists_term_with_nine_l3188_318875

/-- An arithmetic progression with natural number first term and common difference -/
structure ArithmeticProgression :=
  (first_term : ℕ)
  (common_difference : ℕ)

/-- Function to check if a natural number contains the digit 9 -/
def contains_nine (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10^k) % 10 = 9

/-- Theorem stating that there exists a term in the arithmetic progression containing the digit 9 -/
theorem exists_term_with_nine (ap : ArithmeticProgression) :
  ∃ n : ℕ, contains_nine (ap.first_term + n * ap.common_difference) :=
sorry

end NUMINAMATH_CALUDE_exists_term_with_nine_l3188_318875


namespace NUMINAMATH_CALUDE_multiples_of_5_ending_in_0_less_than_200_l3188_318832

def count_multiples_of_5_ending_in_0 (upper_bound : ℕ) : ℕ :=
  (upper_bound - 1) / 10

theorem multiples_of_5_ending_in_0_less_than_200 :
  count_multiples_of_5_ending_in_0 200 = 19 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_5_ending_in_0_less_than_200_l3188_318832


namespace NUMINAMATH_CALUDE_wood_per_table_is_12_l3188_318840

/-- The number of pieces of wood required to make a table -/
def wood_per_table : ℕ := sorry

/-- The total number of pieces of wood available -/
def total_wood : ℕ := 672

/-- The number of pieces of wood required to make a chair -/
def wood_per_chair : ℕ := 8

/-- The number of tables that can be made -/
def num_tables : ℕ := 24

/-- The number of chairs that can be made -/
def num_chairs : ℕ := 48

theorem wood_per_table_is_12 :
  wood_per_table = 12 :=
by sorry

end NUMINAMATH_CALUDE_wood_per_table_is_12_l3188_318840


namespace NUMINAMATH_CALUDE_total_cost_is_21_93_l3188_318823

/-- The amount Alyssa spent on grapes -/
def grapes_cost : ℚ := 12.08

/-- The amount Alyssa spent on cherries -/
def cherries_cost : ℚ := 9.85

/-- The total amount Alyssa spent on fruits -/
def total_cost : ℚ := grapes_cost + cherries_cost

/-- Theorem stating that the total cost is equal to $21.93 -/
theorem total_cost_is_21_93 : total_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_21_93_l3188_318823


namespace NUMINAMATH_CALUDE_max_distance_between_sets_l3188_318843

theorem max_distance_between_sets : ∃ (a b : ℂ),
  (a^4 - 16 = 0) ∧
  (b^4 - 16*b^3 - 16*b + 256 = 0) ∧
  (∀ (x y : ℂ), (x^4 - 16 = 0) → (y^4 - 16*y^3 - 16*y + 256 = 0) →
    Complex.abs (x - y) ≤ Complex.abs (a - b)) ∧
  Complex.abs (a - b) = 2 * Real.sqrt 65 :=
sorry

end NUMINAMATH_CALUDE_max_distance_between_sets_l3188_318843


namespace NUMINAMATH_CALUDE_darrel_took_48_candies_l3188_318883

/-- Represents the number of candies on the table -/
structure CandyCount where
  red : ℕ
  blue : ℕ

/-- Represents the state of candies on the table at different stages -/
structure CandyState where
  initial : CandyCount
  afterDarrel : CandyCount
  afterCloe : CandyCount

/-- Darrel's action of taking candies -/
def darrelAction (x : ℕ) (c : CandyCount) : CandyCount :=
  { red := c.red - x, blue := c.blue - x }

/-- Cloe's action of taking candies -/
def cloeAction (c : CandyCount) : CandyCount :=
  { red := c.red - 12, blue := c.blue - 12 }

/-- The theorem to be proved -/
theorem darrel_took_48_candies (state : CandyState) (x : ℕ) :
  state.initial.red = 3 * state.initial.blue →
  state.afterDarrel = darrelAction x state.initial →
  state.afterDarrel.red = 4 * state.afterDarrel.blue →
  state.afterCloe = cloeAction state.afterDarrel →
  state.afterCloe.red = 5 * state.afterCloe.blue →
  2 * x = 48 := by
  sorry


end NUMINAMATH_CALUDE_darrel_took_48_candies_l3188_318883


namespace NUMINAMATH_CALUDE_first_concert_attendance_proof_l3188_318846

/-- The number of people attending the first concert -/
def first_concert_attendance : ℕ := 65899

/-- The number of people attending the second concert -/
def second_concert_attendance : ℕ := 66018

/-- The difference in attendance between the second and first concerts -/
def attendance_difference : ℕ := 119

theorem first_concert_attendance_proof :
  first_concert_attendance = second_concert_attendance - attendance_difference :=
by sorry

end NUMINAMATH_CALUDE_first_concert_attendance_proof_l3188_318846


namespace NUMINAMATH_CALUDE_inequality_proof_l3188_318865

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6) 
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) : 
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3188_318865


namespace NUMINAMATH_CALUDE_abc_sum_range_l3188_318873

theorem abc_sum_range (a b c : ℝ) (h : a + b + 2*c = 0) :
  (∃ y : ℝ, y < 0 ∧ ab + ac + bc = y) ∧ ab + ac + bc ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_abc_sum_range_l3188_318873


namespace NUMINAMATH_CALUDE_unique_prime_triple_l3188_318854

theorem unique_prime_triple (p : ℕ) : 
  Prime p ∧ Prime (2 * p + 1) ∧ Prime (4 * p + 1) ↔ p = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_triple_l3188_318854


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3188_318848

theorem fraction_equivalence : 
  ∀ (n : ℚ), (2 + n) / (7 + n) = 3 / 8 → n = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3188_318848


namespace NUMINAMATH_CALUDE_jessica_cut_two_roses_l3188_318812

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses initial_orchids final_roses final_orchids : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that Jessica cut 2 roses given the initial and final flower counts -/
theorem jessica_cut_two_roses :
  roses_cut 15 62 17 96 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_two_roses_l3188_318812


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3188_318884

/-- Given a quadratic equation that can be factored into two linear factors, prove the value of m -/
theorem quadratic_factorization (m : ℝ) : 
  (∃ (a b : ℝ), ∀ (x y : ℝ), 
    x^2 + 7*x*y + m*y^2 - 5*x + 43*y - 24 = (x + a*y + 3) * (x + b*y - 8)) → 
  m = -18 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3188_318884


namespace NUMINAMATH_CALUDE_additional_members_needed_club_membership_increase_l3188_318842

/-- The number of additional members needed for a club to reach its desired membership. -/
theorem additional_members_needed (current_members : ℕ) (desired_members : ℕ) : ℕ :=
  desired_members - current_members

/-- Proof that the club needs 15 additional members. -/
theorem club_membership_increase : additional_members_needed 10 25 = 15 := by
  -- The proof goes here
  sorry

#check club_membership_increase

end NUMINAMATH_CALUDE_additional_members_needed_club_membership_increase_l3188_318842


namespace NUMINAMATH_CALUDE_budget_equipment_percentage_l3188_318821

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  research_development : ℝ
  utilities : ℝ
  supplies : ℝ
  salaries : ℝ
  equipment : ℝ

/-- Theorem: Given the budget allocation conditions, the percentage spent on equipment is 4% -/
theorem budget_equipment_percentage
  (budget : BudgetAllocation)
  (h1 : budget.transportation = 15)
  (h2 : budget.research_development = 9)
  (h3 : budget.utilities = 5)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries = (234 / 360) * 100)
  (h6 : budget.transportation + budget.research_development + budget.utilities +
        budget.supplies + budget.salaries + budget.equipment = 100) :
  budget.equipment = 4 := by
  sorry

end NUMINAMATH_CALUDE_budget_equipment_percentage_l3188_318821


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_869_l3188_318867

theorem sqrt_product_plus_one_equals_869 : 
  Real.sqrt ((31 * 30 * 29 * 28) + 1) = 869 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_equals_869_l3188_318867


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3188_318828

def A : Set ℝ := {x | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}

theorem subset_implies_a_values :
  ∀ a : ℝ, B a ⊆ A → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3188_318828


namespace NUMINAMATH_CALUDE_negation_equivalence_l3188_318891

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3188_318891


namespace NUMINAMATH_CALUDE_probability_two_blue_jellybeans_l3188_318894

-- Define the total number of jellybeans and the number of each color
def total_jellybeans : ℕ := 12
def red_jellybeans : ℕ := 3
def blue_jellybeans : ℕ := 4
def white_jellybeans : ℕ := 5

-- Define the number of jellybeans to be picked
def picked_jellybeans : ℕ := 3

-- Define the probability of picking exactly two blue jellybeans
def prob_two_blue : ℚ := 12 / 55

-- Theorem statement
theorem probability_two_blue_jellybeans : 
  prob_two_blue = (Nat.choose blue_jellybeans 2 * Nat.choose (total_jellybeans - blue_jellybeans) 1) / 
                  Nat.choose total_jellybeans picked_jellybeans :=
by sorry

end NUMINAMATH_CALUDE_probability_two_blue_jellybeans_l3188_318894


namespace NUMINAMATH_CALUDE_benzoic_acid_weight_l3188_318816

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound -/
def molecularWeight (formula : MolecularFormula) (weights : AtomicWeights) : ℝ :=
  formula.carbon * weights.carbon +
  formula.hydrogen * weights.hydrogen +
  formula.oxygen * weights.oxygen

/-- Theorem: The molecular weight of 4 moles of Benzoic acid is 488.472 grams -/
theorem benzoic_acid_weight :
  let benzoicAcid : MolecularFormula := { carbon := 7, hydrogen := 6, oxygen := 2 }
  let atomicWeights : AtomicWeights := { carbon := 12.01, hydrogen := 1.008, oxygen := 16.00 }
  (4 : ℝ) * molecularWeight benzoicAcid atomicWeights = 488.472 := by
  sorry


end NUMINAMATH_CALUDE_benzoic_acid_weight_l3188_318816


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3188_318847

theorem basketball_free_throws :
  ∀ (two_points three_points free_throws : ℕ),
    3 * three_points = 2 * two_points →
    free_throws = 2 * two_points - 1 →
    2 * two_points + 3 * three_points + free_throws = 71 →
    free_throws = 23 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3188_318847


namespace NUMINAMATH_CALUDE_number_of_cartons_l3188_318866

/-- Represents the number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- Represents the number of packs in a box -/
def packs_per_box : ℕ := 10

/-- Represents the price of a pack in dollars -/
def price_per_pack : ℕ := 1

/-- Represents the total cost for all cartons in dollars -/
def total_cost : ℕ := 1440

/-- Theorem stating that the number of cartons is 12 -/
theorem number_of_cartons : 
  (total_cost : ℚ) / (boxes_per_carton * packs_per_box * price_per_pack) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cartons_l3188_318866


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l3188_318831

/-- Given a restaurant that made hamburgers and served some, 
    calculate the number of hamburgers left over. -/
theorem hamburgers_left_over 
  (total_made : ℕ) 
  (served : ℕ) 
  (h1 : total_made = 25) 
  (h2 : served = 11) : 
  total_made - served = 14 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l3188_318831


namespace NUMINAMATH_CALUDE_poster_spacing_proof_l3188_318858

/-- Calculates the equal distance between posters and from the ends of the wall -/
def equal_distance (wall_width : ℕ) (num_posters : ℕ) (poster_width : ℕ) : ℕ :=
  (wall_width - num_posters * poster_width) / (num_posters + 1)

/-- Theorem stating that the equal distance is 20 cm given the problem conditions -/
theorem poster_spacing_proof :
  equal_distance 320 6 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_poster_spacing_proof_l3188_318858


namespace NUMINAMATH_CALUDE_lcm_of_1400_and_1050_l3188_318870

theorem lcm_of_1400_and_1050 : Nat.lcm 1400 1050 = 4200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_1400_and_1050_l3188_318870


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3188_318853

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are not collinear -/
def NotCollinear (a b : V) : Prop := ∀ (k : ℝ), k • a ≠ b

theorem vector_equation_solution
  (a b : V) (x : ℝ)
  (h_not_collinear : NotCollinear a b)
  (h_c : ∃ c : V, c = x • a + b)
  (h_d : ∃ d : V, d = a + (2*x - 1) • b)
  (h_collinear : ∃ (k : ℝ) (c d : V), c = x • a + b ∧ d = a + (2*x - 1) • b ∧ d = k • c) :
  x = 1 ∨ x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3188_318853


namespace NUMINAMATH_CALUDE_factorization_equality_l3188_318824

theorem factorization_equality (x : ℝ) : (x + 2) * x - (x + 2) = (x + 2) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3188_318824


namespace NUMINAMATH_CALUDE_reflection_property_l3188_318872

/-- A reflection in R^2 -/
def Reflection (v : ℝ × ℝ) : ℝ × ℝ → ℝ × ℝ := sorry

theorem reflection_property (r : ℝ × ℝ → ℝ × ℝ) :
  r (2, 4) = (10, -2) →
  r (1, 6) = (107/37, -198/37) :=
by sorry

end NUMINAMATH_CALUDE_reflection_property_l3188_318872


namespace NUMINAMATH_CALUDE_opposite_of_three_l3188_318888

theorem opposite_of_three (a : ℝ) : (2 * a + 3) + 3 = 0 → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l3188_318888


namespace NUMINAMATH_CALUDE_percentage_of_chemical_b_in_solution_x_l3188_318826

-- Define the solutions and mixture
def solution_x (a b : ℝ) : Prop := a + b = 1 ∧ a = 0.1
def solution_y : Prop := 0.2 + 0.8 = 1
def mixture (x y : ℝ) : Prop := x + y = 1 ∧ x = 0.8

-- Define the chemical compositions
def chemical_a_in_mixture : ℝ := 0.12
def chemical_b_in_solution_x : ℝ := 0.9

-- State the theorem
theorem percentage_of_chemical_b_in_solution_x 
  (a b x y : ℝ) 
  (hx : solution_x a b) 
  (hy : solution_y)
  (hm : mixture x y)
  (ha : x * a + y * 0.2 = chemical_a_in_mixture) :
  b = chemical_b_in_solution_x :=
sorry

end NUMINAMATH_CALUDE_percentage_of_chemical_b_in_solution_x_l3188_318826


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3188_318887

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 14

-- Theorem statement
theorem conic_is_ellipse :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, conic_equation x y ↔ a*x^2 + b*y^2 + c*x*y + d*x + e*y + f = 0) ∧
    b^2 * c^2 - 4 * a * b * (a * e^2 + b * d^2 - c * d * e + f * (c^2 - 4 * a * b)) > 0 ∧
    a + b ≠ 0 ∧ a * b > 0 :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3188_318887


namespace NUMINAMATH_CALUDE_white_squares_in_row_l3188_318871

/-- Represents a modified stair-step figure where each row begins and ends with a black square
    and has alternating white and black squares. -/
structure ModifiedStairStep where
  /-- The number of squares in the nth row is 2n -/
  squares_in_row : ℕ → ℕ
  /-- Each row begins and ends with a black square -/
  begins_ends_black : ∀ n : ℕ, squares_in_row n ≥ 2
  /-- The number of squares in each row is even -/
  even_squares : ∀ n : ℕ, Even (squares_in_row n)

/-- The number of white squares in the nth row of a modified stair-step figure is equal to n -/
theorem white_squares_in_row (figure : ModifiedStairStep) (n : ℕ) :
  (figure.squares_in_row n) / 2 = n := by
  sorry

end NUMINAMATH_CALUDE_white_squares_in_row_l3188_318871


namespace NUMINAMATH_CALUDE_cyclist_travel_time_is_40_l3188_318811

/-- Represents the tram schedule and cyclist's journey -/
structure TramSchedule where
  /-- Interval between tram departures from Station A (in minutes) -/
  departure_interval : ℕ
  /-- Time for a tram to travel from Station A to Station B (in minutes) -/
  journey_time : ℕ
  /-- Number of trams encountered by the cyclist -/
  trams_encountered : ℕ

/-- Calculates the cyclist's travel time -/
def cyclist_travel_time (schedule : TramSchedule) : ℕ :=
  (schedule.trams_encountered + 2) * schedule.departure_interval

/-- Theorem stating the cyclist's travel time is 40 minutes -/
theorem cyclist_travel_time_is_40 (schedule : TramSchedule)
  (h1 : schedule.departure_interval = 5)
  (h2 : schedule.journey_time = 15)
  (h3 : schedule.trams_encountered = 10) :
  cyclist_travel_time schedule = 40 := by
  sorry

#eval cyclist_travel_time { departure_interval := 5, journey_time := 15, trams_encountered := 10 }

end NUMINAMATH_CALUDE_cyclist_travel_time_is_40_l3188_318811


namespace NUMINAMATH_CALUDE_vector_equation_solution_l3188_318836

-- Define the vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 5)
def c (x : ℝ) : ℝ × ℝ := (3, x)

-- Define the dot product operation
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- State the theorem
theorem vector_equation_solution :
  ∃ x : ℝ, dot_product (vector_sub (scalar_mul 8 a) b) (c x) = 30 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l3188_318836


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3188_318844

/-- Given a hyperbola C and a parabola, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 5 →  -- Focal distance
  (∃ k : ℝ, ∀ x : ℝ, (1/16 * x^2 + 1 - k*x = 0 → 
    (k = b/a ∨ k = -b/a))) →  -- Parabola tangent to asymptotes
  (∀ x y : ℝ, x^2 / 4 - y^2 = 1) :=  -- Conclusion: Specific hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3188_318844


namespace NUMINAMATH_CALUDE_min_sum_squares_l3188_318852

theorem min_sum_squares (x y z : ℝ) (h : x - 2*y - 3*z = 4) :
  ∃ (m : ℝ), m = 8/7 ∧ ∀ (a b c : ℝ), a - 2*b - 3*c = 4 → a^2 + b^2 + c^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3188_318852


namespace NUMINAMATH_CALUDE_ten_row_triangle_count_l3188_318830

/-- Calculates the sum of the first n natural numbers. -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in a triangle with n rows. -/
def rods_count (n : ℕ) : ℕ := 3 * triangular_number n

/-- Calculates the number of connectors in a triangle with n rows of rods. -/
def connectors_count (n : ℕ) : ℕ := triangular_number (n + 1)

/-- The total number of rods and connectors in a triangle with n rows of rods. -/
def total_count (n : ℕ) : ℕ := rods_count n + connectors_count n

theorem ten_row_triangle_count :
  total_count 10 = 231 := by sorry

end NUMINAMATH_CALUDE_ten_row_triangle_count_l3188_318830


namespace NUMINAMATH_CALUDE_twenty_five_percent_problem_l3188_318822

theorem twenty_five_percent_problem : ∃ x : ℝ, (0.75 * 80 = 1.25 * x) ∧ (x = 48) := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_problem_l3188_318822


namespace NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l3188_318895

/-- The amount of cargo loaded in the Bahamas is equal to the difference between the final amount of cargo and the initial amount of cargo. -/
theorem cargo_loaded_in_bahamas (initial_cargo final_cargo : ℕ) 
  (h1 : initial_cargo = 5973)
  (h2 : final_cargo = 14696) :
  final_cargo - initial_cargo = 8723 := by
  sorry

end NUMINAMATH_CALUDE_cargo_loaded_in_bahamas_l3188_318895


namespace NUMINAMATH_CALUDE_min_c_value_l3188_318898

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2 * x + y = 2025 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1013 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l3188_318898


namespace NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l3188_318851

-- Define the function f(x) = |2x-a| + 5x
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + 5*x

-- Part I: Solution set when a = 3
theorem solution_set_part_i :
  ∀ x : ℝ, f 3 x ≥ 5*x + 1 ↔ x ≤ 1 ∨ x ≥ 2 := by sorry

-- Part II: Value of a for given solution set
theorem solution_set_part_ii :
  (∀ x : ℝ, f 3 x ≤ 0 ↔ x ≤ -1) := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_solution_set_part_ii_l3188_318851


namespace NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l3188_318879

-- Define the point type
variable {Point : Type*}

-- Define the distance function
variable (dist : Point → Point → ℝ)

-- Define the volume function for tetrahedrons
variable (volume : Point → Point → Point → Point → ℝ)

-- Theorem statement
theorem tetrahedron_volume_ratio
  (A B C D B' C' D' : Point) :
  volume A B C D / volume A B' C' D' =
  (dist A B * dist A C * dist A D) / (dist A B' * dist A C' * dist A D') :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_ratio_l3188_318879


namespace NUMINAMATH_CALUDE_mrs_hilt_travel_distance_l3188_318800

/-- Calculates the total miles traveled given the initial odometer reading and additional miles --/
def total_miles_traveled (initial_reading : ℝ) (additional_miles : ℝ) : ℝ :=
  initial_reading + additional_miles

/-- Theorem stating that the total miles traveled is 2,210.23 given the specific conditions --/
theorem mrs_hilt_travel_distance :
  total_miles_traveled 1498.76 711.47 = 2210.23 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_travel_distance_l3188_318800


namespace NUMINAMATH_CALUDE_suresh_investment_l3188_318880

/-- Given the total profit, Ramesh's investment, and Ramesh's share of profit, 
    prove that Suresh's investment is Rs. 24,000. -/
theorem suresh_investment 
  (total_profit : ℕ) 
  (ramesh_investment : ℕ) 
  (ramesh_profit : ℕ) 
  (h1 : total_profit = 19000)
  (h2 : ramesh_investment = 40000)
  (h3 : ramesh_profit = 11875) :
  (total_profit - ramesh_profit) * ramesh_investment / ramesh_profit = 24000 := by
  sorry

end NUMINAMATH_CALUDE_suresh_investment_l3188_318880


namespace NUMINAMATH_CALUDE_equation_solution_l3188_318855

theorem equation_solution :
  let f (x : ℚ) := (3 - x) / (x + 2) + (3*x - 6) / (3 - x)
  ∃! x, f x = 2 ∧ x = -7/6 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3188_318855


namespace NUMINAMATH_CALUDE_joes_haircuts_l3188_318893

/-- The number of women's haircuts Joe did -/
def womens_haircuts : ℕ := sorry

/-- The time it takes to cut a woman's hair in minutes -/
def womens_haircut_time : ℕ := 50

/-- The time it takes to cut a man's hair in minutes -/
def mens_haircut_time : ℕ := 15

/-- The time it takes to cut a kid's hair in minutes -/
def kids_haircut_time : ℕ := 25

/-- The number of men's haircuts Joe did -/
def mens_haircuts : ℕ := 2

/-- The number of kids' haircuts Joe did -/
def kids_haircuts : ℕ := 3

/-- The total time Joe spent cutting hair in minutes -/
def total_time : ℕ := 255

theorem joes_haircuts : womens_haircuts = 3 := by sorry

end NUMINAMATH_CALUDE_joes_haircuts_l3188_318893


namespace NUMINAMATH_CALUDE_square_of_cube_of_smallest_prime_l3188_318861

def smallest_prime : ℕ := 2

theorem square_of_cube_of_smallest_prime : 
  (smallest_prime ^ 3) ^ 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_of_cube_of_smallest_prime_l3188_318861


namespace NUMINAMATH_CALUDE_farmer_vegetable_difference_l3188_318829

/-- Calculates the total difference between initial and remaining tomatoes and carrots --/
def total_difference (initial_tomatoes initial_carrots picked_tomatoes picked_carrots given_tomatoes given_carrots : ℕ) : ℕ :=
  (initial_tomatoes - (initial_tomatoes - picked_tomatoes + given_tomatoes)) +
  (initial_carrots - (initial_carrots - picked_carrots + given_carrots))

/-- Theorem stating the total difference for the given problem --/
theorem farmer_vegetable_difference :
  total_difference 17 13 5 6 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_farmer_vegetable_difference_l3188_318829


namespace NUMINAMATH_CALUDE_negation_equivalence_l3188_318885

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, Real.exp x - x - 1 < 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3188_318885


namespace NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l3188_318833

theorem max_value_problem (x : ℝ) (h : x < 1) :
  ∃ y : ℝ, y = (4 * x^2 - 3 * x) / (x - 1) ∧ 
  ∀ z : ℝ, z = (4 * x^2 - 3 * x) / (x - 1) → z ≤ y ∧ y = 1 :=
sorry

theorem min_value_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  ∃ y : ℝ, y = 4 / (a + 1) + 1 / b ∧
  ∀ z : ℝ, z = 4 / (a + 1) + 1 / b → y ≤ z ∧ y = 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_min_value_problem_l3188_318833


namespace NUMINAMATH_CALUDE_cos_sin_sum_zero_implies_double_angle_sum_zero_l3188_318841

theorem cos_sin_sum_zero_implies_double_angle_sum_zero 
  (x y z : ℝ) 
  (h1 : Real.cos x + Real.cos y + Real.cos z = 0)
  (h2 : Real.sin x + Real.sin y + Real.sin z = 0) : 
  Real.cos (2*x) + Real.cos (2*y) + Real.cos (2*z) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_sum_zero_implies_double_angle_sum_zero_l3188_318841


namespace NUMINAMATH_CALUDE_star_arrangement_exists_l3188_318837

-- Define the type for the six-pointed star
def StarArrangement := Fin 12 → Fin 12

-- Define the property that the sum of four numbers on each line equals 26
def ValidArrangement (s : StarArrangement) : Prop :=
  -- Top line
  s 0 + s 1 + s 2 + s 3 = 26 ∧
  -- Left line
  s 4 + s 5 + s 6 + s 7 = 26 ∧
  -- Right line
  s 8 + s 9 + s 10 + s 11 = 26 ∧
  -- All numbers from 1 to 12 are used exactly once
  ∀ n : Fin 12, ∃! i : Fin 12, s i = n.succ

-- Theorem statement
theorem star_arrangement_exists : ∃ s : StarArrangement, ValidArrangement s := by
  sorry

end NUMINAMATH_CALUDE_star_arrangement_exists_l3188_318837


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3188_318839

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 - 4 * x + 3 = 0) ↔ (m ≤ 4/3 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3188_318839


namespace NUMINAMATH_CALUDE_max_consecutive_expressible_l3188_318838

/-- A function that represents the expression x^3 + 2y^2 --/
def f (x y : ℤ) : ℤ := x^3 + 2*y^2

/-- The property of being expressible in the form x^3 + 2y^2 --/
def expressible (n : ℤ) : Prop := ∃ x y : ℤ, f x y = n

/-- A sequence of consecutive integers starting from a given integer --/
def consecutive_seq (start : ℤ) (length : ℕ) : Set ℤ :=
  {n : ℤ | start ≤ n ∧ n < start + length}

/-- The main theorem stating the maximal length of consecutive expressible integers --/
theorem max_consecutive_expressible :
  (∃ start : ℤ, ∀ n ∈ consecutive_seq start 5, expressible n) ∧
  (∀ start : ℤ, ∀ length : ℕ, length > 5 →
    ∃ n ∈ consecutive_seq start length, ¬expressible n) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_expressible_l3188_318838


namespace NUMINAMATH_CALUDE_store_profit_calculation_l3188_318868

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters. -/
theorem store_profit_calculation (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.07
  
  let SP1 := C * (1 + initial_markup)
  let SP2 := SP1 * (1 + new_year_markup)
  let SPF := SP2 * (1 - february_discount)
  let profit := SPF - C
  
  profit / C = 0.395 := by sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l3188_318868


namespace NUMINAMATH_CALUDE_inequality_implication_l3188_318849

theorem inequality_implication (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 2 * (3^(2*x)) - 3^x + a^2 - a - 3 > 0) → 
  a < -1 ∨ a > 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_implication_l3188_318849


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_81_plus_sqrt_81_over_2_l3188_318892

theorem sqrt_of_sqrt_81_plus_sqrt_81_over_2 : 
  Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_81_plus_sqrt_81_over_2_l3188_318892


namespace NUMINAMATH_CALUDE_sequence_a_property_l3188_318876

def sequence_a (n : ℕ) : ℚ := 2 * n^2 - n

theorem sequence_a_property :
  (sequence_a 1 = 1) ∧
  (∀ n m : ℕ, n ≠ 0 → m ≠ 0 → sequence_a m / m - sequence_a n / n = 2 * (m - n)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_a_property_l3188_318876


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l3188_318803

/-- The number of ways to place 10 numbered balls into 10 numbered boxes 
    such that exactly 3 balls do not match the numbers of their boxes -/
def ballPlacementWays : ℕ := 240

/-- Theorem stating that the number of ways to place 10 numbered balls into 10 numbered boxes 
    such that exactly 3 balls do not match the numbers of their boxes is 240 -/
theorem ball_placement_theorem : ballPlacementWays = 240 := by
  sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l3188_318803


namespace NUMINAMATH_CALUDE_smallest_n_value_smallest_n_is_99000_l3188_318877

/-- The number of ordered quadruplets satisfying the conditions -/
def num_quadruplets : ℕ := 91000

/-- The given GCD value for all quadruplets -/
def given_gcd : ℕ := 55

/-- 
Proposition: The smallest positive integer n satisfying the following conditions is 99000:
1. There exist exactly 91000 ordered quadruplets of positive integers (a, b, c, d)
2. For each quadruplet, gcd(a, b, c, d) = 55
3. For each quadruplet, lcm(a, b, c, d) = n
-/
theorem smallest_n_value (n : ℕ) : 
  (∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    S.card = num_quadruplets ∧ 
    ∀ (a b c d : ℕ), (a, b, c, d) ∈ S → 
      Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = given_gcd ∧
      Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = n) →
  n ≥ 99000 :=
by sorry

/-- The smallest value of n satisfying the conditions is indeed 99000 -/
theorem smallest_n_is_99000 : 
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    S.card = num_quadruplets ∧ 
    ∀ (a b c d : ℕ), (a, b, c, d) ∈ S → 
      Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = given_gcd ∧
      Nat.lcm (Nat.lcm (Nat.lcm a b) c) d = 99000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_smallest_n_is_99000_l3188_318877


namespace NUMINAMATH_CALUDE_min_value_expression_l3188_318860

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : a > 0) (h4 : b ≠ 0) :
  ((a + b)^3 + (b - c)^2 + (c - a)^3) / b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3188_318860


namespace NUMINAMATH_CALUDE_remainder_theorem_l3188_318818

/-- The polynomial P(z) = 4z^4 - 9z^3 + 3z^2 - 17z + 7 -/
def P (z : ℂ) : ℂ := 4 * z^4 - 9 * z^3 + 3 * z^2 - 17 * z + 7

/-- The theorem stating that the remainder of P(z) divided by (z - 2) is -23 -/
theorem remainder_theorem :
  ∃ Q : ℂ → ℂ, P = (fun z ↦ (z - 2) * Q z + (-23)) := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3188_318818


namespace NUMINAMATH_CALUDE_exists_shape_with_five_faces_l3188_318869

/-- A geometric shape. -/
structure Shape where
  faces : ℕ

/-- A square pyramid is a shape with 5 faces. -/
def SquarePyramid : Shape :=
  { faces := 5 }

/-- There exists a shape with exactly 5 faces. -/
theorem exists_shape_with_five_faces : ∃ (s : Shape), s.faces = 5 := by
  sorry

end NUMINAMATH_CALUDE_exists_shape_with_five_faces_l3188_318869


namespace NUMINAMATH_CALUDE_smallest_number_in_ratio_l3188_318862

theorem smallest_number_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a * 5 = b * 3 →
  a * 7 = c * 3 →
  c = 56 →
  c - a = 32 →
  a = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_number_in_ratio_l3188_318862


namespace NUMINAMATH_CALUDE_min_value_expression_l3188_318863

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b - a - 2 * b = 0) :
  (a^2 / 4 - 2 / a + b^2 - 1 / b) ≥ 7 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3188_318863


namespace NUMINAMATH_CALUDE_daisy_exchange_impossible_l3188_318827

/-- Represents the number of girls in the row -/
def n : ℕ := 33

/-- Represents the number of places each girl passes her daisy -/
def pass_distance : ℕ := 2

/-- Predicate that checks if a girl at position i receives a daisy -/
def receives_daisy (i : ℕ) : Prop :=
  ∃ j : ℕ, j ≤ n ∧ (i = j + pass_distance ∨ i = j - pass_distance)

/-- Theorem stating it's impossible for every girl to end up with exactly one daisy -/
theorem daisy_exchange_impossible : ¬(∀ i : ℕ, i ≤ n → ∃! j : ℕ, receives_daisy j ∧ i = j) :=
sorry

end NUMINAMATH_CALUDE_daisy_exchange_impossible_l3188_318827


namespace NUMINAMATH_CALUDE_wine_barrels_l3188_318864

theorem wine_barrels (a b : ℝ) : 
  (a + 8 = b) ∧ (b + 3 = 3 * (a - 3)) → a = 10 ∧ b = 18 := by
  sorry

end NUMINAMATH_CALUDE_wine_barrels_l3188_318864


namespace NUMINAMATH_CALUDE_scale_division_l3188_318809

/-- Represents the length of a scale in inches -/
def scale_length : ℕ := 90

/-- Represents the length of each part in inches -/
def part_length : ℕ := 18

/-- Theorem stating that the scale divided into equal parts results in 5 parts -/
theorem scale_division :
  scale_length / part_length = 5 := by sorry

end NUMINAMATH_CALUDE_scale_division_l3188_318809


namespace NUMINAMATH_CALUDE_min_correct_answers_l3188_318845

theorem min_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (min_score : ℕ) : 
  total_questions = 20 →
  correct_points = 10 →
  incorrect_points = 5 →
  min_score = 120 →
  ∃ (x : ℕ), x = 15 ∧ 
    (∀ (y : ℕ), y < x → correct_points * y - incorrect_points * (total_questions - y) ≤ min_score) ∧
    correct_points * x - incorrect_points * (total_questions - x) > min_score :=
by sorry

end NUMINAMATH_CALUDE_min_correct_answers_l3188_318845


namespace NUMINAMATH_CALUDE_car_catchup_l3188_318825

/-- The time (in hours) it takes for the second car to catch up with the first car -/
def catchup_time : ℝ :=
  1.5

/-- The speed of the first car in km/h -/
def speed_first : ℝ :=
  60

/-- The speed of the second car in km/h -/
def speed_second : ℝ :=
  80

/-- The head start of the first car in hours -/
def head_start : ℝ :=
  0.5

theorem car_catchup :
  speed_second * catchup_time = speed_first * (catchup_time + head_start) :=
sorry

end NUMINAMATH_CALUDE_car_catchup_l3188_318825


namespace NUMINAMATH_CALUDE_chocolates_per_box_l3188_318801

-- Define the problem parameters
def total_boxes : ℕ := 20
def total_chocolates : ℕ := 500

-- Theorem statement
theorem chocolates_per_box :
  total_chocolates / total_boxes = 25 :=
by
  sorry -- Proof omitted

end NUMINAMATH_CALUDE_chocolates_per_box_l3188_318801


namespace NUMINAMATH_CALUDE_two_rolls_probability_l3188_318813

/-- A fair six-sided die --/
def FairDie := Fin 6

/-- The probability of rolling a specific number on a fair die --/
def prob_single_roll : ℚ := 1 / 6

/-- The sum of two die rolls --/
def sum_of_rolls (a b : FairDie) : ℕ := a.val + b.val + 2

/-- Whether a number is prime --/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The probability that the sum of two rolls is prime --/
def prob_sum_is_prime : ℚ := 15 / 36

theorem two_rolls_probability (rolls : ℕ) : 
  (rolls = 2 ∧ prob_sum_is_prime = 0.41666666666666663) → rolls = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_rolls_probability_l3188_318813


namespace NUMINAMATH_CALUDE_diamond_three_five_l3188_318814

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 2 * y + x * y

-- Theorem statement
theorem diamond_three_five : diamond 3 5 = 37 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_five_l3188_318814


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l3188_318819

/-- Given a positive real number x equal to 3.3333333333333335, prove that the equation
    x * 10 / y = x^2 is satisfied when y = 3. -/
theorem square_root_equation_solution (x : ℝ) (hx : x = 3.3333333333333335) :
  ∃ y : ℝ, y = 3 ∧ x * 10 / y = x^2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l3188_318819


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3188_318881

theorem arithmetic_calculations : 
  (1 - 2 + 3 + (-4) = -2) ∧ 
  ((-6) / 3 - (-10) - |(-8)| = 0) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3188_318881


namespace NUMINAMATH_CALUDE_max_train_collection_l3188_318817

/-- The number of trains Max receives each year -/
def trains_per_year : ℕ := 3

/-- The number of years Max collects trains -/
def collection_years : ℕ := 5

/-- The factor by which Max's parents increase his collection -/
def parents_gift_factor : ℕ := 2

/-- The total number of trains Max has after the collection period and his parents' gift -/
def total_trains : ℕ := trains_per_year * collection_years * parents_gift_factor

theorem max_train_collection :
  total_trains = 30 := by sorry

end NUMINAMATH_CALUDE_max_train_collection_l3188_318817


namespace NUMINAMATH_CALUDE_louisa_travel_problem_l3188_318835

/-- Louisa's vacation travel problem -/
theorem louisa_travel_problem (first_day_distance : ℝ) (speed : ℝ) (time_difference : ℝ) 
  (h1 : first_day_distance = 100)
  (h2 : speed = 25)
  (h3 : time_difference = 3)
  : ∃ (second_day_distance : ℝ), second_day_distance = 175 := by
  sorry

end NUMINAMATH_CALUDE_louisa_travel_problem_l3188_318835


namespace NUMINAMATH_CALUDE_shirt_ironing_time_l3188_318899

/-- The number of days per week Hayden irons his clothes -/
def days_per_week : ℕ := 5

/-- The number of minutes Hayden spends ironing his pants each day -/
def pants_ironing_time : ℕ := 3

/-- The total number of minutes Hayden spends ironing over 4 weeks -/
def total_ironing_time : ℕ := 160

/-- The number of weeks in the period -/
def num_weeks : ℕ := 4

theorem shirt_ironing_time :
  ∃ (shirt_time : ℕ),
    shirt_time * (days_per_week * num_weeks) = 
      total_ironing_time - (pants_ironing_time * days_per_week * num_weeks) ∧
    shirt_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_shirt_ironing_time_l3188_318899


namespace NUMINAMATH_CALUDE_largest_expression_l3188_318878

theorem largest_expression : 
  let a := 3 + 2 + 1 + 9
  let b := 3 * 2 + 1 + 9
  let c := 3 + 2 * 1 + 9
  let d := 3 + 2 + 1 / 9
  let e := 3 * 2 / 1 + 9
  b ≥ a ∧ b > c ∧ b > d ∧ b ≥ e := by
sorry

end NUMINAMATH_CALUDE_largest_expression_l3188_318878


namespace NUMINAMATH_CALUDE_deployment_plans_count_l3188_318889

def number_of_volunteers : ℕ := 6
def number_of_positions : ℕ := 4
def number_of_restricted_volunteers : ℕ := 2

theorem deployment_plans_count :
  (number_of_volunteers.choose number_of_positions * number_of_positions.factorial) -
  (number_of_restricted_volunteers * ((number_of_volunteers - 1).choose (number_of_positions - 1) * (number_of_positions - 1).factorial)) = 240 :=
sorry

end NUMINAMATH_CALUDE_deployment_plans_count_l3188_318889


namespace NUMINAMATH_CALUDE_divisible_by_thirteen_l3188_318857

theorem divisible_by_thirteen (a b : ℕ) (h : a * 13 = 119268916) :
  119268903 % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_thirteen_l3188_318857


namespace NUMINAMATH_CALUDE_number_of_best_friends_l3188_318806

theorem number_of_best_friends (total_cards : ℕ) (cards_per_friend : ℕ) 
  (h1 : total_cards = 455) 
  (h2 : cards_per_friend = 91) : 
  total_cards / cards_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_best_friends_l3188_318806


namespace NUMINAMATH_CALUDE_kaleb_shirts_l3188_318874

-- Define the initial number of shirts
def initial_shirts : ℕ := 17

-- Define the number of shirts Kaleb would have after getting rid of 7
def remaining_shirts : ℕ := 10

-- Define the number of shirts Kaleb got rid of
def removed_shirts : ℕ := 7

-- Theorem to prove
theorem kaleb_shirts : initial_shirts = remaining_shirts + removed_shirts :=
by sorry

end NUMINAMATH_CALUDE_kaleb_shirts_l3188_318874


namespace NUMINAMATH_CALUDE_correct_operation_l3188_318896

theorem correct_operation (a : ℝ) : 2 * a^3 - a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3188_318896


namespace NUMINAMATH_CALUDE_plan_y_more_economical_min_megabytes_optimal_l3188_318820

/-- Represents the cost of an internet plan in cents -/
def PlanCost (initial_fee : ℕ) (rate : ℕ) (megabytes : ℕ) : ℕ :=
  initial_fee * 100 + rate * megabytes

/-- The minimum number of megabytes for Plan Y to be more economical than Plan X -/
def MinMegabytes : ℕ := 501

theorem plan_y_more_economical :
  ∀ m : ℕ, m ≥ MinMegabytes →
    PlanCost 25 10 m < PlanCost 0 15 m :=
by
  sorry

theorem min_megabytes_optimal :
  ∀ m : ℕ, m < MinMegabytes →
    PlanCost 0 15 m ≤ PlanCost 25 10 m :=
by
  sorry

end NUMINAMATH_CALUDE_plan_y_more_economical_min_megabytes_optimal_l3188_318820


namespace NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_l3188_318834

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_line_implies_parallel_planes 
  (α β : Plane) (l : Line) : 
  (perpendicular l α ∧ perpendicular l β) → parallel α β := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_implies_parallel_planes_l3188_318834


namespace NUMINAMATH_CALUDE_x_squared_plus_inverse_l3188_318804

theorem x_squared_plus_inverse (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_inverse_l3188_318804


namespace NUMINAMATH_CALUDE_total_amount_paid_l3188_318808

theorem total_amount_paid (total_work : ℚ) (ac_portion : ℚ) (b_payment : ℚ) : 
  total_work = 1 ∧ 
  ac_portion = 19/23 ∧ 
  b_payment = 12 →
  (1 - ac_portion) * (total_work * b_payment) / (1 - ac_portion) = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l3188_318808


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3188_318850

theorem tangent_line_to_circle (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 3 → x^2 + y^2 = 1 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      x'^2 + y'^2 = 1 → (x' - x)^2 + (y' - y)^2 < δ^2 → 
        (y' - (k * x' + 3))^2 > ε^2 * ((x' - x)^2 + (y' - y)^2)) →
  k = 2 * Real.sqrt 2 ∨ k = -2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3188_318850


namespace NUMINAMATH_CALUDE_banana_distribution_exists_l3188_318886

-- Define the number of bananas and boxes
def total_bananas : ℕ := 40
def num_boxes : ℕ := 8

-- Define a valid distribution
def is_valid_distribution (dist : List ℕ) : Prop :=
  dist.length = num_boxes ∧
  dist.sum = total_bananas ∧
  dist.Nodup

-- Theorem statement
theorem banana_distribution_exists : 
  ∃ (dist : List ℕ), is_valid_distribution dist :=
sorry

end NUMINAMATH_CALUDE_banana_distribution_exists_l3188_318886


namespace NUMINAMATH_CALUDE_prism_has_315_edges_l3188_318859

/-- A prism is a polyhedron with two congruent and parallel faces (bases) connected by rectangular faces. -/
structure Prism where
  num_edges : ℕ

/-- The number of edges in a prism is always a multiple of 3. -/
axiom prism_edges_multiple_of_three (p : Prism) : ∃ k : ℕ, p.num_edges = 3 * k

/-- The prism has more than 310 edges. -/
axiom edges_greater_than_310 (p : Prism) : p.num_edges > 310

/-- The prism has fewer than 320 edges. -/
axiom edges_less_than_320 (p : Prism) : p.num_edges < 320

/-- The number of edges in the prism is odd. -/
axiom edges_odd (p : Prism) : Odd p.num_edges

theorem prism_has_315_edges (p : Prism) : p.num_edges = 315 := by
  sorry

end NUMINAMATH_CALUDE_prism_has_315_edges_l3188_318859


namespace NUMINAMATH_CALUDE_max_value_on_circle_l3188_318807

theorem max_value_on_circle (x y : ℝ) : 
  Complex.abs (x - 2 + y * Complex.I) = 1 →
  (∃ (x' y' : ℝ), Complex.abs (x' - 2 + y' * Complex.I) = 1 ∧ 
    |3 * x' - y'| ≥ |3 * x - y|) →
  |3 * x - y| ≤ 6 + Real.sqrt 10 :=
sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l3188_318807


namespace NUMINAMATH_CALUDE_power_relation_l3188_318856

theorem power_relation (a : ℝ) (m n : ℤ) (h1 : a^m = 3) (h2 : a^n = 2) :
  a^(m - 2*n) = 3/4 := by
sorry

end NUMINAMATH_CALUDE_power_relation_l3188_318856


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3188_318882

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem stating that for an arithmetic sequence with S₅ = 5 and S₉ = 27, S₇ = 14 -/
theorem arithmetic_sequence_sum (a₁ d : ℚ) 
  (h₁ : S 5 a₁ d = 5)
  (h₂ : S 9 a₁ d = 27) : 
  S 7 a₁ d = 14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3188_318882


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l3188_318815

theorem complex_sum_theorem (x y u v w z : ℂ) : 
  v = 2 → 
  w = -x - u → 
  (x + y * Complex.I) + (u + v * Complex.I) + (w + z * Complex.I) = 2 * Complex.I → 
  z + y = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l3188_318815
