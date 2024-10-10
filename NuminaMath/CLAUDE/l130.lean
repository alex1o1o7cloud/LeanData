import Mathlib

namespace computation_problem_value_l130_13084

theorem computation_problem_value (total_problems : Nat) (word_problem_value : Nat) 
  (total_points : Nat) (computation_problems : Nat) :
  total_problems = 30 →
  word_problem_value = 5 →
  total_points = 110 →
  computation_problems = 20 →
  ∃ (computation_value : Nat),
    computation_value = 3 ∧
    total_points = computation_problems * computation_value + 
      (total_problems - computation_problems) * word_problem_value :=
by sorry

end computation_problem_value_l130_13084


namespace rectangle_vertical_length_l130_13011

/-- Given a rectangle with perimeter 50 cm and horizontal length 13 cm, prove its vertical length is 12 cm -/
theorem rectangle_vertical_length (perimeter : ℝ) (horizontal_length : ℝ) (vertical_length : ℝ) : 
  perimeter = 50 ∧ horizontal_length = 13 → 
  perimeter = 2 * (horizontal_length + vertical_length) →
  vertical_length = 12 := by
sorry

end rectangle_vertical_length_l130_13011


namespace four_engine_safer_than_two_engine_l130_13096

-- Define the success rate of an engine
variable (P : ℝ) 

-- Define the probability of successful flight for a 2-engine airplane
def prob_success_2engine (P : ℝ) : ℝ := P^2 + 2*P*(1-P)

-- Define the probability of successful flight for a 4-engine airplane
def prob_success_4engine (P : ℝ) : ℝ := P^4 + 4*P^3*(1-P) + 6*P^2*(1-P)^2

-- Theorem statement
theorem four_engine_safer_than_two_engine :
  ∀ P, 2/3 < P ∧ P < 1 → prob_success_4engine P > prob_success_2engine P :=
sorry

end four_engine_safer_than_two_engine_l130_13096


namespace system_solution_l130_13000

/-- Given a system of equations:
    1) x = 1.12 * y + 52.8
    2) x = y + 50
    Prove that the solution is approximately x ≈ 26.67 and y ≈ -23.33 -/
theorem system_solution :
  ∃ (x y : ℝ),
    (x = 1.12 * y + 52.8) ∧
    (x = y + 50) ∧
    (abs (x - 26.67) < 0.01) ∧
    (abs (y + 23.33) < 0.01) := by
  sorry

end system_solution_l130_13000


namespace inverse_variation_problem_l130_13066

theorem inverse_variation_problem (x w : ℝ) (k : ℝ) :
  (∀ x w, x^4 * w^(1/4) = k) →
  (3^4 * 16^(1/4) = k) →
  (6^4 * w^(1/4) = k) →
  w = 1 / 4096 :=
by sorry

end inverse_variation_problem_l130_13066


namespace sqrt_equation_solution_l130_13055

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt 32 + Real.sqrt x = Real.sqrt 50 → x = 2 := by
  sorry

end sqrt_equation_solution_l130_13055


namespace city_rental_rate_proof_l130_13006

/-- The cost per mile for City Rentals -/
def city_rental_rate : ℝ := 0.31

/-- The base cost for City Rentals -/
def city_base_cost : ℝ := 38.95

/-- The base cost for Safety Rent A Truck -/
def safety_base_cost : ℝ := 41.95

/-- The cost per mile for Safety Rent A Truck -/
def safety_rental_rate : ℝ := 0.29

/-- The number of miles at which the costs are equal -/
def equal_miles : ℝ := 150.0

theorem city_rental_rate_proof :
  city_base_cost + equal_miles * city_rental_rate =
  safety_base_cost + equal_miles * safety_rental_rate :=
by sorry

end city_rental_rate_proof_l130_13006


namespace perpendicular_lines_a_value_l130_13070

theorem perpendicular_lines_a_value (a : ℝ) :
  (∃ (x y : ℝ), y = a * x - 2) ∧ 
  (∃ (x y : ℝ), y = (a + 2) * x + 1) ∧
  (a * (a + 2) = -1) →
  a = -1 := by
  sorry

end perpendicular_lines_a_value_l130_13070


namespace blue_marble_percent_is_35_l130_13073

/-- Represents the composition of items in an urn -/
structure UrnComposition where
  button_percent : ℝ
  red_marble_percent : ℝ
  blue_marble_percent : ℝ

/-- The percentage of blue marbles in the urn -/
def blue_marble_percentage (urn : UrnComposition) : ℝ :=
  urn.blue_marble_percent

/-- Theorem stating the percentage of blue marbles in the urn -/
theorem blue_marble_percent_is_35 (urn : UrnComposition) 
  (h1 : urn.button_percent = 0.3)
  (h2 : urn.red_marble_percent = 0.5 * (1 - urn.button_percent)) :
  blue_marble_percentage urn = 0.35 := by
  sorry

#check blue_marble_percent_is_35

end blue_marble_percent_is_35_l130_13073


namespace fifteen_more_than_two_thirds_of_120_l130_13075

theorem fifteen_more_than_two_thirds_of_120 : (2 / 3 : ℚ) * 120 + 15 = 95 := by
  sorry

end fifteen_more_than_two_thirds_of_120_l130_13075


namespace sum_abcd_equals_21_l130_13074

theorem sum_abcd_equals_21 
  (a b c d : ℝ) 
  (h1 : a * c + a * d + b * c + b * d = 68) 
  (h2 : c + d = 4) : 
  a + b + c + d = 21 := by
sorry

end sum_abcd_equals_21_l130_13074


namespace video_recorder_wholesale_cost_l130_13035

theorem video_recorder_wholesale_cost :
  ∀ (wholesale_cost retail_price employee_price : ℝ),
    retail_price = 1.2 * wholesale_cost →
    employee_price = 0.7 * retail_price →
    employee_price = 168 →
    wholesale_cost = 200 := by
  sorry

end video_recorder_wholesale_cost_l130_13035


namespace carbon_atoms_in_compound_l130_13099

-- Define atomic weights
def atomic_weight_C : ℝ := 12
def atomic_weight_H : ℝ := 1
def atomic_weight_O : ℝ := 16

-- Define the compound properties
def hydrogen_atoms : ℕ := 6
def oxygen_atoms : ℕ := 2
def molecular_weight : ℝ := 122

-- Theorem to prove
theorem carbon_atoms_in_compound :
  ∃ (carbon_atoms : ℕ),
    (carbon_atoms : ℝ) * atomic_weight_C +
    (hydrogen_atoms : ℝ) * atomic_weight_H +
    (oxygen_atoms : ℝ) * atomic_weight_O =
    molecular_weight ∧
    carbon_atoms = 7 := by
  sorry

end carbon_atoms_in_compound_l130_13099


namespace sufficient_necessary_condition_l130_13081

-- Define the interval (1, 4]
def OpenClosedInterval := { x : ℝ | 1 < x ∧ x ≤ 4 }

-- Define the inequality function
def InequalityFunction (m : ℝ) (x : ℝ) := x^2 - m*x + m > 0

-- State the theorem
theorem sufficient_necessary_condition :
  ∀ m : ℝ, (∀ x ∈ OpenClosedInterval, InequalityFunction m x) ↔ m < 4 := by
  sorry

end sufficient_necessary_condition_l130_13081


namespace hyperbola_asymptote_slope_l130_13088

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + (y + 2)^2) - Real.sqrt ((x - 6)^2 + (y + 2)^2) = 4

-- Define the distance between foci
def distance_between_foci : ℝ := 5

-- Define the semi-major axis
def semi_major_axis : ℝ := 2

-- Define the positive slope of an asymptote
def positive_asymptote_slope : ℝ := 0.75

-- Theorem statement
theorem hyperbola_asymptote_slope :
  positive_asymptote_slope = (Real.sqrt (((distance_between_foci / 2)^2) - semi_major_axis^2)) / semi_major_axis :=
sorry

end hyperbola_asymptote_slope_l130_13088


namespace circle_tangent_sum_of_radii_l130_13050

theorem circle_tangent_sum_of_radii :
  ∀ r : ℝ,
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  ∃ r' : ℝ,
  (r' > 0) ∧
  ((r' - 4)^2 + r'^2 = (r' + 2)^2) ∧
  (r + r' = 12) :=
by sorry

end circle_tangent_sum_of_radii_l130_13050


namespace arithmetic_mean_of_fractions_l130_13060

theorem arithmetic_mean_of_fractions :
  let a := 8 / 11
  let b := 9 / 11
  let c := 7 / 11
  a = (b + c) / 2 := by
sorry

end arithmetic_mean_of_fractions_l130_13060


namespace max_value_rational_function_l130_13041

theorem max_value_rational_function : 
  ∃ (M : ℤ), M = 57 ∧ 
  (∀ (x : ℝ), (3 * x^2 + 9 * x + 21) / (3 * x^2 + 9 * x + 7) ≤ M) ∧
  (∃ (x : ℝ), (3 * x^2 + 9 * x + 21) / (3 * x^2 + 9 * x + 7) > M - 1) :=
by
  sorry

end max_value_rational_function_l130_13041


namespace max_profit_at_84_l130_13054

/-- Defective rate as a function of daily output -/
def defective_rate (x : ℕ) : ℚ :=
  if x ≤ 94 then 1 / (96 - x) else 2/3

/-- Daily profit as a function of daily output and profit per qualified instrument -/
def daily_profit (x : ℕ) (A : ℚ) : ℚ :=
  if x ≤ 94 then
    (x * (1 - defective_rate x) * A) - (x * defective_rate x * (A/2))
  else 0

theorem max_profit_at_84 (A : ℚ) (h : A > 0) :
  ∀ x : ℕ, x ≠ 0 → daily_profit 84 A ≥ daily_profit x A :=
sorry

end max_profit_at_84_l130_13054


namespace greg_age_l130_13086

/-- Given the ages of five people with certain relationships, prove Greg's age --/
theorem greg_age (C D E F G : ℕ) : 
  D = E - 5 →
  E = 2 * C →
  F = C - 1 →
  G = 2 * F →
  D = 15 →
  G = 18 := by
  sorry

#check greg_age

end greg_age_l130_13086


namespace fish_length_difference_l130_13013

theorem fish_length_difference :
  let fish1_length : ℝ := 0.3
  let fish2_length : ℝ := 0.2
  fish1_length - fish2_length = 0.1 := by
  sorry

end fish_length_difference_l130_13013


namespace inequalities_always_true_l130_13001

theorem inequalities_always_true 
  (x y a b : ℝ) 
  (hx : x > 0) (hy : y > 0) (ha : a > 0) (hb : b > 0)
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ 
  (x - y ≤ a - b) ∧ 
  (x * y ≤ a * b) ∧ 
  (x / y ≤ a / b) := by
sorry

end inequalities_always_true_l130_13001


namespace rectangular_solid_length_l130_13017

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (solid : RectangularSolid) : ℝ :=
  2 * (solid.length * solid.width + solid.length * solid.depth + solid.width * solid.depth)

theorem rectangular_solid_length 
  (solid : RectangularSolid) 
  (h1 : solid.width = 5)
  (h2 : solid.depth = 2)
  (h3 : surfaceArea solid = 104) : 
  solid.length = 6 := by
sorry

end rectangular_solid_length_l130_13017


namespace repeating_base_k_representation_l130_13097

theorem repeating_base_k_representation (k : ℕ) (h1 : k > 0) : 
  (4 * k + 5) / (k^2 - 1) = 11 / 143 → k = 52 :=
by sorry

end repeating_base_k_representation_l130_13097


namespace negation_of_proposition_l130_13049

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*|x| ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*|x| < 0) :=
by sorry

end negation_of_proposition_l130_13049


namespace sum_of_squares_219_l130_13012

theorem sum_of_squares_219 (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a^2 + b^2 + c^2 = 219 →
  (a : ℕ) + b + c = 21 := by
sorry

end sum_of_squares_219_l130_13012


namespace pairwise_sum_difference_l130_13037

theorem pairwise_sum_difference (n : ℕ) (x : Fin n → ℝ) 
  (h_n : n ≥ 4) 
  (h_pos : ∀ i, x i > 0) : 
  ∃ (i j k l : Fin n), i ≠ j ∧ k ≠ l ∧ 
    (x i + x j) ≤ (x k + x l) * (2 : ℝ)^(1 / (n - 2 : ℝ)) := by
  sorry

end pairwise_sum_difference_l130_13037


namespace inequality_proof_l130_13051

theorem inequality_proof (a b c : ℝ) (ha : a = 31/32) (hb : b = Real.cos (1/4))
  (hc : c = 4 * Real.sin (1/4)) : c > b ∧ b > a := by
  sorry

end inequality_proof_l130_13051


namespace grid_transformation_impossible_l130_13071

/-- Represents a 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℤ

/-- The initial grid configuration -/
def initial_grid : Grid :=
  fun i j => 
    match i, j with
    | 0, 0 => 1 | 0, 1 => 2 | 0, 2 => 3
    | 1, 0 => 4 | 1, 1 => 5 | 1, 2 => 6
    | 2, 0 => 7 | 2, 1 => 8 | 2, 2 => 9

/-- The target grid configuration -/
def target_grid : Grid :=
  fun i j => 
    match i, j with
    | 0, 0 => 7 | 0, 1 => 9 | 0, 2 => 2
    | 1, 0 => 3 | 1, 1 => 5 | 1, 2 => 6
    | 2, 0 => 1 | 2, 1 => 4 | 2, 2 => 8

/-- Calculates the invariant of a grid -/
def grid_invariant (g : Grid) : ℤ :=
  (g 0 0 + g 0 2 + g 1 1 + g 2 0 + g 2 2) - (g 0 1 + g 1 0 + g 1 2 + g 2 1)

/-- Theorem stating the impossibility of transforming the initial grid to the target grid -/
theorem grid_transformation_impossible : 
  ¬∃ (f : Grid → Grid), (f initial_grid = target_grid ∧ 
    ∀ g : Grid, grid_invariant g = grid_invariant (f g)) :=
by
  sorry


end grid_transformation_impossible_l130_13071


namespace function_value_at_shifted_point_l130_13008

/-- Given a function f(x) = a * tan³(x) + b * sin(x) + 1 where f(4) = 5, prove that f(2π - 4) = -3 -/
theorem function_value_at_shifted_point 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.tan x ^ 3 + b * Real.sin x + 1) 
  (h2 : f 4 = 5) : 
  f (2 * Real.pi - 4) = -3 := by
  sorry

end function_value_at_shifted_point_l130_13008


namespace turner_syndrome_classification_l130_13058

-- Define the types of mutations
inductive MutationType
  | GeneMutation
  | ChromosomalNumberVariation
  | GeneRecombination
  | ChromosomalStructureVariation

-- Define a structure for chromosomes
structure Chromosome where
  isSexChromosome : Bool

-- Define a human genetic condition
structure GeneticCondition where
  name : String
  missingChromosome : Option Chromosome
  mutationType : MutationType

-- Define Turner syndrome
def TurnerSyndrome : GeneticCondition where
  name := "Turner syndrome"
  missingChromosome := some { isSexChromosome := true }
  mutationType := MutationType.ChromosomalNumberVariation

-- Theorem statement
theorem turner_syndrome_classification :
  TurnerSyndrome.mutationType = MutationType.ChromosomalNumberVariation :=
by
  sorry


end turner_syndrome_classification_l130_13058


namespace destination_distance_l130_13036

theorem destination_distance (d : ℝ) : 
  (¬ (d ≥ 8)) →  -- Alice's statement is false
  (¬ (d ≤ 7)) →  -- Bob's statement is false
  (d ≠ 6) →      -- Charlie's statement is false
  7 < d ∧ d < 8 := by
sorry

end destination_distance_l130_13036


namespace root_not_sufficient_for_bisection_l130_13010

-- Define a continuous function on a closed interval
def ContinuousOnInterval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  Continuous f ∧ a ≤ b

-- Define the condition for a function to have a root
def HasRoot (f : ℝ → ℝ) : Prop :=
  ∃ x, f x = 0

-- Define the conditions for the bisection method to be applicable
def BisectionApplicable (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ContinuousOnInterval f a b ∧ f a * f b < 0

-- Theorem statement
theorem root_not_sufficient_for_bisection :
  ∃ f : ℝ → ℝ, HasRoot f ∧ ¬(∃ a b, BisectionApplicable f a b) :=
sorry

end root_not_sufficient_for_bisection_l130_13010


namespace smallest_even_triangle_perimeter_l130_13019

/-- A triangle with consecutive even integer side lengths -/
structure EvenTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : b = a + 2
  h2 : c = b + 2
  h3 : Even a
  h4 : a + b > c
  h5 : a + c > b
  h6 : b + c > a

/-- The perimeter of an EvenTriangle -/
def perimeter (t : EvenTriangle) : ℕ := t.a + t.b + t.c

/-- The statement that 18 is the smallest possible perimeter of an EvenTriangle -/
theorem smallest_even_triangle_perimeter :
  ∀ t : EvenTriangle, perimeter t ≥ 18 ∧ ∃ t₀ : EvenTriangle, perimeter t₀ = 18 := by
  sorry

end smallest_even_triangle_perimeter_l130_13019


namespace sector_max_area_angle_l130_13057

/-- Given a sector with circumference 4 cm, the central angle that maximizes the area is π radians. -/
theorem sector_max_area_angle (r : ℝ) (θ : ℝ) :
  r * θ + 2 * r = 4 →  -- Circumference condition
  (∀ r' θ', r' * θ' + 2 * r' = 4 → 
    (1/2) * r^2 * θ ≥ (1/2) * r'^2 * θ') →  -- Area maximization condition
  θ = π :=
sorry

end sector_max_area_angle_l130_13057


namespace scientists_born_in_july_percentage_l130_13025

theorem scientists_born_in_july_percentage :
  let total_scientists : ℕ := 120
  let july_born_scientists : ℕ := 15
  (july_born_scientists : ℚ) / total_scientists * 100 = 12.5 :=
by sorry

end scientists_born_in_july_percentage_l130_13025


namespace reciprocal_geometric_progression_sum_l130_13091

theorem reciprocal_geometric_progression_sum
  (n : ℕ)  -- number of terms divided by 2
  (r : ℝ)  -- half of the common ratio
  (S : ℝ)  -- sum of the original geometric progression
  (h1 : S = (1 - (2*r)^(2*n)) / (1 - 2*r))  -- definition of S
  : (1 - (1/(2*r))^(2*n)) / (1 - 1/(2*r)) = S / (2^n * r^(2*n-1)) :=
sorry

end reciprocal_geometric_progression_sum_l130_13091


namespace sum_digits_888_base_8_l130_13007

/-- Represents a number in base 8 as a list of digits (least significant digit first) -/
def BaseEightRepresentation := List Nat

/-- Converts a natural number to its base 8 representation -/
def toBaseEight (n : Nat) : BaseEightRepresentation :=
  sorry

/-- Calculates the sum of digits in a base 8 representation -/
def sumDigits (repr : BaseEightRepresentation) : Nat :=
  sorry

theorem sum_digits_888_base_8 :
  sumDigits (toBaseEight 888) = 13 := by
  sorry

end sum_digits_888_base_8_l130_13007


namespace arithmetic_sequence_property_l130_13039

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence a_n, if a_3 + a_8 = 22 and a_6 = 7, then a_5 = 15 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 8 = 22) 
  (h_a6 : a 6 = 7) : 
  a 5 = 15 := by
  sorry

end arithmetic_sequence_property_l130_13039


namespace nested_radical_value_l130_13064

/-- Given a continuous nested radical X = √(x√(y√(z√(x√(y√(z...)))))), 
    prove that X = ∛(x^4 * y^2 * z) -/
theorem nested_radical_value (x y z : ℝ) (X : ℝ) 
  (h : X = Real.sqrt (x * Real.sqrt (y * Real.sqrt (z * X)))) :
  X = (x^4 * y^2 * z)^(1/7) := by
sorry

end nested_radical_value_l130_13064


namespace parabola_translation_l130_13029

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the vertical translation
def vertical_translation (y : ℝ) : ℝ := y + 3

-- Define the horizontal translation
def horizontal_translation (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem parabola_translation :
  ∀ x : ℝ, vertical_translation (original_parabola (horizontal_translation x)) = (x - 1)^2 + 3 :=
by sorry

end parabola_translation_l130_13029


namespace two_distinct_cool_triples_for_odd_x_l130_13080

/-- A cool type triple (x, y, z) consists of positive integers with y ≥ 2 
    and satisfies the equation x^2 - 3y^2 = z^2 - 3 -/
def CoolTriple (x y z : ℕ) : Prop :=
  x > 0 ∧ y ≥ 2 ∧ z > 0 ∧ x^2 - 3*y^2 = z^2 - 3

/-- For every odd x ≥ 5, there exist at least two distinct cool triples -/
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h_odd : Odd x) (h_ge_5 : x ≥ 5) :
  ∃ y1 z1 y2 z2 : ℕ, 
    CoolTriple x y1 z1 ∧ 
    CoolTriple x y2 z2 ∧ 
    (y1 ≠ y2 ∨ z1 ≠ z2) :=
by
  sorry

end two_distinct_cool_triples_for_odd_x_l130_13080


namespace rectangle_diagonal_l130_13067

/-- The diagonal of a rectangle with length 30√3 cm and width 30 cm is 60 cm. -/
theorem rectangle_diagonal : 
  let length : ℝ := 30 * Real.sqrt 3
  let width : ℝ := 30
  let diagonal : ℝ := Real.sqrt (length^2 + width^2)
  diagonal = 60 := by sorry

end rectangle_diagonal_l130_13067


namespace shore_distance_l130_13048

/-- The distance between two shores A and B, given the movement of two boats --/
theorem shore_distance (d : ℝ) : d = 800 :=
  -- Define the meeting points
  let first_meeting : ℝ := 500
  let second_meeting : ℝ := d - 300

  -- Define the distances traveled by each boat at the first meeting
  let boat_m_first : ℝ := first_meeting
  let boat_b_first : ℝ := d - first_meeting

  -- Define the distances traveled by each boat at the second meeting
  let boat_m_second : ℝ := second_meeting
  let boat_b_second : ℝ := 300

  -- The ratio of distances traveled should be equal for both meetings
  have h : boat_m_first / boat_b_first = boat_m_second / boat_b_second := by sorry

  -- The distance d satisfies the equation derived from the equal ratios
  have eq : d * d - 800 * d = 0 := by sorry

  -- The only positive solution to this equation is 800
  sorry


end shore_distance_l130_13048


namespace min_colors_for_triangle_free_edge_coloring_l130_13089

theorem min_colors_for_triangle_free_edge_coloring (n : Nat) (h : n = 2015) :
  ∃ (f : Fin n → Fin n → Fin n),
    (∀ (i j : Fin n), i ≠ j → f i j = f j i) ∧
    (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → f i j ≠ f j k ∨ f j k ≠ f i k ∨ f i k ≠ f i j) ∧
    (∀ (g : Fin n → Fin n → Fin (n - 1)),
      ∃ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ g i j = g j k ∧ g j k = g i k) :=
by sorry

#check min_colors_for_triangle_free_edge_coloring

end min_colors_for_triangle_free_edge_coloring_l130_13089


namespace quadratic_roots_ratio_l130_13015

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 ∧ p / q = 3 / 2 ∧ 
   p + q = -10 ∧ p * q = k ∧ 
   ∀ x : ℝ, x^2 + 10*x + k = 0 ↔ (x = p ∨ x = q)) → 
  k = 24 := by
sorry

end quadratic_roots_ratio_l130_13015


namespace x_value_l130_13046

theorem x_value (x y : ℝ) (h1 : x - y = 18) (h2 : x + y = 10) : x = 14 := by
  sorry

end x_value_l130_13046


namespace solution_set_f_less_than_3_range_of_a_when_f_plus_g_greater_than_1_l130_13090

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - a|
def g (x : ℝ) : ℝ := |x + 1|

-- Statement for part (I)
theorem solution_set_f_less_than_3 (x : ℝ) :
  |3 * x - 4| < 3 ↔ 1/3 < x ∧ x < 7/3 :=
sorry

-- Statement for part (II)
theorem range_of_a_when_f_plus_g_greater_than_1 (a : ℝ) :
  (∀ x : ℝ, f a x + g x > 1) ↔ a < -6 ∨ a > 0 :=
sorry

end solution_set_f_less_than_3_range_of_a_when_f_plus_g_greater_than_1_l130_13090


namespace tan_equality_implies_160_degrees_l130_13045

theorem tan_equality_implies_160_degrees (x : Real) :
  0 ≤ x ∧ x < 360 →
  Real.tan ((150 - x) * π / 180) = (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) / 
                                   (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) →
  x = 160 := by
sorry

end tan_equality_implies_160_degrees_l130_13045


namespace base_equality_proof_l130_13063

/-- Converts a base-6 number to decimal --/
def base6ToDecimal (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 36 + tens * 6 + ones

/-- Converts a number in base b to decimal --/
def baseToDecimal (n : Nat) (b : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

theorem base_equality_proof : 
  ∃! (b : Nat), b > 0 ∧ base6ToDecimal 142 = baseToDecimal 215 b :=
by
  sorry

end base_equality_proof_l130_13063


namespace correct_termination_condition_l130_13085

/-- Represents the state of the program at each iteration --/
structure ProgramState :=
  (i : ℕ)
  (S : ℕ)

/-- Simulates one iteration of the loop --/
def iterate (state : ProgramState) : ProgramState :=
  { i := state.i - 1, S := state.S * state.i }

/-- Checks if the given condition terminates the loop correctly --/
def is_correct_termination (condition : ℕ → Bool) : Prop :=
  let final_state := iterate (iterate (iterate (iterate { i := 12, S := 1 })))
  final_state.S = 11880 ∧ condition final_state.i = true ∧ 
  ∀ n, n < 4 → condition ((iterate^[n] { i := 12, S := 1 }).i) = false

theorem correct_termination_condition :
  is_correct_termination (λ i => i = 8) := by sorry

end correct_termination_condition_l130_13085


namespace first_episode_length_l130_13059

/-- Given a series with four episodes, where the second episode is 62 minutes long,
    the third episode is 65 minutes long, the fourth episode is 55 minutes long,
    and the total duration of all four episodes is 4 hours,
    prove that the first episode is 58 minutes long. -/
theorem first_episode_length :
  ∀ (episode1 episode2 episode3 episode4 : ℕ),
  episode2 = 62 →
  episode3 = 65 →
  episode4 = 55 →
  episode1 + episode2 + episode3 + episode4 = 4 * 60 →
  episode1 = 58 :=
by
  sorry


end first_episode_length_l130_13059


namespace prob_two_s_is_one_tenth_l130_13079

/-- The set of tiles containing letters G, A, U, S, and S -/
def tiles : Finset Char := {'G', 'A', 'U', 'S', 'S'}

/-- The number of S tiles in the set -/
def num_s_tiles : Nat := (tiles.filter (· = 'S')).card

/-- The probability of selecting two S tiles when choosing 2 tiles at random -/
def prob_two_s : ℚ := (num_s_tiles.choose 2 : ℚ) / (tiles.card.choose 2)

/-- Theorem stating that the probability of selecting two S tiles is 1/10 -/
theorem prob_two_s_is_one_tenth : prob_two_s = 1 / 10 := by sorry

end prob_two_s_is_one_tenth_l130_13079


namespace hannahs_appliance_cost_l130_13098

/-- The total cost of a washing machine and dryer after applying a discount -/
def total_cost_after_discount (washing_machine_cost : ℝ) (dryer_cost_difference : ℝ) (discount_rate : ℝ) : ℝ :=
  let dryer_cost := washing_machine_cost - dryer_cost_difference
  let total_cost := washing_machine_cost + dryer_cost
  let discount := total_cost * discount_rate
  total_cost - discount

/-- Theorem stating the total cost after discount for Hannah's purchase -/
theorem hannahs_appliance_cost :
  total_cost_after_discount 100 30 0.1 = 153 := by
  sorry

end hannahs_appliance_cost_l130_13098


namespace correct_grooming_time_l130_13083

/-- Represents the grooming time for a cat -/
structure GroomingTime where
  nailClipTime : ℕ  -- Time to clip one nail in seconds
  earCleanTime : ℕ  -- Time to clean one ear in seconds
  totalTime : ℕ     -- Total grooming time in seconds

/-- Calculates the total grooming time for a cat -/
def calculateGroomingTime (gt : GroomingTime) (numClaws numFeet numEars : ℕ) : ℕ :=
  (gt.nailClipTime * numClaws * numFeet) + (gt.earCleanTime * numEars) + 
  (gt.totalTime - (gt.nailClipTime * numClaws * numFeet) - (gt.earCleanTime * numEars))

/-- Theorem stating that the total grooming time is correct -/
theorem correct_grooming_time (gt : GroomingTime) :
  gt.nailClipTime = 10 → 
  gt.earCleanTime = 90 → 
  gt.totalTime = 640 → 
  calculateGroomingTime gt 4 4 2 = 640 := by
  sorry

#eval calculateGroomingTime { nailClipTime := 10, earCleanTime := 90, totalTime := 640 } 4 4 2

end correct_grooming_time_l130_13083


namespace F_fraction_difference_l130_13023

def F : ℚ := 925 / 999

theorem F_fraction_difference : ∃ (a b : ℕ), 
  F = a / b ∧ 
  (∀ (c d : ℕ), F = c / d → b ≤ d) ∧
  b - a = 2 := by
  sorry

end F_fraction_difference_l130_13023


namespace remainder_calculation_l130_13062

theorem remainder_calculation (P Q R D Q' R' D' D'' Q'' R'' : ℤ)
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : D'' = D' + 1)
  (h4 : P = D'' * Q'' + R'') :
  R'' = R + D * R' - Q'' := by sorry

end remainder_calculation_l130_13062


namespace art_club_artworks_l130_13003

/-- The number of artworks collected by an art club over multiple school years -/
def artworks_collected (num_students : ℕ) (artworks_per_quarter : ℕ) (quarters_per_year : ℕ) (num_years : ℕ) : ℕ :=
  num_students * artworks_per_quarter * quarters_per_year * num_years

/-- Theorem: The art club collects 900 artworks in 3 school years -/
theorem art_club_artworks :
  artworks_collected 25 3 4 3 = 900 := by
  sorry

end art_club_artworks_l130_13003


namespace workshop_technicians_salary_l130_13009

/-- Represents the average salary of technicians in a workshop -/
def average_salary_technicians (total_workers : ℕ) (technicians : ℕ) (avg_salary_all : ℚ) (avg_salary_others : ℚ) : ℚ :=
  let other_workers := total_workers - technicians
  let total_salary := (avg_salary_all * total_workers : ℚ)
  let other_salary := (avg_salary_others * other_workers : ℚ)
  let technicians_salary := total_salary - other_salary
  technicians_salary / technicians

/-- Theorem stating that the average salary of technicians is 1000 given the workshop conditions -/
theorem workshop_technicians_salary :
  average_salary_technicians 22 7 850 780 = 1000 := by
  sorry

end workshop_technicians_salary_l130_13009


namespace total_trees_planted_l130_13072

theorem total_trees_planted (total_gardeners : ℕ) (street_a_gardeners : ℕ) (street_b_gardeners : ℕ) 
  (h1 : total_gardeners = street_a_gardeners + street_b_gardeners)
  (h2 : total_gardeners = 19)
  (h3 : street_a_gardeners = 4)
  (h4 : street_b_gardeners = 15)
  (h5 : ∃ x : ℕ, street_b_gardeners * x - 1 = 4 * (street_a_gardeners * x - 1)) :
  ∃ trees_per_gardener : ℕ, total_gardeners * trees_per_gardener = 57 :=
by sorry

end total_trees_planted_l130_13072


namespace pizza_combinations_l130_13027

theorem pizza_combinations (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 3) :
  Nat.choose n k = 56 := by
  sorry

end pizza_combinations_l130_13027


namespace hexagon_perimeter_l130_13044

/-- A hexagon ABCDEF with specific side lengths -/
structure Hexagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EF : ℝ
  FA : ℝ

/-- The perimeter of a hexagon -/
def perimeter (h : Hexagon) : ℝ :=
  h.AB + h.BC + h.CD + h.DE + h.EF + h.FA

/-- Theorem: The perimeter of the specific hexagon ABCDEF is 13 -/
theorem hexagon_perimeter :
  ∃ (h : Hexagon),
    h.AB = 2 ∧ h.BC = 2 ∧ h.CD = 2 ∧ h.DE = 2 ∧ h.EF = 2 ∧ h.FA = 3 ∧
    perimeter h = 13 := by
  sorry

end hexagon_perimeter_l130_13044


namespace gcf_of_120_180_300_l130_13021

theorem gcf_of_120_180_300 : Nat.gcd 120 (Nat.gcd 180 300) = 60 := by
  sorry

end gcf_of_120_180_300_l130_13021


namespace solve_for_a_l130_13078

-- Define the equation for all a, b, and c
def equation (a b c : ℝ) : Prop :=
  a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)

-- Define the theorem
theorem solve_for_a :
  ∀ a : ℝ, (∀ b c : ℝ, equation a b c) → a * 15 * 2 = 4 → a = 6 :=
by
  sorry


end solve_for_a_l130_13078


namespace distance_between_docks_l130_13002

/-- The distance between docks A and B in kilometers. -/
def distance : ℝ := 105

/-- The speed of the water flow in kilometers per hour. -/
def water_speed : ℝ := 3

/-- The time taken to travel downstream in hours. -/
def downstream_time : ℝ := 5

/-- The time taken to travel upstream in hours. -/
def upstream_time : ℝ := 7

/-- Theorem stating that the distance between docks A and B is 105 kilometers. -/
theorem distance_between_docks :
  distance = 105 ∧
  water_speed = 3 ∧
  downstream_time = 5 ∧
  upstream_time = 7 ∧
  (distance / downstream_time - water_speed = distance / upstream_time + water_speed) :=
by sorry

end distance_between_docks_l130_13002


namespace picture_distance_l130_13042

/-- Proves that for a wall of width 24 feet and a picture of width 4 feet hung in the center,
    the distance from the end of the wall to the nearest edge of the picture is 10 feet. -/
theorem picture_distance (wall_width picture_width : ℝ) (h1 : wall_width = 24) (h2 : picture_width = 4) :
  let distance := (wall_width - picture_width) / 2
  distance = 10 := by
sorry

end picture_distance_l130_13042


namespace percentage_calculation_l130_13053

theorem percentage_calculation (x : ℝ) (p : ℝ) (h1 : x = 60) (h2 : x = (p / 100) * x + 52.8) : p = 12 := by
  sorry

end percentage_calculation_l130_13053


namespace sum_of_radii_tangent_circles_l130_13094

/-- The sum of radii of a circle tangent to x and y axes and externally tangent to another circle -/
theorem sum_of_radii_tangent_circles : ∀ r : ℝ,
  (r > 0) →
  ((r - 5)^2 + r^2 = (r + 2)^2) →
  ∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ (r₁ + r₂ = 14) := by
sorry

end sum_of_radii_tangent_circles_l130_13094


namespace siblings_age_sum_l130_13043

/-- The age difference between siblings -/
def age_gap : ℕ := 5

/-- The current age of the eldest sibling -/
def eldest_age_now : ℕ := 20

/-- The number of years into the future we're considering -/
def years_forward : ℕ := 10

/-- The total age of three siblings after a given number of years -/
def total_age_after (years : ℕ) : ℕ :=
  (eldest_age_now + years) + (eldest_age_now - age_gap + years) + (eldest_age_now - 2 * age_gap + years)

theorem siblings_age_sum :
  total_age_after years_forward = 75 :=
by sorry

end siblings_age_sum_l130_13043


namespace scarf_price_reduction_l130_13065

/-- Calculates the final price of a scarf after two successive price reductions -/
theorem scarf_price_reduction (original_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  original_price = 10 ∧ first_reduction = 0.3 ∧ second_reduction = 0.5 →
  original_price * (1 - first_reduction) * (1 - second_reduction) = 3.5 := by
  sorry

end scarf_price_reduction_l130_13065


namespace cookies_remaining_batches_l130_13040

/-- Given the following conditions:
  * Each batch of cookies requires 2 cups of flour
  * 3 batches of cookies were baked
  * The initial amount of flour was 20 cups
  Prove that 7 additional batches of cookies can be made with the remaining flour -/
theorem cookies_remaining_batches 
  (flour_per_batch : ℕ) 
  (batches_baked : ℕ) 
  (initial_flour : ℕ) : 
  flour_per_batch = 2 →
  batches_baked = 3 →
  initial_flour = 20 →
  (initial_flour - flour_per_batch * batches_baked) / flour_per_batch = 7 :=
by sorry

end cookies_remaining_batches_l130_13040


namespace four_digit_greater_than_three_digit_l130_13032

theorem four_digit_greater_than_three_digit :
  ∀ (a b : ℕ), (1000 ≤ a ∧ a < 10000) → (100 ≤ b ∧ b < 1000) → a > b :=
by
  sorry

end four_digit_greater_than_three_digit_l130_13032


namespace empty_jar_weight_l130_13076

/-- Represents the weight of a jar with water -/
structure JarWeight where
  empty : ℝ  -- Weight of the empty jar
  water : ℝ  -- Weight of water when fully filled

/-- The weight of the jar when partially filled -/
def partialWeight (j : JarWeight) (fraction : ℝ) : ℝ :=
  j.empty + fraction * j.water

theorem empty_jar_weight (j : JarWeight) :
  (partialWeight j (1/5) = 560) →
  (partialWeight j (4/5) = 740) →
  j.empty = 500 := by
  sorry

end empty_jar_weight_l130_13076


namespace circle_square_radius_l130_13031

theorem circle_square_radius (s : ℝ) (r : ℝ) : 
  s^2 = 9/16 →                  -- Area of the square is 9/16
  π * r^2 = 9/16 →              -- Area of the circle is 9/16
  2 * r = s →                   -- Diameter of circle equals side length of square
  r = 3/8 := by                 -- Radius of the circle is 3/8
sorry

end circle_square_radius_l130_13031


namespace circle_collinearity_l130_13038

-- Define the circle ω
def circle_ω (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | dist P ((A + B) / 2) = dist A B / 2}

-- Define a point on the circle
def point_on_circle (O : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  O ∈ ω

-- Define orthogonal projection
def orthogonal_projection (O H : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  (H.1 - A.1) * (B.2 - A.2) = (H.2 - A.2) * (B.1 - A.1) ∧
  (O.1 - H.1) * (B.1 - A.1) + (O.2 - H.2) * (B.2 - A.2) = 0

-- Define the intersection of two circles
def circle_intersection (O H X Y : ℝ × ℝ) (ω : Set (ℝ × ℝ)) : Prop :=
  X ∈ ω ∧ Y ∈ ω ∧
  dist X O = dist O H ∧ dist Y O = dist O H

-- Define collinearity
def collinear (P Q R : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (R.1 - P.1) = (R.2 - P.2) * (Q.1 - P.1)

-- The main theorem
theorem circle_collinearity 
  (A B O H X Y : ℝ × ℝ) (ω : Set (ℝ × ℝ)) :
  ω = circle_ω A B →
  point_on_circle O ω →
  orthogonal_projection O H A B →
  circle_intersection O H X Y ω →
  collinear X Y ((O + H) / 2) :=
sorry

end circle_collinearity_l130_13038


namespace unique_integer_congruence_l130_13052

theorem unique_integer_congruence :
  ∃! n : ℤ, 6 ≤ n ∧ n ≤ 12 ∧ n ≡ 10403 [ZMOD 7] ∧ n = 8 := by
  sorry

end unique_integer_congruence_l130_13052


namespace city_water_consumption_most_suitable_l130_13005

/-- Represents a survey scenario -/
structure SurveyScenario where
  description : String
  population_size : Nat
  practicality_of_sampling : Bool

/-- Determines if a survey scenario is suitable for sampling -/
def is_suitable_for_sampling (scenario : SurveyScenario) : Bool :=
  scenario.population_size > 1000 && scenario.practicality_of_sampling

/-- The list of survey scenarios -/
def survey_scenarios : List SurveyScenario := [
  { description := "Security check for passengers before boarding a plane",
    population_size := 300,
    practicality_of_sampling := false },
  { description := "Survey of the vision of students in Grade 8, Class 1 of a certain school",
    population_size := 40,
    practicality_of_sampling := false },
  { description := "Survey of the average daily water consumption in a certain city",
    population_size := 100000,
    practicality_of_sampling := true },
  { description := "Survey of the sleep time of 20 centenarians in a certain county",
    population_size := 20,
    practicality_of_sampling := false }
]

theorem city_water_consumption_most_suitable :
  ∃ (scenario : SurveyScenario),
    scenario ∈ survey_scenarios ∧
    scenario.description = "Survey of the average daily water consumption in a certain city" ∧
    is_suitable_for_sampling scenario ∧
    ∀ (other : SurveyScenario),
      other ∈ survey_scenarios →
      other ≠ scenario →
      ¬(is_suitable_for_sampling other) :=
by sorry

end city_water_consumption_most_suitable_l130_13005


namespace sum_squares_range_l130_13056

/-- Given a positive constant k and a sequence of positive real numbers x_i whose sum equals k,
    the sum of x_i^2 can take any value in the open interval (0, k^2). -/
theorem sum_squares_range (k : ℝ) (x : ℕ → ℝ) (h_k_pos : k > 0) (h_x_pos : ∀ n, x n > 0)
    (h_sum_x : ∑' n, x n = k) :
  ∀ y, 0 < y ∧ y < k^2 → ∃ x : ℕ → ℝ,
    (∀ n, x n > 0) ∧ (∑' n, x n = k) ∧ (∑' n, (x n)^2 = y) :=
by sorry

end sum_squares_range_l130_13056


namespace range_of_a_l130_13047

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := ∀ x, x ∈ A → x ∈ B a

-- Theorem statement
theorem range_of_a : 
  ∀ a : ℝ, sufficient_condition a ↔ a ≤ -3 :=
sorry

end range_of_a_l130_13047


namespace arithmetic_sequence_separable_special_sequence_a_value_complex_sequence_separable_values_l130_13082

/-- A sequence is m-th degree separable if there exists an n such that a_{m+n} = a_m + a_n -/
def IsNthDegreeSeparable (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∃ n : ℕ, a (m + n) = a m + a n

/-- An arithmetic sequence with first term 2 and common difference 2 -/
def ArithmeticSequence (n : ℕ) : ℝ := 2 * n

/-- A sequence with sum of first n terms S_n = 2^n - a where a > 0 -/
def SpecialSequence (a : ℝ) (n : ℕ) : ℝ := 2^n - a

/-- A sequence defined by a_n = 2^n + n^2 + 12 -/
def ComplexSequence (n : ℕ) : ℝ := 2^n + n^2 + 12

theorem arithmetic_sequence_separable :
  IsNthDegreeSeparable ArithmeticSequence 3 :=
sorry

theorem special_sequence_a_value (a : ℝ) (h : a > 0) :
  IsNthDegreeSeparable (SpecialSequence a) 1 → a = 1 :=
sorry

theorem complex_sequence_separable_values :
  (∃ m : ℕ, IsNthDegreeSeparable ComplexSequence m) ∧
  (∀ m : ℕ, IsNthDegreeSeparable ComplexSequence m → (m = 1 ∨ m = 3)) :=
sorry

end arithmetic_sequence_separable_special_sequence_a_value_complex_sequence_separable_values_l130_13082


namespace remaining_budget_theorem_l130_13016

/-- Represents the annual budget of Centerville --/
def annual_budget : ℝ := 20000

/-- Represents the percentage of the budget spent on the public library --/
def library_percentage : ℝ := 0.15

/-- Represents the amount spent on the public library --/
def library_spending : ℝ := 3000

/-- Represents the percentage of the budget spent on public parks --/
def parks_percentage : ℝ := 0.24

/-- Theorem stating the remaining amount of the budget after library and parks spending --/
theorem remaining_budget_theorem :
  annual_budget * (1 - library_percentage - parks_percentage) = 12200 := by
  sorry


end remaining_budget_theorem_l130_13016


namespace eggs_sold_count_l130_13095

/-- The number of eggs in each tray -/
def eggs_per_tray : ℕ := 30

/-- The initial number of trays to be collected -/
def initial_trays : ℕ := 10

/-- The number of trays dropped -/
def dropped_trays : ℕ := 2

/-- The number of additional trays added after the accident -/
def additional_trays : ℕ := 7

/-- Theorem stating the total number of eggs sold -/
theorem eggs_sold_count : 
  (initial_trays - dropped_trays + additional_trays) * eggs_per_tray = 450 := by
sorry

end eggs_sold_count_l130_13095


namespace polynomial_condition_l130_13077

/-- A polynomial P satisfying the given condition for all real a, b, c is of the form ax² + bx -/
theorem polynomial_condition (P : ℝ → ℝ) : 
  (∀ (a b c : ℝ), P (a + b - 2*c) + P (b + c - 2*a) + P (c + a - 2*b) = 
    3 * P (a - b) + 3 * P (b - c) + 3 * P (c - a)) →
  ∃ (a b : ℝ), ∀ x, P x = a * x^2 + b * x :=
by sorry

end polynomial_condition_l130_13077


namespace f_2017_eq_cos_l130_13022

open Real

/-- Recursive definition of the function sequence f_n -/
noncomputable def f (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => sin
  | n + 1 => deriv (f n)

/-- The 2017th function in the sequence equals cosine -/
theorem f_2017_eq_cos : f 2017 = cos := by
  sorry

end f_2017_eq_cos_l130_13022


namespace milk_bottle_boxes_l130_13020

/-- Given a total number of milk bottles, bottles per bag, and bags per box,
    calculate the total number of boxes. -/
def calculate_boxes (total_bottles : ℕ) (bottles_per_bag : ℕ) (bags_per_box : ℕ) : ℕ :=
  total_bottles / (bottles_per_bag * bags_per_box)

/-- Theorem stating that given 8640 milk bottles, with 12 bottles per bag and 6 bags per box,
    the total number of boxes is equal to 120. -/
theorem milk_bottle_boxes :
  calculate_boxes 8640 12 6 = 120 := by
  sorry

end milk_bottle_boxes_l130_13020


namespace range_of_m_when_a_is_zero_x_minus_one_times_f_nonpositive_l130_13087

noncomputable section

-- Define the function f
def f (m a x : ℝ) : ℝ := -m * (a * x + 1) * Real.log x + x - a

-- Part 1
theorem range_of_m_when_a_is_zero (m : ℝ) :
  (∀ x > 1, f m 0 x ≥ 0) ↔ m ∈ Set.Iic (Real.exp 1) :=
sorry

-- Part 2
theorem x_minus_one_times_f_nonpositive (x : ℝ) (hx : x > 0) :
  (x - 1) * f 1 1 x ≤ 0 :=
sorry

end range_of_m_when_a_is_zero_x_minus_one_times_f_nonpositive_l130_13087


namespace cafe_customers_l130_13024

/-- Prove that the number of customers in a group is 12, given the following conditions:
  * 3 offices ordered 10 sandwiches each
  * Half of the group ordered 4 sandwiches each
  * Total sandwiches made is 54
-/
theorem cafe_customers (num_offices : Nat) (sandwiches_per_office : Nat)
  (sandwiches_per_customer : Nat) (total_sandwiches : Nat) :
  num_offices = 3 →
  sandwiches_per_office = 10 →
  sandwiches_per_customer = 4 →
  total_sandwiches = 54 →
  ∃ (num_customers : Nat),
    num_customers = 12 ∧
    total_sandwiches = num_offices * sandwiches_per_office +
      (num_customers / 2) * sandwiches_per_customer :=
by
  sorry

end cafe_customers_l130_13024


namespace circle_tangent_triangle_l130_13014

/-- Given a circle with radius R externally tangent to triangle ABC, 
    prove that angle C is π/6 and the maximum area is (√3 + 2)/4 * R^2 -/
theorem circle_tangent_triangle (R a b c : ℝ) (A B C : ℝ) :
  R > 0 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  2 * R * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 3 * a - b) * Real.sin B →
  C = π / 6 ∧ 
  ∃ (S : ℝ), S ≤ (Real.sqrt 3 + 2) / 4 * R^2 ∧ 
    (∀ (A' B' C' : ℝ), A' + B' + C' = π → 
      1 / 2 * 2 * R * Real.sin A' * 2 * R * Real.sin B' * Real.sin C' ≤ S) :=
by sorry

end circle_tangent_triangle_l130_13014


namespace cosine_sine_expression_value_l130_13018

theorem cosine_sine_expression_value : 
  Real.cos (10 * π / 180) * Real.sin (70 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end cosine_sine_expression_value_l130_13018


namespace events_mutually_exclusive_but_not_opposite_l130_13026

-- Define the total number of balls and the number of each color
def totalBalls : ℕ := 6
def redBalls : ℕ := 3
def whiteBalls : ℕ := 3

-- Define the number of balls drawn
def ballsDrawn : ℕ := 3

-- Define the events
def atLeastTwoWhite (w : ℕ) : Prop := w ≥ 2
def allRed (r : ℕ) : Prop := r = 3

-- Define mutual exclusivity
def mutuallyExclusive (e1 e2 : Prop) : Prop :=
  ¬(e1 ∧ e2)

-- Define opposite events
def oppositeEvents (e1 e2 : Prop) : Prop :=
  ∀ (outcome : ℕ × ℕ), (e1 ∨ e2) ∧ ¬(e1 ∧ e2)

-- Theorem statement
theorem events_mutually_exclusive_but_not_opposite :
  (mutuallyExclusive (atLeastTwoWhite whiteBalls) (allRed redBalls)) ∧
  ¬(oppositeEvents (atLeastTwoWhite whiteBalls) (allRed redBalls)) :=
by sorry

end events_mutually_exclusive_but_not_opposite_l130_13026


namespace geometric_sequence_ratio_l130_13004

/-- Given a geometric sequence {a_n} with common ratio q = 1/2, prove that S_4 / a_2 = 15/4,
    where S_n is the sum of the first n terms. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) = (1 / 2) * a n) →  -- Common ratio q = 1/2
  (∀ n, S n = a 1 * (1 - (1 / 2)^n) / (1 - (1 / 2))) →  -- Definition of S_n
  S 4 / a 2 = 15 / 4 := by
  sorry

end geometric_sequence_ratio_l130_13004


namespace derivative_x_minus_inverse_x_l130_13069

open Real

theorem derivative_x_minus_inverse_x (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => x - 1 / x) x = 1 + 1 / x^2 :=
sorry

end derivative_x_minus_inverse_x_l130_13069


namespace work_completion_time_l130_13061

/-- Given two workers a and b, where a is twice as fast as b, and b can complete a work in 24 days,
    prove that a and b together can complete the work in 8 days. -/
theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : b * 24 = 1) :
  1 / (a + b) = 8 :=
sorry

end work_completion_time_l130_13061


namespace smallest_number_divisible_l130_13033

theorem smallest_number_divisible (n : ℕ) : n ≥ 1012 ∧ 
  (∀ m : ℕ, m < 1012 → 
    ¬(((m - 4) % 12 = 0) ∧ 
      ((m - 4) % 16 = 0) ∧ 
      ((m - 4) % 18 = 0) ∧ 
      ((m - 4) % 21 = 0) ∧ 
      ((m - 4) % 28 = 0))) →
  ((n - 4) % 12 = 0) ∧ 
  ((n - 4) % 16 = 0) ∧ 
  ((n - 4) % 18 = 0) ∧ 
  ((n - 4) % 21 = 0) ∧ 
  ((n - 4) % 28 = 0) :=
by sorry

#check smallest_number_divisible

end smallest_number_divisible_l130_13033


namespace union_intersection_result_intersection_complement_result_l130_13028

-- Define the universe set U
def U : Set ℤ := {x | -3 ≤ x ∧ x ≤ 3}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {-1, 0, 1}
def C : Set ℤ := {-2, 0, 2}

-- Theorem for the first part
theorem union_intersection_result : A ∪ (B ∩ C) = {0, 1, 2, 3} := by
  sorry

-- Theorem for the second part
theorem intersection_complement_result : A ∩ (U \ (B ∪ C)) = {3} := by
  sorry

end union_intersection_result_intersection_complement_result_l130_13028


namespace diophantine_equation_solutions_l130_13034

theorem diophantine_equation_solutions (x y : ℤ) :
  x^6 - y^2 = 648 ↔ (x = 3 ∧ y = 9) ∨ (x = -3 ∧ y = 9) ∨ (x = 3 ∧ y = -9) ∨ (x = -3 ∧ y = -9) := by
  sorry

end diophantine_equation_solutions_l130_13034


namespace rectangle_dimension_change_l130_13030

/-- Proves that a 45% increase in breadth and 88.5% increase in area results in a 30% increase in length for a rectangle -/
theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h_positive : L > 0 ∧ B > 0) :
  B' = 1.45 * B ∧ L' * B' = 1.885 * (L * B) → L' = 1.3 * L := by
  sorry

end rectangle_dimension_change_l130_13030


namespace jacob_guarantee_sheep_l130_13093

/-- The maximum square number in the list -/
def max_square : Nat := 2021^2

/-- The list of square numbers from 1^2 to 2021^2 -/
def square_list : List Nat := List.range 2021 |>.map (λ x => (x + 1)^2)

/-- The game state, including the current sum on the whiteboard and the remaining numbers -/
structure GameState where
  sum : Nat
  remaining : List Nat

/-- A player's strategy for choosing a number from the list -/
def Strategy := GameState → Nat

/-- The result of playing the game, counting the number of times the sum is divisible by 4 after Jacob's turn -/
def play_game (jacob_strategy : Strategy) (laban_strategy : Strategy) : Nat :=
  sorry

/-- The theorem stating that Jacob can guarantee at least 506 sheep -/
theorem jacob_guarantee_sheep :
  ∃ (jacob_strategy : Strategy),
    ∀ (laban_strategy : Strategy),
      play_game jacob_strategy laban_strategy ≥ 506 := by
  sorry

end jacob_guarantee_sheep_l130_13093


namespace equation_solutions_l130_13092

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => (14*x - x^2)/(x + 2) * (x + (14 - x)/(x + 2))
  ∃ (a b c : ℝ), 
    (f a = 48 ∧ f b = 48 ∧ f c = 48) ∧
    (a = 4 ∧ b = (1 + Real.sqrt 193)/2 ∧ c = (1 - Real.sqrt 193)/2) :=
by sorry

end equation_solutions_l130_13092


namespace triangle_properties_l130_13068

-- Define the triangle ABC
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  -- Given conditions
  b = 4 → c = 5 → A = π / 3 →
  -- Properties to prove
  a = Real.sqrt 21 ∧ Real.sin (2 * B) = 4 * Real.sqrt 3 / 7 :=
by
  sorry

end triangle_properties_l130_13068
