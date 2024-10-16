import Mathlib

namespace NUMINAMATH_CALUDE_daisy_tuesday_toys_l389_38981

/-- The number of dog toys Daisy had on various days --/
structure DaisyToys where
  monday : ℕ
  tuesday_before : ℕ
  tuesday_after : ℕ
  wednesday_new : ℕ
  total_if_found : ℕ

/-- Theorem stating the number of toys Daisy had on Tuesday before new purchases --/
theorem daisy_tuesday_toys (d : DaisyToys)
  (h1 : d.monday = 5)
  (h2 : d.tuesday_after = d.tuesday_before + 3)
  (h3 : d.wednesday_new = 5)
  (h4 : d.total_if_found = 13)
  (h5 : d.total_if_found = d.tuesday_before + 3 + d.wednesday_new) :
  d.tuesday_before = 5 := by
  sorry

#check daisy_tuesday_toys

end NUMINAMATH_CALUDE_daisy_tuesday_toys_l389_38981


namespace NUMINAMATH_CALUDE_prism_volume_l389_38931

/-- A right rectangular prism with given face areas has a volume of 30 cubic inches -/
theorem prism_volume (l w h : ℝ) 
  (face1 : l * w = 10)
  (face2 : w * h = 15)
  (face3 : l * h = 6) :
  l * w * h = 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l389_38931


namespace NUMINAMATH_CALUDE_replacement_preserves_mean_and_variance_l389_38978

def initial_set : List ℤ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
def new_set : List ℤ := [-5, -5, -3, -2, -1, 0, 1, 1, 2, 3, 4, 5]

def mean (s : List ℤ) : ℚ := (s.sum : ℚ) / s.length

def variance (s : List ℤ) : ℚ :=
  let m := mean s
  (s.map (λ x => ((x : ℚ) - m) ^ 2)).sum / s.length

theorem replacement_preserves_mean_and_variance :
  mean initial_set = mean new_set ∧ variance initial_set = variance new_set :=
sorry

end NUMINAMATH_CALUDE_replacement_preserves_mean_and_variance_l389_38978


namespace NUMINAMATH_CALUDE_scaling_transformation_l389_38996

-- Define the original circle equation
def original_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scale_x (x : ℝ) : ℝ := 5 * x
def scale_y (y : ℝ) : ℝ := 3 * y

-- State the theorem
theorem scaling_transformation :
  ∀ x' y' : ℝ, (∃ x y : ℝ, original_equation x y ∧ x' = scale_x x ∧ y' = scale_y y) →
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_scaling_transformation_l389_38996


namespace NUMINAMATH_CALUDE_lcm_of_10_14_20_l389_38908

theorem lcm_of_10_14_20 : Nat.lcm (Nat.lcm 10 14) 20 = 140 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_10_14_20_l389_38908


namespace NUMINAMATH_CALUDE_max_sum_of_four_numbers_l389_38993

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d →
  (c + d) + (b + d) + (a + b + c) + (a + b + d) = 2017 →
  a + b + c + d ≤ 2015 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_numbers_l389_38993


namespace NUMINAMATH_CALUDE_fraction_equality_l389_38980

theorem fraction_equality (x y z : ℝ) (h1 : x / 2 = y / 3) (h2 : x / 2 = z / 5) (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l389_38980


namespace NUMINAMATH_CALUDE_median_of_special_list_l389_38942

def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def isMedian (m : ℕ) : Prop :=
  let N := sumOfSquares 100
  let leftCount := sumOfSquares (m - 1)
  let rightCount := sumOfSquares m
  N / 2 > leftCount ∧ N / 2 ≤ rightCount

theorem median_of_special_list : isMedian 72 := by
  sorry

end NUMINAMATH_CALUDE_median_of_special_list_l389_38942


namespace NUMINAMATH_CALUDE_jumping_jacks_ratio_l389_38967

/-- The ratio of Brooke's jumping jacks to Sidney's jumping jacks is 3:1 -/
theorem jumping_jacks_ratio : 
  let sidney_jj := [20, 36, 40, 50]
  let brooke_jj := 438
  (brooke_jj : ℚ) / (sidney_jj.sum : ℚ) = 3 / 1 := by sorry

end NUMINAMATH_CALUDE_jumping_jacks_ratio_l389_38967


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l389_38907

theorem algebraic_expression_value (x : ℝ) : 2 * x^2 + 2 * x + 5 = 9 → 3 * x^2 + 3 * x - 7 = -1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l389_38907


namespace NUMINAMATH_CALUDE_bungee_cord_extension_l389_38932

/-- The maximum extension of a bungee cord in a bungee jumping scenario -/
theorem bungee_cord_extension
  (m : ℝ) -- mass of the person
  (H : ℝ) -- maximum fall distance
  (k : ℝ) -- spring constant of the bungee cord
  (L₀ : ℝ) -- original length of the bungee cord
  (h : ℝ) -- extension of the bungee cord
  (g : ℝ) -- gravitational acceleration
  (hpos : h > 0)
  (mpos : m > 0)
  (kpos : k > 0)
  (Hpos : H > 0)
  (L₀pos : L₀ > 0)
  (gpos : g > 0)
  (hooke : k * h = 4 * m * g) -- Hooke's law and maximum tension condition
  (energy : m * g * H = (1/2) * k * h^2) -- Conservation of energy
  : h = H / 2 := by
  sorry

end NUMINAMATH_CALUDE_bungee_cord_extension_l389_38932


namespace NUMINAMATH_CALUDE_simplify_fraction_x_squared_minus_y_squared_l389_38997

-- Part 1
theorem simplify_fraction (a : ℝ) (h : a > 0) : 
  1 / (Real.sqrt a + 1) = (Real.sqrt a - 1) / 2 :=
sorry

-- Part 2
theorem x_squared_minus_y_squared (x y : ℝ) 
  (hx : x = 1 / (2 + Real.sqrt 3)) 
  (hy : y = 1 / (2 - Real.sqrt 3)) : 
  x^2 - y^2 = -8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_x_squared_minus_y_squared_l389_38997


namespace NUMINAMATH_CALUDE_fourth_power_divisor_count_l389_38911

theorem fourth_power_divisor_count (n : ℕ+) : ∃ d : ℕ, 
  (∀ k : ℕ, k ∣ n^4 ↔ k ≤ d) ∧ d % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_divisor_count_l389_38911


namespace NUMINAMATH_CALUDE_profit_calculation_l389_38901

theorem profit_calculation (marked_price : ℝ) (cost_price : ℝ) 
  (h1 : cost_price > 0) 
  (h2 : marked_price > 0) 
  (h3 : 0.8 * marked_price = 1.2 * cost_price) : 
  (marked_price - cost_price) / cost_price = 0.5 := by
sorry

end NUMINAMATH_CALUDE_profit_calculation_l389_38901


namespace NUMINAMATH_CALUDE_units_digit_G_1000_l389_38936

/-- The function G_n defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem stating that the units digit of G(1000) is 2 -/
theorem units_digit_G_1000 : unitsDigit (G 1000) = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_1000_l389_38936


namespace NUMINAMATH_CALUDE_lcm_problem_l389_38995

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by sorry

end NUMINAMATH_CALUDE_lcm_problem_l389_38995


namespace NUMINAMATH_CALUDE_hyperbola_and_parabola_properties_l389_38991

-- Define the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := 16 * x^2 - 9 * y^2 = 144

-- Define the parabola
def parabola_equation (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem hyperbola_and_parabola_properties :
  -- Length of real axis
  (∃ a : ℝ, a = 3 ∧ 2 * a = 6) ∧
  -- Length of imaginary axis
  (∃ b : ℝ, b = 4 ∧ 2 * b = 8) ∧
  -- Eccentricity
  (∃ e : ℝ, e = 5 / 3) ∧
  -- Parabola equation
  (∀ x y : ℝ, hyperbola_equation x y →
    (x = 0 ∧ y = 0 → parabola_equation x y) ∧
    (x = -3 ∧ y = 0 → parabola_equation x y)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_parabola_properties_l389_38991


namespace NUMINAMATH_CALUDE_count_valid_pairs_l389_38958

def is_valid_pair (A B : ℕ+) : Prop :=
  12 ∣ A ∧ 12 ∣ B ∧
  20 ∣ A ∧ 20 ∣ B ∧
  45 ∣ A ∧ 45 ∣ B ∧
  Nat.lcm A B = 4320

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ+ × ℕ+)), 
    (∀ p ∈ pairs, is_valid_pair p.1 p.2) ∧
    (∀ A B, is_valid_pair A B → (A, B) ∈ pairs) ∧
    pairs.card = 11 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l389_38958


namespace NUMINAMATH_CALUDE_mangoes_per_neighbor_l389_38903

-- Define the given conditions
def total_mangoes : ℕ := 560
def mangoes_to_family : ℕ := 50
def num_neighbors : ℕ := 12

-- Define the relationship between x and total mangoes
def mangoes_sold (total : ℕ) : ℕ := total / 2

-- Theorem statement
theorem mangoes_per_neighbor : 
  (total_mangoes - mangoes_sold total_mangoes - mangoes_to_family) / num_neighbors = 19 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_per_neighbor_l389_38903


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l389_38970

def A : Set ℝ := {x | x^2 - 4 < 0}

def B : Set ℝ := {x | ∃ n : ℤ, x = 2*n + 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l389_38970


namespace NUMINAMATH_CALUDE_stream_speed_l389_38950

theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 14 →
  distance = 4864 →
  total_time = 700 →
  (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time) →
  stream_speed = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l389_38950


namespace NUMINAMATH_CALUDE_solve_lindas_savings_l389_38905

def lindas_savings_problem (savings : ℚ) : Prop :=
  let furniture_fraction : ℚ := 3 / 5
  let tv_fraction : ℚ := 1 - furniture_fraction
  let tv_cost : ℚ := 400
  tv_fraction * savings = tv_cost ∧ savings = 1000

theorem solve_lindas_savings : ∃ (savings : ℚ), lindas_savings_problem savings :=
sorry

end NUMINAMATH_CALUDE_solve_lindas_savings_l389_38905


namespace NUMINAMATH_CALUDE_bottles_per_box_l389_38921

theorem bottles_per_box 
  (num_boxes : ℕ) 
  (bottle_capacity : ℚ) 
  (fill_ratio : ℚ) 
  (total_water : ℚ) :
  num_boxes = 10 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  total_water = 4500 →
  (total_water / (bottle_capacity * fill_ratio)) / num_boxes = 50 :=
by sorry

end NUMINAMATH_CALUDE_bottles_per_box_l389_38921


namespace NUMINAMATH_CALUDE_parallelogram_area_l389_38986

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 12) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 60 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_area_l389_38986


namespace NUMINAMATH_CALUDE_icosahedron_coloring_count_l389_38913

/-- The number of faces in a regular icosahedron -/
def num_faces : ℕ := 20

/-- The number of colors available -/
def num_colors : ℕ := 10

/-- The order of the rotational symmetry group of a regular icosahedron -/
def icosahedron_symmetry_order : ℕ := 60

/-- The number of rotations around an axis through opposite faces -/
def rotations_per_axis : ℕ := 5

theorem icosahedron_coloring_count :
  (Nat.factorial (num_colors - 1)) / rotations_per_axis =
  72576 := by sorry

end NUMINAMATH_CALUDE_icosahedron_coloring_count_l389_38913


namespace NUMINAMATH_CALUDE_equation_holds_for_three_l389_38944

-- Define the equation we want to prove
def equation (n : ℕ) : Prop :=
  (((17 * Real.sqrt 5 + 38) ^ (1 / n : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1 / n : ℝ))) = 2 * Real.sqrt 5

-- Theorem statement
theorem equation_holds_for_three : 
  equation 3 := by sorry

end NUMINAMATH_CALUDE_equation_holds_for_three_l389_38944


namespace NUMINAMATH_CALUDE_swim_team_girls_l389_38939

theorem swim_team_girls (boys girls : ℕ) : 
  girls = 5 * boys → 
  girls + boys = 96 → 
  girls = 80 := by
sorry

end NUMINAMATH_CALUDE_swim_team_girls_l389_38939


namespace NUMINAMATH_CALUDE_cricket_match_playtime_l389_38964

-- Define the total duration of the match in minutes
def total_duration : ℕ := 12 * 60 + 35

-- Define the lunch break duration in minutes
def lunch_break : ℕ := 15

-- Theorem to prove the actual playtime
theorem cricket_match_playtime :
  total_duration - lunch_break = 740 := by
  sorry

end NUMINAMATH_CALUDE_cricket_match_playtime_l389_38964


namespace NUMINAMATH_CALUDE_absolute_value_equals_negative_l389_38955

theorem absolute_value_equals_negative (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_negative_l389_38955


namespace NUMINAMATH_CALUDE_museum_trip_cost_l389_38902

/-- Calculates the total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem stating that the total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_trip_cost : total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_trip_cost_l389_38902


namespace NUMINAMATH_CALUDE_berry_ratio_l389_38975

/-- Given the distribution of berries among Stacy, Steve, and Sylar, 
    prove that the ratio of Stacy's berries to Steve's berries is 4:1 -/
theorem berry_ratio (total berries_stacy berries_steve berries_sylar : ℕ) :
  total = 1100 →
  berries_stacy = 800 →
  berries_steve = 2 * berries_sylar →
  total = berries_stacy + berries_steve + berries_sylar →
  berries_stacy / berries_steve = 4 := by
  sorry

#check berry_ratio

end NUMINAMATH_CALUDE_berry_ratio_l389_38975


namespace NUMINAMATH_CALUDE_solution_to_equation_l389_38965

theorem solution_to_equation (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h_eq : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
          2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) :
  x = (3 + Real.sqrt 13) / 2 ∧ 
  y = (3 + Real.sqrt 13) / 2 ∧ 
  z = (3 + Real.sqrt 13) / 2 := by
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l389_38965


namespace NUMINAMATH_CALUDE_monday_bonnets_count_l389_38930

/-- Represents the number of bonnets made on each day of the week --/
structure BonnetProduction where
  monday : ℕ
  tuesday_wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the total number of bonnets produced --/
def total_bonnets (bp : BonnetProduction) : ℕ :=
  bp.monday + bp.tuesday_wednesday + bp.thursday + bp.friday

theorem monday_bonnets_count :
  ∃ (bp : BonnetProduction),
    bp.tuesday_wednesday = 2 * bp.monday ∧
    bp.thursday = bp.monday + 5 ∧
    bp.friday = bp.thursday - 5 ∧
    total_bonnets bp = 11 * 5 ∧
    bp.monday = 10 := by
  sorry

end NUMINAMATH_CALUDE_monday_bonnets_count_l389_38930


namespace NUMINAMATH_CALUDE_alien_sequence_valid_l389_38992

/-- Represents a symbol in the alien sequence -/
inductive AlienSymbol
| percent
| exclamation
| ampersand
| plus
| zero

/-- Represents the possible operations -/
inductive Operation
| addition
| subtraction
| multiplication
| division
| exponentiation

/-- Represents a mapping of symbols to digits or operations -/
structure SymbolMapping where
  base : ℕ
  digit_map : AlienSymbol → Fin base
  operation : AlienSymbol → Option Operation
  equality : AlienSymbol

/-- Converts a list of alien symbols to a natural number given a symbol mapping -/
def alien_to_nat (mapping : SymbolMapping) (symbols : List AlienSymbol) : ℕ := sorry

/-- Checks if a list of alien symbols represents a valid equation given a symbol mapping -/
def is_valid_equation (mapping : SymbolMapping) (symbols : List AlienSymbol) : Prop := sorry

/-- The alien sequence -/
def alien_sequence : List AlienSymbol :=
  [AlienSymbol.percent, AlienSymbol.exclamation, AlienSymbol.ampersand,
   AlienSymbol.plus, AlienSymbol.exclamation, AlienSymbol.zero,
   AlienSymbol.plus, AlienSymbol.plus, AlienSymbol.exclamation,
   AlienSymbol.exclamation, AlienSymbol.exclamation]

theorem alien_sequence_valid :
  ∃ (mapping : SymbolMapping), is_valid_equation mapping alien_sequence := by
  sorry

#check alien_sequence_valid

end NUMINAMATH_CALUDE_alien_sequence_valid_l389_38992


namespace NUMINAMATH_CALUDE_fruit_distribution_ways_l389_38961

/-- The number of ways to distribute n indistinguishable items into k distinguishable bins -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of fruits to buy -/
def total_fruits : ℕ := 17

/-- The number of types of fruit -/
def fruit_types : ℕ := 5

/-- The number of fruits remaining after placing one in each type -/
def remaining_fruits : ℕ := total_fruits - fruit_types

theorem fruit_distribution_ways :
  distribute remaining_fruits fruit_types = 1820 :=
sorry

end NUMINAMATH_CALUDE_fruit_distribution_ways_l389_38961


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l389_38935

theorem absolute_value_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l389_38935


namespace NUMINAMATH_CALUDE_coffee_break_probabilities_l389_38904

/-- Represents the state of knowledge among scientists -/
structure ScientistGroup where
  total : Nat
  initial_knowers : Nat
  
/-- Represents the outcome after the coffee break -/
structure CoffeeBreakOutcome where
  final_knowers : Nat

/-- Probability of a specific outcome after the coffee break -/
def probability_of_outcome (group : ScientistGroup) (outcome : CoffeeBreakOutcome) : ℚ :=
  sorry

/-- Expected number of scientists who know the news after the coffee break -/
def expected_final_knowers (group : ScientistGroup) : ℚ :=
  sorry

theorem coffee_break_probabilities (group : ScientistGroup) 
  (h1 : group.total = 18) 
  (h2 : group.initial_knowers = 10) : 
  probability_of_outcome group ⟨13⟩ = 0 ∧ 
  probability_of_outcome group ⟨14⟩ = 1120 / 2431 ∧
  expected_final_knowers group = 14 + 12 / 17 :=
  sorry

end NUMINAMATH_CALUDE_coffee_break_probabilities_l389_38904


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l389_38998

theorem arithmetic_sequence_proof :
  ∀ (a n d : ℤ),
  a = -7 →
  n = 3 →
  d = 5 →
  (a + (n - 1) * d = n) ∧
  (n * (2 * a + (n - 1) * d) / 2 = -6) :=
λ a n d h1 h2 h3 =>
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l389_38998


namespace NUMINAMATH_CALUDE_poster_distance_is_18cm_l389_38909

/-- The number of posters -/
def num_posters : ℕ := 8

/-- The width of each poster in centimeters -/
def poster_width : ℝ := 29.05

/-- The width of the wall in meters -/
def wall_width_m : ℝ := 3.944

/-- The width of the wall in centimeters -/
def wall_width_cm : ℝ := wall_width_m * 100

/-- The number of gaps between posters and wall ends -/
def num_gaps : ℕ := num_posters + 1

/-- The theorem stating that the distance between posters is 18 cm -/
theorem poster_distance_is_18cm : 
  (wall_width_cm - num_posters * poster_width) / num_gaps = 18 := by
  sorry

end NUMINAMATH_CALUDE_poster_distance_is_18cm_l389_38909


namespace NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l389_38971

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8/17)) = 15/17 := by
  sorry

end NUMINAMATH_CALUDE_cos_arcsin_eight_seventeenths_l389_38971


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l389_38949

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l389_38949


namespace NUMINAMATH_CALUDE_double_frosted_cubes_count_l389_38951

/-- Represents a cube with dimensions n × n × n -/
structure Cube (n : ℕ) where
  size : ℕ := n

/-- Represents a cake with frosting on top and sides, but not on bottom -/
structure FrostedCake (n : ℕ) extends Cube n where
  frosted_top : Bool := true
  frosted_sides : Bool := true
  frosted_bottom : Bool := false

/-- Counts the number of 1×1×1 cubes with exactly two frosted faces in a FrostedCake -/
def count_double_frosted_cubes (cake : FrostedCake 4) : ℕ :=
  sorry

theorem double_frosted_cubes_count :
  ∀ (cake : FrostedCake 4), count_double_frosted_cubes cake = 20 :=
by sorry

end NUMINAMATH_CALUDE_double_frosted_cubes_count_l389_38951


namespace NUMINAMATH_CALUDE_school_track_length_l389_38906

/-- Given that 200 steps correspond to 100 meters and 800 steps were walked along a track,
    the length of the track is 400 meters. -/
theorem school_track_length (steps_per_hundred_meters : ℕ) (track_steps : ℕ) : 
  steps_per_hundred_meters = 200 →
  track_steps = 800 →
  (100 : ℝ) / steps_per_hundred_meters * track_steps = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_track_length_l389_38906


namespace NUMINAMATH_CALUDE_trig_identity_l389_38983

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin ((5*π)/6 - x) + (Real.cos ((π/3) - x))^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l389_38983


namespace NUMINAMATH_CALUDE_consecutive_square_roots_l389_38988

theorem consecutive_square_roots (x : ℝ) (n : ℕ) :
  (∃ m : ℕ, n = m ∧ x^2 = m) →
  Real.sqrt ((n + 1 : ℝ)) = Real.sqrt (x^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_consecutive_square_roots_l389_38988


namespace NUMINAMATH_CALUDE_fraction_equality_l389_38956

theorem fraction_equality (p q r s : ℚ) 
  (h1 : p / q = 2)
  (h2 : q / r = 4 / 5)
  (h3 : r / s = 3) :
  s / p = 5 / 24 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l389_38956


namespace NUMINAMATH_CALUDE_prob_two_gray_rabbits_l389_38960

/-- The probability of selecting 2 gray rabbits out of a group of 5 rabbits, 
    where 3 are gray and 2 are white, given that each rabbit has an equal 
    chance of being selected. -/
theorem prob_two_gray_rabbits (total : Nat) (gray : Nat) (white : Nat) 
    (h1 : total = gray + white) 
    (h2 : total = 5) 
    (h3 : gray = 3) 
    (h4 : white = 2) : 
  (Nat.choose gray 2 : ℚ) / (Nat.choose total 2) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_gray_rabbits_l389_38960


namespace NUMINAMATH_CALUDE_machine_production_time_l389_38973

/-- Given a machine that produces 150 items in 2 hours, 
    prove that it takes 0.8 minutes to produce one item. -/
theorem machine_production_time : 
  let total_items : ℕ := 150
  let total_hours : ℝ := 2
  let minutes_per_hour : ℝ := 60
  let total_minutes : ℝ := total_hours * minutes_per_hour
  total_minutes / total_items = 0.8 := by sorry

end NUMINAMATH_CALUDE_machine_production_time_l389_38973


namespace NUMINAMATH_CALUDE_modular_inverse_of_seven_mod_2003_l389_38945

theorem modular_inverse_of_seven_mod_2003 : ∃ x : ℕ, x < 2003 ∧ (7 * x) % 2003 = 1 :=
by
  use 1717
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_seven_mod_2003_l389_38945


namespace NUMINAMATH_CALUDE_min_x_coord_midpoint_l389_38959

/-- Given a segment AB of length 3 with endpoints on the parabola y^2 = x,
    the minimum x-coordinate of the midpoint M of AB is 5/4 -/
theorem min_x_coord_midpoint (A B M : ℝ × ℝ) :
  (A.2^2 = A.1) →  -- A is on the parabola y^2 = x
  (B.2^2 = B.1) →  -- B is on the parabola y^2 = x
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 9 →  -- AB has length 3
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →  -- M is the midpoint of AB
  M.1 ≥ 5/4 :=
sorry

end NUMINAMATH_CALUDE_min_x_coord_midpoint_l389_38959


namespace NUMINAMATH_CALUDE_square_less_than_triple_l389_38979

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l389_38979


namespace NUMINAMATH_CALUDE_prism_volume_l389_38922

theorem prism_volume (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a * b = 100 → a * h = 50 → b * h = 40 → h = 10 →
  a * b * h = 200 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l389_38922


namespace NUMINAMATH_CALUDE_problem_1_l389_38918

theorem problem_1 (m n : ℝ) (h1 : m = 2) (h2 : n = 1) : 
  (2*m^2 - 3*m*n + 8) - (5*m*n - 4*m^2 + 8) = 8 := by sorry

end NUMINAMATH_CALUDE_problem_1_l389_38918


namespace NUMINAMATH_CALUDE_company_employees_l389_38954

theorem company_employees (december_employees : ℕ) (increase_percentage : ℚ) :
  december_employees = 470 →
  increase_percentage = 15 / 100 →
  ∃ (january_employees : ℕ),
    (january_employees : ℚ) * (1 + increase_percentage) = december_employees ∧
    january_employees = 409 :=
by sorry

end NUMINAMATH_CALUDE_company_employees_l389_38954


namespace NUMINAMATH_CALUDE_ticket_distribution_count_l389_38937

/-- The number of ways to distribute 5 consecutive movie tickets to 5 people. -/
def distribute_tickets : ℕ :=
  /- Number of ways to group tickets -/ 4 *
  /- Number of ways to order A and B -/ 2 *
  /- Number of ways to permute remaining tickets -/ 6

/-- Theorem stating that there are 48 ways to distribute the tickets. -/
theorem ticket_distribution_count :
  distribute_tickets = 48 := by
  sorry

end NUMINAMATH_CALUDE_ticket_distribution_count_l389_38937


namespace NUMINAMATH_CALUDE_total_stars_is_580_l389_38923

/-- The number of stars needed to fill all bottles Kyle bought -/
def total_stars : ℕ :=
  let type_a_initial := 3
  let type_a_later := 5
  let type_b := 4
  let type_c := 2
  let capacity_a := 30
  let capacity_b := 50
  let capacity_c := 70
  (type_a_initial + type_a_later) * capacity_a + type_b * capacity_b + type_c * capacity_c

theorem total_stars_is_580 : total_stars = 580 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_is_580_l389_38923


namespace NUMINAMATH_CALUDE_range_of_c_l389_38910

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^y < c^x

def q (c : ℝ) : Prop := ∀ x : ℝ, x^2 + x + (1/2) * c > 0

theorem range_of_c (c : ℝ) (h_c : c > 0) 
  (h_or : p c ∨ q c) (h_not_and : ¬(p c ∧ q c)) : 
  c ∈ Set.Ioc 0 (1/2) ∪ Set.Ici 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l389_38910


namespace NUMINAMATH_CALUDE_unique_solution_for_digit_sum_equation_l389_38968

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem stating that 402 is the only solution to n(S(n) - 1) = 2010 -/
theorem unique_solution_for_digit_sum_equation :
  ∀ n : ℕ, n > 0 → (n * (S n - 1) = 2010) ↔ n = 402 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_digit_sum_equation_l389_38968


namespace NUMINAMATH_CALUDE_circle_inscribed_angles_sum_l389_38920

theorem circle_inscribed_angles_sum (n : ℕ) (x y : ℝ) : 
  n = 18 →
  x = 3 * (360 / n) / 2 →
  y = 5 * (360 / n) / 2 →
  x + y = 80 := by
  sorry

end NUMINAMATH_CALUDE_circle_inscribed_angles_sum_l389_38920


namespace NUMINAMATH_CALUDE_find_r_l389_38963

theorem find_r (m : ℝ) (r : ℝ) 
  (h1 : 5 = m * 3^r) 
  (h2 : 45 = m * 9^(2*r)) : 
  r = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_find_r_l389_38963


namespace NUMINAMATH_CALUDE_player_A_wins_l389_38989

/-- Represents a game state with three piles of matches -/
structure GameState where
  pile1 : Nat
  pile2 : Nat
  pile3 : Nat

/-- Represents a player in the game -/
inductive Player
  | A
  | B

/-- Defines a valid move in the game -/
def ValidMove (state : GameState) (newState : GameState) : Prop :=
  ∃ (i j : Fin 3) (k : Nat),
    i ≠ j ∧
    k > 0 ∧
    k < state.pile1 + state.pile2 + state.pile3 ∧
    newState.pile1 + newState.pile2 + newState.pile3 = state.pile1 + state.pile2 + state.pile3 - k

/-- Defines the winning condition for a player -/
def Wins (player : Player) (initialState : GameState) : Prop :=
  ∀ (state : GameState),
    state = initialState →
    ∃ (strategy : GameState → GameState),
      (∀ (s : GameState), ValidMove s (strategy s)) ∧
      (∀ (opponent : Player → GameState → GameState),
        (∀ (s : GameState), ValidMove s (opponent player s)) →
        ∃ (n : Nat), ¬ValidMove (Nat.iterate (λ s => opponent player (strategy s)) n initialState) (opponent player (Nat.iterate (λ s => opponent player (strategy s)) n initialState)))

/-- The main theorem stating that Player A has a winning strategy -/
theorem player_A_wins :
  Wins Player.A ⟨100, 200, 300⟩ := by
  sorry


end NUMINAMATH_CALUDE_player_A_wins_l389_38989


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l389_38924

theorem sum_of_reciprocals_lower_bound (a b : ℝ) (h1 : a * b > 0) (h2 : a + b = 1) : 
  1 / a + 1 / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_lower_bound_l389_38924


namespace NUMINAMATH_CALUDE_triangle_side_length_l389_38943

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 10 →
  c = 3 →
  Real.cos A = 1/4 →
  b^2 + c^2 - a^2 = 2 * b * c * Real.cos A →
  b = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l389_38943


namespace NUMINAMATH_CALUDE_union_complement_equality_l389_38987

open Set

def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 4}
def N : Finset ℕ := {2, 5}

theorem union_complement_equality : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_complement_equality_l389_38987


namespace NUMINAMATH_CALUDE_smallest_value_between_zero_and_one_l389_38927

theorem smallest_value_between_zero_and_one (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  x^2 < x ∧ x^2 < Real.sqrt x ∧ x^2 < 3*x ∧ x^2 < 1/x := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_between_zero_and_one_l389_38927


namespace NUMINAMATH_CALUDE_alice_bushes_theorem_l389_38946

/-- The number of bushes Alice needs to buy for her yard -/
def bushes_needed (sides : ℕ) (side_length : ℕ) (bush_length : ℕ) : ℕ :=
  (sides * side_length) / bush_length

theorem alice_bushes_theorem :
  bushes_needed 3 16 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_bushes_theorem_l389_38946


namespace NUMINAMATH_CALUDE_triangle_angle_c_60_degrees_l389_38938

theorem triangle_angle_c_60_degrees
  (A B C : Real)
  (triangle_sum : A + B + C = Real.pi)
  (tan_condition : Real.tan A + Real.tan B + Real.sqrt 3 = Real.sqrt 3 * Real.tan A * Real.tan B) :
  C = Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_c_60_degrees_l389_38938


namespace NUMINAMATH_CALUDE_f_even_implies_a_zero_f_min_value_when_a_zero_f_never_odd_l389_38934

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- Part I: If f is even, then a = 0
theorem f_even_implies_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) → a = 0 :=
sorry

-- Part II: When a = 0, the minimum value of f is 1
theorem f_min_value_when_a_zero :
  ∀ x, f 0 x ≥ 1 :=
sorry

-- Part III: f can never be an odd function for any real a
theorem f_never_odd (a : ℝ) :
  ¬(∀ x, f a x = -(f a (-x))) :=
sorry

end NUMINAMATH_CALUDE_f_even_implies_a_zero_f_min_value_when_a_zero_f_never_odd_l389_38934


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l389_38929

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x < 3} = Set.Ioo (-3/2) (3/2) := by sorry

-- Part 2: Range of a for which f(x) ≥ 3 for all x
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 3) ↔ (a ≥ 2 ∨ a ≤ -4) := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l389_38929


namespace NUMINAMATH_CALUDE_ace_of_hearts_or_diamonds_probability_l389_38999

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)

/-- Definition of a standard deck -/
def standardDeck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4 }

/-- Number of Aces of ♥ or ♦ in a standard deck -/
def redAces : Nat := 2

/-- Probability of drawing a specific card from a randomly arranged deck -/
def drawProbability (d : Deck) (favorableOutcomes : Nat) : Rat :=
  favorableOutcomes / d.cards

/-- Theorem: Probability of drawing an Ace of ♥ or ♦ from a standard deck is 1/26 -/
theorem ace_of_hearts_or_diamonds_probability :
  drawProbability standardDeck redAces = 1 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ace_of_hearts_or_diamonds_probability_l389_38999


namespace NUMINAMATH_CALUDE_candies_per_block_l389_38933

theorem candies_per_block (candies_per_house : ℕ) (houses_per_block : ℕ) : 
  candies_per_house = 7 → houses_per_block = 5 → candies_per_house * houses_per_block = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_candies_per_block_l389_38933


namespace NUMINAMATH_CALUDE_president_vp_selection_ways_l389_38926

/-- Represents the composition of a club -/
structure ClubComposition where
  total_members : Nat
  boys : Nat
  girls : Nat
  senior_boys : Nat
  senior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_president_and_vp (club : ClubComposition) : Nat :=
  let boy_pres_girl_vp := club.senior_boys * club.girls
  let girl_pres_boy_vp := club.senior_girls * (club.boys - club.senior_boys)
  boy_pres_girl_vp + girl_pres_boy_vp

/-- Theorem stating the number of ways to choose a president and vice-president -/
theorem president_vp_selection_ways (club : ClubComposition) 
  (h1 : club.total_members = 24)
  (h2 : club.boys = 8)
  (h3 : club.girls = 16)
  (h4 : club.senior_boys = 2)
  (h5 : club.senior_girls = 2)
  (h6 : club.senior_boys + club.senior_girls = 4)
  (h7 : club.boys + club.girls = club.total_members) :
  choose_president_and_vp club = 44 := by
  sorry

end NUMINAMATH_CALUDE_president_vp_selection_ways_l389_38926


namespace NUMINAMATH_CALUDE_unique_grid_solution_l389_38972

-- Define the grid type
def Grid := List (List Nat)

-- Define the visibility type
def Visibility := List Nat

-- Function to check if a grid is valid
def is_valid_grid (g : Grid) : Prop := sorry

-- Function to check if visibility conditions are met
def meets_visibility (g : Grid) (v : Visibility) : Prop := sorry

-- Function to extract the four-digit number from the grid
def extract_number (g : Grid) : Nat := sorry

-- Theorem statement
theorem unique_grid_solution :
  ∀ (g : Grid) (v : Visibility),
    is_valid_grid g ∧ meets_visibility g v →
    extract_number g = 2213 := by sorry

end NUMINAMATH_CALUDE_unique_grid_solution_l389_38972


namespace NUMINAMATH_CALUDE_glass_mass_problem_l389_38925

theorem glass_mass_problem (full_mass : ℝ) (half_removed_mass : ℝ) 
  (h1 : full_mass = 1000)
  (h2 : half_removed_mass = 700) : 
  full_mass - 2 * (full_mass - half_removed_mass) = 400 := by
  sorry

end NUMINAMATH_CALUDE_glass_mass_problem_l389_38925


namespace NUMINAMATH_CALUDE_money_distribution_l389_38919

/-- Given three people A, B, and C with a total amount of money,
    prove that B and C together have a specific amount. -/
theorem money_distribution (total A_C C B_C : ℕ) : 
  total = 1000 →
  A_C = 700 →
  C = 300 →
  B_C = total - (A_C - C) →
  B_C = 600 := by
  sorry

#check money_distribution

end NUMINAMATH_CALUDE_money_distribution_l389_38919


namespace NUMINAMATH_CALUDE_hannahs_purchase_cost_l389_38915

/-- The total cost of purchasing sweatshirts and T-shirts -/
def total_cost (num_sweatshirts num_tshirts sweatshirt_price tshirt_price : ℕ) : ℕ :=
  num_sweatshirts * sweatshirt_price + num_tshirts * tshirt_price

/-- Theorem stating that the total cost of 3 sweatshirts at $15 each and 2 T-shirts at $10 each is $65 -/
theorem hannahs_purchase_cost :
  total_cost 3 2 15 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_purchase_cost_l389_38915


namespace NUMINAMATH_CALUDE_billy_video_count_l389_38914

/-- The number of videos suggested in each round -/
def suggestions_per_round : ℕ := 15

/-- The number of rounds Billy goes through without liking any videos -/
def unsuccessful_rounds : ℕ := 5

/-- The position of the video Billy watches in the final round -/
def final_video_position : ℕ := 5

/-- The total number of videos Billy watches -/
def total_videos_watched : ℕ := suggestions_per_round * unsuccessful_rounds + 1

theorem billy_video_count :
  total_videos_watched = 76 :=
sorry

end NUMINAMATH_CALUDE_billy_video_count_l389_38914


namespace NUMINAMATH_CALUDE_function_range_theorem_l389_38976

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_range_theorem (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ≠ 0, is_odd_function f) →
  (∀ x ≠ 0, f (x + 5/2) * f x = 1) →
  f (-1) > 1 →
  f 2016 = (a + 3) / (a - 3) →
  0 < a ∧ a < 3 := by sorry

end NUMINAMATH_CALUDE_function_range_theorem_l389_38976


namespace NUMINAMATH_CALUDE_exactly_three_combinations_l389_38962

/-- Represents the number of games played -/
def total_games : ℕ := 15

/-- Represents the total points scored -/
def total_points : ℕ := 33

/-- Represents the points earned for a win -/
def win_points : ℕ := 3

/-- Represents the points earned for a draw -/
def draw_points : ℕ := 1

/-- Represents the points earned for a loss -/
def loss_points : ℕ := 0

/-- A combination of wins, draws, and losses -/
structure GameCombination where
  wins : ℕ
  draws : ℕ
  losses : ℕ

/-- Checks if a combination is valid according to the given conditions -/
def is_valid_combination (c : GameCombination) : Prop :=
  c.wins + c.draws + c.losses = total_games ∧
  c.wins * win_points + c.draws * draw_points + c.losses * loss_points = total_points

/-- The theorem to be proved -/
theorem exactly_three_combinations :
  ∃! (combinations : List GameCombination),
    (∀ c ∈ combinations, is_valid_combination c) ∧
    combinations.length = 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_combinations_l389_38962


namespace NUMINAMATH_CALUDE_camping_probability_l389_38953

theorem camping_probability (p_rain p_tents_on_time : ℝ) : 
  p_rain = 1 / 2 →
  p_tents_on_time = 1 / 2 →
  (p_rain * (1 - p_tents_on_time)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_camping_probability_l389_38953


namespace NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l389_38948

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The sum of the units digits of 734^99 and 347^83 is 7 -/
theorem sum_units_digits_734_99_347_83 : 
  (unitsDigit (734^99) + unitsDigit (347^83)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_units_digits_734_99_347_83_l389_38948


namespace NUMINAMATH_CALUDE_sin_translation_l389_38912

/-- Given a function f obtained by translating the graph of y = sin 2x
    1 unit left and 1 unit upward, prove that f(x) = sin(2x+2)+1 for all real x. -/
theorem sin_translation (f : ℝ → ℝ) 
  (h : ∀ x, f x = (fun y ↦ Real.sin (2 * y)) (x + 1) + 1) :
  ∀ x, f x = Real.sin (2 * x + 2) + 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_translation_l389_38912


namespace NUMINAMATH_CALUDE_common_tangents_exist_l389_38966

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a line is a common tangent to two circles -/
def isCommonTangent (l : Line) (c1 c2 : Circle) : Prop := 
  isTangent l c1 ∧ isTangent l c2

/-- The line connecting the centers of two circles -/
def centerLine (c1 c2 : Circle) : Line := sorry

/-- Checks if a line intersects another line -/
def intersects (l1 l2 : Line) : Prop := sorry

/-- Theorem: For any two circles, there exist common tangents in two cases -/
theorem common_tangents_exist (c1 c2 : Circle) : 
  ∃ (l1 l2 : Line), 
    (isCommonTangent l1 c1 c2 ∧ ¬intersects l1 (centerLine c1 c2)) ∧
    (isCommonTangent l2 c1 c2 ∧ intersects l2 (centerLine c1 c2)) := by
  sorry

end NUMINAMATH_CALUDE_common_tangents_exist_l389_38966


namespace NUMINAMATH_CALUDE_number_of_dolls_l389_38940

theorem number_of_dolls (total_toys : ℕ) (action_figure_percentage : ℚ) (number_of_dolls : ℕ) : 
  total_toys = 120 →
  action_figure_percentage = 35 / 100 →
  number_of_dolls = total_toys - (action_figure_percentage * total_toys).floor →
  number_of_dolls = 78 := by
  sorry

end NUMINAMATH_CALUDE_number_of_dolls_l389_38940


namespace NUMINAMATH_CALUDE_voldemort_calorie_limit_l389_38941

/-- Voldemort's daily calorie intake limit -/
def daily_calorie_limit : ℕ := by sorry

/-- Calories from breakfast -/
def breakfast_calories : ℕ := 560

/-- Calories from lunch -/
def lunch_calories : ℕ := 780

/-- Calories from dinner -/
def dinner_calories : ℕ := 110 + 310 + 215

/-- Remaining calories Voldemort can still take -/
def remaining_calories : ℕ := 525

/-- Theorem stating Voldemort's daily calorie intake limit -/
theorem voldemort_calorie_limit :
  daily_calorie_limit = breakfast_calories + lunch_calories + dinner_calories + remaining_calories := by
  sorry

end NUMINAMATH_CALUDE_voldemort_calorie_limit_l389_38941


namespace NUMINAMATH_CALUDE_divisibility_condition_l389_38990

theorem divisibility_condition (m n : ℕ) :
  (1 + (m + n) * m) ∣ ((n + 1) * (m + n) - 1) ↔ (m = 0 ∨ m = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l389_38990


namespace NUMINAMATH_CALUDE_min_value_a_l389_38947

theorem min_value_a (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → (x + y) * (1 / x + a / y) ≥ 16) →
  a ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l389_38947


namespace NUMINAMATH_CALUDE_inverse_of_three_mod_forty_l389_38985

theorem inverse_of_three_mod_forty :
  ∃ x : ℕ, x < 40 ∧ (3 * x) % 40 = 1 :=
by
  use 27
  sorry

end NUMINAMATH_CALUDE_inverse_of_three_mod_forty_l389_38985


namespace NUMINAMATH_CALUDE_inverse_proportion_l389_38952

/-- Given that p and q are inversely proportional, prove that if p = 30 when q = 4, 
    then p = 240/11 when q = 5.5 -/
theorem inverse_proportion (p q : ℝ) (h : ∃ k : ℝ, ∀ x y : ℝ, p * q = k) 
    (h1 : p = 30 ∧ q = 4) : 
    (p = 240/11 ∧ q = 5.5) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l389_38952


namespace NUMINAMATH_CALUDE_spider_movement_limit_l389_38974

/-- Represents the spider's position and movement on the wall --/
structure SpiderPosition :=
  (height : ℝ)  -- Current height of the spider
  (day : ℕ)     -- Current day

/-- Defines the daily movement of the spider --/
def daily_movement (sp : SpiderPosition) : SpiderPosition :=
  ⟨sp.height + 2, sp.day + 1⟩

/-- Checks if the spider can be moved up 3 feet --/
def can_move_up (sp : SpiderPosition) (wall_height : ℝ) : Prop :=
  sp.height + 3 ≤ wall_height

/-- Theorem: Tony runs out of room after 8 days --/
theorem spider_movement_limit :
  ∀ (wall_height : ℝ) (initial_height : ℝ),
  wall_height = 18 → initial_height = 3 →
  ∃ (n : ℕ), n = 8 ∧
  ¬(can_move_up (n.iterate daily_movement ⟨initial_height, 0⟩) wall_height) ∧
  ∀ (m : ℕ), m < n →
  can_move_up (m.iterate daily_movement ⟨initial_height, 0⟩) wall_height :=
by sorry

end NUMINAMATH_CALUDE_spider_movement_limit_l389_38974


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l389_38982

/-- Two perpendicular lines with a given perpendicular foot -/
structure PerpendicularLines where
  a : ℝ
  b : ℝ
  c : ℝ
  line1 : ∀ x y : ℝ, a * x + 4 * y - 2 = 0
  line2 : ∀ x y : ℝ, 2 * x - 5 * y + b = 0
  perpendicular : (a / 4) * (2 / 5) = -1
  foot_on_line1 : a * 1 + 4 * c - 2 = 0
  foot_on_line2 : 2 * 1 - 5 * c + b = 0

/-- The sum of a, b, and c for perpendicular lines with given conditions is -4 -/
theorem perpendicular_lines_sum (l : PerpendicularLines) : l.a + l.b + l.c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l389_38982


namespace NUMINAMATH_CALUDE_least_sum_problem_l389_38916

theorem least_sum_problem (x y z : ℕ+) 
  (h1 : 4 * x.val = 6 * z.val)
  (h2 : ∀ (a b c : ℕ+), 4 * a.val = 6 * c.val → a.val + b.val + c.val ≥ 37)
  (h3 : x.val + y.val + z.val = 37) : 
  y.val = 32 := by
sorry

end NUMINAMATH_CALUDE_least_sum_problem_l389_38916


namespace NUMINAMATH_CALUDE_circle_intersection_parallelogram_l389_38957

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two circles intersect non-tangentially -/
def nonTangentialIntersection (c1 c2 : Circle) : Prop :=
  sorry

/-- Finds the intersection points of two circles -/
def circleIntersection (c1 c2 : Circle) : Set Point :=
  sorry

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (a b c d : Point) : Prop :=
  sorry

theorem circle_intersection_parallelogram 
  (k1 k2 k3 : Circle)
  (P : Point)
  (A B : Point)
  (D C : Point)
  (h1 : k1.radius = k2.radius ∧ k2.radius = k3.radius)
  (h2 : nonTangentialIntersection k1 k2 ∧ nonTangentialIntersection k2 k3 ∧ nonTangentialIntersection k3 k1)
  (h3 : P ∈ circleIntersection k1 k2 ∩ circleIntersection k2 k3 ∩ circleIntersection k3 k1)
  (h4 : A = k1.center)
  (h5 : B = k2.center)
  (h6 : D ∈ circleIntersection k1 k3 ∧ D ≠ P)
  (h7 : C ∈ circleIntersection k2 k3 ∧ C ≠ P)
  : isParallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_parallelogram_l389_38957


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l389_38917

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l389_38917


namespace NUMINAMATH_CALUDE_chlorine_treatment_capacity_l389_38969

/-- Proves that given a rectangular pool with specified dimensions and chlorine costs,
    one quart of chlorine treats 120 cubic feet of water. -/
theorem chlorine_treatment_capacity
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (chlorine_cost : ℝ) (total_spent : ℝ)
  (h1 : length = 10)
  (h2 : width = 8)
  (h3 : depth = 6)
  (h4 : chlorine_cost = 3)
  (h5 : total_spent = 12) :
  (length * width * depth) / (total_spent / chlorine_cost) = 120 := by
  sorry


end NUMINAMATH_CALUDE_chlorine_treatment_capacity_l389_38969


namespace NUMINAMATH_CALUDE_dragon_population_l389_38977

theorem dragon_population (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 117) 
  (h2 : total_legs = 108) : 
  ∃ (three_headed six_headed : ℕ), 
    three_headed = 15 ∧ 
    six_headed = 12 ∧ 
    3 * three_headed + 6 * six_headed = total_heads ∧ 
    4 * (three_headed + six_headed) = total_legs :=
by sorry

end NUMINAMATH_CALUDE_dragon_population_l389_38977


namespace NUMINAMATH_CALUDE_language_study_difference_l389_38994

def total_students : ℕ := 2500

def german_min : ℕ := 1750
def german_max : ℕ := 1875

def russian_min : ℕ := 625
def russian_max : ℕ := 875

theorem language_study_difference : 
  let m := german_min + russian_min - total_students
  let M := german_max + russian_max - total_students
  M - m = 375 := by sorry

end NUMINAMATH_CALUDE_language_study_difference_l389_38994


namespace NUMINAMATH_CALUDE_foot_of_perpendicular_to_yOz_plane_l389_38900

/-- The foot of a perpendicular from a point to a plane -/
def foot_of_perpendicular (P : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  sorry

/-- The yOz plane in ℝ³ -/
def yOz_plane : Set (ℝ × ℝ × ℝ) :=
  {p | p.1 = 0}

theorem foot_of_perpendicular_to_yOz_plane :
  let P : ℝ × ℝ × ℝ := (1, Real.sqrt 2, Real.sqrt 3)
  let Q := foot_of_perpendicular P yOz_plane
  Q = (0, Real.sqrt 2, Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_foot_of_perpendicular_to_yOz_plane_l389_38900


namespace NUMINAMATH_CALUDE_count_true_props_l389_38984

def original_prop : Prop := ∀ x : ℝ, x^2 > 1 → x > 1

def converse_prop : Prop := ∀ x : ℝ, x > 1 → x^2 > 1

def inverse_prop : Prop := ∀ x : ℝ, x^2 ≤ 1 → x ≤ 1

def contrapositive_prop : Prop := ∀ x : ℝ, x ≤ 1 → x^2 ≤ 1

theorem count_true_props :
  (converse_prop ∧ inverse_prop ∧ ¬contrapositive_prop) ∨
  (converse_prop ∧ ¬inverse_prop ∧ contrapositive_prop) ∨
  (¬converse_prop ∧ inverse_prop ∧ contrapositive_prop) :=
sorry

end NUMINAMATH_CALUDE_count_true_props_l389_38984


namespace NUMINAMATH_CALUDE_sin_two_theta_equals_three_fourths_l389_38928

theorem sin_two_theta_equals_three_fourths (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2)
  (h2 : Real.sin (π * Real.cos θ) = Real.cos (π * Real.sin θ)) : 
  Real.sin (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_two_theta_equals_three_fourths_l389_38928
