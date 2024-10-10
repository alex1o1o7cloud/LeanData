import Mathlib

namespace power_of_fraction_l2800_280003

theorem power_of_fraction : (5 / 3 : ℚ) ^ 3 = 125 / 27 := by sorry

end power_of_fraction_l2800_280003


namespace divisible_by_seven_pair_l2800_280031

theorem divisible_by_seven_pair : ∃! (x y : ℕ), x < 10 ∧ y < 10 ∧
  (1000 + 100 * x + 10 * y + 2) % 7 = 0 ∧
  (1000 * x + 120 + y) % 7 = 0 ∧
  x = 6 ∧ y = 5 := by sorry

end divisible_by_seven_pair_l2800_280031


namespace triathlete_average_speed_l2800_280077

/-- Triathlete's average speed for a multi-segment trip -/
theorem triathlete_average_speed (total_distance : ℝ) 
  (run_flat_speed run_uphill_speed run_downhill_speed swim_speed bike_speed : ℝ)
  (run_flat_distance run_uphill_distance run_downhill_distance swim_distance bike_distance : ℝ)
  (h1 : total_distance = run_flat_distance + run_uphill_distance + run_downhill_distance + swim_distance + bike_distance)
  (h2 : run_flat_speed > 0 ∧ run_uphill_speed > 0 ∧ run_downhill_speed > 0 ∧ swim_speed > 0 ∧ bike_speed > 0)
  (h3 : run_flat_distance > 0 ∧ run_uphill_distance > 0 ∧ run_downhill_distance > 0 ∧ swim_distance > 0 ∧ bike_distance > 0)
  (h4 : total_distance = 9)
  (h5 : run_flat_speed = 10)
  (h6 : run_uphill_speed = 6)
  (h7 : run_downhill_speed = 14)
  (h8 : swim_speed = 4)
  (h9 : bike_speed = 12)
  (h10 : run_flat_distance = 1)
  (h11 : run_uphill_distance = 1)
  (h12 : run_downhill_distance = 1)
  (h13 : swim_distance = 3)
  (h14 : bike_distance = 3) :
  ∃ (average_speed : ℝ), abs (average_speed - 0.1121) < 0.0001 ∧
    average_speed = total_distance / (run_flat_distance / run_flat_speed + 
                                      run_uphill_distance / run_uphill_speed + 
                                      run_downhill_distance / run_downhill_speed + 
                                      swim_distance / swim_speed + 
                                      bike_distance / bike_speed) / 60 :=
by sorry

end triathlete_average_speed_l2800_280077


namespace badminton_cost_equality_l2800_280024

/-- Represents the cost calculation for two stores selling badminton equipment -/
theorem badminton_cost_equality (x : ℝ) : x ≥ 5 → (125 + 5*x = 135 + 4.5*x ↔ x = 20) :=
by
  sorry

#check badminton_cost_equality

end badminton_cost_equality_l2800_280024


namespace count_f_50_eq_18_l2800_280069

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range n)).card + 1

-- Define f₁(n)
def f₁ (n : ℕ) : ℕ := 3 * num_divisors n

-- Define fⱼ(n) recursively
def f (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j + 1 => f₁ (f j n)

-- Define the set of n ≤ 60 for which f₅₀(n) = 18
def set_f_50_eq_18 : Finset ℕ :=
  Finset.filter (λ n => f 50 n = 18) (Finset.range 61)

-- State the theorem
theorem count_f_50_eq_18 : (set_f_50_eq_18.card : ℕ) = 13 := by
  sorry

end count_f_50_eq_18_l2800_280069


namespace lloyd_excess_rate_multiple_l2800_280093

/-- Represents Lloyd's work information --/
structure WorkInfo where
  regularHours : Float
  regularRate : Float
  totalHours : Float
  totalEarnings : Float

/-- Calculates the multiple of regular rate for excess hours --/
def excessRateMultiple (info : WorkInfo) : Float :=
  let regularEarnings := info.regularHours * info.regularRate
  let excessHours := info.totalHours - info.regularHours
  let excessEarnings := info.totalEarnings - regularEarnings
  let excessRate := excessEarnings / excessHours
  excessRate / info.regularRate

/-- Theorem stating that the multiple of regular rate for excess hours is 1.5 --/
theorem lloyd_excess_rate_multiple :
  let lloyd : WorkInfo := {
    regularHours := 7.5,
    regularRate := 3.5,
    totalHours := 10.5,
    totalEarnings := 42
  }
  excessRateMultiple lloyd = 1.5 := by
  sorry


end lloyd_excess_rate_multiple_l2800_280093


namespace elective_course_schemes_l2800_280041

theorem elective_course_schemes (n : ℕ) (k : ℕ) : n = 4 ∧ k = 2 → Nat.choose n k = 6 := by
  sorry

end elective_course_schemes_l2800_280041


namespace intersection_A_B_l2800_280066

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {-1, 0} := by sorry

end intersection_A_B_l2800_280066


namespace partition_existence_l2800_280017

theorem partition_existence (p q : ℕ+) (h_coprime : Nat.Coprime p q) (h_neq : p ≠ q) :
  (∃ (A B C : Set ℕ+),
    (∀ z : ℕ+, (z ∈ A ∧ z + p ∈ B ∧ z + q ∈ C) ∨
               (z ∈ B ∧ z + p ∈ C ∧ z + q ∈ A) ∨
               (z ∈ C ∧ z + p ∈ A ∧ z + q ∈ B)) ∧
    (A ∪ B ∪ C = Set.univ) ∧
    (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (C ∩ A = ∅)) ↔
  (3 ∣ p + q) :=
sorry

end partition_existence_l2800_280017


namespace binomial_coefficient_ratio_l2800_280015

theorem binomial_coefficient_ratio (n : ℕ) : 
  (∃ r : ℕ, r + 2 ≤ n ∧ 
    (n.choose r : ℚ) / (n.choose (r + 1)) = 1 / 2 ∧
    (n.choose (r + 1) : ℚ) / (n.choose (r + 2)) = 2 / 3) → 
  n = 14 := by
sorry

end binomial_coefficient_ratio_l2800_280015


namespace jennys_money_l2800_280048

theorem jennys_money (initial_money : ℚ) : 
  (initial_money * (1 - 3/7) = 24) → 
  (initial_money / 2 = 21) := by
sorry

end jennys_money_l2800_280048


namespace binary_representation_1023_l2800_280056

/-- Represents a binary expansion of a natural number -/
def BinaryExpansion (n : ℕ) : List Bool :=
  sorry

/-- Counts the number of true values in a list of booleans -/
def countOnes (l : List Bool) : ℕ :=
  sorry

/-- Calculates the sum of indices where the value is true -/
def sumIndices (l : List Bool) : ℕ :=
  sorry

theorem binary_representation_1023 :
  let binary := BinaryExpansion 1023
  (sumIndices binary = 45) ∧ (countOnes binary = 10) :=
sorry

end binary_representation_1023_l2800_280056


namespace smallest_perimeter_cross_section_area_is_sqrt_6_l2800_280061

/-- Represents a quadrilateral pyramid with a square base -/
structure QuadPyramid where
  base_side : ℝ
  lateral_height : ℝ

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  pyramid : QuadPyramid
  point_on_base : ℝ  -- Distance from A to the point on AB

/-- The area of the cross-section with the smallest perimeter -/
def smallest_perimeter_cross_section_area (p : QuadPyramid) : ℝ := sorry

/-- The theorem stating that the area of the smallest perimeter cross-section is √6 -/
theorem smallest_perimeter_cross_section_area_is_sqrt_6 
  (p : QuadPyramid) 
  (h1 : p.base_side = 2) 
  (h2 : p.lateral_height = 2) : 
  smallest_perimeter_cross_section_area p = Real.sqrt 6 := by sorry

end smallest_perimeter_cross_section_area_is_sqrt_6_l2800_280061


namespace quadratic_factorability_l2800_280096

theorem quadratic_factorability : ∃ (a b c p q : ℤ),
  (∀ x : ℝ, 3 * (x - 3)^2 = x^2 - 9 ↔ a * x^2 + b * x + c = 0) ∧
  (a * x^2 + b * x + c = (x - p) * (x - q)) :=
sorry

end quadratic_factorability_l2800_280096


namespace line_plane_perpendicularity_l2800_280097

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and subset relations
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (l : Line) (m : Line) (α β : Plane) 
  (h1 : l ≠ m) 
  (h2 : α ≠ β) 
  (h3 : subset l α) 
  (h4 : subset m β) :
  perpendicular l β → plane_perpendicular α β :=
sorry

end line_plane_perpendicularity_l2800_280097


namespace simplify_fraction_l2800_280057

theorem simplify_fraction : 24 * (8 / 15) * (5 / 18) = 32 / 9 := by
  sorry

end simplify_fraction_l2800_280057


namespace domain_of_shifted_function_l2800_280075

-- Define the function f with domain [0,2]
def f : Set ℝ := Set.Icc 0 2

-- Define the function g(x) = f(x+1)
def g (x : ℝ) : Prop := ∃ y ∈ f, y = x + 1

-- Theorem statement
theorem domain_of_shifted_function :
  Set.Icc (-1) 1 = {x | g x} := by sorry

end domain_of_shifted_function_l2800_280075


namespace num_unique_heights_equals_multiples_of_five_l2800_280067

/-- Represents the dimensions of a brick in inches -/
structure BrickDimensions where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the configuration of a tower of bricks -/
def TowerConfiguration := List Nat

/-- The number of bricks in the tower -/
def numBricks : Nat := 80

/-- The dimensions of each brick -/
def brickDimensions : BrickDimensions := { small := 3, medium := 8, large := 18 }

/-- Calculate the height of a tower given its configuration -/
def towerHeight (config : TowerConfiguration) : Nat :=
  config.sum

/-- Generate all possible tower configurations -/
def allConfigurations : List TowerConfiguration :=
  sorry

/-- Calculate the number of unique tower heights -/
def numUniqueHeights : Nat :=
  (allConfigurations.map towerHeight).toFinset.card

/-- The main theorem to prove -/
theorem num_unique_heights_equals_multiples_of_five :
  numUniqueHeights = (((numBricks * brickDimensions.large) - (numBricks * brickDimensions.small)) / 5 + 1) := by
  sorry

end num_unique_heights_equals_multiples_of_five_l2800_280067


namespace decimal_51_to_binary_l2800_280010

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Checks if a list of booleans represents the given binary number -/
def isBinaryRepresentation (bits : List Bool) (binaryStr : String) : Prop :=
  bits.reverse.map (fun b => if b then '1' else '0') = binaryStr.toList

theorem decimal_51_to_binary :
  isBinaryRepresentation (toBinary 51) "110011" := by
  sorry

#eval toBinary 51

end decimal_51_to_binary_l2800_280010


namespace smaller_circle_area_smaller_circle_radius_l2800_280013

/-- Two externally tangent circles with common tangents -/
structure TangentCircles where
  R : ℝ  -- radius of larger circle
  r : ℝ  -- radius of smaller circle
  tangent_length : ℝ  -- length of common tangent segment (PA and AB)
  circles_tangent : R = 2 * r  -- condition for external tangency
  common_tangent : tangent_length = 4  -- given PA = AB = 4

/-- The area of the smaller circle in a TangentCircles configuration is 2π -/
theorem smaller_circle_area (tc : TangentCircles) : 
  Real.pi * tc.r^2 = 2 * Real.pi := by
  sorry

/-- Alternative formulation using Real.sqrt -/
theorem smaller_circle_radius (tc : TangentCircles) :
  tc.r = Real.sqrt 2 := by
  sorry

end smaller_circle_area_smaller_circle_radius_l2800_280013


namespace sum_of_reciprocals_plus_one_l2800_280033

theorem sum_of_reciprocals_plus_one (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2 := by
  sorry

end sum_of_reciprocals_plus_one_l2800_280033


namespace solution_satisfies_system_l2800_280099

theorem solution_satisfies_system : ∃ (a b c : ℝ), 
  (a^3 + 3*a*b^2 + 3*a*c^2 - 6*a*b*c = 1) ∧
  (b^3 + 3*b*a^2 + 3*b*c^2 - 6*a*b*c = 1) ∧
  (c^3 + 3*c*a^2 + 3*c*b^2 - 6*a*b*c = 1) ∧
  (a = 1 ∧ b = 1 ∧ c = 1) := by
sorry

end solution_satisfies_system_l2800_280099


namespace propositions_analysis_l2800_280058

-- Proposition 1
def has_real_roots (q : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + q = 0

-- Proposition 2
def both_zero (x y : ℝ) : Prop := x = 0 ∧ y = 0

theorem propositions_analysis :
  -- Proposition 1
  (¬ (∀ q : ℝ, has_real_roots q → q < 1)) ∧  -- Converse is false
  (∀ q : ℝ, ¬(has_real_roots q) → q ≥ 1) ∧  -- Contrapositive is true
  -- Proposition 2
  (∀ x y : ℝ, both_zero x y → x^2 + y^2 = 0) ∧  -- Converse is true
  (∀ x y : ℝ, ¬(both_zero x y) → x^2 + y^2 ≠ 0)  -- Contrapositive is true
  := by sorry

end propositions_analysis_l2800_280058


namespace negative_operation_l2800_280029

theorem negative_operation (a b c d : ℤ) : a = (-7) * (-6) ∧ b = (-7) - (-15) ∧ c = 0 * (-2) * (-3) ∧ d = (-6) + (-4) → d < 0 := by
  sorry

end negative_operation_l2800_280029


namespace billion_scientific_notation_l2800_280038

/-- Represents 1.2 billion in decimal form -/
def billion : ℝ := 1200000000

/-- Represents 1.2 × 10^8 in scientific notation -/
def scientific_notation : ℝ := 1.2 * (10^8)

/-- Theorem stating that 1.2 billion is equal to 1.2 × 10^8 in scientific notation -/
theorem billion_scientific_notation : billion = scientific_notation := by
  sorry

end billion_scientific_notation_l2800_280038


namespace cubic_symmetry_extrema_l2800_280090

/-- A cubic function that is symmetric about the origin -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + (a - 1) * x^2 + 48 * (b - 3) * x + b

/-- The derivative of f -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * (a - 1) * x + 144

/-- The discriminant of f' = 0 -/
def discriminant (a : ℝ) : ℝ := 4 * (a^2 - 434 * a + 1)

theorem cubic_symmetry_extrema (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- symmetry about the origin
  (∃ x_min, ∀ x, f a b x_min ≤ f a b x) ∧ 
  (∃ x_max, ∀ x, f a b x ≤ f a b x_max) := by
  sorry

end cubic_symmetry_extrema_l2800_280090


namespace jelly_bean_matching_probability_l2800_280043

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.blue + jb.yellow

/-- Abe's jelly bean distribution -/
def abe_jelly_beans : JellyBeans :=
  { green := 2, red := 1, blue := 1, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob_jelly_beans : JellyBeans :=
  { green := 3, red := 2, blue := 1, yellow := 2 }

/-- Calculates the probability of both people showing the same color -/
def matching_probability (jb1 jb2 : JellyBeans) : ℚ :=
  let total1 := jb1.total
  let total2 := jb2.total
  (jb1.green * jb2.green + jb1.red * jb2.red + jb1.blue * jb2.blue) / (total1 * total2)

theorem jelly_bean_matching_probability :
  matching_probability abe_jelly_beans bob_jelly_beans = 9 / 32 := by
  sorry

end jelly_bean_matching_probability_l2800_280043


namespace scientific_notation_10374_billion_l2800_280025

/-- Converts a number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ℝ × ℤ :=
  sorry

/-- Checks if two scientific notation representations are equal -/
def scientificNotationEqual (a : ℝ) (b : ℤ) (c : ℝ) (d : ℤ) : Prop :=
  sorry

theorem scientific_notation_10374_billion :
  let result := toScientificNotation (10374 * 1000000000) 3
  scientificNotationEqual result.1 result.2 1.04 13 := by
  sorry

end scientific_notation_10374_billion_l2800_280025


namespace work_completion_l2800_280086

theorem work_completion (x : ℕ) : 
  (x * 40 = (x - 5) * 60) → x = 15 := by
  sorry

end work_completion_l2800_280086


namespace house_profit_percentage_l2800_280091

/-- Proves that given two houses sold at $10,000 each, with a 10% loss on the second house
    and a 17% net profit overall, the profit percentage on the first house is approximately 67.15%. -/
theorem house_profit_percentage (selling_price : ℝ) (loss_percentage : ℝ) (net_profit_percentage : ℝ) :
  selling_price = 10000 →
  loss_percentage = 0.10 →
  net_profit_percentage = 0.17 →
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 0.6715) < 0.0001 :=
by sorry

end house_profit_percentage_l2800_280091


namespace uniform_probability_diff_colors_l2800_280001

def shorts_colors := Fin 3
def jersey_colors := Fin 3

def total_combinations : ℕ := 9

def matching_combinations : ℕ := 2

theorem uniform_probability_diff_colors :
  (total_combinations - matching_combinations) / total_combinations = 7 / 9 := by
  sorry

end uniform_probability_diff_colors_l2800_280001


namespace point_on_graph_l2800_280050

/-- A linear function passing through (0, -3) with slope 2 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The point (2, 1) lies on the graph of f -/
theorem point_on_graph : f 2 = 1 := by sorry

end point_on_graph_l2800_280050


namespace rectangular_field_area_l2800_280052

theorem rectangular_field_area (width length perimeter area : ℝ) : 
  width = (1/3) * length →
  perimeter = 2 * (width + length) →
  perimeter = 60 →
  area = width * length →
  area = 168.75 := by
sorry

end rectangular_field_area_l2800_280052


namespace log_exponent_sum_l2800_280087

theorem log_exponent_sum (a b : ℝ) (h1 : a = Real.log 25) (h2 : b = Real.log 36) :
  (5 : ℝ) ^ (a / b) + (6 : ℝ) ^ (b / a) = 11 := by
  sorry

end log_exponent_sum_l2800_280087


namespace two_trees_remain_l2800_280011

/-- The number of walnut trees remaining after removal -/
def remaining_trees (initial : ℕ) (removed : ℕ) : ℕ :=
  initial - removed

/-- Theorem stating that 2 trees remain after removing 4 from 6 -/
theorem two_trees_remain :
  remaining_trees 6 4 = 2 := by
  sorry

end two_trees_remain_l2800_280011


namespace max_planes_four_points_l2800_280028

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in three-dimensional space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to determine the number of planes formed by four points -/
def numPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

/-- Theorem stating the maximum number of planes determined by four points -/
theorem max_planes_four_points :
  ∃ (p1 p2 p3 p4 : Point3D), numPlanes p1 p2 p3 p4 = 4 ∧
  ∀ (q1 q2 q3 q4 : Point3D), numPlanes q1 q2 q3 q4 ≤ 4 := by sorry

end max_planes_four_points_l2800_280028


namespace triangle_area_l2800_280016

/-- The area of a triangle with side lengths 15, 36, and 39 is 270 -/
theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) :
  (1/2 : ℝ) * a * b = 270 := by
  sorry

end triangle_area_l2800_280016


namespace monotone_increasing_condition_l2800_280039

/-- If f(x) = kx - ln x is monotonically increasing on (1, +∞), then k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) : 
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) → k ≥ 1 := by
  sorry

end monotone_increasing_condition_l2800_280039


namespace cubic_polynomial_value_at_5_l2800_280059

/-- A cubic polynomial satisfying specific conditions -/
def cubic_polynomial (p : ℝ → ℝ) : Prop :=
  (∃ a b c d : ℝ, ∀ x, p x = a*x^3 + b*x^2 + c*x + d) ∧
  (p 1 = 1) ∧
  (p 2 = 1/8) ∧
  (p 3 = 1/27) ∧
  (p 4 = 1/64)

/-- The main theorem -/
theorem cubic_polynomial_value_at_5 (p : ℝ → ℝ) (h : cubic_polynomial p) :
  p 5 = -76/375 := by
  sorry

end cubic_polynomial_value_at_5_l2800_280059


namespace jackson_charity_collection_l2800_280032

/-- Represents the problem of Jackson's charity collection --/
theorem jackson_charity_collection 
  (total_days : ℕ) 
  (goal : ℕ) 
  (monday_earning : ℕ) 
  (tuesday_earning : ℕ) 
  (houses_per_bundle : ℕ) 
  (earning_per_bundle : ℕ) 
  (h1 : total_days = 5)
  (h2 : goal = 1000)
  (h3 : monday_earning = 300)
  (h4 : tuesday_earning = 40)
  (h5 : houses_per_bundle = 4)
  (h6 : earning_per_bundle = 10) :
  ∃ (houses_per_day : ℕ), 
    houses_per_day = 88 ∧ 
    (goal - monday_earning - tuesday_earning) = 
      (total_days - 2) * houses_per_day * (earning_per_bundle / houses_per_bundle) :=
sorry

end jackson_charity_collection_l2800_280032


namespace total_students_olympiad_l2800_280009

/-- Represents a mathematics teacher at Archimedes Academy -/
inductive Teacher
| Euler
| Fibonacci
| Gauss
| Noether

/-- Returns the number of students taking the Math Olympiad for a given teacher -/
def students_in_class (t : Teacher) : Nat :=
  match t with
  | Teacher.Euler => 15
  | Teacher.Fibonacci => 10
  | Teacher.Gauss => 12
  | Teacher.Noether => 7

/-- The list of all teachers at Archimedes Academy -/
def all_teachers : List Teacher :=
  [Teacher.Euler, Teacher.Fibonacci, Teacher.Gauss, Teacher.Noether]

/-- Theorem stating that the total number of students taking the Math Olympiad is 44 -/
theorem total_students_olympiad :
  (all_teachers.map students_in_class).sum = 44 := by
  sorry

end total_students_olympiad_l2800_280009


namespace sum_of_parts_l2800_280018

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 56) (h2 : y = 37.66666666666667) :
  10 * x + 22 * y = 1012 := by
  sorry

end sum_of_parts_l2800_280018


namespace band_percentage_is_twenty_percent_l2800_280051

-- Define the number of students in the band
def students_in_band : ℕ := 168

-- Define the total number of students
def total_students : ℕ := 840

-- Define the percentage of students in the band
def percentage_in_band : ℚ := (students_in_band : ℚ) / total_students * 100

-- Theorem statement
theorem band_percentage_is_twenty_percent :
  percentage_in_band = 20 := by
  sorry

end band_percentage_is_twenty_percent_l2800_280051


namespace student_b_speed_l2800_280007

theorem student_b_speed (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_difference = 1/6 →
  ∃ (speed_b : ℝ),
    distance / speed_b - time_difference = distance / (speed_ratio * speed_b) ∧
    speed_b = 12 :=
by sorry

end student_b_speed_l2800_280007


namespace vasya_driving_distance_l2800_280023

theorem vasya_driving_distance
  (total_distance : ℝ)
  (anton_distance : ℝ)
  (vasya_distance : ℝ)
  (sasha_distance : ℝ)
  (dima_distance : ℝ)
  (h1 : anton_distance = vasya_distance / 2)
  (h2 : sasha_distance = anton_distance + dima_distance)
  (h3 : dima_distance = total_distance / 10)
  (h4 : anton_distance + vasya_distance + sasha_distance + dima_distance = total_distance) :
  vasya_distance = (2 : ℝ) / 5 * total_distance :=
by sorry

end vasya_driving_distance_l2800_280023


namespace log_problem_l2800_280047

theorem log_problem (x : ℝ) : 
  x = (Real.log 4 / Real.log 16) ^ (Real.log 16 / Real.log 4) →
  Real.log x / Real.log 7 = -2 * Real.log 2 / Real.log 7 := by
sorry

end log_problem_l2800_280047


namespace base_representation_equivalence_l2800_280044

/-- Represents a positive integer in different bases -/
structure BaseRepresentation where
  base8 : ℕ  -- Representation in base 8
  base5 : ℕ  -- Representation in base 5
  base10 : ℕ -- Representation in base 10
  is_valid : base8 ≥ 10 ∧ base8 < 100 ∧ base5 ≥ 10 ∧ base5 < 100

/-- Converts a two-digit number in base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  8 * (n / 10) + (n % 10)

/-- Converts a two-digit number in base 5 to base 10 -/
def base5_to_base10 (n : ℕ) : ℕ :=
  5 * (n % 10) + (n / 10)

/-- Theorem stating the equivalence of the representations -/
theorem base_representation_equivalence (n : BaseRepresentation) : 
  base8_to_base10 n.base8 = base5_to_base10 n.base5 ∧ 
  base8_to_base10 n.base8 = n.base10 ∧ 
  n.base10 = 39 := by
  sorry

#check base_representation_equivalence

end base_representation_equivalence_l2800_280044


namespace price_reduction_equation_l2800_280065

/-- Represents the price reduction scenario for an item -/
structure PriceReduction where
  initial_price : ℝ
  final_price : ℝ
  reduction_percentage : ℝ
  num_reductions : ℕ

/-- Theorem stating the relationship between initial price, final price, and reduction percentage -/
theorem price_reduction_equation (pr : PriceReduction) 
  (h1 : pr.initial_price = 150)
  (h2 : pr.final_price = 96)
  (h3 : pr.num_reductions = 2) :
  pr.initial_price * (1 - pr.reduction_percentage)^pr.num_reductions = pr.final_price := by
  sorry

#check price_reduction_equation

end price_reduction_equation_l2800_280065


namespace arrange_crosses_and_zeros_theorem_l2800_280068

def arrange_crosses_and_zeros (n : ℕ) : ℕ :=
  if n = 27 then 14
  else if n = 26 then 105
  else 0

theorem arrange_crosses_and_zeros_theorem :
  (arrange_crosses_and_zeros 27 = 14) ∧
  (arrange_crosses_and_zeros 26 = 105) :=
sorry

end arrange_crosses_and_zeros_theorem_l2800_280068


namespace fraction_proof_l2800_280030

theorem fraction_proof (N : ℝ) (h1 : N = 150) (h2 : N - (3/5) * N = 60) : (3 : ℝ) / 5 = (3 : ℝ) / 5 := by
  sorry

end fraction_proof_l2800_280030


namespace delores_purchase_shortfall_l2800_280022

def initial_amount : ℚ := 450
def computer_cost : ℚ := 500
def computer_discount_rate : ℚ := 0.2
def printer_cost : ℚ := 50
def printer_tax_rate : ℚ := 0.2

def computer_discount : ℚ := computer_cost * computer_discount_rate
def discounted_computer_cost : ℚ := computer_cost - computer_discount
def printer_tax : ℚ := printer_cost * printer_tax_rate
def total_printer_cost : ℚ := printer_cost + printer_tax
def total_spent : ℚ := discounted_computer_cost + total_printer_cost

theorem delores_purchase_shortfall :
  initial_amount - total_spent = -10 := by sorry

end delores_purchase_shortfall_l2800_280022


namespace no_single_digit_quadratic_solution_l2800_280035

theorem no_single_digit_quadratic_solution :
  ¬∃ (A : ℕ), 1 ≤ A ∧ A ≤ 9 ∧
  ∃ (x : ℕ), x > 0 ∧ x^2 - (2*A)*x + (A^2 + 1) = 0 :=
sorry

end no_single_digit_quadratic_solution_l2800_280035


namespace remainder_sum_l2800_280014

theorem remainder_sum (n : ℤ) : n % 20 = 13 → (n % 4 + n % 5 = 4) := by
  sorry

end remainder_sum_l2800_280014


namespace prob_at_least_two_same_l2800_280088

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of at least two dice showing the same number when rolling four fair 8-sided dice -/
theorem prob_at_least_two_same (num_sides : ℕ) (num_dice : ℕ) : 
  num_sides = 8 → num_dice = 4 → 
  (1 - (num_sides * (num_sides - 1) * (num_sides - 2) * (num_sides - 3)) / (num_sides ^ num_dice)) = 151/256 := by
  sorry

end prob_at_least_two_same_l2800_280088


namespace geometric_series_sum_l2800_280084

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h : r ≠ 1) :
  let S := (a * (1 - r^n)) / (1 - r)
  a = -1 → r = -3 → n = 10 → S = 14762 := by
  sorry

end geometric_series_sum_l2800_280084


namespace plane_ticket_price_is_800_l2800_280089

/-- Represents the luggage and ticket pricing scenario -/
structure LuggagePricing where
  totalWeight : ℕ
  freeAllowance : ℕ
  excessChargeRate : ℚ
  luggageTicketPrice : ℕ

/-- Calculates the plane ticket price based on the given luggage pricing scenario -/
def planeTicketPrice (scenario : LuggagePricing) : ℕ :=
  sorry

/-- Theorem stating that the plane ticket price is 800 yuan for the given scenario -/
theorem plane_ticket_price_is_800 :
  planeTicketPrice ⟨30, 20, 3/200, 120⟩ = 800 :=
sorry

end plane_ticket_price_is_800_l2800_280089


namespace residential_building_capacity_l2800_280081

/-- The number of households that can be accommodated in multiple identical residential buildings. -/
def total_households (floors_per_building : ℕ) (households_per_floor : ℕ) (num_buildings : ℕ) : ℕ :=
  floors_per_building * households_per_floor * num_buildings

/-- Theorem stating that 10 buildings with 16 floors and 12 households per floor can accommodate 1920 households. -/
theorem residential_building_capacity :
  total_households 16 12 10 = 1920 := by
  sorry

end residential_building_capacity_l2800_280081


namespace max_distance_sum_l2800_280073

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the point M
def M : ℝ × ℝ := (6, 4)

-- Statement of the theorem
theorem max_distance_sum (a b : ℝ) (F₁ : ℝ × ℝ) :
  ∃ (max : ℝ), ∀ (P : ℝ × ℝ),
    ellipse a b P.1 P.2 →
    dist P M + dist P F₁ ≤ max ∧
    (∃ (Q : ℝ × ℝ), ellipse a b Q.1 Q.2 ∧ dist Q M + dist Q F₁ = max) ∧
    max = 15 :=
sorry

end max_distance_sum_l2800_280073


namespace second_pedal_triangle_rotation_l2800_280046

/-- Represents a triangle with angles in degrees -/
structure Triangle where
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_180 : angle_a + angle_b + angle_c = 180

/-- Computes the angles of the first pedal triangle -/
def first_pedal_triangle (t : Triangle) : Triangle :=
  { angle_a := 2 * t.angle_a,
    angle_b := 2 * t.angle_b,
    angle_c := 2 * t.angle_c - 180,
    sum_180 := by sorry }

/-- Computes the angles of the second pedal triangle -/
def second_pedal_triangle (t : Triangle) : Triangle :=
  let pt := first_pedal_triangle t
  { angle_a := 180 - 2 * pt.angle_a,
    angle_b := 180 - 2 * pt.angle_b,
    angle_c := 180 - 2 * pt.angle_c,
    sum_180 := by sorry }

/-- Computes the rotation angle between two triangles -/
def rotation_angle (t1 t2 : Triangle) : ℝ :=
  (180 - t1.angle_c) + t2.angle_b

/-- Theorem statement -/
theorem second_pedal_triangle_rotation (t : Triangle)
  (h1 : t.angle_a = 12)
  (h2 : t.angle_b = 36)
  (h3 : t.angle_c = 132) :
  rotation_angle t (second_pedal_triangle t) = 120 := by sorry

end second_pedal_triangle_rotation_l2800_280046


namespace total_wage_calculation_l2800_280000

-- Define the basic parameters
def basic_rate : ℝ := 20
def regular_hours : ℕ := 40
def total_hours : ℕ := 48
def overtime_rate_increase : ℝ := 0.25

-- Define the calculation functions
def regular_pay (rate : ℝ) (hours : ℕ) : ℝ := rate * hours
def overtime_rate (rate : ℝ) (increase : ℝ) : ℝ := rate * (1 + increase)
def overtime_hours (total : ℕ) (regular : ℕ) : ℕ := total - regular
def overtime_pay (rate : ℝ) (hours : ℕ) : ℝ := rate * hours

-- Theorem statement
theorem total_wage_calculation :
  let reg_pay := regular_pay basic_rate regular_hours
  let ot_rate := overtime_rate basic_rate overtime_rate_increase
  let ot_hours := overtime_hours total_hours regular_hours
  let ot_pay := overtime_pay ot_rate ot_hours
  reg_pay + ot_pay = 1000 := by
  sorry

end total_wage_calculation_l2800_280000


namespace set_intersection_example_l2800_280071

theorem set_intersection_example : 
  let A : Set ℕ := {1, 3, 9}
  let B : Set ℕ := {1, 5, 9}
  A ∩ B = {1, 9} := by
sorry

end set_intersection_example_l2800_280071


namespace gcd_lcm_sum_36_495_l2800_280021

theorem gcd_lcm_sum_36_495 : Nat.gcd 36 495 + Nat.lcm 36 495 = 1989 := by
  sorry

end gcd_lcm_sum_36_495_l2800_280021


namespace sum_first_three_terms_l2800_280053

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_first_three_terms
  (a : ℕ → ℤ)
  (h_arithmetic : arithmetic_sequence a)
  (h_fifth : a 5 = 7)
  (h_sixth : a 6 = 12)
  (h_seventh : a 7 = 17) :
  a 1 + a 2 + a 3 = -24 :=
sorry

end sum_first_three_terms_l2800_280053


namespace integer_root_condition_l2800_280095

def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 3*x^2 + a*x + 11 = 0

theorem integer_root_condition (a : ℤ) :
  has_integer_root a ↔ a = -155 ∨ a = -15 ∨ a = 13 ∨ a = 87 :=
sorry

end integer_root_condition_l2800_280095


namespace quadratic_inequality_solution_set_l2800_280064

theorem quadratic_inequality_solution_set (x : ℝ) :
  {x | x^2 - 4*x > 44} = {x | x < -4 ∨ x > 11} :=
by sorry

end quadratic_inequality_solution_set_l2800_280064


namespace nursery_school_students_nursery_school_students_is_50_l2800_280037

/-- The number of students in a nursery school satisfying specific age distribution conditions -/
theorem nursery_school_students : ℕ :=
  let S : ℕ := 50  -- Total number of students
  let four_and_older : ℕ := S / 10  -- Students 4 years old or older
  let younger_than_three : ℕ := 20  -- Students younger than 3 years old
  let not_between_three_and_four : ℕ := 25  -- Students not between 3 and 4 years old
  have h1 : four_and_older = S / 10 := by sorry
  have h2 : younger_than_three = 20 := by sorry
  have h3 : not_between_three_and_four = 25 := by sorry
  have h4 : four_and_older + younger_than_three = not_between_three_and_four := by sorry
  S

/-- Proof that the number of students in the nursery school is 50 -/
theorem nursery_school_students_is_50 : nursery_school_students = 50 := by sorry

end nursery_school_students_nursery_school_students_is_50_l2800_280037


namespace stationery_prices_l2800_280040

-- Define the variables
variable (x : ℝ) -- Price of one notebook
variable (y : ℝ) -- Price of one pen

-- Define the theorem
theorem stationery_prices : 
  (3 * x + 5 * y = 30) ∧ 
  (30 - (3 * x + 5 * y + 2 * y) = -0.4) ∧ 
  (30 - (3 * x + 5 * y + 2 * x) = -2) → 
  (x = 3.6 ∧ y = 2.8) :=
by sorry

end stationery_prices_l2800_280040


namespace function_and_range_proof_l2800_280049

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B (f : ℝ → ℝ) : Set ℝ := {x | 1 < f x ∧ f x < 3}

-- Define the theorem
theorem function_and_range_proof 
  (a b : ℝ) 
  (h_a_nonzero : a ≠ 0)
  (h_f : ∀ x, f x = a * x + b)
  (h_f_condition : ∀ x, f (2 * x + 1) = 4 * x + 1)
  (h_subset : B f ⊆ A a) :
  (∀ x, f x = 2 * x - 1) ∧ (1/2 ≤ a ∧ a ≤ 2) :=
by sorry

end function_and_range_proof_l2800_280049


namespace forty_eggs_not_eaten_l2800_280012

/-- Represents the number of eggs in a problem about weekly egg consumption --/
structure EggProblem where
  trays_per_week : ℕ
  eggs_per_tray : ℕ
  children_eggs_per_day : ℕ
  parents_eggs_per_day : ℕ
  days_per_week : ℕ

/-- Calculates the number of eggs not eaten in a week --/
def eggs_not_eaten (p : EggProblem) : ℕ :=
  p.trays_per_week * p.eggs_per_tray - 
  (p.children_eggs_per_day + p.parents_eggs_per_day) * p.days_per_week

/-- Theorem stating that given the problem conditions, 40 eggs are not eaten in a week --/
theorem forty_eggs_not_eaten (p : EggProblem) 
  (h1 : p.trays_per_week = 2)
  (h2 : p.eggs_per_tray = 24)
  (h3 : p.children_eggs_per_day = 4)
  (h4 : p.parents_eggs_per_day = 4)
  (h5 : p.days_per_week = 7) :
  eggs_not_eaten p = 40 := by
  sorry

end forty_eggs_not_eaten_l2800_280012


namespace two_solutions_l2800_280005

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x * (x - 6) = 7

-- Theorem statement
theorem two_solutions :
  ∃ (a b : ℝ), a ≠ b ∧ 
  quadratic_equation a ∧ 
  quadratic_equation b ∧
  ∀ (c : ℝ), quadratic_equation c → (c = a ∨ c = b) :=
sorry

end two_solutions_l2800_280005


namespace prism_properties_l2800_280082

/-- A right triangular prism with rectangular base ABCD and height DE -/
structure RightTriangularPrism where
  AB : ℝ
  BC : ℝ
  DE : ℝ
  ab_eq_dc : AB = 8
  bc_eq : BC = 15
  de_eq : DE = 7

/-- The perimeter of the base ABCD -/
def basePerimeter (p : RightTriangularPrism) : ℝ :=
  2 * (p.AB + p.BC)

/-- The area of the base ABCD -/
def baseArea (p : RightTriangularPrism) : ℝ :=
  p.AB * p.BC

/-- The volume of the right triangular prism -/
def volume (p : RightTriangularPrism) : ℝ :=
  p.AB * p.BC * p.DE

theorem prism_properties (p : RightTriangularPrism) :
  basePerimeter p = 46 ∧ baseArea p = 120 ∧ volume p = 840 := by
  sorry

end prism_properties_l2800_280082


namespace M_divisible_by_40_l2800_280063

/-- M is the number formed by concatenating integers from 1 to 39 -/
def M : ℕ := sorry

/-- Theorem stating that M is divisible by 40 -/
theorem M_divisible_by_40 : 40 ∣ M := by sorry

end M_divisible_by_40_l2800_280063


namespace abc_product_l2800_280002

theorem abc_product (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * (b + c) = 198)
  (h2 : b * (c + a) = 210)
  (h3 : c * (a + b) = 222) :
  a * b * c = 1069 := by
sorry

end abc_product_l2800_280002


namespace solution_range_l2800_280076

theorem solution_range (x : ℝ) : 
  x ≥ 1 → 
  Real.sqrt (x + 2 - 2 * Real.sqrt (x - 1)) + Real.sqrt (x + 5 - 3 * Real.sqrt (x - 1)) = 2 → 
  2 ≤ x ∧ x ≤ 5 := by sorry

end solution_range_l2800_280076


namespace stadium_length_feet_l2800_280019

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℕ → ℕ := λ x => 3 * x

/-- Length of the stadium in yards -/
def stadium_length_yards : ℕ := 80

/-- Theorem stating that the stadium length in feet is 240 -/
theorem stadium_length_feet : yards_to_feet stadium_length_yards = 240 := by
  sorry

end stadium_length_feet_l2800_280019


namespace lesser_solution_quadratic_l2800_280072

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 24 = 0 ∧ ∀ y, y^2 + 10*y - 24 = 0 → x ≤ y → x = -12 :=
by sorry

end lesser_solution_quadratic_l2800_280072


namespace commodity_trade_fair_companies_l2800_280079

theorem commodity_trade_fair_companies : ∃ (n : ℕ), n > 0 ∧ n * (n - 1) / 2 = 45 := by
  sorry

end commodity_trade_fair_companies_l2800_280079


namespace point_A_outside_circle_l2800_280062

/-- The position of point A on the number line after t seconds -/
def position_A (t : ℝ) : ℝ := 2 * t

/-- The center of circle B on the number line -/
def center_B : ℝ := 16

/-- The radius of circle B -/
def radius_B : ℝ := 4

/-- Predicate for point A being outside circle B -/
def is_outside_circle (t : ℝ) : Prop :=
  position_A t < center_B - radius_B ∨ position_A t > center_B + radius_B

theorem point_A_outside_circle (t : ℝ) :
  is_outside_circle t ↔ t < 6 ∨ t > 10 := by sorry

end point_A_outside_circle_l2800_280062


namespace deck_size_l2800_280055

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2/5 →
  (r : ℚ) / (r + b + 7) = 1/3 →
  r + b = 35 := by
sorry

end deck_size_l2800_280055


namespace volleyballs_left_l2800_280092

theorem volleyballs_left (total : ℕ) (lent : ℕ) (left : ℕ) : 
  total = 9 → lent = 5 → left = total - lent → left = 4 := by sorry

end volleyballs_left_l2800_280092


namespace opposite_of_2022_l2800_280027

-- Define the opposite of an integer
def opposite (n : ℤ) : ℤ := -n

-- Theorem stating that the opposite of 2022 is -2022
theorem opposite_of_2022 : opposite 2022 = -2022 := by
  sorry

end opposite_of_2022_l2800_280027


namespace sum_of_four_numbers_l2800_280034

theorem sum_of_four_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end sum_of_four_numbers_l2800_280034


namespace button_probability_l2800_280020

theorem button_probability (initial_red : ℕ) (initial_blue : ℕ) 
  (removed_red : ℕ) (removed_blue : ℕ) :
  initial_red = 8 →
  initial_blue = 12 →
  removed_red = removed_blue →
  (initial_red + initial_blue - removed_red - removed_blue : ℚ) = 
    (5 / 8 : ℚ) * (initial_red + initial_blue : ℚ) →
  ((initial_red - removed_red : ℚ) / (initial_red + initial_blue - removed_red - removed_blue : ℚ)) *
  (removed_red : ℚ) / (removed_red + removed_blue : ℚ) = 4 / 25 := by
  sorry

end button_probability_l2800_280020


namespace rachelle_gpa_probability_l2800_280070

def grade_points (grade : Char) : ℕ :=
  match grade with
  | 'A' => 5
  | 'B' => 4
  | 'C' => 3
  | 'D' => 2
  | _ => 0

def gpa (total_points : ℕ) : ℚ := total_points / 5

def english_prob (grade : Char) : ℚ :=
  match grade with
  | 'A' => 1 / 7
  | 'B' => 1 / 5
  | 'C' => 1 - 1 / 7 - 1 / 5
  | _ => 0

def history_prob (grade : Char) : ℚ :=
  match grade with
  | 'B' => 1 / 3
  | 'C' => 1 / 6
  | 'D' => 1 - 1 / 3 - 1 / 6
  | _ => 0

theorem rachelle_gpa_probability :
  let assured_points := 3 * grade_points 'A'
  let min_total_points := 20
  let required_points := min_total_points - assured_points
  let prob_a_english := english_prob 'A'
  let prob_b_english := english_prob 'B'
  let prob_b_history := history_prob 'B'
  let prob_c_history := history_prob 'C'
  (prob_a_english + prob_b_english * prob_b_history + prob_b_english * prob_c_history) = 17 / 70 := by
  sorry

end rachelle_gpa_probability_l2800_280070


namespace intersection_on_y_axis_l2800_280094

/-- Given two lines in the xy-plane defined by equations 2x + 3y - k = 0 and x - ky + 12 = 0,
    if their intersection point lies on the y-axis, then k = 6 or k = -6. -/
theorem intersection_on_y_axis (k : ℝ) : 
  (∃ y : ℝ, 2 * 0 + 3 * y - k = 0 ∧ 0 - k * y + 12 = 0) →
  k = 6 ∨ k = -6 := by
sorry

end intersection_on_y_axis_l2800_280094


namespace tammy_orange_earnings_l2800_280085

/-- Calculates Tammy's earnings from selling oranges over 3 weeks --/
def tammys_earnings (num_trees : ℕ) (oranges_per_tree : ℕ) (oranges_per_pack : ℕ) 
  (price_per_pack : ℚ) (days : ℕ) : ℚ :=
  let oranges_per_day := num_trees * oranges_per_tree
  let packs_per_day := oranges_per_day / oranges_per_pack
  let packs_in_period := packs_per_day * days
  packs_in_period * price_per_pack

/-- Proves that Tammy's earnings after 3 weeks will be $840 --/
theorem tammy_orange_earnings : 
  tammys_earnings 10 12 6 2 21 = 840 := by sorry

end tammy_orange_earnings_l2800_280085


namespace expanded_volume_of_problem_box_l2800_280042

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of space inside and within one unit of a box -/
def expandedVolume (b : Box) : ℝ := sorry

/-- The specific box in the problem -/
def problemBox : Box := ⟨2, 3, 4⟩

theorem expanded_volume_of_problem_box :
  expandedVolume problemBox = (228 + 31 * Real.pi) / 3 := by sorry

end expanded_volume_of_problem_box_l2800_280042


namespace part_one_part_two_l2800_280080

/-- The quadratic function y in terms of x and a -/
def y (x a : ℝ) : ℝ := x^2 - (a + 2) * x + 4

/-- Part 1 of the theorem -/
theorem part_one (a b : ℝ) (h1 : b > 1) 
  (h2 : ∀ x, y x a < 0 ↔ 1 < x ∧ x < b) : 
  a = 3 ∧ b = 4 := by sorry

/-- Part 2 of the theorem -/
theorem part_two (a : ℝ) 
  (h : ∀ x, 1 ≤ x → x ≤ 4 → y x a ≥ -a - 1) : 
  a ≤ 4 := by sorry

end part_one_part_two_l2800_280080


namespace person_age_puzzle_l2800_280045

theorem person_age_puzzle (x : ℤ) : 3 * (x + 5) - 3 * (x - 5) = x → x = 30 := by
  sorry

end person_age_puzzle_l2800_280045


namespace pond_to_field_area_ratio_l2800_280006

/-- Proves that the ratio of a square pond's area to a rectangular field's area is 1:50 
    given specific dimensions -/
theorem pond_to_field_area_ratio 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (pond_side : ℝ) 
  (h1 : field_length = 80) 
  (h2 : field_width = 40) 
  (h3 : pond_side = 8) : 
  (pond_side ^ 2) / (field_length * field_width) = 1 / 50 := by
  sorry


end pond_to_field_area_ratio_l2800_280006


namespace yulin_school_sampling_l2800_280074

/-- Systematic sampling function that calculates the number of elements to be removed -/
def systematicSamplingRemoval (populationSize sampleSize : ℕ) : ℕ :=
  populationSize % sampleSize

/-- Theorem stating that for the given population and sample size, 
    the number of students to be removed is 2 -/
theorem yulin_school_sampling :
  systematicSamplingRemoval 254 42 = 2 := by
  sorry

end yulin_school_sampling_l2800_280074


namespace age_relation_l2800_280060

/-- Given that p is currently 3 times as old as q and p was 30 years old 3 years ago,
    prove that p will be twice as old as q in 11 years. -/
theorem age_relation (p q : ℕ) (x : ℕ) : 
  p = 3 * q →  -- p is 3 times as old as q
  p = 30 + 3 →  -- p was 30 years old 3 years ago
  p + x = 2 * (q + x) →  -- in x years, p will be twice as old as q
  x = 11 := by
sorry

end age_relation_l2800_280060


namespace tan_sum_product_equals_one_l2800_280054

theorem tan_sum_product_equals_one :
  ∀ (x y : Real),
  (x = 17 * π / 180 ∧ y = 28 * π / 180) →
  (∀ (A B : Real), Real.tan (A + B) = (Real.tan A + Real.tan B) / (1 - Real.tan A * Real.tan B)) →
  (x + y = π / 4) →
  (Real.tan (π / 4) = 1) →
  Real.tan x + Real.tan y + Real.tan x * Real.tan y = 1 :=
by sorry

end tan_sum_product_equals_one_l2800_280054


namespace maximal_planar_iff_3n_minus_6_edges_l2800_280098

structure PlanarGraph where
  n : ℕ
  e : ℕ
  h_vertices : n ≥ 3

def is_maximal_planar (G : PlanarGraph) : Prop :=
  ∀ H : PlanarGraph, G.n = H.n → G.e ≥ H.e

theorem maximal_planar_iff_3n_minus_6_edges (G : PlanarGraph) :
  is_maximal_planar G ↔ G.e = 3 * G.n - 6 := by sorry

end maximal_planar_iff_3n_minus_6_edges_l2800_280098


namespace work_completion_time_l2800_280008

theorem work_completion_time (x : ℕ) : 
  (50 * x = 25 * (x + 20)) → x = 20 := by
  sorry

end work_completion_time_l2800_280008


namespace crease_lines_form_ellipse_l2800_280083

/-- Given a circle with radius R and an interior point A at distance a from the center,
    this theorem states that the set of points on all crease lines formed by folding
    the circle so that a point on the circumference coincides with A is described by
    the equation of an ellipse. -/
theorem crease_lines_form_ellipse (R a : ℝ) (h : 0 < a ∧ a < R) :
  ∀ x y : ℝ, (x - a / 2)^2 / (R / 2)^2 + y^2 / ((R / 2)^2 - (a / 2)^2) = 1 ↔ 
  (∃ A' : ℝ × ℝ, (A'.1^2 + A'.2^2 = R^2) ∧ 
   ((x - a)^2 + y^2 = (x - A'.1)^2 + (y - A'.2)^2)) :=
by sorry

end crease_lines_form_ellipse_l2800_280083


namespace total_jumps_l2800_280078

theorem total_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 :=
by sorry

end total_jumps_l2800_280078


namespace train_speed_problem_l2800_280036

/-- Proves that given two trains of equal length 70 meters, where one train travels at 50 km/hr
    and passes the other train in 36 seconds, the speed of the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 70 →
  faster_speed = 50 →
  passing_time = 36 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    train_length * 2 = (faster_speed - slower_speed) * passing_time * (1000 / 3600) ∧
    slower_speed = 36 := by
  sorry

end train_speed_problem_l2800_280036


namespace evaluate_fraction_l2800_280026

theorem evaluate_fraction : 
  (3 * Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 11) = 
  -(1/6) * (Real.sqrt 21 + Real.sqrt 35 - Real.sqrt 77) - 7/3 := by
  sorry

end evaluate_fraction_l2800_280026


namespace roundness_of_1280000_l2800_280004

/-- Roundness of a positive integer is the sum of exponents in its prime factorization. -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The number we're calculating the roundness for -/
def our_number : ℕ+ := 1280000

/-- Theorem stating that the roundness of 1,280,000 is 19 -/
theorem roundness_of_1280000 : roundness our_number = 19 := by
  sorry

end roundness_of_1280000_l2800_280004
