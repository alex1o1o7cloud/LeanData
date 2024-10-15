import Mathlib

namespace NUMINAMATH_CALUDE_copper_carbonate_molecular_weight_l1609_160953

/-- The molecular weight of Copper(II) carbonate for a given number of moles -/
def molecular_weight (moles : ℝ) : ℝ := sorry

/-- Theorem: The molecular weight of one mole of Copper(II) carbonate is 124 grams/mole -/
theorem copper_carbonate_molecular_weight :
  molecular_weight 1 = 124 :=
by
  have h : molecular_weight 8 = 992 := sorry
  sorry

end NUMINAMATH_CALUDE_copper_carbonate_molecular_weight_l1609_160953


namespace NUMINAMATH_CALUDE_s_99_digits_l1609_160966

/-- s(n) is the n-digit number formed by attaching the first n perfect squares in order -/
def s (n : ℕ) : ℕ := sorry

/-- count_digits n returns the number of digits in the natural number n -/
def count_digits (n : ℕ) : ℕ := sorry

/-- The theorem states that s(99) has 189 digits -/
theorem s_99_digits : count_digits (s 99) = 189 := by sorry

end NUMINAMATH_CALUDE_s_99_digits_l1609_160966


namespace NUMINAMATH_CALUDE_one_pair_probability_l1609_160997

/-- Represents the number of socks -/
def total_socks : ℕ := 10

/-- Represents the number of colors -/
def num_colors : ℕ := 5

/-- Represents the number of socks per color -/
def socks_per_color : ℕ := 2

/-- Represents the number of socks to be selected -/
def socks_selected : ℕ := 4

/-- Calculates the probability of selecting exactly one pair of socks of the same color -/
def prob_one_pair : ℚ := 4 / 7

/-- Proves that the probability of selecting exactly one pair of socks of the same color
    when randomly choosing 4 socks from a set of 10 socks (2 of each of 5 colors) is 4/7 -/
theorem one_pair_probability : 
  total_socks = num_colors * socks_per_color ∧ 
  socks_selected = 4 → 
  prob_one_pair = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_one_pair_probability_l1609_160997


namespace NUMINAMATH_CALUDE_poster_width_l1609_160933

theorem poster_width (height : ℝ) (area : ℝ) (width : ℝ) 
  (h1 : height = 7)
  (h2 : area = 28)
  (h3 : area = width * height) : 
  width = 4 := by sorry

end NUMINAMATH_CALUDE_poster_width_l1609_160933


namespace NUMINAMATH_CALUDE_ratio_problem_l1609_160977

theorem ratio_problem (first_number second_number : ℝ) : 
  first_number / second_number = 20 → first_number = 200 → second_number = 10 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1609_160977


namespace NUMINAMATH_CALUDE_cube_construction_count_l1609_160989

/-- The rotational symmetry group of a cube -/
def CubeRotationGroup : Type := Unit

/-- The order of the rotational symmetry group of a cube -/
def cubeRotationGroupOrder : ℕ := 26

/-- The number of ways to choose 13 items from 27 items -/
def chooseThirteenFromTwentySeven : ℕ := 2333606

/-- The number of configurations fixed by the identity rotation -/
def fixedByIdentity : ℕ := chooseThirteenFromTwentySeven

/-- The number of configurations fixed by rotations around centroids of faces -/
def fixedByCentroidRotations : ℕ := 2

/-- The number of configurations fixed by rotations around vertices and edges -/
def fixedByVertexEdgeRotations : ℕ := 1

/-- The total number of fixed configurations -/
def totalFixedConfigurations : ℕ := fixedByIdentity + fixedByCentroidRotations + fixedByVertexEdgeRotations

/-- The number of rotationally distinct ways to construct the cube -/
def distinctConstructions : ℕ := totalFixedConfigurations / cubeRotationGroupOrder

theorem cube_construction_count :
  distinctConstructions = 89754 :=
sorry

end NUMINAMATH_CALUDE_cube_construction_count_l1609_160989


namespace NUMINAMATH_CALUDE_bike_ride_time_l1609_160970

/-- Given a constant speed where 2 miles are covered in 6 minutes,
    prove that the time required to travel 5 miles at the same speed is 15 minutes. -/
theorem bike_ride_time (speed : ℝ) (h1 : speed > 0) (h2 : 2 / speed = 6) : 5 / speed = 15 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_time_l1609_160970


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l1609_160999

/-- The cost of fencing an irregularly shaped field -/
theorem fencing_cost_theorem (triangle_side1 triangle_side2 triangle_side3 circle_radius : ℝ)
  (triangle_cost_per_meter circle_cost_per_meter : ℝ)
  (h1 : triangle_side1 = 100)
  (h2 : triangle_side2 = 150)
  (h3 : triangle_side3 = 50)
  (h4 : circle_radius = 30)
  (h5 : triangle_cost_per_meter = 5)
  (h6 : circle_cost_per_meter = 7) :
  ∃ (total_cost : ℝ), 
    abs (total_cost - ((triangle_side1 + triangle_side2 + triangle_side3) * triangle_cost_per_meter +
    2 * Real.pi * circle_radius * circle_cost_per_meter)) < 1 ∧
    total_cost = 2819 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_theorem_l1609_160999


namespace NUMINAMATH_CALUDE_triple_q_2000_power_l1609_160993

/-- Sum of digits function -/
def q (n : ℕ) : ℕ :=
  if n < 10 then n else q (n / 10) + n % 10

/-- Theorem: The triple application of q to 2000^2000 results in 4 -/
theorem triple_q_2000_power : q (q (q (2000^2000))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_triple_q_2000_power_l1609_160993


namespace NUMINAMATH_CALUDE_total_big_cats_l1609_160940

def feline_sanctuary (lions tigers : ℕ) : ℕ :=
  let cougars := (lions + tigers) / 2
  lions + tigers + cougars

theorem total_big_cats :
  feline_sanctuary 12 14 = 39 := by
  sorry

end NUMINAMATH_CALUDE_total_big_cats_l1609_160940


namespace NUMINAMATH_CALUDE_function_value_theorem_l1609_160980

theorem function_value_theorem (f : ℝ → ℝ) (h : ∀ x, f ((1/2) * x - 1) = 2 * x + 3) :
  f (-3/4) = 4 :=
by sorry

end NUMINAMATH_CALUDE_function_value_theorem_l1609_160980


namespace NUMINAMATH_CALUDE_intersection_and_range_l1609_160994

def A : Set ℝ := {x | x^2 + 2*x - 8 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem intersection_and_range :
  (A ∩ B 1 = {-4}) ∧
  (∀ a : ℝ, A ∩ B a = B a ↔ a < -1 ∨ a > 3) := by
  sorry

end NUMINAMATH_CALUDE_intersection_and_range_l1609_160994


namespace NUMINAMATH_CALUDE_proportional_increase_l1609_160976

theorem proportional_increase (x y : ℝ) (c : ℝ) (h1 : y = c * x) :
  let x' := 1.3 * x
  let y' := c * x'
  y' = 2.6 * y →
  (y' - y) / y = 1.6 := by
sorry

end NUMINAMATH_CALUDE_proportional_increase_l1609_160976


namespace NUMINAMATH_CALUDE_final_price_approx_l1609_160938

-- Define the initial cost price
def initial_cost : ℝ := 114.94

-- Define the profit percentages
def profit_A : ℝ := 0.35
def profit_B : ℝ := 0.45

-- Define the function to calculate selling price given cost price and profit percentage
def selling_price (cost : ℝ) (profit : ℝ) : ℝ := cost * (1 + profit)

-- Define the final selling price calculation
def final_price : ℝ := selling_price (selling_price initial_cost profit_A) profit_B

-- Theorem to prove
theorem final_price_approx :
  ∃ ε > 0, |final_price - 225| < ε :=
sorry

end NUMINAMATH_CALUDE_final_price_approx_l1609_160938


namespace NUMINAMATH_CALUDE_procedure_arrangement_count_l1609_160944

/-- The number of ways to arrange 6 procedures with specific constraints -/
def arrangement_count : ℕ := 96

/-- The number of procedures -/
def total_procedures : ℕ := 6

/-- The number of ways to place procedure A (first or last) -/
def a_placements : ℕ := 2

/-- The number of ways to arrange B and C within their unit -/
def bc_arrangements : ℕ := 2

/-- The number of elements to arrange (BC unit + 3 other procedures) -/
def elements_to_arrange : ℕ := 4

theorem procedure_arrangement_count :
  arrangement_count = 
    a_placements * elements_to_arrange.factorial * bc_arrangements :=
by sorry

end NUMINAMATH_CALUDE_procedure_arrangement_count_l1609_160944


namespace NUMINAMATH_CALUDE_square_real_implies_a_zero_l1609_160925

theorem square_real_implies_a_zero (a : ℝ) : 
  (Complex.I * a + 2) ^ 2 ∈ Set.range Complex.ofReal → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_real_implies_a_zero_l1609_160925


namespace NUMINAMATH_CALUDE_function_characterization_l1609_160936

-- Define the property that f must satisfy
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x * f y + f (x + y) ≥ (y + 1) * f x + f y

-- Theorem statement
theorem function_characterization (f : ℝ → ℝ) 
  (h : SatisfiesInequality f) : 
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x := by
  sorry

end NUMINAMATH_CALUDE_function_characterization_l1609_160936


namespace NUMINAMATH_CALUDE_min_lines_proof_l1609_160949

/-- The number of regions created by n lines in a plane -/
def regions (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- The minimum number of lines needed to divide a plane into at least 1000 regions -/
def min_lines_for_1000_regions : ℕ := 45

theorem min_lines_proof :
  (∀ k < min_lines_for_1000_regions, regions k < 1000) ∧
  regions min_lines_for_1000_regions ≥ 1000 := by
  sorry

#eval regions min_lines_for_1000_regions

end NUMINAMATH_CALUDE_min_lines_proof_l1609_160949


namespace NUMINAMATH_CALUDE_original_number_proof_l1609_160967

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

theorem original_number_proof :
  let original := 1453789
  let swapped := 8453719
  (∃ i j, swap_digits original i j = swapped) ∧
  (swapped > 3 * original) :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1609_160967


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1609_160995

-- Define the prices and quantities
def shirt_price : ℝ := 15
def shirt_quantity : ℕ := 4
def pants_price : ℝ := 40
def pants_quantity : ℕ := 2
def suit_price : ℝ := 150
def suit_quantity : ℕ := 1
def sweater_price : ℝ := 30
def sweater_quantity : ℕ := 2
def tie_price : ℝ := 20
def tie_quantity : ℕ := 3
def shoes_price : ℝ := 80
def shoes_quantity : ℕ := 1

-- Define the discounts
def shirt_discount : ℝ := 0.2
def pants_discount : ℝ := 0.3
def tie_discount : ℝ := 0.5
def shoes_discount : ℝ := 0.25
def coupon_discount : ℝ := 0.1

-- Define reward points
def reward_points : ℕ := 500
def reward_point_value : ℝ := 0.05

-- Define sales tax
def sales_tax_rate : ℝ := 0.05

-- Define the theorem
theorem total_cost_calculation :
  let shirt_total := shirt_price * shirt_quantity * (1 - shirt_discount)
  let pants_total := pants_price * pants_quantity * (1 - pants_discount)
  let suit_total := suit_price * suit_quantity
  let sweater_total := sweater_price * sweater_quantity
  let tie_total := tie_price * tie_quantity - tie_price * tie_discount
  let shoes_total := shoes_price * shoes_quantity * (1 - shoes_discount)
  let subtotal := shirt_total + pants_total + suit_total + sweater_total + tie_total + shoes_total
  let after_coupon := subtotal * (1 - coupon_discount)
  let after_rewards := after_coupon - (reward_points * reward_point_value)
  let final_total := after_rewards * (1 + sales_tax_rate)
  final_total = 374.43 := by sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1609_160995


namespace NUMINAMATH_CALUDE_two_cubic_feet_equals_3456_cubic_inches_l1609_160983

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℚ := 12

/-- Calculates the volume in cubic inches given the volume in cubic feet -/
def cubic_feet_to_cubic_inches (cf : ℚ) : ℚ :=
  cf * feet_to_inches^3

/-- Theorem stating that 2 cubic feet is equal to 3456 cubic inches -/
theorem two_cubic_feet_equals_3456_cubic_inches :
  cubic_feet_to_cubic_inches 2 = 3456 := by
  sorry

end NUMINAMATH_CALUDE_two_cubic_feet_equals_3456_cubic_inches_l1609_160983


namespace NUMINAMATH_CALUDE_pred_rohem_30_more_pred_rohem_triple_total_sold_is_60_l1609_160948

/-- The number of alarm clocks sold at "Za Rohem" -/
def za_rohem : ℕ := 15

/-- The number of alarm clocks sold at "Před Rohem" -/
def pred_rohem : ℕ := za_rohem + 30

/-- The claim that "Před Rohem" sold 30 more alarm clocks than "Za Rohem" -/
theorem pred_rohem_30_more : pred_rohem = za_rohem + 30 := by sorry

/-- The claim that "Před Rohem" sold three times as many alarm clocks as "Za Rohem" -/
theorem pred_rohem_triple : pred_rohem = 3 * za_rohem := by sorry

/-- The total number of alarm clocks sold at both shops -/
def total_sold : ℕ := za_rohem + pred_rohem

/-- Proof that the total number of alarm clocks sold at both shops is 60 -/
theorem total_sold_is_60 : total_sold = 60 := by sorry

end NUMINAMATH_CALUDE_pred_rohem_30_more_pred_rohem_triple_total_sold_is_60_l1609_160948


namespace NUMINAMATH_CALUDE_floor_equality_iff_interval_l1609_160912

theorem floor_equality_iff_interval (x : ℝ) : 
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋ ↔ 2 ≤ x ∧ x < 7/3 := by sorry

end NUMINAMATH_CALUDE_floor_equality_iff_interval_l1609_160912


namespace NUMINAMATH_CALUDE_certain_number_proof_l1609_160908

theorem certain_number_proof (n x : ℝ) (h1 : n = -4.5) (h2 : 10 * n = x - 2 * n) : x = -54 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1609_160908


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l1609_160982

/-- Represents a 3D solid composed of unit cubes -/
structure CubeSolid where
  cubes : ℕ
  top_layer : ℕ
  bottom_layer : ℕ
  height : ℕ
  length : ℕ
  depth : ℕ

/-- Calculates the surface area of a CubeSolid -/
def surface_area (s : CubeSolid) : ℕ := sorry

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid :=
  { cubes := 15
  , top_layer := 5
  , bottom_layer := 5
  , height := 3
  , length := 5
  , depth := 3 }

/-- Theorem stating that the surface area of the problem_solid is 26 square units -/
theorem problem_solid_surface_area :
  surface_area problem_solid = 26 := by sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l1609_160982


namespace NUMINAMATH_CALUDE_no_solution_arctan_equation_l1609_160991

theorem no_solution_arctan_equation :
  ¬ ∃ (x : ℝ), x > 0 ∧ Real.arctan (1 / x^2) + Real.arctan (1 / x^4) = π / 4 := by
sorry

end NUMINAMATH_CALUDE_no_solution_arctan_equation_l1609_160991


namespace NUMINAMATH_CALUDE_maria_flour_calculation_l1609_160987

/-- The amount of flour needed for a given number of cookies -/
def flour_needed (cookies : ℕ) : ℚ :=
  (3 : ℚ) * cookies / 40

theorem maria_flour_calculation :
  flour_needed 120 = 9 := by sorry

end NUMINAMATH_CALUDE_maria_flour_calculation_l1609_160987


namespace NUMINAMATH_CALUDE_art_show_pricing_l1609_160945

/-- The price of a large painting that satisfies the given conditions -/
def large_painting_price : ℕ → ℕ → ℕ → ℕ → ℕ := λ small_price large_count small_count total_earnings =>
  (total_earnings - small_price * small_count) / large_count

theorem art_show_pricing (small_price large_count small_count total_earnings : ℕ) 
  (h1 : small_price = 80)
  (h2 : large_count = 5)
  (h3 : small_count = 8)
  (h4 : total_earnings = 1140) :
  large_painting_price small_price large_count small_count total_earnings = 100 := by
sorry

#eval large_painting_price 80 5 8 1140

end NUMINAMATH_CALUDE_art_show_pricing_l1609_160945


namespace NUMINAMATH_CALUDE_right_triangle_construction_impossibility_l1609_160926

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a point being inside a circle
def IsInside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ), c = Circle center radius ∧
    (p.1 - center.1)^2 + (p.2 - center.2)^2 < radius^2

-- Define a circle with diameter AB
def CircleWithDiameter (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  Circle ((A.1 + B.1)/2, (A.2 + B.2)/2) (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 2)

-- Define intersection of two sets
def Intersects (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ s1 ∧ p ∈ s2

-- Main theorem
theorem right_triangle_construction_impossibility
  (C : Set (ℝ × ℝ)) (A B : ℝ × ℝ)
  (h_circle : ∃ center radius, C = Circle center radius)
  (h_A_inside : IsInside A C)
  (h_B_inside : IsInside B C) :
  (¬ ∃ P Q R : ℝ × ℝ,
    P ∈ C ∧ Q ∈ C ∧ R ∈ C ∧
    (A.1 - P.1) * (Q.1 - P.1) + (A.2 - P.2) * (Q.2 - P.2) = 0 ∧
    (B.1 - P.1) * (R.1 - P.1) + (B.2 - P.2) * (R.2 - P.2) = 0 ∧
    (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  ↔
  ¬ Intersects (CircleWithDiameter A B) C :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_construction_impossibility_l1609_160926


namespace NUMINAMATH_CALUDE_pet_store_combinations_l1609_160990

def num_puppies : ℕ := 20
def num_kittens : ℕ := 6
def num_hamsters : ℕ := 8

def alice_choices : ℕ := num_puppies
def bob_pet_type_choices : ℕ := 2
def bob_specific_pet_choices : ℕ := num_kittens
def charlie_choices : ℕ := num_hamsters

theorem pet_store_combinations : 
  alice_choices * bob_pet_type_choices * bob_specific_pet_choices * charlie_choices = 1920 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l1609_160990


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1609_160979

/-- Given a circle with center (5, -2) and one endpoint of a diameter at (2, 3),
    prove that the other endpoint of the diameter is at (8, -7). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) (endpoint2 : ℝ × ℝ) : 
  center = (5, -2) → endpoint1 = (2, 3) → endpoint2 = (8, -7) → 
  (center.1 - endpoint1.1 = endpoint2.1 - center.1 ∧ 
   center.2 - endpoint1.2 = endpoint2.2 - center.2) := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1609_160979


namespace NUMINAMATH_CALUDE_max_trig_sum_l1609_160998

theorem max_trig_sum (θ₁ θ₂ θ₃ θ₄ θ₅ : ℝ) :
  Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
  Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
  Real.cos θ₅ * Real.sin θ₁ ≤ 5/2 := by
sorry

end NUMINAMATH_CALUDE_max_trig_sum_l1609_160998


namespace NUMINAMATH_CALUDE_art_count_l1609_160903

/-- The number of Asian art pieces seen -/
def asian_art : ℕ := 465

/-- The number of Egyptian art pieces seen -/
def egyptian_art : ℕ := 527

/-- The total number of art pieces seen -/
def total_art : ℕ := asian_art + egyptian_art

theorem art_count : total_art = 992 := by
  sorry

end NUMINAMATH_CALUDE_art_count_l1609_160903


namespace NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1609_160939

theorem square_sum_zero_implies_both_zero (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_zero_implies_both_zero_l1609_160939


namespace NUMINAMATH_CALUDE_hexagon_pattern_triangle_area_l1609_160906

/-- The area of a triangle formed by centers of alternate hexagons in a hexagonal pattern -/
theorem hexagon_pattern_triangle_area :
  ∀ (hexagon_side_length : ℝ) (triangle_side_length : ℝ),
    hexagon_side_length = 1 →
    triangle_side_length = 3 * hexagon_side_length →
    ∃ (triangle_area : ℝ),
      triangle_area = (9 * Real.sqrt 3) / 4 ∧
      triangle_area = (Real.sqrt 3 / 4) * triangle_side_length^2 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_pattern_triangle_area_l1609_160906


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l1609_160959

theorem subtraction_of_decimals : 7.25 - 3.1 - 1.05 = 3.10 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l1609_160959


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_half_l1609_160934

theorem sin_cos_sum_equals_sqrt_sum_half :
  Real.sin (14 * π / 3) + Real.cos (-25 * π / 4) = (Real.sqrt 3 + Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt_sum_half_l1609_160934


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1609_160911

theorem pirate_treasure_probability :
  let n_islands : ℕ := 8
  let n_treasure : ℕ := 4
  let p_treasure : ℚ := 1/3
  let p_traps : ℚ := 1/6
  let p_neither : ℚ := 1/2
  let choose := fun (n k : ℕ) => (Nat.choose n k : ℚ)
  
  (choose n_islands n_treasure) * p_treasure^n_treasure * p_neither^(n_islands - n_treasure) = 35/648 :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1609_160911


namespace NUMINAMATH_CALUDE_min_side_triangle_l1609_160954

theorem min_side_triangle (S γ : ℝ) (hS : S > 0) (hγ : 0 < γ ∧ γ < π) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (1/2 * a * b * Real.sin γ = S) ∧
  (∀ (a' b' c' : ℝ), a' > 0 → b' > 0 → c' > 0 →
    1/2 * a' * b' * Real.sin γ = S →
    c' ≥ 2 * Real.sqrt (S * Real.tan (γ/2))) :=
sorry

end NUMINAMATH_CALUDE_min_side_triangle_l1609_160954


namespace NUMINAMATH_CALUDE_kona_trip_distance_l1609_160921

/-- The distance from Kona's apartment to the bakery in miles -/
def apartment_to_bakery : ℝ := 9

/-- The distance from the bakery to Kona's grandmother's house in miles -/
def bakery_to_grandma : ℝ := 24

/-- The additional distance of the round trip with bakery stop compared to without -/
def additional_distance : ℝ := 6

/-- The distance from Kona's grandmother's house to his apartment in miles -/
def grandma_to_apartment : ℝ := 27

theorem kona_trip_distance :
  apartment_to_bakery + bakery_to_grandma + grandma_to_apartment =
  2 * grandma_to_apartment + additional_distance :=
sorry

end NUMINAMATH_CALUDE_kona_trip_distance_l1609_160921


namespace NUMINAMATH_CALUDE_closest_angles_to_2013_l1609_160955

theorem closest_angles_to_2013 (x : ℝ) :
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2) →
  (x = 1935 * π / 180 ∨ x = 2025 * π / 180) ∧
  ∀ y : ℝ, (2^(Real.sin y)^2 + 2^(Real.cos y)^2 = 2 * Real.sqrt 2) →
    (1935 * π / 180 < y ∧ y < 2025 * π / 180) →
    (y ≠ 1935 * π / 180 ∧ y ≠ 2025 * π / 180) →
    ¬(∃ n : ℤ, y = n * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_closest_angles_to_2013_l1609_160955


namespace NUMINAMATH_CALUDE_woods_area_calculation_l1609_160988

/-- The area of rectangular woods -/
def woods_area (width : ℝ) (length : ℝ) : ℝ := width * length

/-- Theorem: The area of woods with width 8 miles and length 3 miles is 24 square miles -/
theorem woods_area_calculation :
  woods_area 8 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_woods_area_calculation_l1609_160988


namespace NUMINAMATH_CALUDE_integer_root_count_l1609_160941

theorem integer_root_count : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, ∃ k : ℤ, Real.sqrt (123 - Real.sqrt x) = k) ∧ S.card = 12 :=
sorry

end NUMINAMATH_CALUDE_integer_root_count_l1609_160941


namespace NUMINAMATH_CALUDE_shooting_training_equivalence_l1609_160963

-- Define the propositions
variable (p q : Prop)

-- Define "both shots hit the target"
def both_hit (p q : Prop) : Prop := p ∧ q

-- Define "exactly one shot hits the target"
def exactly_one_hit (p q : Prop) : Prop := (p ∧ ¬q) ∨ (¬p ∧ q)

-- Theorem stating the equivalence
theorem shooting_training_equivalence :
  (both_hit p q ↔ p ∧ q) ∧
  (exactly_one_hit p q ↔ (p ∧ ¬q) ∨ (¬p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_shooting_training_equivalence_l1609_160963


namespace NUMINAMATH_CALUDE_mary_earnings_per_home_l1609_160985

/-- Mary's earnings per home, given total earnings and number of homes cleaned -/
def earnings_per_home (total_earnings : ℕ) (homes_cleaned : ℕ) : ℕ :=
  total_earnings / homes_cleaned

/-- Proof that Mary earns $46 per home -/
theorem mary_earnings_per_home :
  earnings_per_home 276 6 = 46 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_per_home_l1609_160985


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1609_160969

/-- Given two polynomials in p, prove that their difference simplifies to the given result. -/
theorem polynomial_simplification (p : ℝ) :
  (2 * p^4 - 3 * p^3 + 7 * p - 4) - (-6 * p^3 - 5 * p^2 + 4 * p + 3) =
  2 * p^4 + 3 * p^3 + 5 * p^2 + 3 * p - 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1609_160969


namespace NUMINAMATH_CALUDE_max_games_512_3_l1609_160931

/-- Represents a tournament where players must be defeated three times to be eliminated -/
structure Tournament where
  contestants : ℕ
  defeats_to_eliminate : ℕ

/-- Calculates the maximum number of games that could be played in the tournament -/
def max_games (t : Tournament) : ℕ :=
  (t.contestants - 1) * t.defeats_to_eliminate + 2

/-- Theorem stating that for a tournament with 512 contestants and 3 defeats to eliminate,
    the maximum number of games is 1535 -/
theorem max_games_512_3 :
  let t : Tournament := { contestants := 512, defeats_to_eliminate := 3 }
  max_games t = 1535 := by
  sorry

end NUMINAMATH_CALUDE_max_games_512_3_l1609_160931


namespace NUMINAMATH_CALUDE_min_value_a_l1609_160972

theorem min_value_a : 
  (∀ (x y : ℝ), x > 0 → y > 0 → x + Real.sqrt (x * y) ≤ a * (x + y)) → 
  a ≥ (Real.sqrt 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_a_l1609_160972


namespace NUMINAMATH_CALUDE_cards_distribution_l1609_160984

/-- Given a deck of 48 cards dealt as evenly as possible among 9 people,
    the number of people who receive fewer than 6 cards is 6. -/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 48) (h2 : num_people = 9) :
  let cards_per_person := total_cards / num_people
  let remainder := total_cards % num_people
  let people_with_extra := remainder
  let people_with_fewer := num_people - people_with_extra
  people_with_fewer = 6 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l1609_160984


namespace NUMINAMATH_CALUDE_jaydee_typing_time_l1609_160950

/-- Calculates the time needed to type a research paper given specific conditions. -/
def time_to_type_paper (words_per_minute : ℕ) (break_interval : ℕ) (break_duration : ℕ) 
  (words_per_mistake : ℕ) (mistake_correction_time : ℕ) (total_words : ℕ) : ℕ :=
  let typing_time := (total_words + words_per_minute - 1) / words_per_minute
  let breaks := typing_time / break_interval
  let break_time := breaks * break_duration
  let mistakes := (total_words + words_per_mistake - 1) / words_per_mistake
  let correction_time := mistakes * mistake_correction_time
  let total_minutes := typing_time + break_time + correction_time
  (total_minutes + 59) / 60

/-- Theorem stating that Jaydee will take 6 hours to type the research paper. -/
theorem jaydee_typing_time : 
  time_to_type_paper 32 25 5 100 1 7125 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jaydee_typing_time_l1609_160950


namespace NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l1609_160961

theorem largest_n_for_sin_cos_inequality : 
  ∃ n : ℕ+, (∀ m : ℕ+, m > n → ∃ x : ℝ, (Real.sin x + Real.cos x)^(m : ℝ) < 2 / (m : ℝ)) ∧
             (∀ x : ℝ, (Real.sin x + Real.cos x)^(n : ℝ) ≥ 2 / (n : ℝ)) ∧
             n = 4 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_sin_cos_inequality_l1609_160961


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l1609_160917

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) : 
  total = 36 →
  difference = 6 →
  ∃ (girls boys : ℕ),
    girls = boys + difference ∧
    girls + boys = total ∧
    girls * 5 = boys * 7 :=
by sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l1609_160917


namespace NUMINAMATH_CALUDE_digit_207_is_8_l1609_160932

/-- The decimal representation of 3/7 as a sequence of digits -/
def decimal_rep_3_7 : ℕ → Fin 10
  | n => sorry

/-- The length of the repeating sequence in the decimal representation of 3/7 -/
def repeat_length : ℕ := 6

/-- The 207th digit beyond the decimal point in the decimal representation of 3/7 -/
def digit_207 : Fin 10 := decimal_rep_3_7 206

theorem digit_207_is_8 : digit_207 = 8 := by sorry

end NUMINAMATH_CALUDE_digit_207_is_8_l1609_160932


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1609_160952

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 1456 [ZMOD 11]) → n ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1609_160952


namespace NUMINAMATH_CALUDE_blue_face_probability_l1609_160951

/-- A cube with colored faces -/
structure ColoredCube where
  blue_faces : ℕ
  red_faces : ℕ

/-- The probability of rolling a specific color on a colored cube -/
def roll_probability (cube : ColoredCube) (color : String) : ℚ :=
  match color with
  | "blue" => cube.blue_faces / (cube.blue_faces + cube.red_faces)
  | "red" => cube.red_faces / (cube.blue_faces + cube.red_faces)
  | _ => 0

/-- Theorem: The probability of rolling a blue face on a cube with 3 blue faces and 3 red faces is 1/2 -/
theorem blue_face_probability :
  ∀ (cube : ColoredCube),
    cube.blue_faces = 3 →
    cube.red_faces = 3 →
    roll_probability cube "blue" = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_blue_face_probability_l1609_160951


namespace NUMINAMATH_CALUDE_faye_pencil_rows_l1609_160914

/-- Given that Faye has 720 pencils in total and places 24 pencils in each row,
    prove that the number of rows she created is 30. -/
theorem faye_pencil_rows (total_pencils : Nat) (pencils_per_row : Nat) (rows : Nat) :
  total_pencils = 720 →
  pencils_per_row = 24 →
  rows * pencils_per_row = total_pencils →
  rows = 30 := by
  sorry

#check faye_pencil_rows

end NUMINAMATH_CALUDE_faye_pencil_rows_l1609_160914


namespace NUMINAMATH_CALUDE_double_price_increase_l1609_160962

theorem double_price_increase (original_price : ℝ) (increase_percentage : ℝ) :
  let first_increase := original_price * (1 + increase_percentage / 100)
  let second_increase := first_increase * (1 + increase_percentage / 100)
  increase_percentage = 15 →
  second_increase = original_price * (1 + 32.25 / 100) :=
by sorry

end NUMINAMATH_CALUDE_double_price_increase_l1609_160962


namespace NUMINAMATH_CALUDE_P_roots_count_l1609_160968

/-- Recursive definition of the polynomial sequence Pₙ(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | 1, x => x
  | (n+2), x => x * P (n+1) x - P n x

/-- The number of distinct real roots of Pₙ(x) -/
def num_roots (n : ℕ) : ℕ := n

theorem P_roots_count (n : ℕ) : 
  (∃ (s : Finset ℝ), s.card = num_roots n ∧ 
   (∀ x ∈ s, P n x = 0) ∧
   (∀ x : ℝ, P n x = 0 → x ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_P_roots_count_l1609_160968


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1609_160958

/-- Hyperbola with foci and a special point -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ  -- Left focus
  F₂ : ℝ × ℝ  -- Right focus
  P : ℝ × ℝ   -- Special point on the right branch
  h₁ : a > b
  h₂ : b > 0
  h₃ : F₁.1 < 0 ∧ F₁.2 = 0  -- Left focus on negative x-axis
  h₄ : F₂.1 > 0 ∧ F₂.2 = 0  -- Right focus on positive x-axis
  h₅ : P.1 > 0  -- P is on the right branch
  h₆ : P.1^2 / a^2 - P.2^2 / b^2 = 1  -- P satisfies hyperbola equation
  h₇ : (P.1 + F₂.1) * (P.1 - F₂.1) + P.2 * P.2 = 0  -- Dot product condition
  h₈ : (P.1 - F₁.1)^2 + P.2^2 = 4 * ((P.1 - F₂.1)^2 + P.2^2)  -- Distance condition

/-- The eccentricity of a hyperbola with the given properties is √5 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt ((h.F₂.1 - h.F₁.1)^2 / (4 * h.a^2)) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1609_160958


namespace NUMINAMATH_CALUDE_proposition_d_is_false_l1609_160943

/-- Proposition D is false: There exist four mutually different non-zero vectors on a plane 
    such that the sum vector of any two vectors is perpendicular to the sum vector of 
    the remaining two vectors. -/
theorem proposition_d_is_false :
  ∃ (a b c d : ℝ × ℝ), 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a.1 + b.1) * (c.1 + d.1) + (a.2 + b.2) * (c.2 + d.2) = 0 ∧
    (a.1 + c.1) * (b.1 + d.1) + (a.2 + c.2) * (b.2 + d.2) = 0 ∧
    (a.1 + d.1) * (b.1 + c.1) + (a.2 + d.2) * (b.2 + c.2) = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_proposition_d_is_false_l1609_160943


namespace NUMINAMATH_CALUDE_shaded_area_proof_l1609_160996

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem shaded_area_proof : U \ (A ∪ B) = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l1609_160996


namespace NUMINAMATH_CALUDE_equation_unique_solution_l1609_160924

/-- The function representing the left-hand side of the equation -/
def f (y : ℝ) : ℝ := (30 * y + (30 * y + 25) ^ (1/3)) ^ (1/3)

/-- The theorem stating that the equation has a unique solution -/
theorem equation_unique_solution :
  ∃! y : ℝ, f y = 15 ∧ y = 335/3 := by sorry

end NUMINAMATH_CALUDE_equation_unique_solution_l1609_160924


namespace NUMINAMATH_CALUDE_fraction_comparison_l1609_160942

theorem fraction_comparison (x : ℝ) : 
  x > 3/4 → x ≠ 3 → (9 - 3*x ≠ 0) → (5*x + 3 > 9 - 3*x) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1609_160942


namespace NUMINAMATH_CALUDE_triangle_horizontal_line_l1609_160907

/-- Given two intersecting lines and the area of the triangle they form with the x-axis,
    prove the equation of the horizontal line that completes this triangle. -/
theorem triangle_horizontal_line
  (line1 : ℝ → ℝ)
  (line2 : ℝ)
  (area : ℝ)
  (h1 : ∀ x, line1 x = x)
  (h2 : line2 = -9)
  (h3 : area = 40.5)
  : ∃ y : ℝ, y = 9 ∧ 
    (1/2 : ℝ) * |line2| * y = area ∧
    (line1 (-line2) = y) :=
by sorry

end NUMINAMATH_CALUDE_triangle_horizontal_line_l1609_160907


namespace NUMINAMATH_CALUDE_line_slope_and_intercept_l1609_160930

/-- Given a line expressed as (3, -4) · ((x, y) - (-2, 8)) = 0, 
    prove that its slope is 3/4 and its y-intercept is 9.5 -/
theorem line_slope_and_intercept :
  let line := fun (x y : ℝ) => 3 * (x + 2) + (-4) * (y - 8) = 0
  ∃ (m b : ℝ), m = 3/4 ∧ b = 9.5 ∧ ∀ x y, line x y ↔ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_slope_and_intercept_l1609_160930


namespace NUMINAMATH_CALUDE_no_five_solutions_and_divisibility_l1609_160935

theorem no_five_solutions_and_divisibility (k : ℤ) :
  (¬ ∃ (x₁ x₂ x₃ x₄ x₅ y₁ : ℤ),
    y₁^2 - k = x₁^3 ∧
    (y₁ - 1)^2 - k = x₂^3 ∧
    (y₁ - 2)^2 - k = x₃^3 ∧
    (y₁ - 3)^2 - k = x₄^3 ∧
    (y₁ - 4)^2 - k = x₅^3) ∧
  (∀ (x₁ x₂ x₃ x₄ y₁ : ℤ),
    y₁^2 - k = x₁^3 ∧
    (y₁ - 1)^2 - k = x₂^3 ∧
    (y₁ - 2)^2 - k = x₃^3 ∧
    (y₁ - 3)^2 - k = x₄^3 →
    63 ∣ (k - 17)) :=
by sorry

end NUMINAMATH_CALUDE_no_five_solutions_and_divisibility_l1609_160935


namespace NUMINAMATH_CALUDE_largest_digit_sum_quotient_l1609_160965

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

theorem largest_digit_sum_quotient :
  (∀ n : ThreeDigitNumber, (value n : ℚ) / (digitSum n : ℚ) ≤ 100) ∧
  (∃ n : ThreeDigitNumber, (value n : ℚ) / (digitSum n : ℚ) = 100) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_sum_quotient_l1609_160965


namespace NUMINAMATH_CALUDE_sample_size_calculation_l1609_160928

/-- Given a sample with 16 units of model A, and the ratio of quantities of 
    models A, B, and C being 2:3:5, the total sample size n is 80. -/
theorem sample_size_calculation (model_a_count : ℕ) (ratio_a ratio_b ratio_c : ℕ) :
  model_a_count = 16 →
  ratio_a = 2 →
  ratio_b = 3 →
  ratio_c = 5 →
  (ratio_a : ℚ) / (ratio_a + ratio_b + ratio_c : ℚ) * (model_a_count * (ratio_a + ratio_b + ratio_c) / ratio_a) = 80 :=
by sorry

end NUMINAMATH_CALUDE_sample_size_calculation_l1609_160928


namespace NUMINAMATH_CALUDE_floor_tiling_l1609_160974

theorem floor_tiling (n : ℕ) (h1 : 10 < n) (h2 : n < 20) :
  (∃ x : ℕ, n^2 = 9*x) ↔ n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiling_l1609_160974


namespace NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_l1609_160923

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180

-- Define an obtuse triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  ∃ i : Fin 3, t.angles i > 90

-- Define an obtuse angle
def ObtuseAngle (angle : ℝ) : Prop := angle > 90

-- Theorem: An obtuse triangle has exactly one obtuse interior angle
theorem obtuse_triangle_one_obtuse_angle (t : Triangle) (h : ObtuseTriangle t) :
  ∃! i : Fin 3, ObtuseAngle (t.angles i) :=
sorry

end NUMINAMATH_CALUDE_obtuse_triangle_one_obtuse_angle_l1609_160923


namespace NUMINAMATH_CALUDE_bob_local_tax_cents_l1609_160973

/-- Bob's hourly wage in dollars -/
def bob_hourly_wage : ℝ := 25

/-- Local tax rate as a decimal -/
def local_tax_rate : ℝ := 0.025

/-- Conversion rate from dollars to cents -/
def dollars_to_cents : ℝ := 100

/-- Theorem: The amount of Bob's hourly wage used for local taxes is 62.5 cents -/
theorem bob_local_tax_cents : 
  bob_hourly_wage * local_tax_rate * dollars_to_cents = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_bob_local_tax_cents_l1609_160973


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l1609_160978

theorem sqrt_difference_approximation : |Real.sqrt 144 - Real.sqrt 140 - 0.17| < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l1609_160978


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1609_160964

def roll_probability : ℚ := 1 / 12

theorem dice_roll_probability :
  (probability_first_die_three * probability_second_die_odd = roll_probability) :=
by
  sorry

where
  probability_first_die_three : ℚ := 1 / 6
  probability_second_die_odd : ℚ := 1 / 2

end NUMINAMATH_CALUDE_dice_roll_probability_l1609_160964


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1609_160913

-- Define the number of male volunteers
def num_male : Nat := 4

-- Define the number of female volunteers
def num_female : Nat := 2

-- Define the number of elderly people
def num_elderly : Nat := 2

-- Define the total number of people
def total_people : Nat := num_male + num_female + num_elderly

-- Define the function to calculate the number of arrangements
def calculate_arrangements (n_male : Nat) (n_female : Nat) (n_elderly : Nat) : Nat :=
  -- Treat elderly people as one unit
  let n_units := n_male + 1
  -- Calculate arrangements of units
  let unit_arrangements := Nat.factorial n_units
  -- Calculate arrangements of elderly people themselves
  let elderly_arrangements := Nat.factorial n_elderly
  -- Calculate arrangements of female volunteers in the spaces between and around other people
  let female_arrangements := (n_units + 1) * n_units
  unit_arrangements * elderly_arrangements * female_arrangements

-- Theorem statement
theorem number_of_arrangements :
  calculate_arrangements num_male num_female num_elderly = 7200 := by
  sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1609_160913


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l1609_160986

theorem greatest_integer_with_gcf_five : ∃ n : ℕ, n < 200 ∧ Nat.gcd n 30 = 5 ∧ ∀ m : ℕ, m < 200 → Nat.gcd m 30 = 5 → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcf_five_l1609_160986


namespace NUMINAMATH_CALUDE_john_donation_increases_average_l1609_160919

/-- Represents the donation amounts of Alice, Bob, and Carol -/
structure Donations where
  alice : ℝ
  bob : ℝ
  carol : ℝ

/-- The conditions of the problem -/
def donation_conditions (d : Donations) : Prop :=
  d.alice > 0 ∧ d.bob > 0 ∧ d.carol > 0 ∧  -- Each student donated a positive amount
  d.alice ≠ d.bob ∧ d.alice ≠ d.carol ∧ d.bob ≠ d.carol ∧  -- Each student donated a different amount
  d.alice / d.bob = 3 / 2 ∧  -- Ratio of Alice's to Bob's donation is 3:2
  d.carol / d.bob = 5 / 2 ∧  -- Ratio of Carol's to Bob's donation is 5:2
  d.alice + d.bob = 120  -- Sum of Alice's and Bob's donations is $120

/-- John's donation -/
def john_donation (d : Donations) : ℝ :=
  240

/-- The theorem to be proved -/
theorem john_donation_increases_average (d : Donations) 
  (h : donation_conditions d) : 
  (d.alice + d.bob + d.carol + john_donation d) / 4 = 
  1.5 * (d.alice + d.bob + d.carol) / 3 := by
  sorry

end NUMINAMATH_CALUDE_john_donation_increases_average_l1609_160919


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1609_160946

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 7 →
  a * b = t - 6 * Complex.I →
  t = 9 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1609_160946


namespace NUMINAMATH_CALUDE_problem_solution_l1609_160992

/-- Equation I: 2x + y + z = 47, where x, y, z are positive integers -/
def equation_I (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ 2 * x + y + z = 47

/-- Equation II: 2x + y + z + w = 47, where x, y, z, w are positive integers -/
def equation_II (x y z w : ℕ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 2 * x + y + z + w = 47

/-- Consecutive integers -/
def consecutive (a b c : ℕ) : Prop :=
  b = a + 1 ∧ c = a + 2

/-- Four consecutive integers -/
def consecutive_four (a b c d : ℕ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3

/-- Consecutive even integers -/
def consecutive_even (a b c : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * (k + 1) ∧ c = 2 * (k + 2)

/-- Four consecutive even integers -/
def consecutive_even_four (a b c d : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k ∧ b = 2 * (k + 1) ∧ c = 2 * (k + 2) ∧ d = 2 * (k + 3)

/-- Four consecutive odd integers -/
def consecutive_odd_four (a b c d : ℕ) : Prop :=
  ∃ k : ℕ, a = 2 * k + 1 ∧ b = 2 * k + 3 ∧ c = 2 * k + 5 ∧ d = 2 * k + 7

theorem problem_solution :
  (∃ x y z : ℕ, equation_I x y z ∧ consecutive x y z) ∧
  (∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_four x y z w) ∧
  (¬ ∃ x y z : ℕ, equation_I x y z ∧ consecutive_even x y z) ∧
  (¬ ∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_even_four x y z w) ∧
  (¬ ∃ x y z w : ℕ, equation_II x y z w ∧ consecutive_odd_four x y z w) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1609_160992


namespace NUMINAMATH_CALUDE_parallelograms_in_hexagon_l1609_160920

/-- A regular hexagon -/
structure RegularHexagon where
  /-- The number of sides in a regular hexagon -/
  sides : Nat
  /-- The property that a regular hexagon has 6 sides -/
  has_six_sides : sides = 6

/-- A parallelogram formed by two adjacent equilateral triangles in a regular hexagon -/
structure Parallelogram (h : RegularHexagon) where

/-- The number of parallelograms in a regular hexagon -/
def num_parallelograms (h : RegularHexagon) : Nat :=
  h.sides

/-- Theorem: The number of parallelograms in a regular hexagon is 6 -/
theorem parallelograms_in_hexagon (h : RegularHexagon) : 
  num_parallelograms h = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallelograms_in_hexagon_l1609_160920


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1609_160956

theorem sin_cos_difference_equals_half : 
  Real.sin (137 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l1609_160956


namespace NUMINAMATH_CALUDE_direct_variation_with_constant_l1609_160910

/-- A function that varies directly as x plus a constant -/
def f (k c : ℝ) (x : ℝ) : ℝ := k * x + c

/-- Theorem stating that if f(5) = 10 and f(1) = 6, then f(7) = 12 -/
theorem direct_variation_with_constant 
  (k c : ℝ) 
  (h1 : f k c 5 = 10) 
  (h2 : f k c 1 = 6) : 
  f k c 7 = 12 := by
  sorry

#check direct_variation_with_constant

end NUMINAMATH_CALUDE_direct_variation_with_constant_l1609_160910


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_six_l1609_160937

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C,
    this theorem proves that under certain conditions, the perimeter is 6. -/
theorem triangle_perimeter_is_six 
  (a b c : ℝ) 
  (A B C : ℝ)
  (h1 : a * Real.cos C + Real.sqrt 3 * a * Real.sin C - b - c = 0)
  (h2 : a = 2)
  (h3 : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_six_l1609_160937


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1609_160901

open Real

theorem function_inequality_implies_a_bound (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ →
    (log x₁ + a * x₁^2 - (log x₂ + a * x₂^2)) / (x₁ - x₂) > 2) →
  a ≥ (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l1609_160901


namespace NUMINAMATH_CALUDE_three_digit_cube_divisible_by_eight_l1609_160957

theorem three_digit_cube_divisible_by_eight :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, n = m^3 ∧ 8 ∣ n := by sorry

end NUMINAMATH_CALUDE_three_digit_cube_divisible_by_eight_l1609_160957


namespace NUMINAMATH_CALUDE_triangle_side_length_l1609_160947

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  2 * b = a + c →  -- arithmetic sequence condition
  B = π / 6 →  -- 30° in radians
  (1 / 2) * a * c * Real.sin B = 3 / 2 →  -- area condition
  b = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1609_160947


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1609_160971

theorem polynomial_factorization (x : ℝ) :
  x^4 + 2021*x^2 + 2020*x + 2021 = (x^2 + x + 1)*(x^2 - x + 2021) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1609_160971


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l1609_160927

theorem cuboid_edge_length (x : ℝ) : 
  x > 0 → 2 * x * 3 = 30 → x = 5 := by sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l1609_160927


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1609_160929

/-- Given a rectangle where the length is three times the width and the diagonal is 8√10,
    prove that its perimeter is 64. -/
theorem rectangle_perimeter (w l d : ℝ) : 
  l = 3 * w →                 -- length is three times the width
  d = 8 * (10 : ℝ).sqrt →     -- diagonal is 8√10
  w * w + l * l = d * d →     -- Pythagorean theorem
  2 * (w + l) = 64 :=         -- perimeter is 64
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1609_160929


namespace NUMINAMATH_CALUDE_probability_sum_10_four_dice_l1609_160981

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The target sum we're looking for -/
def targetSum : ℕ := 10

/-- The total number of possible outcomes when throwing four dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (combinations that sum to 10) -/
def favorableOutcomes : ℕ := 46

/-- The probability of getting a sum of 10 when throwing four 6-sided dice -/
theorem probability_sum_10_four_dice : 
  (favorableOutcomes : ℚ) / totalOutcomes = 23 / 648 := by sorry

end NUMINAMATH_CALUDE_probability_sum_10_four_dice_l1609_160981


namespace NUMINAMATH_CALUDE_cow_husk_consumption_l1609_160922

/-- Given that 26 cows eat 26 bags of husk in 26 days, prove that one cow will eat one bag of husk in 26 days -/
theorem cow_husk_consumption (cows bags days : ℕ) (h : cows = 26 ∧ bags = 26 ∧ days = 26) :
  (1 : ℕ) * bags = (1 : ℕ) * cows * days := by
  sorry

end NUMINAMATH_CALUDE_cow_husk_consumption_l1609_160922


namespace NUMINAMATH_CALUDE_no_trapezoid_solution_l1609_160960

theorem no_trapezoid_solution : ¬∃ (b₁ b₂ : ℕ), 
  b₁ > 0 ∧ b₂ > 0 ∧
  b₁ % 12 = 0 ∧ b₂ % 12 = 0 ∧
  80 * (b₁ + b₂) / 2 = 2800 :=
sorry

end NUMINAMATH_CALUDE_no_trapezoid_solution_l1609_160960


namespace NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l1609_160905

theorem triangle_angle_not_all_greater_60 :
  ∀ (a b c : Real),
  (a > 0) → (b > 0) → (c > 0) →
  (a + b + c = 180) →
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_not_all_greater_60_l1609_160905


namespace NUMINAMATH_CALUDE_simplify_expression_l1609_160909

theorem simplify_expression (r : ℝ) : 180 * r - 88 * r = 92 * r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1609_160909


namespace NUMINAMATH_CALUDE_two_consecutive_sets_sum_100_l1609_160916

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_100 : start * length + (length * (length - 1)) / 2 = 100
  at_least_two : length ≥ 2

/-- The theorem stating that there are exactly two sets of consecutive positive integers
    whose sum is 100 and contain at least two integers -/
theorem two_consecutive_sets_sum_100 :
  ∃! (sets : Finset ConsecutiveSet), sets.card = 2 ∧ 
    (∀ s ∈ sets, s.start > 0 ∧ s.length ≥ 2 ∧ 
      s.start * s.length + (s.length * (s.length - 1)) / 2 = 100) :=
sorry

end NUMINAMATH_CALUDE_two_consecutive_sets_sum_100_l1609_160916


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1609_160975

universe u

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 4, 5}
def B : Set Nat := {1, 2, 5}

theorem complement_intersection_theorem : (U \ A) ∩ B = {1} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1609_160975


namespace NUMINAMATH_CALUDE_power_function_properties_l1609_160900

-- Define the power function f
noncomputable def f : ℝ → ℝ := λ x => Real.sqrt x

-- State the theorem
theorem power_function_properties :
  (f 9 = 3) →
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x₁ x₂, x₂ > x₁ ∧ x₁ > 0 → (f x₁ + f x₂) / 2 < f ((x₁ + x₂) / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_power_function_properties_l1609_160900


namespace NUMINAMATH_CALUDE_sibling_product_sixteen_l1609_160904

/-- Represents a family with a given number of girls and boys -/
structure Family :=
  (girls : ℕ)
  (boys : ℕ)

/-- Calculates the product of sisters and brothers for a member of the family -/
def siblingProduct (f : Family) : ℕ :=
  (f.girls - 1) * f.boys

/-- Theorem: In a family with 5 girls and 4 boys, the product of sisters and brothers is 16 -/
theorem sibling_product_sixteen (f : Family) (h1 : f.girls = 5) (h2 : f.boys = 4) :
  siblingProduct f = 16 := by
  sorry

end NUMINAMATH_CALUDE_sibling_product_sixteen_l1609_160904


namespace NUMINAMATH_CALUDE_droid_coffee_usage_l1609_160915

/-- The number of bags of coffee beans Droid uses in a week -/
def weekly_coffee_usage (morning_usage : ℕ) (days_per_week : ℕ) : ℕ :=
  let afternoon_usage := 3 * morning_usage
  let evening_usage := 2 * morning_usage
  let daily_usage := morning_usage + afternoon_usage + evening_usage
  daily_usage * days_per_week

/-- Theorem stating that Droid uses 126 bags of coffee beans per week -/
theorem droid_coffee_usage :
  weekly_coffee_usage 3 7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_droid_coffee_usage_l1609_160915


namespace NUMINAMATH_CALUDE_coefficient_x5_expansion_l1609_160918

/-- The coefficient of x^5 in the expansion of (2 + √x - x^2018/2017)^12 -/
def coefficient_x5 : ℕ :=
  -- Define the coefficient here
  264

/-- Theorem stating that the coefficient of x^5 in the expansion of (2 + √x - x^2018/2017)^12 is 264 -/
theorem coefficient_x5_expansion :
  coefficient_x5 = 264 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x5_expansion_l1609_160918


namespace NUMINAMATH_CALUDE_max_value_constraint_l1609_160902

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x + 3*y < 60) :
  xy*(60 - 4*x - 3*y) ≤ 2000/3 ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4*x₀ + 3*y₀ < 60 ∧ x₀*y₀*(60 - 4*x₀ - 3*y₀) = 2000/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l1609_160902
