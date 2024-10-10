import Mathlib

namespace square_side_length_l203_20394

theorem square_side_length : ∃ (x : ℝ), x > 0 ∧ x^2 = 6^2 + 8^2 :=
by
  -- The proof goes here
  sorry

end square_side_length_l203_20394


namespace geometric_sequence_property_l203_20363

/-- A geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)

/-- Theorem: In a geometric sequence where a₁a₈³a₁₅ = 243, the value of a₉³/a₁₁ is 9 -/
theorem geometric_sequence_property (seq : GeometricSequence) 
    (h : seq.a 1 * (seq.a 8)^3 * seq.a 15 = 243) :
    (seq.a 9)^3 / seq.a 11 = 9 := by
  sorry

end geometric_sequence_property_l203_20363


namespace binary_multiplication_theorem_l203_20371

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number as a list of bits (least significant bit first) -/
def binary_1101 : List Bool := [true, false, true, true]
def binary_111 : List Bool := [true, true, true]
def binary_10001111 : List Bool := [true, true, true, true, false, false, false, true]

theorem binary_multiplication_theorem :
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) =
  binary_to_decimal binary_10001111 ∧
  (binary_to_decimal binary_1101) * (binary_to_decimal binary_111) = 143 := by
  sorry

end binary_multiplication_theorem_l203_20371


namespace bacon_suggestion_count_l203_20350

theorem bacon_suggestion_count (mashed_and_bacon : ℕ) (only_bacon : ℕ) : 
  mashed_and_bacon = 218 → only_bacon = 351 → 
  mashed_and_bacon + only_bacon = 569 := by sorry

end bacon_suggestion_count_l203_20350


namespace triangle_minimum_product_l203_20376

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2c cos B = 2a + b and the area of the triangle is (√3/12)c, then ab ≥ 1/3 -/
theorem triangle_minimum_product (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  2 * c * Real.cos B = 2 * a + b →
  (1 / 2) * a * b * Real.sin C = (Real.sqrt 3 / 12) * c →
  a * b ≥ 1 / 3 := by
  sorry


end triangle_minimum_product_l203_20376


namespace sqrt_four_equals_two_l203_20329

theorem sqrt_four_equals_two : Real.sqrt 4 = 2 := by
  sorry

end sqrt_four_equals_two_l203_20329


namespace tangent_slopes_product_l203_20368

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the circle where P lies
def circle_P (x y : ℝ) : Prop := x^2 + y^2 = 7

-- Define the tangent line from P(x₀, y₀) to C with slope k
def tangent_line (x₀ y₀ k x y : ℝ) : Prop := y - y₀ = k * (x - x₀)

-- Define the condition for a line to be tangent to C
def is_tangent (x₀ y₀ k : ℝ) : Prop :=
  ∃ x y, tangent_line x₀ y₀ k x y ∧ ellipse_C x y

-- Main theorem
theorem tangent_slopes_product (x₀ y₀ k₁ k₂ : ℝ) :
  circle_P x₀ y₀ →
  is_tangent x₀ y₀ k₁ →
  is_tangent x₀ y₀ k₂ →
  k₁ ≠ k₂ →
  k₁ * k₂ = -1 :=
sorry

end tangent_slopes_product_l203_20368


namespace nh4cl_formation_l203_20309

-- Define the chemical species
inductive ChemicalSpecies
| NH3
| HCl
| NH4Cl

-- Define a type for chemical reactions
structure Reaction where
  reactants : List (ChemicalSpecies × ℚ)
  products : List (ChemicalSpecies × ℚ)

-- Define the specific reaction
def nh3_hcl_reaction : Reaction :=
  { reactants := [(ChemicalSpecies.NH3, 1), (ChemicalSpecies.HCl, 1)],
    products := [(ChemicalSpecies.NH4Cl, 1)] }

-- Define the available amounts of reactants
def available_nh3 : ℚ := 1
def available_hcl : ℚ := 1

-- Theorem statement
theorem nh4cl_formation :
  let reaction := nh3_hcl_reaction
  let nh3_amount := available_nh3
  let hcl_amount := available_hcl
  let nh4cl_formed := 1
  nh4cl_formed = min nh3_amount hcl_amount := by sorry

end nh4cl_formation_l203_20309


namespace trig_identity_l203_20393

theorem trig_identity (α : Real) (h : Real.sin (π/3 - α) = 1/3) :
  Real.cos (π/3 + 2*α) = -7/9 := by
  sorry

end trig_identity_l203_20393


namespace equal_dice_probability_l203_20362

/-- The number of dice being rolled -/
def num_dice : ℕ := 5

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The probability of a single die showing a number less than or equal to 10 -/
def prob_le_10 : ℚ := 1/2

/-- The probability of a single die showing a number greater than 10 -/
def prob_gt_10 : ℚ := 1/2

/-- The number of ways to choose dice showing numbers less than or equal to 10 -/
def ways_to_choose : ℕ := Nat.choose num_dice (num_dice / 2)

/-- The theorem stating the probability of rolling an equal number of dice showing
    numbers less than or equal to 10 as showing numbers greater than 10 -/
theorem equal_dice_probability :
  (2 * ways_to_choose : ℚ) * (prob_le_10 ^ num_dice) = 5/8 := by sorry

end equal_dice_probability_l203_20362


namespace inequalities_proof_l203_20330

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) ≥ Real.sqrt a + Real.sqrt b ∧
  (a + b = 1 → (1/a) + (1/b) + (1/(a*b)) ≥ 8) := by
  sorry

end inequalities_proof_l203_20330


namespace meal_cost_l203_20361

-- Define variables for the cost of each item
variable (s : ℝ) -- cost of one sandwich
variable (c : ℝ) -- cost of one cup of coffee
variable (p : ℝ) -- cost of one piece of pie

-- Define the given equations
def equation1 : Prop := 5 * s + 8 * c + p = 5
def equation2 : Prop := 7 * s + 12 * c + p = 7.2
def equation3 : Prop := 4 * s + 6 * c + 2 * p = 6

-- Theorem to prove
theorem meal_cost (h1 : equation1 s c p) (h2 : equation2 s c p) (h3 : equation3 s c p) :
  s + c + p = 1.9 := by sorry

end meal_cost_l203_20361


namespace problem_solution_l203_20304

-- Define the line l: x + my + 2√3 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := x + m * y + 2 * Real.sqrt 3 = 0

-- Define the circle O: x² + y² = r² (r > 0)
def circle_O (r : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = r^2 ∧ r > 0

-- Define line l': x = 3
def line_l' (x : ℝ) : Prop := x = 3

theorem problem_solution :
  -- Part 1
  (∀ r : ℝ, (∀ m : ℝ, ∃ x y : ℝ, line_l m x y ∧ circle_O r x y) ↔ r ≥ 2 * Real.sqrt 3) ∧
  -- Part 2
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    circle_O 5 x₁ y₁ ∧ circle_O 5 x₂ y₂ ∧ 
    (∃ m : ℝ, line_l m x₁ y₁ ∧ line_l m x₂ y₂) →
    2 * Real.sqrt 13 ≤ Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≤ 10) ∧
  -- Part 3
  (∀ s t : ℝ, s^2 + t^2 = 1 →
    ∃ x y : ℝ, 
      (x - 3)^2 + (y - (1 - 3*s)/t)^2 = ((3 - s)/t)^2 ∧
      (x = 3 + 2 * Real.sqrt 2 ∧ y = 0 ∨ x = 3 - 2 * Real.sqrt 2 ∧ y = 0)) :=
by sorry

end problem_solution_l203_20304


namespace smallest_staircase_steps_l203_20353

theorem smallest_staircase_steps : ∃ n : ℕ,
  n > 20 ∧
  n % 5 = 4 ∧
  n % 6 = 3 ∧
  n % 7 = 5 ∧
  (∀ m : ℕ, m > 20 → m % 5 = 4 → m % 6 = 3 → m % 7 = 5 → m ≥ n) ∧
  n = 159 :=
by sorry

end smallest_staircase_steps_l203_20353


namespace composite_condition_l203_20320

def is_composite (m : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ m = a * b

theorem composite_condition (n : ℕ) (hn : 0 < n) : 
  is_composite (3^(2*n+1) - 2^(2*n+1) - 6*n) ↔ n > 1 :=
sorry

end composite_condition_l203_20320


namespace rational_equation_solution_l203_20367

theorem rational_equation_solution (x : ℝ) : 
  (x^2 - 7*x + 10) / (x^2 - 9*x + 8) = (x^2 - 3*x - 18) / (x^2 - 4*x - 21) ↔ x = 11 ∨ x = -2 :=
by sorry

end rational_equation_solution_l203_20367


namespace A_inter_B_eq_two_three_l203_20359

def A : Set ℕ := {x | (x - 2) * (x - 4) ≤ 0}

def B : Set ℕ := {x | x ≤ 3}

theorem A_inter_B_eq_two_three : A ∩ B = {2, 3} := by
  sorry

end A_inter_B_eq_two_three_l203_20359


namespace pizza_problem_solution_l203_20395

/-- Represents the number of slices in a pizza --/
structure PizzaSlices where
  small : ℕ
  large : ℕ

/-- Represents the number of pizzas purchased --/
structure PizzasPurchased where
  small : ℕ
  large : ℕ

/-- Represents the number of slices eaten by each person --/
structure SlicesEaten where
  george : ℕ
  bob : ℕ
  susie : ℕ
  bill : ℕ
  fred : ℕ
  mark : ℕ

def pizza_problem (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) (leftover : ℕ) : Prop :=
  slices.small = 4 ∧
  slices.large = 8 ∧
  purchased.large = 2 ∧
  eaten.george = 3 ∧
  eaten.bob = eaten.george + 1 ∧
  eaten.susie = eaten.bob / 2 ∧
  eaten.bill = 3 ∧
  eaten.fred = 3 ∧
  eaten.mark = 3 ∧
  leftover = 10 ∧
  purchased.small * slices.small + purchased.large * slices.large =
    eaten.george + eaten.bob + eaten.susie + eaten.bill + eaten.fred + eaten.mark + leftover

theorem pizza_problem_solution 
  (slices : PizzaSlices) (purchased : PizzasPurchased) (eaten : SlicesEaten) (leftover : ℕ) :
  pizza_problem slices purchased eaten leftover → purchased.small = 3 := by
  sorry

end pizza_problem_solution_l203_20395


namespace investment_final_values_l203_20379

/-- Calculates the final value of an investment after two years --/
def final_value (initial : ℝ) (year1_change : ℝ) (year1_dividend : ℝ) (year2_change : ℝ) : ℝ :=
  (initial * (1 + year1_change) + initial * year1_dividend) * (1 + year2_change)

/-- Proves that the final values of investments D, E, and F are correct --/
theorem investment_final_values :
  let d := final_value 100 0 0.1 0.05
  let e := final_value 100 0.3 0 (-0.1)
  let f := final_value 100 (-0.1) 0 0.2
  d = 115.5 ∧ e = 117 ∧ f = 108 :=
by sorry

#eval final_value 100 0 0.1 0.05
#eval final_value 100 0.3 0 (-0.1)
#eval final_value 100 (-0.1) 0 0.2

end investment_final_values_l203_20379


namespace negation_of_all_divisible_by_five_are_odd_l203_20357

theorem negation_of_all_divisible_by_five_are_odd :
  ¬(∀ n : ℤ, 5 ∣ n → Odd n) ↔ ∃ n : ℤ, 5 ∣ n ∧ ¬(Odd n) :=
by sorry

end negation_of_all_divisible_by_five_are_odd_l203_20357


namespace monochromatic_triangle_in_K17_l203_20321

/-- A coloring of the edges of a complete graph -/
def Coloring (n : ℕ) := Fin n → Fin n → Fin 3

/-- A triangle in a graph -/
def Triangle (n : ℕ) := { t : Fin n × Fin n × Fin n // t.1 ≠ t.2.1 ∧ t.1 ≠ t.2.2 ∧ t.2.1 ≠ t.2.2 }

/-- A monochromatic triangle in a coloring -/
def MonochromaticTriangle (n : ℕ) (c : Coloring n) (t : Triangle n) : Prop :=
  c t.val.1 t.val.2.1 = c t.val.1 t.val.2.2 ∧ c t.val.1 t.val.2.2 = c t.val.2.1 t.val.2.2

/-- The main theorem: any 3-coloring of K₁₇ contains a monochromatic triangle -/
theorem monochromatic_triangle_in_K17 :
  ∀ c : Coloring 17, ∃ t : Triangle 17, MonochromaticTriangle 17 c t := by
  sorry

end monochromatic_triangle_in_K17_l203_20321


namespace razorback_tshirt_sales_l203_20331

/-- The Razorback t-shirt shop problem -/
theorem razorback_tshirt_sales
  (original_price : ℕ)
  (discount : ℕ)
  (num_sold : ℕ)
  (h1 : original_price = 51)
  (h2 : discount = 8)
  (h3 : num_sold = 130) :
  (original_price - discount) * num_sold = 5590 :=
by sorry

end razorback_tshirt_sales_l203_20331


namespace circle_radius_l203_20352

/-- The radius of a circle given by the equation x^2 + y^2 - 4x + 2y - 4 = 0 is 3 -/
theorem circle_radius (x y : ℝ) : 
  x^2 + y^2 - 4*x + 2*y - 4 = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 3^2 := by
  sorry

end circle_radius_l203_20352


namespace repeating_decimal_equals_fraction_l203_20337

/-- Represents the repeating decimal 0.37246̅ -/
def repeating_decimal : ℚ := 37246 / 100000 + (246 / 100000) / (1 - 1/1000)

/-- The fraction we want to prove equality with -/
def target_fraction : ℚ := 37187378 / 99900

/-- Theorem stating that the repeating decimal is equal to the target fraction -/
theorem repeating_decimal_equals_fraction : repeating_decimal = target_fraction := by
  sorry

end repeating_decimal_equals_fraction_l203_20337


namespace white_surface_area_fraction_is_three_fourths_l203_20312

/-- Represents a cube made of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculates the fraction of white surface area for a composite cube -/
def white_surface_area_fraction (c : CompositeCube) : ℚ :=
  sorry

/-- Theorem: The fraction of white surface area for a specific composite cube is 3/4 -/
theorem white_surface_area_fraction_is_three_fourths :
  let c : CompositeCube := {
    edge_length := 4,
    small_cube_count := 64,
    white_cube_count := 48,
    black_cube_count := 16
  }
  white_surface_area_fraction c = 3/4 := by
  sorry

end white_surface_area_fraction_is_three_fourths_l203_20312


namespace abc_inequality_l203_20364

theorem abc_inequality : 
  let a : ℝ := (3/4)^(2/3)
  let b : ℝ := (2/3)^(3/4)
  let c : ℝ := Real.log (4/3) / Real.log (2/3)
  a > b ∧ b > c := by sorry

end abc_inequality_l203_20364


namespace car_arrives_earlier_l203_20336

/-- Represents a vehicle (car or bus) -/
inductive Vehicle
| Car
| Bus

/-- Represents the state of a traffic light -/
inductive LightState
| Green
| Red

/-- Calculates the travel time for a vehicle given the number of blocks -/
def travelTime (v : Vehicle) (blocks : ℕ) : ℕ :=
  match v with
  | Vehicle.Car => blocks
  | Vehicle.Bus => 2 * blocks

/-- Calculates the number of complete light cycles for a given time -/
def completeLightCycles (time : ℕ) : ℕ :=
  time / 4

/-- Calculates the waiting time at red lights for a given travel time -/
def waitingTime (time : ℕ) : ℕ :=
  completeLightCycles time

/-- Calculates the total time to reach the destination for a vehicle -/
def totalTime (v : Vehicle) (blocks : ℕ) : ℕ :=
  let travel := travelTime v blocks
  travel + waitingTime travel

/-- The main theorem to prove -/
theorem car_arrives_earlier (blocks : ℕ) (h : blocks = 12) :
  totalTime Vehicle.Car blocks + 9 = totalTime Vehicle.Bus blocks :=
by sorry

end car_arrives_earlier_l203_20336


namespace students_passed_both_tests_l203_20314

theorem students_passed_both_tests
  (total_students : ℕ)
  (passed_long_jump : ℕ)
  (passed_shot_put : ℕ)
  (failed_both : ℕ)
  (h1 : total_students = 50)
  (h2 : passed_long_jump = 40)
  (h3 : passed_shot_put = 31)
  (h4 : failed_both = 4) :
  total_students - failed_both = passed_long_jump + passed_shot_put - (passed_long_jump + passed_shot_put - (total_students - failed_both)) :=
by sorry

end students_passed_both_tests_l203_20314


namespace function_inequality_equivalence_l203_20351

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x

theorem function_inequality_equivalence (a : ℝ) :
  (a > 0) →
  (∀ m n : ℝ, m > 0 → n > 0 → m ≠ n →
    Real.sqrt (m * n) + (m + n) / 2 > (m - n) / (f a m - f a n)) ↔
  a ≥ 1 / 2 :=
by sorry

end function_inequality_equivalence_l203_20351


namespace age_difference_l203_20377

theorem age_difference (a b c d : ℕ) : 
  (a + b = b + c + 13) →
  (b + d = c + d + 7) →
  (a + d = 2 * c - 12) →
  (a = c + 13) :=
by
  sorry

end age_difference_l203_20377


namespace probability_three_two_correct_l203_20388

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def different_numbers : ℕ := 10

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The probability of drawing exactly 3 slips with one number and 2 slips with another number -/
def probability_three_two : ℚ := 75 / 35313

theorem probability_three_two_correct :
  probability_three_two = (different_numbers.choose 2 * slips_per_number.choose 3 * slips_per_number.choose 2) / total_slips.choose drawn_slips :=
by sorry

end probability_three_two_correct_l203_20388


namespace michael_basketball_points_l203_20399

theorem michael_basketball_points :
  ∀ (junior_points : ℝ),
    (junior_points + (junior_points * 1.2) = 572) →
    junior_points = 260 := by
  sorry

end michael_basketball_points_l203_20399


namespace smallest_angle_satisfies_equation_l203_20386

/-- The smallest positive angle (in degrees) that satisfies the given equation -/
noncomputable def smallest_angle : ℝ :=
  (1 / 4) * Real.arcsin (2 / 9) * (180 / Real.pi)

/-- The equation that the angle must satisfy -/
def equation (x : ℝ) : Prop :=
  9 * Real.sin x * (Real.cos x)^7 - 9 * (Real.sin x)^7 * Real.cos x = 1

theorem smallest_angle_satisfies_equation :
  equation (smallest_angle * (Real.pi / 180)) ∧
  ∀ y, 0 < y ∧ y < smallest_angle → ¬equation (y * (Real.pi / 180)) :=
by sorry

end smallest_angle_satisfies_equation_l203_20386


namespace square_area_five_parts_l203_20370

/-- Given a square divided into five equal areas with side AB of length 3.6 cm,
    the total area of the square is 1156 square centimeters. -/
theorem square_area_five_parts (s : ℝ) (h1 : s > 0) :
  let ab : ℝ := 3.6
  let area : ℝ := s^2
  (∃ (x : ℝ), ab = s * x ∧ x > 0 ∧ x < 1 ∧ 5 * (s * x)^2 = area) →
  area = 1156 := by
sorry

end square_area_five_parts_l203_20370


namespace tiffany_lives_l203_20397

theorem tiffany_lives (x : ℕ) : 
  (x - 14 + 27 = 56) → x = 43 := by
  sorry

end tiffany_lives_l203_20397


namespace combined_area_of_triangle_and_square_l203_20324

theorem combined_area_of_triangle_and_square (triangle_area : ℝ) (base_length : ℝ) : 
  triangle_area = 720 → 
  base_length = 40 → 
  (triangle_area = 1/2 * base_length * (triangle_area / (1/2 * base_length))) →
  (base_length^2 + triangle_area = 2320) := by
sorry

end combined_area_of_triangle_and_square_l203_20324


namespace customers_before_correct_l203_20323

/-- The number of customers before the lunch rush -/
def customers_before : ℝ := 29.0

/-- The number of customers added during the lunch rush -/
def customers_added_lunch : ℝ := 20.0

/-- The number of customers that came in after the lunch rush -/
def customers_after_lunch : ℝ := 34.0

/-- The total number of customers after all additions -/
def total_customers : ℝ := 83.0

/-- Theorem stating that the number of customers before the lunch rush is correct -/
theorem customers_before_correct :
  customers_before + customers_added_lunch + customers_after_lunch = total_customers :=
by sorry

end customers_before_correct_l203_20323


namespace intersection_of_A_and_B_l203_20302

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {y | ∃ x ∈ A, y = x + 1}

theorem intersection_of_A_and_B : A ∩ B = {2, 3, 4} := by
  sorry

end intersection_of_A_and_B_l203_20302


namespace vector_problem_l203_20398

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-1, 3]

def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ (c : ℝ), v = fun i => c * (w i)

theorem vector_problem :
  (∃ k : ℝ, perpendicular (fun i => k * (a i) + (b i)) (fun i => (a i) - 3 * (b i)) ∧ k = -2.5) ∧
  (∃ k : ℝ, parallel (fun i => k * (a i) + (b i)) (fun i => (a i) - 3 * (b i)) ∧ k = -1/3) :=
sorry

end vector_problem_l203_20398


namespace factorization_of_4x_squared_minus_16_l203_20303

theorem factorization_of_4x_squared_minus_16 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_4x_squared_minus_16_l203_20303


namespace no_real_roots_l203_20300

theorem no_real_roots : ¬∃ x : ℝ, Real.sqrt (x + 9) + Real.sqrt (x - 2) = 3 := by
  sorry

end no_real_roots_l203_20300


namespace sum_of_fractions_l203_20378

theorem sum_of_fractions : (2 : ℚ) / 7 + 8 / 10 = 38 / 35 := by sorry

end sum_of_fractions_l203_20378


namespace inequality_theorem_l203_20315

theorem inequality_theorem (p q r : ℝ) : 
  (∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x < -6 ∨ |x - 20| ≤ 2) →
  p < q →
  p + 2*q + 3*r = 44 := by
  sorry

end inequality_theorem_l203_20315


namespace tablecloth_width_l203_20346

/-- Given a rectangular tablecloth and napkins with specified dimensions,
    prove that the width of the tablecloth is 54 inches. -/
theorem tablecloth_width
  (tablecloth_length : ℕ)
  (napkin_length napkin_width : ℕ)
  (num_napkins : ℕ)
  (total_area : ℕ)
  (h1 : tablecloth_length = 102)
  (h2 : napkin_length = 6)
  (h3 : napkin_width = 7)
  (h4 : num_napkins = 8)
  (h5 : total_area = 5844) :
  total_area - num_napkins * napkin_length * napkin_width = 54 * tablecloth_length :=
by sorry

end tablecloth_width_l203_20346


namespace train_crossing_time_l203_20369

/-- Given a train and platform with specific dimensions and time to pass the platform,
    calculate the time it takes for the train to cross a point object (tree). -/
theorem train_crossing_time (train_length platform_length time_to_pass_platform : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 500)
  (h3 : time_to_pass_platform = 170) :
  (train_length / ((train_length + platform_length) / time_to_pass_platform)) = 120 :=
by sorry

end train_crossing_time_l203_20369


namespace time_difference_per_mile_l203_20383

-- Define the given conditions
def young_girl_distance : ℝ := 18  -- miles
def young_girl_time : ℝ := 135     -- minutes (2 hours and 15 minutes)
def current_distance : ℝ := 12     -- miles
def current_time : ℝ := 300        -- minutes (5 hours)

-- Define the theorem
theorem time_difference_per_mile : 
  (current_time / current_distance) - (young_girl_time / young_girl_distance) = 17.5 := by
  sorry

end time_difference_per_mile_l203_20383


namespace range_of_a_l203_20338

theorem range_of_a (x y z a : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 1) (heq : a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  0 < a ∧ a ≤ 7 / 27 :=
by sorry

end range_of_a_l203_20338


namespace factorization_proof_l203_20372

theorem factorization_proof (x : ℝ) : 75 * x^11 + 225 * x^22 = 75 * x^11 * (1 + 3 * x^11) := by
  sorry

end factorization_proof_l203_20372


namespace gwen_birthday_money_l203_20335

/-- Calculates the remaining money after spending -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Proves that Gwen has 5 dollars left after spending 2 dollars from her initial 7 dollars -/
theorem gwen_birthday_money : remaining_money 7 2 = 5 := by
  sorry

end gwen_birthday_money_l203_20335


namespace train_speed_l203_20308

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length time : ℝ) (h1 : length = 320) (h2 : time = 16) :
  length / time = 20 := by
  sorry

end train_speed_l203_20308


namespace circle_equation_AB_l203_20356

/-- Given two points A and B, this function returns the equation of the circle
    with AB as its diameter in the form (x - h)² + (y - k)² = r², where
    (h, k) is the center of the circle and r is its radius. -/
def circle_equation_with_diameter (A B : ℝ × ℝ) : (ℝ → ℝ → Prop) :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let h := (x₁ + x₂) / 2
  let k := (y₁ + y₂) / 2
  let r := ((x₁ - x₂)^2 + (y₁ - y₂)^2).sqrt / 2
  fun x y => (x - h)^2 + (y - k)^2 = r^2

/-- Theorem stating that for points A(3, -2) and B(-5, 4), the equation of the circle
    with AB as its diameter is (x + 1)² + (y - 1)² = 25. -/
theorem circle_equation_AB : 
  circle_equation_with_diameter (3, -2) (-5, 4) = fun x y => (x + 1)^2 + (y - 1)^2 = 25 :=
by sorry

end circle_equation_AB_l203_20356


namespace min_value_ab_l203_20328

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + b + 3) : 
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 3 ∧ a₀ * b₀ = 9 :=
by sorry

end min_value_ab_l203_20328


namespace number_ratio_l203_20360

theorem number_ratio (x : ℝ) (h : x + 5 = 17) : x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_l203_20360


namespace two_letter_selection_count_l203_20382

def word : String := "УЧЕБНИК"

def is_vowel (c : Char) : Bool :=
  c = 'У' || c = 'Е' || c = 'И'

def is_consonant (c : Char) : Bool :=
  c = 'Ч' || c = 'Б' || c = 'Н' || c = 'К'

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def count_consonants (s : String) : Nat :=
  s.toList.filter is_consonant |>.length

theorem two_letter_selection_count :
  count_vowels word * count_consonants word = 12 :=
by sorry

end two_letter_selection_count_l203_20382


namespace arithmetic_sequence_sum_l203_20345

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 2 + a 4 = 6 → a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end arithmetic_sequence_sum_l203_20345


namespace complex_equality_l203_20385

theorem complex_equality (z : ℂ) : Complex.abs (z + 2) = Complex.abs (z - 3) → z = 1/2 := by
  sorry

end complex_equality_l203_20385


namespace circle_equation_through_points_l203_20396

theorem circle_equation_through_points : 
  let circle_eq := (fun (x y : ℝ) => x^2 + y^2 - 4*x - 6*y)
  (circle_eq 0 0 = 0) ∧ 
  (circle_eq 4 0 = 0) ∧ 
  (circle_eq (-1) 1 = 0) := by
  sorry

end circle_equation_through_points_l203_20396


namespace division_problem_l203_20307

theorem division_problem (a b q : ℕ) (h1 : a - b = 1360) (h2 : a = 1614) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end division_problem_l203_20307


namespace least_multiple_17_above_500_l203_20380

theorem least_multiple_17_above_500 : ∃ (n : ℕ), n * 17 = 510 ∧ 
  510 > 500 ∧ 
  (∀ m : ℕ, m * 17 > 500 → m * 17 ≥ 510) := by
  sorry

end least_multiple_17_above_500_l203_20380


namespace equation_solution_range_l203_20343

theorem equation_solution_range (x m : ℝ) : 
  x + 3 = 3 * x - m → x ≥ 0 → m ≥ -3 := by
  sorry

end equation_solution_range_l203_20343


namespace min_b_over_a_l203_20354

theorem min_b_over_a (a b : ℝ) (h : ∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + b) : 
  (∀ c : ℝ, (∀ x > -1, Real.log (x + 1) - 1 ≤ a * x + c) → b / a ≤ c / a) → b / a = 1 - Real.exp 1 :=
sorry

end min_b_over_a_l203_20354


namespace expression_value_l203_20317

theorem expression_value (m : ℝ) (h : m^2 - m = 1) : 
  (m - 1)^2 + (m + 1)*(m - 1) + 2022 = 2024 := by
  sorry

end expression_value_l203_20317


namespace parabola_y_intercept_l203_20342

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  (∀ x y, y = x^2 + b*x + c → 
    ((x = 2 ∧ y = 5) ∨ (x = 4 ∧ y = 9))) → 
  c = 9 := by
  sorry

end parabola_y_intercept_l203_20342


namespace third_divisor_is_three_l203_20344

def smallest_number : ℕ := 1011
def diminished_number : ℕ := smallest_number - 3

theorem third_divisor_is_three :
  ∃ (x : ℕ), x ≠ 12 ∧ x ≠ 16 ∧ x ≠ 21 ∧ x ≠ 28 ∧
  diminished_number % 12 = 0 ∧
  diminished_number % 16 = 0 ∧
  diminished_number % x = 0 ∧
  diminished_number % 21 = 0 ∧
  diminished_number % 28 = 0 ∧
  x = 3 :=
by sorry

end third_divisor_is_three_l203_20344


namespace equal_implies_parallel_l203_20373

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem equal_implies_parallel (a b : V) : a = b → parallel a b := by
  sorry

end equal_implies_parallel_l203_20373


namespace ollie_fewer_than_angus_l203_20391

/-- The number of fish caught by Patrick -/
def patrick_fish : ℕ := 8

/-- The number of fish caught by Ollie -/
def ollie_fish : ℕ := 5

/-- The number of fish caught by Angus -/
def angus_fish : ℕ := patrick_fish + 4

/-- The difference between Angus's and Ollie's fish catch -/
def fish_difference : ℕ := angus_fish - ollie_fish

theorem ollie_fewer_than_angus : fish_difference = 7 := by sorry

end ollie_fewer_than_angus_l203_20391


namespace modular_inverse_of_5_mod_35_l203_20305

theorem modular_inverse_of_5_mod_35 : 
  ∃ x : ℕ, x < 35 ∧ (5 * x) % 35 = 1 :=
by
  use 29
  sorry

end modular_inverse_of_5_mod_35_l203_20305


namespace smallest_number_divisible_l203_20390

theorem smallest_number_divisible (n : ℕ) : n ≥ 62 →
  (∃ (k : ℕ), n - 8 = 18 * k ∧ n - 8 ≥ 44) →
  (∀ (m : ℕ), m < n →
    ¬(∃ (l : ℕ), m - 8 = 18 * l ∧ m - 8 ≥ 44)) :=
by sorry

end smallest_number_divisible_l203_20390


namespace expression_evaluation_l203_20301

theorem expression_evaluation (x y z w : ℝ) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w := by
  sorry

end expression_evaluation_l203_20301


namespace tadpoles_kept_l203_20384

theorem tadpoles_kept (total : ℕ) (release_percentage : ℚ) (kept : ℕ) : 
  total = 180 → 
  release_percentage = 75 / 100 → 
  kept = total - (release_percentage * total).floor → 
  kept = 45 :=
by
  sorry

end tadpoles_kept_l203_20384


namespace gcf_of_75_and_135_l203_20347

theorem gcf_of_75_and_135 : Nat.gcd 75 135 = 15 := by
  sorry

end gcf_of_75_and_135_l203_20347


namespace condition_relationship_l203_20316

theorem condition_relationship :
  (∀ x : ℝ, |x - 1| ≤ 1 → 2 - x ≥ 0) ∧
  (∃ x : ℝ, 2 - x ≥ 0 ∧ |x - 1| > 1) :=
by sorry

end condition_relationship_l203_20316


namespace geometric_sequence_second_term_l203_20358

theorem geometric_sequence_second_term
  (a : ℕ → ℕ)  -- Sequence of natural numbers
  (h1 : a 1 = 1)  -- First term is 1
  (h2 : a 3 = 9)  -- Third term is 9
  (h_ratio : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 3 * a n)  -- Common ratio is 3
  : a 2 = 3 := by
  sorry

end geometric_sequence_second_term_l203_20358


namespace symmetric_points_sum_l203_20313

/-- Two points are symmetric with respect to the y-axis if their y-coordinates are equal
    and their x-coordinates are opposite in sign and equal in magnitude. -/
def symmetric_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  y₁ = y₂ ∧ x₁ = -x₂

/-- Given that point A(a,1) is symmetric to point A'(5,b) with respect to the y-axis,
    prove that a + b = -4. -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_y_axis a 1 5 b → a + b = -4 := by
  sorry

end symmetric_points_sum_l203_20313


namespace range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l203_20333

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - a) * (x - 3 * a) < 0 ∧ a > 0

def q (x : ℝ) : Prop := x^2 - 5*x + 6 < 0

-- Theorem 1: Range of x when a = 1
theorem range_of_x_when_a_is_one :
  ∃ (lower upper : ℝ), lower = 2 ∧ upper = 3 ∧
  ∀ x, p x 1 ∧ q x ↔ lower < x ∧ x < upper :=
sorry

-- Theorem 2: Range of a when p is necessary but not sufficient for q
theorem range_of_a_necessary_not_sufficient :
  ∃ (lower upper : ℝ), lower = 1 ∧ upper = 2 ∧
  ∀ a, (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) ↔ lower ≤ a ∧ a ≤ upper :=
sorry

end range_of_x_when_a_is_one_range_of_a_necessary_not_sufficient_l203_20333


namespace absolute_value_inequality_l203_20375

theorem absolute_value_inequality (x y z : ℝ) : 
  |x| + |y| + |z| ≤ |x + y - z| + |x - y + z| + |-x + y + z| := by
  sorry

end absolute_value_inequality_l203_20375


namespace investment_rate_proof_l203_20389

/-- Calculates the simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_proof (principal : ℝ) (time : ℝ) (rate : ℝ) :
  principal = 7000 →
  time = 2 →
  simpleInterest principal rate time = simpleInterest principal 0.12 time + 420 →
  rate = 0.15 := by
sorry

end investment_rate_proof_l203_20389


namespace unique_number_with_digit_sum_l203_20348

/-- Given a natural number n, returns the sum of its digits. -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number n is a three-digit number. -/
def isThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem unique_number_with_digit_sum : 
  ∃! n : ℕ, isThreeDigitNumber n ∧ n + sumOfDigits n = 328 := by sorry

end unique_number_with_digit_sum_l203_20348


namespace cubic_sum_problem_l203_20340

theorem cubic_sum_problem (a b c : ℂ) 
  (sum_condition : a + b + c = 2)
  (product_sum_condition : a * b + a * c + b * c = -1)
  (product_condition : a * b * c = -8) :
  a^3 + b^3 + c^3 = 69 := by
sorry

end cubic_sum_problem_l203_20340


namespace typhoon_fallen_trees_l203_20392

/-- Represents the number of trees that fell during a typhoon --/
structure FallenTrees where
  narra : ℕ
  mahogany : ℕ

/-- Represents the initial and final state of trees on the farm --/
structure FarmState where
  initialNarra : ℕ
  initialMahogany : ℕ
  finalTotal : ℕ

def replantedTrees (fallen : FallenTrees) : ℕ :=
  2 * fallen.narra + 3 * fallen.mahogany

theorem typhoon_fallen_trees (farm : FarmState) 
  (h1 : farm.initialNarra = 30)
  (h2 : farm.initialMahogany = 50)
  (h3 : farm.finalTotal = 88) :
  ∃ (fallen : FallenTrees),
    fallen.mahogany = fallen.narra + 1 ∧
    farm.finalTotal = 
      farm.initialNarra + farm.initialMahogany - 
      (fallen.narra + fallen.mahogany) + 
      replantedTrees fallen ∧
    fallen.narra + fallen.mahogany = 5 :=
  sorry


end typhoon_fallen_trees_l203_20392


namespace maximize_product_l203_20326

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 50) :
  x^7 * y^3 ≤ 35^7 * 15^3 ∧
  (x^7 * y^3 = 35^7 * 15^3 ↔ x = 35 ∧ y = 15) :=
sorry

end maximize_product_l203_20326


namespace kids_left_playing_l203_20310

theorem kids_left_playing (initial_kids : ℝ) (kids_going_home : ℝ) :
  initial_kids = 22.0 →
  kids_going_home = 14.0 →
  initial_kids - kids_going_home = 8.0 := by
  sorry

end kids_left_playing_l203_20310


namespace geography_english_sum_l203_20339

/-- Represents Henry's test scores -/
structure TestScores where
  geography : ℝ
  math : ℝ
  english : ℝ
  history : ℝ

/-- Henry's test scores satisfy the given conditions -/
def satisfiesConditions (scores : TestScores) : Prop :=
  scores.math = 70 ∧
  scores.history = (scores.geography + scores.math + scores.english) / 3 ∧
  scores.geography + scores.math + scores.english + scores.history = 248

/-- The sum of Henry's Geography and English scores is 116 -/
theorem geography_english_sum (scores : TestScores) 
  (h : satisfiesConditions scores) : scores.geography + scores.english = 116 := by
  sorry

end geography_english_sum_l203_20339


namespace quadratic_equation_roots_l203_20365

/-- Given a quadratic equation x^2 + mx - 2 = 0 where -1 is a root,
    prove that m = -1 and the other root is 2 -/
theorem quadratic_equation_roots (m : ℝ) : 
  ((-1 : ℝ)^2 + m*(-1) - 2 = 0) → 
  (m = -1 ∧ ∃ r : ℝ, r ≠ -1 ∧ r^2 + m*r - 2 = 0 ∧ r = 2) :=
by sorry

end quadratic_equation_roots_l203_20365


namespace negation_of_forall_x_squared_gt_x_l203_20381

theorem negation_of_forall_x_squared_gt_x :
  (¬ ∀ x : ℕ, x^2 > x) ↔ (∃ x₀ : ℕ, x₀^2 ≤ x₀) := by
  sorry

end negation_of_forall_x_squared_gt_x_l203_20381


namespace fraction_of_7000_l203_20306

theorem fraction_of_7000 (x : ℝ) : x = 0.101 →
  x * 7000 - (1 / 1000) * 7000 = 700 := by
  sorry

end fraction_of_7000_l203_20306


namespace arithmetic_sequence_20th_term_l203_20387

theorem arithmetic_sequence_20th_term (a : ℕ → ℤ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_sum_odd : a 1 + a 3 + a 5 = 105)
  (h_sum_even : a 2 + a 4 + a 6 = 99) :
  a 20 = 1 := by
sorry

end arithmetic_sequence_20th_term_l203_20387


namespace sum_of_ten_angles_is_1080_l203_20311

/-- A regular pentagon inscribed in a circle --/
structure RegularPentagonInCircle where
  /-- The measure of each interior angle of the pentagon --/
  interior_angle : ℝ
  /-- The measure of each exterior angle of the pentagon --/
  exterior_angle : ℝ
  /-- The measure of each angle inscribed in the segments outside the pentagon --/
  inscribed_angle : ℝ
  /-- The number of vertices in the pentagon --/
  num_vertices : ℕ
  /-- The interior angle of a regular pentagon is 108° --/
  interior_angle_eq : interior_angle = 108
  /-- The exterior angle is supplementary to the interior angle --/
  exterior_angle_eq : exterior_angle = 180 - interior_angle
  /-- The number of vertices in a pentagon is 5 --/
  num_vertices_eq : num_vertices = 5
  /-- The inscribed angle is half of the central angle --/
  inscribed_angle_eq : inscribed_angle = (360 - exterior_angle) / 2

/-- The sum of the ten angles in a regular pentagon inscribed in a circle --/
def sum_of_ten_angles (p : RegularPentagonInCircle) : ℝ :=
  p.num_vertices * (p.inscribed_angle + p.exterior_angle)

/-- Theorem: The sum of the ten angles is 1080° --/
theorem sum_of_ten_angles_is_1080 (p : RegularPentagonInCircle) :
  sum_of_ten_angles p = 1080 := by
  sorry

end sum_of_ten_angles_is_1080_l203_20311


namespace xy_max_value_l203_20334

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2*y = 2) :
  ∃ (max : ℝ), max = (1/2 : ℝ) ∧ ∀ z, z = x*y → z ≤ max :=
sorry

end xy_max_value_l203_20334


namespace phone_number_pricing_l203_20366

theorem phone_number_pricing (X Y : ℤ) : 
  (0 < X ∧ X < 250) →
  (0 < Y ∧ Y < 250) →
  125 * X - 64 * Y = 5 →
  ((X = 41 ∧ Y = 80) ∨ (X = 105 ∧ Y = 205)) := by
sorry

end phone_number_pricing_l203_20366


namespace average_transformation_l203_20374

theorem average_transformation (x₁ x₂ x₃ : ℝ) (h : (x₁ + x₂ + x₃) / 3 = 2) :
  ((2 * x₁ + 4) + (2 * x₂ + 4) + (2 * x₃ + 4)) / 3 = 8 := by
  sorry

end average_transformation_l203_20374


namespace cos_sin_15_identity_l203_20341

theorem cos_sin_15_identity : 
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 + 
  Real.sin (15 * π / 180) * Real.cos (15 * π / 180) = 
  (1 + 2 * Real.sqrt 3) / 4 := by
  sorry

end cos_sin_15_identity_l203_20341


namespace doughnuts_theorem_l203_20355

/-- The number of boxes of doughnuts -/
def num_boxes : ℕ := 4

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 12

/-- The total number of doughnuts -/
def total_doughnuts : ℕ := num_boxes * doughnuts_per_box

theorem doughnuts_theorem : total_doughnuts = 48 := by
  sorry

end doughnuts_theorem_l203_20355


namespace problem_solution_l203_20332

theorem problem_solution (a b : ℚ) 
  (h1 : 5 + a = 7 - b) 
  (h2 : 7 + b = 12 + a) : 
  5 - a = 13/2 := by sorry

end problem_solution_l203_20332


namespace total_unique_plants_l203_20327

-- Define the sets X, Y, Z as finite sets
variable (X Y Z : Finset ℕ)

-- Define the cardinalities of the sets and their intersections
axiom card_X : X.card = 700
axiom card_Y : Y.card = 600
axiom card_Z : Z.card = 400
axiom card_X_inter_Y : (X ∩ Y).card = 100
axiom card_X_inter_Z : (X ∩ Z).card = 200
axiom card_Y_inter_Z : (Y ∩ Z).card = 50
axiom card_X_inter_Y_inter_Z : (X ∩ Y ∩ Z).card = 25

-- Theorem statement
theorem total_unique_plants : (X ∪ Y ∪ Z).card = 1375 :=
sorry

end total_unique_plants_l203_20327


namespace squirrel_acorns_l203_20322

/- Define the initial number of acorns -/
def initial_acorns : ℕ := 210

/- Define the number of parts the pile was divided into -/
def num_parts : ℕ := 3

/- Define the number of acorns left in each part after removal -/
def acorns_per_part : ℕ := 60

/- Define the total number of acorns removed -/
def total_removed : ℕ := 30

/- Theorem statement -/
theorem squirrel_acorns : 
  (initial_acorns / num_parts - acorns_per_part) * num_parts = total_removed ∧
  initial_acorns % num_parts = 0 := by
  sorry

#check squirrel_acorns

end squirrel_acorns_l203_20322


namespace fraction_meaningful_l203_20349

theorem fraction_meaningful (m : ℝ) : 
  (∃ (x : ℝ), x = 1 / (m + 3)) ↔ m ≠ -3 :=
by sorry

end fraction_meaningful_l203_20349


namespace geometric_sequence_decreasing_condition_l203_20318

/-- A geometric sequence with common ratio q -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- A decreasing sequence -/
def DecreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

/-- The condition "0 < q < 1" is neither sufficient nor necessary for a geometric sequence to be decreasing -/
theorem geometric_sequence_decreasing_condition (a : ℕ → ℝ) (q : ℝ) :
  ¬(((0 < q ∧ q < 1) → DecreasingSequence a) ∧ (DecreasingSequence a → (0 < q ∧ q < 1))) :=
by sorry

end geometric_sequence_decreasing_condition_l203_20318


namespace proportionality_check_l203_20325

/-- Represents a relationship between x and y --/
inductive Relationship
  | DirectProp
  | InverseProp
  | Neither

/-- Determines the relationship between x and y for a given equation --/
def determineRelationship (equation : ℝ → ℝ → Prop) : Relationship :=
  sorry

/-- Equation A: x^2 + y^2 = 16 --/
def equationA (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Equation B: 2xy = 5 --/
def equationB (x y : ℝ) : Prop := 2*x*y = 5

/-- Equation C: x = 3y --/
def equationC (x y : ℝ) : Prop := x = 3*y

/-- Equation D: x^2 = 4y --/
def equationD (x y : ℝ) : Prop := x^2 = 4*y

/-- Equation E: 5x + 2y = 20 --/
def equationE (x y : ℝ) : Prop := 5*x + 2*y = 20

theorem proportionality_check :
  (determineRelationship equationA = Relationship.Neither) ∧
  (determineRelationship equationB = Relationship.InverseProp) ∧
  (determineRelationship equationC = Relationship.DirectProp) ∧
  (determineRelationship equationD = Relationship.Neither) ∧
  (determineRelationship equationE = Relationship.Neither) :=
sorry

end proportionality_check_l203_20325


namespace total_ear_muffs_l203_20319

/-- The number of ear muffs bought before December -/
def before_december : ℕ := 1346

/-- The number of ear muffs bought during December -/
def during_december : ℕ := 6444

/-- The total number of ear muffs bought -/
def total : ℕ := before_december + during_december

theorem total_ear_muffs : total = 7790 := by sorry

end total_ear_muffs_l203_20319
