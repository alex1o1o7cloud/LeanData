import Mathlib

namespace probability_equals_three_over_646_l2444_244417

-- Define the cube
def cube_side_length : ℕ := 5
def total_cubes : ℕ := cube_side_length ^ 3

-- Define the number of cubes with different numbers of painted faces
def cubes_with_three_painted_faces : ℕ := 1
def cubes_with_one_painted_face : ℕ := 36

-- Define the probability calculation function
def probability_one_three_one_face : ℚ :=
  (cubes_with_three_painted_faces * cubes_with_one_painted_face : ℚ) /
  (total_cubes * (total_cubes - 1) / 2)

-- The theorem to prove
theorem probability_equals_three_over_646 :
  probability_one_three_one_face = 3 / 646 := by
  sorry

end probability_equals_three_over_646_l2444_244417


namespace quadratic_equivalence_l2444_244411

theorem quadratic_equivalence : ∀ x : ℝ, x^2 - 8*x - 1 = 0 ↔ (x - 4)^2 = 17 := by
  sorry

end quadratic_equivalence_l2444_244411


namespace function_properties_l2444_244447

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

theorem function_properties (a : ℝ) :
  (∃ x, ∀ y, f a y ≤ f a x) ∧ (∃ x, f a x = 6) →
  a = 6 ∧
  ∀ k, (∀ x t, x ∈ [-2, 2] → t ∈ [-1, 1] → f a x ≥ k * t - 25) ↔ k ∈ [-3, 3] :=
sorry

end function_properties_l2444_244447


namespace tan_theta_value_l2444_244451

theorem tan_theta_value (θ : Real) (x y : Real) : 
  x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 → Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end tan_theta_value_l2444_244451


namespace prime_sum_power_implies_three_power_l2444_244482

theorem prime_sum_power_implies_three_power (n : ℕ) : 
  Nat.Prime (1 + 2^n + 4^n) → ∃ k : ℕ, n = 3^k :=
by sorry

end prime_sum_power_implies_three_power_l2444_244482


namespace binomial_coefficient_ratio_l2444_244494

theorem binomial_coefficient_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₂ + a₄) / (a₁ + a₃) = -3/4 := by
sorry

end binomial_coefficient_ratio_l2444_244494


namespace divisible_by_seven_l2444_244465

theorem divisible_by_seven (n : ℕ) : 7 ∣ (3^(2*n+1) + 2^(n+2)) := by
  sorry

end divisible_by_seven_l2444_244465


namespace solution_system_l2444_244471

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 88) : 
  x^2 + y^2 = 304/9 := by
sorry

end solution_system_l2444_244471


namespace percentage_to_total_l2444_244421

/-- If 25% of an amount is 75 rupees, then the total amount is 300 rupees. -/
theorem percentage_to_total (amount : ℝ) : (25 / 100) * amount = 75 → amount = 300 := by
  sorry

end percentage_to_total_l2444_244421


namespace route_length_difference_l2444_244479

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Represents the trip details of Jerry and Beth -/
structure TripDetails where
  jerry_speed : ℝ
  jerry_time : ℝ
  beth_speed : ℝ
  beth_extra_time : ℝ

/-- Theorem stating the difference in route lengths -/
theorem route_length_difference (trip : TripDetails) : 
  trip.jerry_speed = 40 →
  trip.jerry_time = 0.5 →
  trip.beth_speed = 30 →
  trip.beth_extra_time = 1/3 →
  distance trip.beth_speed (trip.jerry_time + trip.beth_extra_time) - 
  distance trip.jerry_speed trip.jerry_time = 5 := by
  sorry

end route_length_difference_l2444_244479


namespace consecutive_sums_not_prime_l2444_244436

theorem consecutive_sums_not_prime (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ k < 5*n + 10 ∧ (5*n + 10) % k = 0) ∧
  (∃ k : ℕ, k > 1 ∧ k < 5*n^2 + 10 ∧ (5*n^2 + 10) % k = 0) := by
  sorry

#check consecutive_sums_not_prime

end consecutive_sums_not_prime_l2444_244436


namespace probability_three_same_color_l2444_244412

def total_balls : ℕ := 27
def green_balls : ℕ := 15
def white_balls : ℕ := 12

def probability_same_color : ℚ := 3 / 13

theorem probability_three_same_color :
  let total_combinations := Nat.choose total_balls 3
  let green_combinations := Nat.choose green_balls 3
  let white_combinations := Nat.choose white_balls 3
  (green_combinations + white_combinations : ℚ) / total_combinations = probability_same_color :=
by sorry

end probability_three_same_color_l2444_244412


namespace parallelogram_diagonal_squared_l2444_244429

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- Area of the parallelogram
  area : ℝ
  -- Length of PQ (projections of A and C onto BD)
  pq : ℝ
  -- Length of RS (projections of B and D onto AC)
  rs : ℝ
  -- Ensures area is positive
  area_pos : area > 0
  -- Ensures PQ is positive
  pq_pos : pq > 0
  -- Ensures RS is positive
  rs_pos : rs > 0

/-- The main theorem about the longer diagonal of the parallelogram -/
theorem parallelogram_diagonal_squared
  (abcd : Parallelogram)
  (h1 : abcd.area = 24)
  (h2 : abcd.pq = 8)
  (h3 : abcd.rs = 10) :
  ∃ (d : ℝ), d > 0 ∧ d^2 = 62 + 20 * Real.sqrt 61 :=
sorry

end parallelogram_diagonal_squared_l2444_244429


namespace dice_cube_surface_area_l2444_244425

theorem dice_cube_surface_area (num_dice : ℕ) (die_side_length : ℝ) (h1 : num_dice = 27) (h2 : die_side_length = 3) :
  let edge_length : ℝ := die_side_length * (num_dice ^ (1/3 : ℝ))
  let face_area : ℝ := edge_length ^ 2
  let surface_area : ℝ := 6 * face_area
  surface_area = 486 :=
by sorry

end dice_cube_surface_area_l2444_244425


namespace weed_difference_l2444_244441

/-- The number of weeds Sarah pulled on each day --/
structure WeedCount where
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Sarah's weed-pulling over four days --/
def SarahsWeedPulling (w : WeedCount) : Prop :=
  w.tuesday = 25 ∧
  w.wednesday = 3 * w.tuesday ∧
  w.thursday = w.wednesday / 5 ∧
  w.friday < w.thursday ∧
  w.tuesday + w.wednesday + w.thursday + w.friday = 120

/-- Theorem stating the difference in weeds pulled between Thursday and Friday --/
theorem weed_difference (w : WeedCount) (h : SarahsWeedPulling w) :
  w.thursday - w.friday = 10 := by
  sorry

end weed_difference_l2444_244441


namespace remaining_sample_is_nineteen_l2444_244461

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Theorem stating that the remaining sample number is 19 -/
theorem remaining_sample_is_nineteen (s : SystematicSampling)
  (h1 : s.total_students = 56)
  (h2 : s.sample_size = 4)
  (h3 : s.known_samples = [5, 33, 47])
  : ∃ (remaining : ℕ), remaining = 19 ∧ remaining ∉ s.known_samples :=
by sorry

end remaining_sample_is_nineteen_l2444_244461


namespace tan_alpha_plus_pi_fourth_l2444_244487

theorem tan_alpha_plus_pi_fourth (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan (α + π / 4) = -1 / 7 := by
  sorry

end tan_alpha_plus_pi_fourth_l2444_244487


namespace park_outer_diameter_l2444_244410

/-- Represents the diameter of the outer boundary of a circular park with concentric sections -/
def outer_diameter (fountain_diameter : ℝ) (garden_width : ℝ) (path_width : ℝ) : ℝ :=
  fountain_diameter + 2 * (garden_width + path_width)

/-- Theorem stating the diameter of the outer boundary of the jogging path -/
theorem park_outer_diameter :
  outer_diameter 14 12 10 = 58 := by
  sorry

end park_outer_diameter_l2444_244410


namespace sin_sum_to_product_l2444_244489

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (7 * x) = 2 * Real.sin (5 * x) * Real.cos (2 * x) := by
  sorry

-- The sum-to-product identity for sine
axiom sin_sum_to_product_identity (a b : ℝ) : 
  Real.sin a + Real.sin b = 2 * Real.sin ((a + b) / 2) * Real.cos ((a - b) / 2)

end sin_sum_to_product_l2444_244489


namespace min_value_cyclic_fraction_sum_l2444_244433

theorem min_value_cyclic_fraction_sum (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a / b + b / c + c / d + d / a ≥ 4 := by
  sorry

end min_value_cyclic_fraction_sum_l2444_244433


namespace mango_purchase_l2444_244467

theorem mango_purchase (grapes_kg : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) :
  grapes_kg = 10 ∧ 
  grapes_rate = 70 ∧ 
  mango_rate = 55 ∧ 
  total_paid = 1195 →
  ∃ (mango_kg : ℕ), mango_kg = 9 ∧ grapes_kg * grapes_rate + mango_kg * mango_rate = total_paid :=
by sorry

end mango_purchase_l2444_244467


namespace inscribed_circle_radius_isosceles_triangle_l2444_244469

/-- The radius of the inscribed circle in an isosceles triangle -/
theorem inscribed_circle_radius_isosceles_triangle (DE DF EF : ℝ) (h1 : DE = 8) (h2 : DF = 8) (h3 : EF = 10) :
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  let r := K / s
  r = 5 * Real.sqrt 39 / 13 := by
  sorry

end inscribed_circle_radius_isosceles_triangle_l2444_244469


namespace complex_root_quadratic_equation_l2444_244478

theorem complex_root_quadratic_equation (b c : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I * Real.sqrt 2 : ℂ) ^ 2 + b * (1 - Complex.I * Real.sqrt 2) + c = 0 →
  b = -2 ∧ c = 3 := by
sorry

end complex_root_quadratic_equation_l2444_244478


namespace polynomial_expansion_alternating_sum_l2444_244443

theorem polynomial_expansion_alternating_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ - a₁ + a₂ - a₃ + a₄ = 1 := by
sorry

end polynomial_expansion_alternating_sum_l2444_244443


namespace circle_point_x_value_l2444_244409

/-- Given a circle in the xy-plane with diameter endpoints (-3,0) and (21,0),
    if the point (x,12) is on the circle, then x = 9. -/
theorem circle_point_x_value (x : ℝ) : 
  let center : ℝ × ℝ := ((21 - 3) / 2 + -3, 0)
  let radius : ℝ := (21 - (-3)) / 2
  ((x - center.1)^2 + (12 - center.2)^2 = radius^2) → x = 9 := by
  sorry

end circle_point_x_value_l2444_244409


namespace arithmetic_sequence_property_l2444_244477

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Our specific arithmetic sequence satisfying a₃ + a₈ = 6 -/
def our_sequence (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧ a 3 + a 8 = 6

theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : our_sequence a) :
  3 * a 2 + a 16 = 12 := by
  sorry

end arithmetic_sequence_property_l2444_244477


namespace probability_two_females_l2444_244475

def total_students : ℕ := 5
def female_students : ℕ := 3
def male_students : ℕ := 2
def students_to_select : ℕ := 2

theorem probability_two_females :
  (Nat.choose female_students students_to_select : ℚ) / 
  (Nat.choose total_students students_to_select : ℚ) = 3 / 10 := by
  sorry

end probability_two_females_l2444_244475


namespace altitude_not_integer_l2444_244472

/-- Represents a right triangle with integer sides -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  is_right : a^2 + b^2 = c^2  -- Pythagorean theorem

/-- The altitude to the hypotenuse in a right triangle -/
def altitude (t : RightTriangle) : ℚ :=
  (t.a * t.b : ℚ) / t.c

/-- Theorem: In a right triangle with pairwise coprime integer sides, 
    the altitude to the hypotenuse is not an integer -/
theorem altitude_not_integer (t : RightTriangle) 
  (h_coprime : Nat.gcd t.a t.b = 1 ∧ Nat.gcd t.b t.c = 1 ∧ Nat.gcd t.c t.a = 1) : 
  ¬ ∃ (n : ℕ), altitude t = n :=
sorry

end altitude_not_integer_l2444_244472


namespace interest_difference_is_520_l2444_244490

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Theorem stating that the difference between the principal and 
    the simple interest is $520 under the given conditions -/
theorem interest_difference_is_520 :
  let principal : ℝ := 1000
  let rate : ℝ := 0.06
  let time : ℝ := 8
  principal - simple_interest principal rate time = 520 := by
sorry


end interest_difference_is_520_l2444_244490


namespace car_profit_percentage_l2444_244438

/-- Calculates the profit percentage on the original price of a car, given specific buying and selling conditions. -/
theorem car_profit_percentage (P : ℝ) (h : P > 0) : 
  let purchase_price := 0.95 * P
  let taxes := 0.03 * P
  let maintenance := 0.02 * P
  let total_cost := purchase_price + taxes + maintenance
  let selling_price := purchase_price * 1.6
  let profit := selling_price - total_cost
  let profit_percentage := (profit / P) * 100
  profit_percentage = 52 := by sorry

end car_profit_percentage_l2444_244438


namespace sqrt_sum_zero_implies_both_zero_l2444_244427

theorem sqrt_sum_zero_implies_both_zero (x y : ℝ) :
  Real.sqrt x + Real.sqrt y = 0 → x = 0 ∧ y = 0 := by
  sorry

end sqrt_sum_zero_implies_both_zero_l2444_244427


namespace bracelet_large_beads_l2444_244416

/-- Proves the number of large beads per bracelet given the problem conditions -/
theorem bracelet_large_beads (total_beads : ℕ) (num_bracelets : ℕ) : 
  total_beads = 528 →
  num_bracelets = 11 →
  ∃ (large_beads_per_bracelet : ℕ),
    large_beads_per_bracelet * num_bracelets = total_beads / 2 ∧
    large_beads_per_bracelet = 24 := by
  sorry

#check bracelet_large_beads

end bracelet_large_beads_l2444_244416


namespace jeromes_contact_list_ratio_l2444_244414

/-- Proves that the ratio of out of school friends to classmates is 1:2 given the conditions in Jerome's contact list problem -/
theorem jeromes_contact_list_ratio : 
  ∀ (out_of_school_friends : ℕ),
    20 + out_of_school_friends + 2 + 1 = 33 →
    out_of_school_friends = 10 ∧ 
    (out_of_school_friends : ℚ) / 20 = 1 / 2 := by
  sorry


end jeromes_contact_list_ratio_l2444_244414


namespace average_time_per_mile_l2444_244419

/-- Proves that the average time per mile is 9 minutes for a 24-mile run completed in 3 hours and 36 minutes -/
theorem average_time_per_mile (distance : ℕ) (hours : ℕ) (minutes : ℕ) : 
  distance = 24 ∧ hours = 3 ∧ minutes = 36 → 
  (hours * 60 + minutes) / distance = 9 := by
  sorry

end average_time_per_mile_l2444_244419


namespace egyptian_fraction_1991_l2444_244420

theorem egyptian_fraction_1991 : ∃ (k l m : ℕ), 
  Odd k ∧ Odd l ∧ Odd m ∧ 
  (1 : ℚ) / 1991 = 1 / k + 1 / l + 1 / m := by
  sorry

end egyptian_fraction_1991_l2444_244420


namespace common_root_quadratic_equations_l2444_244460

theorem common_root_quadratic_equations (p : ℝ) :
  (p > 0 ∧
   ∃ x : ℝ, (3 * x^2 - 4 * p * x + 9 = 0) ∧ (x^2 - 2 * p * x + 5 = 0)) ↔
  p = 3 :=
by sorry

end common_root_quadratic_equations_l2444_244460


namespace rotation_equivalence_l2444_244431

/-- 
Given that:
1. A point A is rotated 550 degrees clockwise about a center point B to reach point C.
2. The same point A is rotated x degrees counterclockwise about the same center point B to reach point C.
3. x is less than 360 degrees.

Prove that x equals 170 degrees.
-/
theorem rotation_equivalence (x : ℝ) 
  (h1 : x < 360) 
  (h2 : (550 % 360 : ℝ) + x = 360) : x = 170 :=
by sorry

end rotation_equivalence_l2444_244431


namespace no_real_sqrt_negative_quadratic_l2444_244455

theorem no_real_sqrt_negative_quadratic : ∀ x : ℝ, ¬ ∃ y : ℝ, y^2 = -(x^2 + x + 1) :=
by
  sorry

end no_real_sqrt_negative_quadratic_l2444_244455


namespace dog_food_theorem_l2444_244405

/-- The amount of food eaten by Hannah's three dogs -/
def dog_food_problem (first_dog_food second_dog_food third_dog_food : ℝ) : Prop :=
  -- Hannah has three dogs
  -- The first dog eats 1.5 cups of dog food a day
  first_dog_food = 1.5 ∧
  -- The second dog eats twice as much as the first dog
  second_dog_food = 2 * first_dog_food ∧
  -- Hannah prepares 10 cups of dog food in total for her three dogs
  first_dog_food + second_dog_food + third_dog_food = 10 ∧
  -- The difference between the third dog's food and the second dog's food is 2.5 cups
  third_dog_food - second_dog_food = 2.5

theorem dog_food_theorem :
  ∃ (first_dog_food second_dog_food third_dog_food : ℝ),
    dog_food_problem first_dog_food second_dog_food third_dog_food :=
by
  sorry

end dog_food_theorem_l2444_244405


namespace equation_simplification_l2444_244422

theorem equation_simplification (x y z : ℕ) (hx : x > 1) (hy : y > 1) (hz : z > 1) :
  9 * x - (10 / (2 * y) / 3 + 7 * z) * Real.pi = 9 * x - (5 * Real.pi / (3 * y)) - (7 * Real.pi * z) :=
by sorry

end equation_simplification_l2444_244422


namespace gumball_probability_l2444_244457

theorem gumball_probability (blue_prob : ℝ) (pink_prob : ℝ) : 
  (blue_prob ^ 2 = 25 / 49) → 
  (blue_prob + pink_prob = 1) → 
  (pink_prob = 2 / 7) := by
  sorry

end gumball_probability_l2444_244457


namespace field_trip_students_l2444_244407

theorem field_trip_students (van_capacity : ℕ) (num_adults : ℕ) (num_vans : ℕ) : 
  van_capacity = 5 → num_adults = 5 → num_vans = 6 → 
  (num_vans * van_capacity - num_adults : ℕ) = 25 := by
  sorry

end field_trip_students_l2444_244407


namespace seven_trees_planting_methods_l2444_244401

/-- The number of ways to plant n trees in a row, choosing from plane trees and willow trees,
    such that no two adjacent trees are both willows. -/
def valid_planting_methods (n : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 34 valid planting methods for 7 trees. -/
theorem seven_trees_planting_methods :
  valid_planting_methods 7 = 34 :=
sorry

end seven_trees_planting_methods_l2444_244401


namespace complement_of_A_l2444_244468

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x ≥ 1} ∪ {x : ℝ | x < 0}

theorem complement_of_A : Set.compl A = Set.Icc 0 1 := by sorry

end complement_of_A_l2444_244468


namespace root_sum_product_l2444_244462

theorem root_sum_product (a b : ℝ) : 
  (a^4 - 6*a - 2 = 0) → 
  (b^4 - 6*b - 2 = 0) → 
  (a ≠ b) →
  (a*b + a + b = 2 - Real.sqrt 3) :=
by sorry

end root_sum_product_l2444_244462


namespace dice_probabilities_l2444_244413

-- Define the sample space and events
def Ω : ℕ := 216  -- Total number of possible outcomes
def A : ℕ := 120  -- Number of outcomes where all dice show different numbers
def AB : ℕ := 75  -- Number of outcomes satisfying both A and B

-- Define the probabilities
def P_AB : ℚ := AB / Ω
def P_A : ℚ := A / Ω
def P_B_given_A : ℚ := P_AB / P_A

-- State the theorem
theorem dice_probabilities :
  P_AB = 75 / 216 ∧ P_B_given_A = 5 / 8 := by
  sorry


end dice_probabilities_l2444_244413


namespace triangle_properties_l2444_244463

/-- Triangle ABC with given properties -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h1 : a = 3
  h2 : b = 4
  h3 : B = π/2 + A

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) : Real.tan t.B = -4/3 ∧ t.c = 7/5 := by
  sorry


end triangle_properties_l2444_244463


namespace arithmetic_mean_problem_l2444_244464

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 20 + 25 + 7 + 15 + y) / 6 = 15 → y = 15 := by
sorry

end arithmetic_mean_problem_l2444_244464


namespace sphere_and_cylinder_properties_l2444_244418

/-- Given a sphere with volume 72π cubic inches, prove its surface area and the radius of a cylinder with the same volume and height 4 inches. -/
theorem sphere_and_cylinder_properties :
  ∃ (r : ℝ), 
    (4 / 3 * π * r^3 = 72 * π) ∧ 
    (4 * π * r^2 = 36 * 2^(2/3) * π) ∧
    ∃ (r_cyl : ℝ), 
      (π * r_cyl^2 * 4 = 72 * π) ∧ 
      (r_cyl = 3 * Real.sqrt 2) :=
by sorry

end sphere_and_cylinder_properties_l2444_244418


namespace solution_set_of_inequality_l2444_244428

theorem solution_set_of_inequality (x : ℝ) : 
  x * (x + 2) < 3 ↔ -3 < x ∧ x < 1 := by sorry

end solution_set_of_inequality_l2444_244428


namespace rectangle_shorter_side_l2444_244415

theorem rectangle_shorter_side 
  (area : ℝ) 
  (perimeter : ℝ) 
  (h_area : area = 91) 
  (h_perimeter : perimeter = 40) : 
  ∃ (length width : ℝ), 
    length * width = area ∧ 
    2 * (length + width) = perimeter ∧ 
    width = 7 ∧ 
    width ≤ length := by
  sorry

end rectangle_shorter_side_l2444_244415


namespace crude_oil_mixture_theorem_l2444_244456

/-- Represents the percentage of hydrocarbons in crude oil from the first source -/
def first_source_percentage : ℝ := 25

/-- Represents the total amount of crude oil needed in gallons -/
def total_crude_oil : ℝ := 50

/-- Represents the desired percentage of hydrocarbons in the final mixture -/
def final_mixture_percentage : ℝ := 55

/-- Represents the amount of crude oil from the second source in gallons -/
def second_source_amount : ℝ := 30

/-- Represents the percentage of hydrocarbons in crude oil from the second source -/
def second_source_percentage : ℝ := 75

/-- Theorem stating that given the conditions, the percentage of hydrocarbons
    in the first source is 25% -/
theorem crude_oil_mixture_theorem :
  (first_source_percentage / 100 * (total_crude_oil - second_source_amount) +
   second_source_percentage / 100 * second_source_amount) / total_crude_oil * 100 =
  final_mixture_percentage := by
  sorry

end crude_oil_mixture_theorem_l2444_244456


namespace max_non_managers_l2444_244493

theorem max_non_managers (managers : ℕ) (ratio_managers : ℕ) (ratio_non_managers : ℕ) :
  managers = 11 →
  ratio_managers = 7 →
  ratio_non_managers = 37 →
  ∀ non_managers : ℕ,
    (managers : ℚ) / (non_managers : ℚ) > (ratio_managers : ℚ) / (ratio_non_managers : ℚ) →
    non_managers ≤ 58 :=
by sorry

end max_non_managers_l2444_244493


namespace two_number_difference_l2444_244452

theorem two_number_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (triple_minus_quad : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 := by
sorry

end two_number_difference_l2444_244452


namespace max_store_visits_l2444_244408

theorem max_store_visits (total_stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 7) (h2 : total_visits = 21) 
  (h3 : unique_visitors = 11) (h4 : double_visitors = 7) 
  (h5 : double_visitors * 2 ≤ total_visits) 
  (h6 : ∀ v, v ≤ unique_visitors → v ≥ 1) : 
  ∃ max_visits : ℕ, max_visits ≤ total_stores ∧ 
  (∀ v, v ≤ unique_visitors → v ≤ max_visits) ∧ max_visits = 4 :=
by sorry

end max_store_visits_l2444_244408


namespace unique_solution_quadratic_l2444_244439

theorem unique_solution_quadratic (m : ℚ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = m + 3 * x) ↔ m = 23 / 4 := by
  sorry

end unique_solution_quadratic_l2444_244439


namespace wyatt_orange_juice_purchase_l2444_244496

def orange_juice_cartons (initial_money : ℕ) (bread_loaves : ℕ) (bread_cost : ℕ) (juice_cost : ℕ) (remaining_money : ℕ) : ℕ :=
  (initial_money - remaining_money - bread_loaves * bread_cost) / juice_cost

theorem wyatt_orange_juice_purchase :
  orange_juice_cartons 74 5 5 2 41 = 4 := by
  sorry

end wyatt_orange_juice_purchase_l2444_244496


namespace necessary_but_not_sufficient_condition_l2444_244432

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (x^2 - 2*x - 3 < 0) → (-2 < x ∧ x < 3) ∧
  ∃ y : ℝ, -2 < y ∧ y < 3 ∧ ¬(y^2 - 2*y - 3 < 0) :=
by sorry

end necessary_but_not_sufficient_condition_l2444_244432


namespace max_crates_on_trailer_l2444_244423

/-- The maximum number of crates a trailer can carry given weight constraints -/
theorem max_crates_on_trailer (min_crate_weight max_total_weight : ℕ) 
  (h1 : min_crate_weight ≥ 120)
  (h2 : max_total_weight = 720) :
  (max_total_weight / min_crate_weight : ℕ) = 6 := by
  sorry

#check max_crates_on_trailer

end max_crates_on_trailer_l2444_244423


namespace tyler_saltwater_animals_l2444_244488

/-- The number of saltwater aquariums Tyler has -/
def saltwater_aquariums : ℕ := 22

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 46

/-- The total number of saltwater animals Tyler has -/
def total_saltwater_animals : ℕ := saltwater_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_saltwater_animals = 1012 := by
  sorry

end tyler_saltwater_animals_l2444_244488


namespace power_of_half_l2444_244476

theorem power_of_half (some_power k : ℕ) : 
  (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → 
  k = 8 → 
  some_power = 16 := by
sorry

end power_of_half_l2444_244476


namespace carpenter_tables_problem_l2444_244406

theorem carpenter_tables_problem (T : ℕ) : 
  T + (T - 3) = 17 → T = 10 := by sorry

end carpenter_tables_problem_l2444_244406


namespace dental_bill_theorem_l2444_244444

/-- The cost of a dental filling -/
def filling_cost : ℕ := sorry

/-- The cost of a dental cleaning -/
def cleaning_cost : ℕ := 70

/-- The cost of a tooth extraction -/
def extraction_cost : ℕ := 290

/-- The total bill for dental services -/
def total_bill : ℕ := 5 * filling_cost

theorem dental_bill_theorem : 
  filling_cost = 120 ∧ 
  total_bill = cleaning_cost + 2 * filling_cost + extraction_cost := by
  sorry

end dental_bill_theorem_l2444_244444


namespace neon_sign_blink_interval_l2444_244449

theorem neon_sign_blink_interval (t1 t2 : ℕ) : 
  t1 = 9 → 
  t1.lcm t2 = 45 → 
  t2 = 15 := by
sorry

end neon_sign_blink_interval_l2444_244449


namespace cube_volume_from_surface_area_l2444_244450

theorem cube_volume_from_surface_area :
  ∀ s V : ℝ,
  (6 * s^2 = 864) →  -- Surface area condition
  (V = s^3) →        -- Volume definition
  V = 1728 :=
by
  sorry

end cube_volume_from_surface_area_l2444_244450


namespace cos_sin_transformation_l2444_244485

theorem cos_sin_transformation (x : ℝ) :
  3 * Real.cos (2 * x) = 3 * Real.sin (2 * (x + π / 12) + π / 3) :=
by sorry

end cos_sin_transformation_l2444_244485


namespace sum_of_even_indexed_coefficients_l2444_244402

theorem sum_of_even_indexed_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (2*x + 3)^8 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                      a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8) →
  a + a₂ + a₄ + a₆ + a₈ = 3281 := by
sorry

end sum_of_even_indexed_coefficients_l2444_244402


namespace polar_to_cartesian_equivalence_l2444_244448

/-- Prove that the polar equation ρ²cos(2θ) = 16 is equivalent to the Cartesian equation x² - y² = 16 -/
theorem polar_to_cartesian_equivalence (ρ θ x y : ℝ) 
  (h1 : x = ρ * Real.cos θ) 
  (h2 : y = ρ * Real.sin θ) : 
  ρ^2 * Real.cos (2 * θ) = 16 ↔ x^2 - y^2 = 16 := by
  sorry

end polar_to_cartesian_equivalence_l2444_244448


namespace debby_vacation_pictures_l2444_244470

/-- Calculates the number of remaining pictures after deletion -/
def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (zoo_pictures + museum_pictures) - deleted_pictures

/-- Theorem: The number of remaining pictures is correct for Debby's vacation -/
theorem debby_vacation_pictures : remaining_pictures 24 12 14 = 22 := by
  sorry

end debby_vacation_pictures_l2444_244470


namespace h2o_formation_in_neutralization_l2444_244480

/-- Represents a chemical substance -/
structure Substance where
  name : String
  moles : ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Substance
  products : List Substance

/-- Given a balanced chemical equation and the amounts of reactants, 
    calculate the amount of a specific product formed -/
def calculateProductAmount (reaction : Reaction) (product : Substance) : ℝ :=
  sorry

theorem h2o_formation_in_neutralization :
  let hch3co2 := Substance.mk "HCH3CO2" 1
  let naoh := Substance.mk "NaOH" 1
  let h2o := Substance.mk "H2O" 1
  let nach3co2 := Substance.mk "NaCH3CO2" 1
  let reaction := Reaction.mk [hch3co2, naoh] [nach3co2, h2o]
  calculateProductAmount reaction h2o = 1 := by
  sorry

end h2o_formation_in_neutralization_l2444_244480


namespace min_value_of_3a_plus_2_l2444_244466

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 6 * a + 5 = 7) :
  ∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), 8 * x^2 + 6 * x + 5 = 7 → 3 * x + 2 ≥ m :=
by sorry

end min_value_of_3a_plus_2_l2444_244466


namespace profit_maximization_l2444_244484

noncomputable def profit (x : ℝ) : ℝ := 20 - x - 4 / (x + 1)

theorem profit_maximization (a : ℝ) (h_a : a > 0) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ a^2 - 3*a + 3 ∧
  (∀ (y : ℝ), 0 ≤ y ∧ y ≤ a^2 - 3*a + 3 → profit x ≥ profit y) ∧
  ((a ≥ 2 ∨ 0 < a ∧ a ≤ 1) → x = 1) ∧
  (1 < a ∧ a < 2 → x = a^2 - 3*a + 3) :=
sorry

end profit_maximization_l2444_244484


namespace last_three_digits_perfect_square_l2444_244403

theorem last_three_digits_perfect_square (n : ℕ) : 
  ∃ (m : ℕ), m * m % 1000 = 689 ∧ 
  ∀ (k : ℕ), k * k % 1000 ≠ 759 := by
  sorry

end last_three_digits_perfect_square_l2444_244403


namespace irrational_and_rational_numbers_l2444_244483

theorem irrational_and_rational_numbers : 
  (¬ ∃ (p q : ℤ), π = (p : ℚ) / (q : ℚ)) ∧ 
  (∃ (p q : ℤ), (22 : ℚ) / (7 : ℚ) = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), (0 : ℚ) = (p : ℚ) / (q : ℚ)) ∧
  (∃ (p q : ℤ), (-2 : ℚ) = (p : ℚ) / (q : ℚ)) :=
by sorry

end irrational_and_rational_numbers_l2444_244483


namespace function_value_l2444_244491

/-- Given a function f(x) = x^α that passes through (2, √2/2), prove f(4) = 1/2 -/
theorem function_value (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^α) :
  f 2 = Real.sqrt 2 / 2 → f 4 = 1 / 2 := by
  sorry

end function_value_l2444_244491


namespace x_value_proof_l2444_244430

theorem x_value_proof (x : ℝ) (h : 3/4 - 1/2 = 4/x) : x = 16 := by
  sorry

end x_value_proof_l2444_244430


namespace bat_ball_cost_difference_l2444_244498

/-- The cost difference between a ball and a bat -/
def cost_difference (x y : ℝ) : ℝ := y - x

/-- The problem statement -/
theorem bat_ball_cost_difference :
  ∀ x y : ℝ,
  (2 * x + 3 * y = 1300) →
  (3 * x + 2 * y = 1200) →
  cost_difference x y = 100 := by
sorry

end bat_ball_cost_difference_l2444_244498


namespace range_of_a_l2444_244426

theorem range_of_a (a : ℝ) : 
  (a + 1)^(-1/2 : ℝ) < (3 - 2*a)^(-1/2 : ℝ) → 2/3 < a ∧ a < 3/2 :=
by
  sorry

end range_of_a_l2444_244426


namespace fence_length_l2444_244492

/-- The total length of a fence for a land shaped like a rectangle combined with a semicircle,
    given the dimensions and an opening. -/
theorem fence_length
  (rect_length : ℝ)
  (rect_width : ℝ)
  (semicircle_radius : ℝ)
  (opening_length : ℝ)
  (h1 : rect_length = 20)
  (h2 : rect_width = 14)
  (h3 : semicircle_radius = 7)
  (h4 : opening_length = 3)
  : rect_length * 2 + rect_width + π * semicircle_radius + rect_width - opening_length = 73 :=
by
  sorry

end fence_length_l2444_244492


namespace expression_upper_bound_l2444_244497

theorem expression_upper_bound (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 2) :
  Real.sqrt (a^3 + (2-b)^3) + Real.sqrt (b^3 + (2-c)^3) + 
  Real.sqrt (c^3 + (2-d)^3) + Real.sqrt (d^3 + (3-a)^3) ≤ 5 + 2 * Real.sqrt 2 := by
  sorry

end expression_upper_bound_l2444_244497


namespace set_B_proof_l2444_244459

open Set

theorem set_B_proof (U : Set ℕ) (A B : Set ℕ) :
  U = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  (U \ (A ∪ B)) = {1, 3} →
  ((U \ A) ∩ B) = {2, 4} →
  B = {5, 6, 7, 8, 9} :=
by sorry

end set_B_proof_l2444_244459


namespace triangle_properties_l2444_244458

theorem triangle_properties (a b c A B C S : ℝ) : 
  a = 2 → C = π / 3 → 
  (A = π / 4 → c = Real.sqrt 6) ∧ 
  (S = Real.sqrt 3 → b = 2 ∧ c = 2) := by
  sorry

end triangle_properties_l2444_244458


namespace father_daughter_ages_l2444_244434

/-- Represents the ages of a father and daughter at present and in the future. -/
structure FamilyAges where
  daughter_now : ℕ
  father_now : ℕ
  daughter_future : ℕ
  father_future : ℕ

/-- The conditions given in the problem. -/
def age_conditions (ages : FamilyAges) : Prop :=
  ages.father_now = 5 * ages.daughter_now ∧
  ages.daughter_future = ages.daughter_now + 30 ∧
  ages.father_future = ages.father_now + 30 ∧
  ages.father_future = 3 * ages.daughter_future

/-- The theorem stating the solution to the problem. -/
theorem father_daughter_ages :
  ∃ (ages : FamilyAges), age_conditions ages ∧ ages.daughter_now = 30 ∧ ages.father_now = 150 := by
  sorry

end father_daughter_ages_l2444_244434


namespace integer_roots_of_polynomial_l2444_244445

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂*x^2 + b₁*x + 18

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  ∀ x : ℤ, polynomial b₂ b₁ x = 0 →
    x ∈ ({-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} : Set ℤ) := by
  sorry

end integer_roots_of_polynomial_l2444_244445


namespace five_eighteenths_decimal_l2444_244442

theorem five_eighteenths_decimal : 
  (5 : ℚ) / 18 = 0.2777777777777777 :=
by sorry

end five_eighteenths_decimal_l2444_244442


namespace tan_theta_equals_sqrt_three_over_five_l2444_244435

theorem tan_theta_equals_sqrt_three_over_five (θ : Real) : 
  2 * Real.sin (θ + π/3) = 3 * Real.sin (π/3 - θ) → 
  Real.tan θ = Real.sqrt 3 / 5 := by
sorry

end tan_theta_equals_sqrt_three_over_five_l2444_244435


namespace cubic_arithmetic_progression_roots_l2444_244495

/-- A cubic polynomial with coefficients in ℝ -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The roots of a cubic polynomial form an arithmetic progression -/
def roots_in_arithmetic_progression (p : CubicPolynomial) : Prop :=
  ∃ (r d : ℂ), p.a * (r - d)^3 + p.b * (r - d)^2 + p.c * (r - d) + p.d = 0 ∧
                p.a * r^3 + p.b * r^2 + p.c * r + p.d = 0 ∧
                p.a * (r + d)^3 + p.b * (r + d)^2 + p.c * (r + d) + p.d = 0

/-- Two roots of a cubic polynomial are not real -/
def two_roots_not_real (p : CubicPolynomial) : Prop :=
  ∃ (z w : ℂ), p.a * z^3 + p.b * z^2 + p.c * z + p.d = 0 ∧
                p.a * w^3 + p.b * w^2 + p.c * w + p.d = 0 ∧
                z.im ≠ 0 ∧ w.im ≠ 0 ∧ z ≠ w

theorem cubic_arithmetic_progression_roots (a : ℝ) :
  let p := CubicPolynomial.mk 1 (-7) 20 a
  roots_in_arithmetic_progression p ∧ two_roots_not_real p → a = -574/27 := by
  sorry

end cubic_arithmetic_progression_roots_l2444_244495


namespace bryan_total_books_l2444_244499

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 9

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 56

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that Bryan has 504 books in total -/
theorem bryan_total_books : total_books = 504 := by sorry

end bryan_total_books_l2444_244499


namespace unique_solution_trig_equation_l2444_244400

theorem unique_solution_trig_equation :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
    Real.tan ((150 - x) * π / 180) = 
      (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
      (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
    x = 115 := by
  sorry

end unique_solution_trig_equation_l2444_244400


namespace distribution_schemes_count_l2444_244424

/-- The number of ways to distribute 3 people to 7 communities with at most 2 people per community -/
def distribution_schemes : ℕ := sorry

/-- The number of ways to choose 3 communities out of 7 -/
def three_single_communities : ℕ := sorry

/-- The number of ways to choose 2 communities out of 7 and distribute 3 people -/
def one_double_one_single : ℕ := sorry

theorem distribution_schemes_count :
  distribution_schemes = three_single_communities + one_double_one_single ∧
  distribution_schemes = 336 := by sorry

end distribution_schemes_count_l2444_244424


namespace restore_exchange_rate_l2444_244474

/-- The exchange rate between Trade Federation's currency and Naboo's currency -/
structure ExchangeRate :=
  (rate : ℝ)

/-- The money supply of the Trade Federation -/
structure MoneySupply :=
  (supply : ℝ)

/-- The relationship between money supply changes and exchange rate changes -/
def money_supply_effect (ms_change : ℝ) : ℝ := 5 * ms_change

/-- The theorem stating the required change in money supply to restore the exchange rate -/
theorem restore_exchange_rate 
  (initial_rate : ExchangeRate)
  (new_rate : ExchangeRate)
  (money_supply : MoneySupply) :
  initial_rate.rate = 90 →
  new_rate.rate = 100 →
  (∀ (ms_change : ℝ), 
    ExchangeRate.rate (new_rate) * (1 - money_supply_effect ms_change / 100) = 
    ExchangeRate.rate (initial_rate)) →
  ∃ (ms_change : ℝ), ms_change = -2 :=
sorry

end restore_exchange_rate_l2444_244474


namespace intersection_sum_l2444_244454

theorem intersection_sum (c d : ℝ) : 
  (3 = (1/3) * 3 + c) → 
  (3 = (1/3) * 3 + d) → 
  c + d = 4 := by
sorry

end intersection_sum_l2444_244454


namespace complement_A_union_B_when_m_4_range_of_m_for_B_subset_A_l2444_244453

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part I
theorem complement_A_union_B_when_m_4 :
  (Set.univ : Set ℝ) \ (A ∪ B 4) = {x | x < -2 ∨ x > 7} := by sorry

-- Theorem for part II
theorem range_of_m_for_B_subset_A :
  {m : ℝ | (B m).Nonempty ∧ B m ⊆ A} = {m | 2 ≤ m ∧ m ≤ 3} := by sorry

end complement_A_union_B_when_m_4_range_of_m_for_B_subset_A_l2444_244453


namespace triangle_values_theorem_l2444_244440

def triangle (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c * x * y

theorem triangle_values_theorem (a b c d : ℚ) :
  (∀ x : ℚ, triangle a b c x d = x) ∧
  (triangle a b c 1 2 = 3) ∧
  (triangle a b c 2 3 = 4) ∧
  (d ≠ 0) →
  a = 5 ∧ b = 0 ∧ c = -1 ∧ d = 4 := by
  sorry

end triangle_values_theorem_l2444_244440


namespace parabola_intersects_x_axis_once_l2444_244473

/-- A parabola in the xy-plane defined by y = x^2 + 2x + k -/
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2*x + k

/-- Condition for a quadratic equation to have exactly one real root -/
def has_one_root (a b c : ℝ) : Prop := b^2 - 4*a*c = 0

/-- Theorem: The parabola y = x^2 + 2x + k intersects the x-axis at only one point if and only if k = 1 -/
theorem parabola_intersects_x_axis_once (k : ℝ) :
  (∃ x : ℝ, parabola k x = 0 ∧ ∀ y : ℝ, parabola k y = 0 → y = x) ↔ k = 1 :=
sorry

end parabola_intersects_x_axis_once_l2444_244473


namespace complex_modulus_problem_l2444_244481

theorem complex_modulus_problem (z : ℂ) (h : (1 + Complex.I) / z = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l2444_244481


namespace square_sum_problem_l2444_244404

theorem square_sum_problem (a b : ℝ) (h1 : a * b = 30) (h2 : a + b = 10) :
  a^2 + b^2 = 40 := by sorry

end square_sum_problem_l2444_244404


namespace nephews_count_l2444_244486

/-- The number of nephews Alden and Vihaan have altogether -/
def total_nephews (alden_past : ℕ) (increase : ℕ) : ℕ :=
  let alden_current := 2 * alden_past
  let vihaan := alden_current + increase
  alden_current + vihaan

/-- Theorem stating the total number of nephews Alden and Vihaan have -/
theorem nephews_count : total_nephews 50 60 = 260 := by
  sorry

end nephews_count_l2444_244486


namespace larger_number_proof_l2444_244446

theorem larger_number_proof (a b : ℕ+) (x y : ℕ+) 
  (hcf_eq : Nat.gcd a b = 30)
  (x_eq : x = 10)
  (y_eq : y = 15)
  (lcm_eq : Nat.lcm a b = 30 * x * y) :
  max a b = 450 := by
sorry

end larger_number_proof_l2444_244446


namespace sum_of_coefficients_l2444_244437

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                    a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 510 := by
sorry

end sum_of_coefficients_l2444_244437
