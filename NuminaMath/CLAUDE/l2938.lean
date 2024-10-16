import Mathlib

namespace NUMINAMATH_CALUDE_first_divisible_by_three_and_seven_l2938_293832

theorem first_divisible_by_three_and_seven (lower_bound upper_bound : ℕ) 
  (h_lower : lower_bound = 100) (h_upper : upper_bound = 600) :
  ∃ (n : ℕ), 
    n ≥ lower_bound ∧ 
    n ≤ upper_bound ∧ 
    n % 3 = 0 ∧ 
    n % 7 = 0 ∧
    ∀ (m : ℕ), m ≥ lower_bound ∧ m < n → m % 3 ≠ 0 ∨ m % 7 ≠ 0 :=
by
  -- Proof goes here
  sorry

#eval (105 : ℕ)

end NUMINAMATH_CALUDE_first_divisible_by_three_and_seven_l2938_293832


namespace NUMINAMATH_CALUDE_original_group_size_l2938_293893

theorem original_group_size (n : ℕ) (W : ℝ) : 
  W = n * 35 ∧ 
  W + 40 = (n + 1) * 36 →
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_original_group_size_l2938_293893


namespace NUMINAMATH_CALUDE_sine_range_on_interval_l2938_293874

open Real

theorem sine_range_on_interval :
  let f : ℝ → ℝ := λ x ↦ sin x
  let S : Set ℝ := { x | π/6 ≤ x ∧ x ≤ π/2 }
  f '' S = { y | 1/2 ≤ y ∧ y ≤ 1 } := by
  sorry

end NUMINAMATH_CALUDE_sine_range_on_interval_l2938_293874


namespace NUMINAMATH_CALUDE_fibonacci_problem_l2938_293894

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the property of arithmetic sequence for Fibonacci numbers
def is_arithmetic_seq (a b c : ℕ) : Prop :=
  fib c - fib b = fib b - fib a

-- Define the main theorem
theorem fibonacci_problem (b : ℕ) :
  is_arithmetic_seq (b - 3) b (b + 3) →
  (b - 3) + b + (b + 3) = 2253 →
  b = 751 := by
  sorry


end NUMINAMATH_CALUDE_fibonacci_problem_l2938_293894


namespace NUMINAMATH_CALUDE_greatest_third_term_in_arithmetic_sequence_l2938_293823

theorem greatest_third_term_in_arithmetic_sequence :
  ∀ (a d : ℕ),
  a > 0 →
  d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 65 →
  (a + 2*d) = 13 ∧ ∀ (b e : ℕ), b > 0 → e > 0 → 
  b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 65 →
  (b + 2*e) ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_term_in_arithmetic_sequence_l2938_293823


namespace NUMINAMATH_CALUDE_factor_expression_l2938_293877

theorem factor_expression (y : ℝ) : 16 * y^2 + 8 * y = 8 * y * (2 * y + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2938_293877


namespace NUMINAMATH_CALUDE_triangle_base_height_difference_l2938_293822

theorem triangle_base_height_difference (base height : ℚ) : 
  base = 5/6 → height = 4/6 → base - height = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_height_difference_l2938_293822


namespace NUMINAMATH_CALUDE_specific_cone_properties_l2938_293815

/-- Represents a cone with given height and slant height -/
structure Cone where
  height : ℝ
  slant_height : ℝ

/-- The central angle (in degrees) of the unfolded lateral surface of a cone -/
def central_angle (c : Cone) : ℝ := sorry

/-- The lateral surface area of a cone -/
def lateral_surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the properties of a specific cone -/
theorem specific_cone_properties :
  let c := Cone.mk (2 * Real.sqrt 2) 3
  central_angle c = 120 ∧ lateral_surface_area c = 3 * Real.pi := by sorry

end NUMINAMATH_CALUDE_specific_cone_properties_l2938_293815


namespace NUMINAMATH_CALUDE_pictures_deleted_l2938_293871

theorem pictures_deleted (zoo_pics : ℕ) (museum_pics : ℕ) (pics_left : ℕ) : 
  zoo_pics = 50 → museum_pics = 8 → pics_left = 20 → 
  zoo_pics + museum_pics - pics_left = 38 := by
sorry

end NUMINAMATH_CALUDE_pictures_deleted_l2938_293871


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2938_293875

theorem smaller_number_proof (x y : ℝ) : 
  x - y = 9 → x + y = 46 → min x y = 18.5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2938_293875


namespace NUMINAMATH_CALUDE_dans_age_l2938_293873

/-- Dan's present age satisfies the given condition -/
theorem dans_age : ∃ x : ℕ, (x + 18 = 8 * (x - 3) ∧ x = 6) := by sorry

end NUMINAMATH_CALUDE_dans_age_l2938_293873


namespace NUMINAMATH_CALUDE_inverse_mod_89_l2938_293852

theorem inverse_mod_89 (h : (5⁻¹ : ZMod 89) = 39) : (25⁻¹ : ZMod 89) = 8 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_89_l2938_293852


namespace NUMINAMATH_CALUDE_simplify_expression_l2938_293891

theorem simplify_expression : 5 * (18 / 7) * (21 / -45) = -6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2938_293891


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2938_293889

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500 ∧ (n + 1) * (n + 2) ≥ 500) → n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l2938_293889


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_solve_inequality_l2938_293865

def f (a x : ℝ) : ℝ := a * x^2 + (1 + a) * x + a

theorem quadratic_inequality_solutions (a : ℝ) :
  (∃ x : ℝ, f a x ≥ 0) ↔ a ≥ -1/3 :=
sorry

theorem solve_inequality (a : ℝ) (h : a > 0) :
  {x : ℝ | f a x < a - 1} =
    if a < 1 then {x : ℝ | -1/a < x ∧ x < -1}
    else if a = 1 then ∅
    else {x : ℝ | -1 < x ∧ x < -1/a} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_solve_inequality_l2938_293865


namespace NUMINAMATH_CALUDE_system_solution_l2938_293882

theorem system_solution : 
  ∃ (x y : ℝ), 
    (6.751 * x + 3.249 * y = 26.751) ∧ 
    (3.249 * x + 6.751 * y = 23.249) ∧ 
    (x = 3) ∧ 
    (y = 2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2938_293882


namespace NUMINAMATH_CALUDE_lipstick_ratio_l2938_293876

def lipstick_problem (total_students : ℕ) (blue_lipstick : ℕ) : Prop :=
  let red_lipstick := blue_lipstick * 5
  let colored_lipstick := red_lipstick * 4
  colored_lipstick * 2 = total_students

theorem lipstick_ratio :
  lipstick_problem 200 5 :=
sorry

end NUMINAMATH_CALUDE_lipstick_ratio_l2938_293876


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_l2938_293880

theorem systematic_sampling_interval 
  (total_items : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_items = 2005)
  (h2 : sample_size = 20) :
  ∃ (removed : ℕ) (interval : ℕ),
    removed < sample_size ∧
    interval * sample_size = total_items - removed ∧
    interval = 100 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_l2938_293880


namespace NUMINAMATH_CALUDE_program_output_l2938_293818

def program (a b c : ℕ) : (ℕ × ℕ × ℕ) :=
  let a' := b
  let b' := c
  let c' := a'
  (a', b', c')

theorem program_output : program 10 20 30 = (20, 30, 20) := by
  sorry

end NUMINAMATH_CALUDE_program_output_l2938_293818


namespace NUMINAMATH_CALUDE_average_age_increase_l2938_293825

theorem average_age_increase (initial_count : Nat) (replaced_age1 replaced_age2 women_avg_age : ℕ) :
  initial_count = 9 →
  replaced_age1 = 36 →
  replaced_age2 = 32 →
  women_avg_age = 52 →
  (2 * women_avg_age - (replaced_age1 + replaced_age2)) / initial_count = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_average_age_increase_l2938_293825


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2938_293812

theorem fraction_evaluation : 
  (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2938_293812


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2938_293864

/-- Given a quadratic inequality ax^2 + bx + c < 0 with solution set (1/t, t) where t > 0,
    prove certain properties about a, b, c, and related equations. -/
theorem quadratic_inequality_properties
  (a b c t : ℝ)
  (h_solution_set : ∀ x : ℝ, a * x^2 + b * x + c < 0 ↔ 1/t < x ∧ x < t)
  (h_t_pos : t > 0) :
  abc < 0 ∧
  2*a + b < 0 ∧
  (∀ x₁ x₂ : ℝ, (a * x₁ + b * Real.sqrt x₁ + c = 0 ∧
                 a * x₂ + b * Real.sqrt x₂ + c = 0) →
                x₁ + x₂ > t + 1/t) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2938_293864


namespace NUMINAMATH_CALUDE_ice_cream_bill_l2938_293899

/-- The cost of ice cream scoops for Pierre and his mom -/
theorem ice_cream_bill (cost_per_scoop : ℕ) (pierre_scoops : ℕ) (mom_scoops : ℕ) :
  cost_per_scoop = 2 → pierre_scoops = 3 → mom_scoops = 4 →
  cost_per_scoop * (pierre_scoops + mom_scoops) = 14 :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_bill_l2938_293899


namespace NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersection_points_l2938_293831

/-- The number of interior intersection points in an n by n grid of squares -/
def interior_intersection_points (n : ℕ) : ℕ := (n - 1) * (n - 1)

/-- Theorem: The number of interior intersection points in a 12 by 12 grid of squares is 121 -/
theorem twelve_by_twelve_grid_intersection_points :
  interior_intersection_points 12 = 121 := by
  sorry

end NUMINAMATH_CALUDE_twelve_by_twelve_grid_intersection_points_l2938_293831


namespace NUMINAMATH_CALUDE_taxi_fare_problem_l2938_293826

/-- Represents the fare for a taxi ride. -/
structure TaxiFare where
  distance : ℝ  -- Distance traveled in kilometers
  cost : ℝ      -- Cost in dollars
  h_positive : distance > 0

/-- States that taxi fares are directly proportional to the distance traveled. -/
def DirectlyProportional (f₁ f₂ : TaxiFare) : Prop :=
  f₁.cost / f₁.distance = f₂.cost / f₂.distance

theorem taxi_fare_problem (f₁ : TaxiFare) 
    (h₁ : f₁.distance = 80 ∧ f₁.cost = 200) :
    ∃ (f₂ : TaxiFare), 
      f₂.distance = 120 ∧ 
      DirectlyProportional f₁ f₂ ∧ 
      f₂.cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_problem_l2938_293826


namespace NUMINAMATH_CALUDE_sum_of_distances_bound_l2938_293810

/-- A rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_ge_width : length ≥ width

/-- A point inside a rectangle -/
structure PointInRectangle (rect : Rectangle) where
  x : ℝ
  y : ℝ
  x_bounds : 0 ≤ x ∧ x ≤ rect.length
  y_bounds : 0 ≤ y ∧ y ≤ rect.width

/-- The sum of distances from a point to the extensions of all sides of a rectangle -/
def sum_of_distances (rect : Rectangle) (p : PointInRectangle rect) : ℝ :=
  p.x + (rect.length - p.x) + p.y + (rect.width - p.y)

/-- The theorem stating that the sum of distances is at most 2l + 2w -/
theorem sum_of_distances_bound (rect : Rectangle) (p : PointInRectangle rect) :
  sum_of_distances rect p ≤ 2 * rect.length + 2 * rect.width := by
  sorry


end NUMINAMATH_CALUDE_sum_of_distances_bound_l2938_293810


namespace NUMINAMATH_CALUDE_A_enumeration_l2938_293829

def A : Set ℤ := {y | ∃ x : ℕ, y = 6 / (x - 2) ∧ 6 % (x - 2) = 0}

theorem A_enumeration : A = {-3, -6, 6, 3, 2, 1} := by
  sorry

end NUMINAMATH_CALUDE_A_enumeration_l2938_293829


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2938_293811

/-- Given an ellipse C with semi-major axis a and semi-minor axis b,
    and a circle with diameter 2a tangent to a line,
    prove that the eccentricity of C is √(6)/3 -/
theorem ellipse_eccentricity (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 + y^2 / b^2 = 1}
  let L := {(x, y) : ℝ × ℝ | b * x - a * y + 2 * a * b = 0}
  let circle_diameter := 2 * a
  (∃ (p : ℝ × ℝ), p ∈ C ∧ p ∈ L) →  -- The circle is tangent to the line
  let e := Real.sqrt (1 - b^2 / a^2)  -- Eccentricity definition
  e = Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2938_293811


namespace NUMINAMATH_CALUDE_equation_solution_l2938_293817

theorem equation_solution : 
  ∀ x : ℂ, (13*x - x^2)/(x + 1) * (x + (13 - x)/(x + 1)) = 54 ↔ 
  x = 3 ∨ x = 6 ∨ x = (5 + Complex.I * Real.sqrt 11)/2 ∨ x = (5 - Complex.I * Real.sqrt 11)/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2938_293817


namespace NUMINAMATH_CALUDE_linear_equation_m_value_l2938_293869

theorem linear_equation_m_value (m : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (m - 1) * x^(abs m) - 2 = a * x + b) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_m_value_l2938_293869


namespace NUMINAMATH_CALUDE_bertha_family_no_daughters_bertha_family_no_daughters_is_32_l2938_293841

/-- Represents the family structure of Bertha and her descendants --/
structure BerthaFamily where
  daughters : Nat
  granddaughters : Nat
  daughters_with_children : Nat

/-- The properties of Bertha's family --/
def bertha_family : BerthaFamily where
  daughters := 8
  granddaughters := 32
  daughters_with_children := 8

theorem bertha_family_no_daughters : Nat :=
  let total := bertha_family.daughters + bertha_family.granddaughters
  let with_daughters := bertha_family.daughters_with_children
  total - with_daughters
  
#check bertha_family_no_daughters

theorem bertha_family_no_daughters_is_32 :
  bertha_family_no_daughters = 32 := by
  sorry

#check bertha_family_no_daughters_is_32

end NUMINAMATH_CALUDE_bertha_family_no_daughters_bertha_family_no_daughters_is_32_l2938_293841


namespace NUMINAMATH_CALUDE_third_number_problem_l2938_293838

theorem third_number_problem (first second third : ℕ) : 
  (3 * first + 3 * second + 3 * third + 11 = 170) →
  (first = 16) →
  (second = 17) →
  third = 20 := by
sorry

end NUMINAMATH_CALUDE_third_number_problem_l2938_293838


namespace NUMINAMATH_CALUDE_min_skittles_proof_l2938_293837

def min_skittles : ℕ := 150

theorem min_skittles_proof :
  (∀ n : ℕ, n ≥ min_skittles ∧ n % 19 = 17 → n ≥ min_skittles) ∧
  min_skittles % 19 = 17 :=
sorry

end NUMINAMATH_CALUDE_min_skittles_proof_l2938_293837


namespace NUMINAMATH_CALUDE_rat_value_l2938_293813

/-- Represents the alphabet with corresponding numeric values. --/
def alphabet : List (Char × Nat) := [
  ('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7), ('h', 8), ('i', 9), ('j', 10),
  ('k', 11), ('l', 12), ('m', 13), ('n', 14), ('o', 15), ('p', 16), ('q', 17), ('r', 18), ('s', 19),
  ('t', 20), ('u', 21), ('v', 22), ('w', 23), ('x', 24), ('y', 25), ('z', 26)
]

/-- Gets the numeric value of a character based on its position in the alphabet. --/
def letterValue (c : Char) : Nat :=
  (alphabet.find? (fun p => p.1 == c.toLower)).map Prod.snd |>.getD 0

/-- Calculates the number value of a word based on the given rules. --/
def wordValue (word : String) : Nat :=
  let letterSum := word.toList.map letterValue |>.sum
  letterSum * word.length

/-- Theorem stating that the number value of "rat" is 117. --/
theorem rat_value : wordValue "rat" = 117 := by
  sorry

end NUMINAMATH_CALUDE_rat_value_l2938_293813


namespace NUMINAMATH_CALUDE_annual_interest_calculation_l2938_293845

theorem annual_interest_calculation (principal : ℝ) (quarterly_rate : ℝ) :
  principal = 10000 →
  quarterly_rate = 0.05 →
  (principal * quarterly_rate * 4) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_annual_interest_calculation_l2938_293845


namespace NUMINAMATH_CALUDE_cassies_dogs_l2938_293820

/-- The number of parrots Cassie has -/
def num_parrots : ℕ := 8

/-- The number of nails per dog foot -/
def nails_per_dog_foot : ℕ := 4

/-- The number of feet a dog has -/
def dog_feet : ℕ := 4

/-- The number of claws per parrot leg -/
def claws_per_parrot_leg : ℕ := 3

/-- The number of legs a parrot has -/
def parrot_legs : ℕ := 2

/-- The total number of nails Cassie needs to cut -/
def total_nails : ℕ := 113

/-- The number of dogs Cassie has -/
def num_dogs : ℕ := 4

theorem cassies_dogs :
  num_dogs = 4 :=
by sorry

end NUMINAMATH_CALUDE_cassies_dogs_l2938_293820


namespace NUMINAMATH_CALUDE_kangaroo_six_hops_l2938_293824

def hop_distance (n : ℕ) : ℚ :=
  1 - (3/4)^n

theorem kangaroo_six_hops :
  hop_distance 6 = 3367 / 4096 := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_six_hops_l2938_293824


namespace NUMINAMATH_CALUDE_stable_number_theorem_l2938_293878

/-- Definition of a stable number -/
def is_stable (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  (n / 100 ≠ 0) ∧ ((n / 10) % 10 ≠ 0) ∧ (n % 10 ≠ 0) ∧
  (n / 100 + (n / 10) % 10 > n % 10) ∧
  (n / 100 + n % 10 > (n / 10) % 10) ∧
  ((n / 10) % 10 + n % 10 > n / 100)

/-- Definition of F(n) -/
def F (n : ℕ) : ℕ := (n % 10) * 10 + (n / 10) % 10

/-- Definition of Q(n) -/
def Q (n : ℕ) : ℕ := ((n / 10) % 10) * 10 + n / 100

/-- Main theorem -/
theorem stable_number_theorem (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 1 ≤ b ∧ b ≤ 4) :
  let s := 100 * a + 101 * b + 30
  is_stable s ∧ (5 * F s + 2 * Q s) % 11 = 0 → s = 432 ∨ s = 534 := by
  sorry

end NUMINAMATH_CALUDE_stable_number_theorem_l2938_293878


namespace NUMINAMATH_CALUDE_circle_properties_l2938_293809

-- Define the circle equation type
def CircleEquation := ℝ → ℝ → ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the properties of the circles and points
def CircleProperties (f : CircleEquation) (P₁ P₂ : Point) :=
  f P₁.x P₁.y = 0 ∧ f P₂.x P₂.y ≠ 0

-- Define the new circle equation
def NewCircleEquation (f : CircleEquation) (P₁ P₂ : Point) : CircleEquation :=
  fun x y => f x y - f P₁.x P₁.y - f P₂.x P₂.y

-- Theorem statement
theorem circle_properties
  (f : CircleEquation)
  (P₁ P₂ : Point)
  (h : CircleProperties f P₁ P₂) :
  let g := NewCircleEquation f P₁ P₂
  (g P₂.x P₂.y = 0) ∧
  (∀ x y, g x y = 0 → f x y = f P₂.x P₂.y) := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l2938_293809


namespace NUMINAMATH_CALUDE_total_green_is_seven_l2938_293830

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The number of green marbles Tom has -/
def tom_green : ℕ := 4

/-- The total number of green marbles Sara and Tom have together -/
def total_green : ℕ := sara_green + tom_green

/-- Theorem stating that the total number of green marbles is 7 -/
theorem total_green_is_seven : total_green = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_green_is_seven_l2938_293830


namespace NUMINAMATH_CALUDE_sidorov_cash_calculation_l2938_293898

/-- The disposable cash of the Sidorov family as of June 1, 2018 -/
def sidorov_cash : ℝ := 724506.3

/-- The first component of the Sidorov family's cash -/
def cash_component1 : ℝ := 496941.3

/-- The second component of the Sidorov family's cash -/
def cash_component2 : ℝ := 227565.0

/-- Theorem stating that the sum of the two cash components equals the total disposable cash -/
theorem sidorov_cash_calculation : 
  cash_component1 + cash_component2 = sidorov_cash := by
  sorry

end NUMINAMATH_CALUDE_sidorov_cash_calculation_l2938_293898


namespace NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l2938_293887

theorem x_power_2048_minus_reciprocal (x : ℂ) (h : x + 1/x = Complex.I * Real.sqrt 2) :
  x^2048 - 1/x^2048 = 14^512 - 1024 := by
  sorry

end NUMINAMATH_CALUDE_x_power_2048_minus_reciprocal_l2938_293887


namespace NUMINAMATH_CALUDE_bernardo_wins_l2938_293800

theorem bernardo_wins (N : ℕ) : N = 78 ↔ 
  N ∈ Finset.range 1000 ∧ 
  (∀ m : ℕ, m < N → m ∉ Finset.range 1000 ∨ 
    3 * m ≥ 1000 ∨ 
    3 * m + 75 ≥ 1000 ∨ 
    9 * m + 225 ≥ 1000 ∨ 
    9 * m + 300 < 1000) ∧
  3 * N < 1000 ∧
  3 * N + 75 < 1000 ∧
  9 * N + 225 < 1000 ∧
  9 * N + 300 ≥ 1000 := by
sorry

#eval (78 / 10) + (78 % 10)  -- Sum of digits of 78

end NUMINAMATH_CALUDE_bernardo_wins_l2938_293800


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l2938_293851

theorem magical_red_knights_fraction 
  (total_knights : ℕ) 
  (total_knights_pos : total_knights > 0)
  (red_knights : ℕ) 
  (blue_knights : ℕ) 
  (magical_knights : ℕ) 
  (red_knights_fraction : red_knights = (3 * total_knights) / 8)
  (blue_knights_fraction : blue_knights = total_knights - red_knights)
  (magical_knights_fraction : magical_knights = total_knights / 4)
  (magical_ratio : ∃ (p q : ℕ) (p_pos : p > 0) (q_pos : q > 0), 
    red_knights * p * 3 = blue_knights * p * q ∧ 
    red_knights * p + blue_knights * p = magical_knights * q) :
  ∃ (p q : ℕ) (p_pos : p > 0) (q_pos : q > 0), 
    7 * p = 3 * q ∧ 
    red_knights * p = magical_knights * q := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l2938_293851


namespace NUMINAMATH_CALUDE_point_inside_circle_l2938_293890

theorem point_inside_circle (O P : Point) (r OP : ℝ) : 
  r = 4 → OP = 3 → OP < r :=
sorry

end NUMINAMATH_CALUDE_point_inside_circle_l2938_293890


namespace NUMINAMATH_CALUDE_length_F_to_F_prime_l2938_293846

/-- Triangle DEF with vertices D(-1, 3), E(5, -1), and F(-4, -2) is reflected over the y-axis.
    This theorem proves that the length of the segment from F to F' is 8. -/
theorem length_F_to_F_prime (D E F : ℝ × ℝ) : 
  D = (-1, 3) → E = (5, -1) → F = (-4, -2) → 
  let F' := (-(F.1), F.2)
  abs (F'.1 - F.1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_length_F_to_F_prime_l2938_293846


namespace NUMINAMATH_CALUDE_magnitude_sum_perpendicular_vectors_l2938_293886

/-- Given two vectors a and b in R², where a is perpendicular to b,
    prove that the magnitude of a + 2b is 2√10 -/
theorem magnitude_sum_perpendicular_vectors
  (a b : ℝ × ℝ)
  (h1 : a.1 = 4)
  (h2 : b = (1, -2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ‖a + 2 • b‖ = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_sum_perpendicular_vectors_l2938_293886


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2938_293847

/-- A function to check if a number is a palindrome in a given base -/
def isPalindromeInBase (n : ℕ) (base : ℕ) : Prop := sorry

/-- A function to convert a number from base 10 to another base -/
def toBase (n : ℕ) (base : ℕ) : List ℕ := sorry

theorem smallest_dual_base_palindrome : 
  ∀ k : ℕ, k > 15 → isPalindromeInBase k 3 → isPalindromeInBase k 5 → k ≥ 26 :=
sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l2938_293847


namespace NUMINAMATH_CALUDE_at_hash_sum_l2938_293859

def at_operation (a b : ℕ+) : ℚ := (a.val * b.val : ℚ) / (a.val + b.val)

def hash_operation (a b : ℕ+) : ℚ := (a.val + 3 * b.val : ℚ) / (b.val + 3 * a.val)

theorem at_hash_sum :
  (at_operation 3 9) + (hash_operation 3 9) = 47 / 12 := by sorry

end NUMINAMATH_CALUDE_at_hash_sum_l2938_293859


namespace NUMINAMATH_CALUDE_subset_implication_l2938_293863

theorem subset_implication (A B : Set ℕ) (a : ℕ) :
  A = {1, a} ∧ B = {1, 2, 3} →
  (a = 3 → A ⊆ B) := by
  sorry

end NUMINAMATH_CALUDE_subset_implication_l2938_293863


namespace NUMINAMATH_CALUDE_rectangle_equation_l2938_293849

theorem rectangle_equation (x : ℝ) : 
  (∀ L W : ℝ, L * W = 864 ∧ L + W = 60 ∧ L = W + x) →
  (60 - x) / 2 * (60 + x) / 2 = 864 := by
sorry

end NUMINAMATH_CALUDE_rectangle_equation_l2938_293849


namespace NUMINAMATH_CALUDE_complex_fraction_real_implies_zero_l2938_293856

theorem complex_fraction_real_implies_zero (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) + 2 * Complex.I) / ((a : ℂ) - 2 * Complex.I)).im = 0 →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_real_implies_zero_l2938_293856


namespace NUMINAMATH_CALUDE_range_of_a_l2938_293897

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 2 3, x^2 - a ≥ 0) → a ∈ Set.Iic 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2938_293897


namespace NUMINAMATH_CALUDE_expression_value_l2938_293884

theorem expression_value : (4.7 * 13.26 + 4.7 * 9.43 + 4.7 * 77.31) = 470 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2938_293884


namespace NUMINAMATH_CALUDE_ammonia_composition_l2938_293858

/-- The mass percentage of Nitrogen in Ammonia -/
def nitrogen_percentage : ℝ := 77.78

/-- The mass percentage of Hydrogen in Ammonia -/
def hydrogen_percentage : ℝ := 100 - nitrogen_percentage

theorem ammonia_composition :
  hydrogen_percentage = 22.22 := by sorry

end NUMINAMATH_CALUDE_ammonia_composition_l2938_293858


namespace NUMINAMATH_CALUDE_job_completion_time_l2938_293853

/-- Given two workers A and B, where A completes a job in 10 days and B completes it in 6 days,
    prove that they can complete the job together in 3.75 days. -/
theorem job_completion_time (a_time b_time combined_time : ℝ) 
  (ha : a_time = 10) 
  (hb : b_time = 6) 
  (hc : combined_time = (a_time * b_time) / (a_time + b_time)) : 
  combined_time = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2938_293853


namespace NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2938_293839

/-- Given a hyperbola with equation mx^2 + y^2 = 1, if its conjugate axis is twice the length
of its transverse axis, then m = -1/4 -/
theorem hyperbola_axis_ratio (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →  -- Equation of the hyperbola
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧       -- Existence of positive a and b
    (∀ x y : ℝ, y^2 / b^2 - x^2 / a^2 = 1) ∧  -- Standard form of hyperbola
    2 * b = 2 * a) →                -- Conjugate axis is twice the transverse axis
  m = -1/4 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_axis_ratio_l2938_293839


namespace NUMINAMATH_CALUDE_trip_average_speed_l2938_293814

/-- Calculates the average speed given three segments of a trip -/
def average_speed (d1 d2 d3 t1 t2 t3 : ℚ) : ℚ :=
  (d1 + d2 + d3) / (t1 + t2 + t3)

/-- Theorem: The average speed for the given trip is 1200/18 miles per hour -/
theorem trip_average_speed :
  average_speed 420 480 300 6 7 5 = 1200 / 18 := by
  sorry

end NUMINAMATH_CALUDE_trip_average_speed_l2938_293814


namespace NUMINAMATH_CALUDE_distance_between_points_l2938_293895

theorem distance_between_points : Real.sqrt 89 = Real.sqrt ((1 - (-4))^2 + (-3 - 5)^2) := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2938_293895


namespace NUMINAMATH_CALUDE_parallelogram_area_crossing_boards_l2938_293855

/-- The area of a parallelogram formed by two boards crossing at a 45-degree angle -/
theorem parallelogram_area_crossing_boards (board1_width board2_width : ℝ) 
  (h1 : board1_width = 5)
  (h2 : board2_width = 7)
  (h3 : Real.pi / 4 = 45 * Real.pi / 180) : 
  board1_width * (board2_width * Real.sin (Real.pi / 4)) = 35 * Real.sqrt 2 / 2 := by
  sorry

#check parallelogram_area_crossing_boards

end NUMINAMATH_CALUDE_parallelogram_area_crossing_boards_l2938_293855


namespace NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l2938_293854

/-- Given a line passing through points (4, 0) and (-4, -4), 
    prove that the y-coordinate of the point on this line with x-coordinate 10 is 3. -/
theorem line_y_coordinate_at_x_10 (L : Set (ℝ × ℝ)) :
  ((4 : ℝ), 0) ∈ L →
  ((-4 : ℝ), -4) ∈ L →
  ∃ m b : ℝ, ∀ x y : ℝ, (x, y) ∈ L ↔ y = m * x + b →
  (10, 3) ∈ L := by
sorry

end NUMINAMATH_CALUDE_line_y_coordinate_at_x_10_l2938_293854


namespace NUMINAMATH_CALUDE_bucket_weight_l2938_293827

/-- Given a bucket with unknown empty weight and full water weight,
    if it weighs c kilograms when three-quarters full and b kilograms when half-full,
    then its weight when one-third full is (5/3)b - (2/3)c kilograms. -/
theorem bucket_weight (b c : ℝ) : 
  (∃ x y : ℝ, x + 3/4 * y = c ∧ x + 1/2 * y = b) → 
  (∃ z : ℝ, z = 5/3 * b - 2/3 * c ∧ 
    ∀ x y : ℝ, x + 3/4 * y = c → x + 1/2 * y = b → x + 1/3 * y = z) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l2938_293827


namespace NUMINAMATH_CALUDE_donkey_elephant_weight_difference_l2938_293879

-- Define the weights and conversion factor
def elephant_weight_tons : ℝ := 3
def pounds_per_ton : ℝ := 2000
def combined_weight_pounds : ℝ := 6600

-- Define the theorem
theorem donkey_elephant_weight_difference : 
  let elephant_weight_pounds := elephant_weight_tons * pounds_per_ton
  let donkey_weight_pounds := combined_weight_pounds - elephant_weight_pounds
  let weight_difference_percentage := (elephant_weight_pounds - donkey_weight_pounds) / elephant_weight_pounds * 100
  weight_difference_percentage = 90 := by
sorry

end NUMINAMATH_CALUDE_donkey_elephant_weight_difference_l2938_293879


namespace NUMINAMATH_CALUDE_intersection_theorem_l2938_293844

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 0}
def N : Set ℝ := {x : ℝ | x^2 - 4 > 0}

-- State the theorem
theorem intersection_theorem :
  M ∩ (Set.univ \ N) = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l2938_293844


namespace NUMINAMATH_CALUDE_equation_solution_l2938_293892

theorem equation_solution : 
  ∃! x : ℝ, 2 * x - 1 = 3 * x + 2 ∧ x = -3 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2938_293892


namespace NUMINAMATH_CALUDE_toothpicks_for_2003_base_l2938_293806

def small_triangles (n : ℕ) : ℕ := n * (n + 1) / 2

def toothpicks (base : ℕ) : ℕ :=
  let total_triangles := small_triangles base
  3 * total_triangles / 2

theorem toothpicks_for_2003_base :
  toothpicks 2003 = 3010554 :=
by sorry

end NUMINAMATH_CALUDE_toothpicks_for_2003_base_l2938_293806


namespace NUMINAMATH_CALUDE_banana_distribution_problem_l2938_293872

/-- Proves that the number of absent children is 330 given the conditions of the banana distribution problem -/
theorem banana_distribution_problem (total_children : ℕ) 
  (h1 : total_children = 660)
  (h2 : ∃ (total_bananas : ℕ), total_bananas = total_children * 2)
  (h3 : ∃ (present_children : ℕ), present_children * 4 = total_children * 2) :
  total_children - (total_children * 2) / 4 = 330 := by
  sorry

end NUMINAMATH_CALUDE_banana_distribution_problem_l2938_293872


namespace NUMINAMATH_CALUDE_age_score_ratio_l2938_293862

def almas_age : ℕ := 20
def melinas_age : ℕ := 60
def almas_score : ℕ := 40

theorem age_score_ratio :
  (almas_age + melinas_age) / almas_score = 2 ∧
  melinas_age = 3 * almas_age ∧
  melinas_age = 60 ∧
  almas_score = 40 := by
  sorry

end NUMINAMATH_CALUDE_age_score_ratio_l2938_293862


namespace NUMINAMATH_CALUDE_cube_square_equation_solution_l2938_293801

theorem cube_square_equation_solution :
  2^3 - 7 = 3^2 + (-8) := by sorry

end NUMINAMATH_CALUDE_cube_square_equation_solution_l2938_293801


namespace NUMINAMATH_CALUDE_li_cake_purchase_l2938_293842

theorem li_cake_purchase 
  (fruit_price : ℝ) 
  (chocolate_price : ℝ) 
  (total_spent : ℝ) 
  (average_price : ℝ)
  (h1 : fruit_price = 4.8)
  (h2 : chocolate_price = 6.6)
  (h3 : total_spent = 167.4)
  (h4 : average_price = 6.2) :
  ∃ (fruit_count chocolate_count : ℕ),
    fruit_count = 6 ∧ 
    chocolate_count = 21 ∧
    fruit_count * fruit_price + chocolate_count * chocolate_price = total_spent ∧
    (fruit_count + chocolate_count : ℝ) * average_price = total_spent :=
by sorry

end NUMINAMATH_CALUDE_li_cake_purchase_l2938_293842


namespace NUMINAMATH_CALUDE_m_range_l2938_293867

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x, x^2 + m*x + 1 ≥ 0

def q (m : ℝ) : Prop := ∀ x, (8*x + 4*(m - 1)) ≠ 0

-- Define the theorem
theorem m_range (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → ((-2 ≤ m ∧ m < 1) ∨ m > 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2938_293867


namespace NUMINAMATH_CALUDE_ellipse_inequality_l2938_293828

/-- An ellipse with equation ax^2 + by^2 = 1 and foci on the x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  is_ellipse : a > 0 ∧ b > 0
  foci_on_x_axis : True  -- We can't directly express this condition in Lean, so we use True as a placeholder

/-- Theorem: For an ellipse ax^2 + by^2 = 1 with foci on the x-axis, 0 < a < b -/
theorem ellipse_inequality (e : Ellipse) : 0 < e.a ∧ e.a < e.b := by
  sorry

end NUMINAMATH_CALUDE_ellipse_inequality_l2938_293828


namespace NUMINAMATH_CALUDE_polygon_sides_l2938_293896

theorem polygon_sides (n : ℕ) (sum_interior_angles : ℝ) : sum_interior_angles = 1440 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l2938_293896


namespace NUMINAMATH_CALUDE_matching_shoes_probability_l2938_293819

/-- A box containing pairs of shoes -/
structure ShoeBox where
  pairs : ℕ
  total : ℕ
  total_eq_twice_pairs : total = 2 * pairs

/-- The probability of selecting two matching shoes from a ShoeBox -/
def matchingProbability (box : ShoeBox) : ℚ :=
  1 / (box.total - 1)

theorem matching_shoes_probability (box : ShoeBox) 
  (h : box.pairs = 100) : 
  matchingProbability box = 1 / 199 := by
  sorry

end NUMINAMATH_CALUDE_matching_shoes_probability_l2938_293819


namespace NUMINAMATH_CALUDE_sales_percentage_other_l2938_293840

theorem sales_percentage_other (total_percentage : ℝ) (markers_percentage : ℝ) (notebooks_percentage : ℝ)
  (h1 : total_percentage = 100)
  (h2 : markers_percentage = 42)
  (h3 : notebooks_percentage = 22) :
  total_percentage - markers_percentage - notebooks_percentage = 36 := by
sorry

end NUMINAMATH_CALUDE_sales_percentage_other_l2938_293840


namespace NUMINAMATH_CALUDE_certain_number_equation_l2938_293881

theorem certain_number_equation : ∃ x : ℚ, 1038 * x = 173 * 240 ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l2938_293881


namespace NUMINAMATH_CALUDE_plants_given_to_friend_l2938_293866

def initial_plants : ℕ := 3
def months : ℕ := 3
def remaining_plants : ℕ := 20

def plants_after_doubling (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

theorem plants_given_to_friend :
  plants_after_doubling initial_plants months - remaining_plants = 4 := by
  sorry

end NUMINAMATH_CALUDE_plants_given_to_friend_l2938_293866


namespace NUMINAMATH_CALUDE_problem_1_l2938_293857

theorem problem_1 (a : ℝ) : a^3 * a + (2*a^2)^2 = 5*a^4 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2938_293857


namespace NUMINAMATH_CALUDE_circumradius_is_five_l2938_293802

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ := sorry

-- Assert that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the centroid G of triangle F₁PF₂
def G : ℝ × ℝ := sorry

-- Define the incenter I of triangle F₁PF₂
def I : ℝ × ℝ := sorry

-- Assert that G and I are parallel to the x-axis
axiom G_I_parallel_x : G.2 = I.2

-- Define the circumradius of triangle F₁PF₂
def circumradius (F₁ F₂ P : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumradius of triangle F₁PF₂ is 5
theorem circumradius_is_five : circumradius F₁ F₂ P = 5 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_is_five_l2938_293802


namespace NUMINAMATH_CALUDE_deepak_present_age_l2938_293804

/-- The ratio of ages between Rahul, Deepak, and Sameer -/
def age_ratio : Fin 3 → ℕ
  | 0 => 4  -- Rahul
  | 1 => 3  -- Deepak
  | 2 => 5  -- Sameer

/-- The number of years in the future we're considering -/
def years_future : ℕ := 6

/-- Rahul's age after the specified number of years -/
def rahul_future_age : ℕ := 26

/-- Proves that given the age ratio and Rahul's future age, Deepak's present age is 15 years -/
theorem deepak_present_age :
  ∃ (k : ℕ),
    (age_ratio 0 * k + years_future = rahul_future_age) ∧
    (age_ratio 1 * k = 15) := by
  sorry

end NUMINAMATH_CALUDE_deepak_present_age_l2938_293804


namespace NUMINAMATH_CALUDE_factor_expression_l2938_293833

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) + 2 * (y - 4) = (3 * y + 7) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2938_293833


namespace NUMINAMATH_CALUDE_janes_shadow_length_l2938_293835

/-- Given a tree and a person (Jane) casting shadows, this theorem proves
    the length of Jane's shadow based on the heights of the tree and Jane,
    and the length of the tree's shadow. -/
theorem janes_shadow_length
  (tree_height : ℝ)
  (tree_shadow : ℝ)
  (jane_height : ℝ)
  (h_tree_height : tree_height = 30)
  (h_tree_shadow : tree_shadow = 10)
  (h_jane_height : jane_height = 1.5) :
  jane_height * tree_shadow / tree_height = 0.5 := by
  sorry


end NUMINAMATH_CALUDE_janes_shadow_length_l2938_293835


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2938_293888

theorem quadratic_equation_roots (c : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 16 * x + c = 0 ↔ x = (-16 + Real.sqrt 24) / 4 ∨ x = (-16 - Real.sqrt 24) / 4) →
  c = 29 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2938_293888


namespace NUMINAMATH_CALUDE_area_between_tangents_and_curve_l2938_293816

noncomputable section

-- Define the curve
def C (x : ℝ) : ℝ := 1 / x

-- Define the points P and Q
def P (a : ℝ) : ℝ × ℝ := (a, C a)
def Q (a : ℝ) : ℝ × ℝ := (2*a, C (2*a))

-- Define the tangent lines at P and Q
def l (a : ℝ) (x : ℝ) : ℝ := -1/(a^2) * x + 2/a
def m (a : ℝ) (x : ℝ) : ℝ := -1/(4*a^2) * x + 1/a

-- Define the area function
def area (a : ℝ) : ℝ :=
  ∫ x in a..(2*a), (C x - l a x) + (C x - m a x)

-- State the theorem
theorem area_between_tangents_and_curve (a : ℝ) (h : a > 0) :
  area a = 2 * Real.log 2 - 9/8 :=
sorry

end

end NUMINAMATH_CALUDE_area_between_tangents_and_curve_l2938_293816


namespace NUMINAMATH_CALUDE_election_votes_l2938_293885

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (excess_percent : ℚ) 
  (h_total : total_votes = 6720)
  (h_invalid : invalid_percent = 1/5)
  (h_excess : excess_percent = 3/20) :
  ∃ (votes_b : ℕ), votes_b = 2184 ∧ 
  (↑votes_b : ℚ) + (↑votes_b + excess_percent * total_votes) = (1 - invalid_percent) * total_votes :=
sorry

end NUMINAMATH_CALUDE_election_votes_l2938_293885


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2938_293848

theorem negation_of_universal_statement :
  ¬(∀ a : ℝ, ∃ x : ℝ, x > 0 ∧ a * x^2 - 3 * x - a = 0) ↔
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → a * x^2 - 3 * x - a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2938_293848


namespace NUMINAMATH_CALUDE_parabola_directrix_l2938_293870

/-- The directrix of a parabola y^2 = 16x is x = -4 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 16*x → (∃ (a : ℝ), a = 4 ∧ x = -a) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2938_293870


namespace NUMINAMATH_CALUDE_exam_score_calculation_l2938_293805

theorem exam_score_calculation (total_questions : ℕ) (total_marks : ℕ) (correct_answers : ℕ) 
  (h1 : total_questions = 60)
  (h2 : total_marks = 150)
  (h3 : correct_answers = 42)
  (h4 : total_questions = correct_answers + (total_questions - correct_answers)) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l2938_293805


namespace NUMINAMATH_CALUDE_min_value_theorem_l2938_293860

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2938_293860


namespace NUMINAMATH_CALUDE_binomial_15_4_l2938_293807

theorem binomial_15_4 : Nat.choose 15 4 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_4_l2938_293807


namespace NUMINAMATH_CALUDE_caramel_apple_ice_cream_cost_difference_l2938_293821

/-- The cost difference between a caramel apple and an ice cream cone -/
theorem caramel_apple_ice_cream_cost_difference 
  (caramel_apple_cost : ℕ) 
  (ice_cream_cost : ℕ) 
  (h1 : caramel_apple_cost = 25)
  (h2 : ice_cream_cost = 15) : 
  caramel_apple_cost - ice_cream_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_caramel_apple_ice_cream_cost_difference_l2938_293821


namespace NUMINAMATH_CALUDE_wall_bricks_count_l2938_293850

/-- Represents the time (in hours) it takes Ben to build the wall alone -/
def ben_time : ℝ := 12

/-- Represents the time (in hours) it takes Arya to build the wall alone -/
def arya_time : ℝ := 15

/-- Represents the reduction in combined output (in bricks per hour) due to chattiness -/
def chattiness_reduction : ℝ := 15

/-- Represents the time (in hours) it takes Ben and Arya to build the wall together -/
def combined_time : ℝ := 6

/-- Represents the number of bricks in the wall -/
def wall_bricks : ℝ := 900

theorem wall_bricks_count : 
  ben_time * arya_time * (1 / ben_time + 1 / arya_time - chattiness_reduction / wall_bricks) * combined_time = arya_time + ben_time := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l2938_293850


namespace NUMINAMATH_CALUDE_monotonic_functional_equation_implies_f_zero_eq_one_l2938_293803

/-- A function f: ℝ → ℝ is monotonic if for all x, y ∈ ℝ, x ≤ y implies f(x) ≤ f(y) -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- A function f: ℝ → ℝ satisfies the functional equation f(x+y) = f(x)f(y) for all x, y ∈ ℝ -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

theorem monotonic_functional_equation_implies_f_zero_eq_one
  (f : ℝ → ℝ) (h_mono : Monotonic f) (h_eq : SatisfiesFunctionalEquation f) :
  f 0 = 1 :=
sorry

end NUMINAMATH_CALUDE_monotonic_functional_equation_implies_f_zero_eq_one_l2938_293803


namespace NUMINAMATH_CALUDE_strictly_increasing_f_implies_a_nonneg_l2938_293834

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- State the theorem
theorem strictly_increasing_f_implies_a_nonneg 
  (h : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_strictly_increasing_f_implies_a_nonneg_l2938_293834


namespace NUMINAMATH_CALUDE_no_collision_probability_correct_l2938_293883

/-- A regular icosahedron -/
structure Icosahedron :=
  (vertices : Fin 12 → Type)
  (adjacent : Fin 12 → Fin 5 → Fin 12)

/-- An ant on the icosahedron -/
structure Ant :=
  (position : Fin 12)

/-- The probability of an ant moving to a specific adjacent vertex -/
def move_probability : ℚ := 1 / 5

/-- The number of ants -/
def num_ants : ℕ := 12

/-- The probability that no two ants arrive at the same vertex -/
def no_collision_probability (i : Icosahedron) : ℚ :=
  (Nat.factorial num_ants : ℚ) / (5 ^ num_ants)

theorem no_collision_probability_correct (i : Icosahedron) :
  no_collision_probability i = (Nat.factorial num_ants : ℚ) / (5 ^ num_ants) :=
sorry

end NUMINAMATH_CALUDE_no_collision_probability_correct_l2938_293883


namespace NUMINAMATH_CALUDE_expression_evaluation_l2938_293861

theorem expression_evaluation :
  let x : ℚ := -1/3
  let y : ℚ := -1
  ((x - 2*y)^2 - (2*x + y)*(x - 4*y) - (-x + 3*y)*(x + 3*y)) / (-y) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2938_293861


namespace NUMINAMATH_CALUDE_ace_king_probability_l2938_293836

/-- The probability of drawing an Ace first and a King second from a modified deck -/
theorem ace_king_probability (total_cards : ℕ) (num_aces : ℕ) (num_kings : ℕ) 
  (h1 : total_cards = 54) 
  (h2 : num_aces = 5) 
  (h3 : num_kings = 4) : 
  (num_aces : ℚ) / total_cards * num_kings / (total_cards - 1) = 10 / 1426 := by
  sorry

end NUMINAMATH_CALUDE_ace_king_probability_l2938_293836


namespace NUMINAMATH_CALUDE_lettuce_salads_per_plant_l2938_293843

theorem lettuce_salads_per_plant (total_salads : ℕ) (plants : ℕ) (loss_fraction : ℚ) : 
  total_salads = 12 →
  loss_fraction = 1/2 →
  plants = 8 →
  (total_salads / (1 - loss_fraction)) / plants = 3 := by
  sorry

end NUMINAMATH_CALUDE_lettuce_salads_per_plant_l2938_293843


namespace NUMINAMATH_CALUDE_four_integer_average_l2938_293808

theorem four_integer_average (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different integers
  d = 90 ∧                 -- Largest integer is 90
  a ≥ 13 →                 -- Smallest integer is at least 13
  (a + b + c + d) / 4 = 33 -- Average is 33
  := by sorry

end NUMINAMATH_CALUDE_four_integer_average_l2938_293808


namespace NUMINAMATH_CALUDE_gcd_105_45_l2938_293868

theorem gcd_105_45 : Nat.gcd 105 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_105_45_l2938_293868
