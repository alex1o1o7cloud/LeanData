import Mathlib

namespace NUMINAMATH_CALUDE_slope_of_line_slope_of_specific_line_l1917_191713

/-- The slope of a line given by the equation y + ax - b = 0 is -a. -/
theorem slope_of_line (a b : ℝ) : 
  (fun x y : ℝ => y + a * x - b = 0) = (fun x y : ℝ => y = -a * x + b) := by
  sorry

/-- The slope of the line y + 3x - 1 = 0 is -3. -/
theorem slope_of_specific_line : 
  (fun x y : ℝ => y + 3 * x - 1 = 0) = (fun x y : ℝ => y = -3 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_slope_of_specific_line_l1917_191713


namespace NUMINAMATH_CALUDE_tigers_losses_l1917_191767

theorem tigers_losses (total_games wins : ℕ) (h1 : total_games = 56) (h2 : wins = 38) : 
  ∃ losses ties : ℕ, 
    losses + ties + wins = total_games ∧ 
    ties = losses / 2 ∧
    losses = 12 := by
sorry

end NUMINAMATH_CALUDE_tigers_losses_l1917_191767


namespace NUMINAMATH_CALUDE_remaining_money_l1917_191701

def octal_to_decimal (n : ℕ) : ℕ := sorry

def john_savings : ℕ := 5555

def airline_cost : ℕ := 1200
def lodging_cost : ℕ := 800
def food_cost : ℕ := 400

def total_expenses : ℕ := airline_cost + lodging_cost + food_cost

theorem remaining_money :
  octal_to_decimal john_savings - total_expenses = 525 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1917_191701


namespace NUMINAMATH_CALUDE_foot_of_perpendicular_l1917_191766

-- Define the point A
def A : ℝ × ℝ := (1, 2)

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the perpendicular line from A to the x-axis
def perp_line : Set (ℝ × ℝ) := {p | p.1 = A.1}

-- Define point M as the intersection of the perpendicular line and the x-axis
def M : ℝ × ℝ := (A.1, 0)

-- Theorem statement
theorem foot_of_perpendicular : M ∈ x_axis ∧ M ∈ perp_line := by sorry

end NUMINAMATH_CALUDE_foot_of_perpendicular_l1917_191766


namespace NUMINAMATH_CALUDE_exactly_one_real_solution_l1917_191745

theorem exactly_one_real_solution :
  ∃! x : ℝ, ((-4 * (x - 3)^2 : ℝ) ≥ 0) := by sorry

end NUMINAMATH_CALUDE_exactly_one_real_solution_l1917_191745


namespace NUMINAMATH_CALUDE_sam_distance_walked_sam_walks_25_miles_l1917_191765

-- Define the constants
def total_distance : ℝ := 55
def fred_speed : ℝ := 6
def sam_speed : ℝ := 5

-- Define the theorem
theorem sam_distance_walked : ℝ := by
  -- The distance Sam walks
  let d : ℝ := sam_speed * (total_distance / (fred_speed + sam_speed))
  -- Prove that d equals 25
  sorry

-- The main theorem
theorem sam_walks_25_miles :
  sam_distance_walked = 25 := by sorry

end NUMINAMATH_CALUDE_sam_distance_walked_sam_walks_25_miles_l1917_191765


namespace NUMINAMATH_CALUDE_even_function_comparison_l1917_191711

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- A function f is increasing on (-∞, 0) if f(x) < f(y) for all x < y < 0 -/
def IncreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → y < 0 → f x < f y

theorem even_function_comparison (f : ℝ → ℝ) (x₁ x₂ : ℝ)
    (heven : IsEven f)
    (hincr : IncreasingOnNegatives f)
    (hx₁ : x₁ < 0)
    (hsum : x₁ + x₂ > 0) :
    f x₁ > f x₂ := by
  sorry


end NUMINAMATH_CALUDE_even_function_comparison_l1917_191711


namespace NUMINAMATH_CALUDE_rational_root_of_polynomial_l1917_191787

def p (x : ℚ) : ℚ := 3 * x^5 - 4 * x^3 - 7 * x^2 + 2 * x + 1

theorem rational_root_of_polynomial :
  (p (-1/3) = 0) ∧ (∀ q : ℚ, p q = 0 → q = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_rational_root_of_polynomial_l1917_191787


namespace NUMINAMATH_CALUDE_smallest_m_is_13_l1917_191723

/-- The set of complex numbers with real part between 1/2 and √2/2 -/
def T : Set ℂ :=
  {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

/-- Property that for all n ≥ m, there exists z in T such that z^n = 1 -/
def HasNthRoot (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, z ∈ T ∧ z^n = 1

/-- 13 is the smallest positive integer satisfying the HasNthRoot property -/
theorem smallest_m_is_13 :
  HasNthRoot 13 ∧ ∀ m : ℕ, m > 0 → m < 13 → ¬HasNthRoot m :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_13_l1917_191723


namespace NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_intersection_AB_complement_P_l1917_191775

-- Define the sets A, B, and P
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for (ᶜB) ∪ P
theorem union_complement_B_P : (Bᶜ : Set ℝ) ∪ P = {x : ℝ | x ≤ 0 ∨ x ≥ 5/2} := by sorry

-- Theorem for (A ∩ B) ∩ (ᶜP)
theorem intersection_AB_complement_P : (A ∩ B) ∩ (Pᶜ : Set ℝ) = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_complement_B_P_intersection_AB_complement_P_l1917_191775


namespace NUMINAMATH_CALUDE_square_of_good_is_good_l1917_191707

def is_averaging_sequence (a : ℕ → ℤ) : Prop :=
  ∀ k, 2 * a (k + 1) = a k + a (k + 1)

def is_good_sequence (x : ℕ → ℤ) : Prop :=
  ∀ n, is_averaging_sequence (λ k => x (n + k))

theorem square_of_good_is_good (x : ℕ → ℤ) :
  is_good_sequence x → is_good_sequence (λ k => x k ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_square_of_good_is_good_l1917_191707


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l1917_191730

/-- The number of oak trees in a park after planting new trees -/
theorem oak_trees_after_planting (current : ℕ) (new : ℕ) : current = 5 → new = 4 → current + new = 9 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l1917_191730


namespace NUMINAMATH_CALUDE_stating_special_multiples_count_l1917_191749

/-- 
The count of positive integers less than 500 that are multiples of 3 but not multiples of 9.
-/
def count_special_multiples : ℕ := 
  (Finset.filter (fun n => n % 3 = 0 ∧ n % 9 ≠ 0) (Finset.range 500)).card

/-- 
Theorem stating that the count of positive integers less than 500 
that are multiples of 3 but not multiples of 9 is equal to 111.
-/
theorem special_multiples_count : count_special_multiples = 111 := by
  sorry

end NUMINAMATH_CALUDE_stating_special_multiples_count_l1917_191749


namespace NUMINAMATH_CALUDE_last_digit_of_power_difference_l1917_191718

theorem last_digit_of_power_difference : (7^95 - 3^58) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_difference_l1917_191718


namespace NUMINAMATH_CALUDE_probability_of_drawing_k_l1917_191727

/-- The probability of drawing a "K" from a standard deck of 54 playing cards -/
theorem probability_of_drawing_k (total_cards : ℕ) (k_cards : ℕ) : 
  total_cards = 54 → k_cards = 4 → (k_cards : ℚ) / total_cards = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_k_l1917_191727


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1917_191747

theorem inequality_solution_range (a : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℤ), (∀ x : ℤ, (2*x - 1)^2 < a*x^2 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) →
  a ∈ Set.Ioo (25/9) (49/16) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1917_191747


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_two_sufficient_condition_implies_m_range_l1917_191731

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_equals_two :
  ∀ m : ℝ, A ∩ B m = Set.Icc 0 3 → m = 2 := by sorry

-- Theorem 2
theorem sufficient_condition_implies_m_range :
  ∀ m : ℝ, A ⊆ Set.univ \ B m → m > 5 ∨ m < -3 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_two_sufficient_condition_implies_m_range_l1917_191731


namespace NUMINAMATH_CALUDE_spring_bud_cup_value_l1917_191785

theorem spring_bud_cup_value : ∃ x : ℕ, x + x = 578 ∧ x = 289 := by sorry

end NUMINAMATH_CALUDE_spring_bud_cup_value_l1917_191785


namespace NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1917_191725

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sum_first_six_terms :
  let a : ℚ := 1/2
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/6144 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_first_six_terms_l1917_191725


namespace NUMINAMATH_CALUDE_frankies_pets_l1917_191703

/-- The number of pets Frankie has -/
def total_pets (cats : ℕ) : ℕ :=
  let snakes := 2 * cats
  let parrots := cats - 1
  let tortoises := parrots + 1
  let dogs := 2
  let hamsters := 3
  let fish := 5
  cats + snakes + parrots + tortoises + dogs + hamsters + fish

/-- Theorem stating the total number of Frankie's pets -/
theorem frankies_pets :
  ∃ (cats : ℕ),
    2 * cats + cats + 2 = 14 ∧
    total_pets cats = 39 := by
  sorry

end NUMINAMATH_CALUDE_frankies_pets_l1917_191703


namespace NUMINAMATH_CALUDE_inverse_proposition_l1917_191706

theorem inverse_proposition : 
  (∀ a b : ℝ, (a + b = 2 → ¬(a < 1 ∧ b < 1))) ↔ 
  (∀ a b : ℝ, (a < 1 ∧ b < 1 → a + b ≠ 2)) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proposition_l1917_191706


namespace NUMINAMATH_CALUDE_root_product_l1917_191734

theorem root_product (x₁ x₂ : ℝ) (h₁ : x₁ * Real.log x₁ = 2006) (h₂ : x₂ * Real.exp x₂ = 2006) : 
  x₁ * x₂ = 2006 := by
sorry

end NUMINAMATH_CALUDE_root_product_l1917_191734


namespace NUMINAMATH_CALUDE_binary_conversion_l1917_191729

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, false, true]

-- Define the function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define the function to convert decimal to base-7
def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

-- Theorem statement
theorem binary_conversion :
  binary_to_decimal binary_num = 45 ∧
  decimal_to_base7 (binary_to_decimal binary_num) = [6, 3] := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_l1917_191729


namespace NUMINAMATH_CALUDE_integer_expression_l1917_191705

theorem integer_expression (n : ℕ) : ∃ k : ℤ, (n^5 : ℚ) / 5 + (n^3 : ℚ) / 3 + (7 * n : ℚ) / 15 = k := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_l1917_191705


namespace NUMINAMATH_CALUDE_initial_amount_proof_l1917_191704

/-- 
Theorem: If an amount increases by 1/8th of itself each year for two years 
and results in 81000, then the initial amount was 64000.
-/
theorem initial_amount_proof (P : ℚ) : 
  (P * (1 + 1/8) * (1 + 1/8) = 81000) → P = 64000 := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l1917_191704


namespace NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l1917_191781

/-- Given a rectangle with dimensions 6 and 9, prove that the ratio of the volumes of cylinders
    formed by rolling along each side is 3/4, with the larger volume in the numerator. -/
theorem rectangle_cylinder_volume_ratio :
  let rect_width : ℝ := 6
  let rect_height : ℝ := 9
  let volume1 := π * (rect_width / (2 * π))^2 * rect_height
  let volume2 := π * (rect_height / (2 * π))^2 * rect_width
  max volume1 volume2 / min volume1 volume2 = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_cylinder_volume_ratio_l1917_191781


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1917_191798

/-- Sum of a geometric series with n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The given geometric series -/
def givenSeries : List ℚ := [1/4, -1/16, 1/64, -1/256, 1/1024]

theorem geometric_series_sum :
  let a₁ : ℚ := 1/4
  let r : ℚ := -1/4
  let n : ℕ := 5
  geometricSum a₁ r n = 205/1024 ∧ givenSeries.sum = 205/1024 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1917_191798


namespace NUMINAMATH_CALUDE_f_greater_than_exp_l1917_191773

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, deriv f x > f x)
variable (h3 : f 0 = 1)

-- Theorem statement
theorem f_greater_than_exp (x : ℝ) : f x > Real.exp x ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_greater_than_exp_l1917_191773


namespace NUMINAMATH_CALUDE_max_filled_circles_in_120_l1917_191732

/-- Represents the pattern of circles where the number of ○s before each ● increases by 1 each time -/
def circle_pattern (n : ℕ) : ℕ := n * (n + 1) / 2 + n

/-- The maximum number of ● in the first 120 circles -/
def max_filled_circles : ℕ := 14

theorem max_filled_circles_in_120 :
  (∀ k, k > max_filled_circles → circle_pattern k > 120) ∧
  circle_pattern max_filled_circles ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_max_filled_circles_in_120_l1917_191732


namespace NUMINAMATH_CALUDE_unique_tangent_line_l1917_191726

/-- The function whose graph we are considering -/
def f (x : ℝ) : ℝ := x^4 - 4*x^3 - 26*x^2

/-- The line we are trying to prove is unique -/
def L (x : ℝ) : ℝ := -60*x - 225

/-- Predicate to check if a point (x, y) is on or above the line L -/
def onOrAboveLine (x y : ℝ) : Prop := y ≥ L x

/-- Predicate to check if a point (x, y) is on the graph of f -/
def onGraph (x y : ℝ) : Prop := y = f x

/-- The main theorem stating the uniqueness of the line L -/
theorem unique_tangent_line :
  (∀ x y, onGraph x y → onOrAboveLine x y) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ onGraph x₁ (L x₁) ∧ onGraph x₂ (L x₂)) ∧
  (∀ a b, (∀ x y, onGraph x y → y ≥ a*x + b) ∧
          (∃ x₁ x₂, x₁ ≠ x₂ ∧ onGraph x₁ (a*x₁ + b) ∧ onGraph x₂ (a*x₂ + b))
          → a = -60 ∧ b = -225) :=
by sorry

end NUMINAMATH_CALUDE_unique_tangent_line_l1917_191726


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1917_191710

theorem quadratic_roots_property (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c < 0) :
  ∃ (x₁ x₂ : ℝ), 
    (a * x₁^2 + b * x₁ + c = 0) ∧ 
    (a * x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁ > 0) ∧ 
    (x₂ < 0) ∧ 
    (|x₂| > |x₁|) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1917_191710


namespace NUMINAMATH_CALUDE_max_distance_from_origin_l1917_191768

def center : ℝ × ℝ := (5, -2)
def rope_length : ℝ := 12

theorem max_distance_from_origin :
  let max_dist := rope_length + Real.sqrt ((center.1 ^ 2) + (center.2 ^ 2))
  ∀ p : ℝ × ℝ, Real.sqrt ((p.1 - center.1) ^ 2 + (p.2 - center.2) ^ 2) ≤ rope_length →
    Real.sqrt (p.1 ^ 2 + p.2 ^ 2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_from_origin_l1917_191768


namespace NUMINAMATH_CALUDE_one_intersection_values_l1917_191755

/-- The function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 4) * x^2 - 2 * m * x - m - 6

/-- The discriminant of the quadratic function f(x) -/
def discriminant (m : ℝ) : ℝ := 4 * m^2 - 4 * (m - 4) * (-m - 6)

/-- Predicate to check if f(x) has only one intersection with x-axis -/
def has_one_intersection (m : ℝ) : Prop :=
  (m = 4) ∨ (discriminant m = 0)

/-- Theorem stating the values of m for which f(x) has one intersection with x-axis -/
theorem one_intersection_values :
  ∀ m : ℝ, has_one_intersection m ↔ m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end NUMINAMATH_CALUDE_one_intersection_values_l1917_191755


namespace NUMINAMATH_CALUDE_tower_combinations_l1917_191753

/-- The number of different towers of height 7 that can be built using 3 red cubes, 4 blue cubes, and 2 yellow cubes -/
def num_towers : ℕ := 5040

/-- The height of the tower -/
def tower_height : ℕ := 7

/-- The number of red cubes -/
def red_cubes : ℕ := 3

/-- The number of blue cubes -/
def blue_cubes : ℕ := 4

/-- The number of yellow cubes -/
def yellow_cubes : ℕ := 2

/-- The total number of cubes -/
def total_cubes : ℕ := red_cubes + blue_cubes + yellow_cubes

theorem tower_combinations : num_towers = 5040 := by
  sorry

end NUMINAMATH_CALUDE_tower_combinations_l1917_191753


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1917_191742

theorem functional_equation_solution (c : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, (f x + 1) * (f y + 1) = f (x + y) + f (x * y + c)) →
  c = 1 ∨ c = -1 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1917_191742


namespace NUMINAMATH_CALUDE_no_solutions_exist_l1917_191777

theorem no_solutions_exist : ¬∃ (x y z : ℝ), (x + y = 3) ∧ (x * y - z^2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_exist_l1917_191777


namespace NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l1917_191720

-- Define the wheel diameter in feet
def wheel_diameter : ℝ := 8

-- Define one mile in feet
def mile_in_feet : ℝ := 5280

-- Theorem statement
theorem wheel_revolutions_for_one_mile :
  (mile_in_feet / (π * wheel_diameter)) = 660 / π := by
  sorry

end NUMINAMATH_CALUDE_wheel_revolutions_for_one_mile_l1917_191720


namespace NUMINAMATH_CALUDE_perfect_square_in_base_k_l1917_191715

theorem perfect_square_in_base_k (k : ℤ) (h : k ≥ 6) :
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) =
  (k^4 + k^3 + k^2 + k + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_in_base_k_l1917_191715


namespace NUMINAMATH_CALUDE_proposition_analysis_l1917_191709

theorem proposition_analysis :
  (¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) ∧
  (∀ x y a : ℝ, 1 < y ∧ y < x ∧ a < 0 → x^a < y^a) ∧
  ((¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) ∨
   (∀ x y a : ℝ, 1 < y ∧ y < x ∧ a < 0 → x^a < y^a)) ∧
  (¬(∀ x y a : ℝ, 1 < y ∧ y < x ∧ 0 < a ∧ a < 1 → a^(1/x) < a^(1/y))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_analysis_l1917_191709


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1917_191746

noncomputable def f (x : ℝ) := 2 * x^2 - Real.log x

theorem f_decreasing_interval :
  ∀ x : ℝ, x > 0 → x < 1/2 → (deriv f) x < 0 :=
by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1917_191746


namespace NUMINAMATH_CALUDE_total_campers_rowing_hiking_l1917_191761

/-- The total number of campers who went rowing and hiking -/
theorem total_campers_rowing_hiking 
  (morning_rowing : ℕ) 
  (morning_hiking : ℕ) 
  (afternoon_rowing : ℕ) 
  (h1 : morning_rowing = 41)
  (h2 : morning_hiking = 4)
  (h3 : afternoon_rowing = 26) :
  morning_rowing + morning_hiking + afternoon_rowing = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_hiking_l1917_191761


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1917_191769

theorem perfect_square_quadratic (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 2*(k+1)*x + 4 = (x - a)^2) → (k = -3 ∨ k = 1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1917_191769


namespace NUMINAMATH_CALUDE_trapezoid_side_length_l1917_191794

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- Length of side AB
  ab : ℝ
  -- Length of side CD
  cd : ℝ
  -- Ratio of areas of triangles ABC and ADC
  area_ratio : ℝ
  -- The sum of AB and CD is 300
  sum_sides : ab + cd = 300
  -- The ratio of areas is 5:4
  ratio_condition : area_ratio = 5 / 4

/-- 
Theorem: In a trapezoid ABCD, if the ratio of the area of triangle ABC 
to the area of triangle ADC is 5:4, and AB + CD = 300 cm, 
then AB = 500/3 cm.
-/
theorem trapezoid_side_length (t : Trapezoid) : t.ab = 500 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_side_length_l1917_191794


namespace NUMINAMATH_CALUDE_three_consecutive_days_without_class_l1917_191736

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a day in the month -/
structure Day where
  dayNumber : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a month with its properties -/
structure Month where
  days : List Day
  startDay : DayOfWeek
  totalDays : Nat
  classSchedule : List Nat

/-- Main theorem to prove -/
theorem three_consecutive_days_without_class 
  (november2017 : Month)
  (h1 : november2017.startDay = DayOfWeek.Wednesday)
  (h2 : november2017.totalDays = 30)
  (h3 : november2017.classSchedule.length = 11)
  (h4 : ∀ d ∈ november2017.days, 
    d.dayOfWeek = DayOfWeek.Saturday ∨ d.dayOfWeek = DayOfWeek.Sunday → 
    d.dayNumber ∉ november2017.classSchedule) :
  ∃ d1 d2 d3 : Day, 
    d1 ∈ november2017.days ∧ 
    d2 ∈ november2017.days ∧ 
    d3 ∈ november2017.days ∧ 
    d1.dayNumber + 1 = d2.dayNumber ∧ 
    d2.dayNumber + 1 = d3.dayNumber ∧ 
    d1.dayNumber ∉ november2017.classSchedule ∧ 
    d2.dayNumber ∉ november2017.classSchedule ∧ 
    d3.dayNumber ∉ november2017.classSchedule :=
by sorry

end NUMINAMATH_CALUDE_three_consecutive_days_without_class_l1917_191736


namespace NUMINAMATH_CALUDE_odd_symmetric_latin_square_diagonal_l1917_191716

/-- A square matrix of size n × n filled with integers from 1 to n -/
def LatinSquare (n : ℕ) := Matrix (Fin n) (Fin n) (Fin n)

/-- Predicate to check if a LatinSquare has all numbers from 1 to n in each row and column -/
def is_valid_latin_square (A : LatinSquare n) : Prop :=
  ∀ i j : Fin n, (∃ k : Fin n, A i k = j) ∧ (∃ k : Fin n, A k j = i)

/-- Predicate to check if a LatinSquare is symmetric -/
def is_symmetric (A : LatinSquare n) : Prop :=
  ∀ i j : Fin n, A i j = A j i

/-- Predicate to check if all numbers from 1 to n appear on the main diagonal -/
def all_on_diagonal (A : LatinSquare n) : Prop :=
  ∀ k : Fin n, ∃ i : Fin n, A i i = k

/-- Theorem stating that for odd n, a valid symmetric Latin square has all numbers on its diagonal -/
theorem odd_symmetric_latin_square_diagonal (n : ℕ) (hn : Odd n) (A : LatinSquare n)
  (hvalid : is_valid_latin_square A) (hsym : is_symmetric A) :
  all_on_diagonal A :=
sorry

end NUMINAMATH_CALUDE_odd_symmetric_latin_square_diagonal_l1917_191716


namespace NUMINAMATH_CALUDE_log_inequality_l1917_191764

theorem log_inequality : (1 : ℝ) / 3 < Real.log 3 - Real.log 2 ∧ Real.log 3 - Real.log 2 < (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1917_191764


namespace NUMINAMATH_CALUDE_seven_bus_routes_l1917_191724

/-- Represents a bus stop in the network -/
structure BusStop :=
  (id : ℕ)

/-- Represents a bus route in the network -/
structure BusRoute :=
  (id : ℕ)
  (stops : Finset BusStop)

/-- Represents the entire bus network -/
structure BusNetwork :=
  (stops : Finset BusStop)
  (routes : Finset BusRoute)

/-- Every stop is reachable from any other stop without transfer -/
def all_stops_reachable (network : BusNetwork) : Prop :=
  ∀ s₁ s₂ : BusStop, s₁ ∈ network.stops → s₂ ∈ network.stops → 
    ∃ r : BusRoute, r ∈ network.routes ∧ s₁ ∈ r.stops ∧ s₂ ∈ r.stops

/-- Each pair of routes intersects at exactly one unique stop -/
def unique_intersection (network : BusNetwork) : Prop :=
  ∀ r₁ r₂ : BusRoute, r₁ ∈ network.routes → r₂ ∈ network.routes → r₁ ≠ r₂ →
    ∃! s : BusStop, s ∈ r₁.stops ∧ s ∈ r₂.stops

/-- Each route has exactly three stops -/
def three_stops_per_route (network : BusNetwork) : Prop :=
  ∀ r : BusRoute, r ∈ network.routes → Finset.card r.stops = 3

/-- There is more than one route -/
def multiple_routes (network : BusNetwork) : Prop :=
  ∃ r₁ r₂ : BusRoute, r₁ ∈ network.routes ∧ r₂ ∈ network.routes ∧ r₁ ≠ r₂

/-- The main theorem: Given the conditions, prove that there are 7 bus routes -/
theorem seven_bus_routes (network : BusNetwork) 
  (h1 : all_stops_reachable network)
  (h2 : unique_intersection network)
  (h3 : three_stops_per_route network)
  (h4 : multiple_routes network) :
  Finset.card network.routes = 7 :=
sorry

end NUMINAMATH_CALUDE_seven_bus_routes_l1917_191724


namespace NUMINAMATH_CALUDE_m_range_l1917_191786

theorem m_range (m : ℝ) : (∀ x > 0, x + 1/x - m > 0) → m < 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l1917_191786


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1917_191762

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

theorem hyperbola_equation :
  ∀ (x y : ℝ), 
    (∃ (t : ℝ), hyperbola t y ∧ asymptotes t y) →  -- Hyperbola exists with given asymptotes
    hyperbola 4 (Real.sqrt 3) →                    -- Hyperbola passes through (4, √3)
    hyperbola x y                                  -- The equation of the hyperbola
  := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1917_191762


namespace NUMINAMATH_CALUDE_trig_expression_equality_l1917_191791

theorem trig_expression_equality : 
  (Real.sin (40 * π / 180) - Real.sqrt 3 * Real.cos (20 * π / 180)) / Real.cos (10 * π / 180) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l1917_191791


namespace NUMINAMATH_CALUDE_right_triangle_max_area_right_triangle_max_area_achieved_right_triangle_max_area_is_nine_l1917_191700

theorem right_triangle_max_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_hypotenuse : c = 6) :
  a * b ≤ 18 := by
  sorry

theorem right_triangle_max_area_achieved (a b : ℝ) (h_right : a^2 + b^2 = 36) (h_equal : a = b) :
  a * b = 18 := by
  sorry

theorem right_triangle_max_area_is_nine :
  ∃ (a b : ℝ), a^2 + b^2 = 36 ∧ a * b / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_area_right_triangle_max_area_achieved_right_triangle_max_area_is_nine_l1917_191700


namespace NUMINAMATH_CALUDE_dot_product_equals_negative_102_l1917_191789

def vector1 : Fin 4 → ℤ := ![4, -5, 6, -3]
def vector2 : Fin 4 → ℤ := ![-2, 8, -7, 4]

theorem dot_product_equals_negative_102 :
  (Finset.univ.sum fun i => (vector1 i) * (vector2 i)) = -102 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_equals_negative_102_l1917_191789


namespace NUMINAMATH_CALUDE_inequality_proof_l1917_191760

theorem inequality_proof (a b c d : ℝ) :
  (a^2 - a + 1) * (b^2 - b + 1) * (c^2 - c + 1) * (d^2 - d + 1) ≥ 
  9/16 * (a - b) * (b - c) * (c - d) * (d - a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1917_191760


namespace NUMINAMATH_CALUDE_square_and_rectangles_problem_l1917_191776

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the area of a square -/
def Square.area (s : Square) : ℝ := s.side * s.side

/-- Theorem statement for the given problem -/
theorem square_and_rectangles_problem
  (small_square : Square)
  (large_rectangle : Rectangle)
  (R : Rectangle)
  (large_square : Square)
  (h1 : small_square.side = 2)
  (h2 : large_rectangle.width = 2 ∧ large_rectangle.height = 4)
  (h3 : small_square.area + large_rectangle.area + R.area = large_square.area)
  : large_square.side = 4 ∧ R.area = 4 := by
  sorry


end NUMINAMATH_CALUDE_square_and_rectangles_problem_l1917_191776


namespace NUMINAMATH_CALUDE_insurance_cost_over_decade_l1917_191771

/-- The amount spent on car insurance in a year -/
def yearly_insurance_cost : ℕ := 4000

/-- The number of years in a decade -/
def years_in_decade : ℕ := 10

/-- The total cost of car insurance over a decade -/
def decade_insurance_cost : ℕ := yearly_insurance_cost * years_in_decade

theorem insurance_cost_over_decade : 
  decade_insurance_cost = 40000 := by sorry

end NUMINAMATH_CALUDE_insurance_cost_over_decade_l1917_191771


namespace NUMINAMATH_CALUDE_least_possible_third_side_length_l1917_191759

theorem least_possible_third_side_length (a b c : ℝ) : 
  a = 8 → b = 15 → c > 0 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  c ≥ Real.sqrt 161 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_third_side_length_l1917_191759


namespace NUMINAMATH_CALUDE_investment_profit_comparison_l1917_191788

/-- Profit calculation for selling at the beginning of the month -/
def profit_beginning (x : ℝ) : ℝ := 0.265 * x

/-- Profit calculation for selling at the end of the month -/
def profit_end (x : ℝ) : ℝ := 0.3 * x - 700

theorem investment_profit_comparison :
  /- The investment amount where profits are equal is 20,000 yuan -/
  (∃ x : ℝ, x = 20000 ∧ profit_beginning x = profit_end x) ∧
  /- For a 50,000 yuan investment, profit from selling at the end is greater -/
  (profit_end 50000 > profit_beginning 50000) :=
by sorry

end NUMINAMATH_CALUDE_investment_profit_comparison_l1917_191788


namespace NUMINAMATH_CALUDE_tens_digit_of_3_power_205_l1917_191712

theorem tens_digit_of_3_power_205 : ∃ n : ℕ, 3^205 ≡ 40 + n [ZMOD 100] ∧ n < 10 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_3_power_205_l1917_191712


namespace NUMINAMATH_CALUDE_sin_cos_identity_l1917_191708

theorem sin_cos_identity : 
  Real.sin (10 * π / 180) * Real.cos (70 * π / 180) - 
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) = 
  -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l1917_191708


namespace NUMINAMATH_CALUDE_exercise_books_count_l1917_191722

/-- Given a shop with pencils, pens, exercise books, and erasers in the ratio 10 : 2 : 3 : 4,
    where there are 150 pencils, prove that there are 45 exercise books. -/
theorem exercise_books_count (pencils : ℕ) (pens : ℕ) (exercise_books : ℕ) (erasers : ℕ) :
  pencils = 150 →
  10 * pens = 2 * pencils →
  10 * exercise_books = 3 * pencils →
  10 * erasers = 4 * pencils →
  exercise_books = 45 := by
  sorry

end NUMINAMATH_CALUDE_exercise_books_count_l1917_191722


namespace NUMINAMATH_CALUDE_triangle_sides_simplification_l1917_191783

theorem triangle_sides_simplification (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  |a + b - c| - |a - c - b| = 2*a - 2*c := by
  sorry

end NUMINAMATH_CALUDE_triangle_sides_simplification_l1917_191783


namespace NUMINAMATH_CALUDE_min_value_a_min_value_a_achieved_l1917_191719

theorem min_value_a (a b : ℕ) (h1 : a > 0) (h2 : a = b - 2005) 
  (h3 : ∃ x : ℕ, x > 0 ∧ x^2 - a*x + b = 0) : a ≥ 95 := by
  sorry

theorem min_value_a_achieved (a b : ℕ) (h1 : a > 0) (h2 : a = b - 2005) : 
  (∃ x : ℕ, x > 0 ∧ x^2 - 95*x + (95 + 2005) = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_min_value_a_achieved_l1917_191719


namespace NUMINAMATH_CALUDE_range_of_a_l1917_191763

-- Define the propositions P and Q
def P (x a : ℝ) : Prop := |x - a| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the negations of P and Q
def not_P (x a : ℝ) : Prop := ¬(P x a)
def not_Q (x : ℝ) : Prop := ¬(Q x)

-- Define the condition that not_P is sufficient but not necessary for not_Q
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, not_P x a → not_Q x) ∧ (∃ x, not_Q x ∧ ¬(not_P x a))

-- State the theorem
theorem range_of_a :
  ∀ a : ℝ, sufficient_not_necessary a ↔ -1 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1917_191763


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l1917_191751

theorem quadratic_roots_sum_and_product (m n : ℝ) : 
  (m^2 + 4*m - 1 = 0) → 
  (n^2 + 4*n - 1 = 0) → 
  m + n + m*n = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l1917_191751


namespace NUMINAMATH_CALUDE_meat_pie_cost_l1917_191790

/-- The cost of a meat pie given Gerald's initial farthings and remaining pfennigs -/
theorem meat_pie_cost
  (initial_farthings : ℕ)
  (farthings_per_pfennig : ℕ)
  (remaining_pfennigs : ℕ)
  (h1 : initial_farthings = 54)
  (h2 : farthings_per_pfennig = 6)
  (h3 : remaining_pfennigs = 7)
  : (initial_farthings / farthings_per_pfennig) - remaining_pfennigs = 2 := by
  sorry

#check meat_pie_cost

end NUMINAMATH_CALUDE_meat_pie_cost_l1917_191790


namespace NUMINAMATH_CALUDE_cottage_village_price_l1917_191737

/-- The selling price of each house in a cottage village -/
def house_selling_price : ℕ := by sorry

/-- The number of houses in the village -/
def num_houses : ℕ := 15

/-- The total cost of construction for the entire village -/
def total_cost : ℕ := 150 + 105 + 225 + 45

/-- The markup percentage of the construction company -/
def markup_percentage : ℚ := 20 / 100

theorem cottage_village_price :
  (house_selling_price : ℚ) = (total_cost : ℚ) / num_houses * (1 + markup_percentage) ∧
  house_selling_price = 42 := by sorry

end NUMINAMATH_CALUDE_cottage_village_price_l1917_191737


namespace NUMINAMATH_CALUDE_gardeners_mowing_time_l1917_191739

theorem gardeners_mowing_time (rate_A rate_B : ℚ) (h1 : rate_A = 1 / 3) (h2 : rate_B = 1 / 5) :
  1 / (rate_A + rate_B) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_gardeners_mowing_time_l1917_191739


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_27000001_l1917_191792

theorem sum_of_prime_factors_27000001 :
  ∃ (p₁ p₂ p₃ p₄ : Nat),
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
    p₁ * p₂ * p₃ * p₄ = 27000001 ∧
    p₁ + p₂ + p₃ + p₄ = 652 :=
by
  sorry

#check sum_of_prime_factors_27000001

end NUMINAMATH_CALUDE_sum_of_prime_factors_27000001_l1917_191792


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1917_191728

theorem abs_sum_inequality (x : ℝ) :
  (|x + 2| + |x - 2| < x + 7) ↔ (-7/3 < x ∧ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1917_191728


namespace NUMINAMATH_CALUDE_all_equal_cyclic_inequality_l1917_191779

theorem all_equal_cyclic_inequality (a : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
sorry

end NUMINAMATH_CALUDE_all_equal_cyclic_inequality_l1917_191779


namespace NUMINAMATH_CALUDE_four_numbers_product_equality_l1917_191733

theorem four_numbers_product_equality (p : ℝ) (hp : p ≥ 1) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (a : ℝ) > p ∧ (b : ℝ) > p ∧ (c : ℝ) > p ∧ (d : ℝ) > p ∧
    (a : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (b : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (c : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (d : ℝ) < (2 + Real.sqrt (p + 1/4))^2 ∧
    (a * b : ℕ) = c * d :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_product_equality_l1917_191733


namespace NUMINAMATH_CALUDE_intersection_M_N_l1917_191740

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

-- Define the open interval (0, 1]
def open_unit_interval : Set ℝ := {x | 0 < x ∧ x ≤ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = open_unit_interval := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1917_191740


namespace NUMINAMATH_CALUDE_slope_does_not_exist_for_vertical_line_l1917_191799

/-- A line is vertical if its equation can be written in the form x = constant -/
def IsVerticalLine (a b : ℝ) : Prop := a ≠ 0 ∧ ∀ x y : ℝ, a * x + b = 0 → x = -b / a

/-- The slope of a line does not exist if the line is vertical -/
def SlopeDoesNotExist (a b : ℝ) : Prop := IsVerticalLine a b

theorem slope_does_not_exist_for_vertical_line (a b : ℝ) :
  a * x + b = 0 → a ≠ 0 → SlopeDoesNotExist a b := by sorry

end NUMINAMATH_CALUDE_slope_does_not_exist_for_vertical_line_l1917_191799


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1917_191778

theorem simplify_and_evaluate : 
  ∀ x : ℝ, (3*x^2 + 8*x - 6) - (2*x^2 + 4*x - 15) = x^2 + 4*x + 9 ∧ 
  (let x : ℝ := 3; (3*x^2 + 8*x - 6) - (2*x^2 + 4*x - 15) = 30) :=
by
  sorry

#check simplify_and_evaluate

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1917_191778


namespace NUMINAMATH_CALUDE_rod_length_proof_l1917_191774

/-- Given that a 6-meter rod weighs 14.04 kg, prove that a rod weighing 23.4 kg is 10 meters long. -/
theorem rod_length_proof (weight_per_meter : ℝ) (h1 : weight_per_meter = 14.04 / 6) :
  23.4 / weight_per_meter = 10 := by
sorry

end NUMINAMATH_CALUDE_rod_length_proof_l1917_191774


namespace NUMINAMATH_CALUDE_ellipse_m_value_l1917_191758

/-- Given an ellipse with equation x²/25 + y²/m² = 1 (m > 0) and left focus point at (-4, 0), 
    prove that m = 3 -/
theorem ellipse_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2/25 + y^2/m^2 = 1) → 
  (∃ x y : ℝ, x = -4 ∧ y = 0 ∧ (x + 5)^2/25 + y^2/m^2 < 1) → 
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l1917_191758


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l1917_191772

theorem fuel_tank_capacity : ∀ (x : ℚ), 
  (5 / 6 : ℚ) * x - (2 / 3 : ℚ) * x = 15 → x = 90 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l1917_191772


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1917_191754

theorem profit_percent_calculation (selling_price : ℝ) (cost_price : ℝ) (h : cost_price = 0.8 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l1917_191754


namespace NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l1917_191721

theorem fraction_existence_and_nonexistence :
  (∀ n : ℕ+, ∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ a / b ∧ a / b ≤ Real.sqrt (n + 1)) ∧
  (∃ f : ℕ+ → ℕ+, Function.Injective f ∧ ∀ n : ℕ+, ¬∃ a b : ℤ, 0 < b ∧ b ≤ Real.sqrt (f n) ∧ Real.sqrt (f n) ≤ a / b ∧ a / b ≤ Real.sqrt (f n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_fraction_existence_and_nonexistence_l1917_191721


namespace NUMINAMATH_CALUDE_smallest_n_for_equation_l1917_191793

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem smallest_n_for_equation : 
  ∃ (n : ℕ), n > 0 ∧ 2 * n * factorial n + 3 * factorial n = 5040 ∧ 
  ∀ (m : ℕ), m > 0 ∧ m < n → 2 * m * factorial m + 3 * factorial m ≠ 5040 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_equation_l1917_191793


namespace NUMINAMATH_CALUDE_dilation_circle_to_ellipse_l1917_191752

/-- Given a circle A and a dilation transformation, prove the equation of the resulting curve C -/
theorem dilation_circle_to_ellipse :
  let circle_A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let dilation (p : ℝ × ℝ) : ℝ × ℝ := (2 * p.1, 3 * p.2)
  let curve_C : Set (ℝ × ℝ) := {p | p.1^2 / 4 + p.2^2 / 9 = 1}
  (∀ p ∈ circle_A, dilation p ∈ curve_C) ∧
  (∀ q ∈ curve_C, ∃ p ∈ circle_A, dilation p = q) := by
sorry

end NUMINAMATH_CALUDE_dilation_circle_to_ellipse_l1917_191752


namespace NUMINAMATH_CALUDE_nine_caps_per_box_l1917_191796

/-- Given a total number of bottle caps and a number of boxes, 
    calculate the number of bottle caps in each box. -/
def bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℕ) : ℕ :=
  total_caps / num_boxes

/-- Theorem stating that with 54 total bottle caps and 6 boxes, 
    there are 9 bottle caps in each box. -/
theorem nine_caps_per_box :
  bottle_caps_per_box 54 6 = 9 := by
  sorry

#eval bottle_caps_per_box 54 6

end NUMINAMATH_CALUDE_nine_caps_per_box_l1917_191796


namespace NUMINAMATH_CALUDE_stripe_length_on_cylinder_l1917_191795

/-- Proves that the length of a diagonal line on a rectangle with sides 30 inches and 16 inches is 34 inches. -/
theorem stripe_length_on_cylinder (circumference height : ℝ) (h1 : circumference = 30) (h2 : height = 16) :
  Real.sqrt (circumference^2 + height^2) = 34 :=
by sorry

end NUMINAMATH_CALUDE_stripe_length_on_cylinder_l1917_191795


namespace NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l1917_191784

/-- A regular hendecagon is an 11-sided regular polygon -/
def RegularHendecagon : Type := Unit

/-- The number of vertices in a regular hendecagon -/
def num_vertices : ℕ := 11

/-- The number of diagonals in a regular hendecagon -/
def num_diagonals (h : RegularHendecagon) : ℕ := 44

/-- The number of pairs of diagonals in a regular hendecagon -/
def num_diagonal_pairs (h : RegularHendecagon) : ℕ := Nat.choose (num_diagonals h) 2

/-- The number of intersecting diagonal pairs inside a regular hendecagon -/
def num_intersecting_pairs (h : RegularHendecagon) : ℕ := Nat.choose num_vertices 4

/-- The probability that two randomly chosen diagonals intersect inside the hendecagon -/
def intersection_probability (h : RegularHendecagon) : ℚ :=
  (num_intersecting_pairs h : ℚ) / (num_diagonal_pairs h : ℚ)

theorem hendecagon_diagonal_intersection_probability (h : RegularHendecagon) :
  intersection_probability h = 165 / 473 := by
  sorry

end NUMINAMATH_CALUDE_hendecagon_diagonal_intersection_probability_l1917_191784


namespace NUMINAMATH_CALUDE_min_turns_rook_path_l1917_191714

/-- Represents a chessboard --/
structure Chessboard :=
  (files : Nat)
  (ranks : Nat)

/-- Represents a rook's path on a chessboard --/
structure RookPath :=
  (board : Chessboard)
  (turns : Nat)
  (visitsAllSquares : Bool)

/-- Defines a valid rook path that visits all squares exactly once --/
def isValidRookPath (path : RookPath) : Prop :=
  path.board.files = 8 ∧
  path.board.ranks = 8 ∧
  path.visitsAllSquares = true

/-- Theorem: The minimum number of turns for a rook to visit all squares on an 8x8 chessboard is 14 --/
theorem min_turns_rook_path :
  ∀ (path : RookPath), isValidRookPath path → path.turns ≥ 14 :=
by sorry

end NUMINAMATH_CALUDE_min_turns_rook_path_l1917_191714


namespace NUMINAMATH_CALUDE_rational_x_y_l1917_191717

theorem rational_x_y (x y : ℝ) 
  (h : ∀ (p q : ℕ), Prime p → Prime q → Odd p → Odd q → p ≠ q → 
    ∃ (r : ℚ), (x^p + y^q : ℝ) = (r : ℝ)) : 
  ∃ (a b : ℚ), (x = (a : ℝ) ∧ y = (b : ℝ)) := by
sorry

end NUMINAMATH_CALUDE_rational_x_y_l1917_191717


namespace NUMINAMATH_CALUDE_least_x_for_1894x_divisible_by_3_l1917_191780

theorem least_x_for_1894x_divisible_by_3 : 
  ∃ x : ℕ, (∀ y : ℕ, y < x → ¬(3 ∣ 1894 * y)) ∧ (3 ∣ 1894 * x) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_x_for_1894x_divisible_by_3_l1917_191780


namespace NUMINAMATH_CALUDE_complex_division_proof_l1917_191702

theorem complex_division_proof : (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_proof_l1917_191702


namespace NUMINAMATH_CALUDE_goldfish_equality_exists_l1917_191744

theorem goldfish_equality_exists : ∃ n : ℕ+, 
  8 * (5 : ℝ)^n.val = 200 * (3 : ℝ)^n.val + 20 * ((3 : ℝ)^n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_equality_exists_l1917_191744


namespace NUMINAMATH_CALUDE_mara_pink_crayons_percentage_l1917_191770

/-- The percentage of Mara's crayons that are pink -/
def mara_pink_percentage : ℝ := 10

theorem mara_pink_crayons_percentage 
  (mara_total : ℕ) 
  (luna_total : ℕ) 
  (luna_pink_percentage : ℝ) 
  (total_pink : ℕ) 
  (h1 : mara_total = 40)
  (h2 : luna_total = 50)
  (h3 : luna_pink_percentage = 20)
  (h4 : total_pink = 14)
  : mara_pink_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_mara_pink_crayons_percentage_l1917_191770


namespace NUMINAMATH_CALUDE_squarefree_juicy_integers_l1917_191741

def is_juicy (n : ℕ) : Prop :=
  n > 1 ∧ ∀ (d₁ d₂ : ℕ), d₁ ∣ n → d₂ ∣ n → d₁ < d₂ → (d₂ - d₁) ∣ n

def is_squarefree (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p * p ∣ n → p = 1)

theorem squarefree_juicy_integers :
  {n : ℕ | is_squarefree n ∧ is_juicy n} = {2, 6, 42, 1806} :=
sorry

end NUMINAMATH_CALUDE_squarefree_juicy_integers_l1917_191741


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l1917_191750

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l1917_191750


namespace NUMINAMATH_CALUDE_figure3_turns_l1917_191743

/-- Represents a dot in the grid --/
inductive Dot
| Black : Dot
| White : Dot

/-- Represents a turn in the loop --/
inductive Turn
| Right : Turn

/-- Represents a grid with dots --/
structure Grid :=
(dots : List Dot)

/-- Represents a loop in the grid --/
structure Loop :=
(turns : List Turn)

/-- Function to check if a loop is valid for a given grid --/
def is_valid_loop (g : Grid) (l : Loop) : Prop := sorry

/-- Function to count the number of turns in a loop --/
def count_turns (l : Loop) : Nat := l.turns.length

/-- The specific grid configuration for Figure 3 --/
def figure3 : Grid := sorry

/-- Theorem stating that the valid loop for Figure 3 has 20 turns --/
theorem figure3_turns :
  ∃ (l : Loop), is_valid_loop figure3 l ∧ count_turns l = 20 := by sorry

end NUMINAMATH_CALUDE_figure3_turns_l1917_191743


namespace NUMINAMATH_CALUDE_intersection_point_correct_l1917_191757

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -1

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 4

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := 13 / 10

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := 29 / 10

/-- Theorem stating that the intersection point is correct -/
theorem intersection_point_correct : 
  (m₁ * x_intersect + b₁ = y_intersect) ∧ 
  (m₂ * (x_intersect - x₀) = y_intersect - y₀) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l1917_191757


namespace NUMINAMATH_CALUDE_tree_survival_probability_l1917_191735

/-- Probability that at least one tree survives after transplantation -/
theorem tree_survival_probability :
  let survival_rate_A : ℚ := 5/6
  let survival_rate_B : ℚ := 4/5
  let num_trees_A : ℕ := 2
  let num_trees_B : ℕ := 2
  -- Probability that at least one tree survives
  1 - (1 - survival_rate_A) ^ num_trees_A * (1 - survival_rate_B) ^ num_trees_B = 899/900 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_survival_probability_l1917_191735


namespace NUMINAMATH_CALUDE_equation_solutions_l1917_191797

theorem equation_solutions (k : ℤ) (x₁ x₂ x₃ x₄ y₁ : ℤ) :
  (y₁^2 - k = x₁^3) ∧
  ((y₁ - 1)^2 - k = x₂^3) ∧
  ((y₁ - 2)^2 - k = x₃^3) ∧
  ((y₁ - 3)^2 - k = x₄^3) →
  k ≡ 17 [ZMOD 63] :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1917_191797


namespace NUMINAMATH_CALUDE_point_existence_and_uniqueness_l1917_191756

theorem point_existence_and_uniqueness :
  ∃! (x y : ℝ), 
    y = 8 ∧ 
    (x - 3)^2 + (y - 9)^2 = 12^2 ∧ 
    x^2 + y^2 = 14^2 ∧ 
    x > 3 := by
  sorry

end NUMINAMATH_CALUDE_point_existence_and_uniqueness_l1917_191756


namespace NUMINAMATH_CALUDE_f_difference_f_equals_x_plus_3_l1917_191782

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Theorem 1: For any real number a, f(a) - f(a + 1) = -2a - 1
theorem f_difference (a : ℝ) : f a - f (a + 1) = -2 * a - 1 := by
  sorry

-- Theorem 2: If f(x) = x + 3, then x = -1 or x = 2
theorem f_equals_x_plus_3 (x : ℝ) : f x = x + 3 → x = -1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_f_equals_x_plus_3_l1917_191782


namespace NUMINAMATH_CALUDE_leak_empty_time_l1917_191738

/-- Given a pipe that can fill a tank in 6 hours, and with a leak it takes 8 hours to fill the tank,
    prove that the leak alone will empty the full tank in 24 hours. -/
theorem leak_empty_time (fill_rate : ℝ) (combined_rate : ℝ) (leak_rate : ℝ) : 
  fill_rate = 1 / 6 →
  combined_rate = 1 / 8 →
  combined_rate = fill_rate - leak_rate →
  1 / leak_rate = 24 := by
sorry

end NUMINAMATH_CALUDE_leak_empty_time_l1917_191738


namespace NUMINAMATH_CALUDE_base_eight_satisfies_equation_unique_base_satisfies_equation_l1917_191748

/-- Given a base b, converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Checks if the equation 245_b + 132_b = 400_b holds for a given base b --/
def equationHolds (b : Nat) : Prop :=
  toDecimal [2, 4, 5] b + toDecimal [1, 3, 2] b = toDecimal [4, 0, 0] b

theorem base_eight_satisfies_equation :
  equationHolds 8 := by sorry

theorem unique_base_satisfies_equation :
  ∀ b : Nat, b > 1 → equationHolds b → b = 8 := by sorry

end NUMINAMATH_CALUDE_base_eight_satisfies_equation_unique_base_satisfies_equation_l1917_191748
