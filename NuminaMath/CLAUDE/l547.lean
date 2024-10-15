import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_range_l547_54755

/-- Given a quadratic function y = x^2 - 2bx + b^2 + c whose graph intersects
    the line y = 1 - x at only one point, and its vertex is on the graph of
    y = ax^2 (a ≠ 0), prove that the range of values for a is a ≥ -1/5 and a ≠ 0. -/
theorem quadratic_function_range (b c : ℝ) (a : ℝ) 
  (h1 : ∃! x, x^2 - 2*b*x + b^2 + c = 1 - x) 
  (h2 : c = a * b^2) 
  (h3 : a ≠ 0) : 
  a ≥ -1/5 ∧ a ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l547_54755


namespace NUMINAMATH_CALUDE_initial_nickels_l547_54771

/-- Given the current number of nickels and the number of borrowed nickels,
    prove that the initial number of nickels is their sum. -/
theorem initial_nickels (current_nickels borrowed_nickels : ℕ) :
  let initial_nickels := current_nickels + borrowed_nickels
  initial_nickels = current_nickels + borrowed_nickels :=
by sorry

end NUMINAMATH_CALUDE_initial_nickels_l547_54771


namespace NUMINAMATH_CALUDE_cos_20_minus_cos_40_l547_54770

theorem cos_20_minus_cos_40 : Real.cos (20 * π / 180) - Real.cos (40 * π / 180) = -1 / (2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_cos_20_minus_cos_40_l547_54770


namespace NUMINAMATH_CALUDE_superhero_advantage_l547_54729

/-- Superhero's speed in miles per minute -/
def superhero_speed : ℚ := 10 / 4

/-- Supervillain's speed in miles per hour -/
def supervillain_speed : ℚ := 100

/-- Minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem superhero_advantage : 
  (superhero_speed * minutes_per_hour) - supervillain_speed = 50 := by sorry

end NUMINAMATH_CALUDE_superhero_advantage_l547_54729


namespace NUMINAMATH_CALUDE_not_clear_def_not_set_l547_54733

-- Define what it means for a collection to have a clear definition
def has_clear_definition (C : Type → Prop) : Prop :=
  ∀ (x : Type), (C x) ∨ (¬C x)

-- Define what it means to be a set
def is_set (S : Type → Prop) : Prop :=
  has_clear_definition S

-- Theorem: A collection without a clear definition is not a set
theorem not_clear_def_not_set (C : Type → Prop) :
  ¬(has_clear_definition C) → ¬(is_set C) := by
  sorry

end NUMINAMATH_CALUDE_not_clear_def_not_set_l547_54733


namespace NUMINAMATH_CALUDE_speeding_proof_l547_54782

theorem speeding_proof (distance : ℝ) (time : ℝ) (speed_limit : ℝ)
  (h1 : distance = 165)
  (h2 : time = 2)
  (h3 : speed_limit = 80)
  : ∃ t : ℝ, 0 ≤ t ∧ t ≤ time ∧ (distance / time > speed_limit) :=
by
  sorry

#check speeding_proof

end NUMINAMATH_CALUDE_speeding_proof_l547_54782


namespace NUMINAMATH_CALUDE_house_transaction_result_l547_54709

theorem house_transaction_result (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : 
  initial_value = 12000 ∧ 
  loss_percent = 0.15 ∧ 
  gain_percent = 0.20 → 
  initial_value * (1 - loss_percent) * (1 + gain_percent) - initial_value = -240 := by
sorry

end NUMINAMATH_CALUDE_house_transaction_result_l547_54709


namespace NUMINAMATH_CALUDE_at_least_one_good_product_l547_54732

theorem at_least_one_good_product (total : Nat) (defective : Nat) (selected : Nat) 
  (h1 : total = 12)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective < total)
  (h5 : selected ≤ total) :
  ∀ (selection : Finset (Fin total)), selection.card = selected → 
    ∃ (x : Fin total), x ∈ selection ∧ x.val ∉ Finset.range defective :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_good_product_l547_54732


namespace NUMINAMATH_CALUDE_divisor_count_fourth_power_l547_54738

theorem divisor_count_fourth_power (x d : ℕ) : 
  (∃ n : ℕ, x = n^4) → 
  (d = (Finset.filter (· ∣ x) (Finset.range (x + 1))).card) →
  d ≡ 1 [MOD 4] := by
sorry

end NUMINAMATH_CALUDE_divisor_count_fourth_power_l547_54738


namespace NUMINAMATH_CALUDE_sine_derivative_2007_l547_54769

open Real

theorem sine_derivative_2007 (f : ℝ → ℝ) (x : ℝ) :
  f = sin →
  (∀ n : ℕ, deriv^[n] f = fun x ↦ f (x + n * (π / 2))) →
  deriv^[2007] f = fun x ↦ -cos x := by
  sorry

end NUMINAMATH_CALUDE_sine_derivative_2007_l547_54769


namespace NUMINAMATH_CALUDE_platform_length_l547_54760

/-- Calculates the length of a platform given the speed of a train, time to cross the platform,
    and the length of the train. -/
theorem platform_length (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) :
  train_speed = 72 * (5 / 18) →  -- Convert km/hr to m/s
  crossing_time = 26 →
  train_length = 440 →
  train_speed * crossing_time - train_length = 80 := by
  sorry

#check platform_length

end NUMINAMATH_CALUDE_platform_length_l547_54760


namespace NUMINAMATH_CALUDE_eliminate_denominators_l547_54747

theorem eliminate_denominators (x : ℝ) :
  (x - 1) / 3 = 4 - (2 * x + 1) / 2 ↔ 2 * (x - 1) = 24 - 3 * (2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l547_54747


namespace NUMINAMATH_CALUDE_movie_theater_child_price_l547_54718

/-- Proves that the price for children is $4.5 given the conditions of the movie theater problem -/
theorem movie_theater_child_price 
  (adult_price : ℝ) 
  (num_children : ℕ) 
  (child_adult_diff : ℕ) 
  (total_receipts : ℝ) 
  (h1 : adult_price = 6.75)
  (h2 : num_children = 48)
  (h3 : child_adult_diff = 20)
  (h4 : total_receipts = 405) :
  ∃ (child_price : ℝ), 
    child_price = 4.5 ∧ 
    (num_children : ℝ) * child_price + ((num_children : ℝ) - (child_adult_diff : ℝ)) * adult_price = total_receipts :=
by
  sorry

end NUMINAMATH_CALUDE_movie_theater_child_price_l547_54718


namespace NUMINAMATH_CALUDE_power_multiplication_l547_54774

theorem power_multiplication (a : ℝ) : a^3 * a^4 = a^7 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_l547_54774


namespace NUMINAMATH_CALUDE_wy_equals_uv_l547_54712

-- Define the variables
variable (u v w y : ℝ)
variable (α β : ℝ)

-- Define the conditions
axiom sin_roots : (Real.sin α)^2 - u * (Real.sin α) + v = 0 ∧ (Real.sin β)^2 - u * (Real.sin β) + v = 0
axiom cos_roots : (Real.cos α)^2 - w * (Real.cos α) + y = 0 ∧ (Real.cos β)^2 - w * (Real.cos β) + y = 0
axiom right_triangle : Real.sin α = Real.cos β ∧ Real.sin β = Real.cos α

-- State the theorem
theorem wy_equals_uv : wy = uv := by sorry

end NUMINAMATH_CALUDE_wy_equals_uv_l547_54712


namespace NUMINAMATH_CALUDE_optimal_selling_price_l547_54710

/-- Represents the profit optimization problem for a product -/
def ProfitOptimization (initialCost initialPrice initialSales : ℝ) 
                       (priceIncrease salesDecrease : ℝ) : Prop :=
  let profitFunction := fun x : ℝ => 
    (initialPrice + priceIncrease * x - initialCost) * (initialSales - salesDecrease * x)
  ∃ (optimalX : ℝ), 
    (∀ x : ℝ, profitFunction x ≤ profitFunction optimalX) ∧
    initialPrice + priceIncrease * optimalX = 14

/-- The main theorem stating the optimal selling price -/
theorem optimal_selling_price :
  ProfitOptimization 8 10 200 0.5 10 := by
  sorry

#check optimal_selling_price

end NUMINAMATH_CALUDE_optimal_selling_price_l547_54710


namespace NUMINAMATH_CALUDE_twenty_in_base_five_l547_54744

/-- Converts a decimal number to its base-5 representation -/
def to_base_five (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid base-5 number -/
def is_valid_base_five (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, d < 5

/-- Converts a list of base-5 digits to its decimal value -/
def from_base_five (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 5 + d) 0

theorem twenty_in_base_five :
  to_base_five 20 = [4, 0] ∧
  is_valid_base_five [4, 0] ∧
  from_base_five [4, 0] = 20 :=
sorry

end NUMINAMATH_CALUDE_twenty_in_base_five_l547_54744


namespace NUMINAMATH_CALUDE_complement_A_U_eq_l547_54762

-- Define the universal set U
def U : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}

-- Define set A
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 2}

-- Define the complement of A with respect to U
def complement_A_U : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_A_U_eq :
  complement_A_U = {x : ℝ | (-4 < x ∧ x < -3) ∨ (2 ≤ x ∧ x < 4)} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_U_eq_l547_54762


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l547_54761

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-2, x)
  parallel a b → x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l547_54761


namespace NUMINAMATH_CALUDE_difference_of_squares_l547_54754

theorem difference_of_squares (m n : ℝ) : (-m - n) * (-m + n) = (-m)^2 - n^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l547_54754


namespace NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l547_54798

theorem tetrahedron_circumscribed_sphere_area (edge_length : ℝ) : 
  edge_length = 4 → 
  ∃ (sphere_area : ℝ), sphere_area = 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumscribed_sphere_area_l547_54798


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l547_54741

/-- Two 2D vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then x = 3 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l547_54741


namespace NUMINAMATH_CALUDE_square_division_problem_l547_54779

theorem square_division_problem :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 + y^2 = 100 ∧ x/y = 4/3 ∧ x = 8 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_division_problem_l547_54779


namespace NUMINAMATH_CALUDE_garden_land_ratio_l547_54776

/-- Represents a rectangle with width 3/5 of its length -/
structure Rectangle where
  length : ℝ
  width : ℝ
  width_prop : width = 3/5 * length

theorem garden_land_ratio (land garden : Rectangle) 
  (h : garden.length = 3/5 * land.length) :
  (garden.length * garden.width) / (land.length * land.width) = 36/100 := by
  sorry

end NUMINAMATH_CALUDE_garden_land_ratio_l547_54776


namespace NUMINAMATH_CALUDE_soccer_committee_count_l547_54788

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The total number of possible organizing committees -/
def total_committees : ℕ := 34134175

theorem soccer_committee_count :
  (num_teams * (Nat.choose team_size host_selection) *
   (Nat.choose team_size non_host_selection ^ (num_teams - 1))) = total_committees := by
  sorry

end NUMINAMATH_CALUDE_soccer_committee_count_l547_54788


namespace NUMINAMATH_CALUDE_sum_squares_consecutive_integers_l547_54775

theorem sum_squares_consecutive_integers (a : ℤ) :
  let S := (a - 2)^2 + (a - 1)^2 + a^2 + (a + 1)^2 + (a + 2)^2
  ∃ k : ℤ, S = 5 * k ∧ ¬∃ m : ℤ, S = 25 * m :=
by sorry

end NUMINAMATH_CALUDE_sum_squares_consecutive_integers_l547_54775


namespace NUMINAMATH_CALUDE_three_digit_power_of_2_and_5_l547_54763

theorem three_digit_power_of_2_and_5 : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ m : ℕ, n = 2^m) ∧ 
  (∃ k : ℕ, n = 5^k) :=
sorry

end NUMINAMATH_CALUDE_three_digit_power_of_2_and_5_l547_54763


namespace NUMINAMATH_CALUDE_simplify_radical_expression_l547_54792

theorem simplify_radical_expression :
  ∃ (a b c : ℕ+), 
    (((Real.sqrt 2 - 1) ^ (2 - Real.sqrt 3)) / ((Real.sqrt 2 + 1) ^ (2 + Real.sqrt 3)) = 
     (3 + 2 * Real.sqrt 2) ^ Real.sqrt 3) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p ^ 2 ∣ c.val)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_radical_expression_l547_54792


namespace NUMINAMATH_CALUDE_sum_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l547_54795

-- Problem 1
theorem sum_reciprocals (x y : ℝ) (h1 : x + y = 50) (h2 : x * y = 25) :
  1 / x + 1 / y = 2 := by sorry

-- Problem 2
theorem perpendicular_lines (a b : ℝ) 
  (h : ∀ x y, (a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0) → 
    ((-a / 2) * (-3 / b) = -1)) :
  b = -3 := by sorry

-- Problem 3
theorem equilateral_triangle_perimeter (A : ℝ) (h : A = 100 * Real.sqrt 3) :
  let s := Real.sqrt (4 * A / Real.sqrt 3);
  3 * s = 60 := by sorry

-- Problem 4
theorem polynomial_divisibility (p q : ℝ) 
  (h : ∀ x, (x + 2) ∣ (x^3 - 2*x^2 + p*x + q)) 
  (h_p : p = 60) :
  q = 136 := by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_perpendicular_lines_equilateral_triangle_perimeter_polynomial_divisibility_l547_54795


namespace NUMINAMATH_CALUDE_percentage_of_pine_trees_l547_54799

theorem percentage_of_pine_trees (total_trees : ℕ) (non_pine_trees : ℕ) : 
  total_trees = 350 → non_pine_trees = 105 → 
  (((total_trees - non_pine_trees) : ℚ) / total_trees) * 100 = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_pine_trees_l547_54799


namespace NUMINAMATH_CALUDE_longest_side_of_equal_area_rectangles_l547_54725

/-- Represents a rectangle with integer sides -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem longest_side_of_equal_area_rectangles 
  (r1 r2 r3 : Rectangle) 
  (h_equal_areas : area r1 = area r2 ∧ area r2 = area r3)
  (h_one_side_19 : r1.width = 19 ∨ r1.height = 19 ∨ 
                   r2.width = 19 ∨ r2.height = 19 ∨ 
                   r3.width = 19 ∨ r3.height = 19) :
  ∃ (r : Rectangle), (r = r1 ∨ r = r2 ∨ r = r3) ∧ 
    (r.width = 380 ∨ r.height = 380) :=
sorry

end NUMINAMATH_CALUDE_longest_side_of_equal_area_rectangles_l547_54725


namespace NUMINAMATH_CALUDE_open_box_volume_formula_l547_54768

/-- Represents the volume of an open box constructed from a rectangular metal sheet. -/
def boxVolume (sheetLength sheetWidth x : ℝ) : ℝ :=
  (sheetLength - 2*x) * (sheetWidth - 2*x) * x

theorem open_box_volume_formula (x : ℝ) 
  (h1 : x > 0) 
  (h2 : x < 10) : 
  boxVolume 30 20 x = 600*x - 100*x^2 + 4*x^3 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_formula_l547_54768


namespace NUMINAMATH_CALUDE_pentatonic_scale_theorem_l547_54758

/-- Calculates the length of the instrument for the nth note in the pentatonic scale,
    given the initial length and the number of alternations between subtracting
    and adding one-third. -/
def pentatonic_length (initial_length : ℚ) (n : ℕ) : ℚ :=
  initial_length * (2/3)^(n/2) * (4/3)^((n-1)/2)

theorem pentatonic_scale_theorem (a : ℚ) :
  pentatonic_length a 3 = 32 → a = 54 := by
  sorry

#check pentatonic_scale_theorem

end NUMINAMATH_CALUDE_pentatonic_scale_theorem_l547_54758


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l547_54742

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + m) - a n = m * (a (n + 1) - a n)

theorem arithmetic_sequence_middle_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 3 + a 5 = 16) : 
  a 4 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l547_54742


namespace NUMINAMATH_CALUDE_square_diff_plus_six_b_l547_54736

theorem square_diff_plus_six_b (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6*b = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_plus_six_b_l547_54736


namespace NUMINAMATH_CALUDE_simplify_expression_l547_54745

theorem simplify_expression (z y : ℝ) : (4 - 5*z + 2*y) - (6 + 7*z - 3*y) = -2 - 12*z + 5*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l547_54745


namespace NUMINAMATH_CALUDE_cone_volume_theorem_l547_54786

-- Define the cone properties
def base_radius : ℝ := 1
def lateral_area_ratio : ℝ := 2

-- Theorem statement
theorem cone_volume_theorem :
  let r := base_radius
  let l := lateral_area_ratio * r -- slant height
  let h := Real.sqrt (l^2 - r^2) -- height
  (1/3 : ℝ) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_theorem_l547_54786


namespace NUMINAMATH_CALUDE_faye_candy_count_l547_54722

/-- Calculates the final candy count for Faye after eating some and receiving more. -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Proves that Faye's final candy count is 62 pieces. -/
theorem faye_candy_count :
  final_candy_count 47 25 40 = 62 := by
  sorry

end NUMINAMATH_CALUDE_faye_candy_count_l547_54722


namespace NUMINAMATH_CALUDE_gcd_of_48_72_120_l547_54721

theorem gcd_of_48_72_120 : Nat.gcd 48 (Nat.gcd 72 120) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_48_72_120_l547_54721


namespace NUMINAMATH_CALUDE_strap_mask_probability_is_0_12_l547_54778

/-- Represents a mask factory with two types of products -/
structure MaskFactory where
  regularRatio : ℝ
  surgicalRatio : ℝ
  regularStrapRatio : ℝ
  surgicalStrapRatio : ℝ

/-- The probability of selecting a strap mask from the factory -/
def strapMaskProbability (factory : MaskFactory) : ℝ :=
  factory.regularRatio * factory.regularStrapRatio +
  factory.surgicalRatio * factory.surgicalStrapRatio

/-- Theorem stating the probability of selecting a strap mask -/
theorem strap_mask_probability_is_0_12 :
  let factory : MaskFactory := {
    regularRatio := 0.8,
    surgicalRatio := 0.2,
    regularStrapRatio := 0.1,
    surgicalStrapRatio := 0.2
  }
  strapMaskProbability factory = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_strap_mask_probability_is_0_12_l547_54778


namespace NUMINAMATH_CALUDE_dimitri_weekly_calorie_intake_l547_54797

/-- Represents the daily calorie intake from burgers -/
def daily_calorie_intake (burger_a_calories burger_b_calories burger_c_calories : ℕ) 
  (burger_a_count burger_b_count burger_c_count : ℕ) : ℕ :=
  burger_a_calories * burger_a_count + 
  burger_b_calories * burger_b_count + 
  burger_c_calories * burger_c_count

/-- Calculates the weekly calorie intake based on daily intake -/
def weekly_calorie_intake (daily_intake : ℕ) (days_in_week : ℕ) : ℕ :=
  daily_intake * days_in_week

/-- Theorem stating Dimitri's weekly calorie intake from burgers -/
theorem dimitri_weekly_calorie_intake : 
  weekly_calorie_intake 
    (daily_calorie_intake 350 450 550 2 1 3) 
    7 = 19600 := by
  sorry


end NUMINAMATH_CALUDE_dimitri_weekly_calorie_intake_l547_54797


namespace NUMINAMATH_CALUDE_range_of_m_l547_54759

def is_circle (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 - 2*m*x + 2*m^2 - 2*m = 0

def hyperbola_eccentricity_in_range (m : ℝ) : Prop :=
  ∃ (e : ℝ), 1 < e ∧ e < 2 ∧ e^2 = 1 + m/5

def p (m : ℝ) : Prop := is_circle m
def q (m : ℝ) : Prop := hyperbola_eccentricity_in_range m

theorem range_of_m :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (2 ≤ m ∧ m < 15) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l547_54759


namespace NUMINAMATH_CALUDE_truck_distance_l547_54750

/-- Prove that a truck traveling b/4 feet every t seconds will cover 20b/t yards in 4 minutes -/
theorem truck_distance (b t : ℝ) (h1 : t > 0) : 
  let feet_per_t_seconds := b / 4
  let seconds_in_4_minutes := 4 * 60
  let feet_in_yard := 3
  let yards_in_4_minutes := (feet_per_t_seconds * seconds_in_4_minutes / t) / feet_in_yard
  yards_in_4_minutes = 20 * b / t :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_l547_54750


namespace NUMINAMATH_CALUDE_uncommon_card_ratio_l547_54757

/-- Given a number of card packs, cards per pack, and total uncommon cards,
    prove that the ratio of uncommon cards to total cards per pack is 5:2 -/
theorem uncommon_card_ratio
  (num_packs : ℕ)
  (cards_per_pack : ℕ)
  (total_uncommon : ℕ)
  (h1 : num_packs = 10)
  (h2 : cards_per_pack = 20)
  (h3 : total_uncommon = 50) :
  (total_uncommon : ℚ) / (num_packs * cards_per_pack : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_uncommon_card_ratio_l547_54757


namespace NUMINAMATH_CALUDE_no_roots_of_composite_l547_54746

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem no_roots_of_composite (b c : ℝ) :
  (∀ x : ℝ, f b c x ≠ x) →
  (∀ x : ℝ, f b c (f b c x) ≠ x) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_of_composite_l547_54746


namespace NUMINAMATH_CALUDE_quadratic_function_comparison_l547_54749

theorem quadratic_function_comparison (y₁ y₂ : ℝ) : 
  ((-1 : ℝ)^2 - 2*(-1) = y₁) → 
  ((2 : ℝ)^2 - 2*2 = y₂) → 
  y₁ > y₂ := by sorry

end NUMINAMATH_CALUDE_quadratic_function_comparison_l547_54749


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l547_54794

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define points A, B, and D
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (2, 2)
def point_D : ℝ × ℝ := (0, 1)

-- Define line m
def line_m (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Define line l with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x - 0)

-- Define the property that line m bisects circle C
def bisects (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property that a line intersects a circle at two distinct points
def intersects_at_two_points (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the squared distance between two points on a line
def squared_distance (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_and_line_problem (k : ℝ) :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  bisects line_m circle_C ∧
  intersects_at_two_points (line_l k) circle_C ∧
  squared_distance (line_l k) circle_C = 12 →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l547_54794


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l547_54793

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l547_54793


namespace NUMINAMATH_CALUDE_tangent_line_at_pi_l547_54764

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x + 1

theorem tangent_line_at_pi :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y - f π = m * (x - π)) ↔ (x * Real.exp π + y - 1 - π * Real.exp π = 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_pi_l547_54764


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l547_54705

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →
  (∀ k m, a (k + m) - a k = m * (a 2 - a 1)) →
  ((a 2 - 1)^3 + 5*(a 2 - 1) = 1) →
  ((a 2010 - 1)^3 + 5*(a 2010 - 1) = -1) →
  (a 2 + a 2010 = 2 ∧ S 2011 = 2011) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l547_54705


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l547_54756

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 
    Real.sin (Real.exp (x^2 * Real.sin (5/x)) - 1) + x
  else 
    0

theorem f_derivative_at_zero : 
  deriv f 0 = 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l547_54756


namespace NUMINAMATH_CALUDE_jane_reading_probability_l547_54787

theorem jane_reading_probability (p : ℚ) (h : p = 5/8) :
  1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_jane_reading_probability_l547_54787


namespace NUMINAMATH_CALUDE_wooden_block_length_is_3070_l547_54706

/-- The length of a wooden block in centimeters, given that it is 30 cm shorter than 31 meters -/
def wooden_block_length : ℕ :=
  let meters_to_cm : ℕ → ℕ := (· * 100)
  meters_to_cm 31 - 30

theorem wooden_block_length_is_3070 : wooden_block_length = 3070 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_length_is_3070_l547_54706


namespace NUMINAMATH_CALUDE_sampling_properties_l547_54740

/-- Represents a club with male and female members -/
structure Club where
  male_members : ℕ
  female_members : ℕ

/-- Represents a sample drawn from the club -/
structure Sample where
  size : ℕ
  males_selected : ℕ
  females_selected : ℕ

/-- The probability of selecting a male from the club -/
def prob_select_male (c : Club) (s : Sample) : ℚ :=
  s.males_selected / c.male_members

/-- The probability of selecting a female from the club -/
def prob_select_female (c : Club) (s : Sample) : ℚ :=
  s.females_selected / c.female_members

/-- Theorem about the sampling properties of a specific club and sample -/
theorem sampling_properties (c : Club) (s : Sample) 
    (h_male : c.male_members = 30)
    (h_female : c.female_members = 20)
    (h_sample_size : s.size = 5)
    (h_males_selected : s.males_selected = 2)
    (h_females_selected : s.females_selected = 3) :
  (∃ (sampling_method : String), sampling_method = "random") ∧
  (¬ ∃ (sampling_method : String), sampling_method = "stratified") ∧
  prob_select_male c s < prob_select_female c s :=
by sorry

end NUMINAMATH_CALUDE_sampling_properties_l547_54740


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l547_54704

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l547_54704


namespace NUMINAMATH_CALUDE_zach_cookies_theorem_l547_54707

/-- The number of cookies Zach baked over three days --/
def total_cookies (monday_cookies : ℕ) : ℕ :=
  let tuesday_cookies := monday_cookies / 2
  let wednesday_cookies := tuesday_cookies * 3
  monday_cookies + tuesday_cookies + wednesday_cookies - 4

/-- Theorem stating that Zach had 92 cookies at the end of three days --/
theorem zach_cookies_theorem :
  total_cookies 32 = 92 :=
by sorry

end NUMINAMATH_CALUDE_zach_cookies_theorem_l547_54707


namespace NUMINAMATH_CALUDE_katies_speed_l547_54716

/-- Given the running speeds of Eugene, Brianna, Marcus, and Katie, prove Katie's speed -/
theorem katies_speed (eugene_speed : ℝ) (brianna_ratio : ℝ) (marcus_ratio : ℝ) (katie_ratio : ℝ)
  (h1 : eugene_speed = 5)
  (h2 : brianna_ratio = 3/4)
  (h3 : marcus_ratio = 5/6)
  (h4 : katie_ratio = 4/5) :
  katie_ratio * marcus_ratio * brianna_ratio * eugene_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_katies_speed_l547_54716


namespace NUMINAMATH_CALUDE_opponents_total_score_l547_54766

def baseball_problem (team_scores : List ℕ) (games_lost : ℕ) : Prop :=
  let total_games := team_scores.length
  let lost_scores := team_scores.take games_lost
  let won_scores := team_scores.drop games_lost
  
  -- Conditions
  total_games = 7 ∧
  team_scores = [1, 3, 5, 6, 7, 8, 10] ∧
  games_lost = 3 ∧
  
  -- Lost games: opponent scored 2 more than the team
  (List.sum (lost_scores.map (· + 2))) +
  -- Won games: team scored 3 times opponent's score
  (List.sum (won_scores.map (· / 3))) = 24

theorem opponents_total_score :
  baseball_problem [1, 3, 5, 6, 7, 8, 10] 3 := by
  sorry

end NUMINAMATH_CALUDE_opponents_total_score_l547_54766


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l547_54702

theorem subset_implies_a_range (M N : Set ℝ) (a : ℝ) 
  (hM : M = {x : ℝ | x - 2 < 0})
  (hN : N = {x : ℝ | x < a})
  (hSubset : M ⊆ N) :
  a ∈ Set.Ici 2 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l547_54702


namespace NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_sum_of_roots_inequality_l547_54796

theorem cauchy_schwarz_and_inequality (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) ≥ (a*x + b*y + c*z)^2 :=
sorry

theorem sum_of_roots_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  Real.sqrt a + Real.sqrt (2 * b) + Real.sqrt (3 * c) ≤ Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_and_inequality_sum_of_roots_inequality_l547_54796


namespace NUMINAMATH_CALUDE_tic_tac_toe_wins_l547_54728

theorem tic_tac_toe_wins (total_rounds harry_wins william_wins : ℕ) :
  total_rounds = 15 →
  william_wins = harry_wins + 5 →
  william_wins = 10 := by
sorry

end NUMINAMATH_CALUDE_tic_tac_toe_wins_l547_54728


namespace NUMINAMATH_CALUDE_abs_neg_one_third_l547_54752

theorem abs_neg_one_third : |(-1 : ℚ) / 3| = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_third_l547_54752


namespace NUMINAMATH_CALUDE_toothpick_grid_25_15_l547_54731

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := grid.height * grid.width
  horizontal + vertical + diagonal

/-- Theorem stating the total number of toothpicks in a 25x15 grid -/
theorem toothpick_grid_25_15 :
  total_toothpicks ⟨25, 15⟩ = 1165 := by sorry

end NUMINAMATH_CALUDE_toothpick_grid_25_15_l547_54731


namespace NUMINAMATH_CALUDE_simplify_expression_l547_54739

theorem simplify_expression (h : Real.pi < 4) : 
  Real.sqrt ((Real.pi - 4)^2) + Real.pi = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l547_54739


namespace NUMINAMATH_CALUDE_book_has_180_pages_l547_54791

/-- Calculates the number of pages in a book given reading habits and time to finish --/
def book_pages (weekday_pages : ℕ) (weekend_pages : ℕ) (weeks : ℕ) : ℕ :=
  let weekdays := 5 * weeks
  let weekends := 2 * weeks
  weekday_pages * weekdays + weekend_pages * weekends

/-- Theorem stating that a book has 180 pages given specific reading habits and time --/
theorem book_has_180_pages :
  book_pages 10 20 2 = 180 := by
  sorry

#eval book_pages 10 20 2

end NUMINAMATH_CALUDE_book_has_180_pages_l547_54791


namespace NUMINAMATH_CALUDE_volume_polynomial_coefficients_ratio_l547_54701

/-- A right rectangular prism with edge lengths 2, 2, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 2
  height : ℝ := 5

/-- The set of points within distance r of any point in the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume (B : Prism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume polynomial -/
structure VolumeCoefficients where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

theorem volume_polynomial_coefficients_ratio (B : Prism) (coeff : VolumeCoefficients) :
  (∀ r : ℝ, volume B r = coeff.a * r^3 + coeff.b * r^2 + coeff.c * r + coeff.d) →
  (coeff.a > 0 ∧ coeff.b > 0 ∧ coeff.c > 0 ∧ coeff.d > 0) →
  (coeff.b * coeff.c) / (coeff.a * coeff.d) = 50.4 := by
  sorry

end NUMINAMATH_CALUDE_volume_polynomial_coefficients_ratio_l547_54701


namespace NUMINAMATH_CALUDE_parabola_focus_l547_54773

/-- The focus of a parabola y^2 = -4x is at (-1, 0) -/
theorem parabola_focus (x y : ℝ) :
  y^2 = -4*x → (x + 1)^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_l547_54773


namespace NUMINAMATH_CALUDE_andrews_age_l547_54724

theorem andrews_age (andrew_age grandfather_age : ℕ) : 
  grandfather_age = 10 * andrew_age →
  grandfather_age - andrew_age = 63 →
  andrew_age = 7 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l547_54724


namespace NUMINAMATH_CALUDE_pasture_problem_l547_54772

/-- The number of horses b put in the pasture -/
def b_horses : ℕ := 16

/-- The total cost of the pasture in Rs -/
def total_cost : ℕ := 435

/-- The amount b should pay in Rs -/
def b_payment : ℕ := 180

/-- The number of horses a put in -/
def a_horses : ℕ := 12

/-- The number of months a's horses stayed -/
def a_months : ℕ := 8

/-- The number of months b's horses stayed -/
def b_months : ℕ := 9

/-- The number of horses c put in -/
def c_horses : ℕ := 18

/-- The number of months c's horses stayed -/
def c_months : ℕ := 6

theorem pasture_problem :
  b_horses = 16 ∧
  (b_horses * b_months : ℚ) / (a_horses * a_months + b_horses * b_months + c_horses * c_months : ℚ) =
  b_payment / total_cost := by
  sorry

end NUMINAMATH_CALUDE_pasture_problem_l547_54772


namespace NUMINAMATH_CALUDE_log_equation_solution_l547_54783

theorem log_equation_solution :
  ∀ y : ℝ, (Real.log y / Real.log 9 = Real.log 8 / Real.log 2) → y = 729 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l547_54783


namespace NUMINAMATH_CALUDE_apples_per_basket_l547_54713

theorem apples_per_basket (total_baskets : ℕ) (total_apples : ℕ) 
  (h1 : total_baskets = 37) 
  (h2 : total_apples = 629) : 
  total_apples / total_baskets = 17 := by
sorry

end NUMINAMATH_CALUDE_apples_per_basket_l547_54713


namespace NUMINAMATH_CALUDE_tshirt_cost_l547_54765

theorem tshirt_cost (initial_amount : ℕ) (sweater_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 91 →
  sweater_cost = 24 →
  shoes_cost = 11 →
  remaining_amount = 50 →
  initial_amount - remaining_amount - sweater_cost - shoes_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_tshirt_cost_l547_54765


namespace NUMINAMATH_CALUDE_bakery_boxes_sold_l547_54727

/-- Calculates the number of boxes of doughnuts sold by a bakery -/
theorem bakery_boxes_sold
  (doughnuts_per_box : ℕ)
  (total_doughnuts : ℕ)
  (doughnuts_given_away : ℕ)
  (h1 : doughnuts_per_box = 10)
  (h2 : total_doughnuts = 300)
  (h3 : doughnuts_given_away = 30) :
  (total_doughnuts / doughnuts_per_box) - (doughnuts_given_away / doughnuts_per_box) = 27 :=
by sorry

end NUMINAMATH_CALUDE_bakery_boxes_sold_l547_54727


namespace NUMINAMATH_CALUDE_max_telephones_is_210_quality_rate_at_least_90_percent_l547_54723

/-- Represents the quality inspection of a batch of telephones. -/
structure TelephoneBatch where
  first_50_high_quality : Nat := 49
  first_50_total : Nat := 50
  subsequent_high_quality : Nat := 7
  subsequent_total : Nat := 8
  quality_threshold : Rat := 9/10

/-- The maximum number of telephones in the batch satisfying the quality conditions. -/
def max_telephones (batch : TelephoneBatch) : Nat :=
  batch.first_50_total + 20 * batch.subsequent_total

/-- Theorem stating that 210 is the maximum number of telephones in the batch. -/
theorem max_telephones_is_210 (batch : TelephoneBatch) :
  max_telephones batch = 210 :=
by sorry

/-- Theorem stating that the quality rate is at least 90% for the maximum batch size. -/
theorem quality_rate_at_least_90_percent (batch : TelephoneBatch) :
  let total := max_telephones batch
  let high_quality := batch.first_50_high_quality + 20 * batch.subsequent_high_quality
  (high_quality : Rat) / total ≥ batch.quality_threshold :=
by sorry

end NUMINAMATH_CALUDE_max_telephones_is_210_quality_rate_at_least_90_percent_l547_54723


namespace NUMINAMATH_CALUDE_parabola_focus_l547_54715

/-- A parabola is defined by the equation x = -1/8 * y^2. Its focus is at (-2, 0). -/
theorem parabola_focus (x y : ℝ) : 
  x = -1/8 * y^2 → (x + 2 = 0 ∧ y = 0) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_l547_54715


namespace NUMINAMATH_CALUDE_apple_cost_theorem_l547_54781

/-- The cost of groceries for Olivia -/
def grocery_problem (total_cost banana_cost bread_cost milk_cost apple_cost : ℕ) : Prop :=
  total_cost = 42 ∧
  banana_cost = 12 ∧
  bread_cost = 9 ∧
  milk_cost = 7 ∧
  apple_cost = total_cost - (banana_cost + bread_cost + milk_cost)

theorem apple_cost_theorem :
  ∃ (apple_cost : ℕ), grocery_problem 42 12 9 7 apple_cost ∧ apple_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_apple_cost_theorem_l547_54781


namespace NUMINAMATH_CALUDE_four_digit_with_four_or_five_l547_54751

/-- The number of four-digit positive integers -/
def total_four_digit : ℕ := 9000

/-- The number of four-digit positive integers without 4 or 5 -/
def without_four_or_five : ℕ := 3584

/-- The number of four-digit positive integers with at least one 4 or 5 -/
def with_four_or_five : ℕ := total_four_digit - without_four_or_five

theorem four_digit_with_four_or_five : with_four_or_five = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_with_four_or_five_l547_54751


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l547_54719

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℝ, (X^4 + 3•X^2 - 4) = (X^2 + 2) * q + (X^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l547_54719


namespace NUMINAMATH_CALUDE_remainder_problem_l547_54700

theorem remainder_problem (x : ℤ) : x % 63 = 25 → x % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l547_54700


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l547_54743

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x + 2) = 0) ∧
  (∃ x : ℝ, (x - 1) * (x + 2) = 0 ∧ x - 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l547_54743


namespace NUMINAMATH_CALUDE_outfits_count_l547_54780

/-- The number of shirts available. -/
def num_shirts : ℕ := 8

/-- The number of pants available. -/
def num_pants : ℕ := 5

/-- The number of ties available. -/
def num_ties : ℕ := 4

/-- The number of belts available. -/
def num_belts : ℕ := 2

/-- The total number of outfit combinations. -/
def total_outfits : ℕ := num_shirts * num_pants * (num_ties + 1) * (num_belts + 1)

/-- Theorem stating that the total number of outfits is 600. -/
theorem outfits_count : total_outfits = 600 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l547_54780


namespace NUMINAMATH_CALUDE_x_plus_2y_inequality_l547_54785

theorem x_plus_2y_inequality (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 1/y = 1) :
  x + 2*y > m^2 + 2*m ↔ m > -4 ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_x_plus_2y_inequality_l547_54785


namespace NUMINAMATH_CALUDE_piggy_bank_equality_days_l547_54790

def minjoo_initial : ℕ := 12000
def siwoo_initial : ℕ := 4000
def minjoo_daily : ℕ := 300
def siwoo_daily : ℕ := 500

theorem piggy_bank_equality_days : 
  ∃ d : ℕ, d = 40 ∧ 
  minjoo_initial + d * minjoo_daily = siwoo_initial + d * siwoo_daily :=
sorry

end NUMINAMATH_CALUDE_piggy_bank_equality_days_l547_54790


namespace NUMINAMATH_CALUDE_angle_bisector_ratio_l547_54777

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the angle bisectors
def angleBisector (T : Triangle) (vertex : ℝ × ℝ) (side1 side2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the intersection point Q
def intersectionPoint (T : Triangle) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_ratio (T : Triangle) :
  let X := T.X
  let Y := T.Y
  let Z := T.Z
  let U := angleBisector T X Y Z
  let V := angleBisector T Y X Z
  let Q := intersectionPoint T
  distance X Y = 8 ∧ distance X Z = 6 ∧ distance Y Z = 4 →
  distance Y Q / distance Q V = 2 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_ratio_l547_54777


namespace NUMINAMATH_CALUDE_cookie_boxes_problem_l547_54714

theorem cookie_boxes_problem (n : ℕ) : 
  (∃ (mark_sold ann_sold : ℕ), 
    mark_sold = n - 9 ∧ 
    ann_sold = n - 2 ∧ 
    mark_sold ≥ 1 ∧ 
    ann_sold ≥ 1 ∧ 
    mark_sold + ann_sold < n) ↔ 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_cookie_boxes_problem_l547_54714


namespace NUMINAMATH_CALUDE_sector_area_l547_54726

/-- The area of a circular sector with radius 12 meters and central angle 42 degrees -/
theorem sector_area : 
  let r : ℝ := 12
  let θ : ℝ := 42
  let sector_area := (θ / 360) * Real.pi * r^2
  sector_area = (42 / 360) * Real.pi * 12^2 := by sorry

end NUMINAMATH_CALUDE_sector_area_l547_54726


namespace NUMINAMATH_CALUDE_perfect_cube_factors_of_8820_l547_54720

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_cube (n : ℕ) : Prop := sorry

def count_perfect_cube_factors (n : ℕ) : ℕ := sorry

theorem perfect_cube_factors_of_8820 :
  let factorization := prime_factorization 8820
  (factorization = [(2, 2), (3, 2), (5, 1), (7, 2)]) →
  count_perfect_cube_factors 8820 = 1 := by sorry

end NUMINAMATH_CALUDE_perfect_cube_factors_of_8820_l547_54720


namespace NUMINAMATH_CALUDE_no_valid_box_dimensions_l547_54717

theorem no_valid_box_dimensions :
  ¬∃ (a b c : ℕ), 
    (Prime a) ∧ (Prime b) ∧ (Prime c) ∧
    (a ≤ b) ∧ (b ≤ c) ∧
    (a * b * c = 2 * (a * b + b * c + a * c)) ∧
    (Prime (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_box_dimensions_l547_54717


namespace NUMINAMATH_CALUDE_cone_rolling_ratio_l547_54784

/-- 
Theorem: For a right circular cone with base radius r and height h, 
if the cone makes 23 complete rotations when rolled on its side, 
then h/r = 4√33.
-/
theorem cone_rolling_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (2 * Real.pi * Real.sqrt (r^2 + h^2) = 46 * Real.pi * r) → 
  (h / r = 4 * Real.sqrt 33) := by
sorry

end NUMINAMATH_CALUDE_cone_rolling_ratio_l547_54784


namespace NUMINAMATH_CALUDE_circumcircle_equation_l547_54708

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define an equilateral triangle on the parabola
def equilateral_triangle_on_parabola (A B : ℝ × ℝ) : Prop :=
  let O := (0, 0)
  parabola O.1 O.2 ∧ parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - O.1)^2 + (B.2 - O.2)^2 ∧
  (A.1 - O.1)^2 + (A.2 - O.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Theorem statement
theorem circumcircle_equation (A B : ℝ × ℝ) :
  equilateral_triangle_on_parabola A B →
  ∃ x y : ℝ, (x - 4)^2 + y^2 = 16 ∧
            (x - 0)^2 + (y - 0)^2 = (x - A.1)^2 + (y - A.2)^2 ∧
            (x - 0)^2 + (y - 0)^2 = (x - B.1)^2 + (y - B.2)^2 :=
sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l547_54708


namespace NUMINAMATH_CALUDE_unique_solution_factorial_equation_l547_54730

theorem unique_solution_factorial_equation : 
  ∃! (n : ℕ), (Nat.factorial (n + 2) - Nat.factorial (n + 1) - Nat.factorial n = n^2 + n^4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_factorial_equation_l547_54730


namespace NUMINAMATH_CALUDE_problem_statement_l547_54748

theorem problem_statement (x : ℝ) (h : x + 1/x = 7) : 
  (x - 3)^2 + 49 / (x - 3)^2 = 23 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l547_54748


namespace NUMINAMATH_CALUDE_letter_value_puzzle_l547_54735

theorem letter_value_puzzle (L E A D : ℤ) : 
  L = 15 →
  L + E + A + D = 41 →
  D + E + A + L = 45 →
  A + D + D + E + D = 53 →
  D = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_letter_value_puzzle_l547_54735


namespace NUMINAMATH_CALUDE_work_completion_time_l547_54703

theorem work_completion_time (a b c : ℝ) (h1 : b = 6) (h2 : c = 12) 
  (h3 : 1/a + 1/b + 1/c = 7/24) : a = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l547_54703


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_neg_three_fourths_l547_54734

theorem sum_of_solutions_eq_neg_three_fourths :
  let f : ℝ → ℝ := λ x => 243^(x + 1) - 81^(x^2 + 2*x)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_neg_three_fourths_l547_54734


namespace NUMINAMATH_CALUDE_power_equation_solution_l547_54767

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^22 ↔ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l547_54767


namespace NUMINAMATH_CALUDE_no_positive_integer_solutions_l547_54737

theorem no_positive_integer_solutions : ∀ A : ℕ, 
  1 ≤ A → A ≤ 9 → 
  ¬∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p * q = Nat.factorial A ∧ p + q = 10 * A + A := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solutions_l547_54737


namespace NUMINAMATH_CALUDE_total_seedlings_transferred_l547_54789

def seedlings_day1 : ℕ := 200

def seedlings_day2 (day1 : ℕ) : ℕ := 2 * day1

theorem total_seedlings_transferred : 
  seedlings_day1 + seedlings_day2 seedlings_day1 = 600 := by
  sorry

end NUMINAMATH_CALUDE_total_seedlings_transferred_l547_54789


namespace NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l547_54711

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the concept of disjoint circles
def disjoint (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

-- Define the intersection point of external tangents
def external_tangent_intersection (c1 c2 : Circle) : ℝ × ℝ :=
  sorry -- The actual computation is not needed for the statement

-- Define collinearity of three points
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

-- The main theorem
theorem external_tangent_intersections_collinear (C1 C2 C3 : Circle)
  (h12 : disjoint C1 C2) (h23 : disjoint C2 C3) (h31 : disjoint C3 C1) :
  let T12 := external_tangent_intersection C1 C2
  let T23 := external_tangent_intersection C2 C3
  let T31 := external_tangent_intersection C3 C1
  collinear T12 T23 T31 :=
sorry

end NUMINAMATH_CALUDE_external_tangent_intersections_collinear_l547_54711


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l547_54753

theorem sqrt_meaningful_range (m : ℝ) : 
  (∃ (x : ℝ), x^2 = m + 3) ↔ m ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l547_54753
