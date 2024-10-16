import Mathlib

namespace NUMINAMATH_CALUDE_prime_relation_l2707_270700

theorem prime_relation (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → q = 11 * p + 1 → q = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_relation_l2707_270700


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2707_270796

theorem absolute_value_inequality (x : ℝ) :
  |((3 * x + 2) / (x - 2))| ≥ 3 ↔ x ∈ Set.Ici (2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2707_270796


namespace NUMINAMATH_CALUDE_masha_numbers_proof_l2707_270730

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ (a % 2 = 0 ∨ b % 2 = 0)

def is_unique_pair (a b : ℕ) : Prop :=
  ∀ x y : ℕ, x + y = a + b → is_valid_pair x y → (x = a ∧ y = b) ∨ (x = b ∧ y = a)

theorem masha_numbers_proof :
  ∃! (a b : ℕ), is_valid_pair a b ∧ is_unique_pair a b ∧ a + b = 28 :=
sorry

end NUMINAMATH_CALUDE_masha_numbers_proof_l2707_270730


namespace NUMINAMATH_CALUDE_x_value_proof_l2707_270775

theorem x_value_proof (x : ℝ) (h1 : x^2 - 3*x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2707_270775


namespace NUMINAMATH_CALUDE_min_value_z_l2707_270791

theorem min_value_z (x y : ℝ) : 
  x^2 + 2*y^2 + y^3 + 6*x - 4*y + 30 ≥ 20 ∧ 
  ∃ x y : ℝ, x^2 + 2*y^2 + y^3 + 6*x - 4*y + 30 = 20 :=
by sorry

end NUMINAMATH_CALUDE_min_value_z_l2707_270791


namespace NUMINAMATH_CALUDE_sum_of_bases_l2707_270780

-- Define the fractions F₁ and F₂
def F₁ (R : ℕ) : ℚ := (4 * R + 5) / (R^2 - 1)
def F₂ (R : ℕ) : ℚ := (5 * R + 4) / (R^2 - 1)

-- Define the conditions
def conditions (R₁ R₂ : ℕ) : Prop :=
  F₁ R₁ = F₁ R₂ ∧ F₂ R₁ = F₂ R₂ ∧
  R₁ ≥ 2 ∧ R₂ ≥ 2 ∧ -- Ensure bases are valid
  (∃ k : ℕ, F₁ R₁ = k / 11) ∧ -- Represents the repeating decimal 0.454545...
  (∃ k : ℕ, F₂ R₁ = k / 11) ∧ -- Represents the repeating decimal 0.545454...
  (∃ k : ℕ, F₁ R₂ = k / 11) ∧ -- Represents the repeating decimal 0.363636...
  (∃ k : ℕ, F₂ R₂ = k / 11)   -- Represents the repeating decimal 0.636363...

-- State the theorem
theorem sum_of_bases (R₁ R₂ : ℕ) : 
  conditions R₁ R₂ → R₁ + R₂ = 19 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_bases_l2707_270780


namespace NUMINAMATH_CALUDE_alcohol_concentration_in_second_vessel_l2707_270721

/-- 
Given two vessels with different capacities and alcohol concentrations, 
prove that when mixed and diluted to a certain concentration, 
the alcohol percentage in the second vessel can be determined.
-/
theorem alcohol_concentration_in_second_vessel 
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_capacity : ℝ)
  (final_mixture_capacity : ℝ)
  (final_mixture_percentage : ℝ)
  (h1 : vessel1_capacity = 2)
  (h2 : vessel1_alcohol_percentage = 30)
  (h3 : vessel2_capacity = 6)
  (h4 : total_capacity = vessel1_capacity + vessel2_capacity)
  (h5 : final_mixture_capacity = 10)
  (h6 : final_mixture_percentage = 30) :
  ∃ vessel2_alcohol_percentage : ℝ, 
    vessel2_alcohol_percentage = 30 ∧
    vessel1_capacity * (vessel1_alcohol_percentage / 100) + 
    vessel2_capacity * (vessel2_alcohol_percentage / 100) = 
    total_capacity * (final_mixture_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_concentration_in_second_vessel_l2707_270721


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l2707_270740

theorem isosceles_right_triangle_roots (p q : ℂ) (z₁ z₂ : ℂ) : 
  z₁^2 + 2*p*z₁ + q = 0 →
  z₂^2 + 2*p*z₂ + q = 0 →
  z₂ = Complex.I * z₁ →
  p^2 / q = 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l2707_270740


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2707_270739

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 2*x

-- State the theorem
theorem tangent_slope_at_one :
  (deriv f) 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2707_270739


namespace NUMINAMATH_CALUDE_purple_walls_count_l2707_270701

theorem purple_walls_count (total_rooms : ℕ) (walls_per_room : ℕ) (green_ratio : ℚ) : 
  total_rooms = 10 → 
  walls_per_room = 8 → 
  green_ratio = 3/5 → 
  (total_rooms - total_rooms * green_ratio) * walls_per_room = 32 := by
sorry

end NUMINAMATH_CALUDE_purple_walls_count_l2707_270701


namespace NUMINAMATH_CALUDE_cos_negative_135_degrees_l2707_270720

theorem cos_negative_135_degrees : Real.cos ((-135 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_135_degrees_l2707_270720


namespace NUMINAMATH_CALUDE_root_implies_a_range_l2707_270729

def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x + 4

theorem root_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = 0) → a ∈ Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_root_implies_a_range_l2707_270729


namespace NUMINAMATH_CALUDE_sqrt_fifth_root_of_five_sixth_power_l2707_270766

theorem sqrt_fifth_root_of_five_sixth_power :
  (Real.sqrt ((Real.sqrt 5) ^ 5)) ^ 6 = 5 ^ (15 / 2) := by sorry

end NUMINAMATH_CALUDE_sqrt_fifth_root_of_five_sixth_power_l2707_270766


namespace NUMINAMATH_CALUDE_tile_border_ratio_l2707_270724

theorem tile_border_ratio (s d : ℝ) (h_positive : s > 0 ∧ d > 0) : 
  (15 * s)^2 / ((15 * s + 2 * 15 * d)^2) = 3/4 → d/s = 1/13 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l2707_270724


namespace NUMINAMATH_CALUDE_fibonacci_eight_sum_not_equal_single_l2707_270747

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_eight_sum_not_equal_single (k : ℕ) : 
  ¬∃ m : ℕ, 
    fibonacci k + fibonacci (k + 1) + fibonacci (k + 2) + fibonacci (k + 3) + 
    fibonacci (k + 4) + fibonacci (k + 5) + fibonacci (k + 6) + fibonacci (k + 7) = 
    fibonacci m := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_eight_sum_not_equal_single_l2707_270747


namespace NUMINAMATH_CALUDE_arccos_negative_half_equals_two_pi_thirds_l2707_270749

theorem arccos_negative_half_equals_two_pi_thirds : 
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_negative_half_equals_two_pi_thirds_l2707_270749


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2707_270710

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 + b^3 = 35*x^3) 
  (h3 : a^2 - b^2 = 4*x^2) : 
  a = 2*x ∨ a = -2*x := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2707_270710


namespace NUMINAMATH_CALUDE_integral_over_pyramidal_region_l2707_270736

/-- The pyramidal region V -/
def V : Set (Fin 3 → ℝ) :=
  {v | ∀ i, v i ≥ 0 ∧ v 0 + v 1 + v 2 ≤ 1}

/-- The integrand function -/
def f (v : Fin 3 → ℝ) : ℝ :=
  v 0 * v 1^9 * v 2^8 * (1 - v 0 - v 1 - v 2)^4

/-- The theorem statement -/
theorem integral_over_pyramidal_region :
  ∫ v in V, f v = (Nat.factorial 9 * Nat.factorial 8 * Nat.factorial 4) / Nat.factorial 25 := by
  sorry

end NUMINAMATH_CALUDE_integral_over_pyramidal_region_l2707_270736


namespace NUMINAMATH_CALUDE_horner_method_operations_count_l2707_270713

def horner_polynomial (x : ℝ) : ℝ := 9*x^6 + 12*x^5 + 7*x^4 + 54*x^3 + 34*x^2 + 9*x + 1

def horner_method_operations (p : ℝ → ℝ) : ℕ × ℕ :=
  match p with
  | f => (6, 6)  -- Placeholder for the actual implementation

theorem horner_method_operations_count :
  ∀ x : ℝ, horner_method_operations horner_polynomial = (6, 6) := by
  sorry

end NUMINAMATH_CALUDE_horner_method_operations_count_l2707_270713


namespace NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l2707_270711

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a-2)*x - a * log x

theorem f_derivative_positive_at_midpoint (a c x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) 
  (h₄ : f a x₁ = c) (h₅ : f a x₂ = c) : 
  deriv (f a) ((x₁ + x₂) / 2) > 0 := by
sorry

end NUMINAMATH_CALUDE_f_derivative_positive_at_midpoint_l2707_270711


namespace NUMINAMATH_CALUDE_smallest_result_l2707_270734

def S : Set Nat := {2, 3, 5, 7, 11, 13}

theorem smallest_result (a b c : Nat) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (∀ x y z : Nat, x ∈ S → y ∈ S → z ∈ S → x ≠ y ∧ y ≠ z ∧ x ≠ z → 
    22 ≤ (x + x + y) * z) ∧ (∃ x y z : Nat, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ (x + x + y) * z = 22) := by
  sorry

end NUMINAMATH_CALUDE_smallest_result_l2707_270734


namespace NUMINAMATH_CALUDE_circle_translation_l2707_270776

/-- Given a circle equation, prove its center, radius, and translated form -/
theorem circle_translation (x y : ℝ) :
  let original_eq := x^2 + y^2 - 4*x + 6*y - 68 = 0
  let center := (2, -3)
  let radius := 9
  let X := x - 2
  let Y := y + 3
  let translated_eq := X^2 + Y^2 = 81
  original_eq → (
    (x - center.1)^2 + (y - center.2)^2 = radius^2 ∧
    translated_eq
  ) := by sorry

end NUMINAMATH_CALUDE_circle_translation_l2707_270776


namespace NUMINAMATH_CALUDE_sum_distribution_l2707_270725

/-- The sum distribution problem -/
theorem sum_distribution (p q r s t : ℝ) : 
  (q = 0.75 * p) →  -- q gets 75 cents for each dollar p gets
  (r = 0.50 * p) →  -- r gets 50 cents for each dollar p gets
  (s = 0.25 * p) →  -- s gets 25 cents for each dollar p gets
  (t = 0.10 * p) →  -- t gets 10 cents for each dollar p gets
  (s = 25) →        -- The share of s is 25 dollars
  (p + q + r + s + t = 260) := by  -- The total sum is 260 dollars
sorry


end NUMINAMATH_CALUDE_sum_distribution_l2707_270725


namespace NUMINAMATH_CALUDE_town_distance_approx_l2707_270790

/-- Represents the map scale as a fraction of inches per mile -/
def map_scale : ℚ := 7 / (15 * 19)

/-- Represents the distance between two points on the map in inches -/
def map_distance : ℚ := 37 / 8

/-- Calculates the actual distance in miles given the map scale and map distance -/
def actual_distance (scale : ℚ) (distance : ℚ) : ℚ := distance / scale

/-- Theorem stating that the actual distance between the towns is approximately 41.0083 miles -/
theorem town_distance_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/10000 ∧ 
  |actual_distance map_scale map_distance - 41.0083| < ε :=
sorry

end NUMINAMATH_CALUDE_town_distance_approx_l2707_270790


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2707_270703

theorem fraction_inequality_solution_set (x : ℝ) :
  x ≠ 0 → ((x - 1) / x ≤ 0 ↔ 0 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2707_270703


namespace NUMINAMATH_CALUDE_percentage_error_calculation_l2707_270761

theorem percentage_error_calculation : 
  let correct_multiplier : ℚ := 5/3
  let incorrect_multiplier : ℚ := 3/5
  let percentage_error := ((correct_multiplier - incorrect_multiplier) / correct_multiplier) * 100
  percentage_error = 64
  := by sorry

end NUMINAMATH_CALUDE_percentage_error_calculation_l2707_270761


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l2707_270708

theorem rectangular_field_dimensions (m : ℝ) : 
  (3 * m + 8) * (m - 3) = 72 → m = (1 + Real.sqrt 1153) / 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l2707_270708


namespace NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l2707_270765

/-- Given two curves C₁ and C₂, where C₁ is y = ax³ - x² + 2x and C₂ is y = e^x,
    if their tangent lines are perpendicular at x = 1, then a = -1/(3e) -/
theorem perpendicular_tangents_imply_a_value (a : ℝ) :
  let C₁ : ℝ → ℝ := λ x ↦ a * x^3 - x^2 + 2*x
  let C₂ : ℝ → ℝ := λ x ↦ Real.exp x
  let tangent_C₁ : ℝ := 3*a - 2 + 2  -- Derivative of C₁ at x = 1
  let tangent_C₂ : ℝ := Real.exp 1   -- Derivative of C₂ at x = 1
  (tangent_C₁ * tangent_C₂ = -1) →   -- Condition for perpendicular tangents
  a = -1 / (3 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_imply_a_value_l2707_270765


namespace NUMINAMATH_CALUDE_square_park_fencing_cost_l2707_270743

/-- The cost of fencing one side of a square park -/
def cost_per_side : ℕ := 43

/-- The number of sides in a square -/
def num_sides : ℕ := 4

/-- The total cost of fencing a square park -/
def total_cost : ℕ := cost_per_side * num_sides

/-- Theorem: The total cost of fencing a square park is $172 -/
theorem square_park_fencing_cost :
  total_cost = 172 := by
  sorry

end NUMINAMATH_CALUDE_square_park_fencing_cost_l2707_270743


namespace NUMINAMATH_CALUDE_determinant_transformation_l2707_270717

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 12 := by
  sorry

end NUMINAMATH_CALUDE_determinant_transformation_l2707_270717


namespace NUMINAMATH_CALUDE_initial_egg_count_l2707_270715

theorem initial_egg_count (total : ℕ) (taken : ℕ) (left : ℕ) : 
  taken = 5 → left = 42 → total = taken + left → total = 47 := by
  sorry

end NUMINAMATH_CALUDE_initial_egg_count_l2707_270715


namespace NUMINAMATH_CALUDE_inequality_for_natural_numbers_l2707_270754

theorem inequality_for_natural_numbers (n : ℕ) :
  (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n := by sorry

end NUMINAMATH_CALUDE_inequality_for_natural_numbers_l2707_270754


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2707_270748

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2707_270748


namespace NUMINAMATH_CALUDE_total_upload_hours_l2707_270795

def upload_hours (days : ℕ) (videos_per_day : ℕ) (hours_per_video : ℚ) : ℚ :=
  (days : ℚ) * (videos_per_day : ℚ) * hours_per_video

def june_upload_hours : ℚ :=
  upload_hours 10 5 2 +  -- June 1st to June 10th
  upload_hours 10 10 1 + -- June 11th to June 20th
  upload_hours 5 7 3 +   -- June 21st to June 25th
  upload_hours 5 15 (1/2) -- June 26th to June 30th

theorem total_upload_hours : june_upload_hours = 342.5 := by
  sorry

end NUMINAMATH_CALUDE_total_upload_hours_l2707_270795


namespace NUMINAMATH_CALUDE_sum_equals_seven_x_l2707_270707

theorem sum_equals_seven_x (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 2 * y) : 
  x + y + z = 7 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_seven_x_l2707_270707


namespace NUMINAMATH_CALUDE_smallest_unpayable_amount_l2707_270738

/-- Represents the number of coins of each denomination -/
structure CoinCollection where
  fiveP : Nat
  fourP : Nat
  threeP : Nat
  twoP : Nat
  oneP : Nat

/-- Calculates the total value of a coin collection in pence -/
def totalValue (coins : CoinCollection) : Nat :=
  5 * coins.fiveP + 4 * coins.fourP + 3 * coins.threeP + 2 * coins.twoP + coins.oneP

/-- Checks if a given amount can be paid using the coin collection -/
def canPay (coins : CoinCollection) (amount : Nat) : Prop :=
  ∃ (a b c d e : Nat),
    a ≤ coins.fiveP ∧
    b ≤ coins.fourP ∧
    c ≤ coins.threeP ∧
    d ≤ coins.twoP ∧
    e ≤ coins.oneP ∧
    5 * a + 4 * b + 3 * c + 2 * d + e = amount

/-- Edward's coin collection -/
def edwardCoins : CoinCollection :=
  { fiveP := 5, fourP := 4, threeP := 3, twoP := 2, oneP := 1 }

theorem smallest_unpayable_amount :
  (∀ n < 56, canPay edwardCoins n) ∧ ¬(canPay edwardCoins 56) := by
  sorry

end NUMINAMATH_CALUDE_smallest_unpayable_amount_l2707_270738


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l2707_270793

theorem quadratic_roots_imply_c_value (c : ℝ) : 
  (∀ x : ℝ, 4 * x^2 + 8 * x + c = 0 ↔ x = (-8 + Real.sqrt 16) / 8 ∨ x = (-8 - Real.sqrt 16) / 8) →
  c = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l2707_270793


namespace NUMINAMATH_CALUDE_sincos_sum_values_l2707_270792

theorem sincos_sum_values (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  Real.sin x + 3 * Real.cos x = 4 ∨ Real.sin x + 3 * Real.cos x = -4 := by
sorry

end NUMINAMATH_CALUDE_sincos_sum_values_l2707_270792


namespace NUMINAMATH_CALUDE_course_selection_theorem_l2707_270706

def type_a_courses : ℕ := 3
def type_b_courses : ℕ := 4
def total_courses_to_choose : ℕ := 3

def ways_to_choose (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_theorem :
  (ways_to_choose type_a_courses 2 * ways_to_choose type_b_courses 1) +
  (ways_to_choose type_a_courses 1 * ways_to_choose type_b_courses 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l2707_270706


namespace NUMINAMATH_CALUDE_roses_cut_l2707_270758

theorem roses_cut (initial_roses final_roses : ℕ) : 
  initial_roses = 6 → final_roses = 16 → final_roses - initial_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l2707_270758


namespace NUMINAMATH_CALUDE_triangle_height_l2707_270783

theorem triangle_height (base area height : ℝ) : 
  base = 4 ∧ area = 16 ∧ area = (base * height) / 2 → height = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_height_l2707_270783


namespace NUMINAMATH_CALUDE_hyperbola_focus_l2707_270705

/-- Given a hyperbola with equation x^2 - ky^2 = 1 and one focus at (3,0), prove that k = 1/8 -/
theorem hyperbola_focus (k : ℝ) : 
  (∀ x y : ℝ, x^2 - k*y^2 = 1 → (∃ c : ℝ, c^2 = 9 ∧ c^2 = 1 + 1/k)) → 
  k = 1/8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l2707_270705


namespace NUMINAMATH_CALUDE_diamond_calculation_l2707_270771

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_calculation : 
  let x := diamond (diamond 2 3) 4
  let y := diamond 2 (diamond 3 4)
  x - y = -29 / 132 := by sorry

end NUMINAMATH_CALUDE_diamond_calculation_l2707_270771


namespace NUMINAMATH_CALUDE_n_equals_four_l2707_270768

theorem n_equals_four (n : ℝ) (h : 3 * n = 6 * 2) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_n_equals_four_l2707_270768


namespace NUMINAMATH_CALUDE_existence_of_xy_for_function_l2707_270745

open Set

theorem existence_of_xy_for_function (f : ℝ → ℝ) 
  (hf : ∀ x, x > 0 → f x > 0) : 
  ∃ x y, x > 0 ∧ y > 0 ∧ f (x + y) < y * f (f x) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_for_function_l2707_270745


namespace NUMINAMATH_CALUDE_optimal_configuration_prevents_loosening_l2707_270752

/-- A rectangular prism box with trapezoid end faces on a cart -/
structure BoxOnCart where
  d : ℝ  -- Distance between parallel sides of trapezoid
  k : ℝ  -- Width of cart
  b : ℝ  -- Height of trapezoid at one end
  c : ℝ  -- Height of trapezoid at other end
  h_k_gt_d : k > d
  h_b_gt_c : b > c

/-- The optimal configuration of the box on the cart -/
def optimal_configuration (box : BoxOnCart) : Prop :=
  let DC₁ := box.c / (box.b - box.c) * (box.k - box.d)
  let AB₁ := box.b / (box.b + box.c) * (box.k - box.d)
  DC₁ > 0 ∧ AB₁ > 0 ∧ box.k ≤ 2 * box.b * box.d / (box.b - box.c)

/-- The theorem stating the optimal configuration prevents the rope from loosening -/
theorem optimal_configuration_prevents_loosening (box : BoxOnCart) :
  optimal_configuration box →
  ∃ (DC₁ AB₁ : ℝ),
    DC₁ = box.c / (box.b - box.c) * (box.k - box.d) ∧
    AB₁ = box.b / (box.b + box.c) * (box.k - box.d) ∧
    DC₁ > 0 ∧ AB₁ > 0 ∧
    box.k ≤ 2 * box.b * box.d / (box.b - box.c) :=
by sorry

end NUMINAMATH_CALUDE_optimal_configuration_prevents_loosening_l2707_270752


namespace NUMINAMATH_CALUDE_audrey_sleep_theorem_l2707_270722

theorem audrey_sleep_theorem (total_sleep : ℝ) (dream_ratio : ℝ) 
  (h1 : total_sleep = 10)
  (h2 : dream_ratio = 2/5) : 
  total_sleep - (dream_ratio * total_sleep) = 6 := by
  sorry

end NUMINAMATH_CALUDE_audrey_sleep_theorem_l2707_270722


namespace NUMINAMATH_CALUDE_some_number_proof_l2707_270787

def total_prime_factors (n : ℕ) : ℕ := sorry

theorem some_number_proof (x : ℕ) :
  total_prime_factors (x * 11^13 * 7^5) = 29 → x = 2^11 := by
  sorry

end NUMINAMATH_CALUDE_some_number_proof_l2707_270787


namespace NUMINAMATH_CALUDE_profit_percentage_is_20_l2707_270712

-- Define the quantities and prices
def wheat1_quantity : ℝ := 30
def wheat1_price : ℝ := 11.50
def wheat2_quantity : ℝ := 20
def wheat2_price : ℝ := 14.25
def selling_price : ℝ := 15.12

-- Define the theorem
theorem profit_percentage_is_20 : 
  let total_cost := wheat1_quantity * wheat1_price + wheat2_quantity * wheat2_price
  let total_weight := wheat1_quantity + wheat2_quantity
  let cost_price_per_kg := total_cost / total_weight
  let profit_per_kg := selling_price - cost_price_per_kg
  let profit_percentage := (profit_per_kg / cost_price_per_kg) * 100
  profit_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_profit_percentage_is_20_l2707_270712


namespace NUMINAMATH_CALUDE_xiaotong_pe_score_l2707_270785

/-- Calculates the physical education score based on extracurricular activities and final exam scores -/
def physical_education_score (extracurricular_score : ℝ) (final_exam_score : ℝ) : ℝ :=
  0.3 * extracurricular_score + 0.7 * final_exam_score

/-- Xiaotong's physical education score theorem -/
theorem xiaotong_pe_score :
  let max_score : ℝ := 100
  let extracurricular_weight : ℝ := 0.3
  let final_exam_weight : ℝ := 0.7
  let xiaotong_extracurricular_score : ℝ := 90
  let xiaotong_final_exam_score : ℝ := 80
  physical_education_score xiaotong_extracurricular_score xiaotong_final_exam_score = 83 :=
by
  sorry

#eval physical_education_score 90 80

end NUMINAMATH_CALUDE_xiaotong_pe_score_l2707_270785


namespace NUMINAMATH_CALUDE_hyperbola_range_l2707_270756

/-- The equation represents a hyperbola with parameter m -/
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (m + 2) + y^2 / (m - 2) = 1

/-- The range of m for which the equation represents a hyperbola -/
theorem hyperbola_range :
  ∀ m : ℝ, is_hyperbola m ↔ m > -2 ∧ m < 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_range_l2707_270756


namespace NUMINAMATH_CALUDE_unique_solution_E_l2707_270735

/-- Definition of the function E --/
def E (a b c : ℝ) : ℝ := a * b^2 + c

/-- Theorem stating that -1/16 is the unique solution to E(a, 3, 2) = E(a, 5, 3) --/
theorem unique_solution_E :
  ∃! a : ℝ, E a 3 2 = E a 5 3 ∧ a = -1/16 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_E_l2707_270735


namespace NUMINAMATH_CALUDE_minimum_width_for_garden_l2707_270750

theorem minimum_width_for_garden (w : ℝ) : w > 0 → w * (w + 10) ≥ 150 → 
  ∀ x > 0, x * (x + 10) ≥ 150 → 2 * (w + w + 10) ≤ 2 * (x + x + 10) → w = 10 := by
  sorry

end NUMINAMATH_CALUDE_minimum_width_for_garden_l2707_270750


namespace NUMINAMATH_CALUDE_fifth_month_sales_l2707_270798

def sales_1 : ℕ := 5124
def sales_2 : ℕ := 5366
def sales_3 : ℕ := 5808
def sales_4 : ℕ := 5399
def sales_6 : ℕ := 4579
def average_sale : ℕ := 5400
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    (sales_1 + sales_2 + sales_3 + sales_4 + sales_5 + sales_6) / num_months = average_sale ∧
    sales_5 = 6124 :=
by sorry

end NUMINAMATH_CALUDE_fifth_month_sales_l2707_270798


namespace NUMINAMATH_CALUDE_ratio_to_percent_l2707_270759

theorem ratio_to_percent (a b : ℕ) (h : a = 6 ∧ b = 3) :
  (a : ℚ) / b * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percent_l2707_270759


namespace NUMINAMATH_CALUDE_sum_of_sequences_l2707_270737

def sequence1 : List ℕ := [3, 13, 23, 33, 43]
def sequence2 : List ℕ := [11, 21, 31, 41, 51]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum) = 270 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l2707_270737


namespace NUMINAMATH_CALUDE_shopkeeper_theft_loss_l2707_270723

theorem shopkeeper_theft_loss (cost_price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : 
  profit_rate = 0.1 →
  loss_rate = 0.12 →
  cost_price > 0 →
  let selling_price := cost_price * (1 + profit_rate)
  let loss_value := selling_price * loss_rate
  let loss_percentage := (loss_value / cost_price) * 100
  loss_percentage = 13.2 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_theft_loss_l2707_270723


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2707_270769

/-- The cost relationship between bananas and apples -/
def banana_apple_ratio : ℚ := 5 / 3

/-- The cost relationship between apples and oranges -/
def apple_orange_ratio : ℚ := 9 / 6

/-- The number of bananas we want to compare -/
def banana_count : ℕ := 30

/-- The theorem stating the equivalence between bananas and oranges -/
theorem banana_orange_equivalence : 
  ∃ (orange_count : ℕ), 
    (banana_count : ℚ) / banana_apple_ratio / apple_orange_ratio = orange_count ∧ 
    orange_count = 12 := by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2707_270769


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2707_270744

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : r - p = 20) :
  (q + r) / 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2707_270744


namespace NUMINAMATH_CALUDE_strictly_increasing_implies_a_geq_one_l2707_270732

/-- A function f(x) = x^3 - 2x^2 + ax + 3 is strictly increasing on the interval [1, 2] -/
def StrictlyIncreasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x < f y

/-- The function f(x) = x^3 - 2x^2 + ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 + a*x + 3

theorem strictly_increasing_implies_a_geq_one (a : ℝ) :
  StrictlyIncreasing (f a) a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_strictly_increasing_implies_a_geq_one_l2707_270732


namespace NUMINAMATH_CALUDE_shared_bikes_theorem_l2707_270788

def a (n : ℕ+) : ℕ :=
  if n ≤ 3 then 5 * n^4 + 15 else 470 - 10 * n

def b (n : ℕ+) : ℕ := n + 5

def S (n : ℕ+) : ℕ := 8800 - 4 * (n - 46)^2

def remaining_bikes (n : ℕ+) : ℕ := 
  (Finset.range n).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩) - 
  (Finset.range n).sum (λ i => b ⟨i + 1, Nat.succ_pos i⟩)

theorem shared_bikes_theorem :
  remaining_bikes 4 = 945 ∧
  remaining_bikes 42 = 8782 ∧
  S 42 = 8736 ∧
  remaining_bikes 42 > S 42 :=
sorry

end NUMINAMATH_CALUDE_shared_bikes_theorem_l2707_270788


namespace NUMINAMATH_CALUDE_valid_assignment_y_equals_x_plus_1_l2707_270779

/-- Represents a variable name in a programming language --/
def Variable : Type := String

/-- Represents an expression in a programming language --/
inductive Expression
| Var : Variable → Expression
| Num : Int → Expression
| Add : Expression → Expression → Expression

/-- Represents an assignment statement in a programming language --/
structure Assignment :=
  (lhs : Variable)
  (rhs : Expression)

/-- Checks if an assignment statement is valid --/
def is_valid_assignment (a : Assignment) : Prop :=
  ∃ (x : Variable), a.rhs = Expression.Add (Expression.Var x) (Expression.Num 1)

/-- The statement "y = x + 1" is a valid assignment --/
theorem valid_assignment_y_equals_x_plus_1 :
  is_valid_assignment { lhs := "y", rhs := Expression.Add (Expression.Var "x") (Expression.Num 1) } :=
by sorry

end NUMINAMATH_CALUDE_valid_assignment_y_equals_x_plus_1_l2707_270779


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l2707_270704

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- Theorem statement
theorem line_perp_parallel_planes 
  (α β : Plane) (a : Line) 
  (h1 : parallel α β) 
  (h2 : perpendicular a β) : 
  perpendicular a α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l2707_270704


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2707_270770

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (30 * π / 180 + α) = 3/5)
  (h2 : 60 * π / 180 < α)
  (h3 : α < 150 * π / 180) :
  Real.cos α = (3 - 4 * Real.sqrt 3) / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2707_270770


namespace NUMINAMATH_CALUDE_equation_solution_l2707_270797

def solution_set : Set (ℤ × ℤ) :=
  {(-2, 4), (-2, 6), (0, 10), (4, -2), (4, 12), (6, -2), (6, 12), (10, 0), (10, 10), (12, 4), (12, 6)}

theorem equation_solution (x y : ℤ) :
  x + y ≠ 0 →
  (((x^2 + y^2) : ℚ) / (x + y : ℚ) = 10) ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2707_270797


namespace NUMINAMATH_CALUDE_square_difference_identity_l2707_270727

theorem square_difference_identity : (15 + 5)^2 - (15^2 + 5^2) = 150 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l2707_270727


namespace NUMINAMATH_CALUDE_sports_club_theorem_l2707_270786

theorem sports_club_theorem (total_members badminton_players tennis_players both_players : ℕ) 
  (h1 : total_members = 80)
  (h2 : badminton_players = 48)
  (h3 : tennis_players = 46)
  (h4 : both_players = 21) :
  total_members - (badminton_players + tennis_players - both_players) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_theorem_l2707_270786


namespace NUMINAMATH_CALUDE_chord_length_on_unit_circle_l2707_270778

/-- The length of the chord intercepted by the line x-y=0 on the circle x^2 + y^2 = 1 is equal to 2 -/
theorem chord_length_on_unit_circle : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x = y}
  let chord := circle ∩ line
  ∃ (a b : ℝ × ℝ), a ∈ chord ∧ b ∈ chord ∧ a ≠ b ∧ Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_on_unit_circle_l2707_270778


namespace NUMINAMATH_CALUDE_zachary_needs_money_l2707_270782

/-- The amount of additional money Zachary needs to buy football equipment -/
def additional_money_needed (football_cost shorts_cost shoes_cost socks_cost bottle_cost : ℝ)
  (shorts_count socks_count : ℕ) (current_money : ℝ) : ℝ :=
  let total_cost := football_cost + shorts_count * shorts_cost + shoes_cost +
                    socks_count * socks_cost + bottle_cost
  total_cost - current_money

/-- Theorem stating the additional money Zachary needs -/
theorem zachary_needs_money :
  additional_money_needed 3.756 2.498 11.856 1.329 7.834 2 4 24.042 = 9.716 := by
  sorry

end NUMINAMATH_CALUDE_zachary_needs_money_l2707_270782


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l2707_270746

/-- The number of ways to arrange 6 rings out of 10 on 5 fingers -/
def ring_arrangements : ℕ := sorry

/-- The number of ways to choose 6 rings out of 10 -/
def choose_rings : ℕ := sorry

/-- The number of ways to order 6 rings -/
def order_rings : ℕ := sorry

/-- The number of ways to distribute 6 rings among 5 fingers -/
def distribute_rings : ℕ := sorry

theorem ring_arrangement_count :
  ring_arrangements = choose_rings * order_rings * distribute_rings ∧
  ring_arrangements = 31752000 :=
sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l2707_270746


namespace NUMINAMATH_CALUDE_road_repaving_l2707_270767

/-- Given that a construction company repaved 4133 inches of road before today
    and 805 inches today, prove that the total length of road repaved is 4938 inches. -/
theorem road_repaving (inches_before : ℕ) (inches_today : ℕ) 
  (h1 : inches_before = 4133) (h2 : inches_today = 805) :
  inches_before + inches_today = 4938 := by
  sorry

end NUMINAMATH_CALUDE_road_repaving_l2707_270767


namespace NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l2707_270728

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 32 = 0
def equation2 (x : ℝ) : Prop := (x + 4)^3 + 64 = 0

-- Theorem for the first equation
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ ∧ equation1 x₂ ∧ x₁ = 4 ∧ x₂ = -4 :=
sorry

-- Theorem for the second equation
theorem equation2_solution :
  ∃ x : ℝ, equation2 x ∧ x = -8 :=
sorry

end NUMINAMATH_CALUDE_equation1_solutions_equation2_solution_l2707_270728


namespace NUMINAMATH_CALUDE_satellite_sensor_ratio_l2707_270731

theorem satellite_sensor_ratio (total_units : Nat) (upgrade_fraction : Rat) : 
  total_units = 24 → 
  upgrade_fraction = 1 / 7 → 
  (∃ (non_upgraded_per_unit total_upgraded : Nat), 
    (non_upgraded_per_unit : Rat) / (total_upgraded : Rat) = 1 / 4) :=
by sorry

end NUMINAMATH_CALUDE_satellite_sensor_ratio_l2707_270731


namespace NUMINAMATH_CALUDE_candy_mixture_price_l2707_270726

/-- Calculates the selling price per pound of a candy mixture -/
theorem candy_mixture_price (total_weight : ℝ) (cheap_weight : ℝ) (cheap_price : ℝ) (expensive_price : ℝ)
  (h1 : total_weight = 80)
  (h2 : cheap_weight = 64)
  (h3 : cheap_price = 2)
  (h4 : expensive_price = 3)
  : (cheap_weight * cheap_price + (total_weight - cheap_weight) * expensive_price) / total_weight = 2.20 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l2707_270726


namespace NUMINAMATH_CALUDE_exists_zero_sequence_l2707_270784

-- Define the operations
def add_one (x : ℚ) : ℚ := x + 1
def neg_reciprocal (x : ℚ) : ℚ := -1 / x

-- Define a sequence of operations
inductive Operation
| AddOne
| NegReciprocal

def apply_operation (op : Operation) (x : ℚ) : ℚ :=
  match op with
  | Operation.AddOne => add_one x
  | Operation.NegReciprocal => neg_reciprocal x

-- Theorem statement
theorem exists_zero_sequence : ∃ (seq : List Operation), 
  let final_value := seq.foldl (λ acc op => apply_operation op acc) 0
  final_value = 0 ∧ seq.length > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_zero_sequence_l2707_270784


namespace NUMINAMATH_CALUDE_cubic_difference_zero_l2707_270709

theorem cubic_difference_zero (a b : ℝ) (h1 : a - b = 1) (h2 : a * b ≠ 0) :
  a^3 - b^3 - a*b - a^2 - b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_difference_zero_l2707_270709


namespace NUMINAMATH_CALUDE_function_value_nonnegative_l2707_270718

theorem function_value_nonnegative (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  let f : ℝ → ℝ := λ x => x^2 - x
  f m ≥ 0 ∨ f n ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_function_value_nonnegative_l2707_270718


namespace NUMINAMATH_CALUDE_opposite_of_fraction_l2707_270757

theorem opposite_of_fraction (n : ℕ) (hn : n ≠ 0) :
  -(1 : ℚ) / n = -(1 / n) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_fraction_l2707_270757


namespace NUMINAMATH_CALUDE_candy_jar_theorem_l2707_270733

/-- Represents the number of candies of each type in a jar -/
structure CandyJar where
  orange : ℕ
  purple : ℕ
  white : ℕ
  green : ℕ
  black : ℕ

/-- The total number of candies in the jar -/
def CandyJar.total (jar : CandyJar) : ℕ :=
  jar.orange + jar.purple + jar.white + jar.green + jar.black

/-- Replaces a third of purple candies with white candies -/
def CandyJar.replacePurpleWithWhite (jar : CandyJar) : CandyJar :=
  { jar with
    purple := jar.purple - (jar.purple / 3)
    white := jar.white + (jar.purple / 3)
  }

theorem candy_jar_theorem (jar : CandyJar) :
  jar.total = 100 ∧
  jar.orange = 40 ∧
  jar.purple = 30 ∧
  jar.white = 20 ∧
  jar.green = 10 ∧
  jar.black = 10 →
  (jar.replacePurpleWithWhite).white = 30 := by
  sorry


end NUMINAMATH_CALUDE_candy_jar_theorem_l2707_270733


namespace NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l2707_270719

/-- Given a > 0 and f(x) = x^3 - ax is monotonically increasing on [1, +∞),
    prove that the range of values for a is (0, 3]. -/
theorem range_of_a_for_monotonic_f (a : ℝ) (f : ℝ → ℝ) :
  a > 0 →
  (∀ x, f x = x^3 - a*x) →
  (∀ x y, 1 ≤ x → x < y → f x < f y) →
  ∃ S, S = Set.Ioo 0 3 ∧ a ∈ S :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_monotonic_f_l2707_270719


namespace NUMINAMATH_CALUDE_draw_with_even_ball_l2707_270753

/-- The number of balls in the bin -/
def total_balls : ℕ := 15

/-- The number of balls to be drawn -/
def drawn_balls : ℕ := 4

/-- The number of odd-numbered balls -/
def odd_balls : ℕ := 8

/-- Calculate the number of ways to draw n balls from m balls in order -/
def ways_to_draw (m n : ℕ) : ℕ :=
  (List.range n).foldl (fun acc i => acc * (m - i)) 1

/-- The main theorem: number of ways to draw 4 balls with at least one even-numbered ball -/
theorem draw_with_even_ball :
  ways_to_draw total_balls drawn_balls - ways_to_draw odd_balls drawn_balls = 31080 := by
  sorry

end NUMINAMATH_CALUDE_draw_with_even_ball_l2707_270753


namespace NUMINAMATH_CALUDE_right_triangle_vector_relation_l2707_270773

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define vectors
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem right_triangle_vector_relation (t : ℝ) (ABC : Triangle) :
  (ABC.C.1 - ABC.A.1 = 2 ∧ ABC.C.2 - ABC.A.2 = 2) →  -- AC = (2, 2)
  (ABC.B.1 - ABC.A.1 = t ∧ ABC.B.2 - ABC.A.2 = 1) →  -- AB = (t, 1)
  dot_product (vector ABC.A ABC.C) (vector ABC.A ABC.B) = 0 →  -- Angle C is 90 degrees
  t = 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_vector_relation_l2707_270773


namespace NUMINAMATH_CALUDE_verify_scenario1_verify_scenario2_prove_gizmo_production_l2707_270774

/-- Represents the time (in hours) for one worker to produce one gizmo -/
def gizmo_time : ℚ := 1/5

/-- Represents the time (in hours) for one worker to produce one gadget -/
def gadget_time : ℚ := 1/5

/-- Verifies that 80 workers in 1 hour produce 160 gizmos and 240 gadgets -/
theorem verify_scenario1 : 
  80 * (1 / gizmo_time) = 160 ∧ 80 * (1 / gadget_time) = 240 := by sorry

/-- Verifies that 100 workers in 3 hours produce 900 gizmos and 600 gadgets -/
theorem verify_scenario2 : 
  100 * (3 / gizmo_time) = 900 ∧ 100 * (3 / gadget_time) = 600 := by sorry

/-- Proves that 70 workers in 5 hours produce 70 gizmos -/
theorem prove_gizmo_production : 
  70 * (5 / gizmo_time) = 70 := by sorry

end NUMINAMATH_CALUDE_verify_scenario1_verify_scenario2_prove_gizmo_production_l2707_270774


namespace NUMINAMATH_CALUDE_ping_pong_rackets_sold_l2707_270799

/-- The number of pairs of ping pong rackets sold -/
def num_pairs : ℕ := 55

/-- The total amount made from selling rackets in dollars -/
def total_amount : ℚ := 539

/-- The average price of a pair of rackets in dollars -/
def avg_price : ℚ := 9.8

/-- Theorem: The number of pairs of ping pong rackets sold is 55 -/
theorem ping_pong_rackets_sold :
  (total_amount / avg_price : ℚ) = num_pairs := by sorry

end NUMINAMATH_CALUDE_ping_pong_rackets_sold_l2707_270799


namespace NUMINAMATH_CALUDE_half_square_area_l2707_270751

theorem half_square_area (square_area : Real) (h1 : square_area = 100) :
  square_area / 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_half_square_area_l2707_270751


namespace NUMINAMATH_CALUDE_gcd_problem_l2707_270714

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 3883) :
  Int.gcd (4 * b^2 + 35 * b + 56) (3 * b + 8) = 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l2707_270714


namespace NUMINAMATH_CALUDE_exists_function_with_properties_l2707_270762

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the properties of the function
def HasFunctionalEquation (f : RealFunction) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂

def HasNegativeDerivative (f : RealFunction) : Prop :=
  ∀ x : ℝ, x > 0 → (deriv f x) < 0

-- State the theorem
theorem exists_function_with_properties :
  ∃ f : RealFunction,
    HasFunctionalEquation f ∧ HasNegativeDerivative f := by
  sorry

end NUMINAMATH_CALUDE_exists_function_with_properties_l2707_270762


namespace NUMINAMATH_CALUDE_invisible_square_exists_l2707_270755

/-- A lattice point is invisible if the segment from the origin to that point contains another lattice point. -/
def invisible (x y : ℤ) : Prop :=
  ∃ k : ℤ, 1 < k ∧ k < max x.natAbs y.natAbs ∧ (k ∣ x) ∧ (k ∣ y)

/-- For any positive integer L, there exists a square with side length L where all lattice points are invisible. -/
theorem invisible_square_exists (L : ℕ) (hL : 0 < L) :
  ∃ x y : ℤ, ∀ i j : ℕ, i ≤ L → j ≤ L → invisible (x + i) (y + j) := by
  sorry

end NUMINAMATH_CALUDE_invisible_square_exists_l2707_270755


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2707_270794

theorem unique_solution_for_equation : 
  ∃! (n m : ℕ), (n - 1) * 2^(n - 1) + 5 = m^2 + 4*m ∧ n = 6 ∧ m = 11 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2707_270794


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_14_l2707_270777

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 3

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_number_with_digit_sum_14 :
  ∀ n : ℕ, 
    is_valid_number n → 
    digit_sum n = 14 → 
    n ≤ 3222233 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_14_l2707_270777


namespace NUMINAMATH_CALUDE_regression_unit_increase_l2707_270760

/-- Represents a simple linear regression model -/
structure LinearRegression where
  intercept : ℝ
  slope : ℝ

/-- The predicted value for a given x in a linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.intercept + model.slope * x

/-- Theorem: In the given linear regression model, when x increases by 1, y increases by 3 -/
theorem regression_unit_increase (model : LinearRegression) (x : ℝ) 
    (h : model = { intercept := 2, slope := 3 }) :
    predict model (x + 1) - predict model x = 3 := by
  sorry

end NUMINAMATH_CALUDE_regression_unit_increase_l2707_270760


namespace NUMINAMATH_CALUDE_don_bottles_from_shop_c_l2707_270764

/-- The number of bottles Don can buy in total -/
def total_bottles : ℕ := 550

/-- The number of bottles Don buys from Shop A -/
def shop_a_bottles : ℕ := 150

/-- The number of bottles Don buys from Shop B -/
def shop_b_bottles : ℕ := 180

/-- The number of bottles Don buys from Shop C -/
def shop_c_bottles : ℕ := total_bottles - (shop_a_bottles + shop_b_bottles)

theorem don_bottles_from_shop_c : 
  shop_c_bottles = 550 - (150 + 180) := by sorry

end NUMINAMATH_CALUDE_don_bottles_from_shop_c_l2707_270764


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2707_270772

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2707_270772


namespace NUMINAMATH_CALUDE_square_side_increase_l2707_270789

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let side_b := 2 * a
  let area_a := a^2
  let area_b := side_b^2
  let area_c := (area_a + area_b) * 2.45
  let side_c := Real.sqrt area_c
  (side_c - side_b) / side_b = 0.75 := by sorry

end NUMINAMATH_CALUDE_square_side_increase_l2707_270789


namespace NUMINAMATH_CALUDE_nested_square_root_value_l2707_270716

theorem nested_square_root_value (y : ℝ) :
  y = Real.sqrt (2 + y) → y = 2 := by sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l2707_270716


namespace NUMINAMATH_CALUDE_matching_socks_probability_l2707_270741

/-- The number of pairs of socks -/
def num_pairs : ℕ := 5

/-- The total number of socks -/
def total_socks : ℕ := 2 * num_pairs

/-- The number of socks selected each day -/
def socks_per_day : ℕ := 2

/-- The probability of selecting matching socks for the first time on Wednesday -/
def prob_match_wednesday : ℚ :=
  26 / 315

theorem matching_socks_probability :
  let monday_selections := Nat.choose total_socks socks_per_day
  let tuesday_selections := Nat.choose (total_socks - socks_per_day) socks_per_day
  let wednesday_selections := Nat.choose (total_socks - 2 * socks_per_day) socks_per_day
  prob_match_wednesday =
    (monday_selections - num_pairs) / monday_selections *
    ((1 / tuesday_selections * 1 / 5) +
     (12 / tuesday_selections * 2 / 15) +
     (12 / tuesday_selections * 1 / 15)) :=
by sorry

#eval prob_match_wednesday

end NUMINAMATH_CALUDE_matching_socks_probability_l2707_270741


namespace NUMINAMATH_CALUDE_sum_of_specific_S_l2707_270702

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

theorem sum_of_specific_S : S 15 + S 28 + S 39 = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_S_l2707_270702


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2707_270742

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 4 * x

/-- The point of tangency -/
def P : ℝ × ℝ := (-1, 3)

/-- The slope of the tangent line at point P -/
def m : ℝ := f' P.1

/-- The equation of the tangent line -/
def tangent_line (x : ℝ) : ℝ := m * (x - P.1) + P.2

theorem tangent_line_equation :
  ∀ x : ℝ, tangent_line x = -4 * x - 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2707_270742


namespace NUMINAMATH_CALUDE_total_mangoes_l2707_270763

/-- The number of mangoes each person has -/
structure MangoDistribution where
  alexis : ℝ
  dilan : ℝ
  ashley : ℝ
  ben : ℝ

/-- The conditions of the mango distribution problem -/
def mango_conditions (m : MangoDistribution) : Prop :=
  m.alexis = 4 * (m.dilan + m.ashley) ∧
  m.ashley = 2 * m.dilan ∧
  m.alexis = 60 ∧
  m.ben = (m.ashley + m.dilan) / 2

/-- The theorem stating the total number of mangoes -/
theorem total_mangoes (m : MangoDistribution) 
  (h : mango_conditions m) : 
  m.alexis + m.dilan + m.ashley + m.ben = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_total_mangoes_l2707_270763


namespace NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l2707_270781

theorem sum_reciprocals_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 1 / y ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_lower_bound_l2707_270781
