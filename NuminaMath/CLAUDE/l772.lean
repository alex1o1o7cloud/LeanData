import Mathlib

namespace NUMINAMATH_CALUDE_find_x_value_l772_77287

theorem find_x_value (x : ℝ) : 3639 + 11.95 - x = 3054 → x = 596.95 := by
  sorry

end NUMINAMATH_CALUDE_find_x_value_l772_77287


namespace NUMINAMATH_CALUDE_quadratic_point_distance_l772_77227

/-- Given a quadratic function f(x) = ax² - 2ax + b where a > 0,
    and two points (x₁, y₁) and (x₂, y₂) on its graph where y₁ > y₂,
    prove that |x₁ - 1| > |x₂ - 1| -/
theorem quadratic_point_distance (a b x₁ y₁ x₂ y₂ : ℝ) 
  (ha : a > 0)
  (hf₁ : y₁ = a * x₁^2 - 2 * a * x₁ + b)
  (hf₂ : y₂ = a * x₂^2 - 2 * a * x₂ + b)
  (hy : y₁ > y₂) :
  |x₁ - 1| > |x₂ - 1| := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_distance_l772_77227


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l772_77288

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

-- Main theorem
theorem fixed_point_on_line (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  (x₁ ≠ x₂ ∨ y₁ ≠ y₂) →
  perpendicular x₁ y₁ x₂ y₂ →
  ∃ m : ℝ, x₁ = m*y₁ + 8 ∧ x₂ = m*y₂ + 8 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l772_77288


namespace NUMINAMATH_CALUDE_exists_unique_function_satisfying_equation_l772_77253

/-- A functional equation that uniquely determines a function f: ℝ → ℤ --/
def functional_equation (f : ℝ → ℤ) (x₁ x₂ : ℝ) : Prop :=
  0 = (f (-x₁^2 - (x₁ * x₂ - 1)^2))^2 +
      ((f (-x₁^2 - (x₁ * x₂ - 1)^2 + 1) - 1/2)^2 - 1/4)^2 +
      (f (x₁^2 + 2) - 2 * f (x₁^2) + f (x₁^2 - 2))^2 +
      ((f (x₁^2) - f (x₁^2 - 2))^2 - 1)^2 +
      ((f (x₁^2) + f (x₁^2 + 1) - 1/2)^2 - 1/4)^2

/-- The theorem stating the existence of a unique function satisfying the functional equation --/
theorem exists_unique_function_satisfying_equation :
  ∃! f : ℝ → ℤ, (∀ x₁ x₂ : ℝ, functional_equation f x₁ x₂) ∧ Set.range f = Set.univ :=
sorry

end NUMINAMATH_CALUDE_exists_unique_function_satisfying_equation_l772_77253


namespace NUMINAMATH_CALUDE_katie_miles_ran_l772_77254

theorem katie_miles_ran (katie_miles : ℝ) (adam_miles : ℝ) : 
  adam_miles = 3 * katie_miles →
  katie_miles + adam_miles = 240 →
  katie_miles = 60 := by
sorry

end NUMINAMATH_CALUDE_katie_miles_ran_l772_77254


namespace NUMINAMATH_CALUDE_inverse_composition_result_l772_77220

-- Define the function f
def f : Fin 5 → Fin 5
| 1 => 3
| 2 => 5
| 3 => 1
| 4 => 2
| 5 => 4

-- Define the inverse function f⁻¹
def f_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 4
| 3 => 1
| 4 => 5
| 5 => 2

-- State the theorem
theorem inverse_composition_result :
  f_inv (f_inv (f_inv 5)) = 5 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_result_l772_77220


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l772_77242

theorem arithmetic_evaluation : 8 * (6 - 4) + 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l772_77242


namespace NUMINAMATH_CALUDE_resort_tips_fraction_l772_77213

theorem resort_tips_fraction (avg_tips : ℝ) (h : avg_tips > 0) :
  let other_months_tips := 6 * avg_tips
  let august_tips := 6 * avg_tips
  let total_tips := other_months_tips + august_tips
  august_tips / total_tips = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_resort_tips_fraction_l772_77213


namespace NUMINAMATH_CALUDE_unqualified_weight_l772_77244

def flour_label_center : ℝ := 25
def flour_label_tolerance : ℝ := 0.25

def is_qualified (weight : ℝ) : Prop :=
  flour_label_center - flour_label_tolerance ≤ weight ∧ 
  weight ≤ flour_label_center + flour_label_tolerance

theorem unqualified_weight : ¬ (is_qualified 25.26) := by
  sorry

end NUMINAMATH_CALUDE_unqualified_weight_l772_77244


namespace NUMINAMATH_CALUDE_recreational_space_perimeter_l772_77289

-- Define the playground and sandbox dimensions
def playground_width : ℕ := 20
def playground_height : ℕ := 16
def sandbox_width : ℕ := 4
def sandbox_height : ℕ := 3

-- Define the sandbox position
def sandbox_top_distance : ℕ := 6
def sandbox_left_distance : ℕ := 8

-- Define the perimeter calculation function
def calculate_perimeter (playground_width playground_height sandbox_width sandbox_height sandbox_top_distance sandbox_left_distance : ℕ) : ℕ :=
  let right_width := playground_width - sandbox_left_distance - sandbox_width
  let bottom_height := playground_height - sandbox_top_distance - sandbox_height
  let right_perimeter := 2 * (playground_height + right_width)
  let bottom_perimeter := 2 * (bottom_height + sandbox_left_distance)
  let left_perimeter := 2 * (sandbox_top_distance + sandbox_left_distance)
  let overlap := 4 * sandbox_left_distance
  right_perimeter + bottom_perimeter + left_perimeter - overlap

-- Theorem statement
theorem recreational_space_perimeter :
  calculate_perimeter playground_width playground_height sandbox_width sandbox_height sandbox_top_distance sandbox_left_distance = 74 := by
  sorry

end NUMINAMATH_CALUDE_recreational_space_perimeter_l772_77289


namespace NUMINAMATH_CALUDE_division_problem_l772_77245

theorem division_problem (N x : ℕ) : 
  (N / x = 500) → 
  (N % x = 20) → 
  (4 * 500 + 20 = 2020) → 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l772_77245


namespace NUMINAMATH_CALUDE_solve_stick_problem_l772_77264

def stick_problem (dave_sticks amy_sticks ben_sticks total_sticks : ℕ) : Prop :=
  let total_picked := dave_sticks + amy_sticks + ben_sticks
  let sticks_left := total_sticks - total_picked
  total_picked - sticks_left = 5

theorem solve_stick_problem :
  stick_problem 14 9 12 65 := by
  sorry

end NUMINAMATH_CALUDE_solve_stick_problem_l772_77264


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l772_77221

theorem consecutive_non_prime_powers (n : ℕ) : 
  ∃ x : ℤ, ∀ k ∈ Finset.range n, ¬ ∃ (p : ℕ) (m : ℕ), Prime p ∧ x + k.succ = p ^ m := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l772_77221


namespace NUMINAMATH_CALUDE_combined_degrees_sum_l772_77279

/-- The combined number of degrees for Summer and Jolly -/
def combined_degrees (summer_degrees : ℕ) (difference : ℕ) : ℕ :=
  summer_degrees + (summer_degrees - difference)

/-- Theorem stating that given Summer has 150 degrees and 5 more degrees than Jolly,
    the combined number of degrees for Summer and Jolly is 295 -/
theorem combined_degrees_sum (summer_degrees : ℕ) (difference : ℕ)
  (h1 : summer_degrees = 150)
  (h2 : difference = 5) :
  combined_degrees summer_degrees difference = 295 := by
sorry

end NUMINAMATH_CALUDE_combined_degrees_sum_l772_77279


namespace NUMINAMATH_CALUDE_total_donation_theorem_l772_77298

def initial_donation : ℝ := 1707

def percentage_increases : List ℝ := [0.03, 0.05, 0.08, 0.02, 0.10, 0.04, 0.06, 0.09, 0.07, 0.03, 0.05]

def calculate_monthly_donation (prev_donation : ℝ) (percentage_increase : ℝ) : ℝ :=
  prev_donation * (1 + percentage_increase)

def calculate_total_donation (initial : ℝ) (increases : List ℝ) : ℝ :=
  let monthly_donations := increases.scanl calculate_monthly_donation initial
  initial + monthly_donations.sum

theorem total_donation_theorem :
  calculate_total_donation initial_donation percentage_increases = 29906.10 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_theorem_l772_77298


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l772_77272

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = -1) :
  (x - 3*y)^2 - (x - y)*(x + 2*y) = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l772_77272


namespace NUMINAMATH_CALUDE_bacteria_growth_time_l772_77297

/-- The growth factor of bacteria population in one tripling period -/
def tripling_factor : ℕ := 3

/-- The duration in hours of one tripling period -/
def hours_per_tripling : ℕ := 3

/-- The initial number of bacteria -/
def initial_bacteria : ℕ := 300

/-- The final number of bacteria -/
def final_bacteria : ℕ := 72900

/-- The time in hours for bacteria to grow from initial to final count -/
def growth_time : ℕ := 15

theorem bacteria_growth_time :
  (tripling_factor ^ (growth_time / hours_per_tripling)) * initial_bacteria = final_bacteria :=
sorry

end NUMINAMATH_CALUDE_bacteria_growth_time_l772_77297


namespace NUMINAMATH_CALUDE_paul_pencil_production_l772_77210

/-- Calculates the number of pencils made per day given the initial stock, 
    final stock, number of pencils sold, and number of working days. -/
def pencils_per_day (initial_stock final_stock pencils_sold working_days : ℕ) : ℕ :=
  ((final_stock + pencils_sold) - initial_stock) / working_days

/-- Proves that Paul makes 100 pencils per day given the problem conditions. -/
theorem paul_pencil_production : 
  pencils_per_day 80 230 350 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_paul_pencil_production_l772_77210


namespace NUMINAMATH_CALUDE_probability_three_same_color_l772_77222

/-- Represents a person in the block placement scenario -/
structure Person where
  name : String
  blocks : Fin 5 → Color

/-- Represents the colors of the blocks -/
inductive Color
  | Red
  | Blue
  | Yellow
  | White
  | Green

/-- Represents the result of a single trial -/
structure Trial where
  placements : Person → Fin 6 → Option (Fin 5)

/-- The probability of a specific event occurring in the trial -/
def probability (event : Trial → Prop) : ℚ :=
  sorry

/-- Checks if a trial results in at least one box with 3 blocks of the same color -/
def has_three_same_color (t : Trial) : Prop :=
  sorry

/-- The main theorem stating the probability of the event -/
theorem probability_three_same_color 
  (ang ben jasmin : Person)
  (h1 : ang.name = "Ang" ∧ ben.name = "Ben" ∧ jasmin.name = "Jasmin")
  (h2 : ∀ p : Person, p = ang ∨ p = ben ∨ p = jasmin → 
        ∀ i : Fin 5, ∃! c : Color, p.blocks i = c) :
  probability has_three_same_color = 5 / 216 :=
sorry

end NUMINAMATH_CALUDE_probability_three_same_color_l772_77222


namespace NUMINAMATH_CALUDE_equation_solution_iff_m_equals_p_l772_77278

theorem equation_solution_iff_m_equals_p (p m : ℕ) (hp : Prime p) (hm : m ≥ 2) :
  (∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ (x, y) ≠ (1, 1) ∧
    (x^p + y^p) / 2 = ((x + y) / 2)^m) ↔ m = p :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_iff_m_equals_p_l772_77278


namespace NUMINAMATH_CALUDE_ellipse_intersection_slope_sum_l772_77258

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Define the line that intersects the ellipse
def intersecting_line (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 3

-- Define the condition for the sum of slopes
def slope_sum_condition (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    intersecting_line k x₁ y₁ ∧ intersecting_line k x₂ y₂ ∧
    (y₁ - 1) / x₁ + (y₂ - 1) / x₂ = 1

theorem ellipse_intersection_slope_sum (k : ℝ) :
  slope_sum_condition k → k = 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_slope_sum_l772_77258


namespace NUMINAMATH_CALUDE_trigonometric_problem_l772_77294

theorem trigonometric_problem (x y : ℝ) (h_nonzero : x ≠ 0 ∧ y ≠ 0) 
  (h_eq : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  (y / x = 1) ∧
  (∀ A B : ℝ, 0 < A ∧ 0 < B ∧ A + B = 3*π/4 → 
    Real.sin (2*A) + 2 * Real.cos B ≤ 3/2) ∧
  (∃ A B : ℝ, 0 < A ∧ 0 < B ∧ A + B = 3*π/4 ∧ 
    Real.sin (2*A) + 2 * Real.cos B = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l772_77294


namespace NUMINAMATH_CALUDE_value_of_x_l772_77216

theorem value_of_x (x y z : ℚ) : 
  x = (1 / 3) * y → 
  y = (1 / 4) * z → 
  z = 96 → 
  x = 8 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l772_77216


namespace NUMINAMATH_CALUDE_cut_cube_edges_l772_77255

/-- Represents a cube with cut corners -/
structure CutCube where
  originalEdges : Nat
  vertices : Nat
  cutsPerVertex : Nat
  newFacesPerCut : Nat
  newEdgesPerFace : Nat

/-- The number of edges in a cube with cut corners -/
def edgesAfterCut (c : CutCube) : Nat :=
  c.originalEdges + c.vertices * c.cutsPerVertex * c.newEdgesPerFace / 2

/-- Theorem stating that a cube with cut corners has 36 edges -/
theorem cut_cube_edges :
  ∀ c : CutCube,
  c.originalEdges = 12 ∧
  c.vertices = 8 ∧
  c.cutsPerVertex = 1 ∧
  c.newFacesPerCut = 1 ∧
  c.newEdgesPerFace = 4 →
  edgesAfterCut c = 36 := by
  sorry

#check cut_cube_edges

end NUMINAMATH_CALUDE_cut_cube_edges_l772_77255


namespace NUMINAMATH_CALUDE_hyperbola_equation_l772_77211

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and right focus at (c, 0),
    if a circle with radius 4 centered at the right focus passes through
    the origin and the point (a, b) on the asymptote, then a² = 4 and b² = 12 -/
theorem hyperbola_equation (a b c : ℝ) (h1 : c > 0) (h2 : a > 0) (h3 : b > 0) :
  (c = 4) →
  ((a - c)^2 + b^2 = 16) →
  (a^2 + b^2 = c^2) →
  (a^2 = 4 ∧ b^2 = 12) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l772_77211


namespace NUMINAMATH_CALUDE_negative_sqrt_16_l772_77256

theorem negative_sqrt_16 : -Real.sqrt 16 = -4 := by sorry

end NUMINAMATH_CALUDE_negative_sqrt_16_l772_77256


namespace NUMINAMATH_CALUDE_license_plate_palindrome_probability_l772_77291

/-- The probability of a license plate containing at least one palindrome -/
theorem license_plate_palindrome_probability :
  let num_letters : ℕ := 26
  let num_digits : ℕ := 10
  let total_arrangements : ℕ := num_letters^4 * num_digits^4
  let letter_palindromes : ℕ := num_letters^2
  let digit_palindromes : ℕ := num_digits^2
  let prob_letter_palindrome : ℚ := letter_palindromes / (num_letters^4 : ℚ)
  let prob_digit_palindrome : ℚ := digit_palindromes / (num_digits^4 : ℚ)
  let prob_both_palindromes : ℚ := (letter_palindromes * digit_palindromes) / (total_arrangements : ℚ)
  let prob_at_least_one_palindrome : ℚ := prob_letter_palindrome + prob_digit_palindrome - prob_both_palindromes
  prob_at_least_one_palindrome = 131 / 1142 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_probability_l772_77291


namespace NUMINAMATH_CALUDE_A_inter_B_empty_A_union_B_complement_A_inter_complement_B_empty_complement_A_union_complement_B_eq_U_l772_77249

-- Define the universal set U
def U : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set ℝ := {x | -5 ≤ x ∧ x < -1}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem for the intersection of A and B
theorem A_inter_B_empty : A ∩ B = ∅ := by sorry

-- Theorem for the union of A and B
theorem A_union_B : A ∪ B = {x | -5 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for the intersection of complements of A and B
theorem complement_A_inter_complement_B_empty : (U \ A) ∩ (U \ B) = ∅ := by sorry

-- Theorem for the union of complements of A and B
theorem complement_A_union_complement_B_eq_U : (U \ A) ∪ (U \ B) = U := by sorry

end NUMINAMATH_CALUDE_A_inter_B_empty_A_union_B_complement_A_inter_complement_B_empty_complement_A_union_complement_B_eq_U_l772_77249


namespace NUMINAMATH_CALUDE_constants_are_like_terms_l772_77202

/-- Two terms are considered like terms if they have the same variables raised to the same powers, or if they are both constants. -/
def like_terms (t1 t2 : ℝ) : Prop :=
  (∃ (c1 c2 : ℝ), t1 = c1 ∧ t2 = c2) ∨ 
  (∃ (c1 c2 : ℝ) (f : ℝ → ℝ), t1 = c1 * f 0 ∧ t2 = c2 * f 0)

/-- Constants 0 and π are like terms. -/
theorem constants_are_like_terms : like_terms 0 π := by
  sorry

end NUMINAMATH_CALUDE_constants_are_like_terms_l772_77202


namespace NUMINAMATH_CALUDE_cheaper_plan_threshold_min_gigabytes_for_cheaper_plan_y_l772_77206

/-- Represents the cost of an internet plan in cents -/
def PlanCost (initialFee : ℕ) (costPerGB : ℕ) (gigabytes : ℕ) : ℕ :=
  initialFee * 100 + costPerGB * gigabytes

theorem cheaper_plan_threshold :
  ∀ g : ℕ, PlanCost 0 20 g ≤ PlanCost 30 10 g ↔ g ≤ 300 :=
by sorry

theorem min_gigabytes_for_cheaper_plan_y :
  ∃ g : ℕ, g = 301 ∧
    (∀ h : ℕ, PlanCost 0 20 h > PlanCost 30 10 h → h ≥ g) ∧
    PlanCost 0 20 g > PlanCost 30 10 g :=
by sorry

end NUMINAMATH_CALUDE_cheaper_plan_threshold_min_gigabytes_for_cheaper_plan_y_l772_77206


namespace NUMINAMATH_CALUDE_computer_price_increase_l772_77261

theorem computer_price_increase (d : ℝ) (h1 : 2 * d = 540) : 
  d * (1 + 0.3) = 351 := by sorry

end NUMINAMATH_CALUDE_computer_price_increase_l772_77261


namespace NUMINAMATH_CALUDE_sqrt_of_negative_eight_squared_l772_77282

theorem sqrt_of_negative_eight_squared : Real.sqrt ((-8)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_of_negative_eight_squared_l772_77282


namespace NUMINAMATH_CALUDE_odd_integers_between_fractions_l772_77203

theorem odd_integers_between_fractions : 
  let lower_bound : ℚ := 17 / 4
  let upper_bound : ℚ := 35 / 2
  ∃ (S : Finset ℤ), (∀ n ∈ S, (n : ℚ) > lower_bound ∧ (n : ℚ) < upper_bound ∧ Odd n) ∧ 
                    (∀ n : ℤ, (n : ℚ) > lower_bound ∧ (n : ℚ) < upper_bound ∧ Odd n → n ∈ S) ∧
                    Finset.card S = 7 :=
by sorry

end NUMINAMATH_CALUDE_odd_integers_between_fractions_l772_77203


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_9_4_l772_77200

theorem smallest_five_digit_mod_9_4 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≡ 4 [ZMOD 9] → 
    10003 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_9_4_l772_77200


namespace NUMINAMATH_CALUDE_product_expansion_l772_77293

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * ((8 / x) + 12 * x^3 - (2 / x^2)) = (6 / x) + 9 * x^3 - (3 / (2 * x^2)) := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l772_77293


namespace NUMINAMATH_CALUDE_system_solution_l772_77207

theorem system_solution : ∃! (x y : ℝ), x - y = 2 ∧ 2*x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l772_77207


namespace NUMINAMATH_CALUDE_sister_bars_count_l772_77217

/-- Represents the number of granola bars in a pack --/
def pack_size : ℕ := 20

/-- Represents the number of days in a week --/
def days_in_week : ℕ := 7

/-- Represents the number of bars traded to Pete --/
def bars_traded : ℕ := 3

/-- Represents the number of sisters --/
def num_sisters : ℕ := 2

/-- Calculates the number of granola bars each sister receives --/
def bars_per_sister : ℕ := (pack_size - days_in_week - bars_traded) / num_sisters

/-- Proves that each sister receives 5 granola bars --/
theorem sister_bars_count : bars_per_sister = 5 := by
  sorry

#eval bars_per_sister  -- This will evaluate to 5

end NUMINAMATH_CALUDE_sister_bars_count_l772_77217


namespace NUMINAMATH_CALUDE_pirate_count_correct_l772_77252

/-- The number of pirates on the schooner satisfying the given conditions -/
def pirate_count : ℕ := 60

/-- The fraction of pirates who lost a leg -/
def leg_loss_fraction : ℚ := 2/3

/-- The fraction of fight participants who lost an arm -/
def arm_loss_fraction : ℚ := 54/100

/-- The fraction of fight participants who lost both an arm and a leg -/
def both_loss_fraction : ℚ := 34/100

/-- The number of pirates who did not participate in the fight -/
def non_participants : ℕ := 10

theorem pirate_count_correct : 
  ∃ (p : ℕ), p = pirate_count ∧ 
  (leg_loss_fraction : ℚ) * p = (p - non_participants) * both_loss_fraction + 
    ((p - non_participants) * arm_loss_fraction - (p - non_participants) * both_loss_fraction) +
    (leg_loss_fraction * p - (p - non_participants) * both_loss_fraction) :=
sorry

end NUMINAMATH_CALUDE_pirate_count_correct_l772_77252


namespace NUMINAMATH_CALUDE_vessel_mixture_problem_l772_77277

theorem vessel_mixture_problem (x : ℝ) : 
  (0 < x) ∧ (x < 8) →
  (((8 * 0.16 - (8 * 0.16) * (x / 8)) - ((8 * 0.16 - (8 * 0.16) * (x / 8)) * (x / 8))) / 8 = 0.09) →
  x = 2 := by sorry

end NUMINAMATH_CALUDE_vessel_mixture_problem_l772_77277


namespace NUMINAMATH_CALUDE_no_real_solutions_for_complex_norm_equation_l772_77243

theorem no_real_solutions_for_complex_norm_equation :
  ¬∃ c : ℝ, Complex.abs (1 + c - 3*I) = 2 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_complex_norm_equation_l772_77243


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l772_77219

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∀ x, (1 - a) * x > 3 ↔ x < 3 / (1 - a)) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l772_77219


namespace NUMINAMATH_CALUDE_volume_of_region_l772_77276

-- Define the region in space
def Region : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x + y + 2*z| + |x + y - 2*z| ≤ 12) ∧
                   (x ≥ 0) ∧ (y ≥ 0) ∧ (z ≥ 0)}

-- State the theorem
theorem volume_of_region : MeasureTheory.volume Region = 54 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_region_l772_77276


namespace NUMINAMATH_CALUDE_tan_addition_special_case_l772_77209

theorem tan_addition_special_case (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π/3) = (12 * Real.sqrt 3 + 3) / 26 := by
  sorry

end NUMINAMATH_CALUDE_tan_addition_special_case_l772_77209


namespace NUMINAMATH_CALUDE_sin_double_angle_problem_l772_77267

theorem sin_double_angle_problem (x : ℝ) (h : Real.sin (x - π/4) = 3/5) : 
  Real.sin (2*x) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_problem_l772_77267


namespace NUMINAMATH_CALUDE_perfect_square_in_base_n_l772_77248

theorem perfect_square_in_base_n (n : ℕ) (hn : n ≥ 2) :
  ∃ m : ℕ, m^2 = n^4 + n^3 + n^2 + n + 1 ↔ n = 3 := by sorry

end NUMINAMATH_CALUDE_perfect_square_in_base_n_l772_77248


namespace NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l772_77280

def A : Set ℝ := {0, 4}
def B (a : ℝ) : Set ℝ := {2, a^2}

theorem a_equals_2_sufficient_not_necessary :
  (∀ a : ℝ, a = 2 → A ∩ B a = {4}) ∧
  (∃ a : ℝ, a ≠ 2 ∧ A ∩ B a = {4}) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_2_sufficient_not_necessary_l772_77280


namespace NUMINAMATH_CALUDE_base7_divisibility_l772_77233

/-- Converts a base 7 number of the form 25y3₇ to base 10 -/
def base7ToBase10 (y : ℕ) : ℕ := 2 * 7^3 + 5 * 7^2 + y * 7 + 3

/-- Checks if a number is divisible by 19 -/
def isDivisibleBy19 (n : ℕ) : Prop := ∃ k : ℕ, n = 19 * k

theorem base7_divisibility (y : ℕ) : 
  y < 7 → (isDivisibleBy19 (base7ToBase10 y) ↔ y = 3) := by sorry

end NUMINAMATH_CALUDE_base7_divisibility_l772_77233


namespace NUMINAMATH_CALUDE_omitted_angle_measure_l772_77237

theorem omitted_angle_measure (n : ℕ) (sum_without_one : ℝ) : 
  n ≥ 3 → 
  sum_without_one = 1958 → 
  (n - 2) * 180 - sum_without_one = 22 :=
by sorry

end NUMINAMATH_CALUDE_omitted_angle_measure_l772_77237


namespace NUMINAMATH_CALUDE_fraction_equality_l772_77215

theorem fraction_equality : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l772_77215


namespace NUMINAMATH_CALUDE_math_team_combinations_l772_77260

theorem math_team_combinations (girls : ℕ) (boys : ℕ) : 
  girls = 3 → boys = 5 → (girls.choose 2) * (boys.choose 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_math_team_combinations_l772_77260


namespace NUMINAMATH_CALUDE_tan_negative_405_degrees_l772_77292

theorem tan_negative_405_degrees : Real.tan ((-405 : ℝ) * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_negative_405_degrees_l772_77292


namespace NUMINAMATH_CALUDE_max_ballpoint_pens_l772_77204

/-- Represents the number of pens of each type -/
structure PenCounts where
  ballpoint : ℕ
  gel : ℕ
  fountain : ℕ

/-- Checks if the given pen counts satisfy all conditions -/
def satisfiesConditions (counts : PenCounts) : Prop :=
  counts.ballpoint + counts.gel + counts.fountain = 15 ∧
  counts.ballpoint ≥ 1 ∧ counts.gel ≥ 1 ∧ counts.fountain ≥ 1 ∧
  10 * counts.ballpoint + 40 * counts.gel + 60 * counts.fountain = 500

/-- Theorem stating that the maximum number of ballpoint pens is 6 -/
theorem max_ballpoint_pens : 
  (∃ counts : PenCounts, satisfiesConditions counts) →
  (∀ counts : PenCounts, satisfiesConditions counts → counts.ballpoint ≤ 6) ∧
  (∃ counts : PenCounts, satisfiesConditions counts ∧ counts.ballpoint = 6) :=
by sorry

end NUMINAMATH_CALUDE_max_ballpoint_pens_l772_77204


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l772_77232

theorem arithmetic_calculation : (-0.5) - (-3.2) + 2.8 - 6.5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l772_77232


namespace NUMINAMATH_CALUDE_equation_solution_l772_77259

theorem equation_solution :
  ∃ x : ℚ, (5 * x - 3) / (6 * x - 6) = 4 / 3 ∧ x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l772_77259


namespace NUMINAMATH_CALUDE_lucy_fish_purchase_l772_77223

/-- The number of fish Lucy needs to buy to reach her desired total -/
def fish_to_buy (initial : ℕ) (desired : ℕ) : ℕ := desired - initial

/-- Theorem: Lucy needs to buy 68 fish -/
theorem lucy_fish_purchase : fish_to_buy 212 280 = 68 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_purchase_l772_77223


namespace NUMINAMATH_CALUDE_same_color_probability_l772_77290

theorem same_color_probability (total_balls : ℕ) (green_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = green_balls + white_balls)
  (h2 : green_balls = 5)
  (h3 : white_balls = 8) :
  (green_balls * (green_balls - 1) + white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1)) = 19 / 39 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l772_77290


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l772_77214

theorem unique_k_for_prime_roots : ∃! k : ℕ, 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p + q = 73) ∧ (p * q = k) ∧ 
  ∀ x : ℝ, x^2 - 73*x + k = 0 ↔ (x = p ∨ x = q) := by
  sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l772_77214


namespace NUMINAMATH_CALUDE_hotel_expenditure_l772_77246

/-- The total expenditure of 9 persons in a hotel, given specific spending conditions. -/
theorem hotel_expenditure (n : ℕ) (individual_cost : ℕ) (extra_cost : ℕ) : 
  n = 9 → 
  individual_cost = 12 → 
  extra_cost = 8 → 
  (n - 1) * individual_cost + 
  (individual_cost + ((n - 1) * individual_cost + (individual_cost + extra_cost)) / n) = 117 :=
by sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l772_77246


namespace NUMINAMATH_CALUDE_joan_has_five_apples_l772_77262

/-- The number of apples Joan has after giving some away -/
def apples_remaining (initial : ℕ) (given_to_melanie : ℕ) (given_to_sarah : ℕ) : ℕ :=
  initial - given_to_melanie - given_to_sarah

/-- Proof that Joan has 5 apples remaining -/
theorem joan_has_five_apples :
  apples_remaining 43 27 11 = 5 := by
  sorry

end NUMINAMATH_CALUDE_joan_has_five_apples_l772_77262


namespace NUMINAMATH_CALUDE_log_equation_solution_l772_77295

theorem log_equation_solution (x : ℝ) :
  (Real.log x / Real.log 8 + Real.log (x^3) / Real.log 4 = 9) ↔ (x = 2^(54/11)) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_solution_l772_77295


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l772_77296

/-- A geometric sequence of positive integers with first term 5 and fifth term 3125 has its fourth term equal to 625. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 5 →                            -- First term is 5
  a 5 = 3125 →                         -- Fifth term is 3125
  a 4 = 625 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l772_77296


namespace NUMINAMATH_CALUDE_jessica_quarters_l772_77238

theorem jessica_quarters (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 8 → received = 3 → total = initial + received → total = 11 :=
by sorry

end NUMINAMATH_CALUDE_jessica_quarters_l772_77238


namespace NUMINAMATH_CALUDE_tracy_art_fair_sales_l772_77218

theorem tracy_art_fair_sales : 
  let total_customers : ℕ := 20
  let first_group_size : ℕ := 4
  let second_group_size : ℕ := 12
  let third_group_size : ℕ := 4
  let first_group_paintings_per_customer : ℕ := 2
  let second_group_paintings_per_customer : ℕ := 1
  let third_group_paintings_per_customer : ℕ := 4
  first_group_size + second_group_size + third_group_size = total_customers →
  first_group_size * first_group_paintings_per_customer + 
  second_group_size * second_group_paintings_per_customer + 
  third_group_size * third_group_paintings_per_customer = 36 := by
sorry


end NUMINAMATH_CALUDE_tracy_art_fair_sales_l772_77218


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l772_77235

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 10*x + 16) → (∃ y : ℝ, y^2 = 10*y + 16 ∧ x + y = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l772_77235


namespace NUMINAMATH_CALUDE_quadratic_roots_distance_l772_77228

theorem quadratic_roots_distance (t : ℝ) (x₁ x₂ : ℂ) :
  x₁^2 + t*x₁ + 2 = 0 →
  x₂^2 + t*x₂ + 2 = 0 →
  x₁ ≠ x₂ →
  Complex.abs (x₁ - x₂) = 2 * Real.sqrt 2 →
  t = -4 ∨ t = 0 ∨ t = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distance_l772_77228


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l772_77268

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

theorem binary_multiplication_theorem :
  let a := [true, true, false, true, false, true]  -- 110101₂
  let b := [true, true, true, false, true]  -- 11101₂
  let c := [true, false, true, false, true, true, true, false, true, false, true]  -- 10101110101₂
  binary_to_nat a * binary_to_nat b = binary_to_nat c := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l772_77268


namespace NUMINAMATH_CALUDE_university_box_cost_l772_77250

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (dim : BoxDimensions) : ℕ :=
  dim.length * dim.width * dim.height

/-- Calculates the number of boxes needed given the total volume and box volume -/
def boxesNeeded (totalVolume boxVolume : ℕ) : ℕ :=
  (totalVolume + boxVolume - 1) / boxVolume

/-- Calculates the total cost given the number of boxes and cost per box -/
def totalCost (numBoxes : ℕ) (costPerBox : ℚ) : ℚ :=
  (numBoxes : ℚ) * costPerBox

/-- Theorem stating the minimum amount the university must spend on boxes -/
theorem university_box_cost
  (boxDim : BoxDimensions)
  (costPerBox : ℚ)
  (totalVolume : ℕ)
  (h1 : boxDim = ⟨20, 20, 15⟩)
  (h2 : costPerBox = 6/5)
  (h3 : totalVolume = 3060000) :
  totalCost (boxesNeeded totalVolume (boxVolume boxDim)) costPerBox = 612 := by
  sorry


end NUMINAMATH_CALUDE_university_box_cost_l772_77250


namespace NUMINAMATH_CALUDE_pink_highlighters_count_l772_77257

theorem pink_highlighters_count (total yellow blue : ℕ) (h1 : total = 22) (h2 : yellow = 8) (h3 : blue = 5) :
  ∃ pink : ℕ, pink + yellow + blue = total ∧ pink = 9 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_count_l772_77257


namespace NUMINAMATH_CALUDE_coin_and_die_probability_l772_77205

theorem coin_and_die_probability : 
  let coin_prob := 1 / 2  -- Probability of getting heads on a fair coin
  let die_prob := 1 / 6   -- Probability of rolling a multiple of 5 on a 6-sided die
  coin_prob * die_prob = 1 / 12 :=
by sorry

end NUMINAMATH_CALUDE_coin_and_die_probability_l772_77205


namespace NUMINAMATH_CALUDE_cooking_mode_and_median_l772_77269

def cooking_data : List Nat := [4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8]

def mode (data : List Nat) : Nat :=
  sorry

def median (data : List Nat) : Nat :=
  sorry

theorem cooking_mode_and_median :
  mode cooking_data = 6 ∧ median cooking_data = 6 := by
  sorry

end NUMINAMATH_CALUDE_cooking_mode_and_median_l772_77269


namespace NUMINAMATH_CALUDE_civil_servant_dispatch_l772_77201

theorem civil_servant_dispatch (m n k : ℕ) (hm : m = 5) (hn : n = 4) (hk : k = 3) :
  (k.factorial * (Nat.choose (m + n) k - Nat.choose m k - Nat.choose n k)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_civil_servant_dispatch_l772_77201


namespace NUMINAMATH_CALUDE_molecular_weight_CaI2_l772_77236

/-- Given that the molecular weight of 3 moles of CaI2 is 882 g/mol,
    prove that the molecular weight of 1 mole of CaI2 is 294 g/mol. -/
theorem molecular_weight_CaI2 (weight_3_moles : ℝ) (h : weight_3_moles = 882) :
  weight_3_moles / 3 = 294 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_CaI2_l772_77236


namespace NUMINAMATH_CALUDE_sum_234_78_base5_l772_77271

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_234_78_base5 : 
  toBase5 (234 + 78) = [2, 2, 2, 2] := by sorry

end NUMINAMATH_CALUDE_sum_234_78_base5_l772_77271


namespace NUMINAMATH_CALUDE_complement_of_A_l772_77208

def U : Set ℕ := {n | 1 ≤ n ∧ n ≤ 10}
def A : Set ℕ := {1, 2, 3, 5, 8}

theorem complement_of_A : (Aᶜ : Set ℕ) = {4, 6, 7, 9, 10} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l772_77208


namespace NUMINAMATH_CALUDE_susan_jen_time_difference_l772_77270

/-- A relay race with 4 runners -/
structure RelayRace where
  mary_time : ℝ
  susan_time : ℝ
  jen_time : ℝ
  tiffany_time : ℝ

/-- The conditions of the relay race -/
def race_conditions (race : RelayRace) : Prop :=
  race.mary_time = 2 * race.susan_time ∧
  race.susan_time > race.jen_time ∧
  race.jen_time = 30 ∧
  race.tiffany_time = race.mary_time - 7 ∧
  race.mary_time + race.susan_time + race.jen_time + race.tiffany_time = 223

/-- The theorem stating that Susan's time is 10 seconds longer than Jen's time -/
theorem susan_jen_time_difference (race : RelayRace) 
  (h : race_conditions race) : race.susan_time - race.jen_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_susan_jen_time_difference_l772_77270


namespace NUMINAMATH_CALUDE_complex_sum_of_powers_l772_77229

theorem complex_sum_of_powers : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_of_powers_l772_77229


namespace NUMINAMATH_CALUDE_zero_of_function_l772_77226

/-- Given a function f(x) = m + (1/3)^x where f(-2) = 0, prove that m = -9 -/
theorem zero_of_function (m : ℝ) : 
  (let f : ℝ → ℝ := λ x ↦ m + (1/3)^x
   f (-2) = 0) → 
  m = -9 := by sorry

end NUMINAMATH_CALUDE_zero_of_function_l772_77226


namespace NUMINAMATH_CALUDE_course_selection_theorem_l772_77283

/-- The number of ways to select 4 courses out of 9, where 3 specific courses are mutually exclusive -/
def course_selection_schemes (total_courses : ℕ) (exclusive_courses : ℕ) (courses_to_choose : ℕ) : ℕ :=
  (exclusive_courses * Nat.choose (total_courses - exclusive_courses) (courses_to_choose - 1)) +
  Nat.choose (total_courses - exclusive_courses) courses_to_choose

/-- Theorem stating that the number of course selection schemes is 75 -/
theorem course_selection_theorem : course_selection_schemes 9 3 4 = 75 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l772_77283


namespace NUMINAMATH_CALUDE_negation_equivalence_l772_77230

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x - 1 > 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l772_77230


namespace NUMINAMATH_CALUDE_unique_triple_sum_l772_77275

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_triple_sum (a b c : ℕ) 
  (ha : TwoDigitPositiveInt a) 
  (hb : TwoDigitPositiveInt b) 
  (hc : TwoDigitPositiveInt c) 
  (h : a^3 + 3*b^3 + 9*c^3 = 9*a*b*c + 1) : 
  a + b + c = 9 := by
sorry

end NUMINAMATH_CALUDE_unique_triple_sum_l772_77275


namespace NUMINAMATH_CALUDE_complex_equation_solution_l772_77234

theorem complex_equation_solution (a b : ℝ) (z : ℂ) :
  z = a + 4*Complex.I ∧ z / (z + b) = 4*Complex.I → b = 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l772_77234


namespace NUMINAMATH_CALUDE_salt_addition_problem_l772_77231

theorem salt_addition_problem (x : ℝ) (salt_added : ℝ) : 
  x = 104.99999999999997 →
  let initial_salt := 0.2 * x
  let water_after_evaporation := 0.75 * x
  let volume_after_evaporation := water_after_evaporation + initial_salt
  let final_volume := volume_after_evaporation + 7 + salt_added
  let final_salt := initial_salt + salt_added
  (final_salt / final_volume = 1/3) →
  salt_added = 11.375 := by
sorry

#eval (11.375 : Float)

end NUMINAMATH_CALUDE_salt_addition_problem_l772_77231


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l772_77266

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 5 * x^2 + 4 * x + 20 = 0 ↔ x = a + b * I ∨ x = a - b * I) → 
  a + b^2 = 86/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l772_77266


namespace NUMINAMATH_CALUDE_complex_division_simplification_l772_77239

theorem complex_division_simplification : 
  let i : ℂ := Complex.I
  (2 * i) / (1 + i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l772_77239


namespace NUMINAMATH_CALUDE_ellipse_condition_l772_77212

def is_ellipse (k : ℝ) : Prop :=
  1 < k ∧ k < 5 ∧ k ≠ 3

theorem ellipse_condition (k : ℝ) :
  (is_ellipse k → (1 < k ∧ k < 5)) ∧
  ¬(1 < k ∧ k < 5 → is_ellipse k) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l772_77212


namespace NUMINAMATH_CALUDE_function_inequality_l772_77265

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l772_77265


namespace NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l772_77225

/-- A function that reverses a five-digit integer -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 10000
  let b := (n / 1000) % 10
  let c := (n / 100) % 10
  let d := (n / 10) % 10
  let e := n % 10
  e * 10000 + d * 1000 + c * 100 + b * 10 + a

/-- Predicate to check if a natural number has at least one even digit -/
def has_even_digit (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ 2 ∣ d ∧ ∃ k : ℕ, n / (10^k) % 10 = d

theorem sum_with_reverse_has_even_digit (n : ℕ) 
  (h : 10000 ≤ n ∧ n < 100000) : 
  has_even_digit (n + reverse_digits n) :=
sorry

end NUMINAMATH_CALUDE_sum_with_reverse_has_even_digit_l772_77225


namespace NUMINAMATH_CALUDE_binomial_20_2_l772_77286

theorem binomial_20_2 : Nat.choose 20 2 = 190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_2_l772_77286


namespace NUMINAMATH_CALUDE_tv_price_change_l772_77273

theorem tv_price_change (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0) 
  (h2 : x > 0) : 
  (initial_price * 0.8 * (1 + x / 100) = initial_price * 1.12) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_tv_price_change_l772_77273


namespace NUMINAMATH_CALUDE_negation_of_proposition_ln_negation_l772_77284

open Real

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x, p x) ↔ (∃ x, ¬ p x) :=
by sorry

theorem ln_negation :
  (¬ ∀ x : ℝ, log x > 1) ↔ (∃ x : ℝ, log x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_ln_negation_l772_77284


namespace NUMINAMATH_CALUDE_a_5_equals_17_l772_77247

theorem a_5_equals_17 (a : ℕ → ℤ) (h : ∀ n, a n = 4 * n - 3) : a 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_17_l772_77247


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l772_77241

-- Define the hyperbola equation
def hyperbola_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 81 = 1

-- Define the asymptote equation
def asymptote_equation (x y : ℝ) : Prop :=
  y = 3 * x

-- Theorem statement
theorem hyperbola_asymptote_implies_a_value (a : ℝ) :
  (a > 0) →
  (∃ x y : ℝ, hyperbola_equation x y a ∧ asymptote_equation x y) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_a_value_l772_77241


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l772_77274

theorem ellipse_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, x^2/2 + y^2/m = 1) →  -- ellipse equation
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (x^2/a^2 + y^2/b^2 = 1 ↔ x^2/2 + y^2/m = 1) ∧ 
    c^2 = a^2 - b^2 ∧ c/a = 1/2) →  -- eccentricity condition
  m = 3/2 ∨ m = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l772_77274


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l772_77299

theorem exactly_one_greater_than_one 
  (x y z : ℝ) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) 
  (product_one : x * y * z = 1) 
  (sum_inequality : x + y + z > 1/x + 1/y + 1/z) : 
  (x > 1 ∧ y ≤ 1 ∧ z ≤ 1) ∨ 
  (x ≤ 1 ∧ y > 1 ∧ z ≤ 1) ∨ 
  (x ≤ 1 ∧ y ≤ 1 ∧ z > 1) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l772_77299


namespace NUMINAMATH_CALUDE_smallest_sticker_collection_l772_77251

theorem smallest_sticker_collection (S : ℕ) : 
  S > 2 →
  S % 5 = 2 →
  S % 11 = 2 →
  S % 13 = 2 →
  (∀ T : ℕ, T > 2 ∧ T % 5 = 2 ∧ T % 11 = 2 ∧ T % 13 = 2 → S ≤ T) →
  S = 717 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sticker_collection_l772_77251


namespace NUMINAMATH_CALUDE_stating_traffic_light_probability_l772_77285

/-- Represents the duration of a traffic light cycle in seconds. -/
def cycleDuration : ℕ := 80

/-- Represents the duration of time when proceeding is allowed (green + yellow) in seconds. -/
def proceedDuration : ℕ := 50

/-- Represents the duration of time when proceeding is not allowed (red) in seconds. -/
def stopDuration : ℕ := 30

/-- Represents the maximum waiting time in seconds for the probability calculation. -/
def maxWaitTime : ℕ := 10

/-- 
Theorem stating that the probability of waiting no more than 10 seconds to proceed 
in the given traffic light cycle is 3/4.
-/
theorem traffic_light_probability : 
  (proceedDuration + maxWaitTime : ℚ) / cycleDuration = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_stating_traffic_light_probability_l772_77285


namespace NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l772_77263

/-- A positive integer greater than 1 is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop := n > 1 ∧ ∃ k, 1 < k ∧ k < n ∧ k ∣ n

/-- The largest integer that always divides n^4 - n for all composite n is 6 -/
theorem largest_divisor_n4_minus_n (n : ℕ) (h : IsComposite n) :
  (∀ m : ℕ, m > 6 → ¬(m ∣ (n^4 - n))) ∧ (6 ∣ (n^4 - n)) := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_n4_minus_n_l772_77263


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l772_77224

/-- An arithmetic sequence with first term 1 and sum of first three terms 9 has general term 2n - 1 -/
theorem arithmetic_sequence_general_term :
  ∀ (a : ℕ → ℝ), 
    (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) → -- arithmetic sequence condition
    a 1 = 1 → -- first term condition
    a 1 + a 2 + a 3 = 9 → -- sum of first three terms condition
    ∀ n, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l772_77224


namespace NUMINAMATH_CALUDE_range_of_a_l772_77281

/-- The range of non-negative real number a that satisfies the given conditions -/
theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → ∃ (an bn : ℝ), bn = an^3) →  -- Points are on y = x^3
  (∃ (a1 : ℝ), a1 = a ∧ a ≥ 0) →  -- a1 = a and a ≥ 0
  (∀ n : ℕ, n ≥ 1 → ∃ (cn : ℝ), cn = an + an+1) →  -- cn = an + an+1
  (∀ n : ℕ, n ≥ 1 → ∃ (cn an : ℝ), cn = 1/2 * an + 3/2) →  -- cn = 1/2*an + 3/2
  (∀ n : ℕ, an ≠ 1) →  -- All terms of {an} are not equal to 1
  (∀ n : ℕ, n ≥ 1 → ∃ (kn : ℝ), kn = (bn - 1) / (an - 1)) →  -- kn = (bn - 1) / (an - 1)
  (∃ (k0 : ℝ), ∀ n : ℕ, n ≥ 1 → (kn - k0) * (kn+1 - k0) < 0) →  -- Existence of k0
  (0 ≤ a ∧ a < 7 ∧ a ≠ 1) :=
by
  sorry


end NUMINAMATH_CALUDE_range_of_a_l772_77281


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l772_77240

theorem arithmetic_sequence_problem (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  (a 2)^2 + 12*(a 2) - 8 = 0 →  -- a₂ is a root
  (a 10)^2 + 12*(a 10) - 8 = 0 →  -- a₁₀ is a root
  a 2 ≠ a 10 →  -- a₂ and a₁₀ are distinct roots
  a 6 = -6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l772_77240
