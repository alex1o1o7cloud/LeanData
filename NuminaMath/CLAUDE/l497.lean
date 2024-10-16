import Mathlib

namespace NUMINAMATH_CALUDE_johns_daily_wage_without_bonus_l497_49705

/-- John's work scenario -/
structure WorkScenario where
  regular_hours : ℕ
  bonus_hours : ℕ
  bonus_amount : ℕ
  hourly_rate_with_bonus : ℕ

/-- Calculates John's daily wage without bonus -/
def daily_wage_without_bonus (w : WorkScenario) : ℕ :=
  w.hourly_rate_with_bonus * w.bonus_hours - w.bonus_amount

/-- Theorem: John's daily wage without bonus is $80 -/
theorem johns_daily_wage_without_bonus :
  let w : WorkScenario := {
    regular_hours := 8,
    bonus_hours := 10,
    bonus_amount := 20,
    hourly_rate_with_bonus := 10
  }
  daily_wage_without_bonus w = 80 := by
  sorry


end NUMINAMATH_CALUDE_johns_daily_wage_without_bonus_l497_49705


namespace NUMINAMATH_CALUDE_four_circles_common_tangent_l497_49703

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the length of a common tangent between two circles -/
def tangentLength (c1 c2 : Circle) : ℝ := sorry

/-- 
Given four circles α, β, γ, and δ satisfying the tangent length equation,
there exists a circle tangent to all four circles.
-/
theorem four_circles_common_tangent 
  (α β γ δ : Circle)
  (h : tangentLength α β * tangentLength γ δ + 
       tangentLength β γ * tangentLength δ α = 
       tangentLength α γ * tangentLength β δ) :
  ∃ (σ : Circle), 
    (tangentLength σ α = 0) ∧ 
    (tangentLength σ β = 0) ∧ 
    (tangentLength σ γ = 0) ∧ 
    (tangentLength σ δ = 0) :=
sorry

end NUMINAMATH_CALUDE_four_circles_common_tangent_l497_49703


namespace NUMINAMATH_CALUDE_unique_prime_in_special_form_l497_49716

def special_form (n : ℕ) : ℚ :=
  (1 / 11) * ((10^(2*n) - 1) / 9)

theorem unique_prime_in_special_form :
  ∃! p : ℕ, Prime p ∧ ∃ n : ℕ, (special_form n : ℚ) = p ∧ p = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_in_special_form_l497_49716


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l497_49711

theorem inequality_and_equality_conditions (a b : ℝ) :
  (a^2 + b^2 - a - b - a*b + 0.25 ≥ 0) ∧
  (a^2 + b^2 - a - b - a*b + 0.25 = 0 ↔ (a = 0 ∧ b = 0.5) ∨ (a = 0.5 ∧ b = 0)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l497_49711


namespace NUMINAMATH_CALUDE_triangle_area_l497_49719

/-- The area of a triangle with base 7 units and height 3 units is 10.5 square units. -/
theorem triangle_area : 
  let base : ℝ := 7
  let height : ℝ := 3
  let area : ℝ := (1/2) * base * height
  area = 10.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l497_49719


namespace NUMINAMATH_CALUDE_middle_number_proof_l497_49774

theorem middle_number_proof (A B C : ℝ) (hC : C = 56) (hDiff : C - A = 32) (hRatio : B / C = 5 / 7) : B = 40 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l497_49774


namespace NUMINAMATH_CALUDE_xyz_equals_five_l497_49751

theorem xyz_equals_five
  (a b c x y z : ℂ)
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (eq_a : a = (b^2 + c^2) / (x - 3))
  (eq_b : b = (a^2 + c^2) / (y - 3))
  (eq_c : c = (a^2 + b^2) / (z - 3))
  (sum_prod : x*y + y*z + z*x = 11)
  (sum : x + y + z = 5) :
  x * y * z = 5 := by
sorry

end NUMINAMATH_CALUDE_xyz_equals_five_l497_49751


namespace NUMINAMATH_CALUDE_quadratic_intersection_l497_49708

/-- Represents a quadratic function of the form y = x^2 + px + q -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  h : -2 * p + q = 2023

/-- The x-coordinate of the intersection point -/
def intersection_x : ℝ := -2

/-- The y-coordinate of the intersection point -/
def intersection_y : ℝ := 2027

/-- Theorem stating that all quadratic functions satisfying the condition intersect at a single point -/
theorem quadratic_intersection (f : QuadraticFunction) : 
  (intersection_x^2 + f.p * intersection_x + f.q) = intersection_y := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l497_49708


namespace NUMINAMATH_CALUDE_factorization_identity_l497_49729

theorem factorization_identity (x y : ℝ) : (x - y)^2 + 2*y*(x - y) = (x - y)*(x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l497_49729


namespace NUMINAMATH_CALUDE_defective_product_selection_l497_49735

/-- The set of possible numbers of defective products when selecting from a pool --/
def PossibleDefectives (total : ℕ) (defective : ℕ) (selected : ℕ) : Set ℕ :=
  {n : ℕ | n ≤ min defective selected ∧ n ≤ selected ∧ defective - n ≤ total - selected}

/-- Theorem stating the possible values for the number of defective products selected --/
theorem defective_product_selection (total : ℕ) (defective : ℕ) (selected : ℕ)
  (h_total : total = 8)
  (h_defective : defective = 2)
  (h_selected : selected = 3) :
  PossibleDefectives total defective selected = {0, 1, 2} :=
by sorry

end NUMINAMATH_CALUDE_defective_product_selection_l497_49735


namespace NUMINAMATH_CALUDE_smallest_non_prime_sums_l497_49772

theorem smallest_non_prime_sums : ∃ (n : ℕ), n = 7 ∧
  (∀ m : ℕ, m < n →
    (Prime (m + 1 + m + 2 + m + 3) ∨
     Prime (m + m + 2 + m + 3) ∨
     Prime (m + m + 1 + m + 3) ∨
     Prime (m + m + 1 + m + 2))) ∧
  (¬ Prime (n + 1 + n + 2 + n + 3) ∧
   ¬ Prime (n + n + 2 + n + 3) ∧
   ¬ Prime (n + n + 1 + n + 3) ∧
   ¬ Prime (n + n + 1 + n + 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_prime_sums_l497_49772


namespace NUMINAMATH_CALUDE_total_candy_count_l497_49794

theorem total_candy_count (chocolate_boxes gummy_boxes caramel_boxes : ℕ)
  (chocolate_per_box gummy_per_box caramel_per_box : ℕ)
  (h1 : chocolate_boxes = 3)
  (h2 : chocolate_per_box = 6)
  (h3 : caramel_boxes = 5)
  (h4 : caramel_per_box = 8)
  (h5 : gummy_boxes = 4)
  (h6 : gummy_per_box = 10) :
  chocolate_boxes * chocolate_per_box +
  caramel_boxes * caramel_per_box +
  gummy_boxes * gummy_per_box = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_candy_count_l497_49794


namespace NUMINAMATH_CALUDE_digital_earth_properties_l497_49721

-- Define the properties of Digital Earth
structure DigitalEarth where
  is_digitized : Bool
  meets_current_needs : Bool
  main_feature_virtual_reality : Bool
  uses_centralized_storage : Bool

-- Define the correct properties of Digital Earth
def correct_digital_earth : DigitalEarth := {
  is_digitized := true,
  meets_current_needs := false,
  main_feature_virtual_reality := true,
  uses_centralized_storage := false
}

-- Theorem stating the correct properties of Digital Earth
theorem digital_earth_properties :
  correct_digital_earth.is_digitized ∧
  correct_digital_earth.main_feature_virtual_reality ∧
  ¬correct_digital_earth.meets_current_needs ∧
  ¬correct_digital_earth.uses_centralized_storage :=
by sorry


end NUMINAMATH_CALUDE_digital_earth_properties_l497_49721


namespace NUMINAMATH_CALUDE_continuity_at_three_l497_49789

def f (x : ℝ) : ℝ := 2 * x^2 - 4

theorem continuity_at_three :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_continuity_at_three_l497_49789


namespace NUMINAMATH_CALUDE_height_after_growth_spurt_height_approximation_l497_49757

/-- Calculates the height of a person after a year of growth with specific conditions. -/
theorem height_after_growth_spurt (initial_height : ℝ) 
  (initial_growth_rate : ℝ) (initial_growth_months : ℕ) 
  (growth_increase_rate : ℝ) (total_months : ℕ) : ℝ :=
  let inches_to_meters := 0.0254
  let height_after_initial_growth := initial_height + initial_growth_rate * initial_growth_months
  let remaining_months := total_months - initial_growth_months
  let first_variable_growth := initial_growth_rate * (1 + growth_increase_rate)
  let variable_growth_sum := first_variable_growth * 
    (1 - (1 + growth_increase_rate) ^ remaining_months) / growth_increase_rate
  (height_after_initial_growth + variable_growth_sum) * inches_to_meters

/-- The height after growth spurt is approximately 2.59 meters. -/
theorem height_approximation : 
  ∃ ε > 0, |height_after_growth_spurt 66 2 3 0.1 12 - 2.59| < ε :=
sorry

end NUMINAMATH_CALUDE_height_after_growth_spurt_height_approximation_l497_49757


namespace NUMINAMATH_CALUDE_product_inequality_l497_49743

theorem product_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l497_49743


namespace NUMINAMATH_CALUDE_rectangular_prism_surface_area_l497_49762

/-- The surface area of a rectangular prism with dimensions 1, 2, and 2 is 16 -/
theorem rectangular_prism_surface_area :
  let length : ℝ := 1
  let width : ℝ := 2
  let height : ℝ := 2
  let surface_area := 2 * (length * width + length * height + width * height)
  surface_area = 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_surface_area_l497_49762


namespace NUMINAMATH_CALUDE_sqrt_a_plus_one_real_l497_49720

theorem sqrt_a_plus_one_real (a : ℝ) : (∃ (x : ℝ), x ^ 2 = a + 1) ↔ a ≥ -1 := by sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_one_real_l497_49720


namespace NUMINAMATH_CALUDE_base_89_multiple_of_13_l497_49701

theorem base_89_multiple_of_13 (b : ℤ) (h1 : 0 ≤ b) (h2 : b ≤ 20) 
  (h3 : (142536472 : ℤ) ≡ b [ZMOD 13]) : b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_89_multiple_of_13_l497_49701


namespace NUMINAMATH_CALUDE_clothing_distribution_l497_49746

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 59)
  (h2 : first_load = 32)
  (h3 : num_small_loads = 9)
  : (total - first_load) / num_small_loads = 3 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l497_49746


namespace NUMINAMATH_CALUDE_ab_value_l497_49738

theorem ab_value (a b : ℝ) (h : |a + 1| + (b - 3)^2 = 0) : a^b = -1 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l497_49738


namespace NUMINAMATH_CALUDE_train_speed_l497_49797

/-- The speed of a train given its length, time to cross a walking man, and the man's speed --/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmh : ℝ) : 
  train_length = 700 →
  crossing_time = 41.9966402687785 →
  man_speed_kmh = 3 →
  ∃ (train_speed_kmh : ℝ), abs (train_speed_kmh - 63.0036) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_train_speed_l497_49797


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l497_49747

theorem minimum_value_theorem (x : ℝ) (h : x > -2) :
  x + 1 / (x + 2) ≥ 0 ∧ ∃ y > -2, y + 1 / (y + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l497_49747


namespace NUMINAMATH_CALUDE_unique_arrangement_l497_49790

-- Define the types for containers and liquids
inductive Container : Type
  | Bottle
  | Glass
  | Jug
  | Jar

inductive Liquid : Type
  | Milk
  | Lemonade
  | Kvass
  | Water

-- Define the arrangement as a function from Container to Liquid
def Arrangement := Container → Liquid

-- Define the conditions
def water_milk_not_in_bottle (arr : Arrangement) : Prop :=
  arr Container.Bottle ≠ Liquid.Water ∧ arr Container.Bottle ≠ Liquid.Milk

def lemonade_between_jug_and_kvass (arr : Arrangement) : Prop :=
  (arr Container.Bottle = Liquid.Lemonade ∧ arr Container.Jug = Liquid.Milk ∧ arr Container.Jar = Liquid.Kvass) ∨
  (arr Container.Glass = Liquid.Lemonade ∧ arr Container.Bottle = Liquid.Kvass ∧ arr Container.Jar = Liquid.Milk) ∨
  (arr Container.Bottle = Liquid.Milk ∧ arr Container.Glass = Liquid.Lemonade ∧ arr Container.Jug = Liquid.Kvass)

def jar_not_lemonade_or_water (arr : Arrangement) : Prop :=
  arr Container.Jar ≠ Liquid.Lemonade ∧ arr Container.Jar ≠ Liquid.Water

def glass_next_to_jar_and_milk (arr : Arrangement) : Prop :=
  (arr Container.Glass = Liquid.Water ∧ arr Container.Jug = Liquid.Milk) ∨
  (arr Container.Glass = Liquid.Kvass ∧ arr Container.Bottle = Liquid.Milk)

-- Define the correct arrangement
def correct_arrangement : Arrangement :=
  fun c => match c with
  | Container.Bottle => Liquid.Lemonade
  | Container.Glass => Liquid.Water
  | Container.Jug => Liquid.Milk
  | Container.Jar => Liquid.Kvass

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement),
    water_milk_not_in_bottle arr ∧
    lemonade_between_jug_and_kvass arr ∧
    jar_not_lemonade_or_water arr ∧
    glass_next_to_jar_and_milk arr →
    arr = correct_arrangement :=
by sorry

end NUMINAMATH_CALUDE_unique_arrangement_l497_49790


namespace NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_and_s_l497_49742

theorem log_10_7_in_terms_of_r_and_s (r s : ℝ) 
  (hr : Real.log 2 / Real.log 5 = r) 
  (hs : Real.log 7 / Real.log 2 = s) : 
  Real.log 7 / Real.log 10 = s * r / (r + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_10_7_in_terms_of_r_and_s_l497_49742


namespace NUMINAMATH_CALUDE_circle_chord_tangent_relation_l497_49741

/-- Given a circle with radius r, a chord FG extending to meet the tangent at F at point H,
    and a point I on FH such that FI = GH, prove that v^2 = u^3 / (r + u),
    where u is the distance of I from the tangent through G
    and v is the distance of I from the line through chord FG. -/
theorem circle_chord_tangent_relation (r : ℝ) (u v : ℝ) 
  (h_positive : r > 0) 
  (h_u_positive : u > 0) 
  (h_v_positive : v > 0) 
  (h_v_eq_r : v = r) : 
  v^2 = u^3 / (r + u) := by
  sorry

end NUMINAMATH_CALUDE_circle_chord_tangent_relation_l497_49741


namespace NUMINAMATH_CALUDE_x_cube_minus_six_x_squared_l497_49767

theorem x_cube_minus_six_x_squared (x : ℝ) : x = 3 → x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_x_cube_minus_six_x_squared_l497_49767


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l497_49765

theorem fraction_sum_equals_decimal : (3 / 15) + (5 / 125) + (7 / 1000) = 0.247 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l497_49765


namespace NUMINAMATH_CALUDE_max_visible_cubes_is_400_l497_49756

/-- The dimension of the cube --/
def n : ℕ := 12

/-- The number of unit cubes on one face of the cube --/
def face_count : ℕ := n^2

/-- The number of unit cubes along one edge of the cube --/
def edge_count : ℕ := n

/-- The number of visible faces from a corner --/
def visible_faces : ℕ := 3

/-- The number of edges shared between two visible faces --/
def shared_edges : ℕ := 3

/-- The maximum number of visible unit cubes from a single point --/
def max_visible_cubes : ℕ := visible_faces * face_count - shared_edges * (edge_count - 1) + 1

/-- Theorem stating that the maximum number of visible unit cubes is 400 --/
theorem max_visible_cubes_is_400 : max_visible_cubes = 400 := by
  sorry

end NUMINAMATH_CALUDE_max_visible_cubes_is_400_l497_49756


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l497_49766

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 - 30 * x + c = 0) →
  a + c = 41 →
  a < c →
  (a = (41 + Real.sqrt 781) / 2 ∧ c = (41 - Real.sqrt 781) / 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l497_49766


namespace NUMINAMATH_CALUDE_parallelogram_vector_l497_49704

/-- A parallelogram on the complex plane -/
structure Parallelogram :=
  (A B C D : ℂ)
  (parallelogram_condition : (C - A) = (D - B))

/-- The theorem statement -/
theorem parallelogram_vector (ABCD : Parallelogram) 
  (hAC : ABCD.C - ABCD.A = 6 + 8*I) 
  (hBD : ABCD.D - ABCD.B = -4 + 6*I) : 
  ABCD.A - ABCD.D = -1 - 7*I :=
sorry

end NUMINAMATH_CALUDE_parallelogram_vector_l497_49704


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l497_49753

/-- The constant term in the binomial expansion of (ax - 1/√x)^6 -/
def constant_term (a : ℝ) : ℝ := 15 * a^2

theorem binomial_expansion_constant_term (a : ℝ) (h1 : a > 0) (h2 : constant_term a = 120) : 
  a = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l497_49753


namespace NUMINAMATH_CALUDE_coffee_mixture_proof_l497_49768

/-- Given a total mixture and a ratio of coffee to milk, calculate the amount of coffee needed. -/
def coffee_amount (total_mixture : ℕ) (coffee_ratio milk_ratio : ℕ) : ℕ :=
  (total_mixture * coffee_ratio) / (coffee_ratio + milk_ratio)

/-- Theorem stating that for a 4400g mixture with a 2:9 coffee to milk ratio, 800g of coffee is needed. -/
theorem coffee_mixture_proof :
  coffee_amount 4400 2 9 = 800 := by
  sorry

end NUMINAMATH_CALUDE_coffee_mixture_proof_l497_49768


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l497_49723

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l497_49723


namespace NUMINAMATH_CALUDE_two_roots_implies_c_value_l497_49715

/-- A cubic function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

/-- The number of roots of f for a given c -/
def num_roots (c : ℝ) : ℕ := sorry

/-- Theorem stating that if f has exactly two roots, then c is either -2 or 2 -/
theorem two_roots_implies_c_value (c : ℝ) :
  num_roots c = 2 → c = -2 ∨ c = 2 := by sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_value_l497_49715


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l497_49780

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 5 * x + m^2 - 2 * m = 0) ∧ 
  (m * 0^2 + 5 * 0 + m^2 - 2 * m = 0) → 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l497_49780


namespace NUMINAMATH_CALUDE_sugar_amount_is_one_cup_l497_49771

/-- Represents the ratio of ingredients in a recipe --/
structure Ratio :=
  (flour : ℚ)
  (water : ℚ)
  (sugar : ℚ)

/-- The original recipe ratio --/
def originalRatio : Ratio :=
  { flour := 7, water := 2, sugar := 1 }

/-- The new recipe ratio --/
def newRatio : Ratio :=
  { flour := originalRatio.flour * 2,
    water := originalRatio.water,
    sugar := originalRatio.sugar * 2 }

/-- The amount of water in the new recipe (in cups) --/
def newWaterAmount : ℚ := 2

/-- Calculates the amount of sugar needed in the new recipe --/
def sugarNeeded (r : Ratio) (waterAmount : ℚ) : ℚ :=
  (waterAmount * r.sugar) / r.water

/-- Theorem stating that the amount of sugar needed in the new recipe is 1 cup --/
theorem sugar_amount_is_one_cup :
  sugarNeeded newRatio newWaterAmount = 1 := by
  sorry


end NUMINAMATH_CALUDE_sugar_amount_is_one_cup_l497_49771


namespace NUMINAMATH_CALUDE_choose_3_from_12_l497_49709

theorem choose_3_from_12 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_choose_3_from_12_l497_49709


namespace NUMINAMATH_CALUDE_jumps_per_meter_l497_49739

/-- Given the relationships between different units of length, 
    this theorem proves how many jumps are in one meter. -/
theorem jumps_per_meter 
  (x y a b p q s t : ℚ) 
  (hops_to_skips : x * 1 = y)
  (jumps_to_hops : a = b * 1)
  (skips_to_leaps : p * 1 = q)
  (leaps_to_meters : s = t * 1)
  (x_pos : 0 < x) (y_pos : 0 < y) (a_pos : 0 < a) (b_pos : 0 < b)
  (p_pos : 0 < p) (q_pos : 0 < q) (s_pos : 0 < s) (t_pos : 0 < t) :
  1 = (s * p * x * a) / (t * q * y * b) :=
sorry

end NUMINAMATH_CALUDE_jumps_per_meter_l497_49739


namespace NUMINAMATH_CALUDE_circle_C_equation_l497_49730

/-- Given circle is symmetric to (x-1)^2 + y^2 = 1 with respect to y = -x -/
def given_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := y = -x

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 1

/-- Symmetry transformation -/
def symmetric_point (x y : ℝ) : ℝ × ℝ := (-y, -x)

theorem circle_C_equation :
  ∀ x y : ℝ, 
  (∃ x' y' : ℝ, given_circle x' y' ∧ 
   symmetric_point x y = (x', y') ∧
   symmetry_line x y) →
  circle_C x y :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_l497_49730


namespace NUMINAMATH_CALUDE_pharmaceutical_royalties_l497_49727

theorem pharmaceutical_royalties (first_royalties second_royalties second_sales : ℝ)
  (ratio_decrease : ℝ) (h1 : first_royalties = 8)
  (h2 : second_royalties = 9) (h3 : second_sales = 108)
  (h4 : ratio_decrease = 0.7916666666666667) :
  ∃ first_sales : ℝ,
    first_sales = 20 ∧
    (first_royalties / first_sales) - (second_royalties / second_sales) =
      ratio_decrease * (first_royalties / first_sales) :=
by sorry

end NUMINAMATH_CALUDE_pharmaceutical_royalties_l497_49727


namespace NUMINAMATH_CALUDE_paperback_copies_sold_l497_49712

theorem paperback_copies_sold (hardback_copies : ℕ) (total_copies : ℕ) :
  hardback_copies = 36000 →
  total_copies = 440000 →
  ∃ (paperback_copies : ℕ), 
    paperback_copies = 9 * hardback_copies ∧
    total_copies = hardback_copies + paperback_copies ∧
    paperback_copies = 324000 :=
by sorry

end NUMINAMATH_CALUDE_paperback_copies_sold_l497_49712


namespace NUMINAMATH_CALUDE_speed_decrease_percentage_l497_49734

theorem speed_decrease_percentage (distance : ℝ) (fast_speed slow_speed : ℝ) 
  (h_distance_positive : distance > 0)
  (h_fast_speed_positive : fast_speed > 0)
  (h_slow_speed_positive : slow_speed > 0)
  (h_fast_time : distance / fast_speed = 40)
  (h_slow_time : distance / slow_speed = 50) :
  (fast_speed - slow_speed) / fast_speed = 1/5 := by
sorry

end NUMINAMATH_CALUDE_speed_decrease_percentage_l497_49734


namespace NUMINAMATH_CALUDE_mike_seed_count_l497_49725

theorem mike_seed_count (seeds_left : ℕ) (seeds_to_left : ℕ) (seeds_to_new : ℕ) 
  (h1 : seeds_to_left = 20)
  (h2 : seeds_left = 30)
  (h3 : seeds_to_new = 30) :
  seeds_left + seeds_to_left + 2 * seeds_to_left + seeds_to_new = 120 := by
  sorry

#check mike_seed_count

end NUMINAMATH_CALUDE_mike_seed_count_l497_49725


namespace NUMINAMATH_CALUDE_compound_interest_rate_l497_49713

/-- Proves that given the compound interest conditions, the rate of interest is 5% -/
theorem compound_interest_rate (P R : ℝ) 
  (h1 : P * (1 + R / 100) ^ 2 = 17640)
  (h2 : P * (1 + R / 100) ^ 3 = 18522) : 
  R = 5 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l497_49713


namespace NUMINAMATH_CALUDE_average_equals_50y_implies_y_value_l497_49773

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem average_equals_50y_implies_y_value :
  let n := 99
  let sum_1_to_99 := sum_to_n n
  ∀ y : ℚ, (sum_1_to_99 + y) / (n + 1 : ℚ) = 50 * y → y = 4950 / 4999 := by
sorry

end NUMINAMATH_CALUDE_average_equals_50y_implies_y_value_l497_49773


namespace NUMINAMATH_CALUDE_system_solution_l497_49784

theorem system_solution : ∃! (x y : ℝ), 2 * x - y = 3 ∧ x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l497_49784


namespace NUMINAMATH_CALUDE_exists_similar_package_with_ten_boxes_l497_49752

/-- Represents a rectangular box with dimensions a, b, and c -/
structure Box where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- Represents a package containing boxes -/
structure Package where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z

/-- Defines geometric similarity between a package and a box -/
def geometricallySimilar (p : Package) (b : Box) : Prop :=
  ∃ k : ℝ, k > 0 ∧ p.x = k * b.a ∧ p.y = k * b.b ∧ p.z = k * b.c

/-- Defines if a package can contain exactly 10 boxes -/
def canContainTenBoxes (p : Package) (b : Box) : Prop :=
  (p.x = 10 * b.a ∧ p.y = b.b ∧ p.z = b.c) ∨
  (p.x = 5 * b.a ∧ p.y = 2 * b.b ∧ p.z = b.c)

/-- Theorem stating that there exists a package geometrically similar to a box and containing 10 boxes -/
theorem exists_similar_package_with_ten_boxes (b : Box) :
  ∃ p : Package, geometricallySimilar p b ∧ canContainTenBoxes p b := by
  sorry

end NUMINAMATH_CALUDE_exists_similar_package_with_ten_boxes_l497_49752


namespace NUMINAMATH_CALUDE_coefficient_x_squared_l497_49759

/-- The coefficient of x^2 in the expansion of (2x^2 + 3x + 4)(5x^2 + 6x + 7) is 52 -/
theorem coefficient_x_squared (x : ℝ) : 
  (2*x^2 + 3*x + 4) * (5*x^2 + 6*x + 7) = 10*x^4 + 27*x^3 + 52*x^2 + 45*x + 28 := by
  sorry

#check coefficient_x_squared

end NUMINAMATH_CALUDE_coefficient_x_squared_l497_49759


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l497_49745

theorem boys_to_girls_ratio (S G : ℚ) (h : S > 0) (h1 : G > 0) (h2 : (1/2) * G = (1/3) * S) :
  (S - G) / G = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l497_49745


namespace NUMINAMATH_CALUDE_number_puzzle_l497_49799

theorem number_puzzle (x : ℤ) : x + (x - 1) = 33 → 6 * x - 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l497_49799


namespace NUMINAMATH_CALUDE_no_two_digit_even_square_palindromes_l497_49795

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem no_two_digit_even_square_palindromes :
  ¬ ∃ n : ℕ, is_two_digit n ∧ is_perfect_square n ∧ 
    (∃ m : ℕ, n = m * m ∧ is_even m) ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_no_two_digit_even_square_palindromes_l497_49795


namespace NUMINAMATH_CALUDE_smallest_possible_a_l497_49754

theorem smallest_possible_a (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) : 
  (∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ 17) ∧ a ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l497_49754


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l497_49769

theorem angle_sum_is_pi_over_two (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_condition : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l497_49769


namespace NUMINAMATH_CALUDE_no_number_with_2011_quotient_and_remainder_l497_49793

-- Function to calculate the sum of digits of a natural number
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem no_number_with_2011_quotient_and_remainder :
  ¬ ∃ (n : ℕ), 
    let s := sumOfDigits n
    n / s = 2011 ∧ n % s = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_number_with_2011_quotient_and_remainder_l497_49793


namespace NUMINAMATH_CALUDE_original_price_calculation_l497_49726

theorem original_price_calculation (price paid : ℝ) (h1 : paid = 18) (h2 : paid = (1/4) * price) : price = 72 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l497_49726


namespace NUMINAMATH_CALUDE_tangent_beta_minus_two_alpha_l497_49785

theorem tangent_beta_minus_two_alpha 
  (α β : ℝ) 
  (h1 : (1 - Real.cos (2 * α)) / (Real.sin α * Real.cos α) = 1) 
  (h2 : Real.tan (β - α) = -1/3) : 
  Real.tan (β - 2*α) = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_beta_minus_two_alpha_l497_49785


namespace NUMINAMATH_CALUDE_expression_equality_l497_49718

theorem expression_equality (a b c d : ℕ) : 
  a = 12 → b = 13 → c = 16 → d = 11 → 3 * a^2 - 3 * b + 2 * c * d^2 = 4265 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l497_49718


namespace NUMINAMATH_CALUDE_no_five_digit_sum_20_div_9_l497_49755

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem no_five_digit_sum_20_div_9 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → ¬(n % 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_five_digit_sum_20_div_9_l497_49755


namespace NUMINAMATH_CALUDE_no_rational_roots_l497_49775

/-- The polynomial we're investigating -/
def p (x : ℚ) : ℚ := 3 * x^4 + 2 * x^3 - 8 * x^2 - x + 1

/-- Theorem stating that the polynomial has no rational roots -/
theorem no_rational_roots : ∀ x : ℚ, p x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l497_49775


namespace NUMINAMATH_CALUDE_multiply_decimals_l497_49796

theorem multiply_decimals : (4.8 : ℝ) * 0.25 * 0.1 = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_multiply_decimals_l497_49796


namespace NUMINAMATH_CALUDE_box_areas_product_l497_49702

/-- For a rectangular box with dimensions a, b, and c, and a constant k,
    where the areas of the bottom, side, and front are kab, kbc, and kca respectively,
    the product of these areas is equal to k^3 × (abc)^2. -/
theorem box_areas_product (a b c k : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ k > 0) :
  (k * a * b) * (k * b * c) * (k * c * a) = k^3 * (a * b * c)^2 := by
  sorry

end NUMINAMATH_CALUDE_box_areas_product_l497_49702


namespace NUMINAMATH_CALUDE_lucky_years_2023_to_2027_l497_49706

def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem lucky_years_2023_to_2027 : 
  ¬(isLuckyYear 2023) ∧
  (isLuckyYear 2024) ∧
  (isLuckyYear 2025) ∧
  (isLuckyYear 2026) ∧
  (isLuckyYear 2027) := by
  sorry

end NUMINAMATH_CALUDE_lucky_years_2023_to_2027_l497_49706


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l497_49736

/-- Given a point P on the unit circle with coordinates (sin(2π/3), cos(2π/3)) 
    on the terminal side of angle α, prove that the smallest positive value of α is 11π/6 -/
theorem smallest_positive_angle (P : ℝ × ℝ) (α : ℝ) : 
  P.1 = Real.sin (2 * Real.pi / 3) →
  P.2 = Real.cos (2 * Real.pi / 3) →
  P ∈ {Q : ℝ × ℝ | ∃ t : ℝ, Q.1 = Real.cos t ∧ Q.2 = Real.sin t ∧ t ≥ 0 ∧ t < 2 * Real.pi} →
  (∃ k : ℤ, α = 11 * Real.pi / 6 + 2 * Real.pi * k) ∧ 
  (∀ β : ℝ, (∃ k : ℤ, β = α + 2 * Real.pi * k) → β ≥ 11 * Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l497_49736


namespace NUMINAMATH_CALUDE_total_interest_after_ten_years_l497_49748

/-- Calculate the total interest after 10 years given:
  * The simple interest on the initial principal for 10 years is 1400
  * The principal is trebled after 5 years
-/
theorem total_interest_after_ten_years (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 1400) : 
  (P * R * 5 / 100) + (3 * P * R * 5 / 100) = 280 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_after_ten_years_l497_49748


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l497_49728

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 2 then (4 - a) * x + 7 else a^x

-- Define what it means for f to be increasing on ℝ
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem increasing_function_a_range :
  ∀ a : ℝ, (is_increasing (f a)) ↔ (3 ≤ a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l497_49728


namespace NUMINAMATH_CALUDE_probability_second_red_given_first_red_is_five_ninths_l497_49749

/-- Represents the probability of drawing a red ball as the second ball, given that the first ball drawn was red, from a set of 10 balls containing 6 red balls and 4 white balls. -/
def probability_second_red_given_first_red (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) : ℚ :=
  if total_balls = red_balls + white_balls ∧ red_balls > 0 then
    (red_balls - 1) / (total_balls - 1)
  else
    0

/-- Theorem stating that the probability of drawing a red ball as the second ball, given that the first ball drawn was red, from a set of 10 balls containing 6 red balls and 4 white balls, is 5/9. -/
theorem probability_second_red_given_first_red_is_five_ninths :
  probability_second_red_given_first_red 10 6 4 = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_second_red_given_first_red_is_five_ninths_l497_49749


namespace NUMINAMATH_CALUDE_function_attains_minimum_l497_49783

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsAdditive (f : ℝ → ℝ) : Prop := ∀ x y, f (x + y) = f x + f y

theorem function_attains_minimum (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : IsOdd f) 
  (h_additive : IsAdditive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_ab : a < b) :
  ∀ x ∈ Set.Icc a b, f b ≤ f x :=
sorry

end NUMINAMATH_CALUDE_function_attains_minimum_l497_49783


namespace NUMINAMATH_CALUDE_brendas_age_l497_49781

/-- Given the ages of Addison, Brenda, and Janet, prove that Brenda is 3 years old -/
theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B)     -- Addison's age is four times Brenda's age
  (h2 : J = B + 9)     -- Janet is nine years older than Brenda
  (h3 : A = J)         -- Addison and Janet are twins (same age)
  : B = 3 := by        -- Prove that Brenda's age (B) is 3
sorry


end NUMINAMATH_CALUDE_brendas_age_l497_49781


namespace NUMINAMATH_CALUDE_beach_problem_l497_49782

/-- The number of people in the third row at the beach -/
def third_row_count (total_rows : Nat) (initial_first_row : Nat) (left_first_row : Nat) 
  (initial_second_row : Nat) (left_second_row : Nat) (total_left : Nat) : Nat :=
  total_left - ((initial_first_row - left_first_row) + (initial_second_row - left_second_row))

/-- Theorem: The number of people in the third row is 18 -/
theorem beach_problem : 
  third_row_count 3 24 3 20 5 54 = 18 := by
  sorry

end NUMINAMATH_CALUDE_beach_problem_l497_49782


namespace NUMINAMATH_CALUDE_inverse_function_point_l497_49798

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- State that the graph of f passes through (0, 1)
axiom f_point : f 0 = 1

-- Theorem to prove
theorem inverse_function_point :
  (f_inv 1) + 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l497_49798


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l497_49737

theorem train_platform_crossing_time
  (train_length : ℝ)
  (tree_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 120)
  (h3 : platform_length = 800) :
  let train_speed := train_length / tree_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed = 200 :=
by sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l497_49737


namespace NUMINAMATH_CALUDE_jane_rejection_percentage_l497_49778

theorem jane_rejection_percentage 
  (john_rejection_rate : Real) 
  (total_rejection_rate : Real) 
  (jane_inspection_ratio : Real) :
  john_rejection_rate = 0.005 →
  total_rejection_rate = 0.0075 →
  jane_inspection_ratio = 1.25 →
  ∃ jane_rejection_rate : Real,
    jane_rejection_rate = 0.0095 ∧
    john_rejection_rate * 1 + jane_rejection_rate * jane_inspection_ratio = 
      total_rejection_rate * (1 + jane_inspection_ratio) :=
by sorry

end NUMINAMATH_CALUDE_jane_rejection_percentage_l497_49778


namespace NUMINAMATH_CALUDE_inscribed_cylinder_properties_l497_49770

/-- A right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ
  cylinder_height : ℝ
  /-- The cylinder's diameter equals its height -/
  height_eq_diameter : cylinder_height = 2 * cylinder_radius
  /-- The axes of the cylinder and cone coincide -/
  axes_coincide : True

/-- The space left in the cone above the cylinder -/
def space_above_cylinder (c : InscribedCylinder) : ℝ :=
  c.cone_altitude - c.cylinder_height

theorem inscribed_cylinder_properties (c : InscribedCylinder) 
  (h1 : c.cone_diameter = 16) 
  (h2 : c.cone_altitude = 20) : 
  c.cylinder_radius = 40 / 9 ∧ space_above_cylinder c = 100 / 9 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_properties_l497_49770


namespace NUMINAMATH_CALUDE_pancake_fundraiser_l497_49788

/-- The civic league's pancake breakfast fundraiser problem -/
theorem pancake_fundraiser
  (pancake_price : ℝ)
  (bacon_price : ℝ)
  (pancake_stacks : ℕ)
  (bacon_slices : ℕ)
  (h1 : pancake_price = 4)
  (h2 : bacon_price = 2)
  (h3 : pancake_stacks = 60)
  (h4 : bacon_slices = 90) :
  pancake_price * (pancake_stacks : ℝ) + bacon_price * (bacon_slices : ℝ) = 420 := by
  sorry

end NUMINAMATH_CALUDE_pancake_fundraiser_l497_49788


namespace NUMINAMATH_CALUDE_bean_ratio_l497_49777

/-- Proves that the ratio of green beans to remaining beans after removing red and white beans is 1:1 --/
theorem bean_ratio (total : ℕ) (green : ℕ) : 
  total = 572 →
  green = 143 →
  (total - total / 4 - (total - total / 4) / 3 - green) = green :=
by
  sorry

end NUMINAMATH_CALUDE_bean_ratio_l497_49777


namespace NUMINAMATH_CALUDE_min_value_expression_l497_49700

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 81 ∧
  (a^2 + 9*a*b + 9*b^2 + 3*c^2 = 81 ↔ a = 3 ∧ b = 1 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l497_49700


namespace NUMINAMATH_CALUDE_truck_length_l497_49792

/-- The length of a truck given its speed and tunnel transit time -/
theorem truck_length (tunnel_length : ℝ) (transit_time : ℝ) (speed_mph : ℝ) :
  tunnel_length = 330 →
  transit_time = 6 →
  speed_mph = 45 →
  (speed_mph * 5280 / 3600) * transit_time - tunnel_length = 66 :=
by sorry

end NUMINAMATH_CALUDE_truck_length_l497_49792


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l497_49763

/-- Given two points A(1, y₁) and B(2, y₂) on the graph of y = 2/x, prove that y₁ > y₂ -/
theorem inverse_proportion_comparison (y₁ y₂ : ℝ) :
  y₁ = 2 / 1 → y₂ = 2 / 2 → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l497_49763


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l497_49750

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_positive : ∀ n, 0 < a n) 
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = a n * q) 
  (h_sum_1_2 : a 1 + a 2 = 3/4)
  (h_sum_3_to_6 : a 3 + a 4 + a 5 + a 6 = 15) :
  a 7 + a 8 + a 9 = 112 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l497_49750


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l497_49786

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  36 ∣ n^2 ∧ 1024 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 36 ∣ m^2 → 1024 ∣ m^3 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l497_49786


namespace NUMINAMATH_CALUDE_tom_twice_tim_age_l497_49731

/-- Proves that Tom will be twice Tim's age in 3 years -/
theorem tom_twice_tim_age (tom_age tim_age : ℕ) (x : ℕ) : 
  tom_age + tim_age = 21 → 
  tom_age = 15 → 
  tom_age + x = 2 * (tim_age + x) → 
  x = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_twice_tim_age_l497_49731


namespace NUMINAMATH_CALUDE_base_10_to_base_5_512_l497_49758

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base_10_to_base_5_512 :
  toBase5 512 = [4, 0, 2, 2] :=
sorry

end NUMINAMATH_CALUDE_base_10_to_base_5_512_l497_49758


namespace NUMINAMATH_CALUDE_expected_distinct_values_formula_l497_49717

/-- The number of elements in our set -/
def n : ℕ := 2013

/-- The probability of choosing any specific value -/
def p : ℚ := 1 / n

/-- The probability of not choosing a specific value -/
def q : ℚ := 1 - p

/-- The expected number of distinct values in a set of n elements,
    each chosen independently and randomly from {1, ..., n} -/
def expected_distinct_values : ℚ := n * (1 - q^n)

/-- Theorem stating that the expected number of distinct values
    is equal to the formula derived in the solution -/
theorem expected_distinct_values_formula :
  expected_distinct_values = n * (1 - (n - 1 : ℚ)^n / n^n) :=
sorry

end NUMINAMATH_CALUDE_expected_distinct_values_formula_l497_49717


namespace NUMINAMATH_CALUDE_mart_vegetable_count_l497_49776

/-- The number of cucumbers in the mart -/
def cucumbers : ℕ := 58

/-- The number of carrots in the mart -/
def carrots : ℕ := cucumbers - 24

/-- The number of tomatoes in the mart -/
def tomatoes : ℕ := cucumbers + 49

/-- The number of radishes in the mart -/
def radishes : ℕ := carrots

/-- The total number of vegetables in the mart -/
def total_vegetables : ℕ := cucumbers + carrots + tomatoes + radishes

/-- Theorem stating the total number of vegetables in the mart -/
theorem mart_vegetable_count : total_vegetables = 233 := by sorry

end NUMINAMATH_CALUDE_mart_vegetable_count_l497_49776


namespace NUMINAMATH_CALUDE_first_duck_ate_half_l497_49732

/-- The fraction of bread eaten by the first duck -/
def first_duck_fraction (total_bread pieces_left second_duck_pieces third_duck_pieces : ℕ) : ℚ :=
  let eaten := total_bread - pieces_left
  let first_duck_pieces := eaten - (second_duck_pieces + third_duck_pieces)
  first_duck_pieces / total_bread

/-- Theorem stating the fraction of bread eaten by the first duck -/
theorem first_duck_ate_half :
  first_duck_fraction 100 30 13 7 = 1/2 := by
  sorry

#eval first_duck_fraction 100 30 13 7

end NUMINAMATH_CALUDE_first_duck_ate_half_l497_49732


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l497_49707

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (2 : ℂ) / (1 - Complex.I) = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l497_49707


namespace NUMINAMATH_CALUDE_pyramid_edge_ratio_l497_49779

/-- Represents a pyramid with a cross-section parallel to its base -/
structure Pyramid where
  base_area : ℝ
  cross_section_area : ℝ
  upper_edge_length : ℝ
  lower_edge_length : ℝ
  parallel_cross_section : cross_section_area > 0
  area_ratio : cross_section_area / base_area = 4 / 9

/-- 
Theorem: In a pyramid with a cross-section parallel to its base, 
if the ratio of the cross-sectional area to the base area is 4:9, 
then the ratio of the lengths of the upper and lower parts of the lateral edge is 2:3.
-/
theorem pyramid_edge_ratio (p : Pyramid) : 
  p.upper_edge_length / p.lower_edge_length = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_edge_ratio_l497_49779


namespace NUMINAMATH_CALUDE_max_value_theorem_l497_49787

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  2*x + y ≤ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ 2*x₀ + y₀ = 8 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l497_49787


namespace NUMINAMATH_CALUDE_chess_piece_arrangements_l497_49764

theorem chess_piece_arrangements (n m : ℕ) (hn : n = 9) (hm : m = 6) :
  (Finset.card (Finset.univ : Finset (Fin n → Fin m))) = (14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6) := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_arrangements_l497_49764


namespace NUMINAMATH_CALUDE_division_problem_l497_49791

theorem division_problem (dividend quotient divisor remainder x : ℕ) : 
  remainder = 5 →
  divisor = 3 * quotient →
  dividend = 113 →
  divisor = 3 * remainder + x →
  dividend = divisor * quotient + remainder →
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l497_49791


namespace NUMINAMATH_CALUDE_car_speed_time_relations_l497_49733

/-- Represents the speed and time of a car --/
structure CarData where
  speed : ℝ
  time : ℝ

/-- Given conditions and proof goals for the car problem --/
theorem car_speed_time_relations 
  (x y z : CarData) 
  (h1 : y.speed = 3 * x.speed) 
  (h2 : z.speed = (x.speed + y.speed) / 2) 
  (h3 : x.speed * x.time = y.speed * y.time) 
  (h4 : x.speed * x.time = z.speed * z.time) : 
  z.speed = 2 * x.speed ∧ 
  y.time = x.time / 3 ∧ 
  z.time = x.time / 2 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_time_relations_l497_49733


namespace NUMINAMATH_CALUDE_a_range_l497_49714

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 ≤ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}

-- State the theorem
theorem a_range (a : ℝ) : A ∪ B a = B a → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l497_49714


namespace NUMINAMATH_CALUDE_sqrt_four_squared_times_five_to_sixth_l497_49722

theorem sqrt_four_squared_times_five_to_sixth : Real.sqrt (4^2 * 5^6) = 500 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_four_squared_times_five_to_sixth_l497_49722


namespace NUMINAMATH_CALUDE_square_area_l497_49744

theorem square_area (x : ℝ) : 
  (6 * x - 18 = 3 * x + 9) → 
  (6 * x - 18)^2 = 1296 := by
sorry

end NUMINAMATH_CALUDE_square_area_l497_49744


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l497_49740

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / (x + 1) = 0 ∧ x + 1 ≠ 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l497_49740


namespace NUMINAMATH_CALUDE_super_extra_yield_interest_l497_49724

/-- Calculates the interest earned on a compound interest savings account -/
theorem super_extra_yield_interest
  (principal : ℝ)
  (rate : ℝ)
  (years : ℕ)
  (h_principal : principal = 1500)
  (h_rate : rate = 0.02)
  (h_years : years = 5) :
  ⌊(principal * (1 + rate) ^ years - principal)⌋ = 156 := by
  sorry

end NUMINAMATH_CALUDE_super_extra_yield_interest_l497_49724


namespace NUMINAMATH_CALUDE_stating_judge_assignment_count_l497_49761

/-- Represents the number of judges from each grade -/
def judges_per_grade : ℕ := 2

/-- Represents the number of grades -/
def num_grades : ℕ := 3

/-- Represents the number of courts -/
def num_courts : ℕ := 3

/-- Represents the number of judges per court -/
def judges_per_court : ℕ := 2

/-- 
Theorem stating that the number of ways to assign judges to courts 
under the given conditions is 48
-/
theorem judge_assignment_count : 
  (judges_per_grade ^ num_courts) * (Nat.factorial num_courts) = 48 := by
  sorry


end NUMINAMATH_CALUDE_stating_judge_assignment_count_l497_49761


namespace NUMINAMATH_CALUDE_quadratic_equation_conversion_l497_49710

theorem quadratic_equation_conversion :
  ∃ (a b c : ℝ), a = 1 ∧ b = -5 ∧ c = 3 ∧
  ∀ x, (x - 1)^2 = 3*x - 2 ↔ a*x^2 + b*x + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_conversion_l497_49710


namespace NUMINAMATH_CALUDE_inscribed_circle_square_area_l497_49760

/-- Given a circle inscribed in a square, if the circle's area is 314 square inches,
    then the square's area is 400 square inches. -/
theorem inscribed_circle_square_area :
  ∀ (circle_radius square_side : ℝ),
  circle_radius > 0 →
  square_side > 0 →
  circle_radius * 2 = square_side →
  π * circle_radius^2 = 314 →
  square_side^2 = 400 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_square_area_l497_49760
