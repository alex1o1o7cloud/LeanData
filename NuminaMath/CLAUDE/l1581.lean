import Mathlib

namespace digit_replacement_theorem_l1581_158103

def first_number : ℕ := 631927
def second_number : ℕ := 590265
def given_sum : ℕ := 1192192

def replace_digit (n : ℕ) (d e : ℕ) : ℕ := 
  sorry

theorem digit_replacement_theorem :
  ∃ (d e : ℕ), d ≠ e ∧ d < 10 ∧ e < 10 ∧
  (replace_digit first_number d e) + (replace_digit second_number d e) = 
    replace_digit given_sum d e ∧
  d + e = 6 := by
  sorry

end digit_replacement_theorem_l1581_158103


namespace broth_per_serving_is_two_point_five_l1581_158194

/-- Represents the number of cups in one pint -/
def cups_per_pint : ℚ := 2

/-- Represents the number of servings -/
def num_servings : ℕ := 8

/-- Represents the number of pints of vegetables and broth combined for all servings -/
def total_pints : ℚ := 14

/-- Represents the number of cups of vegetables in one serving -/
def vegetables_per_serving : ℚ := 1

/-- Calculates the number of cups of broth in one serving of soup -/
def broth_per_serving : ℚ :=
  (total_pints * cups_per_pint - num_servings * vegetables_per_serving) / num_servings

theorem broth_per_serving_is_two_point_five :
  broth_per_serving = 2.5 := by sorry

end broth_per_serving_is_two_point_five_l1581_158194


namespace carl_driving_hours_l1581_158160

/-- 
Given Carl's initial daily driving hours and additional weekly hours after promotion, 
prove that the total hours he will drive in two weeks is equal to 40 hours.
-/
theorem carl_driving_hours (initial_daily_hours : ℝ) (additional_weekly_hours : ℝ) : 
  initial_daily_hours = 2 ∧ additional_weekly_hours = 6 → 
  (initial_daily_hours * 7 + additional_weekly_hours) * 2 = 40 := by
sorry

end carl_driving_hours_l1581_158160


namespace combined_ratio_theorem_l1581_158114

/-- Represents the ratio of liquids in a vessel -/
structure LiquidRatio :=
  (water : ℚ)
  (milk : ℚ)
  (syrup : ℚ)

/-- Represents a vessel with its volume and liquid ratio -/
structure Vessel :=
  (volume : ℚ)
  (ratio : LiquidRatio)

def combine_vessels (vessels : List Vessel) : LiquidRatio :=
  let total_water := vessels.map (λ v => v.volume * v.ratio.water) |>.sum
  let total_milk := vessels.map (λ v => v.volume * v.ratio.milk) |>.sum
  let total_syrup := vessels.map (λ v => v.volume * v.ratio.syrup) |>.sum
  { water := total_water, milk := total_milk, syrup := total_syrup }

theorem combined_ratio_theorem (v1 v2 v3 : Vessel)
  (h1 : v1.volume = 3 ∧ v2.volume = 5 ∧ v3.volume = 7)
  (h2 : v1.ratio = { water := 1/6, milk := 1/3, syrup := 1/2 })
  (h3 : v2.ratio = { water := 2/7, milk := 4/7, syrup := 1/7 })
  (h4 : v3.ratio = { water := 1/2, milk := 1/6, syrup := 1/3 }) :
  let combined := combine_vessels [v1, v2, v3]
  combined.water / (combined.water + combined.milk + combined.syrup) = 228 / 630 ∧
  combined.milk / (combined.water + combined.milk + combined.syrup) = 211 / 630 ∧
  combined.syrup / (combined.water + combined.milk + combined.syrup) = 191 / 630 := by
  sorry

#check combined_ratio_theorem

end combined_ratio_theorem_l1581_158114


namespace school_fundraiser_distribution_l1581_158107

theorem school_fundraiser_distribution (total_amount : ℚ) (num_charities : ℕ) 
  (h1 : total_amount = 3109)
  (h2 : num_charities = 25) :
  total_amount / num_charities = 124.36 := by
  sorry

end school_fundraiser_distribution_l1581_158107


namespace profit_difference_l1581_158161

def chocolate_cakes_made : ℕ := 40
def vanilla_cakes_made : ℕ := 35
def strawberry_cakes_made : ℕ := 28
def pastries_made : ℕ := 153

def chocolate_cake_price : ℕ := 10
def vanilla_cake_price : ℕ := 12
def strawberry_cake_price : ℕ := 15
def pastry_price : ℕ := 5

def chocolate_cakes_sold : ℕ := 30
def vanilla_cakes_sold : ℕ := 25
def strawberry_cakes_sold : ℕ := 20
def pastries_sold : ℕ := 106

def total_cake_revenue : ℕ := 
  chocolate_cakes_sold * chocolate_cake_price +
  vanilla_cakes_sold * vanilla_cake_price +
  strawberry_cakes_sold * strawberry_cake_price

def total_pastry_revenue : ℕ := pastries_sold * pastry_price

theorem profit_difference : total_cake_revenue - total_pastry_revenue = 370 := by
  sorry

end profit_difference_l1581_158161


namespace shaded_region_perimeter_l1581_158183

/-- Given three identical circles with circumference 48, where each circle touches the other two,
    and the arcs in the shaded region each subtend an angle of 90 degrees at the center of their
    respective circles, the perimeter of the shaded region is equal to 36. -/
theorem shaded_region_perimeter (circle_circumference : ℝ) (arc_angle : ℝ) : 
  circle_circumference = 48 → 
  arc_angle = 90 →
  (3 * (arc_angle / 360) * circle_circumference) = 36 := by
  sorry

#check shaded_region_perimeter

end shaded_region_perimeter_l1581_158183


namespace complement_A_union_B_eq_greater_than_neg_one_l1581_158185

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x | x ≥ 1}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- State the theorem
theorem complement_A_union_B_eq_greater_than_neg_one :
  (Set.compl A) ∪ B = {x : ℝ | x > -1} := by sorry

end complement_A_union_B_eq_greater_than_neg_one_l1581_158185


namespace rabbit_fraction_l1581_158177

theorem rabbit_fraction (initial_cage : ℕ) (added : ℕ) (park : ℕ) : 
  initial_cage = 13 → added = 7 → park = 60 → 
  (initial_cage + added : ℚ) / park = 1/3 := by
  sorry

end rabbit_fraction_l1581_158177


namespace hand_mitt_cost_is_14_l1581_158173

/-- The cost of cooking gear for Eve's nieces --/
def cooking_gear_cost (hand_mitt_cost : ℝ) : Prop :=
  let apron_cost : ℝ := 16
  let utensils_cost : ℝ := 10
  let knife_cost : ℝ := 2 * utensils_cost
  let total_cost_per_niece : ℝ := hand_mitt_cost + apron_cost + utensils_cost + knife_cost
  let discount_rate : ℝ := 0.75
  let number_of_nieces : ℕ := 3
  let total_spent : ℝ := 135
  discount_rate * (number_of_nieces : ℝ) * total_cost_per_niece = total_spent

theorem hand_mitt_cost_is_14 :
  ∃ (hand_mitt_cost : ℝ), cooking_gear_cost hand_mitt_cost ∧ hand_mitt_cost = 14 := by
  sorry

end hand_mitt_cost_is_14_l1581_158173


namespace speed_calculation_l1581_158109

/-- Given a distance of 3.0 miles and a time of 1.5 hours, prove that the speed is 2.0 miles per hour. -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 3.0) 
    (h2 : time = 1.5) 
    (h3 : speed = distance / time) : speed = 2.0 := by
  sorry

end speed_calculation_l1581_158109


namespace quadratic_roots_imply_composite_l1581_158176

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ

/-- The roots of a quadratic polynomial -/
structure Roots where
  x₁ : ℤ
  x₂ : ℤ

/-- Predicate to check if a number is composite -/
def IsComposite (n : ℤ) : Prop :=
  ∃ m k : ℤ, m ≠ 1 ∧ m ≠ -1 ∧ k ≠ 1 ∧ k ≠ -1 ∧ n = m * k

/-- Main theorem -/
theorem quadratic_roots_imply_composite
  (p : QuadraticPolynomial)
  (r : Roots)
  (h₁ : r.x₁ * r.x₁ + p.a * r.x₁ + p.b = 0)
  (h₂ : r.x₂ * r.x₂ + p.a * r.x₂ + p.b = 0)
  (h₃ : |r.x₁| > 2)
  (h₄ : |r.x₂| > 2) :
  IsComposite (p.a + p.b + 1) := by
  sorry

end quadratic_roots_imply_composite_l1581_158176


namespace no_same_line_l1581_158166

/-- Two lines are the same if and only if they have the same slope and y-intercept -/
def same_line (m1 m2 b1 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

/-- The first line equation: ax + 3y + d = 0 -/
def line1 (a d : ℝ) (x y : ℝ) : Prop := a * x + 3 * y + d = 0

/-- The second line equation: 4x - ay + 8 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := 4 * x - a * y + 8 = 0

/-- Theorem: There are no real values of a and d such that ax+3y+d=0 and 4x-ay+8=0 represent the same line -/
theorem no_same_line : ¬∃ (a d : ℝ), ∀ (x y : ℝ), line1 a d x y ↔ line2 a x y := by
  sorry

end no_same_line_l1581_158166


namespace simplify_and_rationalize_l1581_158137

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l1581_158137


namespace hotel_moves_2_8_l1581_158187

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of ways guests can move in a 2 × n grid hotel -/
def hotelMoves (n : ℕ) : ℕ := (fib (n + 1)) ^ 2

/-- Theorem: The number of ways guests can move in a 2 × 8 grid hotel is 3025 -/
theorem hotel_moves_2_8 : hotelMoves 8 = 3025 := by
  sorry

end hotel_moves_2_8_l1581_158187


namespace unique_solution_sin_cos_equation_l1581_158119

theorem unique_solution_sin_cos_equation :
  ∃! x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ Real.sin (Real.cos x) = Real.cos (Real.sin x) := by
  sorry

end unique_solution_sin_cos_equation_l1581_158119


namespace faster_train_speed_l1581_158136

/-- Proves that the speed of the faster train is 20/3 m/s given the specified conditions -/
theorem faster_train_speed
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h_length : train_length = 100)
  (h_time : crossing_time = 20)
  (h_speed_ratio : ∃ (v : ℝ), v > 0 ∧ faster_speed = 2 * v ∧ slower_speed = v)
  (h_relative_speed : relative_speed = faster_speed + slower_speed)
  (h_distance : total_distance = 2 * train_length)
  (h_speed_formula : relative_speed = total_distance / crossing_time) :
  faster_speed = 20 / 3 :=
sorry

end faster_train_speed_l1581_158136


namespace equation_solution_l1581_158188

theorem equation_solution :
  let f : ℝ → ℝ := λ x => x^2 - |x| - 1
  ∀ x : ℝ, f x = 0 ↔ x = (-1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 := by
  sorry

end equation_solution_l1581_158188


namespace comics_in_box_l1581_158100

theorem comics_in_box (pages_per_comic : ℕ) (found_pages : ℕ) (untorn_comics : ℕ) : 
  pages_per_comic = 25 →
  found_pages = 150 →
  untorn_comics = 5 →
  (found_pages / pages_per_comic + untorn_comics : ℕ) = 11 :=
by sorry

end comics_in_box_l1581_158100


namespace modulus_of_z_l1581_158123

-- Define the complex number z
def z : ℂ := 3 + 4 * Complex.I

-- State the theorem
theorem modulus_of_z : Complex.abs z = 5 := by sorry

end modulus_of_z_l1581_158123


namespace lineup_selection_theorem_l1581_158157

/-- The number of ways to select a lineup of 6 players from a team of 15 players -/
def lineup_selection_ways : ℕ := 3603600

/-- The size of the basketball team -/
def team_size : ℕ := 15

/-- The number of positions to be filled -/
def positions_to_fill : ℕ := 6

theorem lineup_selection_theorem :
  (Finset.range team_size).card.factorial / 
  ((team_size - positions_to_fill).factorial) = lineup_selection_ways :=
sorry

end lineup_selection_theorem_l1581_158157


namespace sphere_volume_equal_surface_area_cube_l1581_158151

/-- The volume of a sphere with surface area equal to a cube of side length 2 -/
theorem sphere_volume_equal_surface_area_cube (r : ℝ) : 
  (4 * Real.pi * r^2 = 6 * 2^2) → 
  ((4 / 3) * Real.pi * r^3 = (8 * Real.sqrt 6) / Real.sqrt Real.pi) :=
by sorry

end sphere_volume_equal_surface_area_cube_l1581_158151


namespace largest_n_for_factorization_l1581_158175

/-- 
Theorem: The largest value of n such that 3x^2 + nx + 72 can be factored 
as the product of two linear factors with integer coefficients is 217.
-/
theorem largest_n_for_factorization : 
  ∃ (n : ℤ), n = 217 ∧ 
  (∀ m : ℤ, m > n → 
    ¬∃ (a b c d : ℤ), 3 * X^2 + m * X + 72 = (a * X + b) * (c * X + d)) ∧
  (∃ (a b c d : ℤ), 3 * X^2 + n * X + 72 = (a * X + b) * (c * X + d)) :=
sorry

end largest_n_for_factorization_l1581_158175


namespace min_a_value_l1581_158120

noncomputable def f (x : ℝ) : ℝ := (2 * 2023^x) / (2023^x + 1)

theorem min_a_value (a : ℝ) :
  (∀ x : ℝ, x > 0 → f (a * Real.exp x) ≥ 2 - f (Real.log a - Real.log x)) →
  a ≥ 1 / Real.exp 1 ∧ ∀ b : ℝ, (∀ x : ℝ, x > 0 → f (b * Real.exp x) ≥ 2 - f (Real.log b - Real.log x)) → b ≥ 1 / Real.exp 1 :=
by sorry

end min_a_value_l1581_158120


namespace residue_calculation_l1581_158171

theorem residue_calculation : (207 * 13 - 18 * 8 + 5) % 16 = 8 := by
  sorry

end residue_calculation_l1581_158171


namespace negative_two_less_than_negative_three_halves_l1581_158186

theorem negative_two_less_than_negative_three_halves : -2 < -(3/2) := by
  sorry

end negative_two_less_than_negative_three_halves_l1581_158186


namespace golden_ratio_function_l1581_158135

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_function (f : ℝ → ℝ) :
  (∀ x > 0, Monotone f) →
  (∀ x > 0, f x > 0) →
  (∀ x > 0, f x * f (f x + 1 / x) = 1) →
  f 1 = φ := by
  sorry

end golden_ratio_function_l1581_158135


namespace sqrt_64_minus_neg_2_cubed_equals_16_l1581_158172

theorem sqrt_64_minus_neg_2_cubed_equals_16 : 
  Real.sqrt 64 - (-2)^3 = 16 := by
  sorry

end sqrt_64_minus_neg_2_cubed_equals_16_l1581_158172


namespace area_FYH_specific_l1581_158111

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Calculates the area of triangle FYH in a trapezoid -/
def area_FYH (t : Trapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of triangle FYH in the specific trapezoid -/
theorem area_FYH_specific : 
  let t : Trapezoid := { base1 := 24, base2 := 36, area := 360 }
  area_FYH t = 86.4 := by
  sorry

end area_FYH_specific_l1581_158111


namespace toms_total_amount_l1581_158104

/-- Tom's initial amount in dollars -/
def initial_amount : ℕ := 74

/-- Amount Tom earned from washing cars in dollars -/
def earned_amount : ℕ := 86

/-- Theorem stating Tom's total amount after washing cars -/
theorem toms_total_amount : initial_amount + earned_amount = 160 := by
  sorry

end toms_total_amount_l1581_158104


namespace circle_center_and_radius_prove_center_and_radius_l1581_158133

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-2, 0)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem circle_center_and_radius :
  ∀ (x y : ℝ), circle_equation x y ↔ (x + 2)^2 + y^2 = 4 :=
by sorry

-- Prove that the center and radius are correct
theorem prove_center_and_radius :
  (∀ (x y : ℝ), circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by sorry

end circle_center_and_radius_prove_center_and_radius_l1581_158133


namespace additional_sleep_january_l1581_158199

def sleep_december : ℝ := 6.5
def sleep_january : ℝ := 8.5
def days_in_month : ℕ := 31

theorem additional_sleep_january : 
  (sleep_january - sleep_december) * days_in_month = 62 := by
  sorry

end additional_sleep_january_l1581_158199


namespace min_value_expression_l1581_158142

theorem min_value_expression (a b c k m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hk : k > 0) (hm : m > 0) (hn : n > 0) : 
  (k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a ≥ 6 * k ∧
  ((k * a + m * b) / c + (m * a + n * c) / b + (n * b + k * c) / a = 6 * k ↔ 
    k = m ∧ m = n ∧ a = b ∧ b = c) :=
by sorry

end min_value_expression_l1581_158142


namespace dog_legs_on_street_l1581_158121

theorem dog_legs_on_street (total_animals : ℕ) (cat_fraction : ℚ) (dog_legs : ℕ) : 
  total_animals = 300 →
  cat_fraction = 2 / 3 →
  dog_legs = 4 →
  (total_animals * (1 - cat_fraction) : ℚ).num * dog_legs = 400 :=
by sorry

end dog_legs_on_street_l1581_158121


namespace circle_ratio_l1581_158165

theorem circle_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (h_area : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
  sorry

end circle_ratio_l1581_158165


namespace ring_arrangement_count_l1581_158144

/-- The number of possible six-ring arrangements on four fingers -/
def ring_arrangements : ℕ := 618854400

/-- The number of distinguishable rings -/
def total_rings : ℕ := 10

/-- The number of rings to be arranged -/
def arranged_rings : ℕ := 6

/-- The number of fingers (excluding thumb) -/
def fingers : ℕ := 4

theorem ring_arrangement_count :
  ring_arrangements = (total_rings.choose arranged_rings) * (arranged_rings.factorial) * (fingers ^ arranged_rings) :=
sorry

end ring_arrangement_count_l1581_158144


namespace not_red_card_probability_l1581_158131

/-- Given a deck of cards where the odds of drawing a red card are 1:3,
    the probability of drawing a card that is not red is 3/4. -/
theorem not_red_card_probability (odds : ℚ) (h : odds = 1/3) :
  1 - odds / (1 + odds) = 3/4 := by
  sorry

end not_red_card_probability_l1581_158131


namespace spatial_geometry_l1581_158181

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Plane → Plane → Prop)
variable (intersect : Plane → Plane → Line)
variable (contains : Plane → Line → Prop)
variable (linePerpendicular : Line → Line → Prop)
variable (lineParallel : Line → Line → Prop)
variable (planePerpendicular : Line → Plane → Prop)

-- State the theorem
theorem spatial_geometry 
  (α β : Plane) (l m n : Line) 
  (h1 : perpendicular α β)
  (h2 : intersect α β = l)
  (h3 : contains α m)
  (h4 : contains β n)
  (h5 : linePerpendicular m n) :
  (lineParallel n l → planePerpendicular m β) ∧
  (planePerpendicular m β ∨ planePerpendicular n α) :=
sorry

end spatial_geometry_l1581_158181


namespace line_through_point_parallel_to_line_l1581_158134

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (p : Point) 
  (l1 : Line) 
  (l2 : Line) : 
  p.liesOn l2 ∧ l2.isParallelTo l1 → 
  l2 = Line.mk 1 (-2) 7 :=
by
  sorry

#check line_through_point_parallel_to_line 
  (Point.mk (-1) 3) 
  (Line.mk 1 (-2) 3) 
  (Line.mk 1 (-2) 7)

end line_through_point_parallel_to_line_l1581_158134


namespace no_real_solutions_l1581_158158

theorem no_real_solutions :
  ¬∃ (x : ℝ), (x^4 + 3*x^3)/(x^2 + 3*x + 1) + x = -7 := by
sorry

end no_real_solutions_l1581_158158


namespace S_bounds_l1581_158128

def is_permutation (x : Fin 10 → ℕ) : Prop :=
  ∀ n : ℕ, n < 10 → ∃ i : Fin 10, x i = n

def S (x : Fin 10 → ℕ) : ℕ :=
  x 1 + x 2 + x 3 + x 4 + x 6 + x 7 + x 8

theorem S_bounds (x : Fin 10 → ℕ) (h : is_permutation x) : 
  21 ≤ S x ∧ S x ≤ 25 := by
  sorry

end S_bounds_l1581_158128


namespace two_person_subcommittees_l1581_158139

theorem two_person_subcommittees (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end two_person_subcommittees_l1581_158139


namespace system_solution_existence_l1581_158191

theorem system_solution_existence (k : ℝ) :
  (∃ (x y : ℝ), y = k * x + 4 ∧ y = (3 * k - 2) * x + 5) ↔ k ≠ 1 := by
  sorry

end system_solution_existence_l1581_158191


namespace expression_simplification_l1581_158110

theorem expression_simplification (x y a b c : ℝ) :
  (2 - y) * 24 * (x - y) + 2 * ((a - 2 - 3 * c) * a - 2 * b + c) = 2 + 4 * b^2 - a * b - c^2 := by
  sorry

end expression_simplification_l1581_158110


namespace coffee_stock_solution_l1581_158164

/-- Represents the coffee stock problem -/
def coffee_stock_problem (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) (final_decaf_percent : ℝ) : Prop :=
  ∃ (second_batch : ℝ),
    second_batch > 0 ∧
    (initial_stock * initial_decaf_percent + second_batch * second_batch_decaf_percent) / 
    (initial_stock + second_batch) = final_decaf_percent

/-- The solution to the coffee stock problem -/
theorem coffee_stock_solution :
  coffee_stock_problem 400 0.30 0.60 0.36 → 
  ∃ (second_batch : ℝ), second_batch = 100 := by
  sorry

end coffee_stock_solution_l1581_158164


namespace borrowed_amount_l1581_158118

theorem borrowed_amount (P : ℝ) (interest_rate : ℝ) (total_repayment : ℝ) : 
  interest_rate = 0.1 →
  total_repayment = 1320 →
  total_repayment = P * (1 + interest_rate) →
  P = 1200 := by
  sorry

end borrowed_amount_l1581_158118


namespace arithmetic_square_root_of_nine_l1581_158113

theorem arithmetic_square_root_of_nine :
  ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 9 ∧ (∀ y : ℝ, y ≥ 0 ∧ y^2 = 9 → y = x) ∧ x = 3 :=
by sorry

end arithmetic_square_root_of_nine_l1581_158113


namespace existence_of_special_binary_number_l1581_158106

/-- Represents a binary number as a list of booleans -/
def BinaryNumber := List Bool

/-- Generates all n-digit binary numbers -/
def allNDigitBinaryNumbers (n : Nat) : List BinaryNumber :=
  sorry

/-- Checks if a binary number is a substring of another binary number -/
def isSubstring (sub target : BinaryNumber) : Bool :=
  sorry

/-- Checks if all n-digit binary numbers are substrings of T -/
def allNDigitNumbersAreSubstrings (T : BinaryNumber) (n : Nat) : Prop :=
  ∀ sub, sub ∈ allNDigitBinaryNumbers n → isSubstring sub T

/-- Checks if all n-digit substrings of T are distinct -/
def allNDigitSubstringsAreDistinct (T : BinaryNumber) (n : Nat) : Prop :=
  sorry

theorem existence_of_special_binary_number (n : Nat) :
  ∃ T : BinaryNumber,
    T.length = 2^n + (n - 1) ∧
    allNDigitNumbersAreSubstrings T n ∧
    allNDigitSubstringsAreDistinct T n :=
  sorry

end existence_of_special_binary_number_l1581_158106


namespace josette_purchase_cost_l1581_158149

/-- Calculates the total cost of mineral water bottles with a discount --/
def total_cost (small_count : ℕ) (large_count : ℕ) (small_price : ℚ) (large_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_count := small_count + large_count
  let subtotal := small_count * small_price + large_count * large_price
  if total_count ≥ 5 then
    subtotal * (1 - discount_rate)
  else
    subtotal

/-- The total cost for Josette's purchase is €8.37 --/
theorem josette_purchase_cost :
  total_cost 3 2 (3/2) (12/5) (1/10) = 837/100 := by sorry

end josette_purchase_cost_l1581_158149


namespace team_improvements_minimum_days_team_a_l1581_158152

-- Define the problem parameters
def team_a_rate : ℝ := 15
def team_b_rate : ℝ := 10
def total_days : ℝ := 25
def total_length : ℝ := 300
def team_a_cost : ℝ := 0.6
def team_b_cost : ℝ := 0.8
def max_total_cost : ℝ := 18

-- Theorem for part 1
theorem team_improvements :
  ∃ (x y : ℝ),
    x + y = total_length ∧
    x / team_a_rate + y / team_b_rate = total_days ∧
    x = 150 ∧ y = 150 := by sorry

-- Theorem for part 2
theorem minimum_days_team_a :
  ∃ (m : ℝ),
    m ≥ 10 ∧
    ∀ (n : ℝ),
      n < 10 →
      team_a_cost * n + team_b_cost * ((total_length - team_a_rate * n) / team_b_rate) > max_total_cost := by sorry

end team_improvements_minimum_days_team_a_l1581_158152


namespace perfect_square_sum_l1581_158174

theorem perfect_square_sum (a b : ℤ) 
  (h : ∀ (m n : ℕ), ∃ (k : ℕ), a * m^2 + b * n^2 = k^2) : 
  a * b = 0 := by
  sorry

end perfect_square_sum_l1581_158174


namespace second_range_lower_limit_l1581_158184

theorem second_range_lower_limit (x y : ℝ) 
  (h1 : 3 < x) (h2 : x < 8) (h3 : x > y) (h4 : x < 10) (h5 : x = 7) : 
  3 < y ∧ y ≤ 7 := by
  sorry

end second_range_lower_limit_l1581_158184


namespace derivative_of_sine_function_l1581_158129

open Real

theorem derivative_of_sine_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => 3 * sin (2 * x - π / 6)
  deriv y x = 6 * cos (2 * x - π / 6) := by
sorry

end derivative_of_sine_function_l1581_158129


namespace train_a_speed_l1581_158132

/-- The speed of Train A in miles per hour -/
def speed_train_a : ℝ := 30

/-- The speed of Train B in miles per hour -/
def speed_train_b : ℝ := 36

/-- The time difference in hours between Train A and Train B's departure -/
def time_difference : ℝ := 2

/-- The distance in miles at which Train B overtakes Train A -/
def overtake_distance : ℝ := 360

/-- Theorem stating that the speed of Train A is 30 mph given the conditions -/
theorem train_a_speed :
  ∃ (t : ℝ), 
    t > time_difference ∧
    speed_train_a * t = overtake_distance ∧
    speed_train_b * (t - time_difference) = overtake_distance ∧
    speed_train_a = 30 := by
  sorry

end train_a_speed_l1581_158132


namespace average_attendance_theorem_l1581_158170

/-- Represents the attendance data for a week -/
structure WeekAttendance where
  totalStudents : ℕ
  mondayAbsence : ℚ
  tuesdayAbsence : ℚ
  wednesdayAbsence : ℚ
  thursdayAbsence : ℚ
  fridayAbsence : ℚ

/-- Calculates the average number of students present in a week -/
def averageAttendance (w : WeekAttendance) : ℚ :=
  let mondayPresent := w.totalStudents * (1 - w.mondayAbsence)
  let tuesdayPresent := w.totalStudents * (1 - w.tuesdayAbsence)
  let wednesdayPresent := w.totalStudents * (1 - w.wednesdayAbsence)
  let thursdayPresent := w.totalStudents * (1 - w.thursdayAbsence)
  let fridayPresent := w.totalStudents * (1 - w.fridayAbsence)
  (mondayPresent + tuesdayPresent + wednesdayPresent + thursdayPresent + fridayPresent) / 5

theorem average_attendance_theorem (w : WeekAttendance) 
    (h1 : w.totalStudents = 50)
    (h2 : w.mondayAbsence = 1/10)
    (h3 : w.tuesdayAbsence = 3/25)
    (h4 : w.wednesdayAbsence = 3/20)
    (h5 : w.thursdayAbsence = 1/12.5)
    (h6 : w.fridayAbsence = 1/20) : 
  averageAttendance w = 45 := by
sorry

end average_attendance_theorem_l1581_158170


namespace geometric_sequence_ratio_l1581_158122

/-- Given a geometric sequence {a_n} with common ratio q and sum of first n terms S_n,
    if a_5 = 2S_4 + 3 and a_6 = 2S_5 + 3, then q = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  (∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- Definition of sum of geometric sequence
  a 5 = 2 * S 4 + 3 →
  a 6 = 2 * S 5 + 3 →
  q = 3 := by
sorry


end geometric_sequence_ratio_l1581_158122


namespace ceiling_product_equation_l1581_158167

theorem ceiling_product_equation : ∃ x : ℝ, (⌈x⌉ : ℝ) * x = 204 ∧ x = 13.6 := by
  sorry

end ceiling_product_equation_l1581_158167


namespace parallel_lines_a_value_l1581_158116

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, x + 2*a*y - 1 = 0 ↔ (a-1)*x + a*y + 1 = 0) → 
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ + 2*a*y₁ - 1 = 0 ∧ (a-1)*x₂ + a*y₂ + 1 = 0 ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)) →
  a = 3/2 :=
by sorry

end parallel_lines_a_value_l1581_158116


namespace propositions_p_and_q_are_true_l1581_158126

theorem propositions_p_and_q_are_true :
  (∀ (m : ℝ), (∀ (x : ℝ), x^2 + x + m > 0) → m > 1/4) ∧
  (∀ (A B C : ℝ), 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
    (A > B ↔ Real.sin A > Real.sin B)) := by
  sorry

end propositions_p_and_q_are_true_l1581_158126


namespace female_fox_terriers_count_l1581_158180

theorem female_fox_terriers_count 
  (total_dogs : ℕ) 
  (total_females : ℕ) 
  (total_fox_terriers : ℕ) 
  (male_shih_tzus : ℕ) 
  (h1 : total_dogs = 2012)
  (h2 : total_females = 1110)
  (h3 : total_fox_terriers = 1506)
  (h4 : male_shih_tzus = 202) :
  total_fox_terriers - (total_dogs - total_females - male_shih_tzus) = 806 :=
by
  sorry

end female_fox_terriers_count_l1581_158180


namespace max_value_w_l1581_158159

theorem max_value_w (p q : ℝ) (w : ℝ) 
  (hw : w = Real.sqrt (2 * p - q) + Real.sqrt (3 * q - 2 * p) + Real.sqrt (6 - 2 * q))
  (h1 : 2 * p - q ≥ 0)
  (h2 : 3 * q - 2 * p ≥ 0)
  (h3 : 6 - 2 * q ≥ 0) :
  w ≤ 3 * Real.sqrt 2 := by
  sorry

end max_value_w_l1581_158159


namespace max_a_value_l1581_158196

def is_lattice_point (x y : ℤ) : Prop := True

def line_passes_through_lattice_point (m : ℚ) : Prop :=
  ∃ x y : ℤ, 0 < x ∧ x ≤ 50 ∧ is_lattice_point x y ∧ y = m * x + 5

theorem max_a_value :
  ∀ a : ℚ, (∀ m : ℚ, 2/3 < m → m < a → ¬line_passes_through_lattice_point m) →
    a ≤ 35/51 :=
sorry

end max_a_value_l1581_158196


namespace original_salary_approximation_l1581_158101

/-- Calculates the final salary after applying a sequence of percentage changes --/
def final_salary (original : ℝ) : ℝ :=
  original * 1.12 * 0.93 * 1.09 * 0.94

/-- Theorem stating that the original salary is approximately 981.47 --/
theorem original_salary_approximation :
  ∃ (S : ℝ), S > 0 ∧ final_salary S = 1212 ∧ abs (S - 981.47) < 0.01 := by
  sorry

end original_salary_approximation_l1581_158101


namespace max_submerged_cubes_is_five_l1581_158190

/-- Represents the properties of the cylinder and cubes -/
structure CylinderAndCubes where
  cylinder_diameter : ℝ
  initial_water_height : ℝ
  cube_edge_length : ℝ

/-- Calculates the maximum number of cubes that can be submerged -/
def max_submerged_cubes (props : CylinderAndCubes) : ℕ :=
  -- Implementation details omitted
  sorry

/-- The main theorem stating the maximum number of submerged cubes -/
theorem max_submerged_cubes_is_five (props : CylinderAndCubes) 
  (h1 : props.cylinder_diameter = 2.9)
  (h2 : props.initial_water_height = 4)
  (h3 : props.cube_edge_length = 2) :
  max_submerged_cubes props = 5 := by
  sorry

#check max_submerged_cubes_is_five

end max_submerged_cubes_is_five_l1581_158190


namespace sum_of_roots_quadratic_l1581_158102

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 6*x₁ + 5 = 0) → (x₂^2 - 6*x₂ + 5 = 0) → x₁ + x₂ = 6 :=
by
  sorry

end sum_of_roots_quadratic_l1581_158102


namespace remaining_integers_l1581_158179

theorem remaining_integers (T : Finset ℕ) : 
  T = Finset.range 100 →
  (Finset.filter (fun n => ¬(n % 2 = 0 ∨ n % 3 = 0 ∨ n % 5 = 0)) T).card = 26 :=
by sorry

end remaining_integers_l1581_158179


namespace provisions_last_20_days_l1581_158138

/-- Calculates the number of days provisions will last after reinforcement -/
def provisions_duration (initial_men : ℕ) (initial_days : ℕ) (days_passed : ℕ) (reinforcement : ℕ) : ℚ :=
  let total_provisions := initial_men * initial_days
  let remaining_provisions := total_provisions - (initial_men * days_passed)
  let new_total_men := initial_men + reinforcement
  remaining_provisions / new_total_men

/-- Proves that given the initial conditions, the provisions will last for 20 days after reinforcement -/
theorem provisions_last_20_days :
  provisions_duration 2000 54 21 1300 = 20 := by
  sorry

end provisions_last_20_days_l1581_158138


namespace total_fish_l1581_158143

theorem total_fish (lilly_fish rosy_fish : ℕ) 
  (h1 : lilly_fish = 10) 
  (h2 : rosy_fish = 8) : 
  lilly_fish + rosy_fish = 18 := by
  sorry

end total_fish_l1581_158143


namespace inclination_angle_of_line_l1581_158162

/-- The inclination angle of a line is the angle between the positive x-axis and the line, 
    measured counterclockwise. -/
def inclination_angle (a b c : ℝ) : ℝ := sorry

/-- The equation of the line is ax + by + c = 0 -/
def is_line_equation (a b c : ℝ) : Prop := sorry

theorem inclination_angle_of_line :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := -5
  is_line_equation a b c →
  inclination_angle a b c = 135 * (π / 180) := by sorry

end inclination_angle_of_line_l1581_158162


namespace arithmetic_sequence_sum_l1581_158108

theorem arithmetic_sequence_sum (n : ℕ) (s : ℕ → ℝ) :
  (∀ k, s (k + 1) - s k = s (k + 2) - s (k + 1)) →  -- arithmetic sequence condition
  s n = 48 →                                        -- sum of first n terms
  s (2 * n) = 60 →                                  -- sum of first 2n terms
  s (3 * n) = 36 :=                                 -- sum of first 3n terms
by sorry

end arithmetic_sequence_sum_l1581_158108


namespace regular_polygon_distance_sum_l1581_158156

theorem regular_polygon_distance_sum (n : ℕ) (h : ℝ) (h_list : List ℝ) :
  n > 2 →
  h > 0 →
  h_list.length = n →
  (∀ x ∈ h_list, x > 0) →
  h_list.sum = n * h :=
by sorry

end regular_polygon_distance_sum_l1581_158156


namespace factorial_ratio_l1581_158182

theorem factorial_ratio : Nat.factorial 52 / Nat.factorial 50 = 2652 := by
  sorry

end factorial_ratio_l1581_158182


namespace xy_inequality_l1581_158169

theorem xy_inequality (x y : ℝ) (n : ℕ) (hx : x > 0) (hy : y > 0) :
  x * y ≤ (x^(n+2) + y^(n+2)) / (x^n + y^n) := by
  sorry

end xy_inequality_l1581_158169


namespace molecular_weight_calculation_l1581_158195

/-- Given 6 moles of a compound with a total weight of 252 grams, 
    the molecular weight of the compound is 42 grams/mole. -/
theorem molecular_weight_calculation (moles : ℝ) (total_weight : ℝ) :
  moles = 6 →
  total_weight = 252 →
  total_weight / moles = 42 := by
  sorry

end molecular_weight_calculation_l1581_158195


namespace remainder_theorem_l1581_158145

theorem remainder_theorem : ∃ q : ℕ, 2^202 + 202 = (2^101 + 2^51 + 1) * q + 201 := by
  sorry

end remainder_theorem_l1581_158145


namespace line_contains_point_l1581_158148

/-- Proves that for the line equation 3 - kx = -4y, if the point (3, -2) lies on the line, then k = -5/3 -/
theorem line_contains_point (k : ℚ) : 
  (3 - k * 3 = -4 * (-2)) → k = -5/3 := by
  sorry

end line_contains_point_l1581_158148


namespace cats_on_ship_l1581_158155

/-- Represents the passengers on the Queen Mary II luxury liner -/
structure Passengers where
  sailors : ℕ
  cats : ℕ

/-- The total number of heads on the ship -/
def total_heads (p : Passengers) : ℕ := p.sailors + 1 + 1 + p.cats

/-- The total number of legs on the ship -/
def total_legs (p : Passengers) : ℕ := 2 * p.sailors + 2 + 1 + 4 * p.cats

/-- Theorem stating that there are 7 cats on the ship -/
theorem cats_on_ship : 
  ∃ (p : Passengers), total_heads p = 15 ∧ total_legs p = 43 ∧ p.cats = 7 := by
  sorry

end cats_on_ship_l1581_158155


namespace least_subtraction_for_divisibility_l1581_158147

theorem least_subtraction_for_divisibility (x : ℕ) : 
  (x = 26 ∧ (12702 - x) % 99 = 0) ∧ 
  ∀ y : ℕ, y < x → (12702 - y) % 99 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l1581_158147


namespace sqrt_equation_solution_l1581_158105

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end sqrt_equation_solution_l1581_158105


namespace max_ab_tangent_circles_l1581_158163

/-- Two externally tangent circles -/
structure TangentCircles where
  a : ℝ
  b : ℝ
  c1 : (x : ℝ) → (y : ℝ) → (x - a)^2 + (y + 2)^2 = 4
  c2 : (x : ℝ) → (y : ℝ) → (x + b)^2 + (y + 2)^2 = 1
  tangent : a + b = 3

/-- The maximum value of ab for externally tangent circles -/
theorem max_ab_tangent_circles (tc : TangentCircles) : 
  ∃ (max : ℝ), max = 9/4 ∧ tc.a * tc.b ≤ max :=
by sorry

end max_ab_tangent_circles_l1581_158163


namespace system_solution_l1581_158146

theorem system_solution :
  let f (x y : ℝ) := 7 * x^2 + 7 * y^2 - 3 * x^2 * y^2
  let g (x y : ℝ) := x^4 + y^4 - x^2 * y^2
  ∀ x y : ℝ, (f x y = 7 ∧ g x y = 37) ↔
    ((x = Real.sqrt 7 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
     (x = -Real.sqrt 7 ∧ (y = Real.sqrt 3 ∨ y = -Real.sqrt 3)) ∨
     (x = Real.sqrt 3 ∧ (y = Real.sqrt 7 ∨ y = -Real.sqrt 7)) ∨
     (x = -Real.sqrt 3 ∧ (y = Real.sqrt 7 ∨ y = -Real.sqrt 7))) :=
by sorry

end system_solution_l1581_158146


namespace triangle_angle_relation_l1581_158198

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (angle_a angle_b angle_c : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (positive_angles : 0 < angle_a ∧ 0 < angle_b ∧ 0 < angle_c)
  (angle_sum : angle_a + angle_b + angle_c = Real.pi)
  (law_of_sines : a / Real.sin angle_a = b / Real.sin angle_b)

-- State the theorem
theorem triangle_angle_relation (t : Triangle) 
  (h : t.angle_a = 3 * t.angle_b) :
  (t.a^2 - t.b^2) * (t.a - t.b) = t.b * t.c^2 := by
  sorry


end triangle_angle_relation_l1581_158198


namespace sons_age_l1581_158178

/-- Proves that given the conditions, the son's present age is 25 years. -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 27 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 25 := by
  sorry

end sons_age_l1581_158178


namespace only_solution_is_two_l1581_158124

theorem only_solution_is_two : 
  ∃! (n : ℤ), n + 13 > 15 ∧ -6*n > -18 :=
by sorry

end only_solution_is_two_l1581_158124


namespace smaller_circle_radius_l1581_158140

/-- Given two circles where one has a diameter of 80 cm and its radius is 4 times
    the radius of the other, prove that the radius of the smaller circle is 10 cm. -/
theorem smaller_circle_radius (d : ℝ) (r₁ r₂ : ℝ) : 
  d = 80 → r₁ = d / 2 → r₁ = 4 * r₂ → r₂ = 10 := by
  sorry

end smaller_circle_radius_l1581_158140


namespace bike_distance_theorem_l1581_158117

/-- Calculates the distance traveled by a bike given its speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 3 m/s for 7 seconds covers 21 meters -/
theorem bike_distance_theorem :
  let speed : ℝ := 3
  let time : ℝ := 7
  distance_traveled speed time = 21 := by sorry

end bike_distance_theorem_l1581_158117


namespace percentage_relation_l1581_158115

theorem percentage_relation (A B C : ℝ) (h1 : A = 0.07 * C) (h2 : A = 0.5 * B) :
  B = 0.14 * C := by
  sorry

end percentage_relation_l1581_158115


namespace c_value_satisfies_equation_l1581_158189

/-- Definition of function F -/
def F (a b c d : ℝ) : ℝ := a * b^2 + c * d

/-- Theorem stating that c = 16 satisfies the equation when a = 2 -/
theorem c_value_satisfies_equation :
  ∃ c : ℝ, F 2 3 c 5 = F 2 5 c 3 ∧ c = 16 := by sorry

end c_value_satisfies_equation_l1581_158189


namespace rectangle_circle_square_area_l1581_158197

theorem rectangle_circle_square_area :
  ∀ (rectangle_length rectangle_breadth rectangle_area circle_radius square_side : ℝ),
    rectangle_length = 5 * circle_radius →
    rectangle_breadth = 11 →
    rectangle_area = 220 →
    rectangle_area = rectangle_length * rectangle_breadth →
    circle_radius = square_side →
    square_side ^ 2 = 16 :=
by
  sorry

end rectangle_circle_square_area_l1581_158197


namespace number_of_boys_in_class_l1581_158193

/-- The number of boys in a class given specific height conditions -/
theorem number_of_boys_in_class (n : ℕ) 
  (h1 : (n : ℝ) * 182 = (n : ℝ) * 182 + 166 - 106)
  (h2 : (n : ℝ) * 180 = (n : ℝ) * 182 + 106 - 166) : n = 30 := by
  sorry


end number_of_boys_in_class_l1581_158193


namespace purely_imaginary_z_l1581_158153

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = Complex.I * b) → -- z is purely imaginary
  (∃ c : ℝ, (z + 1)^2 - 2*Complex.I = Complex.I * c) → -- (z+1)^2 - 2i is purely imaginary
  z = -Complex.I := by
sorry

end purely_imaginary_z_l1581_158153


namespace mothers_age_l1581_158127

theorem mothers_age (eunji_current_age eunji_past_age mother_past_age : ℕ) 
  (h1 : eunji_current_age = 16)
  (h2 : eunji_past_age = 8)
  (h3 : mother_past_age = 35) : 
  mother_past_age + (eunji_current_age - eunji_past_age) = 43 := by
  sorry

end mothers_age_l1581_158127


namespace inequality_equivalence_l1581_158130

theorem inequality_equivalence (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by
  sorry

end inequality_equivalence_l1581_158130


namespace weight_range_proof_l1581_158125

/-- Given the weights of Tracy, John, and Jake, prove the range of their weights. -/
theorem weight_range_proof (tracy_weight john_weight jake_weight : ℕ) 
  (h1 : tracy_weight + john_weight + jake_weight = 158)
  (h2 : tracy_weight = 52)
  (h3 : jake_weight = tracy_weight + 8) : 
  (max tracy_weight (max john_weight jake_weight)) - 
  (min tracy_weight (min john_weight jake_weight)) = 14 := by
  sorry

#check weight_range_proof

end weight_range_proof_l1581_158125


namespace sum_of_squares_l1581_158150

theorem sum_of_squares (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^2 + 9) / a = (b^2 + 9) / b ∧ (b^2 + 9) / b = (c^2 + 9) / c) : 
  a^2 + b^2 + c^2 = -27 := by
  sorry

end sum_of_squares_l1581_158150


namespace m_less_than_n_min_sum_a_b_l1581_158192

-- Define the variables and conditions
variables (a b : ℝ) (m n : ℝ)

-- Define the relationships between variables
def m_def : m = a * b + 1 := by sorry
def n_def : n = a + b := by sorry

-- Part 1: Prove m < n when a > 1 and b < 1
theorem m_less_than_n (ha : a > 1) (hb : b < 1) : m < n := by sorry

-- Part 2: Prove minimum value of a + b is 16 when a > 1, b > 1, and m - n = 49
theorem min_sum_a_b (ha : a > 1) (hb : b > 1) (h_diff : m - n = 49) :
  ∃ (min_sum : ℝ), min_sum = 16 ∧ a + b ≥ min_sum := by sorry

end m_less_than_n_min_sum_a_b_l1581_158192


namespace number_puzzle_l1581_158168

theorem number_puzzle : ∃! (N : ℕ), N > 0 ∧ ∃ (Q : ℕ), N = 11 * Q ∧ Q + N + 11 = 71 := by
  sorry

end number_puzzle_l1581_158168


namespace tire_price_problem_l1581_158154

theorem tire_price_problem (total_cost : ℝ) (discount_tire_price : ℝ) 
  (h1 : total_cost = 250)
  (h2 : discount_tire_price = 10) : 
  ∃ (regular_price : ℝ), 3 * regular_price + discount_tire_price = total_cost ∧ regular_price = 80 := by
  sorry

end tire_price_problem_l1581_158154


namespace consecutive_even_numbers_sum_l1581_158141

theorem consecutive_even_numbers_sum (a : ℤ) : 
  (∃ (x : ℤ), 
    (x = a) ∧ 
    (x + (x + 2) + (x + 4) + (x + 6) = 52)) → 
  (a + 4 = 14) := by
  sorry

end consecutive_even_numbers_sum_l1581_158141


namespace stream_speed_l1581_158112

/-- Given a boat's travel times and distances, calculate the stream speed -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 90) 
  (h2 : upstream_distance = 72)
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 3 := by
  sorry

end stream_speed_l1581_158112
