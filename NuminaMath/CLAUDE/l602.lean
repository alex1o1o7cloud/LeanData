import Mathlib

namespace NUMINAMATH_CALUDE_jack_walked_4_miles_l602_60204

/-- The distance Jack walked given his walking time and rate -/
def jack_distance (time_hours : ℝ) (rate : ℝ) : ℝ :=
  time_hours * rate

theorem jack_walked_4_miles :
  let time_hours : ℝ := 1.25  -- 1 hour and 15 minutes in decimal hours
  let rate : ℝ := 3.2         -- miles per hour
  jack_distance time_hours rate = 4 := by
sorry

end NUMINAMATH_CALUDE_jack_walked_4_miles_l602_60204


namespace NUMINAMATH_CALUDE_arrange_seven_white_five_black_l602_60217

/-- The number of ways to arrange white and black balls with no adjacent black balls -/
def arrangeBalls (white black : ℕ) : ℕ :=
  Nat.choose (white + black - black + 1) (black + 1)

/-- Theorem stating that arranging 7 white and 5 black balls with no adjacent black balls results in 56 ways -/
theorem arrange_seven_white_five_black :
  arrangeBalls 7 5 = 56 := by
  sorry

#eval arrangeBalls 7 5

end NUMINAMATH_CALUDE_arrange_seven_white_five_black_l602_60217


namespace NUMINAMATH_CALUDE_eunji_class_size_l602_60221

/-- The number of lines students stand in --/
def num_lines : ℕ := 3

/-- Eunji's position from the front of the line --/
def position_from_front : ℕ := 3

/-- Eunji's position from the back of the line --/
def position_from_back : ℕ := 6

/-- The total number of students in Eunji's line --/
def students_per_line : ℕ := position_from_front + position_from_back - 1

/-- The total number of students in Eunji's class --/
def total_students : ℕ := num_lines * students_per_line

theorem eunji_class_size : total_students = 24 := by
  sorry

end NUMINAMATH_CALUDE_eunji_class_size_l602_60221


namespace NUMINAMATH_CALUDE_green_peaches_count_l602_60206

/-- Given a number of baskets, red peaches per basket, and total peaches,
    calculates the number of green peaches per basket. -/
def green_peaches_per_basket (num_baskets : ℕ) (red_per_basket : ℕ) (total_peaches : ℕ) : ℕ :=
  (total_peaches - num_baskets * red_per_basket) / num_baskets

/-- Proves that there are 4 green peaches in each basket given the problem conditions. -/
theorem green_peaches_count :
  green_peaches_per_basket 15 19 345 = 4 := by
  sorry

#eval green_peaches_per_basket 15 19 345

end NUMINAMATH_CALUDE_green_peaches_count_l602_60206


namespace NUMINAMATH_CALUDE_watson_class_size_l602_60245

/-- The number of students in Ms. Watson's class -/
def total_students (kindergartners first_graders second_graders : ℕ) : ℕ :=
  kindergartners + first_graders + second_graders

/-- Theorem: Ms. Watson has 42 students in her class -/
theorem watson_class_size :
  total_students 14 24 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_watson_class_size_l602_60245


namespace NUMINAMATH_CALUDE_problem_solution_l602_60211

theorem problem_solution (y : ℝ) (h : y + Real.sqrt (y^2 - 4) + 1 / (y - Real.sqrt (y^2 - 4)) = 12) :
  y^2 + Real.sqrt (y^4 - 4) + 1 / (y^2 - Real.sqrt (y^4 - 4)) = 200 / 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l602_60211


namespace NUMINAMATH_CALUDE_german_students_count_l602_60238

theorem german_students_count (total_students : ℕ) 
                               (french_students : ℕ) 
                               (both_students : ℕ) 
                               (neither_students : ℕ) 
                               (h1 : total_students = 60)
                               (h2 : french_students = 41)
                               (h3 : both_students = 9)
                               (h4 : neither_students = 6) :
  ∃ german_students : ℕ, german_students = 22 ∧ 
    german_students + french_students - both_students + neither_students = total_students :=
by
  sorry

end NUMINAMATH_CALUDE_german_students_count_l602_60238


namespace NUMINAMATH_CALUDE_harry_needs_five_spellbooks_l602_60248

/-- Represents the cost and quantity of items Harry needs to buy --/
structure HarrysPurchase where
  spellbookCost : ℕ
  potionKitCost : ℕ
  owlCost : ℕ
  silverToGoldRatio : ℕ
  totalSilver : ℕ
  potionKitQuantity : ℕ

/-- Calculates the number of spellbooks Harry needs to buy --/
def calculateSpellbooks (purchase : HarrysPurchase) : ℕ :=
  let remainingSilver := purchase.totalSilver -
    (purchase.owlCost * purchase.silverToGoldRatio + 
     purchase.potionKitCost * purchase.potionKitQuantity)
  remainingSilver / (purchase.spellbookCost * purchase.silverToGoldRatio)

/-- Theorem stating that Harry needs to buy 5 spellbooks --/
theorem harry_needs_five_spellbooks (purchase : HarrysPurchase) 
  (h1 : purchase.spellbookCost = 5)
  (h2 : purchase.potionKitCost = 20)
  (h3 : purchase.owlCost = 28)
  (h4 : purchase.silverToGoldRatio = 9)
  (h5 : purchase.totalSilver = 537)
  (h6 : purchase.potionKitQuantity = 3) :
  calculateSpellbooks purchase = 5 := by
  sorry


end NUMINAMATH_CALUDE_harry_needs_five_spellbooks_l602_60248


namespace NUMINAMATH_CALUDE_ab_length_in_two_isosceles_triangles_l602_60259

/-- An isosceles triangle with given side lengths -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := t.base + 2 * t.leg

theorem ab_length_in_two_isosceles_triangles 
  (abc cde : IsoscelesTriangle)
  (h1 : perimeter cde = 22)
  (h2 : perimeter abc = 24)
  (h3 : cde.base = 8)
  (h4 : abc.leg = cde.leg) : 
  abc.base = 10 := by sorry

end NUMINAMATH_CALUDE_ab_length_in_two_isosceles_triangles_l602_60259


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l602_60283

/-- Given a quadratic function f(x) = ax^2 - 4x + c where a ≠ 0,
    with range [0, +∞) and f(1) ≤ 4, the maximum value of
    u = a/(c^2+4) + c/(a^2+4) is 7/4. -/
theorem quadratic_function_max_value (a c : ℝ) (h1 : a ≠ 0) :
  let f := fun x => a * x^2 - 4 * x + c
  (∀ y, y ∈ Set.range f → y ≥ 0) →
  (f 1 ≤ 4) →
  (∃ u : ℝ, u = a / (c^2 + 4) + c / (a^2 + 4) ∧
    u ≤ 7/4 ∧
    ∀ v, v = a / (c^2 + 4) + c / (a^2 + 4) → v ≤ u) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l602_60283


namespace NUMINAMATH_CALUDE_equation_solutions_l602_60297

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 6*x + 3 = 0
  let eq2 : ℝ → Prop := λ x ↦ 2*x*(x-1) = 3-3*x
  let sol1 : Set ℝ := {3 + Real.sqrt 6, 3 - Real.sqrt 6}
  let sol2 : Set ℝ := {1, -3/2}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y, eq1 y → y ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y, eq2 y → y ∈ sol2) :=
by sorry


end NUMINAMATH_CALUDE_equation_solutions_l602_60297


namespace NUMINAMATH_CALUDE_tiger_tree_trunk_length_l602_60213

/-- The length of a fallen tree trunk over which a tiger runs --/
theorem tiger_tree_trunk_length (tiger_length : ℝ) (grass_time : ℝ) (trunk_time : ℝ)
  (h_length : tiger_length = 5)
  (h_grass : grass_time = 1)
  (h_trunk : trunk_time = 5) :
  tiger_length * trunk_time = 25 := by
  sorry

end NUMINAMATH_CALUDE_tiger_tree_trunk_length_l602_60213


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_l602_60243

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a - 2)*x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + (a - 2)

-- Theorem statement
theorem tangent_line_at_origin (a : ℝ) 
  (h : ∀ x, f' a x = f' a (-x)) : 
  ∃ m, ∀ x, f a x = m * x + f a 0 ∧ m = -2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_l602_60243


namespace NUMINAMATH_CALUDE_train_length_l602_60263

/-- The length of a train given its speed, bridge length, and crossing time -/
theorem train_length (speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  speed = 45 * 1000 / 3600 →
  bridge_length = 205 →
  crossing_time = 30 →
  speed * crossing_time - bridge_length = 170 := by
  sorry


end NUMINAMATH_CALUDE_train_length_l602_60263


namespace NUMINAMATH_CALUDE_g_formula_and_domain_intersection_points_l602_60298

noncomputable section

-- Define the original function f
def f (x : ℝ) : ℝ := x + 1/x

-- Define the domain of f
def f_domain : Set ℝ := {x | x < 0 ∨ x > 0}

-- Define the symmetric function g
def g (x : ℝ) : ℝ := x - 2 + 1/(x-4)

-- Define the domain of g
def g_domain : Set ℝ := {x | x < 4 ∨ x > 4}

-- Define the symmetry point
def A : ℝ × ℝ := (2, 1)

-- Theorem for the correct formula and domain of g
theorem g_formula_and_domain :
  (∀ x ∈ g_domain, g x = x - 2 + 1/(x-4)) ∧
  (∀ x, x ∈ g_domain ↔ x < 4 ∨ x > 4) :=
sorry

-- Theorem for the intersection points
theorem intersection_points :
  (∀ b : ℝ, (∃! x, g x = b) ↔ b = 4 ∨ b = 0) ∧
  (g 5 = 4 ∧ g 3 = 0) :=
sorry

end

end NUMINAMATH_CALUDE_g_formula_and_domain_intersection_points_l602_60298


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l602_60224

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → 
  a^2019 + b^2018 = -1 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l602_60224


namespace NUMINAMATH_CALUDE_pi_only_irrational_l602_60241

-- Define a function to check if a number is rational
def is_rational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- State the theorem
theorem pi_only_irrational : 
  is_rational (1/7) ∧ 
  ¬(is_rational Real.pi) ∧ 
  is_rational (-1) ∧ 
  is_rational 0 :=
sorry

end NUMINAMATH_CALUDE_pi_only_irrational_l602_60241


namespace NUMINAMATH_CALUDE_derivative_x_sin_x_at_pi_l602_60228

/-- The derivative of f(x) = x * sin(x) evaluated at π is equal to -π. -/
theorem derivative_x_sin_x_at_pi :
  let f : ℝ → ℝ := fun x ↦ x * Real.sin x
  (deriv f) π = -π :=
by sorry

end NUMINAMATH_CALUDE_derivative_x_sin_x_at_pi_l602_60228


namespace NUMINAMATH_CALUDE_cow_chicken_goat_problem_l602_60201

theorem cow_chicken_goat_problem (cows chickens goats : ℕ) : 
  cows + chickens + goats = 12 →
  4 * cows + 2 * chickens + 4 * goats = 18 + 2 * (cows + chickens + goats) →
  cows + goats = 9 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_goat_problem_l602_60201


namespace NUMINAMATH_CALUDE_four_weeks_filming_time_l602_60237

/-- Calculates the total filming time in hours for a given number of weeks -/
def total_filming_time (episode_length : ℕ) (filming_factor : ℚ) (episodes_per_week : ℕ) (weeks : ℕ) : ℚ :=
  let filming_time := episode_length * (1 + filming_factor)
  let total_episodes := episodes_per_week * weeks
  (filming_time * total_episodes) / 60

theorem four_weeks_filming_time :
  total_filming_time 20 (1/2) 5 4 = 10 := by
  sorry

#eval total_filming_time 20 (1/2) 5 4

end NUMINAMATH_CALUDE_four_weeks_filming_time_l602_60237


namespace NUMINAMATH_CALUDE_graph_is_pair_of_lines_l602_60251

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 9*y^2 = 0

/-- Definition of a straight line in slope-intercept form -/
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

/-- The graph consists of two straight lines -/
theorem graph_is_pair_of_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_graph_is_pair_of_lines_l602_60251


namespace NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l602_60277

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius 
  (perimeter : ℝ) 
  (inradius : ℝ) 
  (h_perimeter : perimeter = 40) 
  (h_inradius : inradius = 2.5) : 
  inradius * (perimeter / 2) = 50 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_from_perimeter_and_inradius_l602_60277


namespace NUMINAMATH_CALUDE_pine_cone_weight_l602_60281

/-- The weight of each pine cone given the conditions in Alan's backyard scenario -/
theorem pine_cone_weight (
  trees : ℕ)
  (cones_per_tree : ℕ)
  (roof_percentage : ℚ)
  (total_roof_weight : ℕ)
  (h1 : trees = 8)
  (h2 : cones_per_tree = 200)
  (h3 : roof_percentage = 3/10)
  (h4 : total_roof_weight = 1920)
  : (total_roof_weight : ℚ) / ((trees * cones_per_tree : ℕ) * roof_percentage) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pine_cone_weight_l602_60281


namespace NUMINAMATH_CALUDE_x_over_y_equals_one_l602_60232

theorem x_over_y_equals_one (x y : ℝ) 
  (h1 : 1 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 3) 
  (h3 : ∃ n : ℤ, x / y = n) : 
  x / y = 1 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_equals_one_l602_60232


namespace NUMINAMATH_CALUDE_parabola_properties_l602_60289

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus
def Focus : ℝ × ℝ := (-1, 0)

-- Define the line passing through the focus with slope 45°
def Line (x y : ℝ) : Prop := y = x + 1

-- Define the chord length
def ChordLength : ℝ := 8

theorem parabola_properties :
  -- The parabola passes through (-2, 2√2)
  Parabola (-2) (2 * Real.sqrt 2) ∧
  -- The focus is at (-1, 0)
  Focus = (-1, 0) ∧
  -- The chord formed by the intersection of the parabola and the line has length 8
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    Parabola x₁ y₁ ∧ Parabola x₂ y₂ ∧
    Line x₁ y₁ ∧ Line x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = ChordLength :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l602_60289


namespace NUMINAMATH_CALUDE_solutions_correct_l602_60215

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  3 * x + y = 4 ∧ 3 * x + 2 * y = 6

def system2 (x y : ℝ) : Prop :=
  2 * x + y = 3 ∧ 3 * x - 5 * y = 11

-- State the theorem
theorem solutions_correct :
  (∃ x y : ℝ, system1 x y ∧ x = 2/3 ∧ y = 2) ∧
  (∃ x y : ℝ, system2 x y ∧ x = 2 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_solutions_correct_l602_60215


namespace NUMINAMATH_CALUDE_print_shop_pricing_l602_60293

/-- The price per color copy at print shop Y -/
def price_Y : ℚ := 2.75

/-- The number of color copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X for 40 copies -/
def additional_charge : ℚ := 60

/-- The price per color copy at print shop X -/
def price_X : ℚ := 1.25

theorem print_shop_pricing :
  price_Y * num_copies = price_X * num_copies + additional_charge :=
sorry

end NUMINAMATH_CALUDE_print_shop_pricing_l602_60293


namespace NUMINAMATH_CALUDE_train_platform_time_l602_60229

/-- The time taken for a train to pass a platform -/
def time_to_pass_platform (l : ℝ) (t : ℝ) : ℝ :=
  5 * t

/-- Theorem: The time taken for a train of length l, traveling at a constant velocity, 
    to pass a platform of length 4l is 5 times the time it takes to pass a pole, 
    given that it takes t seconds to pass the pole. -/
theorem train_platform_time (l : ℝ) (t : ℝ) (v : ℝ) :
  l > 0 → t > 0 → v > 0 →
  (l / v = t) →  -- Time to pass pole
  ((l + 4 * l) / v = time_to_pass_platform l t) :=
by sorry

end NUMINAMATH_CALUDE_train_platform_time_l602_60229


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l602_60249

/-- Given that z varies inversely as √w, prove that w = 16 when z = 2, 
    given that z = 4 when w = 4. -/
theorem inverse_variation_sqrt (z w : ℝ) (h : ∃ k : ℝ, ∀ w z, z * Real.sqrt w = k) :
  (4 * Real.sqrt 4 = 4 * Real.sqrt w) → (2 * Real.sqrt w = 4 * Real.sqrt 4) → w = 16 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l602_60249


namespace NUMINAMATH_CALUDE_sin_two_theta_value_l602_60250

theorem sin_two_theta_value (θ : Real) 
  (h1 : 0 < θ ∧ θ < π/2) 
  (h2 : (Real.sin θ + Real.cos θ)^2 + Real.sqrt 3 * Real.cos (2*θ) = 3) : 
  Real.sin (2*θ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_sin_two_theta_value_l602_60250


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l602_60214

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l602_60214


namespace NUMINAMATH_CALUDE_greatest_three_digit_number_condition_l602_60236

theorem greatest_three_digit_number_condition : ∃ n : ℕ,
  (100 ≤ n ∧ n ≤ 999) ∧
  (∃ k : ℕ, n = 11 * k + 2) ∧
  (∃ m : ℕ, n = 7 * m + 4) ∧
  (∀ x : ℕ, (100 ≤ x ∧ x ≤ 999) →
    (∃ a : ℕ, x = 11 * a + 2) →
    (∃ b : ℕ, x = 7 * b + 4) →
    x ≤ n) ∧
  n = 970 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_number_condition_l602_60236


namespace NUMINAMATH_CALUDE_vehicle_travel_time_l602_60275

/-- 
Given two vehicles A and B traveling towards each other, prove that B's total travel time is 7.2 hours
under the following conditions:
1. They meet after 3 hours.
2. A turns back to its starting point, taking 3 hours.
3. A then turns around again and meets B after 0.5 hours.
-/
theorem vehicle_travel_time (v_A v_B : ℝ) (d : ℝ) : 
  v_A > 0 ∧ v_B > 0 ∧ d > 0 → 
  d = 3 * (v_A + v_B) →
  3 * v_A = d / 2 →
  d / 2 + 0.5 * v_A = 3.5 * v_B →
  d / v_B = 7.2 := by
sorry

end NUMINAMATH_CALUDE_vehicle_travel_time_l602_60275


namespace NUMINAMATH_CALUDE_frank_remaining_money_l602_60279

/-- Calculates the remaining money after buying the most expensive lamp -/
def remaining_money (cheapest_lamp_cost most_expensive_factor current_money : ℕ) : ℕ :=
  current_money - (cheapest_lamp_cost * most_expensive_factor)

/-- Proves that Frank will have $30 remaining after buying the most expensive lamp -/
theorem frank_remaining_money :
  remaining_money 20 3 90 = 30 := by sorry

end NUMINAMATH_CALUDE_frank_remaining_money_l602_60279


namespace NUMINAMATH_CALUDE_radical_simplification_l602_60209

theorem radical_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (21 * q) = 21 * q * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_CALUDE_radical_simplification_l602_60209


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l602_60200

/-- A geometric sequence with sum of first n terms Sn -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, n > 0 → a n = a 1 * r^(n-1) ∧ S n = a 1 * (1 - r^n) / (1 - r)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  a 1 + a 3 = 5/2 →
  a 2 + a 4 = 5/4 →
  ∀ n, n > 0 → S n / a n = 2^n - 1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l602_60200


namespace NUMINAMATH_CALUDE_original_average_age_proof_l602_60264

theorem original_average_age_proof (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 ∧
  new_students = 12 ∧
  new_avg = 32 ∧
  avg_decrease = 6 →
  original_avg = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_age_proof_l602_60264


namespace NUMINAMATH_CALUDE_expand_product_l602_60205

theorem expand_product (x : ℝ) : (3*x + 4) * (x - 2) * (x + 6) = 3*x^3 + 16*x^2 - 20*x - 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l602_60205


namespace NUMINAMATH_CALUDE_tan_x_plus_pi_third_l602_60287

theorem tan_x_plus_pi_third (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + π / 3) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_plus_pi_third_l602_60287


namespace NUMINAMATH_CALUDE_dodecahedron_triangle_probability_l602_60257

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Nat
  connections_per_vertex : Nat

/-- The probability of forming a triangle with three randomly chosen vertices -/
def triangle_probability (d : RegularDodecahedron) : ℚ :=
  sorry

/-- Theorem stating the probability of forming a triangle in a regular dodecahedron -/
theorem dodecahedron_triangle_probability :
  let d : RegularDodecahedron := ⟨20, 3⟩
  triangle_probability d = 1 / 57 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_triangle_probability_l602_60257


namespace NUMINAMATH_CALUDE_log_3_infinite_sum_equals_4_l602_60220

theorem log_3_infinite_sum_equals_4 :
  ∃ (x : ℝ), x > 0 ∧ 3^x = x + 81 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_3_infinite_sum_equals_4_l602_60220


namespace NUMINAMATH_CALUDE_maple_logs_solution_l602_60286

/-- The number of logs each maple tree makes -/
def maple_logs : ℕ := 60

theorem maple_logs_solution : 
  ∃ (x : ℕ), x > 0 ∧ 8 * 80 + 3 * x + 4 * 100 = 1220 → x = maple_logs :=
by sorry

end NUMINAMATH_CALUDE_maple_logs_solution_l602_60286


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l602_60230

/-- The minimum distance from any point on a circle to a line --/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + (y - 1)^2 = 4}
  let line := {(x, y) : ℝ × ℝ | x - y + 4 = 0}
  (∃ (d : ℝ), d = 2 * Real.sqrt 2 - 2 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l602_60230


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l602_60282

def polynomial (x : ℤ) : ℤ := x^4 + 2*x^3 - x^2 + 3*x - 30

def possible_roots : Set ℤ := {1, -1, 2, -2, 3, -3, 5, -5, 6, -6, 10, -10, 15, -15, 30, -30}

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = possible_roots := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l602_60282


namespace NUMINAMATH_CALUDE_x_power_ten_plus_inverse_l602_60265

theorem x_power_ten_plus_inverse (x : ℝ) (h : x + 1/x = 5) : x^10 + 1/x^10 = 6430223 := by
  sorry

end NUMINAMATH_CALUDE_x_power_ten_plus_inverse_l602_60265


namespace NUMINAMATH_CALUDE_system_solution_l602_60272

theorem system_solution (x y : ℝ) (k n : ℤ) : 
  (2 * (Real.cos x)^2 - 2 * Real.sqrt 2 * Real.cos x * (Real.cos (8 * x))^2 + (Real.cos (8 * x))^2 = 0 ∧
   Real.sin x = Real.cos y) ↔ 
  ((x = π/4 + 2*π*↑k ∧ (y = π/4 + 2*π*↑n ∨ y = -π/4 + 2*π*↑n)) ∨
   (x = -π/4 + 2*π*↑k ∧ (y = 3*π/4 + 2*π*↑n ∨ y = -3*π/4 + 2*π*↑n))) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l602_60272


namespace NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l602_60262

theorem parabola_focus_vertex_distance (p : ℝ) (h_p : p > 0) : 
  let C : Set (ℝ × ℝ) := {(x, y) | y^2 = 2*p*x}
  let F : ℝ × ℝ := (p/2, 0)
  let l : Set (ℝ × ℝ) := {(x, y) | y = x - p/2}
  let chord_length : ℝ := 4
  let angle_with_axis : ℝ := π/4
  (∀ (x y : ℝ), (x, y) ∈ l → (x - F.1)^2 + (y - F.2)^2 = (x + F.1)^2 + (y - F.2)^2) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ C ∩ l ∧ (x₂, y₂) ∈ C ∩ l ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = chord_length^2) →
  (∀ (x y : ℝ), (x, y) ∈ l → y / x = Real.tan angle_with_axis) →
  F.1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_parabola_focus_vertex_distance_l602_60262


namespace NUMINAMATH_CALUDE_triangle_theorem_l602_60271

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  S : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_angles : A > 0 ∧ B > 0 ∧ C > 0
  h_angle_sum : A + B + C = π
  h_area : S = (1/2) * b * c * Real.sin A

-- Define the main theorem
theorem triangle_theorem (t : Triangle) :
  (3 * t.a^2 - 4 * Real.sqrt 3 * t.S = 3 * t.b^2 + 3 * t.c^2) →
  (t.A = 2 * π / 3) ∧
  (t.a = 3 → 6 < t.a + t.b + t.c ∧ t.a + t.b + t.c < 3 + 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l602_60271


namespace NUMINAMATH_CALUDE_red_chips_count_l602_60225

def total_chips : ℕ := 60
def green_chips : ℕ := 16

def blue_chips : ℕ := total_chips / 6

def red_chips : ℕ := total_chips - blue_chips - green_chips

theorem red_chips_count : red_chips = 34 := by
  sorry

end NUMINAMATH_CALUDE_red_chips_count_l602_60225


namespace NUMINAMATH_CALUDE_adult_elephant_weekly_bananas_eq_630_l602_60276

/-- The number of bananas an adult elephant eats per day -/
def adult_elephant_daily_bananas : ℕ := 90

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of bananas an adult elephant eats in a week -/
def adult_elephant_weekly_bananas : ℕ := adult_elephant_daily_bananas * days_in_week

theorem adult_elephant_weekly_bananas_eq_630 :
  adult_elephant_weekly_bananas = 630 := by
  sorry

end NUMINAMATH_CALUDE_adult_elephant_weekly_bananas_eq_630_l602_60276


namespace NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l602_60203

/-- The sum of interior angles of a hexagon is 720 degrees. -/
theorem sum_interior_angles_hexagon :
  ∀ (n : ℕ) (sum_interior_angles : ℕ → ℝ),
  n = 6 →
  (∀ k : ℕ, sum_interior_angles k = (k - 2) * 180) →
  sum_interior_angles n = 720 := by
sorry

end NUMINAMATH_CALUDE_sum_interior_angles_hexagon_l602_60203


namespace NUMINAMATH_CALUDE_initial_shirts_l602_60290

theorem initial_shirts (S : ℕ) : S + 4 = 16 → S = 12 := by
  sorry

end NUMINAMATH_CALUDE_initial_shirts_l602_60290


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l602_60273

/-- Represents the different sampling methods --/
inductive SamplingMethod
  | Lottery
  | Systematic
  | Stratified
  | RandomNumber

/-- Represents a grade level --/
inductive Grade
  | Third
  | Sixth
  | Ninth

/-- Represents the characteristics of the sampling problem --/
structure SamplingProblem where
  grades : List Grade
  proportionalSampling : Bool
  distinctGroups : Bool

/-- Determines the most appropriate sampling method for a given problem --/
def mostAppropriateMethod (problem : SamplingProblem) : SamplingMethod :=
  if problem.distinctGroups && problem.proportionalSampling then
    SamplingMethod.Stratified
  else
    SamplingMethod.Lottery  -- Default to Lottery for simplicity

/-- The specific problem described in the question --/
def schoolEyesightProblem : SamplingProblem :=
  { grades := [Grade.Third, Grade.Sixth, Grade.Ninth]
  , proportionalSampling := true
  , distinctGroups := true }

theorem stratified_sampling_most_appropriate :
  mostAppropriateMethod schoolEyesightProblem = SamplingMethod.Stratified := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l602_60273


namespace NUMINAMATH_CALUDE_problem_solution_l602_60253

theorem problem_solution (t : ℚ) :
  let x := 3 - 2 * t
  let y := 5 * t + 6
  x = 0 → y = 27 / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l602_60253


namespace NUMINAMATH_CALUDE_fourth_to_third_grade_ratio_l602_60267

/-- Given the number of students in each grade, prove the ratio of 4th to 3rd grade students -/
theorem fourth_to_third_grade_ratio 
  (third_grade : ℕ) 
  (second_grade : ℕ) 
  (total_students : ℕ) 
  (h1 : third_grade = 19) 
  (h2 : second_grade = 29) 
  (h3 : total_students = 86) :
  (total_students - second_grade - third_grade) / third_grade = 2 := by
sorry

end NUMINAMATH_CALUDE_fourth_to_third_grade_ratio_l602_60267


namespace NUMINAMATH_CALUDE_solution_set_correct_l602_60268

def solution_set := {x : ℝ | 0 < x ∧ x < 1}

theorem solution_set_correct : 
  ∀ x : ℝ, x ∈ solution_set ↔ (x * (x + 2) > 0 ∧ |x| < 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_correct_l602_60268


namespace NUMINAMATH_CALUDE_left_handed_fraction_l602_60234

theorem left_handed_fraction (red blue : ℕ) (h_ratio : red = blue) 
  (h_red_left : red / 3 = red.div 3) 
  (h_blue_left : 2 * (blue / 3) = blue.div 3 * 2) : 
  (red.div 3 + blue.div 3 * 2) / (red + blue) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_left_handed_fraction_l602_60234


namespace NUMINAMATH_CALUDE_operation_with_96_percent_error_l602_60223

/-- Given a number N and an operation O(N), if the percentage error between O(N) and 5N is 96%, then O(N) = 0.2N -/
theorem operation_with_96_percent_error (N : ℝ) (O : ℝ → ℝ) :
  (|O N - 5 * N| / (5 * N) = 0.96) → O N = 0.2 * N :=
by sorry

end NUMINAMATH_CALUDE_operation_with_96_percent_error_l602_60223


namespace NUMINAMATH_CALUDE_square_side_length_l602_60270

theorem square_side_length (x : ℝ) (h : x > 0) : 
  x^2 = x * 2 / 2 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l602_60270


namespace NUMINAMATH_CALUDE_star_polygon_points_l602_60280

/-- A regular star polygon with ℓ points -/
structure StarPolygon where
  ℓ : ℕ
  x_angle : Real
  y_angle : Real
  h_x_less_y : x_angle = y_angle - 15
  h_external_sum : ℓ * (x_angle + y_angle) = 360
  h_internal_sum : ℓ * (180 - x_angle - y_angle) = 2 * 360

/-- Theorem: The number of points in the star polygon is 24 -/
theorem star_polygon_points (s : StarPolygon) : s.ℓ = 24 := by
  sorry

end NUMINAMATH_CALUDE_star_polygon_points_l602_60280


namespace NUMINAMATH_CALUDE_inequality_proof_l602_60231

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^5 + 4) ≥ 30 ∧
  ((a^2 + 1) * (b^3 + 2) * (c^5 + 4) = 30 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l602_60231


namespace NUMINAMATH_CALUDE_trapezoid_height_l602_60252

/-- A trapezoid with given area and sum of diagonals has a specific height -/
theorem trapezoid_height (area : ℝ) (sum_diagonals : ℝ) (height : ℝ) :
  area = 2 →
  sum_diagonals = 4 →
  height = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_height_l602_60252


namespace NUMINAMATH_CALUDE_factor_expression_l602_60291

theorem factor_expression (b : ℝ) : 280 * b^2 + 56 * b = 56 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l602_60291


namespace NUMINAMATH_CALUDE_square_field_diagonal_l602_60219

theorem square_field_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 50 → diagonal = 10 := by sorry

end NUMINAMATH_CALUDE_square_field_diagonal_l602_60219


namespace NUMINAMATH_CALUDE_diane_has_27_cents_l602_60212

/-- The amount of money Diane has, given the cost of cookies and the additional amount she needs. -/
def dianes_money (cookie_cost additional_needed : ℕ) : ℕ :=
  cookie_cost - additional_needed

/-- Theorem stating that Diane has 27 cents given the problem conditions. -/
theorem diane_has_27_cents :
  dianes_money 65 38 = 27 := by
  sorry

end NUMINAMATH_CALUDE_diane_has_27_cents_l602_60212


namespace NUMINAMATH_CALUDE_centroid_altitude_distance_l602_60226

/-- Triangle XYZ with sides a, b, c and centroid G -/
structure Triangle where
  a : ℝ  -- side XY
  b : ℝ  -- side XZ
  c : ℝ  -- side YZ
  G : ℝ × ℝ  -- centroid coordinates

/-- The foot of the altitude from a point to a line segment -/
def altitudeFoot (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem: In a triangle with sides 12, 15, and 23, the distance from the centroid
    to the foot of the altitude from the centroid to the longest side is 40/23 -/
theorem centroid_altitude_distance (t : Triangle) 
    (h1 : t.a = 12) (h2 : t.b = 15) (h3 : t.c = 23) : 
    let Q := altitudeFoot t.G (⟨0, 0⟩, ⟨t.c, 0⟩)  -- Assuming YZ is on x-axis
    distance t.G Q = 40 / 23 := by
  sorry

end NUMINAMATH_CALUDE_centroid_altitude_distance_l602_60226


namespace NUMINAMATH_CALUDE_max_students_l602_60242

/-- Represents the relationship between students -/
def knows (n : ℕ) := Fin n → Fin n → Prop

/-- At least two out of any three students know each other -/
def three_two_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c : Fin n, a ≠ b ∧ b ≠ c ∧ a ≠ c →
    k a b ∨ k b c ∨ k a c

/-- At least two out of any four students do not know each other -/
def four_two_dont_know (n : ℕ) (k : knows n) : Prop :=
  ∀ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(k a b) ∨ ¬(k a c) ∨ ¬(k a d) ∨ ¬(k b c) ∨ ¬(k b d) ∨ ¬(k c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students : 
  (∃ (k : knows 8), three_two_know 8 k ∧ four_two_dont_know 8 k) ∧
  (∀ n > 8, ¬∃ (k : knows n), three_two_know n k ∧ four_two_dont_know n k) :=
sorry

end NUMINAMATH_CALUDE_max_students_l602_60242


namespace NUMINAMATH_CALUDE_expression_simplification_l602_60266

theorem expression_simplification (w v : ℝ) :
  3 * w + 5 * w + 7 * v + 9 * w + 11 * v + 15 = 17 * w + 18 * v + 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l602_60266


namespace NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l602_60260

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def factorial (n : ℕ) : ℕ := Nat.factorial n

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_three_digit_non_divisor :
  ∀ k : ℕ, 100 ≤ k → k < 101 →
    is_divisor (sum_of_squares k) (factorial k) →
    ¬ is_divisor (sum_of_squares 101) (factorial 101) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_non_divisor_l602_60260


namespace NUMINAMATH_CALUDE_sum_of_squares_l602_60239

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 50) : x^2 + y^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l602_60239


namespace NUMINAMATH_CALUDE_equality_implies_equal_expressions_l602_60256

theorem equality_implies_equal_expressions (a b : ℝ) : a = b → 2 * (a - 1) = 2 * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_equality_implies_equal_expressions_l602_60256


namespace NUMINAMATH_CALUDE_impossibility_of_broken_line_l602_60258

/-- Represents a segment in the figure -/
structure Segment where
  id : Nat

/-- Represents a region in the figure -/
structure Region where
  segments : Finset Segment

/-- Represents the entire figure -/
structure Figure where
  segments : Finset Segment
  regions : Finset Region

/-- A broken line (polygonal chain) -/
structure BrokenLine where
  intersections : Finset Segment

/-- The theorem statement -/
theorem impossibility_of_broken_line (fig : Figure) 
  (h1 : fig.segments.card = 16)
  (h2 : ∃ r1 r2 r3 : Region, r1 ∈ fig.regions ∧ r2 ∈ fig.regions ∧ r3 ∈ fig.regions ∧ 
        r1.segments.card = 5 ∧ r2.segments.card = 5 ∧ r3.segments.card = 5) :
  ¬∃ (bl : BrokenLine), bl.intersections = fig.segments :=
sorry

end NUMINAMATH_CALUDE_impossibility_of_broken_line_l602_60258


namespace NUMINAMATH_CALUDE_six_digit_nondecreasing_remainder_l602_60274

theorem six_digit_nondecreasing_remainder (n : Nat) (k : Nat) : 
  n = 6 → k = 9 → (Nat.choose (n + k - 1) n) % 1000 = 3 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_nondecreasing_remainder_l602_60274


namespace NUMINAMATH_CALUDE_symmetric_circle_equation_l602_60207

/-- Given two circles C₁ and C₂, where C₁ has equation (x+1)²+(y-1)²=1 and C₂ is symmetric to C₁
    with respect to the line x-y-1=0, prove that the equation of C₂ is (x-2)²+(y+2)²=1 -/
theorem symmetric_circle_equation (x y : ℝ) :
  let C₁ : ℝ → ℝ → Prop := λ x y => (x + 1)^2 + (y - 1)^2 = 1
  let symmetry_line : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let C₂ : ℝ → ℝ → Prop := λ x y => (x - 2)^2 + (y + 2)^2 = 1
  (∀ x y, C₁ x y ↔ (x + 1)^2 + (y - 1)^2 = 1) →
  (∀ x₁ y₁ x₂ y₂, C₁ x₁ y₁ → C₂ x₂ y₂ → 
    ∃ x_sym y_sym, symmetry_line x_sym y_sym ∧
    (x₂ - x_sym = x_sym - x₁) ∧ (y₂ - y_sym = y_sym - y₁)) →
  (∀ x y, C₂ x y ↔ (x - 2)^2 + (y + 2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_circle_equation_l602_60207


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l602_60284

/-- Given a cone with an acute triangular cross-section and slant height 4,
    where the maximum area of cross-sections passing through the vertex is 4√3,
    prove that the central angle of the sector in the lateral surface development is π. -/
theorem cone_lateral_surface_angle (h : ℝ) (θ : ℝ) (r : ℝ) (α : ℝ) : 
  h = 4 → 
  θ < π / 2 →
  (1 / 2) * h * h * Real.sin θ = 4 * Real.sqrt 3 →
  r = 2 →
  α = 2 * π * r / h →
  α = π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l602_60284


namespace NUMINAMATH_CALUDE_equation_solution_l602_60202

theorem equation_solution : ∃ x : ℚ, 25 - 8 = 3 * x + 1 ∧ x = 16/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l602_60202


namespace NUMINAMATH_CALUDE_min_value_a_over_b_l602_60299

theorem min_value_a_over_b (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (h : 2 * Real.sqrt a + b = 1) :
  ∃ (x : ℝ), x = a / b ∧ ∀ (y : ℝ), y = a / b → x ≤ y ∧ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_over_b_l602_60299


namespace NUMINAMATH_CALUDE_ellipse_condition_l602_60285

def is_ellipse_equation (m : ℝ) : Prop :=
  m > 2 ∧ m < 5 ∧ m ≠ 7/2

theorem ellipse_condition (m : ℝ) :
  (2 < m ∧ m < 5) → (is_ellipse_equation m) ∧
  ∃ m', is_ellipse_equation m' ∧ ¬(2 < m' ∧ m' < 5) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l602_60285


namespace NUMINAMATH_CALUDE_fraction_equivalence_l602_60247

theorem fraction_equivalence : (10 : ℝ) / (8 * 60) = 0.1 / (0.8 * 60) := by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l602_60247


namespace NUMINAMATH_CALUDE_selling_multiple_satisfies_profit_equation_l602_60278

/-- The multiple of the value of components that John sells computers for -/
def selling_multiple : ℝ := 1.4

/-- Cost of parts for one computer -/
def parts_cost : ℝ := 800

/-- Number of computers built per month -/
def computers_per_month : ℕ := 60

/-- Monthly rent -/
def monthly_rent : ℝ := 5000

/-- Monthly non-rent extra expenses -/
def extra_expenses : ℝ := 3000

/-- Monthly profit -/
def monthly_profit : ℝ := 11200

/-- Theorem stating that the selling multiple satisfies the profit equation -/
theorem selling_multiple_satisfies_profit_equation :
  computers_per_month * parts_cost * selling_multiple -
  (computers_per_month * parts_cost + monthly_rent + extra_expenses) = monthly_profit := by
  sorry

end NUMINAMATH_CALUDE_selling_multiple_satisfies_profit_equation_l602_60278


namespace NUMINAMATH_CALUDE_line_symmetry_l602_60222

-- Define the lines
def line_l (x y : ℝ) : Prop := x - y - 1 = 0
def line_l1 (x y : ℝ) : Prop := 2*x - y - 2 = 0
def line_l2 (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define symmetry with respect to a line
def symmetric_wrt (l1 l2 l : (ℝ → ℝ → Prop)) : Prop :=
  ∀ (x y : ℝ), l1 x y ↔ ∃ (x' y' : ℝ), l2 x' y' ∧ l ((x + x')/2) ((y + y')/2)

-- Theorem statement
theorem line_symmetry :
  symmetric_wrt line_l1 line_l2 line_l :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l602_60222


namespace NUMINAMATH_CALUDE_sequence_general_term_formula_l602_60246

/-- Given a quadratic equation with real roots, prove the general term formula for a sequence defined by a recurrence relation. -/
theorem sequence_general_term_formula 
  (p q : ℝ) 
  (hq : q ≠ 0) 
  (α β : ℝ) 
  (hroots : α^2 - p*α + q = 0 ∧ β^2 - p*β + q = 0) 
  (a : ℕ → ℝ) 
  (ha1 : a 1 = p) 
  (ha2 : a 2 = p^2 - q) 
  (han : ∀ n : ℕ, n ≥ 3 → a n = p * a (n-1) - q * a (n-2)) :
  ∀ n : ℕ, n ≥ 1 → a n = (α^(n+1) - β^(n+1)) / (α - β) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_formula_l602_60246


namespace NUMINAMATH_CALUDE_third_number_in_sequence_l602_60288

theorem third_number_in_sequence (x : ℕ) 
  (h : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 60) : 
  x + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_third_number_in_sequence_l602_60288


namespace NUMINAMATH_CALUDE_interest_calculation_l602_60227

/-- Given a principal amount and number of years, if the simple interest
    at 5% per annum is Rs. 56 and the compound interest at the same rate
    is Rs. 57.40, then the number of years is 2. -/
theorem interest_calculation (P n : ℝ) : 
  P * n / 20 = 56 →
  P * ((1 + 5/100)^n - 1) = 57.40 →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l602_60227


namespace NUMINAMATH_CALUDE_evaluate_expression_l602_60269

theorem evaluate_expression : 48^3 + 3*(48^2)*4 + 3*48*(4^2) + 4^3 = 140608 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l602_60269


namespace NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l602_60261

theorem sixth_grade_homework_forgetfulness
  (group_a_size : ℕ)
  (group_b_size : ℕ)
  (group_a_forget_rate : ℚ)
  (group_b_forget_rate : ℚ)
  (h1 : group_a_size = 20)
  (h2 : group_b_size = 80)
  (h3 : group_a_forget_rate = 1/5)
  (h4 : group_b_forget_rate = 3/20)
  : (((group_a_size * group_a_forget_rate + group_b_size * group_b_forget_rate) /
     (group_a_size + group_b_size)) : ℚ) = 4/25 :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_homework_forgetfulness_l602_60261


namespace NUMINAMATH_CALUDE_pascal_theorem_l602_60294

-- Define the conic section
structure ConicSection where
  -- Add necessary fields to define a conic section
  -- This is a placeholder and should be replaced with actual definition

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ -- ax + by + c = 0

-- Define the hexagon
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point

-- Function to check if a point lies on a conic section
def pointOnConic (p : Point) (c : ConicSection) : Prop :=
  sorry -- Define the condition for a point to lie on the conic section

-- Function to check if two lines intersect at a point
def linesIntersectAt (l1 l2 : Line) (p : Point) : Prop :=
  sorry -- Define the condition for two lines to intersect at a given point

-- Function to check if three points are collinear
def areCollinear (p1 p2 p3 : Point) : Prop :=
  sorry -- Define the condition for three points to be collinear

-- Theorem statement
theorem pascal_theorem (c : ConicSection) (h : Hexagon) 
  (hInscribed : pointOnConic h.A c ∧ pointOnConic h.B c ∧ pointOnConic h.C c ∧ 
                pointOnConic h.D c ∧ pointOnConic h.E c ∧ pointOnConic h.F c)
  (M N P : Point)
  (hM : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) M) -- AB and DE
  (hN : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) N) -- BC and EF
  (hP : linesIntersectAt (Line.mk 0 0 0) (Line.mk 0 0 0) P) -- CD and FA
  : areCollinear M N P :=
by
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_pascal_theorem_l602_60294


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_x_minus_one_to_eighth_l602_60235

theorem coefficient_x_squared_in_x_minus_one_to_eighth (x : ℝ) : 
  (∃ a b c d e f g : ℝ, (x - 1)^8 = x^8 + 8*x^7 + 28*x^6 + 56*x^5 + 70*x^4 + a*x^3 + b*x^2 + c*x + d) ∧ 
  (∃ p q r s t u v : ℝ, (x - 1)^8 = p*x^7 + q*x^6 + r*x^5 + s*x^4 + t*x^3 + 28*x^2 + u*x + v) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_x_minus_one_to_eighth_l602_60235


namespace NUMINAMATH_CALUDE_last_three_digits_of_2005_power_2005_l602_60240

theorem last_three_digits_of_2005_power_2005 :
  ∃ k : ℕ, 2005^2005 = 1000 * k + 125 :=
sorry

end NUMINAMATH_CALUDE_last_three_digits_of_2005_power_2005_l602_60240


namespace NUMINAMATH_CALUDE_parking_spots_fourth_level_l602_60292

theorem parking_spots_fourth_level 
  (total_levels : Nat) 
  (first_level_spots : Nat) 
  (second_level_diff : Nat) 
  (third_level_diff : Nat) 
  (total_spots : Nat) :
  total_levels = 4 →
  first_level_spots = 4 →
  second_level_diff = 7 →
  third_level_diff = 6 →
  total_spots = 46 →
  let second_level_spots := first_level_spots + second_level_diff
  let third_level_spots := second_level_spots + third_level_diff
  let fourth_level_spots := total_spots - (first_level_spots + second_level_spots + third_level_spots)
  fourth_level_spots = 14 := by
sorry

end NUMINAMATH_CALUDE_parking_spots_fourth_level_l602_60292


namespace NUMINAMATH_CALUDE_square_sum_reciprocal_l602_60254

theorem square_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_reciprocal_l602_60254


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l602_60218

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem geometric_sequence_sum 
  (a : ℕ → ℚ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 2 * a 5 = -3/4) 
  (h_sum : a 2 + a 3 + a 4 + a 5 = 5/4) : 
  1 / a 2 + 1 / a 3 + 1 / a 4 + 1 / a 5 = -5/3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l602_60218


namespace NUMINAMATH_CALUDE_some_club_members_not_debate_team_l602_60216

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Club : U → Prop)  -- x is a club member
variable (Punctual : U → Prop)  -- x is punctual
variable (DebateTeam : U → Prop)  -- x is a debate team member

-- Define the premises
variable (h1 : ∃ x, Club x ∧ ¬Punctual x)  -- Some club members are not punctual
variable (h2 : ∀ x, DebateTeam x → Punctual x)  -- All members of the debate team are punctual

-- State the theorem
theorem some_club_members_not_debate_team :
  ∃ x, Club x ∧ ¬DebateTeam x :=
by sorry

end NUMINAMATH_CALUDE_some_club_members_not_debate_team_l602_60216


namespace NUMINAMATH_CALUDE_addition_subtraction_reduces_system_l602_60208

/-- A method for solving systems of linear equations with two variables -/
inductive SolvingMethod
| Substitution
| AdditionSubtraction

/-- Represents a system of linear equations with two variables -/
structure LinearSystem :=
  (equations : List (LinearEquation))

/-- Represents a linear equation -/
structure LinearEquation :=
  (coefficients : List ℝ)
  (constant : ℝ)

/-- A function that determines if a method reduces a system to a single variable -/
def reduces_to_single_variable (method : SolvingMethod) (system : LinearSystem) : Prop :=
  sorry

/-- The theorem stating that the addition-subtraction method reduces a system to a single variable -/
theorem addition_subtraction_reduces_system :
  ∀ (system : LinearSystem),
    reduces_to_single_variable SolvingMethod.AdditionSubtraction system :=
  sorry

end NUMINAMATH_CALUDE_addition_subtraction_reduces_system_l602_60208


namespace NUMINAMATH_CALUDE_unique_permutations_3_3_3_6_eq_4_l602_60255

/-- The number of unique permutations of a multiset with 4 elements, where 3 elements are identical --/
def unique_permutations_3_3_3_6 : ℕ :=
  Nat.factorial 4 / Nat.factorial 3

theorem unique_permutations_3_3_3_6_eq_4 : 
  unique_permutations_3_3_3_6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_permutations_3_3_3_6_eq_4_l602_60255


namespace NUMINAMATH_CALUDE_inequality_squared_not_always_true_l602_60210

theorem inequality_squared_not_always_true : ¬ ∀ x y : ℝ, x < y → x^2 < y^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_squared_not_always_true_l602_60210


namespace NUMINAMATH_CALUDE_tiles_per_row_l602_60244

-- Define the area of the room in square feet
def room_area : ℝ := 400

-- Define the side length of a tile in inches
def tile_side : ℝ := 8

-- Define the conversion factor from feet to inches
def feet_to_inches : ℝ := 12

-- Theorem to prove
theorem tiles_per_row : 
  ⌊(feet_to_inches * (room_area.sqrt)) / tile_side⌋ = 30 := by
  sorry

end NUMINAMATH_CALUDE_tiles_per_row_l602_60244


namespace NUMINAMATH_CALUDE_soda_price_proof_l602_60295

/-- The regular price per can of soda -/
def regular_price : ℝ := sorry

/-- The discounted price per can when purchased in 24-can cases -/
def discounted_price : ℝ := regular_price * 0.85

/-- The total price for 100 cans purchased in 24-can cases -/
def total_price : ℝ := 34

theorem soda_price_proof :
  discounted_price * 100 = total_price ∧ regular_price = 0.40 :=
sorry

end NUMINAMATH_CALUDE_soda_price_proof_l602_60295


namespace NUMINAMATH_CALUDE_quadratic_equation_proof_l602_60233

theorem quadratic_equation_proof (c : ℝ) : 
  (∃ c_modified : ℝ, c = c_modified + 2 ∧ (-1)^2 + 4*(-1) + c_modified = 0) →
  c = 5 ∧ ∀ x : ℝ, x^2 + 4*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_proof_l602_60233


namespace NUMINAMATH_CALUDE_remainder_fifty_pow_2019_plus_one_mod_seven_l602_60296

theorem remainder_fifty_pow_2019_plus_one_mod_seven (n : ℕ) : (50^2019 + 1) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_fifty_pow_2019_plus_one_mod_seven_l602_60296
