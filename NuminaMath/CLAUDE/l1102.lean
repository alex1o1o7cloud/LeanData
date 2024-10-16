import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_t_range_l1102_110271

/-- A line tangent to a circle and intersecting a parabola at two points -/
structure TangentLineIntersectingParabola where
  k : ℝ
  t : ℝ
  tangent_condition : k^2 = t^2 + 2*t
  distinct_intersections : 16*(t^2 + 2*t) + 16*t > 0

/-- The range of t values for a tangent line intersecting a parabola at two points -/
theorem tangent_line_t_range (l : TangentLineIntersectingParabola) :
  l.t > 0 ∨ l.t < -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_t_range_l1102_110271


namespace NUMINAMATH_CALUDE_round_trip_time_l1102_110244

/-- Calculates the total time for a round trip by boat given the boat's speed, stream speed, and distance. -/
theorem round_trip_time (boat_speed stream_speed distance : ℝ) 
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 2)
  (h3 : distance = 7200)
  (h4 : boat_speed > stream_speed) : 
  ∃ (time : ℝ), abs (time - (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed))) < 0.0001 ∧ 
                abs (time - 914.2857) < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_round_trip_time_l1102_110244


namespace NUMINAMATH_CALUDE_first_digit_is_one_l1102_110242

def base_three_number : List Nat := [1, 2, 0, 2, 2, 1, 0, 1, 1, 2, 2, 2, 1, 0, 1, 2, 2, 1, 0, 2]

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3^i)) 0

def decimal_to_base_nine (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

theorem first_digit_is_one :
  (decimal_to_base_nine (base_three_to_decimal base_three_number)).head? = some 1 := by
  sorry

end NUMINAMATH_CALUDE_first_digit_is_one_l1102_110242


namespace NUMINAMATH_CALUDE_recipe_multiplier_is_six_l1102_110245

/-- Represents the ratio of butter to flour in a recipe -/
structure RecipeRatio where
  butter : ℚ
  flour : ℚ

/-- The original recipe ratio -/
def originalRatio : RecipeRatio := { butter := 2, flour := 5 }

/-- The amount of butter used in the new recipe -/
def newButterAmount : ℚ := 12

/-- Calculates how many times the original recipe is being made -/
def recipeMultiplier (original : RecipeRatio) (newButter : ℚ) : ℚ :=
  newButter / original.butter

theorem recipe_multiplier_is_six :
  recipeMultiplier originalRatio newButterAmount = 6 := by
  sorry

#eval recipeMultiplier originalRatio newButterAmount

end NUMINAMATH_CALUDE_recipe_multiplier_is_six_l1102_110245


namespace NUMINAMATH_CALUDE_largest_power_of_five_dividing_sum_l1102_110204

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_factorials : ℕ := factorial 50 + factorial 52 + factorial 54

theorem largest_power_of_five_dividing_sum : 
  (∃ (k : ℕ), sum_of_factorials = 5^12 * k ∧ ¬(∃ (m : ℕ), sum_of_factorials = 5^13 * m)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_five_dividing_sum_l1102_110204


namespace NUMINAMATH_CALUDE_min_sum_distances_min_sum_distances_equality_l1102_110224

theorem min_sum_distances (u v : ℝ) : 
  Real.sqrt (u^2 + v^2) + Real.sqrt ((u - 1)^2 + v^2) + 
  Real.sqrt (u^2 + (v - 1)^2) + Real.sqrt ((u - 1)^2 + (v - 1)^2) ≥ 2 * Real.sqrt 2 :=
by sorry

theorem min_sum_distances_equality : 
  Real.sqrt ((1/2)^2 + (1/2)^2) + Real.sqrt ((1/2 - 1)^2 + (1/2)^2) + 
  Real.sqrt ((1/2)^2 + (1/2 - 1)^2) + Real.sqrt ((1/2 - 1)^2 + (1/2 - 1)^2) = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_distances_min_sum_distances_equality_l1102_110224


namespace NUMINAMATH_CALUDE_volume_of_rotated_composite_region_l1102_110218

/-- The volume of a solid formed by rotating a composite region about the y-axis -/
theorem volume_of_rotated_composite_region :
  let square_side : ℝ := 4
  let rectangle_width : ℝ := 5
  let rectangle_height : ℝ := 3
  let volume_square : ℝ := π * (square_side / 2)^2 * square_side
  let volume_rectangle : ℝ := π * (rectangle_height / 2)^2 * rectangle_width
  let total_volume : ℝ := volume_square + volume_rectangle
  total_volume = (109 * π) / 4 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_rotated_composite_region_l1102_110218


namespace NUMINAMATH_CALUDE_net_percentage_gain_calculation_l1102_110231

/-- Calculate the net percentage gain from buying and selling glass bowls and ceramic plates --/
theorem net_percentage_gain_calculation 
  (glass_bowls_bought : ℕ) 
  (glass_bowls_price : ℚ) 
  (ceramic_plates_bought : ℕ) 
  (ceramic_plates_price : ℚ) 
  (discount_rate : ℚ) 
  (glass_bowls_sold : ℕ) 
  (glass_bowls_sell_price : ℚ) 
  (ceramic_plates_sold : ℕ) 
  (ceramic_plates_sell_price : ℚ) 
  (glass_bowls_broken : ℕ) 
  (ceramic_plates_broken : ℕ) :
  glass_bowls_bought = 250 →
  glass_bowls_price = 18 →
  ceramic_plates_bought = 150 →
  ceramic_plates_price = 25 →
  discount_rate = 5 / 100 →
  glass_bowls_sold = 200 →
  glass_bowls_sell_price = 25 →
  ceramic_plates_sold = 120 →
  ceramic_plates_sell_price = 32 →
  glass_bowls_broken = 30 →
  ceramic_plates_broken = 10 →
  ∃ (net_percentage_gain : ℚ), 
    abs (net_percentage_gain - (271 / 10000 : ℚ)) < (1 / 1000 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_net_percentage_gain_calculation_l1102_110231


namespace NUMINAMATH_CALUDE_triangle_angle_ratio_range_l1102_110201

theorem triangle_angle_ratio_range (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  A + B + C = π →
  S = (1/2) * b * c * Real.sin A →
  a^2 = 2*S + (b-c)^2 →
  1 - (1/2) * Real.sin A = (b^2 + c^2 - a^2) / (2*b*c) →
  ∃ (l u : ℝ), l = 2 * Real.sqrt 2 ∧ u = 59/15 ∧
    (∀ x, l ≤ x ∧ x < u ↔ 
      ∃ (B' C' : ℝ), 0 < B' ∧ B' < π/2 ∧ 0 < C' ∧ C' < π/2 ∧
        x = (2 * Real.sin B'^2 + Real.sin C'^2) / (Real.sin B' * Real.sin C')) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_ratio_range_l1102_110201


namespace NUMINAMATH_CALUDE_gcd_8251_6105_l1102_110259

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_8251_6105_l1102_110259


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1102_110254

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y, 2*x - 3*y + 6 = 0 → bx - 3*y - 4 = 0 → 
    (2/3) * (b/3) = -1) → 
  b = -9/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1102_110254


namespace NUMINAMATH_CALUDE_safe_mushrooms_l1102_110264

/-- Given the following conditions about mushroom foraging:
  * The total number of mushrooms is 32
  * The number of poisonous mushrooms is twice the number of safe mushrooms
  * There are 5 uncertain mushrooms
  * The sum of safe, poisonous, and uncertain mushrooms equals the total
  Prove that the number of safe mushrooms is 9. -/
theorem safe_mushrooms (total : ℕ) (safe : ℕ) (poisonous : ℕ) (uncertain : ℕ) 
  (h1 : total = 32)
  (h2 : poisonous = 2 * safe)
  (h3 : uncertain = 5)
  (h4 : safe + poisonous + uncertain = total) :
  safe = 9 := by sorry

end NUMINAMATH_CALUDE_safe_mushrooms_l1102_110264


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1102_110250

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1102_110250


namespace NUMINAMATH_CALUDE_foci_distance_of_hyperbola_l1102_110206

def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 18 * x - 16 * y^2 - 32 * y = -144

theorem foci_distance_of_hyperbola :
  ∃ (h : ℝ → ℝ → ℝ), 
    (∀ x y, hyperbola_equation x y → 
      h x y = (let a := 4; let b := 3; Real.sqrt (a^2 + b^2) * 2)) ∧
    (∀ x y, hyperbola_equation x y → h x y = 10) :=
sorry

end NUMINAMATH_CALUDE_foci_distance_of_hyperbola_l1102_110206


namespace NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l1102_110249

theorem no_positive_a_satisfies_inequality :
  ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_a_satisfies_inequality_l1102_110249


namespace NUMINAMATH_CALUDE_randolph_sydney_age_difference_l1102_110270

/-- The age difference between Randolph and Sydney -/
def ageDifference (randolphAge sydneyAge : ℕ) : ℕ := randolphAge - sydneyAge

/-- Theorem stating the age difference between Randolph and Sydney -/
theorem randolph_sydney_age_difference :
  ∀ (sherryAge : ℕ),
    sherryAge = 25 →
    ∀ (sydneyAge : ℕ),
      sydneyAge = 2 * sherryAge →
      ∀ (randolphAge : ℕ),
        randolphAge = 55 →
        ageDifference randolphAge sydneyAge = 5 := by
  sorry

end NUMINAMATH_CALUDE_randolph_sydney_age_difference_l1102_110270


namespace NUMINAMATH_CALUDE_range_of_quadratic_expression_l1102_110215

theorem range_of_quadratic_expression (x y : ℝ) :
  (4 * x^2 - 2 * Real.sqrt 3 * x * y + 4 * y^2 = 13) →
  (10 - 4 * Real.sqrt 3 ≤ x^2 + 4 * y^2) ∧ (x^2 + 4 * y^2 ≤ 10 + 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_quadratic_expression_l1102_110215


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1102_110212

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1102_110212


namespace NUMINAMATH_CALUDE_radford_distance_at_finish_l1102_110293

/-- Represents the race between Radford and Peter -/
structure Race where
  radford_initial_lead : ℝ
  peter_lead_after_3min : ℝ
  race_duration : ℝ
  peter_speed_advantage : ℝ

/-- Calculates the distance between Radford and Peter at the end of the race -/
def final_distance (race : Race) : ℝ :=
  race.peter_lead_after_3min + race.peter_speed_advantage * (race.race_duration - 3)

/-- Theorem stating that Radford is 82 meters behind Peter at the end of the race -/
theorem radford_distance_at_finish (race : Race) 
  (h1 : race.radford_initial_lead = 30)
  (h2 : race.peter_lead_after_3min = 18)
  (h3 : race.race_duration = 7)
  (h4 : race.peter_speed_advantage = 16) :
  final_distance race = 82 := by
  sorry

end NUMINAMATH_CALUDE_radford_distance_at_finish_l1102_110293


namespace NUMINAMATH_CALUDE_initial_number_proof_l1102_110260

theorem initial_number_proof (x : ℝ) : 
  ((5 * x - 20) / 2 - 100 = 4) → x = 45.6 := by
sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1102_110260


namespace NUMINAMATH_CALUDE_mikes_shortfall_l1102_110214

theorem mikes_shortfall (max_marks : ℕ) (mikes_score : ℕ) (passing_percentage : ℚ) : 
  max_marks = 750 → 
  mikes_score = 212 → 
  passing_percentage = 30 / 100 → 
  (↑max_marks * passing_percentage).floor - mikes_score = 13 := by
  sorry

end NUMINAMATH_CALUDE_mikes_shortfall_l1102_110214


namespace NUMINAMATH_CALUDE_pillow_average_cost_l1102_110262

theorem pillow_average_cost (n : ℕ) (avg_cost : ℚ) (additional_cost : ℚ) :
  n = 4 →
  avg_cost = 5 →
  additional_cost = 10 →
  (n * avg_cost + additional_cost) / (n + 1) = 6 := by
sorry

end NUMINAMATH_CALUDE_pillow_average_cost_l1102_110262


namespace NUMINAMATH_CALUDE_sum_and_count_theorem_l1102_110282

def sum_of_range (a b : ℕ) : ℕ := ((b - a + 1) * (a + b)) / 2

def count_even_in_range (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem : 
  sum_of_range 40 60 + count_even_in_range 40 60 = 1061 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_theorem_l1102_110282


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l1102_110226

theorem simplify_and_rationalize (x : ℝ) :
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l1102_110226


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l1102_110269

theorem sqrt_D_irrational (x : ℝ) : 
  ∀ (y : ℝ), y ^ 2 ≠ 3 * (2 * x) ^ 2 + 3 * (2 * x + 1) ^ 2 + (4 * x + 1) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l1102_110269


namespace NUMINAMATH_CALUDE_two_digit_number_property_l1102_110248

theorem two_digit_number_property (x y : ℕ) : 
  x < 10 → y < 10 → x ≠ 0 →
  x^2 + y^2 = 10*x + x*y →
  10*x + y - 36 = 10*y + x →
  10*x + y = 48 ∨ 10*x + y = 37 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l1102_110248


namespace NUMINAMATH_CALUDE_instrument_players_l1102_110238

theorem instrument_players (total_people : ℕ) 
  (fraction_at_least_one : ℚ) 
  (prob_exactly_one : ℚ) 
  (h1 : total_people = 800) 
  (h2 : fraction_at_least_one = 2/5) 
  (h3 : prob_exactly_one = 28/100) : 
  ℕ := by
  sorry

#check instrument_players

end NUMINAMATH_CALUDE_instrument_players_l1102_110238


namespace NUMINAMATH_CALUDE_pie_crust_flour_usage_l1102_110247

theorem pie_crust_flour_usage 
  (original_crusts : ℕ) 
  (original_flour_per_crust : ℚ) 
  (new_crusts : ℕ) :
  original_crusts = 30 →
  original_flour_per_crust = 1/6 →
  new_crusts = 25 →
  (original_crusts * original_flour_per_crust) / new_crusts = 1/5 :=
by sorry

end NUMINAMATH_CALUDE_pie_crust_flour_usage_l1102_110247


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1102_110235

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 7 → b = 24 → c^2 = a^2 + b^2 → c = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1102_110235


namespace NUMINAMATH_CALUDE_g_of_3_eq_6_l1102_110263

/-- A function satisfying the given conditions -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 2 ∧ ∀ x y : ℝ, g (x^2 + y^2) = (x + y) * (g x + g y)

/-- Theorem stating that g(3) = 6 for any function satisfying the conditions -/
theorem g_of_3_eq_6 (g : ℝ → ℝ) (h : special_function g) : g 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_eq_6_l1102_110263


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l1102_110237

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_decreasing : decreasing_on (fun x ↦ f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l1102_110237


namespace NUMINAMATH_CALUDE_pentagon_interior_angle_mean_l1102_110298

/-- The mean value of the measures of the interior angles of a pentagon is 108 degrees. -/
theorem pentagon_interior_angle_mean :
  let n : ℕ := 5  -- number of sides in a pentagon
  let sum_of_angles : ℝ := (n - 2) * 180  -- sum of interior angles
  let mean_angle : ℝ := sum_of_angles / n  -- mean value of interior angles
  mean_angle = 108 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_interior_angle_mean_l1102_110298


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l1102_110291

-- Define the polynomial
def f (x : ℂ) : ℂ := x^3 - x^2 - 1

-- State the theorem
theorem cubic_equation_roots :
  ∃ (a b c : ℂ), 
    (a + b + c = 1) ∧ 
    (a * b + a * c + b * c = 0) ∧ 
    (a * b * c = -1) ∧ 
    (f a = 0) ∧ (f b = 0) ∧ (f c = 0) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l1102_110291


namespace NUMINAMATH_CALUDE_largest_common_divisor_510_399_l1102_110289

theorem largest_common_divisor_510_399 : Nat.gcd 510 399 = 57 := by
  sorry

end NUMINAMATH_CALUDE_largest_common_divisor_510_399_l1102_110289


namespace NUMINAMATH_CALUDE_number_decrease_theorem_l1102_110274

theorem number_decrease_theorem :
  (∃ (N k a x : ℕ), k ≥ 1 ∧ 1 ≤ a ∧ a ≤ 9 ∧ x < 10^k ∧ N = 10^k * a + x ∧ N = 57 * x) ∧
  (¬ ∃ (N k a x : ℕ), k ≥ 1 ∧ 1 ≤ a ∧ a ≤ 9 ∧ x < 10^k ∧ N = 10^k * a + x ∧ N = 58 * x) :=
by sorry

end NUMINAMATH_CALUDE_number_decrease_theorem_l1102_110274


namespace NUMINAMATH_CALUDE_remainder_eleven_pow_thousand_mod_five_hundred_l1102_110257

theorem remainder_eleven_pow_thousand_mod_five_hundred :
  11^1000 % 500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_eleven_pow_thousand_mod_five_hundred_l1102_110257


namespace NUMINAMATH_CALUDE_furniture_cost_prices_l1102_110261

def computer_table_price : ℝ := 8450
def bookshelf_price : ℝ := 6250
def chair_price : ℝ := 3400

def computer_table_markup : ℝ := 0.30
def bookshelf_markup : ℝ := 0.25
def chair_discount : ℝ := 0.15

theorem furniture_cost_prices :
  ∃ (computer_table_cost bookshelf_cost chair_cost : ℝ),
    computer_table_cost = computer_table_price / (1 + computer_table_markup) ∧
    bookshelf_cost = bookshelf_price / (1 + bookshelf_markup) ∧
    chair_cost = chair_price / (1 - chair_discount) ∧
    computer_table_cost = 6500 ∧
    bookshelf_cost = 5000 ∧
    chair_cost = 4000 :=
by sorry

end NUMINAMATH_CALUDE_furniture_cost_prices_l1102_110261


namespace NUMINAMATH_CALUDE_sum_of_cubic_and_quartic_terms_l1102_110277

theorem sum_of_cubic_and_quartic_terms (π : ℝ) : 3 * (3 - π)^3 + 4 * (2 - π)^4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubic_and_quartic_terms_l1102_110277


namespace NUMINAMATH_CALUDE_pond_volume_pond_volume_proof_l1102_110287

/-- The volume of a rectangular prism with dimensions 20 m, 10 m, and 5 m is 1000 cubic meters. -/
theorem pond_volume : ℝ → Prop :=
  fun volume =>
    let length : ℝ := 20
    let width : ℝ := 10
    let depth : ℝ := 5
    volume = length * width * depth ∧ volume = 1000

/-- Proof of the pond volume theorem -/
theorem pond_volume_proof : ∃ volume : ℝ, pond_volume volume := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_pond_volume_proof_l1102_110287


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l1102_110221

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the line segment MN
structure LineSegment :=
  (M N : ℝ × ℝ)

-- Define the parallel property
def isParallel (l1 l2 : LineSegment) : Prop := sorry

-- Define the length of a line segment
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_theorem (XYZ : Triangle) (MN : LineSegment) :
  isParallel MN (LineSegment.mk XYZ.X XYZ.Y) →
  length XYZ.X MN.M = 5 →
  length MN.M XYZ.Y = 8 →
  length MN.N XYZ.Z = 9 →
  length XYZ.Y XYZ.Z = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l1102_110221


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1102_110252

def U : Set ℝ := Set.univ

def A : Set ℝ := {-1, 0, 1, 2, 3}

def B : Set ℝ := {x | x ≥ 2}

theorem intersection_with_complement :
  A ∩ (U \ B) = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1102_110252


namespace NUMINAMATH_CALUDE_parking_lot_cars_l1102_110290

theorem parking_lot_cars : ∃ (total : ℕ), 
  (total / 3 : ℚ) + (total / 2 : ℚ) + 86 = total ∧ total = 516 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_cars_l1102_110290


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1102_110240

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 * a2 + b1 * b2 = 0

-- The theorem to prove
theorem perpendicular_line_equation :
  ∃ (x y : ℝ), intersection_point x y ∧
  perpendicular 2 3 5 2 3 (-7) ∧
  (2 * x + 3 * y - 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1102_110240


namespace NUMINAMATH_CALUDE_certain_number_problem_l1102_110297

theorem certain_number_problem (x : ℚ) : 
  (((x + 5) * 2) / 5) - 5 = 44 / 2 → x = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1102_110297


namespace NUMINAMATH_CALUDE_maintenance_check_increase_l1102_110267

theorem maintenance_check_increase (original_time : ℝ) (increase_percent : ℝ) (new_time : ℝ) :
  original_time = 20 →
  increase_percent = 25 →
  new_time = original_time * (1 + increase_percent / 100) →
  new_time = 25 :=
by sorry

end NUMINAMATH_CALUDE_maintenance_check_increase_l1102_110267


namespace NUMINAMATH_CALUDE_leslie_garden_walkway_area_l1102_110229

/-- Represents Leslie's garden layout --/
structure GardenLayout where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  row_walkway_width : Nat
  column_walkway_width : Nat

/-- Calculates the total area of walkways in the garden --/
def walkway_area (garden : GardenLayout) : Nat :=
  let total_width := garden.columns * garden.bed_width + (garden.columns + 1) * garden.column_walkway_width
  let total_height := garden.rows * garden.bed_height + (garden.rows + 1) * garden.row_walkway_width
  let total_area := total_width * total_height
  let beds_area := garden.rows * garden.columns * garden.bed_width * garden.bed_height
  total_area - beds_area

/-- Leslie's garden layout --/
def leslie_garden : GardenLayout :=
  { rows := 4
  , columns := 3
  , bed_width := 8
  , bed_height := 3
  , row_walkway_width := 1
  , column_walkway_width := 2
  }

/-- Theorem stating that the walkway area in Leslie's garden is 256 square feet --/
theorem leslie_garden_walkway_area :
  walkway_area leslie_garden = 256 := by
  sorry

end NUMINAMATH_CALUDE_leslie_garden_walkway_area_l1102_110229


namespace NUMINAMATH_CALUDE_number_of_subsets_complement_union_l1102_110209

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 3}

-- Define set B
def B : Finset Nat := {1, 2, 4}

-- Theorem statement
theorem number_of_subsets_complement_union (U A B : Finset Nat) : 
  Finset.card (Finset.powerset ((U \ B) ∪ A)) = 8 :=
sorry

end NUMINAMATH_CALUDE_number_of_subsets_complement_union_l1102_110209


namespace NUMINAMATH_CALUDE_call_center_team_ratio_l1102_110222

theorem call_center_team_ratio (a b : ℚ) : 
  (∀ (c : ℚ), a * (3/5 * c) / (b * c) = 3/11) →
  a / b = 5/11 := by
sorry

end NUMINAMATH_CALUDE_call_center_team_ratio_l1102_110222


namespace NUMINAMATH_CALUDE_reciprocal_sum_l1102_110223

theorem reciprocal_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 6 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l1102_110223


namespace NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l1102_110200

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_fourth_fifth_sixth (seq : ArithmeticSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 42 :=
sorry

end NUMINAMATH_CALUDE_sum_fourth_fifth_sixth_l1102_110200


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l1102_110258

theorem quadratic_roots_problem (a b m p q : ℝ) : 
  (a^2 - m*a + 5 = 0) →
  (b^2 - m*b + 5 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + q = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + q = 0) →
  q = 36/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l1102_110258


namespace NUMINAMATH_CALUDE_tan_plus_reciprocal_l1102_110251

theorem tan_plus_reciprocal (α : Real) : 
  Real.tan α + (Real.tan α)⁻¹ = (Real.sin α * Real.cos α)⁻¹ :=
by sorry

end NUMINAMATH_CALUDE_tan_plus_reciprocal_l1102_110251


namespace NUMINAMATH_CALUDE_davids_chemistry_marks_l1102_110292

/-- Given David's marks in various subjects and his average, prove his Chemistry marks -/
theorem davids_chemistry_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 95)
  (h3 : physics_marks = 82)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 93)
  (h6 : (english_marks + math_marks + physics_marks + biology_marks + chemistry_marks : ℚ) / 5 = average_marks) :
  chemistry_marks = 97 :=
by
  sorry

end NUMINAMATH_CALUDE_davids_chemistry_marks_l1102_110292


namespace NUMINAMATH_CALUDE_line_intersects_or_tangent_circle_l1102_110227

/-- A line in 2D space defined by the equation (x+1)m + (y-1)n = 0 --/
structure Line where
  m : ℝ
  n : ℝ

/-- A circle in 2D space defined by the equation x^2 + y^2 = 2 --/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 2}

/-- The point (-1, 1) --/
def M : ℝ × ℝ := (-1, 1)

/-- Theorem stating that the line either intersects or is tangent to the circle --/
theorem line_intersects_or_tangent_circle (l : Line) : 
  (∃ p : ℝ × ℝ, p ∈ Circle ∧ (p.1 + 1) * l.m + (p.2 - 1) * l.n = 0) := by
  sorry

#check line_intersects_or_tangent_circle

end NUMINAMATH_CALUDE_line_intersects_or_tangent_circle_l1102_110227


namespace NUMINAMATH_CALUDE_boris_candy_problem_l1102_110208

/-- Given the initial candy count, amount eaten by daughter, number of bowls, 
    and final count in one bowl, calculate how many pieces Boris took from each bowl. -/
theorem boris_candy_problem (initial_candy : ℕ) (daughter_ate : ℕ) (num_bowls : ℕ) (final_bowl_count : ℕ)
  (h1 : initial_candy = 100)
  (h2 : daughter_ate = 8)
  (h3 : num_bowls = 4)
  (h4 : final_bowl_count = 20)
  (h5 : num_bowls > 0) :
  let remaining_candy := initial_candy - daughter_ate
  let candy_per_bowl := remaining_candy / num_bowls
  candy_per_bowl - final_bowl_count = 3 := by sorry

end NUMINAMATH_CALUDE_boris_candy_problem_l1102_110208


namespace NUMINAMATH_CALUDE_cos_triple_angle_l1102_110295

theorem cos_triple_angle (α : ℝ) : Real.cos (3 * α) = 4 * (Real.cos α)^3 - 3 * Real.cos α := by
  sorry

end NUMINAMATH_CALUDE_cos_triple_angle_l1102_110295


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1102_110265

theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 12
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l1102_110265


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l1102_110286

theorem negative_fraction_comparison : -5/6 < -4/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l1102_110286


namespace NUMINAMATH_CALUDE_jerry_money_left_l1102_110211

-- Define the quantities and prices
def mustard_oil_quantity : ℝ := 2
def mustard_oil_price : ℝ := 13
def pasta_quantity : ℝ := 3
def pasta_price : ℝ := 4
def sauce_quantity : ℝ := 1
def sauce_price : ℝ := 5
def initial_money : ℝ := 50

-- Define the total cost of groceries
def total_cost : ℝ :=
  mustard_oil_quantity * mustard_oil_price +
  pasta_quantity * pasta_price +
  sauce_quantity * sauce_price

-- Define the money left after shopping
def money_left : ℝ := initial_money - total_cost

-- Theorem statement
theorem jerry_money_left : money_left = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_money_left_l1102_110211


namespace NUMINAMATH_CALUDE_probability_calculation_l1102_110255

structure ClassStats where
  total_students : ℕ
  female_percentage : ℚ
  brunette_percentage : ℚ
  short_brunette_percentage : ℚ
  club_participation_percentage : ℚ
  short_club_percentage : ℚ

def probability_short_brunette_club (stats : ClassStats) : ℚ :=
  stats.female_percentage *
  stats.brunette_percentage *
  stats.club_participation_percentage *
  stats.short_club_percentage

theorem probability_calculation (stats : ClassStats) 
  (h1 : stats.total_students = 200)
  (h2 : stats.female_percentage = 3/5)
  (h3 : stats.brunette_percentage = 1/2)
  (h4 : stats.short_brunette_percentage = 1/2)
  (h5 : stats.club_participation_percentage = 2/5)
  (h6 : stats.short_club_percentage = 3/4) :
  probability_short_brunette_club stats = 9/100 := by
  sorry

end NUMINAMATH_CALUDE_probability_calculation_l1102_110255


namespace NUMINAMATH_CALUDE_p_excess_over_q_and_r_l1102_110243

theorem p_excess_over_q_and_r (p q r : ℝ) : 
  p = 47.99999999999999 ∧ 
  q = (1/6) * p ∧ 
  r = (1/6) * p →
  p - (q + r) = 32 :=
by sorry

end NUMINAMATH_CALUDE_p_excess_over_q_and_r_l1102_110243


namespace NUMINAMATH_CALUDE_race_heartbeats_l1102_110234

/-- Calculates the total number of heartbeats during a race given the heart rate, race distance, and pace. -/
def total_heartbeats (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Theorem stating that given specific conditions, the total number of heartbeats during a race is 28800. -/
theorem race_heartbeats :
  let heart_rate : ℕ := 160  -- beats per minute
  let race_distance : ℕ := 30  -- miles
  let pace : ℕ := 6  -- minutes per mile
  total_heartbeats heart_rate race_distance pace = 28800 :=
by sorry

end NUMINAMATH_CALUDE_race_heartbeats_l1102_110234


namespace NUMINAMATH_CALUDE_long_division_problem_l1102_110272

theorem long_division_problem :
  let divisor : ℕ := 12
  let quotient : ℕ := 909809
  let dividend : ℕ := divisor * quotient
  dividend = 10917708 := by
sorry

end NUMINAMATH_CALUDE_long_division_problem_l1102_110272


namespace NUMINAMATH_CALUDE_y_order_l1102_110256

/-- The quadratic function f(x) = -2x² + 4 --/
def f (x : ℝ) : ℝ := -2 * x^2 + 4

/-- Point A on the graph of f --/
def A : ℝ × ℝ := (1, f 1)

/-- Point B on the graph of f --/
def B : ℝ × ℝ := (2, f 2)

/-- Point C on the graph of f --/
def C : ℝ × ℝ := (-3, f (-3))

theorem y_order : A.2 > B.2 ∧ B.2 > C.2 := by sorry

end NUMINAMATH_CALUDE_y_order_l1102_110256


namespace NUMINAMATH_CALUDE_marks_weekly_reading_time_l1102_110230

/-- Given Mark's daily reading time and weekly increase, prove his total weekly reading time -/
theorem marks_weekly_reading_time 
  (daily_reading_time : ℕ) 
  (weekly_increase : ℕ) 
  (h1 : daily_reading_time = 2)
  (h2 : weekly_increase = 4) :
  daily_reading_time * 7 + weekly_increase = 18 := by
  sorry

#check marks_weekly_reading_time

end NUMINAMATH_CALUDE_marks_weekly_reading_time_l1102_110230


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1102_110241

/-- Given that x and y are inversely proportional, prove that when x = -12, y = -56.25 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 60) (h3 : x = 3 * y) :
  x = -12 → y = -56.25 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1102_110241


namespace NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l1102_110253

/-- The sum of the factorials of the first n positive integers -/
def sum_of_factorials (n : ℕ) : ℕ :=
  (List.range n).map Nat.factorial |>.sum

/-- The last two digits of a natural number -/
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

theorem last_two_digits_sum_factorials_15 :
  last_two_digits (sum_of_factorials 15) = 13 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_sum_factorials_15_l1102_110253


namespace NUMINAMATH_CALUDE_sandy_savings_l1102_110296

theorem sandy_savings (last_year_salary : ℝ) (last_year_savings_rate : ℝ) 
  (h1 : last_year_savings_rate > 0)
  (h2 : last_year_savings_rate < 1)
  (h3 : (1.1 * last_year_salary) * 0.09 = 1.65 * (last_year_salary * last_year_savings_rate)) :
  last_year_savings_rate = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sandy_savings_l1102_110296


namespace NUMINAMATH_CALUDE_loss_percentage_calculation_l1102_110232

theorem loss_percentage_calculation (purchase_price selling_price : ℚ) : 
  purchase_price = 490 → 
  selling_price = 465.5 → 
  (purchase_price - selling_price) / purchase_price * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_calculation_l1102_110232


namespace NUMINAMATH_CALUDE_trapezoid_area_maximization_l1102_110284

/-- Given a triangle ABC with sides a, b, c, altitude h, and a point G on the altitude
    at distance x from A, the area of the trapezoid formed by drawing a line parallel
    to the base through G and extending the sides is maximized when
    x = ((b + c) * h) / (2 * (a + b + c)). -/
theorem trapezoid_area_maximization (a b c h x : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧ x > 0 ∧ x < h →
  let t := (1/2) * (a + ((a + b + c) * x / h)) * (h - x)
  ∃ (max_x : ℝ), max_x = ((b + c) * h) / (2 * (a + b + c)) ∧
    ∀ y, 0 < y ∧ y < h → t ≤ (1/2) * (a + ((a + b + c) * max_x / h)) * (h - max_x) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_maximization_l1102_110284


namespace NUMINAMATH_CALUDE_gcf_60_90_l1102_110281

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_90_l1102_110281


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l1102_110228

theorem negative_two_less_than_negative_three_halves : -2 < -3/2 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_three_halves_l1102_110228


namespace NUMINAMATH_CALUDE_round_05019_to_thousandth_l1102_110220

/-- Custom rounding function that rounds to the nearest thousandth as described in the problem -/
def roundToThousandth (x : ℚ) : ℚ :=
  (⌊x * 1000⌋ : ℚ) / 1000

/-- Theorem stating that rounding 0.05019 to the nearest thousandth results in 0.050 -/
theorem round_05019_to_thousandth :
  roundToThousandth (5019 / 100000) = 50 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_round_05019_to_thousandth_l1102_110220


namespace NUMINAMATH_CALUDE_irrigation_system_fluxes_l1102_110216

-- Define the irrigation system
structure IrrigationSystem where
  channels : Set Char
  nodes : Set Char
  flux : Char → Char → ℝ
  water_entry : Char
  water_exit : Char

-- Define the properties of the irrigation system
def is_valid_system (sys : IrrigationSystem) : Prop :=
  -- Channels and nodes
  sys.channels = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'} ∧
  sys.nodes = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'} ∧
  -- Water entry and exit points
  sys.water_entry = 'A' ∧
  sys.water_exit = 'E' ∧
  -- Flux conservation property
  ∀ x y z : Char, x ∈ sys.nodes ∧ y ∈ sys.nodes ∧ z ∈ sys.nodes →
    sys.flux x y + sys.flux y z = sys.flux x z

-- Theorem statement
theorem irrigation_system_fluxes (sys : IrrigationSystem) 
  (h_valid : is_valid_system sys) 
  (h_flux_BC : sys.flux 'B' 'C' = q₀) :
  sys.flux 'A' 'B' = 2 * q₀ ∧
  sys.flux 'A' 'H' = 3/2 * q₀ ∧
  sys.flux 'A' 'B' + sys.flux 'A' 'H' = 7/2 * q₀ :=
by sorry

end NUMINAMATH_CALUDE_irrigation_system_fluxes_l1102_110216


namespace NUMINAMATH_CALUDE_combined_machine_time_order_completion_time_l1102_110219

theorem combined_machine_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  1 / (1 / t1 + 1 / t2) = (t1 * t2) / (t1 + t2) := by sorry

theorem order_completion_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) :
  t1 = 20 → t2 = 30 → 1 / (1 / t1 + 1 / t2) = 12 := by sorry

end NUMINAMATH_CALUDE_combined_machine_time_order_completion_time_l1102_110219


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1102_110217

theorem inequality_and_equality_condition (x y : ℝ) :
  (x^2 + 1) * (y^2 + 1) + 4 * (x - 1) * (y - 1) ≥ 0 ∧
  ((x^2 + 1) * (y^2 + 1) + 4 * (x - 1) * (y - 1) = 0 ↔
    ((x = 1 - Real.sqrt 2 ∧ y = 1 + Real.sqrt 2) ∨
     (x = 1 + Real.sqrt 2 ∧ y = 1 - Real.sqrt 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1102_110217


namespace NUMINAMATH_CALUDE_student_sister_weight_l1102_110268

/-- The combined weight of a student and his sister, given specific conditions --/
theorem student_sister_weight (student_weight sister_weight : ℝ) : 
  student_weight = 79 →
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 116 := by
sorry

end NUMINAMATH_CALUDE_student_sister_weight_l1102_110268


namespace NUMINAMATH_CALUDE_prob_sum_greater_than_9_l1102_110288

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of favorable outcomes (sum > 9) -/
def favorableOutcomes : ℕ := 6

/-- The probability of rolling a sum greater than 9 with two dice -/
def probSumGreaterThan9 : ℚ := favorableOutcomes / totalOutcomes

theorem prob_sum_greater_than_9 : probSumGreaterThan9 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_greater_than_9_l1102_110288


namespace NUMINAMATH_CALUDE_four_number_inequality_equality_condition_l1102_110273

theorem four_number_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a - b) * (a - c)) / (a + b + c) +
  ((b - c) * (b - d)) / (b + c + d) +
  ((c - d) * (c - a)) / (c + d + a) +
  ((d - a) * (d - b)) / (d + a + b) ≥ 0 :=
by sorry

theorem equality_condition (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ((a - b) * (a - c)) / (a + b + c) +
  ((b - c) * (b - d)) / (b + c + d) +
  ((c - d) * (c - a)) / (c + d + a) +
  ((d - a) * (d - b)) / (d + a + b) = 0 ↔
  a = b ∧ b = c ∧ c = d :=
by sorry

end NUMINAMATH_CALUDE_four_number_inequality_equality_condition_l1102_110273


namespace NUMINAMATH_CALUDE_two_color_theorem_l1102_110266

/-- A line in a plane --/
structure Line where
  -- We don't need to define the specifics of a line for this problem

/-- A region in a plane formed by intersecting lines --/
structure Region where
  -- We don't need to define the specifics of a region for this problem

/-- A color used for coloring regions --/
inductive Color
  | Red
  | Blue

/-- A configuration of lines in a plane --/
def Configuration := List Line

/-- A coloring of regions --/
def Coloring := Region → Color

/-- Check if two regions are adjacent --/
def adjacent (r1 r2 : Region) : Prop :=
  sorry -- Definition of adjacency

/-- A valid coloring ensures no adjacent regions have the same color --/
def valid_coloring (c : Configuration) (coloring : Coloring) : Prop :=
  ∀ r1 r2 : Region, adjacent r1 r2 → coloring r1 ≠ coloring r2

/-- The main theorem: for any configuration of lines, there exists a valid coloring --/
theorem two_color_theorem (c : Configuration) : 
  ∃ coloring : Coloring, valid_coloring c coloring :=
sorry

end NUMINAMATH_CALUDE_two_color_theorem_l1102_110266


namespace NUMINAMATH_CALUDE_sam_filled_four_bags_saturday_l1102_110203

/-- The number of bags Sam filled on Saturday -/
def saturday_bags : ℕ := sorry

/-- The number of bags Sam filled on Sunday -/
def sunday_bags : ℕ := 3

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 6

/-- The total number of cans collected -/
def total_cans : ℕ := 42

/-- Theorem stating that Sam filled 4 bags on Saturday -/
theorem sam_filled_four_bags_saturday : saturday_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_sam_filled_four_bags_saturday_l1102_110203


namespace NUMINAMATH_CALUDE_simplify_expression_l1102_110225

theorem simplify_expression (x : ℝ) : 3*x^2 + 4 - 5*x^3 - x^3 + 3 - 3*x^2 = -6*x^3 + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1102_110225


namespace NUMINAMATH_CALUDE_bookstore_editions_l1102_110278

-- Define the universe of books in the bookstore
variable (Book : Type)

-- Define a predicate for new editions
variable (is_new_edition : Book → Prop)

-- Theorem statement
theorem bookstore_editions (h : ¬∀ (b : Book), is_new_edition b) :
  (∃ (b : Book), ¬is_new_edition b) ∧ (¬∀ (b : Book), is_new_edition b) := by
  sorry

end NUMINAMATH_CALUDE_bookstore_editions_l1102_110278


namespace NUMINAMATH_CALUDE_same_price_at_12_sheets_l1102_110202

/-- The price per sheet for John's Photo World -/
def johns_price_per_sheet : ℚ := 275/100

/-- The sitting fee for John's Photo World -/
def johns_sitting_fee : ℚ := 125

/-- The price per sheet for Sam's Picture Emporium -/
def sams_price_per_sheet : ℚ := 150/100

/-- The sitting fee for Sam's Picture Emporium -/
def sams_sitting_fee : ℚ := 140

/-- The total cost for John's Photo World given a number of sheets -/
def johns_total_cost (sheets : ℚ) : ℚ := johns_price_per_sheet * sheets + johns_sitting_fee

/-- The total cost for Sam's Picture Emporium given a number of sheets -/
def sams_total_cost (sheets : ℚ) : ℚ := sams_price_per_sheet * sheets + sams_sitting_fee

theorem same_price_at_12_sheets :
  ∃ (sheets : ℚ), sheets = 12 ∧ johns_total_cost sheets = sams_total_cost sheets :=
by sorry

end NUMINAMATH_CALUDE_same_price_at_12_sheets_l1102_110202


namespace NUMINAMATH_CALUDE_log_equation_sum_l1102_110207

theorem log_equation_sum (a b : ℤ) (h : a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3) : 
  a + 2 * b = 21 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l1102_110207


namespace NUMINAMATH_CALUDE_no_prime_roots_sum_64_l1102_110285

theorem no_prime_roots_sum_64 : ¬∃ (p q k : ℕ), 
  Prime p ∧ Prime q ∧ 
  p * q = k ∧
  p + q = 64 ∧
  p^2 - 64*p + k = 0 ∧
  q^2 - 64*q + k = 0 :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_sum_64_l1102_110285


namespace NUMINAMATH_CALUDE_alicia_tax_deduction_l1102_110294

/-- Represents Alicia's hourly wage in dollars -/
def hourly_wage : ℚ := 25

/-- Represents the local tax rate as a decimal -/
def tax_rate : ℚ := 18 / 1000

/-- Converts dollars to cents -/
def dollars_to_cents (dollars : ℚ) : ℚ := dollars * 100

/-- Calculates the tax deduction in cents -/
def tax_deduction (wage : ℚ) (rate : ℚ) : ℚ :=
  dollars_to_cents (wage * rate)

/-- Theorem stating that Alicia's tax deduction is 45 cents per hour -/
theorem alicia_tax_deduction :
  tax_deduction hourly_wage tax_rate = 45 := by
  sorry

end NUMINAMATH_CALUDE_alicia_tax_deduction_l1102_110294


namespace NUMINAMATH_CALUDE_inequality_solution_l1102_110279

theorem inequality_solution (x : ℝ) : 2 - 1 / (3 * x + 4) < 5 ↔ x > -4/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1102_110279


namespace NUMINAMATH_CALUDE_nested_fourth_root_l1102_110276

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M^(1/4) * M^(1/16)))^(1/4) = M^(21/64) :=
sorry

end NUMINAMATH_CALUDE_nested_fourth_root_l1102_110276


namespace NUMINAMATH_CALUDE_ratio_of_numbers_with_sum_gcd_equal_lcm_l1102_110280

theorem ratio_of_numbers_with_sum_gcd_equal_lcm (A B : ℕ) (h1 : A ≥ B) :
  A + B + Nat.gcd A B = Nat.lcm A B → (A : ℚ) / B = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_with_sum_gcd_equal_lcm_l1102_110280


namespace NUMINAMATH_CALUDE_intersection_A_B_l1102_110283

def A : Set ℝ := {x | ∃ k : ℤ, x = 2 * k + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1102_110283


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l1102_110213

/-- The distance traveled by a boat downstream -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: A boat with speed 13 km/hr in still water, traveling downstream
    in a stream with speed 4 km/hr for 4 hours, covers a distance of 68 km -/
theorem boat_downstream_distance :
  distance_downstream 13 4 4 = 68 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l1102_110213


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1102_110299

theorem walking_speed_problem (x : ℝ) :
  let james_speed := x^2 - 13*x - 30
  let jane_distance := x^2 - 5*x - 66
  let jane_time := x + 6
  let jane_speed := jane_distance / jane_time
  james_speed = jane_speed → james_speed = -4 + 2 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_problem_l1102_110299


namespace NUMINAMATH_CALUDE_car_stop_once_probability_l1102_110210

/-- The probability of a car stopping once at three traffic lights. -/
theorem car_stop_once_probability
  (pA pB pC : ℝ)
  (hA : pA = 1/3)
  (hB : pB = 1/2)
  (hC : pC = 2/3)
  : (1 - pA) * pB * pC + pA * (1 - pB) * pC + pA * pB * (1 - pC) = 7/18 := by
  sorry

end NUMINAMATH_CALUDE_car_stop_once_probability_l1102_110210


namespace NUMINAMATH_CALUDE_op_twice_equals_identity_l1102_110239

-- Define the operation ⊕
def op (x y : ℝ) : ℝ := x^3 - y

-- Statement to prove
theorem op_twice_equals_identity (h : ℝ) : op h (op h h) = h := by
  sorry

end NUMINAMATH_CALUDE_op_twice_equals_identity_l1102_110239


namespace NUMINAMATH_CALUDE_gcd_45345_34534_l1102_110246

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end NUMINAMATH_CALUDE_gcd_45345_34534_l1102_110246


namespace NUMINAMATH_CALUDE_beach_volleyball_max_players_l1102_110275

theorem beach_volleyball_max_players : ∃ (n : ℕ), n > 0 ∧ n ≤ 13 ∧ 
  (∀ (m : ℕ), m > 13 → ¬(
    (∃ (games : Finset (Finset ℕ)), 
      games.card = m ∧ 
      (∀ g ∈ games, g.card = 4) ∧
      (∀ i j, i < m ∧ j < m ∧ i ≠ j → ∃ g ∈ games, i ∈ g ∧ j ∈ g)
    )
  )) := by sorry

end NUMINAMATH_CALUDE_beach_volleyball_max_players_l1102_110275


namespace NUMINAMATH_CALUDE_failed_students_l1102_110233

theorem failed_students (total : ℕ) (passed_percentage : ℚ) 
  (h1 : total = 804)
  (h2 : passed_percentage = 75 / 100) :
  ↑total * (1 - passed_percentage) = 201 := by
  sorry

end NUMINAMATH_CALUDE_failed_students_l1102_110233


namespace NUMINAMATH_CALUDE_fence_perimeter_is_236_l1102_110236

/-- Represents the configuration of a rectangular fence --/
structure FenceConfig where
  total_posts : ℕ
  post_width : ℚ
  gap_width : ℕ
  long_short_ratio : ℕ

/-- Calculates the perimeter of the fence given its configuration --/
def calculate_perimeter (config : FenceConfig) : ℚ :=
  let short_side_posts := config.total_posts / (config.long_short_ratio + 1)
  let long_side_posts := short_side_posts * config.long_short_ratio
  let short_side_length := short_side_posts * config.post_width + (short_side_posts - 1) * config.gap_width
  let long_side_length := long_side_posts * config.post_width + (long_side_posts - 1) * config.gap_width
  2 * (short_side_length + long_side_length)

/-- The main theorem stating that the fence configuration results in a perimeter of 236 feet --/
theorem fence_perimeter_is_236 :
  let config : FenceConfig := {
    total_posts := 36,
    post_width := 1/2,  -- 6 inches = 1/2 foot
    gap_width := 6,
    long_short_ratio := 3
  }
  calculate_perimeter config = 236 := by sorry


end NUMINAMATH_CALUDE_fence_perimeter_is_236_l1102_110236


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1102_110205

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 7*x + 10 < 0 ↔ 2 < x ∧ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1102_110205
