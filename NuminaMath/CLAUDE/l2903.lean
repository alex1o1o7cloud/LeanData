import Mathlib

namespace consecutive_integers_right_triangle_l2903_290377

theorem consecutive_integers_right_triangle (m n : ℕ) (h : n^2 = 2*m + 1) :
  n^2 + m^2 = (m + 1)^2 := by
sorry

end consecutive_integers_right_triangle_l2903_290377


namespace unreachable_from_2_2_2_reachable_from_3_3_3_l2903_290340

/-- The operation that replaces one number with the difference between the sum of the other two and 1 -/
def operation (x y z : ℤ) : ℤ × ℤ × ℤ → Prop :=
  fun w => (w = (y + z - 1, y, z)) ∨ (w = (x, x + z - 1, z)) ∨ (w = (x, y, x + y - 1))

/-- The relation that represents the repeated application of the operation -/
inductive reachable : ℤ × ℤ × ℤ → ℤ × ℤ × ℤ → Prop
  | refl {x} : reachable x x
  | step {x y z} (h : reachable x y) (o : operation y.1 y.2.1 y.2.2 z) : reachable x z

theorem unreachable_from_2_2_2 :
  ¬ reachable (2, 2, 2) (17, 1999, 2105) :=
sorry

theorem reachable_from_3_3_3 :
  reachable (3, 3, 3) (17, 1999, 2105) :=
sorry

end unreachable_from_2_2_2_reachable_from_3_3_3_l2903_290340


namespace min_side_in_triangle_l2903_290381

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if c = 2b and the area of the triangle is 1, then the minimum value of a is √3.
-/
theorem min_side_in_triangle (a b c : ℝ) (A B C : ℝ) :
  c = 2 * b →
  (1 / 2) * b * c * Real.sin A = 1 →
  ∃ (a_min : ℝ), a_min = Real.sqrt 3 ∧ ∀ a', a' ≥ a_min := by
  sorry

end min_side_in_triangle_l2903_290381


namespace smaller_octagon_area_ratio_l2903_290375

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The smaller octagon formed by connecting midpoints of sides of a regular octagon -/
def smallerOctagon (oct : RegularOctagon) : RegularOctagon := sorry

/-- The area of a regular octagon -/
def area (oct : RegularOctagon) : ℝ := sorry

/-- Theorem: The area of the smaller octagon is half the area of the larger octagon -/
theorem smaller_octagon_area_ratio (oct : RegularOctagon) : 
  area (smallerOctagon oct) = (1/2 : ℝ) * area oct := by sorry

end smaller_octagon_area_ratio_l2903_290375


namespace shaded_fraction_is_one_eighth_l2903_290352

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the shaded area within a rectangle -/
structure ShadedRectangle where
  rectangle : Rectangle
  shaded_area : ℝ

theorem shaded_fraction_is_one_eighth 
  (r : Rectangle)
  (sr : ShadedRectangle)
  (h1 : r.length = 15)
  (h2 : r.width = 24)
  (h3 : sr.rectangle = r)
  (h4 : sr.shaded_area = (1 / 4) * (1 / 2) * r.area) :
  sr.shaded_area / r.area = 1 / 8 := by
  sorry

end shaded_fraction_is_one_eighth_l2903_290352


namespace divisible_by_twenty_l2903_290309

theorem divisible_by_twenty (n : ℕ) : ∃ k : ℤ, 9^(8*n+4) - 7^(8*n+4) = 20*k := by
  sorry

end divisible_by_twenty_l2903_290309


namespace sum_inequality_l2903_290365

theorem sum_inequality (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a + c > b + d := by
  sorry

end sum_inequality_l2903_290365


namespace certain_number_problem_l2903_290369

theorem certain_number_problem : ∃ x : ℝ, x * 12 = 0.60 * 900 ∧ x = 45 := by sorry

end certain_number_problem_l2903_290369


namespace second_number_problem_l2903_290322

theorem second_number_problem (a b c : ℚ) : 
  a + b + c = 264 →
  a = 2 * b →
  c = (1 / 3) * a →
  b = 72 := by sorry

end second_number_problem_l2903_290322


namespace point_distance_theorem_l2903_290378

/-- Given a point P with coordinates (x^2 - k, -5), where k is a positive constant,
    if the distance from P to the x-axis is half the distance from P to the y-axis,
    and the total distance from P to both axes is 15 units,
    then k = x^2 - 10. -/
theorem point_distance_theorem (x k : ℝ) (h1 : k > 0) :
  let P : ℝ × ℝ := (x^2 - k, -5)
  abs P.2 = (1/2) * abs P.1 →
  abs P.2 + abs P.1 = 15 →
  k = x^2 - 10 := by
sorry

end point_distance_theorem_l2903_290378


namespace brown_mms_problem_l2903_290348

theorem brown_mms_problem (bag1 bag2 bag3 bag4 bag5 : ℕ) 
  (h1 : bag1 = 9)
  (h2 : bag2 = 12)
  (h5 : bag5 = 3)
  (h_avg : (bag1 + bag2 + bag3 + bag4 + bag5) / 5 = 8) :
  bag3 + bag4 = 16 := by
  sorry

end brown_mms_problem_l2903_290348


namespace gcf_seven_eight_factorial_l2903_290397

theorem gcf_seven_eight_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcf_seven_eight_factorial_l2903_290397


namespace average_playing_time_l2903_290330

/-- The average playing time for children playing table tennis -/
theorem average_playing_time
  (num_children : ℕ)
  (total_time : ℝ)
  (h_num_children : num_children = 5)
  (h_total_time : total_time = 15)
  : total_time / num_children = 3 := by
  sorry

end average_playing_time_l2903_290330


namespace cauchy_functional_equation_l2903_290386

theorem cauchy_functional_equation 
  (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + y) = f x + f y) : 
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x :=
by sorry

end cauchy_functional_equation_l2903_290386


namespace product_divisible_by_14_l2903_290337

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7*a + 8*b = 14*c + 28*d) : 
  14 ∣ (a * b) := by
sorry

end product_divisible_by_14_l2903_290337


namespace difference_in_sums_l2903_290342

def star_list : List Nat := List.range 50 |>.map (· + 1)

def replace_three_with_two (n : Nat) : Nat :=
  let s := toString n
  (s.replace "3" "2").toNat!

def emilio_list : List Nat :=
  star_list.map replace_three_with_two

theorem difference_in_sums : 
  star_list.sum - emilio_list.sum = 105 := by
  sorry

end difference_in_sums_l2903_290342


namespace forecast_variation_determinants_l2903_290301

/-- Represents a variable in regression analysis -/
inductive RegressionVariable
  | Forecast
  | Explanatory
  | Residual

/-- Represents the components that determine the variation of a variable -/
structure VariationDeterminants where
  components : List RegressionVariable

/-- Axiom: In regression analysis, the variation of the forecast variable
    is determined by both explanatory and residual variables -/
axiom regression_variation_determinants :
  VariationDeterminants.components (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]) =
  (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]).components

/-- Theorem: The variation of the forecast variable in regression analysis
    is determined by both explanatory and residual variables -/
theorem forecast_variation_determinants :
  VariationDeterminants.components (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]) =
  (VariationDeterminants.mk [RegressionVariable.Explanatory, RegressionVariable.Residual]).components :=
by sorry

end forecast_variation_determinants_l2903_290301


namespace alcohol_concentration_proof_l2903_290308

/-- Proves that adding 7.5 litres of pure alcohol to a 10 litre solution
    that is 30% alcohol results in a 60% alcohol solution -/
theorem alcohol_concentration_proof :
  let initial_volume : ℝ := 10
  let initial_concentration : ℝ := 0.30
  let added_alcohol : ℝ := 7.5
  let final_concentration : ℝ := 0.60
  let final_volume : ℝ := initial_volume + added_alcohol
  let final_alcohol : ℝ := initial_volume * initial_concentration + added_alcohol
  final_alcohol / final_volume = final_concentration :=
by sorry


end alcohol_concentration_proof_l2903_290308


namespace fraction_equality_l2903_290333

theorem fraction_equality (m n : ℚ) (h : m / n = 3 / 4) : 
  (m + n) / n = 7 / 4 := by
  sorry

end fraction_equality_l2903_290333


namespace quadratic_roots_l2903_290347

theorem quadratic_roots (a : ℝ) : 
  (2 : ℝ)^2 + 2 - a = 0 → (-3 : ℝ)^2 + (-3) - a = 0 := by
  sorry

end quadratic_roots_l2903_290347


namespace quadratic_equation_rational_solutions_l2903_290344

theorem quadratic_equation_rational_solutions :
  ∃! (c₁ c₂ : ℕ+), 
    (∃ (x : ℚ), 7 * x^2 + 13 * x + c₁.val = 0) ∧
    (∃ (x : ℚ), 7 * x^2 + 13 * x + c₂.val = 0) ∧
    c₁ = c₂ ∧ c₁ = 6 := by
  sorry

end quadratic_equation_rational_solutions_l2903_290344


namespace tea_preparation_time_l2903_290334

/-- Represents the time required for each task in minutes -/
structure TaskTimes where
  washKettle : ℕ
  boilWater : ℕ
  washTeapot : ℕ
  washTeacups : ℕ
  getTeaLeaves : ℕ

/-- Calculates the minimum time required to complete all tasks -/
def minTimeRequired (times : TaskTimes) : ℕ :=
  max times.washKettle (times.boilWater + times.washKettle)

/-- Theorem stating that the minimum time required is 16 minutes -/
theorem tea_preparation_time (times : TaskTimes) 
  (h1 : times.washKettle = 1)
  (h2 : times.boilWater = 15)
  (h3 : times.washTeapot = 1)
  (h4 : times.washTeacups = 1)
  (h5 : times.getTeaLeaves = 2) :
  minTimeRequired times = 16 := by
  sorry


end tea_preparation_time_l2903_290334


namespace university_weighted_average_age_l2903_290305

/-- Calculates the weighted average age of a university given the number of arts and technical classes,
    their respective average ages, and assuming each class has the same number of students. -/
theorem university_weighted_average_age
  (num_arts_classes : ℕ)
  (num_tech_classes : ℕ)
  (avg_age_arts : ℝ)
  (avg_age_tech : ℝ)
  (h1 : num_arts_classes = 8)
  (h2 : num_tech_classes = 5)
  (h3 : avg_age_arts = 21)
  (h4 : avg_age_tech = 18) :
  (num_arts_classes * avg_age_arts + num_tech_classes * avg_age_tech) / (num_arts_classes + num_tech_classes) = 258 / 13 := by
sorry

end university_weighted_average_age_l2903_290305


namespace solve_kitchen_supplies_l2903_290387

def kitchen_supplies_problem (angela_pots : ℕ) (angela_plates : ℕ) (angela_cutlery : ℕ) 
  (sharon_total : ℕ) : Prop :=
  angela_pots = 20 ∧
  angela_plates > 3 * angela_pots ∧
  angela_cutlery = angela_plates / 2 ∧
  sharon_total = 254 ∧
  sharon_total = angela_pots / 2 + (3 * angela_plates - 20) + 2 * angela_cutlery ∧
  angela_plates - 3 * angela_pots = 6

theorem solve_kitchen_supplies : 
  ∃ (angela_pots angela_plates angela_cutlery : ℕ),
    kitchen_supplies_problem angela_pots angela_plates angela_cutlery 254 :=
sorry

end solve_kitchen_supplies_l2903_290387


namespace resulting_polygon_sides_bound_resulting_polygon_sides_bound_even_l2903_290372

/-- Represents a convex n-gon with all diagonals drawn --/
structure ConvexNGonWithDiagonals (n : ℕ) where
  -- Add necessary fields here

/-- Represents a polygon resulting from the division of the n-gon by its diagonals --/
structure ResultingPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of sides of a resulting polygon --/
def num_sides (p : ResultingPolygon n) : ℕ := sorry

theorem resulting_polygon_sides_bound (n : ℕ) (ngon : ConvexNGonWithDiagonals n) 
  (p : ResultingPolygon n) : num_sides p ≤ n := by sorry

theorem resulting_polygon_sides_bound_even (n : ℕ) (ngon : ConvexNGonWithDiagonals n) 
  (p : ResultingPolygon n) (h : Even n) : num_sides p ≤ n - 1 := by sorry

end resulting_polygon_sides_bound_resulting_polygon_sides_bound_even_l2903_290372


namespace rationalize_denominator_sqrt3_minus1_l2903_290303

theorem rationalize_denominator_sqrt3_minus1 : 
  (1 : ℝ) / (Real.sqrt 3 - 1) = (Real.sqrt 3 + 1) / 2 := by sorry

end rationalize_denominator_sqrt3_minus1_l2903_290303


namespace simplify_expression_l2903_290338

theorem simplify_expression (a b : ℝ) : 
  (15*a + 45*b) + (20*a + 35*b) - (25*a + 55*b) + (30*a - 5*b) = 40*a + 20*b := by
  sorry

end simplify_expression_l2903_290338


namespace sally_pokemon_cards_l2903_290329

theorem sally_pokemon_cards (initial : ℕ) (new : ℕ) (lost : ℕ) : 
  initial = 27 → new = 41 → lost = 20 → initial + new - lost = 48 := by
  sorry

end sally_pokemon_cards_l2903_290329


namespace ellipse_conditions_l2903_290331

-- Define what it means for an equation to represent an ellipse
def represents_ellipse (a b : ℝ) : Prop :=
  ∃ (h k : ℝ) (A B : ℝ), A ≠ B ∧ A > 0 ∧ B > 0 ∧
    ∀ (x y : ℝ), a * (x - h)^2 + b * (y - k)^2 = 1 ↔ 
      ((x - h)^2 / A^2) + ((y - k)^2 / B^2) = 1

-- State the theorem
theorem ellipse_conditions (a b : ℝ) :
  (a > 0 ∧ b > 0 ∧ represents_ellipse a b) ∧
  ¬(a > 0 ∧ b > 0 → represents_ellipse a b) := by
  sorry


end ellipse_conditions_l2903_290331


namespace linear_equation_solution_l2903_290321

theorem linear_equation_solution (a b : ℝ) : 
  (2 * a + (-1) * b = -1) → (1 + 2 * a - b = 0) := by sorry

end linear_equation_solution_l2903_290321


namespace min_jellybeans_correct_l2903_290326

/-- The smallest number of jellybeans Alex should buy -/
def min_jellybeans : ℕ := 134

/-- Theorem stating that min_jellybeans is the smallest number satisfying the conditions -/
theorem min_jellybeans_correct :
  (min_jellybeans ≥ 120) ∧
  (min_jellybeans % 15 = 14) ∧
  (∀ n : ℕ, n ≥ 120 → n % 15 = 14 → n ≥ min_jellybeans) :=
by sorry

end min_jellybeans_correct_l2903_290326


namespace equal_variance_sequence_properties_l2903_290328

/-- Definition of an equal variance sequence -/
def is_equal_variance_sequence (a : ℕ+ → ℝ) (p : ℝ) :=
  ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

theorem equal_variance_sequence_properties
  (a : ℕ+ → ℝ) (p : ℝ) (h : is_equal_variance_sequence a p) :
  (∀ n : ℕ+, ∃ d : ℝ, a (n + 1) ^ 2 - a n ^ 2 = d) ∧
  is_equal_variance_sequence (fun n ↦ (-1) ^ (n : ℕ)) 0 ∧
  (∀ k : ℕ+, is_equal_variance_sequence (fun n ↦ a (k * n)) (k * p)) :=
by sorry

end equal_variance_sequence_properties_l2903_290328


namespace train_crossing_time_l2903_290368

/-- Given two trains of equal length, prove the time taken by one train to cross a telegraph post. -/
theorem train_crossing_time (train_length : ℝ) (time_second_train : ℝ) (time_crossing_each_other : ℝ) :
  train_length = 120 →
  time_second_train = 15 →
  time_crossing_each_other = 12 →
  ∃ (time_first_train : ℝ),
    time_first_train = 10 ∧
    train_length / time_first_train + train_length / time_second_train =
      2 * train_length / time_crossing_each_other :=
by sorry

end train_crossing_time_l2903_290368


namespace water_remaining_l2903_290354

/-- Given 3 gallons of water and using 5/4 gallons in an experiment, 
    prove that the remaining amount is 7/4 gallons. -/
theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end water_remaining_l2903_290354


namespace square_formation_proof_l2903_290373

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, n = m * m

def piece_sizes : List Nat := [4, 5, 6, 7, 8]

def total_squares : Nat := piece_sizes.sum

theorem square_formation_proof :
  ∃ (removed_piece : Nat),
    removed_piece ∈ piece_sizes ∧
    is_perfect_square (total_squares - removed_piece) ∧
    removed_piece = 5 :=
  sorry

end square_formation_proof_l2903_290373


namespace train_crossing_time_l2903_290343

/-- A train crosses a platform in a certain time -/
theorem train_crossing_time 
  (train_speed : ℝ) 
  (pole_crossing_time : ℝ) 
  (platform_crossing_time : ℝ) : 
  train_speed = 36 → 
  pole_crossing_time = 12 → 
  platform_crossing_time = 49.996960243180546 := by
  sorry

end train_crossing_time_l2903_290343


namespace solve_employee_pay_l2903_290357

def employee_pay_problem (pay_B : ℝ) (percent_A : ℝ) : Prop :=
  let pay_A : ℝ := percent_A * pay_B
  let total_pay : ℝ := pay_A + pay_B
  pay_B = 228 ∧ percent_A = 1.5 → total_pay = 570

theorem solve_employee_pay : employee_pay_problem 228 1.5 := by
  sorry

end solve_employee_pay_l2903_290357


namespace football_season_length_l2903_290358

/-- The number of months in a football season -/
def season_length (total_games : ℕ) (games_per_month : ℕ) : ℕ :=
  total_games / games_per_month

/-- Proof that the football season lasts 17 months -/
theorem football_season_length : season_length 323 19 = 17 := by
  sorry

end football_season_length_l2903_290358


namespace f_properties_l2903_290367

noncomputable def f (x : ℝ) := Real.cos (2 * x) + 2 * Real.sin x * Real.sin x

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x, f (x + p) = f x ∧ ∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  (∃ M : ℝ, ∀ x, f x ≤ M ∧ ∃ x, f x = M) ∧
  (∀ k : ℤ, f (k * Real.pi) = 2) ∧
  (∀ A : ℝ, A > 0 ∧ A < Real.pi / 2 →
    f A = 0 →
    ∀ b a : ℝ, b = 5 ∧ a = 7 →
    ∃ c : ℝ, c > 0 ∧
    (1/2) * b * c * Real.sin A = 10) :=
sorry

end f_properties_l2903_290367


namespace wilsons_theorem_l2903_290316

theorem wilsons_theorem (p : ℕ) (h : p ≥ 2) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end wilsons_theorem_l2903_290316


namespace smallest_k_sum_digits_l2903_290374

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem stating that 9999 is the smallest positive integer k satisfying the condition -/
theorem smallest_k_sum_digits : 
  (∀ m : ℕ, m ∈ Finset.range 2014 → s ((m + 1) * 9999) = s 9999) ∧ 
  (∀ k : ℕ, k < 9999 → ∃ m : ℕ, m ∈ Finset.range 2014 ∧ s ((m + 1) * k) ≠ s k) :=
sorry

end smallest_k_sum_digits_l2903_290374


namespace print_shop_price_X_l2903_290318

/-- The price per color copy at print shop Y -/
def price_Y : ℝ := 1.70

/-- The number of copies in the comparison -/
def num_copies : ℕ := 70

/-- The price difference between shops Y and X for 70 copies -/
def price_difference : ℝ := 35

/-- The price per color copy at print shop X -/
def price_X : ℝ := 1.20

theorem print_shop_price_X :
  price_X = (price_Y * num_copies - price_difference) / num_copies :=
by sorry

end print_shop_price_X_l2903_290318


namespace barney_situp_time_l2903_290312

/-- The number of sit-ups Barney can perform in one minute -/
def barney_situps_per_minute : ℕ := 45

/-- The number of minutes Barney does sit-ups -/
def barney_minutes : ℕ := 1

/-- The number of minutes Carrie does sit-ups -/
def carrie_minutes : ℕ := 2

/-- The number of minutes Jerrie does sit-ups -/
def jerrie_minutes : ℕ := 3

/-- The total number of sit-ups performed by all three -/
def total_situps : ℕ := 510

/-- Theorem stating that given the conditions, Barney did sit-ups for 1 minute -/
theorem barney_situp_time : 
  barney_situps_per_minute * barney_minutes + 
  (2 * barney_situps_per_minute) * carrie_minutes + 
  (2 * barney_situps_per_minute + 5) * jerrie_minutes = total_situps :=
by sorry

end barney_situp_time_l2903_290312


namespace kevins_record_is_72_l2903_290371

/-- Calculates the number of wings in Kevin's hot wing eating record --/
def kevins_record (duration : ℕ) (alans_rate : ℕ) (additional_wings_needed : ℕ) : ℕ :=
  duration * (alans_rate + additional_wings_needed)

theorem kevins_record_is_72 :
  kevins_record 8 5 4 = 72 := by
  sorry

end kevins_record_is_72_l2903_290371


namespace inequality_and_minimum_l2903_290327

theorem inequality_and_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 1/y + 1/(x*y) ≥ 8) := by
  sorry

end inequality_and_minimum_l2903_290327


namespace two_solutions_l2903_290388

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line 2x - 3y + 5 = 0 -/
def onLine (p : Point) : Prop :=
  2 * p.x - 3 * p.y + 5 = 0

/-- The distance between two points is √13 -/
def hasDistance13 (p : Point) : Prop :=
  (p.x - 2)^2 + (p.y - 3)^2 = 13

/-- The two solutions -/
def solution1 : Point := ⟨-1, 1⟩
def solution2 : Point := ⟨5, 5⟩

theorem two_solutions :
  ∀ p : Point, (onLine p ∧ hasDistance13 p) ↔ (p = solution1 ∨ p = solution2) := by
  sorry

end two_solutions_l2903_290388


namespace lottery_not_guaranteed_win_l2903_290314

theorem lottery_not_guaranteed_win (total_tickets : ℕ) (winning_rate : ℝ) (bought_tickets : ℕ) : 
  total_tickets = 1000000 →
  winning_rate = 0.001 →
  bought_tickets = 1000 →
  ∃ p : ℝ, p > 0 ∧ p = (1 - winning_rate) ^ bought_tickets := by
  sorry

end lottery_not_guaranteed_win_l2903_290314


namespace expression_evaluation_l2903_290385

/-- Given x = 3, y = 2, and z = 4, prove that 3 * x - 2 * y + 4 * z = 21 -/
theorem expression_evaluation (x y z : ℕ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  3 * x - 2 * y + 4 * z = 21 := by
  sorry

end expression_evaluation_l2903_290385


namespace problem_solution_l2903_290345

/-- Binary operation ★ on ordered pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) ↦ (a - c, b + d)

/-- Theorem stating that given the conditions, a = 2 -/
theorem problem_solution : 
  ∃ (a b : ℤ), star (5, 2) (1, 1) = (a, b) ∧ star (a, b) (0, 2) = (2, 5) → a = 2 := by
  sorry

end problem_solution_l2903_290345


namespace sin_double_theta_l2903_290360

theorem sin_double_theta (θ : ℝ) :
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (2 * θ) = 6 * Real.sqrt 8 / 25 := by
  sorry

end sin_double_theta_l2903_290360


namespace exponent_difference_equals_204_l2903_290398

theorem exponent_difference_equals_204 : 3^(1*(2+3)) - (3^1 + 3^2 + 3^3) = 204 := by
  sorry

end exponent_difference_equals_204_l2903_290398


namespace smallest_possible_b_l2903_290317

theorem smallest_possible_b (a b : ℝ) : 
  (2 < a ∧ a < b) →
  (2 + a ≤ b) →
  (1/b + 1/a ≤ 2) →
  b ≥ 2 + Real.sqrt 3 :=
sorry

end smallest_possible_b_l2903_290317


namespace blocks_differing_in_two_ways_l2903_290351

def num_materials : Nat := 3
def num_sizes : Nat := 2
def num_colors : Nat := 5
def num_shapes : Nat := 4

def count_different_blocks : Nat :=
  (num_materials - 1) * 1 + -- Material and Size
  (num_materials - 1) * (num_colors - 1) + -- Material and Color
  (num_materials - 1) * (num_shapes - 1) + -- Material and Shape
  1 * (num_colors - 1) + -- Size and Color
  1 * (num_shapes - 1) + -- Size and Shape
  (num_colors - 1) * (num_shapes - 1) -- Color and Shape

theorem blocks_differing_in_two_ways :
  count_different_blocks = 35 := by
  sorry

end blocks_differing_in_two_ways_l2903_290351


namespace tissue_cost_theorem_l2903_290307

/-- Calculates the total cost of tissue boxes given the number of boxes, packs per box, tissues per pack, and cost per tissue. -/
def total_cost (boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℚ) : ℚ :=
  (boxes * packs_per_box * tissues_per_pack : ℚ) * cost_per_tissue

/-- Proves that the total cost of 10 boxes of tissues is $1000 given the specified conditions. -/
theorem tissue_cost_theorem :
  let boxes : ℕ := 10
  let packs_per_box : ℕ := 20
  let tissues_per_pack : ℕ := 100
  let cost_per_tissue : ℚ := 5 / 100
  total_cost boxes packs_per_box tissues_per_pack cost_per_tissue = 1000 := by
  sorry

#eval total_cost 10 20 100 (5 / 100)

end tissue_cost_theorem_l2903_290307


namespace cross_quadrilateral_area_l2903_290390

/-- Given two rectangles ABCD and EFGH forming a cross shape, 
    prove that the area of quadrilateral AFCH is 52.5 -/
theorem cross_quadrilateral_area 
  (AB BC EF FG : ℝ) 
  (h_AB : AB = 9) 
  (h_BC : BC = 5) 
  (h_EF : EF = 3) 
  (h_FG : FG = 10) : 
  Real.sqrt ((AB * FG / 2 + BC * EF / 2) ^ 2 + (AB * BC + EF * FG - BC * EF) ^ 2) = 52.5 := by
  sorry

end cross_quadrilateral_area_l2903_290390


namespace net_error_is_24x_l2903_290355

/-- The net error in cents due to the cashier's miscounting -/
def net_error (x : ℕ) : ℤ :=
  let penny_value : ℤ := 1
  let nickel_value : ℤ := 5
  let dime_value : ℤ := 10
  let quarter_value : ℤ := 25
  let penny_to_nickel_error := x * (nickel_value - penny_value)
  let nickel_to_dime_error := x * (dime_value - nickel_value)
  let dime_to_quarter_error := x * (quarter_value - dime_value)
  penny_to_nickel_error + nickel_to_dime_error + dime_to_quarter_error

theorem net_error_is_24x (x : ℕ) : net_error x = 24 * x :=
sorry

end net_error_is_24x_l2903_290355


namespace earrings_ratio_is_two_to_one_l2903_290320

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on the first day -/
def first_day_pairs : ℕ := 3

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The number of days the gumballs should last -/
def total_days : ℕ := 42

/-- The number of pairs of earrings Kim brings on the second day -/
def second_day_pairs : ℕ := 6

theorem earrings_ratio_is_two_to_one :
  let total_gumballs := gumballs_per_pair * (first_day_pairs + second_day_pairs + (second_day_pairs - 1))
  total_gumballs = gumballs_eaten_per_day * total_days ∧
  second_day_pairs / first_day_pairs = 2 := by
  sorry

end earrings_ratio_is_two_to_one_l2903_290320


namespace triangle_angle_determination_l2903_290335

theorem triangle_angle_determination (a b : ℝ) (A B : Real) :
  a = 40 →
  b = 20 * Real.sqrt 2 →
  A = π / 4 →
  Real.sin B = (b * Real.sin A) / a →
  0 < B →
  B < π / 4 →
  B = π / 6 :=
sorry

end triangle_angle_determination_l2903_290335


namespace fraction_multiplication_one_half_of_one_third_of_one_sixth_of_72_l2903_290302

theorem fraction_multiplication (a b c d : ℚ) :
  a * b * c * d = (a * b * c) * d := by sorry

theorem one_half_of_one_third_of_one_sixth_of_72 :
  (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by sorry

end fraction_multiplication_one_half_of_one_third_of_one_sixth_of_72_l2903_290302


namespace average_speed_calculation_l2903_290389

def initial_reading : ℕ := 2552
def final_reading : ℕ := 2772
def total_time : ℕ := 9

theorem average_speed_calculation :
  (final_reading - initial_reading : ℚ) / total_time = 220 / 9 := by sorry

end average_speed_calculation_l2903_290389


namespace shaded_probability_three_fourths_l2903_290300

-- Define the right-angled triangle
structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse : ℝ
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

-- Define the game board
structure GameBoard where
  triangle : RightTriangle
  total_regions : ℕ
  shaded_regions : ℕ
  regions_by_altitudes : total_regions = 4
  shaded_count : shaded_regions = 3

-- Define the probability function
def probability_shaded (board : GameBoard) : ℚ :=
  board.shaded_regions / board.total_regions

-- Theorem statement
theorem shaded_probability_three_fourths 
  (board : GameBoard) 
  (h1 : board.triangle.leg1 = 6) 
  (h2 : board.triangle.leg2 = 8) : 
  probability_shaded board = 3/4 := by
  sorry


end shaded_probability_three_fourths_l2903_290300


namespace stripe_area_on_cylinder_l2903_290323

/-- The area of a stripe on a cylindrical water tower -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℕ) :
  diameter = 20 ∧ stripe_width = 4 ∧ revolutions = 3 →
  stripe_width * revolutions * π * diameter = 240 * π :=
by sorry

end stripe_area_on_cylinder_l2903_290323


namespace tangent_perpendicular_to_line_l2903_290311

theorem tangent_perpendicular_to_line (a b : ℝ) : 
  b = a^3 →                             -- point (a, b) is on the curve y = x^3
  (3 * a^2) * (-1/3) = -1 →             -- tangent is perpendicular to x + 3y + 1 = 0
  a = 1 ∨ a = -1 :=                     -- conclusion: a = 1 or a = -1
by
  sorry

end tangent_perpendicular_to_line_l2903_290311


namespace multiples_of_6_factors_of_72_l2903_290370

def is_multiple_of_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_factor_of_72 (n : ℕ) : Prop := 72 % n = 0

def solution_set : Set ℕ := {6, 12, 18, 24, 36, 72}

theorem multiples_of_6_factors_of_72 :
  ∀ n : ℕ, (is_multiple_of_6 n ∧ is_factor_of_72 n) ↔ n ∈ solution_set :=
sorry

end multiples_of_6_factors_of_72_l2903_290370


namespace quadratic_integer_root_existence_l2903_290310

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Check if a quadratic polynomial has an integer root -/
def has_integer_root (p : QuadraticPolynomial) : Prop :=
  ∃ x : ℤ, p.a * x * x + p.b * x + p.c = 0

/-- Calculate the cost of changing from one polynomial to another -/
def change_cost (p q : QuadraticPolynomial) : ℕ :=
  (Int.natAbs (p.a - q.a)) + (Int.natAbs (p.b - q.b)) + (Int.natAbs (p.c - q.c))

/-- The main theorem -/
theorem quadratic_integer_root_existence (p : QuadraticPolynomial) 
    (h : p.a + p.b + p.c = 2000) :
    ∃ q : QuadraticPolynomial, has_integer_root q ∧ change_cost p q ≤ 1022 := by
  sorry

end quadratic_integer_root_existence_l2903_290310


namespace car_owners_without_motorcycle_l2903_290391

theorem car_owners_without_motorcycle (total_adults : ℕ) (car_owners : ℕ) (motorcycle_owners : ℕ) (no_vehicle_owners : ℕ) 
  (h1 : total_adults = 560)
  (h2 : car_owners = 520)
  (h3 : motorcycle_owners = 80)
  (h4 : no_vehicle_owners = 10) :
  car_owners - (total_adults - no_vehicle_owners - (car_owners + motorcycle_owners - (total_adults - no_vehicle_owners))) = 470 := by
  sorry

end car_owners_without_motorcycle_l2903_290391


namespace sqrt_calculation_l2903_290350

theorem sqrt_calculation : 
  Real.sqrt 2 * Real.sqrt 6 - 4 * Real.sqrt (1/2) - (1 - Real.sqrt 3)^2 = 
  4 * Real.sqrt 3 - 2 * Real.sqrt 2 - 4 := by
  sorry

end sqrt_calculation_l2903_290350


namespace salary_increase_proof_l2903_290382

/-- Calculates the increase in average salary when adding a manager to a group of employees -/
def salary_increase (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) : ℚ :=
  let new_total := num_employees * initial_avg + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - initial_avg

/-- The increase in average salary when adding a manager's salary of 3300 to a group of 20 employees with an initial average salary of 1200 is equal to 100 -/
theorem salary_increase_proof :
  salary_increase 20 1200 3300 = 100 := by
  sorry

end salary_increase_proof_l2903_290382


namespace min_value_expression_equality_condition_l2903_290341

theorem min_value_expression (x : ℝ) (hx : x > 0) : 2 * Real.sqrt x + 1 / x + x^2 ≥ 4 :=
by sorry

theorem equality_condition : 2 * Real.sqrt 1 + 1 / 1 + 1^2 = 4 :=
by sorry

end min_value_expression_equality_condition_l2903_290341


namespace remainder_theorem_l2903_290325

-- Define the polynomial p(x)
variable (p : ℝ → ℝ)

-- Define the conditions
axiom remainder_x_minus_3 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * q x + 7
axiom remainder_x_plus_2 : ∃ q : ℝ → ℝ, ∀ x, p x = (x + 2) * q x - 3

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * (x + 2) * q x + (2 * x + 1) :=
sorry

end remainder_theorem_l2903_290325


namespace ellipse_theorem_proof_l2903_290313

noncomputable def ellipse_theorem (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  let e := Real.sqrt (a^2 - b^2)
  ∀ (M : ℝ × ℝ),
    M.1^2 / a^2 + M.2^2 / b^2 = 1 →
    let F₁ : ℝ × ℝ := (-e, 0)
    let F₂ : ℝ × ℝ := (e, 0)
    ∃ (A B : ℝ × ℝ),
      A.1^2 / a^2 + A.2^2 / b^2 = 1 ∧
      B.1^2 / a^2 + B.2^2 / b^2 = 1 ∧
      (∃ t : ℝ, A = M + t • (M - F₁)) ∧
      (∃ s : ℝ, B = M + s • (M - F₂)) →
      (b^2 / a^2) * (‖M - F₁‖ / ‖F₁ - A‖ + ‖M - F₂‖ / ‖F₂ - B‖ + 2) = 4

theorem ellipse_theorem_proof (a b : ℝ) (h : a > b ∧ b > 0) :
  ellipse_theorem a b h := by
  sorry

end ellipse_theorem_proof_l2903_290313


namespace transformation_maps_points_l2903_290384

/-- Represents a 2D point -/
structure Point where
  x : ℝ
  y : ℝ

/-- Scales a point by a factor about the origin -/
def scale (p : Point) (factor : ℝ) : Point :=
  { x := p.x * factor, y := p.y * factor }

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Applies scaling followed by reflection across x-axis -/
def scaleAndReflect (p : Point) (factor : ℝ) : Point :=
  reflectX (scale p factor)

theorem transformation_maps_points :
  let C : Point := { x := -5, y := 2 }
  let D : Point := { x := 0, y := 3 }
  let C' : Point := { x := 10, y := -4 }
  let D' : Point := { x := 0, y := -6 }
  (scaleAndReflect C 2 = C') ∧ (scaleAndReflect D 2 = D') := by
  sorry

end transformation_maps_points_l2903_290384


namespace positive_integer_pairs_l2903_290363

theorem positive_integer_pairs (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (∃ k : ℤ, (a^3 * b - 1 : ℤ) = k * (a + 1)) ∧
  (∃ m : ℤ, (b^3 * a + 1 : ℤ) = m * (b - 1)) →
  ((a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3)) :=
by sorry

end positive_integer_pairs_l2903_290363


namespace sum_trailing_zeros_15_factorial_l2903_290376

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def trailingZerosBase10 (n : ℕ) : ℕ := 
  (List.range 5).foldl (fun acc i => acc + n / (5 ^ (i + 1))) 0

def trailingZerosBase12 (n : ℕ) : ℕ := 
  min 
    ((List.range 2).foldl (fun acc i => acc + n / (3 ^ (i + 1))) 0)
    ((List.range 3).foldl (fun acc i => acc + n / (2 ^ (i + 1))) 0 / 2)

theorem sum_trailing_zeros_15_factorial : 
  trailingZerosBase12 (factorial 15) + trailingZerosBase10 (factorial 15) = 8 := by
  sorry

end sum_trailing_zeros_15_factorial_l2903_290376


namespace powerSum7Seq_36th_l2903_290319

/-- Sequence of sums of distinct powers of 7 -/
def powerSum7Seq : ℕ → ℕ
  | 0 => 1
  | n + 1 => powerSum7Seq n + 7^(n.log2)

/-- The 36th number in the sequence is 16856 -/
theorem powerSum7Seq_36th : powerSum7Seq 35 = 16856 := by
  sorry

end powerSum7Seq_36th_l2903_290319


namespace multiply_polynomials_l2903_290356

theorem multiply_polynomials (x : ℝ) : 
  (x^4 + 10*x^2 + 25) * (x^2 - 25) = x^4 + 10*x^2 := by
  sorry

end multiply_polynomials_l2903_290356


namespace quadratic_ratio_l2903_290324

theorem quadratic_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 2600*x + 2600 = (x + d)^2 + e ∧ e / d = -1298 := by
sorry

end quadratic_ratio_l2903_290324


namespace distance_traveled_l2903_290361

-- Define the velocity function
def velocity (t : ℝ) : ℝ := t^2 + 1

-- Define the theorem
theorem distance_traveled (v : ℝ → ℝ) (a b : ℝ) : 
  (v = velocity) → (a = 0) → (b = 3) → ∫ x in a..b, v x = 12 := by
  sorry

end distance_traveled_l2903_290361


namespace three_pairs_same_difference_l2903_290364

theorem three_pairs_same_difference (X : Finset ℕ) 
  (h1 : X ⊆ Finset.range 18 \ {0})
  (h2 : X.card = 8) : 
  ∃ (a b c d e f : ℕ), a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ d ∈ X ∧ e ∈ X ∧ f ∈ X ∧ 
  a ≠ b ∧ c ≠ d ∧ e ≠ f ∧
  (a - b : ℤ) = (c - d : ℤ) ∧ (c - d : ℤ) = (e - f : ℤ) :=
by sorry

end three_pairs_same_difference_l2903_290364


namespace sin_1035_degrees_l2903_290392

theorem sin_1035_degrees : Real.sin (1035 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_1035_degrees_l2903_290392


namespace crayons_left_l2903_290396

theorem crayons_left (initial_crayons lost_crayons : ℕ) 
  (h1 : initial_crayons = 253)
  (h2 : lost_crayons = 70) :
  initial_crayons - lost_crayons = 183 := by
sorry

end crayons_left_l2903_290396


namespace newer_model_travels_200_miles_l2903_290383

/-- The distance traveled by the older model car -/
def older_model_distance : ℝ := 160

/-- The percentage increase in distance for the newer model -/
def newer_model_percentage : ℝ := 0.25

/-- The distance traveled by the newer model car -/
def newer_model_distance : ℝ := older_model_distance * (1 + newer_model_percentage)

/-- Theorem stating that the newer model travels 200 miles -/
theorem newer_model_travels_200_miles :
  newer_model_distance = 200 := by sorry

end newer_model_travels_200_miles_l2903_290383


namespace complex_equation_result_l2903_290353

theorem complex_equation_result (z : ℂ) 
  (h : 15 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 4) + 25) : 
  z + 8 / z = -4 := by
  sorry

end complex_equation_result_l2903_290353


namespace probability_theorem_l2903_290349

def shirts : ℕ := 6
def shorts : ℕ := 8
def socks : ℕ := 7
def total_items : ℕ := shirts + shorts + socks
def items_chosen : ℕ := 4

def probability_specific_combination : ℚ :=
  (Nat.choose shirts 1 * Nat.choose shorts 2 * Nat.choose socks 1) /
  Nat.choose total_items items_chosen

theorem probability_theorem :
  probability_specific_combination = 392 / 1995 := by
  sorry

end probability_theorem_l2903_290349


namespace smallest_among_four_numbers_l2903_290359

theorem smallest_among_four_numbers :
  let a : ℝ := -Real.sqrt 3
  let b : ℝ := 0
  let c : ℝ := 2
  let d : ℝ := -3
  d < a ∧ d < b ∧ d < c := by sorry

end smallest_among_four_numbers_l2903_290359


namespace simplify_expression_l2903_290399

theorem simplify_expression : (512 : ℝ)^(1/3) * (343 : ℝ)^(1/2) = 56 * Real.sqrt 7 := by
  sorry

end simplify_expression_l2903_290399


namespace power_two_gt_square_plus_one_l2903_290379

theorem power_two_gt_square_plus_one (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 := by
  sorry

end power_two_gt_square_plus_one_l2903_290379


namespace will_hero_count_l2903_290346

/-- Represents the number of heroes drawn on a sheet of paper -/
structure HeroCount where
  front : Nat
  back : Nat
  third : Nat

/-- Calculates the total number of heroes drawn -/
def totalHeroes (h : HeroCount) : Nat :=
  h.front + h.back + h.third

/-- Theorem: Given the specific hero counts, the total is 19 -/
theorem will_hero_count :
  ∃ (h : HeroCount), h.front = 4 ∧ h.back = 9 ∧ h.third = 6 ∧ totalHeroes h = 19 :=
by sorry

end will_hero_count_l2903_290346


namespace triangle_height_l2903_290332

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 3 → area = 6 → area = (base * height) / 2 → height = 4 := by
  sorry

end triangle_height_l2903_290332


namespace roosters_count_l2903_290362

/-- Given a total number of chickens and a proportion of roosters to hens to chicks,
    calculate the number of roosters. -/
def count_roosters (total_chickens : ℕ) (rooster_parts hen_parts chick_parts : ℕ) : ℕ :=
  let total_parts := rooster_parts + hen_parts + chick_parts
  let chickens_per_part := total_chickens / total_parts
  rooster_parts * chickens_per_part

/-- Theorem stating that given 9000 total chickens and a proportion of 2:1:3 for
    roosters:hens:chicks, the number of roosters is 3000. -/
theorem roosters_count :
  count_roosters 9000 2 1 3 = 3000 := by
  sorry

end roosters_count_l2903_290362


namespace expression_evaluation_l2903_290394

theorem expression_evaluation : (-2 : ℤ) ^ (4^2) + 1^(3^3) = 65537 := by
  sorry

end expression_evaluation_l2903_290394


namespace count_five_divisors_l2903_290336

theorem count_five_divisors (n : ℕ) (h : n = 50000) : 
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125) + (n / 15625) = 12499 :=
by sorry

end count_five_divisors_l2903_290336


namespace rug_area_l2903_290315

theorem rug_area (w l : ℝ) (h1 : l = w + 8) 
  (h2 : (w + 16) * (l + 16) - w * l = 704) : w * l = 180 := by
  sorry

end rug_area_l2903_290315


namespace video_game_lives_l2903_290393

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 47)
  (h2 : lost_lives = 23)
  (h3 : gained_lives = 46) :
  initial_lives - lost_lives + gained_lives = 70 := by
  sorry

end video_game_lives_l2903_290393


namespace cubic_equation_solution_l2903_290339

theorem cubic_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3) ∧ y = 1000 := by
  sorry

end cubic_equation_solution_l2903_290339


namespace unique_triple_solution_l2903_290366

theorem unique_triple_solution :
  ∀ a b c : ℕ+,
    (∃ k₁ : ℕ, a * b + 1 = k₁ * c) ∧
    (∃ k₂ : ℕ, a * c + 1 = k₂ * b) ∧
    (∃ k₃ : ℕ, b * c + 1 = k₃ * a) →
    a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end unique_triple_solution_l2903_290366


namespace polynomial_inequality_l2903_290304

/-- A polynomial satisfying the given property -/
def GoodPolynomial (p : ℝ → ℝ) : Prop :=
  ∀ x, p (x + 1) - p x = x^100

/-- The main theorem to prove -/
theorem polynomial_inequality (p : ℝ → ℝ) (hp : GoodPolynomial p) :
  ∀ t, 0 ≤ t → t ≤ 1/2 → p (1 - t) ≥ p t := by
  sorry

end polynomial_inequality_l2903_290304


namespace outfit_combinations_l2903_290306

theorem outfit_combinations (shirts : Nat) (ties : Nat) (hats : Nat) :
  shirts = 8 → ties = 6 → hats = 4 → shirts * ties * hats = 192 := by
  sorry

end outfit_combinations_l2903_290306


namespace negation_equivalence_l2903_290395

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end negation_equivalence_l2903_290395


namespace lottery_profit_lilys_profit_l2903_290380

/-- Calculates the profit from selling lottery tickets -/
theorem lottery_profit (n : ℕ) (first_price : ℕ) (prize : ℕ) : ℕ :=
  let total_revenue := n * (2 * first_price + (n - 1)) / 2
  total_revenue - prize

/-- Proves that Lily's profit is $4 given the specified conditions -/
theorem lilys_profit :
  lottery_profit 5 1 11 = 4 := by
  sorry

end lottery_profit_lilys_profit_l2903_290380
