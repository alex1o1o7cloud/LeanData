import Mathlib

namespace quadratic_equation_roots_l3683_368354

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) :
  (∀ x, x^2 - 2*k*x + k^2 - k - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  (k = 5 → x₁*x₂^2 + x₁^2*x₂ = 190) ∧
  (x₁ - 3*x₂ = 2 → k = 3) :=
by sorry

end quadratic_equation_roots_l3683_368354


namespace cos_alpha_for_point_in_third_quadrant_l3683_368321

theorem cos_alpha_for_point_in_third_quadrant (a : ℝ) (α : ℝ) :
  a < 0 →
  ∃ (P : ℝ × ℝ), P = (3*a, 4*a) ∧ 
  (∃ (r : ℝ), r > 0 ∧ P = (r * Real.cos α, r * Real.sin α)) →
  Real.cos α = -3/5 := by
  sorry

end cos_alpha_for_point_in_third_quadrant_l3683_368321


namespace compound_propositions_truth_l3683_368350

-- Define proposition p
def p : Prop := ∀ x : ℝ, (x^2 - 3*x + 2 = 0 → x = 1) ∧ (x ≠ 1 → x^2 - 3*x + 2 ≠ 0)

-- Define proposition q
def q : Prop := ∀ a b : ℝ, a^(1/2) > b^(1/2) ↔ Real.log a > Real.log b

theorem compound_propositions_truth : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ((¬p) ∨ (¬q)) ∧ (p ∧ (¬q)) := by sorry

end compound_propositions_truth_l3683_368350


namespace square_difference_divisible_by_13_l3683_368376

theorem square_difference_divisible_by_13 (a b : ℕ) :
  a ∈ Finset.range 1001 →
  b ∈ Finset.range 1001 →
  a + b = 1001 →
  13 ∣ (a^2 - b^2) :=
by sorry

end square_difference_divisible_by_13_l3683_368376


namespace coefficient_x_cubed_in_expansion_l3683_368328

/-- The coefficient of x^3 in the expansion of (2x + √x)^5 is 10 -/
theorem coefficient_x_cubed_in_expansion : ℕ := by
  sorry

end coefficient_x_cubed_in_expansion_l3683_368328


namespace right_triangle_hypotenuse_l3683_368370

/-- A right triangle with the given cone volume properties has a hypotenuse of approximately 21.3 cm. -/
theorem right_triangle_hypotenuse (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (1 / 3 * π * y^2 * x = 1250 * π) →
  (1 / 3 * π * x^2 * y = 2700 * π) →
  abs (Real.sqrt (x^2 + y^2) - 21.3) < 0.1 := by
  sorry

#check right_triangle_hypotenuse

end right_triangle_hypotenuse_l3683_368370


namespace chinese_character_sum_l3683_368387

theorem chinese_character_sum : ∃! (a b c d e f g : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
   e ≠ f ∧ e ≠ g ∧
   f ≠ g) ∧
  (1000 * a + 100 * b + 10 * c + d + 100 * e + 10 * f + g = 2013) ∧
  (a + b + c + d + e + f + g = 24) :=
by sorry

end chinese_character_sum_l3683_368387


namespace involutive_function_theorem_l3683_368303

/-- A function f is involutive if f(f(x)) = x for all x in its domain -/
def Involutive (f : ℝ → ℝ) : Prop :=
  ∀ x, f (f x) = x

/-- The main theorem -/
theorem involutive_function_theorem (a b c d : ℝ) 
    (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) :
    let f := fun x => (2 * a * x + b) / (3 * c * x + 2 * d)
    Involutive f → 2 * a + 2 * d = 0 := by
  sorry


end involutive_function_theorem_l3683_368303


namespace bugs_eating_flowers_l3683_368342

/-- Given 3 bugs, each eating 2 flowers, prove that the total number of flowers eaten is 6. -/
theorem bugs_eating_flowers : 
  let num_bugs : ℕ := 3
  let flowers_per_bug : ℕ := 2
  num_bugs * flowers_per_bug = 6 := by
  sorry

end bugs_eating_flowers_l3683_368342


namespace inequality_solution_l3683_368390

theorem inequality_solution (x : ℝ) : 
  1 / (x^2 + 1) > 3 / x + 17 / 10 ↔ -2 < x ∧ x < 0 :=
by sorry

end inequality_solution_l3683_368390


namespace max_intersection_points_l3683_368335

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two polygons in a plane -/
structure PolygonConfiguration where
  Q₁ : ConvexPolygon
  Q₂ : ConvexPolygon
  no_shared_segments : Bool
  potentially_intersect : Bool

/-- Theorem: Maximum number of intersection points between two convex polygons -/
theorem max_intersection_points (config : PolygonConfiguration) 
  (h1 : config.Q₁.convex = true)
  (h2 : config.Q₂.convex = true)
  (h3 : config.Q₂.sides ≥ config.Q₁.sides + 3)
  (h4 : config.no_shared_segments = true)
  (h5 : config.potentially_intersect = true) :
  (max_intersections : ℕ) → max_intersections = config.Q₁.sides * config.Q₂.sides :=
by sorry

end max_intersection_points_l3683_368335


namespace new_average_is_250_l3683_368375

/-- A salesperson's commission information -/
structure SalesCommission where
  totalSales : ℕ
  lastCommission : ℝ
  averageIncrease : ℝ

/-- Calculate the new average commission after a big sale -/
def newAverageCommission (sc : SalesCommission) : ℝ :=
  sorry

/-- Theorem stating the new average commission is $250 under given conditions -/
theorem new_average_is_250 (sc : SalesCommission) 
  (h1 : sc.totalSales = 6)
  (h2 : sc.lastCommission = 1000)
  (h3 : sc.averageIncrease = 150) :
  newAverageCommission sc = 250 :=
sorry

end new_average_is_250_l3683_368375


namespace correct_calculation_l3683_368326

theorem correct_calculation : (1/3) + (-1/2) = -1/6 := by
  sorry

end correct_calculation_l3683_368326


namespace remainder_theorem_l3683_368363

theorem remainder_theorem : (8 * 20^34 + 3^34) % 7 = 5 := by
  sorry

end remainder_theorem_l3683_368363


namespace john_skateboard_distance_l3683_368349

/-- Represents the distance John traveled in miles -/
structure JohnTrip where
  skateboard_to_park : ℕ
  walk_to_park : ℕ

/-- Calculates the total distance John skateboarded -/
def total_skateboard_distance (trip : JohnTrip) : ℕ :=
  2 * trip.skateboard_to_park + trip.skateboard_to_park

theorem john_skateboard_distance :
  ∀ (trip : JohnTrip),
    trip.skateboard_to_park = 10 ∧ trip.walk_to_park = 4 →
    total_skateboard_distance trip = 24 :=
by
  sorry

#check john_skateboard_distance

end john_skateboard_distance_l3683_368349


namespace desired_average_grade_l3683_368317

def first_test_score : ℚ := 95
def second_test_score : ℚ := 80
def third_test_score : ℚ := 95

def average_grade : ℚ := (first_test_score + second_test_score + third_test_score) / 3

theorem desired_average_grade :
  average_grade = 90 := by sorry

end desired_average_grade_l3683_368317


namespace min_odd_integers_l3683_368325

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 22)
  (sum2 : a + b + c + d = 36)
  (sum3 : a + b + c + d + e + f = 50) :
  ∃ (a' b' c' d' e' f' : ℤ), 
    a' % 2 = 0 ∧ b' % 2 = 0 ∧ c' % 2 = 0 ∧ d' % 2 = 0 ∧ e' % 2 = 0 ∧ f' % 2 = 0 ∧
    a' + b' = 22 ∧
    a' + b' + c' + d' = 36 ∧
    a' + b' + c' + d' + e' + f' = 50 :=
by sorry

end min_odd_integers_l3683_368325


namespace probability_second_shiny_penny_l3683_368398

def total_pennies : ℕ := 7
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 3

def probability_more_than_three_draws : ℚ :=
  (Nat.choose 3 1 * Nat.choose 4 1 + Nat.choose 3 0 * Nat.choose 4 2) / Nat.choose total_pennies shiny_pennies

theorem probability_second_shiny_penny :
  probability_more_than_three_draws = 18 / 35 := by sorry

end probability_second_shiny_penny_l3683_368398


namespace exists_M_with_properties_l3683_368309

def is_valid_last_four_digits (n : ℕ) : Prop :=
  n < 10000 ∧ 
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 4*a - 3*b ∧
  (n / 1000 ≠ (n / 100) % 10) ∧
  (n / 1000 ≠ (n / 10) % 10) ∧
  (n / 1000 ≠ n % 10) ∧
  (n / 100 % 10 ≠ (n / 10) % 10) ∧
  (n / 100 % 10 ≠ n % 10) ∧
  ((n / 10) % 10 ≠ n % 10)

theorem exists_M_with_properties :
  ∃ (M : ℕ), 
    M % 8 = 0 ∧
    M % 16 ≠ 0 ∧
    is_valid_last_four_digits (M % 10000) ∧
    M % 1000 = 624 :=
sorry

end exists_M_with_properties_l3683_368309


namespace assemble_cook_time_is_five_l3683_368360

/-- The time it takes to assemble and cook one omelet -/
def assemble_cook_time (
  pepper_chop_time : ℕ)  -- Time to chop one pepper
  (onion_chop_time : ℕ)  -- Time to chop one onion
  (cheese_grate_time : ℕ)  -- Time to grate cheese for one omelet
  (num_peppers : ℕ)  -- Number of peppers to chop
  (num_onions : ℕ)  -- Number of onions to chop
  (num_omelets : ℕ)  -- Number of omelets to make
  (total_time : ℕ)  -- Total time for preparing and cooking all omelets
  : ℕ :=
  let prep_time := pepper_chop_time * num_peppers + onion_chop_time * num_onions + cheese_grate_time * num_omelets
  (total_time - prep_time) / num_omelets

/-- Theorem stating that it takes 5 minutes to assemble and cook one omelet -/
theorem assemble_cook_time_is_five :
  assemble_cook_time 3 4 1 4 2 5 50 = 5 := by
  sorry


end assemble_cook_time_is_five_l3683_368360


namespace pat_earned_stickers_l3683_368322

/-- The number of stickers Pat had at the start of the week -/
def initial_stickers : ℕ := 39

/-- The number of stickers Pat had at the end of the week -/
def final_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := final_stickers - initial_stickers

theorem pat_earned_stickers : earned_stickers = 22 := by
  sorry

end pat_earned_stickers_l3683_368322


namespace car_travel_time_l3683_368385

/-- Proves that a car traveling 715 kilometers at an average speed of 65.0 km/h takes 11 hours -/
theorem car_travel_time (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 715 →
  speed = 65 →
  time = distance / speed →
  time = 11 :=
by sorry

end car_travel_time_l3683_368385


namespace opposite_sides_m_range_l3683_368319

/-- Given two points on opposite sides of a line, prove the range of m -/
theorem opposite_sides_m_range :
  ∀ (m : ℝ),
  (2 * 1 + 3 + m) * (2 * (-4) + (-2) + m) < 0 →
  m ∈ Set.Ioo (-5 : ℝ) 10 :=
by sorry

end opposite_sides_m_range_l3683_368319


namespace blueberry_pie_count_l3683_368386

theorem blueberry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 36 →
  apple_ratio = 2 →
  blueberry_ratio = 5 →
  cherry_ratio = 3 →
  blueberry_ratio * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 18 :=
by sorry

end blueberry_pie_count_l3683_368386


namespace nyc_streetlights_l3683_368388

/-- Given the total number of streetlights bought, the number of squares, and the number of streetlights per square, 
    calculate the number of unused streetlights. -/
def unused_streetlights (total : ℕ) (squares : ℕ) (per_square : ℕ) : ℕ :=
  total - squares * per_square

/-- Theorem stating that with 200 total streetlights, 15 squares, and 12 streetlights per square, 
    there will be 20 unused streetlights. -/
theorem nyc_streetlights : unused_streetlights 200 15 12 = 20 := by
  sorry

end nyc_streetlights_l3683_368388


namespace max_square_plots_is_48_l3683_368364

/-- Represents the dimensions and constraints of the field --/
structure FieldParameters where
  length : ℝ
  width : ℝ
  pathwayWidth : ℝ
  availableFencing : ℝ

/-- Calculates the maximum number of square test plots --/
def maxSquarePlots (params : FieldParameters) : ℕ :=
  sorry

/-- Theorem stating the maximum number of square test plots --/
theorem max_square_plots_is_48 (params : FieldParameters) :
  params.length = 45 ∧ 
  params.width = 30 ∧ 
  params.pathwayWidth = 5 ∧ 
  params.availableFencing = 2700 →
  maxSquarePlots params = 48 :=
sorry

end max_square_plots_is_48_l3683_368364


namespace ellipse_point_coordinates_l3683_368302

theorem ellipse_point_coordinates (x y α : ℝ) : 
  x = 4 * Real.cos α → 
  y = 2 * Real.sqrt 3 * Real.sin α → 
  x > 0 → 
  y > 0 → 
  y / x = Real.sqrt 3 → 
  (x, y) = (4 * Real.sqrt 5 / 5, 4 * Real.sqrt 15 / 5) := by
sorry

end ellipse_point_coordinates_l3683_368302


namespace polynomial_real_root_iff_b_negative_l3683_368392

/-- The polynomial x^3 + bx^2 - x + b = 0 has at least one real root if and only if b < 0 -/
theorem polynomial_real_root_iff_b_negative :
  ∀ b : ℝ, (∃ x : ℝ, x^3 + b*x^2 - x + b = 0) ↔ b < 0 := by sorry

end polynomial_real_root_iff_b_negative_l3683_368392


namespace function_intersection_and_tangency_l3683_368369

/-- Given two functions f and g, prove that under certain conditions, 
    the coefficients a, b, and c have specific values. -/
theorem function_intersection_and_tangency 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = 2 * x^3 + a * x)
  (h2 : ∀ x, g x = b * x^2 + c)
  (h3 : f 2 = 0)
  (h4 : g 2 = 0)
  (h5 : (deriv f) 2 = (deriv g) 2) : 
  a = -8 ∧ b = 4 ∧ c = -16 := by
  sorry

end function_intersection_and_tangency_l3683_368369


namespace emily_beads_count_l3683_368373

/-- The number of necklaces Emily made -/
def num_necklaces : ℕ := 11

/-- The number of beads required for each necklace -/
def beads_per_necklace : ℕ := 28

/-- The total number of beads Emily had -/
def total_beads : ℕ := num_necklaces * beads_per_necklace

theorem emily_beads_count : total_beads = 308 := by
  sorry

end emily_beads_count_l3683_368373


namespace largest_c_for_inequality_l3683_368353

theorem largest_c_for_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ c : ℝ, c = |Real.log (a / b)| ∧
  (∀ x α : ℝ, 0 < |x| → |x| ≤ c → 0 < α → α < 1 →
    a^α * b^(1-α) ≤ a * (Real.sinh (α*x) / Real.sinh x) + b * (Real.sinh ((1-α)*x) / Real.sinh x)) ∧
  (∀ c' : ℝ, c' > c →
    ∃ x α : ℝ, 0 < |x| ∧ |x| ≤ c' ∧ 0 < α ∧ α < 1 ∧
      a^α * b^(1-α) > a * (Real.sinh (α*x) / Real.sinh x) + b * (Real.sinh ((1-α)*x) / Real.sinh x)) :=
by sorry

end largest_c_for_inequality_l3683_368353


namespace cone_height_from_circular_sector_l3683_368359

theorem cone_height_from_circular_sector (r : ℝ) (h : r = 8) :
  let sector_arc_length := 2 * π * r / 4
  let cone_base_radius := sector_arc_length / (2 * π)
  let cone_slant_height := r
  cone_slant_height ^ 2 - cone_base_radius ^ 2 = (2 * Real.sqrt 15) ^ 2 :=
by sorry

end cone_height_from_circular_sector_l3683_368359


namespace good_numbers_exist_exist_good_sum_not_good_l3683_368324

/-- A number is "good" if it can be expressed as a^2 + 161b^2 for some integers a and b -/
def is_good_number (n : ℤ) : Prop :=
  ∃ a b : ℤ, n = a^2 + 161 * b^2

theorem good_numbers_exist : is_good_number 100 ∧ is_good_number 2010 := by sorry

theorem exist_good_sum_not_good :
  ∃ x y : ℕ+, is_good_number (x^161 + y^161) ∧ ¬is_good_number (x + y) := by sorry

end good_numbers_exist_exist_good_sum_not_good_l3683_368324


namespace garden_area_l3683_368394

theorem garden_area (total_distance : ℝ) (length_walks : ℕ) (perimeter_walks : ℕ) :
  total_distance = 1500 →
  length_walks = 30 →
  perimeter_walks = 12 →
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * length_walks = total_distance ∧
    2 * (length + width) * perimeter_walks = total_distance ∧
    length * width = 625 := by
  sorry


end garden_area_l3683_368394


namespace pet_parasites_l3683_368358

theorem pet_parasites (dog_burrs : ℕ) : ℕ :=
  let dog_ticks := 6 * dog_burrs
  let dog_fleas := 3 * dog_ticks
  let cat_burrs := 2 * dog_burrs
  let cat_ticks := dog_ticks / 3
  let cat_fleas := 4 * cat_ticks
  let total_parasites := dog_burrs + dog_ticks + dog_fleas + cat_burrs + cat_ticks + cat_fleas
  
  by
  -- Assuming dog_burrs = 12
  have h : dog_burrs = 12 := by sorry
  -- Proof goes here
  sorry

-- The theorem states that given the number of burrs on the dog (which we know is 12),
-- we can calculate the total number of parasites on both pets.
-- The proof would show that this total is indeed 444.

end pet_parasites_l3683_368358


namespace abs_neg_nine_l3683_368352

theorem abs_neg_nine : |(-9 : ℤ)| = 9 := by sorry

end abs_neg_nine_l3683_368352


namespace solution_and_inequality_l3683_368313

-- Define the function f
def f (t : ℝ) (x : ℝ) : ℝ := |x - 4| - t

-- State the theorem
theorem solution_and_inequality (t : ℝ) 
  (h1 : Set.Icc (-1 : ℝ) 5 = {x | f t (x + 2) ≤ 2}) 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h2 : a + b + c = t) : 
  t = 1 ∧ a^2 / b + b^2 / c + c^2 / a ≥ 1 := by
  sorry

end solution_and_inequality_l3683_368313


namespace unique_intersection_point_l3683_368372

open Real

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - m * log x

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := x^2 - (m + 1) * x

noncomputable def h (m : ℝ) (x : ℝ) : ℝ := f m x - g m x

theorem unique_intersection_point (m : ℝ) (hm : m ≥ 1) :
  ∃! x, x > 0 ∧ h m x = 0 := by sorry

end unique_intersection_point_l3683_368372


namespace imaginary_power_sum_l3683_368306

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^18 + i^23 + i^28 + i^33 = i := by
  sorry

end imaginary_power_sum_l3683_368306


namespace eight_people_arrangement_l3683_368377

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people must sit together -/
def restrictedArrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where two specific people cannot sit next to each other -/
def acceptableArrangements (n : ℕ) : ℕ := totalArrangements n - restrictedArrangements n

theorem eight_people_arrangement :
  acceptableArrangements 8 = 30240 := by
  sorry

end eight_people_arrangement_l3683_368377


namespace cube_sum_equals_sum_l3683_368391

theorem cube_sum_equals_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end cube_sum_equals_sum_l3683_368391


namespace platform_length_l3683_368336

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_cross_time : ℝ) (pole_cross_time : ℝ) :
  train_length = 300 →
  platform_cross_time = 27 →
  pole_cross_time = 18 →
  (train_length + (train_length / pole_cross_time * platform_cross_time - train_length)) = 450 :=
by sorry

end platform_length_l3683_368336


namespace product_of_integers_l3683_368384

theorem product_of_integers (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p + q + r = 30 →
  1 / p + 1 / q + 1 / r + 450 / (p * q * r) = 1 →
  p * q * r = 1920 := by
sorry

end product_of_integers_l3683_368384


namespace simplify_expression_l3683_368310

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  sorry

end simplify_expression_l3683_368310


namespace perpendicular_transitivity_l3683_368380

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the different relation for planes and lines
variable (different : Plane → Plane → Prop)
variable (different_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (α β γ : Plane) (m n l : Line)
  (h_diff_planes : different α β ∧ different α γ ∧ different β γ)
  (h_diff_lines : different_line m n ∧ different_line m l ∧ different_line n l)
  (h_n_perp_α : perp n α)
  (h_n_perp_β : perp n β)
  (h_m_perp_α : perp m α) :
  perp m β :=
sorry

end perpendicular_transitivity_l3683_368380


namespace log_problem_l3683_368397

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_problem :
  4 * lg 2 + 3 * lg 5 - lg (1/5) = 4 := by
  sorry

end log_problem_l3683_368397


namespace ratio_12min_to_1hour_is_1_to_5_l3683_368315

/-- The ratio of 12 minutes to 1 hour -/
def ratio_12min_to_1hour : ℚ × ℚ :=
  sorry

/-- One hour in minutes -/
def minutes_per_hour : ℕ := 60

theorem ratio_12min_to_1hour_is_1_to_5 :
  ratio_12min_to_1hour = (1, 5) := by
  sorry

end ratio_12min_to_1hour_is_1_to_5_l3683_368315


namespace tangent_line_at_e_l3683_368327

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e :
  let x₀ : ℝ := Real.exp 1
  let y₀ : ℝ := f x₀
  let m : ℝ := Real.exp 1 * (1 / Real.exp 1) + Real.log (Real.exp 1)
  (λ x y => y = m * (x - x₀) + y₀) = (λ x y => y = 2 * x - Real.exp 1) :=
by sorry

end tangent_line_at_e_l3683_368327


namespace two_pipes_fill_time_l3683_368307

def fill_time (num_pipes : ℕ) (time : ℝ) : Prop :=
  num_pipes > 0 ∧ time > 0 ∧ num_pipes * time = 36

theorem two_pipes_fill_time :
  fill_time 3 12 → fill_time 2 18 :=
by
  sorry

end two_pipes_fill_time_l3683_368307


namespace sculpture_surface_area_l3683_368355

/-- Represents a sculpture made of unit cubes -/
structure Sculpture where
  totalCubes : Nat
  layer1 : Nat
  layer2 : Nat
  layer3 : Nat
  layer4 : Nat

/-- Calculate the exposed surface area of the sculpture -/
def exposedSurfaceArea (s : Sculpture) : Nat :=
  5 * s.layer1 + 4 * s.layer2 + s.layer3 + 3 * s.layer4

/-- The main theorem stating the exposed surface area of the specific sculpture -/
theorem sculpture_surface_area :
  ∃ (s : Sculpture),
    s.totalCubes = 20 ∧
    s.layer1 = 1 ∧
    s.layer2 = 4 ∧
    s.layer3 = 9 ∧
    s.layer4 = 6 ∧
    exposedSurfaceArea s = 48 := by
  sorry


end sculpture_surface_area_l3683_368355


namespace reciprocal_problem_l3683_368346

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 10) : 50 * (1 / x) = 40 := by
  sorry

end reciprocal_problem_l3683_368346


namespace min_guests_for_cheaper_second_planner_l3683_368348

/-- Represents the pricing model of an event planner -/
structure EventPlanner where
  flatFee : ℕ
  perGuestFee : ℕ

/-- Calculates the total cost for a given number of guests -/
def totalCost (planner : EventPlanner) (guests : ℕ) : ℕ :=
  planner.flatFee + planner.perGuestFee * guests

/-- Defines the two event planners -/
def planner1 : EventPlanner := { flatFee := 120, perGuestFee := 18 }
def planner2 : EventPlanner := { flatFee := 250, perGuestFee := 15 }

/-- Theorem stating the minimum number of guests for the second planner to be less expensive -/
theorem min_guests_for_cheaper_second_planner :
  ∀ n : ℕ, (n ≥ 44 → totalCost planner2 n < totalCost planner1 n) ∧
           (n < 44 → totalCost planner2 n ≥ totalCost planner1 n) := by
  sorry

end min_guests_for_cheaper_second_planner_l3683_368348


namespace circle_center_correct_l3683_368381

/-- The equation of a circle in the form ax^2 + bx + cy^2 + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, find its center -/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct (eq : CircleEquation) :
  eq.a = 1 ∧ eq.b = -10 ∧ eq.c = 1 ∧ eq.d = -4 ∧ eq.e = -20 →
  let center := findCircleCenter eq
  center.x = 5 ∧ center.y = 2 :=
sorry

end circle_center_correct_l3683_368381


namespace train_length_l3683_368347

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 54 → time = 7 → speed * time * (1000 / 3600) = 105 := by
  sorry

end train_length_l3683_368347


namespace rectangle_area_arithmetic_progression_l3683_368344

/-- The area of a rectangle with sides in arithmetic progression -/
theorem rectangle_area_arithmetic_progression (a d : ℚ) :
  let shorter_side := a
  let longer_side := a + d
  shorter_side > 0 → longer_side > shorter_side →
  (shorter_side * longer_side : ℚ) = a^2 + a*d :=
by sorry

end rectangle_area_arithmetic_progression_l3683_368344


namespace matchsticks_left_proof_l3683_368331

/-- The number of matchsticks left in the box after Elvis and Ralph make their squares -/
def matchsticks_left (total : ℕ) (elvis_squares : ℕ) (ralph_squares : ℕ) 
  (elvis_per_square : ℕ) (ralph_per_square : ℕ) : ℕ :=
  total - (elvis_squares * elvis_per_square + ralph_squares * ralph_per_square)

/-- Theorem stating that 6 matchsticks will be left in the box -/
theorem matchsticks_left_proof :
  matchsticks_left 50 5 3 4 8 = 6 := by
  sorry

#eval matchsticks_left 50 5 3 4 8

end matchsticks_left_proof_l3683_368331


namespace lcm_gcd_product_l3683_368396

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = 180 := by
  sorry

end lcm_gcd_product_l3683_368396


namespace subset_of_all_implies_zero_l3683_368323

theorem subset_of_all_implies_zero (a : ℝ) :
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end subset_of_all_implies_zero_l3683_368323


namespace floor_sum_example_l3683_368368

theorem floor_sum_example : ⌊(17.2 : ℝ)⌋ + ⌊(-17.2 : ℝ)⌋ = -1 := by sorry

end floor_sum_example_l3683_368368


namespace quadratic_roots_condition_l3683_368337

theorem quadratic_roots_condition (a : ℝ) : 
  (a ∈ Set.Ici 2 → ∃ x : ℝ, x^2 - a*x + 1 = 0) ∧ 
  (∃ a : ℝ, a ∉ Set.Ici 2 ∧ ∃ x : ℝ, x^2 - a*x + 1 = 0) :=
by sorry

end quadratic_roots_condition_l3683_368337


namespace aloh3_molecular_weight_l3683_368357

/-- The molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (alWeight oWeight hWeight : ℝ) (moles : ℝ) : ℝ :=
  moles * (alWeight + 3 * oWeight + 3 * hWeight)

/-- Theorem stating the molecular weight of 7 moles of Al(OH)3 -/
theorem aloh3_molecular_weight :
  molecularWeight 26.98 16.00 1.01 7 = 546.07 := by
  sorry

end aloh3_molecular_weight_l3683_368357


namespace geometric_sequence_common_ratio_l3683_368362

/-- Given a geometric sequence {a_n} where the sum of the first n terms
    is S_n = 3 × 2^n + m, prove that the common ratio is 2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ n, S n = 3 * 2^n + m) 
  (h2 : ∀ n, n ≥ 2 → a n = S n - S (n-1)) :
  ∀ n, n ≥ 2 → a (n+1) / a n = 2 :=
sorry

end geometric_sequence_common_ratio_l3683_368362


namespace condition_sufficient_not_necessary_l3683_368374

-- Define the condition
def condition (A : ℝ × ℝ) : Prop :=
  ∃ k : ℤ, A = (k * Real.pi, 0)

-- Define the statement
def statement (A : ℝ × ℝ) : Prop :=
  ∀ x : ℝ, Real.tan (A.1 + x) = -Real.tan (A.1 - x)

-- Theorem stating the condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ A : ℝ × ℝ, condition A → statement A) ∧
  ¬(∀ A : ℝ × ℝ, statement A → condition A) :=
sorry

end condition_sufficient_not_necessary_l3683_368374


namespace pascal_triangle_51st_row_third_number_l3683_368379

theorem pascal_triangle_51st_row_third_number : 
  let n : ℕ := 50  -- The row with 51 numbers corresponds to (x+y)^50
  let k : ℕ := 2   -- The third number (0-indexed) corresponds to k=2
  Nat.choose n k = 1225 := by
sorry

end pascal_triangle_51st_row_third_number_l3683_368379


namespace equation_transformation_l3683_368316

theorem equation_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = x^2 * (y^2 + y - 7) :=
by sorry

end equation_transformation_l3683_368316


namespace slope_intercept_sum_l3683_368389

/-- Given points X, Y, Z, and G as the midpoint of XY, prove that the sum of the slope
    and y-intercept of the line passing through Z and G is 18/5 -/
theorem slope_intercept_sum (X Y Z G : ℝ × ℝ) : 
  X = (0, 8) → Y = (0, 0) → Z = (10, 0) → 
  G = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) →
  let m := (G.2 - Z.2) / (G.1 - Z.1)
  let b := G.2
  m + b = 18 / 5 := by
sorry

end slope_intercept_sum_l3683_368389


namespace sequence_divisibility_l3683_368356

theorem sequence_divisibility (n : ℕ) : 
  (∃ k, k > 0 ∧ k * (k + 1) ≤ 14520 ∧ 120 ∣ (k * (k + 1))) ↔ 
  (∃ m, m ≥ 1 ∧ m ≤ 8 ∧ 120 ∣ (n * (n + 1))) :=
by sorry

end sequence_divisibility_l3683_368356


namespace sqrt_equation_solution_l3683_368399

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (1 - 3 * x) = 7 → x = -16 := by
  sorry

end sqrt_equation_solution_l3683_368399


namespace special_right_triangle_area_l3683_368301

/-- Represents a right triangle with an incircle that evenly trisects a median -/
structure SpecialRightTriangle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Incircle radius
  r : ℝ
  -- Median length
  m : ℝ
  -- Conditions
  right_angle : a^2 + b^2 = c^2
  hypotenuse : c = 24
  trisected_median : m = 3 * r
  area_condition : a * b = 288

/-- The main theorem -/
theorem special_right_triangle_area (t : SpecialRightTriangle) : 
  ∃ (m n : ℕ), t.a * t.b / 2 = m * Real.sqrt n ∧ m = 144 ∧ n = 1 ∧ ¬ ∃ (p : ℕ), Prime p ∧ n % (p^2) = 0 :=
sorry

end special_right_triangle_area_l3683_368301


namespace tuition_fee_agreement_percentage_l3683_368333

theorem tuition_fee_agreement_percentage (total_parents : ℕ) (disagree_parents : ℕ) 
  (h1 : total_parents = 800) (h2 : disagree_parents = 640) : 
  (total_parents - disagree_parents : ℝ) / total_parents * 100 = 20 := by
  sorry

end tuition_fee_agreement_percentage_l3683_368333


namespace money_distribution_l3683_368366

theorem money_distribution (x : ℝ) (x_pos : x > 0) : 
  let adriano_initial := 5 * x
  let bruno_initial := 4 * x
  let cesar_initial := 3 * x
  let total_initial := adriano_initial + bruno_initial + cesar_initial
  let daniel_received := x + x + x
  daniel_received / total_initial = 1 / 4 := by sorry

end money_distribution_l3683_368366


namespace additional_amount_needed_l3683_368361

/-- The cost of the perfume --/
def perfume_cost : ℚ := 75

/-- The amount Christian saved --/
def christian_saved : ℚ := 5

/-- The amount Sue saved --/
def sue_saved : ℚ := 7

/-- The number of yards Christian mowed --/
def yards_mowed : ℕ := 6

/-- The price Christian charged per yard --/
def price_per_yard : ℚ := 6

/-- The number of dogs Sue walked --/
def dogs_walked : ℕ := 8

/-- The price Sue charged per dog --/
def price_per_dog : ℚ := 3

/-- The theorem stating the additional amount needed --/
theorem additional_amount_needed : 
  perfume_cost - (christian_saved + sue_saved + yards_mowed * price_per_yard + dogs_walked * price_per_dog) = 3 := by
  sorry

end additional_amount_needed_l3683_368361


namespace log_sum_property_l3683_368318

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h_log : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
sorry

end log_sum_property_l3683_368318


namespace problem_statement_l3683_368365

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 1 - x else Real.log x / Real.log 0.2

-- Theorem statement
theorem problem_statement (a : ℝ) (h : f (a + 5) = -1) : f a = 1 := by
  sorry

end problem_statement_l3683_368365


namespace tangent_angle_range_l3683_368338

open Real

noncomputable def curve (x : ℝ) : ℝ := 4 / (exp x + 1)

theorem tangent_angle_range :
  ∀ (x : ℝ), 
  let y := curve x
  let α := Real.arctan (deriv curve x)
  3 * π / 4 ≤ α ∧ α < π :=
by sorry

end tangent_angle_range_l3683_368338


namespace star_seven_three_l3683_368378

-- Define the binary operation *
def star (a b : ℝ) : ℝ := 5*a + 3*b - 2*a*b

-- Theorem statement
theorem star_seven_three : star 7 3 = 2 := by
  sorry

end star_seven_three_l3683_368378


namespace download_time_is_450_minutes_l3683_368367

-- Define the problem parameters
def min_speed : ℝ := 20
def max_speed : ℝ := 40
def avg_speed : ℝ := 30
def program_a_size : ℝ := 450
def program_b_size : ℝ := 240
def program_c_size : ℝ := 120
def mb_per_gb : ℝ := 1000
def seconds_per_minute : ℝ := 60

-- State the theorem
theorem download_time_is_450_minutes :
  let total_size := (program_a_size + program_b_size + program_c_size) * mb_per_gb
  let download_time_seconds := total_size / avg_speed
  let download_time_minutes := download_time_seconds / seconds_per_minute
  download_time_minutes = 450 := by
sorry

end download_time_is_450_minutes_l3683_368367


namespace ladder_length_l3683_368345

theorem ladder_length (angle : Real) (adjacent : Real) (hypotenuse : Real) : 
  angle = 60 * π / 180 →
  adjacent = 4.6 →
  Real.cos angle = adjacent / hypotenuse →
  hypotenuse = 9.2 := by
sorry

end ladder_length_l3683_368345


namespace identical_solutions_l3683_368312

theorem identical_solutions (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 ∧ y = 3*x + k) ↔ k = -9/4 := by
  sorry

end identical_solutions_l3683_368312


namespace total_pastries_count_l3683_368332

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := 13

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies

theorem total_pastries_count : total_pastries = 73 := by
  sorry

end total_pastries_count_l3683_368332


namespace seat_to_right_of_xiaofang_l3683_368383

/-- Represents a seat position as an ordered pair of integers -/
structure SeatPosition :=
  (column : ℤ)
  (row : ℤ)

/-- Returns the seat position to the right of a given seat -/
def seatToRight (seat : SeatPosition) : SeatPosition :=
  { column := seat.column + 1, row := seat.row }

/-- Xiaofang's seat position -/
def xiaofangSeat : SeatPosition := { column := 3, row := 5 }

theorem seat_to_right_of_xiaofang :
  seatToRight xiaofangSeat = { column := 4, row := 5 } := by sorry

end seat_to_right_of_xiaofang_l3683_368383


namespace apples_needed_proof_l3683_368329

/-- The number of additional apples Tessa needs to make a pie -/
def additional_apples_needed (initial : ℕ) (received : ℕ) (required : ℕ) : ℕ :=
  required - (initial + received)

/-- Theorem: Given Tessa's initial apples, apples received from Anita, and apples needed for a pie,
    the number of additional apples needed is equal to the apples required for a pie
    minus the sum of initial apples and received apples. -/
theorem apples_needed_proof (initial : ℕ) (received : ℕ) (required : ℕ)
    (h1 : initial = 4)
    (h2 : received = 5)
    (h3 : required = 10) :
  additional_apples_needed initial received required = 1 := by
  sorry

end apples_needed_proof_l3683_368329


namespace sinusoidal_amplitude_l3683_368320

/-- Given a sinusoidal function y = a * sin(b * x + c) + d that oscillates between 5 and -3,
    prove that the amplitude a is equal to 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) →
  a = 4 :=
by sorry

end sinusoidal_amplitude_l3683_368320


namespace inequality_proof_l3683_368334

/-- The function f(x) = |x - a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- The theorem to be proved -/
theorem inequality_proof (m n : ℝ) (h1 : m > 0) (h2 : n > 0)
  (h3 : Set.Icc 0 2 = {x : ℝ | f 1 x ≤ 1})
  (h4 : 1/m + 1/(2*n) = 1) :
  m + 4*n ≥ 2*Real.sqrt 2 + 3 := by
  sorry

end inequality_proof_l3683_368334


namespace imaginary_part_of_complex_number_l3683_368300

theorem imaginary_part_of_complex_number : 
  let z : ℂ := 1 / (2 + Complex.I)^2
  Complex.im z = -4/25 := by sorry

end imaginary_part_of_complex_number_l3683_368300


namespace max_value_of_expression_l3683_368304

/-- The set of digits to be used -/
def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The expression to be maximized -/
def expression (a b c d e f : ℕ) : ℚ :=
  a / b + c / d + e / f

/-- The theorem stating the maximum value of the expression -/
theorem max_value_of_expression :
  ∃ (a b c d e f : ℕ),
    a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧ f ∈ Digits ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    expression a b c d e f = 59 / 6 ∧
    ∀ (x y z w u v : ℕ),
      x ∈ Digits → y ∈ Digits → z ∈ Digits → w ∈ Digits → u ∈ Digits → v ∈ Digits →
      x ≠ y → x ≠ z → x ≠ w → x ≠ u → x ≠ v →
      y ≠ z → y ≠ w → y ≠ u → y ≠ v →
      z ≠ w → z ≠ u → z ≠ v →
      w ≠ u → w ≠ v →
      u ≠ v →
      expression x y z w u v ≤ 59 / 6 :=
by
  sorry

end max_value_of_expression_l3683_368304


namespace smallest_number_with_given_remainders_l3683_368330

theorem smallest_number_with_given_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 13 = 11 ∧ 
  n % 17 = 9 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 13 = 11 ∧ m % 17 = 9 → m ≥ n :=
by sorry

end smallest_number_with_given_remainders_l3683_368330


namespace train_length_l3683_368341

/-- The length of a train given its speed, the speed of a man walking in the same direction,
    and the time it takes for the train to cross the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) :
  train_speed = 63 →
  man_speed = 3 →
  crossing_time = 71.99424046076314 →
  (train_speed - man_speed) * (5 / 18) * crossing_time = 1199.9040076793857 := by
  sorry

end train_length_l3683_368341


namespace isosceles_triangle_angle_measure_l3683_368339

theorem isosceles_triangle_angle_measure :
  ∀ (D E F : ℝ),
  D = E →  -- Isosceles triangle condition
  F = 2 * D - 40 →  -- Relationship between F and D
  D + E + F = 180 →  -- Sum of angles in a triangle
  F = 70 := by
sorry

end isosceles_triangle_angle_measure_l3683_368339


namespace calculation_proof_l3683_368311

theorem calculation_proof : (2014 * 2014 + 2012) - 2013 * 2013 = 6039 := by
  sorry

end calculation_proof_l3683_368311


namespace rectangular_box_volume_l3683_368308

theorem rectangular_box_volume (l w h : ℝ) (area1 area2 area3 : ℝ) : 
  l > 0 → w > 0 → h > 0 →
  area1 = l * w →
  area2 = w * h →
  area3 = l * h →
  area1 = 30 →
  area2 = 18 →
  area3 = 10 →
  l * w * h = 90 := by
sorry

end rectangular_box_volume_l3683_368308


namespace even_number_decomposition_theorem_l3683_368393

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m ∧ n > 0

def even_number_decomposition (k : ℤ) : Prop :=
  (∃ a b : ℤ, 2 * k = a + b ∧ is_perfect_square (a * b)) ∨
  (∃ c d : ℤ, 2 * k = c - d ∧ is_perfect_square (c * d))

theorem even_number_decomposition_theorem :
  ∃ S : Set ℤ, S.Finite ∧ ∀ k : ℤ, k ∉ S → even_number_decomposition k :=
sorry

end even_number_decomposition_theorem_l3683_368393


namespace y_influenced_by_other_factors_other_factors_lead_to_random_errors_l3683_368340

/-- Linear regression model -/
structure LinearRegressionModel where
  y : ℝ → ℝ  -- Dependent variable
  x : ℝ      -- Independent variable
  b : ℝ      -- Slope
  a : ℝ      -- Intercept
  e : ℝ      -- Random error

/-- Definition of the linear regression model equation -/
def model_equation (m : LinearRegressionModel) : ℝ → ℝ :=
  fun x => m.b * x + m.a + m.e

/-- Theorem stating that y is influenced by factors other than x -/
theorem y_influenced_by_other_factors (m : LinearRegressionModel) :
  ∃ (factor : ℝ), factor ≠ m.x ∧ m.y m.x ≠ m.b * m.x + m.a :=
sorry

/-- Theorem stating that other factors can lead to random errors -/
theorem other_factors_lead_to_random_errors (m : LinearRegressionModel) :
  ∃ (factor : ℝ), factor ≠ m.x ∧ m.e ≠ 0 :=
sorry

end y_influenced_by_other_factors_other_factors_lead_to_random_errors_l3683_368340


namespace complement_of_120_degrees_l3683_368395

-- Define the angle in degrees
def given_angle : ℝ := 120

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem complement_of_120_degrees :
  complement given_angle = 60 := by
  sorry

end complement_of_120_degrees_l3683_368395


namespace max_distance_complex_l3683_368343

theorem max_distance_complex (w : ℂ) (h : Complex.abs w = 3) :
  ∃ (max_dist : ℝ), max_dist = 9 * Real.sqrt 61 + 81 ∧
    ∀ (z : ℂ), Complex.abs z = 3 → Complex.abs ((6 + 5*Complex.I)*z^2 - z^4) ≤ max_dist :=
by sorry

end max_distance_complex_l3683_368343


namespace sin_cos_relation_l3683_368314

theorem sin_cos_relation (α : ℝ) : 
  2 * Real.sin (α - π/3) = (2 - Real.sqrt 3) * Real.cos α → 
  Real.sin (2*α) + 3 * (Real.cos α)^2 = 7/5 := by
  sorry

end sin_cos_relation_l3683_368314


namespace seven_eighths_of_48_l3683_368305

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end seven_eighths_of_48_l3683_368305


namespace expression_takes_many_values_l3683_368382

theorem expression_takes_many_values :
  ∀ (x : ℝ), x ≠ -2 → x ≠ 3 →
  ∃ (y : ℝ), y ≠ x ∧
    (3 + 6 / (3 - x)) ≠ (3 + 6 / (3 - y)) :=
by sorry

end expression_takes_many_values_l3683_368382


namespace initial_number_proof_l3683_368371

theorem initial_number_proof : ∃ x : ℝ, (x / 34) * 15 + 270 = 405 ∧ x = 306 := by
  sorry

end initial_number_proof_l3683_368371


namespace kennedy_car_drive_l3683_368351

theorem kennedy_car_drive (miles_per_gallon : ℝ) (initial_gas : ℝ) 
  (to_school : ℝ) (to_softball : ℝ) (to_restaurant : ℝ) (to_home : ℝ) 
  (to_friend : ℝ) : 
  miles_per_gallon = 19 →
  initial_gas = 2 →
  to_school = 15 →
  to_softball = 6 →
  to_restaurant = 2 →
  to_home = 11 →
  miles_per_gallon * initial_gas = to_school + to_softball + to_restaurant + to_friend + to_home →
  to_friend = 4 := by sorry

end kennedy_car_drive_l3683_368351
