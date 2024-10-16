import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_product_equality_l1780_178024

theorem sqrt_product_equality : Real.sqrt 6 * Real.sqrt 2 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l1780_178024


namespace NUMINAMATH_CALUDE_problem_solution_l1780_178058

theorem problem_solution (x y a : ℝ) 
  (h1 : |x + 1| + (y + 2)^2 = 0)
  (h2 : a * x - 3 * a * y = 1) : 
  a = 0.2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1780_178058


namespace NUMINAMATH_CALUDE_sloth_shoe_pairs_needed_l1780_178099

/-- Represents the number of feet a sloth has -/
def sloth_feet : ℕ := 3

/-- Represents the number of shoes in a complete set for the sloth -/
def shoes_per_set : ℕ := 3

/-- Represents the number of shoes in a pair -/
def shoes_per_pair : ℕ := 2

/-- Represents the number of complete sets the sloth already owns -/
def owned_sets : ℕ := 1

/-- Represents the total number of complete sets the sloth needs -/
def total_sets_needed : ℕ := 5

/-- Theorem stating the number of pairs of shoes the sloth needs to buy -/
theorem sloth_shoe_pairs_needed :
  (total_sets_needed - owned_sets) * shoes_per_set / shoes_per_pair = 6 := by
  sorry

end NUMINAMATH_CALUDE_sloth_shoe_pairs_needed_l1780_178099


namespace NUMINAMATH_CALUDE_trig_simplification_l1780_178034

theorem trig_simplification :
  (Real.cos (20 * π / 180) * Real.sqrt (1 - Real.cos (40 * π / 180))) / Real.cos (50 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l1780_178034


namespace NUMINAMATH_CALUDE_xy_yz_zx_bounds_l1780_178081

theorem xy_yz_zx_bounds (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2 + 1) : 
  ∃ (N n : ℝ), (∀ t : ℝ, t = x*y + y*z + z*x → t ≤ N ∧ n ≤ t) ∧ 11 < N + 6*n ∧ N + 6*n < 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_yz_zx_bounds_l1780_178081


namespace NUMINAMATH_CALUDE_multiple_is_two_l1780_178018

/-- The multiple of Period 2 students compared to Period 1 students -/
def multiple_of_period2 (period1_students period2_students : ℕ) : ℚ :=
  (period1_students + 5) / period2_students

theorem multiple_is_two :
  let period1_students : ℕ := 11
  let period2_students : ℕ := 8
  multiple_of_period2 period1_students period2_students = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_is_two_l1780_178018


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l1780_178037

theorem hyperbola_k_range (k : ℝ) :
  (∃ x y : ℝ, x^2 / (1 + k) - y^2 / (1 - k) = 1) →
  -1 < k ∧ k < 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l1780_178037


namespace NUMINAMATH_CALUDE_percentage_equation_solution_l1780_178000

/-- The solution to the equation (47% of 1442 - x% of 1412) + 63 = 252 is approximately 34.63% -/
theorem percentage_equation_solution : 
  ∃ x : ℝ, abs (x - 34.63) < 0.01 ∧ 
  ((47 / 100) * 1442 - (x / 100) * 1412) + 63 = 252 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_solution_l1780_178000


namespace NUMINAMATH_CALUDE_taxi_fare_for_8_2km_l1780_178013

/-- Calculates the taxi fare for a given distance -/
def taxiFare (distance : Float) : Float :=
  let baseFare := 6
  let midRateDistance := 4
  let midRate := 1
  let highRate := 0.8
  let baseDistance := 3
  let midDistanceEnd := 7
  if distance ≤ baseDistance then
    baseFare
  else if distance ≤ midDistanceEnd then
    baseFare + midRate * (Float.ceil (distance - baseDistance))
  else
    baseFare + midRate * midRateDistance + highRate * (Float.ceil (distance - midDistanceEnd))

theorem taxi_fare_for_8_2km :
  taxiFare 8.2 = 11.6 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_for_8_2km_l1780_178013


namespace NUMINAMATH_CALUDE_cats_left_after_sale_l1780_178015

/-- Calculates the number of cats left after a sale --/
theorem cats_left_after_sale (siamese house persian maine_coon : ℕ)
  (siamese_sold house_sold persian_sold maine_coon_sold : ℚ)
  (h_siamese : siamese = 38)
  (h_house : house = 25)
  (h_persian : persian = 15)
  (h_maine_coon : maine_coon = 12)
  (h_siamese_sold : siamese_sold = 60 / 100)
  (h_house_sold : house_sold = 40 / 100)
  (h_persian_sold : persian_sold = 75 / 100)
  (h_maine_coon_sold : maine_coon_sold = 50 / 100) :
  ⌊siamese - siamese * siamese_sold⌋ +
  ⌊house - house * house_sold⌋ +
  ⌊persian - persian * persian_sold⌋ +
  ⌊maine_coon - maine_coon * maine_coon_sold⌋ = 41 := by
  sorry


end NUMINAMATH_CALUDE_cats_left_after_sale_l1780_178015


namespace NUMINAMATH_CALUDE_investment_proof_l1780_178054

/-- Compound interest function -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

theorem investment_proof : 
  let principal : ℝ := 1000
  let rate : ℝ := 0.06
  let time : ℕ := 8
  let final_balance : ℝ := 1593.85
  compound_interest principal rate time = final_balance := by
sorry

end NUMINAMATH_CALUDE_investment_proof_l1780_178054


namespace NUMINAMATH_CALUDE_sum_of_nine_and_number_l1780_178076

theorem sum_of_nine_and_number (x : ℝ) : 
  (9 - x = 1) → (x < 10) → (9 + x = 17) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_nine_and_number_l1780_178076


namespace NUMINAMATH_CALUDE_fractional_parts_inequality_l1780_178080

theorem fractional_parts_inequality (q : ℕ+) (hq : ¬ ∃ (m : ℕ), q = m^3) :
  ∃ (c : ℝ), c > 0 ∧
  ∀ (n : ℕ+), 
    (n : ℝ) * q.val ^ (1/3 : ℝ) - ⌊(n : ℝ) * q.val ^ (1/3 : ℝ)⌋ +
    (n : ℝ) * q.val ^ (2/3 : ℝ) - ⌊(n : ℝ) * q.val ^ (2/3 : ℝ)⌋ ≥
    c * (n : ℝ) ^ (-1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_fractional_parts_inequality_l1780_178080


namespace NUMINAMATH_CALUDE_points_in_segment_l1780_178011

theorem points_in_segment (n : ℕ) : 
  1 < (n^4 + n^2 + 2) / (n^4 + n^2 + 1) ∧ (n^4 + n^2 + 2) / (n^4 + n^2 + 1) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_points_in_segment_l1780_178011


namespace NUMINAMATH_CALUDE_inverse_sum_lower_bound_l1780_178074

theorem inverse_sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (hab_sum : a + b = 1) :
  1 / a + 1 / b > 4 := by
sorry

end NUMINAMATH_CALUDE_inverse_sum_lower_bound_l1780_178074


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l1780_178068

/-- Represents a quadratic function of the form y = a(x-m)(x-m-k) -/
def quadratic_function (a m k x : ℝ) : ℝ := a * (x - m) * (x - m - k)

/-- The minimum value of the quadratic function when k = 2 -/
def min_value (a m : ℝ) : ℝ := -a

theorem quadratic_minimum_value (a m : ℝ) (h : a > 0) :
  ∃ x, quadratic_function a m 2 x = min_value a m ∧
  ∀ y, quadratic_function a m 2 y ≥ min_value a m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l1780_178068


namespace NUMINAMATH_CALUDE_circle_radius_with_inscribed_square_l1780_178063

/-- Given a circle with a chord of length 6 and an inscribed square of side length 2 in the segment
    corresponding to the chord, prove that the radius of the circle is √10. -/
theorem circle_radius_with_inscribed_square (r : ℝ) 
  (h1 : ∃ (chord : ℝ), chord = 6 ∧ chord ≤ 2 * r)
  (h2 : ∃ (square_side : ℝ), square_side = 2 ∧ 
        square_side ≤ (r + r - chord) ∧ 
        square_side * square_side ≤ chord * (2 * r - chord)) :
  r = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_inscribed_square_l1780_178063


namespace NUMINAMATH_CALUDE_point_difference_on_plane_l1780_178040

/-- Given two points on a plane, prove that the difference in their x and z coordinates are 3 and 0 respectively. -/
theorem point_difference_on_plane (m n z p q : ℝ) (k : ℝ) (hk : k ≠ 0) :
  (m = n / 6 - 2 / 5 + z / k) →
  (m + p = (n + 18) / 6 - 2 / 5 + (z + q) / k) →
  p = 3 ∧ q = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_difference_on_plane_l1780_178040


namespace NUMINAMATH_CALUDE_squared_one_necessary_not_sufficient_l1780_178051

theorem squared_one_necessary_not_sufficient (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) := by sorry

end NUMINAMATH_CALUDE_squared_one_necessary_not_sufficient_l1780_178051


namespace NUMINAMATH_CALUDE_calculation_proofs_l1780_178029

theorem calculation_proofs :
  (4800 / 125 = 38.4) ∧ (13 * 74 + 27 * 13 - 13 = 1300) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proofs_l1780_178029


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1780_178071

/-- The number of schools --/
def num_schools : ℕ := 4

/-- The number of members per school --/
def members_per_school : ℕ := 5

/-- The number of representatives from the host school --/
def host_representatives : ℕ := 3

/-- The number of representatives from each non-host school --/
def non_host_representatives : ℕ := 1

/-- The total number of ways to arrange a presidency meeting --/
def meeting_arrangements : ℕ := num_schools * (Nat.choose members_per_school host_representatives) * (Nat.choose members_per_school non_host_representatives)^(num_schools - 1)

theorem presidency_meeting_arrangements :
  meeting_arrangements = 5000 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l1780_178071


namespace NUMINAMATH_CALUDE_inequality_proof_l1780_178052

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 0) :
  (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y + y :=
by sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l1780_178052


namespace NUMINAMATH_CALUDE_bullet_hole_displacement_l1780_178033

/-- The displacement of the second hole relative to the first hole when a bullet is fired perpendicular to a moving train -/
theorem bullet_hole_displacement 
  (c : Real) -- speed of the train in km/h
  (c_prime : Real) -- speed of the bullet in m/s
  (a : Real) -- width of the train car in meters
  (h1 : c = 60) -- train speed is 60 km/h
  (h2 : c_prime = 40) -- bullet speed is 40 m/s
  (h3 : a = 4) -- train car width is 4 meters
  : (a * c * 1000 / 3600) / c_prime = 1.667 := by sorry

end NUMINAMATH_CALUDE_bullet_hole_displacement_l1780_178033


namespace NUMINAMATH_CALUDE_sector_tangent_problem_l1780_178070

theorem sector_tangent_problem (θ φ : Real) (h1 : 0 < θ) (h2 : θ < 2 * Real.pi) : 
  (1/2 * θ * 4^2 = 2 * Real.pi) → (Real.tan (θ + φ) = 3) → Real.tan φ = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sector_tangent_problem_l1780_178070


namespace NUMINAMATH_CALUDE_savings_percentage_l1780_178008

/-- Represents a person's financial situation over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- Calculates the expenditure for a given year --/
def expenditure (income : ℝ) (savings : ℝ) : ℝ :=
  income - savings

/-- Theorem stating the conditions and the result to be proved --/
theorem savings_percentage (f : FinancialSituation) 
  (h1 : f.income_year2 = 1.2 * f.income_year1)
  (h2 : f.savings_year2 = 2 * f.savings_year1)
  (h3 : expenditure f.income_year1 f.savings_year1 + 
        expenditure f.income_year2 f.savings_year2 = 
        2 * expenditure f.income_year1 f.savings_year1) :
  f.savings_year1 / f.income_year1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_savings_percentage_l1780_178008


namespace NUMINAMATH_CALUDE_sphere_volume_in_cube_l1780_178088

/-- The volume of a sphere inscribed in a cube with surface area 6 cm² is (1/6)π cm³ -/
theorem sphere_volume_in_cube (cube_surface_area : ℝ) (sphere_volume : ℝ) :
  cube_surface_area = 6 →
  sphere_volume = (1 / 6) * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_in_cube_l1780_178088


namespace NUMINAMATH_CALUDE_parabola_equation_and_vertex_l1780_178006

/-- A parabola passing through points (1, 0) and (3, 0) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ b c : ℝ, y = -x^2 + b*x + c ∧ 0 = -1 + b + c ∧ 0 = -9 + 3*b + c

theorem parabola_equation_and_vertex :
  (∀ x y : ℝ, Parabola x y ↔ y = -x^2 + 4*x - 3) ∧
  (∃ x y : ℝ, Parabola x y ∧ x = 2 ∧ y = 1 ∧
    ∀ x' y' : ℝ, Parabola x' y' → y' ≤ y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_and_vertex_l1780_178006


namespace NUMINAMATH_CALUDE_ellipse_max_value_l1780_178021

/-- The maximum value of x + 2y for points on the ellipse x^2/16 + y^2/12 = 1 is 8 -/
theorem ellipse_max_value (x y : ℝ) : 
  x^2/16 + y^2/12 = 1 → x + 2*y ≤ 8 := by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l1780_178021


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1780_178047

/-- 
Given an arithmetic sequence where:
- The first term is 2/3
- The second term is 1
- The third term is 4/3

Prove that the eighth term of this sequence is 3.
-/
theorem arithmetic_sequence_eighth_term : 
  ∀ (a : ℕ → ℚ), 
    (a 1 = 2/3) →
    (a 2 = 1) →
    (a 3 = 4/3) →
    (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
    a 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l1780_178047


namespace NUMINAMATH_CALUDE_consecutive_sets_summing_to_150_l1780_178094

/-- A structure representing a set of consecutive integers -/
structure ConsecutiveSet where
  start : ℕ
  length : ℕ
  sum_is_150 : start * length + (length * (length - 1)) / 2 = 150
  at_least_two : length ≥ 2

/-- The theorem stating that there are exactly 3 sets of consecutive positive integers summing to 150 -/
theorem consecutive_sets_summing_to_150 : 
  ∃! (sets : Finset ConsecutiveSet), sets.card = 3 ∧ 
    (∀ s ∈ sets, s.start > 0 ∧ s.length ≥ 2 ∧ 
      s.start * s.length + (s.length * (s.length - 1)) / 2 = 150) ∧
    (∀ a b : ℕ, a > 0 → b ≥ 2 → 
      (a * b + (b * (b - 1)) / 2 = 150 → ∃ s ∈ sets, s.start = a ∧ s.length = b)) :=
sorry

end NUMINAMATH_CALUDE_consecutive_sets_summing_to_150_l1780_178094


namespace NUMINAMATH_CALUDE_consecutive_squares_divisible_by_five_l1780_178027

theorem consecutive_squares_divisible_by_five (n : ℤ) :
  ∃ k : ℤ, (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 5 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_divisible_by_five_l1780_178027


namespace NUMINAMATH_CALUDE_hockey_league_games_l1780_178089

/-- Represents the number of games played between two groups of teams -/
def games_between (n m : ℕ) (games_per_pair : ℕ) : ℕ := n * m * games_per_pair

/-- Represents the number of games played within a group of teams -/
def games_within (n : ℕ) (games_per_pair : ℕ) : ℕ := n * (n - 1) * games_per_pair / 2

/-- The total number of games played in the hockey league season -/
def total_games : ℕ :=
  let top5 := 5
  let mid5 := 5
  let bottom5 := 5
  let top_vs_top := games_within top5 12
  let top_vs_rest := games_between top5 (mid5 + bottom5) 8
  let mid_vs_mid := games_within mid5 10
  let mid_vs_bottom := games_between mid5 bottom5 6
  let bottom_vs_bottom := games_within bottom5 8
  top_vs_top + top_vs_rest + mid_vs_mid + mid_vs_bottom + bottom_vs_bottom

theorem hockey_league_games :
  total_games = 850 := by sorry

end NUMINAMATH_CALUDE_hockey_league_games_l1780_178089


namespace NUMINAMATH_CALUDE_feuerbach_theorem_l1780_178095

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The midpoint circle of a triangle -/
def midpointCircle (t : Triangle) : Circle := sorry

/-- The incircle of a triangle -/
def incircle (t : Triangle) : Circle := sorry

/-- The excircles of a triangle -/
def excircles (t : Triangle) : Fin 3 → Circle := sorry

/-- Two circles are tangent -/
def areTangent (c1 c2 : Circle) : Prop := sorry

/-- Feuerbach's theorem -/
theorem feuerbach_theorem (t : Triangle) : 
  (areTangent (midpointCircle t) (incircle t)) ∧ 
  (∀ i : Fin 3, areTangent (midpointCircle t) (excircles t i)) := by
  sorry

end NUMINAMATH_CALUDE_feuerbach_theorem_l1780_178095


namespace NUMINAMATH_CALUDE_ninety_percent_greater_than_thirty_by_twelve_l1780_178031

theorem ninety_percent_greater_than_thirty_by_twelve (x : ℝ) : 
  0.9 * x > 0.8 * 30 + 12 → x > 40 := by
  sorry

end NUMINAMATH_CALUDE_ninety_percent_greater_than_thirty_by_twelve_l1780_178031


namespace NUMINAMATH_CALUDE_total_stars_l1780_178020

theorem total_stars (num_students : ℕ) (stars_per_student : ℕ) 
  (h1 : num_students = 186) 
  (h2 : stars_per_student = 5) : 
  num_students * stars_per_student = 930 := by
  sorry

end NUMINAMATH_CALUDE_total_stars_l1780_178020


namespace NUMINAMATH_CALUDE_park_outer_diameter_l1780_178096

/-- Represents the dimensions of a circular park with concentric areas. -/
structure ParkDimensions where
  pond_diameter : ℝ
  seating_width : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park. -/
def outer_diameter (park : ParkDimensions) : ℝ :=
  park.pond_diameter + 2 * (park.seating_width + park.garden_width + park.path_width)

/-- Theorem stating that for a park with given dimensions, the outer diameter is 64 feet. -/
theorem park_outer_diameter :
  let park := ParkDimensions.mk 20 4 10 8
  outer_diameter park = 64 := by
  sorry


end NUMINAMATH_CALUDE_park_outer_diameter_l1780_178096


namespace NUMINAMATH_CALUDE_equation_solution_l1780_178066

theorem equation_solution (a b : ℝ) (h : a ≠ -1) :
  let x := (a^2 - b^2 + 2*a - 2*b) / (2*(a+1))
  x^2 + (b+1)^2 = (a+1 - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1780_178066


namespace NUMINAMATH_CALUDE_total_turnips_l1780_178060

theorem total_turnips (keith_turnips alyssa_turnips : ℕ) 
  (h1 : keith_turnips = 6) 
  (h2 : alyssa_turnips = 9) : 
  keith_turnips + alyssa_turnips = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_l1780_178060


namespace NUMINAMATH_CALUDE_cabinet_area_l1780_178075

theorem cabinet_area : 
  ∀ (width length area : ℝ),
  width = 1.2 →
  length = 1.8 →
  area = width * length →
  area = 2.16 := by
sorry

end NUMINAMATH_CALUDE_cabinet_area_l1780_178075


namespace NUMINAMATH_CALUDE_age_ratio_l1780_178032

def age_problem (a b c : ℕ) : Prop :=
  (a = b + 2) ∧ (a + b + c = 27) ∧ (b = 10)

theorem age_ratio (a b c : ℕ) (h : age_problem a b c) :
  b = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l1780_178032


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l1780_178030

theorem roots_of_quadratic (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (∀ x : ℝ, x^2 - (a + b + c) * x + (a * b + b * c + c * a) = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l1780_178030


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l1780_178045

/-- Proves that mixing 8 liters of 25% alcohol solution with 2 liters of 12% alcohol solution results in a 22.4% alcohol solution -/
theorem alcohol_mixture_percentage :
  let volume1 : ℝ := 8
  let concentration1 : ℝ := 0.25
  let volume2 : ℝ := 2
  let concentration2 : ℝ := 0.12
  let total_volume : ℝ := volume1 + volume2
  let total_alcohol : ℝ := volume1 * concentration1 + volume2 * concentration2
  total_alcohol / total_volume = 0.224 := by
sorry


end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l1780_178045


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1780_178012

/-- Two vectors a and b in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, perpendicular (x, 2) (1, -1) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l1780_178012


namespace NUMINAMATH_CALUDE_library_book_count_l1780_178007

/-- The number of shelves in the library -/
def num_shelves : ℕ := 14240

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 8

/-- The total number of books in the library -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem library_book_count : total_books = 113920 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l1780_178007


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l1780_178004

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-2) 3

-- Define the new function g
def g (x : ℝ) : ℝ := f (2 * x - 1)

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc (-1/2) 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l1780_178004


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1780_178059

theorem inequality_solution_set (t : ℝ) (h : 0 < t ∧ t < 1) :
  {x : ℝ | (t - x) * (x - 1/t) > 0} = {x : ℝ | t < x ∧ x < 1/t} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1780_178059


namespace NUMINAMATH_CALUDE_equation_solution_l1780_178043

theorem equation_solution (k x m n : ℝ) :
  (∃ x, ∀ k, 2 * k * x + 2 * m = 6 - 2 * x + n * k) →
  4 * m + 2 * n = 12 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1780_178043


namespace NUMINAMATH_CALUDE_total_work_seconds_l1780_178044

/-- The total number of seconds worked by four people given their work hours relationships -/
theorem total_work_seconds 
  (bianca_hours : ℝ) 
  (h1 : bianca_hours = 12.5)
  (h2 : ∃ celeste_hours : ℝ, celeste_hours = 2 * bianca_hours)
  (h3 : ∃ mcclain_hours : ℝ, mcclain_hours = celeste_hours - 8.5)
  (h4 : ∃ omar_hours : ℝ, omar_hours = bianca_hours + 3)
  : ∃ total_seconds : ℝ, total_seconds = 250200 := by
  sorry


end NUMINAMATH_CALUDE_total_work_seconds_l1780_178044


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1780_178039

def solution_set : Set ℝ := {x | x < -1 ∨ (-1 < x ∧ x < 0) ∨ x > 2}

def inequality (x : ℝ) : Prop := x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1)

theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1780_178039


namespace NUMINAMATH_CALUDE_students_over_capacity_l1780_178041

/-- Calculates the number of students over capacity given the initial conditions --/
theorem students_over_capacity
  (ratio : ℚ)
  (teachers : ℕ)
  (increase_percent : ℚ)
  (capacity : ℕ)
  (h_ratio : ratio = 27.5)
  (h_teachers : teachers = 42)
  (h_increase : increase_percent = 0.15)
  (h_capacity : capacity = 1300) :
  ⌊(ratio * teachers) * (1 + increase_percent)⌋ - capacity = 28 :=
by sorry

end NUMINAMATH_CALUDE_students_over_capacity_l1780_178041


namespace NUMINAMATH_CALUDE_barbier_theorem_for_delta_curves_l1780_178083

-- Define a Δ-curve
class DeltaCurve where
  height : ℝ
  is_convex : Bool
  can_rotate_in_triangle : Bool
  always_touches_sides : Bool

-- Define the length of a Δ-curve
def length_of_delta_curve (K : DeltaCurve) : ℝ := sorry

-- Define the approximation of a Δ-curve by circular arcs
def approximate_by_circular_arcs (K : DeltaCurve) (n : ℕ) : DeltaCurve := sorry

-- Theorem: The length of any Δ-curve with height h is 2πh/3
theorem barbier_theorem_for_delta_curves (K : DeltaCurve) :
  length_of_delta_curve K = 2 * Real.pi * K.height / 3 := by sorry

end NUMINAMATH_CALUDE_barbier_theorem_for_delta_curves_l1780_178083


namespace NUMINAMATH_CALUDE_floor_sqrt_23_squared_l1780_178010

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_23_squared_l1780_178010


namespace NUMINAMATH_CALUDE_solve_candy_problem_l1780_178050

def candy_problem (megan_candy : ℕ) (mary_multiplier : ℕ) (mary_additional : ℕ) : Prop :=
  let mary_initial := mary_multiplier * megan_candy
  let mary_total := mary_initial + mary_additional
  mary_total = 25

theorem solve_candy_problem : 
  candy_problem 5 3 10 := by sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l1780_178050


namespace NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l1780_178061

theorem average_weight_of_class (group1_count : ℕ) (group1_avg : ℚ) 
                                (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count

theorem class_average_weight :
  average_weight_of_class 24 (50.25 : ℚ) 8 (45.15 : ℚ) = 49 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_of_class_class_average_weight_l1780_178061


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l1780_178057

theorem diophantine_equation_solutions :
  ∀ x y z : ℕ+,
    x > y ∧ y > z →
    (1 : ℚ) / x + 2 / y + 3 / z = 1 →
    ((x = 36 ∧ y = 9 ∧ z = 4) ∨
     (x = 20 ∧ y = 10 ∧ z = 4) ∨
     (x = 15 ∧ y = 6 ∧ z = 5)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l1780_178057


namespace NUMINAMATH_CALUDE_crayon_selection_ways_l1780_178036

/-- The number of ways to choose k items from n items, where order doesn't matter -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of crayons in the box -/
def total_crayons : ℕ := 15

/-- The number of crayons to be selected -/
def selected_crayons : ℕ := 5

theorem crayon_selection_ways : 
  choose total_crayons selected_crayons = 3003 := by sorry

end NUMINAMATH_CALUDE_crayon_selection_ways_l1780_178036


namespace NUMINAMATH_CALUDE_extra_bananas_per_child_l1780_178072

/-- Given the total number of children, number of absent children, and original banana allocation,
    calculate the number of extra bananas each present child received. -/
theorem extra_bananas_per_child 
  (total_children : ℕ) 
  (absent_children : ℕ) 
  (original_allocation : ℕ) 
  (h1 : total_children = 780)
  (h2 : absent_children = 390)
  (h3 : original_allocation = 2)
  (h4 : absent_children < total_children) :
  (total_children * original_allocation) / (total_children - absent_children) - original_allocation = 2 :=
by sorry

end NUMINAMATH_CALUDE_extra_bananas_per_child_l1780_178072


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l1780_178079

def number_of_packages : ℕ := 5
def number_of_houses : ℕ := 5

theorem correct_delivery_probability :
  let total_arrangements := number_of_packages.factorial
  let correct_three_arrangements := (number_of_packages.choose 3) * 1 * 1
  (correct_three_arrangements : ℚ) / total_arrangements = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l1780_178079


namespace NUMINAMATH_CALUDE_movie_admission_price_l1780_178077

theorem movie_admission_price (regular_price : ℝ) : 
  (∀ discounted_price : ℝ, 
    discounted_price = regular_price - 3 →
    6 * discounted_price = 30) →
  regular_price = 8 := by
sorry

end NUMINAMATH_CALUDE_movie_admission_price_l1780_178077


namespace NUMINAMATH_CALUDE_agricultural_machinery_growth_rate_l1780_178069

/-- The average growth rate for May and June in an agricultural machinery factory --/
theorem agricultural_machinery_growth_rate :
  ∀ (april_production : ℕ) (total_production : ℕ) (growth_rate : ℝ),
  april_production = 500 →
  total_production = 1820 →
  april_production + 
    april_production * (1 + growth_rate) + 
    april_production * (1 + growth_rate)^2 = total_production →
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_agricultural_machinery_growth_rate_l1780_178069


namespace NUMINAMATH_CALUDE_nickel_count_l1780_178090

/-- Proves that given $4 in quarters, dimes, and nickels, with 10 quarters and 12 dimes, the number of nickels is 6. -/
theorem nickel_count (total : ℚ) (quarters dimes : ℕ) : 
  total = 4 → 
  quarters = 10 → 
  dimes = 12 → 
  ∃ (nickels : ℕ), 
    total = (0.25 * quarters + 0.1 * dimes + 0.05 * nickels) ∧ 
    nickels = 6 := by sorry

end NUMINAMATH_CALUDE_nickel_count_l1780_178090


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1780_178048

theorem cube_root_equation_solution :
  ∀ x : ℝ, (7 - 3 / (3 + x))^(1/3) = -2 → x = -14/5 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1780_178048


namespace NUMINAMATH_CALUDE_wall_width_proof_l1780_178067

/-- Proves that the width of a wall is 2 meters given specific brick and wall dimensions --/
theorem wall_width_proof (brick_length : Real) (brick_width : Real) (brick_height : Real)
  (wall_length : Real) (wall_height : Real) (num_bricks : Nat) :
  brick_length = 0.2 →
  brick_width = 0.1 →
  brick_height = 0.075 →
  wall_length = 27 →
  wall_height = 0.75 →
  num_bricks = 27000 →
  ∃ (wall_width : Real), wall_width = 2 ∧
    brick_length * brick_width * brick_height * num_bricks =
    wall_length * wall_width * wall_height := by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_l1780_178067


namespace NUMINAMATH_CALUDE_set_A_properties_l1780_178086

def A : Set ℝ := {x | x^2 - 4 = 0}

theorem set_A_properties :
  (A = {-2, 2}) ∧ (2 ∈ A) ∧ (-2 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_set_A_properties_l1780_178086


namespace NUMINAMATH_CALUDE_polygon_arrangement_exists_l1780_178009

/-- A polygon constructed from squares and equilateral triangles -/
structure PolygonArrangement where
  squares : ℕ
  triangles : ℕ
  side_length : ℝ
  perimeter : ℝ

/-- The existence of a polygon arrangement with the given properties -/
theorem polygon_arrangement_exists : ∃ (p : PolygonArrangement), 
  p.squares = 9 ∧ 
  p.triangles = 19 ∧ 
  p.side_length = 1 ∧ 
  p.perimeter = 15 := by
  sorry

end NUMINAMATH_CALUDE_polygon_arrangement_exists_l1780_178009


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_plus_c_l1780_178005

theorem min_value_a_plus_b_plus_c (a b c : ℝ) 
  (h1 : a^2 + b^2 ≤ c) (h2 : c ≤ 1) : 
  ∀ x y z : ℝ, x^2 + y^2 ≤ z ∧ z ≤ 1 → a + b + c ≤ x + y + z ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀^2 + b₀^2 ≤ c₀ ∧ c₀ ≤ 1 ∧ a₀ + b₀ + c₀ = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_plus_c_l1780_178005


namespace NUMINAMATH_CALUDE_max_min_s_values_l1780_178055

theorem max_min_s_values (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  let s := x^2 + y^2
  (∀ a b : ℝ, 4 * a^2 - 5 * a * b + 4 * b^2 = 5 → a^2 + b^2 ≤ 10/3) ∧
  (∃ c d : ℝ, 4 * c^2 - 5 * c * d + 4 * d^2 = 5 ∧ c^2 + d^2 = 10/3) ∧
  (∀ a b : ℝ, 4 * a^2 - 5 * a * b + 4 * b^2 = 5 → a^2 + b^2 ≥ 10/13) ∧
  (∃ e f : ℝ, 4 * e^2 - 5 * e * f + 4 * f^2 = 5 ∧ e^2 + f^2 = 10/13) :=
by sorry

end NUMINAMATH_CALUDE_max_min_s_values_l1780_178055


namespace NUMINAMATH_CALUDE_village_birth_probability_l1780_178025

/-- Represents the gender of a child -/
inductive Gender
| Boy
| Girl

/-- A village with a custom of having children until a boy is born -/
structure Village where
  /-- The probability of having a boy in a single birth -/
  prob_boy : ℝ
  /-- The probability of having a girl in a single birth -/
  prob_girl : ℝ
  /-- The proportion of boys to girls in the village after some time -/
  boy_girl_ratio : ℝ
  /-- The probabilities sum to 1 -/
  prob_sum_one : prob_boy + prob_girl = 1
  /-- The proportion of boys to girls is 1:1 -/
  equal_ratio : boy_girl_ratio = 1

/-- Theorem: In a village with the given custom, the probability of having a boy or a girl is 1/2 -/
theorem village_birth_probability (v : Village) : v.prob_boy = 1/2 ∧ v.prob_girl = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_village_birth_probability_l1780_178025


namespace NUMINAMATH_CALUDE_milk_processing_profit_comparison_l1780_178087

/-- Represents the profit calculation for a milk processing factory --/
theorem milk_processing_profit_comparison :
  let total_milk : ℝ := 9
  let fresh_milk_profit : ℝ := 500
  let yogurt_profit : ℝ := 1200
  let milk_slice_profit : ℝ := 2000
  let yogurt_capacity : ℝ := 3
  let milk_slice_capacity : ℝ := 1
  let processing_days : ℝ := 4

  let plan1_profit := milk_slice_capacity * processing_days * milk_slice_profit + 
                      (total_milk - milk_slice_capacity * processing_days) * fresh_milk_profit

  let plan2_milk_slice : ℝ := 1.5
  let plan2_yogurt : ℝ := 7.5
  let plan2_profit := plan2_milk_slice * milk_slice_profit + plan2_yogurt * yogurt_profit

  plan2_profit > plan1_profit ∧ 
  plan2_milk_slice + plan2_yogurt = total_milk ∧
  plan2_milk_slice / milk_slice_capacity + plan2_yogurt / yogurt_capacity = processing_days :=
by sorry

end NUMINAMATH_CALUDE_milk_processing_profit_comparison_l1780_178087


namespace NUMINAMATH_CALUDE_h_of_3_eq_3_l1780_178026

/-- The function h(x) is defined implicitly by this equation -/
def h_equation (x : ℝ) (h : ℝ → ℝ) : Prop :=
  (x^(2^2007 - 1) - 1) * h x = (x + 1) * (x^2 + 1) * (x^4 + 1) * (x^(2^2006) + 1) - 1

/-- The theorem states that h(3) = 3 for the function h defined by h_equation -/
theorem h_of_3_eq_3 :
  ∃ h : ℝ → ℝ, h_equation 3 h ∧ h 3 = 3 := by sorry

end NUMINAMATH_CALUDE_h_of_3_eq_3_l1780_178026


namespace NUMINAMATH_CALUDE_shortest_to_longest_diagonal_ratio_l1780_178093

/-- A regular octagon -/
structure RegularOctagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- The shortest diagonal of a regular octagon -/
def shortest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The longest diagonal of a regular octagon -/
def longest_diagonal (o : RegularOctagon) : ℝ :=
  sorry

/-- The ratio of the shortest diagonal to the longest diagonal in a regular octagon is 1/2 -/
theorem shortest_to_longest_diagonal_ratio (o : RegularOctagon) :
  shortest_diagonal o / longest_diagonal o = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_shortest_to_longest_diagonal_ratio_l1780_178093


namespace NUMINAMATH_CALUDE_inscribed_circle_circumference_l1780_178002

/-- Given a circle with radius R and an arc subtending 120°, 
    the radius r of the circle inscribed between this arc and its tangents 
    satisfies 2πr = (2πR)/3 -/
theorem inscribed_circle_circumference (R r : ℝ) : r = R / 3 → 2 * π * r = 2 * π * R / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_circumference_l1780_178002


namespace NUMINAMATH_CALUDE_divisibility_of_7386038_l1780_178082

theorem divisibility_of_7386038 : ∃ (k : ℕ), 7386038 = 7 * k := by sorry

end NUMINAMATH_CALUDE_divisibility_of_7386038_l1780_178082


namespace NUMINAMATH_CALUDE_expression_evaluation_l1780_178092

theorem expression_evaluation : -20 + 8 * (10 / 2) - 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1780_178092


namespace NUMINAMATH_CALUDE_inequality_solution_l1780_178042

theorem inequality_solution (x : ℝ) : 
  (x - 1)^2 < 12 - x ↔ (1 - 3 * Real.sqrt 5) / 2 < x ∧ x < (1 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1780_178042


namespace NUMINAMATH_CALUDE_min_z_value_l1780_178014

theorem min_z_value (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  z - x ≥ 9 → 
  (∀ w, w < z → ¬(x < w ∧ w < z ∧ w - x > 5 ∧ Odd w)) →
  z = 9 := by
sorry

end NUMINAMATH_CALUDE_min_z_value_l1780_178014


namespace NUMINAMATH_CALUDE_exists_number_with_2001_trailing_zeros_l1780_178019

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of all divisors of a natural number -/
def productOfDivisors (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with 2001 trailing zeros in its product of divisors -/
theorem exists_number_with_2001_trailing_zeros : 
  ∃ n : ℕ, trailingZeros (productOfDivisors n) = 2001 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_2001_trailing_zeros_l1780_178019


namespace NUMINAMATH_CALUDE_c_profit_share_l1780_178017

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_profit_share (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

theorem c_profit_share :
  let a_investment := 12000
  let b_investment := 16000
  let c_investment := 20000
  let total_investment := a_investment + b_investment + c_investment
  let total_profit := 86400
  calculate_profit_share c_investment total_investment total_profit = 36000 := by
sorry

#eval calculate_profit_share 20000 (12000 + 16000 + 20000) 86400

end NUMINAMATH_CALUDE_c_profit_share_l1780_178017


namespace NUMINAMATH_CALUDE_median_in_third_interval_l1780_178064

/-- Represents the distribution of students across score intervals --/
structure ScoreDistribution where
  total_students : ℕ
  intervals : List ℕ
  h_total : total_students = intervals.sum

/-- The index of the interval containing the median --/
def median_interval_index (sd : ScoreDistribution) : ℕ :=
  sd.intervals.foldl
    (λ acc count =>
      if acc.1 < sd.total_students / 2 then (acc.1 + count, acc.2 + 1)
      else acc)
    (0, 0)
  |>.2

theorem median_in_third_interval (sd : ScoreDistribution) :
  sd.total_students = 100 ∧
  sd.intervals = [20, 18, 15, 22, 14, 11] →
  median_interval_index sd = 3 := by
  sorry

#eval median_interval_index ⟨100, [20, 18, 15, 22, 14, 11], rfl⟩

end NUMINAMATH_CALUDE_median_in_third_interval_l1780_178064


namespace NUMINAMATH_CALUDE_toms_age_problem_l1780_178056

/-- Tom's age problem -/
theorem toms_age_problem (T N : ℝ) : 
  T > 0 ∧ N > 0 ∧ 
  (∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = T) ∧
  T - N = 3 * (T - 4 * N) →
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_problem_l1780_178056


namespace NUMINAMATH_CALUDE_ABC_reflection_collinear_l1780_178053

-- Define the basic structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the triangle ABC
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry

-- Define point P and line γ
def P : Point := sorry
def γ : Line := sorry

-- Define the reflection of a line with respect to another line
def reflect (l₁ l₂ : Line) : Line := sorry

-- Define the intersection of two lines
def intersect (l₁ l₂ : Line) : Point := sorry

-- Define lines PA, PB, PC
def PA : Line := sorry
def PB : Line := sorry
def PC : Line := sorry

-- Define lines BC, AC, AB
def BC : Line := sorry
def AC : Line := sorry
def AB : Line := sorry

-- Define points A', B', C'
def A' : Point := intersect (reflect PA γ) BC
def B' : Point := intersect (reflect PB γ) AC
def C' : Point := intersect (reflect PC γ) AB

-- Define collinearity
def collinear (p q r : Point) : Prop := sorry

-- The theorem to be proved
theorem ABC_reflection_collinear : collinear A' B' C' := by sorry

end NUMINAMATH_CALUDE_ABC_reflection_collinear_l1780_178053


namespace NUMINAMATH_CALUDE_remainder_problem_l1780_178016

theorem remainder_problem (k : ℕ+) (h : 120 % (k^2 : ℕ) = 8) : 150 % (k : ℕ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1780_178016


namespace NUMINAMATH_CALUDE_min_value_expression_l1780_178038

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 128) :
  x^2 + 8*x*y + 4*y^2 + 8*z^2 ≥ 384 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 128 ∧ x₀^2 + 8*x₀*y₀ + 4*y₀^2 + 8*z₀^2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1780_178038


namespace NUMINAMATH_CALUDE_money_left_after_taxes_l1780_178085

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

theorem money_left_after_taxes : 
  annual_income * (1 - tax_rate) = 49200 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_taxes_l1780_178085


namespace NUMINAMATH_CALUDE_integer_solutions_of_inequality_system_l1780_178084

theorem integer_solutions_of_inequality_system :
  {x : ℤ | x + 2 > 0 ∧ 2 * x - 1 ≤ 0} = {-1, 0} := by
sorry

end NUMINAMATH_CALUDE_integer_solutions_of_inequality_system_l1780_178084


namespace NUMINAMATH_CALUDE_total_dinners_sold_l1780_178065

def monday_sales : ℕ := 40

def tuesday_sales : ℕ := monday_sales + 40

def wednesday_sales : ℕ := tuesday_sales / 2

def thursday_sales : ℕ := wednesday_sales + 3

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales

theorem total_dinners_sold : total_sales = 203 := by
  sorry

end NUMINAMATH_CALUDE_total_dinners_sold_l1780_178065


namespace NUMINAMATH_CALUDE_optimal_price_for_max_revenue_l1780_178098

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The theorem stating the optimal price for maximum revenue --/
theorem optimal_price_for_max_revenue :
  ∃ (p : ℝ), 0 < p ∧ p ≤ 37.5 ∧
  ∀ (q : ℝ), 0 < q → q ≤ 37.5 → R p ≥ R q ∧
  p = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_for_max_revenue_l1780_178098


namespace NUMINAMATH_CALUDE_fourth_shot_probability_l1780_178028

/-- The probability of making a shot given the previous shot was made -/
def p_make_given_make : ℚ := 2/3

/-- The probability of making a shot given the previous shot was missed -/
def p_make_given_miss : ℚ := 1/3

/-- The probability of making the first shot -/
def p_first_shot : ℚ := 2/3

/-- The probability of making the n-th shot -/
def p_nth_shot (n : ℕ) : ℚ :=
  1/2 * (1 + 1 / 3^n)

theorem fourth_shot_probability :
  p_nth_shot 4 = 41/81 :=
sorry

end NUMINAMATH_CALUDE_fourth_shot_probability_l1780_178028


namespace NUMINAMATH_CALUDE_hunting_duration_is_three_weeks_l1780_178022

/-- Represents the hunting scenario in the forest -/
structure ForestHunt where
  initialWeasels : ℕ
  initialRabbits : ℕ
  foxes : ℕ
  weaselsPerFoxPerWeek : ℕ
  rabbitsPerFoxPerWeek : ℕ
  remainingRodents : ℕ

/-- Calculates the hunting duration in weeks -/
def huntingDuration (hunt : ForestHunt) : ℚ :=
  let initialRodents := hunt.initialWeasels + hunt.initialRabbits
  let rodentsCaughtPerWeek := hunt.foxes * (hunt.weaselsPerFoxPerWeek + hunt.rabbitsPerFoxPerWeek)
  let totalRodentsCaught := initialRodents - hunt.remainingRodents
  totalRodentsCaught / rodentsCaughtPerWeek

/-- Theorem stating that the hunting duration is 3 weeks for the given scenario -/
theorem hunting_duration_is_three_weeks (hunt : ForestHunt) 
    (h1 : hunt.initialWeasels = 100)
    (h2 : hunt.initialRabbits = 50)
    (h3 : hunt.foxes = 3)
    (h4 : hunt.weaselsPerFoxPerWeek = 4)
    (h5 : hunt.rabbitsPerFoxPerWeek = 2)
    (h6 : hunt.remainingRodents = 96) :
    huntingDuration hunt = 3 := by
  sorry


end NUMINAMATH_CALUDE_hunting_duration_is_three_weeks_l1780_178022


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l1780_178097

theorem simplify_sqrt_difference : 
  (Real.sqrt 800 / Real.sqrt 50) - (Real.sqrt 288 / Real.sqrt 72) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l1780_178097


namespace NUMINAMATH_CALUDE_range_of_a_l1780_178003

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ∈ Set.Iic (-2) ∪ {1} := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1780_178003


namespace NUMINAMATH_CALUDE_spring_properties_l1780_178062

-- Define the spring's properties
def initial_length : ℝ := 20
def rate_of_change : ℝ := 0.5

-- Define the relationship between weight and length
def spring_length (weight : ℝ) : ℝ := initial_length + rate_of_change * weight

-- Theorem stating the properties of the spring
theorem spring_properties :
  (∀ w : ℝ, w ≥ 0 → spring_length w ≥ initial_length) ∧
  (∀ w1 w2 : ℝ, w1 < w2 → spring_length w1 < spring_length w2) ∧
  (∀ w : ℝ, (spring_length (w + 1) - spring_length w) = rate_of_change) :=
by sorry

end NUMINAMATH_CALUDE_spring_properties_l1780_178062


namespace NUMINAMATH_CALUDE_travel_time_difference_l1780_178046

/-- Represents the travel times for different modes of transportation --/
structure TravelTimes where
  drivingTimeMinutes : ℕ
  driveToAirportMinutes : ℕ
  waitToBoardMinutes : ℕ
  exitPlaneMinutes : ℕ

/-- Calculates the total airplane travel time --/
def airplaneTravelTime (t : TravelTimes) : ℕ :=
  t.driveToAirportMinutes + t.waitToBoardMinutes + (t.drivingTimeMinutes / 3) + t.exitPlaneMinutes

/-- Theorem stating the time difference between driving and flying --/
theorem travel_time_difference (t : TravelTimes) 
  (h1 : t.drivingTimeMinutes = 195)
  (h2 : t.driveToAirportMinutes = 10)
  (h3 : t.waitToBoardMinutes = 20)
  (h4 : t.exitPlaneMinutes = 10) :
  t.drivingTimeMinutes - airplaneTravelTime t = 90 := by
  sorry


end NUMINAMATH_CALUDE_travel_time_difference_l1780_178046


namespace NUMINAMATH_CALUDE_min_determinant_2x2_matrix_l1780_178049

def S : Set ℤ := {-1, 1, 2}

theorem min_determinant_2x2_matrix :
  ∀ a b c d : ℤ, a ∈ S → b ∈ S → c ∈ S → d ∈ S →
  ∀ x : ℤ, (∃ a' b' c' d' : ℤ, a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ x = a' * d' - b' * c') →
  x ≥ -6 :=
by sorry

end NUMINAMATH_CALUDE_min_determinant_2x2_matrix_l1780_178049


namespace NUMINAMATH_CALUDE_initial_volume_calculation_l1780_178001

theorem initial_volume_calculation (initial_milk_percentage : Real)
                                   (final_milk_percentage : Real)
                                   (added_water : Real) :
  initial_milk_percentage = 0.84 →
  final_milk_percentage = 0.64 →
  added_water = 18.75 →
  ∃ (initial_volume : Real),
    initial_volume * initial_milk_percentage = 
    final_milk_percentage * (initial_volume + added_water) ∧
    initial_volume = 225 := by
  sorry

end NUMINAMATH_CALUDE_initial_volume_calculation_l1780_178001


namespace NUMINAMATH_CALUDE_inequality_relation_l1780_178023

theorem inequality_relation (x y : ℝ) : 2*x - 5 < 2*y - 5 → x < y := by
  sorry

end NUMINAMATH_CALUDE_inequality_relation_l1780_178023


namespace NUMINAMATH_CALUDE_bottle_cap_cost_l1780_178078

theorem bottle_cap_cost (cost_per_cap : ℝ) (num_caps : ℕ) : 
  cost_per_cap = 5 → num_caps = 5 → cost_per_cap * (num_caps : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_cost_l1780_178078


namespace NUMINAMATH_CALUDE_tailwind_speed_l1780_178035

def plane_speed_with_tailwind : ℝ := 460
def plane_speed_against_tailwind : ℝ := 310

theorem tailwind_speed : ∃ (plane_speed tailwind_speed : ℝ),
  plane_speed + tailwind_speed = plane_speed_with_tailwind ∧
  plane_speed - tailwind_speed = plane_speed_against_tailwind ∧
  tailwind_speed = 75 := by
  sorry

end NUMINAMATH_CALUDE_tailwind_speed_l1780_178035


namespace NUMINAMATH_CALUDE_max_product_for_maximized_fraction_l1780_178073

def Digits := Fin 8

def validDigit (d : Digits) : ℕ := d.val + 2

theorem max_product_for_maximized_fraction :
  ∃ (A B C D : Digits),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (∀ (A' B' C' D' : Digits),
      A' ≠ B' ∧ A' ≠ C' ∧ A' ≠ D' ∧ B' ≠ C' ∧ B' ≠ D' ∧ C' ≠ D' →
      (validDigit A' * validDigit B') / (validDigit C' * validDigit D' : ℚ) ≤
      (validDigit A * validDigit B) / (validDigit C * validDigit D : ℚ)) ∧
    validDigit A * validDigit B = 72 :=
by sorry

end NUMINAMATH_CALUDE_max_product_for_maximized_fraction_l1780_178073


namespace NUMINAMATH_CALUDE_sum_equals_three_halves_l1780_178091

theorem sum_equals_three_halves : 
  let original_sum := (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 + 1 / 13 + 1 / 15
  let removed_terms := 1 / 13 + 1 / 15
  original_sum - removed_terms = 3 / 2 →
  (1 : ℚ) / 3 + 1 / 5 + 1 / 7 + 1 / 9 + 1 / 11 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_three_halves_l1780_178091
