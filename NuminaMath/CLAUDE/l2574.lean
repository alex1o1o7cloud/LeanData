import Mathlib

namespace NUMINAMATH_CALUDE_two_true_propositions_l2574_257421

theorem two_true_propositions :
  let P : ℝ → Prop := λ a => a > -3
  let Q : ℝ → Prop := λ a => a > -6
  let original := ∀ a, P a → Q a
  let converse := ∀ a, Q a → P a
  let inverse := ∀ a, ¬(P a) → ¬(Q a)
  let contrapositive := ∀ a, ¬(Q a) → ¬(P a)
  (original ∧ ¬converse ∧ ¬inverse ∧ contrapositive) :=
by sorry

end NUMINAMATH_CALUDE_two_true_propositions_l2574_257421


namespace NUMINAMATH_CALUDE_well_digging_cost_l2574_257436

/-- The cost of digging a cylindrical well -/
theorem well_digging_cost (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : 
  depth = 14 → diameter = 3 → cost_per_cubic_meter = 16 →
  ∃ (total_cost : ℝ), abs (total_cost - 1584.24) < 0.01 ∧ 
  total_cost = cost_per_cubic_meter * Real.pi * (diameter / 2)^2 * depth := by
sorry

end NUMINAMATH_CALUDE_well_digging_cost_l2574_257436


namespace NUMINAMATH_CALUDE_square_area_from_smaller_squares_l2574_257434

/-- The area of a square composed of smaller squares -/
theorem square_area_from_smaller_squares
  (n : ℕ) -- number of smaller squares
  (side_length : ℝ) -- side length of each smaller square
  (h_n : n = 8) -- there are 8 smaller squares
  (h_side : side_length = 2) -- side length of each smaller square is 2 cm
  : n * side_length^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_smaller_squares_l2574_257434


namespace NUMINAMATH_CALUDE_evaluate_expression_l2574_257439

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2574_257439


namespace NUMINAMATH_CALUDE_loan_period_duration_l2574_257478

/-- The amount of money lent (in Rs.) -/
def loanAmount : ℝ := 3150

/-- The interest rate A charges B (as a decimal) -/
def rateAtoB : ℝ := 0.08

/-- The interest rate B charges C (as a decimal) -/
def rateBtoC : ℝ := 0.125

/-- B's total gain over the period (in Rs.) -/
def totalGain : ℝ := 283.5

/-- The duration of the period in years -/
def periodYears : ℝ := 2

theorem loan_period_duration :
  periodYears * (rateBtoC * loanAmount - rateAtoB * loanAmount) = totalGain :=
by sorry

end NUMINAMATH_CALUDE_loan_period_duration_l2574_257478


namespace NUMINAMATH_CALUDE_calculation_proof_l2574_257423

theorem calculation_proof : 3 * 15 + 3 * 16 + 3 * 19 + 11 = 161 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2574_257423


namespace NUMINAMATH_CALUDE_shirt_price_l2574_257460

/-- Given a shirt and coat with a total cost of 600 dollars, where the shirt costs
    one-third the price of the coat, prove that the shirt costs 150 dollars. -/
theorem shirt_price (total_cost : ℝ) (shirt_price : ℝ) (coat_price : ℝ) 
  (h1 : total_cost = 600)
  (h2 : shirt_price + coat_price = total_cost)
  (h3 : shirt_price = (1/3) * coat_price) :
  shirt_price = 150 := by
  sorry

end NUMINAMATH_CALUDE_shirt_price_l2574_257460


namespace NUMINAMATH_CALUDE_car_trip_duration_l2574_257435

theorem car_trip_duration :
  ∀ (total_time : ℝ) (second_part_time : ℝ),
    total_time > 0 →
    second_part_time ≥ 0 →
    total_time = 5 + second_part_time →
    (30 * 5 + 42 * second_part_time) / total_time = 34 →
    total_time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2574_257435


namespace NUMINAMATH_CALUDE_probability_exactly_one_instrument_l2574_257464

/-- The probability of playing exactly one instrument in a group -/
theorem probability_exactly_one_instrument 
  (total_people : ℕ) 
  (at_least_one_fraction : ℚ) 
  (two_or_more : ℕ) 
  (h1 : total_people = 800) 
  (h2 : at_least_one_fraction = 1 / 5) 
  (h3 : two_or_more = 64) : 
  (↑((at_least_one_fraction * ↑total_people).num - two_or_more) / ↑total_people : ℚ) = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_one_instrument_l2574_257464


namespace NUMINAMATH_CALUDE_africa_asia_difference_l2574_257479

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 8

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 42

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 31

/-- Theorem: The difference between the number of bird families that flew to Africa
    and the number of bird families that flew to Asia is 11 -/
theorem africa_asia_difference : africa_families - asia_families = 11 := by
  sorry

end NUMINAMATH_CALUDE_africa_asia_difference_l2574_257479


namespace NUMINAMATH_CALUDE_solution_inequality_l2574_257409

theorem solution_inequality (x : ℝ) : 
  x > 0 → x * Real.sqrt (15 - x) + Real.sqrt (15 * x - x^3) ≥ 15 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_inequality_l2574_257409


namespace NUMINAMATH_CALUDE_triangle_properties_l2574_257468

/-- Triangle ABC with vertices A(-1,-1), B(3,2), C(7,-7) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The specific triangle ABC given in the problem -/
def triangleABC : Triangle :=
  { A := (-1, -1)
    B := (3, 2)
    C := (7, -7) }

/-- Altitude from a point to a line -/
def altitude (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ → ℝ := sorry

/-- Area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Main theorem stating the properties of triangle ABC -/
theorem triangle_properties (t : Triangle) (h : t = triangleABC) :
  (altitude t.C (λ x => (3/4) * x - 5/4) = λ x => (-4/3) * x + 19/3) ∧
  triangleArea t = 24 := by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2574_257468


namespace NUMINAMATH_CALUDE_total_pencils_l2574_257481

/-- Given that each child has 2 pencils and there are 8 children, 
    prove that the total number of pencils is 16. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) 
  (h2 : num_children = 8) : 
  pencils_per_child * num_children = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l2574_257481


namespace NUMINAMATH_CALUDE_locus_and_fixed_point_l2574_257499

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point N
def point_N : ℝ × ℝ := (-2, 0)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1

-- Define points A₁ and A₂
def point_A1 : ℝ × ℝ := (-1, 0)
def point_A2 : ℝ × ℝ := (1, 0)

-- Define the line x = 2
def line_x_2 (x y : ℝ) : Prop := x = 2

-- Theorem statement
theorem locus_and_fixed_point :
  ∀ (P : ℝ × ℝ) (Q : ℝ × ℝ) (E : ℝ × ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ),
  circle_M P.1 P.2 →
  line_x_2 E.1 E.2 ∧ line_x_2 F.1 F.2 →
  E.2 = -F.2 →
  curve_C Q.1 Q.2 →
  curve_C A.1 A.2 ∧ curve_C B.1 B.2 →
  (∃ (m : ℝ), A.2 - point_A1.2 = m * (A.1 - point_A1.1) ∧
               E.2 - point_A1.2 = m * (E.1 - point_A1.1)) →
  (∃ (n : ℝ), B.2 - point_A2.2 = n * (B.1 - point_A2.1) ∧
               F.2 - point_A2.2 = n * (F.1 - point_A2.1)) →
  (∃ (k : ℝ), B.2 - A.2 = k * (B.1 - A.1) ∧ 0 = k * (2 - A.1) + A.2) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_point_l2574_257499


namespace NUMINAMATH_CALUDE_sum_of_digits_mod_9_C_mod_9_eq_5_l2574_257492

/-- The sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A is the sum of digits of 4568^7777 -/
def A : ℕ := sumOfDigits (4568^7777)

/-- B is the sum of digits of A -/
def B : ℕ := sumOfDigits A

/-- C is the sum of digits of B -/
def C : ℕ := sumOfDigits B

/-- Theorem stating that the sum of digits of a number is congruent to the number modulo 9 -/
theorem sum_of_digits_mod_9 (n : ℕ) : sumOfDigits n ≡ n [MOD 9] := sorry

/-- Main theorem to prove -/
theorem C_mod_9_eq_5 : C ≡ 5 [MOD 9] := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_mod_9_C_mod_9_eq_5_l2574_257492


namespace NUMINAMATH_CALUDE_compare_n_squared_and_two_to_n_l2574_257498

theorem compare_n_squared_and_two_to_n (n : ℕ+) :
  (n = 1 → n.val^2 < 2^n.val) ∧
  (n = 2 → n.val^2 = 2^n.val) ∧
  (n = 3 → n.val^2 > 2^n.val) ∧
  (n = 4 → n.val^2 = 2^n.val) ∧
  (n ≥ 5 → n.val^2 < 2^n.val) := by
  sorry

end NUMINAMATH_CALUDE_compare_n_squared_and_two_to_n_l2574_257498


namespace NUMINAMATH_CALUDE_new_mixture_ratio_l2574_257426

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℚ
  water : ℚ

/-- Calculates the ratio of alcohol to water in a mixture -/
def ratioAlcoholToWater (m : Mixture) : ℚ := m.alcohol / m.water

/-- First jar with 3:1 ratio and 4 liters total -/
def jar1 : Mixture := { alcohol := 3, water := 1 }

/-- Second jar with 2:1 ratio and 6 liters total -/
def jar2 : Mixture := { alcohol := 4, water := 2 }

/-- Amount taken from first jar -/
def amount1 : ℚ := 1

/-- Amount taken from second jar -/
def amount2 : ℚ := 2

/-- New mixture created from combining portions of jar1 and jar2 -/
def newMixture : Mixture := {
  alcohol := amount1 * (jar1.alcohol / (jar1.alcohol + jar1.water)) + 
             amount2 * (jar2.alcohol / (jar2.alcohol + jar2.water)),
  water := amount1 * (jar1.water / (jar1.alcohol + jar1.water)) + 
           amount2 * (jar2.water / (jar2.alcohol + jar2.water))
}

theorem new_mixture_ratio : 
  ratioAlcoholToWater newMixture = 41 / 19 := by sorry

end NUMINAMATH_CALUDE_new_mixture_ratio_l2574_257426


namespace NUMINAMATH_CALUDE_sqrt_280_between_16_and_17_l2574_257420

theorem sqrt_280_between_16_and_17 : 16 < Real.sqrt 280 ∧ Real.sqrt 280 < 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_280_between_16_and_17_l2574_257420


namespace NUMINAMATH_CALUDE_jenny_ate_65_squares_l2574_257465

/-- The number of chocolate squares Mike ate -/
def mike_squares : ℕ := 20

/-- The number of chocolate squares Jenny ate -/
def jenny_squares : ℕ := 3 * mike_squares + 5

/-- Theorem stating that Jenny ate 65 chocolate squares -/
theorem jenny_ate_65_squares : jenny_squares = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_squares_l2574_257465


namespace NUMINAMATH_CALUDE_cube_coverage_l2574_257429

/-- Represents a rectangular strip of size 1 × 2 -/
structure Rectangle where
  length : Nat
  width : Nat

/-- Represents a cube of size n × n × n -/
structure Cube where
  size : Nat

/-- Predicate to check if a rectangle abuts exactly five others -/
def abutsFiveOthers (r : Rectangle) : Prop :=
  sorry

/-- Predicate to check if a cube's surface can be covered with rectangles -/
def canBeCovered (c : Cube) (r : Rectangle) : Prop :=
  sorry

theorem cube_coverage (n : Nat) :
  (∃ c : Cube, c.size = n ∧ ∃ r : Rectangle, r.length = 2 ∧ r.width = 1 ∧
    canBeCovered c r ∧ abutsFiveOthers r) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_cube_coverage_l2574_257429


namespace NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l2574_257480

theorem meals_neither_kosher_nor_vegan 
  (total_clients : ℕ) 
  (vegan_clients : ℕ) 
  (kosher_clients : ℕ) 
  (both_vegan_and_kosher : ℕ) 
  (h1 : total_clients = 30) 
  (h2 : vegan_clients = 7) 
  (h3 : kosher_clients = 8) 
  (h4 : both_vegan_and_kosher = 3) : 
  total_clients - (vegan_clients + kosher_clients - both_vegan_and_kosher) = 18 :=
by sorry

end NUMINAMATH_CALUDE_meals_neither_kosher_nor_vegan_l2574_257480


namespace NUMINAMATH_CALUDE_integer_less_than_sqrt_10_l2574_257482

theorem integer_less_than_sqrt_10 : ∃ n : ℤ, (n : ℝ) < Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_integer_less_than_sqrt_10_l2574_257482


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l2574_257454

/-- 
If the quadratic equation 2x^2 - 5x + m = 0 has two equal real roots,
then m = 25/8.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - 5 * x + m = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - 5 * y + m = 0 → y = x) → 
  m = 25/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l2574_257454


namespace NUMINAMATH_CALUDE_animal_arrangement_count_l2574_257425

/-- The number of ways to arrange animals in cages -/
def arrange_animals (num_chickens num_dogs num_cats num_rabbits : ℕ) : ℕ :=
  Nat.factorial 4 * 
  Nat.factorial num_chickens * 
  Nat.factorial num_dogs * 
  Nat.factorial num_cats * 
  Nat.factorial num_rabbits

/-- Theorem stating the number of arrangements for the given animals -/
theorem animal_arrangement_count :
  arrange_animals 3 3 4 2 = 41472 := by
  sorry

end NUMINAMATH_CALUDE_animal_arrangement_count_l2574_257425


namespace NUMINAMATH_CALUDE_flower_beds_count_l2574_257431

/-- Given that 10 seeds are put in each flower bed and 60 seeds were planted altogether,
    prove that the number of flower beds is 6. -/
theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ)
  (h1 : seeds_per_bed = 10)
  (h2 : total_seeds = 60)
  (h3 : num_beds * seeds_per_bed = total_seeds) :
  num_beds = 6 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_count_l2574_257431


namespace NUMINAMATH_CALUDE_median_name_length_is_five_l2574_257447

/-- Represents the distribution of name lengths -/
structure NameLengthDistribution where
  fourLetters : Nat
  fiveLetters : Nat
  sixLetters : Nat
  sevenLetters : Nat

/-- Calculates the median of a list of numbers -/
def median (list : List Nat) : Rat :=
  sorry

/-- Generates a list of name lengths based on the distribution -/
def generateNameLengthList (dist : NameLengthDistribution) : List Nat :=
  sorry

theorem median_name_length_is_five (dist : NameLengthDistribution) : 
  dist.fourLetters = 9 ∧ 
  dist.fiveLetters = 6 ∧ 
  dist.sixLetters = 2 ∧ 
  dist.sevenLetters = 7 → 
  median (generateNameLengthList dist) = 5 := by
  sorry

end NUMINAMATH_CALUDE_median_name_length_is_five_l2574_257447


namespace NUMINAMATH_CALUDE_order_of_numbers_l2574_257462

theorem order_of_numbers : ∀ (a b c : ℝ), 
  a = 6^(1/2) → b = (1/2)^6 → c = Real.log 6 / Real.log (1/2) →
  c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l2574_257462


namespace NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l2574_257463

theorem banana_pear_weight_equivalence (banana_weight pear_weight : ℝ) 
  (h1 : 9 * banana_weight = 6 * pear_weight) :
  36 * banana_weight = 24 * pear_weight := by
  sorry

end NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l2574_257463


namespace NUMINAMATH_CALUDE_probability_point_in_sphere_l2574_257496

/-- The probability that a randomly selected point (x, y, z) in a cube with side length 2
    centered at the origin lies within a unit sphere centered at the origin. -/
theorem probability_point_in_sphere : 
  let cube_volume : ℝ := 8
  let sphere_volume : ℝ := (4 / 3) * Real.pi
  let prob : ℝ := sphere_volume / cube_volume
  prob = Real.pi / 6 := by sorry

end NUMINAMATH_CALUDE_probability_point_in_sphere_l2574_257496


namespace NUMINAMATH_CALUDE_A_characterization_l2574_257432

/-- The set A defined by the quadratic equation kx^2 - 3x + 2 = 0 -/
def A (k : ℝ) : Set ℝ := {x | k * x^2 - 3 * x + 2 = 0}

/-- Theorem stating the conditions for A to be empty or contain exactly one element -/
theorem A_characterization (k : ℝ) :
  (A k = ∅ ↔ k > 9/8) ∧
  (∃! x, x ∈ A k ↔ k = 0 ∨ k = 9/8) ∧
  (k = 0 → A k = {2/3}) ∧
  (k = 9/8 → A k = {4/3}) :=
sorry

end NUMINAMATH_CALUDE_A_characterization_l2574_257432


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_cube_condition_nonzero_sum_of_squares_iff_not_both_zero_l2574_257404

-- Statement 1
theorem necessary_not_sufficient_cube_condition (x : ℝ) :
  (x^3 = -27 → x^2 = 9) ∧ ¬(x^2 = 9 → x^3 = -27) :=
sorry

-- Statement 2
theorem nonzero_sum_of_squares_iff_not_both_zero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ ¬(a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_cube_condition_nonzero_sum_of_squares_iff_not_both_zero_l2574_257404


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l2574_257408

theorem smallest_gcd_multiple (m n : ℕ+) (h : Nat.gcd m n = 18) :
  ∃ (k : ℕ+), k = Nat.gcd (12 * m) (20 * n) ∧ 
  (∀ (l : ℕ+), l = Nat.gcd (12 * m) (20 * n) → k ≤ l) ∧
  k = 72 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l2574_257408


namespace NUMINAMATH_CALUDE_circles_radius_order_l2574_257477

noncomputable def circle_A_radius : ℝ := 3

noncomputable def circle_B_area : ℝ := 12 * Real.pi

noncomputable def circle_C_area : ℝ := 28 * Real.pi

noncomputable def circle_B_radius : ℝ := Real.sqrt (circle_B_area / Real.pi)

noncomputable def circle_C_radius : ℝ := Real.sqrt (circle_C_area / Real.pi)

theorem circles_radius_order :
  circle_A_radius < circle_B_radius ∧ circle_B_radius < circle_C_radius := by
  sorry

end NUMINAMATH_CALUDE_circles_radius_order_l2574_257477


namespace NUMINAMATH_CALUDE_library_books_before_grant_l2574_257453

/-- The number of books purchased with the grant -/
def books_purchased : ℕ := 2647

/-- The total number of books after the purchase -/
def total_books : ℕ := 8582

/-- The number of books before the grant -/
def books_before : ℕ := total_books - books_purchased

theorem library_books_before_grant :
  books_before = 5935 :=
sorry

end NUMINAMATH_CALUDE_library_books_before_grant_l2574_257453


namespace NUMINAMATH_CALUDE_prob_13_11_is_quarter_l2574_257469

/-- Represents a table tennis game with specific scoring probabilities -/
structure TableTennisGame where
  /-- Probability of player A scoring when A serves -/
  prob_a_scores_on_a_serve : ℝ
  /-- Probability of player A scoring when B serves -/
  prob_a_scores_on_b_serve : ℝ

/-- Calculates the probability of reaching a 13:11 score from a 10:10 tie -/
def prob_13_11 (game : TableTennisGame) : ℝ :=
  sorry

/-- The main theorem stating the probability of reaching 13:11 is 1/4 -/
theorem prob_13_11_is_quarter (game : TableTennisGame) 
  (h1 : game.prob_a_scores_on_a_serve = 2/3)
  (h2 : game.prob_a_scores_on_b_serve = 1/2) :
  prob_13_11 game = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_13_11_is_quarter_l2574_257469


namespace NUMINAMATH_CALUDE_line_property_l2574_257438

/-- Given a line passing through points (1, -1) and (3, 7), 
    prove that 3m - b = 17, where m is the slope and b is the y-intercept. -/
theorem line_property (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (1 : ℝ) * m + b = -1 →        -- Point (1, -1) satisfies the equation
  (3 : ℝ) * m + b = 7 →         -- Point (3, 7) satisfies the equation
  3 * m - b = 17 := by
sorry

end NUMINAMATH_CALUDE_line_property_l2574_257438


namespace NUMINAMATH_CALUDE_math_club_team_selection_l2574_257403

def mathClubSize : ℕ := 15
def teamSize : ℕ := 5

theorem math_club_team_selection :
  Nat.choose mathClubSize teamSize = 3003 := by
  sorry

end NUMINAMATH_CALUDE_math_club_team_selection_l2574_257403


namespace NUMINAMATH_CALUDE_parabola_properties_l2574_257475

/-- Given a parabola y = x^2 - 8x + 12, prove its properties -/
theorem parabola_properties :
  let f (x : ℝ) := x^2 - 8*x + 12
  ∃ (axis vertex_x vertex_y x1 x2 : ℝ),
    -- The axis of symmetry
    axis = 4 ∧
    -- The vertex coordinates
    f vertex_x = vertex_y ∧
    vertex_x = 4 ∧
    vertex_y = -4 ∧
    -- The x-axis intersection points
    f x1 = 0 ∧
    f x2 = 0 ∧
    x1 = 2 ∧
    x2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2574_257475


namespace NUMINAMATH_CALUDE_solution_set_of_sin_equation_l2574_257459

theorem solution_set_of_sin_equation :
  let S : Set ℝ := {x | 2 * Real.sin ((2/3) * x) = 1}
  S = {x | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_sin_equation_l2574_257459


namespace NUMINAMATH_CALUDE_addition_problem_l2574_257401

theorem addition_problem : ∃! x : ℝ, 8 + x = -5 ∧ x = -13 := by sorry

end NUMINAMATH_CALUDE_addition_problem_l2574_257401


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2574_257407

/-- A circle with center O and radius r -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Relationship between a line and a circle -/
inductive LineCircleRelation
  | Disjoint
  | Tangent
  | Intersect

theorem line_intersects_circle (O : Circle) (l : Line) :
  O.radius = 4 →
  distancePointToLine O.center l = 3 →
  LineCircleRelation.Intersect = 
    match O.radius, distancePointToLine O.center l with
    | r, d => if r > d then LineCircleRelation.Intersect
              else if r = d then LineCircleRelation.Tangent
              else LineCircleRelation.Disjoint :=
by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2574_257407


namespace NUMINAMATH_CALUDE_janet_income_difference_l2574_257428

/-- Calculates the difference in monthly income between freelancing and current job for Janet --/
theorem janet_income_difference :
  let hours_per_week : ℕ := 40
  let weeks_per_month : ℕ := 4
  let current_hourly_rate : ℚ := 30
  let freelance_hourly_rate : ℚ := 40
  let extra_fica_per_week : ℚ := 25
  let healthcare_premium_per_month : ℚ := 400

  let current_monthly_income : ℚ := hours_per_week * weeks_per_month * current_hourly_rate
  let freelance_gross_monthly_income : ℚ := hours_per_week * weeks_per_month * freelance_hourly_rate
  let additional_monthly_costs : ℚ := extra_fica_per_week * weeks_per_month + healthcare_premium_per_month
  let freelance_net_monthly_income : ℚ := freelance_gross_monthly_income - additional_monthly_costs

  freelance_net_monthly_income - current_monthly_income = 1100 := by
  sorry

end NUMINAMATH_CALUDE_janet_income_difference_l2574_257428


namespace NUMINAMATH_CALUDE_probability_two_heads_and_three_l2574_257415

def coin_outcomes : ℕ := 2
def die_outcomes : ℕ := 6

def total_outcomes : ℕ := coin_outcomes * coin_outcomes * die_outcomes

def favorable_outcome : ℕ := 1

theorem probability_two_heads_and_three : 
  (favorable_outcome : ℚ) / total_outcomes = 1 / 24 := by sorry

end NUMINAMATH_CALUDE_probability_two_heads_and_three_l2574_257415


namespace NUMINAMATH_CALUDE_class_fraction_proof_l2574_257473

theorem class_fraction_proof (G : ℚ) (B : ℚ) (T : ℚ) (F : ℚ) :
  B / G = 7 / 3 →
  T = B + G →
  (2 / 3) * G = F * T →
  F = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_class_fraction_proof_l2574_257473


namespace NUMINAMATH_CALUDE_oranges_count_l2574_257451

/-- The number of oranges initially in Tom's fruit bowl -/
def initial_oranges : ℕ := 3

/-- The number of lemons initially in Tom's fruit bowl -/
def initial_lemons : ℕ := 6

/-- The number of fruits Tom eats -/
def fruits_eaten : ℕ := 3

/-- The number of fruits remaining after Tom eats -/
def remaining_fruits : ℕ := 6

/-- Theorem stating that the number of oranges initially in the fruit bowl is 3 -/
theorem oranges_count : initial_oranges = 3 :=
  by
    have h1 : initial_oranges + initial_lemons = remaining_fruits + fruits_eaten :=
      by sorry
    have h2 : initial_oranges + 6 = 6 + 3 :=
      by sorry
    show initial_oranges = 3
    sorry

#check oranges_count

end NUMINAMATH_CALUDE_oranges_count_l2574_257451


namespace NUMINAMATH_CALUDE_new_average_mark_l2574_257476

theorem new_average_mark (n : ℕ) (initial_avg : ℚ) (excluded_n : ℕ) (excluded_avg : ℚ) :
  n = 9 →
  initial_avg = 60 →
  excluded_n = 5 →
  excluded_avg = 44 →
  let total_marks := n * initial_avg
  let excluded_marks := excluded_n * excluded_avg
  let remaining_marks := total_marks - excluded_marks
  let remaining_n := n - excluded_n
  (remaining_marks / remaining_n : ℚ) = 80 := by
  sorry

end NUMINAMATH_CALUDE_new_average_mark_l2574_257476


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l2574_257452

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if a line is tangent to a circle -/
def tangent_to_circle (l : Line) (c : Circle) : Prop :=
  (abs (l.a * c.h + l.b * c.k + l.c) / Real.sqrt (l.a^2 + l.b^2)) = c.r

theorem tangent_lines_to_circle (given_line : Line) (c : Circle) :
  given_line = Line.mk 1 2 1 →
  c = Circle.mk 0 0 (Real.sqrt 5) →
  ∃ (l1 l2 : Line),
    parallel l1 given_line ∧
    parallel l2 given_line ∧
    tangent_to_circle l1 c ∧
    tangent_to_circle l2 c ∧
    ((l1 = Line.mk 1 2 5 ∧ l2 = Line.mk 1 2 (-5)) ∨
     (l1 = Line.mk 1 2 (-5) ∧ l2 = Line.mk 1 2 5)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l2574_257452


namespace NUMINAMATH_CALUDE_gwen_book_collection_total_l2574_257444

/-- Represents Gwen's book collection --/
structure BookCollection where
  mystery_shelves : ℕ
  mystery_books_per_shelf : ℕ
  picture_shelves : ℕ
  picture_books_per_shelf : ℕ
  scifi_shelves : ℕ
  scifi_books_per_shelf : ℕ
  nonfiction_shelves : ℕ
  nonfiction_books_per_shelf : ℕ
  mystery_books_lent : ℕ
  scifi_books_lent : ℕ
  picture_books_borrowed : ℕ

/-- Calculates the total number of books in Gwen's collection --/
def total_books (collection : BookCollection) : ℕ :=
  (collection.mystery_shelves * collection.mystery_books_per_shelf - collection.mystery_books_lent) +
  (collection.picture_shelves * collection.picture_books_per_shelf) +
  (collection.scifi_shelves * collection.scifi_books_per_shelf - collection.scifi_books_lent) +
  (collection.nonfiction_shelves * collection.nonfiction_books_per_shelf)

/-- Theorem stating that Gwen's book collection contains 106 books --/
theorem gwen_book_collection_total :
  ∃ (collection : BookCollection),
    collection.mystery_shelves = 8 ∧
    collection.mystery_books_per_shelf = 6 ∧
    collection.picture_shelves = 5 ∧
    collection.picture_books_per_shelf = 4 ∧
    collection.scifi_shelves = 4 ∧
    collection.scifi_books_per_shelf = 7 ∧
    collection.nonfiction_shelves = 3 ∧
    collection.nonfiction_books_per_shelf = 5 ∧
    collection.mystery_books_lent = 2 ∧
    collection.scifi_books_lent = 3 ∧
    collection.picture_books_borrowed = 5 ∧
    total_books collection = 106 :=
by sorry


end NUMINAMATH_CALUDE_gwen_book_collection_total_l2574_257444


namespace NUMINAMATH_CALUDE_equation_solution_l2574_257441

theorem equation_solution : ∃! x : ℝ, 
  x ≠ 2 ∧ x ≠ 3 ∧ (x^3 - 4*x^2)/(x^2 - 5*x + 6) - x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2574_257441


namespace NUMINAMATH_CALUDE_pyramid_intersection_volume_l2574_257483

/-- The length of each edge of the pyramids -/
def edge_length : ℝ := 12

/-- The volume of the solid of intersection of two regular square pyramids -/
def intersection_volume : ℝ := 72

/-- Theorem stating the volume of the solid of intersection of two regular square pyramids -/
theorem pyramid_intersection_volume :
  let pyramids : ℕ := 2
  let base_parallel : Prop := True  -- Represents that bases are parallel
  let edges_parallel : Prop := True  -- Represents that edges are parallel
  let apex_at_center : Prop := True  -- Represents that each apex is at the center of the other base
  intersection_volume = 72 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_intersection_volume_l2574_257483


namespace NUMINAMATH_CALUDE_wall_building_time_l2574_257489

/-- Given that 60 workers can build a wall in 3 days, prove that 30 workers 
    will take 6 days to build the same wall, assuming consistent work rate and conditions. -/
theorem wall_building_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 60) 
  (h2 : days_initial = 3) 
  (h3 : workers_new = 30) :
  (workers_initial * days_initial) / workers_new = 6 := by
sorry

end NUMINAMATH_CALUDE_wall_building_time_l2574_257489


namespace NUMINAMATH_CALUDE_diamonds_sequence_property_diamonds_10th_figure_l2574_257474

/-- The number of diamonds in the nth figure of the sequence -/
def diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 5
  else 1 + 8 * (n - 1) * n

/-- The sequence satisfies the given conditions -/
theorem diamonds_sequence_property (n : ℕ) (h : n ≥ 3) :
  diamonds n = diamonds (n-1) + 8 * (n-1) :=
sorry

/-- The total number of diamonds in the 10th figure is 721 -/
theorem diamonds_10th_figure :
  diamonds 10 = 721 :=
sorry

end NUMINAMATH_CALUDE_diamonds_sequence_property_diamonds_10th_figure_l2574_257474


namespace NUMINAMATH_CALUDE_prove_my_current_age_l2574_257466

/-- The age at which my dog was born -/
def age_when_dog_born : ℕ := 15

/-- The age my dog will be in two years -/
def dog_age_in_two_years : ℕ := 4

/-- My current age -/
def my_current_age : ℕ := age_when_dog_born + (dog_age_in_two_years - 2)

theorem prove_my_current_age : my_current_age = 17 := by
  sorry

end NUMINAMATH_CALUDE_prove_my_current_age_l2574_257466


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l2574_257485

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_third : a 3 = 1)
  (h_product : a 5 * a 6 * a 7 = 8) :
  a 9 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l2574_257485


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l2574_257487

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x.val y.val = 360) (h2 : Nat.gcd x.val z.val = 1176) :
  ∃ (k : ℕ+), (∀ (w : ℕ+), Nat.gcd y.val z.val ≥ k.val) ∧ Nat.gcd y.val z.val = k.val :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l2574_257487


namespace NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l2574_257412

def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f t x ≤ 2) ∧ (∃ x ∈ Set.Icc 0 3, f t x = 2) →
  t = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l2574_257412


namespace NUMINAMATH_CALUDE_tan_15_identity_l2574_257449

theorem tan_15_identity : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_15_identity_l2574_257449


namespace NUMINAMATH_CALUDE_fraction_geq_one_iff_x_in_range_l2574_257446

theorem fraction_geq_one_iff_x_in_range (x : ℝ) : 2 / x ≥ 1 ↔ 0 < x ∧ x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_geq_one_iff_x_in_range_l2574_257446


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2574_257445

/-- The perimeter of a right-angled triangle formed by the difference between two squares -/
theorem triangle_perimeter (x y : ℝ) (h : 0 < x ∧ x < y) :
  let side := (y - x) / 2
  let hypotenuse := (y - x) / Real.sqrt 2
  2 * side + hypotenuse = (y - x) * (1 + Real.sqrt 2) / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2574_257445


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2574_257448

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 54) :
  let side_length : ℝ := Real.sqrt area
  let perimeter : ℝ := 4 * side_length
  let total_cost : ℝ := perimeter * price_per_foot
  total_cost = 3672 := by
sorry


end NUMINAMATH_CALUDE_fence_cost_square_plot_l2574_257448


namespace NUMINAMATH_CALUDE_scissors_count_l2574_257417

theorem scissors_count (initial : Nat) (added : Nat) (total : Nat) : 
  initial = 54 → added = 22 → total = initial + added → total = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l2574_257417


namespace NUMINAMATH_CALUDE_fraction_problem_l2574_257410

theorem fraction_problem (f : ℝ) : f * 50.0 - 4 = 6 → f = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2574_257410


namespace NUMINAMATH_CALUDE_hd_ha_ratio_specific_triangle_l2574_257490

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The foot of the altitude from a vertex to the opposite side -/
def altitude_foot (t : Triangle) (vertex : ℕ) : ℝ × ℝ := sorry

/-- The vertex of a triangle -/
def vertex (t : Triangle) (v : ℕ) : ℝ × ℝ := sorry

/-- The ratio of distances HD:HA in the triangle -/
def hd_ha_ratio (t : Triangle) : ℝ × ℝ := sorry

theorem hd_ha_ratio_specific_triangle :
  let t : Triangle := ⟨11, 13, 20, sorry, sorry, sorry, sorry, sorry, sorry⟩
  let h := orthocenter t
  let d := altitude_foot t 0  -- Assuming 0 represents vertex A
  let a := vertex t 0
  hd_ha_ratio t = (0, 6.6) := by sorry

end NUMINAMATH_CALUDE_hd_ha_ratio_specific_triangle_l2574_257490


namespace NUMINAMATH_CALUDE_minimal_intercept_line_properties_l2574_257418

/-- A line that passes through (1, 4) with positive intercepts and minimal sum of intercepts -/
def minimal_intercept_line (x y : ℝ) : Prop :=
  x + y = 5

theorem minimal_intercept_line_properties :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  minimal_intercept_line 1 4 ∧
  (∀ x y, minimal_intercept_line x y → x = 0 ∨ y = 0 → x = a ∨ y = b) ∧
  (∀ c d : ℝ, c > 0 → d > 0 →
    (∃ x y, x + y = c + d ∧ (x = 0 ∨ y = 0)) →
    a + b ≤ c + d) :=
by sorry

end NUMINAMATH_CALUDE_minimal_intercept_line_properties_l2574_257418


namespace NUMINAMATH_CALUDE_student_minimum_earnings_l2574_257405

/-- Represents the student's work situation -/
structure WorkSituation where
  library_rate : ℝ
  construction_rate : ℝ
  total_hours : ℝ
  library_hours : ℝ

/-- Calculates the minimum weekly earnings for the student -/
def minimum_weekly_earnings (w : WorkSituation) : ℝ :=
  w.library_rate * w.library_hours + 
  w.construction_rate * (w.total_hours - w.library_hours)

/-- Theorem stating the minimum weekly earnings for the given work situation -/
theorem student_minimum_earnings : 
  let w : WorkSituation := {
    library_rate := 8,
    construction_rate := 15,
    total_hours := 25,
    library_hours := 10
  }
  minimum_weekly_earnings w = 305 := by sorry

end NUMINAMATH_CALUDE_student_minimum_earnings_l2574_257405


namespace NUMINAMATH_CALUDE_layla_fish_food_total_l2574_257456

/-- The total amount of food Layla needs to give her fish -/
def total_fish_food (goldfish_count : ℕ) (goldfish_food : ℚ) 
                    (swordtail_count : ℕ) (swordtail_food : ℚ) 
                    (guppy_count : ℕ) (guppy_food : ℚ) : ℚ :=
  goldfish_count * goldfish_food + swordtail_count * swordtail_food + guppy_count * guppy_food

/-- Theorem stating the total amount of food Layla needs to give her fish -/
theorem layla_fish_food_total : 
  total_fish_food 2 1 3 2 8 (1/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_layla_fish_food_total_l2574_257456


namespace NUMINAMATH_CALUDE_salary_calculation_l2574_257433

theorem salary_calculation (salary : ℚ) : 
  (salary * (1 - 0.2) * (1 - 0.1) * (1 - 0.1) = 1377) → 
  salary = 2125 := by
sorry

end NUMINAMATH_CALUDE_salary_calculation_l2574_257433


namespace NUMINAMATH_CALUDE_function_eq_zero_l2574_257443

theorem function_eq_zero (f : ℝ → ℝ) 
  (h1 : ∀ u v : ℝ, f (2 * u) = f (u + v) * f (v - u) + f (u - v) * f (-u - v))
  (h2 : ∀ u : ℝ, f u ≥ 0) :
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_function_eq_zero_l2574_257443


namespace NUMINAMATH_CALUDE_sqrt_last_digit_exists_l2574_257424

/-- A p-adic number -/
structure PAdicNumber (p : ℕ) where
  digits : ℕ → ℕ
  last_digit : ℕ

/-- The concept of square root in p-arithmetic -/
def has_sqrt_p_adic (α : PAdicNumber p) : Prop :=
  ∃ β : PAdicNumber p, β.digits 0 ^ 2 ≡ α.digits 0 [MOD p]

/-- The main theorem -/
theorem sqrt_last_digit_exists (p : ℕ) (α : PAdicNumber p) :
  has_sqrt_p_adic α → ∃ x : ℕ, x ^ 2 ≡ α.last_digit [MOD p] :=
sorry

end NUMINAMATH_CALUDE_sqrt_last_digit_exists_l2574_257424


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_half_l2574_257472

theorem angle_sum_is_pi_half (α β : Real) (h_acute_α : 0 < α ∧ α < π/2) (h_acute_β : 0 < β ∧ β < π/2) 
  (h_eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1) 
  (h_eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) : 
  α + 2 * β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_half_l2574_257472


namespace NUMINAMATH_CALUDE_brandon_squirrel_count_l2574_257471

/-- The number of squirrels Brandon can catch in an hour -/
def S : ℕ := sorry

/-- The number of rabbits Brandon can catch in an hour -/
def R : ℕ := 2

/-- The calorie content of a squirrel -/
def squirrel_calories : ℕ := 300

/-- The calorie content of a rabbit -/
def rabbit_calories : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits -/
def additional_calories : ℕ := 200

theorem brandon_squirrel_count :
  S * squirrel_calories = R * rabbit_calories + additional_calories ∧ S = 6 := by
  sorry

end NUMINAMATH_CALUDE_brandon_squirrel_count_l2574_257471


namespace NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_bisector_correct_l2574_257411

-- Define the points
def P : ℝ × ℝ := (-1, 3)
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the parallel line through P
def parallel_line (x y : ℝ) : Prop := x - 2*y + 7 = 0

-- Define the perpendicular bisector of AB
def perpendicular_bisector (x y : ℝ) : Prop := 4*x - 2*y - 5 = 0

-- Theorem 1: The parallel line passes through P and is parallel to the original line
theorem parallel_line_correct :
  parallel_line P.1 P.2 ∧
  ∀ (x y : ℝ), parallel_line x y ↔ ∃ (k : ℝ), original_line (x + k) (y + k/2) :=
sorry

-- Theorem 2: The perpendicular bisector is correct
theorem perpendicular_bisector_correct :
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  perpendicular_bisector midpoint.1 midpoint.2 ∧
  (B.2 - A.2) * (B.1 - A.1) * 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_correct_perpendicular_bisector_correct_l2574_257411


namespace NUMINAMATH_CALUDE_paper_edge_length_l2574_257416

theorem paper_edge_length (cube_edge : ℝ) (num_papers : ℕ) :
  cube_edge = 12 →
  num_papers = 54 →
  ∃ (paper_edge : ℝ),
    paper_edge^2 * num_papers = 6 * cube_edge^2 ∧
    paper_edge = 4 := by
  sorry

end NUMINAMATH_CALUDE_paper_edge_length_l2574_257416


namespace NUMINAMATH_CALUDE_is_center_of_hyperbola_l2574_257422

/-- The equation of a hyperbola in the form ((4y-8)^2 / 7^2) - ((2x+6)^2 / 9^2) = 1 -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * y - 8)^2 / 7^2 - (2 * x + 6)^2 / 9^2 = 1

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (-3, 2)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ x y : ℝ, hyperbola_equation x y ↔ hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) :=
by sorry

end NUMINAMATH_CALUDE_is_center_of_hyperbola_l2574_257422


namespace NUMINAMATH_CALUDE_difference_of_squares_division_l2574_257486

theorem difference_of_squares_division : (121^2 - 112^2) / 9 = 233 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_division_l2574_257486


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l2574_257497

/-- Given a line intersecting y = x^2 at (x₁, x₁²) and (x₂, x₂²), and the x-axis at (x₃, 0),
    prove that 1/x₁ + 1/x₂ = 1/x₃ -/
theorem line_parabola_intersection (x₁ x₂ x₃ : ℝ) (h₁ : x₁ ≠ 0) (h₂ : x₂ ≠ 0) (h₃ : x₃ ≠ 0)
  (h_line : ∃ (k m : ℝ), x₁^2 = k * x₁ + m ∧ x₂^2 = k * x₂ + m ∧ 0 = k * x₃ + m) :
  1 / x₁ + 1 / x₂ = 1 / x₃ := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l2574_257497


namespace NUMINAMATH_CALUDE_congruent_integers_count_l2574_257406

theorem congruent_integers_count : 
  (Finset.filter (fun n => n > 0 ∧ n < 2000 ∧ n % 13 = 6) (Finset.range 2000)).card = 154 :=
by sorry

end NUMINAMATH_CALUDE_congruent_integers_count_l2574_257406


namespace NUMINAMATH_CALUDE_rose_bushes_count_l2574_257400

/-- The number of rose bushes in the park after planting -/
def final_roses : ℕ := 6

/-- The number of new rose bushes to be planted -/
def new_roses : ℕ := 4

/-- The number of rose bushes currently in the park -/
def current_roses : ℕ := final_roses - new_roses

theorem rose_bushes_count : current_roses = 2 := by
  sorry

end NUMINAMATH_CALUDE_rose_bushes_count_l2574_257400


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2574_257414

def solution_set (x y : ℝ) : Prop :=
  (x ≤ -1 ∧ y ≤ x + 2 ∧ y ≥ -x - 2) ∨
  (-1 < x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1) ∨
  (x > 1 ∧ y ≤ 2 - x ∧ y ≥ x - 2)

theorem inequality_equivalence (x y : ℝ) :
  |x - 1| + |x + 1| + |2 * y| ≤ 4 ↔ solution_set x y := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2574_257414


namespace NUMINAMATH_CALUDE_expression_value_l2574_257488

theorem expression_value (x : ℝ) (h : x = 5) : 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2574_257488


namespace NUMINAMATH_CALUDE_remainder_2n_mod_4_l2574_257467

theorem remainder_2n_mod_4 (n : ℕ) (h : n % 4 = 3) : (2 * n) % 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2n_mod_4_l2574_257467


namespace NUMINAMATH_CALUDE_incorrect_fraction_equality_l2574_257442

theorem incorrect_fraction_equality (a b : ℝ) (h : 0.7 * a ≠ b) :
  (0.2 * a + b) / (0.7 * a - b) ≠ (2 * a + b) / (7 * a - b) :=
sorry

end NUMINAMATH_CALUDE_incorrect_fraction_equality_l2574_257442


namespace NUMINAMATH_CALUDE_base8_addition_l2574_257430

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10_to_base8 (n : ℕ) : ℕ := sorry

/-- Addition in base-8 --/
def add_base8 (a b c : ℕ) : ℕ :=
  base10_to_base8 (base8_to_base10 a + base8_to_base10 b + base8_to_base10 c)

theorem base8_addition :
  add_base8 246 573 62 = 1123 := by sorry

end NUMINAMATH_CALUDE_base8_addition_l2574_257430


namespace NUMINAMATH_CALUDE_smallest_base_for_repeating_decimal_l2574_257484

/-- Represents a repeating decimal in base k -/
def RepeatingDecimal (k : ℕ) (n : ℕ) := (k : ℚ) ^ 2 / ((k : ℚ) ^ 2 - 1) * (4 * k + 1)

/-- The smallest integer k > 10 such that 17/85 has a repeating decimal representation of 0.414141... in base k -/
theorem smallest_base_for_repeating_decimal :
  ∃ (k : ℕ), k > 10 ∧ RepeatingDecimal k 2 = 17 / 85 ∧
  ∀ (m : ℕ), m > 10 ∧ m < k → RepeatingDecimal m 2 ≠ 17 / 85 := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_for_repeating_decimal_l2574_257484


namespace NUMINAMATH_CALUDE_fayes_initial_money_l2574_257437

/-- Proves that Faye's initial amount of money was $20 --/
theorem fayes_initial_money :
  ∀ (X : ℝ),
  (X + 2*X - (10*1.5 + 5*3) = 30) →
  X = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_fayes_initial_money_l2574_257437


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2574_257455

theorem fraction_inequality_solution_set :
  {x : ℝ | (x + 1) / (x + 2) < 0} = {x : ℝ | -2 < x ∧ x < -1} :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l2574_257455


namespace NUMINAMATH_CALUDE_park_outer_diameter_l2574_257427

/-- Represents the dimensions of a circular park -/
structure CircularPark where
  pond_diameter : ℝ
  garden_width : ℝ
  path_width : ℝ

/-- Calculates the diameter of the outer boundary of a circular park -/
def outer_boundary_diameter (park : CircularPark) : ℝ :=
  park.pond_diameter + 2 * (park.garden_width + park.path_width)

/-- Theorem stating that for a park with given dimensions, the outer boundary diameter is 60 feet -/
theorem park_outer_diameter (park : CircularPark) 
  (h1 : park.pond_diameter = 16)
  (h2 : park.garden_width = 12)
  (h3 : park.path_width = 10) : 
  outer_boundary_diameter park = 60 := by sorry

end NUMINAMATH_CALUDE_park_outer_diameter_l2574_257427


namespace NUMINAMATH_CALUDE_ellipse_arithmetic_sequence_eccentricity_l2574_257458

/-- An ellipse with focal length, minor axis length, and major axis length in arithmetic sequence has eccentricity 3/5 -/
theorem ellipse_arithmetic_sequence_eccentricity :
  ∀ (a b c : ℝ),
    a > b ∧ b > 0 →  -- Conditions for a valid ellipse
    b = (a + c) / 2 →  -- Arithmetic sequence condition
    c^2 = a^2 - b^2 →  -- Relation between focal length and axes lengths
    c / a = 3 / 5 :=
by sorry


end NUMINAMATH_CALUDE_ellipse_arithmetic_sequence_eccentricity_l2574_257458


namespace NUMINAMATH_CALUDE_product_of_roots_quadratic_l2574_257419

theorem product_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 + 2*x₁ - 4 = 0) → (x₂^2 + 2*x₂ - 4 = 0) → x₁ * x₂ = -4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_quadratic_l2574_257419


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2574_257495

-- Define the set M
def M : Set ℝ := {x | 0 < x ∧ x < 27}

-- Define the set N
def N : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem statement
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 0 5 := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2574_257495


namespace NUMINAMATH_CALUDE_student_marks_l2574_257440

theorem student_marks (M P C : ℕ) : 
  C = P + 20 →
  (M + C) / 2 = 25 →
  M + P = 30 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_l2574_257440


namespace NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l2574_257450

theorem last_two_digits_of_seven_power (n : ℕ) : 7^(5^6) ≡ 7 [MOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l2574_257450


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2574_257402

theorem right_triangle_side_length (A B C : Real) (tanA : Real) (AC : Real) :
  tanA = 3 / 5 →
  AC = 10 →
  A^2 + B^2 = C^2 →
  A / C = tanA →
  B = 2 * Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2574_257402


namespace NUMINAMATH_CALUDE_game_configurations_l2574_257461

/-- Represents the game state -/
structure GameState where
  blackPosition : Nat
  whiteCheckers : Nat

/-- The game rules -/
def gameRules : Prop :=
  ∃ (initialState : GameState),
    initialState.blackPosition = 3 ∧
    initialState.whiteCheckers = 2 ∧
    ∀ (moves : Nat),
      moves ≤ 2008 →
      ∃ (finalState : GameState),
        finalState.blackPosition = initialState.blackPosition + moves ∧
        finalState.whiteCheckers ≥ initialState.whiteCheckers ∧
        finalState.whiteCheckers ≤ initialState.whiteCheckers + moves

/-- The theorem to be proved -/
theorem game_configurations (rules : gameRules) :
  ∃ (finalConfigurations : Nat),
    finalConfigurations = 2009 ∧
    ∀ (state : GameState),
      state.blackPosition = 2011 →
      state.whiteCheckers ≥ 2 ∧
      state.whiteCheckers ≤ 2010 :=
sorry

end NUMINAMATH_CALUDE_game_configurations_l2574_257461


namespace NUMINAMATH_CALUDE_perfect_cube_from_sum_l2574_257491

theorem perfect_cube_from_sum (a b c : ℤ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) 
  (h_sum : ∃ (n : ℤ), a / b + b / c + c / a = n) : 
  ∃ (m : ℤ), a * b * c = m^3 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_from_sum_l2574_257491


namespace NUMINAMATH_CALUDE_theater_revenue_l2574_257457

/-- The total number of tickets sold -/
def total_tickets : ℕ := 800

/-- The price of an advanced ticket in cents -/
def advanced_price : ℕ := 1450

/-- The price of a door ticket in cents -/
def door_price : ℕ := 2200

/-- The number of tickets sold at the door -/
def door_tickets : ℕ := 672

/-- The total money taken in cents -/
def total_money : ℕ := 1664000

theorem theater_revenue :
  total_money = 
    (total_tickets - door_tickets) * advanced_price +
    door_tickets * door_price :=
by sorry

end NUMINAMATH_CALUDE_theater_revenue_l2574_257457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2574_257413

/-- An arithmetic sequence with given properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_a5 : a 5 = 8)
  (h_a9 : a 9 = 24) :
  ∃ (d : ℝ), d = 4 ∧ ∀ n, a n = a 1 + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2574_257413


namespace NUMINAMATH_CALUDE_jeanette_juggling_progress_l2574_257470

/-- Calculates the number of objects Jeanette can juggle after a given number of weeks -/
def juggle_objects (initial_objects : ℕ) (weekly_increase : ℕ) (sessions_per_week : ℕ) (session_increase : ℕ) (weeks : ℕ) : ℕ :=
  initial_objects + weeks * (weekly_increase + sessions_per_week * session_increase)

/-- Proves that Jeanette can juggle 21 objects by the end of the 5th week -/
theorem jeanette_juggling_progress : 
  juggle_objects 3 2 3 1 5 = 21 := by
  sorry

#eval juggle_objects 3 2 3 1 5

end NUMINAMATH_CALUDE_jeanette_juggling_progress_l2574_257470


namespace NUMINAMATH_CALUDE_range_of_m_for_equation_l2574_257494

/-- Given that the equation e^(mx) = x^2 has two distinct real roots in the interval (0, 16),
    prove that the range of values for the real number m is (ln(2)/2, 2/e). -/
theorem range_of_m_for_equation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 16 ∧ 
   Real.exp (m * x₁) = x₁^2 ∧ Real.exp (m * x₂) = x₂^2) →
  (Real.log 2 / 2 < m ∧ m < 2 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_equation_l2574_257494


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2574_257493

theorem trigonometric_simplification (α : ℝ) : 
  (1 + Real.tan (2 * α))^2 - 2 * (Real.tan (2 * α))^2 / (1 + (Real.tan (2 * α))^2) - 
  Real.sin (4 * α) - 1 = -2 * (Real.sin (2 * α))^2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2574_257493
