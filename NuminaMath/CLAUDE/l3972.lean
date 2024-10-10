import Mathlib

namespace symmetric_points_on_ellipse_l3972_397273

/-- Given an ellipse C and a line l, prove the range of m for which there are always two points on C symmetric with respect to l -/
theorem symmetric_points_on_ellipse (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 / 4 + y₁^2 / 3 = 1) ∧ 
    (x₂^2 / 4 + y₂^2 / 3 = 1) ∧
    (y₁ = 4*x₁ + m) ∧ 
    (y₂ = 4*x₂ + m) ∧ 
    (x₁ ≠ x₂) ∧
    (∃ (x₀ y₀ : ℝ), x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2 ∧ y₀ = 4*x₀ + m)) ↔ 
  (-2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13) :=
sorry

end symmetric_points_on_ellipse_l3972_397273


namespace restaurant_ratio_proof_l3972_397282

/-- Proves that the original ratio of cooks to waiters was 1:3 given the conditions -/
theorem restaurant_ratio_proof (cooks : ℕ) (waiters : ℕ) :
  cooks = 9 →
  (cooks : ℚ) / (waiters + 12 : ℚ) = 1 / 5 →
  (cooks : ℚ) / (waiters : ℚ) = 1 / 3 := by
  sorry

end restaurant_ratio_proof_l3972_397282


namespace inequality_solution_system_of_inequalities_solution_l3972_397275

-- Part 1
theorem inequality_solution (x : ℝ) :
  (1/3 * x - (3*x + 4)/6 ≤ 2/3) ↔ (x ≥ -8) :=
sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (4*(x+1) ≤ 7*x + 13) ∧ ((x+2)/3 - x/2 > 1) ↔ (-3 ≤ x ∧ x < -2) :=
sorry

end inequality_solution_system_of_inequalities_solution_l3972_397275


namespace bowling_pins_difference_l3972_397286

theorem bowling_pins_difference (patrick_first : ℕ) (richard_first_diff : ℕ) (richard_second_diff : ℕ) : 
  patrick_first = 70 →
  richard_first_diff = 15 →
  richard_second_diff = 3 →
  (patrick_first + richard_first_diff + (2 * (patrick_first + richard_first_diff) - richard_second_diff)) -
  (patrick_first + 2 * (patrick_first + richard_first_diff)) = 12 := by
  sorry

end bowling_pins_difference_l3972_397286


namespace simplify_fraction_l3972_397260

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := by sorry

end simplify_fraction_l3972_397260


namespace cube_volume_from_face_diagonal_l3972_397252

theorem cube_volume_from_face_diagonal (d : ℝ) (h : d = 5 * Real.sqrt 2) :
  ∃ (s : ℝ), s > 0 ∧ s * Real.sqrt 2 = d ∧ s^3 = 125 := by
  sorry

end cube_volume_from_face_diagonal_l3972_397252


namespace expression_equality_l3972_397232

theorem expression_equality : (1/2)⁻¹ + (Real.pi + 2023)^0 - 2 * Real.cos (Real.pi / 3) + Real.sqrt 9 = 5 := by
  sorry

end expression_equality_l3972_397232


namespace hall_length_is_30_l3972_397225

/-- Represents a rectangular hall with specific properties -/
structure RectangularHall where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_breadth_relation : length = breadth + 5
  area_formula : area = length * breadth

/-- Theorem stating that a rectangular hall with the given properties has a length of 30 meters -/
theorem hall_length_is_30 (hall : RectangularHall) (h : hall.area = 750) : hall.length = 30 := by
  sorry

#check hall_length_is_30

end hall_length_is_30_l3972_397225


namespace six_power_plus_one_all_digits_same_l3972_397229

/-- A number has all digits the same in its decimal representation -/
def all_digits_same (m : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ ∀ k : ℕ, (m / 10^k) % 10 = d ∨ m / 10^k = 0

/-- The set of positive integers n for which 6^n + 1 has all digits the same -/
def S : Set ℕ := {n : ℕ | n > 0 ∧ all_digits_same (6^n + 1)}

theorem six_power_plus_one_all_digits_same :
  S = {1, 5} :=
sorry

end six_power_plus_one_all_digits_same_l3972_397229


namespace average_speed_calculation_l3972_397289

theorem average_speed_calculation (distance1 distance2 speed1 speed2 : ℝ) 
  (h1 : distance1 = 20)
  (h2 : distance2 = 40)
  (h3 : speed1 = 8)
  (h4 : speed2 = 20) :
  (distance1 + distance2) / (distance1 / speed1 + distance2 / speed2) = 40/3 :=
sorry

end average_speed_calculation_l3972_397289


namespace complement_A_intersect_B_equals_result_l3972_397220

-- Define the universe set U
def U : Set ℤ := {x : ℤ | x^2 - x - 12 ≤ 0}

-- Define set A
def A : Set ℤ := {-2, -1, 3}

-- Define set B
def B : Set ℤ := {0, 1, 3, 4}

-- Define the result set
def result : Set ℤ := {0, 1, 4}

-- Theorem statement
theorem complement_A_intersect_B_equals_result :
  (U \ A) ∩ B = result := by sorry

end complement_A_intersect_B_equals_result_l3972_397220


namespace similar_triangle_perimeter_l3972_397233

/-- Given a right triangle with sides 15 and 20, similar to a larger triangle
    where one side is twice a rectangle's shorter side (30), 
    prove the perimeter of the larger triangle is 240. -/
theorem similar_triangle_perimeter : 
  ∀ (small_triangle large_triangle : Set ℝ) 
    (rectangle : Set (ℝ × ℝ)),
  (∃ a b c : ℝ, small_triangle = {a, b, c} ∧ 
    a = 15 ∧ b = 20 ∧ c^2 = a^2 + b^2) →
  (∃ x y : ℝ, rectangle = {(30, 60), (x, y)}) →
  (∃ d e f : ℝ, large_triangle = {d, e, f} ∧
    d = 2 * 30 ∧ 
    (d / 15 = e / 20 ∧ d / 15 = f / (15^2 + 20^2).sqrt)) →
  (∃ p : ℝ, p = d + e + f ∧ p = 240) :=
by sorry

end similar_triangle_perimeter_l3972_397233


namespace total_intersection_points_l3972_397244

/-- Regular polygon inscribed in a circle -/
structure RegularPolygon where
  sides : ℕ
  inscribed : Bool

/-- Represents the configuration of regular polygons in a circle -/
structure PolygonConfiguration where
  square : RegularPolygon
  hexagon : RegularPolygon
  octagon : RegularPolygon
  shared_vertices : ℕ
  no_triple_intersections : Bool

/-- Calculates the number of intersection points between two polygons -/
def intersectionPoints (p1 p2 : RegularPolygon) (shared : Bool) : ℕ :=
  sorry

/-- Theorem stating the total number of intersection points -/
theorem total_intersection_points (config : PolygonConfiguration) : 
  config.square.sides = 4 ∧ 
  config.hexagon.sides = 6 ∧ 
  config.octagon.sides = 8 ∧
  config.square.inscribed ∧
  config.hexagon.inscribed ∧
  config.octagon.inscribed ∧
  config.shared_vertices ≤ 3 ∧
  config.no_triple_intersections →
  intersectionPoints config.square config.hexagon (config.shared_vertices > 0) +
  intersectionPoints config.square config.octagon (config.shared_vertices > 1) +
  intersectionPoints config.hexagon config.octagon (config.shared_vertices > 2) = 164 :=
sorry

end total_intersection_points_l3972_397244


namespace equation_equivalence_l3972_397245

theorem equation_equivalence (x y : ℝ) : 
  (5 * x + y = 1) ↔ (y = 1 - 5 * x) := by sorry

end equation_equivalence_l3972_397245


namespace function_value_at_negative_a_l3972_397219

/-- Given a function f(x) = ax³ + bx, prove that if f(a) = 8, then f(-a) = -8 -/
theorem function_value_at_negative_a (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x
  f a = 8 → f (-a) = -8 := by
  sorry

end function_value_at_negative_a_l3972_397219


namespace arithmetic_sequence_proof_l3972_397203

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℚ := (n^2 + 3*n) / 2

-- Define the general term of the arithmetic sequence
def a (n : ℕ) : ℚ := n + 1

-- Define the terms of the sequence b_n
def b (n : ℕ) : ℚ := 1 / (a (2*n - 1) * a (2*n + 1))

-- Define the sum of the first n terms of the sequence b_n
def T (n : ℕ) : ℚ := n / (4*n + 4)

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → a n = n + 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = n / (4*n + 4)) :=
by sorry

end arithmetic_sequence_proof_l3972_397203


namespace apple_distribution_theorem_l3972_397274

/-- Represents the distribution of apples in bags -/
structure AppleDistribution where
  totalApples : Nat
  totalBags : Nat
  xApples : Nat
  threeAppleBags : Nat
  xAppleBags : Nat

/-- Checks if the apple distribution is valid -/
def isValidDistribution (d : AppleDistribution) : Prop :=
  d.totalApples = 109 ∧
  d.totalBags = 20 ∧
  d.threeAppleBags + d.xAppleBags = d.totalBags ∧
  d.xApples * d.xAppleBags + 3 * d.threeAppleBags = d.totalApples

/-- Theorem stating the possible values of x -/
theorem apple_distribution_theorem :
  ∀ d : AppleDistribution,
    isValidDistribution d →
    d.xApples = 10 ∨ d.xApples = 52 :=
by sorry

end apple_distribution_theorem_l3972_397274


namespace sum_three_numbers_l3972_397213

theorem sum_three_numbers (a b c M : ℚ) : 
  a + b + c = 120 ∧ 
  a - 9 = M ∧ 
  b + 9 = M ∧ 
  9 * c = M → 
  M = 1080 / 19 := by
sorry

end sum_three_numbers_l3972_397213


namespace scorpion_millipede_calculation_l3972_397216

/-- Calculates the number of additional millipedes needed to reach a daily segment requirement -/
theorem scorpion_millipede_calculation 
  (daily_requirement : ℕ) 
  (eaten_millipede_segments : ℕ) 
  (eaten_long_millipedes : ℕ) 
  (additional_millipede_segments : ℕ) 
  (h1 : daily_requirement = 800)
  (h2 : eaten_millipede_segments = 60)
  (h3 : eaten_long_millipedes = 2)
  (h4 : additional_millipede_segments = 50) :
  (daily_requirement - (eaten_millipede_segments + eaten_long_millipedes * eaten_millipede_segments * 2)) / additional_millipede_segments = 10 := by
  sorry

end scorpion_millipede_calculation_l3972_397216


namespace sqrt_product_simplification_l3972_397214

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (12 * q) * Real.sqrt (8 * q^3) * Real.sqrt (9 * q^5) = 6 * q^4 * Real.sqrt (6 * q) := by
  sorry

end sqrt_product_simplification_l3972_397214


namespace chord_length_l3972_397263

/-- The length of the chord intercepted by a line on a circle -/
theorem chord_length (t : ℝ) : 
  let line : ℝ → ℝ × ℝ := λ t => (1 + 2*t, 2 + t)
  let circle : ℝ × ℝ → Prop := λ p => p.1^2 + p.2^2 = 9
  let chord_length := 
    Real.sqrt (4 * (9 - (3 / Real.sqrt 5)^2))
  chord_length = 12/5 * Real.sqrt 5 := by
  sorry

end chord_length_l3972_397263


namespace cos_squared_30_minus_2_minus_pi_to_0_l3972_397297

theorem cos_squared_30_minus_2_minus_pi_to_0 :
  Real.cos (30 * π / 180) ^ 2 - (2 - π) ^ 0 = -1/4 := by
  sorry

end cos_squared_30_minus_2_minus_pi_to_0_l3972_397297


namespace f_min_value_l3972_397256

/-- The function f(x) defined as (x+1)(x+2)(x+3)(x+4) + 35 -/
def f (x : ℝ) : ℝ := (x+1)*(x+2)*(x+3)*(x+4) + 35

/-- Theorem stating that the minimum value of f(x) is 34 -/
theorem f_min_value : ∀ x : ℝ, f x ≥ 34 ∧ ∃ x₀ : ℝ, f x₀ = 34 :=
sorry

end f_min_value_l3972_397256


namespace age_difference_is_54_l3972_397257

/-- Represents a person's age with tens and units digits -/
structure Age where
  tens : Nat
  units : Nat
  tens_nonzero : tens ≠ 0

/-- The problem statement -/
theorem age_difference_is_54 
  (jack : Age) 
  (bill : Age) 
  (h1 : jack.tens * 10 + jack.units + 10 = 3 * (bill.tens * 10 + bill.units + 10))
  (h2 : jack.tens = bill.units ∧ jack.units = bill.tens) :
  (jack.tens * 10 + jack.units) - (bill.tens * 10 + bill.units) = 54 := by
  sorry

end age_difference_is_54_l3972_397257


namespace johns_hats_cost_l3972_397237

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_per_week * cost_per_hat

theorem johns_hats_cost :
  total_cost = 700 :=
by sorry

end johns_hats_cost_l3972_397237


namespace photos_per_page_l3972_397250

theorem photos_per_page (total_photos : ℕ) (total_pages : ℕ) (h1 : total_photos = 736) (h2 : total_pages = 122) :
  total_photos / total_pages = 6 := by
  sorry

end photos_per_page_l3972_397250


namespace square_sum_xy_l3972_397266

theorem square_sum_xy (x y a b : ℝ) 
  (h1 : x * y = b^2) 
  (h2 : 1 / x^2 + 1 / y^2 = a) : 
  (x + y)^2 = a * b^4 + 2 * b^2 := by
  sorry

end square_sum_xy_l3972_397266


namespace line_not_in_plane_necessary_not_sufficient_l3972_397240

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- State the theorem
theorem line_not_in_plane_necessary_not_sufficient
  (a b : Line) (α : Plane)
  (h : contained_in a α) :
  (¬ contained_in b α ∧ ¬ (∀ b, ¬ contained_in b α → skew a b)) ∧
  (∀ b, skew a b → ¬ contained_in b α) := by
sorry

end line_not_in_plane_necessary_not_sufficient_l3972_397240


namespace infinitely_many_solutions_l3972_397247

theorem infinitely_many_solutions (a b : ℤ) (h_coprime : Nat.Coprime a.natAbs b.natAbs) :
  ∃ (S : Set (ℤ × ℤ × ℤ)), Set.Infinite S ∧
    ∀ (x y z : ℤ), (x, y, z) ∈ S →
      a * x^2 + b * y^2 = z^3 ∧ Nat.Coprime x.natAbs y.natAbs :=
by sorry


end infinitely_many_solutions_l3972_397247


namespace sector_central_angle_l3972_397228

theorem sector_central_angle (perimeter : ℝ) (area : ℝ) (angle : ℝ) : 
  perimeter = 4 → area = 1 → angle = 2 := by sorry

end sector_central_angle_l3972_397228


namespace susan_menu_fraction_l3972_397218

theorem susan_menu_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (vegan_with_nuts : ℕ) : 
  vegan_dishes = total_dishes / 3 →
  vegan_dishes = 6 →
  vegan_with_nuts = 4 →
  (vegan_dishes - vegan_with_nuts : ℚ) / total_dishes = 1 / 9 := by
  sorry

end susan_menu_fraction_l3972_397218


namespace polynomial_value_l3972_397284

theorem polynomial_value : 
  let x : ℚ := 1/2
  2*x^2 - 5*x + x^2 + 4*x - 3*x^2 - 2 = -5/2 := by sorry

end polynomial_value_l3972_397284


namespace base6_243_equals_base10_99_l3972_397267

/-- Converts a base 6 number represented as a list of digits to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

theorem base6_243_equals_base10_99 :
  base6ToBase10 [3, 4, 2] = 99 := by
  sorry

end base6_243_equals_base10_99_l3972_397267


namespace max_min_sum_of_f_l3972_397241

noncomputable def f (x : ℝ) : ℝ := ((x + 1)^2 + Real.sin x) / (x^2 + 1)

theorem max_min_sum_of_f :
  ∃ (M m : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧
               (∀ x, m ≤ f x) ∧ (∃ x, f x = m) ∧
               M + m = 2 :=
sorry

end max_min_sum_of_f_l3972_397241


namespace alice_probability_after_two_turns_l3972_397212

/-- Represents the probability of Alice having the ball after two turns in the basketball game. -/
def alice_has_ball_after_two_turns (
  alice_toss_prob : ℚ)  -- Probability of Alice tossing the ball to Bob
  (alice_keep_prob : ℚ)  -- Probability of Alice keeping the ball
  (bob_toss_prob : ℚ)    -- Probability of Bob tossing the ball to Alice
  (bob_keep_prob : ℚ) : ℚ :=  -- Probability of Bob keeping the ball
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob

/-- Theorem stating the probability of Alice having the ball after two turns -/
theorem alice_probability_after_two_turns :
  alice_has_ball_after_two_turns (2/3) (1/3) (1/4) (3/4) = 5/18 := by
  sorry

end alice_probability_after_two_turns_l3972_397212


namespace value_of_x_l3972_397255

theorem value_of_x (z y x : ℚ) : z = 48 → y = z / 4 → x = y / 3 → x = 4 := by
  sorry

end value_of_x_l3972_397255


namespace water_volume_is_fifty_l3972_397298

/-- A cubical tank partially filled with water -/
structure CubicalTank where
  side_length : ℝ
  water_level : ℝ
  capacity_fraction : ℝ

/-- The volume of water in the tank -/
def water_volume (tank : CubicalTank) : ℝ :=
  tank.capacity_fraction * tank.side_length^3

theorem water_volume_is_fifty (tank : CubicalTank) 
  (h1 : tank.water_level = 2)
  (h2 : tank.capacity_fraction = 0.4)
  (h3 : tank.water_level = tank.capacity_fraction * tank.side_length) :
  water_volume tank = 50 := by
  sorry

end water_volume_is_fifty_l3972_397298


namespace price_per_small_bottle_l3972_397208

/-- Calculates the price per small bottle given the number of large and small bottles,
    the price of large bottles, and the average price of all bottles. -/
theorem price_per_small_bottle
  (num_large : ℕ)
  (num_small : ℕ)
  (price_large : ℚ)
  (avg_price : ℚ)
  (h1 : num_large = 1325)
  (h2 : num_small = 750)
  (h3 : price_large = 189/100)
  (h4 : avg_price = 17057/10000) :
  ∃ (price_small : ℚ),
    abs (price_small - 13828/10000) < 1/10000 ∧
    (num_large * price_large + num_small * price_small) / (num_large + num_small) = avg_price :=
sorry

end price_per_small_bottle_l3972_397208


namespace complex_number_in_third_quadrant_l3972_397204

theorem complex_number_in_third_quadrant :
  let z : ℂ := ((-1 : ℂ) - 2*I) / (1 - 2*I)
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l3972_397204


namespace probability_prime_sum_three_dice_l3972_397246

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The set of possible prime sums when rolling three 6-sided dice -/
def primeSums : Set ℕ := {3, 5, 7, 11, 13, 17}

/-- The total number of possible outcomes when rolling three 6-sided dice -/
def totalOutcomes : ℕ := numSides ^ 3

/-- The number of ways to roll a prime sum with three 6-sided dice -/
def primeOutcomes : ℕ := 58

/-- The probability of rolling a prime sum with three 6-sided dice -/
theorem probability_prime_sum_three_dice :
  (primeOutcomes : ℚ) / totalOutcomes = 58 / 216 := by
  sorry


end probability_prime_sum_three_dice_l3972_397246


namespace sum_of_powers_implies_sum_power_l3972_397278

theorem sum_of_powers_implies_sum_power (a b : ℝ) :
  a^2009 + b^2009 = 0 → (a + b)^2009 = 0 := by sorry

end sum_of_powers_implies_sum_power_l3972_397278


namespace polynomial_expansion_l3972_397259

theorem polynomial_expansion (w : ℝ) : 
  (3 * w^3 + 4 * w^2 - 7) * (2 * w^3 - 3 * w^2 + 1) = 
  6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 := by
  sorry

end polynomial_expansion_l3972_397259


namespace systematic_sample_validity_l3972_397283

def is_valid_systematic_sample (sample : List Nat) (population_size : Nat) (sample_size : Nat) : Prop :=
  sample.length = sample_size ∧
  (∀ i j, i < j → i < sample.length → j < sample.length → 
    sample[i]! < sample[j]! ∧ 
    (sample[j]! - sample[i]!) = (population_size / sample_size) * (j - i)) ∧
  (∀ n, n ∈ sample → n < population_size)

theorem systematic_sample_validity :
  is_valid_systematic_sample [1, 11, 21, 31, 41] 50 5 :=
by sorry

end systematic_sample_validity_l3972_397283


namespace prime_differences_l3972_397217

theorem prime_differences (x y : ℝ) 
  (h1 : Prime (x - y))
  (h2 : Prime (x^2 - y^2))
  (h3 : Prime (x^3 - y^3)) :
  x - y = 3 := by
sorry

end prime_differences_l3972_397217


namespace lincoln_county_houses_l3972_397271

/-- The total number of houses in Lincoln County after the housing boom -/
def total_houses (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem: The total number of houses in Lincoln County after the housing boom is 118558 -/
theorem lincoln_county_houses :
  total_houses 20817 97741 = 118558 := by
  sorry

end lincoln_county_houses_l3972_397271


namespace min_value_cubic_expression_l3972_397223

theorem min_value_cubic_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) ≥ 4 ∧
  (8 * a^3 + 12 * b^3 + 27 * c^3 + 1 / (9 * a * b * c) = 4 ↔ 
    a = 1 / Real.rpow 8 (1/3) ∧ b = 1 / Real.rpow 12 (1/3) ∧ c = 1 / Real.rpow 27 (1/3)) :=
by sorry

end min_value_cubic_expression_l3972_397223


namespace oil_usage_l3972_397276

theorem oil_usage (rons_oil sara_usage : ℚ) 
  (h1 : rons_oil = 3/8)
  (h2 : sara_usage = 5/6 * rons_oil) : 
  sara_usage = 5/16 := by
  sorry

end oil_usage_l3972_397276


namespace trig_identity_proof_l3972_397293

theorem trig_identity_proof :
  Real.cos (14 * π / 180) * Real.cos (59 * π / 180) +
  Real.sin (14 * π / 180) * Real.sin (121 * π / 180) =
  Real.sqrt 2 / 2 := by
sorry

end trig_identity_proof_l3972_397293


namespace class_size_l3972_397268

/-- The number of students in a class with French and German courses -/
def total_students (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (french - both) + (german - both) + both + neither

/-- Theorem: The total number of students in the class is 87 -/
theorem class_size : total_students 41 22 9 33 = 87 := by
  sorry

end class_size_l3972_397268


namespace floor_abs_sum_equals_eleven_l3972_397200

theorem floor_abs_sum_equals_eleven :
  ⌊|(-5.7 : ℝ)|⌋ + |⌊(-5.7 : ℝ)⌋| = 11 := by
  sorry

end floor_abs_sum_equals_eleven_l3972_397200


namespace expand_expression_l3972_397234

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 5) = x^4 - 4*x^2 - 45 := by
  sorry

end expand_expression_l3972_397234


namespace consecutive_sum_iff_not_power_of_two_l3972_397280

/-- A positive integer n cannot be represented as a sum of two or more consecutive integers
    if and only if n is a power of 2. -/
theorem consecutive_sum_iff_not_power_of_two (n : ℕ) (hn : n > 0) :
  (∃ (k m : ℕ), k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ ¬(∃ i : ℕ, n = 2^i) :=
by sorry

end consecutive_sum_iff_not_power_of_two_l3972_397280


namespace tangent_line_at_negative_one_l3972_397215

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - 3*x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 4*x^3 - 6*x

-- Theorem statement
theorem tangent_line_at_negative_one :
  ∃ (m b : ℝ), 
    (f' (-1) = m) ∧ 
    (f (-1) = -2) ∧ 
    (∀ x y : ℝ, y = m * (x + 1) - 2 ↔ m * x - y + b = 0) ∧
    (2 * x - y = 0 ↔ m * x - y + b = 0) :=
by sorry

end tangent_line_at_negative_one_l3972_397215


namespace A_eq_ge_1989_l3972_397242

/-- The set of functions f: ℕ → ℕ satisfying f(f(x)) - 2f(x) + x = 0 for all x ∈ ℕ -/
def F : Set (ℕ → ℕ) :=
  {f | ∀ x : ℕ, f (f x) - 2 * f x + x = 0}

/-- The set A = {f(1989) | f ∈ F} -/
def A : Set ℕ :=
  {y | ∃ f ∈ F, f 1989 = y}

/-- Theorem stating that A is equal to {k : k ≥ 1989} -/
theorem A_eq_ge_1989 : A = {k : ℕ | k ≥ 1989} := by
  sorry


end A_eq_ge_1989_l3972_397242


namespace path_length_is_twenty_l3972_397295

/-- A circle with diameter AB and points C, D on AB, and P on the circle. -/
structure CircleWithPoints where
  /-- The diameter of the circle -/
  diameter : ℝ
  /-- The distance from A to C -/
  ac_distance : ℝ
  /-- The distance from B to D -/
  bd_distance : ℝ

/-- The length of the broken-line path CPD when P is at B -/
def path_length (circle : CircleWithPoints) : ℝ :=
  circle.ac_distance + circle.diameter + circle.bd_distance

/-- Theorem stating that the path length is 20 units for the given conditions -/
theorem path_length_is_twenty (circle : CircleWithPoints)
  (h1 : circle.diameter = 12)
  (h2 : circle.ac_distance = 3)
  (h3 : circle.bd_distance = 5) :
  path_length circle = 20 := by
  sorry

end path_length_is_twenty_l3972_397295


namespace coffee_percentage_contribution_l3972_397209

def pancake_price : ℚ := 4
def bacon_price : ℚ := 2
def egg_price : ℚ := 3/2
def coffee_price : ℚ := 1

def pancake_sold : ℕ := 60
def bacon_sold : ℕ := 90
def egg_sold : ℕ := 75
def coffee_sold : ℕ := 50

def total_sales : ℚ := 
  pancake_price * pancake_sold + 
  bacon_price * bacon_sold + 
  egg_price * egg_sold + 
  coffee_price * coffee_sold

def coffee_contribution : ℚ := coffee_price * coffee_sold / total_sales

theorem coffee_percentage_contribution : 
  coffee_contribution * 100 = 858/100 := by sorry

end coffee_percentage_contribution_l3972_397209


namespace deadlift_percentage_increase_l3972_397281

/-- Bobby's initial deadlift at age 13 in pounds -/
def initial_deadlift : ℝ := 300

/-- Bobby's annual deadlift increase in pounds -/
def annual_increase : ℝ := 110

/-- Number of years between age 13 and 18 -/
def years : ℕ := 5

/-- Bobby's deadlift at age 18 in pounds -/
def deadlift_at_18 : ℝ := initial_deadlift + (annual_increase * years)

/-- The percentage increase we're looking for -/
def P : ℝ := sorry

/-- Theorem stating the relationship between Bobby's deadlift at 18 and the percentage increase -/
theorem deadlift_percentage_increase : deadlift_at_18 * (1 + P / 100) = deadlift_at_18 + 100 := by
  sorry

end deadlift_percentage_increase_l3972_397281


namespace max_value_of_f_l3972_397272

def f (x : ℝ) : ℝ := -3 * (x - 2)^2 - 3

theorem max_value_of_f :
  ∃ (m : ℝ), m = -3 ∧ ∀ (x : ℝ), f x ≤ m := by sorry

end max_value_of_f_l3972_397272


namespace not_p_sufficient_not_necessary_for_not_q_l3972_397254

-- Define the conditions p and q
def p (x : ℝ) : Prop := abs (x + 1) > 2
def q (x : ℝ) : Prop := x ≥ 2

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  ¬(∀ x : ℝ, not_q x → not_p x) :=
sorry

end not_p_sufficient_not_necessary_for_not_q_l3972_397254


namespace expression_value_l3972_397235

theorem expression_value (x : ℝ) (h : 4 * x^2 - 2 * x + 5 = 7) :
  2 * (x^2 - x) - (x - 1) + (2 * x + 3) = 5 := by
  sorry

end expression_value_l3972_397235


namespace polynomial_equality_l3972_397291

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x + Real.sqrt 2) ^ 4 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 + a₄ * x ^ 4) →
  (a₀ + a₂ + a₄) ^ 2 - (a₁ + a₃) ^ 2 = 1 := by
  sorry

end polynomial_equality_l3972_397291


namespace log_problem_l3972_397262

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_problem (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 2 = 1) :
  a = 2 ∧ 
  f a 1 = 0 ∧ 
  ∀ x > 0, f a x < 1 ↔ x < 2 := by
  sorry

end log_problem_l3972_397262


namespace ryegrass_percentage_in_mixture_X_l3972_397202

/-- Proves that the percentage of ryegrass in seed mixture X is 40% -/
theorem ryegrass_percentage_in_mixture_X : ∀ (x : ℝ),
  -- Seed mixture X has x% ryegrass and 60% bluegrass
  x + 60 = 100 →
  -- A mixture of 86.67% X and 13.33% Y contains 38% ryegrass
  0.8667 * x + 0.1333 * 25 = 38 →
  -- The percentage of ryegrass in seed mixture X is 40%
  x = 40 :=
by
  sorry


end ryegrass_percentage_in_mixture_X_l3972_397202


namespace altitude_bisector_median_inequality_l3972_397205

/-- Triangle structure with altitude, angle bisector, and median from vertex A -/
structure Triangle :=
  (A B C : Point)
  (ha : ℝ) -- altitude from A to BC
  (βa : ℝ) -- angle bisector from A to BC
  (ma : ℝ) -- median from A to BC

/-- Theorem stating the inequality between altitude, angle bisector, and median -/
theorem altitude_bisector_median_inequality (t : Triangle) : t.ha ≤ t.βa ∧ t.βa ≤ t.ma := by
  sorry

end altitude_bisector_median_inequality_l3972_397205


namespace complex_square_roots_l3972_397221

theorem complex_square_roots : 
  ∀ z : ℂ, z^2 = -99 - 40*I ↔ z = 2 - 10*I ∨ z = -2 + 10*I :=
by sorry

end complex_square_roots_l3972_397221


namespace equation_solutions_l3972_397270

theorem equation_solutions :
  (∀ x : ℝ, 2 * (2 * x - 1)^2 = 32 ↔ x = 5/2 ∨ x = -3/2) ∧
  (∀ x : ℝ, -x^2 + 2*x + 1 = 0 ↔ x = -1 - Real.sqrt 2 ∨ x = -1 + Real.sqrt 2) ∧
  (∀ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1) :=
by sorry

end equation_solutions_l3972_397270


namespace unique_prime_f_l3972_397201

/-- The polynomial function f(n) = n^3 - 7n^2 + 18n - 10 -/
def f (n : ℕ) : ℤ := n^3 - 7*n^2 + 18*n - 10

/-- Theorem stating that there exists exactly one positive integer n such that f(n) is prime -/
theorem unique_prime_f : ∃! (n : ℕ+), Nat.Prime (Int.natAbs (f n)) := by sorry

end unique_prime_f_l3972_397201


namespace cube_sphere_comparison_l3972_397294

theorem cube_sphere_comparison (a b R : ℝ) 
  (h1 : 6 * a^2 = 4 * Real.pi * R^2) 
  (h2 : b^3 = (4/3) * Real.pi * R^3) :
  a < b :=
by sorry

end cube_sphere_comparison_l3972_397294


namespace proportionality_coefficient_l3972_397207

/-- Given variables x, y, z and a positive integer k, satisfying the following conditions:
    1. z - y = k * x
    2. x - z = k * y
    3. z = 5/3 * (x - y)
    Prove that k = 3 -/
theorem proportionality_coefficient (x y z : ℝ) (k : ℕ+) 
  (h1 : z - y = k * x)
  (h2 : x - z = k * y)
  (h3 : z = 5/3 * (x - y)) :
  k = 3 := by sorry

end proportionality_coefficient_l3972_397207


namespace a_gt_abs_b_sufficient_not_necessary_l3972_397249

theorem a_gt_abs_b_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) :=
by sorry

end a_gt_abs_b_sufficient_not_necessary_l3972_397249


namespace eleven_sided_polygon_diagonals_l3972_397226

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of diagonals in a polygon with n sides --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A polygon has an obtuse angle --/
def has_obtuse_angle (p : ConvexPolygon n) : Prop := sorry

theorem eleven_sided_polygon_diagonals :
  ∀ (p : ConvexPolygon 11), has_obtuse_angle p → num_diagonals 11 = 44 := by
  sorry

end eleven_sided_polygon_diagonals_l3972_397226


namespace min_value_quadratic_form_l3972_397285

theorem min_value_quadratic_form :
  (∀ x y z : ℝ, x^2 + x*y + y^2 + z^2 ≥ 0) ∧
  (∃ x y z : ℝ, x^2 + x*y + y^2 + z^2 = 0) :=
by sorry

end min_value_quadratic_form_l3972_397285


namespace min_max_values_of_f_l3972_397206

def f (x : ℝ) := -2 * x + 1

theorem min_max_values_of_f :
  ∀ x ∈ Set.Icc 0 5,
    (∃ y ∈ Set.Icc 0 5, f y ≤ f x) ∧
    (∃ z ∈ Set.Icc 0 5, f x ≤ f z) ∧
    f 5 = -9 ∧
    f 0 = 1 ∧
    (∀ w ∈ Set.Icc 0 5, -9 ≤ f w ∧ f w ≤ 1) :=
by sorry

end min_max_values_of_f_l3972_397206


namespace no_solution_for_diophantine_equation_l3972_397269

theorem no_solution_for_diophantine_equation (d : ℤ) (h : d % 4 = 3) :
  ∀ (x y : ℕ), x^2 - d * y^2 ≠ -1 := by
  sorry

end no_solution_for_diophantine_equation_l3972_397269


namespace greatest_integer_radius_eight_is_greatest_l3972_397258

theorem greatest_integer_radius (r : ℕ) : r ^ 2 < 75 → r ≤ 8 := by
  sorry

theorem eight_is_greatest : ∃ (r : ℕ), r ^ 2 < 75 ∧ r = 8 := by
  sorry

end greatest_integer_radius_eight_is_greatest_l3972_397258


namespace polynomial_difference_l3972_397230

/-- A polynomial of degree 5 with specific properties -/
def f (a₁ a₂ a₃ a₄ a₅ : ℝ) (x : ℝ) : ℝ :=
  x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅

/-- The theorem statement -/
theorem polynomial_difference (a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ m : ℝ, m ∈ ({1, 2, 3, 4} : Set ℝ) → f a₁ a₂ a₃ a₄ a₅ m = 2017 * m) →
  f a₁ a₂ a₃ a₄ a₅ 10 - f a₁ a₂ a₃ a₄ a₅ (-5) = 75615 :=
by sorry

end polynomial_difference_l3972_397230


namespace fraction_sum_equation_l3972_397287

theorem fraction_sum_equation (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 15 * y) / (y + 15 * x) = 3) : 
  x / y = 0.8 := by sorry

end fraction_sum_equation_l3972_397287


namespace benny_lunch_payment_l3972_397253

/-- The cost of a lunch special -/
def lunch_special_cost : ℕ := 8

/-- The number of people having lunch -/
def number_of_people : ℕ := 3

/-- The total cost Benny will pay for lunch -/
def total_cost : ℕ := number_of_people * lunch_special_cost

theorem benny_lunch_payment : total_cost = 24 := by
  sorry

end benny_lunch_payment_l3972_397253


namespace empty_solution_set_iff_a_nonnegative_l3972_397296

theorem empty_solution_set_iff_a_nonnegative (a : ℝ) :
  (∀ x : ℝ, ¬(2*x < 5 - 3*x ∧ (x-1)/2 > a)) ↔ a ≥ 0 := by
  sorry

end empty_solution_set_iff_a_nonnegative_l3972_397296


namespace verify_statement_with_flipped_cards_l3972_397222

/-- Represents a card with a letter on one side and a natural number on the other. -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a character is a vowel. -/
def isVowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U', 'a', 'e', 'i', 'o', 'u']

/-- Checks if a natural number is even. -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Represents the set of cards on the table. -/
def cardsOnTable : List Card := [
  { letter := 'A', number := 0 },
  { letter := 'B', number := 0 },
  { letter := 'C', number := 4 },
  { letter := 'D', number := 5 }
]

/-- The statement to verify for each card. -/
def statementToVerify (c : Card) : Prop :=
  isVowel c.letter → isEven c.number

/-- The set of cards that need to be flipped to verify the statement. -/
def cardsToFlip : List Card :=
  cardsOnTable.filter (fun c => c.letter = 'A' ∨ c.number = 4 ∨ c.number = 5)

/-- Theorem stating that flipping the cards A, 4, and 5 is necessary and sufficient
    to verify the given statement for all cards on the table. -/
theorem verify_statement_with_flipped_cards :
  (∀ c ∈ cardsOnTable, statementToVerify c) ↔
  (∀ c ∈ cardsToFlip, statementToVerify c) :=
sorry

end verify_statement_with_flipped_cards_l3972_397222


namespace line_perp_parallel_implies_planes_perp_l3972_397239

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planesPerpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_implies_planes_perp
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel a β) :
  planesPerpendicular α β :=
sorry

end line_perp_parallel_implies_planes_perp_l3972_397239


namespace twelve_by_twelve_grid_intersections_l3972_397279

/-- Represents a square grid -/
structure SquareGrid :=
  (n : ℕ)

/-- Number of interior vertical or horizontal lines in a square grid -/
def interior_lines (g : SquareGrid) : ℕ := g.n - 1

/-- Number of interior intersection points in a square grid -/
def interior_intersections (g : SquareGrid) : ℕ :=
  (interior_lines g) * (interior_lines g)

/-- Theorem: The number of interior intersection points on a 12 by 12 grid of squares is 121 -/
theorem twelve_by_twelve_grid_intersections :
  ∃ (g : SquareGrid), g.n = 12 ∧ interior_intersections g = 121 := by
  sorry

end twelve_by_twelve_grid_intersections_l3972_397279


namespace ticket_multiple_calculation_l3972_397227

/-- The multiple of fair tickets compared to baseball game tickets -/
def ticket_multiple (fair_tickets baseball_tickets : ℕ) : ℚ :=
  (fair_tickets - 6 : ℚ) / baseball_tickets

theorem ticket_multiple_calculation (fair_tickets baseball_tickets : ℕ) 
  (h1 : fair_tickets = ticket_multiple fair_tickets baseball_tickets * baseball_tickets + 6)
  (h2 : fair_tickets = 25)
  (h3 : baseball_tickets = 56) :
  ticket_multiple fair_tickets baseball_tickets = 19 / 56 := by
  sorry

#eval ticket_multiple 25 56

end ticket_multiple_calculation_l3972_397227


namespace two_numbers_difference_l3972_397248

theorem two_numbers_difference (a b : ℝ) 
  (sum_condition : a + b = 9)
  (square_difference_condition : a^2 - b^2 = 45) :
  |a - b| = 5 := by sorry

end two_numbers_difference_l3972_397248


namespace browser_windows_l3972_397231

theorem browser_windows (num_browsers : Nat) (tabs_per_window : Nat) (total_tabs : Nat) :
  num_browsers = 2 →
  tabs_per_window = 10 →
  total_tabs = 60 →
  ∃ (windows_per_browser : Nat),
    windows_per_browser * tabs_per_window * num_browsers = total_tabs ∧
    windows_per_browser = 3 := by
  sorry

end browser_windows_l3972_397231


namespace parabola_c_value_l3972_397299

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines a parabola with given properties -/
def parabola : QuadraticFunction :=
  { a := sorry
    b := sorry
    c := sorry }

theorem parabola_c_value :
  (parabola.a * 3^2 + parabola.b * 3 + parabola.c = -5) ∧
  (parabola.a * 5^2 + parabola.b * 5 + parabola.c = -3) →
  parabola.c = -1/2 := by
  sorry

end parabola_c_value_l3972_397299


namespace f_value_at_100_l3972_397236

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem f_value_at_100 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x * f (x + 3) = 12)
  (h2 : f 1 = 4) :
  f 100 = 3 := by
  sorry

end f_value_at_100_l3972_397236


namespace hyperbola_asymptotes_l3972_397264

/-- Given a hyperbola with equation x²/9 - y²/4 = 1, 
    its asymptotes have the equation y = ±(2/3)x -/
theorem hyperbola_asymptotes : 
  ∃ (f : ℝ → ℝ), 
    (∀ x y : ℝ, x^2/9 - y^2/4 = 1 → 
      (y = f x ∨ y = -f x) ∧ 
      f x = (2/3) * x) := by
sorry

end hyperbola_asymptotes_l3972_397264


namespace transport_probabilities_l3972_397288

structure TransportProbabilities where
  train : ℝ
  ship : ℝ
  car : ℝ
  airplane : ℝ
  mutually_exclusive : train + ship + car + airplane = 1
  going_probability : ℝ

def prob : TransportProbabilities :=
  { train := 0.3
  , ship := 0.2
  , car := 0.1
  , airplane := 0.4
  , mutually_exclusive := by sorry
  , going_probability := 0.5
  }

theorem transport_probabilities (p : TransportProbabilities) :
  (p.train + p.airplane = 0.7) ∧
  (1 - p.ship = 0.8) ∧
  ((p.train + p.ship = p.going_probability) ∨ (p.car + p.airplane = p.going_probability)) :=
by sorry

end transport_probabilities_l3972_397288


namespace geometric_sequence_product_l3972_397277

/-- A geometric sequence with a_4 = 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧ a 4 = 4

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 2 * a 6 = 16 := by
  sorry

end geometric_sequence_product_l3972_397277


namespace hoseok_minyoung_problem_l3972_397238

/-- Given a line of students, calculate the number of students between two specified positions. -/
def students_between (total : ℕ) (right_pos : ℕ) (left_pos : ℕ) : ℕ :=
  left_pos - (total - right_pos + 1) - 1

/-- Theorem: In a line of 13 students, with one student 9th from the right and another 8th from the left, 
    there are 2 students between them. -/
theorem hoseok_minyoung_problem :
  students_between 13 9 8 = 2 := by
  sorry

end hoseok_minyoung_problem_l3972_397238


namespace find_m_l3972_397224

theorem find_m (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120)
  (h2 : Nat.lcm m 45 = 180) :
  m = 24 := by
  sorry

end find_m_l3972_397224


namespace lunch_spending_difference_l3972_397210

theorem lunch_spending_difference (total_spent friend_spent : ℕ) : 
  total_spent = 17 →
  friend_spent = 10 →
  friend_spent > total_spent - friend_spent →
  friend_spent - (total_spent - friend_spent) = 3 := by
  sorry

end lunch_spending_difference_l3972_397210


namespace dodecagon_diagonals_l3972_397211

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals : 
  num_diagonals dodecagon_sides = 54 := by sorry

end dodecagon_diagonals_l3972_397211


namespace smallest_rational_l3972_397251

theorem smallest_rational (S : Set ℚ) (h : S = {-1, 0, 3, -1/3}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -1 := by
  sorry

end smallest_rational_l3972_397251


namespace symmetric_circle_equation_l3972_397243

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- Symmetry about the origin -/
def symmetricAboutOrigin (c1 c2 : Circle) : Prop :=
  c2.center = (-c1.center.1, -c1.center.2) ∧ c2.radius = c1.radius

/-- The main theorem -/
theorem symmetric_circle_equation (c1 c2 : Circle) :
  c1.equation = λ x y => (x + 2)^2 + y^2 = 5 →
  symmetricAboutOrigin c1 c2 →
  c2.equation = λ x y => (x - 2)^2 + y^2 = 5 :=
by sorry

end symmetric_circle_equation_l3972_397243


namespace always_two_real_roots_find_m_value_l3972_397292

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - (m+1)*x + (3*m-6)

-- Theorem 1: The quadratic equation always has two real roots
theorem always_two_real_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 :=
sorry

-- Theorem 2: Given the condition, m = 3
theorem find_m_value (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : quadratic m x₁ = 0) 
  (h₂ : quadratic m x₂ = 0)
  (h₃ : x₁ + x₂ + x₁*x₂ = 7) : 
  m = 3 :=
sorry

end always_two_real_roots_find_m_value_l3972_397292


namespace P_complement_subset_Q_l3972_397290

def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x > -1}
def P_complement : Set ℝ := {x | x ≥ 1}

theorem P_complement_subset_Q : P_complement ⊆ Q := by
  sorry

end P_complement_subset_Q_l3972_397290


namespace solve_for_C_l3972_397265

theorem solve_for_C : ∃ C : ℤ, 80 - (5 - (6 + 2 * (7 - C - 5))) = 89 ∧ C = -2 := by
  sorry

end solve_for_C_l3972_397265


namespace floor_sqrt_50_squared_l3972_397261

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end floor_sqrt_50_squared_l3972_397261
