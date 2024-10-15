import Mathlib

namespace NUMINAMATH_CALUDE_zero_exponent_l3433_343388

theorem zero_exponent (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l3433_343388


namespace NUMINAMATH_CALUDE_smallest_other_integer_l3433_343322

theorem smallest_other_integer (m n x : ℕ+) : 
  (m = 50 ∨ n = 50) →
  Nat.gcd m.val n.val = x.val + 5 →
  Nat.lcm m.val n.val = x.val * (x.val + 5) →
  (m ≠ 50 → m ≥ 10) ∧ (n ≠ 50 → n ≥ 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l3433_343322


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l3433_343316

/-- Proves that for a rectangular plot with given dimensions and total fencing cost,
    the cost per meter of fencing is as calculated. -/
theorem fencing_cost_per_meter
  (length : ℝ) (breadth : ℝ) (total_cost : ℝ)
  (h1 : length = 56)
  (h2 : breadth = 44)
  (h3 : total_cost = 5300) :
  total_cost / (2 * (length + breadth)) = 26.5 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l3433_343316


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3433_343320

theorem quadratic_roots_relation (d e : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 6 = 0 ∧ 2 * s^2 - 4 * s - 6 = 0 ∧
   ∀ x : ℝ, x^2 + d * x + e = 0 ↔ x = r - 3 ∨ x = s - 3) →
  e = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3433_343320


namespace NUMINAMATH_CALUDE_chips_after_steps_chips_after_25_steps_l3433_343372

/-- Represents the state of trays with chips -/
def TrayState := List Bool

/-- Converts a natural number to its binary representation -/
def toBinary (n : Nat) : TrayState :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Counts the number of true values in a list of booleans -/
def countTrueValues (l : List Bool) : Nat :=
  l.filter id |>.length

/-- The number of chips after n steps is equal to the number of 1s in the binary representation of n -/
theorem chips_after_steps (n : Nat) : 
  countTrueValues (toBinary n) = countTrueValues (toBinary n) := by sorry

/-- The number of chips after 25 steps is equal to the number of 1s in the binary representation of 25 -/
theorem chips_after_25_steps : 
  countTrueValues (toBinary 25) = 3 := by sorry

end NUMINAMATH_CALUDE_chips_after_steps_chips_after_25_steps_l3433_343372


namespace NUMINAMATH_CALUDE_volume_expanded_parallelepiped_eq_l3433_343344

/-- The volume of a set of points inside or within one unit of a 2x3x4 rectangular parallelepiped -/
def volume_expanded_parallelepiped : ℝ := sorry

/-- The dimension of the parallelepiped along the x-axis -/
def x_dim : ℝ := 2

/-- The dimension of the parallelepiped along the y-axis -/
def y_dim : ℝ := 3

/-- The dimension of the parallelepiped along the z-axis -/
def z_dim : ℝ := 4

/-- The radius of the expanded region around the parallelepiped -/
def expansion_radius : ℝ := 1

theorem volume_expanded_parallelepiped_eq :
  volume_expanded_parallelepiped = (228 + 31 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_volume_expanded_parallelepiped_eq_l3433_343344


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3433_343339

theorem least_n_satisfying_inequality : 
  (∀ k : ℕ, k > 0 ∧ k < 10 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 100) ∧
  ((1 : ℚ) / 10 - (1 : ℚ) / 11 < (1 : ℚ) / 100) :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l3433_343339


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3433_343383

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := (1 + m * Complex.I) / (1 - Complex.I)
  (∃ (y : ℝ), z = Complex.I * y) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3433_343383


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l3433_343330

theorem problem_1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 := by sorry

theorem problem_2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = -(n - m)^12 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l3433_343330


namespace NUMINAMATH_CALUDE_least_multiple_32_over_500_l3433_343357

theorem least_multiple_32_over_500 : ∃ (n : ℕ), n * 32 > 500 ∧ n * 32 = 512 ∧ ∀ (m : ℕ), m * 32 > 500 → m * 32 ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_32_over_500_l3433_343357


namespace NUMINAMATH_CALUDE_part_one_part_two_l3433_343354

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}

-- Part 1
theorem part_one :
  let a := 2
  (A a ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  ((Set.univ \ A a) ∪ (Set.univ \ B) = {x | x < 3 ∨ x ≥ 4}) := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  A a ∩ B = A a → a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3433_343354


namespace NUMINAMATH_CALUDE_large_planks_nails_l3433_343329

/-- The number of nails needed for large planks in John's house wall construction -/
def nails_for_large_planks (total_nails : ℕ) (nails_for_small_planks : ℕ) : ℕ :=
  total_nails - nails_for_small_planks

/-- Theorem stating that the number of nails for large planks is 15 -/
theorem large_planks_nails :
  nails_for_large_planks 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_large_planks_nails_l3433_343329


namespace NUMINAMATH_CALUDE_product_price_relationship_l3433_343389

/-- Proves the relationship between fall and spring prices of a product given specific conditions -/
theorem product_price_relationship (fall_amount : ℝ) (total_cost : ℝ) (spring_difference : ℝ) : 
  fall_amount = 550 ∧ 
  total_cost = 825 ∧ 
  spring_difference = 220 →
  ∃ (spring_price : ℝ),
    spring_price = total_cost / (fall_amount - spring_difference) ∧
    spring_price = total_cost / fall_amount + 1 ∧
    spring_price = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_product_price_relationship_l3433_343389


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3433_343300

theorem fourteenth_root_of_unity (n : ℕ) (hn : n ≤ 13) : 
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) = 
  Complex.exp (Complex.I * (5 * Real.pi / 7)) :=
sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l3433_343300


namespace NUMINAMATH_CALUDE_perpendicular_line_angle_l3433_343309

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x - Real.sqrt 3 * y + 2 = 0

-- Define the angle of inclination of a line perpendicular to l
def perpendicular_angle (θ : ℝ) : Prop :=
  Real.tan θ = -(Real.sqrt 3 / 3)

-- Theorem statement
theorem perpendicular_line_angle :
  ∃ θ, perpendicular_angle θ ∧ θ = 150 * (π / 180) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_angle_l3433_343309


namespace NUMINAMATH_CALUDE_cos_two_alpha_minus_pi_sixth_l3433_343368

theorem cos_two_alpha_minus_pi_sixth (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/3) 
  (h3 : Real.sin (α + π/6) = 2 * Real.sqrt 5 / 5) : 
  Real.cos (2*α - π/6) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_alpha_minus_pi_sixth_l3433_343368


namespace NUMINAMATH_CALUDE_distributor_profit_percentage_l3433_343304

/-- Proves that the distributor's profit percentage is 65% given the specified conditions -/
theorem distributor_profit_percentage
  (commission_rate : Real)
  (producer_price : Real)
  (final_price : Real)
  (h1 : commission_rate = 0.2)
  (h2 : producer_price = 15)
  (h3 : final_price = 19.8) :
  (((final_price / (1 - commission_rate)) - producer_price) / producer_price) * 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_distributor_profit_percentage_l3433_343304


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l3433_343386

/-- Calculate the total cost for a group to eat at a restaurant -/
theorem restaurant_bill_calculation 
  (adult_meal_cost : ℕ) 
  (total_people : ℕ) 
  (kids_count : ℕ) 
  (h1 : adult_meal_cost = 3)
  (h2 : total_people = 12)
  (h3 : kids_count = 7) :
  (total_people - kids_count) * adult_meal_cost = 15 := by
  sorry

#check restaurant_bill_calculation

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l3433_343386


namespace NUMINAMATH_CALUDE_max_value_condition_l3433_343301

/-- 
Given that x and y are real numbers, prove that when 2005 - (x + y)^2 takes its maximum value, x = -y.
-/
theorem max_value_condition (x y : ℝ) : 
  (∀ a b : ℝ, 2005 - (x + y)^2 ≥ 2005 - (a + b)^2) → x = -y := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l3433_343301


namespace NUMINAMATH_CALUDE_volume_added_equals_expression_l3433_343364

/-- Represents a cylindrical tank lying on its side -/
structure CylindricalTank where
  radius : ℝ
  length : ℝ

/-- Calculates the volume of water added to the tank -/
def volumeAdded (tank : CylindricalTank) (initialDepth finalDepth : ℝ) : ℝ := sorry

/-- The main theorem to prove -/
theorem volume_added_equals_expression (tank : CylindricalTank) :
  tank.radius = 10 →
  tank.length = 30 →
  volumeAdded tank 5 (10 + 5 * Real.sqrt 2) = 1250 * Real.pi + 1500 + 750 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_added_equals_expression_l3433_343364


namespace NUMINAMATH_CALUDE_dog_distance_theorem_l3433_343303

/-- The problem of calculating the distance run by a dog between two people --/
theorem dog_distance_theorem 
  (anderson_speed baxter_speed dog_speed : ℝ)
  (head_start : ℝ)
  (h_anderson_speed : anderson_speed = 2)
  (h_baxter_speed : baxter_speed = 4)
  (h_dog_speed : dog_speed = 10)
  (h_head_start : head_start = 1) :
  let initial_distance := anderson_speed * head_start
  let relative_speed := baxter_speed - anderson_speed
  let catch_up_time := initial_distance / relative_speed
  dog_speed * catch_up_time = 10 := by sorry

end NUMINAMATH_CALUDE_dog_distance_theorem_l3433_343303


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l3433_343334

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) :
  arithmetic_sequence a (-2) →
  (a 1 + a 5) / 2 = -1 →
  a 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l3433_343334


namespace NUMINAMATH_CALUDE_pool_volume_calculation_l3433_343380

/-- Calculates the total volume of a pool given its draining parameters -/
theorem pool_volume_calculation 
  (drain_rate : ℝ) 
  (drain_time : ℝ) 
  (initial_capacity_percentage : ℝ) : 
  drain_rate * drain_time / initial_capacity_percentage = 90000 :=
by
  sorry

#check pool_volume_calculation 60 1200 0.8

end NUMINAMATH_CALUDE_pool_volume_calculation_l3433_343380


namespace NUMINAMATH_CALUDE_matching_color_probability_l3433_343349

-- Define the number of jelly beans for each person
def abe_green : ℕ := 1
def abe_red : ℕ := 2
def bob_green : ℕ := 2
def bob_yellow : ℕ := 1
def bob_red : ℕ := 1

-- Define the total number of jelly beans for each person
def abe_total : ℕ := abe_green + abe_red
def bob_total : ℕ := bob_green + bob_yellow + bob_red

-- Define the probability of matching colors
def prob_match : ℚ := (abe_green * bob_green + abe_red * bob_red) / (abe_total * bob_total)

-- Theorem statement
theorem matching_color_probability :
  prob_match = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_matching_color_probability_l3433_343349


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3433_343323

theorem divisibility_by_five (a b : ℕ+) : 
  (5 ∣ (a * b)) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3433_343323


namespace NUMINAMATH_CALUDE_q_is_false_l3433_343311

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_q_is_false_l3433_343311


namespace NUMINAMATH_CALUDE_vegetable_planting_methods_l3433_343306

/-- The number of vegetable types available --/
def total_vegetables : ℕ := 4

/-- The number of vegetable types to be chosen --/
def chosen_vegetables : ℕ := 3

/-- The number of soil types --/
def soil_types : ℕ := 3

/-- The number of vegetables to be chosen excluding cucumber --/
def vegetables_to_choose : ℕ := chosen_vegetables - 1

/-- The number of remaining vegetables to choose from --/
def remaining_vegetables : ℕ := total_vegetables - 1

theorem vegetable_planting_methods :
  (Nat.choose remaining_vegetables vegetables_to_choose) * (Nat.factorial chosen_vegetables) = 18 :=
sorry

end NUMINAMATH_CALUDE_vegetable_planting_methods_l3433_343306


namespace NUMINAMATH_CALUDE_election_votes_calculation_l3433_343325

theorem election_votes_calculation (total_votes : ℕ) : 
  (85 : ℚ) / 100 * ((85 : ℚ) / 100 * total_votes) = 404600 →
  total_votes = 560000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l3433_343325


namespace NUMINAMATH_CALUDE_james_hall_of_mirrors_glass_area_l3433_343347

/-- The total area of glass needed for three walls in a hall of mirrors --/
def total_glass_area (long_wall_length long_wall_height short_wall_length short_wall_height : ℝ) : ℝ :=
  2 * (long_wall_length * long_wall_height) + (short_wall_length * short_wall_height)

/-- Theorem: The total area of glass needed for James' hall of mirrors is 960 square feet --/
theorem james_hall_of_mirrors_glass_area :
  total_glass_area 30 12 20 12 = 960 := by
  sorry

end NUMINAMATH_CALUDE_james_hall_of_mirrors_glass_area_l3433_343347


namespace NUMINAMATH_CALUDE_parkway_elementary_girls_not_soccer_l3433_343356

theorem parkway_elementary_girls_not_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : soccer_players = 250)
  (h4 : (86 : ℚ) / 100 * soccer_players = ↑(boys_playing_soccer))
  (boys_playing_soccer : ℕ) :
  total_students - soccer_players - (boys - boys_playing_soccer) = 89 := by
  sorry

#check parkway_elementary_girls_not_soccer

end NUMINAMATH_CALUDE_parkway_elementary_girls_not_soccer_l3433_343356


namespace NUMINAMATH_CALUDE_unique_prime_pair_sum_59_l3433_343346

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

theorem unique_prime_pair_sum_59 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ p + q = 59 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_sum_59_l3433_343346


namespace NUMINAMATH_CALUDE_favorite_movies_total_length_l3433_343337

theorem favorite_movies_total_length : 
  ∀ (michael joyce nikki ryn sam alex : ℝ),
    nikki = 30 →
    michael = nikki / 3 →
    joyce = michael + 2 →
    ryn = nikki * (4/5) →
    sam = joyce * 1.5 →
    alex = 2 * (min michael (min joyce (min nikki (min ryn sam)))) →
    michael + joyce + nikki + ryn + sam + alex = 114 :=
by
  sorry

end NUMINAMATH_CALUDE_favorite_movies_total_length_l3433_343337


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l3433_343374

-- Define the slopes and y-intercept
def m₁ : ℚ := 8
def m₂ : ℚ := 4
def b : ℚ := 5

-- Define the x-intercepts
def s : ℚ := -b / m₁
def t : ℚ := -b / m₂

-- Theorem statement
theorem x_intercept_ratio :
  s / t = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l3433_343374


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l3433_343353

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * cos(B) - b * cos(A) = (4/5) * c, then tan(A) / tan(B) = 9 -/
theorem triangle_tangent_ratio (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.cos B - b * Real.cos A = (4/5) * c →
  Real.tan A / Real.tan B = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l3433_343353


namespace NUMINAMATH_CALUDE_last_two_digits_of_nine_to_h_l3433_343324

def a : ℕ := 1
def b : ℕ := 2^a
def c : ℕ := 3^b
def d : ℕ := 4^c
def e : ℕ := 5^d
def f : ℕ := 6^e
def g : ℕ := 7^f
def h : ℕ := 8^g

theorem last_two_digits_of_nine_to_h (a b c d e f g h : ℕ) 
  (ha : a = 1)
  (hb : b = 2^a)
  (hc : c = 3^b)
  (hd : d = 4^c)
  (he : e = 5^d)
  (hf : f = 6^e)
  (hg : g = 7^f)
  (hh : h = 8^g) :
  9^h % 100 = 21 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_nine_to_h_l3433_343324


namespace NUMINAMATH_CALUDE_printing_machines_equation_l3433_343361

theorem printing_machines_equation (x : ℝ) : x > 0 → 
  (1000 / 15 : ℝ) + 1000 / x = 1000 / 5 ↔ 1 / 15 + 1 / x = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_printing_machines_equation_l3433_343361


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l3433_343398

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define our function f
noncomputable def f (x : ℝ) : ℝ :=
  (floor x : ℝ) + Real.sqrt (x - floor x)

-- State the theorem
theorem f_strictly_increasing :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l3433_343398


namespace NUMINAMATH_CALUDE_inequality_theorem_largest_c_k_zero_largest_c_k_two_l3433_343397

theorem inequality_theorem :
  ∀ (k : ℝ), 
  (∃ (c_k : ℝ), c_k > 0 ∧ 
    (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 
      (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k)) ↔ 
  (0 ≤ k ∧ k ≤ 2) :=
sorry

theorem largest_c_k_zero :
  ∀ (c : ℝ), c > 0 →
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c) →
  c ≤ 1 :=
sorry

theorem largest_c_k_two :
  ∀ (c : ℝ), c > 0 →
  (∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c * (x + y + z)^2) →
  c ≤ 8/9 :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_largest_c_k_zero_largest_c_k_two_l3433_343397


namespace NUMINAMATH_CALUDE_square_root_problem_l3433_343345

theorem square_root_problem (a : ℝ) (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt x = 2*a - 1) (h3 : Real.sqrt x = -a + 3) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l3433_343345


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3433_343394

theorem square_plus_reciprocal_square (n : ℝ) (h : n + 1/n = 10) : n^2 + 1/n^2 + 4 = 102 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3433_343394


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l3433_343355

theorem z_in_first_quadrant (z : ℂ) : 
  ((1 + 2*Complex.I) / (z - 3) = -Complex.I) → 
  (z.re > 0 ∧ z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l3433_343355


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l3433_343375

/-- A geometric sequence with common ratio 2 and all positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : GeometricSequence a) (h_prod : a 3 * a 11 = 16) :
  a 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l3433_343375


namespace NUMINAMATH_CALUDE_exactly_three_red_marbles_l3433_343385

def total_marbles : ℕ := 15
def red_marbles : ℕ := 8
def blue_marbles : ℕ := 7
def trials : ℕ := 6
def target_red : ℕ := 3

def probability_red : ℚ := red_marbles / total_marbles
def probability_blue : ℚ := blue_marbles / total_marbles

theorem exactly_three_red_marbles :
  (Nat.choose trials target_red : ℚ) *
  probability_red ^ target_red *
  probability_blue ^ (trials - target_red) =
  6881280 / 38107875 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_red_marbles_l3433_343385


namespace NUMINAMATH_CALUDE_sum_of_x_equals_two_l3433_343391

/-- The function f(x) = |x+1| + |x-3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x - 3|

/-- Theorem stating that if there exist two distinct real numbers x₁ and x₂ 
    such that f(x₁) = f(x₂) = 101, then their sum is 2 -/
theorem sum_of_x_equals_two (x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ = 101) (h₃ : f x₂ = 101) :
  x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_equals_two_l3433_343391


namespace NUMINAMATH_CALUDE_molly_sunday_swim_l3433_343314

/-- Represents the distance Molly swam on Sunday -/
def sunday_swim (saturday_swim total_swim : ℕ) : ℕ :=
  total_swim - saturday_swim

/-- Proves that Molly swam 28 meters on Sunday -/
theorem molly_sunday_swim :
  let saturday_swim : ℕ := 45
  let total_swim : ℕ := 73
  let pool_length : ℕ := 25
  sunday_swim saturday_swim total_swim = 28 := by
  sorry

end NUMINAMATH_CALUDE_molly_sunday_swim_l3433_343314


namespace NUMINAMATH_CALUDE_problem_solution_l3433_343351

def M (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem problem_solution :
  (M 1 = {x | 0 < x ∧ x < 2}) ∧
  ({a : ℝ | M a ⊆ N} = Set.Icc (-2) 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3433_343351


namespace NUMINAMATH_CALUDE_noah_large_paintings_l3433_343326

/-- Represents the number of large paintings sold last month -/
def L : ℕ := sorry

/-- Price of a large painting -/
def large_price : ℕ := 60

/-- Price of a small painting -/
def small_price : ℕ := 30

/-- Number of small paintings sold last month -/
def small_paintings_last_month : ℕ := 4

/-- Total sales this month -/
def sales_this_month : ℕ := 1200

/-- Theorem stating that Noah sold 8 large paintings last month -/
theorem noah_large_paintings : L = 8 := by
  sorry

end NUMINAMATH_CALUDE_noah_large_paintings_l3433_343326


namespace NUMINAMATH_CALUDE_c_d_not_dine_city_center_l3433_343310

-- Define the participants
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define locations
inductive Location : Type
| CityCenter : Location
| NearAHome : Location

-- Define the dining relation
def dines_together (p1 p2 : Person) (l : Location) : Prop := sorry

-- Define participation in dining
def participates (p : Person) : Prop := sorry

-- Condition 1: Only if A participates, B and C will dine together
axiom cond1 : ∀ (l : Location), dines_together Person.B Person.C l → participates Person.A

-- Condition 2: A only dines at restaurants near their home
axiom cond2 : ∀ (p : Person) (l : Location), 
  dines_together Person.A p l → l = Location.NearAHome

-- Condition 3: Only if B participates, D will go to the restaurant to dine
axiom cond3 : ∀ (p : Person) (l : Location), 
  dines_together Person.D p l → participates Person.B

-- Theorem to prove
theorem c_d_not_dine_city_center : 
  ¬(dines_together Person.C Person.D Location.CityCenter) :=
sorry

end NUMINAMATH_CALUDE_c_d_not_dine_city_center_l3433_343310


namespace NUMINAMATH_CALUDE_triangle_median_altitude_equations_l3433_343307

/-- Triangle ABC in the Cartesian coordinate system -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of the median from a vertex to the opposite side -/
def median (t : Triangle) (v : ℝ × ℝ) : Line := sorry

/-- Definition of the altitude from a vertex to the opposite side -/
def altitude (t : Triangle) (v : ℝ × ℝ) : Line := sorry

theorem triangle_median_altitude_equations :
  let t : Triangle := { A := (7, 8), B := (10, 4), C := (2, -4) }
  (median t t.B = { a := 8, b := -1, c := -48 }) ∧
  (altitude t t.B = { a := 1, b := 1, c := -15 }) := by sorry

end NUMINAMATH_CALUDE_triangle_median_altitude_equations_l3433_343307


namespace NUMINAMATH_CALUDE_no_linear_term_implies_k_equals_four_l3433_343308

theorem no_linear_term_implies_k_equals_four (k : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + k) * (x - 4) = a * x^2 + b) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_k_equals_four_l3433_343308


namespace NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l3433_343338

/-- Given a function g(x) = 3x^4 + 2x^3 - x^2 - 4x + s, prove that g(-1) = 0 when s = -4 -/
theorem g_equals_zero_at_negative_one (s : ℝ) : 
  let g : ℝ → ℝ := λ x => 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s
  g (-1) = 0 ↔ s = -4 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_zero_at_negative_one_l3433_343338


namespace NUMINAMATH_CALUDE_jane_score_is_14_l3433_343340

/-- Represents a mathematics competition with a scoring system. -/
structure MathCompetition where
  totalQuestions : ℕ
  correctAnswers : ℕ
  incorrectAnswers : ℕ
  unansweredQuestions : ℕ
  correctPoints : ℚ
  incorrectPenalty : ℚ

/-- Calculates the total score for a given math competition. -/
def calculateScore (comp : MathCompetition) : ℚ :=
  comp.correctAnswers * comp.correctPoints - comp.incorrectAnswers * comp.incorrectPenalty

/-- Theorem stating that Jane's score in the competition is 14 points. -/
theorem jane_score_is_14 (comp : MathCompetition)
  (h1 : comp.totalQuestions = 35)
  (h2 : comp.correctAnswers = 17)
  (h3 : comp.incorrectAnswers = 12)
  (h4 : comp.unansweredQuestions = 6)
  (h5 : comp.correctPoints = 1)
  (h6 : comp.incorrectPenalty = 1/4)
  (h7 : comp.totalQuestions = comp.correctAnswers + comp.incorrectAnswers + comp.unansweredQuestions) :
  calculateScore comp = 14 := by
  sorry

end NUMINAMATH_CALUDE_jane_score_is_14_l3433_343340


namespace NUMINAMATH_CALUDE_sum_of_powers_of_two_l3433_343378

theorem sum_of_powers_of_two (n : ℕ) : 
  (1 : ℚ) / 2^10 + 1 / 2^9 + 1 / 2^8 = n / 2^10 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_two_l3433_343378


namespace NUMINAMATH_CALUDE_blue_ball_probability_l3433_343305

/-- The probability of selecting 3 blue balls from a jar containing 6 red and 4 blue balls -/
theorem blue_ball_probability (red_balls blue_balls selected : ℕ) 
  (h1 : red_balls = 6)
  (h2 : blue_balls = 4)
  (h3 : selected = 3) :
  (Nat.choose blue_balls selected) / (Nat.choose (red_balls + blue_balls) selected) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l3433_343305


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l3433_343312

theorem sibling_ages_sum (a b c : ℕ+) : 
  a < b → b < c → a * b * c = 72 → a + b + c = 13 := by sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l3433_343312


namespace NUMINAMATH_CALUDE_percentage_of_pistachios_with_shells_l3433_343360

theorem percentage_of_pistachios_with_shells 
  (total_pistachios : ℕ)
  (opened_shell_ratio : ℚ)
  (opened_shell_count : ℕ)
  (h1 : total_pistachios = 80)
  (h2 : opened_shell_ratio = 3/4)
  (h3 : opened_shell_count = 57) :
  (↑opened_shell_count / (↑total_pistachios * opened_shell_ratio) : ℚ) = 95/100 :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_pistachios_with_shells_l3433_343360


namespace NUMINAMATH_CALUDE_percentage_saved_approx_11_percent_l3433_343328

def original_price : ℝ := 30
def amount_saved : ℝ := 3
def amount_spent : ℝ := 24

theorem percentage_saved_approx_11_percent :
  let actual_price := amount_spent + amount_saved
  let percentage_saved := (amount_saved / actual_price) * 100
  ∃ ε > 0, abs (percentage_saved - 11) < ε :=
by sorry

end NUMINAMATH_CALUDE_percentage_saved_approx_11_percent_l3433_343328


namespace NUMINAMATH_CALUDE_exists_solution_a4_eq_b3_plus_c2_l3433_343384

theorem exists_solution_a4_eq_b3_plus_c2 : 
  ∃ (a b c : ℕ+), (a : ℝ)^4 = (b : ℝ)^3 + (c : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_a4_eq_b3_plus_c2_l3433_343384


namespace NUMINAMATH_CALUDE_trapezium_side_length_l3433_343302

/-- Given a trapezium with the following properties:
  * One parallel side is 18 cm long
  * The distance between parallel sides is 11 cm
  * The area is 209 cm²
  Then the length of the other parallel side is 20 cm -/
theorem trapezium_side_length (a b h : ℝ) (hb : b = 18) (hh : h = 11) (harea : (a + b) * h / 2 = 209) :
  a = 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l3433_343302


namespace NUMINAMATH_CALUDE_river_depth_calculation_l3433_343381

theorem river_depth_calculation (depth_mid_may : ℝ) : 
  let depth_mid_june := depth_mid_may + 10
  let depth_june_20 := depth_mid_june - 5
  let depth_july_5 := depth_june_20 + 8
  let depth_mid_july := depth_july_5
  depth_mid_july = 45 → depth_mid_may = 32 := by sorry

end NUMINAMATH_CALUDE_river_depth_calculation_l3433_343381


namespace NUMINAMATH_CALUDE_circle_inequality_abc_inequality_l3433_343369

-- Problem I
theorem circle_inequality (x y : ℝ) (h : x^2 + y^2 = 1) : 
  -Real.sqrt 13 ≤ 2*x + 3*y ∧ 2*x + 3*y ≤ Real.sqrt 13 := by
  sorry

-- Problem II
theorem abc_inequality (a b c : ℝ) (h : a^2 + b^2 + c^2 - 2*a - 2*b - 2*c = 0) :
  2*a - b - c ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_abc_inequality_l3433_343369


namespace NUMINAMATH_CALUDE_committee_selection_l3433_343342

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l3433_343342


namespace NUMINAMATH_CALUDE_f_neg_two_equals_ten_l3433_343387

/-- Given a function f(x) = x^2 - 3x, prove that f(-2) = 10 -/
theorem f_neg_two_equals_ten (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 3*x) : f (-2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_equals_ten_l3433_343387


namespace NUMINAMATH_CALUDE_extremum_of_f_under_constraint_l3433_343327

-- Define the function f
def f (x y : ℝ) : ℝ := x^2 + y^2 - 2*x - y

-- Define the constraint function φ
def φ (x y : ℝ) : ℝ := x + y - 1

-- State the theorem
theorem extremum_of_f_under_constraint :
  ∃ (x y : ℝ),
    φ x y = 0 ∧
    (∀ (x' y' : ℝ), φ x' y' = 0 → f x' y' ≥ f x y) ∧
    x = 3/4 ∧ y = 1/4 ∧ f x y = -9/8 :=
sorry

end NUMINAMATH_CALUDE_extremum_of_f_under_constraint_l3433_343327


namespace NUMINAMATH_CALUDE_lcm_75_120_l3433_343399

theorem lcm_75_120 : Nat.lcm 75 120 = 600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_75_120_l3433_343399


namespace NUMINAMATH_CALUDE_mans_current_age_l3433_343335

/-- Given a man and his son, where the man is thrice as old as his son now,
    and after 12 years he will be twice as old as his son,
    prove that the man's current age is 36 years. -/
theorem mans_current_age (man_age son_age : ℕ) : 
  man_age = 3 * son_age →
  man_age + 12 = 2 * (son_age + 12) →
  man_age = 36 := by
  sorry

end NUMINAMATH_CALUDE_mans_current_age_l3433_343335


namespace NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l3433_343350

theorem min_value_fraction (x : ℝ) (h : x > 8) : x^2 / (x - 8) ≥ 32 :=
by sorry

theorem equality_condition (x : ℝ) (h : x > 8) : 
  x^2 / (x - 8) = 32 ↔ x = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_equality_condition_l3433_343350


namespace NUMINAMATH_CALUDE_page_number_added_twice_l3433_343318

theorem page_number_added_twice 
  (n : ℕ) 
  (h1 : 60 ≤ n ∧ n ≤ 70) 
  (h2 : ∃ k : ℕ, k ≤ n ∧ n * (n + 1) / 2 + k = 2378) : 
  ∃ k : ℕ, k ≤ n ∧ n * (n + 1) / 2 + k = 2378 ∧ k = 32 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l3433_343318


namespace NUMINAMATH_CALUDE_camel_count_l3433_343336

/-- The cost of an elephant in rupees -/
def elephant_cost : ℚ := 12000

/-- The cost of an ox in rupees -/
def ox_cost : ℚ := 8000

/-- The cost of a horse in rupees -/
def horse_cost : ℚ := 2000

/-- The cost of a camel in rupees -/
def camel_cost : ℚ := 4800

/-- The number of camels -/
def num_camels : ℚ := 10

theorem camel_count :
  (6 * ox_cost = 4 * elephant_cost) →
  (16 * horse_cost = 4 * ox_cost) →
  (24 * horse_cost = num_camels * camel_cost) →
  (10 * elephant_cost = 120000) →
  num_camels = 10 := by
  sorry

end NUMINAMATH_CALUDE_camel_count_l3433_343336


namespace NUMINAMATH_CALUDE_bus_speed_is_45_l3433_343393

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  h_valid : tens ≥ 1 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

def initial_number : TwoDigitNumber := sorry

def one_hour_later : ThreeDigitNumber := sorry

def two_hours_later : ThreeDigitNumber := sorry

/-- The speed of the bus in km/h -/
def bus_speed : Nat := sorry

theorem bus_speed_is_45 :
  (one_hour_later.hundreds = initial_number.ones) ∧
  (one_hour_later.tens = 0) ∧
  (one_hour_later.ones = initial_number.tens) ∧
  (two_hours_later.hundreds = one_hour_later.hundreds) ∧
  (two_hours_later.ones = one_hour_later.ones) ∧
  (two_hours_later.tens ≠ 0) →
  bus_speed = 45 := by sorry

end NUMINAMATH_CALUDE_bus_speed_is_45_l3433_343393


namespace NUMINAMATH_CALUDE_distribute_6_3_l3433_343343

/-- The number of ways to distribute n items among k categories, 
    with each category receiving at least one item. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 10 ways to distribute 6 items among 3 categories, 
    with each category receiving at least one item. -/
theorem distribute_6_3 : distribute 6 3 = 10 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_l3433_343343


namespace NUMINAMATH_CALUDE_log_inequality_implies_base_inequality_l3433_343348

theorem log_inequality_implies_base_inequality (a b : ℝ) 
  (h1 : (Real.log 3 / Real.log a) > (Real.log 3 / Real.log b)) 
  (h2 : (Real.log 3 / Real.log b) > 0) : b > a ∧ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_implies_base_inequality_l3433_343348


namespace NUMINAMATH_CALUDE_cars_meeting_time_l3433_343376

/-- Given a scenario with two cars and a train, calculate the time for the cars to meet -/
theorem cars_meeting_time (train_length : ℝ) (train_speed_kmh : ℝ) (time_between_encounters : ℝ)
  (time_pass_A : ℝ) (time_pass_B : ℝ)
  (h1 : train_length = 180)
  (h2 : train_speed_kmh = 60)
  (h3 : time_between_encounters = 5)
  (h4 : time_pass_A = 30 / 60)
  (h5 : time_pass_B = 6 / 60) :
  let train_speed := train_speed_kmh * 1000 / 3600
  let car_A_speed := train_speed - train_length / time_pass_A
  let car_B_speed := train_length / time_pass_B - train_speed
  let distance := time_between_encounters * (train_speed - car_A_speed)
  distance / (car_A_speed + car_B_speed) = 1.25 := by
  sorry

end NUMINAMATH_CALUDE_cars_meeting_time_l3433_343376


namespace NUMINAMATH_CALUDE_cost_of_700_pieces_l3433_343313

/-- The cost function for gum pieces -/
def gum_cost (pieces : ℕ) : ℚ :=
  if pieces ≤ 500 then
    pieces / 100
  else
    5 + (pieces - 500) * 8 / 1000

/-- Theorem stating the cost of 700 pieces of gum -/
theorem cost_of_700_pieces : gum_cost 700 = 33/5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_700_pieces_l3433_343313


namespace NUMINAMATH_CALUDE_odd_power_sum_divisible_l3433_343341

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ n : ℕ, n > 0 → Odd n → (x^n + y^n) % (x + y) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_odd_power_sum_divisible_l3433_343341


namespace NUMINAMATH_CALUDE_files_remaining_after_deletion_l3433_343331

theorem files_remaining_after_deletion (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 16)
  (h2 : video_files = 48)
  (h3 : deleted_files = 30) :
  music_files + video_files - deleted_files = 34 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_after_deletion_l3433_343331


namespace NUMINAMATH_CALUDE_jackson_earnings_l3433_343377

def usd_per_hour : ℝ := 5
def gbp_per_hour : ℝ := 3
def jpy_per_hour : ℝ := 400

def vacuum_hours : ℝ := 2 * 2
def dishes_hours : ℝ := 0.5
def bathroom_hours : ℝ := 0.5 * 3

def gbp_to_usd : ℝ := 1.35
def jpy_to_usd : ℝ := 0.009

theorem jackson_earnings : 
  (vacuum_hours * usd_per_hour) + 
  (dishes_hours * gbp_per_hour * gbp_to_usd) + 
  (bathroom_hours * jpy_per_hour * jpy_to_usd) = 27.425 := by
sorry

end NUMINAMATH_CALUDE_jackson_earnings_l3433_343377


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l3433_343315

theorem triangle_angle_sum (first_angle second_angle third_angle : ℝ) : 
  second_angle = 2 * first_angle →
  third_angle = 15 →
  first_angle = third_angle + 40 →
  first_angle + second_angle = 165 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l3433_343315


namespace NUMINAMATH_CALUDE_range_of_a_in_acute_triangle_l3433_343332

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b^2 - a^2 = ac and c = 2, then 2/3 < a < 2 -/
theorem range_of_a_in_acute_triangle (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π / 2 →
  0 < C ∧ C < π / 2 →
  a > 0 ∧ b > 0 ∧ c > 0 →
  b^2 - a^2 = a * c →
  c = 2 →
  2/3 < a ∧ a < 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_in_acute_triangle_l3433_343332


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l3433_343379

/-- A rhombus with given area and diagonal ratio has a specific longer diagonal length -/
theorem rhombus_diagonal_length 
  (area : ℝ) 
  (diagonal_ratio : ℚ) 
  (h_area : area = 150) 
  (h_ratio : diagonal_ratio = 4 / 3) : 
  ∃ (d1 d2 : ℝ), d1 > d2 ∧ d1 / d2 = diagonal_ratio ∧ area = (d1 * d2) / 2 ∧ d1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l3433_343379


namespace NUMINAMATH_CALUDE_expression_undefined_at_eight_l3433_343363

/-- The expression is undefined when x = 8 -/
theorem expression_undefined_at_eight :
  ∀ x : ℝ, x = 8 → (x^2 - 16*x + 64 = 0) := by sorry

end NUMINAMATH_CALUDE_expression_undefined_at_eight_l3433_343363


namespace NUMINAMATH_CALUDE_motorcycles_in_parking_lot_l3433_343371

theorem motorcycles_in_parking_lot :
  let total_wheels : ℕ := 117
  let num_cars : ℕ := 19
  let wheels_per_car : ℕ := 5
  let wheels_per_motorcycle : ℕ := 2
  let num_motorcycles : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_motorcycle
  num_motorcycles = 11 := by
  sorry

end NUMINAMATH_CALUDE_motorcycles_in_parking_lot_l3433_343371


namespace NUMINAMATH_CALUDE_olympic_items_problem_l3433_343333

/-- Olympic Commemorative Items Problem -/
theorem olympic_items_problem 
  (total_items : ℕ) 
  (figurine_cost pendant_cost : ℚ) 
  (total_spent : ℚ) 
  (figurine_price pendant_price : ℚ) 
  (min_profit : ℚ) 
  (h1 : total_items = 180)
  (h2 : figurine_cost = 80)
  (h3 : pendant_cost = 50)
  (h4 : total_spent = 11400)
  (h5 : figurine_price = 100)
  (h6 : pendant_price = 60)
  (h7 : min_profit = 2900) :
  ∃ (figurines pendants max_pendants : ℕ),
    figurines + pendants = total_items ∧
    figurine_cost * figurines + pendant_cost * pendants = total_spent ∧
    figurines = 80 ∧
    pendants = 100 ∧
    max_pendants = 70 ∧
    ∀ m : ℕ, m ≤ max_pendants →
      (pendant_price - pendant_cost) * m + 
      (figurine_price - figurine_cost) * (total_items - m) ≥ min_profit :=
by
  sorry


end NUMINAMATH_CALUDE_olympic_items_problem_l3433_343333


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3433_343359

/-- Calculate the combined tax rate for two individuals given their incomes and tax rates -/
theorem combined_tax_rate
  (john_income : ℝ)
  (john_tax_rate : ℝ)
  (ingrid_income : ℝ)
  (ingrid_tax_rate : ℝ)
  (h1 : john_income = 56000)
  (h2 : john_tax_rate = 0.30)
  (h3 : ingrid_income = 72000)
  (h4 : ingrid_tax_rate = 0.40) :
  (john_income * john_tax_rate + ingrid_income * ingrid_tax_rate) / (john_income + ingrid_income) = 0.35625 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3433_343359


namespace NUMINAMATH_CALUDE_triangle_similarity_problem_l3433_343365

-- Define the triangles and their properties
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (height : ℝ)

-- Define the similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- State the theorem
theorem triangle_similarity_problem 
  (FGH IJH : Triangle)
  (h_similar : similar FGH IJH)
  (h_GH : FGH.side1 = 25)
  (h_JH : IJH.side1 = 15)
  (h_height : FGH.height = 15) :
  IJH.side2 = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_similarity_problem_l3433_343365


namespace NUMINAMATH_CALUDE_negation_of_all_x_squared_nonnegative_l3433_343382

theorem negation_of_all_x_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_all_x_squared_nonnegative_l3433_343382


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l3433_343319

theorem largest_coefficient_binomial_expansion :
  let n : ℕ := 7
  let expansion := fun (k : ℕ) => Nat.choose n k
  (∃ k : ℕ, k ≤ n ∧ expansion k = Finset.sup (Finset.range (n + 1)) expansion) →
  Finset.sup (Finset.range (n + 1)) expansion = 35 :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l3433_343319


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l3433_343317

/-- The ratio of the larger volume to the smaller volume of cylinders formed by rolling a 6 × 10 rectangle -/
theorem cylinder_volume_ratio : 
  let rectangle_width : ℝ := 6
  let rectangle_length : ℝ := 10
  let cylinder1_volume := π * (rectangle_width / (2 * π))^2 * rectangle_length
  let cylinder2_volume := π * (rectangle_length / (2 * π))^2 * rectangle_width
  max cylinder1_volume cylinder2_volume / min cylinder1_volume cylinder2_volume = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l3433_343317


namespace NUMINAMATH_CALUDE_solve_equation_l3433_343367

theorem solve_equation (x : ℝ) : (x / 5) + 3 = 4 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3433_343367


namespace NUMINAMATH_CALUDE_inequalities_theorem_l3433_343358

theorem inequalities_theorem (a b c : ℝ) 
  (h1 : a < 0) (h2 : b > 0) (h3 : a < b) (h4 : b < c) : 
  (a * b < b * c) ∧ 
  (a * c < b * c) ∧ 
  (a * b < a * c) ∧ 
  (a + b < b + c) ∧ 
  (c / a < c / b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_theorem_l3433_343358


namespace NUMINAMATH_CALUDE_abs_square_of_complex_l3433_343321

theorem abs_square_of_complex (z : ℂ) : z = 5 + 2*I → Complex.abs (z^2) = 29 := by
  sorry

end NUMINAMATH_CALUDE_abs_square_of_complex_l3433_343321


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3433_343373

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem f_increasing_on_interval :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → x₁ < 1 → x₂ < 1 → f x₁ < f x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3433_343373


namespace NUMINAMATH_CALUDE_inequality_proof_l3433_343362

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3433_343362


namespace NUMINAMATH_CALUDE_product_of_integers_l3433_343390

theorem product_of_integers (A B C D : ℕ+) : 
  A + B + C + D = 51 →
  A = 2 * C - 3 →
  B = 2 * C + 3 →
  D = 5 * C + 1 →
  A * B * C * D = 14910 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l3433_343390


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l3433_343366

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) :
  let a := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  let r := (d1 * d2) / (4 * a)
  r = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l3433_343366


namespace NUMINAMATH_CALUDE_platform_length_l3433_343352

/-- Given a train of length 200 meters that crosses a platform in 50 seconds
    and a signal pole in 42 seconds, the length of the platform is 38 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 200)
  (h2 : platform_crossing_time = 50)
  (h3 : pole_crossing_time = 42) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 38 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l3433_343352


namespace NUMINAMATH_CALUDE_region_contains_point_c_l3433_343370

def point_in_region (x y : ℝ) : Prop :=
  x + 2*y - 1 > 0 ∧ x - y + 3 < 0

theorem region_contains_point_c :
  point_in_region 0 4 ∧
  ¬point_in_region (-4) 1 ∧
  ¬point_in_region 2 2 ∧
  ¬point_in_region (-2) 1 := by
  sorry

#check region_contains_point_c

end NUMINAMATH_CALUDE_region_contains_point_c_l3433_343370


namespace NUMINAMATH_CALUDE_lemon_heads_distribution_l3433_343395

/-- Given 72 Lemon Heads distributed equally among 6 friends, prove that each friend receives 12 Lemon Heads. -/
theorem lemon_heads_distribution (total : ℕ) (friends : ℕ) (each : ℕ) 
  (h1 : total = 72) 
  (h2 : friends = 6) 
  (h3 : total = friends * each) : 
  each = 12 := by
  sorry

end NUMINAMATH_CALUDE_lemon_heads_distribution_l3433_343395


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3433_343396

theorem quadratic_factorization (x : ℝ) : 2*x^2 - 4*x + 2 = 2*(x-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3433_343396


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l3433_343392

theorem rectangular_plot_dimensions :
  ∀ (width length area : ℕ),
    length = width + 1 →
    area = width * length →
    1000 ≤ area ∧ area < 10000 →
    (∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ area = 1000 * a + 100 * a + 10 * b + b) →
    width ∈ ({33, 66, 99} : Set ℕ) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l3433_343392
