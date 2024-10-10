import Mathlib

namespace molecular_weight_3_moles_N2O3_l3751_375139

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in a molecule of Dinitrogen trioxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in a molecule of Dinitrogen trioxide -/
def O_count : ℕ := 3

/-- The number of moles of Dinitrogen trioxide -/
def mole_count : ℝ := 3

/-- The molecular weight of Dinitrogen trioxide in g/mol -/
def molecular_weight_N2O3 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

/-- Theorem: The molecular weight of 3 moles of Dinitrogen trioxide is 228.06 grams -/
theorem molecular_weight_3_moles_N2O3 : 
  mole_count * molecular_weight_N2O3 = 228.06 := by sorry

end molecular_weight_3_moles_N2O3_l3751_375139


namespace not_cheap_necessary_for_good_quality_l3751_375112

-- Define the universe of goods
variable (Goods : Type)

-- Define predicates for "cheap" and "good quality"
variable (cheap : Goods → Prop)
variable (good_quality : Goods → Prop)

-- State the given condition
variable (h : ∀ g : Goods, cheap g → ¬(good_quality g))

-- Theorem statement
theorem not_cheap_necessary_for_good_quality :
  ∀ g : Goods, good_quality g → ¬(cheap g) :=
by
  sorry

end not_cheap_necessary_for_good_quality_l3751_375112


namespace unique_x_with_three_prime_divisors_l3751_375177

theorem unique_x_with_three_prime_divisors (x n : ℕ) : 
  x = 8^n - 1 →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 31 ∧ q ≠ 31 ∧ x = 31 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 31 ∨ r = p ∨ r = q) →
  x = 32767 :=
by sorry

end unique_x_with_three_prime_divisors_l3751_375177


namespace hearts_on_card_l3751_375117

/-- The number of hearts on each card in a hypothetical deck -/
def hearts_per_card : ℕ := sorry

/-- The number of cows in Devonshire -/
def num_cows : ℕ := 2 * hearts_per_card

/-- The cost of each cow in dollars -/
def cost_per_cow : ℕ := 200

/-- The total cost of all cows in dollars -/
def total_cost : ℕ := 83200

theorem hearts_on_card :
  hearts_per_card = 208 :=
sorry

end hearts_on_card_l3751_375117


namespace not_all_triangles_divisible_to_square_l3751_375168

/-- A triangle with base and height -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- A square with side length -/
structure Square where
  side : ℝ

/-- Represents a division of a shape into parts -/
structure Division where
  parts : ℕ

/-- Represents the ability to form a shape from parts -/
def can_form (d : Division) (s : Square) : Prop := sorry

/-- The theorem stating that not all triangles can be divided into 1000 parts to form a square -/
theorem not_all_triangles_divisible_to_square :
  ∃ t : Triangle, ¬ ∃ (d : Division) (s : Square), d.parts = 1000 ∧ can_form d s := by sorry

end not_all_triangles_divisible_to_square_l3751_375168


namespace root_sum_squared_l3751_375115

theorem root_sum_squared (a b : ℝ) : 
  (a^2 + 2*a - 2016 = 0) → 
  (b^2 + 2*b - 2016 = 0) → 
  (a + b = -2) → 
  (a^2 + 3*a + b = 2014) := by
sorry

end root_sum_squared_l3751_375115


namespace students_disliking_both_l3751_375199

theorem students_disliking_both (total : ℕ) (fries : ℕ) (burgers : ℕ) (both : ℕ) :
  total = 25 →
  fries = 15 →
  burgers = 10 →
  both = 6 →
  total - (fries + burgers - both) = 6 :=
by
  sorry

end students_disliking_both_l3751_375199


namespace greatest_integer_satisfying_inequality_l3751_375169

theorem greatest_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), y - 5 > 4*y - 1 → y ≤ x) ∧ (x - 5 > 4*x - 1) ∧ x = -2 := by
  sorry

end greatest_integer_satisfying_inequality_l3751_375169


namespace x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal_l3751_375167

theorem x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  let y := x + 1/x
  (x^2 + 1/x^2 = y^2 - 2) ∧ (x^3 + 1/x^3 = y^3 - 3*y) := by
  sorry

#check x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal

end x_squared_plus_reciprocal_and_x_cubed_plus_reciprocal_l3751_375167


namespace percentage_of_210_l3751_375149

theorem percentage_of_210 : (33 + 1/3 : ℚ) / 100 * 210 = 70 := by
  sorry

end percentage_of_210_l3751_375149


namespace four_intersections_iff_l3751_375120

/-- The number of intersection points between x^2 + y^2 = a^2 and y = x^2 - a - 1 -/
def intersection_count (a : ℝ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the condition for exactly four intersection points -/
theorem four_intersections_iff (a : ℝ) :
  intersection_count a = 4 ↔ a > -1/2 :=
by sorry

end four_intersections_iff_l3751_375120


namespace projection_area_l3751_375185

/-- A polygon in 3D space -/
structure Polygon3D where
  -- Define the polygon structure (this is a simplification)
  area : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane structure (this is a simplification)

/-- The angle between two planes -/
def angle_between_planes (p1 p2 : Plane3D) : ℝ := sorry

/-- The projection of a polygon onto a plane -/
def project_polygon (poly : Polygon3D) (plane : Plane3D) : Polygon3D := sorry

/-- Theorem: The area of a polygon's projection is the original area times the cosine of the angle between planes -/
theorem projection_area (poly : Polygon3D) (plane : Plane3D) : 
  (project_polygon poly plane).area = poly.area * Real.cos (angle_between_planes (Plane3D.mk) plane) := by
  sorry

end projection_area_l3751_375185


namespace divisible_by_three_l3751_375129

theorem divisible_by_three (n : ℕ) : ∃ k : ℤ, 2 * 7^n + 1 = 3 * k := by
  sorry

end divisible_by_three_l3751_375129


namespace thirtieth_term_value_l3751_375164

def arithmeticGeometricSequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  if n ≤ 4 then
    a₁ + (n - 1) * d
  else
    2 * arithmeticGeometricSequence a₁ d (n - 1)

theorem thirtieth_term_value :
  arithmeticGeometricSequence 4 3 30 = 436207104 := by
  sorry

end thirtieth_term_value_l3751_375164


namespace coffee_break_theorem_coffee_break_converse_l3751_375133

/-- Represents the number of participants who went for coffee -/
def coffee_drinkers : Finset ℕ := {6, 8, 10, 12}

/-- Represents the total number of participants -/
def total_participants : ℕ := 14

theorem coffee_break_theorem (n : ℕ) (hn : n ∈ coffee_drinkers) :
  ∃ (k : ℕ),
    -- k represents the number of pairs of participants who stayed
    0 < k ∧ 
    k < total_participants / 2 ∧
    -- n is the number of participants who left
    n = total_participants - 2 * k ∧
    -- Each remaining participant has exactly one neighbor who left
    ∀ (i : ℕ), i < total_participants → 
      (i % 2 = 0 → (i + 1) % total_participants < n) ∧
      (i % 2 = 1 → i < n) :=
by sorry

theorem coffee_break_converse :
  ∀ (n : ℕ),
    (∃ (k : ℕ),
      0 < k ∧ 
      k < total_participants / 2 ∧
      n = total_participants - 2 * k ∧
      ∀ (i : ℕ), i < total_participants → 
        (i % 2 = 0 → (i + 1) % total_participants < n) ∧
        (i % 2 = 1 → i < n)) →
    n ∈ coffee_drinkers :=
by sorry

end coffee_break_theorem_coffee_break_converse_l3751_375133


namespace progression_existence_l3751_375155

theorem progression_existence (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 6) (h3 : c = 4.5) :
  (∃ q : ℝ, q > 0 ∧ b = a * q ∧ c = b * q) ∧
  ¬(∃ d : ℝ, b = a + d ∧ c = b + d) := by
  sorry

end progression_existence_l3751_375155


namespace complex_fraction_simplification_l3751_375137

theorem complex_fraction_simplification :
  (2 + 2 * Complex.I) / (-3 + 4 * Complex.I) = -14/25 - 14/25 * Complex.I :=
by sorry

end complex_fraction_simplification_l3751_375137


namespace roundness_of_eight_million_l3751_375188

/-- Roundness of a positive integer is the sum of the exponents in its prime factorization -/
def roundness (n : ℕ+) : ℕ := sorry

/-- The roundness of 8,000,000 is 15 -/
theorem roundness_of_eight_million : roundness 8000000 = 15 := by sorry

end roundness_of_eight_million_l3751_375188


namespace probability_independent_events_l3751_375160

theorem probability_independent_events (a b : Set α) (p : Set α → ℝ) 
  (h1 : p a = 4/7)
  (h2 : p (a ∩ b) = 0.22857142857142856)
  (h3 : p (a ∩ b) = p a * p b) : 
  p b = 0.4 := by
  sorry

end probability_independent_events_l3751_375160


namespace train_crossing_time_l3751_375161

/-- Proves the time it takes for a train to cross a stationary man on a platform --/
theorem train_crossing_time (train_speed_kmph : ℝ) (train_speed_mps : ℝ) 
  (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  platform_length = 300 →
  platform_crossing_time = 33 →
  ∃ (train_length : ℝ),
    train_length = train_speed_mps * platform_crossing_time - platform_length ∧
    train_length / train_speed_mps = 18 :=
by sorry

end train_crossing_time_l3751_375161


namespace inequality_holds_l3751_375107

-- Define the real number a
variable (a : ℝ)

-- Define functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f, g, and a
axiom a_gt_one : a > 1
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_minus_g : ∀ x, f x - g x = a^x

-- State the theorem
theorem inequality_holds : g 0 < f 2 ∧ f 2 < f 3 := by sorry

end inequality_holds_l3751_375107


namespace new_students_average_age_l3751_375165

/-- Proves that the average age of new students is 32 years given the problem conditions --/
theorem new_students_average_age
  (original_average : ℝ)
  (original_strength : ℕ)
  (new_students : ℕ)
  (average_decrease : ℝ)
  (h1 : original_average = 40)
  (h2 : original_strength = 12)
  (h3 : new_students = 12)
  (h4 : average_decrease = 4) :
  let new_average := original_average - average_decrease
  let total_original := original_average * original_strength
  let total_new := new_average * (original_strength + new_students) - total_original
  total_new / new_students = 32 := by
  sorry

#check new_students_average_age

end new_students_average_age_l3751_375165


namespace min_value_of_a_l3751_375104

theorem min_value_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, x > 1 → x + a / (x - 1) ≥ 5) : 
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, x > 1 → x + b / (x - 1) ≥ 5) → b ≥ a) ∧ a = 4 :=
by sorry

end min_value_of_a_l3751_375104


namespace coefficient_x_squared_is_60_l3751_375187

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the general term of the expansion
def generalTerm (r : ℕ) : ℚ :=
  (-1)^r * binomial 6 r * 2^r

-- Theorem statement
theorem coefficient_x_squared_is_60 :
  generalTerm 2 = 60 := by sorry

end coefficient_x_squared_is_60_l3751_375187


namespace imaginary_part_of_complex_l3751_375162

theorem imaginary_part_of_complex (z : ℂ) : z = 1 - 2*I → Complex.im z = -2 := by
  sorry

end imaginary_part_of_complex_l3751_375162


namespace geometric_sequence_ratio_l3751_375143

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 5 * a 7 = 2) →
  (a 2 + a 10 = 3) →
  (a 12 / a 4 = 2 ∨ a 12 / a 4 = 1/2) :=
by sorry

end geometric_sequence_ratio_l3751_375143


namespace inequality_solution_range_l3751_375138

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3*a) ↔ (a < -1 ∨ a > 4) := by
  sorry

end inequality_solution_range_l3751_375138


namespace multiplication_preserves_odd_positives_l3751_375154

def P : Set ℕ := {n : ℕ | n % 2 = 1 ∧ n > 0}

def M : Set ℕ := {x : ℕ | ∃ (a b : ℕ), a ∈ P ∧ b ∈ P ∧ x = a * b}

theorem multiplication_preserves_odd_positives (h : M ⊆ P) :
  ∀ (a b : ℕ), a ∈ P → b ∈ P → (a * b) ∈ P := by
  sorry

end multiplication_preserves_odd_positives_l3751_375154


namespace medicine_A_count_l3751_375125

/-- The number of tablets of medicine B in the box -/
def medicine_B : ℕ := 16

/-- The minimum number of tablets to extract to ensure at least two of each kind -/
def min_extract : ℕ := 18

/-- The number of tablets of medicine A in the box -/
def medicine_A : ℕ := 3

theorem medicine_A_count : 
  ∀ (x : ℕ), 
  (x = medicine_A) ↔ 
  (x > 0 ∧ 
   x + medicine_B ≥ min_extract ∧ 
   x - 1 + medicine_B < min_extract) :=
by sorry

end medicine_A_count_l3751_375125


namespace rotate_point_on_circle_l3751_375108

/-- Given a circle with radius 5 centered at the origin, 
    prove that rotating the point (3,4) by 45 degrees counterclockwise 
    results in the point (-√2/2, 7√2/2) -/
theorem rotate_point_on_circle (P Q : ℝ × ℝ) : 
  P.1^2 + P.2^2 = 25 →  -- P is on the circle
  P = (3, 4) →  -- P starts at (3,4)
  Q.1 = P.1 * (Real.sqrt 2 / 2) - P.2 * (Real.sqrt 2 / 2) →  -- Q is P rotated 45°
  Q.2 = P.1 * (Real.sqrt 2 / 2) + P.2 * (Real.sqrt 2 / 2) →
  Q = (-Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2) := by
  sorry

end rotate_point_on_circle_l3751_375108


namespace inequality_proof_l3751_375136

theorem inequality_proof (a b c x y z k : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0) 
  (h_sum : a + x = k ∧ b + y = k ∧ c + z = k) : 
  a * x + b * y + c * z < k^2 := by
sorry

end inequality_proof_l3751_375136


namespace recipe_eggs_l3751_375183

theorem recipe_eggs (total_eggs : ℕ) (rotten_eggs : ℕ) (prob_all_rotten : ℝ) :
  total_eggs = 36 →
  rotten_eggs = 3 →
  prob_all_rotten = 0.0047619047619047615 →
  ∃ (n : ℕ), (rotten_eggs : ℝ) / (total_eggs : ℝ) ^ n = prob_all_rotten ∧ n = 2 :=
by sorry

end recipe_eggs_l3751_375183


namespace bouncing_ball_distance_l3751_375196

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := (List.range (bounces + 1)).map (λ i => initialHeight * bounceRatio^i)
  let ascendDistances := (List.range bounces).map (λ i => initialHeight * bounceRatio^(i+1))
  (descendDistances.sum + ascendDistances.sum)

/-- Theorem: A ball dropped from 25 meters, bouncing 2/3 of its previous height each time,
    and caught after the 4th bounce, travels approximately 88 meters. -/
theorem bouncing_ball_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |totalDistance 25 (2/3) 4 - 88| < ε :=
sorry

end bouncing_ball_distance_l3751_375196


namespace sum_of_roots_cubic_equation_l3751_375193

theorem sum_of_roots_cubic_equation : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 12*x) / (x + 3)
  ∃ (x₁ x₂ : ℝ), (f x₁ = 7 ∧ f x₂ = 7 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 4 :=
by sorry

end sum_of_roots_cubic_equation_l3751_375193


namespace triangle_side_calculation_l3751_375124

-- Define a triangle with sides and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_side_calculation (t : Triangle) 
  (h1 : t.A = 30 * π / 180)
  (h2 : t.C = 45 * π / 180)
  (h3 : t.a = 4) :
  t.c = 4 * Real.sqrt 2 := by
  sorry

end triangle_side_calculation_l3751_375124


namespace tank_fill_time_l3751_375195

/-- Represents the time (in hours) it takes for a pipe to empty or fill the tank when working alone -/
structure PipeTime where
  A : ℝ  -- Time for pipe A to empty the tank
  B : ℝ  -- Time for pipe B to empty the tank
  C : ℝ  -- Time for pipe C to fill the tank

/-- Conditions for the tank filling problem -/
def TankConditions (t : PipeTime) : Prop :=
  (1 / t.C - 1 / t.A) * 2 = 1 ∧
  (1 / t.C - 1 / t.B) * 4 = 1 ∧
  1 / t.C * 5 - (1 / t.A + 1 / t.B) * 8 = 0

/-- The main theorem stating the time to fill the tank using only pipe C -/
theorem tank_fill_time (t : PipeTime) (h : TankConditions t) : t.C = 11/6 := by
  sorry

end tank_fill_time_l3751_375195


namespace terms_before_five_l3751_375152

/-- An arithmetic sequence with first term 95 and common difference -5 -/
def arithmeticSequence (n : ℕ) : ℤ := 95 - 5 * (n - 1)

theorem terms_before_five : 
  (∃ n : ℕ, arithmeticSequence n = 5) ∧ 
  (∀ k : ℕ, k < 19 → arithmeticSequence k > 5) :=
by sorry

end terms_before_five_l3751_375152


namespace quadratic_inequality_counterexample_l3751_375121

theorem quadratic_inequality_counterexample :
  ∃ (a b c : ℝ), b^2 - 4*a*c ≤ 0 ∧ ∃ (x : ℝ), a*x^2 + b*x + c < 0 :=
sorry

end quadratic_inequality_counterexample_l3751_375121


namespace parabola_directrix_l3751_375171

/-- A parabola with equation y² = -8x that opens to the left has a directrix with equation x = 2 -/
theorem parabola_directrix (y x : ℝ) : 
  (y^2 = -8*x) → 
  (∃ p : ℝ, y^2 = -4*p*x ∧ p > 0) → 
  (∃ a : ℝ, a = 2 ∧ ∀ x₀ y₀ : ℝ, y₀^2 = -8*x₀ → |x₀ - a| = |y₀|/4) :=
by sorry

end parabola_directrix_l3751_375171


namespace line_ellipse_intersection_slopes_l3751_375142

theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 8) ↔ 
  m ≤ -Real.sqrt 2.4 ∨ m ≥ Real.sqrt 2.4 := by
sorry

end line_ellipse_intersection_slopes_l3751_375142


namespace quadratic_discriminant_zero_implies_perfect_square_l3751_375114

/-- 
If the discriminant of a quadratic equation ax^2 + bx + c = 0 is zero,
then the left-hand side is a perfect square.
-/
theorem quadratic_discriminant_zero_implies_perfect_square
  (a b c : ℝ) (h : b^2 - 4*a*c = 0) :
  ∃ k : ℝ, ∀ x : ℝ, a*x^2 + b*x + c = k*(2*a*x + b)^2 := by
  sorry

end quadratic_discriminant_zero_implies_perfect_square_l3751_375114


namespace count_three_digit_numbers_l3751_375170

def digit := Fin 4

def valid_first_digit (d : digit) : Prop := d.val ≠ 0

def three_digit_number := { n : ℕ | 100 ≤ n ∧ n < 1000 }

def count_valid_numbers : ℕ := sorry

theorem count_three_digit_numbers : count_valid_numbers = 48 := by sorry

end count_three_digit_numbers_l3751_375170


namespace eggs_per_friend_l3751_375101

/-- Proves that sharing 16 eggs equally among 8 friends results in 2 eggs per friend -/
theorem eggs_per_friend (total_eggs : ℕ) (num_friends : ℕ) (eggs_per_friend : ℕ) 
  (h1 : total_eggs = 16) 
  (h2 : num_friends = 8) 
  (h3 : eggs_per_friend * num_friends = total_eggs) : 
  eggs_per_friend = 2 := by
  sorry

end eggs_per_friend_l3751_375101


namespace no_fraction_satisfies_condition_l3751_375172

theorem no_fraction_satisfies_condition : ¬∃ (x y : ℕ+), 
  (Nat.gcd x.val y.val = 1) ∧ 
  ((x + 2 : ℚ) / (y + 2) = 1.2 * (x : ℚ) / y) := by
  sorry

end no_fraction_satisfies_condition_l3751_375172


namespace only_valid_number_l3751_375103

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  n = (n / 10)^2 + (n % 10)^2 + 13

theorem only_valid_number : ∀ n : ℕ, is_valid_number n ↔ n = 54 := by sorry

end only_valid_number_l3751_375103


namespace freshWaterCostForFamily_l3751_375184

/-- The cost of fresh water for a day for a family, given the cost per gallon, 
    daily water need per person, and number of family members. -/
def freshWaterCost (costPerGallon : ℚ) (dailyNeedPerPerson : ℚ) (familySize : ℕ) : ℚ :=
  costPerGallon * dailyNeedPerPerson * familySize

/-- Theorem stating that the cost of fresh water for a day for a family of 6 is $3, 
    given the specified conditions. -/
theorem freshWaterCostForFamily : 
  freshWaterCost 1 (1/2) 6 = 3 := by
  sorry


end freshWaterCostForFamily_l3751_375184


namespace unique_bases_sum_l3751_375194

theorem unique_bases_sum : ∃! (R₃ R₄ : ℕ), 
  (R₃ > 0 ∧ R₄ > 0) ∧
  ((4 * R₃ + 6) * (R₄^2 - 1) = (4 * R₄ + 9) * (R₃^2 - 1)) ∧
  ((6 * R₃ + 4) * (R₄^2 - 1) = (9 * R₄ + 4) * (R₃^2 - 1)) ∧
  (R₃ + R₄ = 23) := by
  sorry

end unique_bases_sum_l3751_375194


namespace average_age_proof_l3751_375150

/-- Given the ages of John, Mary, and Tonya with specific relationships, prove their average age --/
theorem average_age_proof (tonya mary john : ℕ) 
  (h1 : john = 2 * mary)
  (h2 : john * 2 = tonya)
  (h3 : tonya = 60) : 
  (tonya + john + mary) / 3 = 35 := by
  sorry

end average_age_proof_l3751_375150


namespace last_nonzero_digit_of_b_d_is_five_l3751_375180

/-- Definition of b_n -/
def b (n : ℕ+) : ℕ := 2 * (Nat.factorial (n + 10) / Nat.factorial (n + 2))

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The smallest positive integer d such that the last nonzero digit of b(d) is odd -/
def d : ℕ+ := sorry

theorem last_nonzero_digit_of_b_d_is_five :
  lastNonzeroDigit (b d) = 5 := by sorry

end last_nonzero_digit_of_b_d_is_five_l3751_375180


namespace find_h_of_x_l3751_375182

theorem find_h_of_x (x : ℝ) (h : ℝ → ℝ) : 
  (9 * x^3 - 3 * x + 1 + h x = 3 * x^2 - 5 * x + 3) → 
  (h x = -9 * x^3 + 3 * x^2 - 2 * x + 2) := by
sorry

end find_h_of_x_l3751_375182


namespace stratified_sampling_population_size_l3751_375113

theorem stratified_sampling_population_size
  (x : ℕ) -- number of individuals in stratum A
  (y : ℕ) -- number of individuals in stratum B
  (h1 : (20 : ℚ) * y / (x + y) = (1 : ℚ) / 12 * y) -- equation from stratified sampling
  : x + y = 240 := by
  sorry

end stratified_sampling_population_size_l3751_375113


namespace equation_represents_hyperbola_l3751_375189

/-- The equation (x+y)^2 = x^2 + y^2 + 4 represents a hyperbola in the xy-plane. -/
theorem equation_represents_hyperbola :
  ∃ (f : ℝ → ℝ → Prop), (∀ x y : ℝ, f x y ↔ (x + y)^2 = x^2 + y^2 + 4) ∧
  (∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ x * y = a) :=
by sorry

end equation_represents_hyperbola_l3751_375189


namespace round_0_0984_to_two_sig_figs_l3751_375153

/-- Rounds a number to a specified number of significant figures -/
def roundToSignificantFigures (x : ℝ) (n : ℕ) : ℝ :=
  sorry

/-- Theorem: Rounding 0.0984 to two significant figures results in 0.098 -/
theorem round_0_0984_to_two_sig_figs :
  roundToSignificantFigures 0.0984 2 = 0.098 := by
  sorry

end round_0_0984_to_two_sig_figs_l3751_375153


namespace inequality_not_hold_l3751_375176

theorem inequality_not_hold (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  ¬(1 / (a - b) > 1 / a) := by
  sorry

end inequality_not_hold_l3751_375176


namespace distance_representation_l3751_375135

theorem distance_representation (a : ℝ) : 
  |a + 1| = |a - (-1)| := by sorry

-- The statement proves that |a + 1| is equal to the distance between a and -1,
-- which represents the distance between points A and C on the number line.

end distance_representation_l3751_375135


namespace dice_probability_l3751_375122

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice re-rolled -/
def numRerolled : ℕ := 3

/-- The total number of possible outcomes when re-rolling -/
def totalOutcomes : ℕ := numFaces ^ numRerolled

/-- The number of ways the re-rolled dice can not match the pair -/
def waysNotMatchingPair : ℕ := (numFaces - 1) ^ numRerolled

/-- The number of ways at least one re-rolled die matches the pair -/
def waysMatchingPair : ℕ := totalOutcomes - waysNotMatchingPair

/-- The number of ways all re-rolled dice match each other -/
def waysAllMatch : ℕ := numFaces

/-- The number of successful outcomes -/
def successfulOutcomes : ℕ := waysMatchingPair + waysAllMatch - 1

/-- The probability of at least three dice showing the same value after re-rolling -/
def probability : ℚ := successfulOutcomes / totalOutcomes

theorem dice_probability : probability = 4 / 9 := by
  sorry

end dice_probability_l3751_375122


namespace carpet_square_cost_l3751_375145

/-- Calculates the cost of each carpet square given the floor dimensions, carpet square dimensions, and total cost. -/
theorem carpet_square_cost
  (floor_length : ℝ)
  (floor_width : ℝ)
  (square_side : ℝ)
  (total_cost : ℝ)
  (h1 : floor_length = 24)
  (h2 : floor_width = 64)
  (h3 : square_side = 8)
  (h4 : total_cost = 576)
  : (total_cost / ((floor_length * floor_width) / (square_side * square_side))) = 24 :=
by
  sorry

#check carpet_square_cost

end carpet_square_cost_l3751_375145


namespace water_barrel_problem_l3751_375156

theorem water_barrel_problem :
  ∀ (bucket_capacity : ℕ),
    bucket_capacity > 0 →
    bucket_capacity / 2 +
    bucket_capacity / 3 +
    bucket_capacity / 4 +
    bucket_capacity / 5 +
    bucket_capacity / 6 = 29 →
    bucket_capacity ≤ 30 →
    29 ≤ 30 :=
by
  sorry

end water_barrel_problem_l3751_375156


namespace probability_two_females_selected_l3751_375179

def total_contestants : ℕ := 7
def female_contestants : ℕ := 4
def male_contestants : ℕ := 3

theorem probability_two_females_selected :
  (Nat.choose female_contestants 2 : ℚ) / (Nat.choose total_contestants 2 : ℚ) = 2 / 7 := by
  sorry

end probability_two_females_selected_l3751_375179


namespace brand_y_pen_price_l3751_375126

theorem brand_y_pen_price
  (price_x : ℝ)
  (total_pens : ℕ)
  (total_cost : ℝ)
  (num_x_pens : ℕ)
  (h1 : price_x = 4)
  (h2 : total_pens = 12)
  (h3 : total_cost = 40)
  (h4 : num_x_pens = 8) :
  (total_cost - price_x * num_x_pens) / (total_pens - num_x_pens) = 2 := by
  sorry

end brand_y_pen_price_l3751_375126


namespace odd_k_triple_f_81_l3751_375197

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then 2 * n + 3
  else if n % 3 = 0 ∧ n % 2 ≠ 0 then n / 3
  else n  -- This case is not specified in the original problem, so we leave n unchanged

theorem odd_k_triple_f_81 (k : ℤ) (h_odd : k % 2 = 1) (h_triple_f : f (f (f k)) = 81) : k = 57 := by
  sorry

end odd_k_triple_f_81_l3751_375197


namespace base_8_to_10_conversion_l3751_375134

-- Define the base 8 number as a list of digits
def base_8_number : List Nat := [2, 4, 6]

-- Define the conversion function from base 8 to base 10
def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

-- Theorem statement
theorem base_8_to_10_conversion :
  base_8_to_10 base_8_number = 166 := by sorry

end base_8_to_10_conversion_l3751_375134


namespace angle_bisector_exists_l3751_375105

-- Define the basic structures
structure Point :=
  (x : ℝ) (y : ℝ)

structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define the given lines
def L1 : Line := sorry
def L2 : Line := sorry

-- Define the property of inaccessible intersection
def inaccessibleIntersection (l1 l2 : Line) : Prop := sorry

-- Define angle bisector
def isAngleBisector (bisector : Line) (l1 l2 : Line) : Prop := sorry

-- Theorem statement
theorem angle_bisector_exists (h : inaccessibleIntersection L1 L2) :
  ∃ bisector : Line, isAngleBisector bisector L1 L2 := by
  sorry

end angle_bisector_exists_l3751_375105


namespace red_balls_count_l3751_375118

theorem red_balls_count (total : ℕ) (prob : ℚ) : 
  total = 15 → 
  prob = 1 / 35 →
  (∃ r : ℕ, r ≤ total ∧ 
    prob = (r * (r - 1) * (r - 2)) / (total * (total - 1) * (total - 2))) →
  (∃ r : ℕ, r = 7 ∧ r ≤ total ∧ 
    prob = (r * (r - 1) * (r - 2)) / (total * (total - 1) * (total - 2))) :=
by sorry

end red_balls_count_l3751_375118


namespace softball_team_size_l3751_375132

/-- Proves that a co-ed softball team with given conditions has 20 total players -/
theorem softball_team_size : ∀ (men women : ℕ),
  women = men + 4 →
  (men : ℚ) / (women : ℚ) = 2/3 →
  men + women = 20 := by
sorry

end softball_team_size_l3751_375132


namespace distance_swum_back_l3751_375191

/-- Calculates the distance swum against the current given swimming speed, water speed, and time -/
def distance_against_current (swimming_speed water_speed : ℝ) (time : ℝ) : ℝ :=
  (swimming_speed - water_speed) * time

/-- Proves that the distance swum against the current is 8 km given the specified conditions -/
theorem distance_swum_back (swimming_speed water_speed time : ℝ) 
  (h1 : swimming_speed = 12)
  (h2 : water_speed = 10)
  (h3 : time = 4) :
  distance_against_current swimming_speed water_speed time = 8 := by
  sorry

#check distance_swum_back

end distance_swum_back_l3751_375191


namespace ivan_bought_ten_cards_l3751_375123

/-- The number of Uno Giant Family Cards Ivan bought -/
def num_cards : ℕ := 10

/-- The original price of each card in dollars -/
def original_price : ℚ := 12

/-- The discount per card in dollars -/
def discount : ℚ := 2

/-- The total amount Ivan paid in dollars -/
def total_paid : ℚ := 100

/-- Theorem stating that Ivan bought 10 Uno Giant Family Cards -/
theorem ivan_bought_ten_cards :
  (original_price - discount) * num_cards = total_paid :=
by sorry

end ivan_bought_ten_cards_l3751_375123


namespace digits_of_3_pow_15_times_5_pow_10_l3751_375159

theorem digits_of_3_pow_15_times_5_pow_10 : 
  (Nat.log 10 (3^15 * 5^10) + 1 : ℕ) = 18 :=
sorry

end digits_of_3_pow_15_times_5_pow_10_l3751_375159


namespace find_n_l3751_375178

theorem find_n : ∃ n : ℕ, 
  50 ≤ n ∧ n ≤ 120 ∧ 
  n % 8 = 0 ∧ 
  n % 12 = 4 ∧ 
  n % 7 = 4 ∧
  n = 88 := by
  sorry

end find_n_l3751_375178


namespace five_people_lineup_l3751_375111

/-- The number of ways to arrange n people in a line where k people cannot be first -/
def arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n - k) * n.factorial

/-- The problem statement -/
theorem five_people_lineup : arrangements 5 2 = 72 := by
  sorry

end five_people_lineup_l3751_375111


namespace radical_axis_properties_l3751_375157

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def power (p : ℝ × ℝ) (c : Circle) : ℝ :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 - c.radius^2

def perpendicular (l1 l2 : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let ((x1, y1), (x2, y2)) := l1
  let ((x3, y3), (x4, y4)) := l2
  (x2 - x1) * (x4 - x3) + (y2 - y1) * (y4 - y3) = 0

def on_line (p : ℝ × ℝ) (l : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (x, y) := p
  let ((x1, y1), (x2, y2)) := l
  (y2 - y1) * (x - x1) = (x2 - x1) * (y - y1)

-- Define the theorem
theorem radical_axis_properties 
  (k₁ k₂ : Circle) 
  (P Q : ℝ × ℝ) 
  (h_non_intersect : k₁ ≠ k₂) 
  (h_power_P : power P k₁ = power P k₂)
  (h_power_Q : power Q k₁ = power Q k₂) :
  let O₁ := k₁.center
  let O₂ := k₂.center
  (perpendicular (P, Q) (O₁, O₂)) ∧ 
  (∀ S, (power S k₁ = power S k₂) ↔ on_line S (P, Q)) ∧
  (∀ k : Circle, 
    (∃ x y, power (x, y) k = power (x, y) k₁ ∧ power (x, y) k = power (x, y) k₂) →
    (∃ M, (power M k = power M k₁) ∧ (power M k = power M k₂) ∧ 
          (on_line M (P, Q) ∨ perpendicular (P, Q) (k.center, M)))) := by
  sorry

end radical_axis_properties_l3751_375157


namespace arithmetic_mean_of_special_set_l3751_375148

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 1) :
  let special_number := 1 + 1 / n
  let regular_number := 1
  let sum := special_number + (n - 1) * regular_number
  sum / n = 1 + 1 / n^2 := by sorry

end arithmetic_mean_of_special_set_l3751_375148


namespace trees_planted_today_is_41_l3751_375190

-- Define the initial number of trees
def initial_trees : Nat := 39

-- Define the number of trees to be planted tomorrow
def trees_tomorrow : Nat := 20

-- Define the final number of trees
def final_trees : Nat := 100

-- Define the number of trees planted today
def trees_planted_today : Nat := final_trees - initial_trees - trees_tomorrow

-- Theorem to prove
theorem trees_planted_today_is_41 : trees_planted_today = 41 := by
  sorry

end trees_planted_today_is_41_l3751_375190


namespace base_conversion_sum_l3751_375151

/-- Converts a number from base 11 to base 10 -/
def base11ToBase10 (n : Nat) : Nat :=
  (n / 100) * 121 + ((n / 10) % 10) * 11 + (n % 10)

/-- Converts a number from base 12 to base 10 -/
def base12ToBase10 (n : Nat) (A B : Nat) : Nat :=
  (n / 100) * 144 + ((n / 10) % 10) * 12 + (n % 10)

theorem base_conversion_sum :
  let n1 : Nat := 249
  let n2 : Nat := 3 * 100 + 10 * 10 + 11
  let A : Nat := 10
  let B : Nat := 11
  base11ToBase10 n1 + base12ToBase10 n2 A B = 858 := by
  sorry

end base_conversion_sum_l3751_375151


namespace function_satisfying_lcm_gcd_condition_l3751_375130

theorem function_satisfying_lcm_gcd_condition :
  ∀ (f : ℕ → ℕ),
    (∀ (m n : ℕ), m > 0 ∧ n > 0 → f (m * n) = Nat.lcm m n * Nat.gcd (f m) (f n)) →
    ∃ (k : ℕ), k > 0 ∧ ∀ (x : ℕ), f x = k * x :=
by sorry

end function_satisfying_lcm_gcd_condition_l3751_375130


namespace origami_paper_distribution_l3751_375173

theorem origami_paper_distribution (total_papers : ℝ) (num_cousins : ℝ) 
  (h1 : total_papers = 48.0)
  (h2 : num_cousins = 6.0)
  (h3 : num_cousins ≠ 0) :
  total_papers / num_cousins = 8.0 := by
sorry

end origami_paper_distribution_l3751_375173


namespace sum_distances_specific_triangle_l3751_375141

/-- The sum of distances from a point to the vertices of a triangle, expressed as x + y√z --/
def sum_distances (A B C P : ℝ × ℝ) : ℝ × ℝ × ℕ :=
  sorry

/-- Theorem stating the sum of distances for specific triangle and point --/
theorem sum_distances_specific_triangle :
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (10, 2)
  let C : ℝ × ℝ := (5, 4)
  let P : ℝ × ℝ := (3, 1)
  let (x, y, z) := sum_distances A B C P
  x + y + z = 16 :=
by sorry

end sum_distances_specific_triangle_l3751_375141


namespace dropped_players_not_necessarily_played_each_other_l3751_375110

/-- Represents a round-robin chess tournament --/
structure ChessTournament where
  n : ℕ  -- Total number of participants
  games_played : ℕ  -- Total number of games played
  dropped_players : ℕ  -- Number of players who dropped out

/-- Calculates the total number of games in a round-robin tournament --/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a specific tournament scenario, dropped players didn't necessarily play each other --/
theorem dropped_players_not_necessarily_played_each_other 
  (t : ChessTournament) 
  (h1 : t.games_played = 23) 
  (h2 : t.dropped_players = 2) 
  (h3 : ∃ k : ℕ, t.n = k + t.dropped_players) 
  (h4 : ∃ m : ℕ, m * t.dropped_players = t.games_played - total_games (t.n - t.dropped_players)) :
  ¬ (∀ dropped_player_games : ℕ, dropped_player_games * t.dropped_players = t.games_played - total_games (t.n - t.dropped_players) → dropped_player_games = t.n - t.dropped_players - 1) :=
sorry

end dropped_players_not_necessarily_played_each_other_l3751_375110


namespace coin_not_touching_lines_l3751_375144

/-- The probability that a randomly tossed coin doesn't touch parallel lines -/
theorem coin_not_touching_lines (a r : ℝ) (h : r < a) :
  let p := (a - r) / a
  0 ≤ p ∧ p ≤ 1 ∧ p = (a - r) / a :=
by sorry

end coin_not_touching_lines_l3751_375144


namespace rocky_path_trail_length_l3751_375147

/-- Represents the length of Phoenix's hike on the Rocky Path Trail -/
def rocky_path_trail (a b c d e : ℝ) : Prop :=
  a + b = 24 ∧
  b + c = 28 ∧
  c + d + e = 36 ∧
  a + c = 28

theorem rocky_path_trail_length :
  ∀ a b c d e : ℝ, rocky_path_trail a b c d e → a + b + c + d + e = 60 :=
by
  sorry

end rocky_path_trail_length_l3751_375147


namespace total_candidates_l3751_375106

theorem total_candidates (girls : ℕ) (boys_fail_rate : ℝ) (girls_fail_rate : ℝ) (total_fail_rate : ℝ) :
  girls = 900 →
  boys_fail_rate = 0.7 →
  girls_fail_rate = 0.68 →
  total_fail_rate = 0.691 →
  ∃ (total : ℕ), total = 2000 ∧ 
    (boys_fail_rate * (total - girls) + girls_fail_rate * girls) / total = total_fail_rate :=
by sorry

end total_candidates_l3751_375106


namespace ellipse_focal_length_l3751_375181

/-- The focal length of the ellipse 2x^2 + 3y^2 = 6 is 2 -/
theorem ellipse_focal_length : 
  let ellipse := {(x, y) : ℝ × ℝ | 2 * x^2 + 3 * y^2 = 6}
  ∃ (f : ℝ), f = 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ ellipse → 
      ∃ (c₁ c₂ : ℝ × ℝ), 
        (c₁.1 - x)^2 + (c₁.2 - y)^2 + (c₂.1 - x)^2 + (c₂.2 - y)^2 = f^2 :=
by sorry

end ellipse_focal_length_l3751_375181


namespace dog_kennel_problem_l3751_375166

theorem dog_kennel_problem (total : ℕ) (long_fur : ℕ) (brown : ℕ) (neither : ℕ) 
  (h1 : total = 45)
  (h2 : long_fur = 36)
  (h3 : brown = 27)
  (h4 : neither = 8)
  : total - neither - (long_fur + brown - (total - neither)) = 26 := by
  sorry

end dog_kennel_problem_l3751_375166


namespace rachel_total_steps_l3751_375102

/-- The total number of steps Rachel took during her trip to the Eiffel Tower -/
def total_steps (steps_up steps_down : ℕ) : ℕ := steps_up + steps_down

/-- Theorem stating that Rachel took 892 steps in total -/
theorem rachel_total_steps : total_steps 567 325 = 892 := by
  sorry

end rachel_total_steps_l3751_375102


namespace largest_x_satisfying_equation_l3751_375146

theorem largest_x_satisfying_equation : ∃ (x : ℝ), 
  (∀ y : ℝ, ⌊y⌋ / y = 8 / 9 → y ≤ x) ∧ 
  ⌊x⌋ / x = 8 / 9 ∧ 
  x = 63 / 8 := by
  sorry

end largest_x_satisfying_equation_l3751_375146


namespace xiaoming_age_l3751_375116

def is_valid_age (birth_year : ℕ) (current_year : ℕ) : Prop :=
  current_year - birth_year = (birth_year / 1000) + ((birth_year / 100) % 10) + ((birth_year / 10) % 10) + (birth_year % 10)

theorem xiaoming_age :
  ∃ (age : ℕ), (age = 22 ∨ age = 4) ∧
  ∃ (birth_year : ℕ),
    birth_year ≥ 1900 ∧
    birth_year < 2015 ∧
    is_valid_age birth_year 2015 ∧
    age = 2015 - birth_year :=
by sorry

end xiaoming_age_l3751_375116


namespace girls_exceed_boys_by_69_l3751_375140

/-- Proves that in a class of 485 students with 208 boys, the number of girls exceeds the number of boys by 69 -/
theorem girls_exceed_boys_by_69 :
  let total_students : ℕ := 485
  let num_boys : ℕ := 208
  let num_girls : ℕ := total_students - num_boys
  num_girls - num_boys = 69 := by sorry

end girls_exceed_boys_by_69_l3751_375140


namespace reshuffling_theorem_l3751_375175

def total_employees : ℕ := 10000

def current_proportions : List (String × ℚ) := [
  ("Senior Managers", 2/5),
  ("Junior Managers", 3/10),
  ("Engineers", 1/5),
  ("Marketing Team", 1/10)
]

def desired_proportions : List (String × ℚ) := [
  ("Senior Managers", 7/20),
  ("Junior Managers", 1/5),
  ("Engineers", 1/4),
  ("Marketing Team", 1/5)
]

def calculate_changes (current : List (String × ℚ)) (desired : List (String × ℚ)) (total : ℕ) : 
  List (String × ℤ) :=
  sorry

theorem reshuffling_theorem : 
  calculate_changes current_proportions desired_proportions total_employees = 
    [("Senior Managers", -500), 
     ("Junior Managers", -1000), 
     ("Engineers", 500), 
     ("Marketing Team", 1000)] :=
by sorry

end reshuffling_theorem_l3751_375175


namespace units_digit_of_2_power_2018_l3751_375128

theorem units_digit_of_2_power_2018 : ∃ (f : ℕ → ℕ), 
  (∀ n, f n = n % 4) ∧ 
  (∀ n, n > 0 → (2^n % 10 = 2 ∨ 2^n % 10 = 4 ∨ 2^n % 10 = 8 ∨ 2^n % 10 = 6)) ∧
  (2^2018 % 10 = 4) := by
  sorry

end units_digit_of_2_power_2018_l3751_375128


namespace system_1_solution_l3751_375163

theorem system_1_solution (x y : ℝ) :
  x = y + 1 ∧ 4 * x - 3 * y = 5 → x = 2 ∧ y = 1 := by sorry

end system_1_solution_l3751_375163


namespace cubic_range_l3751_375192

theorem cubic_range (x : ℝ) (h : x^2 - 5*x + 6 < 0) :
  41 < x^3 + 5*x^2 + 6*x + 1 ∧ x^3 + 5*x^2 + 6*x + 1 < 91 := by
  sorry

end cubic_range_l3751_375192


namespace problem_statement_l3751_375198

theorem problem_statement (p : ℝ) (h : 126 * 3^8 = p) : 126 * 3^6 = (1/9) * p := by
  sorry

end problem_statement_l3751_375198


namespace gcf_of_40_120_80_l3751_375158

theorem gcf_of_40_120_80 : Nat.gcd 40 (Nat.gcd 120 80) = 40 := by
  sorry

end gcf_of_40_120_80_l3751_375158


namespace commercial_break_duration_l3751_375127

theorem commercial_break_duration :
  let five_minute_commercials : ℕ := 3
  let two_minute_commercials : ℕ := 11
  let five_minute_duration : ℕ := 5
  let two_minute_duration : ℕ := 2
  (five_minute_commercials * five_minute_duration + two_minute_commercials * two_minute_duration) = 37 := by
  sorry

end commercial_break_duration_l3751_375127


namespace total_garden_area_l3751_375131

-- Define the garden dimensions and counts for each person
def mancino_gardens : ℕ := 4
def mancino_length : ℕ := 16
def mancino_width : ℕ := 5

def marquita_gardens : ℕ := 3
def marquita_length : ℕ := 8
def marquita_width : ℕ := 4

def matteo_gardens : ℕ := 2
def matteo_length : ℕ := 12
def matteo_width : ℕ := 6

def martina_gardens : ℕ := 5
def martina_length : ℕ := 10
def martina_width : ℕ := 3

-- Theorem stating the total square footage of all gardens
theorem total_garden_area :
  mancino_gardens * mancino_length * mancino_width +
  marquita_gardens * marquita_length * marquita_width +
  matteo_gardens * matteo_length * matteo_width +
  martina_gardens * martina_length * martina_width = 710 := by
  sorry

end total_garden_area_l3751_375131


namespace max_profit_is_45_6_l3751_375109

/-- Profit function for location A -/
def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def profit_B (x : ℝ) : ℝ := 2 * x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ 15 ∧ 
  (∀ y : ℝ, y ≥ 0 → y ≤ 15 → total_profit y ≤ total_profit x) ∧
  total_profit x = 45.6 :=
sorry

end max_profit_is_45_6_l3751_375109


namespace square_root_squared_specific_square_root_squared_l3751_375119

theorem square_root_squared (n : ℝ) (h : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by
  sorry

theorem specific_square_root_squared : (Real.sqrt 625681) ^ 2 = 625681 := by
  sorry

end square_root_squared_specific_square_root_squared_l3751_375119


namespace matrix_inverse_scalar_multiple_l3751_375100

/-- Given a 2x2 matrix A with elements [[1, 3], [4, d]] where A⁻¹ = k * A,
    prove that d = 6 and k = 1/6 -/
theorem matrix_inverse_scalar_multiple (d k : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 3; 4, d]
  A⁻¹ = k • A → d = 6 ∧ k = 1/6 := by
  sorry

end matrix_inverse_scalar_multiple_l3751_375100


namespace decimal_to_fraction_035_l3751_375174

def decimal_to_fraction (d : ℚ) : ℕ × ℕ := sorry

theorem decimal_to_fraction_035 :
  (decimal_to_fraction 0.35).1 = 7 := by sorry

end decimal_to_fraction_035_l3751_375174


namespace parallelogram_probability_theorem_l3751_375186

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- The probability of a point in a parallelogram being not below the x-axis -/
def probability_not_below_x_axis (para : Parallelogram) : ℝ := 
  sorry

theorem parallelogram_probability_theorem (para : Parallelogram) :
  para.P = Point.mk 4 4 →
  para.Q = Point.mk (-2) (-2) →
  para.R = Point.mk (-8) (-2) →
  para.S = Point.mk (-2) 4 →
  probability_not_below_x_axis para = 1/2 := by
  sorry

end parallelogram_probability_theorem_l3751_375186
