import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3741_374122

theorem sum_of_x_and_y (a x y : ℝ) (hx : a / x = 1 / 3) (hy : a / y = 1 / 4) :
  x + y = 7 * a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3741_374122


namespace NUMINAMATH_CALUDE_proposition_negations_and_converses_l3741_374131

-- Definition of odd numbers
def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Definition of even numbers
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Definition of prime numbers
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem proposition_negations_and_converses :
  -- Proposition 1
  (¬(∀ x y : ℤ, isOdd x ∧ isOdd y → isEven (x + y)) = False) ∧
  ((∀ x y : ℤ, ¬(isOdd x ∧ isOdd y) → ¬(isEven (x + y))) = False) ∧
  -- Proposition 2
  (¬(∀ x y : ℝ, x * y = 0 → x = 0 ∨ y = 0) = False) ∧
  ((∀ x y : ℝ, x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0) = True) ∧
  -- Proposition 3
  (¬(∀ n : ℕ, isPrime n → isOdd n) = False) ∧
  ((∀ n : ℕ, ¬(isPrime n) → ¬(isOdd n)) = False) := by
sorry

end NUMINAMATH_CALUDE_proposition_negations_and_converses_l3741_374131


namespace NUMINAMATH_CALUDE_beef_purchase_l3741_374169

theorem beef_purchase (initial_budget : ℕ) (chicken_cost : ℕ) (beef_cost_per_pound : ℕ) (remaining_budget : ℕ)
  (h1 : initial_budget = 80)
  (h2 : chicken_cost = 12)
  (h3 : beef_cost_per_pound = 3)
  (h4 : remaining_budget = 53) :
  (initial_budget - remaining_budget - chicken_cost) / beef_cost_per_pound = 5 := by
  sorry

end NUMINAMATH_CALUDE_beef_purchase_l3741_374169


namespace NUMINAMATH_CALUDE_sunglasses_cost_theorem_l3741_374176

/-- The cost of sunglasses for a vendor -/
theorem sunglasses_cost_theorem 
  (selling_price : ℝ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℝ) 
  (h1 : selling_price = 30)
  (h2 : pairs_sold = 10)
  (h3 : sign_cost = 20)
  (h4 : (pairs_sold : ℝ) * (selling_price - cost_per_pair) / 2 = sign_cost) :
  cost_per_pair = 26 := by
  sorry

#check sunglasses_cost_theorem

end NUMINAMATH_CALUDE_sunglasses_cost_theorem_l3741_374176


namespace NUMINAMATH_CALUDE_parabolic_trajectory_falls_within_interval_l3741_374179

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabolic trajectory -/
structure Trajectory where
  a : ℝ
  c : ℝ

/-- Check if a trajectory passes through a point -/
def passesThrough (t : Trajectory) (p : Point) : Prop :=
  p.y = t.a * p.x^2 + t.c

/-- Check if a trajectory intersects with a given x-coordinate at or below a certain y-coordinate -/
def intersectsAt (t : Trajectory) (x y : ℝ) : Prop :=
  t.a * x^2 + t.c ≤ y

theorem parabolic_trajectory_falls_within_interval 
  (t : Trajectory) 
  (A : Point) 
  (P : Point) 
  (D : Point) :
  t.a < 0 →
  A.x = 0 ∧ A.y = 9 →
  P.x = 2 ∧ P.y = 8.1 →
  D.x = 6 ∧ D.y = 7 →
  passesThrough t A →
  passesThrough t P →
  intersectsAt t D.x D.y :=
by sorry

end NUMINAMATH_CALUDE_parabolic_trajectory_falls_within_interval_l3741_374179


namespace NUMINAMATH_CALUDE_parametric_to_slope_intercept_l3741_374195

/-- A line parameterized by (x, y) = (3t + 6, 5t - 7) where t is a real number -/
def parametric_line (t : ℝ) : ℝ × ℝ := (3 * t + 6, 5 * t - 7)

/-- The slope-intercept form of a line -/
def slope_intercept_form (m b : ℝ) (x : ℝ) : ℝ := m * x + b

theorem parametric_to_slope_intercept :
  ∀ (x y : ℝ), (∃ t : ℝ, parametric_line t = (x, y)) →
  y = slope_intercept_form (5/3) (-17) x :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_slope_intercept_l3741_374195


namespace NUMINAMATH_CALUDE_program_duration_l3741_374194

/-- Proves that the duration of each program is 30 minutes -/
theorem program_duration (num_programs : ℕ) (commercial_fraction : ℚ) (total_commercial_time : ℕ) :
  num_programs = 6 →
  commercial_fraction = 1/4 →
  total_commercial_time = 45 →
  ∃ (program_duration : ℕ),
    program_duration = 30 ∧
    (↑num_programs * commercial_fraction * ↑program_duration = ↑total_commercial_time) :=
by
  sorry

end NUMINAMATH_CALUDE_program_duration_l3741_374194


namespace NUMINAMATH_CALUDE_common_tangent_sum_l3741_374147

-- Define the parabolas
def P₁ (x y : ℚ) : Prop := y = x^2 + 51/50
def P₂ (x y : ℚ) : Prop := x = y^2 + 19/2

-- Define the tangent line
def TangentLine (a b c : ℕ) (x y : ℚ) : Prop := a * x + b * y = c

-- Define the property of being a common tangent to both parabolas
def CommonTangent (a b c : ℕ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℚ), 
    P₁ x₁ y₁ ∧ P₂ x₂ y₂ ∧ 
    TangentLine a b c x₁ y₁ ∧ 
    TangentLine a b c x₂ y₂

-- The main theorem
theorem common_tangent_sum :
  ∀ (a b c : ℕ), 
    a > 0 → b > 0 → c > 0 →
    Nat.gcd a (Nat.gcd b c) = 1 →
    CommonTangent a b c →
    (a : ℤ) + b + c = 37 := by sorry

end NUMINAMATH_CALUDE_common_tangent_sum_l3741_374147


namespace NUMINAMATH_CALUDE_hexagon_side_length_l3741_374132

/-- A hexagon is a polygon with 6 sides -/
def Hexagon := { n : ℕ // n = 6 }

/-- The perimeter of a regular polygon is the sum of the lengths of all its sides -/
def perimeter (sides : ℕ) (side_length : ℝ) : ℝ := sides * side_length

theorem hexagon_side_length (h : Hexagon) (p : ℝ) (h_perimeter : p = 12) :
  ∃ (s : ℝ), perimeter h.val s = p ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_side_length_l3741_374132


namespace NUMINAMATH_CALUDE_cube_difference_l3741_374154

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 40) : 
  a^3 - b^3 = 248.5 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l3741_374154


namespace NUMINAMATH_CALUDE_white_balls_count_l3741_374127

theorem white_balls_count (a b c : ℕ) : 
  a + b + c = 20 → -- Total number of balls
  (a : ℚ) / (20 + b) = a / 20 - 1 / 25 → -- Probability change when doubling blue balls
  b / (20 - a) = b / 20 + 1 / 16 → -- Probability change when removing white balls
  a = 4 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l3741_374127


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3741_374177

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3741_374177


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_and_five_l3741_374134

theorem smallest_four_digit_divisible_by_four_and_five : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 4 = 0 ∧ n % 5 = 0 → n ≥ 1020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_four_and_five_l3741_374134


namespace NUMINAMATH_CALUDE_gcd_1989_1547_l3741_374156

theorem gcd_1989_1547 : Nat.gcd 1989 1547 = 221 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1989_1547_l3741_374156


namespace NUMINAMATH_CALUDE_train_distance_problem_l3741_374136

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 70) :
  let t := d / (v2 - v1)
  let x := v1 * t
  (x + (x + d)) = 630 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l3741_374136


namespace NUMINAMATH_CALUDE_roses_picked_later_l3741_374114

/-- Calculates the number of roses picked later by a florist -/
theorem roses_picked_later (initial : ℕ) (sold : ℕ) (final : ℕ) : 
  initial ≥ sold → final > initial - sold → final - (initial - sold) = 21 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_later_l3741_374114


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l3741_374120

theorem negation_of_existence (p : ℕ → Prop) :
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_greater_than_100 :
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_100_l3741_374120


namespace NUMINAMATH_CALUDE_terminal_side_in_quadrant_II_l3741_374123

def α : Real := 2

theorem terminal_side_in_quadrant_II :
  π / 2 < α ∧ α < π :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_quadrant_II_l3741_374123


namespace NUMINAMATH_CALUDE_undefined_expression_l3741_374171

theorem undefined_expression (y : ℝ) : 
  y^2 - 16*y + 64 = 0 → y = 8 :=
by sorry

end NUMINAMATH_CALUDE_undefined_expression_l3741_374171


namespace NUMINAMATH_CALUDE_quadratic_root_implies_p_value_l3741_374161

theorem quadratic_root_implies_p_value (q p : ℝ) (h : Complex.I * Complex.I = -1) :
  (3 : ℂ) * (4 + Complex.I)^2 - q * (4 + Complex.I) + p = 0 → p = 51 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_p_value_l3741_374161


namespace NUMINAMATH_CALUDE_inequality_proof_l3741_374185

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3741_374185


namespace NUMINAMATH_CALUDE_parabola_directrix_l3741_374128

/-- The equation of the directrix of the parabola x² = 4y is y = -1 -/
theorem parabola_directrix (x y : ℝ) : 
  (∀ x y, x^2 = 4*y → ∃ k, y = -k ∧ k = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3741_374128


namespace NUMINAMATH_CALUDE_peach_boxes_count_l3741_374115

def peaches_per_basket : ℕ := 23
def num_baskets : ℕ := 7
def peaches_eaten : ℕ := 7
def peaches_per_box : ℕ := 13

theorem peach_boxes_count :
  let total_peaches := peaches_per_basket * num_baskets
  let remaining_peaches := total_peaches - peaches_eaten
  (remaining_peaches / peaches_per_box : ℕ) = 11 := by
  sorry

end NUMINAMATH_CALUDE_peach_boxes_count_l3741_374115


namespace NUMINAMATH_CALUDE_assignment_plans_eq_48_l3741_374125

/-- Represents the number of umpires from each country -/
def umpires_per_country : ℕ := 2

/-- Represents the number of countries -/
def num_countries : ℕ := 3

/-- Represents the number of venues -/
def num_venues : ℕ := 3

/-- Calculates the number of ways to assign umpires to venues -/
def assignment_plans : ℕ := sorry

/-- Theorem stating that the number of assignment plans is 48 -/
theorem assignment_plans_eq_48 : assignment_plans = 48 := by sorry

end NUMINAMATH_CALUDE_assignment_plans_eq_48_l3741_374125


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3741_374188

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The property that the units digit of any power of 5 is 5 -/
axiom units_digit_power_of_five (k : ℕ) : units_digit (5^k) = 5

/-- The main theorem: The units digit of 5^11 * 2^3 is 0 -/
theorem units_digit_of_product : units_digit (5^11 * 2^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3741_374188


namespace NUMINAMATH_CALUDE_trig_identity_l3741_374100

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3741_374100


namespace NUMINAMATH_CALUDE_answer_determines_sanity_not_species_l3741_374118

-- Define the species of the interlocutor
inductive Species
| Human
| Ghoul

-- Define the mental state of the interlocutor
inductive MentalState
| Sane
| Insane

-- Define the possible answers
inductive Answer
| Yes
| No

-- Define a function that determines the answer based on species and mental state
def getAnswer (s : Species) (m : MentalState) : Answer :=
  match m with
  | MentalState.Sane => Answer.Yes
  | MentalState.Insane => Answer.No

-- Theorem stating that the answer determines sanity but not species
theorem answer_determines_sanity_not_species :
  ∀ (s1 s2 : Species) (m1 m2 : MentalState),
    getAnswer s1 m1 = getAnswer s2 m2 →
    m1 = m2 ∧ (s1 = s2 ∨ s1 ≠ s2) :=
by sorry

end NUMINAMATH_CALUDE_answer_determines_sanity_not_species_l3741_374118


namespace NUMINAMATH_CALUDE_max_candy_leftover_l3741_374143

theorem max_candy_leftover (x : ℕ) : ∃ (q : ℕ), x = 11 * q + 10 ∧ ∀ (r : ℕ), r < 11 → x ≠ 11 * q + r + 1 :=
sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l3741_374143


namespace NUMINAMATH_CALUDE_range_of_m_l3741_374119

def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m < -2 ∨ m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3741_374119


namespace NUMINAMATH_CALUDE_food_combo_discount_percentage_l3741_374164

/-- Calculates the discount percentage on food combos during a special offer. -/
theorem food_combo_discount_percentage
  (evening_ticket_cost : ℚ)
  (food_combo_cost : ℚ)
  (ticket_discount_percent : ℚ)
  (total_savings : ℚ)
  (h1 : evening_ticket_cost = 10)
  (h2 : food_combo_cost = 10)
  (h3 : ticket_discount_percent = 20)
  (h4 : total_savings = 7) :
  (total_savings - ticket_discount_percent / 100 * evening_ticket_cost) / food_combo_cost * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_food_combo_discount_percentage_l3741_374164


namespace NUMINAMATH_CALUDE_almost_perfect_numbers_l3741_374172

def d (n : ℕ) : ℕ := (Nat.divisors n).card

def f (n : ℕ) : ℕ := (Nat.divisors n).sum d

def is_almost_perfect (n : ℕ) : Prop := n > 1 ∧ f n = n

theorem almost_perfect_numbers :
  ∀ n : ℕ, is_almost_perfect n ↔ n = 3 ∨ n = 18 ∨ n = 36 := by sorry

end NUMINAMATH_CALUDE_almost_perfect_numbers_l3741_374172


namespace NUMINAMATH_CALUDE_apple_box_weight_l3741_374112

/-- Given a box of apples with total weight and weight after removing half the apples,
    prove the weight of the box and the weight of the apples. -/
theorem apple_box_weight (total_weight : ℝ) (half_removed_weight : ℝ)
    (h1 : total_weight = 62.8)
    (h2 : half_removed_weight = 31.8) :
    ∃ (box_weight apple_weight : ℝ),
      box_weight = 0.8 ∧
      apple_weight = 62 ∧
      total_weight = box_weight + apple_weight ∧
      half_removed_weight = box_weight + apple_weight / 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_box_weight_l3741_374112


namespace NUMINAMATH_CALUDE_smallest_n_square_cube_l3741_374197

theorem smallest_n_square_cube : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (k : ℕ), 4 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^3) ∧ 
  (∀ (x : ℕ), x > 0 ∧ x < n → ¬(∃ (y : ℕ), 4 * x = y^2) ∨ ¬(∃ (z : ℕ), 5 * x = z^3)) ∧
  n = 100 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_cube_l3741_374197


namespace NUMINAMATH_CALUDE_opposite_roots_quadratic_l3741_374198

theorem opposite_roots_quadratic (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + (k^2 - 4)*x₁ + k + 1 = 0 ∧ 
    x₂^2 + (k^2 - 4)*x₂ + k + 1 = 0 ∧
    x₁ = -x₂) → 
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_opposite_roots_quadratic_l3741_374198


namespace NUMINAMATH_CALUDE_melody_civics_pages_l3741_374151

/-- The number of pages Melody needs to read for her English class -/
def english_pages : ℕ := 20

/-- The number of pages Melody needs to read for her Science class -/
def science_pages : ℕ := 16

/-- The number of pages Melody needs to read for her Chinese class -/
def chinese_pages : ℕ := 12

/-- The fraction of pages Melody will read tomorrow for each class -/
def read_fraction : ℚ := 1/4

/-- The total number of pages Melody will read tomorrow -/
def total_pages_tomorrow : ℕ := 14

/-- The number of pages Melody needs to read for her Civics class -/
def civics_pages : ℕ := 8

theorem melody_civics_pages :
  (english_pages : ℚ) * read_fraction +
  (science_pages : ℚ) * read_fraction +
  (chinese_pages : ℚ) * read_fraction +
  (civics_pages : ℚ) * read_fraction = total_pages_tomorrow :=
sorry

end NUMINAMATH_CALUDE_melody_civics_pages_l3741_374151


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3741_374166

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def geometric_sequence_terms (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : arithmetic_sequence a d) 
  (h3 : geometric_sequence_terms (a 1) (a 3) (a 9)) : 
  3 * (a 3) / (a 16) = 9 / 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3741_374166


namespace NUMINAMATH_CALUDE_beach_house_rent_l3741_374116

/-- The total amount paid for rent by a group of people -/
def total_rent (num_people : ℕ) (rent_per_person : ℚ) : ℚ :=
  num_people * rent_per_person

/-- Proof that 7 people paying $70.00 each results in a total of $490.00 -/
theorem beach_house_rent :
  total_rent 7 70 = 490 := by
  sorry

end NUMINAMATH_CALUDE_beach_house_rent_l3741_374116


namespace NUMINAMATH_CALUDE_max_value_fraction_l3741_374121

theorem max_value_fraction (x : ℝ) : 
  (3 * x^2 + 9 * x + 20) / (3 * x^2 + 9 * x + 7) ≤ 53 ∧ 
  ∀ ε > 0, ∃ y : ℝ, (3 * y^2 + 9 * y + 20) / (3 * y^2 + 9 * y + 7) > 53 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3741_374121


namespace NUMINAMATH_CALUDE_interval_length_implies_k_l3741_374196

theorem interval_length_implies_k (k : ℝ) : 
  k > 0 → 
  (Set.Icc (-3) 3 : Set ℝ) = {x : ℝ | x^2 + k * |x| ≤ 2019} → 
  k = 670 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_k_l3741_374196


namespace NUMINAMATH_CALUDE_root_in_interval_l3741_374193

-- Define the function f(x) = x^3 + x - 1
def f (x : ℝ) : ℝ := x^3 + x - 1

-- State the theorem
theorem root_in_interval :
  f 0.6 < 0 → f 0.7 > 0 → ∃ x ∈ Set.Ioo 0.6 0.7, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3741_374193


namespace NUMINAMATH_CALUDE_second_car_speed_l3741_374124

/-- Given two cars traveling in the same direction for 3 hours, with one car
    traveling at 50 mph and ending up 60 miles ahead of the other car,
    prove that the speed of the second car is 30 mph. -/
theorem second_car_speed (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (distance_diff : ℝ)
    (h1 : speed1 = 50)
    (h2 : time = 3)
    (h3 : distance_diff = 60)
    (h4 : speed1 * time - speed2 * time = distance_diff) :
    speed2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_car_speed_l3741_374124


namespace NUMINAMATH_CALUDE_fruit_purchase_total_l3741_374157

/-- Calculate the total amount paid for fruits given their quantities and rates --/
theorem fruit_purchase_total (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) :
  grape_quantity = 9 →
  grape_rate = 70 →
  mango_quantity = 9 →
  mango_rate = 55 →
  grape_quantity * grape_rate + mango_quantity * mango_rate = 1125 := by
sorry

end NUMINAMATH_CALUDE_fruit_purchase_total_l3741_374157


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l3741_374182

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the open interval (0, 1)
def open_interval_zero_one : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = open_interval_zero_one := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l3741_374182


namespace NUMINAMATH_CALUDE_circle_ratio_theorem_l3741_374104

theorem circle_ratio_theorem (r₁ r₂ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) 
  (h : π * r₂^2 - π * r₁^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_theorem_l3741_374104


namespace NUMINAMATH_CALUDE_amount_of_b_l3741_374129

theorem amount_of_b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 100) (h4 : (3/10) * a = (1/5) * b) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_amount_of_b_l3741_374129


namespace NUMINAMATH_CALUDE_initial_kittens_l3741_374109

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem: Tim's initial number of kittens was 18 -/
theorem initial_kittens : 
  kittens_to_jessica + kittens_to_sara + kittens_left = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_kittens_l3741_374109


namespace NUMINAMATH_CALUDE_least_common_period_l3741_374184

-- Define the property of the function
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

-- Define what it means for a function to be periodic with period p
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- The main theorem
theorem least_common_period :
  ∃ p : ℝ, p > 0 ∧
  (∀ f : ℝ → ℝ, satisfies_condition f → is_periodic f p) ∧
  (∀ q : ℝ, 0 < q ∧ q < p →
    ∃ g : ℝ → ℝ, satisfies_condition g ∧ ¬is_periodic g q) ∧
  p = 30 :=
sorry

end NUMINAMATH_CALUDE_least_common_period_l3741_374184


namespace NUMINAMATH_CALUDE_water_transfer_equilibrium_l3741_374142

theorem water_transfer_equilibrium (total : ℕ) (a b : ℕ) : 
  total = 48 →
  a = 30 →
  b = 18 →
  a + b = total →
  let a' := a - 2 * a
  let b' := b + 2 * a
  let a'' := a' + 2 * a'
  let b'' := b' - 2 * a'
  a'' = b'' := by sorry

end NUMINAMATH_CALUDE_water_transfer_equilibrium_l3741_374142


namespace NUMINAMATH_CALUDE_larger_number_proof_l3741_374190

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 23) →
  (∃ (k : ℕ+), Nat.lcm a b = 23 * 15 * 16 * k) →
  (max a b = 368) :=
by sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3741_374190


namespace NUMINAMATH_CALUDE_energy_increase_with_center_charge_l3741_374155

/-- Represents the energy stored between two charges -/
structure EnergyBetweenCharges where
  charge1 : ℝ
  charge2 : ℝ
  distance : ℝ
  energy : ℝ
  proportionality : energy = (charge1 * charge2) / distance

/-- Configuration of charges on a square -/
structure SquareChargeConfiguration where
  sideLength : ℝ
  chargeValue : ℝ
  totalEnergy : ℝ

/-- Configuration with one charge moved to the center -/
structure CenterChargeConfiguration where
  sideLength : ℝ
  chargeValue : ℝ
  totalEnergy : ℝ

theorem energy_increase_with_center_charge 
  (initial : SquareChargeConfiguration)
  (final : CenterChargeConfiguration)
  (h1 : initial.totalEnergy = 20)
  (h2 : initial.sideLength = final.sideLength)
  (h3 : initial.chargeValue = final.chargeValue)
  : final.totalEnergy - initial.totalEnergy = 40 := by
  sorry

end NUMINAMATH_CALUDE_energy_increase_with_center_charge_l3741_374155


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l3741_374106

theorem quadratic_equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 5*x₁ + 6 = 0 ∧ x₂^2 - 5*x₂ + 6 = 0 ∧ x₁ = 2 ∧ x₂ = 3) ∧
  (∃ x₁ x₂ : ℝ, 2*x₁^2 - 4*x₁ - 1 = 0 ∧ 2*x₂^2 - 4*x₂ - 1 = 0 ∧ 
    x₁ = (4 + Real.sqrt 24) / 4 ∧ x₂ = (4 - Real.sqrt 24) / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l3741_374106


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3741_374102

theorem fraction_sum_equality (a b c : ℝ) (n : ℕ) 
  (h1 : 1/a + 1/b + 1/c = 1/(a + b + c))
  (h2 : Odd n) (h3 : n > 0) :
  1/a^n + 1/b^n + 1/c^n = 1/(a^n + b^n + c^n) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3741_374102


namespace NUMINAMATH_CALUDE_continued_fraction_evaluation_l3741_374145

theorem continued_fraction_evaluation :
  let x : ℚ := 1 + (3 / (4 + (5 / (6 + (7/8)))))
  x = 85/52 := by
sorry

end NUMINAMATH_CALUDE_continued_fraction_evaluation_l3741_374145


namespace NUMINAMATH_CALUDE_point_not_above_curve_l3741_374163

theorem point_not_above_curve :
  ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 →
  ¬(b > a * b^3 - b * b^2) := by
  sorry

end NUMINAMATH_CALUDE_point_not_above_curve_l3741_374163


namespace NUMINAMATH_CALUDE_linear_function_properties_l3741_374180

def f (x : ℝ) : ℝ := -2 * x + 2

theorem linear_function_properties :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Second quadrant
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Fourth quadrant
  (f 1 = 0) ∧                               -- x-intercept
  (∀ x > 0, f x < 2) ∧                      -- y < 2 when x > 0
  (∀ x1 x2, x1 < x2 → f x1 > f x2)          -- y decreases as x increases
  := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l3741_374180


namespace NUMINAMATH_CALUDE_cos_sum_24_144_264_l3741_374148

theorem cos_sum_24_144_264 :
  Real.cos (24 * π / 180) + Real.cos (144 * π / 180) + Real.cos (264 * π / 180) =
    (3 - Real.sqrt 5) / 4 - Real.sin (3 * π / 180) * Real.sqrt (10 + 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_24_144_264_l3741_374148


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3741_374117

theorem arithmetic_expression_equality : 4 * 6 + 8 * 3 - 28 / 2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3741_374117


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3741_374160

/-- Given a sum P at simple interest rate R for 5 years, if increasing the rate by 5% 
    results in Rs. 250 more interest, then P = 1000 -/
theorem interest_rate_problem (P R : ℝ) (h : P > 0) (k : R > 0) :
  (P * (R + 5) * 5) / 100 - (P * R * 5) / 100 = 250 → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3741_374160


namespace NUMINAMATH_CALUDE_bread_cost_l3741_374130

/-- The cost of each loaf of bread given the total number of loaves, 
    cost of peanut butter, initial amount of money, and amount left over. -/
theorem bread_cost (num_loaves : ℕ) (peanut_butter_cost initial_money leftover : ℚ) :
  num_loaves = 3 ∧ 
  peanut_butter_cost = 2 ∧ 
  initial_money = 14 ∧ 
  leftover = 5.25 →
  (initial_money - leftover - peanut_butter_cost) / num_loaves = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_bread_cost_l3741_374130


namespace NUMINAMATH_CALUDE_inductive_reasoning_is_specific_to_general_l3741_374191

-- Define the types of reasoning
inductive ReasoningType
  | Analogical
  | Deductive
  | Inductive
  | Emotional

-- Define the direction of reasoning
inductive ReasoningDirection
  | SpecificToGeneral
  | GeneralToSpecific
  | Other

-- Function to get the direction of a reasoning type
def getReasoningDirection (rt : ReasoningType) : ReasoningDirection :=
  match rt with
  | ReasoningType.Inductive => ReasoningDirection.SpecificToGeneral
  | _ => ReasoningDirection.Other

-- Theorem statement
theorem inductive_reasoning_is_specific_to_general :
  ∃ (rt : ReasoningType), getReasoningDirection rt = ReasoningDirection.SpecificToGeneral ∧
  rt = ReasoningType.Inductive :=
sorry

end NUMINAMATH_CALUDE_inductive_reasoning_is_specific_to_general_l3741_374191


namespace NUMINAMATH_CALUDE_brians_purchased_animals_ratio_l3741_374178

theorem brians_purchased_animals_ratio (initial_horses : ℕ) (initial_sheep : ℕ) (initial_chickens : ℕ) (gifted_goats : ℕ) (male_animals : ℕ) : 
  initial_horses = 100 →
  initial_sheep = 29 →
  initial_chickens = 9 →
  gifted_goats = 37 →
  male_animals = 53 →
  (initial_horses + initial_sheep + initial_chickens - (2 * male_animals - gifted_goats)) * 2 = initial_horses + initial_sheep + initial_chickens :=
by sorry

end NUMINAMATH_CALUDE_brians_purchased_animals_ratio_l3741_374178


namespace NUMINAMATH_CALUDE_divisors_of_x_15_minus_1_l3741_374146

theorem divisors_of_x_15_minus_1 :
  ∀ k : ℕ, k ≤ 14 →
    ∃ p : Polynomial ℤ, (Polynomial.degree p = k) ∧ (p ∣ (X ^ 15 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_x_15_minus_1_l3741_374146


namespace NUMINAMATH_CALUDE_hyperbola_tangent_property_l3741_374199

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the point P
def P : ℝ × ℝ := (2, 3)

-- Define the line AB
def line_AB (x y : ℝ) : Prop := 2*x - 3*y = 2

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define points A and B on the hyperbola
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define the angle between two vectors
def angle (v₁ v₂ : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem hyperbola_tangent_property :
  ∀ (A B : ℝ × ℝ),
  hyperbola A.1 A.2 →
  hyperbola B.1 B.2 →
  A.1 < B.1 →
  line_AB A.1 A.2 →
  line_AB B.1 B.2 →
  line_AB P.1 P.2 →
  (∀ (x y : ℝ), hyperbola x y → line_AB x y → (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2)) →
  (line_AB A.1 A.2 ∧ line_AB B.1 B.2) ∧
  angle (A.1 - F₁.1, A.2 - F₁.2) (P.1 - F₁.1, P.2 - F₁.2) =
  angle (B.1 - F₂.1, B.2 - F₂.2) (P.1 - F₂.1, P.2 - F₂.2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_tangent_property_l3741_374199


namespace NUMINAMATH_CALUDE_rectangles_count_l3741_374107

/-- The number of rectangles in a strip of height 1 and width n --/
def rectanglesInStrip (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- The total number of rectangles in the given grid --/
def totalRectangles : ℕ :=
  rectanglesInStrip 5 + rectanglesInStrip 4 - 1

theorem rectangles_count : totalRectangles = 24 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_count_l3741_374107


namespace NUMINAMATH_CALUDE_not_property_P_if_cong_4_mod_9_l3741_374183

/-- Property P: An integer n has property P if there exist integers x, y, z 
    such that n = x³ + y³ + z³ - 3xyz -/
def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3*x*y*z

/-- Theorem: If an integer n is congruent to 4 modulo 9, then it does not have property P -/
theorem not_property_P_if_cong_4_mod_9 (n : ℤ) (h : n % 9 = 4) : 
  ¬(has_property_P n) := by
  sorry

#check not_property_P_if_cong_4_mod_9

end NUMINAMATH_CALUDE_not_property_P_if_cong_4_mod_9_l3741_374183


namespace NUMINAMATH_CALUDE_pizza_problem_l3741_374187

theorem pizza_problem (total_money : ℕ) (pizza_cost : ℕ) (bill_initial : ℕ) (bill_final : ℕ) :
  total_money = 42 →
  pizza_cost = 11 →
  bill_initial = 30 →
  bill_final = 39 →
  (total_money - (bill_final - bill_initial)) / pizza_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_problem_l3741_374187


namespace NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l3741_374170

/-- Proves that the point (-2, 2) lies on the graph of the inverse proportion function y = -4/x -/
theorem point_on_inverse_proportion_graph :
  let f : ℝ → ℝ := λ x => -4 / x
  f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l3741_374170


namespace NUMINAMATH_CALUDE_emily_candy_from_neighbors_l3741_374158

/-- The number of candy pieces Emily received from her sister -/
def candy_from_sister : ℕ := 13

/-- The number of candy pieces Emily ate per day -/
def candy_eaten_per_day : ℕ := 9

/-- The number of days Emily's candy lasted -/
def days_candy_lasted : ℕ := 2

/-- The number of candy pieces Emily received from neighbors -/
def candy_from_neighbors : ℕ := (candy_eaten_per_day * days_candy_lasted) - candy_from_sister

theorem emily_candy_from_neighbors : candy_from_neighbors = 5 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_from_neighbors_l3741_374158


namespace NUMINAMATH_CALUDE_divisibility_by_a_minus_one_squared_l3741_374103

theorem divisibility_by_a_minus_one_squared (a : ℤ) (n : ℕ) (hn : n > 0) :
  ∃ k : ℤ, a^(n+1) - n*(a-1) - a = k * (a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_a_minus_one_squared_l3741_374103


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3741_374189

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  (X^4 : Polynomial ℝ) + 3 * X^2 - 2 = (X^2 - 4 * X + 3) * q + r ∧ 
  r = 88 * X - 59 ∧ 
  r.degree < (X^2 - 4 * X + 3).degree := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3741_374189


namespace NUMINAMATH_CALUDE_cube_cutting_l3741_374135

theorem cube_cutting (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 = 98 + b^3) : b = 3 :=
sorry

end NUMINAMATH_CALUDE_cube_cutting_l3741_374135


namespace NUMINAMATH_CALUDE_large_number_with_specific_divisors_l3741_374162

/-- A function that returns the list of divisors of a natural number -/
def divisors (n : ℕ) : List ℕ := sorry

/-- A predicate that checks if a list of natural numbers has alternating parity -/
def hasAlternatingParity (l : List ℕ) : Prop := sorry

theorem large_number_with_specific_divisors (n : ℕ) 
  (h1 : (divisors n).length = 1000)
  (h2 : hasAlternatingParity (divisors n)) :
  n > 10^150 := by sorry

end NUMINAMATH_CALUDE_large_number_with_specific_divisors_l3741_374162


namespace NUMINAMATH_CALUDE_two_circles_exist_l3741_374139

/-- The parabola y^2 = 4x with focus F(1,0) and directrix x = -1 -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola -/
def Directrix : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- Point M -/
def M : ℝ × ℝ := (4, 4)

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate for a circle passing through two points and tangent to a line -/
def CirclePassesThroughAndTangent (c : Circle) (p1 p2 : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  (c.center.1 - p1.1)^2 + (c.center.2 - p1.2)^2 = c.radius^2 ∧
  (c.center.1 - p2.1)^2 + (c.center.2 - p2.2)^2 = c.radius^2 ∧
  ∃ (q : ℝ × ℝ), q ∈ l ∧ (c.center.1 - q.1)^2 + (c.center.2 - q.2)^2 = c.radius^2

theorem two_circles_exist : ∃ (c1 c2 : Circle),
  CirclePassesThroughAndTangent c1 F M Directrix ∧
  CirclePassesThroughAndTangent c2 F M Directrix ∧
  c1 ≠ c2 ∧
  ∀ (c : Circle), CirclePassesThroughAndTangent c F M Directrix → c = c1 ∨ c = c2 :=
sorry

end NUMINAMATH_CALUDE_two_circles_exist_l3741_374139


namespace NUMINAMATH_CALUDE_fathers_age_l3741_374186

/-- Proves that given the conditions about the father's and Ming Ming's ages, the father's age this year is 35 -/
theorem fathers_age (ming_age ming_age_3_years_ago father_age father_age_3_years_ago : ℕ) :
  father_age_3_years_ago = 8 * ming_age_3_years_ago →
  father_age = 5 * ming_age →
  father_age = ming_age + 3 →
  father_age_3_years_ago = father_age - 3 →
  father_age = 35 := by
sorry


end NUMINAMATH_CALUDE_fathers_age_l3741_374186


namespace NUMINAMATH_CALUDE_total_cost_is_1400_l3741_374167

def cost_of_suits (off_the_rack_cost : ℕ) (tailoring_cost : ℕ) : ℕ :=
  off_the_rack_cost + (3 * off_the_rack_cost + tailoring_cost)

theorem total_cost_is_1400 :
  cost_of_suits 300 200 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_1400_l3741_374167


namespace NUMINAMATH_CALUDE_solve_halloween_decorations_l3741_374108

/-- Represents the Halloween decoration problem --/
def halloween_decorations 
  (skulls : ℕ) 
  (broomsticks : ℕ) 
  (spiderwebs : ℕ) 
  (cauldrons : ℕ) 
  (total_planned : ℕ) : Prop :=
  let pumpkins := 2 * spiderwebs
  let total_put_up := skulls + broomsticks + spiderwebs + pumpkins + cauldrons
  let left_to_put_up := total_planned - total_put_up
  left_to_put_up = 30

/-- Theorem stating the solution to the Halloween decoration problem --/
theorem solve_halloween_decorations : 
  halloween_decorations 12 4 12 1 83 :=
by
  sorry

#check solve_halloween_decorations

end NUMINAMATH_CALUDE_solve_halloween_decorations_l3741_374108


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l3741_374144

/-- The problem of Jerry's action figures --/
theorem jerrys_action_figures 
  (final_count : ℕ) 
  (added_count : ℕ) 
  (h1 : final_count = 10) 
  (h2 : added_count = 2) :
  final_count - added_count = 8 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l3741_374144


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l3741_374149

theorem sqrt_expression_equals_three_halves :
  (Real.sqrt 48 + (1/4) * Real.sqrt 12) / Real.sqrt 27 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_halves_l3741_374149


namespace NUMINAMATH_CALUDE_probability_gears_from_algebras_l3741_374175

/-- The set of letters in "ALGEBRAS" -/
def algebras : Finset Char := {'A', 'L', 'G', 'E', 'B', 'R', 'S'}

/-- The set of letters in "GEARS" -/
def gears : Finset Char := {'G', 'E', 'A', 'R', 'S'}

/-- The probability of selecting a tile with a letter from "GEARS" out of the tiles from "ALGEBRAS" -/
theorem probability_gears_from_algebras :
  (algebras.filter (λ c => c ∈ gears)).card / algebras.card = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_gears_from_algebras_l3741_374175


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3741_374140

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The value of n that maximizes S n -/
def n_max : ℕ := sorry

theorem arithmetic_sequence_max_sum :
  (S 16 > 0) → (S 17 < 0) → n_max = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l3741_374140


namespace NUMINAMATH_CALUDE_divisibility_sequence_eventually_periodic_l3741_374126

/-- A sequence of positive integers satisfying the given divisibility property -/
def DivisibilitySequence (a : ℕ → ℕ+) : Prop :=
  ∀ n m : ℕ, (a (n + 2*m)).val ∣ (a n).val + (a (n + m)).val

/-- The sequence is eventually periodic -/
def EventuallyPeriodic (a : ℕ → ℕ+) : Prop :=
  ∃ N d : ℕ, d > 0 ∧ ∀ n : ℕ, n > N → a n = a (n + d)

/-- Main theorem: A sequence satisfying the divisibility property is eventually periodic -/
theorem divisibility_sequence_eventually_periodic (a : ℕ → ℕ+) :
  DivisibilitySequence a → EventuallyPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_divisibility_sequence_eventually_periodic_l3741_374126


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3741_374138

-- Define sets A and B
def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3741_374138


namespace NUMINAMATH_CALUDE_jacob_painting_fraction_l3741_374105

/-- Jacob's painting rate in houses per minute -/
def painting_rate : ℚ := 1 / 60

/-- Time given to paint in minutes -/
def paint_time : ℚ := 15

/-- Theorem: If Jacob can paint a house in 60 minutes, then he can paint 1/4 of the house in 15 minutes -/
theorem jacob_painting_fraction :
  painting_rate * paint_time = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_jacob_painting_fraction_l3741_374105


namespace NUMINAMATH_CALUDE_weight_system_properties_l3741_374165

/-- Represents a set of weights -/
def Weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be measured -/
def MaxWeight : ℕ := 40

/-- Checks if a weight can be represented by a combination of given weights -/
def isRepresentable (n : ℕ) (weights : List ℕ) : Prop :=
  ∃ (combination : List Bool), 
    combination.length = weights.length ∧ 
    (List.zip combination weights).foldl (λ sum (b, w) => sum + if b then w else 0) 0 = n

theorem weight_system_properties :
  (∀ n : ℕ, n ≤ MaxWeight → isRepresentable n Weights) ∧
  (∀ n : ℕ, n > MaxWeight → ¬ isRepresentable n Weights) :=
sorry

end NUMINAMATH_CALUDE_weight_system_properties_l3741_374165


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3741_374111

/-- An arithmetic sequence of integers -/
def arithmeticSeq (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- An increasing sequence -/
def increasingSeq (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_problem (b : ℕ → ℤ) 
    (h_arith : arithmeticSeq b)
    (h_incr : increasingSeq b)
    (h_prod : b 4 * b 5 = 30) : 
  b 3 * b 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3741_374111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3741_374133

-- Define an arithmetic sequence and its sum
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence a →
  sum_of_arithmetic_sequence a S →
  m > 0 →
  S (m - 1) = -2 →
  S m = 0 →
  S (m + 1) = 3 →
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3741_374133


namespace NUMINAMATH_CALUDE_prob_at_least_one_heart_or_joker_correct_l3741_374174

/-- The number of cards in a standard deck plus two jokers -/
def total_cards : ℕ := 54

/-- The number of hearts and jokers combined -/
def heart_or_joker : ℕ := 15

/-- The probability of drawing at least one heart or joker in two draws with replacement -/
def prob_at_least_one_heart_or_joker : ℚ := 155 / 324

theorem prob_at_least_one_heart_or_joker_correct :
  (1 : ℚ) - (1 - heart_or_joker / total_cards) ^ 2 = prob_at_least_one_heart_or_joker :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_heart_or_joker_correct_l3741_374174


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l3741_374153

theorem max_value_sum_of_roots (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -3) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 34 ∧
    Real.sqrt (2 * a + 2) + Real.sqrt (4 * b + 8) + Real.sqrt (6 * c + 18) ≤ max ∧
    ∃ (a' b' c' : ℝ), a' + b' + c' = 3 ∧ a' ≥ -1 ∧ b' ≥ -2 ∧ c' ≥ -3 ∧
      Real.sqrt (2 * a' + 2) + Real.sqrt (4 * b' + 8) + Real.sqrt (6 * c' + 18) = max :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l3741_374153


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3741_374168

theorem triangle_angle_measure (a b : ℝ) (A B : ℝ) :
  a > 0 → b > 0 → 0 < A → A < π → 0 < B → B < π →
  Real.sqrt 3 * a = 2 * b * Real.sin A →
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3741_374168


namespace NUMINAMATH_CALUDE_hundredth_ring_squares_l3741_374192

/-- The number of unit squares in the nth ring around a 1x2 rectangle -/
def ring_squares (n : ℕ) : ℕ := 8 * n + 2

/-- The first ring contains 10 unit squares -/
axiom first_ring : ring_squares 1 = 10

theorem hundredth_ring_squares : ring_squares 100 = 802 := by sorry

end NUMINAMATH_CALUDE_hundredth_ring_squares_l3741_374192


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3741_374150

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3741_374150


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3741_374159

theorem system_of_equations_solution (x y : ℚ) 
  (eq1 : 3 * x - 2 * y = 7)
  (eq2 : 2 * x + 3 * y = 8) : 
  x = 37 / 13 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3741_374159


namespace NUMINAMATH_CALUDE_overlapping_area_area_covered_by_both_strips_l3741_374173

/-- The area covered by both strips -/
def S : ℝ := 13.5

/-- The length of the original rectangular strip -/
def total_length : ℝ := 16

/-- The length of the left strip -/
def left_length : ℝ := 9

/-- The length of the right strip -/
def right_length : ℝ := 7

/-- The area covered only by the left strip -/
def left_area : ℝ := 27

/-- The area covered only by the right strip -/
def right_area : ℝ := 18

theorem overlapping_area :
  (left_area + S) / (right_area + S) = left_length / right_length :=
by sorry

theorem area_covered_by_both_strips : S = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_overlapping_area_area_covered_by_both_strips_l3741_374173


namespace NUMINAMATH_CALUDE_area_ratio_quadrilateral_to_dodecagon_l3741_374113

/-- Regular dodecagon with vertices ABCDEFGHIJKL -/
structure RegularDodecagon where
  vertices : Fin 12 → ℝ × ℝ
  is_regular : sorry

/-- Area of a regular dodecagon -/
def area_dodecagon (d : RegularDodecagon) : ℝ := sorry

/-- Area of quadrilateral ACEG in a regular dodecagon -/
def area_quadrilateral_ACEG (d : RegularDodecagon) : ℝ := sorry

/-- Theorem: The ratio of the area of quadrilateral ACEG to the area of a regular dodecagon is 1/(3√3) -/
theorem area_ratio_quadrilateral_to_dodecagon (d : RegularDodecagon) :
  area_quadrilateral_ACEG d / area_dodecagon d = 1 / (3 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_area_ratio_quadrilateral_to_dodecagon_l3741_374113


namespace NUMINAMATH_CALUDE_song_book_cost_l3741_374152

theorem song_book_cost (total_spent : ℝ) (trumpet_cost : ℝ) (song_book_cost : ℝ) :
  total_spent = 151 →
  trumpet_cost = 145.16 →
  total_spent = trumpet_cost + song_book_cost →
  song_book_cost = 5.84 := by
sorry

end NUMINAMATH_CALUDE_song_book_cost_l3741_374152


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l3741_374137

theorem greatest_four_digit_multiple_of_17 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 17 ∣ n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l3741_374137


namespace NUMINAMATH_CALUDE_simplify_complex_square_root_l3741_374110

theorem simplify_complex_square_root : 
  Real.sqrt ((9^8 + 3^14) / (9^6 + 3^15)) = Real.sqrt (15/14) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_square_root_l3741_374110


namespace NUMINAMATH_CALUDE_mary_fruit_expenses_l3741_374101

/-- The total cost of fruits Mary bought -/
def total_cost : ℚ := 34.72

/-- The cost of berries Mary bought -/
def berries_cost : ℚ := 11.08

/-- The cost of apples Mary bought -/
def apples_cost : ℚ := 14.33

/-- The cost of peaches Mary bought -/
def peaches_cost : ℚ := 9.31

/-- Theorem stating that the total cost is the sum of individual fruit costs -/
theorem mary_fruit_expenses : 
  total_cost = berries_cost + apples_cost + peaches_cost := by
  sorry

end NUMINAMATH_CALUDE_mary_fruit_expenses_l3741_374101


namespace NUMINAMATH_CALUDE_parallelogram_sum_xy_l3741_374141

/-- A parallelogram with sides measuring 10, 4x+2, 12y-2, and 10 units consecutively has x + y = 3 -/
theorem parallelogram_sum_xy (x y : ℝ) : 
  (10 : ℝ) = 4*x + 2 ∧ (10 : ℝ) = 12*y - 2 → x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sum_xy_l3741_374141


namespace NUMINAMATH_CALUDE_percentage_problem_l3741_374181

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : 3 * (x / 100 * x) = 18) : x = 10 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3741_374181
