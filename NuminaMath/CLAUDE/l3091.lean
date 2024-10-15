import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_center_x_coordinate_l3091_309174

/-- An ellipse in the first quadrant tangent to both axes with foci at (3,4) and (3,12) -/
structure Ellipse where
  /-- The ellipse is in the first quadrant -/
  first_quadrant : True
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : True
  /-- One focus is at (3,4) -/
  focus1 : ℝ × ℝ := (3, 4)
  /-- The other focus is at (3,12) -/
  focus2 : ℝ × ℝ := (3, 12)

/-- The x-coordinate of the center of the ellipse is 3 -/
theorem ellipse_center_x_coordinate (e : Ellipse) : ∃ (y : ℝ), e.focus1.1 = 3 ∧ e.focus2.1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_center_x_coordinate_l3091_309174


namespace NUMINAMATH_CALUDE_max_elevation_is_288_l3091_309131

-- Define the elevation function
def s (t : ℝ) : ℝ := 144 * t - 18 * t^2

-- Theorem stating that the maximum elevation is 288
theorem max_elevation_is_288 : 
  ∃ t_max : ℝ, ∀ t : ℝ, s t ≤ s t_max ∧ s t_max = 288 :=
sorry

end NUMINAMATH_CALUDE_max_elevation_is_288_l3091_309131


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3091_309175

theorem min_value_of_fraction (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (∀ x y : ℝ, x < 0 → y < 0 → a / (a + 2*b) + b / (a + b) ≥ x / (x + 2*y) + y / (x + y)) →
  a / (a + 2*b) + b / (a + b) = 2 * (Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3091_309175


namespace NUMINAMATH_CALUDE_num_factors_of_2000_l3091_309194

/-- The number of positive factors of a natural number n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- 2000 expressed as a product of prime factors -/
def two_thousand_factorization : ℕ := 2^4 * 5^3

/-- Theorem stating that the number of positive factors of 2000 is 20 -/
theorem num_factors_of_2000 : num_factors two_thousand_factorization = 20 := by sorry

end NUMINAMATH_CALUDE_num_factors_of_2000_l3091_309194


namespace NUMINAMATH_CALUDE_soft_drink_bottles_sold_l3091_309163

theorem soft_drink_bottles_sold (small_bottles : ℕ) (big_bottles : ℕ) 
  (small_sold_percent : ℚ) (total_remaining : ℕ) : 
  small_bottles = 6000 →
  big_bottles = 14000 →
  small_sold_percent = 1/5 →
  total_remaining = 15580 →
  (big_bottles - (total_remaining - (small_bottles - small_bottles * small_sold_percent))) / big_bottles = 23/100 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_bottles_sold_l3091_309163


namespace NUMINAMATH_CALUDE_average_of_remaining_digits_l3091_309164

theorem average_of_remaining_digits 
  (total_count : Nat) 
  (subset_count : Nat) 
  (total_average : ℝ) 
  (subset_average : ℝ) 
  (h1 : total_count = 20) 
  (h2 : subset_count = 14) 
  (h3 : total_average = 500) 
  (h4 : subset_average = 390) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 756.67 := by
sorry

#eval (20 * 500 - 14 * 390) / 6

end NUMINAMATH_CALUDE_average_of_remaining_digits_l3091_309164


namespace NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l3091_309129

def is_sum_of_consecutive_integers (n : ℕ) (k : ℕ) : Prop :=
  ∃ a : ℕ, n = (k * (2 * a + k - 1)) / 2

theorem sum_of_150_consecutive_integers :
  is_sum_of_consecutive_integers 4692583675 150 ∧
  ¬ is_sum_of_consecutive_integers 1627386425 150 ∧
  ¬ is_sum_of_consecutive_integers 2345680925 150 ∧
  ¬ is_sum_of_consecutive_integers 3579113450 150 ∧
  ¬ is_sum_of_consecutive_integers 5815939525 150 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_150_consecutive_integers_l3091_309129


namespace NUMINAMATH_CALUDE_special_pair_sum_l3091_309142

theorem special_pair_sum (a b : ℕ+) (q r : ℕ) : 
  a^2 + b^2 = q * (a + b) + r →
  0 ≤ r →
  r < a + b →
  q^2 + r = 1977 →
  ((a = 50 ∧ b = 37) ∨ (a = 37 ∧ b = 50)) :=
sorry

end NUMINAMATH_CALUDE_special_pair_sum_l3091_309142


namespace NUMINAMATH_CALUDE_milk_replacement_l3091_309184

theorem milk_replacement (x : ℝ) : 
  x > 0 ∧ x < 30 →
  30 - x - (x - x^2/30) = 14.7 →
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_milk_replacement_l3091_309184


namespace NUMINAMATH_CALUDE_debate_club_girls_l3091_309189

theorem debate_club_girls (total_members : ℕ) (present_members : ℕ) 
  (h_total : total_members = 22)
  (h_present : present_members = 14)
  (h_attendance : ∃ (boys girls : ℕ), 
    boys + girls = total_members ∧
    boys + (girls / 3) = present_members) :
  ∃ (girls : ℕ), girls = 12 ∧ 
    ∃ (boys : ℕ), boys + girls = total_members ∧
      boys + (girls / 3) = present_members := by
sorry

end NUMINAMATH_CALUDE_debate_club_girls_l3091_309189


namespace NUMINAMATH_CALUDE_investment_rate_problem_l3091_309173

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (r : ℝ) : 
  simple_interest 900 0.045 7 = simple_interest 900 (r / 100) 7 + 31.50 →
  r = 4 := by
  sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l3091_309173


namespace NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l3091_309192

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

theorem folded_paper_perimeter_ratio :
  let original := Rectangle.mk 6 8
  let folded := Rectangle.mk original.width (original.height / 2)
  let small := Rectangle.mk (folded.width / 2) folded.height
  let large := Rectangle.mk folded.width folded.height
  perimeter small / perimeter large = 7 / 10 := by
sorry

end NUMINAMATH_CALUDE_folded_paper_perimeter_ratio_l3091_309192


namespace NUMINAMATH_CALUDE_inequality_solution_l3091_309137

theorem inequality_solution (x : ℝ) : 
  (1 / (x - 1) - 3 / (x - 2) + 3 / (x - 3) - 1 / (x - 4) < 1 / 24) ↔ 
  (x < 1 ∨ (2 < x ∧ x < 3) ∨ 4 < x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3091_309137


namespace NUMINAMATH_CALUDE_integer_ratio_difference_l3091_309170

theorem integer_ratio_difference (a b c : ℕ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_90 : a + b + c = 90)
  (ratio : 3 * a = 2 * b ∧ 5 * a = 2 * c) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |((c : ℝ) - (a : ℝ)) - 12.846| < ε :=
by sorry

end NUMINAMATH_CALUDE_integer_ratio_difference_l3091_309170


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l3091_309109

/-- Two sequences of positive integers satisfying the given conditions -/
def SequencePair : Type :=
  { pair : (ℕ → ℕ) × (ℕ → ℕ) //
    (∀ n, pair.1 n > 0 ∧ pair.2 n > 0) ∧
    pair.1 0 ≥ 2 ∧ pair.2 0 ≥ 2 ∧
    (∀ n, pair.1 (n + 1) = Nat.gcd (pair.1 n) (pair.2 n) + 1) ∧
    (∀ n, pair.2 (n + 1) = Nat.lcm (pair.1 n) (pair.2 n) - 1) }

/-- The sequence a_n is eventually periodic -/
theorem sequence_eventually_periodic (seq : SequencePair) :
  ∃ (N t : ℕ), t > 0 ∧ ∀ n ≥ N, seq.1.1 (n + t) = seq.1.1 n :=
sorry

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l3091_309109


namespace NUMINAMATH_CALUDE_smallest_divisible_by_14_15_18_l3091_309104

theorem smallest_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ 14 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n ∧ ∀ m : ℕ, m > 0 → 14 ∣ m → 15 ∣ m → 18 ∣ m → n ≤ m :=
by
  use 630
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_14_15_18_l3091_309104


namespace NUMINAMATH_CALUDE_fraction_equality_l3091_309132

theorem fraction_equality (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 8) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -89 / 181 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3091_309132


namespace NUMINAMATH_CALUDE_total_highlighters_l3091_309123

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : pink = 10) (h2 : yellow = 15) (h3 : blue = 8) : 
  pink + yellow + blue = 33 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l3091_309123


namespace NUMINAMATH_CALUDE_tourist_contact_probability_l3091_309150

/-- The probability that two groups of tourists can contact each other -/
def contact_probability (p : ℝ) : ℝ :=
  1 - (1 - p) ^ 40

/-- Theorem: Given two groups of tourists with 5 and 8 members respectively,
    and the probability p that a tourist from the first group has the phone number
    of a tourist from the second group, the probability that the two groups
    will be able to contact each other is 1 - (1-p)^40. -/
theorem tourist_contact_probability (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) : 
  contact_probability p = 1 - (1 - p) ^ 40 := by
  sorry

end NUMINAMATH_CALUDE_tourist_contact_probability_l3091_309150


namespace NUMINAMATH_CALUDE_multiples_count_l3091_309117

theorem multiples_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, 1 ≤ n ∧ n ≤ 300 ∧ 2 ∣ n ∧ 5 ∣ n ∧ ¬(3 ∣ n) ∧ ¬(11 ∣ n)) ∧
  (∀ n, 1 ≤ n ∧ n ≤ 300 ∧ 2 ∣ n ∧ 5 ∣ n ∧ ¬(3 ∣ n) ∧ ¬(11 ∣ n) → n ∈ S) ∧
  S.card = 18 :=
by sorry

end NUMINAMATH_CALUDE_multiples_count_l3091_309117


namespace NUMINAMATH_CALUDE_negative_two_power_sum_l3091_309153

theorem negative_two_power_sum : (-2)^2004 + (-2)^2005 = -2^2004 := by sorry

end NUMINAMATH_CALUDE_negative_two_power_sum_l3091_309153


namespace NUMINAMATH_CALUDE_colleen_paid_more_than_joy_l3091_309115

/-- The amount of money Colleen paid more than Joy for pencils -/
def extra_cost (joy_pencils colleen_pencils price_per_pencil : ℕ) : ℕ :=
  (colleen_pencils - joy_pencils) * price_per_pencil

/-- Proof that Colleen paid $80 more than Joy for pencils -/
theorem colleen_paid_more_than_joy :
  extra_cost 30 50 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_colleen_paid_more_than_joy_l3091_309115


namespace NUMINAMATH_CALUDE_halloween_candy_distribution_l3091_309112

theorem halloween_candy_distribution (initial_candies given_away remaining_candies : ℕ) : 
  initial_candies = 60 → given_away = 40 → remaining_candies = initial_candies - given_away → remaining_candies = 20 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_distribution_l3091_309112


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l3091_309190

-- Define the given parameters
def train_length : ℝ := 140
def train_speed_kmh : ℝ := 45
def crossing_time : ℝ := 30

-- Define the theorem
theorem bridge_length_calculation :
  let train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time
  let bridge_length : ℝ := total_distance - train_length
  bridge_length = 235 := by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l3091_309190


namespace NUMINAMATH_CALUDE_triangle_right_angle_l3091_309144

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_right_angle (t : Triangle) 
  (h : t.b - t.a * Real.cos t.B = t.a * Real.cos t.C - t.c) : 
  t.A = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_right_angle_l3091_309144


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l3091_309161

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  (a = 3 ∧ b = 4) ∨ (a = 3 ∧ c = 4) ∨ (b = 3 ∧ c = 4) →
  a^2 + b^2 = c^2 →
  c = 5 ∨ c = Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l3091_309161


namespace NUMINAMATH_CALUDE_pants_cost_l3091_309133

theorem pants_cost (total_spent shirt_cost tie_cost : ℕ) 
  (h1 : total_spent = 198)
  (h2 : shirt_cost = 43)
  (h3 : tie_cost = 15) : 
  total_spent - (shirt_cost + tie_cost) = 140 := by
  sorry

end NUMINAMATH_CALUDE_pants_cost_l3091_309133


namespace NUMINAMATH_CALUDE_three_coins_same_probability_l3091_309100

def coin_flip := Bool

def total_outcomes (n : ℕ) : ℕ := 2^n

def favorable_outcomes (n : ℕ) : ℕ := 2 * 2^(n - 3)

theorem three_coins_same_probability (n : ℕ) (h : n = 6) :
  (favorable_outcomes n : ℚ) / (total_outcomes n : ℚ) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_three_coins_same_probability_l3091_309100


namespace NUMINAMATH_CALUDE_g_of_one_equals_fifteen_l3091_309130

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the theorem
theorem g_of_one_equals_fifteen :
  (∀ x : ℝ, g (2 * x - 3) = 3 * x + 9) →
  g 1 = 15 := by sorry

end NUMINAMATH_CALUDE_g_of_one_equals_fifteen_l3091_309130


namespace NUMINAMATH_CALUDE_wall_width_proof_l3091_309128

theorem wall_width_proof (wall_height : ℝ) (painting_width : ℝ) (painting_height : ℝ) (painting_percentage : ℝ) :
  wall_height = 5 →
  painting_width = 2 →
  painting_height = 4 →
  painting_percentage = 0.16 →
  ∃ (wall_width : ℝ), 
    wall_width = 10 ∧
    painting_width * painting_height = painting_percentage * (wall_height * wall_width) :=
by
  sorry

end NUMINAMATH_CALUDE_wall_width_proof_l3091_309128


namespace NUMINAMATH_CALUDE_sam_picked_42_cans_l3091_309102

/-- The number of cans Sam picked up in total -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem: Sam picked up 42 cans in total -/
theorem sam_picked_42_cans :
  total_cans 4 3 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_sam_picked_42_cans_l3091_309102


namespace NUMINAMATH_CALUDE_existence_of_m_l3091_309139

/-- The number of factors of 2 in m! -/
def n (m : ℕ) : ℕ := sorry

/-- Theorem stating the existence of m satisfying the given conditions -/
theorem existence_of_m : ∃ m : ℕ, m > 1990^1990 ∧ m = 3^1990 + n m := by sorry

end NUMINAMATH_CALUDE_existence_of_m_l3091_309139


namespace NUMINAMATH_CALUDE_sons_present_age_l3091_309179

/-- Proves that given the conditions about a father and son's ages, the son's present age is 22 years -/
theorem sons_present_age (son_age father_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

#check sons_present_age

end NUMINAMATH_CALUDE_sons_present_age_l3091_309179


namespace NUMINAMATH_CALUDE_grass_field_width_l3091_309127

/-- Proves that given a rectangular grass field with length 75 m, surrounded by a 2.5 m wide path,
    if the cost of constructing the path is Rs. 6750 at Rs. 10 per sq m,
    then the width of the grass field is 55 m. -/
theorem grass_field_width (field_length : ℝ) (path_width : ℝ) (path_cost : ℝ) (cost_per_sqm : ℝ) :
  field_length = 75 →
  path_width = 2.5 →
  path_cost = 6750 →
  cost_per_sqm = 10 →
  ∃ (field_width : ℝ),
    field_width = 55 ∧
    path_cost = cost_per_sqm * (
      (field_length + 2 * path_width) * (field_width + 2 * path_width) -
      field_length * field_width
    ) := by
  sorry

end NUMINAMATH_CALUDE_grass_field_width_l3091_309127


namespace NUMINAMATH_CALUDE_child_b_share_after_investment_l3091_309157

def total_amount : ℝ := 4500
def ratio_sum : ℕ := 2 + 3 + 4
def child_b_ratio : ℕ := 3
def interest_rate : ℝ := 0.04
def time_period : ℝ := 1

theorem child_b_share_after_investment :
  let principal := (child_b_ratio : ℝ) / ratio_sum * total_amount
  let interest := principal * interest_rate * time_period
  principal + interest = 1560 := by sorry

end NUMINAMATH_CALUDE_child_b_share_after_investment_l3091_309157


namespace NUMINAMATH_CALUDE_not_square_sum_of_square_and_divisor_l3091_309145

theorem not_square_sum_of_square_and_divisor (A B : ℕ) (hA : A ≠ 0) (hAsq : ∃ n : ℕ, A = n^2) (hB : B ∣ A) :
  ¬ ∃ m : ℕ, A + B = m^2 := by
sorry

end NUMINAMATH_CALUDE_not_square_sum_of_square_and_divisor_l3091_309145


namespace NUMINAMATH_CALUDE_inverse_composition_equals_two_l3091_309119

-- Define the function f
def f : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 2
| 4 => 5
| 5 => 1

-- Assume f has an inverse
axiom f_has_inverse : Function.Bijective f

-- Define f⁻¹ using the inverse of f
noncomputable def f_inv : Fin 5 → Fin 5 := Function.invFun f

-- State the theorem
theorem inverse_composition_equals_two :
  f_inv (f_inv (f_inv 3)) = 2 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_two_l3091_309119


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3091_309177

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 12 = 1

/-- The foci of the hyperbola -/
def foci : Set (ℝ × ℝ) :=
  {(4, 0), (-4, 0)}

/-- Theorem: The foci of the given hyperbola are (4,0) and (-4,0) -/
theorem hyperbola_foci :
  ∀ (p : ℝ × ℝ), p ∈ foci ↔
    (∃ (c : ℝ), ∀ (x y : ℝ),
      hyperbola_equation x y →
      (x - p.1)^2 + y^2 = (x + p.1)^2 + y^2 + 4*c) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3091_309177


namespace NUMINAMATH_CALUDE_janous_inequality_l3091_309143

theorem janous_inequality (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ 16 * (x / z + z / x + 2) ∧
  ((1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) = 16 * (x / z + z / x + 2) ↔
   x = y ∧ y = z ∧ z = Real.sqrt (α / 3)) :=
by sorry

end NUMINAMATH_CALUDE_janous_inequality_l3091_309143


namespace NUMINAMATH_CALUDE_no_roots_implies_non_integer_difference_l3091_309154

theorem no_roots_implies_non_integer_difference (a b : ℝ) : 
  a ≠ b → 
  (∀ x : ℝ, (x^2 + 20*a*x + 10*b) * (x^2 + 20*b*x + 10*a) ≠ 0) → 
  ¬(∃ n : ℤ, 20*(b - a) = n) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_implies_non_integer_difference_l3091_309154


namespace NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l3091_309198

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line passing through the focus
def line_through_focus (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ∃ (t : ℝ), x₁ = t ∧ x₂ = t ∧ y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂

-- Theorem statement
theorem midpoint_distance_to_y_axis 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_parabola₁ : parabola x₁ y₁) 
  (h_parabola₂ : parabola x₂ y₂)
  (h_line : line_through_focus x₁ y₁ x₂ y₂)
  (h_sum : x₁ + x₂ = 3) :
  (x₁ + x₂) / 2 = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_midpoint_distance_to_y_axis_l3091_309198


namespace NUMINAMATH_CALUDE_product_evaluation_l3091_309141

theorem product_evaluation : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l3091_309141


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l3091_309134

/-- The length of a train given its speed and the time it takes to cross a platform. -/
theorem train_length (platform_length : ℝ) (crossing_time : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (5 / 18)
  let total_distance := train_speed_mps * crossing_time
  total_distance - platform_length

/-- Proof that the train length is approximately 110 meters. -/
theorem train_length_proof :
  ∃ ε > 0, abs (train_length 165 7.499400047996161 132 - 110) < ε :=
by
  sorry


end NUMINAMATH_CALUDE_train_length_train_length_proof_l3091_309134


namespace NUMINAMATH_CALUDE_expression_equality_l3091_309188

theorem expression_equality : 
  Real.sqrt 4 + |Real.sqrt 3 - 3| + 2 * Real.sin (π / 6) - (π - 2023)^0 = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3091_309188


namespace NUMINAMATH_CALUDE_pencils_purchased_l3091_309113

theorem pencils_purchased (num_pens : ℕ) (total_cost : ℝ) (pencil_price : ℝ) (pen_price : ℝ) :
  num_pens = 30 →
  total_cost = 690 →
  pencil_price = 2 →
  pen_price = 18 →
  (total_cost - num_pens * pen_price) / pencil_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_pencils_purchased_l3091_309113


namespace NUMINAMATH_CALUDE_simplify_fraction_l3091_309146

theorem simplify_fraction : (24 : ℚ) / 32 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3091_309146


namespace NUMINAMATH_CALUDE_car_speed_first_hour_l3091_309124

/-- Given a car's speed over two hours, prove its speed in the first hour -/
theorem car_speed_first_hour (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 30 →
  average_speed = 65 →
  ∃ speed_first_hour : ℝ,
    speed_first_hour = 100 ∧
    average_speed = (speed_first_hour + speed_second_hour) / 2 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_first_hour_l3091_309124


namespace NUMINAMATH_CALUDE_largest_s_value_l3091_309151

theorem largest_s_value (r s : ℕ) : 
  r ≥ s ∧ s ≥ 3 ∧ 
  (r - 2) * s * 5 = (s - 2) * r * 4 →
  s ≤ 130 ∧ ∃ (r' : ℕ), r' ≥ 130 ∧ (r' - 2) * 130 * 5 = (130 - 2) * r' * 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_s_value_l3091_309151


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3091_309197

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = 3 + 7 * Complex.I) : 
  z.im = 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3091_309197


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3091_309116

theorem geometric_series_ratio (a r : ℝ) : 
  (∃ (S : ℕ → ℝ), (∀ n, S n = a * r^n) ∧ 
   (∑' n, S n) = 18 ∧ 
   (∑' n, S (2*n + 1)) = 8) →
  r = 4/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3091_309116


namespace NUMINAMATH_CALUDE_max_third_side_length_l3091_309156

theorem max_third_side_length (a b c : ℕ) (ha : a = 7) (hb : b = 12) : 
  (a + b > c ∧ a + c > b ∧ b + c > a) → c ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l3091_309156


namespace NUMINAMATH_CALUDE_craftsman_production_l3091_309120

/-- The number of parts manufactured by a master craftsman during a shift -/
def total_parts : ℕ := by sorry

/-- The number of parts manufactured in the first hour -/
def first_hour_parts : ℕ := 35

/-- The increase in production rate (parts per hour) -/
def rate_increase : ℕ := 15

/-- The time saved by increasing the production rate (in hours) -/
def time_saved : ℚ := 1.5

theorem craftsman_production :
  let initial_rate := first_hour_parts
  let new_rate := initial_rate + rate_increase
  let remaining_parts := total_parts - first_hour_parts
  (remaining_parts : ℚ) / initial_rate - (remaining_parts : ℚ) / new_rate = time_saved →
  total_parts = 210 := by sorry

end NUMINAMATH_CALUDE_craftsman_production_l3091_309120


namespace NUMINAMATH_CALUDE_unique_a_value_l3091_309185

/-- A quadratic function of the form y = 3x^2 + 2(a-1)x + b -/
def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * (a - 1) * x + b

/-- The derivative of the quadratic function -/
def quadratic_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * x + 2 * (a - 1)

theorem unique_a_value (a b : ℝ) :
  (∀ x < 1, quadratic_derivative a x < 0) →
  (∀ x ≥ 1, quadratic_derivative a x ≥ 0) →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l3091_309185


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l3091_309182

/-- The sum of digits of a number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A number consisting of n repetitions of a digit d in base 10 -/
def repeatDigit (d : ℕ) (n : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : 
  let a := repeatDigit 8 2000
  let b := repeatDigit 5 2000
  sumOfDigits (9 * a * b) = 18005 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l3091_309182


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3091_309168

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = 2 ∧ x₁^2 - 8*x₁ + 12 = 0 ∧ x₂^2 - 8*x₂ + 12 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 3 ∧ y₂ = -3 ∧ (y₁ - 3)^2 = 2*y₁*(y₁ - 3) ∧ (y₂ - 3)^2 = 2*y₂*(y₂ - 3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3091_309168


namespace NUMINAMATH_CALUDE_fishes_from_superior_is_44_l3091_309148

/-- The number of fishes taken from Lake Superior -/
def fishes_from_superior (total : ℕ) (ontario_erie : ℕ) (huron_michigan : ℕ) : ℕ :=
  total - ontario_erie - huron_michigan

/-- Theorem: Given the conditions from the problem, prove that the number of fishes
    taken from Lake Superior is 44 -/
theorem fishes_from_superior_is_44 :
  fishes_from_superior 97 23 30 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fishes_from_superior_is_44_l3091_309148


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3091_309162

theorem no_prime_roots_for_quadratic : ¬∃ (k : ℤ), ∃ (p q : ℕ), 
  Prime p ∧ Prime q ∧ p ≠ q ∧ p + q = 65 ∧ p * q = k := by
  sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l3091_309162


namespace NUMINAMATH_CALUDE_expected_value_of_sum_is_seven_l3091_309167

def marbles : Finset Nat := {1, 2, 3, 4, 5, 6}

def pairs : Finset (Nat × Nat) :=
  (marbles.product marbles).filter (fun (a, b) => a < b)

def sum_pair (p : Nat × Nat) : Nat := p.1 + p.2

theorem expected_value_of_sum_is_seven :
  (pairs.sum sum_pair) / pairs.card = 7 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_is_seven_l3091_309167


namespace NUMINAMATH_CALUDE_negation_equivalence_l3091_309111

theorem negation_equivalence : 
  (¬∃ x : ℝ, x^2 - 2*x + 4 > 0) ↔ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3091_309111


namespace NUMINAMATH_CALUDE_mango_production_l3091_309155

/-- Prove that the total produce of mangoes is 400 kg -/
theorem mango_production (apple_production mango_production orange_production : ℕ) 
  (h1 : apple_production = 2 * mango_production)
  (h2 : orange_production = mango_production + 200)
  (h3 : 50 * (apple_production + mango_production + orange_production) = 90000) :
  mango_production = 400 := by
  sorry

end NUMINAMATH_CALUDE_mango_production_l3091_309155


namespace NUMINAMATH_CALUDE_opposite_hands_at_343_l3091_309159

/-- Represents the number of degrees the minute hand moves per minute -/
def minute_hand_speed : ℝ := 6

/-- Represents the number of degrees the hour hand moves per minute -/
def hour_hand_speed : ℝ := 0.5

/-- Represents the number of minutes past 3:00 -/
def minutes_past_three : ℝ := 43

/-- The position of the minute hand after 5 minutes -/
def minute_hand_position (t : ℝ) : ℝ :=
  minute_hand_speed * (t + 5)

/-- The position of the hour hand 4 minutes ago -/
def hour_hand_position (t : ℝ) : ℝ :=
  90 + hour_hand_speed * (t - 4)

/-- Two angles are opposite if their absolute difference is 180 degrees -/
def are_opposite (a b : ℝ) : Prop :=
  abs (a - b) = 180

theorem opposite_hands_at_343 :
  are_opposite 
    (minute_hand_position minutes_past_three) 
    (hour_hand_position minutes_past_three) := by
  sorry

end NUMINAMATH_CALUDE_opposite_hands_at_343_l3091_309159


namespace NUMINAMATH_CALUDE_odd_sum_representation_l3091_309149

theorem odd_sum_representation (a b : ℤ) (h : Odd (a + b)) :
  ∀ n : ℤ, ∃ x y : ℤ, n = x^2 - y^2 + a*x + b*y :=
by sorry

end NUMINAMATH_CALUDE_odd_sum_representation_l3091_309149


namespace NUMINAMATH_CALUDE_variance_of_transformed_binomial_l3091_309176

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialRandomVariable (n : ℕ) (p : ℝ) where
  X : ℝ

/-- The variance of a binomial random variable -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- The variance of a linear transformation of a random variable -/
def linearTransformVariance (a : ℝ) (X : ℝ) : ℝ := a^2 * X

theorem variance_of_transformed_binomial :
  let n : ℕ := 10
  let p : ℝ := 0.8
  let X : BinomialRandomVariable n p := ⟨0⟩  -- The actual value doesn't matter for this theorem
  let var_X : ℝ := binomialVariance n p
  let var_2X_plus_1 : ℝ := linearTransformVariance 2 var_X
  var_2X_plus_1 = 6.4 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_binomial_l3091_309176


namespace NUMINAMATH_CALUDE_total_cookies_count_l3091_309178

/-- Represents a pack of cookies -/
structure CookiePack where
  name : String
  cookies : Nat

/-- Represents a person's cookie purchase -/
structure Purchase where
  packs : List (CookiePack × Nat)

def packA : CookiePack := ⟨"A", 15⟩
def packB : CookiePack := ⟨"B", 30⟩
def packC : CookiePack := ⟨"C", 45⟩
def packD : CookiePack := ⟨"D", 60⟩

def paulPurchase : Purchase := ⟨[(packB, 2), (packA, 1)]⟩
def paulaPurchase : Purchase := ⟨[(packA, 1), (packC, 1)]⟩

def countCookies (purchase : Purchase) : Nat :=
  purchase.packs.foldl (fun acc (pack, quantity) => acc + pack.cookies * quantity) 0

theorem total_cookies_count :
  countCookies paulPurchase + countCookies paulaPurchase = 135 := by
  sorry

#eval countCookies paulPurchase + countCookies paulaPurchase

end NUMINAMATH_CALUDE_total_cookies_count_l3091_309178


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l3091_309105

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_convex (q : Quadrilateral) : Prop := sorry

def are_opposite (q : Quadrilateral) (v1 v2 : ℝ × ℝ) : Prop := sorry

def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

def angle_measure (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_side_length 
  (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_opposite : are_opposite q q.A q.C)
  (h_BC : side_length q.B q.C = 4)
  (h_ADC : angle_measure q.A q.D q.C = π / 3)
  (h_BAD : angle_measure q.B q.A q.D = π / 2)
  (h_area : area q = (side_length q.A q.B * side_length q.C q.D + 
                      side_length q.B q.C * side_length q.A q.D) / 2) :
  side_length q.C q.D = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_side_length_l3091_309105


namespace NUMINAMATH_CALUDE_count_special_sequences_l3091_309180

def sequence_length : ℕ := 15

-- Define a function that counts sequences with all ones consecutive
def count_all_ones_consecutive (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2 - 1

-- Define a function that counts sequences with all zeros consecutive
def count_all_zeros_consecutive (n : ℕ) : ℕ :=
  count_all_ones_consecutive n

-- Define a function that counts sequences with both all zeros and all ones consecutive
def count_both_consecutive : ℕ := 2

-- Theorem statement
theorem count_special_sequences :
  count_all_ones_consecutive sequence_length +
  count_all_zeros_consecutive sequence_length -
  count_both_consecutive = 268 := by
  sorry

end NUMINAMATH_CALUDE_count_special_sequences_l3091_309180


namespace NUMINAMATH_CALUDE_factorization_sum_l3091_309108

theorem factorization_sum (a b c d e f g h j k : ℤ) :
  (∀ x y : ℝ, 64 * x^6 - 729 * y^6 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + j*x*y + k*y^2)) →
  a + b + c + d + e + f + g + h + j + k = 30 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l3091_309108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a13_l3091_309152

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a13
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a13_l3091_309152


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63_l3091_309107

def reverse_number (n : ℕ) : ℕ :=
  let digits := Nat.digits 10 n
  Nat.ofDigits 10 (List.reverse digits)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63 :
  ∃ (p : ℕ),
    is_four_digit p ∧
    p % 63 = 0 ∧
    (reverse_number p) % 63 = 0 ∧
    p % 11 = 0 ∧
    ∀ (q : ℕ),
      is_four_digit q ∧
      q % 63 = 0 ∧
      (reverse_number q) % 63 = 0 ∧
      q % 11 = 0 →
      q ≤ p ∧
    p = 7623 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_11_with_reverse_divisible_by_63_l3091_309107


namespace NUMINAMATH_CALUDE_last_digit_of_2_pow_2010_l3091_309193

-- Define the function that gives the last digit of 2^n
def lastDigitOf2Pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | _ => unreachable!

-- Theorem statement
theorem last_digit_of_2_pow_2010 : lastDigitOf2Pow 2010 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_2_pow_2010_l3091_309193


namespace NUMINAMATH_CALUDE_batsman_average_l3091_309160

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_average * 10 = previous_total →
  (previous_total + 90) / 11 = previous_average + 5 →
  (previous_total + 90) / 11 = 40 := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l3091_309160


namespace NUMINAMATH_CALUDE_diamond_jewel_percentage_is_35_percent_l3091_309196

/-- Represents the composition of objects in an urn -/
structure UrnComposition where
  bead_percent : ℝ
  ruby_jewel_percent : ℝ
  diamond_jewel_percent : ℝ

/-- Calculates the percentage of diamond jewels in the urn -/
def diamond_jewel_percentage (u : UrnComposition) : ℝ :=
  u.diamond_jewel_percent

/-- The theorem stating the percentage of diamond jewels in the urn -/
theorem diamond_jewel_percentage_is_35_percent (u : UrnComposition) 
  (h1 : u.bead_percent = 30)
  (h2 : u.ruby_jewel_percent = 35)
  (h3 : u.bead_percent + u.ruby_jewel_percent + u.diamond_jewel_percent = 100) :
  diamond_jewel_percentage u = 35 := by
  sorry

#check diamond_jewel_percentage_is_35_percent

end NUMINAMATH_CALUDE_diamond_jewel_percentage_is_35_percent_l3091_309196


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3091_309169

theorem sqrt_product_plus_one : 
  Real.sqrt (31 * 30 * 29 * 28 + 1) = 869 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3091_309169


namespace NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l3091_309121

def total_amount : ℝ := 137500
def raw_materials : ℝ := 80000
def machinery : ℝ := 30000

def cash : ℝ := total_amount - (raw_materials + machinery)

theorem cash_percentage_is_twenty_percent :
  (cash / total_amount) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cash_percentage_is_twenty_percent_l3091_309121


namespace NUMINAMATH_CALUDE_stone_collision_distance_l3091_309135

/-- The initial distance between two colliding stones -/
theorem stone_collision_distance (v₀ H g : ℝ) (h_v₀_pos : 0 < v₀) (h_H_pos : 0 < H) (h_g_pos : 0 < g) :
  let t := H / v₀
  let y₁ := H - (1/2) * g * t^2
  let y₂ := v₀ * t - (1/2) * g * t^2
  let x₁ := v₀ * t
  y₁ = y₂ →
  Real.sqrt (H^2 + x₁^2) = H * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_stone_collision_distance_l3091_309135


namespace NUMINAMATH_CALUDE_f_properties_l3091_309166

def f (x : ℝ) := 4 - x^2

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, f x ≥ 3*x ↔ -4 ≤ x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3091_309166


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l3091_309122

/-- Calculates the number of axles for a truck given the total number of wheels,
    number of wheels on the front axle, and number of wheels on other axles -/
def calculateAxles (totalWheels : Nat) (frontAxleWheels : Nat) (otherAxleWheels : Nat) : Nat :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck given the number of axles -/
def calculateToll (axles : Nat) : Real :=
  1.50 + 1.50 * (axles - 2)

theorem truck_toll_calculation :
  let totalWheels : Nat := 18
  let frontAxleWheels : Nat := 2
  let otherAxleWheels : Nat := 4
  let axles := calculateAxles totalWheels frontAxleWheels otherAxleWheels
  calculateToll axles = 6.00 := by
  sorry

end NUMINAMATH_CALUDE_truck_toll_calculation_l3091_309122


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l3091_309126

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = (25*a / (73*b)) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l3091_309126


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3091_309195

/-- Given a line l: y = k(x + 1/2) and a circle C: x^2 + y^2 = 1,
    prove that the line always intersects the circle for any real k. -/
theorem line_intersects_circle (k : ℝ) : 
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
  sorry

#check line_intersects_circle

end NUMINAMATH_CALUDE_line_intersects_circle_l3091_309195


namespace NUMINAMATH_CALUDE_missing_bulbs_l3091_309187

theorem missing_bulbs (total_fixtures : ℕ) (capacity_per_fixture : ℕ) 
  (fixtures_with_4 : ℕ) (fixtures_with_3 : ℕ) (fixtures_with_1 : ℕ) (fixtures_with_0 : ℕ) :
  total_fixtures = 24 →
  capacity_per_fixture = 4 →
  fixtures_with_1 = 2 * fixtures_with_4 →
  fixtures_with_0 = fixtures_with_3 / 2 →
  fixtures_with_4 + fixtures_with_3 + (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1 + fixtures_with_0 = total_fixtures →
  4 * fixtures_with_4 + 3 * fixtures_with_3 + 2 * (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1 = total_fixtures * capacity_per_fixture / 2 →
  total_fixtures * capacity_per_fixture - (4 * fixtures_with_4 + 3 * fixtures_with_3 + 2 * (total_fixtures - fixtures_with_4 - fixtures_with_3 - fixtures_with_1) + fixtures_with_1) = 48 :=
by sorry

end NUMINAMATH_CALUDE_missing_bulbs_l3091_309187


namespace NUMINAMATH_CALUDE_expansion_properties_l3091_309165

theorem expansion_properties :
  let f := fun x => (1 - 2*x)^6
  ∃ (c : ℚ) (p : Polynomial ℚ), 
    (f = fun x => p.eval x) ∧ 
    (p.coeff 2 = 60) ∧ 
    (p.eval 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_expansion_properties_l3091_309165


namespace NUMINAMATH_CALUDE_oscar_fish_count_l3091_309181

/-- Represents the initial number of Oscar fish in Danny's fish tank. -/
def initial_oscar_fish : ℕ := 58

/-- Theorem stating that the initial number of Oscar fish was 58. -/
theorem oscar_fish_count :
  let initial_guppies : ℕ := 94
  let initial_angelfish : ℕ := 76
  let initial_tiger_sharks : ℕ := 89
  let sold_guppies : ℕ := 30
  let sold_angelfish : ℕ := 48
  let sold_tiger_sharks : ℕ := 17
  let sold_oscar_fish : ℕ := 24
  let remaining_fish : ℕ := 198
  initial_oscar_fish = 
    remaining_fish - 
    ((initial_guppies - sold_guppies) + 
     (initial_angelfish - sold_angelfish) + 
     (initial_tiger_sharks - sold_tiger_sharks)) + 
    sold_oscar_fish :=
by sorry

end NUMINAMATH_CALUDE_oscar_fish_count_l3091_309181


namespace NUMINAMATH_CALUDE_sum_and_fraction_difference_l3091_309172

theorem sum_and_fraction_difference (x y : ℝ) 
  (h1 : x + y = 480) 
  (h2 : x / y = 0.8) : 
  y - x = 53.34 := by sorry

end NUMINAMATH_CALUDE_sum_and_fraction_difference_l3091_309172


namespace NUMINAMATH_CALUDE_increasing_function_inequality_l3091_309183

/-- A function f is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Main theorem: If f is increasing and f(2m) > f(-m+9), then m > 3 -/
theorem increasing_function_inequality (f : ℝ → ℝ) (m : ℝ) 
    (h_incr : IsIncreasing f) (h_ineq : f (2 * m) > f (-m + 9)) : 
    m > 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_inequality_l3091_309183


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3091_309114

theorem complex_equation_solution (a b c : ℂ) (h_real : a.im = 0) 
  (h_sum : a + b + c = 5)
  (h_prod_sum : a * b + b * c + c * a = 5)
  (h_prod : a * b * c = 5) :
  a = 4 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3091_309114


namespace NUMINAMATH_CALUDE_projection_implies_y_value_l3091_309186

/-- Given two vectors v and w in R², where v = (2, y) and w = (7, 2),
    if the projection of v onto w is (8, 16/7), then y = 163/7. -/
theorem projection_implies_y_value (y : ℝ) :
  let v : ℝ × ℝ := (2, y)
  let w : ℝ × ℝ := (7, 2)
  let proj_w_v : ℝ × ℝ := ((v.1 * w.1 + v.2 * w.2) / (w.1^2 + w.2^2)) • w
  proj_w_v = (8, 16/7) →
  y = 163/7 := by
sorry

end NUMINAMATH_CALUDE_projection_implies_y_value_l3091_309186


namespace NUMINAMATH_CALUDE_product_comparison_l3091_309147

theorem product_comparison (a b c d : ℝ) (h1 : a ≥ b) (h2 : c ≥ d) :
  (∃ (p : ℕ), p ≥ 3 ∧ (a > 0 ∨ b > 0) ∧ (a > 0 ∨ c > 0) ∧ (a > 0 ∨ d > 0) ∧
               (b > 0 ∨ c > 0) ∧ (b > 0 ∨ d > 0) ∧ (c > 0 ∨ d > 0)) →
    a * c ≥ b * d ∧
  (∃ (n : ℕ), n ≥ 3 ∧ (a < 0 ∨ b < 0) ∧ (a < 0 ∨ c < 0) ∧ (a < 0 ∨ d < 0) ∧
               (b < 0 ∨ c < 0) ∧ (b < 0 ∨ d < 0) ∧ (c < 0 ∨ d < 0)) →
    a * c ≤ b * d ∧
  (((a > 0 ∧ b > 0) ∨ (a > 0 ∧ c > 0) ∨ (a > 0 ∧ d > 0) ∨ (b > 0 ∧ c > 0) ∨
    (b > 0 ∧ d > 0) ∨ (c > 0 ∧ d > 0)) ∧
   ((a < 0 ∧ b < 0) ∨ (a < 0 ∧ c < 0) ∨ (a < 0 ∧ d < 0) ∨ (b < 0 ∧ c < 0) ∨
    (b < 0 ∧ d < 0) ∨ (c < 0 ∧ d < 0))) →
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x = y) ∧
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x < y) ∧
    ¬(∀ x y : ℝ, (x = a * c ∧ y = b * d) → x > y) :=
by sorry

end NUMINAMATH_CALUDE_product_comparison_l3091_309147


namespace NUMINAMATH_CALUDE_special_function_characterization_l3091_309138

/-- A function satisfying the given properties -/
def IsSpecialFunction (f : ℝ → ℝ) : Prop :=
  f 1 = 0 ∧ ∀ x y : ℝ, |f x - f y| = |x - y|

/-- The theorem stating that any function satisfying the given properties
    must be either x - 1 or 1 - x -/
theorem special_function_characterization (f : ℝ → ℝ) (hf : IsSpecialFunction f) :
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = 1 - x) := by
  sorry

end NUMINAMATH_CALUDE_special_function_characterization_l3091_309138


namespace NUMINAMATH_CALUDE_zeros_after_decimal_for_one_over_twelve_to_twelve_l3091_309110

-- Define the function to count zeros after decimal point
def count_zeros_after_decimal (x : ℚ) : ℕ :=
  sorry

-- Theorem statement
theorem zeros_after_decimal_for_one_over_twelve_to_twelve :
  count_zeros_after_decimal (1 / (12^12)) = 11 :=
sorry

end NUMINAMATH_CALUDE_zeros_after_decimal_for_one_over_twelve_to_twelve_l3091_309110


namespace NUMINAMATH_CALUDE_inverse_square_relation_l3091_309158

/-- Given that x varies inversely as the square of y, prove that y = 6 when x = 0.25,
    given that y = 3 when x = 1. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) (h1 : x = k / (y ^ 2)) 
    (h2 : 1 = k / (3 ^ 2)) (h3 : 0.25 = k / (y ^ 2)) : y = 6 := by
  sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l3091_309158


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3091_309118

theorem smallest_multiple_of_6_and_15 :
  ∃ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ (x : ℕ), x > 0 ∧ 6 ∣ x ∧ 15 ∣ x → b ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3091_309118


namespace NUMINAMATH_CALUDE_multiplier_problem_l3091_309125

theorem multiplier_problem (n : ℝ) (m : ℝ) : 
  n = 3 → 7 * n = m * n + 12 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_problem_l3091_309125


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l3091_309106

theorem least_sum_of_bases (a b : ℕ+) : 
  (4 * a.val + 7 = 7 * b.val + 4) →  -- 47 in base a equals 74 in base b
  (∀ (x y : ℕ+), (4 * x.val + 7 = 7 * y.val + 4) → (x.val + y.val ≥ a.val + b.val)) →
  (a.val + b.val = 24) :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l3091_309106


namespace NUMINAMATH_CALUDE_linear_coefficient_of_given_quadratic_l3091_309140

/-- The coefficient of the linear term in a quadratic equation ax^2 + bx + c = 0 is b. -/
def linearCoefficient (a b c : ℝ) : ℝ := b

/-- The quadratic equation x^2 - 2x - 1 = 0 -/
def quadraticEquation (x : ℝ) : Prop := x^2 - 2*x - 1 = 0

theorem linear_coefficient_of_given_quadratic :
  linearCoefficient 1 (-2) (-1) = -2 := by sorry

end NUMINAMATH_CALUDE_linear_coefficient_of_given_quadratic_l3091_309140


namespace NUMINAMATH_CALUDE_first_row_desks_l3091_309199

/-- Calculates the number of desks in the first row given the total number of rows,
    the increase in desks per row, and the total number of students that can be seated. -/
def desks_in_first_row (total_rows : ℕ) (increase_per_row : ℕ) (total_students : ℕ) : ℕ :=
  (2 * total_students - total_rows * (total_rows - 1) * increase_per_row) / (2 * total_rows)

/-- Theorem stating that given 8 rows of desks, where each subsequent row has 2 more desks
    than the previous row, and a total of 136 students can be seated, the number of desks
    in the first row is 10. -/
theorem first_row_desks :
  desks_in_first_row 8 2 136 = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_row_desks_l3091_309199


namespace NUMINAMATH_CALUDE_work_completion_time_l3091_309101

/-- 
Given a piece of work that can be completed by 9 laborers in 15 days, 
this theorem proves that it would take 9 days for 15 laborers to complete the same work.
-/
theorem work_completion_time 
  (total_laborers : ℕ) 
  (available_laborers : ℕ) 
  (actual_days : ℕ) 
  (h1 : total_laborers = 15)
  (h2 : available_laborers = total_laborers - 6)
  (h3 : actual_days = 15)
  : (available_laborers * actual_days) / total_laborers = 9 := by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l3091_309101


namespace NUMINAMATH_CALUDE_unique_number_l3091_309171

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2*k + 1

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13*k

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

theorem unique_number : ∃! n : ℕ, 
  is_two_digit n ∧ 
  is_odd n ∧ 
  is_multiple_of_13 n ∧ 
  is_perfect_square (digit_product n) ∧
  n = 91 := by sorry

end NUMINAMATH_CALUDE_unique_number_l3091_309171


namespace NUMINAMATH_CALUDE_transformed_roots_l3091_309191

-- Define the polynomial and its roots
def P (b : ℝ) (x : ℝ) : ℝ := x^4 - b*x^2 - 6

-- Define the roots of P
def roots (b : ℝ) : Set ℝ := {x | P b x = 0}

-- Define the transformed equation
def Q (b : ℝ) (y : ℝ) : ℝ := 6*y^2 + b*y + 1

-- Theorem statement
theorem transformed_roots (b : ℝ) (a c d : ℝ) (ha : a ∈ roots b) (hc : c ∈ roots b) (hd : d ∈ roots b) :
  Q b ((a + c) / b^3) = 0 ∧ Q b ((a + b) / c^3) = 0 ∧ Q b ((b + c) / a^3) = 0 ∧ Q b ((a + b + c) / d^3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_transformed_roots_l3091_309191


namespace NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_pqr_l3091_309103

theorem log_ten_seven_in_terms_of_pqr (p q r : ℝ) 
  (hp : Real.log 3 / Real.log 8 = p)
  (hq : Real.log 5 / Real.log 3 = q)
  (hr : Real.log 7 / Real.log 4 = r) :
  Real.log 7 / Real.log 10 = 2 * r / (1 + 4 * q * p) := by
  sorry

end NUMINAMATH_CALUDE_log_ten_seven_in_terms_of_pqr_l3091_309103


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3091_309136

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The total number of possible triangles formed by selecting 3 vertices from a decagon -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles with exactly one side coinciding with a side of the decagon -/
def triangles_one_side : ℕ := n * (n - 4)

/-- The number of triangles with two sides coinciding with sides of the decagon -/
def triangles_two_sides : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side coinciding with a side of the decagon) -/
def favorable_outcomes : ℕ := triangles_one_side + triangles_two_sides

/-- The probability of randomly selecting three vertices to form a triangle with at least one side coinciding with a side of the decagon -/
theorem decagon_triangle_probability : 
  (favorable_outcomes : ℚ) / total_triangles = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3091_309136
