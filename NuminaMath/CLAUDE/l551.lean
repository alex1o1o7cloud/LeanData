import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_m_range_l551_55110

-- Define the equation
def is_hyperbola (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / m + y^2 / (5 + m) = 1 ∧ m * (5 + m) < 0

-- State the theorem
theorem hyperbola_m_range :
  ∀ m : ℝ, is_hyperbola m ↔ -5 < m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_m_range_l551_55110


namespace NUMINAMATH_CALUDE_system_solution_l551_55104

theorem system_solution :
  ∃ (A B C D : ℚ),
    A = 1/42 ∧
    B = 1/7 ∧
    C = 1/3 ∧
    D = 1/2 ∧
    A = B * C * D ∧
    A + B = C * D ∧
    A + B + C = D ∧
    A + B + C + D = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l551_55104


namespace NUMINAMATH_CALUDE_investment_growth_equation_l551_55164

/-- Represents the average growth rate equation for a two-year investment period -/
theorem investment_growth_equation (initial_investment : ℝ) (final_investment : ℝ) (x : ℝ) :
  initial_investment = 20000 →
  final_investment = 25000 →
  20 * (1 + x)^2 = 25 :=
by sorry

end NUMINAMATH_CALUDE_investment_growth_equation_l551_55164


namespace NUMINAMATH_CALUDE_tan_identity_l551_55177

theorem tan_identity (α β γ n : Real) 
  (h : Real.sin (2 * (α + γ)) = n * Real.sin (2 * β)) : 
  Real.tan (α + β + γ) = (n + 1) / (n - 1) * Real.tan (α - β + γ) := by
  sorry

end NUMINAMATH_CALUDE_tan_identity_l551_55177


namespace NUMINAMATH_CALUDE_bridget_apples_theorem_l551_55152

/-- The number of apples Bridget originally bought -/
def original_apples : ℕ := 14

/-- The number of apples Bridget gives to Ann -/
def apples_to_ann : ℕ := original_apples / 2

/-- The number of apples Bridget gives to Cassie -/
def apples_to_cassie : ℕ := 5

/-- The number of apples Bridget keeps for herself -/
def apples_for_bridget : ℕ := 2

theorem bridget_apples_theorem :
  original_apples = apples_to_ann * 2 ∧
  original_apples = apples_to_ann + apples_to_cassie + apples_for_bridget :=
by sorry

end NUMINAMATH_CALUDE_bridget_apples_theorem_l551_55152


namespace NUMINAMATH_CALUDE_circle_C_equation_l551_55180

def symmetric_point (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = p.2 + q.2 ∧ p.1 - q.1 = q.2 - p.2

def circle_equation (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

theorem circle_C_equation (center : ℝ × ℝ) :
  symmetric_point center (1, 0) →
  circle_equation center 1 x y ↔ x^2 + (y - 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_circle_C_equation_l551_55180


namespace NUMINAMATH_CALUDE_circle_centered_at_parabola_focus_l551_55109

/-- The focus of the parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The radius of the circle -/
def circle_radius : ℝ := 2

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  (x - parabola_focus.1)^2 + (y - parabola_focus.2)^2 = circle_radius^2

theorem circle_centered_at_parabola_focus :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + y^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_centered_at_parabola_focus_l551_55109


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l551_55126

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation x^2 + 3x - 1 = 0 -/
def a : ℝ := 1
def b : ℝ := 3
def c : ℝ := -1

theorem quadratic_discriminant : discriminant a b c = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l551_55126


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l551_55127

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 = -6*x :=
sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l551_55127


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l551_55129

/-- The total surface area of a cylinder with height 12 and radius 4 is 128π. -/
theorem cylinder_surface_area :
  let h : ℝ := 12
  let r : ℝ := 4
  let circle_area := π * r^2
  let lateral_area := 2 * π * r * h
  circle_area * 2 + lateral_area = 128 * π :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l551_55129


namespace NUMINAMATH_CALUDE_container_volume_ratio_l551_55141

theorem container_volume_ratio : 
  ∀ (C D : ℝ), C > 0 → D > 0 → (3/4 * C = 2/3 * D) → C / D = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l551_55141


namespace NUMINAMATH_CALUDE_inequality_solution_l551_55123

theorem inequality_solution :
  ∃! (a b : ℝ), ∀ x : ℝ, x ∈ Set.Icc 0 1 →
    |a * x + b - Real.sqrt (1 - x^2)| ≤ (Real.sqrt 2 - 1) / 2 ∧
    a = 0 ∧ b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l551_55123


namespace NUMINAMATH_CALUDE_village_lasts_five_weeks_l551_55145

/-- The number of weeks a village lasts given supernatural predators -/
def village_duration (village_population : ℕ) 
  (lead_vampire_drain : ℕ) (vampire_group_size : ℕ) (vampire_group_drain : ℕ)
  (alpha_werewolf_eat : ℕ) (werewolf_pack_size : ℕ) (werewolf_pack_eat : ℕ)
  (ghost_feed : ℕ) : ℕ :=
  let total_consumed_per_week := 
    lead_vampire_drain + 
    (vampire_group_size * vampire_group_drain) + 
    alpha_werewolf_eat + 
    (werewolf_pack_size * werewolf_pack_eat) + 
    ghost_feed
  village_population / total_consumed_per_week

theorem village_lasts_five_weeks :
  village_duration 200 5 3 5 7 2 5 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_village_lasts_five_weeks_l551_55145


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l551_55146

-- Define a point in a 2D Cartesian coordinate system
def Point := ℝ × ℝ

-- Define the given point
def givenPoint : Point := (-2, 3)

-- Theorem stating that the coordinates of the given point with respect to the origin are (-2, 3)
theorem coordinates_wrt_origin (p : Point) (h : p = givenPoint) : p = (-2, 3) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l551_55146


namespace NUMINAMATH_CALUDE_fraction_simplification_l551_55115

theorem fraction_simplification (m : ℝ) (hm : m ≠ 0) (hm1 : m ≠ 1) (hm2 : m ≠ -1) :
  ((m - 1) / m) / ((m^2 - 1) / m^2) = m / (m + 1) := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l551_55115


namespace NUMINAMATH_CALUDE_count_flippable_numbers_is_1500_l551_55137

/-- A digit that remains valid when flipped -/
inductive ValidDigit
| Zero
| One
| Eight
| Six
| Nine

/-- A nine-digit number that remains unchanged when flipped -/
structure FlippableNumber :=
(d1 d2 d3 d4 d5 : ValidDigit)

/-- The count of FlippableNumbers -/
def count_flippable_numbers : ℕ := sorry

/-- The first digit cannot be zero -/
axiom first_digit_nonzero :
  ∀ (n : FlippableNumber), n.d1 ≠ ValidDigit.Zero

/-- The theorem to be proved -/
theorem count_flippable_numbers_is_1500 :
  count_flippable_numbers = 1500 := by sorry

end NUMINAMATH_CALUDE_count_flippable_numbers_is_1500_l551_55137


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l551_55149

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 24 → 
  Even heart → 
  (∀ h c : ℕ, h * c = 24 → Even h → heart + club ≥ h + c) →
  heart + club = 14 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l551_55149


namespace NUMINAMATH_CALUDE_triangle_area_l551_55132

/-- The area of the triangle formed by the x-axis, y-axis, and the line 3x + ay = 12 is 3/2 square units. -/
theorem triangle_area (a : ℝ) : 
  let x_intercept : ℝ := 12 / 3
  let y_intercept : ℝ := 12 / a
  let triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept
  triangle_area = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l551_55132


namespace NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l551_55142

theorem cubic_roots_sum_of_squares (a b c t : ℝ) : 
  (∀ x, x^3 - 8*x^2 + 14*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 16*t^2 - 12*t = -8*Real.sqrt 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_of_squares_l551_55142


namespace NUMINAMATH_CALUDE_average_percent_increase_l551_55106

theorem average_percent_increase (initial_population final_population : ℕ) 
  (years : ℕ) (h1 : initial_population = 175000) (h2 : final_population = 262500) 
  (h3 : years = 10) :
  (((final_population - initial_population) / years) / initial_population) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_average_percent_increase_l551_55106


namespace NUMINAMATH_CALUDE_sqrt_six_greater_than_two_l551_55167

theorem sqrt_six_greater_than_two : Real.sqrt 6 > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_greater_than_two_l551_55167


namespace NUMINAMATH_CALUDE_remaining_red_cards_l551_55158

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (removed_red_cards : ℕ)

/-- A standard deck with half red cards and 10 red cards removed -/
def standard_deck : Deck :=
  { total_cards := 52,
    red_cards := 52 / 2,
    removed_red_cards := 10 }

/-- Theorem: The number of remaining red cards in the standard deck after removal is 16 -/
theorem remaining_red_cards (d : Deck := standard_deck) :
  d.red_cards - d.removed_red_cards = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_red_cards_l551_55158


namespace NUMINAMATH_CALUDE_ken_to_don_ratio_l551_55198

-- Define the painting rates
def don_rate : ℕ := 3
def ken_rate : ℕ := don_rate + 2
def laura_rate : ℕ := 2 * ken_rate
def kim_rate : ℕ := laura_rate - 3

-- Define the total tiles painted in 15 minutes
def total_tiles : ℕ := 375

-- Theorem statement
theorem ken_to_don_ratio : 
  15 * (don_rate + ken_rate + laura_rate + kim_rate) = total_tiles →
  ken_rate / don_rate = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_ken_to_don_ratio_l551_55198


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l551_55117

/-- Given a geometric sequence {a_n} with common ratio q > 0,
    where a_2 = 1 and a_{n+2} + a_{n+1} = 6a_n,
    prove that the sum of the first four terms (S_4) is equal to 15/2. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_q_pos : q > 0)
  (h_geometric : ∀ n, a (n + 1) = a n * q)
  (h_a2 : a 2 = 1)
  (h_relation : ∀ n, a (n + 2) + a (n + 1) = 6 * a n) :
  a 1 + a 2 + a 3 + a 4 = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l551_55117


namespace NUMINAMATH_CALUDE_special_offer_cost_l551_55182

/-- Represents the cost of a T-shirt in pence -/
def TShirtCost : ℕ := 1650

/-- Represents the savings per T-shirt in pence -/
def SavingsPerShirt : ℕ := 550

/-- Represents the number of T-shirts in the offer -/
def NumShirts : ℕ := 3

/-- Represents the number of T-shirts paid for in the offer -/
def PaidShirts : ℕ := 2

theorem special_offer_cost :
  PaidShirts * TShirtCost = 3300 := by sorry

end NUMINAMATH_CALUDE_special_offer_cost_l551_55182


namespace NUMINAMATH_CALUDE_water_addition_changes_ratio_l551_55144

/-- Given a mixture of alcohol and water, prove that adding 2 liters of water
    changes the ratio from 4:3 to 4:5 when the initial amount of alcohol is 4 liters. -/
theorem water_addition_changes_ratio :
  let initial_alcohol : ℝ := 4
  let initial_water : ℝ := 3
  let water_added : ℝ := 2
  let final_water : ℝ := initial_water + water_added
  let initial_ratio : ℝ := initial_alcohol / initial_water
  let final_ratio : ℝ := initial_alcohol / final_water
  initial_ratio = 4/3 ∧ final_ratio = 4/5 := by
  sorry

#check water_addition_changes_ratio

end NUMINAMATH_CALUDE_water_addition_changes_ratio_l551_55144


namespace NUMINAMATH_CALUDE_green_pill_cost_proof_l551_55128

/-- The cost of a green pill in dollars -/
def green_pill_cost : ℚ := 22.5

/-- The cost of a pink pill in dollars -/
def pink_pill_cost : ℚ := green_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℚ := 903

theorem green_pill_cost_proof :
  green_pill_cost = 22.5 ∧
  pink_pill_cost = green_pill_cost - 2 ∧
  treatment_days = 21 ∧
  total_cost = 903 ∧
  total_cost = treatment_days * (green_pill_cost + pink_pill_cost) :=
by sorry

end NUMINAMATH_CALUDE_green_pill_cost_proof_l551_55128


namespace NUMINAMATH_CALUDE_remainder_of_60_div_18_l551_55130

theorem remainder_of_60_div_18 : ∃ q : ℕ, 60 = 18 * q + 6 := by
  sorry

#check remainder_of_60_div_18

end NUMINAMATH_CALUDE_remainder_of_60_div_18_l551_55130


namespace NUMINAMATH_CALUDE_distribution_count_correct_l551_55188

/-- The number of ways to distribute 5 indistinguishable objects into 4 distinguishable containers,
    where 2 containers are of type A and 2 are of type B,
    with at least one object in a type A container. -/
def distribution_count : ℕ := 30

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The total number of rooms -/
def num_rooms : ℕ := 4

/-- The number of rooms with a garden view -/
def num_garden_view : ℕ := 2

/-- The number of rooms without a garden view -/
def num_no_garden_view : ℕ := 2

/-- Theorem stating that the distribution count is correct -/
theorem distribution_count_correct :
  distribution_count = 30 ∧
  num_cousins = 5 ∧
  num_rooms = 4 ∧
  num_garden_view = 2 ∧
  num_no_garden_view = 2 ∧
  num_garden_view + num_no_garden_view = num_rooms :=
by sorry

end NUMINAMATH_CALUDE_distribution_count_correct_l551_55188


namespace NUMINAMATH_CALUDE_remaining_roots_equation_l551_55120

theorem remaining_roots_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hab : a ≠ b) :
  ∃ x₁ : ℝ, (x₁^2 + a*x₁ + b*c = 0 ∧ x₁^2 + b*x₁ + c*a = 0) →
  ∃ x₂ x₃ : ℝ, x₂ ≠ x₁ ∧ x₃ ≠ x₁ ∧ x₂^2 + c*x₂ + a*b = 0 ∧ x₃^2 + c*x₃ + a*b = 0 :=
sorry

end NUMINAMATH_CALUDE_remaining_roots_equation_l551_55120


namespace NUMINAMATH_CALUDE_zack_traveled_to_18_countries_l551_55133

-- Define the number of countries each person traveled to
def george_countries : ℕ := 6
def joseph_countries : ℕ := george_countries / 2
def patrick_countries : ℕ := joseph_countries * 3
def zack_countries : ℕ := patrick_countries * 2

-- Theorem statement
theorem zack_traveled_to_18_countries :
  zack_countries = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_zack_traveled_to_18_countries_l551_55133


namespace NUMINAMATH_CALUDE_fruit_pie_theorem_l551_55166

/-- Represents the number of fruits needed for different types of pies -/
structure FruitRequirement where
  apples : ℕ
  pears : ℕ
  peaches : ℕ

/-- Calculates the total fruits needed for a given number of pies -/
def total_fruits (req : FruitRequirement) (num_pies : ℕ) : FruitRequirement :=
  { apples := req.apples * num_pies
  , pears := req.pears * num_pies
  , peaches := req.peaches * num_pies }

/-- Adds two FruitRequirement structures -/
def add_requirements (a b : FruitRequirement) : FruitRequirement :=
  { apples := a.apples + b.apples
  , pears := a.pears + b.pears
  , peaches := a.peaches + b.peaches }

theorem fruit_pie_theorem :
  let fruit_pie_req : FruitRequirement := { apples := 4, pears := 3, peaches := 0 }
  let apple_peach_pie_req : FruitRequirement := { apples := 6, pears := 0, peaches := 2 }
  let fruit_pies := 357
  let apple_peach_pies := 712
  let total_req := add_requirements (total_fruits fruit_pie_req fruit_pies) (total_fruits apple_peach_pie_req apple_peach_pies)
  total_req.apples = 5700 ∧ total_req.pears = 1071 ∧ total_req.peaches = 1424 := by
  sorry

end NUMINAMATH_CALUDE_fruit_pie_theorem_l551_55166


namespace NUMINAMATH_CALUDE_polygon_division_euler_characteristic_l551_55108

/-- A polygon division represents the result of dividing a polygon into several polygons. -/
structure PolygonDivision where
  p : ℕ  -- number of resulting polygons
  q : ℕ  -- number of segments that are the sides of these polygons
  r : ℕ  -- number of points that are their vertices

/-- The Euler characteristic of a polygon division is always 1. -/
theorem polygon_division_euler_characteristic (d : PolygonDivision) : 
  d.p - d.q + d.r = 1 := by
  sorry

end NUMINAMATH_CALUDE_polygon_division_euler_characteristic_l551_55108


namespace NUMINAMATH_CALUDE_initial_student_count_l551_55168

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) :
  initial_avg = 61.5 →
  new_avg = 64.0 →
  dropped_score = 24 →
  ∃ n : ℕ, n * initial_avg = (n - 1) * new_avg + dropped_score ∧ n = 16 :=
by sorry

end NUMINAMATH_CALUDE_initial_student_count_l551_55168


namespace NUMINAMATH_CALUDE_complement_subset_relation_l551_55121

open Set

theorem complement_subset_relation (P Q : Set ℝ) : 
  (P = {x : ℝ | 0 < x ∧ x < 1}) → 
  (Q = {x : ℝ | x^2 + x - 2 ≤ 0}) → 
  ((compl Q) ⊆ (compl P)) :=
by
  sorry

end NUMINAMATH_CALUDE_complement_subset_relation_l551_55121


namespace NUMINAMATH_CALUDE_log_equation_solution_l551_55105

theorem log_equation_solution (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 → y = 3^(10/3) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l551_55105


namespace NUMINAMATH_CALUDE_tree_planting_group_size_l551_55119

/-- Proves that the number of people in the first group is 3, given the conditions of the tree planting activity. -/
theorem tree_planting_group_size :
  ∀ (x : ℕ), 
    (12 : ℚ) / x = (36 : ℚ) / (x + 6) →
    x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_group_size_l551_55119


namespace NUMINAMATH_CALUDE_prime_pythagorean_triple_l551_55170

theorem prime_pythagorean_triple (p m n : ℕ) 
  (hp : Nat.Prime p) 
  (hm : m > 0) 
  (hn : n > 0) 
  (heq : p^2 + m^2 = n^2) : 
  m > p := by
  sorry

end NUMINAMATH_CALUDE_prime_pythagorean_triple_l551_55170


namespace NUMINAMATH_CALUDE_ratio_problem_l551_55169

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l551_55169


namespace NUMINAMATH_CALUDE_part_one_part_two_l551_55181

-- Define the conditions P and Q
def P (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def Q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Part 1
theorem part_one (x : ℝ) (h : P x 1 ∧ Q x) : 2 < x ∧ x < 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) 
  (h : ∀ x, ¬(P x a) → ¬(Q x)) 
  (h_not_nec : ∃ x, ¬(P x a) ∧ Q x) : 
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l551_55181


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l551_55187

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a line of the form y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The main theorem stating the properties of the parabola and line -/
theorem parabola_and_line_properties
  (p : Parabola)
  (l : Line)
  (passes_through_A : p.a + p.b + p.c = 0)
  (axis_of_symmetry : ∀ x, p.a * (x - 3)^2 + p.b * (x - 3) + p.c = p.a * x^2 + p.b * x + p.c)
  (line_passes_through_A : l.k + l.m = 0)
  (line_passes_through_B : ∃ x, p.a * x^2 + p.b * x + p.c = l.k * x + l.m ∧ x ≠ 1)
  (triangle_area : |l.m| = 4) :
  ((l.k = -4 ∧ l.m = 4 ∧ p.a = 2 ∧ p.b = -12 ∧ p.c = 10) ∨
   (l.k = 4 ∧ l.m = -4 ∧ p.a = -2 ∧ p.b = 12 ∧ p.c = -10)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l551_55187


namespace NUMINAMATH_CALUDE_problem_solution_l551_55165

theorem problem_solution (x y z : ℝ) 
  (h1 : |x| + x + y = 12)
  (h2 : x + |y| - y = 10)
  (h3 : x - y + z = 5) :
  x + y + z = 9/5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l551_55165


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_negative_four_l551_55111

theorem at_least_one_not_greater_than_negative_four
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_negative_four_l551_55111


namespace NUMINAMATH_CALUDE_sara_movie_purchase_cost_l551_55163

/-- The amount Sara spent on movie theater tickets -/
def theater_ticket_cost : ℚ := 10.62

/-- The number of movie theater tickets Sara bought -/
def number_of_tickets : ℕ := 2

/-- The cost of renting a movie -/
def rental_cost : ℚ := 1.59

/-- The total amount Sara spent on movies -/
def total_spent : ℚ := 36.78

/-- Theorem: Given the conditions, Sara spent $13.95 on buying the movie -/
theorem sara_movie_purchase_cost :
  total_spent - (theater_ticket_cost * number_of_tickets + rental_cost) = 13.95 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_purchase_cost_l551_55163


namespace NUMINAMATH_CALUDE_dice_product_composite_probability_l551_55100

def num_dice : ℕ := 6
def num_sides : ℕ := 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

def total_outcomes : ℕ := num_sides ^ num_dice

def non_composite_outcomes : ℕ := 25

theorem dice_product_composite_probability :
  (total_outcomes - non_composite_outcomes) / total_outcomes = 262119 / 262144 :=
sorry

end NUMINAMATH_CALUDE_dice_product_composite_probability_l551_55100


namespace NUMINAMATH_CALUDE_wireless_mice_ratio_l551_55134

/-- Proves that the ratio of wireless mice to total mice sold is 1:2 -/
theorem wireless_mice_ratio (total_mice : ℕ) (optical_mice : ℕ) (trackball_mice : ℕ) :
  total_mice = 80 →
  optical_mice = total_mice / 4 →
  trackball_mice = 20 →
  let wireless_mice := total_mice - (optical_mice + trackball_mice)
  (wireless_mice : ℚ) / total_mice = 1 / 2 := by
  sorry

#check wireless_mice_ratio

end NUMINAMATH_CALUDE_wireless_mice_ratio_l551_55134


namespace NUMINAMATH_CALUDE_cookies_distribution_l551_55155

theorem cookies_distribution (total_cookies : ℕ) (num_people : ℕ) (cookies_per_person : ℕ) : 
  total_cookies = 35 → 
  num_people = 5 → 
  total_cookies = num_people * cookies_per_person → 
  cookies_per_person = 7 := by
sorry

end NUMINAMATH_CALUDE_cookies_distribution_l551_55155


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l551_55122

theorem pascal_triangle_interior_sum (row_6_sum : ℕ) (row_8_sum : ℕ) : 
  row_6_sum = 30 → row_8_sum = 126 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l551_55122


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l551_55107

theorem sqrt_sum_equals_eleven_sqrt_two_over_six :
  Real.sqrt (9/2) + Real.sqrt (2/9) = 11 * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_eleven_sqrt_two_over_six_l551_55107


namespace NUMINAMATH_CALUDE_average_pages_is_23_l551_55139

/-- The number of pages in the storybook Taesoo read -/
def total_pages : ℕ := 161

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The average number of pages read per day -/
def average_pages : ℚ := total_pages / days_in_week

/-- Theorem stating that the average number of pages read per day is 23 -/
theorem average_pages_is_23 : average_pages = 23 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_is_23_l551_55139


namespace NUMINAMATH_CALUDE_no_lattice_equilateral_triangle_l551_55150

-- Define a lattice point as a point with integer coordinates
def LatticePoint (p : ℝ × ℝ) : Prop :=
  ∃ (x y : ℤ), p = (↑x, ↑y)

-- Define an equilateral triangle
def Equilateral (a b c : ℝ × ℝ) : Prop :=
  let d := (fun (p q : ℝ × ℝ) => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2))
  d a b = d b c ∧ d b c = d c a

-- Theorem statement
theorem no_lattice_equilateral_triangle :
  ¬ ∃ (a b c : ℝ × ℝ), LatticePoint a ∧ LatticePoint b ∧ LatticePoint c ∧ Equilateral a b c :=
sorry

end NUMINAMATH_CALUDE_no_lattice_equilateral_triangle_l551_55150


namespace NUMINAMATH_CALUDE_inequality_system_solution_l551_55175

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x - a > 2 ∧ b - 2*x > 0) ↔ (-1 < x ∧ x < 1)) →
  (a + b)^2021 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l551_55175


namespace NUMINAMATH_CALUDE_remainder_theorem_l551_55194

theorem remainder_theorem (r : ℝ) : (r^13 + 1) % (r - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l551_55194


namespace NUMINAMATH_CALUDE_pizza_sharing_l551_55189

theorem pizza_sharing (total pizza_jovin pizza_anna pizza_olivia : ℚ) : 
  total = 1 →
  pizza_jovin = 1/3 →
  pizza_anna = 1/6 →
  pizza_olivia = 1/4 →
  total - (pizza_jovin + pizza_anna + pizza_olivia) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_sharing_l551_55189


namespace NUMINAMATH_CALUDE_rest_area_distance_l551_55186

theorem rest_area_distance (d : ℝ) : 
  (¬ (d ≥ 8)) →  -- David's statement is false
  (¬ (d ≤ 7)) →  -- Ellen's statement is false
  (¬ (d ≤ 6)) →  -- Frank's statement is false
  (7 < d ∧ d < 8) := by
sorry

end NUMINAMATH_CALUDE_rest_area_distance_l551_55186


namespace NUMINAMATH_CALUDE_no_solution_iff_a_leq_8_l551_55116

theorem no_solution_iff_a_leq_8 :
  ∀ a : ℝ, (∀ x : ℝ, ¬(|x - 5| + |x + 3| < a)) ↔ a ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_a_leq_8_l551_55116


namespace NUMINAMATH_CALUDE_conic_section_eccentricity_l551_55101

/-- The conic section defined by the equation 10x - 2xy - 2y + 1 = 0 -/
def ConicSection (x y : ℝ) : Prop :=
  10 * x - 2 * x * y - 2 * y + 1 = 0

/-- The eccentricity of a conic section -/
def Eccentricity (e : ℝ) : Prop :=
  e = Real.sqrt 2

theorem conic_section_eccentricity :
  ∀ x y : ℝ, ConicSection x y → ∃ e : ℝ, Eccentricity e := by
  sorry

end NUMINAMATH_CALUDE_conic_section_eccentricity_l551_55101


namespace NUMINAMATH_CALUDE_james_total_toys_l551_55160

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

/-- Theorem stating that the total number of toys James buys is 60 -/
theorem james_total_toys : total_toys = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_total_toys_l551_55160


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l551_55112

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_necessary_not_sufficient 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicularToPlane l α) 
  (h2 : subset m β) :
  (∀ (α β : Plane), parallel α β → perpendicular l m) ∧ 
  (∃ (l m : Line) (α β : Plane), 
    perpendicularToPlane l α ∧ 
    subset m β ∧ 
    perpendicular l m ∧ 
    ¬(parallel α β)) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l551_55112


namespace NUMINAMATH_CALUDE_nikolai_is_petrs_son_l551_55135

/-- Represents a person who went fishing -/
structure Fisher where
  name : String
  fish_caught : ℕ

/-- Represents a father-son pair who went fishing -/
structure FishingPair where
  father : Fisher
  son : Fisher

/-- The total number of fish caught by all fishers -/
def total_fish : ℕ := 25

/-- Theorem stating that given the conditions, Nikolai must be Petr's son -/
theorem nikolai_is_petrs_son (pair1 pair2 : FishingPair) 
  (h1 : pair1.father.name = "Petr")
  (h2 : pair1.father.fish_caught = 3 * pair1.son.fish_caught)
  (h3 : pair2.father.fish_caught = pair2.son.fish_caught)
  (h4 : pair1.father.fish_caught + pair1.son.fish_caught + 
        pair2.father.fish_caught + pair2.son.fish_caught = total_fish)
  : pair1.son.name = "Nikolai" := by
  sorry

end NUMINAMATH_CALUDE_nikolai_is_petrs_son_l551_55135


namespace NUMINAMATH_CALUDE_weekday_classes_count_l551_55196

/-- Represents the Diving Club's class schedule --/
structure DivingClub where
  weekdayClasses : ℕ
  weekendClassesPerDay : ℕ
  peoplePerClass : ℕ
  totalWeeks : ℕ
  totalPeople : ℕ

/-- Calculates the total number of people who can take classes --/
def totalCapacity (club : DivingClub) : ℕ :=
  (club.weekdayClasses * club.totalWeeks + 
   club.weekendClassesPerDay * 2 * club.totalWeeks) * club.peoplePerClass

/-- Theorem stating the number of weekday classes --/
theorem weekday_classes_count (club : DivingClub) 
  (h1 : club.weekendClassesPerDay = 4)
  (h2 : club.peoplePerClass = 5)
  (h3 : club.totalWeeks = 3)
  (h4 : club.totalPeople = 270)
  (h5 : totalCapacity club = club.totalPeople) :
  club.weekdayClasses = 10 := by
  sorry

#check weekday_classes_count

end NUMINAMATH_CALUDE_weekday_classes_count_l551_55196


namespace NUMINAMATH_CALUDE_ribbon_boxes_theorem_l551_55103

theorem ribbon_boxes_theorem (total_ribbon : ℝ) (ribbon_per_box : ℝ) (leftover : ℝ) :
  total_ribbon = 12.5 ∧ 
  ribbon_per_box = 1.75 ∧ 
  leftover = 0.3 → 
  ⌊total_ribbon / (ribbon_per_box + leftover)⌋ = 6 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_boxes_theorem_l551_55103


namespace NUMINAMATH_CALUDE_watch_time_theorem_l551_55124

/-- Represents a season of the TV show -/
structure Season where
  episodes : Nat
  minutesPerEpisode : Nat

/-- Calculates the total number of days needed to watch the show -/
def daysToWatchShow (seasons : List Season) (hoursPerDay : Nat) : Nat :=
  let totalMinutes := seasons.foldl (fun acc s => acc + s.episodes * s.minutesPerEpisode) 0
  let minutesPerDay := hoursPerDay * 60
  (totalMinutes + minutesPerDay - 1) / minutesPerDay

/-- The main theorem stating it takes 35 days to watch the show -/
theorem watch_time_theorem (seasons : List Season) (hoursPerDay : Nat) :
  seasons = [
    ⟨30, 22⟩, ⟨28, 25⟩, ⟨27, 29⟩, ⟨20, 31⟩, ⟨25, 27⟩, ⟨20, 35⟩
  ] →
  hoursPerDay = 2 →
  daysToWatchShow seasons hoursPerDay = 35 := by
  sorry

#eval daysToWatchShow [
  ⟨30, 22⟩, ⟨28, 25⟩, ⟨27, 29⟩, ⟨20, 31⟩, ⟨25, 27⟩, ⟨20, 35⟩
] 2

end NUMINAMATH_CALUDE_watch_time_theorem_l551_55124


namespace NUMINAMATH_CALUDE_smallest_cross_family_bound_l551_55184

/-- A family of subsets A of a finite set X is a cross family if for every subset B of X,
    B is comparable with at least one subset in A. -/
def IsCrossFamily (X : Finset α) (A : Finset (Finset α)) : Prop :=
  ∀ B : Finset α, B ⊆ X → ∃ A' ∈ A, A' ⊆ B ∨ B ⊆ A'

/-- A is the smallest cross family if no proper subfamily of A is a cross family. -/
def IsSmallestCrossFamily (X : Finset α) (A : Finset (Finset α)) : Prop :=
  IsCrossFamily X A ∧ ∀ A' ⊂ A, ¬IsCrossFamily X A'

theorem smallest_cross_family_bound {α : Type*} [DecidableEq α] (X : Finset α) (A : Finset (Finset α)) :
  IsSmallestCrossFamily X A → A.card ≤ Nat.choose X.card (X.card / 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_cross_family_bound_l551_55184


namespace NUMINAMATH_CALUDE_derivative_at_pi_sixth_l551_55178

theorem derivative_at_pi_sixth (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = Real.cos x - Real.sin x) →
  (∀ x, HasDerivAt f (f' x) x) →
  f' (π/6) = -(1 + Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_pi_sixth_l551_55178


namespace NUMINAMATH_CALUDE_trajectory_equation_l551_55199

/-- The ellipse on which points M and N lie -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The condition for the product of slopes of OM and ON -/
def slope_product (a b : ℝ) (m_slope n_slope : ℝ) : Prop :=
  m_slope * n_slope = b^2 / a^2

/-- The trajectory equation for point P -/
def trajectory (a b m n : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = m^2 + n^2

/-- The main theorem -/
theorem trajectory_equation (a b : ℝ) (m n : ℕ+) (x y : ℝ) :
  a > b ∧ b > 0 →
  ∃ (mx my nx ny : ℝ),
    ellipse a b mx my ∧
    ellipse a b nx ny ∧
    ∃ (m_slope n_slope : ℝ),
      slope_product a b m_slope n_slope →
      x = m * mx + n * nx ∧
      y = m * my + n * ny →
      trajectory a b m n x y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l551_55199


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l551_55183

theorem interest_rate_calculation (initial_charge : ℝ) (final_amount : ℝ) (time : ℝ) :
  initial_charge = 75 →
  final_amount = 80.25 →
  time = 1 →
  (final_amount - initial_charge) / (initial_charge * time) = 0.07 :=
by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l551_55183


namespace NUMINAMATH_CALUDE_quadratic_polynomial_roots_l551_55114

theorem quadratic_polynomial_roots (x₁ x₂ : ℝ) (h_sum : x₁ + x₂ = 8) (h_product : x₁ * x₂ = 16) :
  x₁ * x₂ = 16 ∧ x₁ + x₂ = 8 ↔ x₁^2 - 8*x₁ + 16 = 0 ∧ x₂^2 - 8*x₂ + 16 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_roots_l551_55114


namespace NUMINAMATH_CALUDE_unique_nine_digit_number_l551_55162

/-- A permutation of digits 1 to 9 -/
def Permutation9 := Fin 9 → Fin 9

/-- Checks if a function is a valid permutation of digits 1 to 9 -/
def is_valid_permutation (p : Permutation9) : Prop :=
  Function.Injective p ∧ Function.Surjective p

/-- Converts a permutation to a natural number -/
def permutation_to_nat (p : Permutation9) : ℕ :=
  (List.range 9).foldl (fun acc i => acc * 10 + (p i).val + 1) 0

/-- The property that a permutation decreases by 8 times after rearrangement -/
def decreases_by_8_times (p : Permutation9) : Prop :=
  ∃ q : Permutation9, is_valid_permutation q ∧ permutation_to_nat p = 8 * permutation_to_nat q

theorem unique_nine_digit_number :
  ∃! p : Permutation9, is_valid_permutation p ∧ decreases_by_8_times p ∧ permutation_to_nat p = 123456789 :=
sorry

end NUMINAMATH_CALUDE_unique_nine_digit_number_l551_55162


namespace NUMINAMATH_CALUDE_median_name_length_and_syllables_l551_55179

theorem median_name_length_and_syllables :
  let total_names : ℕ := 23
  let names_4_1 : ℕ := 8  -- 8 names of length 4 and 1 syllable
  let names_5_2 : ℕ := 5  -- 5 names of length 5 and 2 syllables
  let names_3_1 : ℕ := 3  -- 3 names of length 3 and 1 syllable
  let names_6_2 : ℕ := 4  -- 4 names of length 6 and 2 syllables
  let names_7_3 : ℕ := 3  -- 3 names of length 7 and 3 syllables
  
  let median_position : ℕ := (total_names + 1) / 2
  
  let lengths : List ℕ := [3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7]
  let syllables : List ℕ := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
  
  (median_position = 12) ∧
  (lengths.get! (median_position - 1) = 5) ∧
  (syllables.get! (median_position - 1) = 1) :=
by sorry

end NUMINAMATH_CALUDE_median_name_length_and_syllables_l551_55179


namespace NUMINAMATH_CALUDE_internet_discount_percentage_l551_55138

theorem internet_discount_percentage
  (monthly_rate : ℝ)
  (total_payment : ℝ)
  (num_months : ℕ)
  (h1 : monthly_rate = 50)
  (h2 : total_payment = 190)
  (h3 : num_months = 4) :
  (monthly_rate - total_payment / num_months) / monthly_rate * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_internet_discount_percentage_l551_55138


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l551_55140

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l551_55140


namespace NUMINAMATH_CALUDE_line_circle_intersection_l551_55161

theorem line_circle_intersection (k : ℝ) : 
  ∃ x y : ℝ, y = k * (x + 1/2) ∧ x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l551_55161


namespace NUMINAMATH_CALUDE_unique_solution_cubic_rational_equation_l551_55125

theorem unique_solution_cubic_rational_equation :
  ∃! x : ℝ, (x^3 - 3*x^2 + 2*x)/(x^2 + 2*x + 1) + 2*x = -8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_rational_equation_l551_55125


namespace NUMINAMATH_CALUDE_round_trip_time_ratio_l551_55148

/-- Proves that for a round trip with given average speeds, the ratio of return to outbound journey times is 3:2 -/
theorem round_trip_time_ratio 
  (distance : ℝ) 
  (speed_to_destination : ℝ) 
  (average_speed_round_trip : ℝ) 
  (h1 : speed_to_destination = 54) 
  (h2 : average_speed_round_trip = 36) 
  (h3 : distance > 0) 
  (h4 : speed_to_destination > 0) 
  (h5 : average_speed_round_trip > 0) : 
  (distance / average_speed_round_trip - distance / speed_to_destination) / (distance / speed_to_destination) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_time_ratio_l551_55148


namespace NUMINAMATH_CALUDE_water_drinking_time_l551_55193

/-- Proves that given a goal of drinking 3 liters of water and drinking 500 milliliters every 2 hours, it will take 12 hours to reach the goal. -/
theorem water_drinking_time (goal : ℕ) (intake : ℕ) (frequency : ℕ) (h1 : goal = 3) (h2 : intake = 500) (h3 : frequency = 2) : 
  (goal * 1000) / intake * frequency = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_drinking_time_l551_55193


namespace NUMINAMATH_CALUDE_coin_value_difference_l551_55191

-- Define the coin types
inductive Coin
| Penny
| Nickel
| Dime

-- Define the function to calculate the value of a coin in cents
def coinValue : Coin → Nat
| Coin.Penny => 1
| Coin.Nickel => 5
| Coin.Dime => 10

-- Define the total number of coins
def totalCoins : Nat := 3000

-- Define the theorem
theorem coin_value_difference :
  ∃ (p n d : Nat),
    p + n + d = totalCoins ∧
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧
    (∀ (p' n' d' : Nat),
      p' + n' + d' = totalCoins →
      p' ≥ 1 → n' ≥ 1 → d' ≥ 1 →
      coinValue Coin.Penny * p' + coinValue Coin.Nickel * n' + coinValue Coin.Dime * d' ≤
      coinValue Coin.Penny * p + coinValue Coin.Nickel * n + coinValue Coin.Dime * d) ∧
    (∀ (p' n' d' : Nat),
      p' + n' + d' = totalCoins →
      p' ≥ 1 → n' ≥ 1 → d' ≥ 1 →
      coinValue Coin.Penny * p + coinValue Coin.Nickel * n + coinValue Coin.Dime * d -
      (coinValue Coin.Penny * p' + coinValue Coin.Nickel * n' + coinValue Coin.Dime * d') = 26973) :=
by sorry


end NUMINAMATH_CALUDE_coin_value_difference_l551_55191


namespace NUMINAMATH_CALUDE_different_color_probability_is_two_thirds_l551_55197

/-- The number of possible colors for the shorts -/
def shorts_colors : ℕ := 2

/-- The number of possible colors for the jersey -/
def jersey_colors : ℕ := 3

/-- The total number of possible color combinations -/
def total_combinations : ℕ := shorts_colors * jersey_colors

/-- The number of combinations where the shorts and jersey colors are different -/
def different_color_combinations : ℕ := shorts_colors * (jersey_colors - 1)

/-- The probability that the shorts will be a different color than the jersey -/
def different_color_probability : ℚ := different_color_combinations / total_combinations

theorem different_color_probability_is_two_thirds :
  different_color_probability = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_different_color_probability_is_two_thirds_l551_55197


namespace NUMINAMATH_CALUDE_inequality_direction_change_l551_55173

theorem inequality_direction_change : ∃ (a b c : ℝ), a < b ∧ c * a > c * b :=
sorry

end NUMINAMATH_CALUDE_inequality_direction_change_l551_55173


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l551_55185

theorem unique_solution_cube_equation : 
  ∃! (x : ℝ), x ≠ 0 ∧ (3 * x)^5 = (9 * x)^4 ∧ x = 27 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l551_55185


namespace NUMINAMATH_CALUDE_james_work_hours_l551_55190

def minimum_wage : ℚ := 8
def meat_pounds : ℕ := 20
def meat_price : ℚ := 5
def fruit_veg_pounds : ℕ := 15
def fruit_veg_price : ℚ := 4
def bread_pounds : ℕ := 60
def bread_price : ℚ := 3/2
def janitor_hours : ℕ := 10
def janitor_wage : ℚ := 10

def total_cost : ℚ := 
  meat_pounds * meat_price + 
  fruit_veg_pounds * fruit_veg_price + 
  bread_pounds * bread_price + 
  janitor_hours * (janitor_wage * 3/2)

theorem james_work_hours : 
  total_cost / minimum_wage = 50 := by sorry

end NUMINAMATH_CALUDE_james_work_hours_l551_55190


namespace NUMINAMATH_CALUDE_abs_neg_x_eq_2023_l551_55156

theorem abs_neg_x_eq_2023 (x : ℝ) :
  |(-x)| = 2023 → x = 2023 ∨ x = -2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_x_eq_2023_l551_55156


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l551_55171

def y : ℕ := 2^63^74^95^86^47^5

theorem smallest_multiplier_for_perfect_cube (n : ℕ) :
  (∀ m : ℕ, 0 < m ∧ m < 18 → ¬ ∃ k : ℕ, y * m = k^3) ∧
  ∃ k : ℕ, y * 18 = k^3 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_cube_l551_55171


namespace NUMINAMATH_CALUDE_inequality_range_l551_55143

theorem inequality_range (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| > a) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l551_55143


namespace NUMINAMATH_CALUDE_notepad_lasts_four_days_l551_55154

/-- Calculates the number of days a notepad lasts given the specified conditions -/
def notepadDuration (piecesPerNotepad : ℕ) (folds : ℕ) (notesPerDay : ℕ) : ℕ :=
  let sectionsPerPiece := 2^folds
  let totalNotes := piecesPerNotepad * sectionsPerPiece
  totalNotes / notesPerDay

/-- Theorem stating that under the given conditions, a notepad lasts 4 days -/
theorem notepad_lasts_four_days :
  notepadDuration 5 3 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_notepad_lasts_four_days_l551_55154


namespace NUMINAMATH_CALUDE_rachel_milk_consumption_l551_55151

theorem rachel_milk_consumption 
  (bottle1 : ℚ) (bottle2 : ℚ) (rachel_fraction : ℚ) :
  bottle1 = 3/8 →
  bottle2 = 1/4 →
  rachel_fraction = 3/4 →
  rachel_fraction * (bottle1 + bottle2) = 15/32 := by
  sorry

end NUMINAMATH_CALUDE_rachel_milk_consumption_l551_55151


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_zero_l551_55174

/-- The equation of the tangent line to y = x^3 - 2x + 1 at (1, 0) is y = x - 1 -/
theorem tangent_line_at_point_one_zero (x y : ℝ) :
  let f : ℝ → ℝ := λ t => t^3 - 2*t + 1
  let f' : ℝ → ℝ := λ t => 3*t^2 - 2
  let tangent_line : ℝ → ℝ := λ t => t - 1
  f 1 = 0 ∧ f' 1 = (tangent_line 1 - tangent_line 0) → 
  ∀ t, tangent_line t = f 1 + f' 1 * (t - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_zero_l551_55174


namespace NUMINAMATH_CALUDE_compound_oxygen_count_l551_55153

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  hydrogen : ℕ
  bromine : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (h_weight o_weight br_weight : ℕ) : ℕ :=
  c.hydrogen * h_weight + c.bromine * br_weight + c.oxygen * o_weight

theorem compound_oxygen_count :
  ∀ (c : Compound),
    c.hydrogen = 1 →
    c.bromine = 1 →
    molecularWeight c 1 16 80 = 129 →
    c.oxygen = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_count_l551_55153


namespace NUMINAMATH_CALUDE_sons_age_l551_55172

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 20 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 18 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l551_55172


namespace NUMINAMATH_CALUDE_polar_rectangular_equivalence_l551_55176

theorem polar_rectangular_equivalence (ρ θ x y : ℝ) :
  y = ρ * Real.sin θ ∧ x = ρ * Real.cos θ →
  (y^2 = 12 * x ↔ ρ * Real.sin θ^2 = 12 * Real.cos θ) :=
by sorry

end NUMINAMATH_CALUDE_polar_rectangular_equivalence_l551_55176


namespace NUMINAMATH_CALUDE_problem_1_l551_55102

theorem problem_1 : 2 * Real.sqrt 28 + 7 * Real.sqrt 7 - Real.sqrt 7 * Real.sqrt (4/7) = 11 * Real.sqrt 7 - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l551_55102


namespace NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l551_55147

/-- Represents the time to empty a water tank given specific conditions. -/
def time_to_empty_tank (initial_fill : ℚ) (fill_rate : ℚ) (empty_rate : ℚ) : ℚ :=
  initial_fill / (empty_rate - fill_rate)

/-- Theorem stating that under given conditions, the tank will be emptied in 3 minutes. -/
theorem tank_emptied_in_three_minutes :
  let initial_fill : ℚ := 1/5
  let fill_rate : ℚ := 1/10
  let empty_rate : ℚ := 1/6
  time_to_empty_tank initial_fill fill_rate empty_rate = 3 := by
  sorry

#eval time_to_empty_tank (1/5) (1/10) (1/6)

end NUMINAMATH_CALUDE_tank_emptied_in_three_minutes_l551_55147


namespace NUMINAMATH_CALUDE_product_increased_equals_nineteen_l551_55195

theorem product_increased_equals_nineteen (x : ℝ) : 5 * x + 4 = 19 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_increased_equals_nineteen_l551_55195


namespace NUMINAMATH_CALUDE_cafeteria_red_apples_l551_55159

/-- The number of red apples ordered by the cafeteria -/
def red_apples : ℕ := sorry

/-- The number of green apples ordered by the cafeteria -/
def green_apples : ℕ := 17

/-- The number of students who took apples -/
def students_took_apples : ℕ := 10

/-- The number of extra apples left -/
def extra_apples : ℕ := 32

/-- The total number of apples ordered by the cafeteria -/
def total_apples : ℕ := red_apples + green_apples

theorem cafeteria_red_apples :
  red_apples = 25 :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_red_apples_l551_55159


namespace NUMINAMATH_CALUDE_jerusha_earnings_l551_55192

theorem jerusha_earnings (L : ℝ) : 
  L + 4 * L = 85 → 4 * L = 68 := by
  sorry

end NUMINAMATH_CALUDE_jerusha_earnings_l551_55192


namespace NUMINAMATH_CALUDE_age_of_b_l551_55136

theorem age_of_b (a b c : ℕ) : 
  (a + b + c) / 3 = 29 →
  (a + c) / 2 = 32 →
  b = 23 := by
sorry

end NUMINAMATH_CALUDE_age_of_b_l551_55136


namespace NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l551_55131

/-- Converts kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) (seconds_per_hour : ℕ) : ℝ :=
  speed_km_per_second * (seconds_per_hour : ℝ)

/-- Theorem: A space shuttle orbiting at 9 km/s is equivalent to 32400 km/h -/
theorem space_shuttle_speed_conversion :
  km_per_second_to_km_per_hour 9 3600 = 32400 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_conversion_l551_55131


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l551_55113

/-- Proves that adding 3.6 liters of pure alcohol to a 6-liter solution
    that is 20% alcohol results in a solution that is 50% alcohol. -/
theorem alcohol_mixture_proof
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_alcohol : ℝ)
  (final_concentration : ℝ)
  (h1 : initial_volume = 6)
  (h2 : initial_concentration = 0.20)
  (h3 : added_alcohol = 3.6)
  (h4 : final_concentration = 0.50) :
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l551_55113


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l551_55118

/-- The equation of a line passing through (0, 2) with slope 2 is 2x - y + 2 = 0 -/
theorem line_equation_through_point_with_slope (x y : ℝ) :
  (y - 2 = 2 * (x - 0)) ↔ (2 * x - y + 2 = 0) := by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_slope_l551_55118


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l551_55157

/-- A regular polygon with exterior angles measuring 18 degrees has 20 sides
    and the sum of its interior angles is 3240 degrees. -/
theorem regular_polygon_properties (n : ℕ) (exterior_angle : ℝ) :
  exterior_angle = 18 →
  (360 : ℝ) / exterior_angle = n →
  n = 20 ∧
  180 * (n - 2) = 3240 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l551_55157
