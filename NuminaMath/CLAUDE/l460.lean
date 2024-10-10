import Mathlib

namespace tyrone_gave_fifteen_marbles_l460_46084

/-- Represents the marble redistribution problem between Tyrone and Eric -/
def marble_redistribution (x : ℕ) : Prop :=
  let tyrone_initial : ℕ := 150
  let eric_initial : ℕ := 30
  let tyrone_final : ℕ := tyrone_initial - x
  let eric_final : ℕ := eric_initial + x
  (tyrone_final = 3 * eric_final) ∧ (x > 0) ∧ (x < tyrone_initial)

/-- The theorem stating that Tyrone gave 15 marbles to Eric -/
theorem tyrone_gave_fifteen_marbles : 
  ∃ (x : ℕ), marble_redistribution x ∧ x = 15 := by
sorry

end tyrone_gave_fifteen_marbles_l460_46084


namespace missing_fraction_sum_l460_46062

theorem missing_fraction_sum (x : ℚ) : 
  x = 7/15 → 
  (1/3 : ℚ) + (1/2 : ℚ) + (-5/6 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + x = 
  (13333333333333333 : ℚ) / (100000000000000000 : ℚ) := by
  sorry

end missing_fraction_sum_l460_46062


namespace positive_number_problem_l460_46043

theorem positive_number_problem : ∃ (n : ℕ), n > 0 ∧ 3 * n + n^2 = 300 ∧ n = 16 := by
  sorry

end positive_number_problem_l460_46043


namespace sara_letters_ratio_l460_46001

theorem sara_letters_ratio (january february total : ℕ) 
  (h1 : january = 6)
  (h2 : february = 9)
  (h3 : total = 33) :
  (total - january - february) / january = 3 := by
  sorry

end sara_letters_ratio_l460_46001


namespace solution_set_implies_ab_l460_46059

theorem solution_set_implies_ab (a b : ℝ) : 
  (∀ x, x^2 + a*x + b ≤ 0 ↔ -1 ≤ x ∧ x ≤ 3) → a*b = 6 := by
  sorry

end solution_set_implies_ab_l460_46059


namespace exists_divisible_by_n_l460_46015

theorem exists_divisible_by_n (n : ℕ) (h_odd : Odd n) (h_gt_one : n > 1) :
  ∃ k : ℕ, k < n ∧ (n ∣ 2^k - 1) := by
  sorry

end exists_divisible_by_n_l460_46015


namespace acute_angle_equation_l460_46009

theorem acute_angle_equation (x : Real) : 
  x = π/3 → (Real.sin (2*x) + Real.cos x) * (Real.sin x - Real.cos x) = Real.cos x := by
  sorry

end acute_angle_equation_l460_46009


namespace range_of_m_l460_46057

theorem range_of_m (x m : ℝ) : 
  (∀ x, 1/3 < x ∧ x < 1/2 → |x - m| < 1) ∧ 
  (∃ x, |x - m| < 1 ∧ ¬(1/3 < x ∧ x < 1/2)) →
  -1/2 ≤ m ∧ m ≤ 4/3 :=
sorry

end range_of_m_l460_46057


namespace pizza_payment_difference_l460_46041

/-- Pizza sharing problem -/
theorem pizza_payment_difference :
  let total_slices : ℕ := 12
  let plain_pizza_cost : ℚ := 12
  let bacon_cost : ℚ := 3
  let bacon_slices : ℕ := 9
  let dave_plain_slices : ℕ := 1
  let dave_total_slices : ℕ := bacon_slices + dave_plain_slices
  let doug_slices : ℕ := total_slices - dave_total_slices
  let total_cost : ℚ := plain_pizza_cost + bacon_cost
  let cost_per_slice : ℚ := total_cost / total_slices
  let dave_payment : ℚ := cost_per_slice * dave_total_slices
  let doug_payment : ℚ := cost_per_slice * doug_slices
  dave_payment - doug_payment = 10 :=
by sorry

end pizza_payment_difference_l460_46041


namespace xyz_value_l460_46000

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19)
  (h3 : x + y + z = 7) : 
  x * y * z = 10 := by
sorry

end xyz_value_l460_46000


namespace probability_of_green_ball_l460_46044

/-- The probability of drawing a green ball from a bag with specified conditions -/
theorem probability_of_green_ball (total : ℕ) (red : ℕ) (blue : ℕ) 
  (h_total : total = 10)
  (h_red : red = 3)
  (h_blue : blue = 2) :
  (total - red - blue) / total = 1 / 2 := by
  sorry

end probability_of_green_ball_l460_46044


namespace expression_value_at_four_l460_46089

theorem expression_value_at_four :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 4 = 6 := by sorry

end expression_value_at_four_l460_46089


namespace total_age_proof_l460_46068

/-- Given three people a, b, and c, where a is two years older than b, 
    b is twice as old as c, and b is 20 years old, 
    prove that the total of their ages is 52 years. -/
theorem total_age_proof (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  b = 20 → 
  a + b + c = 52 := by
  sorry

end total_age_proof_l460_46068


namespace jennifer_spending_l460_46070

theorem jennifer_spending (initial_amount : ℚ) : 
  initial_amount - (initial_amount / 5 + initial_amount / 6 + initial_amount / 2) = 24 →
  initial_amount = 180 := by
sorry

end jennifer_spending_l460_46070


namespace max_sum_of_factors_l460_46085

theorem max_sum_of_factors (heart club : ℕ) : 
  heart * club = 48 → (∀ x y : ℕ, x * y = 48 → heart + club ≥ x + y) → heart + club = 49 := by
  sorry

end max_sum_of_factors_l460_46085


namespace number_of_employees_l460_46008

def gift_cost : ℕ := 100
def boss_contribution : ℕ := 15
def employee_contribution : ℕ := 11

theorem number_of_employees : 
  ∃ (n : ℕ), 
    gift_cost = boss_contribution + 2 * boss_contribution + n * employee_contribution ∧ 
    n = 5 := by
  sorry

end number_of_employees_l460_46008


namespace work_completion_theorem_l460_46028

theorem work_completion_theorem (original_days : ℕ) (reduced_days : ℕ) (additional_men : ℕ) : ∃ (original_men : ℕ), 
  original_days = 10 ∧ 
  reduced_days = 7 ∧ 
  additional_men = 10 ∧
  original_men * original_days = (original_men + additional_men) * reduced_days ∧
  original_men = 24 := by
sorry

end work_completion_theorem_l460_46028


namespace elena_bouquet_petals_l460_46091

/-- Represents the number of flowers of each type in Elena's garden -/
structure FlowerCounts where
  lilies : ℕ
  tulips : ℕ
  roses : ℕ
  daisies : ℕ

/-- Represents the number of petals for each type of flower -/
structure PetalCounts where
  lily_petals : ℕ
  tulip_petals : ℕ
  rose_petals : ℕ
  daisy_petals : ℕ

/-- Calculates the number of flowers to take for the bouquet -/
def bouquet_flowers (garden : FlowerCounts) : FlowerCounts :=
  let min_count := min (garden.lilies / 2) (min (garden.tulips / 2) (min (garden.roses / 2) (garden.daisies / 2)))
  { lilies := min_count
    tulips := min_count
    roses := min_count
    daisies := min_count }

/-- Calculates the total number of petals in the bouquet -/
def total_petals (flowers : FlowerCounts) (petals : PetalCounts) : ℕ :=
  flowers.lilies * petals.lily_petals +
  flowers.tulips * petals.tulip_petals +
  flowers.roses * petals.rose_petals +
  flowers.daisies * petals.daisy_petals

/-- Elena's garden and petal counts -/
def elena_garden : FlowerCounts := { lilies := 8, tulips := 5, roses := 4, daisies := 3 }
def elena_petals : PetalCounts := { lily_petals := 6, tulip_petals := 3, rose_petals := 5, daisy_petals := 12 }

theorem elena_bouquet_petals :
  total_petals (bouquet_flowers elena_garden) elena_petals = 52 := by
  sorry


end elena_bouquet_petals_l460_46091


namespace S_when_m_is_one_l_range_when_m_is_neg_half_m_range_when_l_is_half_l460_46038

-- Define the set S
def S (m l : ℝ) : Set ℝ := {x : ℝ | m ≤ x ∧ x ≤ l}

-- State the condition that if x ∈ S, then x^2 ∈ S
axiom S_closed_square (m l : ℝ) : ∀ x ∈ S m l, x^2 ∈ S m l

-- Theorem 1
theorem S_when_m_is_one (l : ℝ) : 
  S 1 l = {1} := by sorry

-- Theorem 2
theorem l_range_when_m_is_neg_half : 
  ∀ l, S (-1/2) l ≠ ∅ ↔ 1/4 ≤ l ∧ l ≤ 1 := by sorry

-- Theorem 3
theorem m_range_when_l_is_half : 
  ∀ m, S m (1/2) ≠ ∅ ↔ -Real.sqrt 2 / 2 ≤ m ∧ m ≤ 0 := by sorry

end S_when_m_is_one_l_range_when_m_is_neg_half_m_range_when_l_is_half_l460_46038


namespace base_edge_length_is_six_l460_46095

/-- A square pyramid with a hemisphere resting on its base -/
structure PyramidWithHemisphere where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The radius of the hemisphere -/
  radius : ℝ
  /-- The hemisphere is tangent to the other four faces of the pyramid -/
  is_tangent : Bool

/-- The edge length of the base of the pyramid -/
def base_edge_length (p : PyramidWithHemisphere) : ℝ :=
  sorry

/-- Theorem stating the edge length of the base of the pyramid is 6 -/
theorem base_edge_length_is_six (p : PyramidWithHemisphere) 
  (h1 : p.height = 12)
  (h2 : p.radius = 4)
  (h3 : p.is_tangent = true) : 
  base_edge_length p = 6 := by
  sorry

end base_edge_length_is_six_l460_46095


namespace class_size_l460_46065

theorem class_size (female_students : ℕ) (male_students : ℕ) : 
  female_students = 13 → 
  male_students = 3 * female_students → 
  female_students + male_students = 52 := by
sorry

end class_size_l460_46065


namespace original_selling_price_l460_46098

/-- Given an article with a cost price of $15000, prove that the original selling price
    that would result in an 8% profit if discounted by 10% is $18000. -/
theorem original_selling_price (cost_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  cost_price = 15000 →
  discount_rate = 0.1 →
  profit_rate = 0.08 →
  ∃ (selling_price : ℝ),
    selling_price * (1 - discount_rate) = cost_price * (1 + profit_rate) ∧
    selling_price = 18000 := by
  sorry

end original_selling_price_l460_46098


namespace bulbs_needed_l460_46048

/-- Represents the number of bulbs required for each type of ceiling light. -/
structure BulbRequirement where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of each type of ceiling light. -/
structure CeilingLights where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of bulbs needed given the requirements and quantities. -/
def totalBulbs (req : BulbRequirement) (lights : CeilingLights) : Nat :=
  req.small * lights.small + req.medium * lights.medium + req.large * lights.large

/-- The main theorem stating the total number of bulbs needed. -/
theorem bulbs_needed :
  ∀ (req : BulbRequirement) (lights : CeilingLights),
    req.small = 1 ∧ req.medium = 2 ∧ req.large = 3 ∧
    lights.medium = 12 ∧
    lights.large = 2 * lights.medium ∧
    lights.small = lights.medium + 10 →
    totalBulbs req lights = 118 := by
  sorry


end bulbs_needed_l460_46048


namespace magician_marbles_left_l460_46083

theorem magician_marbles_left (red_initial : Nat) (blue_initial : Nat) 
  (red_taken : Nat) (blue_taken_multiplier : Nat) : 
  red_initial = 20 → 
  blue_initial = 30 → 
  red_taken = 3 → 
  blue_taken_multiplier = 4 → 
  (red_initial - red_taken) + (blue_initial - (blue_taken_multiplier * red_taken)) = 35 := by
  sorry

end magician_marbles_left_l460_46083


namespace equation_roots_l460_46051

theorem equation_roots : ∃ (x y : ℝ), x < 0 ∧ y = 0 ∧
  3^x + x^2 + 2*x - 1 = 0 ∧
  3^y + y^2 + 2*y - 1 = 0 ∧
  ∀ (z : ℝ), (3^z + z^2 + 2*z - 1 = 0) → (z = x ∨ z = y) :=
by sorry

end equation_roots_l460_46051


namespace triangle_inequality_l460_46036

/-- Given a triangle ABC with circumradius R, inradius r, and semiperimeter p,
    prove that 16 R r - 5 r^2 ≤ p^2 ≤ 4 R^2 + 4 R r + 3 r^2 --/
theorem triangle_inequality (R r p : ℝ) (hR : R > 0) (hr : r > 0) (hp : p > 0) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 := by
  sorry

end triangle_inequality_l460_46036


namespace existence_of_abc_l460_46093

theorem existence_of_abc (n : ℕ) : ∃ (a b c : ℤ),
  n = Int.gcd a b * (c^2 - a*b) + Int.gcd b c * (a^2 - b*c) + Int.gcd c a * (b^2 - c*a) := by
  sorry

end existence_of_abc_l460_46093


namespace max_min_ratio_l460_46045

/-- The curve on which point P moves --/
def curve (x y : ℝ) : Prop := y = 3 * Real.sqrt (1 - x^2 / 4)

/-- The expression we're maximizing and minimizing --/
def expr (x y : ℝ) : ℝ := 2 * x - y

/-- Theorem stating the ratio of max to min values of the expression --/
theorem max_min_ratio :
  ∃ (max min : ℝ),
    (∀ x y : ℝ, curve x y → expr x y ≤ max) ∧
    (∃ x y : ℝ, curve x y ∧ expr x y = max) ∧
    (∀ x y : ℝ, curve x y → expr x y ≥ min) ∧
    (∃ x y : ℝ, curve x y ∧ expr x y = min) ∧
    max / min = -4 / 5 :=
by sorry

end max_min_ratio_l460_46045


namespace decimal_equivalent_of_one_fourth_cubed_l460_46040

theorem decimal_equivalent_of_one_fourth_cubed : (1 / 4 : ℚ) ^ 3 = 0.015625 := by
  sorry

end decimal_equivalent_of_one_fourth_cubed_l460_46040


namespace binomial_8_5_l460_46046

theorem binomial_8_5 : Nat.choose 8 5 = 56 := by sorry

end binomial_8_5_l460_46046


namespace area_of_special_trapezoid_l460_46023

/-- An isosceles trapezoid with a circle inscribed in it -/
structure InscribedCircleTrapezoid where
  /-- Length of the shorter base -/
  a : ℝ
  /-- Length of the longer base -/
  b : ℝ
  /-- Length of the leg -/
  c : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : c > 0
  /-- A circle can be inscribed in the trapezoid -/
  hasInscribedCircle : a + b = 2 * c

/-- The area of an isosceles trapezoid with bases 2 and 8, in which a circle can be inscribed, is 20 -/
theorem area_of_special_trapezoid :
  ∀ t : InscribedCircleTrapezoid, t.a = 2 ∧ t.b = 8 → 
  (1/2 : ℝ) * (t.a + t.b) * Real.sqrt (t.c^2 - ((t.b - t.a)/2)^2) = 20 := by
  sorry

end area_of_special_trapezoid_l460_46023


namespace inequality_proof_l460_46058

theorem inequality_proof (a b c : ℝ) : 
  a = Real.sin (14 * π / 180) + Real.cos (14 * π / 180) →
  b = 2 * Real.sqrt 2 * Real.sin (30.5 * π / 180) * Real.cos (30.5 * π / 180) →
  c = Real.sqrt 6 / 2 →
  a < c ∧ c < b := by
  sorry

end inequality_proof_l460_46058


namespace sqrt_sum_representation_l460_46026

theorem sqrt_sum_representation : ∃ (a b c : ℕ+),
  (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) = 
   (a.val * Real.sqrt 3 + b.val * Real.sqrt 11) / c.val) ∧
  (∀ (a' b' c' : ℕ+),
    (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) = 
     (a'.val * Real.sqrt 3 + b'.val * Real.sqrt 11) / c'.val) →
    c'.val ≥ c.val) ∧
  a.val = 44 ∧ b.val = 36 ∧ c.val = 33 :=
by sorry

end sqrt_sum_representation_l460_46026


namespace miyeon_gets_48_sheets_l460_46088

/-- The number of sheets Miyeon gets given the conditions of the paper sharing problem -/
def miyeon_sheets (total_sheets : ℕ) (pink_sheets : ℕ) : ℕ :=
  (total_sheets - pink_sheets) / 2 + pink_sheets

/-- Theorem stating that Miyeon gets 48 sheets under the given conditions -/
theorem miyeon_gets_48_sheets :
  miyeon_sheets 85 11 = 48 :=
by
  sorry

end miyeon_gets_48_sheets_l460_46088


namespace simplify_sqrt_expression_l460_46061

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) : 
  Real.sqrt (1 + ((x^6 - 1) / (3 * x^3))^2) = (Real.sqrt (x^12 + 7*x^6 + 1)) / (3 * x^3) :=
by sorry

end simplify_sqrt_expression_l460_46061


namespace investment_condition_l460_46042

/-- Represents the investment scenario with three banks -/
structure InvestmentScenario where
  national_investment : ℝ
  national_rate : ℝ
  a_rate : ℝ
  b_rate : ℝ
  total_rate : ℝ

/-- The given investment scenario -/
def given_scenario : InvestmentScenario :=
  { national_investment := 7500
  , national_rate := 0.09
  , a_rate := 0.12
  , b_rate := 0.14
  , total_rate := 0.11 }

/-- The total annual income from all three banks -/
def total_income (s : InvestmentScenario) (a b : ℝ) : ℝ :=
  s.national_rate * s.national_investment + s.a_rate * a + s.b_rate * b

/-- The total investment across all three banks -/
def total_investment (s : InvestmentScenario) (a b : ℝ) : ℝ :=
  s.national_investment + a + b

/-- The theorem stating the condition for the desired total annual income -/
theorem investment_condition (s : InvestmentScenario) (a b : ℝ) :
  total_income s a b = s.total_rate * total_investment s a b ↔ 0.01 * a + 0.03 * b = 150 :=
by sorry

end investment_condition_l460_46042


namespace sum_difference_equality_l460_46021

theorem sum_difference_equality : 291 + 503 - 91 + 492 - 103 - 392 = 700 := by
  sorry

end sum_difference_equality_l460_46021


namespace tan_thirteen_pi_fourths_l460_46067

theorem tan_thirteen_pi_fourths : Real.tan (13 * π / 4) = 1 := by
  sorry

end tan_thirteen_pi_fourths_l460_46067


namespace quadratic_coefficient_l460_46052

/-- Proves that for a quadratic function y = ax^2 + bx + c with integer coefficients,
    if the vertex is at (2, 5) and the point (3, 8) lies on the parabola, then a = 3. -/
theorem quadratic_coefficient (a b c : ℤ) : 
  (∀ x : ℝ, ∃ y : ℝ, y = a * x^2 + b * x + c) →
  (∃ y : ℝ, 5 = a * 2^2 + b * 2 + c ∧ 5 ≥ y) →
  (8 = a * 3^2 + b * 3 + c) →
  a = 3 := by
sorry

end quadratic_coefficient_l460_46052


namespace quadratic_root_m_value_l460_46012

theorem quadratic_root_m_value :
  ∀ m : ℝ, (1 : ℝ)^2 + m * 1 + 2 = 0 → m = -3 := by
  sorry

end quadratic_root_m_value_l460_46012


namespace trick_deck_cost_l460_46034

theorem trick_deck_cost (frank_decks : ℕ) (friend_decks : ℕ) (total_spent : ℕ) :
  frank_decks = 3 →
  friend_decks = 2 →
  total_spent = 35 →
  ∃ (cost : ℕ), frank_decks * cost + friend_decks * cost = total_spent ∧ cost = 7 := by
  sorry

end trick_deck_cost_l460_46034


namespace cone_volume_divided_by_pi_l460_46060

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20, when divided by π, equals 375√7 -/
theorem cone_volume_divided_by_pi (r : Real) (h : Real) :
  r = 15 →
  h = 5 * Real.sqrt 7 →
  (1 / 3 * π * r^2 * h) / π = 375 * Real.sqrt 7 := by
sorry

end cone_volume_divided_by_pi_l460_46060


namespace sqrt_x_minus_9_real_l460_46080

theorem sqrt_x_minus_9_real (x : ℝ) : (∃ y : ℝ, y^2 = x - 9) ↔ x ≥ 9 := by sorry

end sqrt_x_minus_9_real_l460_46080


namespace rectangle_area_change_l460_46020

theorem rectangle_area_change (L W : ℝ) (h1 : L * W = 600) : 
  (0.8 * L) * (1.15 * W) = 552 := by
  sorry

end rectangle_area_change_l460_46020


namespace parabola_equation_l460_46054

/-- Represents a parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  a : ℝ
  eq : ∀ x y : ℝ, y^2 = a * x

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ∀ x y : ℝ, y = m * x + b

/-- The chord length of a parabola intercepted by a line -/
def chordLength (p : Parabola) (l : Line) : ℝ := sorry

theorem parabola_equation (p : Parabola) (l : Line) :
  l.m = 2 ∧ l.b = -4 ∧ chordLength p l = 3 * Real.sqrt 5 →
  p.a = 4 ∨ p.a = -36 := by sorry

end parabola_equation_l460_46054


namespace sufficient_but_not_necessary_l460_46074

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {-1, a^2}
def B : Set ℝ := {2, 4}

-- Define the property we want to prove
def property (a : ℝ) : Prop := A a ∩ B = {4}

-- Theorem statement
theorem sufficient_but_not_necessary :
  (∀ a : ℝ, a = -2 → property a) ∧
  ¬(∀ a : ℝ, property a → a = -2) :=
sorry

end sufficient_but_not_necessary_l460_46074


namespace break_room_capacity_l460_46056

/-- The number of people that can be seated at each table -/
def people_per_table : ℕ := 8

/-- The number of tables in the break room -/
def number_of_tables : ℕ := 4

/-- The total number of people that can be seated in the break room -/
def total_seating_capacity : ℕ := people_per_table * number_of_tables

theorem break_room_capacity : total_seating_capacity = 32 := by
  sorry

end break_room_capacity_l460_46056


namespace lower_interest_rate_l460_46022

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem lower_interest_rate 
  (principal : ℚ) 
  (time : ℚ) 
  (higher_rate : ℚ) 
  (interest_difference : ℚ) : 
  principal = 5000 → 
  time = 2 → 
  higher_rate = 18 → 
  interest_difference = 600 → 
  ∃ (lower_rate : ℚ), 
    simpleInterest principal higher_rate time - simpleInterest principal lower_rate time = interest_difference ∧ 
    lower_rate = 12 := by
  sorry

end lower_interest_rate_l460_46022


namespace perfect_square_trinomial_factorization_l460_46047

theorem perfect_square_trinomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 = (x + 3)^2 := by
  sorry

end perfect_square_trinomial_factorization_l460_46047


namespace relationship_abc_l460_46097

theorem relationship_abc (a b c : ℝ) 
  (ha : a = 1 / 2022)
  (hb : b = Real.exp (-2021 / 2022))
  (hc : c = Real.log (2023 / 2022)) :
  c < a ∧ a < b := by
  sorry

end relationship_abc_l460_46097


namespace no_linear_function_satisfies_inequality_l460_46049

theorem no_linear_function_satisfies_inequality :
  ¬ ∃ (a b : ℝ), ∀ x ∈ Set.Icc 0 (2 * Real.pi),
    (a * x + b)^2 - Real.cos x * (a * x + b) < (1/4) * Real.sin x^2 := by
  sorry

end no_linear_function_satisfies_inequality_l460_46049


namespace C_sufficient_for_A_l460_46079

-- Define propositions A, B, and C
variable (A B C : Prop)

-- Define the conditions
variable (h1 : A ↔ B)
variable (h2 : C → B)
variable (h3 : ¬(B → C))

-- Theorem statement
theorem C_sufficient_for_A : C → A := by
  sorry

end C_sufficient_for_A_l460_46079


namespace sum_of_base9_series_l460_46039

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 9 -/
def base10ToBase9 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of an arithmetic series in base 10 -/
def arithmeticSeriesSum (n : ℕ) (a1 : ℕ) (an : ℕ) : ℕ := sorry

theorem sum_of_base9_series :
  let n : ℕ := 36
  let a1 : ℕ := base9ToBase10 1
  let an : ℕ := base9ToBase10 36
  let sum : ℕ := arithmeticSeriesSum n a1 an
  base10ToBase9 sum = 750 := by sorry

end sum_of_base9_series_l460_46039


namespace inverse_function_property_l460_46066

theorem inverse_function_property (f : ℝ → ℝ) (hf : Function.Bijective f) 
  (h : ∀ x : ℝ, f x + f (1 - x) = 2) :
  ∀ x : ℝ, Function.invFun f (x - 2) + Function.invFun f (4 - x) = 1 := by
  sorry

end inverse_function_property_l460_46066


namespace smallest_n_for_sqrt_difference_l460_46018

theorem smallest_n_for_sqrt_difference (n : ℕ) : 
  (n > 0) → 
  (∀ m : ℕ, m > 0 → m < 626 → Real.sqrt m - Real.sqrt (m - 1) ≥ 0.02) → 
  (Real.sqrt 626 - Real.sqrt 625 < 0.02) → 
  (626 = n) :=
sorry

end smallest_n_for_sqrt_difference_l460_46018


namespace journey_distance_l460_46029

theorem journey_distance (train_fraction bus_fraction : ℚ) (walk_distance : ℝ) 
  (h1 : train_fraction = 3/5)
  (h2 : bus_fraction = 7/20)
  (h3 : walk_distance = 6.5)
  (h4 : train_fraction + bus_fraction + (walk_distance / total_distance) = 1) :
  total_distance = 130 :=
by
  sorry

#check journey_distance

end journey_distance_l460_46029


namespace min_variance_sum_l460_46053

theorem min_variance_sum (a b c : ℕ) : 
  70 ≤ a ∧ a < 80 →
  80 ≤ b ∧ b < 90 →
  90 ≤ c ∧ c ≤ 100 →
  let variance := (a - (a + b + c) / 3)^2 + (b - (a + b + c) / 3)^2 + (c - (a + b + c) / 3)^2
  (∀ a' b' c' : ℕ, 
    70 ≤ a' ∧ a' < 80 →
    80 ≤ b' ∧ b' < 90 →
    90 ≤ c' ∧ c' ≤ 100 →
    variance ≤ (a' - (a' + b' + c') / 3)^2 + (b' - (a' + b' + c') / 3)^2 + (c' - (a' + b' + c') / 3)^2) →
  a + b + c = 253 ∨ a + b + c = 254 :=
sorry

end min_variance_sum_l460_46053


namespace fraction_to_decimal_l460_46075

theorem fraction_to_decimal :
  (5 : ℚ) / 16 = (3125 : ℚ) / 10000 :=
by sorry

end fraction_to_decimal_l460_46075


namespace quadratic_inequality_empty_solution_set_l460_46035

theorem quadratic_inequality_empty_solution_set (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 ≥ 0) → k ∈ Set.Icc (-2) 2 := by
  sorry

end quadratic_inequality_empty_solution_set_l460_46035


namespace cricket_team_size_l460_46010

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  n : ℕ  -- number of team members
  captain_age : ℕ
  wicket_keeper_age : ℕ
  team_avg_age : ℚ
  remaining_avg_age : ℚ

/-- The cricket team satisfies the given conditions -/
def satisfies_conditions (team : CricketTeam) : Prop :=
  team.captain_age = 27 ∧
  team.wicket_keeper_age = team.captain_age + 3 ∧
  team.team_avg_age = 24 ∧
  team.remaining_avg_age = team.team_avg_age - 1 ∧
  team.n * team.team_avg_age = team.captain_age + team.wicket_keeper_age + (team.n - 2) * team.remaining_avg_age

/-- The number of members in the cricket team that satisfies the conditions is 11 -/
theorem cricket_team_size :
  ∃ (team : CricketTeam), satisfies_conditions team ∧ team.n = 11 :=
by
  sorry

end cricket_team_size_l460_46010


namespace volleyball_tournament_winner_l460_46002

/-- Represents a volleyball tournament -/
structure VolleyballTournament where
  /-- The number of teams in the tournament -/
  num_teams : ℕ
  /-- The number of games each team plays -/
  games_per_team : ℕ
  /-- The total number of games played in the tournament -/
  total_games : ℕ
  /-- There are no draws in the tournament -/
  no_draws : Bool

/-- Theorem stating that in a volleyball tournament with 6 teams, 
    where each team plays against every other team once and there are no draws, 
    at least one team must win 3 or more games -/
theorem volleyball_tournament_winner (t : VolleyballTournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.games_per_team = 5)
  (h3 : t.total_games = t.num_teams * t.games_per_team / 2)
  (h4 : t.no_draws = true) :
  ∃ (team : ℕ), team ≤ t.num_teams ∧ (∃ (wins : ℕ), wins ≥ 3) :=
sorry

end volleyball_tournament_winner_l460_46002


namespace circumscribed_sphere_area_folded_equilateral_triangle_l460_46024

/-- The surface area of the circumscribed sphere of a tetrahedron formed by folding an equilateral triangle --/
theorem circumscribed_sphere_area_folded_equilateral_triangle :
  let side_length : ℝ := 2
  let height : ℝ := Real.sqrt 3
  let tetrahedron_edge1 : ℝ := 1
  let tetrahedron_edge2 : ℝ := 1
  let tetrahedron_edge3 : ℝ := height
  let sphere_radius : ℝ := Real.sqrt 5 / 2
  let sphere_surface_area : ℝ := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 5 * Real.pi :=
by
  sorry


end circumscribed_sphere_area_folded_equilateral_triangle_l460_46024


namespace probability_independent_of_shape_l460_46025

/-- A geometric model related to area -/
structure GeometricModel where
  area : ℝ
  shape : Type

/-- The probability of a geometric model -/
def probability (model : GeometricModel) : ℝ := sorry

theorem probability_independent_of_shape (model1 model2 : GeometricModel) 
  (h : model1.area = model2.area) : 
  probability model1 = probability model2 := by sorry

end probability_independent_of_shape_l460_46025


namespace largest_square_and_rectangle_in_right_triangle_l460_46081

/-- Given a right triangle ABC with legs AC = a and CB = b, prove:
    1. The side length of the largest square (with vertex C) that lies entirely within the triangle ABC is ab/(a+b)
    2. The dimensions of the largest rectangle (with vertex C) that lies entirely within the triangle ABC are a/2 and b/2 -/
theorem largest_square_and_rectangle_in_right_triangle 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let square_side := a * b / (a + b)
  let rectangle_width := a / 2
  let rectangle_height := b / 2
  (∀ s, s > 0 → s * s ≤ square_side * square_side) ∧
  (∀ w h, w > 0 → h > 0 → w * h ≤ rectangle_width * rectangle_height) := by
  sorry

end largest_square_and_rectangle_in_right_triangle_l460_46081


namespace right_triangle_inequality_l460_46082

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (right_triangle : a^2 + b^2 = c^2) (b_larger : b > a) (tan_condition : b/a < 2) :
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) > 4/9 := by
  sorry

end right_triangle_inequality_l460_46082


namespace cube_root_of_product_l460_46005

theorem cube_root_of_product (a b c : ℕ) : 
  (2^9 * 3^6 * 7^3 : ℝ)^(1/3) = 504 := by
  sorry

end cube_root_of_product_l460_46005


namespace male_democrat_ratio_l460_46069

theorem male_democrat_ratio (total_participants : ℕ) 
  (female_democrats : ℕ) (h1 : total_participants = 840) 
  (h2 : female_democrats = 140) 
  (h3 : female_democrats * 2 ≤ total_participants) : 
  (total_participants / 3 - female_democrats) * 4 = 
  (total_participants - female_democrats * 2) := by
  sorry

#check male_democrat_ratio

end male_democrat_ratio_l460_46069


namespace sum_of_binary_digits_222_l460_46050

/-- The sum of the digits in the binary representation of a natural number -/
def sum_of_binary_digits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- Theorem: The sum of the digits in the binary representation of 222 is 6 -/
theorem sum_of_binary_digits_222 :
  sum_of_binary_digits 222 = 6 := by
  sorry

#eval sum_of_binary_digits 222  -- This should output 6

end sum_of_binary_digits_222_l460_46050


namespace sams_recycling_cans_l460_46086

theorem sams_recycling_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : 
  saturday_bags = 4 → sunday_bags = 3 → cans_per_bag = 6 →
  (saturday_bags + sunday_bags) * cans_per_bag = 42 := by
  sorry

end sams_recycling_cans_l460_46086


namespace even_function_implies_a_equals_one_l460_46037

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |a * x - 1|

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem even_function_implies_a_equals_one :
  ∀ a : ℝ, is_even (f a) → a = 1 := by
sorry

end even_function_implies_a_equals_one_l460_46037


namespace negative_difference_l460_46014

theorem negative_difference (P Q R S T : ℝ) 
  (h1 : P < Q) (h2 : Q < R) (h3 : R < S) (h4 : S < T) 
  (h5 : P ≠ 0) (h6 : Q ≠ 0) (h7 : R ≠ 0) (h8 : S ≠ 0) (h9 : T ≠ 0) : 
  P - Q < 0 := by
  sorry

end negative_difference_l460_46014


namespace remainder_two_pow_33_mod_9_l460_46016

theorem remainder_two_pow_33_mod_9 : 2^33 % 9 = 8 := by sorry

end remainder_two_pow_33_mod_9_l460_46016


namespace slope_product_on_hyperbola_l460_46076

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

theorem slope_product_on_hyperbola 
  (M N P : ℝ × ℝ) 
  (hM : hyperbola M.1 M.2) 
  (hN : hyperbola N.1 N.2) 
  (hP : hyperbola P.1 P.2) 
  (hMN : N = (-M.1, -M.2)) :
  let k_PM := (P.2 - M.2) / (P.1 - M.1)
  let k_PN := (P.2 - N.2) / (P.1 - N.1)
  k_PM * k_PN = (Real.sqrt 5 + 1) / 2 := by
  sorry

end slope_product_on_hyperbola_l460_46076


namespace alex_walking_distance_l460_46063

/-- Represents the bike trip with given conditions -/
structure BikeTrip where
  total_distance : ℝ
  flat_time : ℝ
  flat_speed : ℝ
  uphill_time : ℝ
  uphill_speed : ℝ
  downhill_time : ℝ
  downhill_speed : ℝ

/-- Calculates the distance walked given a BikeTrip -/
def distance_walked (trip : BikeTrip) : ℝ :=
  trip.total_distance - (
    trip.flat_time * trip.flat_speed +
    trip.uphill_time * trip.uphill_speed +
    trip.downhill_time * trip.downhill_speed
  )

/-- Proves that Alex walked 8 miles given the conditions of the problem -/
theorem alex_walking_distance :
  let trip : BikeTrip := {
    total_distance := 164,
    flat_time := 4.5,
    flat_speed := 20,
    uphill_time := 2.5,
    uphill_speed := 12,
    downhill_time := 1.5,
    downhill_speed := 24
  }
  distance_walked trip = 8 := by
  sorry

end alex_walking_distance_l460_46063


namespace part1_part2_l460_46078

-- Define the quadratic inequality
def quadratic_inequality (a b x : ℝ) : Prop := a * x^2 + b * x - 1 ≥ 0

-- Part 1
theorem part1 (a b : ℝ) :
  (∀ x, quadratic_inequality a b x ↔ (3 ≤ x ∧ x ≤ 4)) →
  a + b = 1/2 := by sorry

-- Part 2
theorem part2 (b : ℝ) :
  let solution_set := {x : ℝ | quadratic_inequality (-1) b x}
  if -2 < b ∧ b < 2 then
    solution_set = ∅
  else if b = -2 then
    solution_set = {-1}
  else if b = 2 then
    solution_set = {1}
  else
    ∃ (l u : ℝ), l = (b - Real.sqrt (b^2 - 4)) / 2 ∧
                 u = (b + Real.sqrt (b^2 - 4)) / 2 ∧
                 solution_set = {x | l ≤ x ∧ x ≤ u} := by sorry

end part1_part2_l460_46078


namespace union_equality_implies_range_l460_46030

def A : Set ℝ := {x | |x| > 1}
def B (a : ℝ) : Set ℝ := {x | x < a}

theorem union_equality_implies_range (a : ℝ) : A ∪ B a = A → a ≤ -1 := by
  sorry

end union_equality_implies_range_l460_46030


namespace quadratic_two_distinct_zeros_l460_46073

/-- A quadratic function of the form y = ax^2 + 4x - 2 has two distinct zeros if and only if a > -2 and a ≠ 0 -/
theorem quadratic_two_distinct_zeros (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + 4 * x₁ - 2 = 0 ∧ a * x₂^2 + 4 * x₂ - 2 = 0) ↔
  (a > -2 ∧ a ≠ 0) :=
by sorry

end quadratic_two_distinct_zeros_l460_46073


namespace max_volume_at_one_sixth_l460_46031

/-- The volume of an open-topped box made from a square sheet of cardboard --/
def boxVolume (a x : ℝ) : ℝ := (a - 2*x)^2 * x

/-- The theorem stating that the volume is maximized when the cutout side length is a/6 --/
theorem max_volume_at_one_sixth (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ x < a/2 ∧
  ∀ (y : ℝ), y > 0 → y < a/2 → boxVolume a x ≥ boxVolume a y ∧
  x = a/6 :=
sorry

end max_volume_at_one_sixth_l460_46031


namespace arithmetic_sequence_sufficient_not_necessary_l460_46006

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- Given sequences a and b with the relation b_n = a_n + a_{n+1} -/
def sequence_relation (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a n + a (n + 1)

/-- Theorem stating that {a_n} being arithmetic is sufficient but not necessary for {b_n} to be arithmetic -/
theorem arithmetic_sequence_sufficient_not_necessary
  (a b : ℕ → ℝ) (h : sequence_relation a b) :
  (is_arithmetic_sequence a → is_arithmetic_sequence b) ∧
  ¬(is_arithmetic_sequence b → is_arithmetic_sequence a) := by
  sorry


end arithmetic_sequence_sufficient_not_necessary_l460_46006


namespace median_mode_of_scores_l460_46099

def scores : List ℕ := [7, 10, 9, 8, 9, 9, 8]

def median (l : List ℕ) : ℚ := sorry

def mode (l : List ℕ) : ℕ := sorry

theorem median_mode_of_scores :
  median scores = 9 ∧ mode scores = 9 := by sorry

end median_mode_of_scores_l460_46099


namespace longest_crafting_pattern_length_l460_46077

/-- Represents the lengths of ribbons in inches -/
structure RibbonLengths where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ
  purple : ℕ

/-- Calculates the remaining lengths of ribbons -/
def remainingLengths (initial used : RibbonLengths) : RibbonLengths :=
  { red := initial.red - used.red,
    blue := initial.blue - used.blue,
    green := initial.green - used.green,
    yellow := initial.yellow - used.yellow,
    purple := initial.purple - used.purple }

/-- Finds the minimum length among all ribbon colors -/
def minLength (lengths : RibbonLengths) : ℕ :=
  min lengths.red (min lengths.blue (min lengths.green (min lengths.yellow lengths.purple)))

/-- The initial lengths of ribbons -/
def initialLengths : RibbonLengths :=
  { red := 84, blue := 96, green := 112, yellow := 54, purple := 120 }

/-- The used lengths of ribbons -/
def usedLengths : RibbonLengths :=
  { red := 46, blue := 58, green := 72, yellow := 30, purple := 90 }

theorem longest_crafting_pattern_length :
  minLength (remainingLengths initialLengths usedLengths) = 24 := by
  sorry

#eval minLength (remainingLengths initialLengths usedLengths)

end longest_crafting_pattern_length_l460_46077


namespace a_plus_b_equals_seven_thirds_l460_46072

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -3 * x + 2

-- State the theorem
theorem a_plus_b_equals_seven_thirds 
  (a b : ℝ) 
  (h : ∀ x, g (f a b x) = -2 * x - 3) : 
  a + b = 7/3 := by
  sorry

end a_plus_b_equals_seven_thirds_l460_46072


namespace triangle_cosine_l460_46092

theorem triangle_cosine (A B C : Real) :
  -- Triangle conditions
  A + B + C = Real.pi →
  -- Given conditions
  Real.sin A = 3 / 5 →
  Real.cos B = 5 / 13 →
  -- Conclusion
  Real.cos C = 16 / 65 := by
  sorry

end triangle_cosine_l460_46092


namespace max_brand_A_is_15_l460_46027

/-- The price difference between brand A and brand B soccer balls -/
def price_difference : ℕ := 10

/-- The number of brand A soccer balls in the initial purchase -/
def initial_brand_A : ℕ := 20

/-- The number of brand B soccer balls in the initial purchase -/
def initial_brand_B : ℕ := 15

/-- The total cost of the initial purchase -/
def initial_total_cost : ℕ := 3350

/-- The total number of soccer balls to be purchased -/
def total_balls : ℕ := 50

/-- The maximum total cost for the new purchase -/
def max_total_cost : ℕ := 4650

/-- The price of a brand A soccer ball -/
def price_A : ℕ := initial_total_cost / (initial_brand_A + initial_brand_B)

/-- The price of a brand B soccer ball -/
def price_B : ℕ := price_A - price_difference

/-- The maximum number of brand A soccer balls that can be purchased -/
def max_brand_A : ℕ := (max_total_cost - price_B * total_balls) / (price_A - price_B)

theorem max_brand_A_is_15 : max_brand_A = 15 := by
  sorry

end max_brand_A_is_15_l460_46027


namespace student_failed_marks_l460_46011

def total_marks : ℕ := 400
def passing_percentage : ℚ := 45 / 100
def obtained_marks : ℕ := 150

theorem student_failed_marks :
  (total_marks * passing_percentage).floor - obtained_marks = 30 := by
  sorry

end student_failed_marks_l460_46011


namespace robbie_rice_solution_l460_46003

/-- Robbie's daily rice consumption and fat intake --/
def robbie_rice_problem (x : ℝ) : Prop :=
  let morning_rice := x
  let afternoon_rice := 2
  let evening_rice := 5
  let fat_per_cup := 10
  let weekly_fat := 700
  let daily_rice := morning_rice + afternoon_rice + evening_rice
  let daily_fat := daily_rice * fat_per_cup
  daily_fat * 7 = weekly_fat

/-- The solution to Robbie's rice consumption problem --/
theorem robbie_rice_solution :
  ∃ x : ℝ, robbie_rice_problem x ∧ x = 3 :=
sorry

end robbie_rice_solution_l460_46003


namespace kimberly_skittles_l460_46064

/-- The number of Skittles Kimberly bought -/
def skittles_bought (initial : ℕ) (final : ℕ) : ℕ := final - initial

/-- Proof that Kimberly bought 7 Skittles -/
theorem kimberly_skittles : skittles_bought 5 12 = 7 := by
  sorry

end kimberly_skittles_l460_46064


namespace sum_odd_product_even_l460_46087

theorem sum_odd_product_even (a b : ℤ) : 
  Odd (a + b) → Even (a * b) := by
  sorry

end sum_odd_product_even_l460_46087


namespace polynomial_factorization_l460_46019

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 1) * (x^2 + 6*x + 37) := by
  sorry

end polynomial_factorization_l460_46019


namespace uninsured_part_time_percentage_l460_46013

/-- Represents the survey data and calculates the percentage of uninsured part-time employees -/
def survey_data (total : ℕ) (uninsured : ℕ) (part_time : ℕ) (neither_prob : ℚ) : ℚ :=
  let uninsured_part_time := total - (neither_prob * total).num - uninsured - part_time
  (uninsured_part_time / uninsured) * 100

/-- Theorem stating that given the survey conditions, the percentage of uninsured employees
    who work part-time is approximately 12.5% -/
theorem uninsured_part_time_percentage :
  let result := survey_data 330 104 54 (559606060606060606 / 1000000000000000000)
  ∃ (ε : ℚ), abs (result - 125/10) < ε ∧ ε < 1/10 := by
  sorry

end uninsured_part_time_percentage_l460_46013


namespace externally_tangent_circles_distance_l460_46090

/-- The distance between the centers of two externally tangent circles
    is equal to the sum of their radii -/
theorem externally_tangent_circles_distance
  (r₁ r₂ d : ℝ) 
  (h₁ : r₁ = 2)
  (h₂ : r₂ = 3)
  (h_tangent : d = r₁ + r₂) :
  d = 5 := by sorry

end externally_tangent_circles_distance_l460_46090


namespace min_students_with_blue_eyes_and_backpack_l460_46017

theorem min_students_with_blue_eyes_and_backpack
  (total_students : ℕ)
  (blue_eyes : ℕ)
  (backpack : ℕ)
  (h1 : total_students = 25)
  (h2 : blue_eyes = 15)
  (h3 : backpack = 18)
  : ∃ (both : ℕ), both ≥ 7 ∧ both ≤ min blue_eyes backpack :=
by
  sorry

end min_students_with_blue_eyes_and_backpack_l460_46017


namespace percentage_sum_l460_46004

theorem percentage_sum (P Q R x y : ℝ) 
  (h_pos_P : P > 0) (h_pos_Q : Q > 0) (h_pos_R : R > 0)
  (h_PQ : P = (1 + x / 100) * Q)
  (h_QR : Q = (1 + y / 100) * R)
  (h_PR : P = 2.4 * R) : 
  x + y = 140 := by sorry

end percentage_sum_l460_46004


namespace gcd_370_1332_l460_46096

theorem gcd_370_1332 : Nat.gcd 370 1332 = 74 := by
  sorry

end gcd_370_1332_l460_46096


namespace polynomial_real_root_l460_46071

-- Define the polynomial
def P (b x : ℝ) : ℝ := x^4 + b*x^3 - 3*x^2 + b*x + 1

-- State the theorem
theorem polynomial_real_root (b : ℝ) :
  (∃ x : ℝ, P b x = 0) ↔ b ≤ -1/2 := by sorry

end polynomial_real_root_l460_46071


namespace factor_polynomial_l460_46033

theorem factor_polynomial (x : ℝ) : 
  x^2 + 6*x + 9 - 16*x^4 = (-4*x^2 + 2*x + 3)*(4*x^2 + 2*x + 3) := by
  sorry

end factor_polynomial_l460_46033


namespace tan_sin_cos_relation_l460_46094

theorem tan_sin_cos_relation (α : Real) (h : Real.tan α = -3) :
  (Real.sin α = 3 * Real.sqrt 10 / 10 ∨ Real.sin α = -3 * Real.sqrt 10 / 10) ∧
  (Real.cos α = Real.sqrt 10 / 10 ∨ Real.cos α = -Real.sqrt 10 / 10) :=
by sorry

end tan_sin_cos_relation_l460_46094


namespace final_diaries_count_l460_46007

def calculate_final_diaries (initial : ℕ) : ℕ :=
  let after_buying := initial + 3 * initial
  let lost := (3 * after_buying) / 5
  after_buying - lost

theorem final_diaries_count : calculate_final_diaries 15 = 24 := by
  sorry

end final_diaries_count_l460_46007


namespace hollow_cube_side_length_l460_46032

/-- The number of cubes required to construct a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ := n^3 - (n-2)^3

/-- Theorem: A hollow cube made of 98 unit cubes has a side length of 5 -/
theorem hollow_cube_side_length :
  ∃ (n : ℕ), n > 0 ∧ hollow_cube_cubes n = 98 → n = 5 := by
sorry

end hollow_cube_side_length_l460_46032


namespace coin_difference_is_ten_l460_46055

def coin_values : List ℕ := [5, 10, 25, 50]
def target_amount : ℕ := 60

def min_coins (values : List ℕ) (target : ℕ) : ℕ := sorry
def max_coins (values : List ℕ) (target : ℕ) : ℕ := sorry

theorem coin_difference_is_ten :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 10 := by sorry

end coin_difference_is_ten_l460_46055
