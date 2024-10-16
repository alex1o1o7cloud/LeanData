import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_l933_93301

theorem circle_radius (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y = 1 ↔ (x + 2)^2 + (y - 1)^2 = r^2) → 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 2*y = 1 ↔ (x + 2)^2 + (y - 1)^2 = r^2 ∧ r = Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l933_93301


namespace NUMINAMATH_CALUDE_volunteer_transfer_l933_93343

theorem volunteer_transfer (initial_group1 initial_group2 : ℕ) 
  (h1 : initial_group1 = 20)
  (h2 : initial_group2 = 26) :
  ∃ x : ℚ, x = 32 / 3 ∧ 
    initial_group1 + x = 2 * (initial_group2 - x) := by
  sorry

end NUMINAMATH_CALUDE_volunteer_transfer_l933_93343


namespace NUMINAMATH_CALUDE_complex_equation_square_sum_l933_93369

theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) : 
  i * i = -1 → (a - i) * i = b - i → a^2 + b^2 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_equation_square_sum_l933_93369


namespace NUMINAMATH_CALUDE_extremum_condition_increasing_interval_two_roots_condition_l933_93316

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + a * x^2 + 6 * x

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 6

theorem extremum_condition (a : ℝ) : f' a 3 = 0 := by sorry

theorem increasing_interval (a : ℝ) :
  (∀ m : ℝ, (∀ x ∈ Set.Ioo m (m + 2), f' a x > 0) ↔ m ∈ Set.Iic 0 ∪ Set.Ici 3) := by sorry

theorem two_roots_condition (a : ℝ) :
  (∀ m : ℝ, (∃ x y : ℝ, x ∈ Set.Icc 1 3 ∧ y ∈ Set.Icc 1 3 ∧ x ≠ y ∧ f a x + m = 0 ∧ f a y + m = 0) ↔
  m ∈ Set.Ioo (-14/3) (-9/2)) := by sorry

end NUMINAMATH_CALUDE_extremum_condition_increasing_interval_two_roots_condition_l933_93316


namespace NUMINAMATH_CALUDE_inequality_proof_l933_93387

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 1) : 
  3 - Real.sqrt 3 + x^2 / y + y^2 / z + z^2 / x ≥ (x + y + z)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l933_93387


namespace NUMINAMATH_CALUDE_speech_competition_proof_l933_93398

def scores : List ℝ := [91, 89, 88, 92, 90]

theorem speech_competition_proof :
  let n : ℕ := 5
  let avg : ℝ := 90
  let variance : ℝ := (1 : ℝ) / n * (scores.map (λ x => (x - avg)^2)).sum
  (scores.sum / n = avg) ∧ (variance = 2) :=
by sorry

end NUMINAMATH_CALUDE_speech_competition_proof_l933_93398


namespace NUMINAMATH_CALUDE_log_product_equation_l933_93311

theorem log_product_equation (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 4 → x = 2401 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equation_l933_93311


namespace NUMINAMATH_CALUDE_air_conditioner_installation_rates_l933_93346

theorem air_conditioner_installation_rates 
  (total_A : ℕ) (total_B : ℕ) (diff : ℕ) :
  total_A = 66 →
  total_B = 60 →
  diff = 2 →
  ∃ (days : ℕ) (rate_A : ℕ) (rate_B : ℕ),
    rate_A = rate_B + diff ∧
    rate_A * days = total_A ∧
    rate_B * days = total_B ∧
    rate_A = 22 ∧
    rate_B = 20 :=
by sorry

end NUMINAMATH_CALUDE_air_conditioner_installation_rates_l933_93346


namespace NUMINAMATH_CALUDE_alcohol_water_ratio_in_combined_mixture_l933_93361

/-- Given two containers A and B with alcohol mixtures, this theorem proves
    the ratio of pure alcohol to water in the combined mixture. -/
theorem alcohol_water_ratio_in_combined_mixture
  (v₁ v₂ m₁ n₁ m₂ n₂ : ℝ)
  (hv₁ : v₁ > 0)
  (hv₂ : v₂ > 0)
  (hm₁ : m₁ > 0)
  (hn₁ : n₁ > 0)
  (hm₂ : m₂ > 0)
  (hn₂ : n₂ > 0) :
  let pure_alcohol_A := v₁ * m₁ / (m₁ + n₁)
  let water_A := v₁ * n₁ / (m₁ + n₁)
  let pure_alcohol_B := v₂ * m₂ / (m₂ + n₂)
  let water_B := v₂ * n₂ / (m₂ + n₂)
  let total_pure_alcohol := pure_alcohol_A + pure_alcohol_B
  let total_water := water_A + water_B
  (total_pure_alcohol / total_water) = 
    (v₁*m₁*m₂ + v₁*m₁*n₂ + v₂*m₁*m₂ + v₂*m₂*n₁) / 
    (v₁*m₂*n₁ + v₁*n₁*n₂ + v₂*m₁*n₂ + v₂*n₁*n₂) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_water_ratio_in_combined_mixture_l933_93361


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l933_93379

/-- The distance between the foci of a hyperbola defined by x^2 - 2xy + y^2 = 2 is 4 -/
theorem hyperbola_foci_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 - 2*x*y + y^2 = 2}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ hyperbola ∧ f₂ ∈ hyperbola ∧
    (∀ p ∈ hyperbola, dist p f₁ - dist p f₂ = 2 ∨ dist p f₂ - dist p f₁ = 2) ∧
    dist f₁ f₂ = 4 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l933_93379


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l933_93383

/-- Given that x^4 varies inversely with the fourth root of w, 
    prove that when x = 6, w = 1/4096, given that x = 3 when w = 16 -/
theorem inverse_variation_problem (x w : ℝ) (k : ℝ) (h1 : x^4 * w^(1/4) = k) 
  (h2 : 3^4 * 16^(1/4) = k) : 
  x = 6 → w = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l933_93383


namespace NUMINAMATH_CALUDE_hanks_pancakes_l933_93332

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered short stack pancakes -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack pancakes -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes Hank needs to make -/
def total_pancakes : ℕ := short_stack_orders * short_stack + big_stack_orders * big_stack

theorem hanks_pancakes : total_pancakes = 57 := by
  sorry

end NUMINAMATH_CALUDE_hanks_pancakes_l933_93332


namespace NUMINAMATH_CALUDE_expression_simplification_l933_93312

theorem expression_simplification (k : ℚ) :
  (6 * k + 12) / 6 = k + 2 ∧
  ∃ (a b : ℤ), k + 2 = a * k + b ∧ a = 1 ∧ b = 2 ∧ a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l933_93312


namespace NUMINAMATH_CALUDE_initial_men_count_l933_93305

/-- The number of men initially colouring the cloth -/
def M : ℕ := sorry

/-- The length of cloth coloured by M men in 2 days -/
def initial_cloth_length : ℝ := 48

/-- The time taken by M men to colour the initial cloth length -/
def initial_time : ℝ := 2

/-- The length of cloth coloured by 8 men in 0.75 days -/
def new_cloth_length : ℝ := 36

/-- The time taken by 8 men to colour the new cloth length -/
def new_time : ℝ := 0.75

/-- The number of men in the new scenario -/
def new_men : ℕ := 8

theorem initial_men_count : M = 4 := by sorry

end NUMINAMATH_CALUDE_initial_men_count_l933_93305


namespace NUMINAMATH_CALUDE_book_price_problem_l933_93310

theorem book_price_problem (n : ℕ) (a : ℕ → ℝ) :
  n = 41 ∧
  a 1 = 7 ∧
  (∀ i, 1 ≤ i ∧ i < n → a (i + 1) = a i + 3) ∧
  a n = a ((n + 1) / 2) + a (((n + 1) / 2) + 1) →
  a ((n + 1) / 2) = 67 := by
sorry

end NUMINAMATH_CALUDE_book_price_problem_l933_93310


namespace NUMINAMATH_CALUDE_cos_two_pi_third_plus_two_alpha_l933_93366

theorem cos_two_pi_third_plus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_pi_third_plus_two_alpha_l933_93366


namespace NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l933_93359

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The perpendicular bisector of a line segment -/
def isPerpBisector (p q : Point) (a b c : ℝ) : Prop :=
  let midpoint : Point := ⟨(p.x + q.x) / 2, (p.y + q.y) / 2⟩
  a * midpoint.x + b * midpoint.y = c ∧
  (q.y - p.y) * a = (q.x - p.x) * b

/-- The theorem stating that b = 6 given the conditions -/
theorem perpendicular_bisector_b_value :
  let p : Point := ⟨0, 0⟩
  let q : Point := ⟨4, 8⟩
  ∀ b : ℝ, isPerpBisector p q 1 1 b → b = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_b_value_l933_93359


namespace NUMINAMATH_CALUDE_negation_of_positive_square_l933_93344

theorem negation_of_positive_square (a : ℝ) :
  ¬(a > 0 → a^2 > 0) ↔ (a ≤ 0 → a^2 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_positive_square_l933_93344


namespace NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l933_93334

theorem modulus_of_complex_reciprocal (z : ℂ) : 
  Complex.abs (1 / (1 + Complex.I * Real.sqrt 3)) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_reciprocal_l933_93334


namespace NUMINAMATH_CALUDE_largest_angle_measure_l933_93350

/-- Represents a pentagon with angles in the ratio 3:3:3:4:5 -/
structure RatioPentagon where
  /-- The common factor for the angle measures -/
  x : ℝ
  /-- The sum of interior angles of a pentagon is 540° -/
  angle_sum : 3*x + 3*x + 3*x + 4*x + 5*x = 540

/-- Theorem: The largest angle in a RatioPentagon is 150° -/
theorem largest_angle_measure (p : RatioPentagon) : 5 * p.x = 150 := by
  sorry

#check largest_angle_measure

end NUMINAMATH_CALUDE_largest_angle_measure_l933_93350


namespace NUMINAMATH_CALUDE_largest_multiple_of_45_with_nine_and_zero_m_div_45_l933_93326

/-- A function that checks if a natural number consists only of digits 9 and 0 -/
def only_nine_and_zero (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 9 ∨ d = 0

/-- The largest positive integer that is a multiple of 45 and consists only of digits 9 and 0 -/
def m : ℕ := 99990

theorem largest_multiple_of_45_with_nine_and_zero :
  m % 45 = 0 ∧
  only_nine_and_zero m ∧
  ∀ n : ℕ, n % 45 = 0 → only_nine_and_zero n → n ≤ m :=
sorry

theorem m_div_45 : m / 45 = 2222 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_45_with_nine_and_zero_m_div_45_l933_93326


namespace NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l933_93391

theorem smallest_number_with_remainder_two : ∃! n : ℕ,
  n > 1 ∧
  (∀ d ∈ ({3, 4, 5, 6, 7} : Set ℕ), n % d = 2) ∧
  (∀ m : ℕ, m > 1 ∧ (∀ d ∈ ({3, 4, 5, 6, 7} : Set ℕ), m % d = 2) → m ≥ n) ∧
  n = 422 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainder_two_l933_93391


namespace NUMINAMATH_CALUDE_jeff_fills_130_boxes_l933_93389

/-- Calculates the number of boxes Jeff can fill with remaining donuts --/
def donut_boxes : ℕ :=
  let total_donuts := 50 * 30
  let jeff_eats := 3 * 30
  let friends_eat := 10 + 12 + 8
  let given_away := 25 + 50
  let unavailable := jeff_eats + friends_eat + given_away
  let remaining := total_donuts - unavailable
  remaining / 10

/-- Theorem stating that Jeff can fill 130 boxes with remaining donuts --/
theorem jeff_fills_130_boxes : donut_boxes = 130 := by
  sorry

end NUMINAMATH_CALUDE_jeff_fills_130_boxes_l933_93389


namespace NUMINAMATH_CALUDE_collins_savings_l933_93357

def cans_per_dollar : ℚ := 4

def cans_at_home : ℕ := 12
def cans_at_grandparents : ℕ := 3 * cans_at_home
def cans_from_neighbor : ℕ := 46
def cans_from_office : ℕ := 250

def total_cans : ℕ := cans_at_home + cans_at_grandparents + cans_from_neighbor + cans_from_office

def total_money : ℚ := (total_cans : ℚ) / cans_per_dollar

def savings_amount : ℚ := total_money / 2

theorem collins_savings : savings_amount = 43 := by sorry

end NUMINAMATH_CALUDE_collins_savings_l933_93357


namespace NUMINAMATH_CALUDE_jenny_sweets_division_l933_93339

theorem jenny_sweets_division :
  ∃ n : ℕ, n ≠ 5 ∧ n ≠ 12 ∧ 30 % n = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_sweets_division_l933_93339


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l933_93308

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_identification :
  ¬(is_pythagorean_triple 1 2 3) ∧
  ¬(is_pythagorean_triple 4 5 6) ∧
  ¬(is_pythagorean_triple 6 8 9) ∧
  is_pythagorean_triple 7 24 25 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l933_93308


namespace NUMINAMATH_CALUDE_range_of_a_l933_93333

noncomputable section

-- Define the piecewise function f
def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

-- Define the property of f being increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem range_of_a (a : ℝ) 
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : is_increasing (f a)) :
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l933_93333


namespace NUMINAMATH_CALUDE_line_x_intercept_l933_93313

/-- The x-intercept of a straight line passing through points (2, -4) and (6, 8) is 10/3 -/
theorem line_x_intercept :
  let p1 : ℝ × ℝ := (2, -4)
  let p2 : ℝ × ℝ := (6, 8)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let x_intercept : ℝ := -b / m
  x_intercept = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_l933_93313


namespace NUMINAMATH_CALUDE_tens_digit_of_13_pow_1987_l933_93354

theorem tens_digit_of_13_pow_1987 : ∃ n : ℕ, 13^1987 ≡ 10 * n + 7 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_13_pow_1987_l933_93354


namespace NUMINAMATH_CALUDE_new_people_weight_sum_l933_93378

/-- Given a group of 8 people with average weight W kg, prove that when two people weighing 68 kg each
    leave and are replaced by two new people, causing the average weight to increase by 5.5 kg,
    the sum of the weights of the two new people is 180 kg. -/
theorem new_people_weight_sum (W : ℝ) : 
  let original_total := 8 * W
  let remaining_total := original_total - 2 * 68
  let new_total := 8 * (W + 5.5)
  new_total - remaining_total = 180 := by
  sorry

/-- The sum of the weights of the two new people is no more than 180 kg. -/
axiom new_people_weight_bound (x y : ℝ) : x + y ≤ 180

/-- Each of the new people weighs more than the original average weight. -/
axiom new_people_weight_lower_bound (x y : ℝ) (W : ℝ) : x > W ∧ y > W

end NUMINAMATH_CALUDE_new_people_weight_sum_l933_93378


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l933_93318

theorem binomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  55 * p^9 * q^2 = 165 * p^8 * q^3 → 
  p = 3/4 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l933_93318


namespace NUMINAMATH_CALUDE_ratio_chain_l933_93363

theorem ratio_chain (a b c d : ℚ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (hcd : c / d = 5 / 7) :
  a / d = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_ratio_chain_l933_93363


namespace NUMINAMATH_CALUDE_team_a_more_uniform_l933_93321

/-- Represents a dance team -/
structure DanceTeam where
  name : String
  mean_height : ℝ
  height_variance : ℝ

/-- Define the concept of height uniformity -/
def more_uniform_heights (team1 team2 : DanceTeam) : Prop :=
  team1.height_variance < team2.height_variance

theorem team_a_more_uniform : 
  ∀ (team_a team_b : DanceTeam),
    team_a.name = "A" →
    team_b.name = "B" →
    team_a.mean_height = 1.65 →
    team_b.mean_height = 1.65 →
    team_a.height_variance = 1.5 →
    team_b.height_variance = 2.4 →
    more_uniform_heights team_a team_b :=
by sorry

end NUMINAMATH_CALUDE_team_a_more_uniform_l933_93321


namespace NUMINAMATH_CALUDE_divisibility_property_l933_93367

theorem divisibility_property (a b : ℕ+) 
  (h : ∀ n : ℕ, n ≥ 1 → (a ^ n : ℕ) ∣ (b ^ (n + 1) : ℕ)) : 
  (a : ℕ) ∣ (b : ℕ) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l933_93367


namespace NUMINAMATH_CALUDE_conditional_probability_in_box_l933_93395

/-- A box containing products of different classes -/
structure Box where
  total : ℕ
  firstClass : ℕ
  secondClass : ℕ

/-- The probability of drawing a first-class product followed by another first-class product -/
def probBothFirstClass (b : Box) : ℚ :=
  (b.firstClass : ℚ) * ((b.firstClass - 1) : ℚ) / ((b.total : ℚ) * ((b.total - 1) : ℚ))

/-- The probability of drawing a first-class product first -/
def probFirstClassFirst (b : Box) : ℚ :=
  (b.firstClass : ℚ) / (b.total : ℚ)

/-- The conditional probability of drawing a first-class product second, given that the first draw was a first-class product -/
def conditionalProbability (b : Box) : ℚ :=
  probBothFirstClass b / probFirstClassFirst b

theorem conditional_probability_in_box (b : Box) 
  (h1 : b.total = 4)
  (h2 : b.firstClass = 3)
  (h3 : b.secondClass = 1)
  (h4 : b.firstClass + b.secondClass = b.total) :
  conditionalProbability b = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_in_box_l933_93395


namespace NUMINAMATH_CALUDE_homework_problem_count_l933_93302

theorem homework_problem_count (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : 
  math_pages = 2 → reading_pages = 4 → problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
sorry

end NUMINAMATH_CALUDE_homework_problem_count_l933_93302


namespace NUMINAMATH_CALUDE_sandy_siding_cost_l933_93340

-- Define the dimensions and costs
def wall_width : ℝ := 10
def wall_height : ℝ := 8
def roof_base : ℝ := 10
def roof_height : ℝ := 6
def siding_section_size : ℝ := 100  -- 10 ft x 10 ft = 100 sq ft
def siding_section_cost : ℝ := 30

-- Theorem to prove
theorem sandy_siding_cost :
  let wall_area := wall_width * wall_height
  let roof_area := roof_base * roof_height
  let total_area := wall_area + roof_area
  let sections_needed := ⌈total_area / siding_section_size⌉
  let total_cost := sections_needed * siding_section_cost
  total_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_sandy_siding_cost_l933_93340


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l933_93372

theorem fraction_sum_equality (n : ℕ) (hn : n > 1) :
  ∃ (i j : ℕ), (1 : ℚ) / n = (1 : ℚ) / i - (1 : ℚ) / (j + 1) :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l933_93372


namespace NUMINAMATH_CALUDE_equation_solution_l933_93386

theorem equation_solution :
  ∃ x : ℚ, (2 * x + 3 * x = 500 - (4 * x + 5 * x) + 20) ∧ (x = 520 / 14) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l933_93386


namespace NUMINAMATH_CALUDE_product_of_roots_roots_product_of_equation_l933_93362

theorem product_of_roots (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  f r₁ = 0 ∧ f r₂ = 0 → r₁ * r₂ = c / a :=
by sorry

theorem roots_product_of_equation :
  let f : ℝ → ℝ := λ x => x^2 + 14*x + 52
  let r₁ := (-14 + Real.sqrt (14^2 - 4*1*52)) / (2*1)
  let r₂ := (-14 - Real.sqrt (14^2 - 4*1*52)) / (2*1)
  f r₁ = 0 ∧ f r₂ = 0 → r₁ * r₂ = 48 :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_roots_product_of_equation_l933_93362


namespace NUMINAMATH_CALUDE_larger_circle_tangent_to_line_and_axes_l933_93303

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

/-- Checks if a circle is tangent to a line ax + by = c -/
def isTangentToLine (circle : Circle) (a b c : ℝ) : Prop :=
  let (x, y) := circle.center
  |a * x + b * y - c| / Real.sqrt (a^2 + b^2) = circle.radius

/-- Checks if a circle is tangent to both coordinate axes -/
def isTangentToAxes (circle : Circle) : Prop :=
  circle.center.1 = circle.radius ∧ circle.center.2 = circle.radius

/-- The theorem to be proved -/
theorem larger_circle_tangent_to_line_and_axes :
  ∃ (circle : Circle),
    circle.center = (5/2, 5/2) ∧
    circle.radius = 5/2 ∧
    isInFirstQuadrant circle.center ∧
    isTangentToLine circle 3 4 5 ∧
    isTangentToAxes circle ∧
    (∀ (other : Circle),
      isInFirstQuadrant other.center →
      isTangentToLine other 3 4 5 →
      isTangentToAxes other →
      other.radius ≤ circle.radius) :=
  sorry

end NUMINAMATH_CALUDE_larger_circle_tangent_to_line_and_axes_l933_93303


namespace NUMINAMATH_CALUDE_banana_group_size_l933_93322

theorem banana_group_size (total_bananas : ℕ) (num_groups : ℕ) (h1 : total_bananas = 180) (h2 : num_groups = 10) :
  total_bananas / num_groups = 18 := by
  sorry

end NUMINAMATH_CALUDE_banana_group_size_l933_93322


namespace NUMINAMATH_CALUDE_lucca_basketball_percentage_proof_l933_93317

/-- The percentage of Lucca's balls that are basketballs -/
def lucca_basketball_percentage : ℝ := 10

theorem lucca_basketball_percentage_proof :
  let lucca_total_balls : ℕ := 100
  let lucien_total_balls : ℕ := 200
  let lucien_basketball_percentage : ℝ := 20
  let total_basketballs : ℕ := 50
  lucca_basketball_percentage = 10 := by
  sorry

end NUMINAMATH_CALUDE_lucca_basketball_percentage_proof_l933_93317


namespace NUMINAMATH_CALUDE_marks_trees_l933_93381

theorem marks_trees (current_trees : ℕ) 
  (h : current_trees + 12 = 25) : current_trees = 13 := by
  sorry

end NUMINAMATH_CALUDE_marks_trees_l933_93381


namespace NUMINAMATH_CALUDE_partnership_profit_share_l933_93356

/-- 
Given:
- A, B, and C are in a partnership
- A invests 3 times as much as B
- B invests two-thirds of what C invests
- The total profit is 4400

Prove that B's share of the profit is 800
-/
theorem partnership_profit_share (c : ℝ) (total_profit : ℝ) 
  (h1 : c > 0)
  (h2 : total_profit = 4400) :
  let b := (2/3) * c
  let a := 3 * b
  let total_investment := a + b + c
  b / total_investment * total_profit = 800 := by
sorry

end NUMINAMATH_CALUDE_partnership_profit_share_l933_93356


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l933_93384

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l933_93384


namespace NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l933_93399

/-- 
Given an isosceles triangle with side lengths m, n, and 4, where m and n are 
roots of x^2 - 6x + k + 2 = 0, prove that k = 7 or k = 6.
-/
theorem isosceles_triangle_quadratic_roots (m n k : ℝ) : 
  (m > 0 ∧ n > 0) →  -- m and n are positive (side lengths)
  (m = n ∨ m = 4 ∨ n = 4) →  -- isosceles condition
  (m ≠ n ∨ m ≠ 4) →  -- not equilateral
  m^2 - 6*m + k + 2 = 0 →  -- m is a root
  n^2 - 6*n + k + 2 = 0 →  -- n is a root
  k = 7 ∨ k = 6 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_quadratic_roots_l933_93399


namespace NUMINAMATH_CALUDE_probability_one_from_a_is_11_21_l933_93348

/-- Represents the number of factories in each area -/
structure FactoryCounts where
  areaA : Nat
  areaB : Nat
  areaC : Nat

/-- Represents the number of factories selected from each area -/
structure SelectedCounts where
  areaA : Nat
  areaB : Nat
  areaC : Nat

/-- Calculates the probability of selecting at least one factory from area A
    when choosing 2 out of 7 stratified sampled factories -/
def probabilityAtLeastOneFromA (counts : FactoryCounts) (selected : SelectedCounts) : Rat :=
  sorry

/-- The main theorem stating the probability is 11/21 given the specific conditions -/
theorem probability_one_from_a_is_11_21 :
  let counts : FactoryCounts := ⟨18, 27, 18⟩
  let selected : SelectedCounts := ⟨2, 3, 2⟩
  probabilityAtLeastOneFromA counts selected = 11 / 21 := by sorry

end NUMINAMATH_CALUDE_probability_one_from_a_is_11_21_l933_93348


namespace NUMINAMATH_CALUDE_remainder_theorem_l933_93347

theorem remainder_theorem : (7 * 10^15 + 3^15) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l933_93347


namespace NUMINAMATH_CALUDE_sqrt_10_plus_2_range_l933_93300

theorem sqrt_10_plus_2_range : 5 < Real.sqrt 10 + 2 ∧ Real.sqrt 10 + 2 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_10_plus_2_range_l933_93300


namespace NUMINAMATH_CALUDE_brandons_cash_sales_l933_93330

theorem brandons_cash_sales (total_sales : ℝ) (credit_sales_fraction : ℝ) (cash_sales : ℝ) : 
  total_sales = 80 →
  credit_sales_fraction = 2/5 →
  cash_sales = total_sales * (1 - credit_sales_fraction) →
  cash_sales = 48 := by
sorry

end NUMINAMATH_CALUDE_brandons_cash_sales_l933_93330


namespace NUMINAMATH_CALUDE_staff_age_calculation_l933_93341

theorem staff_age_calculation (num_students : ℕ) (student_avg_age : ℕ) (num_staff : ℕ) (age_increase : ℕ) :
  num_students = 50 →
  student_avg_age = 25 →
  num_staff = 5 →
  age_increase = 2 →
  (num_students * student_avg_age + num_staff * ((student_avg_age + age_increase) * (num_students + num_staff) - num_students * student_avg_age)) / num_staff = 235 := by
  sorry

end NUMINAMATH_CALUDE_staff_age_calculation_l933_93341


namespace NUMINAMATH_CALUDE_triangle_count_on_circle_l933_93323

theorem triangle_count_on_circle (n : ℕ) (h : n = 10) : 
  Nat.choose n 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_on_circle_l933_93323


namespace NUMINAMATH_CALUDE_total_water_in_boxes_l933_93329

theorem total_water_in_boxes (num_boxes : ℕ) (bottles_per_box : ℕ) (bottle_capacity : ℚ) (fill_ratio : ℚ) : 
  num_boxes = 10 →
  bottles_per_box = 50 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  (num_boxes * bottles_per_box * bottle_capacity * fill_ratio : ℚ) = 4500 := by
sorry

end NUMINAMATH_CALUDE_total_water_in_boxes_l933_93329


namespace NUMINAMATH_CALUDE_f_960_minus_f_640_l933_93371

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(960) - f(640) = 5/8 -/
theorem f_960_minus_f_640 : f 960 - f 640 = 5/8 := by sorry

end NUMINAMATH_CALUDE_f_960_minus_f_640_l933_93371


namespace NUMINAMATH_CALUDE_train_arrival_time_l933_93331

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem train_arrival_time 
  (departure : Time)
  (journey_duration : Nat)
  (h1 : departure = { hours := 9, minutes := 45 })
  (h2 : journey_duration = 15) :
  addMinutes departure journey_duration = { hours := 10, minutes := 0 } :=
sorry

end NUMINAMATH_CALUDE_train_arrival_time_l933_93331


namespace NUMINAMATH_CALUDE_infinite_triplets_exist_l933_93368

theorem infinite_triplets_exist : 
  ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 0 ∧ a^4 + b^4 + c^4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_infinite_triplets_exist_l933_93368


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l933_93370

theorem difference_of_reciprocals (x y : ℝ) : 
  x = Real.sqrt 5 - 1 → y = Real.sqrt 5 + 1 → 1 / x - 1 / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l933_93370


namespace NUMINAMATH_CALUDE_triangle_sides_proof_l933_93352

theorem triangle_sides_proof (a b c : ℝ) (h : ℝ) (x : ℝ) :
  b - c = 3 →
  h = 10 →
  (a / 2 + 6) - (a / 2 - 6) = 12 →
  a^2 = 427 / 3 ∧
  b = Real.sqrt (427 / 3) + 3 / 2 ∧
  c = Real.sqrt (427 / 3) - 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_proof_l933_93352


namespace NUMINAMATH_CALUDE_interest_rate_is_10_percent_l933_93394

/-- Calculates the simple interest rate given principal, amount, and time. -/
def calculate_interest_rate (principal amount : ℚ) (time : ℕ) : ℚ :=
  ((amount - principal) * 100) / (principal * time)

/-- Theorem: Given the conditions, the interest rate is 10%. -/
theorem interest_rate_is_10_percent :
  let principal : ℚ := 750
  let amount : ℚ := 900
  let time : ℕ := 2
  calculate_interest_rate principal amount time = 10 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_10_percent_l933_93394


namespace NUMINAMATH_CALUDE_new_supervisor_salary_range_l933_93342

theorem new_supervisor_salary_range (
  old_average : ℝ) 
  (old_supervisor_salary : ℝ) 
  (new_average : ℝ) 
  (min_worker_salary : ℝ) 
  (max_worker_salary : ℝ) 
  (min_supervisor_salary : ℝ) 
  (max_supervisor_salary : ℝ) :
  old_average = 430 →
  old_supervisor_salary = 870 →
  new_average = 410 →
  min_worker_salary = 300 →
  max_worker_salary = 500 →
  min_supervisor_salary = 800 →
  max_supervisor_salary = 1100 →
  ∃ (new_supervisor_salary : ℝ),
    min_supervisor_salary ≤ new_supervisor_salary ∧
    new_supervisor_salary ≤ max_supervisor_salary ∧
    (9 * new_average - 8 * old_average + old_supervisor_salary = new_supervisor_salary) :=
by sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_range_l933_93342


namespace NUMINAMATH_CALUDE_water_bottle_cost_l933_93345

def initial_amount : ℕ := 50
def final_amount : ℕ := 44
def num_baguettes : ℕ := 2
def cost_per_baguette : ℕ := 2
def num_water_bottles : ℕ := 2

theorem water_bottle_cost :
  (initial_amount - final_amount - num_baguettes * cost_per_baguette) / num_water_bottles = 1 :=
by sorry

end NUMINAMATH_CALUDE_water_bottle_cost_l933_93345


namespace NUMINAMATH_CALUDE_safe_journey_exists_l933_93364

-- Define the duration of the journey
def road_duration : ℕ := 4
def trail_duration : ℕ := 4

-- Define the eruption patterns
def crater1_cycle : ℕ := 18
def crater2_cycle : ℕ := 10

-- Define the safety condition
def is_safe (t : ℕ) : Prop :=
  (t % crater1_cycle ≠ 0) ∧ 
  ((t % crater2_cycle ≠ 0) → (t < road_duration ∨ t ≥ road_duration + trail_duration))

-- Theorem statement
theorem safe_journey_exists :
  ∃ start : ℕ, 
    (∀ t : ℕ, t ≥ start ∧ t < start + 2 * (road_duration + trail_duration) → is_safe t) :=
sorry

end NUMINAMATH_CALUDE_safe_journey_exists_l933_93364


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l933_93375

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → (∃ k : ℕ, n = k^2) → 2 ∣ n → 3 ∣ n → 5 ∣ n → n ≥ 900 :=
by sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l933_93375


namespace NUMINAMATH_CALUDE_complex_magnitude_proof_l933_93351

theorem complex_magnitude_proof : Complex.abs (3/5 - 5/4 * Complex.I) = Real.sqrt 769 / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_proof_l933_93351


namespace NUMINAMATH_CALUDE_assignment_validity_l933_93307

/-- Represents a variable in a programming language -/
structure Variable where
  name : String

/-- Represents an expression in a programming language -/
inductive Expression
  | Var : Variable → Expression
  | Product : Expression → Expression → Expression
  | Literal : Int → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Expression
  rhs : Expression

/-- Predicate to check if an expression is a single variable -/
def isSingleVariable : Expression → Prop
  | Expression.Var _ => True
  | _ => False

/-- Theorem: An assignment statement is valid if and only if its left-hand side is a single variable -/
theorem assignment_validity (a : Assignment) :
  isSingleVariable a.lhs ↔ True :=
sorry

#check assignment_validity

end NUMINAMATH_CALUDE_assignment_validity_l933_93307


namespace NUMINAMATH_CALUDE_supermarket_eggs_l933_93304

/-- Represents the number of egg cartons in the supermarket -/
def num_cartons : ℕ := 28

/-- Represents the length of the egg array in each carton -/
def carton_length : ℕ := 33

/-- Represents the width of the egg array in each carton -/
def carton_width : ℕ := 4

/-- Calculates the total number of eggs in the supermarket -/
def total_eggs : ℕ := num_cartons * carton_length * carton_width

/-- Theorem stating that the total number of eggs in the supermarket is 3696 -/
theorem supermarket_eggs : total_eggs = 3696 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_eggs_l933_93304


namespace NUMINAMATH_CALUDE_pizza_fraction_l933_93377

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice = 1/3 →
  (whole_slices : ℚ) / total_slices + shared_slice / total_slices = 7/48 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l933_93377


namespace NUMINAMATH_CALUDE_hands_in_peters_class_l933_93397

/-- The number of hands in Peter's class, excluding his own. -/
def handsInClass (totalStudents : ℕ) (handsPerStudent : ℕ) : ℕ :=
  (totalStudents * handsPerStudent) - handsPerStudent

/-- Theorem: The number of hands in Peter's class, excluding his own, is 20. -/
theorem hands_in_peters_class :
  handsInClass 11 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hands_in_peters_class_l933_93397


namespace NUMINAMATH_CALUDE_largest_common_divisor_l933_93393

theorem largest_common_divisor : ∃ (n : ℕ), n = 60 ∧ 
  n ∣ 660 ∧ n < 100 ∧ n ∣ 120 ∧ 
  ∀ (m : ℕ), m ∣ 660 ∧ m < 100 ∧ m ∣ 120 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l933_93393


namespace NUMINAMATH_CALUDE_solve_flower_problem_l933_93336

def flower_problem (minyoung_flowers : ℕ) (ratio : ℕ) : Prop :=
  let yoojung_flowers := minyoung_flowers / ratio
  minyoung_flowers + yoojung_flowers = 30

theorem solve_flower_problem :
  flower_problem 24 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_flower_problem_l933_93336


namespace NUMINAMATH_CALUDE_cole_average_speed_back_home_l933_93385

/-- Proves that Cole's average speed back home was 120 km/h given the conditions of his round trip. -/
theorem cole_average_speed_back_home 
  (speed_to_work : ℝ) 
  (total_time : ℝ) 
  (time_to_work : ℝ) 
  (h1 : speed_to_work = 80) 
  (h2 : total_time = 3) 
  (h3 : time_to_work = 108 / 60) : 
  (speed_to_work * time_to_work) / (total_time - time_to_work) = 120 := by
  sorry

#check cole_average_speed_back_home

end NUMINAMATH_CALUDE_cole_average_speed_back_home_l933_93385


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l933_93392

-- Define the conversion factor
def yards_to_feet : ℝ := 3

-- Define the volume in cubic yards
def cubic_yards : ℝ := 5

-- Theorem to prove
theorem cubic_yards_to_cubic_feet :
  cubic_yards * yards_to_feet^3 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l933_93392


namespace NUMINAMATH_CALUDE_five_digit_four_digit_division_l933_93380

theorem five_digit_four_digit_division (a b : ℕ) : 
  (a * 11111 = 16 * (b * 1111) + (a * 1111 - 16 * (b * 111) + 2000)) →
  (a ≤ 9) →
  (b ≤ 9) →
  (a * 11111 ≥ b * 1111) →
  (a * 1111 ≥ b * 111) →
  (a = 5 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_five_digit_four_digit_division_l933_93380


namespace NUMINAMATH_CALUDE_planes_parallel_l933_93360

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (lineParallel : Line → Line → Prop)

-- State the theorem
theorem planes_parallel (α β γ : Plane) (a b : Line) :
  (parallel α γ ∧ parallel β γ) ∧
  (perpendicular a α ∧ perpendicular b β ∧ lineParallel a b) →
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_l933_93360


namespace NUMINAMATH_CALUDE_diana_age_is_22_l933_93374

-- Define the ages as natural numbers
def anna_age : ℕ := 48

-- Define the relationships between ages
def brianna_age : ℕ := anna_age / 2
def caitlin_age : ℕ := brianna_age - 5
def diana_age : ℕ := caitlin_age + 3

-- Theorem to prove Diana's age
theorem diana_age_is_22 : diana_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_diana_age_is_22_l933_93374


namespace NUMINAMATH_CALUDE_three_config_m_separable_l933_93327

/-- A 3-configuration of a set is m-separable if it can be partitioned into m subsets
    such that no three elements of the configuration are in the same subset. -/
def is_m_separable (A : Set α) (m : ℕ) : Prop :=
  ∃ (f : α → Fin m), ∀ (x y z : α), x ∈ A → y ∈ A → z ∈ A →
    x ≠ y → y ≠ z → x ≠ z → f x ≠ f y ∨ f y ≠ f z ∨ f x ≠ f z

/-- A 3-configuration of a set A is a subset of A with exactly 3 elements. -/
def is_3_configuration (S : Set α) (A : Set α) : Prop :=
  S ⊆ A ∧ S.ncard = 3

theorem three_config_m_separable
  (A : Set α) (n m : ℕ) (h_card : A.ncard = n) (h_m : m ≥ n / 2) :
  ∀ S : Set α, is_3_configuration S A → is_m_separable S m :=
sorry

end NUMINAMATH_CALUDE_three_config_m_separable_l933_93327


namespace NUMINAMATH_CALUDE_root_sum_of_coefficients_l933_93338

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic (p q : ℝ) (x : ℂ) : Prop :=
  x^2 + p * x + q = 0

-- State the theorem
theorem root_sum_of_coefficients (p q : ℝ) :
  quadratic p q (1 + i) → p + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_of_coefficients_l933_93338


namespace NUMINAMATH_CALUDE_selection_options_count_l933_93376

/-- Represents the number of people skilled in the first method -/
def skilled_in_first_method : ℕ := 5

/-- Represents the number of people skilled in the second method -/
def skilled_in_second_method : ℕ := 4

/-- Represents the total number of people -/
def total_people : ℕ := skilled_in_first_method + skilled_in_second_method

/-- Theorem: The number of ways to select one person from the group is equal to the total number of people -/
theorem selection_options_count : 
  (skilled_in_first_method + skilled_in_second_method) = total_people := by
  sorry

end NUMINAMATH_CALUDE_selection_options_count_l933_93376


namespace NUMINAMATH_CALUDE_fraction_sum_l933_93355

theorem fraction_sum (p q : ℚ) (h : p / q = 4 / 5) : 
  1 / 7 + (2 * q - p) / (2 * q + p) = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l933_93355


namespace NUMINAMATH_CALUDE_gcd_168_486_l933_93315

def continuedProportionateReduction (a b : ℕ) : ℕ :=
  if a = 0 then b
  else if b = 0 then a
  else if a ≥ b then continuedProportionateReduction (a - b) b
  else continuedProportionateReduction a (b - a)

theorem gcd_168_486 :
  continuedProportionateReduction 168 486 = 6 ∧ 
  (∀ d : ℕ, d ∣ 168 ∧ d ∣ 486 → d ≤ 6) := by sorry

end NUMINAMATH_CALUDE_gcd_168_486_l933_93315


namespace NUMINAMATH_CALUDE_absolute_difference_of_roots_l933_93319

theorem absolute_difference_of_roots (p q : ℝ) : 
  p^2 - 6*p + 8 = 0 → q^2 - 6*q + 8 = 0 → |p - q| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_roots_l933_93319


namespace NUMINAMATH_CALUDE_angle_measure_l933_93358

-- Define a type for angles
structure Angle where
  degrees : ℕ
  minutes : ℕ

-- Define vertical angles
def vertical_angles (a1 a2 : Angle) : Prop := a1 = a2

-- Define complementary angle
def complementary_angle (a : Angle) : Angle :=
  ⟨90 - a.degrees, 60 - a.minutes⟩

-- Theorem statement
theorem angle_measure :
  ∀ (angle1 angle2 : Angle),
  vertical_angles angle1 angle2 →
  complementary_angle angle1 = ⟨79, 32⟩ →
  angle2 = ⟨100, 28⟩ :=
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l933_93358


namespace NUMINAMATH_CALUDE_no_single_digit_divisor_l933_93309

theorem no_single_digit_divisor (n : ℤ) (d : ℤ) :
  1 < d → d < 10 → ¬(∃ k : ℤ, 2 * n^2 - 31 = d * k) := by
  sorry

end NUMINAMATH_CALUDE_no_single_digit_divisor_l933_93309


namespace NUMINAMATH_CALUDE_triangle_proof_l933_93349

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

theorem triangle_proof (t : Triangle) 
  (h1 : 2 * t.c - t.b = 2 * t.a * Real.cos t.B)
  (h2 : 1/2 * t.b * t.c * Real.sin t.A = 3/2 * Real.sqrt 3)
  (h3 : t.c = Real.sqrt 3) :
  t.A = Real.pi / 3 ∧ t.B = Real.pi / 2 := by
  sorry

#check triangle_proof

end NUMINAMATH_CALUDE_triangle_proof_l933_93349


namespace NUMINAMATH_CALUDE_a_6_equals_12_l933_93365

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_6_equals_12 
  (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 9 = (1/2) * a 12 + 6) : 
  a 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_a_6_equals_12_l933_93365


namespace NUMINAMATH_CALUDE_square_root_equality_l933_93388

theorem square_root_equality (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ Real.sqrt x = a + 2 ∧ Real.sqrt x = 2*a - 5) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equality_l933_93388


namespace NUMINAMATH_CALUDE_convexity_inequality_equality_conditions_l933_93328

theorem convexity_inequality (x y a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : a ≥ 0) 
  (h3 : b ≥ 0) : 
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 := by
  sorry

theorem equality_conditions (x y a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : a ≥ 0) 
  (h3 : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2 ↔ (a = 0 ∨ b = 0 ∨ x = y) := by
  sorry

end NUMINAMATH_CALUDE_convexity_inequality_equality_conditions_l933_93328


namespace NUMINAMATH_CALUDE_green_hat_cost_l933_93335

theorem green_hat_cost (total_hats : ℕ) (green_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) :
  total_hats = 85 →
  green_hats = 20 →
  blue_hat_cost = 6 →
  total_price = 530 →
  (total_hats - green_hats) * blue_hat_cost + green_hats * 7 = total_price :=
by sorry

end NUMINAMATH_CALUDE_green_hat_cost_l933_93335


namespace NUMINAMATH_CALUDE_mark_car_repair_cost_l933_93353

/-- Calculates the total cost of car repair for Mark -/
theorem mark_car_repair_cost :
  let labor_hours : ℝ := 2
  let labor_rate : ℝ := 75
  let part_cost : ℝ := 150
  let cleaning_hours : ℝ := 1
  let cleaning_rate : ℝ := 60
  let labor_discount : ℝ := 0.1
  let tax_rate : ℝ := 0.08

  let labor_cost := labor_hours * labor_rate
  let discounted_labor := labor_cost * (1 - labor_discount)
  let cleaning_cost := cleaning_hours * cleaning_rate
  let subtotal := discounted_labor + part_cost + cleaning_cost
  let total_cost := subtotal * (1 + tax_rate)

  total_cost = 372.60 := by sorry

end NUMINAMATH_CALUDE_mark_car_repair_cost_l933_93353


namespace NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l933_93396

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- We don't need to define the specifics of a regular polygon for this statement
  -- Just the number of sides is sufficient

/-- The number of diagonals that can be drawn from a single vertex in a regular polygon -/
def diagonalsFromVertex (p : RegularPolygon n) : ℕ := n - 3

/-- Theorem: In a regular decagon, 7 diagonals can be drawn from any vertex -/
theorem decagon_diagonals_from_vertex :
  ∀ (p : RegularPolygon 10), diagonalsFromVertex p = 7 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_from_vertex_l933_93396


namespace NUMINAMATH_CALUDE_inequality_proof_l933_93314

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum_inv : 1/a + 1/b = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l933_93314


namespace NUMINAMATH_CALUDE_non_collinear_triples_count_l933_93324

/-- The total number of points -/
def total_points : ℕ := 60

/-- The number of collinear triples -/
def collinear_triples : ℕ := 30

/-- The number of ways to choose three points from the total points -/
def total_triples : ℕ := total_points.choose 3

/-- The number of ways to choose three non-collinear points -/
def non_collinear_triples : ℕ := total_triples - collinear_triples

theorem non_collinear_triples_count : non_collinear_triples = 34190 := by
  sorry

end NUMINAMATH_CALUDE_non_collinear_triples_count_l933_93324


namespace NUMINAMATH_CALUDE_brocard_point_characterization_l933_93306

open Real

/-- Triangle structure with side lengths a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

/-- Point structure with coordinates x, y -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the area of a triangle given side lengths -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the area of a triangle given three points -/
def areaFromPoints (p1 p2 p3 : Point) : ℝ := sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Definition of Brocard point -/
def isBrocardPoint (p : Point) (t : Triangle) : Prop :=
  let s_abc := triangleArea t
  let s_pbc := areaFromPoints p (Point.mk 0 0) (Point.mk t.c 0)
  let s_pca := areaFromPoints p (Point.mk t.c 0) (Point.mk 0 t.a)
  let s_pab := areaFromPoints p (Point.mk 0 0) (Point.mk 0 t.a)
  isInside p t ∧
  (s_pbc / (t.c^2 * t.a^2) = s_pca / (t.a^2 * t.b^2)) ∧
  (s_pca / (t.a^2 * t.b^2) = s_pab / (t.b^2 * t.c^2)) ∧
  (s_pab / (t.b^2 * t.c^2) = s_abc / (t.a^2 * t.b^2 + t.b^2 * t.c^2 + t.c^2 * t.a^2))

/-- Theorem: Characterization of Brocard point -/
theorem brocard_point_characterization (t : Triangle) (p : Point) :
  isBrocardPoint p t ↔
  (let s_abc := triangleArea t
   let s_pbc := areaFromPoints p (Point.mk 0 0) (Point.mk t.c 0)
   let s_pca := areaFromPoints p (Point.mk t.c 0) (Point.mk 0 t.a)
   let s_pab := areaFromPoints p (Point.mk 0 0) (Point.mk 0 t.a)
   isInside p t ∧
   (s_pbc / (t.c^2 * t.a^2) = s_pca / (t.a^2 * t.b^2)) ∧
   (s_pca / (t.a^2 * t.b^2) = s_pab / (t.b^2 * t.c^2)) ∧
   (s_pab / (t.b^2 * t.c^2) = s_abc / (t.a^2 * t.b^2 + t.b^2 * t.c^2 + t.c^2 * t.a^2))) :=
by sorry

end NUMINAMATH_CALUDE_brocard_point_characterization_l933_93306


namespace NUMINAMATH_CALUDE_buy_three_items_ways_l933_93382

/-- The number of headphones available for sale. -/
def headphones : ℕ := 9

/-- The number of computer mice available for sale. -/
def mice : ℕ := 13

/-- The number of keyboards available for sale. -/
def keyboards : ℕ := 5

/-- The number of "keyboard and mouse" sets available. -/
def keyboard_mouse_sets : ℕ := 4

/-- The number of "headphones and mouse" sets available. -/
def headphones_mouse_sets : ℕ := 5

/-- The total number of ways to buy three items: headphones, a keyboard, and a mouse. -/
def total_ways : ℕ := 646

/-- Theorem stating that the total number of ways to buy three items
    (headphones, keyboard, and mouse) is 646. -/
theorem buy_three_items_ways :
  headphones * keyboard_mouse_sets +
  keyboards * headphones_mouse_sets +
  headphones * mice * keyboards = total_ways := by
  sorry

end NUMINAMATH_CALUDE_buy_three_items_ways_l933_93382


namespace NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_four_l933_93390

theorem least_positive_integer_for_multiple_of_four :
  ∃ (n : ℕ), n > 0 ∧ (575 + n) % 4 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (575 + m) % 4 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_for_multiple_of_four_l933_93390


namespace NUMINAMATH_CALUDE_sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth_l933_93373

theorem sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth : 
  10 * 56 * (3/2 + 5/4 + 9/8 + 17/16 + 33/32 + 65/64 - 7) = -1/64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_minus_seven_equals_negative_one_sixty_fourth_l933_93373


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l933_93337

/-- Represents a country --/
inductive Country
| Italy
| Germany

/-- Represents a decade --/
inductive Decade
| Fifties
| Sixties

/-- The price of a stamp in cents --/
def stampPrice (c : Country) : ℕ :=
  match c with
  | Country.Italy => 7
  | Country.Germany => 5

/-- The number of stamps Juan has from a given country and decade --/
def stampCount (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Italy, Decade.Fifties => 5
  | Country.Italy, Decade.Sixties => 8
  | Country.Germany, Decade.Fifties => 7
  | Country.Germany, Decade.Sixties => 6

/-- The total cost of Juan's European stamps from Italy and Germany issued before the 70's --/
def totalCost : ℚ :=
  let italyTotal := (stampCount Country.Italy Decade.Fifties + stampCount Country.Italy Decade.Sixties) * stampPrice Country.Italy
  let germanyTotal := (stampCount Country.Germany Decade.Fifties + stampCount Country.Germany Decade.Sixties) * stampPrice Country.Germany
  (italyTotal + germanyTotal : ℚ) / 100

theorem total_cost_is_correct : totalCost = 156 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l933_93337


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l933_93320

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- Define the set {1, 2}
def set_1_2 : Set ℝ := {1, 2}

-- Statement to prove
theorem sufficient_not_necessary_condition :
  (∀ m ∈ set_1_2, log10 m < 1) ∧
  (∃ m : ℝ, log10 m < 1 ∧ m ∉ set_1_2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l933_93320


namespace NUMINAMATH_CALUDE_bowling_team_size_l933_93325

/-- The number of original players in a bowling team -/
def original_players : ℕ := 7

/-- The original average weight of the team in kg -/
def original_avg : ℚ := 94

/-- The weight of the first new player in kg -/
def new_player1 : ℚ := 110

/-- The weight of the second new player in kg -/
def new_player2 : ℚ := 60

/-- The new average weight of the team after adding two players, in kg -/
def new_avg : ℚ := 92

theorem bowling_team_size :
  (original_avg * original_players + new_player1 + new_player2) / (original_players + 2) = new_avg :=
sorry

end NUMINAMATH_CALUDE_bowling_team_size_l933_93325
