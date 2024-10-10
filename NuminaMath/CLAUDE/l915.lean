import Mathlib

namespace ten_thousandths_digit_of_437_div_128_l915_91542

theorem ten_thousandths_digit_of_437_div_128 :
  (437 : ℚ) / 128 = 3 + 4/10 + 1/100 + 4/1000 + 6/10000 + 8/100000 + 7/1000000 + 5/10000000 :=
by sorry

end ten_thousandths_digit_of_437_div_128_l915_91542


namespace cosine_sine_sum_equals_half_l915_91582

theorem cosine_sine_sum_equals_half : 
  Real.cos (80 * π / 180) * Real.cos (20 * π / 180) + 
  Real.sin (100 * π / 180) * Real.sin (380 * π / 180) = 1 / 2 := by
  sorry

end cosine_sine_sum_equals_half_l915_91582


namespace not_in_range_iff_b_in_interval_l915_91576

/-- The function f(x) = x^2 + bx + 3 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- Theorem: -3 is not in the range of f(x) = x^2 + bx + 3 if and only if b ∈ (-2√6, 2√6) -/
theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, f b x ≠ -3) ↔ b ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) :=
sorry

end not_in_range_iff_b_in_interval_l915_91576


namespace arithmetic_sequence_sum_l915_91580

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 32 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 10 + a 11 = 32

/-- Theorem: If a is an arithmetic sequence satisfying the sum condition,
    then the sum of the 6th and 7th terms is 16 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : a 6 + a 7 = 16 := by
  sorry

end arithmetic_sequence_sum_l915_91580


namespace octal_calculation_l915_91509

/-- Represents a number in base 8 --/
def OctalNumber := ℕ

/-- Converts a natural number to its octal representation --/
def to_octal (n : ℕ) : OctalNumber :=
  sorry

/-- Performs subtraction in base 8 --/
def octal_sub (a b : OctalNumber) : OctalNumber :=
  sorry

/-- Theorem stating the result of the given octal calculation --/
theorem octal_calculation : 
  octal_sub (octal_sub (to_octal 123) (to_octal 51)) (to_octal 15) = to_octal 25 :=
sorry

end octal_calculation_l915_91509


namespace second_player_can_win_l915_91545

/-- A function representing a player's strategy for choosing digits. -/
def Strategy := Nat → Fin 5

/-- The result of a game where two players alternate choosing digits. -/
def GameResult (s1 s2 : Strategy) : Fin 9 :=
  (List.range 30).foldl
    (λ acc i => (acc + if i % 2 = 0 then s1 i else s2 i) % 9)
    0

/-- Theorem stating that the second player can always ensure divisibility by 9. -/
theorem second_player_can_win :
  ∀ s1 : Strategy, ∃ s2 : Strategy, GameResult s1 s2 = 0 :=
sorry

end second_player_can_win_l915_91545


namespace parallelogram_base_l915_91588

/-- Given a parallelogram with area 864 square cm and height 24 cm, its base is 36 cm. -/
theorem parallelogram_base (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 864 ∧ height = 24 ∧ area = base * height → base = 36 := by
  sorry

end parallelogram_base_l915_91588


namespace function_value_at_pi_over_four_l915_91568

theorem function_value_at_pi_over_four (φ : ℝ) :
  let f : ℝ → ℝ := λ x => Real.sin (x + 2 * φ) - 2 * Real.sin φ * Real.cos (x + φ)
  f (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end function_value_at_pi_over_four_l915_91568


namespace trapezoid_area_l915_91540

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the isosceles triangle PQR -/
structure IsoscelesTriangle extends Triangle

/-- Represents the trapezoid TQRS -/
structure Trapezoid where
  area : ℝ

/-- The problem setup -/
axiom smallest_triangle : Triangle
axiom smallest_triangle_area : smallest_triangle.area = 2

axiom PQR : IsoscelesTriangle
axiom PQR_area : PQR.area = 72

axiom PTQ : Triangle
axiom PTQ_composition : PTQ.area = 5 * smallest_triangle.area

axiom TQRS : Trapezoid
axiom TQRS_formation : TQRS.area = PQR.area - PTQ.area

/-- The theorem to prove -/
theorem trapezoid_area : TQRS.area = 62 := by
  sorry

end trapezoid_area_l915_91540


namespace cubic_root_implies_p_value_l915_91595

theorem cubic_root_implies_p_value : ∀ p : ℝ, (3 : ℝ)^3 + p * 3 - 18 = 0 → p = -3 := by
  sorry

end cubic_root_implies_p_value_l915_91595


namespace g_inequality_solution_set_range_of_a_l915_91530

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - 5*a| + |2*x + 1|
def g (x : ℝ) : ℝ := |x - 1| + 3

-- Theorem for the solution set of |g(x)| < 8
theorem g_inequality_solution_set :
  {x : ℝ | |g x| < 8} = {x : ℝ | -4 < x ∧ x < 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (h : ∀ x₁ : ℝ, ∃ x₂ : ℝ, f a x₁ = g x₂) :
  a ≥ 0.4 ∨ a ≤ -0.8 := by sorry

end g_inequality_solution_set_range_of_a_l915_91530


namespace simplify_complex_expression_l915_91546

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_expression :
  6 * (2 - i) + 4 * i * (6 - i) = 16 + 18 * i :=
by sorry

end simplify_complex_expression_l915_91546


namespace three_digit_divisible_by_eight_consecutive_l915_91533

theorem three_digit_divisible_by_eight_consecutive : ∃ n : ℕ, 
  100 ≤ n ∧ n ≤ 999 ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 8 → k ∣ n) :=
by sorry

end three_digit_divisible_by_eight_consecutive_l915_91533


namespace arithmetic_sequence_inequality_l915_91507

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, if 0 < a_1 < a_2, then a_2 > √(a_1 * a_3) -/
theorem arithmetic_sequence_inequality (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  0 < a 1 → a 1 < a 2 → a 2 > Real.sqrt (a 1 * a 3) := by
  sorry

end arithmetic_sequence_inequality_l915_91507


namespace sum_of_squares_of_coefficients_l915_91579

def polynomial (x : ℝ) := 3 * (x^4 + 2*x^3 + 5*x^2 + 2)

theorem sum_of_squares_of_coefficients : 
  (3^2 : ℝ) + (6^2 : ℝ) + (15^2 : ℝ) + (6^2 : ℝ) = 306 := by sorry

end sum_of_squares_of_coefficients_l915_91579


namespace calculate_ants_monroe_ants_l915_91591

/-- Given a collection of spiders and ants, calculate the number of ants -/
theorem calculate_ants (num_spiders : ℕ) (total_legs : ℕ) (spider_legs : ℕ) (ant_legs : ℕ) : ℕ :=
  let num_ants := (total_legs - num_spiders * spider_legs) / ant_legs
  num_ants

/-- Prove that Monroe has 12 ants in his collection -/
theorem monroe_ants : 
  let num_spiders : ℕ := 8
  let total_legs : ℕ := 136
  let spider_legs : ℕ := 8
  let ant_legs : ℕ := 6
  calculate_ants num_spiders total_legs spider_legs ant_legs = 12 := by
  sorry

end calculate_ants_monroe_ants_l915_91591


namespace class_height_most_suitable_for_census_l915_91581

/-- Represents a scenario that could be investigated --/
inductive Scenario
| WaterQuality
| StudentMentalHealth
| ClassHeight
| TVRatings

/-- Characteristics of a scenario --/
structure ScenarioCharacteristics where
  population_size : ℕ
  accessibility : Bool
  feasibility : Bool

/-- Defines what makes a scenario suitable for a census --/
def suitable_for_census (c : ScenarioCharacteristics) : Prop :=
  c.population_size ≤ 100 ∧ c.accessibility ∧ c.feasibility

/-- Assigns characteristics to each scenario --/
def scenario_characteristics : Scenario → ScenarioCharacteristics
| Scenario.WaterQuality => ⟨1000, false, false⟩
| Scenario.StudentMentalHealth => ⟨1000000, false, false⟩
| Scenario.ClassHeight => ⟨30, true, true⟩
| Scenario.TVRatings => ⟨10000000, false, false⟩

theorem class_height_most_suitable_for_census :
  ∀ s : Scenario, s ≠ Scenario.ClassHeight →
    ¬(suitable_for_census (scenario_characteristics s)) ∧
    suitable_for_census (scenario_characteristics Scenario.ClassHeight) :=
by sorry

end class_height_most_suitable_for_census_l915_91581


namespace sample_grade_10_is_15_l915_91521

/-- Represents a school with stratified sampling -/
structure School where
  total_students : ℕ
  grade_10_students : ℕ
  sample_size : ℕ

/-- Calculates the number of Grade 10 students to be sampled -/
def sample_grade_10 (school : School) : ℕ :=
  (school.sample_size * school.grade_10_students) / school.total_students

/-- Theorem stating that for the given school parameters, 
    the number of Grade 10 students to be sampled is 15 -/
theorem sample_grade_10_is_15 (school : School) 
  (h1 : school.total_students = 2000)
  (h2 : school.grade_10_students = 600)
  (h3 : school.sample_size = 50) :
  sample_grade_10 school = 15 := by
  sorry

#eval sample_grade_10 ⟨2000, 600, 50⟩

end sample_grade_10_is_15_l915_91521


namespace min_rectangles_theorem_l915_91566

/-- The minimum number of rectangles that can be placed on an n × n grid -/
def min_rectangles (k n : ℕ) : ℕ :=
  if n = k then k
  else min n (2*n - 2*k + 2)

/-- Theorem stating the minimum number of rectangles that can be placed -/
theorem min_rectangles_theorem (k n : ℕ) (h1 : k ≥ 2) (h2 : k ≤ n) (h3 : n ≤ 2*k - 1) :
  min_rectangles k n = 
    if n = k then k
    else min n (2*n - 2*k + 2) := by
  sorry

#check min_rectangles_theorem

end min_rectangles_theorem_l915_91566


namespace melissa_banana_count_l915_91561

/-- Calculates the final number of bananas Melissa has -/
def melissasFinalBananas (initialBananas buyMultiplier sharedBananas : ℕ) : ℕ :=
  let remainingBananas := initialBananas - sharedBananas
  let boughtBananas := buyMultiplier * remainingBananas
  remainingBananas + boughtBananas

theorem melissa_banana_count :
  melissasFinalBananas 88 3 4 = 336 := by
  sorry

end melissa_banana_count_l915_91561


namespace function_inequality_l915_91552

open Real

theorem function_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → (x + 1) * f x + x * f' x ≥ 0) :
  f 1 < 2 * ℯ * f 2 := by
  sorry

end function_inequality_l915_91552


namespace min_n_value_l915_91531

theorem min_n_value (m n : ℝ) : 
  (∃ x : ℝ, x^2 + (m - 2023) * x + (n - 1) = 0 ∧ 
   ∀ y : ℝ, y^2 + (m - 2023) * y + (n - 1) = 0 → y = x) → 
  n ≥ 1 ∧ ∃ m₀ : ℝ, ∃ x₀ : ℝ, x₀^2 + (m₀ - 2023) * x₀ = 0 :=
by sorry

end min_n_value_l915_91531


namespace min_value_theorem_l915_91583

theorem min_value_theorem (a : ℝ) (h1 : 0 < a) (h2 : a < 2) :
  (4 / a) + (1 / (2 - a)) ≥ (9 / 2) := by
  sorry

end min_value_theorem_l915_91583


namespace product_selection_theorem_l915_91500

def total_products : ℕ := 10
def defective_products : ℕ := 3
def good_products : ℕ := 7
def products_drawn : ℕ := 5

theorem product_selection_theorem :
  (∃ (no_defective : ℕ) (exactly_two_defective : ℕ) (at_least_one_defective : ℕ),
    -- No defective products
    no_defective = Nat.choose good_products products_drawn ∧
    -- Exactly 2 defective products
    exactly_two_defective = Nat.choose defective_products 2 * Nat.choose good_products 3 ∧
    -- At least 1 defective product
    at_least_one_defective = Nat.choose total_products products_drawn - Nat.choose good_products products_drawn) :=
by
  sorry

end product_selection_theorem_l915_91500


namespace smallest_seating_l915_91584

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  chairs : Nat
  seated : Nat

/-- Checks if the seating arrangement satisfies the condition that any new person must sit next to someone already seated. -/
def satisfiesCondition (table : CircularTable) : Prop :=
  ∀ (new_seat : Nat), new_seat < table.chairs → 
    ∃ (adjacent_seat : Nat), adjacent_seat < table.chairs ∧ 
      (adjacent_seat = (new_seat + 1) % table.chairs ∨ 
       adjacent_seat = (new_seat + table.chairs - 1) % table.chairs)

/-- Theorem stating the smallest number of people that can be seated to satisfy the condition. -/
theorem smallest_seating (table : CircularTable) : 
  table.chairs = 90 → 
  (∀ n < 23, ¬(satisfiesCondition ⟨90, n⟩)) ∧ 
  satisfiesCondition ⟨90, 23⟩ := by
  sorry

end smallest_seating_l915_91584


namespace james_weekly_nut_spending_l915_91518

/-- Represents the cost and consumption of nuts -/
structure NutInfo where
  price : ℚ
  weight : ℚ
  consumption : ℚ
  days : ℕ

/-- Calculates the weekly cost for a type of nut -/
def weeklyCost (nut : NutInfo) : ℚ :=
  (nut.consumption / nut.days) * 7 * (nut.price / nut.weight)

/-- Theorem stating James' weekly spending on nuts -/
theorem james_weekly_nut_spending :
  let pistachios : NutInfo := ⟨10, 5, 30, 5⟩
  let almonds : NutInfo := ⟨8, 4, 24, 4⟩
  let walnuts : NutInfo := ⟨12, 6, 18, 3⟩
  weeklyCost pistachios + weeklyCost almonds + weeklyCost walnuts = 252 := by
  sorry

end james_weekly_nut_spending_l915_91518


namespace union_of_A_and_B_l915_91555

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 1}
def B : Set ℝ := {x | x^2 - 3*x < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -2 < x ∧ x < 3} := by sorry

end union_of_A_and_B_l915_91555


namespace sum_of_squares_l915_91590

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 131)
  (h2 : a + b + c = 21) : 
  a^2 + b^2 + c^2 = 179 := by
sorry

end sum_of_squares_l915_91590


namespace calculator_prices_and_relations_l915_91543

/-- The price of two A-brand and three B-brand calculators -/
def total_price_1 : ℝ := 156

/-- The price of three A-brand and one B-brand calculator -/
def total_price_2 : ℝ := 122

/-- The discount rate for A-brand calculators during promotion -/
def discount_rate_A : ℝ := 0.8

/-- The discount rate for B-brand calculators during promotion -/
def discount_rate_B : ℝ := 0.875

/-- The unit price of A-brand calculators -/
def price_A : ℝ := 30

/-- The unit price of B-brand calculators -/
def price_B : ℝ := 32

/-- The function relation for A-brand calculators during promotion -/
def y1 (x : ℝ) : ℝ := 24 * x

/-- The function relation for B-brand calculators during promotion -/
def y2 (x : ℝ) : ℝ := 28 * x

theorem calculator_prices_and_relations :
  (2 * price_A + 3 * price_B = total_price_1) ∧
  (3 * price_A + price_B = total_price_2) ∧
  (∀ x, y1 x = discount_rate_A * price_A * x) ∧
  (∀ x, y2 x = discount_rate_B * price_B * x) :=
sorry

end calculator_prices_and_relations_l915_91543


namespace triangle_number_assignment_l915_91536

theorem triangle_number_assignment :
  ∀ (A B C D E F : ℕ),
    ({A, B, C, D, E, F} : Finset ℕ) = {1, 2, 3, 4, 5, 6} →
    B + D + E = 14 →
    C + E + F = 12 →
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 :=
by sorry

end triangle_number_assignment_l915_91536


namespace cube_sum_inequality_l915_91586

theorem cube_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) :
  a + b ≤ 2 := by
  sorry

end cube_sum_inequality_l915_91586


namespace all_flowers_bloom_monday_l915_91526

-- Define the days of the week
inductive Day : Type
| monday : Day
| tuesday : Day
| wednesday : Day
| thursday : Day
| friday : Day
| saturday : Day
| sunday : Day

-- Define the flower types
inductive Flower : Type
| sunflower : Flower
| lily : Flower
| peony : Flower

-- Define a function to check if a flower blooms on a given day
def blooms (f : Flower) (d : Day) : Prop := sorry

-- Define the conditions
axiom one_day_all_bloom : ∃! d : Day, ∀ f : Flower, blooms f d

axiom no_three_consecutive_days : 
  ∀ f : Flower, ∀ d1 d2 d3 : Day, 
    (blooms f d1 ∧ blooms f d2 ∧ blooms f d3) → 
    (d1 ≠ Day.monday ∨ d2 ≠ Day.tuesday ∨ d3 ≠ Day.wednesday) ∧
    (d1 ≠ Day.tuesday ∨ d2 ≠ Day.wednesday ∨ d3 ≠ Day.thursday) ∧
    (d1 ≠ Day.wednesday ∨ d2 ≠ Day.thursday ∨ d3 ≠ Day.friday) ∧
    (d1 ≠ Day.thursday ∨ d2 ≠ Day.friday ∨ d3 ≠ Day.saturday) ∧
    (d1 ≠ Day.friday ∨ d2 ≠ Day.saturday ∨ d3 ≠ Day.sunday) ∧
    (d1 ≠ Day.saturday ∨ d2 ≠ Day.sunday ∨ d3 ≠ Day.monday) ∧
    (d1 ≠ Day.sunday ∨ d2 ≠ Day.monday ∨ d3 ≠ Day.tuesday)

axiom two_flowers_not_bloom : 
  ∀ f1 f2 : Flower, f1 ≠ f2 → 
    (∃! d : Day, ¬(blooms f1 d ∧ blooms f2 d))

axiom sunflowers_not_bloom : 
  ¬blooms Flower.sunflower Day.tuesday ∧ 
  ¬blooms Flower.sunflower Day.thursday ∧ 
  ¬blooms Flower.sunflower Day.sunday

axiom lilies_not_bloom : 
  ¬blooms Flower.lily Day.thursday ∧ 
  ¬blooms Flower.lily Day.saturday

axiom peonies_not_bloom : 
  ¬blooms Flower.peony Day.sunday

-- The theorem to prove
theorem all_flowers_bloom_monday : 
  ∀ f : Flower, blooms f Day.monday ∧ 
  (∀ d : Day, d ≠ Day.monday → ¬(∀ f : Flower, blooms f d)) :=
by sorry

end all_flowers_bloom_monday_l915_91526


namespace rebeccas_haircut_price_l915_91529

/-- Rebecca's hair salon pricing and earnings --/
theorem rebeccas_haircut_price 
  (perm_price : ℕ) 
  (dye_job_price : ℕ) 
  (dye_cost : ℕ) 
  (haircuts : ℕ) 
  (perms : ℕ) 
  (dye_jobs : ℕ) 
  (tips : ℕ) 
  (total_earnings : ℕ) 
  (h : perm_price = 40) 
  (i : dye_job_price = 60) 
  (j : dye_cost = 10) 
  (k : haircuts = 4) 
  (l : perms = 1) 
  (m : dye_jobs = 2) 
  (n : tips = 50) 
  (o : total_earnings = 310) : 
  ∃ (haircut_price : ℕ), 
    haircut_price * haircuts + 
    perm_price * perms + 
    dye_job_price * dye_jobs + 
    tips - 
    dye_cost * dye_jobs = total_earnings ∧ 
    haircut_price = 30 := by
  sorry

end rebeccas_haircut_price_l915_91529


namespace chairs_to_remove_proof_l915_91593

/-- Calculates the number of chairs to remove given the initial setup and expected attendance --/
def chairs_to_remove (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_attendees : ℕ) : ℕ :=
  let rows_needed := (expected_attendees + chairs_per_row - 1) / chairs_per_row
  let chairs_needed := rows_needed * chairs_per_row
  total_chairs - chairs_needed

/-- Proves that given the specific conditions, 105 chairs should be removed --/
theorem chairs_to_remove_proof :
  chairs_to_remove 15 300 180 = 105 := by
  sorry

#eval chairs_to_remove 15 300 180

end chairs_to_remove_proof_l915_91593


namespace octagon_area_l915_91514

/-- Given two concentric squares with side length 2 and a line segment AB of length 3/4 between the squares,
    the area of the resulting octagon ABCDEFGH is 6. -/
theorem octagon_area (square_side : ℝ) (AB_length : ℝ) (h1 : square_side = 2) (h2 : AB_length = 3/4) :
  let triangle_area := (1/2) * square_side * AB_length
  let octagon_area := 8 * triangle_area
  octagon_area = 6 := by sorry

end octagon_area_l915_91514


namespace perfect_square_15AB9_l915_91592

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

def has_form_15AB9 (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ n = 15000 + A * 100 + B * 10 + 9

theorem perfect_square_15AB9 (n : ℕ) (h1 : is_five_digit n) (h2 : has_form_15AB9 n) (h3 : is_perfect_square n) :
  ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ n = 15000 + A * 100 + B * 10 + 9 ∧ A + B = 3 :=
sorry

end perfect_square_15AB9_l915_91592


namespace next_two_juicy_numbers_l915_91553

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a * b * c * d = n ∧ 1 = 1/a + 1/b + 1/c + 1/d

theorem next_two_juicy_numbers :
  (∀ n < 6, ¬ is_juicy n) ∧
  is_juicy 6 ∧
  is_juicy 12 ∧
  is_juicy 20 ∧
  (∀ n, 6 < n ∧ n < 12 → ¬ is_juicy n) ∧
  (∀ n, 12 < n ∧ n < 20 → ¬ is_juicy n) :=
sorry

end next_two_juicy_numbers_l915_91553


namespace complex_magnitude_l915_91559

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 3 - 4 * Complex.I) : Complex.abs z = 5 := by
  sorry

end complex_magnitude_l915_91559


namespace min_cuts_for_100_pieces_l915_91519

/-- Represents the number of pieces a cube is divided into after making cuts -/
def num_pieces (a b c : ℕ) : ℕ := (a + 1) * (b + 1) * (c + 1)

/-- Theorem stating that 11 is the minimum number of cuts needed to divide a cube into 100 pieces -/
theorem min_cuts_for_100_pieces :
  ∃ (a b c : ℕ), num_pieces a b c = 100 ∧ a + b + c = 11 ∧
  (∀ (x y z : ℕ), num_pieces x y z ≥ 100 → x + y + z ≥ 11) :=
sorry

end min_cuts_for_100_pieces_l915_91519


namespace goldfish_fed_by_four_scoops_l915_91516

/-- The number of goldfish that can be fed by one scoop of fish food -/
def goldfish_per_scoop : ℕ := 8

/-- The number of scoops of fish food -/
def number_of_scoops : ℕ := 4

/-- Theorem: 4 scoops of fish food can feed 32 goldfish -/
theorem goldfish_fed_by_four_scoops : 
  number_of_scoops * goldfish_per_scoop = 32 := by
  sorry

end goldfish_fed_by_four_scoops_l915_91516


namespace monotonic_decreasing_interval_l915_91508

def f (x : ℝ) : ℝ := x^2 - 6*x + 5

theorem monotonic_decreasing_interval :
  ∀ x y : ℝ, x < y → y ≤ 3 → f x > f y :=
by sorry

end monotonic_decreasing_interval_l915_91508


namespace equation_solution_l915_91556

theorem equation_solution (a b : ℤ) : 
  (((a + 2 : ℚ) / (b + 1) + (a + 1 : ℚ) / (b + 2) = 1 + 6 / (a + b + 1)) ∧ 
   (b + 1 ≠ 0) ∧ (b + 2 ≠ 0) ∧ (a + b + 1 ≠ 0)) ↔ 
  ((∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ∨ (a = 1 ∧ b = 0)) :=
by sorry

end equation_solution_l915_91556


namespace logarithmic_function_problem_l915_91524

open Real

theorem logarithmic_function_problem (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) :
  let f := fun x => |log x|
  (f a = f b) →
  (∀ x ∈ Set.Icc (a^2) b, f x ≤ 2) →
  (∃ x ∈ Set.Icc (a^2) b, f x = 2) →
  2 * a + b = 2 / Real.exp 1 + Real.exp 1 := by
sorry

end logarithmic_function_problem_l915_91524


namespace line_tangent_to_circle_l915_91570

/-- The line x + ay = 3 is tangent to the circle (x-1)² + y² = 2 if and only if a = ±1 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (x + a * y = 3) → ((x - 1)^2 + y^2 = 2) → 
   (∀ x' y' : ℝ, (x' + a * y' = 3) → ((x' - 1)^2 + y'^2 ≥ 2))) ↔ 
  (a = 1 ∨ a = -1) :=
sorry

end line_tangent_to_circle_l915_91570


namespace minimum_at_one_positive_when_minimum_less_than_one_l915_91562

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := (2 * x + 1) * (a * x - 1) / x

-- Theorem 1: If the minimum point of f(x) is at x_0 = 1, then a = 1
theorem minimum_at_one (a : ℝ) :
  (∀ x > 0, f a x ≥ f a 1) → a = 1 := by sorry

-- Theorem 2: If 0 < x_0 < 1, where x_0 is the minimum point of f(x), then f(x) > 0 for all x > 0
theorem positive_when_minimum_less_than_one (a : ℝ) (x_0 : ℝ) :
  (0 < x_0 ∧ x_0 < 1) →
  (∀ x > 0, f a x ≥ f a x_0) →
  (∀ x > 0, f a x > 0) := by sorry

end minimum_at_one_positive_when_minimum_less_than_one_l915_91562


namespace geometric_sequence_product_l915_91541

/-- A geometric sequence with positive terms where a_1 and a_{99} are roots of x^2 - 10x + 16 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) ∧
  a 1 * a 99 = 16 ∧
  a 1 + a 99 = 10

theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) : 
  a 20 * a 50 * a 80 = 64 := by
sorry

end geometric_sequence_product_l915_91541


namespace sum_of_a_and_b_range_of_c_l915_91569

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x + b

-- Define the inequality
def inequality (a b : ℝ) := {x : ℝ | f a b x < 0}

-- Define the second quadratic function
def g (b c x : ℝ) := -x^2 + b*x + c

-- Theorem 1
theorem sum_of_a_and_b (a b : ℝ) : 
  inequality a b = {x | 2 < x ∧ x < 3} → a + b = 11 := by sorry

-- Theorem 2
theorem range_of_c (c : ℝ) : 
  (∀ x, g 6 c x ≤ 0) → c ≤ -9 := by sorry

end sum_of_a_and_b_range_of_c_l915_91569


namespace sum_of_coefficients_l915_91517

/-- A polynomial with real coefficients -/
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

/-- Theorem stating that if g(1+i) = 0 and g(3i) = 0, then p + q + r + s = 9 -/
theorem sum_of_coefficients (p q r s : ℝ) :
  g p q r s (1 + Complex.I) = 0 →
  g p q r s (3 * Complex.I) = 0 →
  p + q + r + s = 9 := by
  sorry

end sum_of_coefficients_l915_91517


namespace cubic_equation_root_b_value_l915_91527

theorem cubic_equation_root_b_value :
  ∀ (a b : ℚ),
  (∃ (x : ℂ), x = 1 + Real.sqrt 2 ∧ x^3 + a*x^2 + b*x + 6 = 0) →
  b = 11 := by
sorry

end cubic_equation_root_b_value_l915_91527


namespace at_least_one_irrational_l915_91548

theorem at_least_one_irrational (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) :
  ¬(∃ (q r : ℚ), (a = ↑q ∧ b = ↑r)) :=
sorry

end at_least_one_irrational_l915_91548


namespace continuous_fraction_solution_l915_91503

theorem continuous_fraction_solution :
  ∃ y : ℝ, y > 0 ∧ y = 3 + 3 / (2 + 3 / y) ∧ y = (3 + 3 * Real.sqrt 3) / 2 := by
  sorry

end continuous_fraction_solution_l915_91503


namespace initial_erasers_eq_taken_plus_left_l915_91538

/-- The initial number of erasers in the box -/
def initial_erasers : ℕ := 69

/-- The number of erasers Doris took out of the box -/
def erasers_taken : ℕ := 54

/-- The number of erasers left in the box -/
def erasers_left : ℕ := 15

/-- Theorem stating that the initial number of erasers is equal to
    the sum of erasers taken and erasers left -/
theorem initial_erasers_eq_taken_plus_left :
  initial_erasers = erasers_taken + erasers_left := by
  sorry

end initial_erasers_eq_taken_plus_left_l915_91538


namespace alternating_fraction_value_l915_91539

theorem alternating_fraction_value :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = 5 / 3 := by
  sorry

end alternating_fraction_value_l915_91539


namespace grocery_store_costs_l915_91512

/-- Calculates the money paid for orders given total costs and fractions for salary and delivery --/
def money_paid_for_orders (total_costs : ℝ) (salary_fraction : ℝ) (delivery_fraction : ℝ) : ℝ :=
  let salary := salary_fraction * total_costs
  let remaining := total_costs - salary
  let delivery := delivery_fraction * remaining
  total_costs - salary - delivery

/-- Proves that given the specified conditions, the money paid for orders is $1800 --/
theorem grocery_store_costs : 
  money_paid_for_orders 4000 (2/5) (1/4) = 1800 := by
  sorry

end grocery_store_costs_l915_91512


namespace full_price_revenue_is_1250_l915_91567

/-- Represents the revenue from full-price tickets in a school club's ticket sale. -/
def revenue_full_price (full_price : ℚ) (num_full_price : ℕ) : ℚ :=
  full_price * num_full_price

/-- Represents the total revenue from all tickets sold. -/
def total_revenue (full_price : ℚ) (num_full_price : ℕ) (num_discount_price : ℕ) : ℚ :=
  revenue_full_price full_price num_full_price + (full_price / 3) * num_discount_price

/-- Theorem stating that the revenue from full-price tickets is $1250. -/
theorem full_price_revenue_is_1250 :
  ∃ (full_price : ℚ) (num_full_price num_discount_price : ℕ),
    num_full_price + num_discount_price = 200 ∧
    total_revenue full_price num_full_price num_discount_price = 2500 ∧
    revenue_full_price full_price num_full_price = 1250 :=
sorry

end full_price_revenue_is_1250_l915_91567


namespace sum_of_digits_difference_l915_91577

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Sum of digits for all numbers in a list -/
def sumOfDigitsForList (list : List ℕ) : ℕ := sorry

/-- List of odd numbers from 1 to 99 -/
def oddNumbers : List ℕ := sorry

/-- List of even numbers from 2 to 100 -/
def evenNumbers : List ℕ := sorry

theorem sum_of_digits_difference : 
  sumOfDigitsForList oddNumbers - sumOfDigitsForList evenNumbers = 49 := by sorry

end sum_of_digits_difference_l915_91577


namespace francis_fruit_cups_l915_91585

/-- The cost of a breakfast given the number of muffins and fruit cups -/
def breakfast_cost (muffins fruit_cups : ℕ) : ℕ := 2 * muffins + 3 * fruit_cups

/-- The problem statement -/
theorem francis_fruit_cups : ∃ f : ℕ, 
  breakfast_cost 2 f + breakfast_cost 2 1 = 17 ∧ f = 2 := by sorry

end francis_fruit_cups_l915_91585


namespace complementary_angle_of_25_l915_91565

def complementary_angle (x : ℝ) : ℝ := 90 - x

theorem complementary_angle_of_25 :
  complementary_angle 25 = 65 :=
by sorry

end complementary_angle_of_25_l915_91565


namespace chipped_marbles_count_l915_91598

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [18, 19, 21, 23, 25, 34]

/-- The total number of marbles -/
def total_marbles : Nat := bags.sum

/-- Predicate to check if a number is divisible by 3 -/
def divisible_by_three (n : Nat) : Prop := n % 3 = 0

/-- The number of bags Jane takes -/
def jane_bags : Nat := 3

/-- The number of bags George takes -/
def george_bags : Nat := 2

/-- The number of bags that remain -/
def remaining_bags : Nat := bags.length - jane_bags - george_bags

/-- Theorem stating the number of chipped marbles -/
theorem chipped_marbles_count : 
  ∃ (chipped : Nat) (jane george : List Nat),
    chipped ∈ bags ∧
    jane.length = jane_bags ∧
    george.length = george_bags ∧
    (jane.sum = 2 * george.sum) ∧
    (∀ m ∈ jane ++ george, m ≠ chipped) ∧
    divisible_by_three (total_marbles - chipped) ∧
    chipped = 23 := by
  sorry

end chipped_marbles_count_l915_91598


namespace opposite_greater_implies_negative_l915_91596

theorem opposite_greater_implies_negative (x : ℝ) : -x > x → x < 0 := by
  sorry

end opposite_greater_implies_negative_l915_91596


namespace three_integers_sum_and_ratio_l915_91557

theorem three_integers_sum_and_ratio : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 90 ∧
  2 * b = 3 * a ∧
  2 * c = 5 * a ∧
  a = 18 ∧ b = 27 ∧ c = 45 := by
sorry

end three_integers_sum_and_ratio_l915_91557


namespace problem_solution_l915_91520

theorem problem_solution (x y : ℚ) 
  (eq1 : 102 * x - 5 * y = 25) 
  (eq2 : 3 * y - x = 10) : 
  10 - x = 2885 / 301 := by
sorry

end problem_solution_l915_91520


namespace sodium_reduction_is_one_third_l915_91589

def sodium_reduction_fraction (salt_teaspoons : ℕ) (parmesan_oz : ℕ) 
  (salt_sodium_per_tsp : ℕ) (parmesan_sodium_per_oz : ℕ) 
  (parmesan_reduction_oz : ℕ) : ℚ :=
  let original_sodium := salt_teaspoons * salt_sodium_per_tsp + parmesan_oz * parmesan_sodium_per_oz
  let reduced_sodium := salt_teaspoons * salt_sodium_per_tsp + (parmesan_oz - parmesan_reduction_oz) * parmesan_sodium_per_oz
  (original_sodium - reduced_sodium : ℚ) / original_sodium

theorem sodium_reduction_is_one_third :
  sodium_reduction_fraction 2 8 50 25 4 = 1/3 := by
  sorry

end sodium_reduction_is_one_third_l915_91589


namespace adams_shelves_l915_91573

theorem adams_shelves (action_figures_per_shelf : ℕ) (num_shelves : ℕ) (total_capacity : ℕ) :
  action_figures_per_shelf = 11 →
  num_shelves = 4 →
  total_capacity = action_figures_per_shelf * num_shelves →
  total_capacity = 44 :=
by sorry

end adams_shelves_l915_91573


namespace angle_with_special_complement_supplement_l915_91535

theorem angle_with_special_complement_supplement : 
  ∀ x : ℝ, 
  (0 ≤ x) ∧ (x ≤ 180) →
  (180 - x = 3 * (90 - x)) →
  x = 45 := by
sorry

end angle_with_special_complement_supplement_l915_91535


namespace streamer_earnings_l915_91534

/-- Calculates the weekly earnings of a streamer given their schedule and hourly rate. -/
def weekly_earnings (days_off : ℕ) (hours_per_stream : ℕ) (hourly_rate : ℕ) : ℕ :=
  (7 - days_off) * hours_per_stream * hourly_rate

/-- Theorem stating that a streamer with the given schedule earns $160 per week. -/
theorem streamer_earnings :
  weekly_earnings 3 4 10 = 160 := by
  sorry

end streamer_earnings_l915_91534


namespace base_10_to_base_7_conversion_l915_91510

-- Define the base 10 number
def base_10_num : ℕ := 3500

-- Define the base 7 representation
def base_7_repr : List ℕ := [1, 3, 1, 3, 0]

-- Function to convert a list of digits in base 7 to a natural number
def to_nat (digits : List ℕ) : ℕ :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem stating the equivalence
theorem base_10_to_base_7_conversion :
  base_10_num = to_nat base_7_repr :=
by sorry

end base_10_to_base_7_conversion_l915_91510


namespace pizza_division_l915_91558

theorem pizza_division (total_pizza : ℚ) (num_friends : ℕ) : 
  total_pizza = 5/6 ∧ num_friends = 4 → 
  total_pizza / num_friends = 5/24 := by
  sorry

end pizza_division_l915_91558


namespace solution_set_is_correct_l915_91571

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x : ℝ, (f.deriv.deriv) x < f x)
variable (h2 : f 2 = 1)

-- Define the solution set
def solution_set := {x : ℝ | f x > Real.exp (x - 2)}

-- State the theorem
theorem solution_set_is_correct : solution_set f = Set.Iio 2 := by sorry

end solution_set_is_correct_l915_91571


namespace lines_perpendicular_l915_91513

-- Define the slopes of the two lines
def slope1 : ℚ := 3 / 4
def slope2 : ℚ := -4 / 3

-- Define the equations of the two lines
def line1 (x y : ℚ) : Prop := 4 * y - 3 * x = 16
def line2 (x y : ℚ) : Prop := 3 * y + 4 * x = 15

-- Theorem: The two lines are perpendicular
theorem lines_perpendicular : slope1 * slope2 = -1 := by sorry

end lines_perpendicular_l915_91513


namespace no_x_term_condition_l915_91578

theorem no_x_term_condition (a : ℝ) : 
  (∀ x, (-2*x + a)*(x - 1) = -2*x^2 - a) → a = -2 := by
  sorry

end no_x_term_condition_l915_91578


namespace jack_total_travel_time_l915_91564

/-- Represents the time spent in a country during travel -/
structure CountryTime where
  customsHours : ℕ
  quarantineDays : ℕ

/-- Calculates the total hours spent in a country -/
def totalHoursInCountry (ct : CountryTime) : ℕ :=
  ct.customsHours + 24 * ct.quarantineDays

/-- The time Jack spent in each country -/
def jackTravelTime : List CountryTime := [
  { customsHours := 20, quarantineDays := 14 },  -- Canada
  { customsHours := 15, quarantineDays := 10 },  -- Australia
  { customsHours := 10, quarantineDays := 7 }    -- Japan
]

/-- Theorem stating the total time Jack spent in customs and quarantine -/
theorem jack_total_travel_time :
  List.foldl (λ acc ct => acc + totalHoursInCountry ct) 0 jackTravelTime = 789 :=
by sorry

end jack_total_travel_time_l915_91564


namespace students_walking_home_l915_91525

theorem students_walking_home (bus auto bike skate : ℚ) 
  (h_bus : bus = 1/3)
  (h_auto : auto = 1/5)
  (h_bike : bike = 1/8)
  (h_skate : skate = 1/15) :
  1 - (bus + auto + bike + skate) = 11/40 := by
sorry

end students_walking_home_l915_91525


namespace arrival_time_difference_l915_91522

/-- The distance to the campsite in miles -/
def distance : ℝ := 3

/-- Jill's hiking speed in miles per hour -/
def jill_speed : ℝ := 6

/-- Jack's hiking speed in miles per hour -/
def jack_speed : ℝ := 3

/-- Conversion factor from hours to minutes -/
def minutes_per_hour : ℝ := 60

/-- The time difference in minutes between Jill and Jack's arrival at the campsite -/
theorem arrival_time_difference : 
  (distance / jack_speed - distance / jill_speed) * minutes_per_hour = 30 := by
  sorry

end arrival_time_difference_l915_91522


namespace common_tangents_count_l915_91532

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + 9 = 0

-- Define the function to count common tangent lines
def count_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem statement
theorem common_tangents_count :
  count_common_tangents circle1 circle2 = 3 := by sorry

end common_tangents_count_l915_91532


namespace f_satisfies_conditions_l915_91537

-- Define the function f
def f (x : ℝ) : ℝ := x

-- Theorem stating that f satisfies all conditions
theorem f_satisfies_conditions :
  (∀ x : ℝ, f (-x) + f x = 0) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0) :=
by sorry

#check f_satisfies_conditions

end f_satisfies_conditions_l915_91537


namespace geometric_sequence_sum_l915_91501

/-- A geometric sequence where the sum of every two consecutive terms forms another geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 2) + a (n + 3) = r * (a n + a (n + 1))

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 1 + a 2 = 1 →
  a 3 + a 4 = 2 →
  a 9 + a 10 = 16 := by
sorry

end geometric_sequence_sum_l915_91501


namespace solution_satisfies_conditions_l915_91599

/-- Represents the number of teeth each person has -/
structure TeethCount where
  dima : ℕ
  yulia : ℕ
  kolya : ℕ
  vanya : ℕ

/-- Checks if the given teeth count satisfies all conditions of the problem -/
def satisfiesConditions (tc : TeethCount) : Prop :=
  tc.dima = tc.yulia + 2 ∧
  tc.kolya = tc.dima + tc.yulia ∧
  tc.vanya = 2 * tc.kolya ∧
  tc.dima + tc.yulia + tc.kolya + tc.vanya = 64

/-- The theorem stating that the solution satisfies all conditions -/
theorem solution_satisfies_conditions : 
  satisfiesConditions ⟨9, 7, 16, 32⟩ := by sorry

end solution_satisfies_conditions_l915_91599


namespace root_sum_squares_l915_91547

theorem root_sum_squares (a : ℝ) : 
  (∃ x y : ℝ, x^2 - 3*a*x + a^2 = 0 ∧ y^2 - 3*a*y + a^2 = 0 ∧ x^2 + y^2 = 1.75) → 
  a = 0.5 ∨ a = -0.5 := by
sorry

end root_sum_squares_l915_91547


namespace problem_statement_l915_91594

theorem problem_statement (a b : ℝ) (h1 : a - b = 1) (h2 : a^2 - b^2 = -1) :
  3 * a^2008 - 5 * b^2008 = -5 := by
  sorry

end problem_statement_l915_91594


namespace chips_left_uneaten_l915_91544

/-- Calculates the number of chips left uneaten when half of a batch of cookies is consumed. -/
theorem chips_left_uneaten (chips_per_cookie : ℕ) (dozens : ℕ) : 
  chips_per_cookie = 7 → dozens = 4 → (dozens * 12 / 2) * chips_per_cookie = 168 := by
  sorry

#check chips_left_uneaten

end chips_left_uneaten_l915_91544


namespace max_value_of_fraction_l915_91597

theorem max_value_of_fraction (x y : ℝ) (hx : -5 ≤ x ∧ x ≤ -3) (hy : 3 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -5 ≤ x' ∧ x' ≤ -3 ∧ 3 ≤ y' ∧ y' ≤ 5 → (x' + y' + 1) / x' ≤ (x + y + 1) / x) →
  (x + y + 1) / x = -1/5 :=
sorry

end max_value_of_fraction_l915_91597


namespace sean_initial_apples_l915_91575

theorem sean_initial_apples (initial : ℕ) (received : ℕ) (total : ℕ) : 
  received = 8 → total = 17 → initial + received = total → initial = 9 := by
  sorry

end sean_initial_apples_l915_91575


namespace modulus_of_i_times_one_plus_i_l915_91511

theorem modulus_of_i_times_one_plus_i : Complex.abs (Complex.I * (1 + Complex.I)) = 1 := by sorry

end modulus_of_i_times_one_plus_i_l915_91511


namespace triplet_satisfies_equations_l915_91528

theorem triplet_satisfies_equations : ∃ (x y z : ℂ),
  x + y + z = 5 ∧
  x^2 + y^2 + z^2 = 19 ∧
  x^3 + y^3 + z^3 = 53 ∧
  x = -1 ∧ y = Complex.I * Real.sqrt 3 ∧ z = -Complex.I * Real.sqrt 3 := by
  sorry

end triplet_satisfies_equations_l915_91528


namespace tan_alpha_value_l915_91554

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) : 
  Real.tan α = 13 / 16 := by
  sorry

end tan_alpha_value_l915_91554


namespace power_of_three_equivalence_l915_91523

theorem power_of_three_equivalence : 
  (1 / 2 : ℝ) * (3 : ℝ)^21 - (1 / 3 : ℝ) * (3 : ℝ)^20 = (7 / 6 : ℝ) * (3 : ℝ)^20 := by
  sorry

end power_of_three_equivalence_l915_91523


namespace exists_common_divisor_l915_91551

/-- A function from positive integers to positive integers greater than 1 -/
def PositiveIntegerFunction : Type := ℕ+ → ℕ+

/-- The property that f(m+n) divides f(m) + f(n) for all positive integers m and n -/
def HasDivisibilityProperty (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : ℕ+, (f (m + n)) ∣ (f m + f n)

/-- The theorem stating that there exists a common divisor greater than 1 for all values of f -/
theorem exists_common_divisor (f : PositiveIntegerFunction) 
  (h : HasDivisibilityProperty f) : 
  ∃ c : ℕ+, c > 1 ∧ ∀ n : ℕ+, c ∣ f n := by
  sorry

end exists_common_divisor_l915_91551


namespace flour_amount_l915_91549

def recipe_flour (flour_added : ℕ) (flour_to_add : ℕ) : ℕ :=
  flour_added + flour_to_add

theorem flour_amount : recipe_flour 6 4 = 10 := by
  sorry

end flour_amount_l915_91549


namespace max_divisors_1_to_20_l915_91560

def divisorCount (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def maxDivisorCount : ℕ → ℕ
  | 0 => 0
  | n + 1 => max (maxDivisorCount n) (divisorCount (n + 1))

theorem max_divisors_1_to_20 :
  maxDivisorCount 20 = 6 ∧
  divisorCount 12 = 6 ∧
  divisorCount 18 = 6 ∧
  divisorCount 20 = 6 ∧
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → divisorCount n ≤ 6 :=
by sorry

end max_divisors_1_to_20_l915_91560


namespace max_value_when_a_zero_range_of_a_for_local_max_l915_91563

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.exp x + a * (x - 1)^2

-- Theorem for part 1
theorem max_value_when_a_zero :
  ∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x ∧ f 0 x = 1 / Real.exp 1 :=
sorry

-- Theorem for part 2
theorem range_of_a_for_local_max :
  ∀ (a : ℝ), (∃ (x : ℝ), ∀ (y : ℝ), f a y ≤ f a x ∧ f a x ≤ 1/2) ↔
  (a < 1 / (2 * Real.exp 1) ∨ (a > 1 / (2 * Real.exp 1) ∧ a ≤ 1/2)) :=
sorry

end max_value_when_a_zero_range_of_a_for_local_max_l915_91563


namespace tangent_line_to_exponential_l915_91574

/-- If the line y = x + t is tangent to the curve y = e^x, then t = 1 -/
theorem tangent_line_to_exponential (t : ℝ) : 
  (∃ x₀ : ℝ, (x₀ + t = Real.exp x₀) ∧ 
             (1 = Real.exp x₀)) → 
  t = 1 := by
  sorry

end tangent_line_to_exponential_l915_91574


namespace solve_for_b_l915_91550

theorem solve_for_b (a b : ℝ) (h1 : a * b = 2 * (a + b) + 10) (h2 : b - a = 5) : b = 9 := by
  sorry

end solve_for_b_l915_91550


namespace unique_three_digit_odd_sum_27_l915_91505

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The digit sum of a natural number is the sum of its digits. -/
def DigitSum (n : ℕ) : ℕ := sorry

/-- A number is odd if it leaves a remainder of 1 when divided by 2. -/
def IsOdd (n : ℕ) : Prop := n % 2 = 1

/-- There is exactly one three-digit number with a digit sum of 27 that is odd. -/
theorem unique_three_digit_odd_sum_27 : 
  ∃! n : ℕ, ThreeDigitNumber n ∧ DigitSum n = 27 ∧ IsOdd n := by sorry

end unique_three_digit_odd_sum_27_l915_91505


namespace compound_composition_l915_91572

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_composition :
  ∃ (c : Compound),
    c.hydrogen = 8 ∧
    c.oxygen = 7 ∧
    molecularWeight c 12.01 1.01 16.00 = 192 ∧
    c.carbon = 6 := by
  sorry

end compound_composition_l915_91572


namespace office_network_connections_l915_91515

/-- Represents a computer network with switches and connections -/
structure ComputerNetwork where
  num_switches : ℕ
  connections_per_switch : ℕ
  num_crucial_switches : ℕ

/-- Calculates the total number of connections in the network -/
def total_connections (network : ComputerNetwork) : ℕ :=
  (network.num_switches * network.connections_per_switch) / 2 + network.num_crucial_switches

/-- Theorem: The total number of connections in the given network is 65 -/
theorem office_network_connections :
  let network : ComputerNetwork := {
    num_switches := 30,
    connections_per_switch := 4,
    num_crucial_switches := 5
  }
  total_connections network = 65 := by
  sorry

end office_network_connections_l915_91515


namespace initial_tomatoes_count_l915_91506

def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_remaining_after_yesterday : ℕ := 104

theorem initial_tomatoes_count :
  tomatoes_picked_yesterday + tomatoes_picked_today + tomatoes_remaining_after_yesterday = 201 :=
by
  sorry

end initial_tomatoes_count_l915_91506


namespace sin_180_degrees_l915_91502

theorem sin_180_degrees : Real.sin (180 * π / 180) = 0 := by
  sorry

end sin_180_degrees_l915_91502


namespace max_area_rectangle_l915_91587

/-- A rectangle with non-negative length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ
  length_nonneg : 0 ≤ length
  width_nonneg : 0 ≤ width

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℝ := r.length * r.width

/-- A rectangle with perimeter at least 80. -/
def RectangleWithLargePerimeter := {r : Rectangle // r.perimeter ≥ 80}

theorem max_area_rectangle (r : RectangleWithLargePerimeter) :
  r.val.area ≤ 400 ∧ 
  (r.val.area = 400 ↔ r.val.length = 20 ∧ r.val.width = 20) := by
sorry

end max_area_rectangle_l915_91587


namespace even_mono_decreasing_range_l915_91504

/-- A function f: ℝ → ℝ is even -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f: ℝ → ℝ is monotonically decreasing on [0,+∞) -/
def IsMonoDecreasingOnNonnegative (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem even_mono_decreasing_range (f : ℝ → ℝ) (m : ℝ) 
    (h_even : IsEven f) 
    (h_mono : IsMonoDecreasingOnNonnegative f) 
    (h_ineq : f m > f (1 - m)) : 
  m < 1/2 := by
  sorry

end even_mono_decreasing_range_l915_91504
