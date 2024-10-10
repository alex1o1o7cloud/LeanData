import Mathlib

namespace right_triangle_min_leg_sum_l561_56139

theorem right_triangle_min_leg_sum (a b : ℝ) (h_right : a > 0 ∧ b > 0) (h_area : (1/2) * a * b = 50) :
  a + b ≥ 20 ∧ (a + b = 20 ↔ a = 10 ∧ b = 10) :=
sorry

end right_triangle_min_leg_sum_l561_56139


namespace a_squared_gt_b_squared_neither_sufficient_nor_necessary_l561_56170

theorem a_squared_gt_b_squared_neither_sufficient_nor_necessary (a b : ℝ) :
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) ∧ ¬(∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end a_squared_gt_b_squared_neither_sufficient_nor_necessary_l561_56170


namespace daps_equivalent_to_24_dips_l561_56168

-- Define the units
variable (dap dop dip : ℝ)

-- Define the relationships between units
axiom dap_to_dop : 5 * dap = 4 * dop
axiom dop_to_dip : 3 * dop = 8 * dip

-- Theorem to prove
theorem daps_equivalent_to_24_dips : 
  24 * dip = (45/4) * dap := by sorry

end daps_equivalent_to_24_dips_l561_56168


namespace range_of_m_l561_56143

open Set Real

theorem range_of_m (m : ℝ) : 
  let A : Set ℝ := {x | -1 < x ∧ x < 7}
  let B : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 3 * m + 1}
  let p := A ∩ B = B
  let q := ∃! x, x^2 + 2*m*x + 2*m ≤ 0
  ¬(p ∨ q) → m ∈ Ici 2 :=
by
  sorry


end range_of_m_l561_56143


namespace yoojeong_drank_most_l561_56174

def yoojeong_milk : ℚ := 7/10
def eunji_milk : ℚ := 1/2
def yuna_milk : ℚ := 6/10

theorem yoojeong_drank_most : 
  yoojeong_milk > eunji_milk ∧ yoojeong_milk > yuna_milk := by
  sorry

end yoojeong_drank_most_l561_56174


namespace acute_angle_inequalities_l561_56117

theorem acute_angle_inequalities (α β : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_α_lt_β : α < β) : 
  (α - Real.sin α < β - Real.sin β) ∧ 
  (Real.tan α - α < Real.tan β - β) := by
  sorry

end acute_angle_inequalities_l561_56117


namespace M_is_solution_set_inequality_holds_l561_56171

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1|

-- Define the set M
def M : Set ℝ := {x | x < -1 ∨ x > 1}

-- Statement 1: M is the solution set for f(x) < |2x+1| - 1
theorem M_is_solution_set : ∀ x : ℝ, x ∈ M ↔ f x < |2*x + 1| - 1 :=
sorry

-- Statement 2: For any a, b ∈ M, f(ab) > f(a) - f(-b)
theorem inequality_holds : ∀ a b : ℝ, a ∈ M → b ∈ M → f (a*b) > f a - f (-b) :=
sorry

end M_is_solution_set_inequality_holds_l561_56171


namespace water_trough_theorem_l561_56164

/-- Calculates the final amount of water in a trough after a given number of days -/
def water_trough_calculation (initial_amount : ℝ) (evaporation_rate : ℝ) (refill_rate : ℝ) (days : ℕ) : ℝ :=
  initial_amount - (evaporation_rate - refill_rate) * days

/-- Theorem stating the final amount of water in the trough after 45 days -/
theorem water_trough_theorem :
  water_trough_calculation 350 1 0.4 45 = 323 := by
  sorry

#eval water_trough_calculation 350 1 0.4 45

end water_trough_theorem_l561_56164


namespace two_cars_total_distance_l561_56163

/-- Proves that given two cars with specified fuel efficiencies and consumption,
    the total distance driven is 1750 miles. -/
theorem two_cars_total_distance
  (efficiency1 : ℝ) (efficiency2 : ℝ) (total_consumption : ℝ) (consumption1 : ℝ)
  (h1 : efficiency1 = 25)
  (h2 : efficiency2 = 40)
  (h3 : total_consumption = 55)
  (h4 : consumption1 = 30) :
  efficiency1 * consumption1 + efficiency2 * (total_consumption - consumption1) = 1750 :=
by sorry

end two_cars_total_distance_l561_56163


namespace triangle_side_length_l561_56178

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Given conditions
  c = Real.sqrt 3 →
  A = π / 4 →  -- 45° in radians
  C = π / 3 →  -- 60° in radians
  -- Law of Sines
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  -- Conclusion
  a = Real.sqrt 2 := by
sorry

end triangle_side_length_l561_56178


namespace circular_track_length_l561_56111

/-- The length of the circular track in meters -/
def track_length : ℝ := 480

/-- Alex's speed in meters per unit time -/
def alex_speed : ℝ := 4

/-- Jamie's speed in meters per unit time -/
def jamie_speed : ℝ := 3

/-- Distance Alex runs to first meeting point in meters -/
def alex_first_meeting : ℝ := 150

/-- Distance Jamie runs after first meeting to second meeting point in meters -/
def jamie_second_meeting : ℝ := 180

theorem circular_track_length :
  track_length = 480 ∧
  alex_speed / jamie_speed = 4 / 3 ∧
  alex_first_meeting = 150 ∧
  jamie_second_meeting = 180 ∧
  track_length / 2 + alex_first_meeting = 
    (track_length / 2 - alex_first_meeting) + jamie_second_meeting + track_length / 2 :=
by sorry

end circular_track_length_l561_56111


namespace fraction_simplification_l561_56157

theorem fraction_simplification :
  (154 : ℚ) / 10780 = 1 / 70 := by sorry

end fraction_simplification_l561_56157


namespace prime_sum_2019_power_l561_56109

theorem prime_sum_2019_power (p q : ℕ) : 
  Prime p → Prime q → p + q = 2019 → (p - 1)^(q - 1) = 1 ∨ (p - 1)^(q - 1) = 2016 := by
  sorry

end prime_sum_2019_power_l561_56109


namespace correct_relative_pronoun_l561_56180

/-- Represents a relative pronoun -/
inductive RelativePronoun
| When
| That
| Where
| Which

/-- Represents the context of an opportunity -/
structure OpportunityContext where
  universal : Bool
  independentOfAge : Bool
  independentOfProfession : Bool
  independentOfReligion : Bool
  independentOfBackground : Bool

/-- Represents the function of a relative pronoun in a sentence -/
structure PronounFunction where
  modifiesNoun : Bool
  describesCircumstances : Bool
  introducesAdjectiveClause : Bool

/-- Determines if a relative pronoun is correct for the given sentence -/
def isCorrectPronoun (pronoun : RelativePronoun) (context : OpportunityContext) (function : PronounFunction) : Prop :=
  context.universal ∧
  context.independentOfAge ∧
  context.independentOfProfession ∧
  context.independentOfReligion ∧
  context.independentOfBackground ∧
  function.modifiesNoun ∧
  function.describesCircumstances ∧
  function.introducesAdjectiveClause ∧
  pronoun = RelativePronoun.Where

theorem correct_relative_pronoun (context : OpportunityContext) (function : PronounFunction) :
  isCorrectPronoun RelativePronoun.Where context function :=
by sorry

end correct_relative_pronoun_l561_56180


namespace vitamin_a_intake_in_grams_l561_56191

/-- Conversion factor from grams to milligrams -/
def gram_to_mg : ℝ := 1000

/-- Conversion factor from milligrams to micrograms -/
def mg_to_μg : ℝ := 1000

/-- Daily intake of vitamin A for adult women in micrograms -/
def vitamin_a_intake : ℝ := 750

/-- Theorem stating that 750 micrograms is equal to 7.5 × 10^-4 grams -/
theorem vitamin_a_intake_in_grams :
  (vitamin_a_intake / (gram_to_mg * mg_to_μg)) = 7.5e-4 := by
  sorry

end vitamin_a_intake_in_grams_l561_56191


namespace unique_sundaes_count_l561_56156

/-- The number of flavors available -/
def n : ℕ := 8

/-- The number of flavors in each sundae -/
def k : ℕ := 2

/-- The number of unique two scoop sundaes -/
def unique_sundaes : ℕ := Nat.choose n k

theorem unique_sundaes_count : unique_sundaes = 28 := by
  sorry

end unique_sundaes_count_l561_56156


namespace otimes_properties_l561_56125

-- Define the operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Statement of the theorem
theorem otimes_properties :
  (otimes 2 (-2) = 6) ∧ 
  (∃ a b : ℝ, otimes a b ≠ otimes b a) ∧
  (∀ a b : ℝ, a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  (∃ a b : ℝ, otimes a b = 0 ∧ a ≠ 0) := by
  sorry


end otimes_properties_l561_56125


namespace relay_team_permutations_l561_56106

def team_size : ℕ := 4
def fixed_positions : ℕ := 1
def remaining_positions : ℕ := team_size - fixed_positions

theorem relay_team_permutations :
  Nat.factorial remaining_positions = 6 :=
by sorry

end relay_team_permutations_l561_56106


namespace workers_in_first_group_l561_56199

/-- Given two groups of workers building walls, this theorem proves the number of workers in the first group. -/
theorem workers_in_first_group 
  (wall_length_1 : ℝ) 
  (days_1 : ℝ) 
  (wall_length_2 : ℝ) 
  (days_2 : ℝ) 
  (workers_2 : ℕ) 
  (h1 : wall_length_1 = 66) 
  (h2 : days_1 = 12) 
  (h3 : wall_length_2 = 189.2) 
  (h4 : days_2 = 8) 
  (h5 : workers_2 = 86) :
  ∃ (workers_1 : ℕ), workers_1 = 57 ∧ 
    (workers_1 : ℝ) * days_1 * wall_length_2 = (workers_2 : ℝ) * days_2 * wall_length_1 :=
by sorry

end workers_in_first_group_l561_56199


namespace ratio_to_eight_l561_56148

theorem ratio_to_eight : ∃ x : ℚ, (5 : ℚ) / 1 = x / 8 ∧ x = 40 := by
  sorry

end ratio_to_eight_l561_56148


namespace sum_of_roots_cubic_equation_l561_56186

theorem sum_of_roots_cubic_equation :
  let f (x : ℝ) := 3 * x^3 - 6 * x^2 - 9 * x
  ∃ (r₁ r₂ r₃ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧ r₁ + r₂ + r₃ = 2 :=
by sorry

end sum_of_roots_cubic_equation_l561_56186


namespace asymptote_parabola_intersection_distance_l561_56126

/-- The distance between the two points where the asymptotes of the hyperbola x^2 - y^2 = 1
    intersect with the parabola y^2 = 4x is 8, given that one intersection point is the origin. -/
theorem asymptote_parabola_intersection_distance : 
  let hyperbola := fun (x y : ℝ) => x^2 - y^2 = 1
  let parabola := fun (x y : ℝ) => y^2 = 4*x
  let asymptote1 := fun (x : ℝ) => x
  let asymptote2 := fun (x : ℝ) => -x
  let O : ℝ × ℝ := (0, 0)
  let A : ℝ × ℝ := (4, 4)
  let B : ℝ × ℝ := (4, -4)
  (hyperbola O.1 O.2) ∧ 
  (parabola O.1 O.2) ∧
  (parabola A.1 A.2) ∧ 
  (parabola B.1 B.2) ∧
  (A.2 = asymptote1 A.1) ∧
  (B.2 = asymptote2 B.1) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
sorry

end asymptote_parabola_intersection_distance_l561_56126


namespace half_filled_cylindrical_tank_volume_l561_56110

/-- The volume of water in a half-filled cylindrical tank lying on its side -/
theorem half_filled_cylindrical_tank_volume
  (r : ℝ) -- radius of the tank
  (h : ℝ) -- height (length) of the tank
  (hr : r = 5) -- given radius is 5 feet
  (hh : h = 10) -- given height is 10 feet
  : (1 / 2 * π * r^2 * h) = 125 * π := by
  sorry

end half_filled_cylindrical_tank_volume_l561_56110


namespace power_of_four_exponent_l561_56103

theorem power_of_four_exponent (n : ℕ) (x : ℕ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^x) 
  (h2 : n = 17) : 
  x = 18 := by
  sorry

end power_of_four_exponent_l561_56103


namespace greatest_divisor_with_remainders_l561_56155

theorem greatest_divisor_with_remainders : Nat.gcd (1442 - 12) (1816 - 6) = 10 := by
  sorry

end greatest_divisor_with_remainders_l561_56155


namespace book_selection_theorem_l561_56190

theorem book_selection_theorem (math_books : Nat) (physics_books : Nat) : 
  math_books = 3 → physics_books = 2 → math_books * physics_books = 6 := by
  sorry

end book_selection_theorem_l561_56190


namespace trig_simplification_l561_56198

theorem trig_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 :=
by sorry

end trig_simplification_l561_56198


namespace square_sum_given_difference_and_product_l561_56146

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : (x - y)^2 = 49) (h2 : x * y = 8) : x^2 + y^2 = 65 := by
  sorry

end square_sum_given_difference_and_product_l561_56146


namespace quadratic_equation_property_l561_56105

/-- 
A quadratic equation with coefficients a, b, and c, where a ≠ 0,
satisfying a + b + c = 0 and having two equal real roots.
-/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  sum_zero : a + b + c = 0
  equal_roots : ∃ x : ℝ, ∀ y : ℝ, a * y^2 + b * y + c = 0 ↔ y = x

theorem quadratic_equation_property (eq : QuadraticEquation) : eq.a = eq.c := by
  sorry

end quadratic_equation_property_l561_56105


namespace rental_hours_proof_l561_56184

/-- Represents a bike rental service with a base cost and hourly rate. -/
structure BikeRental where
  baseCost : ℕ
  hourlyRate : ℕ

/-- Calculates the total cost for a given number of hours. -/
def totalCost (rental : BikeRental) (hours : ℕ) : ℕ :=
  rental.baseCost + rental.hourlyRate * hours

/-- Proves that for the given bike rental conditions and total cost, the number of hours rented is 9. -/
theorem rental_hours_proof (rental : BikeRental) 
    (h1 : rental.baseCost = 17)
    (h2 : rental.hourlyRate = 7)
    (h3 : totalCost rental 9 = 80) : 
  ∃ (hours : ℕ), totalCost rental hours = 80 ∧ hours = 9 := by
  sorry

#check rental_hours_proof

end rental_hours_proof_l561_56184


namespace shopkeeper_total_cards_l561_56151

/-- The number of complete decks of standard playing cards -/
def standard_decks : ℕ := 3

/-- The number of cards in a complete deck of standard playing cards -/
def cards_per_standard_deck : ℕ := 52

/-- The number of incomplete decks of tarot cards -/
def tarot_decks : ℕ := 2

/-- The number of cards in each incomplete tarot deck -/
def cards_per_tarot_deck : ℕ := 72

/-- The number of sets of trading cards -/
def trading_card_sets : ℕ := 5

/-- The number of cards in each trading card set -/
def cards_per_trading_set : ℕ := 100

/-- The number of additional random cards -/
def random_cards : ℕ := 27

/-- The total number of cards the shopkeeper has -/
def total_cards : ℕ := 
  standard_decks * cards_per_standard_deck + 
  tarot_decks * cards_per_tarot_deck + 
  trading_card_sets * cards_per_trading_set + 
  random_cards

theorem shopkeeper_total_cards : total_cards = 827 := by
  sorry

end shopkeeper_total_cards_l561_56151


namespace sum_of_fraction_is_correct_l561_56149

/-- The repeating decimal 0.̅14 as a real number -/
def repeating_decimal : ℚ := 14 / 99

/-- The sum of numerator and denominator of the fraction representation of 0.̅14 -/
def sum_of_fraction : ℕ := 113

/-- Theorem stating that the sum of numerator and denominator of 0.̅14 in lowest terms is 113 -/
theorem sum_of_fraction_is_correct : 
  ∃ (n d : ℕ), d ≠ 0 ∧ repeating_decimal = n / d ∧ Nat.gcd n d = 1 ∧ n + d = sum_of_fraction := by
  sorry

end sum_of_fraction_is_correct_l561_56149


namespace part_one_part_two_l561_56114

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 3|

-- Part 1
theorem part_one (m : ℝ) : 
  (∀ x, f m x > 2 ↔ 2 < x ∧ x < 4) → m = 3 := by sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, |x - a| ≥ f 3 x) → (a ≥ 6 ∨ a ≤ 0) := by sorry

end part_one_part_two_l561_56114


namespace mom_bought_packages_l561_56166

def shirts_per_package : ℕ := 6
def total_shirts : ℕ := 426

theorem mom_bought_packages : 
  ∃ (packages : ℕ), packages * shirts_per_package = total_shirts ∧ packages = 71 := by
  sorry

end mom_bought_packages_l561_56166


namespace restaurant_group_adults_l561_56187

/-- Calculates the number of adults in a restaurant group given the total bill, 
    number of children, and cost per meal. -/
theorem restaurant_group_adults 
  (total_bill : ℕ) 
  (num_children : ℕ) 
  (cost_per_meal : ℕ) : 
  total_bill = 56 → 
  num_children = 5 → 
  cost_per_meal = 8 → 
  (total_bill - num_children * cost_per_meal) / cost_per_meal = 2 := by
  sorry

end restaurant_group_adults_l561_56187


namespace locus_and_line_equations_l561_56107

-- Define the circles F₁ and F₂
def F₁ (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def F₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 / 16 + y^2 / 12 = 1 ∧ x ≠ -4

-- Define the line l
def l (x y : ℝ) : Prop := 3 * x - 2 * y - 8 = 0

-- Define the point Q
def Q : ℝ × ℝ := (2, -1)

-- Theorem statement
theorem locus_and_line_equations :
  ∃ (M : ℝ × ℝ → Prop),
    (∀ x y, M (x, y) → F₁ x y) ∧
    (∀ x y, M (x, y) → F₂ x y) ∧
    (∀ x y, C x y ↔ ∃ r > 0, M (x, y) ∧ r = 2) ∧
    (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ l A.1 A.2 ∧ l B.1 B.2 ∧ Q = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :=
sorry

end locus_and_line_equations_l561_56107


namespace sams_remaining_seashells_l561_56179

-- Define the initial number of seashells Sam found
def initial_seashells : ℕ := 35

-- Define the number of seashells Sam gave to Joan
def given_away : ℕ := 18

-- Theorem stating how many seashells Sam has now
theorem sams_remaining_seashells : 
  initial_seashells - given_away = 17 := by sorry

end sams_remaining_seashells_l561_56179


namespace line_slope_proof_l561_56144

/-- Given two vectors in a Cartesian coordinate plane and a line with certain properties,
    prove that the slope of the line is 2/5. -/
theorem line_slope_proof (OA OB : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  OA = (1, 4) →
  OB = (-3, 1) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y = k * x) →
  (∀ (v : ℝ × ℝ), v ∈ l → v.2 > 0) →
  (∀ (C : ℝ × ℝ), C ∈ l → OA.1 * C.1 + OA.2 * C.2 = OB.1 * C.1 + OB.2 * C.2) →
  (∃ (k : ℝ), ∀ (x y : ℝ), (x, y) ∈ l ↔ y = k * x ∧ k = 2/5) :=
by sorry

end line_slope_proof_l561_56144


namespace john_lewis_meeting_point_l561_56196

/-- Represents the journey between two cities --/
structure Journey where
  distance : ℝ
  johnSpeed : ℝ
  lewisOutboundSpeed : ℝ
  lewisReturnSpeed : ℝ
  johnBreakFrequency : ℝ
  johnBreakDuration : ℝ
  lewisBreakFrequency : ℝ
  lewisBreakDuration : ℝ

/-- Calculates the meeting point of John and Lewis --/
def meetingPoint (j : Journey) : ℝ :=
  sorry

/-- Theorem stating the meeting point of John and Lewis --/
theorem john_lewis_meeting_point :
  let j : Journey := {
    distance := 240,
    johnSpeed := 40,
    lewisOutboundSpeed := 60,
    lewisReturnSpeed := 50,
    johnBreakFrequency := 2,
    johnBreakDuration := 0.25,
    lewisBreakFrequency := 2.5,
    lewisBreakDuration := 1/3
  }
  ∃ (ε : ℝ), ε > 0 ∧ |meetingPoint j - 23.33| < ε :=
sorry

end john_lewis_meeting_point_l561_56196


namespace greatest_three_digit_multiple_of_17_l561_56169

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by sorry

end greatest_three_digit_multiple_of_17_l561_56169


namespace f_derivative_l561_56138

noncomputable def f (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem f_derivative (x : ℝ) : 
  deriv f x = -2 * Real.exp x * (Real.sin x + Real.cos x) := by
  sorry

end f_derivative_l561_56138


namespace toothpick_grid_50_40_l561_56145

/-- Calculates the total number of toothpicks in a rectangular grid -/
def toothpick_grid_count (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem stating that a 50x40 toothpick grid contains 4090 toothpicks -/
theorem toothpick_grid_50_40 :
  toothpick_grid_count 50 40 = 4090 := by
  sorry

end toothpick_grid_50_40_l561_56145


namespace function_identity_l561_56102

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 1 > 0)
  (h2 : ∀ m n : ℕ, f (m^2 + n^2) = (f m)^2 + (f n)^2) :
  ∀ n : ℕ, f n = n :=
by sorry

end function_identity_l561_56102


namespace oranges_for_juice_l561_56167

/-- Given that 18 oranges make 27 liters of orange juice, 
    prove that 6 oranges are needed to make 9 liters of orange juice. -/
theorem oranges_for_juice (oranges : ℕ) (juice : ℕ) 
  (h : 18 * juice = 27 * oranges) : 
  6 * juice = 9 * oranges :=
by sorry

end oranges_for_juice_l561_56167


namespace shoes_lost_example_l561_56130

/-- Given an initial number of shoe pairs and a maximum number of remaining pairs,
    calculate the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (max_remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * max_remaining_pairs

/-- Theorem: Given 27 initial pairs of shoes and 22 maximum remaining pairs,
    the number of individual shoes lost is 10. -/
theorem shoes_lost_example : shoes_lost 27 22 = 10 := by
  sorry

end shoes_lost_example_l561_56130


namespace factor_implies_values_l561_56176

theorem factor_implies_values (p q : ℝ) : 
  (∃ (a b c : ℝ), (X^4 + p*X^2 + q) = (X^2 + 2*X + 5) * (a*X^2 + b*X + c)) → 
  p = 6 ∧ q = 25 := by
  sorry

end factor_implies_values_l561_56176


namespace parallelogram_area_l561_56150

/-- The area of a parallelogram with given side lengths and angle between them -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (ha : a = 15) (hb : b = 20) (hθ : θ = 35 * π / 180) :
  abs (a * b * Real.sin θ - 172.08) < 0.01 := by
  sorry

end parallelogram_area_l561_56150


namespace books_from_first_shop_l561_56108

theorem books_from_first_shop (total_spent : ℕ) (second_shop_books : ℕ) (avg_price : ℕ) :
  total_spent = 768 →
  second_shop_books = 22 →
  avg_price = 12 →
  ∃ first_shop_books : ℕ,
    first_shop_books = 42 ∧
    total_spent = avg_price * (first_shop_books + second_shop_books) :=
by
  sorry

end books_from_first_shop_l561_56108


namespace remainder_98_35_mod_100_l561_56101

theorem remainder_98_35_mod_100 : 98^35 ≡ -24 [ZMOD 100] := by sorry

end remainder_98_35_mod_100_l561_56101


namespace toys_after_game_purchase_l561_56183

theorem toys_after_game_purchase (initial_amount : ℕ) (game_cost : ℕ) (toy_cost : ℕ) : 
  initial_amount = 57 → game_cost = 27 → toy_cost = 6 → 
  (initial_amount - game_cost) / toy_cost = 5 := by
  sorry

end toys_after_game_purchase_l561_56183


namespace area_traced_on_concentric_spheres_l561_56136

/-- Theorem: Area traced by a sphere on concentric spheres
  Given:
  - Two concentric spheres with radii R₁ and R₂
  - A smaller sphere that traces areas on both spheres
  - The area traced on the inner sphere is A₁
  Prove:
  The area A₂ traced on the outer sphere is equal to A₁ * (R₂/R₁)²
-/
theorem area_traced_on_concentric_spheres
  (R₁ R₂ A₁ : ℝ)
  (h₁ : 0 < R₁)
  (h₂ : 0 < R₂)
  (h₃ : 0 < A₁)
  (h₄ : R₁ < R₂) :
  ∃ A₂ : ℝ, A₂ = A₁ * (R₂/R₁)^2 := by
  sorry

end area_traced_on_concentric_spheres_l561_56136


namespace linear_program_coefficient_l561_56158

/-- Given a set of linear constraints and a linear objective function,
    prove that the value of the coefficient m in the objective function
    is -2/3 when the minimum value of the function is -3. -/
theorem linear_program_coefficient (x y : ℝ) (m : ℝ) : 
  (x + y - 2 ≥ 0) →
  (x - y + 1 ≥ 0) →
  (x ≤ 3) →
  (∀ x y, x + y - 2 ≥ 0 → x - y + 1 ≥ 0 → x ≤ 3 → m * x + y ≥ -3) →
  (∃ x y, x + y - 2 ≥ 0 ∧ x - y + 1 ≥ 0 ∧ x ≤ 3 ∧ m * x + y = -3) →
  m = -2/3 := by
sorry

end linear_program_coefficient_l561_56158


namespace cooking_shopping_combinations_l561_56127

theorem cooking_shopping_combinations (n : ℕ) (k : ℕ) (h : n = 5 ∧ k = 3) : 
  Nat.choose n k = 10 := by
  sorry

end cooking_shopping_combinations_l561_56127


namespace square_tiling_for_n_ge_5_l561_56104

/-- A rectangle is dominant if it is similar to a 2 × 1 rectangle -/
def DominantRectangle (r : Rectangle) : Prop := sorry

/-- A tiling of a square with n dominant rectangles -/
def SquareTiling (n : ℕ) : Prop := sorry

/-- Theorem: For all integers n ≥ 5, it is possible to tile a square with n dominant rectangles -/
theorem square_tiling_for_n_ge_5 (n : ℕ) (h : n ≥ 5) : SquareTiling n := by
  sorry

end square_tiling_for_n_ge_5_l561_56104


namespace igloo_bottom_row_bricks_l561_56118

/-- Represents the structure of an igloo --/
structure Igloo where
  total_rows : ℕ
  top_row_bricks : ℕ
  total_bricks : ℕ

/-- Calculates the number of bricks in each row of the bottom half of the igloo --/
def bottom_row_bricks (igloo : Igloo) : ℕ :=
  let bottom_rows := igloo.total_rows / 2
  let top_bricks := bottom_rows * igloo.top_row_bricks
  (igloo.total_bricks - top_bricks) / bottom_rows

/-- Theorem stating that for the given igloo specifications, 
    the number of bricks in each row of the bottom half is 12 --/
theorem igloo_bottom_row_bricks :
  let igloo : Igloo := { total_rows := 10, top_row_bricks := 8, total_bricks := 100 }
  bottom_row_bricks igloo = 12 := by
  sorry


end igloo_bottom_row_bricks_l561_56118


namespace square_implies_composite_l561_56165

theorem square_implies_composite (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_square : ∃ n : ℕ, x^2 + x*y - y = n^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ x + y + 1 = a * b :=
by sorry

end square_implies_composite_l561_56165


namespace two_cars_gas_consumption_l561_56153

/-- Represents the gas consumption and mileage of a car for a week -/
structure CarData where
  mpg : ℝ
  gallons_consumed : ℝ
  miles_driven : ℝ

/-- Calculates the total gas consumption for two cars in a week -/
def total_gas_consumption (car1 : CarData) (car2 : CarData) : ℝ :=
  car1.gallons_consumed + car2.gallons_consumed

/-- Theorem stating the total gas consumption of two cars given specific conditions -/
theorem two_cars_gas_consumption
  (car1 : CarData)
  (car2 : CarData)
  (h1 : car1.mpg = 25)
  (h2 : car2.mpg = 40)
  (h3 : car1.gallons_consumed = 30)
  (h4 : car1.miles_driven + car2.miles_driven = 1825)
  (h5 : car1.miles_driven = car1.mpg * car1.gallons_consumed)
  (h6 : car2.miles_driven = car2.mpg * car2.gallons_consumed) :
  total_gas_consumption car1 car2 = 56.875 := by
  sorry

#eval Float.round ((25 : Float) * 30 + (1825 - 25 * 30) / 40) * 1000 / 1000

end two_cars_gas_consumption_l561_56153


namespace derivative_of_linear_function_l561_56192

theorem derivative_of_linear_function (x : ℝ) :
  let f : ℝ → ℝ := λ x => 3 * x + 5
  HasDerivAt f 3 x := by sorry

end derivative_of_linear_function_l561_56192


namespace product_of_geometric_terms_l561_56193

def arithmetic_sequence (n : ℕ) : ℕ := 2 * n - 1

def geometric_sequence (n : ℕ) : ℕ := 2^(n - 1)

theorem product_of_geometric_terms : 
  geometric_sequence (arithmetic_sequence 1) * 
  geometric_sequence (arithmetic_sequence 3) * 
  geometric_sequence (arithmetic_sequence 5) = 4096 := by
sorry

end product_of_geometric_terms_l561_56193


namespace inscribe_square_in_circle_l561_56113

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- A line in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if four points form a square -/
def is_square (p1 p2 p3 p4 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d34 := (p3.x - p4.x)^2 + (p3.y - p4.y)^2
  let d41 := (p4.x - p1.x)^2 + (p4.y - p1.y)^2
  let d13 := (p1.x - p3.x)^2 + (p1.y - p3.y)^2
  let d24 := (p2.x - p4.x)^2 + (p2.y - p4.y)^2
  (d12 = d23) ∧ (d23 = d34) ∧ (d34 = d41) ∧ (d13 = d24)

/-- Check if a point lies on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Construct a line through two points -/
def line_through_points (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  let l := line_through_points p1 p2
  l.a * p3.x + l.b * p3.y + l.c = 0

/-- Theorem: Given a circle with a marked center, it is possible to construct
    four points on the circle that form the vertices of a square using only
    straightedge constructions -/
theorem inscribe_square_in_circle (c : Circle) :
  ∃ (p1 p2 p3 p4 : Point),
    on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c ∧ on_circle p4 c ∧
    is_square p1 p2 p3 p4 :=
sorry

end inscribe_square_in_circle_l561_56113


namespace trigonometric_calculation_quadratic_equation_solution_l561_56128

-- Problem 1
theorem trigonometric_calculation :
  3 * Real.tan (45 * π / 180) - (1 / 3)⁻¹ + (Real.sin (30 * π / 180) - 2022)^0 + |Real.cos (30 * π / 180) - Real.sqrt 3 / 2| = 1 := by
  sorry

-- Problem 2
theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x * (x + 3) - 5 * (x + 3)
  (f 5 = 0 ∧ f (-3) = 0) ∧ ∀ x, f x = 0 → x = 5 ∨ x = -3 := by
  sorry

end trigonometric_calculation_quadratic_equation_solution_l561_56128


namespace f_value_at_2017_5_l561_56115

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def f_on_unit_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

theorem f_value_at_2017_5 (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period_2 f) 
  (h_unit : f_on_unit_interval f) : 
  f 2017.5 = -1/2 := by
sorry

end f_value_at_2017_5_l561_56115


namespace inequality_proof_l561_56160

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 2*x*y*z ∧ x*y + y*z + z*x - 2*x*y*z ≤ 7/27 := by
  sorry

end inequality_proof_l561_56160


namespace spinner_direction_l561_56100

-- Define the possible directions
inductive Direction
  | North
  | East
  | South
  | West

-- Define the rotation function
def rotate (initial : Direction) (clockwise : Rat) (counterclockwise : Rat) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction :
  let initial_direction := Direction.North
  let clockwise_rotation : Rat := 7/4
  let counterclockwise_rotation : Rat := 5/2
  rotate initial_direction clockwise_rotation counterclockwise_rotation = Direction.East :=
by sorry

end spinner_direction_l561_56100


namespace limestone_amount_l561_56195

/-- Represents the composition of a cement compound -/
structure CementCompound where
  limestone : ℝ
  shale : ℝ
  total_weight : ℝ
  limestone_cost : ℝ
  shale_cost : ℝ
  compound_cost : ℝ

/-- Theorem stating the correct amount of limestone in the compound -/
theorem limestone_amount (c : CementCompound) 
  (h1 : c.total_weight = 100)
  (h2 : c.limestone_cost = 3)
  (h3 : c.shale_cost = 5)
  (h4 : c.compound_cost = 4.25)
  (h5 : c.limestone + c.shale = c.total_weight)
  (h6 : c.limestone * c.limestone_cost + c.shale * c.shale_cost = c.total_weight * c.compound_cost) :
  c.limestone = 37.5 := by
  sorry

end limestone_amount_l561_56195


namespace exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary_l561_56175

/-- Represents the possible outcomes when selecting 2 students from a group of 2 boys and 2 girls -/
inductive Outcome
  | TwoBoys
  | OneGirlOneBoy
  | TwoGirls

/-- The sample space of all possible outcomes -/
def SampleSpace : Set Outcome := {Outcome.TwoBoys, Outcome.OneGirlOneBoy, Outcome.TwoGirls}

/-- The event "Exactly 1 girl" -/
def ExactlyOneGirl : Set Outcome := {Outcome.OneGirlOneBoy}

/-- The event "Exactly 2 girls" -/
def ExactlyTwoGirls : Set Outcome := {Outcome.TwoGirls}

/-- Theorem stating that "Exactly 1 girl" and "Exactly 2 girls" are mutually exclusive but not contrary -/
theorem exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary :
  (ExactlyOneGirl ∩ ExactlyTwoGirls = ∅) ∧
  (ExactlyOneGirl ∪ ExactlyTwoGirls ≠ SampleSpace) := by
  sorry

end exactly_one_girl_and_exactly_two_girls_mutually_exclusive_but_not_contrary_l561_56175


namespace farm_area_calculation_l561_56181

/-- The total area of a farm with given sections and section area -/
def farm_total_area (num_sections : ℕ) (section_area : ℕ) : ℕ :=
  num_sections * section_area

/-- Theorem: The total area of a farm with 5 sections of 60 acres each is 300 acres -/
theorem farm_area_calculation :
  farm_total_area 5 60 = 300 := by
  sorry

end farm_area_calculation_l561_56181


namespace inequality_proof_l561_56131

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b / a > b - a / b) ∧ (1 / a + c < 1 / b + c) := by
  sorry

end inequality_proof_l561_56131


namespace complex_product_theorem_l561_56189

theorem complex_product_theorem (α₁ α₂ α₃ : ℝ) : 
  let z₁ : ℂ := Complex.exp (I * α₁)
  let z₂ : ℂ := Complex.exp (I * α₂)
  let z₃ : ℂ := Complex.exp (I * α₃)
  z₁ * z₂ = Complex.exp (I * (α₁ + α₂)) →
  z₂ * z₃ = Complex.exp (I * (α₂ + α₃)) →
  z₁ * z₂ * z₃ = Complex.exp (I * (α₁ + α₂ + α₃)) :=
by sorry

end complex_product_theorem_l561_56189


namespace percentage_of_women_employees_l561_56172

theorem percentage_of_women_employees (men_with_degree : ℝ) (men_without_degree : ℕ) (total_women : ℕ) : 
  men_with_degree = 0.75 * (men_with_degree + men_without_degree) →
  men_without_degree = 8 →
  total_women = 48 →
  (total_women : ℝ) / ((men_with_degree + men_without_degree : ℝ) + total_women) * 100 = 60 := by
sorry

end percentage_of_women_employees_l561_56172


namespace y_intercept_of_parallel_line_through_point_l561_56182

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Check if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line :=
  { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line_through_point :
  ∃ (b : Line), 
    parallel b givenLine ∧ 
    pointOnLine b 4 (-2) ∧ 
    b.yIntercept = 10 := by
  sorry

end y_intercept_of_parallel_line_through_point_l561_56182


namespace arithmetic_sequence_sum_property_l561_56137

def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℤ)
  (h_arith : arithmeticSequence a)
  (h_a9 : a 9 = -2012)
  (h_a17 : a 17 = -2012) :
  a 1 + a 25 < 0 := by
  sorry

end arithmetic_sequence_sum_property_l561_56137


namespace composite_function_solution_l561_56194

theorem composite_function_solution (h k : ℝ → ℝ) (b : ℝ) :
  (∀ x, h x = x / 3 + 2) →
  (∀ x, k x = 5 - 2 * x) →
  h (k b) = 4 →
  b = -1 / 2 := by
sorry

end composite_function_solution_l561_56194


namespace proposition_all_lines_proposition_line_planes_proposition_all_planes_l561_56162

-- Define the basic types
inductive GeometricObject
| Line
| Plane

-- Define the relationships
def perpendicular (a b : GeometricObject) : Prop := sorry
def parallel (a b : GeometricObject) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricObject) : Prop :=
  perpendicular x y ∧ parallel y z → perpendicular x z

-- Theorem for case 1: all lines
theorem proposition_all_lines :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Line ∧ 
  y = GeometricObject.Line ∧ 
  z = GeometricObject.Line →
  proposition x y z :=
sorry

-- Theorem for case 2: x is line, y and z are planes
theorem proposition_line_planes :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Line ∧ 
  y = GeometricObject.Plane ∧ 
  z = GeometricObject.Plane →
  proposition x y z :=
sorry

-- Theorem for case 3: all planes
theorem proposition_all_planes :
  ∀ x y z : GeometricObject,
  x = GeometricObject.Plane ∧ 
  y = GeometricObject.Plane ∧ 
  z = GeometricObject.Plane →
  proposition x y z :=
sorry

end proposition_all_lines_proposition_line_planes_proposition_all_planes_l561_56162


namespace oreo_multiple_l561_56140

def total_oreos : ℕ := 52
def james_oreos : ℕ := 43

theorem oreo_multiple :
  ∃ (multiple : ℕ) (jordan_oreos : ℕ),
    james_oreos = multiple * jordan_oreos + 7 ∧
    total_oreos = james_oreos + jordan_oreos ∧
    multiple = 4 := by
  sorry

end oreo_multiple_l561_56140


namespace shaded_area_between_circles_l561_56159

theorem shaded_area_between_circles (R : ℝ) (r : ℝ) : 
  R = 10 → r = 4 → π * R^2 - 2 * π * r^2 = 68 * π := by
  sorry

end shaded_area_between_circles_l561_56159


namespace remainder_98_power_50_mod_50_l561_56141

theorem remainder_98_power_50_mod_50 : 98^50 % 50 = 24 := by
  sorry

end remainder_98_power_50_mod_50_l561_56141


namespace complex_equation_solution_l561_56132

/-- Given that (z - 2i)(2 - i) = 5, prove that z = 2 + 3i -/
theorem complex_equation_solution (z : ℂ) (h : (z - 2*Complex.I)*(2 - Complex.I) = 5) :
  z = 2 + 3*Complex.I := by
  sorry

end complex_equation_solution_l561_56132


namespace yellow_marble_probability_l561_56188

/-- Represents a bag of marbles -/
structure Bag where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def Bag.total (b : Bag) : ℕ := b.white + b.black + b.yellow + b.blue

/-- Calculates the probability of drawing a specific color from a bag -/
def Bag.prob (b : Bag) (color : ℕ) : ℚ :=
  (color : ℚ) / (b.total : ℚ)

/-- The configuration of Bag A -/
def bagA : Bag := { white := 5, black := 6 }

/-- The configuration of Bag B -/
def bagB : Bag := { yellow := 8, blue := 6 }

/-- The configuration of Bag C -/
def bagC : Bag := { yellow := 3, blue := 9 }

/-- The probability of drawing a yellow marble as the second marble -/
def yellowProbability : ℚ :=
  bagA.prob bagA.white * bagB.prob bagB.yellow +
  bagA.prob bagA.black * bagC.prob bagC.yellow

theorem yellow_marble_probability :
  yellowProbability = 61 / 154 := by
  sorry

end yellow_marble_probability_l561_56188


namespace johns_money_to_mother_l561_56122

theorem johns_money_to_mother (initial_amount : ℝ) (father_fraction : ℝ) (amount_left : ℝ) :
  initial_amount = 200 →
  father_fraction = 3 / 10 →
  amount_left = 65 →
  ∃ (mother_fraction : ℝ), 
    mother_fraction = 3 / 8 ∧
    amount_left = initial_amount * (1 - (mother_fraction + father_fraction)) :=
by sorry

end johns_money_to_mother_l561_56122


namespace max_ships_on_board_l561_56116

/-- Represents a ship placement on a board -/
structure ShipPlacement where
  board_size : Nat × Nat
  ship_size : Nat × Nat
  ship_count : Nat

/-- Checks if a ship placement is valid -/
def is_valid_placement (p : ShipPlacement) : Prop :=
  p.board_size.1 = 10 ∧
  p.board_size.2 = 10 ∧
  p.ship_size.1 = 1 ∧
  p.ship_size.2 = 4 ∧
  p.ship_count ≤ 25

/-- Theorem stating the maximum number of ships -/
theorem max_ships_on_board :
  ∃ (p : ShipPlacement), is_valid_placement p ∧
    ∀ (q : ShipPlacement), is_valid_placement q → q.ship_count ≤ p.ship_count :=
sorry

end max_ships_on_board_l561_56116


namespace reach_probability_is_5_128_l561_56197

/-- Represents a point in the 2D coordinate plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a step direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of taking a specific step -/
def stepProbability : ℚ := 1 / 4

/-- The starting point -/
def start : Point := ⟨0, 0⟩

/-- The target point -/
def target : Point := ⟨3, 1⟩

/-- The maximum number of steps allowed -/
def maxSteps : ℕ := 5

/-- Calculates the probability of reaching the target point from the start point
    in at most maxSteps steps -/
def reachProbability (start target : Point) (maxSteps : ℕ) : ℚ :=
  sorry

theorem reach_probability_is_5_128 :
  reachProbability start target maxSteps = 5 / 128 :=
sorry

end reach_probability_is_5_128_l561_56197


namespace cube_difference_l561_56161

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 53) : a^3 - b^3 = 385 := by
  sorry

end cube_difference_l561_56161


namespace total_ear_muffs_bought_l561_56177

theorem total_ear_muffs_bought (before_december : ℕ) (during_december : ℕ)
  (h1 : before_december = 1346)
  (h2 : during_december = 6444) :
  before_december + during_december = 7790 := by
  sorry

end total_ear_muffs_bought_l561_56177


namespace equation_solution_l561_56121

theorem equation_solution :
  ∀ x : ℚ,
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3/4 := by
sorry

end equation_solution_l561_56121


namespace chocolate_distribution_l561_56152

theorem chocolate_distribution (total_chocolates : ℕ) (boys_chocolates : ℕ) (girls_chocolates : ℕ) 
  (num_boys : ℕ) (num_girls : ℕ) :
  total_chocolates = 3000 →
  boys_chocolates = 2 →
  girls_chocolates = 3 →
  num_boys = 60 →
  num_girls = 60 →
  num_boys * boys_chocolates + num_girls * girls_chocolates = total_chocolates →
  num_boys + num_girls = 120 := by
sorry

end chocolate_distribution_l561_56152


namespace xiao_ming_current_age_l561_56133

/-- Xiao Ming's age this year -/
def xiao_ming_age : ℕ := sorry

/-- Xiao Ming's mother's age this year -/
def mother_age : ℕ := sorry

/-- Xiao Ming's age three years from now -/
def xiao_ming_age_future : ℕ := sorry

/-- Xiao Ming's mother's age three years from now -/
def mother_age_future : ℕ := sorry

/-- The theorem stating Xiao Ming's age this year -/
theorem xiao_ming_current_age :
  (mother_age = 3 * xiao_ming_age) ∧
  (mother_age_future = 2 * xiao_ming_age_future + 10) ∧
  (xiao_ming_age_future = xiao_ming_age + 3) ∧
  (mother_age_future = mother_age + 3) →
  xiao_ming_age = 13 := by
  sorry

end xiao_ming_current_age_l561_56133


namespace savings_duration_l561_56112

/-- Thomas and Joseph's savings problem -/
theorem savings_duration : 
  ∀ (thomas_monthly joseph_monthly total_savings : ℚ),
  thomas_monthly = 40 →
  joseph_monthly = (3/5) * thomas_monthly →
  total_savings = 4608 →
  ∃ (months : ℕ), 
    (thomas_monthly + joseph_monthly) * months = total_savings ∧ 
    months = 72 := by
  sorry

end savings_duration_l561_56112


namespace orange_bucket_difference_l561_56119

/-- Proves that the difference between the number of oranges in the second and first buckets is 17 -/
theorem orange_bucket_difference :
  ∀ (second_bucket : ℕ),
  22 + second_bucket + (second_bucket - 11) = 89 →
  second_bucket - 22 = 17 :=
by
  sorry

end orange_bucket_difference_l561_56119


namespace evaluate_sqrt_fraction_l561_56147

theorem evaluate_sqrt_fraction (x : ℝ) (h : x < -1) :
  Real.sqrt (x / (1 - (x - 2) / (x + 1))) = -x * (x + 1) / Real.sqrt 3 := by
  sorry

end evaluate_sqrt_fraction_l561_56147


namespace multiply_and_add_equality_l561_56120

theorem multiply_and_add_equality : 52 * 46 + 104 * 52 = 7800 := by
  sorry

end multiply_and_add_equality_l561_56120


namespace round_trip_speed_l561_56154

theorem round_trip_speed (x : ℝ) : 
  x > 0 →
  (2 : ℝ) / ((1 / x) + (1 / 3)) = 5 →
  x = 15 := by
sorry

end round_trip_speed_l561_56154


namespace fraction_meaningful_l561_56185

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 5)) ↔ x ≠ 5 := by sorry

end fraction_meaningful_l561_56185


namespace greatest_integer_fraction_l561_56124

theorem greatest_integer_fraction (x : ℤ) : (5 : ℚ) / 8 > (x : ℚ) / 15 ↔ x ≤ 9 := by sorry

end greatest_integer_fraction_l561_56124


namespace cake_distribution_l561_56173

theorem cake_distribution (total_cakes : ℕ) (num_children : ℕ) : 
  total_cakes = 18 → num_children = 3 → 
  ∃ (oldest middle youngest : ℕ),
    oldest = (2 * total_cakes / 5 : ℕ) ∧
    middle = total_cakes / 3 ∧
    youngest = total_cakes - (oldest + middle) ∧
    oldest = 7 ∧ middle = 6 ∧ youngest = 5 := by
  sorry

#check cake_distribution

end cake_distribution_l561_56173


namespace cubic_factorization_l561_56135

theorem cubic_factorization (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by
  sorry

end cubic_factorization_l561_56135


namespace sales_tax_rate_zero_l561_56142

theorem sales_tax_rate_zero (sale_price_with_tax : ℝ) (profit_percentage : ℝ) (cost_price : ℝ)
  (h1 : sale_price_with_tax = 616)
  (h2 : profit_percentage = 16)
  (h3 : cost_price = 531.03) :
  let profit := (profit_percentage / 100) * cost_price
  let sale_price_before_tax := cost_price + profit
  let sales_tax_rate := ((sale_price_with_tax - sale_price_before_tax) / sale_price_before_tax) * 100
  sales_tax_rate = 0 := by sorry

end sales_tax_rate_zero_l561_56142


namespace quadratic_function_sum_of_coefficients_l561_56129

theorem quadratic_function_sum_of_coefficients 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : (1 : ℝ) = a * (1 : ℝ)^2 + b * (1 : ℝ) - 1) : 
  a + b = 2 :=
by sorry

end quadratic_function_sum_of_coefficients_l561_56129


namespace smallest_m_divisible_by_15_l561_56134

/-- The largest prime with 2015 digits -/
def q : ℕ := sorry

theorem smallest_m_divisible_by_15 :
  ∃ m : ℕ, m > 0 ∧ 15 ∣ (q^2 - m) ∧ ∀ k : ℕ, 0 < k ∧ k < m → ¬(15 ∣ (q^2 - k)) :=
by sorry

end smallest_m_divisible_by_15_l561_56134


namespace friendly_group_has_complete_subgroup_l561_56123

/-- Represents the property of two people knowing each other -/
def knows (people : Type) : people → people → Prop := sorry

/-- A group of people satisfying the condition that among any three, two know each other -/
structure FriendlyGroup (people : Type) where
  size : Nat
  members : Finset people
  size_eq : members.card = size
  friendly : ∀ (a b c : people), a ∈ members → b ∈ members → c ∈ members →
    a ≠ b → b ≠ c → a ≠ c → (knows people a b ∨ knows people b c ∨ knows people a c)

/-- A complete subgroup where every pair knows each other -/
def CompleteSubgroup {people : Type} (group : FriendlyGroup people) (subgroup : Finset people) : Prop :=
  subgroup ⊆ group.members ∧ ∀ (a b : people), a ∈ subgroup → b ∈ subgroup → a ≠ b → knows people a b

/-- The main theorem: In a group of 9 people satisfying the friendly condition,
    there exists a complete subgroup of 4 people -/
theorem friendly_group_has_complete_subgroup 
  {people : Type} (group : FriendlyGroup people) (h : group.size = 9) :
  ∃ (subgroup : Finset people), subgroup.card = 4 ∧ CompleteSubgroup group subgroup := by
  sorry

end friendly_group_has_complete_subgroup_l561_56123
