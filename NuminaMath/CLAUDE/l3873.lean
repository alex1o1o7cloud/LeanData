import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3873_387355

theorem divisibility_implies_equality (a b : ℕ+) :
  (4 * a * b - 1) ∣ (4 * a^2 - 1)^2 → a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3873_387355


namespace NUMINAMATH_CALUDE_cyclist_energized_time_l3873_387335

/-- Given a cyclist who rides at different speeds when energized and exhausted,
    prove the time spent energized for a specific total distance and time. -/
theorem cyclist_energized_time
  (speed_energized : ℝ)
  (speed_exhausted : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (h_speed_energized : speed_energized = 22)
  (h_speed_exhausted : speed_exhausted = 15)
  (h_total_distance : total_distance = 154)
  (h_total_time : total_time = 9)
  : ∃ (time_energized : ℝ),
    time_energized * speed_energized +
    (total_time - time_energized) * speed_exhausted = total_distance ∧
    time_energized = 19 / 7 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_energized_time_l3873_387335


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3873_387359

/-- Given a binomial expansion (2x + 1/√x)^n where n is a positive integer,
    if the ratio of binomial coefficients of the second term to the third term is 2:5,
    then n = 6, the coefficient of x^3 is 240, and the sum of binomial terms is 728 -/
theorem binomial_expansion_properties (n : ℕ+) :
  (Nat.choose n 1 : ℚ) / (Nat.choose n 2 : ℚ) = 2 / 5 →
  (n = 6 ∧
   (Nat.choose 6 2 : ℕ) * 2^4 = 240 ∧
   (2^6 * Nat.choose 6 0 + 2^5 * Nat.choose 6 1 + 2^4 * Nat.choose 6 2 +
    2^3 * Nat.choose 6 3 + 2^2 * Nat.choose 6 4 + 2 * Nat.choose 6 5) = 728) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3873_387359


namespace NUMINAMATH_CALUDE_watermelon_stand_problem_l3873_387364

/-- A watermelon stand problem -/
theorem watermelon_stand_problem (total_melons : ℕ) 
  (single_melon_customers : ℕ) (triple_melon_customers : ℕ) :
  total_melons = 46 →
  single_melon_customers = 17 →
  triple_melon_customers = 3 →
  total_melons - (single_melon_customers * 1 + triple_melon_customers * 3) = 20 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_stand_problem_l3873_387364


namespace NUMINAMATH_CALUDE_scale_division_l3873_387304

/-- Proves that dividing a scale of length 80 inches into 4 equal parts results in each part having a length of 20 inches. -/
theorem scale_division (scale_length : ℕ) (num_parts : ℕ) (part_length : ℕ) 
  (h1 : scale_length = 80) 
  (h2 : num_parts = 4) 
  (h3 : part_length * num_parts = scale_length) : 
  part_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l3873_387304


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3873_387362

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((2 + Complex.I) / (1 - 2 * Complex.I)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3873_387362


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3873_387346

theorem triangle_angle_measure (a b c : ℝ) (A B C : ℝ) :
  0 < B → B < π →
  a = b * Real.cos C + c * Real.sin B →
  B = π / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3873_387346


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3873_387343

/-- Proves that mixing 100 mL of 10% alcohol solution with 300 mL of 30% alcohol solution 
    results in a 25% alcohol solution -/
theorem alcohol_mixture_percentage :
  let x_volume : ℝ := 100
  let x_percentage : ℝ := 10
  let y_volume : ℝ := 300
  let y_percentage : ℝ := 30
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := (x_volume * x_percentage + y_volume * y_percentage) / 100
  total_alcohol / total_volume * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l3873_387343


namespace NUMINAMATH_CALUDE_solve_for_t_l3873_387308

-- Define the variables
variable (s t : ℝ)

-- State the theorem
theorem solve_for_t (eq1 : 7 * s + 3 * t = 82) (eq2 : s = 2 * t - 3) : t = 103 / 17 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l3873_387308


namespace NUMINAMATH_CALUDE_num_groupings_l3873_387354

/-- The number of ways to distribute n items into 2 non-empty groups -/
def distribute (n : ℕ) : ℕ :=
  2^n - 2

/-- The number of tour guides -/
def num_guides : ℕ := 2

/-- The number of tourists -/
def num_tourists : ℕ := 6

/-- Each guide must have at least one tourist -/
axiom guides_not_empty : distribute num_tourists ≥ 1

/-- Theorem: The number of ways to distribute 6 tourists between 2 guides, 
    with each guide having at least one tourist, is 62 -/
theorem num_groupings : distribute num_tourists = 62 := by
  sorry

end NUMINAMATH_CALUDE_num_groupings_l3873_387354


namespace NUMINAMATH_CALUDE_profit_loss_ratio_l3873_387353

theorem profit_loss_ratio (c x y : ℝ) (hx : x = 0.85 * c) (hy : y = 1.15 * c) :
  y / x = 23 / 17 := by
  sorry

end NUMINAMATH_CALUDE_profit_loss_ratio_l3873_387353


namespace NUMINAMATH_CALUDE_q_necessary_not_sufficient_l3873_387363

-- Define the propositions
def p (b : ℝ) : Prop := ∃ r : ℝ, r ≠ 0 ∧ b = 1 * r ∧ 9 = b * r

def q (b : ℝ) : Prop := b = 3

-- State the theorem
theorem q_necessary_not_sufficient :
  (∀ b : ℝ, p b → q b) ∧ (∃ b : ℝ, p b ∧ ¬q b) := by
  sorry

end NUMINAMATH_CALUDE_q_necessary_not_sufficient_l3873_387363


namespace NUMINAMATH_CALUDE_cost_per_item_l3873_387323

theorem cost_per_item (total_customers : ℕ) (purchase_percentage : ℚ) (total_profit : ℚ) : 
  total_customers = 100 → 
  purchase_percentage = 80 / 100 → 
  total_profit = 1000 → 
  total_profit / (total_customers * purchase_percentage) = 25 / 2 := by
sorry

end NUMINAMATH_CALUDE_cost_per_item_l3873_387323


namespace NUMINAMATH_CALUDE_kyles_weight_lifting_ratio_l3873_387371

/-- 
Given:
- Kyle can lift 60 more pounds this year
- He can now lift 80 pounds in total
Prove that the ratio of the additional weight to the weight he could lift last year is 3
-/
theorem kyles_weight_lifting_ratio : 
  ∀ (last_year_weight additional_weight total_weight : ℕ),
  additional_weight = 60 →
  total_weight = 80 →
  total_weight = last_year_weight + additional_weight →
  (additional_weight : ℚ) / last_year_weight = 3 := by
sorry

end NUMINAMATH_CALUDE_kyles_weight_lifting_ratio_l3873_387371


namespace NUMINAMATH_CALUDE_inverse_of_proposition_l3873_387376

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  x < 0 → x^2 > 0

-- Define the inverse proposition
def inverse_proposition (x : ℝ) : Prop :=
  x^2 > 0 → x < 0

-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition
theorem inverse_of_proposition :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, inverse_proposition x) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_proposition_l3873_387376


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3873_387380

def P : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def Q : Set ℝ := {x | x ≥ 3}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 3 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3873_387380


namespace NUMINAMATH_CALUDE_NaNO3_formed_l3873_387377

-- Define the chemical reaction
structure Reaction where
  reactant1 : String
  reactant2 : String
  product1 : String
  product2 : String

-- Define the moles of each substance
structure Moles where
  AgNO3 : ℝ
  NaOH : ℝ
  AgOH : ℝ
  NaNO3 : ℝ

-- Define the chemical equation
def chemicalEquation : Reaction :=
  { reactant1 := "AgNO3"
  , reactant2 := "NaOH"
  , product1 := "AgOH"
  , product2 := "NaNO3" }

-- Define the initial moles
def initialMoles : Moles :=
  { AgNO3 := 1
  , NaOH := 1
  , AgOH := 0
  , NaNO3 := 0 }

-- Define the reaction completion condition
def reactionComplete (initial : Moles) (final : Moles) : Prop :=
  final.AgNO3 = 0 ∨ final.NaOH = 0

-- Define the no side reactions condition
def noSideReactions (initial : Moles) (final : Moles) : Prop :=
  initial.AgNO3 + initial.NaOH = final.AgOH + final.NaNO3

-- Theorem statement
theorem NaNO3_formed
  (reaction : Reaction)
  (initial : Moles)
  (final : Moles)
  (hReaction : reaction = chemicalEquation)
  (hInitial : initial = initialMoles)
  (hComplete : reactionComplete initial final)
  (hNoSide : noSideReactions initial final) :
  final.NaNO3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_NaNO3_formed_l3873_387377


namespace NUMINAMATH_CALUDE_cosine_graph_shift_l3873_387302

/-- Given a cosine function f(x) = 3cos(2x), prove that shifting its graph π/6 units 
    to the right results in the graph of g(x) = 3cos(2x - π/3) -/
theorem cosine_graph_shift (x : ℝ) :
  3 * Real.cos (2 * (x - π / 6)) = 3 * Real.cos (2 * x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_cosine_graph_shift_l3873_387302


namespace NUMINAMATH_CALUDE_complex_in_second_quadrant_l3873_387325

def second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_in_second_quadrant :
  let z : ℂ := -2 + I
  second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_complex_in_second_quadrant_l3873_387325


namespace NUMINAMATH_CALUDE_circle_radius_l3873_387368

theorem circle_radius (x y : ℝ) :
  x^2 - 8*x + y^2 + 4*y + 9 = 0 →
  ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 11 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l3873_387368


namespace NUMINAMATH_CALUDE_shrimp_earnings_l3873_387360

/-- Calculates the earnings of each boy from catching and selling shrimp --/
theorem shrimp_earnings (victor_shrimp : ℕ) (austin_diff : ℕ) (price : ℚ) (price_per : ℕ) :
  victor_shrimp = 26 →
  austin_diff = 8 →
  price = 7 →
  price_per = 11 →
  let austin_shrimp := victor_shrimp - austin_diff
  let brian_shrimp := (victor_shrimp + austin_shrimp) / 2
  let total_shrimp := victor_shrimp + austin_shrimp + brian_shrimp
  let total_earnings := (total_shrimp / price_per : ℚ) * price
  total_earnings / 3 = 14 := by
sorry


end NUMINAMATH_CALUDE_shrimp_earnings_l3873_387360


namespace NUMINAMATH_CALUDE_equalize_piles_in_three_moves_l3873_387357

/-- Represents a configuration of pin piles -/
structure PinPiles :=
  (pile1 : Nat) (pile2 : Nat) (pile3 : Nat)

/-- Represents a move between two piles -/
inductive Move
  | one_to_two
  | one_to_three
  | two_to_one
  | two_to_three
  | three_to_one
  | three_to_two

/-- Applies a move to a given configuration -/
def apply_move (piles : PinPiles) (move : Move) : PinPiles :=
  match move with
  | Move.one_to_two => PinPiles.mk (piles.pile1 - piles.pile2) (piles.pile2 * 2) piles.pile3
  | Move.one_to_three => PinPiles.mk (piles.pile1 - piles.pile3) piles.pile2 (piles.pile3 * 2)
  | Move.two_to_one => PinPiles.mk (piles.pile1 * 2) (piles.pile2 - piles.pile1) piles.pile3
  | Move.two_to_three => PinPiles.mk piles.pile1 (piles.pile2 - piles.pile3) (piles.pile3 * 2)
  | Move.three_to_one => PinPiles.mk (piles.pile1 * 2) piles.pile2 (piles.pile3 - piles.pile1)
  | Move.three_to_two => PinPiles.mk piles.pile1 (piles.pile2 * 2) (piles.pile3 - piles.pile2)

/-- The main theorem to be proved -/
theorem equalize_piles_in_three_moves :
  ∃ (m1 m2 m3 : Move),
    let initial := PinPiles.mk 11 7 6
    let step1 := apply_move initial m1
    let step2 := apply_move step1 m2
    let step3 := apply_move step2 m3
    step3 = PinPiles.mk 8 8 8 :=
by
  sorry

end NUMINAMATH_CALUDE_equalize_piles_in_three_moves_l3873_387357


namespace NUMINAMATH_CALUDE_division_ways_count_l3873_387310

def number_of_people : ℕ := 6
def number_of_cars : ℕ := 2
def max_capacity_per_car : ℕ := 4

theorem division_ways_count :
  (Finset.sum (Finset.range (min number_of_people (max_capacity_per_car + 1)))
    (λ i => (number_of_people.choose i) * ((number_of_people - i).choose (number_of_people - i)))) = 60 := by
  sorry

end NUMINAMATH_CALUDE_division_ways_count_l3873_387310


namespace NUMINAMATH_CALUDE_cubic_function_has_three_roots_l3873_387345

-- Define the cubic function
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Theorem statement
theorem cubic_function_has_three_roots :
  ∃ (a b c : ℝ), a < b ∧ b < c ∧
  f a = 0 ∧ f b = 0 ∧ f c = 0 ∧
  ∀ x, f x = 0 → x = a ∨ x = b ∨ x = c :=
sorry

end NUMINAMATH_CALUDE_cubic_function_has_three_roots_l3873_387345


namespace NUMINAMATH_CALUDE_unique_cube_property_l3873_387326

theorem unique_cube_property :
  ∃! (n : ℕ), n > 0 ∧ n^3 / 1000 = n :=
by sorry

end NUMINAMATH_CALUDE_unique_cube_property_l3873_387326


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3873_387301

/-- An arithmetic sequence with common difference 3 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 3

/-- a_2, a_4, and a_8 form a geometric sequence -/
def geometric_subseq (a : ℕ → ℝ) : Prop :=
  (a 4) ^ 2 = a 2 * a 8

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  arithmetic_seq a → geometric_subseq a → a 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3873_387301


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l3873_387344

theorem prime_square_sum_equation :
  ∃ (p q r : ℕ), Prime p ∧ Prime q ∧ Prime r ∧ p^2 + 1 = q^2 + r^2 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_sum_equation_l3873_387344


namespace NUMINAMATH_CALUDE_giants_playoff_fraction_l3873_387300

theorem giants_playoff_fraction :
  let games_played : ℕ := 20
  let games_won : ℕ := 12
  let games_left : ℕ := 10
  let additional_wins_needed : ℕ := 8
  let total_games : ℕ := games_played + games_left
  let total_wins_needed : ℕ := games_won + additional_wins_needed
  (total_wins_needed : ℚ) / total_games = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_giants_playoff_fraction_l3873_387300


namespace NUMINAMATH_CALUDE_triangle_isosceles_l3873_387309

-- Define a structure for a triangle
structure Triangle where
  p : ℝ
  q : ℝ
  r : ℝ
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r

-- Define the condition for triangle existence
def triangleExists (t : Triangle) (n : ℕ) : Prop :=
  t.p^n + t.q^n > t.r^n ∧ t.q^n + t.r^n > t.p^n ∧ t.r^n + t.p^n > t.q^n

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop :=
  t.p = t.q ∨ t.q = t.r ∨ t.r = t.p

-- The main theorem
theorem triangle_isosceles (t : Triangle) 
  (h : ∀ n : ℕ, triangleExists t n) : isIsosceles t := by
  sorry

end NUMINAMATH_CALUDE_triangle_isosceles_l3873_387309


namespace NUMINAMATH_CALUDE_high_school_total_students_l3873_387350

/-- Represents a high school with three grades -/
structure HighSchool :=
  (freshman_count : ℕ)
  (sophomore_count : ℕ)
  (senior_count : ℕ)

/-- Represents a stratified sample from the high school -/
structure StratifiedSample :=
  (freshman_sample : ℕ)
  (sophomore_sample : ℕ)
  (senior_sample : ℕ)

/-- The total number of students in the high school -/
def total_students (hs : HighSchool) : ℕ :=
  hs.freshman_count + hs.sophomore_count + hs.senior_count

/-- The total number of students in the sample -/
def total_sample (s : StratifiedSample) : ℕ :=
  s.freshman_sample + s.sophomore_sample + s.senior_sample

theorem high_school_total_students 
  (hs : HighSchool) 
  (sample : StratifiedSample) 
  (h1 : hs.freshman_count = 400)
  (h2 : sample.sophomore_sample = 15)
  (h3 : sample.senior_sample = 10)
  (h4 : total_sample sample = 45) :
  total_students hs = 900 :=
sorry

end NUMINAMATH_CALUDE_high_school_total_students_l3873_387350


namespace NUMINAMATH_CALUDE_sony_games_to_give_away_l3873_387336

theorem sony_games_to_give_away (current_sony_games : ℕ) (target_sony_games : ℕ) :
  current_sony_games = 132 → target_sony_games = 31 →
  current_sony_games - target_sony_games = 101 :=
by
  sorry


end NUMINAMATH_CALUDE_sony_games_to_give_away_l3873_387336


namespace NUMINAMATH_CALUDE_invitations_per_package_l3873_387399

theorem invitations_per_package (friends : ℕ) (packs : ℕ) (h1 : friends = 10) (h2 : packs = 5) :
  friends / packs = 2 := by
sorry

end NUMINAMATH_CALUDE_invitations_per_package_l3873_387399


namespace NUMINAMATH_CALUDE_eight_power_15_divided_by_64_power_6_l3873_387375

theorem eight_power_15_divided_by_64_power_6 : 8^15 / 64^6 = 512 := by sorry

end NUMINAMATH_CALUDE_eight_power_15_divided_by_64_power_6_l3873_387375


namespace NUMINAMATH_CALUDE_smallest_power_comparison_l3873_387321

theorem smallest_power_comparison : 127^8 < 63^10 ∧ 63^10 < 33^12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_power_comparison_l3873_387321


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3873_387320

theorem contrapositive_equivalence :
  (∀ x : ℝ, (x^2 ≥ 4 → x ≤ -2 ∨ x ≥ 2)) ↔
  (∀ x : ℝ, (-2 < x ∧ x < 2 → x^2 < 4)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3873_387320


namespace NUMINAMATH_CALUDE_farm_milk_production_l3873_387339

/-- Calculates the weekly milk production for a farm -/
def weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * 7

/-- Theorem: A farm with 52 cows, each producing 5 liters of milk per day, produces 1820 liters of milk in a week -/
theorem farm_milk_production :
  weekly_milk_production 52 5 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_farm_milk_production_l3873_387339


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_six_l3873_387330

/-- Given a non-isosceles triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its perimeter is 6 under certain conditions. -/
theorem triangle_perimeter_is_six 
  (a b c A B C : ℝ) 
  (h_non_isosceles : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_angles : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a * (Real.cos (C / 2))^2 + c * (Real.cos (A / 2))^2 = 3 * c / 2)
  (h_sines : 2 * Real.sin (A - B) + b * Real.sin B = a * Real.sin A) :
  a + b + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_six_l3873_387330


namespace NUMINAMATH_CALUDE_function_equation_solution_l3873_387351

theorem function_equation_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y) →
  ((∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l3873_387351


namespace NUMINAMATH_CALUDE_exists_non_monochromatic_coloring_l3873_387390

/-- Represents a coloring of numbers using 4 colors -/
def Coloring := Fin 2008 → Fin 4

/-- An arithmetic progression of 10 terms -/
def ArithmeticProgression := Fin 10 → Fin 2008

/-- Checks if an arithmetic progression is valid (within the range 1 to 2008) -/
def isValidAP (ap : ArithmeticProgression) : Prop :=
  ∀ i : Fin 10, ap i < 2008

/-- Checks if an arithmetic progression is monochromatic under a given coloring -/
def isMonochromatic (c : Coloring) (ap : ArithmeticProgression) : Prop :=
  ∃ color : Fin 4, ∀ i : Fin 10, c (ap i) = color

/-- The main theorem statement -/
theorem exists_non_monochromatic_coloring :
  ∃ c : Coloring, ∀ ap : ArithmeticProgression, isValidAP ap → ¬isMonochromatic c ap := by
  sorry

end NUMINAMATH_CALUDE_exists_non_monochromatic_coloring_l3873_387390


namespace NUMINAMATH_CALUDE_unique_number_with_gcd_l3873_387396

theorem unique_number_with_gcd : ∃! n : ℕ, 70 ≤ n ∧ n < 80 ∧ Nat.gcd 30 n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_gcd_l3873_387396


namespace NUMINAMATH_CALUDE_ants_eyes_count_l3873_387369

theorem ants_eyes_count (spider_count : ℕ) (ant_count : ℕ) (eyes_per_spider : ℕ) (total_eyes : ℕ)
  (h1 : spider_count = 3)
  (h2 : ant_count = 50)
  (h3 : eyes_per_spider = 8)
  (h4 : total_eyes = 124) :
  (total_eyes - spider_count * eyes_per_spider) / ant_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_ants_eyes_count_l3873_387369


namespace NUMINAMATH_CALUDE_no_valid_number_l3873_387307

theorem no_valid_number : ¬∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧  -- 3-digit number
  (∃ (x : ℕ), x < 10 ∧ n = 520 + x) ∧  -- in the form 52x where x is a digit
  (n % 6 = 0) ∧  -- divisible by 6
  (n % 10 = 6)  -- last digit is 6
  := by sorry

end NUMINAMATH_CALUDE_no_valid_number_l3873_387307


namespace NUMINAMATH_CALUDE_allison_video_uploads_l3873_387333

/-- Prove that Allison uploaded 10 one-hour videos daily during the first half of June. -/
theorem allison_video_uploads :
  ∀ (x : ℕ),
  (15 * x + 15 * (2 * x) = 450) →
  x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_allison_video_uploads_l3873_387333


namespace NUMINAMATH_CALUDE_sequence_general_term_l3873_387394

/-- Given a sequence {aₙ} with sum of first n terms Sₙ = (2/3)n² - (1/3)n,
    prove that the general term is aₙ = (4/3)n - 1 -/
theorem sequence_general_term (a : ℕ → ℚ) (S : ℕ → ℚ) 
    (h : ∀ n, S n = 2/3 * n^2 - 1/3 * n) :
  ∀ n, a n = 4/3 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3873_387394


namespace NUMINAMATH_CALUDE_two_distinct_prime_products_count_l3873_387378

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that counts the number of integers less than n that are the product of exactly two distinct primes -/
def countTwoDistinctPrimeProducts (n : ℕ) : ℕ := sorry

/-- Theorem stating that the count of numbers less than 1,000,000 that are the product of exactly two distinct primes is 209867 -/
theorem two_distinct_prime_products_count :
  countTwoDistinctPrimeProducts 1000000 = 209867 := by sorry

end NUMINAMATH_CALUDE_two_distinct_prime_products_count_l3873_387378


namespace NUMINAMATH_CALUDE_river_current_speed_l3873_387312

/-- Proves that the speed of the river's current is half the swimmer's speed in still water --/
theorem river_current_speed (x y : ℝ) : 
  x > 0 → -- swimmer's speed in still water is positive
  x = 10 → -- swimmer's speed in still water is 10 km/h
  (x + y) > 0 → -- downstream speed is positive
  (x - y) > 0 → -- upstream speed is positive
  (1 / (x - y)) = (3 * (1 / (x + y))) → -- upstream time is 3 times downstream time
  y = x / 2 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l3873_387312


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3873_387385

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 12) 
  (eq2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3873_387385


namespace NUMINAMATH_CALUDE_solution_count_correct_l3873_387313

/-- The number of integers n satisfying the equation 1 + ⌊(100n)/103⌋ = ⌈(97n)/100⌉ -/
def solution_count : ℕ := 10300

/-- Function g(n) defined as ⌈(97n)/100⌉ - ⌊(100n)/103⌋ -/
def g (n : ℤ) : ℤ := ⌈(97 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 103⌋

/-- The main theorem stating that the number of solutions is equal to solution_count -/
theorem solution_count_correct :
  (∑' n : ℤ, if 1 + ⌊(100 * n : ℚ) / 103⌋ = ⌈(97 * n : ℚ) / 100⌉ then 1 else 0) = solution_count :=
sorry

/-- Lemma showing the periodic behavior of g(n) -/
lemma g_periodic (n : ℤ) : g (n + 10300) = g n + 3 :=
sorry

/-- Lemma stating that for each residue class modulo 10300, there exists a unique solution -/
lemma unique_solution_per_residue_class (r : ℤ) :
  ∃! n : ℤ, g n = 1 ∧ n ≡ r [ZMOD 10300] :=
sorry

end NUMINAMATH_CALUDE_solution_count_correct_l3873_387313


namespace NUMINAMATH_CALUDE_valid_queue_arrangements_correct_l3873_387349

/-- Represents the number of valid queue arrangements for a concert ticket purchase scenario. -/
def validQueueArrangements (m n : ℕ) : ℚ :=
  if n ≥ m then
    (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1))
  else 0

/-- Theorem stating the correctness of the validQueueArrangements function. -/
theorem valid_queue_arrangements_correct (m n : ℕ) (h : n ≥ m) :
  validQueueArrangements m n = (Nat.factorial (m + n) * (n - m + 1)) / (Nat.factorial m * Nat.factorial (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_valid_queue_arrangements_correct_l3873_387349


namespace NUMINAMATH_CALUDE_routes_between_plains_cities_l3873_387340

theorem routes_between_plains_cities 
  (total_cities : ℕ) 
  (mountainous_cities : ℕ) 
  (plains_cities : ℕ) 
  (total_routes : ℕ) 
  (mountainous_routes : ℕ) 
  (h1 : total_cities = 100)
  (h2 : mountainous_cities = 30)
  (h3 : plains_cities = 70)
  (h4 : mountainous_cities + plains_cities = total_cities)
  (h5 : total_routes = 150)
  (h6 : mountainous_routes = 21) :
  total_routes - mountainous_routes - (mountainous_cities * 3 - mountainous_routes * 2) / 2 = 81 := by
  sorry

end NUMINAMATH_CALUDE_routes_between_plains_cities_l3873_387340


namespace NUMINAMATH_CALUDE_min_value_a2_plus_b2_l3873_387365

theorem min_value_a2_plus_b2 (a b : ℝ) (h : (9 : ℝ) / a^2 + (4 : ℝ) / b^2 = 1) :
  ∀ x y : ℝ, (9 : ℝ) / x^2 + (4 : ℝ) / y^2 = 1 → x^2 + y^2 ≥ 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a2_plus_b2_l3873_387365


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3873_387372

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3873_387372


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l3873_387314

theorem min_value_sum_of_squares (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 9 ∧
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l3873_387314


namespace NUMINAMATH_CALUDE_first_group_number_is_five_l3873_387329

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  groupSize : ℕ
  numberFromGroup17 : ℕ

/-- The systematic sampling scheme for the given problem -/
def problemSampling : SystematicSampling :=
  { totalStudents := 140
  , sampleSize := 20
  , groupSize := 7
  , numberFromGroup17 := 117
  }

/-- The number drawn from the first group in a systematic sampling -/
def firstGroupNumber (s : SystematicSampling) : ℕ :=
  s.numberFromGroup17 - s.groupSize * (17 - 1)

/-- Theorem stating that the number drawn from the first group is 5 -/
theorem first_group_number_is_five :
  firstGroupNumber problemSampling = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_group_number_is_five_l3873_387329


namespace NUMINAMATH_CALUDE_largest_y_value_l3873_387322

/-- The largest possible value of y for regular polygons Q1 (x-gon) and Q2 (y-gon) -/
theorem largest_y_value (x y : ℕ) : 
  x ≥ y → 
  y ≥ 3 → 
  (x - 2) * y * 29 = (y - 2) * x * 28 → 
  y ≤ 57 :=
by sorry

end NUMINAMATH_CALUDE_largest_y_value_l3873_387322


namespace NUMINAMATH_CALUDE_smallest_integer_proof_l3873_387342

/-- The smallest positive integer that can be represented as CC₆ and DD₈ -/
def smallest_integer : ℕ := 63

/-- Conversion from base 6 to base 10 -/
def base_6_to_10 (c : ℕ) : ℕ := 6 * c + c

/-- Conversion from base 8 to base 10 -/
def base_8_to_10 (d : ℕ) : ℕ := 8 * d + d

/-- Theorem stating that 63 is the smallest positive integer representable as CC₆ and DD₈ -/
theorem smallest_integer_proof :
  ∃ (c d : ℕ),
    c < 6 ∧ d < 8 ∧
    base_6_to_10 c = smallest_integer ∧
    base_8_to_10 d = smallest_integer ∧
    ∀ (n : ℕ), n > 0 ∧ (∃ (c' d' : ℕ), c' < 6 ∧ d' < 8 ∧ base_6_to_10 c' = n ∧ base_8_to_10 d' = n) →
      n ≥ smallest_integer :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_proof_l3873_387342


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3873_387347

theorem x_squared_minus_y_squared (x y : ℝ) 
  (sum : x + y = 20) 
  (diff : x - y = 4) : 
  x^2 - y^2 = 80 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l3873_387347


namespace NUMINAMATH_CALUDE_negative_irrational_less_than_neg_three_l3873_387384

theorem negative_irrational_less_than_neg_three :
  ∃ x : ℝ, x < -3 ∧ Irrational x ∧ x < 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_negative_irrational_less_than_neg_three_l3873_387384


namespace NUMINAMATH_CALUDE_derived_sequence_not_arithmetic_nor_geometric_l3873_387370

/-- A sequence {a_n} defined by its partial sums s_n = aq^n -/
def PartialSumSequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^n

/-- The sequence {a_n} derived from the partial sums -/
def DerivedSequence (a q : ℝ) : ℕ → ℝ :=
  fun n => if n = 1 then a * q else a * (q - 1) * q^(n - 1)

/-- Theorem stating that the derived sequence is neither arithmetic nor geometric -/
theorem derived_sequence_not_arithmetic_nor_geometric (a q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) (hq_nonzero : q ≠ 0) :
  ¬ (∃ d : ℝ, ∀ n : ℕ, n ≥ 2 → DerivedSequence a q (n + 1) - DerivedSequence a q n = d) ∧
  ¬ (∃ r : ℝ, ∀ n : ℕ, n ≥ 1 → DerivedSequence a q (n + 1) / DerivedSequence a q n = r) :=
by sorry


end NUMINAMATH_CALUDE_derived_sequence_not_arithmetic_nor_geometric_l3873_387370


namespace NUMINAMATH_CALUDE_skips_mode_is_165_l3873_387356

def skips : List ℕ := [165, 165, 165, 165, 165, 170, 170, 145, 150, 150]

def mode (l : List ℕ) : ℕ := 
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem skips_mode_is_165 : mode skips = 165 := by sorry

end NUMINAMATH_CALUDE_skips_mode_is_165_l3873_387356


namespace NUMINAMATH_CALUDE_range_of_fraction_l3873_387397

theorem range_of_fraction (a b : ℝ) (ha : 1 < a ∧ a < 2) (hb : -2 < b ∧ b < -1) :
  ∃ x, -2 < x ∧ x < -1/2 ∧ x = a/b :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l3873_387397


namespace NUMINAMATH_CALUDE_combined_salaries_l3873_387392

/-- The combined salaries of B, C, D, and E given A's salary and the average salary of all five -/
theorem combined_salaries 
  (salary_A : ℕ) 
  (average_salary : ℕ) 
  (h1 : salary_A = 8000)
  (h2 : average_salary = 8600) :
  (5 * average_salary) - salary_A = 35000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salaries_l3873_387392


namespace NUMINAMATH_CALUDE_mall_entrance_exit_ways_l3873_387388

theorem mall_entrance_exit_ways (n : Nat) (h : n = 4) : 
  (n * (n - 1) : Nat) = 12 := by
  sorry

#check mall_entrance_exit_ways

end NUMINAMATH_CALUDE_mall_entrance_exit_ways_l3873_387388


namespace NUMINAMATH_CALUDE_cosine_product_sqrt_l3873_387382

theorem cosine_product_sqrt (π : Real) : 
  Real.sqrt ((3 - Real.cos (π / 9)^2) * (3 - Real.cos (2 * π / 9)^2) * (3 - Real.cos (4 * π / 9)^2)) = 39 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_sqrt_l3873_387382


namespace NUMINAMATH_CALUDE_two_integers_sum_l3873_387318

theorem two_integers_sum (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 120) :
  x + y = 15 := by sorry

end NUMINAMATH_CALUDE_two_integers_sum_l3873_387318


namespace NUMINAMATH_CALUDE_max_stores_visited_l3873_387303

theorem max_stores_visited (
  total_stores : ℕ)
  (total_shoppers : ℕ)
  (double_visitors : ℕ)
  (total_visits : ℕ)
  (h1 : total_stores = 7)
  (h2 : total_shoppers = 11)
  (h3 : double_visitors = 7)
  (h4 : total_visits = 21)
  (h5 : double_visitors * 2 + (total_shoppers - double_visitors) ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧
    ∀ (individual_visits : ℕ), individual_visits ≤ max_visits :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l3873_387303


namespace NUMINAMATH_CALUDE_function_minimum_and_inequality_l3873_387383

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |5 - x|

-- State the theorem
theorem function_minimum_and_inequality :
  ∃ (m : ℝ), 
    (∀ x, f x ≥ m) ∧ 
    (∃ x, f x = m) ∧
    m = 9/2 ∧
    ∀ (a b : ℝ), a ≥ 0 → b ≥ 0 → a + b = (2/3) * m → 
      1 / (a + 1) + 1 / (b + 2) ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_and_inequality_l3873_387383


namespace NUMINAMATH_CALUDE_tiffany_score_l3873_387317

/-- The score for each treasure found -/
def points_per_treasure : ℕ := 6

/-- The number of treasures found on the first level -/
def treasures_level1 : ℕ := 3

/-- The number of treasures found on the second level -/
def treasures_level2 : ℕ := 5

/-- Tiffany's total score -/
def total_score : ℕ := points_per_treasure * (treasures_level1 + treasures_level2)

theorem tiffany_score : total_score = 48 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_score_l3873_387317


namespace NUMINAMATH_CALUDE_wendy_first_level_treasures_l3873_387316

/-- Represents the game scenario where Wendy finds treasures on two levels --/
structure GameScenario where
  pointsPerTreasure : ℕ
  treasuresOnSecondLevel : ℕ
  totalScore : ℕ

/-- Calculates the number of treasures found on the first level --/
def treasuresOnFirstLevel (game : GameScenario) : ℕ :=
  (game.totalScore - game.pointsPerTreasure * game.treasuresOnSecondLevel) / game.pointsPerTreasure

/-- Theorem stating that Wendy found 4 treasures on the first level --/
theorem wendy_first_level_treasures :
  let game : GameScenario := {
    pointsPerTreasure := 5,
    treasuresOnSecondLevel := 3,
    totalScore := 35
  }
  treasuresOnFirstLevel game = 4 := by sorry

end NUMINAMATH_CALUDE_wendy_first_level_treasures_l3873_387316


namespace NUMINAMATH_CALUDE_cubic_function_property_l3873_387395

/-- Given a cubic function f(x) = ax³ + bx² + cx + d where f(1) = 4,
    prove that 12a - 6b + 3c - 2d = 40 -/
theorem cubic_function_property (a b c d : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^3 + b * x^2 + c * x + d)
  (h_f1 : f 1 = 4) :
  12 * a - 6 * b + 3 * c - 2 * d = 40 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3873_387395


namespace NUMINAMATH_CALUDE_franks_decks_l3873_387393

theorem franks_decks (deck_cost : ℕ) (friend_decks : ℕ) (total_spent : ℕ) :
  deck_cost = 7 →
  friend_decks = 2 →
  total_spent = 35 →
  ∃ (frank_decks : ℕ), frank_decks * deck_cost + friend_decks * deck_cost = total_spent ∧ frank_decks = 3 :=
by sorry

end NUMINAMATH_CALUDE_franks_decks_l3873_387393


namespace NUMINAMATH_CALUDE_simplify_calculations_l3873_387367

theorem simplify_calculations :
  (329 * 101 = 33229) ∧
  (54 * 98 + 46 * 98 = 9800) ∧
  (98 * 125 = 12250) ∧
  (37 * 29 + 37 = 1110) := by
  sorry

end NUMINAMATH_CALUDE_simplify_calculations_l3873_387367


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3873_387374

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3*x⌋ + 1/3⌋ = ⌊x + 3⌋) ↔ (4/3 ≤ x ∧ x < 5/3) := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3873_387374


namespace NUMINAMATH_CALUDE_interview_probability_l3873_387311

/-- The number of students enrolled in at least one language class -/
def total_students : ℕ := 30

/-- The number of students enrolled in the German class -/
def german_students : ℕ := 20

/-- The number of students enrolled in the Italian class -/
def italian_students : ℕ := 22

/-- The probability of selecting two students such that at least one is enrolled in German
    and at least one is enrolled in Italian -/
def prob_both_classes : ℚ := 362 / 435

theorem interview_probability :
  prob_both_classes = 1 - (Nat.choose (german_students + italian_students - total_students) 2 +
                           Nat.choose (german_students - (german_students + italian_students - total_students)) 2 +
                           Nat.choose (italian_students - (german_students + italian_students - total_students)) 2) /
                          Nat.choose total_students 2 :=
by sorry

end NUMINAMATH_CALUDE_interview_probability_l3873_387311


namespace NUMINAMATH_CALUDE_john_payment_amount_l3873_387305

/-- The final amount John needs to pay after late charges -/
def final_amount (original_bill : ℝ) (first_charge : ℝ) (second_charge : ℝ) (third_charge : ℝ) : ℝ :=
  original_bill * (1 + first_charge) * (1 + second_charge) * (1 + third_charge)

/-- Theorem stating the final amount John needs to pay -/
theorem john_payment_amount :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |final_amount 500 0.02 0.03 0.025 - 538.43| < ε :=
sorry

end NUMINAMATH_CALUDE_john_payment_amount_l3873_387305


namespace NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l3873_387358

-- Define the number of faces on a die
def die_faces : ℕ := 6

-- Define the number of dice
def num_dice : ℕ := 4

-- Define the possible common differences
def common_differences : List ℕ := [1, 2]

-- Define a function to calculate the total number of outcomes
def total_outcomes : ℕ := die_faces ^ num_dice

-- Define a function to calculate the favorable outcomes
def favorable_outcomes : ℕ := sorry

-- The main theorem
theorem dice_arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l3873_387358


namespace NUMINAMATH_CALUDE_no_further_simplification_l3873_387352

theorem no_further_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  ∀ (f : ℝ → ℝ), f (a/b - b/a + a^2*b^2) = a/b - b/a + a^2*b^2 → f = id := by
  sorry

end NUMINAMATH_CALUDE_no_further_simplification_l3873_387352


namespace NUMINAMATH_CALUDE_velocity_at_t_1_is_zero_l3873_387373

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2*t

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := -2*t + 2

-- Theorem statement
theorem velocity_at_t_1_is_zero : v 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_velocity_at_t_1_is_zero_l3873_387373


namespace NUMINAMATH_CALUDE_sin_translation_to_cos_l3873_387387

theorem sin_translation_to_cos (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x + π / 6)
  let g (x : ℝ) := f (x + π / 6)
  g x = Real.cos (2 * x) := by
sorry

end NUMINAMATH_CALUDE_sin_translation_to_cos_l3873_387387


namespace NUMINAMATH_CALUDE_xy_equals_one_l3873_387361

theorem xy_equals_one (x y : ℝ) : 
  x > 0 → y > 0 → x / y = 36 → y = 0.16666666666666666 → x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_one_l3873_387361


namespace NUMINAMATH_CALUDE_sum_difference_equality_l3873_387315

theorem sum_difference_equality : 3.59 + 2.4 - 1.67 = 4.32 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_equality_l3873_387315


namespace NUMINAMATH_CALUDE_students_neither_music_nor_art_l3873_387341

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 30) 
  (h3 : art = 10) 
  (h4 : both = 10) : 
  total - (music + art - both) = 470 := by
  sorry

end NUMINAMATH_CALUDE_students_neither_music_nor_art_l3873_387341


namespace NUMINAMATH_CALUDE_shirts_made_yesterday_is_nine_l3873_387334

/-- The number of shirts a machine can make per minute -/
def shirts_per_minute : ℕ := 3

/-- The number of minutes the machine worked yesterday -/
def minutes_worked_yesterday : ℕ := 3

/-- The number of shirts made yesterday -/
def shirts_made_yesterday : ℕ := shirts_per_minute * minutes_worked_yesterday

theorem shirts_made_yesterday_is_nine : shirts_made_yesterday = 9 := by
  sorry

end NUMINAMATH_CALUDE_shirts_made_yesterday_is_nine_l3873_387334


namespace NUMINAMATH_CALUDE_total_spent_on_fruits_l3873_387327

def total_fruits : ℕ := 32
def plum_cost : ℕ := 2
def peach_cost : ℕ := 1
def plums_bought : ℕ := 20

theorem total_spent_on_fruits : 
  plums_bought * plum_cost + (total_fruits - plums_bought) * peach_cost = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_on_fruits_l3873_387327


namespace NUMINAMATH_CALUDE_count_divisible_by_four_l3873_387381

theorem count_divisible_by_four : 
  (Finset.filter (fun n : Fin 10 => (748 * 10 + n : ℕ) % 4 = 0) Finset.univ).card = 3 :=
by sorry

end NUMINAMATH_CALUDE_count_divisible_by_four_l3873_387381


namespace NUMINAMATH_CALUDE_election_results_l3873_387319

/-- Represents a candidate in the election -/
inductive Candidate
  | Montoran
  | AjudaPinto
  | VidameOfOussel

/-- Represents a voter group with their preferences -/
structure VoterGroup where
  size : Nat
  preferences : List Candidate

/-- Represents the election setup -/
structure Election where
  totalVoters : Nat
  candidates : List Candidate
  voterGroups : List VoterGroup

/-- One-round voting system -/
def oneRoundWinner (e : Election) : Candidate := sorry

/-- Two-round voting system -/
def twoRoundWinner (e : Election) : Candidate := sorry

/-- Three-round voting system -/
def threeRoundWinner (e : Election) : Candidate := sorry

/-- The election setup based on the problem description -/
def electionSetup : Election :=
  { totalVoters := 100000
  , candidates := [Candidate.Montoran, Candidate.AjudaPinto, Candidate.VidameOfOussel]
  , voterGroups :=
    [ { size := 33000
      , preferences := [Candidate.Montoran, Candidate.AjudaPinto, Candidate.VidameOfOussel]
      }
    , { size := 18000
      , preferences := [Candidate.AjudaPinto, Candidate.Montoran, Candidate.VidameOfOussel]
      }
    , { size := 12000
      , preferences := [Candidate.AjudaPinto, Candidate.VidameOfOussel, Candidate.Montoran]
      }
    , { size := 37000
      , preferences := [Candidate.VidameOfOussel, Candidate.AjudaPinto, Candidate.Montoran]
      }
    ]
  }

theorem election_results (e : Election) :
  e = electionSetup →
  oneRoundWinner e = Candidate.VidameOfOussel ∧
  twoRoundWinner e = Candidate.Montoran ∧
  threeRoundWinner e = Candidate.AjudaPinto :=
sorry

end NUMINAMATH_CALUDE_election_results_l3873_387319


namespace NUMINAMATH_CALUDE_girls_fraction_is_half_l3873_387331

/-- Given a class of students, prove that the fraction of the number of girls
    that equals 1/3 of the total number of students is 1/2, when the ratio of
    boys to girls is 1/2. -/
theorem girls_fraction_is_half (T G B : ℚ) : 
  T > 0 → G > 0 → B > 0 →
  T = G + B →
  B / G = 1 / 2 →
  ∃ (f : ℚ), f * G = (1 / 3) * T ∧ f = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_is_half_l3873_387331


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3873_387338

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_ab : a + b = 35)
  (sum_bc : b + c = 57)
  (sum_ca : c + a = 62) :
  a + b + c = 77 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3873_387338


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3873_387391

/-- Proves the simplification of two algebraic expressions -/
theorem algebraic_simplification (a b : ℝ) :
  (3 * a - (4 * b - 2 * a + 1) = 5 * a - 4 * b - 1) ∧
  (2 * (5 * a - 3 * b) - 3 * (a^2 - 2 * b) = 10 * a - 3 * a^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3873_387391


namespace NUMINAMATH_CALUDE_functional_relationship_l3873_387348

/-- Given a function y that is the sum of two components y₁ and y₂,
    where y₁ is directly proportional to x and y₂ is inversely proportional to (x-2),
    prove that y = x + 2/(x-2) when y = -1 at x = 1 and y = 5 at x = 3. -/
theorem functional_relationship (y y₁ y₂ : ℝ → ℝ) (k₁ k₂ : ℝ) :
  (∀ x, y x = y₁ x + y₂ x) →
  (∀ x, y₁ x = k₁ * x) →
  (∀ x, y₂ x = k₂ / (x - 2)) →
  y 1 = -1 →
  y 3 = 5 →
  ∀ x, y x = x + 2 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_functional_relationship_l3873_387348


namespace NUMINAMATH_CALUDE_count_integers_eq_25_l3873_387389

/-- The number of integers between 100 and 200 (exclusive) that have the same remainder when divided by 6 and 8 -/
def count_integers : ℕ :=
  (Finset.filter (λ n : ℕ => 
    100 < n ∧ n < 200 ∧ n % 6 = n % 8
  ) (Finset.range 200)).card

/-- Theorem stating that there are exactly 25 such integers -/
theorem count_integers_eq_25 : count_integers = 25 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_eq_25_l3873_387389


namespace NUMINAMATH_CALUDE_star_interior_angle_sum_formula_l3873_387328

/-- An n-pointed star is formed from a convex n-gon by extending each side k
    to intersect with side k+3 (modulo n). This function calculates the
    sum of interior angles at the n vertices of the resulting star. -/
def starInteriorAngleSum (n : ℕ) : ℝ :=
  180 * (n - 6 : ℝ)

/-- Theorem stating that for an n-pointed star (n ≥ 5), the sum of
    interior angles at the n vertices is 180(n-6) degrees. -/
theorem star_interior_angle_sum_formula {n : ℕ} (h : n ≥ 5) :
  starInteriorAngleSum n = 180 * (n - 6 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_star_interior_angle_sum_formula_l3873_387328


namespace NUMINAMATH_CALUDE_intersection_x_is_seven_l3873_387379

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- The x-axis -/
def x_axis : Line := { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Point C -/
def C : Point := ⟨7, 5⟩

/-- Point D -/
def D : Point := ⟨7, -3⟩

/-- The line passing through points C and D -/
def line_CD : Line := { p1 := C, p2 := D }

/-- The x-coordinate of the intersection point between a line and the x-axis -/
def intersection_x (l : Line) : ℝ := sorry

theorem intersection_x_is_seven : intersection_x line_CD = 7 := by sorry

end NUMINAMATH_CALUDE_intersection_x_is_seven_l3873_387379


namespace NUMINAMATH_CALUDE_divisibility_of_consecutive_ones_l3873_387386

/-- A number consisting of n consecutive ones -/
def consecutive_ones (n : ℕ) : ℕ :=
  (10^n - 1) / 9

theorem divisibility_of_consecutive_ones :
  ∃ k : ℕ, consecutive_ones 1998 = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_consecutive_ones_l3873_387386


namespace NUMINAMATH_CALUDE_vacuuming_time_ratio_l3873_387306

theorem vacuuming_time_ratio : 
  ∀ (time_downstairs : ℝ),
  time_downstairs > 0 →
  27 = time_downstairs + 5 →
  38 = 27 + time_downstairs →
  (27 : ℝ) / time_downstairs = 27 / 22 :=
by
  sorry

end NUMINAMATH_CALUDE_vacuuming_time_ratio_l3873_387306


namespace NUMINAMATH_CALUDE_probability_two_boys_or_two_girls_l3873_387398

/-- The probability of selecting either two boys or two girls from a group of 5 students -/
theorem probability_two_boys_or_two_girls (total_students : ℕ) (num_boys : ℕ) (num_girls : ℕ) :
  total_students = 5 →
  num_boys = 2 →
  num_girls = 3 →
  (Nat.choose num_girls 2 + Nat.choose num_boys 2) / Nat.choose total_students 2 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_boys_or_two_girls_l3873_387398


namespace NUMINAMATH_CALUDE_exists_scores_with_median_16_l3873_387324

/-- Represents a set of basketball scores -/
def BasketballScores := List ℕ

/-- Calculates the median of a list of natural numbers -/
def median (scores : BasketballScores) : ℚ :=
  sorry

/-- Theorem: There exists a set of basketball scores with a median of 16 -/
theorem exists_scores_with_median_16 : 
  ∃ (scores : BasketballScores), median scores = 16 := by
  sorry

end NUMINAMATH_CALUDE_exists_scores_with_median_16_l3873_387324


namespace NUMINAMATH_CALUDE_bakery_boxes_l3873_387337

theorem bakery_boxes (total_muffins : ℕ) (muffins_per_box : ℕ) (additional_boxes : ℕ) 
  (h1 : total_muffins = 95)
  (h2 : muffins_per_box = 5)
  (h3 : additional_boxes = 9) :
  total_muffins / muffins_per_box - additional_boxes = 10 :=
by sorry

end NUMINAMATH_CALUDE_bakery_boxes_l3873_387337


namespace NUMINAMATH_CALUDE_f_value_at_one_l3873_387366

def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem f_value_at_one (m : ℝ) :
  (∀ x ≥ -2, ∀ y ≥ -2, x < y → f m x < f m y) →
  (∀ x ≤ -2, ∀ y ≤ -2, x < y → f m x > f m y) →
  f m 1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_one_l3873_387366


namespace NUMINAMATH_CALUDE_number_problem_l3873_387332

theorem number_problem (n : ℝ) (h : (1/3) * (1/4) * n = 18) : (3/10) * n = 64.8 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3873_387332
