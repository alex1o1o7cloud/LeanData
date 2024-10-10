import Mathlib

namespace consecutive_integers_problem_l290_29038

theorem consecutive_integers_problem (a b c : ℕ) : 
  a.succ = b → b.succ = c → 
  a > 0 → b > 0 → c > 0 → 
  a^2 = 97344 → c^2 = 98596 → 
  b = 313 := by
sorry

end consecutive_integers_problem_l290_29038


namespace probability_three_black_cards_l290_29073

/-- The probability of drawing three black cards consecutively from a standard deck --/
theorem probability_three_black_cards (total_cards : ℕ) (black_cards : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : black_cards = 26) : 
  (black_cards * (black_cards - 1) * (black_cards - 2)) / 
  (total_cards * (total_cards - 1) * (total_cards - 2)) = 4 / 17 := by
sorry

end probability_three_black_cards_l290_29073


namespace arithmetic_sequence_first_term_l290_29095

/-- Given an arithmetic sequence {a_n} where a_3 = 1 and a_4 + a_10 = 18, prove that a_1 = -3 -/
theorem arithmetic_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a3 : a 3 = 1) 
  (h_sum : a 4 + a 10 = 18) : 
  a 1 = -3 := by sorry

end arithmetic_sequence_first_term_l290_29095


namespace candies_needed_to_fill_bags_l290_29013

theorem candies_needed_to_fill_bags (total_candies : ℕ) (bag_capacity : ℕ) (h1 : total_candies = 254) (h2 : bag_capacity = 30) : 
  (bag_capacity - (total_candies % bag_capacity)) = 16 := by
sorry

end candies_needed_to_fill_bags_l290_29013


namespace prob_not_blue_from_odds_l290_29094

-- Define the odds ratio
def odds_blue : ℚ := 5 / 6

-- Define the probability of not obtaining a blue ball
def prob_not_blue : ℚ := 6 / 11

-- Theorem statement
theorem prob_not_blue_from_odds :
  odds_blue = 5 / 6 → prob_not_blue = 6 / 11 := by
  sorry

end prob_not_blue_from_odds_l290_29094


namespace empty_quadratic_inequality_implies_a_range_l290_29061

theorem empty_quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + 4 ≥ 0) → a ∈ Set.Icc (-4) 4 := by
  sorry

end empty_quadratic_inequality_implies_a_range_l290_29061


namespace hotel_expenditure_l290_29093

theorem hotel_expenditure (num_men : ℕ) (standard_cost : ℚ) (extra_cost : ℚ) :
  num_men = 9 →
  standard_cost = 3 →
  extra_cost = 2 →
  (((num_men - 1) * standard_cost + 
    (standard_cost + extra_cost + 
      ((num_men - 1) * standard_cost + (standard_cost + extra_cost)) / num_men)) = 29.25) := by
  sorry

end hotel_expenditure_l290_29093


namespace rectangle_bisector_slope_l290_29021

/-- The slope of a line passing through the origin and the center of a rectangle
    with vertices (1, 0), (5, 0), (1, 2), and (5, 2) is 1/3. -/
theorem rectangle_bisector_slope :
  let vertices : List (ℝ × ℝ) := [(1, 0), (5, 0), (1, 2), (5, 2)]
  let center : ℝ × ℝ := (
    (vertices[0].1 + vertices[3].1) / 2,
    (vertices[0].2 + vertices[3].2) / 2
  )
  let slope : ℝ := center.2 / center.1
  slope = 1 / 3 := by
  sorry

end rectangle_bisector_slope_l290_29021


namespace infinitely_many_primes_mod_3_eq_2_l290_29060

theorem infinitely_many_primes_mod_3_eq_2 : Set.Infinite {p : ℕ | Nat.Prime p ∧ p % 3 = 2} := by
  sorry

end infinitely_many_primes_mod_3_eq_2_l290_29060


namespace total_students_l290_29070

theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 16) 
  (h2 : rank_from_left = 6) : 
  rank_from_right + rank_from_left - 1 = 21 := by
  sorry

end total_students_l290_29070


namespace john_savings_proof_l290_29045

/-- Calculates the monthly savings amount given the total savings period, amount spent, and remaining amount. -/
def monthly_savings (savings_period_years : ℕ) (amount_spent : ℕ) (amount_remaining : ℕ) : ℚ :=
  let total_saved : ℕ := amount_spent + amount_remaining
  let total_months : ℕ := savings_period_years * 12
  (total_saved : ℚ) / total_months

/-- Proves that given a savings period of 2 years, $400 spent, and $200 remaining, the monthly savings amount is $25. -/
theorem john_savings_proof :
  monthly_savings 2 400 200 = 25 := by
  sorry

end john_savings_proof_l290_29045


namespace daps_equivalent_to_48_dips_l290_29028

/-- Represents the number of units of a currency -/
structure Currency where
  amount : ℚ
  name : String

/-- Defines the exchange rate between two currencies -/
def exchange_rate (a b : Currency) : ℚ := a.amount / b.amount

/-- Given conditions of the problem -/
axiom daps_to_dops : exchange_rate (Currency.mk 5 "daps") (Currency.mk 4 "dops") = 1
axiom dops_to_dips : exchange_rate (Currency.mk 3 "dops") (Currency.mk 8 "dips") = 1

/-- The theorem to be proved -/
theorem daps_equivalent_to_48_dips :
  exchange_rate (Currency.mk 22.5 "daps") (Currency.mk 48 "dips") = 1 := by
  sorry

end daps_equivalent_to_48_dips_l290_29028


namespace hall_dimension_l290_29062

/-- Represents a square rug with a given side length. -/
structure Rug where
  side : ℝ
  square : side > 0

/-- Represents a square hall containing two rugs. -/
structure Hall where
  small_rug : Rug
  large_rug : Rug
  opposite_overlap : ℝ
  adjacent_overlap : ℝ
  hall_side : ℝ

/-- The theorem stating the conditions and the conclusion about the hall's dimensions. -/
theorem hall_dimension (h : Hall) : 
  h.large_rug.side = 2 * h.small_rug.side ∧ 
  h.opposite_overlap = 4 ∧ 
  h.adjacent_overlap = 14 → 
  h.hall_side = 19 := by
  sorry


end hall_dimension_l290_29062


namespace bird_nest_problem_l290_29080

theorem bird_nest_problem (first_bird_initial : Nat) (first_bird_additional : Nat)
                          (second_bird_initial : Nat) (second_bird_additional : Nat)
                          (third_bird_initial : Nat) (third_bird_additional : Nat)
                          (first_bird_carry_capacity : Nat) (tree_drop_fraction : Nat) :
  first_bird_initial = 12 →
  first_bird_additional = 6 →
  second_bird_initial = 15 →
  second_bird_additional = 8 →
  third_bird_initial = 10 →
  third_bird_additional = 4 →
  first_bird_carry_capacity = 3 →
  tree_drop_fraction = 3 →
  (first_bird_initial * first_bird_additional +
   second_bird_initial * second_bird_additional +
   third_bird_initial * third_bird_additional = 232) ∧
  (((first_bird_initial * first_bird_additional) -
    (first_bird_initial * first_bird_additional / tree_drop_fraction)) /
    first_bird_carry_capacity = 16) :=
by sorry

end bird_nest_problem_l290_29080


namespace sum_of_coefficients_l290_29091

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^8 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + 
    a₄*(x - 1)^4 + a₅*(x - 1)^5 + a₆*(x - 1)^6 + a₇*(x - 1)^7 + a₈*(x - 1)^8 + 
    a₉*(x - 1)^9 + a₁₀*(x - 1)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end sum_of_coefficients_l290_29091


namespace henry_jill_age_ratio_l290_29083

/-- Proves that the ratio of Henry's age to Jill's age 11 years ago is 2:1 -/
theorem henry_jill_age_ratio : 
  ∀ (henry_age jill_age : ℕ),
  henry_age + jill_age = 40 →
  henry_age = 23 →
  jill_age = 17 →
  ∃ (k : ℕ), (henry_age - 11) = k * (jill_age - 11) →
  (henry_age - 11) / (jill_age - 11) = 2 :=
by
  sorry

end henry_jill_age_ratio_l290_29083


namespace planting_cost_l290_29018

def flower_cost : ℕ := 9
def clay_pot_cost : ℕ := flower_cost + 20
def soil_cost : ℕ := flower_cost - 2

def total_cost : ℕ := flower_cost + clay_pot_cost + soil_cost

theorem planting_cost : total_cost = 45 := by
  sorry

end planting_cost_l290_29018


namespace product_congruence_zero_mod_17_l290_29088

theorem product_congruence_zero_mod_17 : 
  (2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) % 17 = 0 := by
  sorry

end product_congruence_zero_mod_17_l290_29088


namespace true_compound_proposition_l290_29081

-- Define the propositions
def p : Prop := ∀ x : ℝ, x < 0 → x^3 < 0
def q : Prop := ∀ x : ℝ, x > 0 → Real.log x < 0

-- Theorem to prove
theorem true_compound_proposition : (¬p) ∨ (¬q) := by
  sorry

end true_compound_proposition_l290_29081


namespace geometric_sequence_common_ratio_l290_29054

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : is_geometric_sequence a q)
  (h_q_bounds : 0 < q ∧ q < 1/2)
  (h_property : ∀ k : ℕ, k > 0 → ∃ n : ℕ, a k - (a (k+1) + a (k+2)) = a n) :
  q = Real.sqrt 2 - 1 :=
sorry

end geometric_sequence_common_ratio_l290_29054


namespace fraction_problem_l290_29040

theorem fraction_problem (x : ℚ) : 
  x / (4 * x + 5) = 3 / 7 → x = -3 := by
  sorry

end fraction_problem_l290_29040


namespace exponent_division_l290_29079

theorem exponent_division (a : ℝ) (m n : ℕ) : a ^ m / a ^ n = a ^ (m - n) := by
  sorry

end exponent_division_l290_29079


namespace tile_c_in_rectangle_three_l290_29077

/-- Represents the four sides of a tile -/
structure TileSides :=
  (top : Nat)
  (right : Nat)
  (bottom : Nat)
  (left : Nat)

/-- Represents a tile with its sides -/
inductive Tile
| A
| B
| C
| D

/-- Represents the four rectangles -/
inductive Rectangle
| One
| Two
| Three
| Four

/-- Function to get the sides of a tile -/
def getTileSides (t : Tile) : TileSides :=
  match t with
  | Tile.A => ⟨6, 1, 3, 2⟩
  | Tile.B => ⟨3, 6, 2, 0⟩
  | Tile.C => ⟨4, 0, 5, 6⟩
  | Tile.D => ⟨2, 5, 1, 4⟩

/-- Predicate to check if two tiles can be placed adjacent to each other -/
def canBePlacedAdjacent (t1 t2 : Tile) (side : Nat → Nat) : Prop :=
  side (getTileSides t1).right = side (getTileSides t2).left

/-- The main theorem stating that Tile C must be placed in Rectangle 3 -/
theorem tile_c_in_rectangle_three :
  ∃ (placement : Tile → Rectangle),
    placement Tile.C = Rectangle.Three ∧
    (∀ t1 t2 : Tile, t1 ≠ t2 → placement t1 ≠ placement t2) ∧
    (∀ t1 t2 : Tile, 
      (placement t1 = Rectangle.One ∧ placement t2 = Rectangle.Two) ∨
      (placement t1 = Rectangle.Two ∧ placement t2 = Rectangle.Three) ∨
      (placement t1 = Rectangle.Three ∧ placement t2 = Rectangle.Four) →
      canBePlacedAdjacent t1 t2 id) := by
  sorry

end tile_c_in_rectangle_three_l290_29077


namespace pet_weights_l290_29012

/-- Given the weights of pets owned by Evan, Ivan, and Kara, prove their total weight -/
theorem pet_weights (evan_dog : ℕ) (ivan_dog : ℕ) (kara_cat : ℕ)
  (h1 : evan_dog = 63)
  (h2 : evan_dog = 7 * ivan_dog)
  (h3 : kara_cat = 5 * (evan_dog + ivan_dog)) :
  evan_dog + ivan_dog + kara_cat = 432 := by
  sorry

end pet_weights_l290_29012


namespace expression_evaluation_l290_29036

theorem expression_evaluation (a b : ℚ) (h1 : a = -1) (h2 : b = 1/2) :
  ((2*a + b)^2 - (2*a + b)*(2*a - b)) / (2*b) = -3/2 := by
  sorry

end expression_evaluation_l290_29036


namespace max_ratio_squared_l290_29008

theorem max_ratio_squared (a b c y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≥ b) (hbc : b ≥ c)
  (hy : 0 ≤ y ∧ y < a) (hz : 0 ≤ z ∧ z < c)
  (heq : a^2 + z^2 = c^2 + y^2 ∧ c^2 + y^2 = (a - y)^2 + (c - z)^2) :
  (a / c)^2 ≤ 4/3 :=
sorry

end max_ratio_squared_l290_29008


namespace outfit_combinations_l290_29092

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (hats : ℕ) : 
  shirts = 5 → pants = 4 → hats = 2 → shirts * pants * hats = 40 := by
  sorry

end outfit_combinations_l290_29092


namespace contrapositive_equivalence_l290_29009

theorem contrapositive_equivalence (a b : ℝ) : 
  (¬(a - 1 > b - 2) → ¬(a > b)) ↔ (a - 1 ≤ b - 2 → a ≤ b) := by sorry

end contrapositive_equivalence_l290_29009


namespace x_value_proof_l290_29037

theorem x_value_proof (x y : ℤ) (h1 : x + y = 20) (h2 : x - y = 36) : x = 28 := by
  sorry

end x_value_proof_l290_29037


namespace coefficient_x21_eq_932_l290_29014

open Nat BigOperators Finset

/-- The coefficient of x^21 in the expansion of (1 + x + x^2 + ... + x^20)(1 + x + x^2 + ... + x^10)^3 -/
def coefficient_x21 : ℕ :=
  (Finset.range 22).sum (λ i => i.choose 3) -
  3 * ((Finset.range 15).sum (λ i => i.choose 3)) +
  1

/-- The geometric series (1 + x + x^2 + ... + x^n) -/
def geometric_sum (n : ℕ) (x : ℝ) : ℝ :=
  (Finset.range (n + 1)).sum (λ i => x ^ i)

theorem coefficient_x21_eq_932 :
  coefficient_x21 = 932 :=
sorry

end coefficient_x21_eq_932_l290_29014


namespace smart_mart_science_kits_l290_29078

/-- The number of puzzles sold by Smart Mart -/
def puzzles_sold : ℕ := 36

/-- The difference between science kits and puzzles sold -/
def difference : ℕ := 9

/-- The number of science kits sold by Smart Mart -/
def science_kits_sold : ℕ := puzzles_sold + difference

/-- Theorem stating that Smart Mart sold 45 science kits -/
theorem smart_mart_science_kits : science_kits_sold = 45 := by
  sorry

end smart_mart_science_kits_l290_29078


namespace power_of_power_l290_29033

theorem power_of_power : (3^2)^4 = 6561 := by
  sorry

end power_of_power_l290_29033


namespace sum_digits_base8_888_l290_29024

/-- Converts a natural number to its base 8 representation as a list of digits -/
def toBase8 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Sums the digits in a list -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.sum

theorem sum_digits_base8_888 : sumDigits (toBase8 888) = 13 := by
  sorry

end sum_digits_base8_888_l290_29024


namespace expression_evaluation_l290_29010

theorem expression_evaluation : 
  81 + (128 / 16) + (15 * 12) - 250 - (180 / 3)^2 = -3581 := by
  sorry

end expression_evaluation_l290_29010


namespace y_squared_value_l290_29019

theorem y_squared_value (y : ℝ) (h : Real.sqrt (y + 16) - Real.sqrt (y - 16) = 2) : 
  y^2 = 9216 := by
sorry

end y_squared_value_l290_29019


namespace cube_sum_and_reciprocal_l290_29047

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end cube_sum_and_reciprocal_l290_29047


namespace equation_satisfied_l290_29042

theorem equation_satisfied (a b c : ℤ) (h1 : a = b) (h2 : b = c + 1) :
  a * (a - b) + b * (b - c) + c * (c - a) = 3 := by
  sorry

end equation_satisfied_l290_29042


namespace video_views_proof_l290_29097

/-- Calculates the total number of views for a video given initial views,
    a multiplier for increase after 4 days, and additional views after 2 more days. -/
def totalViews (initialViews : ℕ) (increaseMultiplier : ℕ) (additionalViews : ℕ) : ℕ :=
  initialViews + (increaseMultiplier * initialViews) + additionalViews

/-- Proves that given the specific conditions from the problem,
    the total number of views is 94000. -/
theorem video_views_proof :
  totalViews 4000 10 50000 = 94000 := by
  sorry

end video_views_proof_l290_29097


namespace total_toes_is_164_l290_29035

/-- Represents a race on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toes_per_hand (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of students of each race on the bus -/
def students (r : Race) : Nat :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes for a single being of a given race -/
def toes_per_being (r : Race) : Nat :=
  hands r * toes_per_hand r

/-- Total number of toes on the bus for a given race -/
def total_toes_per_race (r : Race) : Nat :=
  students r * toes_per_being r

/-- Total number of toes on the Popton school bus -/
def total_toes_on_bus : Nat :=
  total_toes_per_race Race.Hoopit + total_toes_per_race Race.Neglart

theorem total_toes_is_164 : total_toes_on_bus = 164 := by
  sorry

end total_toes_is_164_l290_29035


namespace monthly_repayment_l290_29089

theorem monthly_repayment (T M : ℝ) 
  (h1 : T / 2 = 6 * M)
  (h2 : T / 2 - 4 * M = 20) : 
  M = 10 := by sorry

end monthly_repayment_l290_29089


namespace pyramid_top_value_l290_29025

/-- Represents a pyramid structure with four levels -/
structure Pyramid :=
  (bottom_left : ℕ)
  (bottom_right : ℕ)
  (second_left : ℕ)
  (second_right : ℕ)
  (third_left : ℕ)
  (third_right : ℕ)
  (top : ℕ)

/-- Checks if the pyramid satisfies the product rule -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.bottom_left = p.second_left * p.third_left ∧
  p.bottom_right = p.second_right * p.third_right ∧
  p.second_left = p.top * p.third_left ∧
  p.second_right = p.top * p.third_right

theorem pyramid_top_value (p : Pyramid) :
  p.bottom_left = 300 ∧ 
  p.bottom_right = 1800 ∧ 
  p.second_left = 6 ∧ 
  p.second_right = 30 ∧
  is_valid_pyramid p →
  p.top = 60 := by
  sorry

end pyramid_top_value_l290_29025


namespace diagonal_bd_length_l290_29026

/-- A trapezoid with specific properties -/
structure SpecialTrapezoid where
  /-- The length of base AD -/
  ad : ℝ
  /-- The length of base BC -/
  bc : ℝ
  /-- The length of diagonal AC -/
  ac : ℝ
  /-- The circles on AB, BC, and CD as diameters intersect at one point -/
  circles_intersect : Prop

/-- The theorem about the length of diagonal BD in a special trapezoid -/
theorem diagonal_bd_length (t : SpecialTrapezoid)
    (h_ad : t.ad = 20)
    (h_bc : t.bc = 14)
    (h_ac : t.ac = 16) :
  ∃ (bd : ℝ), bd = 30 ∧ bd * bd = t.ac * t.ac + (t.ad - t.bc) * (t.ad - t.bc) / 4 := by
  sorry

end diagonal_bd_length_l290_29026


namespace symmetric_points_product_l290_29051

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric with respect to the origin if x₁ = -x₂ and y₁ = -y₂ -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

theorem symmetric_points_product (a b : ℝ) :
  symmetric_wrt_origin (a + 2) 2 4 (-b) →
  a * b = -12 := by
sorry

end symmetric_points_product_l290_29051


namespace sausage_cutting_theorem_l290_29099

/-- Represents the number of pieces produced when cutting along rings of a single color -/
def PiecesFromSingleColor : ℕ → ℕ := λ n => n + 1

/-- Represents the total number of pieces produced when cutting along rings of multiple colors -/
def TotalPieces (cuts : List ℕ) : ℕ :=
  (cuts.sum) + 1

theorem sausage_cutting_theorem (red yellow green : ℕ) 
  (h_red : PiecesFromSingleColor red = 5)
  (h_yellow : PiecesFromSingleColor yellow = 7)
  (h_green : PiecesFromSingleColor green = 11) :
  TotalPieces [red, yellow, green] = 21 := by
  sorry

#check sausage_cutting_theorem

end sausage_cutting_theorem_l290_29099


namespace crayon_selection_problem_l290_29030

theorem crayon_selection_problem :
  let n : ℕ := 20  -- Total number of crayons
  let k : ℕ := 6   -- Number of crayons to select
  Nat.choose n k = 38760 := by
  sorry

end crayon_selection_problem_l290_29030


namespace no_real_solutions_for_matrix_equation_l290_29056

theorem no_real_solutions_for_matrix_equation : 
  ¬∃ (x : ℝ), (3 * x * x - 8 = 2 * x^2 - 3 * x - 4) := by
  sorry

end no_real_solutions_for_matrix_equation_l290_29056


namespace seventh_term_is_384_l290_29082

/-- The nth term of a geometric sequence -/
def geometricSequenceTerm (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

/-- The seventh term of the specific geometric sequence -/
def seventhTerm : ℝ :=
  geometricSequenceTerm 6 (-2) 7

theorem seventh_term_is_384 : seventhTerm = 384 := by
  sorry

end seventh_term_is_384_l290_29082


namespace problem_statement_l290_29058

theorem problem_statement (x y : ℕ) (h1 : y > 3) 
  (h2 : x^2 + y^4 = 2*((x-6)^2 + (y+1)^2)) : 
  x^2 + y^4 = 1994 := by
  sorry

end problem_statement_l290_29058


namespace total_balloons_l290_29069

theorem total_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end total_balloons_l290_29069


namespace grunters_win_probability_l290_29046

def number_of_games : ℕ := 5
def win_probability : ℚ := 3/5

theorem grunters_win_probability :
  let p := win_probability
  let n := number_of_games
  (n.choose 4 * p^4 * (1-p)^1) + p^n = 1053/3125 := by sorry

end grunters_win_probability_l290_29046


namespace max_b_theorem_l290_29000

def is_lattice_point (x y : ℤ) : Prop := True

def line_equation (m : ℚ) (x : ℚ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x ∧ x ≤ 200 → ¬(is_lattice_point x y ∧ line_equation m x = y)

def max_b : ℚ := 68 / 203

theorem max_b_theorem :
  (∀ m : ℚ, 1/3 < m → m < max_b → no_lattice_points m) ∧
  (∀ b : ℚ, b > max_b → ∃ m : ℚ, 1/3 < m ∧ m < b ∧ ¬(no_lattice_points m)) :=
sorry

end max_b_theorem_l290_29000


namespace infinite_integer_solutions_l290_29072

theorem infinite_integer_solutions :
  ∃ (S : Set ℕ+), (Set.Infinite S) ∧
  (∀ n ∈ S, ¬ ∃ m : ℕ, n = m^3) ∧
  (∀ n ∈ S,
    let a : ℝ := (n : ℝ)^(1/3)
    let b : ℝ := 1 / (a - ⌊a⌋)
    let c : ℝ := 1 / (b - ⌊b⌋)
    ∃ r s t : ℤ, (r ≠ 0 ∨ s ≠ 0 ∨ t ≠ 0) ∧ r * a + s * b + t * c = 0) :=
by sorry

end infinite_integer_solutions_l290_29072


namespace sum_of_exponentials_l290_29064

theorem sum_of_exponentials (x y : ℝ) :
  (3^x + 3^(y+1) = 5 * Real.sqrt 3) →
  (3^(x+1) + 3^y = 3 * Real.sqrt 3) →
  3^x + 3^y = 2 * Real.sqrt 3 := by
sorry

end sum_of_exponentials_l290_29064


namespace smallest_m_for_nth_root_in_T_l290_29031

def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_root_in_T : 
  (∀ n : ℕ, n ≥ 12 → ∃ z ∈ T, z ^ n = 1) ∧ 
  (∀ m : ℕ, m < 12 → ∃ n : ℕ, n ≥ m ∧ ∀ z ∈ T, z ^ n ≠ 1) :=
sorry

end smallest_m_for_nth_root_in_T_l290_29031


namespace set_inclusion_iff_m_range_l290_29067

theorem set_inclusion_iff_m_range (m : ℝ) :
  ({x : ℝ | x^2 - 2*x - 3 ≤ 0} ⊆ {x : ℝ | |x - m| > 3}) ↔ 
  (m < -4 ∨ m > 6) :=
sorry

end set_inclusion_iff_m_range_l290_29067


namespace coefficient_x_squared_in_expansion_l290_29096

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (λ k => (Nat.choose 6 k) * (-2)^(6 - k) * x^k) = 
  240 * x^2 + (Finset.range 7).sum (λ k => if k ≠ 2 then (Nat.choose 6 k) * (-2)^(6 - k) * x^k else 0) := by
sorry

end coefficient_x_squared_in_expansion_l290_29096


namespace function_sum_positive_l290_29015

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, x < y → x < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = -f (-x + 2))

-- Define the theorem
theorem function_sum_positive (x₁ x₂ : ℝ) 
  (hx₁ : x₁ < 2) (hx₂ : x₂ > 2) (h : |x₁ - 2| < |x₂ - 2|) :
  f x₁ + f x₂ > 0 := by
  sorry

end function_sum_positive_l290_29015


namespace triangle_angle_measure_l290_29017

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 →
  E = 4 * F + 30 →
  D + E + F = 180 →
  F = 15 := by
sorry

end triangle_angle_measure_l290_29017


namespace pocket_money_problem_l290_29011

theorem pocket_money_problem (older_initial : ℕ) (younger_initial : ℕ) (difference : ℕ) (amount_given : ℕ) : 
  older_initial = 2800 →
  younger_initial = 1500 →
  older_initial - amount_given = younger_initial + amount_given + difference →
  difference = 360 →
  amount_given = 470 := by
  sorry

end pocket_money_problem_l290_29011


namespace cubic_roots_arithmetic_imply_p_eq_two_l290_29043

/-- A cubic polynomial with coefficient p -/
def cubic_poly (p : ℝ) (x : ℝ) : ℝ := x^3 - 6*p*x^2 + 5*p*x + 88

/-- The roots of the cubic polynomial form an arithmetic sequence -/
def roots_form_arithmetic_sequence (p : ℝ) : Prop :=
  ∃ (a d : ℝ), Set.range (λ i : Fin 3 => a + i.val * d) = {x | cubic_poly p x = 0}

/-- If the roots of x³ - 6px² + 5px + 88 = 0 form an arithmetic sequence, then p = 2 -/
theorem cubic_roots_arithmetic_imply_p_eq_two :
  ∀ p : ℝ, roots_form_arithmetic_sequence p → p = 2 := by
  sorry

end cubic_roots_arithmetic_imply_p_eq_two_l290_29043


namespace max_knights_between_knights_is_32_l290_29002

/-- Represents a seating arrangement of knights and samurais around a round table. -/
structure SeatingArrangement where
  total_knights : ℕ
  total_samurais : ℕ
  knights_with_samurai_right : ℕ

/-- The maximum number of knights that could be seated next to two other knights. -/
def max_knights_between_knights (arrangement : SeatingArrangement) : ℕ :=
  arrangement.total_knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights that could be seated next to two other knights
    in the given arrangement. -/
theorem max_knights_between_knights_is_32 (arrangement : SeatingArrangement) 
  (h1 : arrangement.total_knights = 40)
  (h2 : arrangement.total_samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#check max_knights_between_knights_is_32

end max_knights_between_knights_is_32_l290_29002


namespace spatial_pythagorean_quadruplet_l290_29068

theorem spatial_pythagorean_quadruplet (m n p q : ℤ) : 
  let x := 2 * m * p + 2 * n * q
  let y := |2 * m * q - 2 * n * p|
  let z := |m^2 + n^2 - p^2 - q^2|
  let u := m^2 + n^2 + p^2 + q^2
  (x^2 + y^2 + z^2 = u^2) →
  (∀ d : ℤ, d > 1 → ¬(d ∣ x ∧ d ∣ y ∧ d ∣ z ∧ d ∣ u)) →
  (∀ d : ℤ, d > 1 → ¬(d ∣ m ∧ d ∣ n ∧ d ∣ p ∧ d ∣ q)) :=
by sorry

end spatial_pythagorean_quadruplet_l290_29068


namespace miles_driven_l290_29057

-- Define the efficiency of the car in miles per gallon
def miles_per_gallon : ℝ := 45

-- Define the price of gas per gallon
def price_per_gallon : ℝ := 5

-- Define the amount spent on gas
def amount_spent : ℝ := 25

-- Theorem to prove
theorem miles_driven : 
  miles_per_gallon * (amount_spent / price_per_gallon) = 225 := by
sorry

end miles_driven_l290_29057


namespace arithmetic_progression_of_primes_l290_29085

theorem arithmetic_progression_of_primes (p : ℕ) (a : ℕ → ℕ) (d : ℕ) :
  Prime p →
  (∀ i, i ∈ Finset.range p → Prime (a i)) →
  (∀ i j, i < j → j < p → a j - a i = (j - i) * d) →
  a 0 > p →
  p ∣ d :=
by sorry

end arithmetic_progression_of_primes_l290_29085


namespace investment_growth_l290_29048

/-- Calculates the total amount after compound interest is applied for a given number of periods -/
def compound_interest (principal : ℝ) (rate : ℝ) (periods : ℕ) : ℝ :=
  principal * (1 + rate) ^ periods

/-- The problem statement -/
theorem investment_growth : compound_interest 300 0.1 2 = 363 := by
  sorry

end investment_growth_l290_29048


namespace unique_solution_system_l290_29063

theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = z + w + z * w * z ∧
    y = w + x + w * z * x ∧
    z = x + y + x * y * x ∧
    w = y + z + z * y * z :=
by sorry

end unique_solution_system_l290_29063


namespace negation_of_implication_l290_29086

theorem negation_of_implication (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) := by sorry

end negation_of_implication_l290_29086


namespace percent_composition_l290_29041

-- Define the % operations
def percent_right (x : ℤ) : ℤ := 8 - x
def percent_left (x : ℤ) : ℤ := x - 8

-- Theorem statement
theorem percent_composition : percent_left (percent_right 10) = -10 := by
  sorry

end percent_composition_l290_29041


namespace trapezoid_diagonal_triangles_l290_29076

/-- Represents a trapezoid with given area and bases -/
structure Trapezoid where
  area : ℝ
  base1 : ℝ
  base2 : ℝ

/-- Represents the areas of triangles formed by diagonals in a trapezoid -/
structure DiagonalTriangles where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ

/-- 
Given a trapezoid with area 3 and bases 1 and 2, 
the areas of the triangles formed by its diagonals are 1/3, 2/3, 2/3, and 4/3
-/
theorem trapezoid_diagonal_triangles (t : Trapezoid) 
  (h1 : t.area = 3) 
  (h2 : t.base1 = 1) 
  (h3 : t.base2 = 2) : 
  ∃ (d : DiagonalTriangles), 
    d.area1 = 1/3 ∧ 
    d.area2 = 2/3 ∧ 
    d.area3 = 2/3 ∧ 
    d.area4 = 4/3 := by
  sorry


end trapezoid_diagonal_triangles_l290_29076


namespace geometric_sequence_sum_l290_29039

/-- The sum of the first n terms of a geometric sequence with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of our geometric sequence -/
def a : ℚ := 1/4

/-- The common ratio of our geometric sequence -/
def r : ℚ := 1/4

/-- The number of terms we're summing -/
def n : ℕ := 6

theorem geometric_sequence_sum : 
  geometricSum a r n = 4095/12288 := by sorry

end geometric_sequence_sum_l290_29039


namespace least_k_cubed_divisible_by_336_l290_29003

theorem least_k_cubed_divisible_by_336 :
  ∃ (k : ℕ), k > 0 ∧ k^3 % 336 = 0 ∧ ∀ (m : ℕ), m > 0 → m^3 % 336 = 0 → k ≤ m :=
by
  -- The proof would go here
  sorry

end least_k_cubed_divisible_by_336_l290_29003


namespace locus_not_hyperbola_ellipse_intersection_l290_29004

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent (c1 c2 : Circle) : Prop :=
  dist c1.center c2.center = c1.radius + c2.radius ∨
  dist c1.center c2.center = |c1.radius - c2.radius|

def locus (O₁ O₂ : Circle) : Set (ℝ × ℝ) :=
  {P | ∃ (r : ℝ), tangent O₁ ⟨P, r⟩ ∧ tangent O₂ ⟨P, r⟩}

def hyperbola (f₁ f₂ : ℝ × ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {P | |dist P f₁ - dist P f₂| = 2 * a}

def ellipse (f₁ f₂ : ℝ × ℝ) (a : ℝ) : Set (ℝ × ℝ) :=
  {P | dist P f₁ + dist P f₂ = 2 * a}

theorem locus_not_hyperbola_ellipse_intersection
  (O₁ O₂ : Circle) (f₁ f₂ g₁ g₂ : ℝ × ℝ) (a b : ℝ) :
  locus O₁ O₂ ≠ hyperbola f₁ f₂ a ∩ ellipse g₁ g₂ b :=
sorry

end locus_not_hyperbola_ellipse_intersection_l290_29004


namespace ten_thousandths_place_of_5_32_l290_29075

theorem ten_thousandths_place_of_5_32 : 
  ∃ (n : ℕ), (5 : ℚ) / 32 = (n * 10000 + 2) / 100000 ∧ n < 10000 := by
  sorry

end ten_thousandths_place_of_5_32_l290_29075


namespace pie_division_l290_29001

theorem pie_division (total_pie : ℚ) (people : ℕ) : 
  total_pie = 8 / 9 → people = 3 → total_pie / people = 8 / 27 := by
  sorry

end pie_division_l290_29001


namespace line_through_two_points_l290_29087

/-- Given a line x = 6y + 5 passing through points (m, n) and (m + 2, n + p) in a rectangular coordinate system, prove that p = 1/3 -/
theorem line_through_two_points (m n : ℝ) : 
  (m = 6 * n + 5) → (m + 2 = 6 * (n + p) + 5) → p = 1/3 :=
by
  sorry

end line_through_two_points_l290_29087


namespace seashells_problem_l290_29059

theorem seashells_problem (given_away : ℕ) (remaining : ℕ) :
  given_away = 18 → remaining = 17 → given_away + remaining = 35 :=
by sorry

end seashells_problem_l290_29059


namespace exist_consecutive_lucky_tickets_l290_29098

/-- A function that calculates the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a number is a six-digit number -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- A predicate that checks if a number is lucky (sum of digits divisible by 7) -/
def is_lucky (n : ℕ) : Prop := sum_of_digits n % 7 = 0

/-- Theorem stating that there exist two consecutive six-digit numbers that are both lucky -/
theorem exist_consecutive_lucky_tickets : 
  ∃ n : ℕ, is_six_digit n ∧ is_six_digit (n + 1) ∧ is_lucky n ∧ is_lucky (n + 1) := by
  sorry

end exist_consecutive_lucky_tickets_l290_29098


namespace sum_lent_problem_l290_29029

/-- Proves that given a sum P lent at 5% per annum simple interest for 8 years,
    if the interest is $360 less than P, then P equals $600. -/
theorem sum_lent_problem (P : ℝ) : 
  (P * 0.05 * 8 = P - 360) → P = 600 := by
  sorry

end sum_lent_problem_l290_29029


namespace real_estate_commission_l290_29053

/-- Calculate the commission for a real estate agent given the selling price and commission rate -/
def calculate_commission (selling_price : ℝ) (commission_rate : ℝ) : ℝ :=
  selling_price * commission_rate

/-- Theorem stating that the commission for a house sold at $148,000 with a 6% commission rate is $8,880 -/
theorem real_estate_commission :
  let selling_price : ℝ := 148000
  let commission_rate : ℝ := 0.06
  calculate_commission selling_price commission_rate = 8880 := by
  sorry

end real_estate_commission_l290_29053


namespace round_trip_speed_l290_29049

/-- Proves that given the conditions of the round trip, the speed from B to A is 45 miles per hour -/
theorem round_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) (speed_ba : ℝ) : 
  distance = 180 →
  speed_ab = 90 →
  avg_speed = 60 →
  speed_ba = (2 * distance * avg_speed) / (2 * distance - avg_speed * (distance / speed_ab)) →
  speed_ba = 45 :=
by sorry

end round_trip_speed_l290_29049


namespace quadratic_inequality_range_l290_29084

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) := by
  sorry

end quadratic_inequality_range_l290_29084


namespace geometric_sequence_101st_term_l290_29034

def geometricSequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * r^(n - 1)

theorem geometric_sequence_101st_term :
  let a := 12
  let second_term := -36
  let r := second_term / a
  let n := 101
  geometricSequence a r n = 12 * 3^100 := by
sorry

end geometric_sequence_101st_term_l290_29034


namespace similar_triangles_leg_sum_l290_29007

theorem similar_triangles_leg_sum (A₁ A₂ : ℝ) (h : ℝ) (s : ℝ) :
  A₁ = 18 →
  A₂ = 288 →
  h = 9 →
  (A₂ / A₁ = (s / h) ^ 2) →
  s = 4 * Real.sqrt 153 :=
by sorry

end similar_triangles_leg_sum_l290_29007


namespace functional_equation_solution_l290_29066

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x - f y) = f x * f y) : 
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end functional_equation_solution_l290_29066


namespace eighth_hexagonal_number_l290_29027

/-- Definition of hexagonal numbers -/
def hexagonal (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The 8th hexagonal number is 120 -/
theorem eighth_hexagonal_number : hexagonal 8 = 120 := by
  sorry

end eighth_hexagonal_number_l290_29027


namespace abs_rational_nonnegative_l290_29074

theorem abs_rational_nonnegative (a : ℚ) : 0 ≤ |a| := by
  sorry

end abs_rational_nonnegative_l290_29074


namespace triangle_height_proof_l290_29032

/-- Given a square with side length s, a rectangle with base s and height h,
    and an isosceles triangle with base s and height h,
    prove that h = 2s/3 when the combined area of the rectangle and triangle
    equals the area of the square. -/
theorem triangle_height_proof (s : ℝ) (h : ℝ) : 
  s > 0 → h > 0 → s * h + (s * h) / 2 = s^2 → h = 2 * s / 3 := by
  sorry

end triangle_height_proof_l290_29032


namespace circle_point_range_l290_29044

/-- Given a point A(0,-3) and a circle C: (x-a)^2 + (y-a+2)^2 = 1,
    if there exists a point M on C such that MA = 2MO,
    then 0 ≤ a ≤ 3 -/
theorem circle_point_range (a : ℝ) :
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧
              (x^2 + (y + 3)^2) = 4 * (x^2 + y^2)) →
  0 ≤ a ∧ a ≤ 3 := by
sorry

end circle_point_range_l290_29044


namespace triangle_area_l290_29005

/-- Given a triangle ABC where BC = 10 cm and the height from A to BC is 12 cm,
    prove that the area of triangle ABC is 60 square centimeters. -/
theorem triangle_area (BC height : ℝ) (h1 : BC = 10) (h2 : height = 12) :
  (1 / 2) * BC * height = 60 :=
by sorry

end triangle_area_l290_29005


namespace circle_radius_l290_29052

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 34 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (4, 5)

/-- The theorem stating that the radius of the circle is √7 -/
theorem circle_radius :
  ∃ (r : ℝ), r > 0 ∧
  (∀ (x y : ℝ), circle_equation x y ↔ 
    (x - circle_center.1)^2 + (y - circle_center.2)^2 = r^2) ∧
  r = Real.sqrt 7 :=
sorry

end circle_radius_l290_29052


namespace robins_haircut_l290_29016

/-- Given Robin's initial hair length and current hair length, 
    prove the amount of hair cut is the difference between these lengths. -/
theorem robins_haircut (initial_length current_length : ℕ) 
  (h1 : initial_length = 17)
  (h2 : current_length = 13) :
  initial_length - current_length = 4 := by
  sorry

end robins_haircut_l290_29016


namespace pathway_layers_l290_29065

def bricks_in_layer (n : ℕ) : ℕ :=
  if n % 2 = 1 then 4 else 4 * 2^((n / 2) - 1)

def total_bricks (n : ℕ) : ℕ :=
  (List.range n).map (λ i => bricks_in_layer (i + 1)) |> List.sum

theorem pathway_layers : ∃ n : ℕ, n > 0 ∧ total_bricks n = 280 :=
  sorry

end pathway_layers_l290_29065


namespace function_constraint_l290_29090

theorem function_constraint (a : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → a * x + 6 ≤ 10) ↔ (a = -4 ∨ a = 2 ∨ a = 0) :=
by sorry

end function_constraint_l290_29090


namespace cosine_product_equals_one_eighth_two_minus_sqrt_two_l290_29020

theorem cosine_product_equals_one_eighth_two_minus_sqrt_two :
  (1 + Real.cos (π / 9)) * (1 + Real.cos (4 * π / 9)) *
  (1 + Real.cos (5 * π / 9)) * (1 + Real.cos (8 * π / 9)) =
  1 / 8 * (2 - Real.sqrt 2) := by
sorry

end cosine_product_equals_one_eighth_two_minus_sqrt_two_l290_29020


namespace expected_worth_unfair_coin_l290_29006

/-- An unfair coin with given probabilities and payoffs -/
structure UnfairCoin where
  probHeads : ℚ
  probTails : ℚ
  payoffHeads : ℚ
  payoffTails : ℚ
  prob_sum : probHeads + probTails = 1

/-- The expected value of a flip of the unfair coin -/
def expectedValue (coin : UnfairCoin) : ℚ :=
  coin.probHeads * coin.payoffHeads + coin.probTails * coin.payoffTails

/-- Theorem: The expected worth of a specific unfair coin flip -/
theorem expected_worth_unfair_coin :
  ∃ (coin : UnfairCoin),
    coin.probHeads = 2/3 ∧
    coin.probTails = 1/3 ∧
    coin.payoffHeads = 5 ∧
    coin.payoffTails = -9 ∧
    expectedValue coin = 1/3 := by
  sorry

end expected_worth_unfair_coin_l290_29006


namespace positive_numbers_equality_l290_29055

theorem positive_numbers_equality (a b : ℝ) : 
  0 < a → 0 < b → a^b = b^a → b = 3*a → a = Real.sqrt 3 := by
  sorry

end positive_numbers_equality_l290_29055


namespace inequality_proof_l290_29023

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b * (a + b) + b * c * (b + c) + a * c * (a + c) - 6 * a * b * c ≥ 0 := by
  sorry

end inequality_proof_l290_29023


namespace new_building_windows_l290_29050

/-- The number of windows needed for a new building --/
def total_windows (installed : ℕ) (hours_per_window : ℕ) (remaining_hours : ℕ) : ℕ :=
  installed + remaining_hours / hours_per_window

/-- Proof that the total number of windows needed is 14 --/
theorem new_building_windows :
  total_windows 5 4 36 = 14 :=
by sorry

end new_building_windows_l290_29050


namespace cubic_polynomial_root_l290_29071

theorem cubic_polynomial_root (a b c : ℚ) :
  (∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 2 - Real.sqrt 5) →
  (∀ x y : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ y^3 + a*y^2 + b*y + c = 0 → x + y = 4) →
  (-4 : ℝ)^3 + a*(-4 : ℝ)^2 + b*(-4 : ℝ) + c = 0 :=
by sorry

end cubic_polynomial_root_l290_29071


namespace max_imaginary_part_at_84_degrees_l290_29022

/-- The polynomial whose roots we're investigating -/
def f (z : ℂ) : ℂ := z^12 - z^9 + z^6 - z^3 + 1

/-- The set of roots of the polynomial -/
def roots : Set ℂ := {z : ℂ | f z = 0}

/-- The set of angles corresponding to the roots -/
def root_angles : Set Real := {θ : Real | ∃ z ∈ roots, z = Complex.exp (θ * Complex.I)}

/-- The theorem stating the maximum imaginary part occurs at 84 degrees -/
theorem max_imaginary_part_at_84_degrees :
  ∃ θ ∈ root_angles,
    θ * Real.pi / 180 = 84 * Real.pi / 180 ∧
    ∀ φ ∈ root_angles, -Real.pi/2 ≤ φ ∧ φ ≤ Real.pi/2 →
      Complex.abs (Complex.sin (Complex.ofReal φ)) ≤ 
      Complex.abs (Complex.sin (Complex.ofReal (θ * Real.pi / 180))) :=
sorry

end max_imaginary_part_at_84_degrees_l290_29022
