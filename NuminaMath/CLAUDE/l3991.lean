import Mathlib

namespace NUMINAMATH_CALUDE_total_apples_in_basket_l3991_399171

def initial_apples : Nat := 8
def added_apples : Nat := 7

theorem total_apples_in_basket : initial_apples + added_apples = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_in_basket_l3991_399171


namespace NUMINAMATH_CALUDE_negation_of_implication_l3991_399123

theorem negation_of_implication (a b c : ℝ) : 
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) ↔ (a + b + c = 3 ∧ a^2 + b^2 + c^2 < 3) :=
sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3991_399123


namespace NUMINAMATH_CALUDE_total_good_vegetables_l3991_399100

def carrots_day1 : ℕ := 23
def carrots_day2 : ℕ := 47
def rotten_carrots_day1 : ℕ := 10
def rotten_carrots_day2 : ℕ := 15

def tomatoes_day1 : ℕ := 34
def tomatoes_day2 : ℕ := 50
def rotten_tomatoes_day1 : ℕ := 5
def rotten_tomatoes_day2 : ℕ := 7

def cucumbers_day1 : ℕ := 42
def cucumbers_day2 : ℕ := 38
def rotten_cucumbers_day1 : ℕ := 7
def rotten_cucumbers_day2 : ℕ := 12

theorem total_good_vegetables :
  (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2) +
  (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2) +
  (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2) = 178 :=
by sorry

end NUMINAMATH_CALUDE_total_good_vegetables_l3991_399100


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3991_399154

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed from the decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 70

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3991_399154


namespace NUMINAMATH_CALUDE_polygon_with_1800_degree_sum_is_dodecagon_l3991_399130

theorem polygon_with_1800_degree_sum_is_dodecagon :
  ∀ n : ℕ, 
  n ≥ 3 →
  (n - 2) * 180 = 1800 →
  n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_with_1800_degree_sum_is_dodecagon_l3991_399130


namespace NUMINAMATH_CALUDE_special_ellipse_intersecting_line_l3991_399177

/-- An ellipse with its upper vertex and left focus on a given line --/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 + y^2 / b^2 = 1
  vertex_focus_line : ℝ → ℝ → Prop := fun x y => x - y + 2 = 0

/-- A line intersecting the ellipse --/
structure IntersectingLine (E : SpecialEllipse) where
  l : ℝ → ℝ → Prop
  P : ℝ × ℝ
  Q : ℝ × ℝ
  h_P : E.equation P.1 P.2 ∧ l P.1 P.2
  h_Q : E.equation Q.1 Q.2 ∧ l Q.1 Q.2
  h_midpoint : (P.1 + Q.1) / 2 = -1 ∧ (P.2 + Q.2) / 2 = 1

/-- The main theorem --/
theorem special_ellipse_intersecting_line 
  (E : SpecialEllipse) 
  (h_E : E.a^2 = 8 ∧ E.b^2 = 4) 
  (l : IntersectingLine E) : 
  l.l = fun x y => x - 2*y + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_special_ellipse_intersecting_line_l3991_399177


namespace NUMINAMATH_CALUDE_mikes_games_l3991_399180

theorem mikes_games (initial_amount : ℕ) (spent_amount : ℕ) (game_cost : ℕ) : 
  initial_amount = 101 →
  spent_amount = 47 →
  game_cost = 6 →
  (initial_amount - spent_amount) / game_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_mikes_games_l3991_399180


namespace NUMINAMATH_CALUDE_square_flags_count_l3991_399110

theorem square_flags_count (total_fabric : ℕ) (square_size wide_size tall_size : ℕ × ℕ)
  (wide_count tall_count : ℕ) (fabric_left : ℕ) :
  total_fabric = 1000 →
  square_size = (4, 4) →
  wide_size = (5, 3) →
  tall_size = (3, 5) →
  wide_count = 20 →
  tall_count = 10 →
  fabric_left = 294 →
  ∃ (square_count : ℕ),
    square_count = 16 ∧
    square_count * (square_size.1 * square_size.2) +
    wide_count * (wide_size.1 * wide_size.2) +
    tall_count * (tall_size.1 * tall_size.2) +
    fabric_left = total_fabric :=
by sorry

end NUMINAMATH_CALUDE_square_flags_count_l3991_399110


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3991_399105

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (b < -1 → |a| + |b| > 1) ∧ 
  ∃ a b : ℝ, |a| + |b| > 1 ∧ b ≥ -1 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3991_399105


namespace NUMINAMATH_CALUDE_triangle_area_specific_l3991_399117

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

theorem triangle_area_specific : 
  ∀ (a b c : ℝ) (A B C : ℝ),
    b = 2 →
    c = 2 * Real.sqrt 2 →
    C = π / 4 →
    triangle_area a b c A B C = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_specific_l3991_399117


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3991_399131

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3991_399131


namespace NUMINAMATH_CALUDE_domain_of_g_l3991_399196

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-12) 3

-- Define the function g in terms of f
def g (x : ℝ) : ℝ := f (3 * x)

-- State the theorem
theorem domain_of_g : 
  {x : ℝ | g x ∈ Set.range f} = Set.Icc (-4) 1 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l3991_399196


namespace NUMINAMATH_CALUDE_chess_group_players_l3991_399194

theorem chess_group_players (n : ℕ) : n * (n - 1) / 2 = 1225 → n = 50 := by
  sorry

end NUMINAMATH_CALUDE_chess_group_players_l3991_399194


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l3991_399179

theorem parabola_fixed_point :
  ∀ t : ℝ, 3 * (2 : ℝ)^2 + t * 2 - 2 * t = 12 := by
  sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l3991_399179


namespace NUMINAMATH_CALUDE_cristine_lemons_left_l3991_399145

def dozen : ℕ := 12

def lemons_given_to_neighbor (total : ℕ) : ℕ := total / 4

def lemons_exchanged_for_oranges : ℕ := 2

theorem cristine_lemons_left (initial_lemons : ℕ) 
  (h1 : initial_lemons = dozen) 
  (h2 : lemons_given_to_neighbor initial_lemons = initial_lemons / 4) 
  (h3 : lemons_exchanged_for_oranges = 2) : 
  initial_lemons - lemons_given_to_neighbor initial_lemons - lemons_exchanged_for_oranges = 7 := by
  sorry

end NUMINAMATH_CALUDE_cristine_lemons_left_l3991_399145


namespace NUMINAMATH_CALUDE_remainder_sum_l3991_399111

theorem remainder_sum (c d : ℤ) 
  (hc : c % 100 = 78) 
  (hd : d % 150 = 123) : 
  (c + d) % 50 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l3991_399111


namespace NUMINAMATH_CALUDE_T_greater_than_N_l3991_399102

/-- Represents an 8x8 chessboard -/
def Board := Fin 8 → Fin 8 → Bool

/-- Represents a domino placement on the board -/
def DominoPlacement := List (Fin 8 × Fin 8 × Bool)

/-- Returns true if the given domino placement is valid on the board -/
def isValidPlacement (board : Board) (placement : DominoPlacement) : Prop :=
  sorry

/-- Counts the number of valid domino placements for a given number of dominoes -/
def countPlacements (n : Nat) : Nat :=
  sorry

/-- The number of ways to place 32 dominoes -/
def N : Nat := countPlacements 32

/-- The number of ways to place 24 dominoes -/
def T : Nat := countPlacements 24

/-- Theorem stating that T is greater than N -/
theorem T_greater_than_N : T > N := by
  sorry

end NUMINAMATH_CALUDE_T_greater_than_N_l3991_399102


namespace NUMINAMATH_CALUDE_emily_purchase_cost_l3991_399170

/-- Calculate the total cost of Emily's purchase including discount, tax, and installation fee -/
theorem emily_purchase_cost :
  let curtain_price : ℚ := 30
  let curtain_quantity : ℕ := 2
  let print_price : ℚ := 15
  let print_quantity : ℕ := 9
  let discount_rate : ℚ := 0.1
  let tax_rate : ℚ := 0.08
  let installation_fee : ℚ := 50

  let subtotal : ℚ := curtain_price * curtain_quantity + print_price * print_quantity
  let discounted_total : ℚ := subtotal * (1 - discount_rate)
  let taxed_total : ℚ := discounted_total * (1 + tax_rate)
  let total_cost : ℚ := taxed_total + installation_fee

  total_cost = 239.54 := by sorry

end NUMINAMATH_CALUDE_emily_purchase_cost_l3991_399170


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_five_digit_palindrome_product_l3991_399113

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem statement -/
theorem smallest_three_digit_palindrome_non_five_digit_palindrome_product :
  isThreeDigitPalindrome 131 ∧
  ¬(isFiveDigitPalindrome (131 * 103)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 131 → isFiveDigitPalindrome (n * 103) :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_non_five_digit_palindrome_product_l3991_399113


namespace NUMINAMATH_CALUDE_pizza_slices_l3991_399139

theorem pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) (h1 : num_pizzas = 7) (h2 : slices_per_pizza = 2) :
  num_pizzas * slices_per_pizza = 14 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l3991_399139


namespace NUMINAMATH_CALUDE_conjecture_proof_l3991_399166

theorem conjecture_proof (n : ℕ) (h : n ≥ 1) :
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_conjecture_proof_l3991_399166


namespace NUMINAMATH_CALUDE_problem_statement_l3991_399161

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 11) :
  c + 1 / b = 5 / 19 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3991_399161


namespace NUMINAMATH_CALUDE_restaurant_bill_total_l3991_399158

theorem restaurant_bill_total (number_of_people : ℕ) (individual_payment : ℕ) : 
  number_of_people = 3 → individual_payment = 45 → number_of_people * individual_payment = 135 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_total_l3991_399158


namespace NUMINAMATH_CALUDE_interest_rate_is_one_percent_l3991_399174

/-- Calculates the interest rate given principal, time, and total simple interest -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (total_interest : ℚ) : ℚ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that given the problem conditions, the interest rate is 1% -/
theorem interest_rate_is_one_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (total_interest : ℚ) 
  (h1 : principal = 44625)
  (h2 : time = 9)
  (h3 : total_interest = 4016.25) :
  calculate_interest_rate principal time total_interest = 1 := by
  sorry

#eval calculate_interest_rate 44625 9 4016.25

end NUMINAMATH_CALUDE_interest_rate_is_one_percent_l3991_399174


namespace NUMINAMATH_CALUDE_cube_inequality_l3991_399109

theorem cube_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l3991_399109


namespace NUMINAMATH_CALUDE_custom_mult_comm_custom_mult_comm_complex_l3991_399112

/-- Custom multiplication operation -/
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating the commutativity of the custom multiplication -/
theorem custom_mult_comm (a b : ℝ) : custom_mult a b = custom_mult b a := by
  sorry

/-- Theorem stating the commutativity of the custom multiplication with a complex expression -/
theorem custom_mult_comm_complex (a b c : ℝ) : custom_mult a (b - c) = custom_mult (b - c) a := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_comm_custom_mult_comm_complex_l3991_399112


namespace NUMINAMATH_CALUDE_papaya_tree_growth_ratio_l3991_399186

/-- Papaya tree growth problem -/
theorem papaya_tree_growth_ratio : 
  ∀ (growth_1 growth_2 growth_3 growth_4 growth_5 : ℝ),
  growth_1 = 2 →
  growth_2 = growth_1 * 1.5 →
  growth_3 = growth_2 * 1.5 →
  growth_5 = growth_4 / 2 →
  growth_1 + growth_2 + growth_3 + growth_4 + growth_5 = 23 →
  growth_4 / growth_3 = 2 := by
sorry


end NUMINAMATH_CALUDE_papaya_tree_growth_ratio_l3991_399186


namespace NUMINAMATH_CALUDE_min_players_distinct_scores_l3991_399108

/-- A round robin chess tournament where each player plays every other player exactly once. -/
structure Tournament (n : ℕ) where
  scores : Fin n → ℚ

/-- Property P(m) for a tournament -/
def hasPropertyP (t : Tournament n) (m : ℕ) : Prop :=
  ∀ (S : Finset (Fin n)), S.card = m →
    (∃ (w : Fin n), w ∈ S ∧ ∀ (x : Fin n), x ∈ S ∧ x ≠ w → t.scores w > t.scores x) ∧
    (∃ (l : Fin n), l ∈ S ∧ ∀ (x : Fin n), x ∈ S ∧ x ≠ l → t.scores l < t.scores x)

/-- All scores in the tournament are distinct -/
def hasDistinctScores (t : Tournament n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → t.scores i ≠ t.scores j

/-- The main theorem -/
theorem min_players_distinct_scores (m : ℕ) (h : m ≥ 4) :
  (∀ (n : ℕ), n ≥ 2*m - 3 →
    ∀ (t : Tournament n), hasPropertyP t m → hasDistinctScores t) ∧
  (∃ (t : Tournament (2*m - 4)), hasPropertyP t m ∧ ¬hasDistinctScores t) :=
sorry

end NUMINAMATH_CALUDE_min_players_distinct_scores_l3991_399108


namespace NUMINAMATH_CALUDE_eggplant_seed_distribution_l3991_399114

theorem eggplant_seed_distribution (total_seeds : ℕ) (num_pots : ℕ) (seeds_in_last_pot : ℕ) :
  total_seeds = 10 →
  num_pots = 4 →
  seeds_in_last_pot = 1 →
  ∃ (seeds_per_pot : ℕ),
    seeds_per_pot * (num_pots - 1) + seeds_in_last_pot = total_seeds ∧
    seeds_per_pot = 3 :=
by sorry

end NUMINAMATH_CALUDE_eggplant_seed_distribution_l3991_399114


namespace NUMINAMATH_CALUDE_walking_biking_time_difference_l3991_399107

/-- Proves that the difference between walking and biking time is 4 minutes -/
theorem walking_biking_time_difference :
  let blocks : ℕ := 6
  let walk_time_per_block : ℚ := 1
  let bike_time_per_block : ℚ := 20 / 60
  (blocks * walk_time_per_block) - (blocks * bike_time_per_block) = 4 := by
  sorry

end NUMINAMATH_CALUDE_walking_biking_time_difference_l3991_399107


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l3991_399199

theorem unique_solution_implies_a_equals_two (a : ℝ) : 
  (∃! x : ℝ, 0 ≤ x^2 - a*x + a ∧ x^2 - a*x + a ≤ 1) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_equals_two_l3991_399199


namespace NUMINAMATH_CALUDE_min_product_of_tangents_l3991_399148

theorem min_product_of_tangents (α β γ : Real) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_acute_γ : 0 < γ ∧ γ < π / 2)
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  Real.tan α * Real.tan β * Real.tan γ ≥ 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_tangents_l3991_399148


namespace NUMINAMATH_CALUDE_aluminum_mass_calculation_l3991_399126

/-- Given two parts with equal volume but different densities, 
    calculate the mass of one part given the mass difference. -/
theorem aluminum_mass_calculation 
  (ρA ρM : ℝ) -- densities of aluminum and copper
  (Δm : ℝ) -- mass difference
  (h1 : ρA = 2700) -- density of aluminum
  (h2 : ρM = 8900) -- density of copper
  (h3 : Δm = 0.06) -- mass difference in kg
  (h4 : ρM > ρA) -- copper is denser than aluminum
  : ∃ (mA : ℝ), mA = (ρA * Δm) / (ρM - ρA) :=
by sorry

end NUMINAMATH_CALUDE_aluminum_mass_calculation_l3991_399126


namespace NUMINAMATH_CALUDE_number_difference_l3991_399198

theorem number_difference (x y : ℝ) : 
  x + y = 50 →
  3 * max x y - 5 * min x y = 10 →
  |x - y| = 15 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l3991_399198


namespace NUMINAMATH_CALUDE_trig_simplification_l3991_399141

theorem trig_simplification (x y : ℝ) :
  (Real.cos x)^2 + (Real.sin x)^2 + (Real.cos (x + y))^2 - 
  2 * (Real.cos x) * (Real.cos y) * (Real.cos (x + y)) - 
  (Real.sin x) * (Real.sin y) = (Real.sin (x - y))^2 := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3991_399141


namespace NUMINAMATH_CALUDE_four_digit_sum_problem_l3991_399119

def is_valid_digit (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 9

def to_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d

theorem four_digit_sum_problem (a b c d : ℕ) :
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  to_number a b c d + to_number d c b a = 11990 →
  (a = 1 ∧ b = 9 ∧ c = 9 ∧ d = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_problem_l3991_399119


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3991_399195

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem complement_intersection_theorem : (U \ A) ∩ B = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3991_399195


namespace NUMINAMATH_CALUDE_equation_solution_l3991_399189

theorem equation_solution (x : ℝ) : 
  (∀ z : ℝ, 10 * x * z - 15 * z + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3991_399189


namespace NUMINAMATH_CALUDE_valid_outfit_choices_l3991_399125

/-- Represents the number of valid outfit choices given specific clothing items and constraints. -/
theorem valid_outfit_choices : 
  -- Define the number of shirts, pants, and their colors
  let num_shirts : ℕ := 6
  let num_pants : ℕ := 6
  let num_colors : ℕ := 6
  
  -- Define the number of hats
  let num_patterned_hats : ℕ := 6
  let num_solid_hats : ℕ := 6
  let total_hats : ℕ := num_patterned_hats + num_solid_hats
  
  -- Calculate total combinations
  let total_combinations : ℕ := num_shirts * num_pants * total_hats
  
  -- Calculate invalid combinations
  let same_color_combinations : ℕ := num_colors
  let pattern_mismatch_combinations : ℕ := num_patterned_hats * num_shirts * (num_pants - 1)
  
  -- Calculate valid combinations
  let valid_combinations : ℕ := total_combinations - same_color_combinations - pattern_mismatch_combinations
  
  -- Prove that the number of valid outfit choices is 246
  valid_combinations = 246 := by
    sorry

end NUMINAMATH_CALUDE_valid_outfit_choices_l3991_399125


namespace NUMINAMATH_CALUDE_f_minimum_value_l3991_399185

def f (x : ℝ) : ℝ := |2*x - 1| + |3*x - 2| + |4*x - 3| + |5*x - 4|

theorem f_minimum_value :
  (∀ x : ℝ, f x ≥ 1) ∧ (∃ x : ℝ, f x = 1) := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l3991_399185


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l3991_399178

/-- A decagon is a polygon with 10 vertices -/
def Decagon : ℕ := 10

/-- The number of vertices adjacent to any vertex in a decagon -/
def AdjacentVertices : ℕ := 2

/-- The probability of selecting two adjacent vertices in a decagon -/
def ProbAdjacentVertices : ℚ := 2 / 9

theorem adjacent_vertices_probability (d : ℕ) (av : ℕ) (p : ℚ) 
  (h1 : d = Decagon) 
  (h2 : av = AdjacentVertices) 
  (h3 : p = ProbAdjacentVertices) : 
  p = av / (d - 1) := by
  sorry

#check adjacent_vertices_probability

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l3991_399178


namespace NUMINAMATH_CALUDE_unread_books_l3991_399157

theorem unread_books (total_books read_books : ℕ) : 
  total_books = 20 → read_books = 15 → total_books - read_books = 5 := by
  sorry

end NUMINAMATH_CALUDE_unread_books_l3991_399157


namespace NUMINAMATH_CALUDE_alcohol_concentration_l3991_399164

/-- Prove that the concentration of alcohol in the final mixture is 30% --/
theorem alcohol_concentration (vessel1_capacity : ℝ) (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percent : ℝ)
  (total_liquid : ℝ) (final_vessel_capacity : ℝ) :
  vessel1_capacity = 2 →
  vessel1_alcohol_percent = 30 →
  vessel2_capacity = 6 →
  vessel2_alcohol_percent = 40 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  let total_alcohol := (vessel1_capacity * vessel1_alcohol_percent / 100) +
                       (vessel2_capacity * vessel2_alcohol_percent / 100)
  (total_alcohol / final_vessel_capacity) * 100 = 30 := by
  sorry

#check alcohol_concentration

end NUMINAMATH_CALUDE_alcohol_concentration_l3991_399164


namespace NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l3991_399184

/-- A parabola is defined by its coefficients a, b, and c. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Function f is the original parabola translated 3 units to the right. -/
def f (p : Parabola) (x : ℝ) : ℝ :=
  p.a * (x - 3)^2 + p.b * (x - 3) + p.c

/-- Function g is the reflected parabola translated 3 units to the left. -/
def g (p : Parabola) (x : ℝ) : ℝ :=
  -p.a * (x + 3)^2 - p.b * (x + 3) - p.c

/-- The sum of f and g is a non-horizontal line. -/
theorem sum_is_non_horizontal_line (p : Parabola) :
  ∃ m k : ℝ, m ≠ 0 ∧ ∀ x, f p x + g p x = m * x + k := by
  sorry

end NUMINAMATH_CALUDE_sum_is_non_horizontal_line_l3991_399184


namespace NUMINAMATH_CALUDE_pet_store_black_cats_l3991_399136

/-- Given a pet store with white, black, and gray cats, prove the number of black cats. -/
theorem pet_store_black_cats 
  (total_cats : ℕ) 
  (white_cats : ℕ) 
  (gray_cats : ℕ) 
  (h_total : total_cats = 15) 
  (h_white : white_cats = 2) 
  (h_gray : gray_cats = 3) :
  total_cats - white_cats - gray_cats = 10 :=
by
  sorry

#check pet_store_black_cats

end NUMINAMATH_CALUDE_pet_store_black_cats_l3991_399136


namespace NUMINAMATH_CALUDE_inequality_proof_l3991_399138

theorem inequality_proof (x y : ℝ) (hx : x > 1) (hy : y > 0) :
  (4 * (x^2 * y^2 + x * y^3 + 4 * y^2 + 4 * x * y)) / (x + y) > 3 * x^2 * y + y :=
by sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l3991_399138


namespace NUMINAMATH_CALUDE_cos_equality_angle_l3991_399137

theorem cos_equality_angle (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_angle_l3991_399137


namespace NUMINAMATH_CALUDE_money_left_after_taxes_l3991_399146

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18

theorem money_left_after_taxes : 
  annual_income * (1 - tax_rate) = 49200 := by
  sorry

end NUMINAMATH_CALUDE_money_left_after_taxes_l3991_399146


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3991_399172

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ n < 500 → n ≤ 495 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l3991_399172


namespace NUMINAMATH_CALUDE_bottle_cap_cost_l3991_399128

theorem bottle_cap_cost (cost_per_cap : ℝ) (num_caps : ℕ) : 
  cost_per_cap = 5 → num_caps = 5 → cost_per_cap * (num_caps : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_cost_l3991_399128


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_0813_l3991_399132

theorem scientific_notation_of_0_0813 :
  ∃ (a : ℝ) (n : ℤ), 0.0813 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ n = -2 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_0813_l3991_399132


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3991_399176

theorem quadratic_solution_sum (m n : ℝ) (h1 : m ≠ 0) 
  (h2 : m * 1^2 + n * 1 - 2022 = 0) : m + n + 1 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3991_399176


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l3991_399165

def f (x : ℝ) := x^2 - 4*x + 3

theorem f_decreasing_on_interval :
  ∀ x y : ℝ, x < y → y ≤ 2 → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l3991_399165


namespace NUMINAMATH_CALUDE_train_speed_theorem_l3991_399101

-- Define the given constants
def train_length : ℝ := 110
def bridge_length : ℝ := 170
def crossing_time : ℝ := 16.7986561075114

-- Define the theorem
theorem train_speed_theorem :
  let total_distance := train_length + bridge_length
  let speed_ms := total_distance / crossing_time
  let speed_kmh := speed_ms * 3.6
  speed_kmh = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_theorem_l3991_399101


namespace NUMINAMATH_CALUDE_equation_solution_l3991_399127

theorem equation_solution : ∃! x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3991_399127


namespace NUMINAMATH_CALUDE_apple_cost_price_l3991_399152

theorem apple_cost_price (selling_price : ℝ) (loss_fraction : ℝ) (cost_price : ℝ) : 
  selling_price = 19 →
  loss_fraction = 1/6 →
  selling_price = cost_price - loss_fraction * cost_price →
  cost_price = 22.8 := by
sorry

end NUMINAMATH_CALUDE_apple_cost_price_l3991_399152


namespace NUMINAMATH_CALUDE_loan_duration_b_l3991_399106

/-- Proves that the loan duration for B is 2 years given the problem conditions -/
theorem loan_duration_b (principal_b : ℕ) (principal_c : ℕ) (rate : ℚ) 
  (duration_c : ℕ) (total_interest : ℕ) :
  principal_b = 5000 →
  principal_c = 3000 →
  rate = 8/100 →
  duration_c = 4 →
  total_interest = 1760 →
  ∃ (n : ℕ), n = 2 ∧ 
    (principal_b * rate * n + principal_c * rate * duration_c = total_interest) :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_b_l3991_399106


namespace NUMINAMATH_CALUDE_police_officers_on_duty_l3991_399191

theorem police_officers_on_duty 
  (total_female_officers : ℕ) 
  (female_duty_percentage : ℚ)
  (female_duty_ratio : ℚ) :
  total_female_officers = 400 →
  female_duty_percentage = 19 / 100 →
  female_duty_ratio = 1 / 2 →
  ∃ (officers_on_duty : ℕ), 
    officers_on_duty = 152 ∧ 
    (female_duty_percentage * total_female_officers : ℚ) = (female_duty_ratio * officers_on_duty : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_police_officers_on_duty_l3991_399191


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3991_399162

theorem least_three_digit_multiple_of_eight : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  n % 8 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ m % 8 = 0 → n ≤ m) ∧ 
  n = 104 :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_eight_l3991_399162


namespace NUMINAMATH_CALUDE_trefoils_per_case_l3991_399133

theorem trefoils_per_case (total_boxes : ℕ) (total_cases : ℕ) (boxes_per_case : ℕ) : 
  total_boxes = 54 → total_cases = 9 → boxes_per_case = total_boxes / total_cases → boxes_per_case = 6 := by
  sorry

end NUMINAMATH_CALUDE_trefoils_per_case_l3991_399133


namespace NUMINAMATH_CALUDE_candle_ratio_l3991_399149

/-- Proves the ratio of candles Alyssa used to total candles -/
theorem candle_ratio :
  ∀ (total candles_used_by_alyssa : ℕ) (chelsea_usage_percent : ℚ),
  total = 40 →
  chelsea_usage_percent = 70 / 100 →
  candles_used_by_alyssa + 
    (chelsea_usage_percent * (total - candles_used_by_alyssa)).floor + 6 = total →
  candles_used_by_alyssa * 2 = total := by
  sorry

#check candle_ratio

end NUMINAMATH_CALUDE_candle_ratio_l3991_399149


namespace NUMINAMATH_CALUDE_restaurant_problem_l3991_399155

theorem restaurant_problem (people : ℕ) 
  (h1 : 7 * 10 + (88 / people + 7) = 88) : people = 8 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_problem_l3991_399155


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3991_399116

theorem sufficient_but_not_necessary (p q : Prop) : 
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l3991_399116


namespace NUMINAMATH_CALUDE_total_pupils_across_schools_l3991_399197

/-- The total number of pupils across three schools -/
def total_pupils (school_a_girls school_a_boys school_b_girls school_b_boys school_c_girls school_c_boys : ℕ) : ℕ :=
  school_a_girls + school_a_boys + school_b_girls + school_b_boys + school_c_girls + school_c_boys

/-- Theorem stating that the total number of pupils across the three schools is 3120 -/
theorem total_pupils_across_schools :
  total_pupils 542 387 713 489 628 361 = 3120 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_across_schools_l3991_399197


namespace NUMINAMATH_CALUDE_imaginary_unit_cube_l3991_399115

theorem imaginary_unit_cube (i : ℂ) (hi : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_cube_l3991_399115


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l3991_399142

def stamp_price : ℕ := 25
def available_amount : ℕ := 4000

theorem max_stamps_purchasable :
  ∀ n : ℕ, n * stamp_price ≤ available_amount ↔ n ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l3991_399142


namespace NUMINAMATH_CALUDE_square_sum_equals_two_l3991_399190

theorem square_sum_equals_two (a b : ℝ) 
  (h1 : (a + b)^2 = 4) 
  (h2 : a * b = 1) : 
  a^2 + b^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_two_l3991_399190


namespace NUMINAMATH_CALUDE_line_equation_equivalence_l3991_399103

theorem line_equation_equivalence (x y : ℝ) (h : 2*x - 5*y - 3 = 0) : 
  -4*x + 10*y + 3 = 0 := by sorry

end NUMINAMATH_CALUDE_line_equation_equivalence_l3991_399103


namespace NUMINAMATH_CALUDE_polynomial_identity_l3991_399187

-- Define a polynomial function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem polynomial_identity 
  (h : ∀ x : ℝ, f (x^2 + 2) = x^4 + 5*x^2 + 1) :
  ∀ x : ℝ, f (x^2 - 2) = x^4 - 3*x^2 - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l3991_399187


namespace NUMINAMATH_CALUDE_erased_number_theorem_l3991_399169

theorem erased_number_theorem (n : ℕ) (h1 : n = 20) :
  ∀ x ∈ Finset.range n,
    (∃ y ∈ Finset.range n \ {x}, (n * (n + 1) / 2 - x : ℚ) / (n - 1) = y) ↔ 
    x = 1 ∨ x = n :=
by sorry

end NUMINAMATH_CALUDE_erased_number_theorem_l3991_399169


namespace NUMINAMATH_CALUDE_not_always_parallel_lines_l3991_399160

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane_line : Plane → Line → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem not_always_parallel_lines 
  (l m : Line) (α : Plane) 
  (h1 : parallel_plane_line α l) 
  (h2 : subset m α) : 
  ¬ (∀ l m α, parallel_plane_line α l → subset m α → parallel l m) :=
sorry

end NUMINAMATH_CALUDE_not_always_parallel_lines_l3991_399160


namespace NUMINAMATH_CALUDE_work_completion_time_l3991_399168

/-- The number of days it takes y to complete the work -/
def y_days : ℝ := 40

/-- The number of days it takes x and y together to complete the work -/
def combined_days : ℝ := 13.333333333333332

/-- The number of days it takes x to complete the work -/
def x_days : ℝ := 20

theorem work_completion_time :
  1 / x_days + 1 / y_days = 1 / combined_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3991_399168


namespace NUMINAMATH_CALUDE_equation_solution_l3991_399163

theorem equation_solution : 
  ∀ x : ℝ, (x - 1) * (x + 1) = x - 1 ↔ x = 0 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3991_399163


namespace NUMINAMATH_CALUDE_relay_race_time_reduction_l3991_399118

theorem relay_race_time_reduction (T : ℝ) (T1 T2 T3 T4 T5 : ℝ) :
  T > 0 ∧ T1 > 0 ∧ T2 > 0 ∧ T3 > 0 ∧ T4 > 0 ∧ T5 > 0 ∧
  T = T1 + T2 + T3 + T4 + T5 ∧
  T1/2 + T2 + T3 + T4 + T5 = 0.95 * T ∧
  T1 + T2/2 + T3 + T4 + T5 = 0.9 * T ∧
  T1 + T2 + T3/2 + T4 + T5 = 0.88 * T ∧
  T1 + T2 + T3 + T4/2 + T5 = 0.85 * T →
  T1 + T2 + T3 + T4 + T5/2 = 0.92 * T := by
sorry

end NUMINAMATH_CALUDE_relay_race_time_reduction_l3991_399118


namespace NUMINAMATH_CALUDE_straight_line_properties_l3991_399193

-- Define a straight line in a Cartesian coordinate system
structure StraightLine where
  -- We don't define the line using a specific equation to keep it general
  slope_angle : Real
  has_defined_slope : Bool

-- Theorem statement
theorem straight_line_properties (l : StraightLine) : 
  (0 ≤ l.slope_angle ∧ l.slope_angle < π) ∧ 
  (l.has_defined_slope = false → l.slope_angle = π/2) :=
by sorry

end NUMINAMATH_CALUDE_straight_line_properties_l3991_399193


namespace NUMINAMATH_CALUDE_unique_congruence_solution_l3991_399175

theorem unique_congruence_solution : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ 10389 [ZMOD 12] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_solution_l3991_399175


namespace NUMINAMATH_CALUDE_complex_sum_reciprocals_l3991_399192

theorem complex_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 5) :
  Complex.abs (1 / z + 1 / w) = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocals_l3991_399192


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3991_399173

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x - 1| < 2 → -3 < x ∧ x < 3) ∧
  (∃ x : ℝ, -3 < x ∧ x < 3 ∧ ¬(|x - 1| < 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3991_399173


namespace NUMINAMATH_CALUDE_circle_area_theorem_l3991_399104

def A : ℝ × ℝ := (8, 15)
def B : ℝ × ℝ := (14, 9)

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def tangent_line (c : Circle) (p : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

def intersect_x_axis (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

theorem circle_area_theorem (ω : Circle) :
  on_circle A ω →
  on_circle B ω →
  (∃ (p : ℝ × ℝ), p.2 = 0 ∧ p ∈ tangent_line ω A ∧ p ∈ tangent_line ω B) →
  ω.radius^2 * Real.pi = 306 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l3991_399104


namespace NUMINAMATH_CALUDE_current_average_age_l3991_399181

-- Define the number of people in the initial group
def initial_group : ℕ := 6

-- Define the average age of the initial group after two years
def future_average_age : ℕ := 43

-- Define the age of the new person joining the group
def new_person_age : ℕ := 69

-- Define the total number of people after the new person joins
def total_people : ℕ := initial_group + 1

-- Theorem to prove
theorem current_average_age :
  (initial_group * future_average_age - initial_group * 2 + new_person_age) / total_people = 45 :=
by sorry

end NUMINAMATH_CALUDE_current_average_age_l3991_399181


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l3991_399156

def ticket_price (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (service_fee : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 + service_fee

theorem zoo_ticket_price :
  ticket_price 15 0.4 0.1 2 = 10.1 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l3991_399156


namespace NUMINAMATH_CALUDE_probability_two_teachers_in_A_proof_l3991_399135

/-- The probability of exactly two out of three teachers being assigned to place A -/
def probability_two_teachers_in_A : ℚ := 3/8

/-- The number of teachers -/
def num_teachers : ℕ := 3

/-- The number of places -/
def num_places : ℕ := 2

theorem probability_two_teachers_in_A_proof :
  probability_two_teachers_in_A = 
    (Nat.choose num_teachers 2 : ℚ) / (num_places ^ num_teachers : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_two_teachers_in_A_proof_l3991_399135


namespace NUMINAMATH_CALUDE_equation_solution_l3991_399121

theorem equation_solution (a b : ℕ+) :
  2 * a ^ 2 = 3 * b ^ 3 ↔ ∃ k : ℕ+, a = 18 * k ^ 3 ∧ b = 6 * k ^ 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3991_399121


namespace NUMINAMATH_CALUDE_treehouse_planks_l3991_399159

theorem treehouse_planks :
  ∀ (T : ℕ),
  (T / 4 : ℚ) + (T / 2 : ℚ) + 20 + 30 = T →
  T = 200 := by
sorry

end NUMINAMATH_CALUDE_treehouse_planks_l3991_399159


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3991_399122

/-- If x^2 + mx + 16 is a perfect square trinomial, then m = ±8 -/
theorem perfect_square_trinomial (m : ℝ) : 
  (∀ x, ∃ k, x^2 + m*x + 16 = k^2) → m = 8 ∨ m = -8 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3991_399122


namespace NUMINAMATH_CALUDE_min_value_expression_l3991_399153

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 3) (h_rel : x = 2 * y) :
  ∃ (min : ℝ), min = 4/3 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 3 → x = 2 * y →
    (x + y) / (x * y * z) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3991_399153


namespace NUMINAMATH_CALUDE_two_face_painted_count_l3991_399143

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_length : Nat
  painted_faces : Nat

/-- Counts the number of smaller cubes painted on exactly two faces -/
def count_two_face_painted (c : CutCube) : Nat :=
  if c.side_length = 3 ∧ c.painted_faces = 6 then
    24
  else
    0

theorem two_face_painted_count (c : CutCube) :
  c.side_length = 3 ∧ c.painted_faces = 6 → count_two_face_painted c = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_face_painted_count_l3991_399143


namespace NUMINAMATH_CALUDE_inequality_not_always_correct_l3991_399167

theorem inequality_not_always_correct
  (x y z w : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hxy : x > y)
  (hz : z ≠ 0)
  (hw : w ≠ 0) :
  ∃ (x' y' z' w' : ℝ),
    x' > 0 ∧ y' > 0 ∧ x' > y' ∧ z' ≠ 0 ∧ w' ≠ 0 ∧
    x' * z' ≤ y' * w' * z' :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_correct_l3991_399167


namespace NUMINAMATH_CALUDE_correct_delivery_probability_l3991_399129

def number_of_packages : ℕ := 5
def number_of_houses : ℕ := 5

theorem correct_delivery_probability :
  let total_arrangements := number_of_packages.factorial
  let correct_three_arrangements := (number_of_packages.choose 3) * 1 * 1
  (correct_three_arrangements : ℚ) / total_arrangements = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_delivery_probability_l3991_399129


namespace NUMINAMATH_CALUDE_complex_cube_sum_ratio_l3991_399134

theorem complex_cube_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_squared_diff : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 13 :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_sum_ratio_l3991_399134


namespace NUMINAMATH_CALUDE_lines_parallel_l3991_399151

/-- The value of k that makes the given lines parallel -/
def k : ℚ := 16/5

/-- The first line's direction vector -/
def v1 : Fin 2 → ℚ := ![5, -8]

/-- The second line's direction vector -/
def v2 : Fin 2 → ℚ := ![-2, k]

/-- Theorem stating that k makes the lines parallel -/
theorem lines_parallel : ∃ (c : ℚ), v1 = c • v2 := by sorry

end NUMINAMATH_CALUDE_lines_parallel_l3991_399151


namespace NUMINAMATH_CALUDE_probability_theorem_l3991_399182

/-- Represents the total number of products -/
def total_products : ℕ := 7

/-- Represents the number of genuine products -/
def genuine_products : ℕ := 4

/-- Represents the number of defective products -/
def defective_products : ℕ := 3

/-- The probability of selecting a genuine product on the second draw,
    given that a defective product was selected on the first draw -/
def probability_genuine_second_given_defective_first : ℚ := 2/3

/-- Theorem stating the probability of selecting a genuine product on the second draw,
    given that a defective product was selected on the first draw -/
theorem probability_theorem :
  probability_genuine_second_given_defective_first = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l3991_399182


namespace NUMINAMATH_CALUDE_min_sum_squares_l3991_399120

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + 3 * x₂ + 4 * x₃ = 72) (h_rel : x₁ = 3 * x₂) :
  x₁^2 + x₂^2 + x₃^2 ≥ 347.04 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l3991_399120


namespace NUMINAMATH_CALUDE_rebecca_gave_two_caps_l3991_399150

def initial_caps : ℕ := 7
def final_caps : ℕ := 9

theorem rebecca_gave_two_caps : final_caps - initial_caps = 2 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_gave_two_caps_l3991_399150


namespace NUMINAMATH_CALUDE_right_square_pyramid_base_neq_lateral_l3991_399183

/-- A right square pyramid -/
structure RightSquarePyramid where
  baseEdge : ℝ
  lateralEdge : ℝ
  height : ℝ

/-- Theorem: In a right square pyramid, the base edge length cannot be equal to the lateral edge length -/
theorem right_square_pyramid_base_neq_lateral (p : RightSquarePyramid) : 
  p.baseEdge ≠ p.lateralEdge :=
sorry

end NUMINAMATH_CALUDE_right_square_pyramid_base_neq_lateral_l3991_399183


namespace NUMINAMATH_CALUDE_total_triangle_area_is_36_l3991_399147

/-- Represents a square in the grid -/
structure Square where
  x : Nat
  y : Nat
  deriving Repr

/-- Represents a triangle in a square -/
structure Triangle where
  square : Square
  deriving Repr

/-- The size of the grid -/
def gridSize : Nat := 6

/-- Calculate the area of a single triangle -/
def triangleArea : ℝ := 0.5

/-- Calculate the number of triangles in a square -/
def trianglesPerSquare : Nat := 2

/-- Calculate the total number of squares in the grid -/
def totalSquares : Nat := gridSize * gridSize

/-- Calculate the total area of all triangles in the grid -/
def totalTriangleArea : ℝ :=
  (totalSquares : ℝ) * (trianglesPerSquare : ℝ) * triangleArea

/-- Theorem stating that the total area of triangles in the grid is 36 -/
theorem total_triangle_area_is_36 : totalTriangleArea = 36 := by
  sorry

#eval totalTriangleArea

end NUMINAMATH_CALUDE_total_triangle_area_is_36_l3991_399147


namespace NUMINAMATH_CALUDE_trip_duration_l3991_399124

theorem trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 70 ∧
  initial_time = 4 ∧
  additional_speed = 60 ∧
  average_speed = 65 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed ∧
    total_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_trip_duration_l3991_399124


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3991_399140

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l3991_399140


namespace NUMINAMATH_CALUDE_detergent_needed_l3991_399144

/-- The amount of detergent used per pound of clothes -/
def detergent_per_pound : ℝ := 2

/-- The amount of clothes to be washed, in pounds -/
def clothes_amount : ℝ := 9

/-- Theorem stating the amount of detergent needed for a given amount of clothes -/
theorem detergent_needed (detergent_per_pound : ℝ) (clothes_amount : ℝ) :
  detergent_per_pound * clothes_amount = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_detergent_needed_l3991_399144


namespace NUMINAMATH_CALUDE_chess_match_probability_l3991_399188

theorem chess_match_probability (p_win p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_chess_match_probability_l3991_399188
