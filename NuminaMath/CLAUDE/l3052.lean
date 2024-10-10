import Mathlib

namespace henry_collection_cost_l3052_305262

/-- The amount of money Henry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Henry needs $144 to finish his collection -/
theorem henry_collection_cost :
  money_needed 3 15 12 = 144 := by
  sorry

end henry_collection_cost_l3052_305262


namespace digit_sum_problem_l3052_305265

theorem digit_sum_problem (A B C D E F : ℕ) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
  2*A + 3*B + 2*C + 2*D + 2*E + 2*F = 47 →
  B = 5 := by
sorry

end digit_sum_problem_l3052_305265


namespace complex_modulus_sqrt5_l3052_305260

theorem complex_modulus_sqrt5 (a : ℝ) : 
  Complex.abs (1 + a * Complex.I) = Real.sqrt 5 ↔ a = 2 ∨ a = -2 := by
sorry

end complex_modulus_sqrt5_l3052_305260


namespace q_value_l3052_305272

/-- The coordinates of point A -/
def A : ℝ × ℝ := (0, 12)

/-- The coordinates of point Q -/
def Q : ℝ × ℝ := (3, 12)

/-- The coordinates of point B -/
def B : ℝ × ℝ := (15, 0)

/-- The coordinates of point O -/
def O : ℝ × ℝ := (0, 0)

/-- The coordinates of point C -/
def C (q : ℝ) : ℝ × ℝ := (0, q)

/-- The area of triangle ABC -/
def area_ABC : ℝ := 36

/-- Theorem: If the area of triangle ABC is 36 and the points have the given coordinates, then q = 9 -/
theorem q_value : ∃ q : ℝ, C q = (0, q) ∧ area_ABC = 36 → q = 9 := by
  sorry

end q_value_l3052_305272


namespace scientific_notation_of_234_1_million_l3052_305243

theorem scientific_notation_of_234_1_million :
  let million : ℝ := 10^6
  234.1 * million = 2.341 * 10^6 := by sorry

end scientific_notation_of_234_1_million_l3052_305243


namespace vote_alteration_l3052_305270

theorem vote_alteration (got twi tad : ℕ) (x : ℚ) : 
  got = 10 →
  twi = 12 →
  tad = 20 →
  2 * got = got + twi / 2 + tad * (1 - x / 100) →
  x = 80 := by
sorry

end vote_alteration_l3052_305270


namespace division_subtraction_l3052_305216

theorem division_subtraction : ((-150) / (-50)) - 15 = -12 := by
  sorry

end division_subtraction_l3052_305216


namespace subcommittee_count_l3052_305287

theorem subcommittee_count (n m k t : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 5) (h4 : t = 5) :
  (Nat.choose n k) - (Nat.choose (n - t) k) = 771 :=
by sorry

end subcommittee_count_l3052_305287


namespace arithmetic_sequence_common_difference_l3052_305219

theorem arithmetic_sequence_common_difference
  (a : ℝ)  -- first term
  (an : ℝ) -- last term
  (s : ℝ)  -- sum of all terms
  (h1 : a = 7)
  (h2 : an = 88)
  (h3 : s = 570)
  : ∃ n : ℕ, n > 1 ∧ (an - a) / (n - 1) = 81 / 11 :=
by
  sorry

end arithmetic_sequence_common_difference_l3052_305219


namespace gummy_worms_problem_l3052_305284

theorem gummy_worms_problem (x : ℝ) : (x / 2^4 = 4) → x = 64 := by
  sorry

end gummy_worms_problem_l3052_305284


namespace smallest_x_absolute_value_equation_l3052_305212

theorem smallest_x_absolute_value_equation : 
  ∀ x : ℝ, |4*x + 12| = 40 → x ≥ -13 :=
by
  sorry

end smallest_x_absolute_value_equation_l3052_305212


namespace jersey_tshirt_cost_difference_l3052_305266

/-- Calculates the final cost difference between a jersey and a t-shirt --/
theorem jersey_tshirt_cost_difference :
  let jersey_price : ℚ := 115
  let tshirt_price : ℚ := 25
  let jersey_discount : ℚ := 10 / 100
  let tshirt_discount : ℚ := 15 / 100
  let sales_tax : ℚ := 8 / 100
  let jersey_shipping : ℚ := 5
  let tshirt_shipping : ℚ := 3

  let jersey_discounted := jersey_price * (1 - jersey_discount)
  let tshirt_discounted := tshirt_price * (1 - tshirt_discount)

  let jersey_with_tax := jersey_discounted * (1 + sales_tax)
  let tshirt_with_tax := tshirt_discounted * (1 + sales_tax)

  let jersey_final := jersey_with_tax + jersey_shipping
  let tshirt_final := tshirt_with_tax + tshirt_shipping

  jersey_final - tshirt_final = 90.83 := by sorry

end jersey_tshirt_cost_difference_l3052_305266


namespace magnitude_of_complex_fraction_l3052_305291

theorem magnitude_of_complex_fraction (z : ℂ) : z = (2 - I) / (1 + 2*I) → Complex.abs z = 1 := by
  sorry

end magnitude_of_complex_fraction_l3052_305291


namespace bowl_glass_pairings_l3052_305241

/-- The number of possible pairings when choosing one bowl from a set of distinct bowls
    and one glass from a set of distinct glasses -/
def num_pairings (num_bowls : ℕ) (num_glasses : ℕ) : ℕ :=
  num_bowls * num_glasses

/-- Theorem stating that with 5 distinct bowls and 6 distinct glasses,
    the number of possible pairings is 30 -/
theorem bowl_glass_pairings :
  num_pairings 5 6 = 30 := by
  sorry

end bowl_glass_pairings_l3052_305241


namespace expression_simplification_l3052_305207

theorem expression_simplification (x : ℚ) (h : x = -3) :
  (1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x^2 - 1)) = 2/5 :=
by sorry

end expression_simplification_l3052_305207


namespace octal_2016_to_binary_l3052_305220

/-- Converts an octal number to decimal --/
def octal_to_decimal (octal : ℕ) : ℕ := sorry

/-- Converts a decimal number to binary --/
def decimal_to_binary (decimal : ℕ) : List ℕ := sorry

/-- Converts a list of binary digits to a natural number --/
def binary_list_to_nat (binary : List ℕ) : ℕ := sorry

theorem octal_2016_to_binary :
  let octal := 2016
  let decimal := octal_to_decimal octal
  let binary := decimal_to_binary decimal
  binary_list_to_nat binary = binary_list_to_nat [1,0,0,0,0,0,0,1,1,1,0] := by sorry

end octal_2016_to_binary_l3052_305220


namespace third_stick_length_l3052_305294

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem third_stick_length (x : ℕ) 
  (h1 : is_even x)
  (h2 : x + 10 > 2)
  (h3 : x + 2 > 10)
  (h4 : 10 + 2 > x) : 
  x = 10 := by
  sorry

end third_stick_length_l3052_305294


namespace benzene_homolog_bonds_l3052_305275

/-- Represents the number of bonds in a molecule -/
def num_bonds (n : ℕ) : ℕ := 3 * n - 3

/-- Represents the number of valence electrons in a molecule -/
def num_valence_electrons (n : ℕ) : ℕ := 4 * n + (2 * n - 6)

/-- Theorem stating the relationship between carbon atoms and bonds in benzene homologs -/
theorem benzene_homolog_bonds (n : ℕ) : 
  num_bonds n = (num_valence_electrons n) / 2 := by
  sorry

end benzene_homolog_bonds_l3052_305275


namespace fourth_player_score_zero_l3052_305234

/-- Represents the score of a player in the chess tournament -/
structure PlayerScore :=
  (score : ℕ)

/-- Represents the scores of all players in the tournament -/
structure TournamentScores :=
  (players : Fin 4 → PlayerScore)

/-- The total points awarded in a tournament with 4 players -/
def totalPoints : ℕ := 12

/-- Theorem stating that if three players have scores 6, 4, and 2, the fourth must have 0 -/
theorem fourth_player_score_zero (t : TournamentScores) :
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (t.players i).score = 6 ∧ 
    (t.players j).score = 4 ∧ 
    (t.players k).score = 2) →
  (∃ (l : Fin 4), (∀ m : Fin 4, m ≠ l → 
    (t.players m).score = 6 ∨ 
    (t.players m).score = 4 ∨ 
    (t.players m).score = 2) ∧ 
  (t.players l).score = 0) :=
by sorry

end fourth_player_score_zero_l3052_305234


namespace hyperbola_eccentricity_l3052_305200

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    ∃ B C : ℝ × ℝ, 
      B.1 = c ∧ C.1 = c ∧ 
      B.2^2 = (b^2 / a^2) * (c^2 - a^2) ∧ 
      C.2^2 = (b^2 / a^2) * (c^2 - a^2) ∧
      (B.2 - C.2)^2 = 2 * (c + a)^2) →
  c / a = 2 := by
sorry

end hyperbola_eccentricity_l3052_305200


namespace cookie_distribution_l3052_305285

theorem cookie_distribution (total : ℝ) (blue green red : ℝ) : 
  blue = (1/4) * total ∧ 
  green = (5/9) * (total - blue) → 
  (blue + green) / total = 2/3 := by
sorry

end cookie_distribution_l3052_305285


namespace committee_probability_l3052_305205

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

theorem committee_probability :
  let total_combinations := Nat.choose total_members committee_size
  let all_boys_or_all_girls := 2 * Nat.choose boys committee_size
  (total_combinations - all_boys_or_all_girls : ℚ) / total_combinations = 5115 / 5313 := by
  sorry

end committee_probability_l3052_305205


namespace pure_imaginary_modulus_l3052_305230

theorem pure_imaginary_modulus (b : ℝ) : 
  let z : ℂ := (3 + b * Complex.I) * (1 + Complex.I) - 2
  (z.re = 0 ∧ z.im ≠ 0) → Complex.abs z = 4 := by
  sorry

end pure_imaginary_modulus_l3052_305230


namespace namjoon_has_greater_sum_l3052_305264

def jimin_numbers : List Nat := [1, 7]
def namjoon_numbers : List Nat := [6, 3]

theorem namjoon_has_greater_sum :
  List.sum namjoon_numbers > List.sum jimin_numbers := by
  sorry

end namjoon_has_greater_sum_l3052_305264


namespace smallest_number_proof_smallest_number_is_4725_l3052_305235

theorem smallest_number_proof (x : ℕ) : 
  (x + 3 = 4728) ∧ 
  (∃ k₁ : ℕ, (x + 3) = 27 * k₁) ∧ 
  (∃ k₂ : ℕ, (x + 3) = 35 * k₂) ∧ 
  (∃ k₃ : ℕ, (x + 3) = 25 * k₃) →
  x ≥ 4725 :=
by sorry

theorem smallest_number_is_4725 : 
  (4725 + 3 = 4728) ∧ 
  (∃ k₁ : ℕ, (4725 + 3) = 27 * k₁) ∧ 
  (∃ k₂ : ℕ, (4725 + 3) = 35 * k₂) ∧ 
  (∃ k₃ : ℕ, (4725 + 3) = 25 * k₃) :=
by sorry

end smallest_number_proof_smallest_number_is_4725_l3052_305235


namespace simplify_expression_l3052_305202

theorem simplify_expression (n : ℕ) : 
  (3 * 2^(n+5) - 5 * 2^n) / (4 * 2^(n+2)) = 91 / 16 := by
  sorry

end simplify_expression_l3052_305202


namespace log_sum_problem_l3052_305237

theorem log_sum_problem (x y : ℝ) (h1 : Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1/2) (h2 : x = 12) :
  y = 1/6 := by
  sorry

end log_sum_problem_l3052_305237


namespace rectangle_area_change_l3052_305259

theorem rectangle_area_change (L B : ℝ) (h₁ : L > 0) (h₂ : B > 0) :
  let A := L * B
  let L' := 1.20 * L
  let B' := 0.95 * B
  let A' := L' * B'
  A' = 1.14 * A := by sorry

end rectangle_area_change_l3052_305259


namespace farm_animals_l3052_305281

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) (ducks : ℕ) (horses : ℕ) : 
  total_animals = 11 →
  total_legs = 30 →
  ducks + horses = total_animals →
  2 * ducks + 4 * horses = total_legs →
  ducks = 7 := by
sorry

end farm_animals_l3052_305281


namespace problem_solution_l3052_305208

def f (m : ℝ) (x : ℝ) : ℝ := |x - m| - |x + 3*m|

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ≤ -3/2) ∧
  ((∀ x t : ℝ, f m x < |2 + t| + |t - 1|) → 0 < m ∧ m < 3/4) :=
sorry

end problem_solution_l3052_305208


namespace no_root_greater_than_sqrt29_div_2_l3052_305210

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 + 3 = 53
def equation2 (x : ℝ) : Prop := (3*x - 1)^2 = (x - 2)^2
def equation3 (x : ℝ) : Prop := Real.sqrt (x^2 - 9) ≥ Real.sqrt (x - 2)

-- Define a function to check if a number is a root of any equation
def is_root (x : ℝ) : Prop :=
  equation1 x ∨ equation2 x ∨ equation3 x

-- Theorem statement
theorem no_root_greater_than_sqrt29_div_2 :
  ∀ x : ℝ, is_root x → x ≤ Real.sqrt 29 / 2 :=
by sorry

end no_root_greater_than_sqrt29_div_2_l3052_305210


namespace interest_difference_approx_l3052_305227

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

/-- The positive difference between compound and simple interest balances -/
def interest_difference (principal : ℝ) (compound_rate : ℝ) (simple_rate : ℝ) (time : ℕ) : ℝ :=
  |simple_interest principal simple_rate time - compound_interest principal compound_rate time|

theorem interest_difference_approx :
  ∃ ε > 0, |interest_difference 10000 0.04 0.06 12 - 1189| < ε :=
sorry

end interest_difference_approx_l3052_305227


namespace solution_set_f_max_value_g_l3052_305247

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Define the function g(x)
def g (x : ℝ) : ℝ := f x - x^2 + x

-- Theorem 1: Solution set of f(x) ≥ 1
theorem solution_set_f (x : ℝ) : f x ≥ 1 ↔ x ≥ 1 := by sorry

-- Theorem 2: Maximum value of g(x)
theorem max_value_g : ∃ (x : ℝ), ∀ (y : ℝ), g y ≤ g x ∧ g x = 5/4 := by sorry

end solution_set_f_max_value_g_l3052_305247


namespace max_edges_triangle_free_30_max_edges_k4_free_30_l3052_305244

/-- The maximum number of edges in a triangle-free graph with 30 vertices -/
def max_edges_triangle_free (n : ℕ) : ℕ :=
  if n = 30 then 225 else 0

/-- The maximum number of edges in a K₄-free graph with 30 vertices -/
def max_edges_k4_free (n : ℕ) : ℕ :=
  if n = 30 then 300 else 0

/-- Theorem stating the maximum number of edges in a triangle-free graph with 30 vertices -/
theorem max_edges_triangle_free_30 :
  max_edges_triangle_free 30 = 225 := by sorry

/-- Theorem stating the maximum number of edges in a K₄-free graph with 30 vertices -/
theorem max_edges_k4_free_30 :
  max_edges_k4_free 30 = 300 := by sorry

end max_edges_triangle_free_30_max_edges_k4_free_30_l3052_305244


namespace divide_multiply_add_subtract_l3052_305245

theorem divide_multiply_add_subtract (x n : ℝ) : x = 40 → ((x / n) * 5 + 10 - 12 = 48 ↔ n = 4) := by
  sorry

end divide_multiply_add_subtract_l3052_305245


namespace second_agency_cost_per_mile_l3052_305269

/-- The cost per mile for the second agency that makes both agencies' costs equal at 25.0 miles -/
theorem second_agency_cost_per_mile :
  let first_agency_daily_rate : ℚ := 20.25
  let first_agency_per_mile : ℚ := 0.14
  let second_agency_daily_rate : ℚ := 18.25
  let miles_driven : ℚ := 25.0
  let second_agency_per_mile : ℚ := (first_agency_daily_rate - second_agency_daily_rate + first_agency_per_mile * miles_driven) / miles_driven
  second_agency_per_mile = 0.22 := by sorry

end second_agency_cost_per_mile_l3052_305269


namespace right_triangle_leg_ratio_l3052_305283

theorem right_triangle_leg_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_projection : (c - b^2 / c) / (b^2 / c) = 4) : b / a = 2 := by
  sorry

end right_triangle_leg_ratio_l3052_305283


namespace solution_set_when_a_is_one_range_of_a_l3052_305238

-- Define the function f
def f (x a : ℝ) := |2*x - a| + |x - 3*a|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f x 1 ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ |x - a/2| + a^2 + 1} = 
  {a : ℝ | (-2 ≤ a ∧ a ≤ -1/2) ∨ (1/2 ≤ a ∧ a ≤ 2)} := by sorry

end solution_set_when_a_is_one_range_of_a_l3052_305238


namespace complex_number_location_l3052_305239

theorem complex_number_location (z : ℂ) (h : (z - 1) * Complex.I = Complex.I + 1) : 
  0 < z.re ∧ z.im < 0 := by
sorry

end complex_number_location_l3052_305239


namespace unique_solution_system_l3052_305248

theorem unique_solution_system (x y : ℝ) : 
  (3 * x ≥ 2 * y + 16 ∧ 
   x^4 + 2 * x^2 * y^2 + y^4 + 25 - 26 * x^2 - 26 * y^2 = 72 * x * y) ↔ 
  (x = 6 ∧ y = 1) :=
by sorry

end unique_solution_system_l3052_305248


namespace simplify_and_evaluate_l3052_305267

theorem simplify_and_evaluate :
  let a : ℝ := (1/2 : ℝ) + Real.sqrt (1/2)
  (a + Real.sqrt 3) * (a - Real.sqrt 3) - a * (a - 6) = 3 * Real.sqrt 2 := by
  sorry

end simplify_and_evaluate_l3052_305267


namespace negation_of_proposition_l3052_305297

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, 2 * x^2 - 1 > 0) ↔ (∃ x : ℝ, 2 * x^2 - 1 ≤ 0) := by
  sorry

end negation_of_proposition_l3052_305297


namespace second_discount_percentage_second_discount_percentage_proof_l3052_305289

theorem second_discount_percentage (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 528)
  (h2 : first_discount = 0.2)
  (h3 : final_price = 380.16) : ℝ :=
let price_after_first_discount := original_price * (1 - first_discount)
let second_discount := (price_after_first_discount - final_price) / price_after_first_discount
0.1

theorem second_discount_percentage_proof 
  (original_price first_discount final_price : ℝ) 
  (h1 : original_price = 528)
  (h2 : first_discount = 0.2)
  (h3 : final_price = 380.16) : 
  second_discount_percentage original_price first_discount final_price h1 h2 h3 = 0.1 := by
sorry

end second_discount_percentage_second_discount_percentage_proof_l3052_305289


namespace inscribed_circle_diameter_l3052_305223

/-- Given a square with an inscribed circle, if the perimeter of the square (in inches) 
    equals the area of the circle (in square inches), then the diameter of the circle 
    is 16/π inches. -/
theorem inscribed_circle_diameter (s : ℝ) (r : ℝ) (h : s > 0) :
  (4 * s = π * r^2) → (2 * r = 16 / π) := by sorry

end inscribed_circle_diameter_l3052_305223


namespace nickels_left_l3052_305232

def initial_cents : ℕ := 475
def exchanged_cents : ℕ := 75
def cents_per_nickel : ℕ := 5
def cents_per_dime : ℕ := 10

def peter_proportion : ℚ := 2/5
def randi_proportion : ℚ := 3/5
def paula_proportion : ℚ := 1/10

theorem nickels_left : ℕ := by
  -- Prove that Ray is left with 82 nickels
  sorry

end nickels_left_l3052_305232


namespace inequalities_satisfied_l3052_305213

theorem inequalities_satisfied (a b c x y z : ℝ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x*y + y*z + z*x ≤ a*b + b*c + c*a + 3) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 + 3) ∧ 
  (x*y*z ≤ a*b*c + 1) := by
  sorry

end inequalities_satisfied_l3052_305213


namespace green_shirts_count_l3052_305217

/-- Proves that the number of green shirts is 17 given the total number of shirts and the number of blue shirts. -/
theorem green_shirts_count (total_shirts : ℕ) (blue_shirts : ℕ) (h1 : total_shirts = 23) (h2 : blue_shirts = 6) :
  total_shirts - blue_shirts = 17 := by
  sorry

#check green_shirts_count

end green_shirts_count_l3052_305217


namespace charles_earnings_l3052_305293

/-- Charles' earnings problem -/
theorem charles_earnings (housesitting_rate : ℝ) (housesitting_hours : ℝ) (dogs_walked : ℝ) (total_earnings : ℝ) :
  housesitting_rate = 15 →
  housesitting_hours = 10 →
  dogs_walked = 3 →
  total_earnings = 216 →
  (total_earnings - housesitting_rate * housesitting_hours) / dogs_walked = 22 := by
  sorry

end charles_earnings_l3052_305293


namespace product_expansion_l3052_305201

theorem product_expansion (x : ℝ) : 
  (2 * x^2 - 3 * x + 4) * (2 * x^2 + 3 * x + 4) = 4 * x^4 + 7 * x^2 + 16 := by
  sorry

end product_expansion_l3052_305201


namespace closest_cube_approximation_l3052_305286

def x : Real := 0.48017

theorem closest_cube_approximation :
  ∀ y ∈ ({0.011, 1.10, 11.0, 110} : Set Real),
  |x^3 - 0.110| < |x^3 - y| := by sorry

end closest_cube_approximation_l3052_305286


namespace club_leadership_combinations_l3052_305278

/-- Represents the total number of members in the club -/
def total_members : ℕ := 24

/-- Represents the number of boys in the club -/
def num_boys : ℕ := 12

/-- Represents the number of girls in the club -/
def num_girls : ℕ := 12

/-- Represents the number of age groups -/
def num_age_groups : ℕ := 2

/-- Represents the number of members in each gender and age group combination -/
def members_per_group : ℕ := 6

/-- Theorem stating the number of ways to choose a president and vice-president -/
theorem club_leadership_combinations : 
  (num_boys * members_per_group + num_girls * members_per_group) = 144 := by
  sorry

end club_leadership_combinations_l3052_305278


namespace power_function_m_value_l3052_305206

theorem power_function_m_value (m : ℕ+) (f : ℝ → ℝ) : 
  (∀ x, f x = x ^ (m.val ^ 2 + m.val)) → 
  f (Real.sqrt 2) = 2 → 
  m = 1 := by sorry

end power_function_m_value_l3052_305206


namespace k_range_l3052_305251

theorem k_range (x y k : ℝ) : 
  (3 * x + y = k + 1) → 
  (x + 3 * y = 3) → 
  (0 < x + y) → 
  (x + y < 1) → 
  (-4 < k ∧ k < 0) := by
sorry

end k_range_l3052_305251


namespace geometric_sequence_ratio_sum_l3052_305263

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : k ≠ 0) (h2 : p ≠ r) :
  k * p^2 - k * r^2 = 5 * (k * p - k * r) → p + r = 5 := by
sorry

end geometric_sequence_ratio_sum_l3052_305263


namespace sphere_volume_from_surface_area_l3052_305277

/-- Given a sphere with surface area 4π, its volume is 4π/3 -/
theorem sphere_volume_from_surface_area :
  ∀ r : ℝ, 4 * Real.pi * r^2 = 4 * Real.pi → (4 / 3) * Real.pi * r^3 = (4 / 3) * Real.pi := by
  sorry

end sphere_volume_from_surface_area_l3052_305277


namespace solution_set_of_inequality_l3052_305218

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is increasing on (0, +∞) if f(x) < f(y) for all 0 < x < y -/
def IsIncreasingOnPositiveReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

theorem solution_set_of_inequality (f : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_incr : IsIncreasingOnPositiveReals f)
    (h_zero : f (-3) = 0) :
    {x : ℝ | (x - 2) * f x < 0} = Set.Ioo (-3) 0 ∪ Set.Ioo 2 3 := by
  sorry

end solution_set_of_inequality_l3052_305218


namespace equation_solution_l3052_305246

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = -4 ∧ x₂ = -2) ∧ 
  (∀ x : ℝ, (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solution_l3052_305246


namespace power_equation_solution_l3052_305225

theorem power_equation_solution :
  ∀ m : ℤ, 3 * 2^2000 - 5 * 2^1999 + 4 * 2^1998 - 2^1997 = m * 2^1997 → m = 11 := by
  sorry

end power_equation_solution_l3052_305225


namespace horner_v2_equals_10_l3052_305298

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^7 + x^6 + x^4 + x^2 + 1 -/
def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [1, 0, 1, 0, 1, 0, 1, 2]

/-- Theorem: V_2 in Horner's method for f(x) when x = 2 is 10 -/
theorem horner_v2_equals_10 : 
  (horner (coeffs.take 3) 2) = 10 := by sorry

end horner_v2_equals_10_l3052_305298


namespace intersection_of_A_and_B_l3052_305222

-- Define set A
def A : Set ℝ := {x | |x| ≤ 1}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = x^2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by sorry

end intersection_of_A_and_B_l3052_305222


namespace largest_integer_with_remainder_l3052_305224

theorem largest_integer_with_remainder (n : ℕ) : n < 100 ∧ n % 9 = 5 ∧ ∀ m, m < 100 ∧ m % 9 = 5 → m ≤ n ↔ n = 95 := by
  sorry

end largest_integer_with_remainder_l3052_305224


namespace greatest_three_digit_multiple_of_17_l3052_305292

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 17 = 0 → n ≤ 986 :=
by sorry

end greatest_three_digit_multiple_of_17_l3052_305292


namespace allan_balloons_l3052_305226

def park_balloon_problem (jake_initial : ℕ) (jake_bought : ℕ) (difference : ℕ) : ℕ :=
  let jake_total := jake_initial + jake_bought
  jake_total - difference

theorem allan_balloons :
  park_balloon_problem 3 4 1 = 6 :=
by
  sorry

end allan_balloons_l3052_305226


namespace negative_a_cubed_times_negative_a_fourth_l3052_305295

theorem negative_a_cubed_times_negative_a_fourth (a : ℝ) : -a^3 * (-a)^4 = -a^7 := by
  sorry

end negative_a_cubed_times_negative_a_fourth_l3052_305295


namespace box_volume_formula_l3052_305290

/-- The volume of an open box formed from a rectangular sheet --/
def boxVolume (x y : ℝ) : ℝ := (16 - 2*x) * (12 - 2*y) * y

/-- Theorem stating the volume of the box --/
theorem box_volume_formula (x y : ℝ) :
  boxVolume x y = 192*y - 32*y^2 - 24*x*y + 4*x*y^2 := by
  sorry

end box_volume_formula_l3052_305290


namespace remainder_x_50_divided_by_x_plus_1_cubed_l3052_305214

theorem remainder_x_50_divided_by_x_plus_1_cubed (x : ℚ) :
  (x^50) % (x + 1)^3 = 1225*x^2 + 2450*x + 1176 := by
  sorry

end remainder_x_50_divided_by_x_plus_1_cubed_l3052_305214


namespace add_decimal_numbers_l3052_305204

theorem add_decimal_numbers : 0.45 + 57.25 = 57.70 := by
  sorry

end add_decimal_numbers_l3052_305204


namespace remainder_777_444_mod_13_l3052_305257

theorem remainder_777_444_mod_13 : 777^444 ≡ 1 [ZMOD 13] := by
  sorry

end remainder_777_444_mod_13_l3052_305257


namespace students_passing_both_subjects_l3052_305256

theorem students_passing_both_subjects (total_english : ℕ) (total_math : ℕ) (diff_only_english : ℕ) :
  total_english = 30 →
  total_math = 20 →
  diff_only_english = 10 →
  ∃ (both : ℕ),
    both = 10 ∧
    total_english = both + (both + diff_only_english) ∧
    total_math = both + both :=
by sorry

end students_passing_both_subjects_l3052_305256


namespace opposite_abs_sum_l3052_305282

theorem opposite_abs_sum (a m n : ℝ) : 
  (|a - 2| + |m + n + 3| = 0) → (a + m + n = -1) := by
sorry

end opposite_abs_sum_l3052_305282


namespace complex_equality_l3052_305228

theorem complex_equality (a : ℝ) : 
  (Complex.re ((1 + 2*Complex.I) * (a + Complex.I)) = Complex.im ((1 + 2*Complex.I) * (a + Complex.I))) → 
  a = -3 := by
sorry

end complex_equality_l3052_305228


namespace multiply_93_107_l3052_305240

theorem multiply_93_107 : 93 * 107 = 9951 := by
  sorry

end multiply_93_107_l3052_305240


namespace min_even_integers_l3052_305215

theorem min_even_integers (a b c d e f g h : ℤ) : 
  a + b + c = 30 → 
  a + b + c + d + e = 49 → 
  a + b + c + d + e + f + g + h = 78 → 
  ∃ (evens : Finset ℤ), evens ⊆ {a, b, c, d, e, f, g, h} ∧ 
                         evens.card = 2 ∧
                         (∀ x ∈ evens, Even x) ∧
                         (∀ (other_evens : Finset ℤ), 
                           other_evens ⊆ {a, b, c, d, e, f, g, h} → 
                           (∀ x ∈ other_evens, Even x) → 
                           other_evens.card ≥ 2) :=
by sorry

end min_even_integers_l3052_305215


namespace circle_area_and_circumference_l3052_305211

/-- Given a circle described by the polar equation r = 4 cos θ + 3 sin θ,
    prove that its area is 25π/4 and its circumference is 5π. -/
theorem circle_area_and_circumference :
  ∀ θ : ℝ, ∃ r : ℝ, r = 4 * Real.cos θ + 3 * Real.sin θ →
  ∃ A C : ℝ, A = (25 * Real.pi) / 4 ∧ C = 5 * Real.pi := by
  sorry

end circle_area_and_circumference_l3052_305211


namespace line_parallel_plane_neither_necessary_nor_sufficient_l3052_305288

/-- Two lines are perpendicular -/
def perpendicular (l₁ l₂ : Line) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem line_parallel_plane_neither_necessary_nor_sufficient
  (m n : Line) (α : Plane) (h : perpendicular m n) :
  ¬(∀ (m n : Line) (α : Plane), perpendicular m n → (line_parallel_plane n α ↔ line_perp_plane m α)) :=
by sorry

end line_parallel_plane_neither_necessary_nor_sufficient_l3052_305288


namespace arccos_one_half_l3052_305250

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_l3052_305250


namespace parabola_intercept_sum_l3052_305233

/-- Represents a parabola of the form x = 3y^2 - 9y + 5 --/
def Parabola (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- The x-intercept of the parabola --/
def x_intercept (a : ℝ) : Prop := Parabola a 0

/-- The y-intercepts of the parabola --/
def y_intercepts (b c : ℝ) : Prop := Parabola 0 b ∧ Parabola 0 c ∧ b ≠ c

theorem parabola_intercept_sum (a b c : ℝ) :
  x_intercept a → y_intercepts b c → a + b + c = 8 := by
  sorry

end parabola_intercept_sum_l3052_305233


namespace sum_fractions_equals_11111_l3052_305253

theorem sum_fractions_equals_11111 : 
  4/5 + 9 * (4/5) + 99 * (4/5) + 999 * (4/5) + 9999 * (4/5) + 1 = 11111 := by
  sorry

end sum_fractions_equals_11111_l3052_305253


namespace flour_needed_for_one_batch_l3052_305296

/-- The number of cups of flour needed for one batch of cookies -/
def flour_per_batch : ℝ := 4

/-- The number of cups of sugar needed for one batch of cookies -/
def sugar_per_batch : ℝ := 1.5

/-- The total number of cups of flour and sugar needed for 8 batches -/
def total_for_eight_batches : ℝ := 44

theorem flour_needed_for_one_batch :
  flour_per_batch = 4 :=
by
  have h1 : sugar_per_batch = 1.5 := rfl
  have h2 : total_for_eight_batches = 44 := rfl
  have h3 : 8 * flour_per_batch + 8 * sugar_per_batch = total_for_eight_batches := by sorry
  sorry

end flour_needed_for_one_batch_l3052_305296


namespace jane_sequins_count_l3052_305231

/-- The number of rows of blue sequins -/
def blue_rows : Nat := 6

/-- The number of blue sequins in each row -/
def blue_per_row : Nat := 8

/-- The number of rows of purple sequins -/
def purple_rows : Nat := 5

/-- The number of purple sequins in each row -/
def purple_per_row : Nat := 12

/-- The number of rows of green sequins -/
def green_rows : Nat := 9

/-- The number of green sequins in each row -/
def green_per_row : Nat := 6

/-- The total number of sequins Jane adds to her costume -/
def total_sequins : Nat := blue_rows * blue_per_row + purple_rows * purple_per_row + green_rows * green_per_row

theorem jane_sequins_count : total_sequins = 162 := by
  sorry

end jane_sequins_count_l3052_305231


namespace furniture_legs_l3052_305274

theorem furniture_legs 
  (total_tables : ℕ) 
  (total_legs : ℕ) 
  (four_leg_tables : ℕ) 
  (h1 : total_tables = 36)
  (h2 : total_legs = 124)
  (h3 : four_leg_tables = 16) :
  (total_legs - 4 * four_leg_tables) / (total_tables - four_leg_tables) = 3 := by
sorry

end furniture_legs_l3052_305274


namespace inequality_proof_l3052_305280

noncomputable section

variables (a : ℝ) (x₁ x₂ : ℝ)

def f (x : ℝ) := x^2 + 2/x + a * Real.log x

theorem inequality_proof (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) (h₄ : a ≤ 0) :
  (f a x₁ + f a x₂) / 2 > f a ((x₁ + x₂) / 2) :=
sorry

end

end inequality_proof_l3052_305280


namespace boxes_remaining_l3052_305229

theorem boxes_remaining (total : ℕ) (filled : ℕ) (h1 : total = 13) (h2 : filled = 8) :
  total - filled = 5 := by
  sorry

end boxes_remaining_l3052_305229


namespace min_groups_for_club_l3052_305261

/-- Given a club with 30 members and a maximum group size of 12,
    the minimum number of groups required is 3. -/
theorem min_groups_for_club (total_members : ℕ) (max_group_size : ℕ) :
  total_members = 30 →
  max_group_size = 12 →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_members ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) →
  (∃ (num_groups : ℕ), 
    num_groups * max_group_size ≥ total_members ∧
    ∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) ∧
  (∀ (num_groups : ℕ),
    num_groups * max_group_size ≥ total_members ∧
    (∀ (k : ℕ), k * max_group_size ≥ total_members → k ≥ num_groups) →
    num_groups = 3) :=
by
  sorry


end min_groups_for_club_l3052_305261


namespace complement_A_in_U_is_correct_l3052_305299

-- Define the universal set U
def U : Set Int := {x | -2 ≤ x ∧ x ≤ 6}

-- Define set A
def A : Set Int := {x | ∃ n : Nat, x = 2 * n ∧ n ≤ 3}

-- Define the complement of A in U
def complement_A_in_U : Set Int := U \ A

-- Theorem to prove
theorem complement_A_in_U_is_correct :
  complement_A_in_U = {-2, -1, 1, 3, 5} := by sorry

end complement_A_in_U_is_correct_l3052_305299


namespace mixed_doubles_pairing_methods_l3052_305273

theorem mixed_doubles_pairing_methods (total_players : Nat) (male_players : Nat) (female_players : Nat) 
  (selected_male : Nat) (selected_female : Nat) :
  total_players = male_players + female_players →
  male_players = 5 →
  female_players = 4 →
  selected_male = 2 →
  selected_female = 2 →
  (Nat.choose male_players selected_male) * (Nat.choose female_players selected_female) * 
  (Nat.factorial selected_male) = 120 := by
sorry

end mixed_doubles_pairing_methods_l3052_305273


namespace sin_cos_identity_l3052_305268

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1/2 := by
  sorry

end sin_cos_identity_l3052_305268


namespace line_ellipse_intersection_range_l3052_305209

/-- The range of m for which the line y = kx + 1 always intersects the ellipse x²/5 + y²/m = 1 at two points -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), (∀ (x y : ℝ), (y = k*x + 1 ∧ x^2/5 + y^2/m = 1) → ∃! (p q : ℝ × ℝ), p ≠ q ∧ 
    (p.1^2/5 + p.2^2/m = 1) ∧ (q.1^2/5 + q.2^2/m = 1) ∧ 
    p.2 = k*p.1 + 1 ∧ q.2 = k*q.1 + 1) ↔ 
  (m > 1 ∧ m < 5) ∨ m > 5 :=
sorry

end line_ellipse_intersection_range_l3052_305209


namespace mirror_pieces_l3052_305271

theorem mirror_pieces (total : ℕ) (swept : ℕ) (stolen : ℕ) (picked_fraction : ℚ) : 
  total = 60 →
  swept = total / 2 →
  stolen = 3 →
  picked_fraction = 1 / 3 →
  (total - swept - stolen) * picked_fraction = 9 := by
sorry

end mirror_pieces_l3052_305271


namespace number_count_l3052_305203

theorem number_count (avg_all : Real) (avg1 : Real) (avg2 : Real) (avg3 : Real) 
  (h1 : avg_all = 3.95)
  (h2 : avg1 = 3.8)
  (h3 : avg2 = 3.85)
  (h4 : avg3 = 4.200000000000001)
  (h5 : 2 * avg1 + 2 * avg2 + 2 * avg3 = avg_all * 6) :
  6 = (2 * avg1 + 2 * avg2 + 2 * avg3) / avg_all := by
  sorry

end number_count_l3052_305203


namespace f_cos_10_deg_l3052_305279

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem f_cos_10_deg : 
  (∀ x, f (Real.sin x) = Real.cos (3 * x)) → 
  f (Real.cos (10 * π / 180)) = -1/2 := by sorry

end f_cos_10_deg_l3052_305279


namespace quadratic_always_intersects_x_axis_l3052_305236

theorem quadratic_always_intersects_x_axis (a : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3*a + 1) * x + 3 = 0 :=
by sorry

end quadratic_always_intersects_x_axis_l3052_305236


namespace jakes_weight_l3052_305254

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 33 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 153) : 
  jake_weight = 113 := by
sorry

end jakes_weight_l3052_305254


namespace sum_of_divisors_91_l3052_305221

/-- The sum of all positive divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all positive divisors of 91 is 112 -/
theorem sum_of_divisors_91 : sum_of_divisors 91 = 112 := by sorry

end sum_of_divisors_91_l3052_305221


namespace hotel_rooms_l3052_305249

theorem hotel_rooms (total_rooms : ℕ) (single_cost double_cost : ℕ) (total_revenue : ℕ) :
  total_rooms = 260 ∧
  single_cost = 35 ∧
  double_cost = 60 ∧
  total_revenue = 14000 →
  ∃ (single_rooms double_rooms : ℕ),
    single_rooms + double_rooms = total_rooms ∧
    single_cost * single_rooms + double_cost * double_rooms = total_revenue ∧
    single_rooms = 64 :=
by sorry

end hotel_rooms_l3052_305249


namespace arithmetic_sequence_ratio_l3052_305258

/-- Given two arithmetic sequences {a_n} and {b_n} with sums S_n and T_n respectively,
    if S_n/T_n = 2n/(3n+1) for all natural numbers n, then a_5/b_5 = 9/14 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2) →
  (∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2) →
  (∀ n : ℕ, S n / T n = (2 * n : ℚ) / (3 * n + 1)) →
  a 5 / b 5 = 9 / 14 := by
  sorry

end arithmetic_sequence_ratio_l3052_305258


namespace boxes_on_pallet_l3052_305252

/-- 
Given a pallet of boxes with a total weight and the weight of each box,
calculate the number of boxes on the pallet.
-/
theorem boxes_on_pallet (total_weight : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267)
  (h2 : box_weight = 89) :
  total_weight / box_weight = 3 :=
by sorry

end boxes_on_pallet_l3052_305252


namespace horner_v4_equals_80_l3052_305276

/-- Horner's Rule for polynomial evaluation --/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 12x^5 + 60x^4 - 160x^3 + 240x^2 - 192x + 64 --/
def f (x : ℝ) : ℝ :=
  x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

/-- The coefficients of the polynomial in reverse order --/
def coeffs : List ℝ := [64, -192, 240, -160, 60, -12, 1]

/-- The value of x for which we're evaluating the polynomial --/
def x : ℝ := 2

/-- The intermediate value v_4 in Horner's Rule calculation --/
def v_4 : ℝ := ((-80 * x) + 240)

theorem horner_v4_equals_80 : v_4 = 80 := by
  sorry

#eval v_4

end horner_v4_equals_80_l3052_305276


namespace boats_by_april_l3052_305242

def boats_in_month (n : Nat) : Nat :=
  match n with
  | 0 => 4  -- January
  | 1 => 2  -- February
  | m + 2 => 3 * boats_in_month (m + 1)  -- March onwards

def total_boats (n : Nat) : Nat :=
  match n with
  | 0 => boats_in_month 0
  | m + 1 => boats_in_month (m + 1) + total_boats m

theorem boats_by_april : total_boats 3 = 30 := by
  sorry

end boats_by_april_l3052_305242


namespace regression_analysis_considerations_l3052_305255

/-- Represents the key considerations in regression analysis predictions -/
inductive RegressionConsideration
  | ApplicabilityToSamplePopulation
  | Temporality
  | InfluenceOfSampleRange
  | PredictionPrecision

/-- Represents a regression analysis model -/
structure RegressionModel where
  considerations : List RegressionConsideration

/-- Theorem stating the key considerations in regression analysis predictions -/
theorem regression_analysis_considerations (model : RegressionModel) :
  model.considerations = [
    RegressionConsideration.ApplicabilityToSamplePopulation,
    RegressionConsideration.Temporality,
    RegressionConsideration.InfluenceOfSampleRange,
    RegressionConsideration.PredictionPrecision
  ] := by sorry


end regression_analysis_considerations_l3052_305255
