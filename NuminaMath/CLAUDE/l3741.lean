import Mathlib

namespace quadratic_inequality_theorem_l3741_374154

-- Define the quadratic function
def f (k x : ℝ) := k * x^2 - 2 * x + 6 * k

-- Define the solution set
def solution_set (k : ℝ) := {x : ℝ | f k x < 0}

-- Define the interval (2, 3)
def interval := {x : ℝ | 2 < x ∧ x < 3}

theorem quadratic_inequality_theorem (k : ℝ) (h : k > 0) :
  (solution_set k = interval → k = 2/5) ∧
  (∀ x ∈ interval, f k x < 0 → 0 < k ∧ k ≤ 2/5) ∧
  (solution_set k ⊆ interval → 2/5 ≤ k) := by sorry

end quadratic_inequality_theorem_l3741_374154


namespace expression_equality_l3741_374110

theorem expression_equality : (1/4)⁻¹ - |Real.sqrt 3 - 2| + 2 * (-Real.sqrt 3) = 2 - Real.sqrt 3 := by
  sorry

end expression_equality_l3741_374110


namespace division_result_l3741_374130

theorem division_result : (24 : ℝ) / (52 - 40) = 2 := by
  sorry

end division_result_l3741_374130


namespace absolute_value_fraction_l3741_374140

theorem absolute_value_fraction : (|3| / |(-2)^3|) = -2 := by sorry

end absolute_value_fraction_l3741_374140


namespace wilcoxon_rank_sum_test_result_l3741_374131

def sample1 : List ℝ := [3, 4, 6, 10, 13, 17]
def sample2 : List ℝ := [1, 2, 5, 7, 16, 20, 22]

def significanceLevel : ℝ := 0.01

def calculateRankSum (sample : List ℝ) (allValues : List ℝ) : ℕ :=
  sorry

def wilcoxonRankSumTest (sample1 sample2 : List ℝ) (significanceLevel : ℝ) : Bool :=
  sorry

theorem wilcoxon_rank_sum_test_result :
  let n1 := sample1.length
  let n2 := sample2.length
  let allValues := sample1 ++ sample2
  let W1 := calculateRankSum sample1 allValues
  let Wlower := 24  -- Critical value from Wilcoxon rank-sum test table
  let Wupper := (n1 + n2 + 1) * n1 - Wlower
  Wlower < W1 ∧ W1 < Wupper ∧ wilcoxonRankSumTest sample1 sample2 significanceLevel = false :=
by
  sorry

end wilcoxon_rank_sum_test_result_l3741_374131


namespace selling_price_is_twenty_l3741_374175

/-- Calculates the selling price per phone given the total number of phones, 
    total cost, and desired profit ratio. -/
def selling_price_per_phone (total_phones : ℕ) (total_cost : ℚ) (profit_ratio : ℚ) : ℚ :=
  let cost_per_phone := total_cost / total_phones
  let profit_per_phone := (total_cost * profit_ratio) / total_phones
  cost_per_phone + profit_per_phone

/-- Theorem stating that the selling price per phone is $20 given the problem conditions. -/
theorem selling_price_is_twenty :
  selling_price_per_phone 200 3000 (1/3) = 20 := by
  sorry

end selling_price_is_twenty_l3741_374175


namespace max_cities_is_four_l3741_374191

/-- Represents the modes of transportation --/
inductive TransportMode
| Bus
| Train
| Airplane

/-- Represents a city in the country --/
structure City where
  id : Nat

/-- Represents the transportation network of the country --/
structure TransportNetwork where
  cities : List City
  connections : List City → List City → TransportMode → Prop

/-- Checks if the network satisfies the condition that no city is serviced by all three types of transportation --/
def noTripleService (network : TransportNetwork) : Prop :=
  ∀ c : City, c ∈ network.cities →
    ¬(∃ (c1 c2 c3 : City), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
      network.connections [c, c1] [c, c1] TransportMode.Bus ∧
      network.connections [c, c2] [c, c2] TransportMode.Train ∧
      network.connections [c, c3] [c, c3] TransportMode.Airplane)

/-- Checks if the network satisfies the condition that no three cities are connected by the same mode of transportation --/
def noTripleConnection (network : TransportNetwork) : Prop :=
  ∀ mode : TransportMode, ¬(∃ (c1 c2 c3 : City), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧
    network.connections [c1, c2] [c1, c2] mode ∧
    network.connections [c2, c3] [c2, c3] mode ∧
    network.connections [c1, c3] [c1, c3] mode)

/-- The main theorem stating that the maximum number of cities is 4 --/
theorem max_cities_is_four :
  ∀ (network : TransportNetwork),
    (∀ (c1 c2 : City), c1 ≠ c2 → ∃ (mode : TransportMode), network.connections [c1, c2] [c1, c2] mode) →
    noTripleService network →
    noTripleConnection network →
    List.length network.cities ≤ 4 :=
sorry

end max_cities_is_four_l3741_374191


namespace financial_equation_solution_l3741_374116

/-- Given a financial equation and some conditions, prove the value of p -/
theorem financial_equation_solution (q v : ℂ) (h1 : 3 * q - v = 5000) (h2 : q = 3) (h3 : v = 3 + 75 * Complex.I) :
  ∃ p : ℂ, p = 1667 + 25 * Complex.I := by
  sorry

end financial_equation_solution_l3741_374116


namespace sandys_phone_bill_l3741_374145

theorem sandys_phone_bill (kim_age : ℕ) (sandy_age : ℕ) (sandy_bill : ℕ) :
  kim_age = 10 →
  sandy_age + 2 = 3 * (kim_age + 2) →
  sandy_bill = 10 * sandy_age →
  sandy_bill = 340 := by
  sorry

end sandys_phone_bill_l3741_374145


namespace two_red_cards_probability_l3741_374164

/-- The probability of drawing two red cards in succession from a deck of 100 cards
    containing 50 red cards and 50 black cards, without replacement. -/
theorem two_red_cards_probability (total_cards : ℕ) (red_cards : ℕ) (black_cards : ℕ) 
    (h1 : total_cards = 100)
    (h2 : red_cards = 50)
    (h3 : black_cards = 50)
    (h4 : total_cards = red_cards + black_cards) :
    (red_cards : ℚ) / total_cards * ((red_cards - 1) : ℚ) / (total_cards - 1) = 49 / 198 := by
  sorry

end two_red_cards_probability_l3741_374164


namespace sum_sqrt_squared_pairs_geq_sqrt2_l3741_374181

theorem sum_sqrt_squared_pairs_geq_sqrt2 (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : 
  Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + a^2) ≥ Real.sqrt 2 := by
  sorry

end sum_sqrt_squared_pairs_geq_sqrt2_l3741_374181


namespace dining_bill_calculation_l3741_374136

def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  let total := price_with_tax * (1 + tip_rate)
  total

theorem dining_bill_calculation :
  total_amount_spent 100 0.1 0.2 = 132 := by
  sorry

end dining_bill_calculation_l3741_374136


namespace lucy_money_theorem_l3741_374173

def lucy_money_problem (initial_amount : ℚ) : ℚ :=
  let remaining_after_loss := initial_amount * (1 - 1/3)
  let spent := remaining_after_loss * (1/4)
  remaining_after_loss - spent

theorem lucy_money_theorem :
  lucy_money_problem 30 = 15 := by sorry

end lucy_money_theorem_l3741_374173


namespace jims_journey_distance_l3741_374166

/-- The total distance of Jim's journey -/
def total_distance (driven : ℕ) (remaining : ℕ) : ℕ :=
  driven + remaining

/-- Theorem stating the total distance of Jim's journey -/
theorem jims_journey_distance :
  total_distance 215 985 = 1200 := by
  sorry

end jims_journey_distance_l3741_374166


namespace transformed_area_doubled_l3741_374132

-- Define a function representing the area under a curve
noncomputable def areaUnderCurve (f : ℝ → ℝ) (a b : ℝ) : ℝ := sorry

-- Define the original function g
variable (g : ℝ → ℝ)

-- Define the interval [a, b] over which we're measuring the area
variable (a b : ℝ)

-- Theorem statement
theorem transformed_area_doubled 
  (h : areaUnderCurve g a b = 15) : 
  areaUnderCurve (fun x ↦ 2 * g (x + 3)) a b = 30 := by
  sorry

end transformed_area_doubled_l3741_374132


namespace no_valid_right_triangle_with_prime_angles_l3741_374199

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_valid_right_triangle_with_prime_angles :
  ¬ ∃ (x : ℕ), 
    x > 0 ∧ 
    3 * x < 90 ∧ 
    x + 3 * x = 90 ∧ 
    is_prime x ∧ 
    is_prime (3 * x) :=
sorry

end no_valid_right_triangle_with_prime_angles_l3741_374199


namespace square_of_six_y_minus_two_l3741_374172

theorem square_of_six_y_minus_two (y : ℝ) (h : 3 * y^2 + 6 = 2 * y + 10) : (6 * y - 2)^2 = 52 := by
  sorry

end square_of_six_y_minus_two_l3741_374172


namespace share_multiple_l3741_374101

theorem share_multiple (total : ℕ) (c_share : ℕ) (k : ℕ) : 
  total = 880 → c_share = 160 → 
  (∃ (a_share b_share : ℕ), 
    a_share + b_share + c_share = total ∧ 
    4 * a_share = k * b_share ∧ 
    k * b_share = 10 * c_share) → 
  k = 5 := by
sorry

end share_multiple_l3741_374101


namespace oplus_three_two_l3741_374123

def oplus (a b : ℕ) : ℕ := a + b + a * b - 1

theorem oplus_three_two : oplus 3 2 = 10 := by
  sorry

end oplus_three_two_l3741_374123


namespace election_majority_proof_l3741_374139

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 455 → 
  winning_percentage = 70 / 100 → 
  ⌊(2 * winning_percentage - 1) * total_votes⌋ = 182 := by
  sorry

end election_majority_proof_l3741_374139


namespace manuscript_typing_cost_l3741_374142

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptCost (totalPages : ℕ) (initialCost revisonCost : ℚ) 
  (revisedOnce revisedTwice : ℕ) : ℚ :=
  let initialTypingCost := totalPages * initialCost
  let firstRevisionCost := revisedOnce * revisonCost
  let secondRevisionCost := revisedTwice * (2 * revisonCost)
  initialTypingCost + firstRevisionCost + secondRevisionCost

/-- Theorem stating that the total cost of typing the manuscript is $1360. -/
theorem manuscript_typing_cost : 
  manuscriptCost 200 5 3 80 20 = 1360 := by
  sorry

end manuscript_typing_cost_l3741_374142


namespace arithmetic_sequence_sum_l3741_374180

theorem arithmetic_sequence_sum (n : ℕ) : 
  (Finset.range (n + 3)).sum (fun i => 2 * i + 3) = n^2 + 8*n + 15 := by
  sorry

end arithmetic_sequence_sum_l3741_374180


namespace interval_intersection_l3741_374106

theorem interval_intersection (x : ℝ) : 
  (2 < 4*x ∧ 4*x < 4) ∧ (2 < 5*x ∧ 5*x < 4) ↔ (1/2 < x ∧ x < 4/5) :=
by sorry

end interval_intersection_l3741_374106


namespace water_purifier_filtration_layers_l3741_374111

theorem water_purifier_filtration_layers (initial_impurities : ℝ) (target_impurities : ℝ) 
  (filter_efficiency : ℝ) (h1 : initial_impurities = 80) (h2 : target_impurities = 2) 
  (h3 : filter_efficiency = 1/3) : 
  ∃ n : ℕ, (initial_impurities * (1 - filter_efficiency)^n ≤ target_impurities ∧ 
  ∀ m : ℕ, m < n → initial_impurities * (1 - filter_efficiency)^m > target_impurities) :=
sorry

end water_purifier_filtration_layers_l3741_374111


namespace power_of_three_mod_eight_l3741_374113

theorem power_of_three_mod_eight : 3^2028 % 8 = 1 := by
  sorry

end power_of_three_mod_eight_l3741_374113


namespace quadruple_pieces_sold_l3741_374158

/-- Represents the number of pieces sold for each type --/
structure PiecesSold where
  single : Nat
  double : Nat
  triple : Nat
  quadruple : Nat

/-- Calculates the total earnings in cents --/
def totalEarnings (pieces : PiecesSold) : Nat :=
  pieces.single + 2 * pieces.double + 3 * pieces.triple + 4 * pieces.quadruple

/-- The main theorem to prove --/
theorem quadruple_pieces_sold (pieces : PiecesSold) :
  pieces.single = 100 ∧ 
  pieces.double = 45 ∧ 
  pieces.triple = 50 ∧ 
  totalEarnings pieces = 1000 →
  pieces.quadruple = 165 := by
  sorry

#eval totalEarnings { single := 100, double := 45, triple := 50, quadruple := 165 }

end quadruple_pieces_sold_l3741_374158


namespace vector_distance_inequality_l3741_374190

noncomputable def max_T : ℝ := (Real.sqrt 6 - Real.sqrt 2) / 4

theorem vector_distance_inequality (a b : ℝ × ℝ) :
  (∀ m n : ℝ, let c := (m, 1 - m)
               let d := (n, 1 - n)
               (a.1 - c.1)^2 + (a.2 - c.2)^2 + (b.1 - d.1)^2 + (b.2 - d.2)^2 ≥ max_T^2) →
  (norm a = 1 ∧ norm b = 1 ∧ a.1 * b.1 + a.2 * b.2 = 1/2) :=
sorry

end vector_distance_inequality_l3741_374190


namespace negative_of_difference_l3741_374102

theorem negative_of_difference (a b : ℝ) : -(a - b) = -a + b := by sorry

end negative_of_difference_l3741_374102


namespace arithmetic_sequence_problem_l3741_374176

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 8 = 9) 
  (h_a4 : a 4 = 3) : 
  a 5 = 6 := by
  sorry

end arithmetic_sequence_problem_l3741_374176


namespace investment_interest_calculation_l3741_374144

/-- Calculates the total interest earned on an investment -/
def total_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- The problem statement -/
theorem investment_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let time : ℕ := 5
  abs (total_interest principal rate time - 938.66) < 0.01 := by
  sorry

end investment_interest_calculation_l3741_374144


namespace fiftieth_term_of_specific_sequence_l3741_374174

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 50th term of the arithmetic sequence with first term 3 and common difference 7 is 346 -/
theorem fiftieth_term_of_specific_sequence : 
  arithmeticSequenceTerm 3 7 50 = 346 := by
  sorry

#check fiftieth_term_of_specific_sequence

end fiftieth_term_of_specific_sequence_l3741_374174


namespace complex_number_location_l3741_374186

theorem complex_number_location (z : ℂ) (h : z * Complex.I = 2 - Complex.I) :
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end complex_number_location_l3741_374186


namespace housing_units_without_cable_or_vcr_l3741_374198

theorem housing_units_without_cable_or_vcr 
  (total : ℝ) 
  (cable : ℝ) 
  (vcr : ℝ) 
  (both : ℝ) 
  (h1 : cable = (1 / 5) * total) 
  (h2 : vcr = (1 / 10) * total) 
  (h3 : both = (1 / 3) * cable) :
  (total - (cable + vcr - both)) / total = 23 / 30 := by
sorry

end housing_units_without_cable_or_vcr_l3741_374198


namespace chocolates_needed_to_fill_last_box_l3741_374178

def chocolates_per_box : ℕ := 30
def total_chocolates : ℕ := 254

theorem chocolates_needed_to_fill_last_box : 
  (chocolates_per_box - (total_chocolates % chocolates_per_box)) = 16 := by
  sorry

end chocolates_needed_to_fill_last_box_l3741_374178


namespace import_tax_problem_l3741_374122

/-- The import tax rate as a decimal -/
def tax_rate : ℝ := 0.07

/-- The threshold above which the tax is applied -/
def tax_threshold : ℝ := 1000

/-- The amount of tax paid -/
def tax_paid : ℝ := 112.70

/-- The total value of the item -/
def total_value : ℝ := 2610

theorem import_tax_problem :
  tax_rate * (total_value - tax_threshold) = tax_paid :=
by sorry

end import_tax_problem_l3741_374122


namespace complex_modulus_of_iz_eq_one_l3741_374127

theorem complex_modulus_of_iz_eq_one (z : ℂ) (h : Complex.I * z = 1) : Complex.abs z = 1 := by
  sorry

end complex_modulus_of_iz_eq_one_l3741_374127


namespace boxes_with_neither_pens_nor_pencils_l3741_374135

theorem boxes_with_neither_pens_nor_pencils 
  (total_boxes : ℕ) 
  (boxes_with_pencils : ℕ) 
  (boxes_with_pens : ℕ) 
  (boxes_with_both : ℕ) 
  (h1 : total_boxes = 15)
  (h2 : boxes_with_pencils = 7)
  (h3 : boxes_with_pens = 4)
  (h4 : boxes_with_both = 3) :
  total_boxes - (boxes_with_pencils + boxes_with_pens - boxes_with_both) = 7 :=
by sorry

end boxes_with_neither_pens_nor_pencils_l3741_374135


namespace expression_simplification_l3741_374195

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (1 - (x + 1) / x) / ((x^2 - 1) / (x^2 - x)) = -Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3741_374195


namespace hyperbola_eccentricity_l3741_374108

/-- The eccentricity of a hyperbola with equation mx² + y² = 1 and eccentricity √2 is -1 -/
theorem hyperbola_eccentricity (m : ℝ) : 
  (∀ x y : ℝ, m * x^2 + y^2 = 1) →  -- Equation of the hyperbola
  (∃ a c : ℝ, a > 0 ∧ c > a ∧ (c/a)^2 = 2) →  -- Eccentricity is √2
  m = -1 :=
by sorry

end hyperbola_eccentricity_l3741_374108


namespace factor_of_quadratic_l3741_374138

theorem factor_of_quadratic (m x : ℤ) : 
  (∃ k : ℤ, (m - x) * k = m^2 - 5*m - 24) → x = 8 := by
  sorry

end factor_of_quadratic_l3741_374138


namespace power_calculation_l3741_374134

theorem power_calculation : (2 : ℝ)^2021 * (-1/2 : ℝ)^2022 = 1/2 := by sorry

end power_calculation_l3741_374134


namespace triangle_area_from_perimeter_and_inradius_l3741_374150

/-- Theorem: Area of a triangle with given perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius
  (perimeter : ℝ) (inradius : ℝ) (h_perimeter : perimeter = 39)
  (h_inradius : inradius = 1.5) :
  perimeter * inradius / 4 = 29.25 := by
  sorry

end triangle_area_from_perimeter_and_inradius_l3741_374150


namespace chemistry_mean_marks_l3741_374182

/-- Proves that the mean mark in the second section is 60 given the conditions of the problem -/
theorem chemistry_mean_marks (n₁ n₂ n₃ n₄ : ℕ) (m₁ m₃ m₄ : ℚ) (overall_avg : ℚ) :
  n₁ = 60 →
  n₂ = 35 →
  n₃ = 45 →
  n₄ = 42 →
  m₁ = 50 →
  m₃ = 55 →
  m₄ = 45 →
  overall_avg = 52005494505494504/1000000000000000 →
  ∃ m₂ : ℚ, m₂ = 60 ∧ 
    overall_avg * (n₁ + n₂ + n₃ + n₄ : ℚ) = n₁ * m₁ + n₂ * m₂ + n₃ * m₃ + n₄ * m₄ :=
by sorry


end chemistry_mean_marks_l3741_374182


namespace symmetry_line_equation_l3741_374133

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two points are symmetric about a line -/
def symmetric_about (P Q : Point) (l : Line) : Prop :=
  -- Definition of symmetry (to be implemented)
  sorry

/-- The problem statement -/
theorem symmetry_line_equation :
  let P : Point := ⟨3, 2⟩
  let Q : Point := ⟨1, 4⟩
  let l : Line := ⟨1, -1, 1⟩  -- Represents x - y + 1 = 0
  symmetric_about P Q l → l = ⟨1, -1, 1⟩ := by
  sorry

end symmetry_line_equation_l3741_374133


namespace geometric_sequence_proof_l3741_374157

theorem geometric_sequence_proof (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence condition
  a 1 + a 3 = 10 →                                  -- first given condition
  a 2 + a 4 = 5 →                                   -- second given condition
  ∀ n, a n = 2^(4 - n) :=                           -- conclusion to prove
by
  sorry

end geometric_sequence_proof_l3741_374157


namespace solve_for_a_l3741_374185

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x else x + 1

theorem solve_for_a (a : ℝ) (h : f a + f 1 = 0) : a = -3 := by
  sorry

end solve_for_a_l3741_374185


namespace emily_total_score_l3741_374137

/-- A game with five rounds and specific scoring rules. -/
structure Game where
  round1_score : ℤ
  round2_score : ℤ
  round3_score : ℤ
  round4_score : ℤ
  round5_score : ℤ

/-- Emily's game performance -/
def emily_game : Game where
  round1_score := 16
  round2_score := 32
  round3_score := -27
  round4_score := 46
  round5_score := 12

/-- Calculate the total score for a game -/
def total_score (g : Game) : ℤ :=
  g.round1_score + g.round2_score + g.round3_score + (g.round4_score * 2) + (g.round5_score / 3)

/-- Theorem stating that Emily's total score is 117 -/
theorem emily_total_score : total_score emily_game = 117 := by
  sorry


end emily_total_score_l3741_374137


namespace max_value_of_expression_l3741_374118

theorem max_value_of_expression (x : ℝ) :
  x^4 / (x^8 + 4*x^6 - 8*x^4 + 16*x^2 + 64) ≤ 1/24 ∧
  ∃ y : ℝ, y^4 / (y^8 + 4*y^6 - 8*y^4 + 16*y^2 + 64) = 1/24 :=
by sorry

end max_value_of_expression_l3741_374118


namespace range_of_a_l3741_374105

open Set Real

theorem range_of_a (p q : Prop) (h : p ∧ q) : 
  (∀ x ∈ Icc 1 2, x^2 ≥ a) ∧ (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by sorry

end range_of_a_l3741_374105


namespace intersection_of_A_and_B_l3741_374151

-- Define set A
def A : Set ℝ := {x | |x + 3| + |x - 4| ≤ 9}

-- Define set B
def B : Set ℝ := {x | ∃ t > 0, x = 4*t + 1/t - 6}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | -2 ≤ x ∧ x ≤ 5} := by
  sorry

end intersection_of_A_and_B_l3741_374151


namespace decimal_subtraction_equality_l3741_374170

def repeating_decimal_789 : ℚ := 789 / 999
def repeating_decimal_456 : ℚ := 456 / 999
def repeating_decimal_123 : ℚ := 123 / 999

theorem decimal_subtraction_equality : 
  repeating_decimal_789 - repeating_decimal_456 - repeating_decimal_123 = 70 / 333 := by
  sorry

end decimal_subtraction_equality_l3741_374170


namespace mr_martin_bagels_l3741_374121

/-- Represents the purchase of coffee and bagels -/
structure Purchase where
  coffee : ℕ
  bagels : ℕ
  total : ℚ

/-- Represents the cost of items -/
structure Prices where
  coffee : ℚ
  bagel : ℚ

def mrs_martin : Purchase := { coffee := 3, bagels := 2, total := 12.75 }
def mr_martin (x : ℕ) : Purchase := { coffee := 2, bagels := x, total := 14 }

def prices : Prices := { coffee := 3.25, bagel := 1.5 }

theorem mr_martin_bagels :
  ∃ x : ℕ, 
    (mr_martin x).total = (mr_martin x).coffee • prices.coffee + (mr_martin x).bagels • prices.bagel ∧
    mrs_martin.total = mrs_martin.coffee • prices.coffee + mrs_martin.bagels • prices.bagel ∧
    x = 5 := by
  sorry

#check mr_martin_bagels

end mr_martin_bagels_l3741_374121


namespace honda_second_shift_production_l3741_374141

/-- Represents the production of cars in a Honda factory --/
structure CarProduction where
  second_shift : ℕ
  day_shift : ℕ
  total : ℕ

/-- The conditions of the Honda car production problem --/
def honda_production : CarProduction :=
  { second_shift := 0,  -- placeholder, will be proven
    day_shift := 0,     -- placeholder, will be proven
    total := 5500 }

/-- The theorem stating the solution to the Honda car production problem --/
theorem honda_second_shift_production :
  ∃ (p : CarProduction),
    p.day_shift = 4 * p.second_shift ∧
    p.total = p.day_shift + p.second_shift ∧
    p.total = honda_production.total ∧
    p.second_shift = 1100 := by
  sorry

end honda_second_shift_production_l3741_374141


namespace max_y_value_max_y_achieved_l3741_374187

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = 4) : y ≤ 7 := by
  sorry

theorem max_y_achieved : ∃ x y : ℤ, x * y + 3 * x + 2 * y = 4 ∧ y = 7 := by
  sorry

end max_y_value_max_y_achieved_l3741_374187


namespace remainder_equality_l3741_374177

/-- Represents a natural number as a list of its digits in reverse order -/
def DigitList := List Nat

/-- Converts a natural number to its digit list representation -/
def toDigitList (n : Nat) : DigitList :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : DigitList) : DigitList :=
      if m = 0 then acc
      else aux (m / 10) ((m % 10) :: acc)
    aux n []

/-- Pairs digits from right to left, allowing the leftmost pair to be a single digit -/
def pairDigits (dl : DigitList) : List Nat :=
  match dl with
  | [] => []
  | [x] => [x]
  | x :: y :: rest => (x + 10 * y) :: pairDigits rest

/-- Sums a list of natural numbers -/
def sumList (l : List Nat) : Nat := l.foldl (·+·) 0

/-- The main theorem statement -/
theorem remainder_equality (n : Nat) :
  n % 99 = (sumList (pairDigits (toDigitList n))) % 99 := by
  sorry

end remainder_equality_l3741_374177


namespace polynomial_root_sum_squares_l3741_374100

theorem polynomial_root_sum_squares (a b c t : ℝ) : 
  (∀ x : ℝ, x^3 - 6*x^2 + 8*x - 2 = 0 ↔ (x = a ∨ x = b ∨ x = c)) →
  t = Real.sqrt a + Real.sqrt b + Real.sqrt c →
  t^4 - 12*t^2 - 4*t = -4 := by
sorry

end polynomial_root_sum_squares_l3741_374100


namespace fraction_comparison_l3741_374163

theorem fraction_comparison : -8/21 > -3/7 := by
  sorry

end fraction_comparison_l3741_374163


namespace base_side_length_l3741_374168

/-- Represents a right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ
  lateral_face_area : ℝ

/-- The lateral face area of a right pyramid is half the product of its base side and slant height -/
axiom lateral_face_area_formula (p : RightPyramid) : 
  p.lateral_face_area = (1/2) * p.base_side * p.slant_height

/-- 
Given a right pyramid with a square base, if its lateral face area is 120 square meters 
and its slant height is 24 meters, then the length of a side of its base is 10 meters.
-/
theorem base_side_length (p : RightPyramid) 
  (h1 : p.lateral_face_area = 120) 
  (h2 : p.slant_height = 24) : 
  p.base_side = 10 := by
sorry

end base_side_length_l3741_374168


namespace min_reciprocal_sum_l3741_374192

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (∀ x y, x > 0 → y > 0 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 4 :=
by sorry

end min_reciprocal_sum_l3741_374192


namespace special_isosceles_triangle_base_l3741_374117

/-- An isosceles triangle with a specific configuration of inscribed circles -/
structure SpecialIsoscelesTriangle where
  /-- The radius of the incircle of the triangle -/
  r₁ : ℝ
  /-- The radius of the smaller circle tangent to the incircle and congruent sides -/
  r₂ : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- Condition that r₁ = 3 -/
  h₁ : r₁ = 3
  /-- Condition that r₂ = 2 -/
  h₂ : r₂ = 2

/-- The theorem stating that the base of the special isosceles triangle is 3√6 -/
theorem special_isosceles_triangle_base (t : SpecialIsoscelesTriangle) : t.base = 3 * Real.sqrt 6 := by
  sorry

end special_isosceles_triangle_base_l3741_374117


namespace fraction_simplification_l3741_374159

theorem fraction_simplification (y : ℝ) (h : y = 3) : 
  (y^8 + 18*y^4 + 81) / (y^4 + 9) = 90 := by
sorry

end fraction_simplification_l3741_374159


namespace expression_evaluation_l3741_374197

theorem expression_evaluation (x : ℕ) (h : x = 3) :
  x + x * (x^x) + (x^(x^x)) = 7625597485071 := by
  sorry

end expression_evaluation_l3741_374197


namespace sum_of_e_values_l3741_374169

theorem sum_of_e_values (e : ℝ) : (|2 - e| = 5) → (∃ (e₁ e₂ : ℝ), (|2 - e₁| = 5 ∧ |2 - e₂| = 5 ∧ e₁ + e₂ = 4)) := by
  sorry

end sum_of_e_values_l3741_374169


namespace sandy_total_marks_l3741_374194

/-- Sandy's marking system and attempt results -/
structure SandyAttempt where
  correct_marks : ℕ  -- Marks for each correct sum
  incorrect_penalty : ℕ  -- Marks lost for each incorrect sum
  total_attempts : ℕ  -- Total number of sums attempted
  correct_attempts : ℕ  -- Number of correct sums

/-- Calculate Sandy's total marks -/
def calculate_total_marks (s : SandyAttempt) : ℤ :=
  (s.correct_attempts * s.correct_marks : ℤ) -
  ((s.total_attempts - s.correct_attempts) * s.incorrect_penalty : ℤ)

/-- Theorem stating that Sandy's total marks is 65 -/
theorem sandy_total_marks :
  let s : SandyAttempt := {
    correct_marks := 3,
    incorrect_penalty := 2,
    total_attempts := 30,
    correct_attempts := 25
  }
  calculate_total_marks s = 65 := by
  sorry

end sandy_total_marks_l3741_374194


namespace inequality_proof_l3741_374155

theorem inequality_proof (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a*b ∧ -a*b > b^2 := by
  sorry

end inequality_proof_l3741_374155


namespace free_throw_contest_ratio_l3741_374115

theorem free_throw_contest_ratio (alex sandra hector : ℕ) : 
  alex = 8 →
  hector = 2 * sandra →
  alex + sandra + hector = 80 →
  sandra = 3 * alex :=
by
  sorry

end free_throw_contest_ratio_l3741_374115


namespace cameron_questions_total_l3741_374153

/-- Represents a tour group with a number of regular tourists and inquisitive tourists -/
structure TourGroup where
  regular_tourists : ℕ
  inquisitive_tourists : ℕ

/-- Calculates the number of questions answered for a tour group -/
def questions_answered (group : TourGroup) (questions_per_tourist : ℕ) : ℕ :=
  group.regular_tourists * questions_per_tourist + 
  group.inquisitive_tourists * (questions_per_tourist * 3)

theorem cameron_questions_total : 
  let questions_per_tourist := 2
  let tour1 := TourGroup.mk 6 0
  let tour2 := TourGroup.mk 11 0
  let tour3 := TourGroup.mk 7 1
  let tour4 := TourGroup.mk 7 0
  questions_answered tour1 questions_per_tourist +
  questions_answered tour2 questions_per_tourist +
  questions_answered tour3 questions_per_tourist +
  questions_answered tour4 questions_per_tourist = 68 := by
  sorry

end cameron_questions_total_l3741_374153


namespace jar_capacity_l3741_374193

/-- Proves that the capacity of each jar James needs to buy is 0.5 liters -/
theorem jar_capacity
  (num_hives : ℕ)
  (honey_per_hive : ℝ)
  (num_jars : ℕ)
  (h1 : num_hives = 5)
  (h2 : honey_per_hive = 20)
  (h3 : num_jars = 100)
  : (num_hives * honey_per_hive / 2) / num_jars = 0.5 := by
  sorry

end jar_capacity_l3741_374193


namespace reflection_envelope_is_half_nephroid_l3741_374148

/-- A point on the complex plane -/
def ComplexPoint := ℂ

/-- A line in the complex plane -/
def Line := ComplexPoint → Prop

/-- The unit circle centered at the origin -/
def UnitCircle : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- A bundle of parallel rays -/
def ParallelRays : Set Line := sorry

/-- The reflection of a ray off the unit circle -/
def ReflectedRay (ray : Line) : Line := sorry

/-- The envelope of a family of lines -/
def Envelope (family : Set Line) : Set ComplexPoint := sorry

/-- Half of a nephroid -/
def HalfNephroid : Set ComplexPoint := sorry

/-- The theorem statement -/
theorem reflection_envelope_is_half_nephroid :
  Envelope (ReflectedRay '' ParallelRays) = HalfNephroid := by sorry

end reflection_envelope_is_half_nephroid_l3741_374148


namespace chocolate_doughnut_students_correct_l3741_374165

/-- The number of students wanting chocolate doughnuts given the conditions -/
def chocolate_doughnut_students : ℕ :=
  let total_students : ℕ := 25
  let chocolate_cost : ℕ := 2
  let glazed_cost : ℕ := 1
  let total_cost : ℕ := 35
  -- The number of students wanting chocolate doughnuts
  10

/-- Theorem stating that the number of students wanting chocolate doughnuts is correct -/
theorem chocolate_doughnut_students_correct :
  let c := chocolate_doughnut_students
  let g := 25 - c
  c + g = 25 ∧ 2 * c + g = 35 := by sorry

end chocolate_doughnut_students_correct_l3741_374165


namespace parabola_properties_l3741_374162

/-- A parabola with specific properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_a_pos : a > 0
  h_axis : b = 2 * a
  h_intercept : a * m^2 + b * m + c = 0
  h_m_bounds : 0 < m ∧ m < 1

/-- Theorem stating properties of the parabola -/
theorem parabola_properties (p : Parabola) :
  (4 * p.a + p.c > 0) ∧
  (∀ t : ℝ, p.a - p.b * t ≤ p.a * t^2 + p.b) :=
sorry

end parabola_properties_l3741_374162


namespace clarissa_photos_eq_14_l3741_374124

/-- The number of slots in the photo album -/
def total_slots : ℕ := 40

/-- The number of photos Cristina brings -/
def cristina_photos : ℕ := 7

/-- The number of photos John brings -/
def john_photos : ℕ := 10

/-- The number of photos Sarah brings -/
def sarah_photos : ℕ := 9

/-- The number of photos Clarissa needs to bring -/
def clarissa_photos : ℕ := total_slots - (cristina_photos + john_photos + sarah_photos)

theorem clarissa_photos_eq_14 : clarissa_photos = 14 := by
  sorry

end clarissa_photos_eq_14_l3741_374124


namespace cat_weight_problem_l3741_374160

theorem cat_weight_problem (total_weight : ℕ) (cat1_weight : ℕ) (cat2_weight : ℕ) 
  (h1 : total_weight = 13)
  (h2 : cat1_weight = 2)
  (h3 : cat2_weight = 7) :
  total_weight - cat1_weight - cat2_weight = 4 := by
  sorry

end cat_weight_problem_l3741_374160


namespace x_less_than_negative_one_l3741_374109

theorem x_less_than_negative_one (a b : ℚ) 
  (ha : -2 < a ∧ a < -1.5) 
  (hb : 0.5 < b ∧ b < 1) : 
  let x := (a - 5*b) / (a + 5*b)
  x < -1 := by
sorry

end x_less_than_negative_one_l3741_374109


namespace triangle_proof_l3741_374126

noncomputable section

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given equation for the triangle -/
def triangle_equation (t : Triangle) : Prop :=
  t.b^2 - (2 * Real.sqrt 3 / 3) * t.b * t.c * Real.sin t.A + t.c^2 = t.a^2

theorem triangle_proof (t : Triangle) 
  (h_eq : triangle_equation t) 
  (h_b : t.b = 2) 
  (h_c : t.c = 3) :
  t.A = π/3 ∧ t.a = Real.sqrt 7 ∧ Real.sin (2*t.B - t.A) = 3*Real.sqrt 3/14 := by
  sorry

end

end triangle_proof_l3741_374126


namespace number_of_female_employees_l3741_374103

/-- Given a company with employees, prove that the number of female employees is 500 -/
theorem number_of_female_employees 
  (E : ℕ) -- Total number of employees
  (F : ℕ) -- Number of female employees
  (M : ℕ) -- Number of male employees
  (h1 : E = F + M) -- Total employees is sum of female and male employees
  (h2 : (2 : ℚ) / 5 * E = 200 + (2 : ℚ) / 5 * M) -- Equation for total managers
  (h3 : (200 : ℚ) = F - (2 : ℚ) / 5 * M) -- Equation for female managers
  : F = 500 := by
  sorry

end number_of_female_employees_l3741_374103


namespace tetrahedron_non_coplanar_choices_l3741_374183

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertices : Fin 4 → Point3D
  midpoints : Fin 6 → Point3D

/-- Checks if four points are coplanar -/
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- The set of all points (vertices and midpoints) of a tetrahedron -/
def tetrahedron_points (t : Tetrahedron) : Finset Point3D := sorry

/-- The number of ways to choose 4 non-coplanar points from a tetrahedron's points -/
def non_coplanar_choices (t : Tetrahedron) : ℕ := sorry

theorem tetrahedron_non_coplanar_choices :
  ∀ t : Tetrahedron, non_coplanar_choices t = 141 := sorry

end tetrahedron_non_coplanar_choices_l3741_374183


namespace sum_of_A_and_B_l3741_374112

theorem sum_of_A_and_B (A B : ℕ) (h1 : 3 * 7 = 7 * A) (h2 : 7 * A = B) : A + B = 24 := by
  sorry

end sum_of_A_and_B_l3741_374112


namespace quadratic_inequality_range_l3741_374161

theorem quadratic_inequality_range : 
  ∀ x : ℝ, x^2 - 7*x + 12 < 0 → 
  ∃ y : ℝ, y ∈ Set.Ioo 42 56 ∧ y = x^2 + 7*x + 12 :=
by sorry

end quadratic_inequality_range_l3741_374161


namespace johann_mail_delivery_l3741_374128

theorem johann_mail_delivery (total_mail : ℕ) (friend_mail : ℕ) (num_friends : ℕ) :
  total_mail = 180 →
  friend_mail = 41 →
  num_friends = 2 →
  total_mail - (friend_mail * num_friends) = 98 := by
  sorry

end johann_mail_delivery_l3741_374128


namespace gcf_of_40_120_45_l3741_374149

theorem gcf_of_40_120_45 : Nat.gcd 40 (Nat.gcd 120 45) = 5 := by
  sorry

end gcf_of_40_120_45_l3741_374149


namespace probability_ratio_l3741_374189

-- Define the total number of slips
def total_slips : ℕ := 30

-- Define the number of different numbers on the slips
def num_options : ℕ := 6

-- Define the number of slips for each number
def slips_per_number : ℕ := 5

-- Define the number of slips drawn
def drawn_slips : ℕ := 4

-- Define the probability of drawing four slips with the same number
def p : ℚ := (num_options * slips_per_number) / Nat.choose total_slips drawn_slips

-- Define the probability of drawing two pairs of slips with different numbers
def q : ℚ := (Nat.choose num_options 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 2) / Nat.choose total_slips drawn_slips

-- Theorem statement
theorem probability_ratio : q / p = 50 := by sorry

end probability_ratio_l3741_374189


namespace arithmetic_computation_l3741_374104

theorem arithmetic_computation : 6^2 + 2*(5) - 4^2 = 30 := by
  sorry

end arithmetic_computation_l3741_374104


namespace smallest_fraction_greater_than_target_l3741_374129

/-- A fraction with two-digit numerator and denominator -/
structure TwoDigitFraction :=
  (numerator : Nat)
  (denominator : Nat)
  (num_two_digit : 10 ≤ numerator ∧ numerator ≤ 99)
  (den_two_digit : 10 ≤ denominator ∧ denominator ≤ 99)

/-- The fraction 4/9 -/
def target : Rat := 4 / 9

/-- The fraction 41/92 -/
def smallest : TwoDigitFraction :=
  { numerator := 41
  , denominator := 92
  , num_two_digit := by sorry
  , den_two_digit := by sorry }

theorem smallest_fraction_greater_than_target :
  (smallest.numerator : Rat) / smallest.denominator > target ∧
  ∀ (f : TwoDigitFraction), 
    (f.numerator : Rat) / f.denominator > target → 
    (smallest.numerator : Rat) / smallest.denominator ≤ (f.numerator : Rat) / f.denominator :=
by sorry

end smallest_fraction_greater_than_target_l3741_374129


namespace xenia_earnings_and_wage_l3741_374143

/-- Xenia's work schedule and earnings --/
structure WorkSchedule where
  week1_hours : ℝ
  week2_hours : ℝ
  week3_hours : ℝ
  week2_extra_earnings : ℝ
  week3_bonus : ℝ

/-- Calculate Xenia's total earnings and hourly wage --/
def calculate_earnings_and_wage (schedule : WorkSchedule) : ℝ × ℝ := by
  sorry

/-- Theorem stating Xenia's total earnings and hourly wage --/
theorem xenia_earnings_and_wage (schedule : WorkSchedule)
  (h1 : schedule.week1_hours = 18)
  (h2 : schedule.week2_hours = 25)
  (h3 : schedule.week3_hours = 28)
  (h4 : schedule.week2_extra_earnings = 60)
  (h5 : schedule.week3_bonus = 30) :
  let (total_earnings, hourly_wage) := calculate_earnings_and_wage schedule
  total_earnings = 639.47 ∧ hourly_wage = 8.57 := by
  sorry

end xenia_earnings_and_wage_l3741_374143


namespace flowers_per_bouquet_l3741_374120

theorem flowers_per_bouquet 
  (total_flowers : ℕ) 
  (wilted_flowers : ℕ) 
  (num_bouquets : ℕ) 
  (h1 : total_flowers = 45) 
  (h2 : wilted_flowers = 35) 
  (h3 : num_bouquets = 2) : 
  (total_flowers - wilted_flowers) / num_bouquets = 5 := by
sorry

end flowers_per_bouquet_l3741_374120


namespace min_disks_for_files_l3741_374125

theorem min_disks_for_files : 
  let total_files : ℕ := 35
  let disk_capacity : ℚ := 1.44
  let files_0_6MB : ℕ := 5
  let files_0_5MB : ℕ := 18
  let files_0_3MB : ℕ := total_files - files_0_6MB - files_0_5MB
  let size_0_6MB : ℚ := 0.6
  let size_0_5MB : ℚ := 0.5
  let size_0_3MB : ℚ := 0.3
  ∀ n : ℕ, 
    (n * disk_capacity ≥ 
      files_0_6MB * size_0_6MB + 
      files_0_5MB * size_0_5MB + 
      files_0_3MB * size_0_3MB) →
    n ≥ 12 :=
by sorry

end min_disks_for_files_l3741_374125


namespace ac_squared_gt_bc_squared_sufficient_not_necessary_l3741_374156

theorem ac_squared_gt_bc_squared_sufficient_not_necessary
  (a b c : ℝ) (h : c ≠ 0) :
  (∀ a b, a * c^2 > b * c^2 → a > b) ∧
  ¬(∀ a b, a > b → a * c^2 > b * c^2) :=
sorry

end ac_squared_gt_bc_squared_sufficient_not_necessary_l3741_374156


namespace binomial_coefficient_equality_l3741_374119

theorem binomial_coefficient_equality (n : ℕ) : 
  (Nat.choose 12 n = Nat.choose 12 (2*n - 3)) → (n = 3 ∨ n = 5) := by
  sorry

end binomial_coefficient_equality_l3741_374119


namespace arithmetic_sequence_property_l3741_374179

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) 
  (h2 : a 1 = 2) 
  (h3 : a 3 + a 5 = 10) : 
  a 7 = 8 := by
  sorry

end arithmetic_sequence_property_l3741_374179


namespace power_product_equals_sum_of_exponents_l3741_374107

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by sorry

end power_product_equals_sum_of_exponents_l3741_374107


namespace union_M_N_l3741_374114

-- Define the universe set U
def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N

-- Theorem statement
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end union_M_N_l3741_374114


namespace smaller_angle_at_5_oclock_l3741_374147

/-- The number of hour marks on a clock. -/
def num_hours : ℕ := 12

/-- The number of degrees in a full circle. -/
def full_circle : ℕ := 360

/-- The time in hours. -/
def time : ℕ := 5

/-- The angle between adjacent hour marks on a clock. -/
def angle_per_hour : ℕ := full_circle / num_hours

/-- The angle between the hour hand and 12 o'clock position at the given time. -/
def hour_hand_angle : ℕ := time * angle_per_hour

/-- The smaller angle between the hour hand and minute hand at 5 o'clock. -/
theorem smaller_angle_at_5_oclock : hour_hand_angle = 150 := by
  sorry

end smaller_angle_at_5_oclock_l3741_374147


namespace route_number_theorem_l3741_374146

/-- Represents a digit on a seven-segment display -/
inductive Digit
| Zero | One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- Represents a three-digit number -/
structure ThreeDigitNumber :=
  (hundreds : Digit)
  (tens : Digit)
  (units : Digit)

/-- Converts a ThreeDigitNumber to a natural number -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  match n.hundreds, n.tens, n.units with
  | Digit.Three, Digit.Five, Digit.One => 351
  | Digit.Three, Digit.Five, Digit.Four => 354
  | Digit.Three, Digit.Five, Digit.Seven => 357
  | Digit.Three, Digit.Six, Digit.One => 361
  | Digit.Three, Digit.Six, Digit.Seven => 367
  | Digit.Three, Digit.Eight, Digit.One => 381
  | Digit.Three, Digit.Nine, Digit.One => 391
  | Digit.Three, Digit.Nine, Digit.Seven => 397
  | Digit.Eight, Digit.Five, Digit.One => 851
  | Digit.Nine, Digit.Five, Digit.One => 951
  | Digit.Nine, Digit.Five, Digit.Seven => 957
  | Digit.Nine, Digit.Six, Digit.One => 961
  | Digit.Nine, Digit.Nine, Digit.One => 991
  | _, _, _ => 0  -- Default case, should not occur in our problem

/-- The set of possible route numbers -/
def possibleRouteNumbers : Set Nat :=
  {351, 354, 357, 361, 367, 381, 391, 397, 851, 951, 957, 961, 991}

/-- The theorem stating that the displayed number 351 with two non-working segments
    can only result in the numbers in the possibleRouteNumbers set -/
theorem route_number_theorem (n : ThreeDigitNumber) 
    (h : n.toNat ∈ possibleRouteNumbers) : 
    ∃ (broken_segments : Nat), broken_segments ≤ 2 ∧ 
    n.toNat ∈ possibleRouteNumbers :=
  sorry

end route_number_theorem_l3741_374146


namespace basswood_figurines_count_l3741_374171

/-- The number of figurines that can be created from a block of basswood -/
def basswood_figurines : ℕ := 3

/-- The number of figurines that can be created from a block of butternut wood -/
def butternut_figurines : ℕ := 4

/-- The number of figurines that can be created from a block of Aspen wood -/
def aspen_figurines : ℕ := 2 * basswood_figurines

/-- The total number of figurines Adam can make -/
def total_figurines : ℕ := 245

/-- The number of basswood blocks Adam has -/
def basswood_blocks : ℕ := 15

/-- The number of butternut wood blocks Adam has -/
def butternut_blocks : ℕ := 20

/-- The number of Aspen wood blocks Adam has -/
def aspen_blocks : ℕ := 20

theorem basswood_figurines_count : 
  basswood_blocks * basswood_figurines + 
  butternut_blocks * butternut_figurines + 
  aspen_blocks * aspen_figurines = total_figurines :=
by sorry

end basswood_figurines_count_l3741_374171


namespace total_money_left_l3741_374184

def monthly_income : ℕ := 1000

def savings_june : ℕ := monthly_income * 25 / 100
def savings_july : ℕ := monthly_income * 20 / 100
def savings_august : ℕ := monthly_income * 30 / 100

def expenses_june : ℕ := 200 + monthly_income * 5 / 100
def expenses_july : ℕ := 250 + monthly_income * 15 / 100
def expenses_august : ℕ := 300 + monthly_income * 10 / 100

def gift_august : ℕ := 50

def money_left_june : ℕ := monthly_income - savings_june - expenses_june
def money_left_july : ℕ := monthly_income - savings_july - expenses_july
def money_left_august : ℕ := monthly_income - savings_august - expenses_august + gift_august

theorem total_money_left : 
  money_left_june + money_left_july + money_left_august = 1250 := by
  sorry

end total_money_left_l3741_374184


namespace closed_equinumerous_to_halfopen_l3741_374152

-- Define the closed interval [0,1]
def closedInterval : Set ℝ := Set.Icc 0 1

-- Define the half-open interval [0,1)
def halfOpenInterval : Set ℝ := Set.Ico 0 1

-- Statement: There exists a bijective function from [0,1] to [0,1)
theorem closed_equinumerous_to_halfopen :
  ∃ f : closedInterval → halfOpenInterval, Function.Bijective f := by
  sorry

end closed_equinumerous_to_halfopen_l3741_374152


namespace sequence_integer_condition_l3741_374196

def sequence_condition (x : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → x n = (x (n - 2) * x (n - 1)) / (2 * x (n - 2) - x (n - 1))

def infinitely_many_integers (x : ℕ → ℝ) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ ∃ k : ℤ, x n = k

theorem sequence_integer_condition (x : ℕ → ℝ) :
  (∀ n : ℕ, x n ≠ 0) →
  sequence_condition x →
  (infinitely_many_integers x ↔ ∃ k : ℤ, k ≠ 0 ∧ x 1 = k ∧ x 2 = k) :=
sorry

end sequence_integer_condition_l3741_374196


namespace intersection_of_A_and_B_l3741_374167

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 3) * (x + 1) ≤ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | 2 * x > 2}

-- Theorem statement
theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | 1 < x ∧ x ≤ 3} := by sorry

end intersection_of_A_and_B_l3741_374167


namespace max_b_is_maximum_l3741_374188

def is_lattice_point (x y : ℤ) : Prop := true

def line_equation (m : ℚ) (x : ℤ) : ℚ := m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x y : ℤ, 0 < x → x ≤ 50 → is_lattice_point x y → line_equation m x ≠ y

def max_b : ℚ := 11/51

theorem max_b_is_maximum :
  (∀ m : ℚ, 2/5 < m → m < max_b → no_lattice_points m) ∧
  ∀ b : ℚ, b > max_b → ∃ m : ℚ, 2/5 < m ∧ m < b ∧ ¬(no_lattice_points m) :=
sorry

end max_b_is_maximum_l3741_374188
