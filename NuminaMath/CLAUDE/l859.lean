import Mathlib

namespace regular_polygon_sides_l859_85904

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 24 → n * exterior_angle = 360 → n = 15 := by
  sorry

end regular_polygon_sides_l859_85904


namespace dispersion_measure_properties_l859_85937

-- Define a type for datasets
structure Dataset where
  data : List ℝ

-- Define a type for dispersion measures
structure DispersionMeasure where
  measure : Dataset → ℝ

-- Statement 1: Multiple values can be used to describe the degree of dispersion
def multipleValuesUsed (d : DispersionMeasure) : Prop :=
  ∃ (d1 d2 : DispersionMeasure), d1 ≠ d2

-- Statement 2: One should make full use of the obtained data
def fullDataUsed (d : DispersionMeasure) : Prop :=
  ∀ (dataset : Dataset), d.measure dataset = d.measure dataset

-- Statement 3: For different datasets, when the degree of dispersion is large, this value should be smaller (incorrect statement)
def incorrectDispersionRelation (d : DispersionMeasure) : Prop :=
  ∃ (dataset1 dataset2 : Dataset),
    d.measure dataset1 > d.measure dataset2 →
    d.measure dataset1 < d.measure dataset2

theorem dispersion_measure_properties :
  ∃ (d : DispersionMeasure),
    multipleValuesUsed d ∧
    fullDataUsed d ∧
    ¬incorrectDispersionRelation d :=
  sorry

end dispersion_measure_properties_l859_85937


namespace sum_of_three_numbers_l859_85951

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 267) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 23 := by
  sorry

end sum_of_three_numbers_l859_85951


namespace mr_martin_purchase_cost_l859_85973

/-- The cost of Mrs. Martin's purchase -/
def mrs_martin_cost : ℝ := 12.75

/-- The number of coffee cups Mrs. Martin bought -/
def mrs_martin_coffee : ℕ := 3

/-- The number of bagels Mrs. Martin bought -/
def mrs_martin_bagels : ℕ := 2

/-- The cost of one bagel -/
def bagel_cost : ℝ := 1.5

/-- The number of coffee cups Mr. Martin bought -/
def mr_martin_coffee : ℕ := 2

/-- The number of bagels Mr. Martin bought -/
def mr_martin_bagels : ℕ := 5

/-- Theorem stating that Mr. Martin's purchase costs $14.00 -/
theorem mr_martin_purchase_cost : 
  ∃ (coffee_cost : ℝ), 
    mrs_martin_cost = mrs_martin_coffee * coffee_cost + mrs_martin_bagels * bagel_cost ∧
    mr_martin_coffee * coffee_cost + mr_martin_bagels * bagel_cost = 14 :=
by sorry

end mr_martin_purchase_cost_l859_85973


namespace compute_expression_l859_85915

theorem compute_expression : 3 * ((25 + 15)^2 - (25 - 15)^2) = 4500 := by
  sorry

end compute_expression_l859_85915


namespace laps_run_l859_85920

/-- Proves the number of laps run in a track given total distance, lap length, and remaining laps --/
theorem laps_run (total_distance : ℕ) (lap_length : ℕ) (remaining_laps : ℕ) 
  (h1 : total_distance = 2400)
  (h2 : lap_length = 150)
  (h3 : remaining_laps = 4) :
  (total_distance / lap_length) - remaining_laps = 12 := by
  sorry

#check laps_run

end laps_run_l859_85920


namespace product_ratio_l859_85958

def range_start : Int := -2020
def range_end : Int := 2019

theorem product_ratio :
  let smallest_product := range_start * (range_start + 1) * (range_start + 2)
  let largest_product := (range_end - 2) * (range_end - 1) * range_end
  (smallest_product : ℚ) / largest_product = -2020 / 2017 := by
sorry

end product_ratio_l859_85958


namespace anna_scores_proof_l859_85984

def anna_scores : List ℕ := [94, 87, 86, 78, 71, 58]

theorem anna_scores_proof :
  -- The list has 6 elements
  anna_scores.length = 6 ∧
  -- All elements are less than 95
  (∀ x ∈ anna_scores, x < 95) ∧
  -- All elements are different
  anna_scores.Nodup ∧
  -- The list is sorted in descending order
  anna_scores.Sorted (· ≥ ·) ∧
  -- The first three scores are 86, 78, and 71
  [86, 78, 71].Sublist anna_scores ∧
  -- The mean of all scores is 79
  anna_scores.sum / anna_scores.length = 79 := by
sorry

end anna_scores_proof_l859_85984


namespace evaluate_expression_l859_85927

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4)
  (hy : y = 1/3)
  (hz : z = -12)
  (hw : w = 5) :
  x^2 * y^3 * z + w = 179/36 := by
  sorry

end evaluate_expression_l859_85927


namespace ball_attendees_l859_85968

theorem ball_attendees :
  ∀ n m : ℕ,
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end ball_attendees_l859_85968


namespace geometric_sequence_problem_l859_85907

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 1 + a 2 = 4/9 →  -- First condition
  a 3 + a 4 + a 5 + a 6 = 40 →  -- Second condition
  (a 7 + a 8 + a 9) / 9 = 117 := by
sorry

end geometric_sequence_problem_l859_85907


namespace nicky_trade_profit_l859_85916

/-- Calculates Nicky's profit or loss in a baseball card trade with Jill --/
theorem nicky_trade_profit :
  let nicky_card1_value : ℚ := 8
  let nicky_card1_count : ℕ := 2
  let nicky_card2_value : ℚ := 5
  let nicky_card2_count : ℕ := 3
  let jill_card1_value_cad : ℚ := 21
  let jill_card1_count : ℕ := 1
  let jill_card2_value_cad : ℚ := 6
  let jill_card2_count : ℕ := 2
  let exchange_rate_usd_per_cad : ℚ := 0.8
  let tax_rate : ℚ := 0.05

  let nicky_total_value := nicky_card1_value * nicky_card1_count + nicky_card2_value * nicky_card2_count
  let jill_total_value_cad := jill_card1_value_cad * jill_card1_count + jill_card2_value_cad * jill_card2_count
  let jill_total_value_usd := jill_total_value_cad * exchange_rate_usd_per_cad
  let total_trade_value_usd := nicky_total_value + jill_total_value_usd
  let tax_amount := total_trade_value_usd * tax_rate
  let nicky_profit := jill_total_value_usd - (nicky_total_value + tax_amount)

  nicky_profit = -7.47 := by sorry

end nicky_trade_profit_l859_85916


namespace total_games_cost_is_13800_l859_85961

/-- Calculates the total cost of games owned by Katie and her friends -/
def totalGamesCost (katieGames : ℕ) (newFriends oldFriends : ℕ) (newFriendGames oldFriendGames : ℕ) (costPerGame : ℕ) : ℕ :=
  let totalGames := katieGames + newFriends * newFriendGames + oldFriends * oldFriendGames
  totalGames * costPerGame

/-- Theorem stating that the total cost of games is $13,800 -/
theorem total_games_cost_is_13800 :
  totalGamesCost 91 5 3 88 53 20 = 13800 := by
  sorry

#eval totalGamesCost 91 5 3 88 53 20

end total_games_cost_is_13800_l859_85961


namespace initial_speed_calculation_l859_85979

/-- Proves that the initial speed satisfies the given equation under the problem conditions -/
theorem initial_speed_calculation (D T : ℝ) (hD : D > 0) (hT : T > 0) 
  (h_time_constraint : T/3 + (D/3) / 25 = T) : ∃ S : ℝ, S = 2*D/T ∧ S = 100 := by
  sorry

end initial_speed_calculation_l859_85979


namespace bookshop_online_sales_l859_85903

theorem bookshop_online_sales (initial_books : ℕ) (saturday_instore : ℕ) (sunday_instore : ℕ)
  (sunday_online_increase : ℕ) (shipment : ℕ) (final_books : ℕ) :
  initial_books = 743 →
  saturday_instore = 37 →
  sunday_instore = 2 * saturday_instore →
  sunday_online_increase = 34 →
  shipment = 160 →
  final_books = 502 →
  ∃ (saturday_online : ℕ),
    final_books = initial_books - saturday_instore - saturday_online -
      sunday_instore - (saturday_online + sunday_online_increase) + shipment ∧
    saturday_online = 128 := by
  sorry

end bookshop_online_sales_l859_85903


namespace horseshoe_profit_is_5000_l859_85946

/-- Calculates the profit for a horseshoe manufacturing company --/
def horseshoe_profit (initial_outlay : ℕ) (cost_per_set : ℕ) (price_per_set : ℕ) (num_sets : ℕ) : ℤ :=
  (price_per_set * num_sets : ℤ) - (initial_outlay + cost_per_set * num_sets : ℤ)

/-- Proves that the profit for the given conditions is $5,000 --/
theorem horseshoe_profit_is_5000 :
  horseshoe_profit 10000 20 50 500 = 5000 := by
  sorry

#eval horseshoe_profit 10000 20 50 500

end horseshoe_profit_is_5000_l859_85946


namespace min_value_not_e_squared_minus_2m_l859_85952

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.exp x - (m / 2) * x^2 - m * x

theorem min_value_not_e_squared_minus_2m (m : ℝ) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 2 ∧ f m x < Real.exp 2 - 2 * m :=
sorry

end min_value_not_e_squared_minus_2m_l859_85952


namespace chocolate_bar_expense_l859_85932

def chocolate_bar_cost : ℚ := 3/2  -- $1.50 represented as a rational number
def smores_per_bar : ℕ := 3
def num_scouts : ℕ := 15
def smores_per_scout : ℕ := 2

theorem chocolate_bar_expense : 
  ↑num_scouts * ↑smores_per_scout / ↑smores_per_bar * chocolate_bar_cost = 15 := by
  sorry

end chocolate_bar_expense_l859_85932


namespace solution_set_implies_m_range_l859_85964

theorem solution_set_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - m| > 4) → (m > 3 ∨ m < -5) :=
by sorry

end solution_set_implies_m_range_l859_85964


namespace car_journey_time_l859_85970

theorem car_journey_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 450)
  (h2 : new_speed = 50)
  (h3 : time_ratio = 3/2)
  (h4 : distance = new_speed * (time_ratio * initial_time)) :
  initial_time = 6 := by
  sorry

end car_journey_time_l859_85970


namespace complex_expression_simplification_l859_85977

theorem complex_expression_simplification (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
sorry

end complex_expression_simplification_l859_85977


namespace f_monotonicity_and_extrema_l859_85959

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 - 3*x + 1

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -3 → f x < f y) ∧
  (∀ x y, -3 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  (∀ x y, 1 < x ∧ x < y → f x < f y) ∧
  (∀ x, f x ≤ 10) ∧
  (∀ x, f x ≥ -2/3) ∧
  (∃ x, f x = 10) ∧
  (∃ x, f x = -2/3) :=
sorry

end f_monotonicity_and_extrema_l859_85959


namespace sum_of_last_two_digits_of_8_pow_351_l859_85963

theorem sum_of_last_two_digits_of_8_pow_351 :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ (8^351) % 100 = 10 * a + b ∧ a + b = 11 := by
  sorry

end sum_of_last_two_digits_of_8_pow_351_l859_85963


namespace happyTails_cats_count_l859_85901

/-- Represents the number of cats that can perform a specific combination of tricks -/
structure CatTricks where
  jump : Nat
  fetch : Nat
  spin : Nat
  jumpFetch : Nat
  fetchSpin : Nat
  jumpSpin : Nat
  allThree : Nat
  none : Nat

/-- Calculates the total number of cats at HappyTails Training Center -/
def totalCats (ct : CatTricks) : Nat :=
  ct.jump + ct.fetch + ct.spin - ct.jumpFetch - ct.fetchSpin - ct.jumpSpin + ct.allThree + ct.none

/-- Theorem stating that the total number of cats at HappyTails Training Center is 70 -/
theorem happyTails_cats_count (ct : CatTricks)
  (h1 : ct.jump = 40)
  (h2 : ct.fetch = 25)
  (h3 : ct.spin = 30)
  (h4 : ct.jumpFetch = 15)
  (h5 : ct.fetchSpin = 10)
  (h6 : ct.jumpSpin = 12)
  (h7 : ct.allThree = 5)
  (h8 : ct.none = 7) :
  totalCats ct = 70 := by
  sorry

end happyTails_cats_count_l859_85901


namespace adjacent_sum_of_six_l859_85945

/-- Represents a 3x3 table filled with numbers 1 to 9 --/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Returns the list of adjacent positions for a given position --/
def adjacent_positions (row col : Fin 3) : List (Fin 3 × Fin 3) := sorry

/-- Returns the sum of adjacent numbers for a given number in the table --/
def adjacent_sum (t : Table) (n : Fin 9) : ℕ := sorry

/-- Checks if the table satisfies the given conditions --/
def valid_table (t : Table) : Prop :=
  (t 0 0 = 0) ∧ (t 2 0 = 1) ∧ (t 0 2 = 2) ∧ (t 2 2 = 3) ∧
  (adjacent_sum t 4 = 9) ∧
  (∀ i j : Fin 3, ∀ k : Fin 9, (t i j = k) → (∀ i' j' : Fin 3, (i ≠ i' ∨ j ≠ j') → t i' j' ≠ k))

theorem adjacent_sum_of_six (t : Table) (h : valid_table t) : adjacent_sum t 5 = 29 := by
  sorry

end adjacent_sum_of_six_l859_85945


namespace complex_equation_solution_l859_85944

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop :=
  z * i = ((i + 1) / (i - 1)) ^ 2016

-- Theorem statement
theorem complex_equation_solution :
  ∃ (z : ℂ), equation z ∧ z = -i :=
sorry

end complex_equation_solution_l859_85944


namespace constant_remainder_iff_a_eq_neg_seven_l859_85930

/-- The dividend polynomial -/
def dividend (a : ℝ) (x : ℝ) : ℝ := 10 * x^3 - 7 * x^2 + a * x + 6

/-- The divisor polynomial -/
def divisor (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- The remainder of the polynomial division -/
def remainder (a : ℝ) (x : ℝ) : ℝ := (a + 7) * x + 2

theorem constant_remainder_iff_a_eq_neg_seven :
  ∀ a : ℝ, (∀ x : ℝ, ∃ q : ℝ, dividend a x = divisor x * q + remainder a x) ↔ a = -7 := by
  sorry

end constant_remainder_iff_a_eq_neg_seven_l859_85930


namespace vector_minimization_and_angle_l859_85966

/-- Given vectors OP, OA, OB, and a point C on line OP, prove that OC minimizes CA · CB and calculate cos ∠ACB -/
theorem vector_minimization_and_angle (O P A B C : ℝ × ℝ) : 
  O = (0, 0) →
  P = (2, 1) →
  A = (1, 7) →
  B = (5, 1) →
  (∃ t : ℝ, C = (t * 2, t * 1)) →
  (∀ D : ℝ × ℝ, (∃ s : ℝ, D = (s * 2, s * 1)) → 
    (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) ≤ 
    (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2)) →
  C = (4, 2) ∧ 
  (((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / 
   (((A.1 - C.1)^2 + (A.2 - C.2)^2) * ((B.1 - C.1)^2 + (B.2 - C.2)^2))^(1/2) = -4 * 17^(1/2) / 17) :=
by sorry


end vector_minimization_and_angle_l859_85966


namespace purely_imaginary_value_l859_85918

-- Define a complex number z as a function of real number m
def z (m : ℝ) : ℂ := m + 2 + (m - 1) * Complex.I

-- State the theorem
theorem purely_imaginary_value (m : ℝ) : 
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = -2 := by
  sorry

end purely_imaginary_value_l859_85918


namespace base_number_proof_l859_85999

theorem base_number_proof (x : ℝ) (n : ℕ) 
  (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^28) 
  (h2 : n = 27) : 
  x = 2 := by
  sorry

end base_number_proof_l859_85999


namespace wrapping_and_ribbons_fractions_l859_85975

/-- Given a roll of wrapping paper, prove the fractions used for wrapping and ribbons on each present -/
theorem wrapping_and_ribbons_fractions
  (total_wrap : ℚ) -- Total fraction of roll used for wrapping
  (total_ribbon : ℚ) -- Total fraction of roll used for ribbons
  (num_presents : ℕ) -- Number of presents
  (h1 : total_wrap = 2/5) -- Condition: 2/5 of roll used for wrapping
  (h2 : total_ribbon = 1/5) -- Condition: 1/5 of roll used for ribbons
  (h3 : num_presents = 5) -- Condition: 5 presents
  : (total_wrap / num_presents = 2/25) ∧ (total_ribbon / num_presents = 1/25) := by
  sorry


end wrapping_and_ribbons_fractions_l859_85975


namespace total_selling_price_cloth_l859_85940

/-- Calculate the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
theorem total_selling_price_cloth (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) :
  quantity = 85 →
  profit_per_meter = 15 →
  cost_price_per_meter = 85 →
  quantity * (cost_price_per_meter + profit_per_meter) = 8500 := by
  sorry

end total_selling_price_cloth_l859_85940


namespace cube_side_length_is_one_l859_85983

/-- The surface area of a cuboid formed by joining two cubes with side length s is 10 -/
def cuboid_surface_area (s : ℝ) : ℝ := 10 * s^2

/-- Theorem: If two cubes with side length s are joined to form a cuboid with surface area 10, then s = 1 -/
theorem cube_side_length_is_one :
  ∃ (s : ℝ), s > 0 ∧ cuboid_surface_area s = 10 → s = 1 :=
by sorry

end cube_side_length_is_one_l859_85983


namespace min_ABFG_value_l859_85925

/-- Represents a seven-digit number ABCDEFG -/
structure SevenDigitNumber where
  digits : Fin 7 → Nat
  is_valid : ∀ i, digits i < 10

/-- Extracts a five-digit number from a seven-digit number -/
def extract_five_digits (n : SevenDigitNumber) (start : Fin 3) : Nat :=
  (n.digits start) * 10000 + (n.digits (start + 1)) * 1000 + 
  (n.digits (start + 2)) * 100 + (n.digits (start + 3)) * 10 + 
  (n.digits (start + 4))

/-- Extracts a four-digit number ABFG from a seven-digit number ABCDEFG -/
def extract_ABFG (n : SevenDigitNumber) : Nat :=
  (n.digits 0) * 1000 + (n.digits 1) * 100 + (n.digits 5) * 10 + (n.digits 6)

/-- The main theorem -/
theorem min_ABFG_value (n : SevenDigitNumber) 
  (h1 : extract_five_digits n 1 % 2013 = 0)
  (h2 : extract_five_digits n 3 % 1221 = 0) :
  3036 ≤ extract_ABFG n :=
sorry

end min_ABFG_value_l859_85925


namespace child_grandmother_weight_ratio_l859_85950

def family_weights (total_weight daughter_weight daughter_child_weight : ℝ) : Prop :=
  total_weight = 120 ∧ daughter_child_weight = 60 ∧ daughter_weight = 48

theorem child_grandmother_weight_ratio 
  (total_weight daughter_weight daughter_child_weight : ℝ) 
  (h : family_weights total_weight daughter_weight daughter_child_weight) : 
  (daughter_child_weight - daughter_weight) / (total_weight - daughter_child_weight) = 1/5 := by
  sorry

end child_grandmother_weight_ratio_l859_85950


namespace only_caseD_has_two_solutions_l859_85981

-- Define a structure for triangle cases
structure TriangleCase where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given cases
def caseA : TriangleCase := { a := 0, b := 10, c := 0, A := 45, B := 70, C := 0 }
def caseB : TriangleCase := { a := 60, b := 0, c := 48, A := 0, B := 100, C := 0 }
def caseC : TriangleCase := { a := 14, b := 16, c := 0, A := 45, B := 0, C := 0 }
def caseD : TriangleCase := { a := 7, b := 5, c := 0, A := 80, B := 0, C := 0 }

-- Function to determine if a case has two solutions
def hasTwoSolutions (t : TriangleCase) : Prop :=
  ∃ (B1 B2 : ℝ), B1 ≠ B2 ∧ 
    0 < B1 ∧ B1 < 180 ∧
    0 < B2 ∧ B2 < 180 ∧
    t.a / Real.sin t.A = t.b / Real.sin B1 ∧
    t.a / Real.sin t.A = t.b / Real.sin B2

-- Theorem stating that only case D has two solutions
theorem only_caseD_has_two_solutions :
  ¬(hasTwoSolutions caseA) ∧
  ¬(hasTwoSolutions caseB) ∧
  ¬(hasTwoSolutions caseC) ∧
  hasTwoSolutions caseD :=
sorry

end only_caseD_has_two_solutions_l859_85981


namespace ratio_sum_theorem_l859_85954

theorem ratio_sum_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 3 / 4) (h4 : b = 12) : a + b = 21 := by
  sorry

end ratio_sum_theorem_l859_85954


namespace distance_XY_is_1000_l859_85972

/-- The distance between two points X and Y --/
def distance : ℝ := sorry

/-- The time taken to travel from X to Y --/
def time_XY : ℝ := 10

/-- The time taken to travel from Y to X --/
def time_YX : ℝ := 4

/-- The average speed for the entire journey --/
def avg_speed : ℝ := 142.85714285714286

/-- Theorem stating that the distance between X and Y is 1000 miles --/
theorem distance_XY_is_1000 : distance = 1000 := by sorry

end distance_XY_is_1000_l859_85972


namespace max_trig_sum_value_l859_85955

theorem max_trig_sum_value (θ₁ θ₂ θ₃ θ₄ θ₅ θ₆ : ℝ) :
  (∀ φ₁ φ₂ φ₃ φ₄ φ₅ φ₆ : ℝ,
    (Real.cos φ₁ * Real.sin φ₂ + Real.cos φ₂ * Real.sin φ₃ + 
     Real.cos φ₃ * Real.sin φ₄ + Real.cos φ₄ * Real.sin φ₅ + 
     Real.cos φ₅ * Real.sin φ₆ + Real.cos φ₆ * Real.sin φ₁) ≤
    (Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
     Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
     Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁)) ∧
  (Real.cos θ₁ * Real.sin θ₂ + Real.cos θ₂ * Real.sin θ₃ + 
   Real.cos θ₃ * Real.sin θ₄ + Real.cos θ₄ * Real.sin θ₅ + 
   Real.cos θ₅ * Real.sin θ₆ + Real.cos θ₆ * Real.sin θ₁) = 3 + 3 * Real.sqrt 2 / 2 := by
  sorry

end max_trig_sum_value_l859_85955


namespace one_third_recipe_ingredients_l859_85921

def full_recipe_flour : ℚ := 27/4  -- 6 3/4 cups of flour
def full_recipe_sugar : ℚ := 5/2   -- 2 1/2 cups of sugar
def recipe_fraction : ℚ := 1/3     -- One-third of the recipe

theorem one_third_recipe_ingredients :
  recipe_fraction * full_recipe_flour = 9/4 ∧
  recipe_fraction * full_recipe_sugar = 5/6 := by
  sorry

end one_third_recipe_ingredients_l859_85921


namespace alpha_one_sufficient_not_necessary_l859_85929

-- Define sets A and B
def A : Set ℝ := {x | 2 < x ∧ x < 3}
def B (α : ℝ) : Set ℝ := {x | (x + 2) * (x - α) < 0}

-- Statement to prove
theorem alpha_one_sufficient_not_necessary :
  (∀ x, x ∈ A ∩ B 1 → False) ∧
  (∃ α, α ≠ 1 ∧ ∀ x, x ∈ A ∩ B α → False) := by sorry

end alpha_one_sufficient_not_necessary_l859_85929


namespace right_triangle_side_ratio_range_l859_85986

theorem right_triangle_side_ratio_range (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →        -- Pythagorean theorem
  a + b = c * x →          -- Given condition
  x ∈ Set.Ioo 1 (Real.sqrt 2) := by
  sorry

end right_triangle_side_ratio_range_l859_85986


namespace book_distribution_ways_l859_85943

/-- The number of ways to distribute books to students -/
def distribute_books (num_book_types : ℕ) (num_students : ℕ) (min_copies : ℕ) : ℕ :=
  num_book_types ^ num_students

/-- Theorem: There are 125 ways to distribute 3 books to 3 students from 5 types of books -/
theorem book_distribution_ways :
  distribute_books 5 3 3 = 125 := by
  sorry

end book_distribution_ways_l859_85943


namespace total_savings_l859_85956

/-- The total savings from buying discounted milk and cereal with promotions -/
theorem total_savings (M C : ℝ) : ℝ := by
  -- M: original price of a gallon of milk
  -- C: original price of a box of cereal
  -- Milk discount: 25%
  -- Cereal promotion: buy two, get one 50% off
  -- Buying 3 gallons of milk and 6 boxes of cereal

  /- Define the milk discount -/
  let milk_discount_percent : ℝ := 0.25

  /- Define the cereal promotion discount -/
  let cereal_promotion_discount : ℝ := 0.5

  /- Calculate the savings on milk -/
  let milk_savings : ℝ := 3 * M * milk_discount_percent

  /- Calculate the savings on cereal -/
  let cereal_savings : ℝ := 2 * C * cereal_promotion_discount

  /- Calculate the total savings -/
  let total_savings : ℝ := milk_savings + cereal_savings

  /- Prove that the total savings equals (0.75 * M) + C -/
  have : total_savings = (0.75 * M) + C := by sorry

  /- Return the total savings -/
  exact total_savings

end total_savings_l859_85956


namespace probability_of_passing_is_correct_l859_85923

-- Define the number of shots in the test
def num_shots : ℕ := 3

-- Define the minimum number of successful shots required to pass
def min_successful_shots : ℕ := 2

-- Define the probability of making a single shot
def single_shot_probability : ℝ := 0.6

-- Define the function to calculate the probability of passing the test
def probability_of_passing : ℝ := sorry

-- Theorem stating that the probability of passing is 0.648
theorem probability_of_passing_is_correct : probability_of_passing = 0.648 := by sorry

end probability_of_passing_is_correct_l859_85923


namespace no_integer_satisfies_conditions_l859_85969

theorem no_integer_satisfies_conditions : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), n = 16 * k) ∧ 
  (23 < Real.sqrt n) ∧ 
  (Real.sqrt n < 23.2) := by
sorry

end no_integer_satisfies_conditions_l859_85969


namespace suna_travel_distance_l859_85989

theorem suna_travel_distance (D : ℝ) 
  (h1 : (1 - 7/15) * (1 - 5/8) * (1 - 2/3) * D = 2.6) : D = 39 := by
  sorry

end suna_travel_distance_l859_85989


namespace opposite_reciprocal_expression_l859_85928

theorem opposite_reciprocal_expression (m n p q : ℝ) 
  (h1 : m + n = 0) 
  (h2 : p * q = 1) : 
  -2023 * m + 3 / (p * q) - 2023 * n = 3 := by
sorry

end opposite_reciprocal_expression_l859_85928


namespace max_sundays_in_45_days_l859_85934

/-- The number of days we're considering at the start of the year -/
def days_considered : ℕ := 45

/-- The day of the week, represented as a number from 0 to 6 -/
inductive DayOfWeek : Type
  | Sunday : DayOfWeek
  | Monday : DayOfWeek
  | Tuesday : DayOfWeek
  | Wednesday : DayOfWeek
  | Thursday : DayOfWeek
  | Friday : DayOfWeek
  | Saturday : DayOfWeek

/-- Function to count the number of Sundays in the first n days of a year starting on a given day -/
def count_sundays (start_day : DayOfWeek) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of Sundays in the first 45 days of a year starting on Sunday is 7 -/
theorem max_sundays_in_45_days : 
  count_sundays DayOfWeek.Sunday days_considered = 7 :=
sorry

end max_sundays_in_45_days_l859_85934


namespace system_solution_l859_85974

theorem system_solution :
  let solutions : List (ℤ × ℤ) := [(-5, -3), (-3, -5), (3, 5), (5, 3)]
  ∀ x y : ℤ, (x^2 - x*y + y^2 = 19 ∧ x^4 + x^2*y^2 + y^4 = 931) ↔ (x, y) ∈ solutions :=
by sorry

end system_solution_l859_85974


namespace intersection_point_is_unique_solution_l859_85936

/-- The system of equations representing two lines -/
def line_system (x y : ℚ) : Prop :=
  2 * y = -x + 3 ∧ -y = 5 * x + 1

/-- The intersection point of the two lines -/
def intersection_point : ℚ × ℚ := (-5/9, 16/9)

/-- Theorem stating that the intersection point is the unique solution to the system of equations -/
theorem intersection_point_is_unique_solution :
  line_system intersection_point.1 intersection_point.2 ∧
  ∀ x y, line_system x y → (x, y) = intersection_point := by
  sorry

end intersection_point_is_unique_solution_l859_85936


namespace problem_solution_l859_85976

theorem problem_solution (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 
  (a = Real.sqrt 2 + 1) ∧ 
  (a^2 - 2*a = 1) ∧ 
  (2*a^3 - 4*a^2 - 1 = 2 * Real.sqrt 2 + 1) := by
  sorry

end problem_solution_l859_85976


namespace loan_amount_to_c_l859_85980

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  amount_b : ℝ  -- Amount lent to B
  amount_c : ℝ  -- Amount lent to C (to be determined)
  years_b : ℝ   -- Years for B's loan
  years_c : ℝ   -- Years for C's loan
  rate : ℝ      -- Annual interest rate
  total_interest : ℝ  -- Total interest received from both B and C

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The main theorem to prove --/
theorem loan_amount_to_c 
  (loan : LoanDetails) 
  (h1 : loan.amount_b = 5000)
  (h2 : loan.years_b = 2)
  (h3 : loan.years_c = 4)
  (h4 : loan.rate = 0.09)
  (h5 : loan.total_interest = 1980)
  (h6 : simple_interest loan.amount_b loan.rate loan.years_b + 
        simple_interest loan.amount_c loan.rate loan.years_c = loan.total_interest) :
  loan.amount_c = 500 := by
  sorry


end loan_amount_to_c_l859_85980


namespace divisibility_implies_equality_l859_85908

theorem divisibility_implies_equality (a b : ℕ) 
  (h : (a^2 + a*b + 1) % (b^2 + b*a + 1) = 0) : a = b := by
  sorry

end divisibility_implies_equality_l859_85908


namespace solution_system_equations_l859_85900

theorem solution_system_equations (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0)
  (eq₁ : x₁ + x₂ = x₃^2)
  (eq₂ : x₂ + x₃ = x₄^2)
  (eq₃ : x₃ + x₄ = x₅^2)
  (eq₄ : x₄ + x₅ = x₁^2)
  (eq₅ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 :=
by sorry

end solution_system_equations_l859_85900


namespace share_difference_example_l859_85922

/-- Given a total profit and proportions for distribution among three parties,
    calculate the difference between the shares of the second and third parties. -/
def shareDifference (totalProfit : ℕ) (propA propB propC : ℕ) : ℕ :=
  let totalParts := propA + propB + propC
  let partValue := totalProfit / totalParts
  let shareB := propB * partValue
  let shareC := propC * partValue
  shareC - shareB

/-- Prove that for a total profit of 20000 distributed in the proportion 2:3:5,
    the difference between C's and B's shares is 4000. -/
theorem share_difference_example : shareDifference 20000 2 3 5 = 4000 := by
  sorry

end share_difference_example_l859_85922


namespace shen_win_probability_correct_l859_85978

/-- Represents a player in the game -/
inductive Player
| Shen
| Ling
| Ru

/-- The number of slips each player puts in the bucket initially -/
def initial_slips : Nat := 4

/-- The total number of slips in the bucket -/
def total_slips : Nat := 13

/-- The number of slips Shen needs to win -/
def shen_win_condition : Nat := 4

/-- Calculates the probability of Shen winning the game -/
def shen_win_probability : Rat :=
  67 / 117

/-- Theorem stating that the calculated probability is correct -/
theorem shen_win_probability_correct :
  shen_win_probability = 67 / 117 := by sorry

end shen_win_probability_correct_l859_85978


namespace minimum_point_of_translated_abs_value_function_l859_85953

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 3) - 4 + 4

-- Theorem statement
theorem minimum_point_of_translated_abs_value_function :
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 0 ∧ x = 3 :=
sorry

end minimum_point_of_translated_abs_value_function_l859_85953


namespace smallest_multiples_product_l859_85997

theorem smallest_multiples_product (c d : ℕ) : 
  (c ≥ 10 ∧ c < 100 ∧ c % 7 = 0 ∧ ∀ x, x ≥ 10 ∧ x < 100 ∧ x % 7 = 0 → c ≤ x) →
  (d ≥ 100 ∧ d < 1000 ∧ d % 5 = 0 ∧ ∀ y, y ≥ 100 ∧ y < 1000 ∧ y % 5 = 0 → d ≤ y) →
  (c * d) - 100 = 1300 := by
sorry

end smallest_multiples_product_l859_85997


namespace cost_of_480_chocolates_l859_85912

/-- The cost of buying a given number of chocolates, given the box size and box cost -/
def chocolate_cost (total_chocolates : ℕ) (box_size : ℕ) (box_cost : ℕ) : ℕ :=
  (total_chocolates / box_size) * box_cost

/-- Theorem: The cost of 480 chocolates is $96, given that a box of 40 chocolates costs $8 -/
theorem cost_of_480_chocolates :
  chocolate_cost 480 40 8 = 96 := by
  sorry

end cost_of_480_chocolates_l859_85912


namespace drawings_on_last_page_l859_85996

theorem drawings_on_last_page 
  (initial_notebooks : Nat) 
  (pages_per_notebook : Nat) 
  (initial_drawings_per_page : Nat) 
  (reorganized_drawings_per_page : Nat)
  (filled_notebooks : Nat)
  (filled_pages_last_notebook : Nat) :
  initial_notebooks = 12 →
  pages_per_notebook = 35 →
  initial_drawings_per_page = 4 →
  reorganized_drawings_per_page = 7 →
  filled_notebooks = 6 →
  filled_pages_last_notebook = 25 →
  (initial_notebooks * pages_per_notebook * initial_drawings_per_page) -
  (filled_notebooks * pages_per_notebook * reorganized_drawings_per_page) -
  (filled_pages_last_notebook * reorganized_drawings_per_page) = 5 := by
  sorry

end drawings_on_last_page_l859_85996


namespace specific_management_structure_count_l859_85935

/-- The number of ways to form a management structure --/
def management_structure_count (total_employees : ℕ) (ceo_count : ℕ) (vp_count : ℕ) (managers_per_vp : ℕ) : ℕ :=
  total_employees * 
  (Nat.choose (total_employees - 1) vp_count) * 
  (Nat.choose (total_employees - 1 - vp_count) managers_per_vp) * 
  (Nat.choose (total_employees - 1 - vp_count - managers_per_vp) managers_per_vp)

/-- Theorem stating the number of ways to form the specific management structure --/
theorem specific_management_structure_count :
  management_structure_count 13 1 2 3 = 349800 := by
  sorry

end specific_management_structure_count_l859_85935


namespace square_area_l859_85993

-- Define the square WXYZ
structure Square (W X Y Z : ℝ × ℝ) : Prop where
  is_square : true  -- We assume WXYZ is a square

-- Define the points P and Q
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry

-- Define the properties of the square and points
def square_properties (W X Y Z : ℝ × ℝ) (P Q : ℝ × ℝ) : Prop :=
  Square W X Y Z ∧
  (∃ (t : ℝ), t > 0 ∧ t < 1 ∧ P = (1 - t) • X + t • Y) ∧  -- P is on XY
  (∃ (s : ℝ), s > 0 ∧ s < 1 ∧ Q = (1 - s) • W + s • Z) ∧  -- Q is on WZ
  (Y.1 - P.1)^2 + (Y.2 - P.2)^2 = 16 ∧  -- YP = 4
  (Q.1 - Z.1)^2 + (Q.2 - Z.2)^2 = 9    -- QZ = 3

-- Define the angle trisection property
def angle_trisected (W P Q : ℝ × ℝ) : Prop :=
  ∃ (θ : ℝ), θ > 0 ∧ 
    (P.2 - W.2) / (P.1 - W.1) = Real.tan θ ∧
    (Q.2 - W.2) / (Q.1 - W.1) = Real.tan (2 * θ)

-- Theorem statement
theorem square_area (W X Y Z : ℝ × ℝ) (P Q : ℝ × ℝ) :
  square_properties W X Y Z P Q →
  angle_trisected W P Q →
  (Y.1 - W.1)^2 + (Y.2 - W.2)^2 = 48 := by
  sorry

end square_area_l859_85993


namespace ratio_difference_l859_85941

theorem ratio_difference (a b c : ℕ) (h1 : a + b + c > 0) : 
  (a : ℚ) / (a + b + c) = 3 / 15 →
  (b : ℚ) / (a + b + c) = 5 / 15 →
  (c : ℚ) / (a + b + c) = 7 / 15 →
  c = 70 →
  c - a = 40 := by
sorry

end ratio_difference_l859_85941


namespace max_k_value_l859_85995

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * (x^2 / y^2 + y^2 / x^2 + 2) + 2 * k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 6) / 2 := by
  sorry

end max_k_value_l859_85995


namespace pirate_count_l859_85905

/-- Represents the total number of pirates on the schooner -/
def total_pirates : ℕ := 30

/-- Represents the number of pirates who did not participate in the battle -/
def non_participants : ℕ := 10

/-- Represents the percentage of battle participants who lost an arm -/
def arm_loss_percentage : ℚ := 54 / 100

/-- Represents the percentage of battle participants who lost both an arm and a leg -/
def both_loss_percentage : ℚ := 34 / 100

/-- Represents the fraction of all pirates who lost a leg -/
def leg_loss_fraction : ℚ := 2 / 3

theorem pirate_count : 
  total_pirates = 30 ∧
  non_participants = 10 ∧
  arm_loss_percentage = 54 / 100 ∧
  both_loss_percentage = 34 / 100 ∧
  leg_loss_fraction = 2 / 3 →
  total_pirates = 30 :=
by sorry

end pirate_count_l859_85905


namespace quadratic_inequality_roots_l859_85994

theorem quadratic_inequality_roots (c : ℝ) : 
  (∀ x, -x^2 + c*x + 3 < 0 ↔ x < -3 ∨ x > 2) → c = 5 := by
  sorry

end quadratic_inequality_roots_l859_85994


namespace camp_kids_count_l859_85988

theorem camp_kids_count : ℕ :=
  let total_kids : ℕ := 2000
  let soccer_kids : ℕ := total_kids / 2
  let morning_soccer_kids : ℕ := soccer_kids / 4
  let afternoon_soccer_kids : ℕ := 750
  have h1 : soccer_kids = total_kids / 2 := by sorry
  have h2 : morning_soccer_kids = soccer_kids / 4 := by sorry
  have h3 : afternoon_soccer_kids = 750 := by sorry
  have h4 : morning_soccer_kids + afternoon_soccer_kids = soccer_kids := by sorry
  total_kids

/- Proof
sorry
-/

end camp_kids_count_l859_85988


namespace problem_solution_l859_85902

theorem problem_solution :
  let A : ℝ → ℝ → ℝ := λ x y => -4 * x^2 - 4 * x * y + 1
  let B : ℝ → ℝ → ℝ := λ x y => x^2 + x * y - 5
  let x : ℝ := 1
  let y : ℝ := -1
  2 * B x y - A x y = -11 := by
  sorry

end problem_solution_l859_85902


namespace prime_cube_sum_of_squares_l859_85967

theorem prime_cube_sum_of_squares (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p^3 = p^2 + q^2 + r^2 → 
  p = 3 ∧ q = 3 ∧ r = 3 := by
sorry

end prime_cube_sum_of_squares_l859_85967


namespace exists_k_configuration_l859_85914

/-- A configuration of black cells on an infinite white checker plane. -/
structure BlackCellConfiguration where
  cells : Set (ℤ × ℤ)
  finite : Set.Finite cells

/-- A line on the infinite plane (vertical, horizontal, or diagonal). -/
inductive Line
  | Vertical (x : ℤ) : Line
  | Horizontal (y : ℤ) : Line
  | Diagonal (a b c : ℤ) : Line

/-- The number of black cells on a given line for a given configuration. -/
def blackCellsOnLine (config : BlackCellConfiguration) (line : Line) : ℕ :=
  sorry

/-- A configuration satisfies the k-condition if every line contains
    either k black cells or no black cells. -/
def satisfiesKCondition (config : BlackCellConfiguration) (k : ℕ+) : Prop :=
  ∀ line, blackCellsOnLine config line = k ∨ blackCellsOnLine config line = 0

/-- For any positive integer k, there exists a configuration of black cells
    satisfying the k-condition. -/
theorem exists_k_configuration (k : ℕ+) :
  ∃ (config : BlackCellConfiguration), satisfiesKCondition config k :=
sorry

end exists_k_configuration_l859_85914


namespace male_red_ants_percentage_l859_85947

/-- Represents the percentage of red ants in the total population -/
def red_percentage : ℝ := 0.85

/-- Represents the percentage of female ants among red ants -/
def female_red_percentage : ℝ := 0.45

/-- Calculates the percentage of male red ants in the total population -/
def male_red_percentage : ℝ := red_percentage * (1 - female_red_percentage)

/-- Theorem stating that the percentage of male red ants in the total population is 46.75% -/
theorem male_red_ants_percentage : 
  male_red_percentage = 0.4675 := by sorry

end male_red_ants_percentage_l859_85947


namespace sum_f_mod_1000_l859_85948

-- Define the function f
def f (n : ℕ) : ℕ := 
  (Finset.filter (fun d => d < n ∨ Nat.gcd d n ≠ 1) (Nat.divisors (2024^2024))).card

-- State the theorem
theorem sum_f_mod_1000 : 
  (Finset.sum (Finset.range (2024^2024 + 1)) f) % 1000 = 224 := by sorry

end sum_f_mod_1000_l859_85948


namespace arithmetic_geometric_sequence_relation_l859_85960

theorem arithmetic_geometric_sequence_relation :
  ∃ (a g : ℕ → ℕ) (n : ℕ),
    (∀ k, a (k + 1) = a k + (a 2 - a 1)) ∧  -- arithmetic sequence
    (∀ k, g (k + 1) = g k * (g 2 / g 1)) ∧  -- geometric sequence
    n = 14 ∧
    a 1 = g 1 ∧ a 2 = g 2 ∧ a 5 = g 3 ∧ a n = g 4 ∧
    g 1 + g 2 + g 3 + g 4 = 80 ∧
    a 1 = 2 ∧ a 2 = 6 ∧ a 5 = 18 ∧ a n = 54 ∧
    g 1 = 2 ∧ g 2 = 6 ∧ g 3 = 18 ∧ g 4 = 54 :=
by sorry


end arithmetic_geometric_sequence_relation_l859_85960


namespace correct_articles_l859_85939

/-- Represents an article in English --/
inductive Article
| None
| A
| The

/-- Represents a sentence with two blanks for articles --/
structure Sentence :=
  (first_blank : Article)
  (second_blank : Article)

/-- Checks if a given sentence has the correct articles --/
def is_correct_sentence (s : Sentence) : Prop :=
  s.first_blank = Article.None ∧ s.second_blank = Article.A

/-- The theorem stating that the correct sentence has no article in the first blank and "a" in the second blank --/
theorem correct_articles : 
  ∃ (s : Sentence), is_correct_sentence s :=
sorry

end correct_articles_l859_85939


namespace new_student_weight_l859_85911

theorem new_student_weight (initial_count : ℕ) (initial_avg : ℝ) (new_count : ℕ) (new_avg : ℝ) : 
  initial_count = 19 →
  initial_avg = 15 →
  new_count = initial_count + 1 →
  new_avg = 14.9 →
  (initial_count : ℝ) * initial_avg + (new_count * new_avg - initial_count * initial_avg) = 13 := by
  sorry

end new_student_weight_l859_85911


namespace book_to_bookmark_ratio_l859_85992

def books : ℕ := 72
def bookmarks : ℕ := 16

theorem book_to_bookmark_ratio : 
  (books / (Nat.gcd books bookmarks)) / (bookmarks / (Nat.gcd books bookmarks)) = 9 / 2 := by
  sorry

end book_to_bookmark_ratio_l859_85992


namespace inverse_proportionality_l859_85917

theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * 10 = k) :
  40 * (5/4 : ℝ) = k := by sorry

end inverse_proportionality_l859_85917


namespace prove_smallest_positive_angle_l859_85919

def smallest_positive_angle_theorem : Prop :=
  ∃ θ : Real,
    θ > 0 ∧
    θ < 2 * Real.pi ∧
    Real.cos θ = Real.sin (60 * Real.pi / 180) + Real.cos (42 * Real.pi / 180) - 
                 Real.sin (12 * Real.pi / 180) - Real.cos (6 * Real.pi / 180) ∧
    θ = 66 * Real.pi / 180 ∧
    ∀ φ : Real, 
      φ > 0 → 
      φ < 2 * Real.pi → 
      Real.cos φ = Real.sin (60 * Real.pi / 180) + Real.cos (42 * Real.pi / 180) - 
                   Real.sin (12 * Real.pi / 180) - Real.cos (6 * Real.pi / 180) → 
      φ ≥ θ

theorem prove_smallest_positive_angle : smallest_positive_angle_theorem :=
sorry

end prove_smallest_positive_angle_l859_85919


namespace two_card_picks_from_two_decks_l859_85982

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)

/-- Represents the total collection of cards from two shuffled decks -/
def ShuffledDecks (d : Deck) : Nat :=
  2 * d.cards

/-- The number of ways to pick two different cards from shuffled decks -/
def PickTwoCards (total : Nat) : Nat :=
  total * (total - 1)

theorem two_card_picks_from_two_decks :
  let standard_deck : Deck := { cards := 52, suits := 4, cards_per_suit := 13 }
  let shuffled_total := ShuffledDecks standard_deck
  PickTwoCards shuffled_total = 10692 := by
  sorry

end two_card_picks_from_two_decks_l859_85982


namespace range_of_k_l859_85957

-- Define the inequality condition
def inequality_condition (k : ℝ) : Prop :=
  ∀ x > 0, Real.exp (x + 1) - (Real.log x + 2 * k) / x - k ≥ 0

-- Theorem statement
theorem range_of_k (k : ℝ) :
  inequality_condition k → k ∈ Set.Iic 1 :=
by sorry

end range_of_k_l859_85957


namespace sqrt_meaningful_iff_x_geq_3_l859_85971

theorem sqrt_meaningful_iff_x_geq_3 (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_iff_x_geq_3_l859_85971


namespace largest_square_divisor_l859_85998

theorem largest_square_divisor : 
  ∃ (x : ℕ), x = 12 ∧ 
  x^2 ∣ (24 * 35 * 46 * 57) ∧ 
  ∀ (y : ℕ), y > x → ¬(y^2 ∣ (24 * 35 * 46 * 57)) := by
  sorry

end largest_square_divisor_l859_85998


namespace train_speed_increase_l859_85942

theorem train_speed_increase (old_time new_time : ℝ) 
  (hold : old_time = 16 ∧ new_time = 14) : 
  (1 / new_time - 1 / old_time) / (1 / old_time) = 
  (1 / 14 - 1 / 16) / (1 / 16) := by
  sorry

end train_speed_increase_l859_85942


namespace expression_simplification_l859_85909

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 5 + 2) :
  (x + 2) / (x - 1) / (x + 1 - 3 / (x - 1)) = Real.sqrt 5 / 5 := by
  sorry

end expression_simplification_l859_85909


namespace similar_triangles_height_cycle_height_problem_l859_85990

theorem similar_triangles_height (h₁ : ℝ) (b₁ : ℝ) (b₂ : ℝ) (h₁_pos : h₁ > 0) (b₁_pos : b₁ > 0) (b₂_pos : b₂ > 0) :
  h₁ / b₁ = (h₁ * b₂ / b₁) / b₂ :=
by sorry

theorem cycle_height_problem (h₁ : ℝ) (b₁ : ℝ) (b₂ : ℝ) 
  (h₁_val : h₁ = 2.5) (b₁_val : b₁ = 5) (b₂_val : b₂ = 4) :
  h₁ * b₂ / b₁ = 2 :=
by sorry

end similar_triangles_height_cycle_height_problem_l859_85990


namespace max_volume_rotating_cube_max_volume_is_eight_l859_85931

/-- The maximum volume of a cube that can rotate freely inside a cube with edge length 2 -/
theorem max_volume_rotating_cube (outer_edge : ℝ) (h : outer_edge = 2) :
  ∃ (inner_edge : ℝ),
    inner_edge > 0 ∧
    inner_edge * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 ∧
    ∀ (x : ℝ), x > 0 → x * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 → x^3 ≤ inner_edge^3 :=
by
  sorry

/-- The maximum volume of the rotating cube is 8 -/
theorem max_volume_is_eight (outer_edge : ℝ) (h : outer_edge = 2) :
  ∃ (inner_edge : ℝ),
    inner_edge > 0 ∧
    inner_edge * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 ∧
    inner_edge^3 = 8 ∧
    ∀ (x : ℝ), x > 0 → x * Real.sqrt 3 ≤ outer_edge * Real.sqrt 3 → x^3 ≤ 8 :=
by
  sorry

end max_volume_rotating_cube_max_volume_is_eight_l859_85931


namespace root_ratio_quadratic_equation_l859_85938

theorem root_ratio_quadratic_equation (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x/y = 2/3 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0) :
  6*b^2 = 25*a*c := by
  sorry

end root_ratio_quadratic_equation_l859_85938


namespace parametric_to_cartesian_l859_85913

theorem parametric_to_cartesian :
  ∀ x y θ : ℝ,
  x = Real.sin θ →
  y = Real.cos (2 * θ) →
  -1 ≤ x ∧ x ≤ 1 →
  y = 1 - 2 * x^2 :=
by
  sorry

end parametric_to_cartesian_l859_85913


namespace complex_fraction_real_l859_85949

/-- Given that i is the imaginary unit and (a+2i)/(1+i) is a real number, prove that a = 2 -/
theorem complex_fraction_real (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑a + 2 * Complex.I) / (1 + Complex.I) ∈ Set.range (Complex.ofReal : ℝ → ℂ) → 
  a = 2 := by
  sorry

end complex_fraction_real_l859_85949


namespace kiwis_equal_lemons_l859_85987

/-- Represents the contents of a fruit basket -/
structure FruitBasket where
  mangoes : Nat
  pears : Nat
  pawpaws : Nat
  lemons : Nat
  kiwis : Nat

/-- Represents Tania's collection of fruit baskets -/
def TaniaBaskets : List FruitBasket :=
  [
    { mangoes := 18, pears := 0, pawpaws := 0, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 10, pawpaws := 0, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 0, pawpaws := 12, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 0, pawpaws := 0, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 0, pawpaws := 0, lemons := 0, kiwis := 0 }
  ]

/-- The total number of fruits in all baskets -/
def totalFruits : Nat := 58

/-- The total number of lemons in all baskets -/
def totalLemons : Nat := 9

/-- Theorem stating that the number of kiwis equals the number of lemons in the last two baskets -/
theorem kiwis_equal_lemons (h1 : List.length TaniaBaskets = 5)
    (h2 : (TaniaBaskets.map (fun b => b.mangoes + b.pears + b.pawpaws + b.lemons + b.kiwis)).sum = totalFruits)
    (h3 : (TaniaBaskets.map (fun b => b.lemons)).sum = totalLemons) :
    (List.take 2 (List.reverse TaniaBaskets)).map (fun b => b.kiwis) = 
    (List.take 2 (List.reverse TaniaBaskets)).map (fun b => b.lemons) := by
  sorry

end kiwis_equal_lemons_l859_85987


namespace basketball_team_average_weight_l859_85962

/-- The average weight of a basketball team after adding new players -/
theorem basketball_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1 : ℝ) 
  (new_player2 : ℝ) 
  (new_player3 : ℝ) 
  (new_player4 : ℝ) 
  (h1 : original_players = 8) 
  (h2 : original_average = 105.5) 
  (h3 : new_player1 = 110.3) 
  (h4 : new_player2 = 99.7) 
  (h5 : new_player3 = 103.2) 
  (h6 : new_player4 = 115.4) : 
  (original_players * original_average + new_player1 + new_player2 + new_player3 + new_player4) / (original_players + 4) = 106.05 := by
  sorry

end basketball_team_average_weight_l859_85962


namespace subset_implies_a_equals_one_l859_85924

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) :
  A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l859_85924


namespace banana_sharing_l859_85933

/-- Proves that sharing 21 bananas equally among 3 friends results in 7 bananas per friend -/
theorem banana_sharing (total_bananas : ℕ) (num_friends : ℕ) (bananas_per_friend : ℕ) :
  total_bananas = 21 →
  num_friends = 3 →
  bananas_per_friend = total_bananas / num_friends →
  bananas_per_friend = 7 :=
by
  sorry

end banana_sharing_l859_85933


namespace fraction_to_decimal_l859_85991

theorem fraction_to_decimal (n : ℕ) (d : ℕ) (h : d = 2^3 * 5^7) :
  (n : ℚ) / d = 0.0006625 ↔ n = 53 :=
sorry

end fraction_to_decimal_l859_85991


namespace toucan_female_fraction_l859_85985

theorem toucan_female_fraction (total_birds : ℝ) (h1 : total_birds > 0) :
  let parrot_fraction : ℝ := 3/5
  let toucan_fraction : ℝ := 1 - parrot_fraction
  let female_parrot_fraction : ℝ := 1/3
  let male_bird_fraction : ℝ := 1/2
  let female_toucan_count : ℝ := toucan_fraction * total_birds * female_toucan_fraction
  let female_parrot_count : ℝ := parrot_fraction * total_birds * female_parrot_fraction
  let total_female_count : ℝ := female_toucan_count + female_parrot_count
  female_toucan_count + female_parrot_count = male_bird_fraction * total_birds →
  female_toucan_fraction = 3/4 :=
by sorry

end toucan_female_fraction_l859_85985


namespace faster_train_speed_faster_train_speed_result_l859_85906

/-- Calculates the speed of the faster train given the conditions of the problem -/
theorem faster_train_speed (length_train1 length_train2 : ℝ)
                            (speed_slower : ℝ)
                            (crossing_time : ℝ) : ℝ :=
  let total_length := length_train1 + length_train2
  let total_length_km := total_length / 1000
  let crossing_time_hours := crossing_time / 3600
  let relative_speed := total_length_km / crossing_time_hours
  speed_slower + relative_speed

/-- The speed of the faster train is approximately 45.95 kmph -/
theorem faster_train_speed_result :
  ∃ ε > 0, |faster_train_speed 200 150 40 210 - 45.95| < ε :=
by
  sorry

end faster_train_speed_faster_train_speed_result_l859_85906


namespace contrapositive_equivalence_l859_85926

theorem contrapositive_equivalence (p q : Prop) :
  (p → ¬q) ↔ (q → ¬p) := by
  sorry

end contrapositive_equivalence_l859_85926


namespace similar_triangle_coordinates_l859_85965

-- Define the vertices of triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def ratio : ℝ := 2

-- Define the possible coordinates of C'
def C'_pos : ℝ × ℝ := (6, 4)
def C'_neg : ℝ × ℝ := (-6, -4)

-- Theorem statement
theorem similar_triangle_coordinates :
  ∀ (C' : ℝ × ℝ), 
    (∃ (k : ℝ), k = ratio ∧ C' = (k * C.1, k * C.2)) ∨
    (∃ (k : ℝ), k = -ratio ∧ C' = (k * C.1, k * C.2)) →
    C' = C'_pos ∨ C' = C'_neg :=
by sorry

end similar_triangle_coordinates_l859_85965


namespace sum_of_squares_zero_l859_85910

theorem sum_of_squares_zero (a b c : ℝ) :
  (a - 6)^2 + (b - 3)^2 + (c - 2)^2 = 0 → a + b + c = 11 := by
  sorry

end sum_of_squares_zero_l859_85910
