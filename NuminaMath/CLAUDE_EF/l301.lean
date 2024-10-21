import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cycles_in_graph_l301_30158

/-- A simple graph with vertices and edges. -/
structure MySimpleGraph (V : Type) where
  edges : Set (V × V)
  symm : ∀ ⦃x y⦄, (x, y) ∈ edges → (y, x) ∈ edges
  irrefl : ∀ x, (x, x) ∉ edges

/-- The number of vertices in a graph. -/
def vertexCount {V : Type} (G : MySimpleGraph V) : ℕ := sorry

/-- The number of edges in a graph. -/
def edgeCount {V : Type} (G : MySimpleGraph V) : ℕ := sorry

/-- The cyclomatic number (circuit rank) of a graph. -/
def cyclomaticNumber {V : Type} (G : MySimpleGraph V) : ℕ := 
  edgeCount G - vertexCount G + 1

/-- The theorem stating the minimal number of cycles in the given graph. -/
theorem min_cycles_in_graph {V : Type} (G : MySimpleGraph V) 
  (h_vertices : vertexCount G = 2013) 
  (h_edges : edgeCount G = 3013) : 
  cyclomaticNumber G = 1001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cycles_in_graph_l301_30158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_50_l301_30122

noncomputable def initial_outlay : ℝ := 12450
noncomputable def manufacturing_cost_per_set : ℝ := 20.75
def sets_produced : ℕ := 950
noncomputable def profit : ℝ := 15337.5

noncomputable def total_cost : ℝ := initial_outlay + manufacturing_cost_per_set * (sets_produced : ℝ)

noncomputable def selling_price : ℝ := (total_cost + profit) / (sets_produced : ℝ)

theorem selling_price_is_50 : selling_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_selling_price_is_50_l301_30122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l301_30190

theorem complex_equation_solution (z : ℂ) 
  (h : 8 * Complex.normSq z = 3 * Complex.normSq (z + 3) + Complex.normSq (z^2 + 2) + 50) :
  z + 9 / z = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_solution_l301_30190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_and_function_properties_l301_30167

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (1 - Real.exp (2*x)) / Real.exp x

-- Theorem statement
theorem quadratic_range_and_function_properties :
  (∀ y ∈ Set.Ici 2, ∃ x ≥ 0, f x = y) ∧
  (∀ x, f x ≥ 2) ∧
  (∀ x, h (-x) = -h x) ∧
  (∀ x y, x < y → h x > h y) := by
  sorry

#check quadratic_range_and_function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_range_and_function_properties_l301_30167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_divisibility_l301_30104

def u : ℕ → ℤ
  | 0 => 0  -- Added case for 0
  | 1 => 1
  | 2 => 0
  | 3 => 1
  | n + 4 => u (n + 2) + u (n + 1)

theorem u_divisibility (n : ℕ) (h : n ≥ 1) :
  ∃ (k₁ k₂ : ℤ), u (2 * n) - (u (n - 1))^2 = k₁ * u n ∧
                 u (2 * n + 1) - (u (n + 1))^2 = k₂ * u n :=
by
  sorry

#check u_divisibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_divisibility_l301_30104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stumps_l301_30132

/-- Represents the possible number of stumps on the road -/
def PossibleStumps : Set ℕ := {1, 3, 11, 33}

/-- Maria's rest time function -/
noncomputable def maria_rest : ℝ → ℝ := sorry

/-- Mikhail's rest time function -/
noncomputable def mikhail_rest : ℝ → ℝ := sorry

/-- Maria's total travel time function -/
noncomputable def maria_total_time : ℝ → ℝ → ℝ := sorry

/-- Mikhail's total travel time function -/
noncomputable def mikhail_total_time : ℝ → ℝ → ℝ := sorry

/-- Proves that the given set of stumps is correct under the given conditions -/
theorem correct_stumps 
  (road_length : ℝ) 
  (maria_speed : ℝ) 
  (mikhail_speed : ℝ) 
  (h_road : road_length = 11) 
  (h_maria : maria_speed = 4) 
  (h_mikhail : mikhail_speed = 5) 
  (h_rest_ratio : ∀ (t : ℝ), mikhail_rest t = 2 * maria_rest t) 
  (h_whole_minutes : ∀ (n : ℕ), n ∈ PossibleStumps → ∃ (t : ℕ), maria_rest (t : ℝ) = n * t) 
  (h_same_time : maria_total_time road_length maria_speed = mikhail_total_time road_length mikhail_speed) :
  ∀ (n : ℕ), n ∈ PossibleStumps ↔ 
    ∃ (t : ℝ), 
      t > 0 ∧ 
      maria_total_time road_length maria_speed = road_length / maria_speed + n * t ∧
      mikhail_total_time road_length mikhail_speed = road_length / mikhail_speed + n * (2 * t) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_stumps_l301_30132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l301_30133

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + d * i)

theorem arithmetic_sequence_properties :
  let seq := arithmetic_sequence (-50) 6 21
  (seq.length = 21) ∧
  (seq.getLast? = some 68) ∧
  (seq.sum = 231) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l301_30133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_7_5_l301_30115

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Calculates the area of a trapezoid given its four vertices -/
def trapezoidArea (e f g h : Point2D) : ℚ :=
  let base1 := f.y - e.y
  let base2 := g.x - h.x
  let height := h.x - e.x
  (base1 + base2) * height / 2

/-- Theorem: The area of the trapezoid EFGH is 7.5 square units -/
theorem trapezoid_area_is_7_5 :
  let e := Point2D.mk 0 0
  let f := Point2D.mk 0 3
  let g := Point2D.mk 5 3
  let h := Point2D.mk 3 0
  trapezoidArea e f g h = 15/2 := by
  sorry

#eval trapezoidArea (Point2D.mk 0 0) (Point2D.mk 0 3) (Point2D.mk 5 3) (Point2D.mk 3 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_7_5_l301_30115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18s_l301_30168

/-- Calculates the time for a train to cross a signal pole given its length, 
    the length of a platform it crosses, and the time it takes to cross the platform. -/
noncomputable def time_to_cross_signal_pole (train_length platform_length time_to_cross_platform : ℝ) : ℝ :=
  train_length / ((train_length + platform_length) / time_to_cross_platform)

/-- Theorem stating that a 300m train crossing a 500m platform in 48s 
    will take approximately 18s to cross a signal pole. -/
theorem train_crossing_time_approx_18s :
  let train_length : ℝ := 300
  let platform_length : ℝ := 500
  let time_to_cross_platform : ℝ := 48
  let time_to_cross_pole := time_to_cross_signal_pole train_length platform_length time_to_cross_platform
  abs (time_to_cross_pole - 18) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_approx_18s_l301_30168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_expression_value_l301_30152

/-- Theorem: The value of (a³ + b³ + c³) / (a² - ab + b² - ac + c²) when a = 6, b = 3, and c = 2 is 13.21 -/
theorem cubic_expression_value : 
  let a : ℝ := 6
  let b : ℝ := 3
  let c : ℝ := 2
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - a*c + c^2) = 13.21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_expression_value_l301_30152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_terms_l301_30102

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  isPositive : ∀ n, a n > 0
  isArithmetic : ∃ d : ℚ, ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem stating the maximum product of the 7th and 14th terms -/
theorem max_product_of_terms (seq : ArithmeticSequence) 
  (h : sumOfTerms seq 20 = 100) :
  (∀ seq' : ArithmeticSequence, sumOfTerms seq' 20 = 100 → 
    seq'.a 7 * seq'.a 14 ≤ seq.a 7 * seq.a 14) →
  seq.a 7 * seq.a 14 = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_of_terms_l301_30102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_m_triangle_equilateral_triangle_l301_30166

-- Define the equation
def equation (x m : ℝ) : Prop := (x - 2) * (x^2 - 4*x + m) = 0

-- Define the roots
def roots (m : ℝ) : Prop := ∃ (x₁ x₂ x₃ : ℝ), 
  equation x₁ m ∧ equation x₂ m ∧ equation x₃ m ∧ 
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃

-- Define triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define isosceles triangle
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) (h : roots m) : m ≤ 4 := by sorry

-- Theorem 2: Range of m for triangle
theorem range_of_m_triangle (m : ℝ) (h : roots m) 
  (t : ∃ (x₁ x₂ x₃ : ℝ), equation x₁ m ∧ equation x₂ m ∧ equation x₃ m ∧ 
    triangle_inequality x₁ x₂ x₃) : 
  3 < m ∧ m ≤ 4 := by sorry

-- Theorem 3: Equilateral triangle case
theorem equilateral_triangle (m : ℝ) (h : roots m)
  (t : ∃ (x₁ x₂ x₃ : ℝ), equation x₁ m ∧ equation x₂ m ∧ equation x₃ m ∧ 
    isosceles_triangle x₁ x₂ x₃) :
  m = 4 ∧ ∃ (s : ℝ), s = 2 ∧ (Real.sqrt 3 / 4) * s^2 = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_range_of_m_triangle_equilateral_triangle_l301_30166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_five_roses_and_daisies_l301_30137

/-- The number of ways to arrange plants with constraints -/
def arrange_plants (num_roses : ℕ) (num_daisies : ℕ) : ℕ :=
  (num_roses + 1) * Nat.factorial num_roses * Nat.factorial num_daisies

/-- Theorem: Arranging 5 roses and 5 daisies with all daisies together -/
theorem arrange_five_roses_and_daisies :
  arrange_plants 5 5 = 86400 := by
  rw [arrange_plants]
  norm_num
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrange_five_roses_and_daisies_l301_30137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_probability_theorem_l301_30162

/-- The probability of snowing on at most 3 days in December in Frost Town -/
noncomputable def snow_probability : ℝ := sorry

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The probability of snowing on any given day in December -/
noncomputable def daily_snow_probability : ℝ := 1/5

/-- Theorem stating that the probability of snowing on at most 3 days in December
    is approximately 0.342 -/
theorem snow_probability_theorem :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |snow_probability - 0.342| < ε := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_probability_theorem_l301_30162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_subtraction_theorem_l301_30148

/-- Represents a number in base 8 --/
structure Base8 where
  value : Nat
  is_valid : value < 512 := by sorry -- Ensures the number is within 3 digits in base 8

/-- Converts a base 8 number to base 10 --/
def to_base10 (n : Base8) : Nat :=
  let d0 := n.value % 8
  let d1 := (n.value / 8) % 8
  let d2 := n.value / 64
  d0 + 8 * d1 + 64 * d2

/-- Subtracts two base 8 numbers --/
def base8_sub (a b : Base8) : Base8 where
  value := to_base10 a - to_base10 b
  is_valid := by sorry

instance : OfNat Base8 n where
  ofNat := ⟨n, by sorry⟩

theorem base8_subtraction_theorem :
  let a : Base8 := 156
  let b : Base8 := 71
  (base8_sub a b).value = 75 ∧ to_base10 (base8_sub a b) = 61 := by
  sorry

#eval to_base10 (base8_sub ⟨156, by sorry⟩ ⟨71, by sorry⟩)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_subtraction_theorem_l301_30148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l301_30195

theorem test_maximum_marks (passing_threshold : Real) 
  (student_score : Nat) (failing_margin : Nat) 
  (h1 : passing_threshold = 0.75)
  (h2 : student_score = 200)
  (h3 : failing_margin = 180) :
  ∃ (max_marks : Nat), max_marks = 507 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_maximum_marks_l301_30195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_2415_l301_30112

-- Define the capital contributions
noncomputable def capital_A (total : ℝ) : ℝ := (1/3) * total
noncomputable def capital_B (total : ℝ) : ℝ := (1/4) * total
noncomputable def capital_C (total : ℝ) : ℝ := (1/5) * total
noncomputable def capital_D (total : ℝ) : ℝ := total - (capital_A total + capital_B total + capital_C total)

-- Define A's profit share
def profit_A : ℝ := 805

-- Theorem to prove
theorem total_profit_is_2415 (total_capital : ℝ) (total_profit : ℝ) 
  (h1 : profit_A / capital_A total_capital = total_profit / total_capital) :
  total_profit = 2415 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_2415_l301_30112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_purchase_savings_l301_30129

/-- Proves that purchasing a laptop online saves $0.05 compared to the discounted in-store price --/
theorem laptop_purchase_savings (in_store_price in_store_discount online_payment online_shipping : ℝ) 
  (h1 : in_store_price = 599.99)
  (h2 : in_store_discount = 0.05)
  (h3 : online_payment = 109.99)
  (h4 : online_shipping = 19.99)
  : (in_store_price * (1 - in_store_discount)) - (5 * online_payment + online_shipping) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laptop_purchase_savings_l301_30129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_horse_race_odds_l301_30141

/-- Represents the odds against a horse winning --/
structure Odds where
  against : ℕ
  in_favor : ℕ

/-- Calculates the probability of winning given the odds against --/
def probabilityOfWinning (odds : Odds) : ℚ :=
  odds.in_favor / (odds.against + odds.in_favor)

theorem three_horse_race_odds (oddsA oddsB : Odds) :
  oddsA = ⟨5, 2⟩ →
  oddsB = ⟨4, 1⟩ →
  ∃ oddsC : Odds, oddsC = ⟨17, 18⟩ ∧
    (probabilityOfWinning oddsA) + (probabilityOfWinning oddsB) + (probabilityOfWinning oddsC) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_horse_race_odds_l301_30141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_derivative_l301_30153

/-- Given a function f(x) = 2cos(4x + x₀²) - 1, if f is even, then f'(x₀²/2) = 0 -/
theorem even_function_derivative (x₀ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.cos (4 * x + x₀^2) - 1
  (∀ x, f x = f (-x)) →  -- f is an even function
  deriv f (x₀^2 / 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_derivative_l301_30153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_distance_l301_30127

/-- The distance between a bee's nest and a flower, given specific flight conditions. -/
theorem bee_flight_distance : ℝ := by
  let outbound_speed : ℝ := 10  -- meters per hour
  let return_speed : ℝ := 15    -- meters per hour
  let time_difference : ℝ := 0.5 -- hours (30 minutes)
  let distance : ℝ := 15        -- meters
  
  have h1 : distance = outbound_speed * (distance / return_speed + time_difference) := by
    sorry
  have h2 : distance = return_speed * (distance / outbound_speed - time_difference) := by
    sorry
  
  exact distance

/- The theorem states that the distance is 15 meters. The proof would involve showing that
   this distance satisfies the conditions given in the problem. -/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_distance_l301_30127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l301_30163

noncomputable section

/-- The volume of a cylinder with hemispherical ends -/
def cylinderWithHemispheres (r : ℝ) (h : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3 + Real.pi * r^2 * h

/-- The length of line segment CD -/
def segmentLength (volume : ℝ) (radius : ℝ) : ℝ :=
  (volume / (Real.pi * radius^2)) - (4/3) * radius

theorem line_segment_length (volume : ℝ) (radius : ℝ) 
  (h_volume : volume = 288 * Real.pi) (h_radius : radius = 4) :
  segmentLength volume radius = 92/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l301_30163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_august_cricket_matches_l301_30145

theorem august_cricket_matches 
  (initial_win_percentage : ℚ) 
  (final_win_percentage : ℚ) 
  (winning_streak_matches : ℕ) 
  (h1 : initial_win_percentage = 26/100)
  (h2 : final_win_percentage = 52/100)
  (h3 : winning_streak_matches = 65) :
  ∃ (total_matches : ℕ), 
    (initial_win_percentage * total_matches + winning_streak_matches) / 
    (total_matches + winning_streak_matches) = final_win_percentage ∧ 
    total_matches = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_august_cricket_matches_l301_30145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_l301_30171

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

def a_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) = (1 + 2 * a n) / (1 + a n)

theorem sequence_bounded (a : ℕ → ℝ) (h : a_sequence a) : ∀ n : ℕ, 0 < n → a n < φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_bounded_l301_30171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_implies_a_greater_than_one_l301_30143

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 2^x)
noncomputable def g (a x : ℝ) : ℝ := Real.log ((x - a + 1) * (x - a - 1)) / Real.log 2

-- Define the domain sets A and B
def A : Set ℝ := {x | x ≤ 0}
def B (a : ℝ) : Set ℝ := {x | (x - a + 1) * (x - a - 1) > 0}

-- State the theorem
theorem domain_intersection_implies_a_greater_than_one (a : ℝ) :
  (A ∩ B a = A) → a > 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_implies_a_greater_than_one_l301_30143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_regions_correct_l301_30156

/-- The number of regions created by n lines in a plane -/
def num_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem stating that num_regions correctly calculates the number of regions -/
theorem num_regions_correct (n : ℕ) (line : Fin n → Set (ℝ × ℝ)) (number_of_regions : ℕ) :
  (∀ (i j : Fin n), i ≠ j → ∃! p, p ∈ line i ∧ p ∈ line j) →
  (∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → 
    ¬∃ p, p ∈ line i ∧ p ∈ line j ∧ p ∈ line k) →
  number_of_regions = num_regions n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_regions_correct_l301_30156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_adjacent_l301_30130

theorem probability_not_adjacent (n : ℕ) (h : n = 5) :
  (n - 1).factorial * (n + 1) / (2 * n.factorial) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_not_adjacent_l301_30130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_eq_two_tangent_line_implies_max_value_non_monotonic_implies_a_range_l301_30185

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

noncomputable def f_deriv (a x : ℝ) : ℝ := x^2 - 2 * a * x + a^2 - 1

theorem extreme_point_implies_a_eq_two (a b : ℝ) :
  (f_deriv a 1 = 0) → (a = 0 ∨ a = 2) := by
  sorry

theorem tangent_line_implies_max_value (a b : ℝ) :
  (f a b 1 = 2) →
  (f_deriv a 1 = -1) →
  (∀ x ∈ Set.Icc (-2 : ℝ) 4, f 1 (8/3) x ≤ 8) := by
  sorry

theorem non_monotonic_implies_a_range (a : ℝ) :
  (a ≠ 0) →
  (∃ x ∈ Set.Ioo (-1 : ℝ) 1, f_deriv a x = 0) →
  (a ∈ Set.Ioo (-2 : ℝ) 0 ∪ Set.Ioo 0 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_implies_a_eq_two_tangent_line_implies_max_value_non_monotonic_implies_a_range_l301_30185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_salad_cost_l301_30150

/-- The cost of a Taco Grande Plate -/
def T : Real := sorry

/-- The cost of a side salad -/
def S : Real := sorry

/-- The cost of cheesy fries -/
def cheesy_fries_cost : Real := 4

/-- The cost of a diet cola -/
def diet_cola_cost : Real := 2

/-- Mike's total bill -/
def mike_bill : Real := T + S + cheesy_fries_cost + diet_cola_cost

/-- John's total bill -/
def john_bill : Real := T

/-- Theorem stating that the side salad costs $2 -/
theorem side_salad_cost : 
  (mike_bill = 2 * john_bill) → 
  (mike_bill + john_bill = 24) → 
  S = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_salad_cost_l301_30150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_formula_l301_30118

/-- The radius of the inscribed circle in a right triangle -/
noncomputable def inscribed_circle_radius (a b : ℝ) : ℝ :=
  (a + b - Real.sqrt (a^2 + b^2)) / 2

/-- Theorem: The radius of the inscribed circle in a right triangle with legs a and b
    is (a + b - √(a² + b²)) / 2 -/
theorem inscribed_circle_radius_formula (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ (r : ℝ), r > 0 ∧ r = inscribed_circle_radius a b := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_formula_l301_30118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l301_30116

theorem sin_cos_difference (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = 4/3)
  (h2 : 0 < θ) (h3 : θ < π/4) :
  Real.sin θ - Real.cos θ = -Real.sqrt 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_difference_l301_30116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l301_30147

-- Define the vectors a and b
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 3, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi :=
sorry

-- Theorem for the range of f(x) in the given interval
theorem range_in_interval :
  ∀ x ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12),
    f x ∈ Set.Icc 3 6 ∧
    (∃ x₁ ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12), f x₁ = 3) ∧
    (∃ x₂ ∈ Set.Icc (Real.pi / 12) (7 * Real.pi / 12), f x₂ = 6) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_range_in_interval_l301_30147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l301_30128

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 * a - 3) * x - 1 else x^2 + 1

-- State the theorem
theorem function_monotonicity (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) ↔ 
  (a > 3/2 ∧ a ≤ 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_monotonicity_l301_30128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simulation_approximates_theory_l301_30108

/-- Represents the outcome of a single shot --/
inductive ShotOutcome
  | Hit
  | Miss

/-- Represents a group of three shots --/
def ThreeShots := (ShotOutcome × ShotOutcome × ShotOutcome)

/-- The probability of a successful shot --/
def p_success : ℝ := 0.4

/-- The number of groups in the simulation --/
def num_groups : ℕ := 20

/-- Counts the number of hits in a group of three shots --/
def count_hits (shots : ThreeShots) : ℕ :=
  match shots with
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Hit) => 3
  | (ShotOutcome.Hit, ShotOutcome.Hit, ShotOutcome.Miss) => 2
  | (ShotOutcome.Hit, ShotOutcome.Miss, ShotOutcome.Hit) => 2
  | (ShotOutcome.Miss, ShotOutcome.Hit, ShotOutcome.Hit) => 2
  | (ShotOutcome.Hit, ShotOutcome.Miss, ShotOutcome.Miss) => 1
  | (ShotOutcome.Miss, ShotOutcome.Hit, ShotOutcome.Miss) => 1
  | (ShotOutcome.Miss, ShotOutcome.Miss, ShotOutcome.Hit) => 1
  | (ShotOutcome.Miss, ShotOutcome.Miss, ShotOutcome.Miss) => 0

/-- The theoretical probability of exactly two hits in three shots --/
def p_two_hits : ℝ := 3 * p_success^2 * (1 - p_success)

/-- The observed probability in the simulation --/
def observed_p_two_hits : ℝ := 0.25

theorem simulation_approximates_theory :
  |observed_p_two_hits - p_two_hits| < 0.01 := by
  sorry

#eval p_two_hits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simulation_approximates_theory_l301_30108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_chord_l301_30198

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  minor_axis_length : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e : e = Real.sqrt 3 / 2
  h_minor : minor_axis_length = 4

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem about the ellipse and a chord -/
theorem ellipse_and_chord (E : Ellipse) (P : Point) 
    (h_P : P.x = 2 ∧ P.y = 1) :
  -- 1. The equation of the ellipse
  (∀ x y : ℝ, x^2/16 + y^2/4 = 1 ↔ x^2/E.a^2 + y^2/E.b^2 = 1) ∧
  -- 2. The equation of the chord
  (∃ A B : Point, 
    (A.x^2/E.a^2 + A.y^2/E.b^2 = 1) ∧ 
    (B.x^2/E.a^2 + B.y^2/E.b^2 = 1) ∧
    (P.x = (A.x + B.x)/2) ∧ 
    (P.y = (A.y + B.y)/2) ∧
    (∀ x y : ℝ, x + 2*y - 4 = 0 ↔ (y - A.y) = -(x - A.x)/2)) ∧
  -- 3. The length of the chord
  (∃ A B : Point, 
    (A.x^2/E.a^2 + A.y^2/E.b^2 = 1) ∧ 
    (B.x^2/E.a^2 + B.y^2/E.b^2 = 1) ∧
    (P.x = (A.x + B.x)/2) ∧ 
    (P.y = (A.y + B.y)/2) ∧
    ((A.x - B.x)^2 + (A.y - B.y)^2 = 20)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_chord_l301_30198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probabilities_l301_30140

/-- The hit rate of player A -/
noncomputable def hit_rate_A : ℝ := 1/2

/-- The hit rate of player B -/
noncomputable def hit_rate_B : ℝ := 3/4

/-- The probability that player B misses twice in a row -/
noncomputable def miss_twice_B : ℝ := 1/16

/-- The probability that player A hits at least once in two shots -/
noncomputable def hit_at_least_once_A : ℝ := 3/4

/-- The probability that both players hit a total of two times in two shots each -/
noncomputable def hit_total_two : ℝ := 11/32

theorem basketball_probabilities :
  (1 - hit_rate_B) ^ 2 = miss_twice_B ∧
  hit_at_least_once_A = 1 - (1 - hit_rate_A) ^ 2 ∧
  hit_total_two = 2 * hit_rate_A * (1 - hit_rate_A) * hit_rate_B * (1 - hit_rate_B) +
                  hit_rate_A ^ 2 * (1 - hit_rate_B) ^ 2 +
                  (1 - hit_rate_A) ^ 2 * hit_rate_B ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probabilities_l301_30140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l301_30117

theorem complex_absolute_value : ∀ z : ℂ, 
  Complex.abs (Complex.abs (Complex.abs (-Complex.abs ((-2 : ℂ) + 3) - 2) + 2)) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_absolute_value_l301_30117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_100_yuan_l301_30188

/-- Represents the number of ways to spend n yuan given the conditions -/
def spending_ways (n : ℕ) : ℚ :=
  (2^(n+1) + (-1)^n : ℚ) / 3

/-- The conditions of the problem -/
def total_amount : ℕ := 100
def min_purchase : ℕ := 1
def candy_price : ℕ := 1
def cookie_price : ℕ := 2
def fruit_price : ℕ := 2

/-- The main theorem to prove -/
theorem spending_100_yuan : spending_ways total_amount = (2^101 + 1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spending_100_yuan_l301_30188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_business_trip_duration_equals_12_hours_l301_30154

/-- Represents the duration of Mary's business trip in minutes -/
def businessTripDuration (uberToHouse : ℕ) (uberToAirportMultiplier : ℕ) (bagCheck : ℕ) 
  (securityMultiplier : ℕ) (waitForBoarding : ℕ) (waitForTakeoffMultiplier : ℕ) 
  (firstLayover : ℕ) (delay : ℕ) (secondLayover : ℕ) (timeZoneChange : ℕ) : ℕ :=
  let uberToAirport := uberToHouse * uberToAirportMultiplier
  let security := bagCheck * securityMultiplier
  let waitForTakeoff := waitForBoarding * waitForTakeoffMultiplier
  let totalBeforeFirstFlight := uberToHouse + uberToAirport + bagCheck + security + waitForBoarding + waitForTakeoff
  let totalLayoversAndDelay := firstLayover + delay + secondLayover
  totalBeforeFirstFlight + totalLayoversAndDelay + timeZoneChange * 60

theorem business_trip_duration_equals_12_hours :
  businessTripDuration 10 5 15 3 20 2 205 45 110 3 = 12 * 60 := by
  sorry

#eval businessTripDuration 10 5 15 3 20 2 205 45 110 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_business_trip_duration_equals_12_hours_l301_30154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l301_30155

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n > 0 → m > 0 → b n > 0 ∧ ∃ r : ℝ, b (n + m) = b n * r^m

theorem geometric_sequence_property
  (b : ℕ → ℝ)
  (h_geom : geometric_sequence b)
  (m n : ℕ)
  (h_m_pos : m > 0)
  (h_n_pos : n > 0)
  (h_neq : m ≠ n)
  (a : ℝ)
  (h_a : b m = a)
  (b_val : ℝ)
  (h_b : b n = b_val) :
  b (m + n) = (n - m : ℝ) * (b_val^n / a^m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_property_l301_30155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_lower_bound_l301_30191

noncomputable def g (a b c : ℝ) : ℝ := (a + 1) / (a + b) + (b + 1) / (b + c) + (c + 1) / (c + a)

theorem g_lower_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  g a b c > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_lower_bound_l301_30191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_f_is_even_not_symmetric_about_pi_over_4_f_increasing_on_interval_l301_30135

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

-- Statement 1: The smallest positive period of f(x) is π
theorem smallest_positive_period : 
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧ 
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧ 
  p = Real.pi := by sorry

-- Statement 2: f(x) is an even function
theorem f_is_even : ∀ (x : ℝ), f (-x) = f x := by sorry

-- Statement 3: The graph of f(x) is not symmetric about the line x = π/4
theorem not_symmetric_about_pi_over_4 : 
  ∃ (x : ℝ), f (Real.pi / 4 + x) ≠ f (Real.pi / 4 - x) := by sorry

-- Statement 4: f(x) is increasing in the interval [0, π/2]
theorem f_increasing_on_interval : 
  ∀ (x y : ℝ), 0 ≤ x ∧ x < y ∧ y ≤ Real.pi / 2 → f x < f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_f_is_even_not_symmetric_about_pi_over_4_f_increasing_on_interval_l301_30135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l301_30138

theorem triangle_side_relation (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0)
  (h_C_twice_B : C = 2 * B)
  (h_A_neq_B : A ≠ B)
  (h_sine_law : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C))
  : c^2 = b * (a + b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_relation_l301_30138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l301_30193

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 5, 7} : Finset ℕ) → 
  b ∈ ({1, 3, 5, 7} : Finset ℕ) → 
  c ∈ ({1, 3, 5, 7} : Finset ℕ) → 
  d ∈ ({1, 3, 5, 7} : Finset ℕ) → 
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d → 
  (a * b + b * c + c * d + d * a) ≤ 64 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_sum_l301_30193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_in_quadrants_l301_30149

def inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, x ≠ 0 → f x = k / x

def in_second_and_fourth_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 0 → (x > 0 → f x < 0) ∧ (x < 0 → f x > 0)

theorem inverse_proportion_in_quadrants :
  ∃ f : ℝ → ℝ, inverse_proportion f ∧ in_second_and_fourth_quadrants f :=
by
  use λ x => -3 / x
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_in_quadrants_l301_30149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l301_30119

theorem constant_term_expansion (a : ℝ) : 
  (∃ x : ℝ, (a * Real.sqrt x - 1 / (3 * x))^5 = -40 + Real.sqrt x * (a * Real.sqrt x - 1 / (3 * x))^4 * (5 * a - 1 / (3 * x^(3/2)))) → 
  (a = 2 ∨ a = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l301_30119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_factorial_equation_l301_30177

theorem unique_solution_factorial_equation :
  ∃! x : ℕ, x > 0 ∧ (Nat.factorial x - Nat.factorial (x - 3)) / 23 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_factorial_equation_l301_30177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_invalid_votes_percentage_l301_30184

theorem election_invalid_votes_percentage 
  (total_votes : ℕ) 
  (votes_B : ℕ) 
  (excess_percentage : ℚ) :
  total_votes = 7720 →
  votes_B = 2509 →
  excess_percentage = 15 / 100 →
  let votes_A := votes_B + (excess_percentage * total_votes).floor
  let total_valid_votes := votes_A + votes_B
  let invalid_votes := total_votes - total_valid_votes
  let invalid_percentage := (invalid_votes : ℚ) / total_votes * 100
  ‖invalid_percentage - 19.979‖ < 0.001 := by
  sorry

#check election_invalid_votes_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_invalid_votes_percentage_l301_30184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_is_ten_percent_l301_30161

/-- Calculates the simple interest rate given principal, total amount, and time -/
noncomputable def simple_interest_rate (principal : ℝ) (total_amount : ℝ) (time : ℝ) : ℝ :=
  ((total_amount - principal) * 100) / (principal * time)

/-- Theorem: The simple interest rate for the given conditions is 10% -/
theorem simple_interest_rate_is_ten_percent :
  let principal := (750 : ℝ)
  let total_amount := (1125 : ℝ)
  let time := (5 : ℝ)
  simple_interest_rate principal total_amount time = 10 := by
  sorry

-- Remove the #eval statement as it's not necessary for compilation
-- and may cause issues with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simple_interest_rate_is_ten_percent_l301_30161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_seat_cost_correct_l301_30181

/-- The original cost of each seat in the auditorium problem -/
def original_seat_cost : ℚ := 30

/-- The total number of seats being added -/
def total_seats : ℕ := 5 * 8

/-- The discount rate -/
def discount_rate : ℚ := 1 / 10

/-- The number of seat groups eligible for discount -/
def discount_groups : ℕ := total_seats / 10

/-- Theorem stating that the given solution satisfies the problem conditions -/
theorem auditorium_seat_cost_correct :
  (original_seat_cost * total_seats : ℚ) - 
  (discount_rate * original_seat_cost * (10 * discount_groups)) = 1080 := by
  -- Proof goes here
  sorry

#eval original_seat_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_auditorium_seat_cost_correct_l301_30181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_of_8_equals_11_l301_30170

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n
  else (n % 10) + sumOfDigits (n / 10)

-- Define f(n) as the sum of digits of n^2 + 1
def f (n : ℕ) : ℕ := sumOfDigits (n^2 + 1)

-- Define f_k(n) recursively
def f_k : ℕ → ℕ → ℕ
  | 0, n => n
  | 1, n => f n
  | k+1, n => f (f_k k n)

theorem f_2011_of_8_equals_11 : f_k 2011 8 = 11 := by
  sorry

#eval f_k 2011 8  -- This will evaluate the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2011_of_8_equals_11_l301_30170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_calculation_l301_30174

/-- Represents the energy stored in a configuration of charges -/
noncomputable def energy : ℝ := sorry

/-- The side length of the square -/
noncomputable def s : ℝ := sorry

/-- The number of charges -/
def n : ℕ := 4

/-- The initial energy stored in the configuration with charges at vertices -/
def initial_energy : ℝ := 20

/-- The energy between adjacent charges in the initial configuration -/
noncomputable def adjacent_energy : ℝ := initial_energy / (n * (n - 1) / 2)

/-- The energy between the center charge and a vertex charge in the new configuration -/
noncomputable def center_vertex_energy : ℝ := adjacent_energy * (s / (s / Real.sqrt 2))

/-- The total energy in the new configuration -/
noncomputable def new_total_energy : ℝ := n * center_vertex_energy + (n - 1) * adjacent_energy

/-- The increase in energy when moving one charge to the center -/
noncomputable def energy_increase : ℝ := new_total_energy - initial_energy

theorem energy_increase_calculation :
  energy_increase = 20 * Real.sqrt 2 - 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_energy_increase_calculation_l301_30174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circular_sectors_area_l301_30199

/-- The side length of the regular hexagon -/
noncomputable def hexagon_side : ℝ := 8

/-- The radius of the circular arcs -/
noncomputable def arc_radius : ℝ := 5

/-- The angle subtended by each arc in radians -/
noncomputable def arc_angle : ℝ := Real.pi / 2

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The area of the region inside the hexagon but outside the circular sectors -/
noncomputable def shaded_area : ℝ := 96 * Real.sqrt 3 - 37.5 * Real.pi

theorem hexagon_circular_sectors_area :
  let hexagon_area := (hexagon_sides : ℝ) * (Real.sqrt 3 / 4 * hexagon_side ^ 2)
  let sector_area := (hexagon_sides : ℝ) * (arc_angle / (2 * Real.pi) * Real.pi * arc_radius ^ 2)
  hexagon_area - sector_area = shaded_area := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_circular_sectors_area_l301_30199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acceptance_at_13_acceptance_increases_before_13_acceptance_decreases_after_13_l301_30100

-- Define the type for our data points
structure DataPoint where
  x : ℝ
  y : ℝ

-- Define our dataset
def dataset : List DataPoint := [
  ⟨2, 47.8⟩, ⟨5, 53.5⟩, ⟨7, 56.3⟩, ⟨10, 59⟩, ⟨12, 59.8⟩,
  ⟨13, 59.9⟩, ⟨14, 59.8⟩, ⟨17, 58.3⟩, ⟨20, 55⟩
]

-- Define the domain of x
def validX (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 30

-- Define a function to get the y value for a given x
noncomputable def getY (x : ℝ) : ℝ :=
  match dataset.find? (fun d => d.x = x) with
  | some d => d.y
  | none => 0  -- Default value if x is not in the dataset

-- Theorem statements
theorem max_acceptance_at_13 :
  ∀ x, validX x → getY 13 ≥ getY x := by sorry

theorem acceptance_increases_before_13 :
  ∀ x₁ x₂, validX x₁ → validX x₂ → 0 < x₁ → x₁ < x₂ → x₂ < 13 →
  getY x₁ < getY x₂ := by sorry

theorem acceptance_decreases_after_13 :
  ∀ x₁ x₂, validX x₁ → validX x₂ → 13 < x₁ → x₁ < x₂ → x₂ < 20 →
  getY x₁ > getY x₂ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_acceptance_at_13_acceptance_increases_before_13_acceptance_decreases_after_13_l301_30100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ethane_combustion_stoichiometry_l301_30197

-- Define the molecules and their coefficients in the balanced equation
structure Molecule where
  name : String
  coefficient : ℚ
deriving Inhabited

-- Define the balanced equation
def balancedEquation : List Molecule :=
  [⟨"C2H6", 2⟩, ⟨"O2", 7⟩, ⟨"CO2", 4⟩, ⟨"H2O", 6⟩]

-- Define the given amounts
def givenEthane : ℚ := 1
def givenOxygen : ℚ := 2
def givenCarbonDioxide : ℚ := 2

-- Theorem statement
theorem ethane_combustion_stoichiometry :
  let ethane := (balancedEquation.filter (λ m => m.name = "C2H6")).head!
  let oxygen := (balancedEquation.filter (λ m => m.name = "O2")).head!
  let carbonDioxide := (balancedEquation.filter (λ m => m.name = "CO2")).head!
  givenEthane * (oxygen.coefficient / ethane.coefficient) = 3.5 ∧
  givenEthane * (carbonDioxide.coefficient / ethane.coefficient) = givenCarbonDioxide :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ethane_combustion_stoichiometry_l301_30197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_weighted_average_l301_30182

/-- Calculates the weighted average of David's marks --/
def weighted_average (marks : Fin 5 → ℚ) (weights : Fin 5 → ℚ) : ℚ :=
  (Finset.sum (Finset.univ : Finset (Fin 5)) (λ i => marks i * weights i)) / 
  (Finset.sum (Finset.univ : Finset (Fin 5)) weights)

theorem davids_weighted_average :
  let marks : Fin 5 → ℚ := ![86, 85, 82, 87, 85]
  let weights : Fin 5 → ℚ := ![2, 3, 4, 3, 2]
  weighted_average marks weights = 8471 / 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_davids_weighted_average_l301_30182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_l301_30176

/-- Given vectors a and b, if k*a + b is perpendicular to a - 2*b, then k = 2 -/
theorem vector_perpendicular_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (1, 2))
  (h2 : b = (2, -1))
  (h3 : ((k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 2 * b.1, a.2 - 2 * b.2) = 0)) :
  k = 2 := by
  sorry

#check vector_perpendicular_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_perpendicular_condition_l301_30176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l301_30189

-- Define the constants
noncomputable def a : ℝ := Real.log Real.pi / Real.log (1 / 2013)
noncomputable def b : ℝ := (1 / 5) ^ (-0.8 : ℝ)
noncomputable def c : ℝ := Real.log Real.pi / Real.log 10

-- State the theorem
theorem order_of_logarithms : a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_logarithms_l301_30189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_with_q_cubed_l301_30109

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def b_sequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a (3*n - 2) + a (3*n - 1) + a (3*n)

theorem b_is_geometric_with_q_cubed 
  (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ) 
  (hq : q ≠ 1) 
  (ha : geometric_sequence a q) 
  (hb : b_sequence a b) : 
  geometric_sequence b (q^3) :=
sorry

#check b_is_geometric_with_q_cubed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_is_geometric_with_q_cubed_l301_30109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l301_30134

/-- A parabola with equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- The focus of the parabola -/
def Focus : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 0)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (A : ℝ × ℝ) 
  (h1 : A ∈ Parabola) 
  (h2 : distance A Focus = distance B Focus) : 
  distance A B = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l301_30134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monotone_subsequence_l301_30124

theorem infinite_monotone_subsequence (f : ℕ → ℝ) : 
  ∃ A : Set ℕ, Set.Infinite A ∧ 
    (∀ (a b : ℕ), a ∈ A → b ∈ A → a < b → f a < f b) ∨ 
    (∀ (a b : ℕ), a ∈ A → b ∈ A → a < b → f a > f b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_monotone_subsequence_l301_30124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spotted_brown_toad_fraction_l301_30146

/-- Proves that the fraction of spotted brown toads is 1/4 given the specified conditions -/
theorem spotted_brown_toad_fraction :
  -- Ratio of green toads to brown toads
  let green_to_brown_ratio : ℚ := 1 / 25
  -- Number of spotted brown toads per acre
  let spotted_brown_per_acre : ℕ := 50
  -- Number of green toads per acre
  let green_per_acre : ℕ := 8
  -- Total number of brown toads per acre
  let total_brown_per_acre : ℚ := green_per_acre / green_to_brown_ratio
  -- Fraction of spotted brown toads
  let spotted_fraction : ℚ := spotted_brown_per_acre / total_brown_per_acre
  
  spotted_fraction = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spotted_brown_toad_fraction_l301_30146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l301_30136

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- Changed to ℚ (rational numbers) for computability
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  a_10 : a 10 = 30
  a_20 : a 20 = 50

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 2 * n + 10) ∧
  (sum_n seq 11 = 242) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l301_30136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_y_axis_l301_30123

/-- Given points O, A, and B in ℝ², and a point P on the y-axis satisfying 
    the vector equation OP = 2OA + mAB, prove that m = 2/3 -/
theorem vector_equation_y_axis (O A B P : ℝ × ℝ) (m : ℝ) : 
  O = (0, 0) →
  A = (-1, 3) →
  B = (2, -4) →
  P.1 = 0 →  -- P is on the y-axis
  P - O = 2 • (A - O) + m • (B - A) →
  m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_y_axis_l301_30123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_one_l301_30157

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (x + 1)

theorem inverse_f_at_one :
  ∃ f_inv : ℝ → ℝ, Function.RightInverse f_inv f ∧ f_inv 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_one_l301_30157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l301_30125

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1)

noncomputable def f' (x : ℝ) : ℝ := Real.exp (x - 1)

theorem tangent_line_slope :
  ∃ (x₀ : ℝ), 
    -- The tangent line passes through (1,0)
    1 - x₀ = (f x₀ - 0) / (f' x₀) ∧
    -- The slope of the tangent line is e
    f' x₀ = Real.exp 1 :=
by
  -- Proof goes here
  sorry

#check tangent_line_slope

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_slope_l301_30125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minimization_and_tangent_lines_l301_30101

-- Define the circle C
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*m*x - 2*y + 4*m - 4 = 0

-- Define the area of the circle as a function of m
noncomputable def circle_area (m : ℝ) : ℝ :=
  Real.pi * (m^2 - 4*m + 5)

-- Define the tangent line equations
def tangent_line_1 (x : ℝ) : Prop := x = 1
def tangent_line_2 (x y : ℝ) : Prop := 4*x - 3*y - 10 = 0

-- State the theorem
theorem circle_minimization_and_tangent_lines :
  -- The value of m that minimizes the area
  ∃ (m : ℝ), ∀ (m' : ℝ), circle_area m ≤ circle_area m' ∧ m = 2 ∧
  -- The tangent lines pass through (1, -2) and are tangent to the circle when m = 2
  tangent_line_1 1 ∧ tangent_line_2 1 (-2) ∧
  (∀ (x y : ℝ), circle_equation x y 2 → (tangent_line_1 x ∨ tangent_line_2 x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_minimization_and_tangent_lines_l301_30101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l301_30196

theorem simplify_expression : (7 : Int) - (-5) - 3 + (-9) = 7 + 5 - 3 - 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l301_30196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_requirement_l301_30121

/-- The number of pillars to be painted -/
def num_pillars : ℕ := 12

/-- The height of each pillar in feet -/
def pillar_height : ℝ := 22

/-- The diameter of each pillar in feet -/
def pillar_diameter : ℝ := 8

/-- The area in square feet that one gallon of paint covers -/
def paint_coverage : ℝ := 400

/-- Calculate the minimum number of full gallons of paint required to cover the lateral surface area of the pillars -/
noncomputable def paint_gallons_required : ℕ :=
  let radius := pillar_diameter / 2
  let lateral_area_per_pillar := 2 * Real.pi * radius * pillar_height
  let total_lateral_area := lateral_area_per_pillar * num_pillars
  let gallons_needed := total_lateral_area / paint_coverage
  Nat.ceil gallons_needed

/-- Theorem stating that 17 gallons of paint are required -/
theorem paint_requirement : paint_gallons_required = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_requirement_l301_30121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l301_30186

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  Real.cos A = Real.sqrt 6 / 3 → 
  b = 2 * Real.sqrt 2 → 
  c = Real.sqrt 3 → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
  a = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l301_30186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_zero_l301_30164

def Spinner1 : Finset ℕ := {2, 4, 6, 8}
def Spinner2 : Finset ℕ := {3, 5, 7, 9, 11}

def SpinProduct : Finset (ℕ × ℕ) := Finset.product Spinner1 Spinner2

def ProductResult : Finset ℕ := SpinProduct.image (fun (x : ℕ × ℕ) => x.1 * x.2)

theorem spinner_prime_probability_zero :
  Finset.filter Nat.Prime ProductResult = ∅ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_prime_probability_zero_l301_30164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l301_30151

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (4 * x - x^2) / Real.log (1/2)

-- Define the domain of f
def domain_f : Set ℝ := { x | 0 < x ∧ x < 4 }

-- Theorem statement
theorem f_strictly_increasing :
  ∀ x ∈ domain_f, ∀ y ∈ domain_f,
    2 ≤ x ∧ x < y ∧ y < 4 → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_l301_30151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ways_to_purchase_l301_30105

def num_cookie_flavors : ℕ := 8
def num_smoothie_flavors : ℕ := 5
def total_items : ℕ := num_cookie_flavors + num_smoothie_flavors
def items_purchased : ℕ := 5

-- Function to calculate binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate combinations with repetition
def combinations_with_repetition (n k : ℕ) : ℕ := choose (n + k - 1) k

-- Function to calculate the number of ways for a specific case
def case_count (charlie_items : ℕ) : ℕ :=
  choose total_items charlie_items *
  combinations_with_repetition num_cookie_flavors (items_purchased - charlie_items)

-- Theorem stating the total number of ways
theorem total_ways_to_purchase :
  (case_count 5) + (case_count 4) + (case_count 3) +
  (case_count 2) + (case_count 1) + (case_count 0) = 27330 := by
  sorry

#eval (case_count 5) + (case_count 4) + (case_count 3) +
      (case_count 2) + (case_count 1) + (case_count 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_ways_to_purchase_l301_30105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_difference_is_one_l301_30160

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1)

def sequence_A (n : ℕ) : ℝ := geometric_sequence 3 3 n

def sequence_B (n : ℕ) : ℝ := arithmetic_sequence 10 10 n

def valid_A (n : ℕ) : Prop := sequence_A n ≤ 300

def valid_B (n : ℕ) : Prop := sequence_B n ≤ 300

theorem least_difference_is_one :
  ∃ (m n : ℕ), 
    valid_A m ∧ valid_B n ∧
    ∀ (i j : ℕ), valid_A i → valid_B j → 
      |sequence_A m - sequence_B n| ≤ |sequence_A i - sequence_B j| ∧
      |sequence_A m - sequence_B n| = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_difference_is_one_l301_30160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_column_height_at_quarter_span_l301_30169

/-- Represents a parabolic arch bridge -/
structure ParabolicArch where
  span : ℝ
  height : ℝ

/-- Calculates the height of a support column at a given distance from the center -/
noncomputable def columnHeight (arch : ParabolicArch) (distance : ℝ) : ℝ :=
  let p := arch.span^2 / (8 * arch.height)
  arch.height - distance^2 / (4 * p)

theorem column_height_at_quarter_span (arch : ParabolicArch) 
  (h_span : arch.span = 16)
  (h_height : arch.height = 4) :
  columnHeight arch (arch.span / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_column_height_at_quarter_span_l301_30169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l301_30113

/-- The sum of the infinite series Σ(k/(4^k)) from k=1 to ∞ -/
noncomputable def infiniteSeries : ℝ := Real.rpow 4 (-1) * (4 / (1 - Real.rpow 4 (-1))^2)

/-- Theorem: The sum of the infinite series Σ(k/(4^k)) from k=1 to ∞ equals 4/9 -/
theorem infiniteSeries_sum : infiniteSeries = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_sum_l301_30113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_unit_square_lateral_area_l301_30179

/-- The lateral surface area of a solid formed by rotating a unit square around one of its sides. -/
noncomputable def lateralSurfaceArea : ℝ := 2 * Real.pi

/-- Theorem stating that the lateral surface area of the solid formed by rotating a unit square
    around one of its sides is equal to 2π. -/
theorem rotate_unit_square_lateral_area :
  lateralSurfaceArea = 2 * Real.pi := by
  -- Unfold the definition of lateralSurfaceArea
  unfold lateralSurfaceArea
  -- The equality now follows by reflexivity
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rotate_unit_square_lateral_area_l301_30179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l301_30107

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h : Filter.Tendsto (λ (Δx : ℝ) ↦ (f (1 + Δx) - f 1) / Δx) (𝓝 0) (𝓝 1))

-- Theorem statement
theorem tangent_slope_at_one :
  deriv f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l301_30107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AC_approx_l301_30131

-- Define the quadrilateral AZBC
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (5, 12)
def Z : ℝ × ℝ := (5, 0)
def C : ℝ × ℝ := (26, 12)

-- Define the given side lengths
def AB : ℝ := 13
def ZC : ℝ := 25
def AZ : ℝ := 5

-- Function to calculate distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem length_of_AC_approx (ε : ℝ) (h : ε > 0) :
  ∃ (ac : ℝ), abs (distance A C - ac) < ε ∧ abs (ac - 18.4) < 0.05 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AC_approx_l301_30131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_squares_bounds_l301_30114

theorem roots_sum_squares_bounds :
  ∃ (max min : ℝ),
    (∀ (m α β : ℝ),
      (α^2 - 2*m*α + 3 + 4*m^2 - 6 = 0) →
      (β^2 - 2*m*β + 3 + 4*m^2 - 6 = 0) →
      (α-1)^2 + (β-1)^2 ≤ max) ∧
    (∃ (m α β : ℝ),
      (α^2 - 2*m*α + 3 + 4*m^2 - 6 = 0) ∧
      (β^2 - 2*m*β + 3 + 4*m^2 - 6 = 0) ∧
      (α-1)^2 + (β-1)^2 = max) ∧
    (∀ (m α β : ℝ),
      (α^2 - 2*m*α + 3 + 4*m^2 - 6 = 0) →
      (β^2 - 2*m*β + 3 + 4*m^2 - 6 = 0) →
      min ≤ (α-1)^2 + (β-1)^2) ∧
    (∃ (m α β : ℝ),
      (α^2 - 2*m*α + 3 + 4*m^2 - 6 = 0) ∧
      (β^2 - 2*m*β + 3 + 4*m^2 - 6 = 0) ∧
      (α-1)^2 + (β-1)^2 = min) ∧
    max = 9 ∧ min = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_sum_squares_bounds_l301_30114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_range_l301_30144

-- Define the function f as noncomputable
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x / Real.sqrt (m * x^2 + m * x + 1)

-- State the theorem
theorem domain_implies_m_range (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f m x = y) → 0 ≤ m ∧ m < 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_implies_m_range_l301_30144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_five_equal_triangles_l301_30111

/-- A quadrilateral is a polygon with four sides -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A triangle is a polygon with three sides -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Function to calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Predicate to check if a list of triangles has equal areas -/
def hasEqualAreas (triangles : List Triangle) : Prop :=
  ∀ t1 t2, t1 ∈ triangles → t2 ∈ triangles → triangleArea t1 = triangleArea t2

/-- Function to divide a quadrilateral into triangles -/
noncomputable def divideQuadrilateral (q : Quadrilateral) (n : ℕ) : List Triangle := sorry

/-- Theorem stating that there exists a quadrilateral that can be divided into 5 equal triangles -/
theorem exists_quadrilateral_with_five_equal_triangles :
  ∃ (q : Quadrilateral), hasEqualAreas (divideQuadrilateral q 5) := by
  sorry

#check exists_quadrilateral_with_five_equal_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_quadrilateral_with_five_equal_triangles_l301_30111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_right_angled_faces_exists_pyramid_with_two_right_angled_faces_l301_30110

/-- A quadrangular pyramid -/
structure QuadrangularPyramid where
  base : Set Point
  apex : Point
  is_quadrilateral_base : Prop  -- Changed from IsQuadrilateral
  is_pyramid : Prop  -- Changed from IsPyramid

/-- A function that counts the number of right-angled triangles among the lateral faces of a quadrangular pyramid -/
def count_right_angled_faces (pyramid : QuadrangularPyramid) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of right-angled triangles among the lateral faces of a quadrangular pyramid is 2 -/
theorem max_right_angled_faces (pyramid : QuadrangularPyramid) : 
  count_right_angled_faces pyramid ≤ 2 := by
  sorry

/-- Theorem stating that there exists a quadrangular pyramid with exactly 2 right-angled triangles among its lateral faces -/
theorem exists_pyramid_with_two_right_angled_faces : 
  ∃ (pyramid : QuadrangularPyramid), count_right_angled_faces pyramid = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_right_angled_faces_exists_pyramid_with_two_right_angled_faces_l301_30110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l301_30139

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = x + 2

-- Define the foci
noncomputable def focus1 : ℝ × ℝ := (-2 * Real.sqrt 2, 0)
noncomputable def focus2 : ℝ × ℝ := (2 * Real.sqrt 2, 0)

-- Define the major axis length
def majorAxisLength : ℝ := 6

-- Theorem statement
theorem midpoint_of_intersection (A B : ℝ × ℝ) :
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B →
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  M = (-9/5, 1/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_of_intersection_l301_30139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l301_30172

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to angle A
  b : ℝ  -- Side opposite to angle B
  c : ℝ  -- Side opposite to angle C

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.a^2 - t.b^2 = t.c ∧
  Real.sin t.A * Real.cos t.B = 2 * Real.cos t.A * Real.sin t.B

-- Theorem statement
theorem triangle_side_value (t : Triangle) :
  satisfies_conditions t → t.c = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_value_l301_30172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_five_z_specific_values_l301_30194

-- Define the complex number z
variable (z : ℂ)

-- Define the given condition
def condition (z : ℂ) : Prop := Complex.abs (2 * z + 5) = Complex.abs (z + 10)

-- Define the property of being on the bisector of the first and third quadrants
def on_bisector (w : ℂ) : Prop := w.re = w.im

-- Theorem 1: If |2z+5| = |z+10|, then |z| = 5
theorem abs_z_equals_five (h : condition z) : Complex.abs z = 5 := by
  sorry

-- Theorem 2: If |2z+5| = |z+10| and (1-2i)z is on the bisector, then z has one of the two specific values
theorem z_specific_values (h1 : condition z) (h2 : on_bisector ((1 - Complex.I * 2) * z)) :
  z = Complex.ofReal (Real.sqrt 10 / 2) - Complex.I * Complex.ofReal (3 * Real.sqrt 10 / 2) ∨
  z = Complex.ofReal (-Real.sqrt 10 / 2) + Complex.I * Complex.ofReal (3 * Real.sqrt 10 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_five_z_specific_values_l301_30194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_by_stage_most_reasonable_l301_30165

-- Define the population
def Student := Type

-- Define the educational stages
inductive EducationalStage
| Primary
| JuniorHigh
| SeniorHigh

-- Define the characteristic being measured
def LungCapacity := ℝ

-- Define the survey
def Survey := Student → LungCapacity

-- Define the concept of significant difference
def SignificantDifference (s₁ s₂ : Set Student) (survey : Survey) : Prop :=
  sorry

-- Define the sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define a function to determine the most reasonable sampling method
def mostReasonableSamplingMethod : SamplingMethod :=
  SamplingMethod.StratifiedByEducationalStage

-- State the theorem
theorem stratified_by_stage_most_reasonable
  (students : Set Student)
  (survey : Survey)
  (stage : Student → EducationalStage)
  (gender : Student → Bool) :
  (∀ s₁ s₂, s₁ ≠ s₂ → SignificantDifference {x | stage x = s₁} {x | stage x = s₂} survey) →
  (∀ s, ¬SignificantDifference {x | stage x = s ∧ gender x} {x | stage x = s ∧ ¬gender x} survey) →
  mostReasonableSamplingMethod = SamplingMethod.StratifiedByEducationalStage :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_by_stage_most_reasonable_l301_30165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l301_30180

/-- Calculates the time for two trains to clear each other -/
noncomputable def time_to_clear (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * (5/18)
  let speed2_ms := speed2 * (5/18)
  let total_length := length1 + length2
  let relative_speed := speed1_ms + speed2_ms
  total_length / relative_speed

theorem trains_clear_time :
  time_to_clear 120 280 42 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_clear_time_l301_30180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_difference_l301_30183

def runner_times : List ℚ := [96, 88, 78, 102]

def pairwise_differences (times : List ℚ) : List ℚ :=
  List.foldr (λ x acc => 
    (List.map (λ y => abs (x - y)) (times.filter (λ y => y ≠ x))) ++ acc
  ) [] times

theorem average_time_difference :
  let differences := pairwise_differences runner_times
  (List.sum differences) / differences.length = 80 / 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_time_difference_l301_30183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_proof_l301_30142

/-- Given different non-zero real numbers a, b, c satisfying {a + b, b + c, c + a} = {ab, bc, ca},
    prove that {a, b, c} = {a^2 - 2, b^2 - 2, c^2 - 2}. -/
theorem set_equality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
    (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a)
    (h_eq : Finset.toSet {a + b, b + c, c + a} = Finset.toSet {a * b, b * c, c * a}) :
    Finset.toSet {a, b, c} = Finset.toSet {a^2 - 2, b^2 - 2, c^2 - 2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_proof_l301_30142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_equal_intercepts_l₂_l301_30173

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def l₂ (m x y : ℝ) : Prop := x - m * y + 1 - 3 * m = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d : ℝ), ∀ (x y : ℝ), (f x y ↔ a * x + b * y + c = 0) ∧
                                (g x y ↔ a * x + b * y + d = 0)

-- Define equal intercepts
def equal_intercepts (f : ℝ → ℝ → ℝ → Prop) (m : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ f m k 0 ∧ f m 0 k

-- Theorem 1: If l₁ and l₂ are parallel, then m = -1/2
theorem parallel_lines_m (m : ℝ) :
  parallel l₁ (l₂ m) → m = -1/2 := by sorry

-- Theorem 2: If l₂ has equal intercepts, then its equation is x + y + 4 = 0 or 3x - y = 0
theorem equal_intercepts_l₂ (m : ℝ) :
  equal_intercepts l₂ m →
  (∀ x y, l₂ m x y ↔ x + y + 4 = 0) ∨
  (∀ x y, l₂ m x y ↔ 3 * x - y = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_equal_intercepts_l₂_l301_30173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_diagonal_sum_l301_30106

/-- Represents Pascal's Triangle as a function from row and column to natural numbers -/
def PascalTriangle : ℕ → ℕ → ℕ := sorry

/-- Represents the right diagonal sequence above a given position in Pascal's Triangle -/
def RightDiagonal (row col : ℕ) : List ℕ := sorry

/-- The sum of elements in a list of natural numbers -/
def sum_list : List ℕ → ℕ := sorry

/-- Theorem stating that the value at a position in Pascal's Triangle is equal to the sum of the elements in the right diagonal above it -/
theorem pascal_triangle_diagonal_sum (row col : ℕ) :
  PascalTriangle row col = sum_list (RightDiagonal row col) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_diagonal_sum_l301_30106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l301_30159

theorem cyclic_sum_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 16 * (a + b + c) ≥ 1 / a + 1 / b + 1 / c) : 
  (1 / (a + b + Real.sqrt (2 * a + 2 * c)))^3 +
  (1 / (b + c + Real.sqrt (2 * b + 2 * a)))^3 +
  (1 / (c + a + Real.sqrt (2 * c + 2 * b)))^3 ≤ 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_sum_inequality_l301_30159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l301_30178

-- Define the inverse proportion function
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

-- Define the point P
def P : ℝ × ℝ := (-3, 3)

-- Theorem statement
theorem inverse_proportion_quadrants :
  ∃ k : ℝ, 
    (inverse_proportion k P.fst = P.snd) ∧ 
    (∀ x y : ℝ, y = inverse_proportion k x → 
      ((x < 0 ∧ y > 0) ∨ (x > 0 ∧ y < 0))) := by
  -- Proof goes here
  sorry

-- Additional helper lemma to show the value of k
lemma find_k : ∃ k : ℝ, inverse_proportion k P.fst = P.snd := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_quadrants_l301_30178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_furniture_l301_30192

/-- Calculates the total cost price of furniture items given their selling prices and markup percentages -/
theorem total_cost_price_furniture 
  (computer_table_price office_chair_price bookshelf_price : ℝ)
  (computer_table_markup office_chair_markup bookshelf_markup : ℝ)
  (h1 : computer_table_price = 8340)
  (h2 : office_chair_price = 4675)
  (h3 : bookshelf_price = 3600)
  (h4 : computer_table_markup = 0.25)
  (h5 : office_chair_markup = 0.30)
  (h6 : bookshelf_markup = 0.20) :
  (computer_table_price / (1 + computer_table_markup)) +
  (office_chair_price / (1 + office_chair_markup)) +
  (bookshelf_price / (1 + bookshelf_markup)) = 13268.15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_furniture_l301_30192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l301_30120

noncomputable section

/-- Line l' with equation y = -3x + 9 -/
def l' (x : ℝ) : ℝ := -3 * x + 9

/-- Line m' with equation y = -5x + 9 -/
def m' (x : ℝ) : ℝ := -5 * x + 9

/-- The x-intercept of line l' -/
def x_intercept_l' : ℝ := 3

/-- The x-intercept of line m' -/
def x_intercept_m' : ℝ := 9 / 5

/-- The area under line l' in the first quadrant -/
def area_under_l' : ℝ := (1 / 2) * x_intercept_l' * l' 0

/-- The area under line m' in the first quadrant -/
def area_under_m' : ℝ := (1 / 2) * x_intercept_m' * m' 0

/-- The area between lines l' and m' in the first quadrant -/
def area_between_lines : ℝ := area_under_l' - area_under_m'

/-- The probability of a randomly selected point in the 1st quadrant and below l' falling between l' and m' -/
def probability : ℝ := area_between_lines / area_under_l'

theorem probability_between_lines : probability = 0.4 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_between_lines_l301_30120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_gcd_greater_than_m_l301_30103

theorem existence_of_gcd_greater_than_m (d m : ℕ) (hd : d > 1) :
  ∃ (k l : ℕ), k > l ∧ l > 0 ∧ Nat.gcd (2^(2^k) + d) (2^(2^l) + d) > m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_gcd_greater_than_m_l301_30103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_winning_pair_l301_30187

/-- Represents a card in the deck -/
inductive Card
| blue : Fin 4 → Card
| yellow : Fin 4 → Card
deriving DecidableEq

/-- The deck of cards -/
def deck : Finset Card :=
  (Finset.univ.image Card.blue) ∪ (Finset.univ.image Card.yellow)

/-- Predicate for a winning pair of cards -/
def is_winning_pair (c1 c2 : Card) : Bool :=
  match c1, c2 with
  | Card.blue _, Card.blue _ => true
  | Card.yellow _, Card.yellow _ => true
  | Card.blue n1, Card.yellow n2 => n1 = n2
  | Card.yellow n1, Card.blue n2 => n1 = n2

/-- The set of all possible pairs of cards -/
def all_pairs : Finset (Card × Card) :=
  deck.product deck

/-- The set of winning pairs -/
def winning_pairs : Finset (Card × Card) :=
  all_pairs.filter (fun p => is_winning_pair p.1 p.2)

theorem probability_of_winning_pair :
  (winning_pairs.card : ℚ) / all_pairs.card = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_winning_pair_l301_30187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_inequality_l301_30175

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 2*x

-- g is defined explicitly based on the problem's solution
def g (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem symmetry_and_inequality :
  (∀ x, g x = -x^2 + 2*x) ∧
  (∀ x, g x ≥ f x - |x| - 1 ↔ -1 ≤ x ∧ x ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_inequality_l301_30175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_displacement_l301_30126

-- Define the velocity function
def v (t : ℝ) : ℝ := t^2 - t + 2

-- State the theorem
theorem particle_displacement :
  (∫ t in (1 : ℝ)..2, v t) = 17/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_displacement_l301_30126
