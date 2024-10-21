import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_charges_calculation_l136_13678

theorem transportation_charges_calculation (purchase_price repair_cost actual_selling_price : ℕ)
  (profit_percentage : ℚ) : 
  purchase_price = 12000 →
  repair_cost = 5000 →
  profit_percentage = 1/2 →
  actual_selling_price = 27000 →
  let total_cost_before_transport := purchase_price + repair_cost
  let selling_price_with_profit := total_cost_before_transport + (profit_percentage * total_cost_before_transport).floor
  actual_selling_price - selling_price_with_profit = 1500 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transportation_charges_calculation_l136_13678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_circle_circumference_l136_13619

-- Define the Rectangle structure
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the Circle structure
structure Circle where
  radius : ℝ

-- Define the inscribed_in relation
def Rectangle.inscribed_in (r : Rectangle) (c : Circle) : Prop :=
  r.width^2 + r.height^2 = (2 * c.radius)^2

theorem inscribed_rectangle_circle_circumference :
  ∀ (r : Rectangle) (c : Circle),
  r.inscribed_in c →
  r.width = 9 →
  r.height = 12 →
  2 * π * c.radius = 15 * π :=
by
  intros r c h_inscribed h_width h_height
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_circle_circumference_l136_13619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equality_l136_13626

theorem ceiling_sum_equality : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_sum_equality_l136_13626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_difference_l136_13657

theorem sin_cos_fourth_power_difference (α : Real) (h : Real.sin α = Real.sqrt 5 / 5) :
  Real.sin α ^ 4 - Real.cos α ^ 4 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_fourth_power_difference_l136_13657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_visible_pairs_155_birds_l136_13650

/-- Represents the number of birds at each position on the circle -/
def BirdDistribution := List Nat

/-- Calculates the number of mutually visible pairs for a given distribution -/
def visiblePairs (dist : BirdDistribution) : Nat :=
  dist.foldl (fun acc n => acc + n * (n - 1) / 2) 0

/-- Checks if a distribution is valid (sums to total birds and fits within positions) -/
def isValidDistribution (dist : BirdDistribution) (totalBirds : Nat) (positions : Nat) : Prop :=
  dist.sum = totalBirds ∧ dist.length ≤ positions

theorem min_visible_pairs_155_birds :
  ∃ (dist : BirdDistribution),
    isValidDistribution dist 155 35 ∧
    visiblePairs dist = 270 ∧
    ∀ (other : BirdDistribution),
      isValidDistribution other 155 35 →
      visiblePairs other ≥ 270 := by
  sorry

#eval visiblePairs [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_visible_pairs_155_birds_l136_13650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_correct_l136_13638

-- Define the original proposition
def original_proposition (a b : ℝ) : Prop :=
  a > b → (2 : ℝ)^a > (2 : ℝ)^b - 1

-- Define the negation of the original proposition
def negation_proposition (a b : ℝ) : Prop :=
  a ≤ b → (2 : ℝ)^a ≤ (2 : ℝ)^b - 1

-- Theorem stating that the negation is correct
theorem negation_correct :
  ∀ a b : ℝ, ¬(original_proposition a b) ↔ negation_proposition a b :=
by
  sorry

#check negation_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_correct_l136_13638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l136_13625

-- Define the expression
noncomputable def f (x : ℝ) : ℝ := (x^2 - 2*x + 3) / (2*x - 2)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1 ∧ 
  (∀ x : ℝ, 0 < x → x < 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = M) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l136_13625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l136_13641

noncomputable def loss_percentage (cost_price selling_price : ℝ) : ℝ :=
  ((cost_price - selling_price) / cost_price) * 100

theorem radio_loss_percentage :
  let cost_price := (1500 : ℝ)
  let selling_price := (1110 : ℝ)
  loss_percentage cost_price selling_price = 26 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radio_loss_percentage_l136_13641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_resisting_arrest_years_l136_13602

/-- Calculates the additional years for resisting arrest given the base sentence rate,
    third offense increase, amount stolen, and total sentence. -/
noncomputable def additional_years_for_resisting_arrest 
  (base_sentence_rate : ℝ) 
  (third_offense_increase : ℝ) 
  (amount_stolen : ℝ) 
  (total_sentence : ℝ) : ℝ :=
  let base_sentence := amount_stolen / (base_sentence_rate * 5000)
  let increased_sentence := base_sentence * (1 + third_offense_increase)
  total_sentence - increased_sentence

/-- Theorem stating that given the specified conditions, 
    the additional years for resisting arrest is 2. -/
theorem resisting_arrest_years : 
  additional_years_for_resisting_arrest 1 0.25 40000 12 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_resisting_arrest_years_l136_13602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_all_ones_row_correct_l136_13611

/-- Represents the 0-1 triangle derived from Pascal's triangle -/
def ZeroOneTriangle : ℕ → ℕ → ℕ := sorry

/-- Checks if a row in the 0-1 triangle consists entirely of 1s -/
def is_all_ones (row : ℕ) : Prop :=
  ∀ i, ZeroOneTriangle row i = 1

/-- The row number of the nth time the entire row consists of 1s -/
def nth_all_ones_row (n : ℕ) : ℕ := 2^n - 1

/-- Theorem stating that the nth time a row consists entirely of 1s occurs at row 2^n - 1 -/
theorem nth_all_ones_row_correct (n : ℕ) : 
  is_all_ones (nth_all_ones_row n) ∧ 
  (∀ k < nth_all_ones_row n, ¬is_all_ones k ∨ ∃ m < n, k = nth_all_ones_row m) :=
sorry

#check nth_all_ones_row_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nth_all_ones_row_correct_l136_13611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_room_arrangements_six_three_l136_13658

/-- Function that calculates the number of arrangements for given travelers and rooms -/
def traveler_room_arrangements : ℕ → ℕ → ℕ 
  | n, k => sorry  -- Placeholder implementation

/-- Theorem stating that for 6 travelers and 3 rooms, there are 240 arrangements -/
theorem traveler_room_arrangements_six_three :
  traveler_room_arrangements 6 3 = 240 := by
  sorry

/- Here's what the definitions and theorem mean:

traveler_room_arrangements : ℕ → ℕ → ℕ
This defines a function that takes two natural numbers (number of travelers and number of rooms) 
and returns a natural number (number of possible arrangements).

traveler_room_arrangements_six_three :
  traveler_room_arrangements 6 3 = 240
This theorem states that when we have 6 travelers and 3 rooms, 
the number of possible arrangements is 240.

The actual proof is omitted (replaced by 'sorry') as requested.
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traveler_room_arrangements_six_three_l136_13658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fayes_money_ratio_l136_13633

/-- Represents the problem of calculating the ratio of money given by Faye's mother to Faye's initial money --/
theorem fayes_money_ratio :
  ∃ (initial_money cupcake_price cookie_box_price total_spent total_after_gift mothers_gift : ℚ)
    (cupcake_quantity cookie_box_quantity : ℕ)
    (money_left : ℚ),
    initial_money = 20 ∧
    cupcake_price = 1.5 ∧
    cupcake_quantity = 10 ∧
    cookie_box_price = 3 ∧
    cookie_box_quantity = 5 ∧
    money_left = 30 ∧
    total_spent = cupcake_price * cupcake_quantity + cookie_box_price * cookie_box_quantity ∧
    total_after_gift = total_spent + money_left ∧
    mothers_gift = total_after_gift - initial_money ∧
    mothers_gift / initial_money = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fayes_money_ratio_l136_13633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_problem_l136_13661

/-- Chess tournament problem -/
theorem chess_tournament_problem 
  (n : ℕ) -- number of 9th grade students
  (m : ℕ) -- points scored by 9th grade students
  (h1 : n > 0) -- at least one 9th grade student
  (h2 : 10 * n > 0) -- at least one 10th grade student (10 times more than 9th)
  (h3 : m = n * (11 * n - 1)) -- total points formula for 9th graders
  (h4 : (11 : ℚ) * m = (11 * n * (11 * n - 1)) / 2) -- total points formula for all students
  : n = 1 ∧ m = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_problem_l136_13661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_circle_number_l136_13634

theorem central_circle_number (a b c d e f g : ℕ) : 
  ({a, b, c, d, e, f, g} : Finset ℕ) = Finset.range 7 →
  (∃ (k : ℕ), a + c + d = k ∧ b + d + e = k + 1 ∧ d + f + g = k + 2) →
  d = 1 ∨ d = 4 ∨ d = 7 := by
  sorry

#check central_circle_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_circle_number_l136_13634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_current_age_l136_13681

/-- Jessica's age when her mother died -/
def J : ℕ := sorry

/-- Jessica's mother's age when she died -/
def M : ℕ := sorry

/-- Jessica's current age -/
def current_age : ℕ := J + 10

theorem jessica_current_age :
  (J = M / 2) →  -- Jessica was half her mother's age when her mother died
  (M = 70 - 10) →  -- If her mother were alive now (10 years later), she would have been 70
  (current_age = 40) -- Jessica's current age is 40
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jessica_current_age_l136_13681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_sum_equivalence_l136_13630

def has_square_sum_factors (n k : ℕ) : Prop :=
  ∃ (factors : Finset ℕ), factors.card = k ∧ 
    (∀ x ∈ factors, x ∣ n) ∧ 
    (factors.sum (λ x ↦ x^2) = n)

def has_sum_factors (n k : ℕ) : Prop :=
  ∃ (factors : Finset ℕ), factors.card = k ∧ 
    (∀ x ∈ factors, x ∣ n) ∧ 
    (factors.sum id = n)

theorem factor_sum_equivalence :
  {k : ℕ | k > 0 ∧ ∀ n : ℕ, has_square_sum_factors n k → has_sum_factors n k} = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factor_sum_equivalence_l136_13630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_composite_polynomials_l136_13642

/-- The degree of a polynomial -/
def degree (p : Polynomial ℝ) : WithBot ℕ := Polynomial.degree p

/-- The result of substituting x^n for x in a polynomial -/
noncomputable def substPow (p : Polynomial ℝ) (n : ℕ) : Polynomial ℝ := p.comp (Polynomial.X ^ n)

theorem degree_of_composite_polynomials (f g h : Polynomial ℝ) 
  (hf : degree f = 3) (hg : degree g = 6) (hh : degree h = 2) :
  degree (substPow f 2 * substPow g 5 * substPow h 3) = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_composite_polynomials_l136_13642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_discount_l136_13601

theorem shopkeeper_discount (cost_price : ℝ) (markup_percentage : ℝ) (net_profit_percentage : ℝ) :
  markup_percentage = 20 →
  net_profit_percentage = 2 →
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := cost_price * (1 + net_profit_percentage / 100)
  let discount := marked_price - selling_price
  let discount_percentage := (discount / marked_price) * 100
  discount_percentage = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_discount_l136_13601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l136_13648

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (3*θ) = -23/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l136_13648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_x₀_with_finite_values_l136_13600

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

def sequenceF (x₀ : ℝ) : ℕ → ℝ
  | 0 => x₀
  | n + 1 => f (sequenceF x₀ n)

def hasFiniteValues (x₀ : ℝ) : Prop :=
  ∃ (S : Set ℝ), (Set.Finite S) ∧ (∀ n, sequenceF x₀ n ∈ S)

theorem infinite_x₀_with_finite_values :
  ∃ (S : Set ℝ), Set.Infinite S ∧ (∀ x₀ ∈ S, x₀ ≥ 0 ∧ hasFiniteValues x₀) := by
  sorry

#check infinite_x₀_with_finite_values

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_x₀_with_finite_values_l136_13600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_range_l136_13615

/-- A parabola with equation y² = x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = p.1}

/-- The fixed point B -/
def B : ℝ × ℝ := (1, 1)

/-- Perpendicular condition for vectors -/
def Perpendicular (v w : ℝ × ℝ) : Prop := v.1 * w.1 + v.2 * w.2 = 0

/-- The range of vertical coordinates for Q -/
def QRange : Set ℝ := Set.Iic (-1) ∪ Set.Ici 3

theorem parabola_perpendicular_range :
  ∀ P Q : ℝ × ℝ,
  P ∈ Parabola →
  Q ∈ Parabola →
  Perpendicular (P.1 - B.1, P.2 - B.2) (Q.1 - P.1, Q.2 - P.2) →
  Q.2 ∈ QRange :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_perpendicular_range_l136_13615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonoverlap_area_difference_value_l136_13698

/-- The positive difference between the areas of the nonoverlapping portions of a circle
    with radius 3 and a square with side length 2, where the circle crosses the center of the square. -/
noncomputable def nonoverlap_area_difference : ℝ :=
  let circle_radius : ℝ := 3
  let square_side : ℝ := 2
  let circle_area : ℝ := Real.pi * circle_radius^2
  let square_area : ℝ := square_side^2
  circle_area - square_area

/-- The positive difference between the areas of the nonoverlapping portions of a circle
    with radius 3 and a square with side length 2, where the circle crosses the center of the square,
    is equal to 9π - 4. -/
theorem nonoverlap_area_difference_value : nonoverlap_area_difference = 9 * Real.pi - 4 := by
  unfold nonoverlap_area_difference
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonoverlap_area_difference_value_l136_13698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l136_13697

noncomputable section

open Real

def OA : ℝ := 2
def OB : ℝ := 3
def OC : ℝ := 2 * sqrt 5

def angle_AOC : ℝ := arctan 2
def angle_BOC : ℝ := π / 3  -- 60 degrees in radians

def p : ℝ := 5 / 2
def q : ℝ := (3 * sqrt 5) / 2

theorem vector_decomposition :
  OC * (cos angle_AOC) = p * OA + q * OB * (cos (angle_AOC + angle_BOC)) ∧
  OC * (sin angle_AOC) = q * OB * (sin (angle_AOC + angle_BOC)) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_l136_13697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l136_13655

/-- Represents the average speed of a traveler -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The problem setup -/
def problem_setup : Prop :=
  ∃ (eddy_distance freddy_distance eddy_time freddy_time : ℝ),
    eddy_distance = 600 ∧
    freddy_distance = 460 ∧
    eddy_time = 3 ∧
    freddy_time = 4 ∧
    0 < eddy_time ∧
    0 < freddy_time

/-- The main theorem -/
theorem speed_ratio (h : problem_setup) :
  ∃ (eddy_distance freddy_distance eddy_time freddy_time : ℝ),
    eddy_distance = 600 ∧
    freddy_distance = 460 ∧
    eddy_time = 3 ∧
    freddy_time = 4 ∧
    (average_speed eddy_distance eddy_time) / (average_speed freddy_distance freddy_time) = 40 / 23 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l136_13655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_and_b_l136_13620

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def S (n : ℕ) : ℝ := 2 * n^2 - n

noncomputable def b (n : ℕ) (c : ℝ) : ℝ := S n / (n + c)

theorem arithmetic_sequence_sum_and_b (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1)^2 - 6 * (a 1) + 5 = 0 →
  (a 2)^2 - 6 * (a 2) + 5 = 0 →
  a 1 < a 2 →
  (∀ n : ℕ, S n = 2 * n^2 - n) ∧
  (∀ n : ℕ, b (n + 1) (-1/2) - b n (-1/2) = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_and_b_l136_13620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_log_expansion_l136_13617

-- Define the logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem binomial_log_expansion : (lg 2 + lg 5)^20 = 1 := by
  -- Convert lg to natural logarithm
  have h1 : ∀ x, lg x = Real.log x / Real.log 10 := by sorry
  
  -- Use the properties of logarithms
  have h2 : lg 2 + lg 5 = lg 10 := by
    rw [h1, h1, h1]
    sorry
  
  -- Simplify lg 10
  have h3 : lg 10 = 1 := by
    rw [h1]
    sorry
  
  -- Rewrite the left-hand side
  rw [h2, h3]
  
  -- Simplify 1^20
  simp


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_log_expansion_l136_13617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l136_13645

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x * (Real.sqrt 3 * Real.cos x + Real.sin x) + 1/2

-- Define the theorem
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  c = Real.sqrt 3 →
  f C = 2 →
  Real.sin B = 2 * Real.sin A →
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l136_13645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_1729_l136_13621

theorem largest_prime_factor_of_1729 : (Nat.factors 1729).maximum? = some 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_1729_l136_13621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_trig_inequality_l136_13616

theorem largest_n_for_trig_inequality :
  ∀ n : ℕ, n > 0 → (∀ x : ℝ, (Real.sin x)^(n : ℝ) + (Real.cos x)^(n : ℝ) ≥ 2 / n) ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_for_trig_inequality_l136_13616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_nth_root_is_polynomial_root_l136_13664

theorem no_rational_nth_root_is_polynomial_root :
  ∀ (n : ℕ) (q : ℚ),
  n > 0 →
  ¬ ∃ (x : ℂ), (x^n = q) ∧ (x^5 - x^4 - 4*x^3 + 4*x^2 + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_rational_nth_root_is_polynomial_root_l136_13664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_permutations_l136_13660

/-- A move on an n×n grid of counters -/
inductive Move (n : ℕ)
  | horizontal (row : Fin n) (direction : Bool)
  | vertical (col : Fin n) (direction : Bool)

/-- A returning sequence of moves -/
def ReturningSequence (n : ℕ) := List (Move n)

/-- The effect of a returning sequence on the arrangement of counters -/
def permutation (n : ℕ) (seq : ReturningSequence n) : Equiv (Fin n × Fin n) (Fin n × Fin n) :=
  sorry

/-- Sign of a permutation on n^2 elements -/
def permSign (n : ℕ) (p : Equiv (Fin n × Fin n) (Fin n × Fin n)) : Int :=
  sorry

theorem counter_permutations (n : ℕ) (h : n > 1) :
  (∀ (p : Equiv (Fin n × Fin n) (Fin n × Fin n)), permSign n p = 1 →
    ∃ (seq : ReturningSequence n), permutation n seq = p) ∧
  (∀ (seq : ReturningSequence n), permSign n (permutation n seq) = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_counter_permutations_l136_13660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_problem_l136_13632

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧ 
  A + B + C = Real.pi

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_eq : a * Real.sin B = Real.sqrt 3 * b * Real.cos A)
  (h_a : a = Real.sqrt 7)
  (h_b : b = 2) :
  A = Real.pi/3 ∧ 
  (1/2 : ℝ) * b * c * Real.sin A = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_problem_l136_13632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_meet_participation_l136_13607

/-- The number of students participating in both ball games and track and field competitions -/
def students_in_ball_and_track (total students_swimming students_track students_ball
  students_swimming_and_track students_swimming_and_ball : ℕ) : ℕ :=
  students_swimming + students_track + students_ball -
  students_swimming_and_track - students_swimming_and_ball - total

theorem sports_meet_participation
  (total : ℕ)
  (students_swimming : ℕ)
  (students_track : ℕ)
  (students_ball : ℕ)
  (students_swimming_and_track : ℕ)
  (students_swimming_and_ball : ℕ)
  (h1 : total = 26)
  (h2 : students_swimming = 15)
  (h3 : students_track = 8)
  (h4 : students_ball = 14)
  (h5 : students_swimming_and_track = 3)
  (h6 : students_swimming_and_ball = 3)
  (h7 : ∀ s, s ∈ (Set.univ : Set ℕ) → ¬(s ∈ (Set.inter (Set.inter {s | s ≤ students_swimming} {s | s ≤ students_track}) {s | s ≤ students_ball}))) :
  students_in_ball_and_track total students_swimming students_track students_ball
    students_swimming_and_track students_swimming_and_ball = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sports_meet_participation_l136_13607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l136_13624

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ) - Real.cos (ω * x + φ)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (4 * x - Real.pi / 3)

theorem function_properties (ω φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) 
  (h3 : ∀ x, f ω φ x = -f ω φ (-x)) 
  (h4 : ∃ k : ℤ, ∀ x, f ω φ (x + Real.pi / (2 * ω)) = f ω φ x) :
  (∀ x ∈ Set.Ioo (-Real.pi/2) (-Real.pi/4), ∀ y ∈ Set.Ioo (-Real.pi/2) (-Real.pi/4), 
    x < y → f ω φ y < f ω φ x) ∧
  (Set.range (g ∘ (fun x => x)) = Set.Icc (-2) (Real.sqrt 3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l136_13624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_case_l136_13627

/-- The length of a wire stretched between two vertical poles -/
noncomputable def wire_length (horizontal_distance : ℝ) (height_difference : ℝ) (pole1_height : ℝ) (pole2_height : ℝ) : ℝ :=
  Real.sqrt (horizontal_distance ^ 2 + (height_difference + pole2_height - pole1_height) ^ 2)

/-- Theorem stating the length of the wire in the given scenario -/
theorem wire_length_specific_case : 
  wire_length 20 3 8 18 = Real.sqrt 569 := by
  unfold wire_length
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_length_specific_case_l136_13627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiple_l136_13688

theorem matrix_scalar_multiple (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, N.mulVec v = (5 : ℝ) • v) ↔
  N = ![![5, 0, 0], ![0, 5, 0], ![0, 0, 5]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiple_l136_13688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_no_zeros_iff_sum_of_logs_gt_two_l136_13677

/-- The function f(x) = ln x - kx --/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * x

theorem tangent_line_at_one (k : ℝ) (h : k = 2) :
  ∃ m b : ℝ, ∀ x y : ℝ, y = m * (x - 1) + f k 1 ↔ x + y + 1 = 0 :=
sorry

theorem no_zeros_iff (k : ℝ) :
  (∀ x : ℝ, x > 0 → f k x ≠ 0) ↔ k > Real.exp (-1) :=
sorry

theorem sum_of_logs_gt_two (k : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) 
  (hz₁ : f k x₁ = 0) (hz₂ : f k x₂ = 0) :
  Real.log x₁ + Real.log x₂ > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_no_zeros_iff_sum_of_logs_gt_two_l136_13677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_30_l136_13614

/-- An isosceles right triangle with legs of length 10 -/
structure IsoscelesRightTriangle where
  leg_length : ℝ
  is_ten : leg_length = 10

/-- The triangle is divided into 25 smaller congruent triangles -/
def num_small_triangles : ℕ := 25

/-- The shaded area consists of 15 of these smaller triangles -/
def num_shaded_triangles : ℕ := 15

/-- Calculate the shaded area of the triangle -/
noncomputable def shaded_area (t : IsoscelesRightTriangle) : ℝ :=
  (t.leg_length ^ 2 / 2) * (num_shaded_triangles : ℝ) / (num_small_triangles : ℝ)

/-- The shaded area is equal to 30 -/
theorem shaded_area_is_30 (t : IsoscelesRightTriangle) : shaded_area t = 30 := by
  sorry

#eval num_small_triangles -- This line is added to check if the definition is working

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_30_l136_13614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_12sided_die_l136_13653

def is_prime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (λ m => ¬(n % (m + 2) = 0))

def count_primes (n : ℕ) : ℕ :=
  (List.range n).filter (λ x => is_prime (x + 1)) |>.length

theorem probability_prime_12sided_die :
  (count_primes 12 : ℚ) / 12 = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_12sided_die_l136_13653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l136_13628

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (Real.sqrt (2*x - 1) ≤ 1 → (x - a)*(x - (a + 1)) ≤ 0)) ∧ 
  (∃ x : ℝ, (x - a)*(x - (a + 1)) ≤ 0 ∧ Real.sqrt (2*x - 1) > 1) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l136_13628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_greater_than_4_l136_13654

/-- The number of possible outcomes when rolling a single fair die -/
def dieOutcomes : ℕ := 6

/-- The total number of possible outcomes when rolling two fair dice simultaneously -/
def totalOutcomes : ℕ := dieOutcomes * dieOutcomes

/-- The number of outcomes where the sum of face values is not greater than 4 -/
def outcomesNotGreaterThan4 : ℕ := 6

/-- The probability of the sum of face values being greater than 4 when rolling two fair dice simultaneously -/
theorem prob_sum_greater_than_4 : 1 - (outcomesNotGreaterThan4 : ℚ) / totalOutcomes = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_sum_greater_than_4_l136_13654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l136_13695

-- Define the ⊗ operation
noncomputable def otimes (a b : ℝ) : ℝ := if a ≤ b then a else b

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := otimes (Real.sin x) (Real.cos x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = Real.sqrt 2 / 2 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (y : ℝ), f y = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l136_13695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_negative_seven_ninths_l136_13665

noncomputable def g (x y : ℝ) : ℝ :=
  if x + y ≤ 5 then
    (x * y - x - 3) / (3 * x)
  else
    (x * y - y + 1) / (-3 * y)

theorem g_sum_equals_negative_seven_ninths :
  g 1 4 + g 3 3 = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_negative_seven_ninths_l136_13665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_both_subjects_difference_l136_13696

def total_students : ℤ := 3000

def chemistry_range (x : ℤ) : Prop :=
  1500 ≤ x ∧ x ≤ 1800

def physics_range (x : ℤ) : Prop :=
  750 ≤ x ∧ x ≤ 1050

theorem students_in_both_subjects_difference :
  ∃ (n N : ℤ),
    (∀ c p : ℤ, chemistry_range c → physics_range p →
      n ≤ c + p - total_students ∧ c + p - total_students ≤ N) ∧
    N - n = -600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_students_in_both_subjects_difference_l136_13696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_f_of_two_l136_13639

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -(x + 3)^2 else x + 5

-- State the theorem
theorem five_fold_f_of_two : f (f (f (f (f 2)))) = -5 := by
  -- Evaluate f(2)
  have h1 : f 2 = -25 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(2))
  have h2 : f (f 2) = -20 := by
    simp [f, h1]
    norm_num
  
  -- Evaluate f(f(f(2)))
  have h3 : f (f (f 2)) = -15 := by
    simp [f, h2]
    norm_num
  
  -- Evaluate f(f(f(f(2))))
  have h4 : f (f (f (f 2))) = -10 := by
    simp [f, h3]
    norm_num
  
  -- Final evaluation
  simp [f, h4]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_f_of_two_l136_13639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_card_area_after_shortening_l136_13672

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle --/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The theorem to be proved --/
theorem card_area_after_shortening (initial : Rectangle)
  (h1 : initial.length = 5 ∧ initial.width = 7)
  (h2 : ∃ (shortened : Rectangle), 
    (shortened.length = initial.length - 2 ∨ shortened.width = initial.width - 2) ∧
    area shortened = 21) :
  ∃ (other_shortened : Rectangle),
    (other_shortened.length = initial.length - 2 ∨ other_shortened.width = initial.width - 2) ∧
    area other_shortened = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_card_area_after_shortening_l136_13672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_8_f_inverse_8_eq_2_l136_13652

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 10^(x-1) - 2

-- State the theorem
theorem inverse_f_at_8 : f (2 : ℝ) = 8 := by
  sorry

-- State the inverse theorem
theorem f_inverse_8_eq_2 : f⁻¹ 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_at_8_f_inverse_8_eq_2_l136_13652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_squares_ratio_l136_13674

-- Define the squares and points
noncomputable def Square (s : ℝ) := s * s

-- Define the positions of points C and D
noncomputable def C_position : ℝ := 1 / 3
noncomputable def D_position : ℝ := 2 / 3

-- Define the area of pentagon AJICB
noncomputable def area_pentagon (s : ℝ) : ℝ := s * s

-- Define the total area of three squares
noncomputable def total_area_squares (s : ℝ) : ℝ := 3 * Square s

-- Theorem statement
theorem pentagon_to_squares_ratio :
  ∀ s : ℝ, s > 0 →
  (area_pentagon s) / (total_area_squares s) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_to_squares_ratio_l136_13674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_H_value_l136_13623

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the addition problem -/
structure AdditionProblem where
  F : Digit
  I : Digit
  V : Digit
  T : Digit
  H : Digit
  R : Digit
  E : Digit
  G : Digit
  all_different : F ≠ I ∧ F ≠ V ∧ F ≠ T ∧ F ≠ H ∧ F ≠ R ∧ F ≠ E ∧ F ≠ G ∧
                  I ≠ V ∧ I ≠ T ∧ I ≠ H ∧ I ≠ R ∧ I ≠ E ∧ I ≠ G ∧
                  V ≠ T ∧ V ≠ H ∧ V ≠ R ∧ V ≠ E ∧ V ≠ G ∧
                  T ≠ H ∧ T ≠ R ∧ T ≠ E ∧ T ≠ G ∧
                  H ≠ R ∧ H ≠ E ∧ H ≠ G ∧
                  R ≠ E ∧ R ≠ G ∧
                  E ≠ G
  F_is_9 : F.val = 9
  I_is_odd : I.val % 2 = 1
  sum_equation : 100 * F.val + 10 * I.val + V.val + 100 * T.val + 10 * H.val + R.val = 
                 1000 * E.val + 100 * I.val + 10 * G.val + H.val

theorem unique_H_value (p : AdditionProblem) : p.H.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_H_value_l136_13623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l136_13691

noncomputable def f (x : ℝ) := 2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * (Real.cos x) ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ x ∈ Set.Icc (-π/3) (π/3), 0 ≤ f x) ∧
  (∃ x ∈ Set.Icc (-π/3) (π/3), f x = 0) ∧
  (∀ x ∈ Set.Icc (-π/3) (π/3), f x ≤ 2 + Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc (-π/3) (π/3), f x = 2 + Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l136_13691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_max_value_l136_13635

theorem cos_max_value (a b : ℝ) (h : Real.cos (a - b) = Real.cos a + Real.cos b) : 
  ∃ (max_cos_a : ℝ), (∀ x, Real.cos x ≤ max_cos_a) ∧ (∃ y, Real.cos y = max_cos_a) ∧ max_cos_a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_max_value_l136_13635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_is_one_l136_13685

-- Define the interval [1/2, 2]
def I : Set ℝ := Set.Icc (1/2 : ℝ) 2

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / x + x * Real.log x
def g (x : ℝ) : ℝ := x^3 - x^2 - 3

-- State the theorem
theorem min_a_is_one :
  ∃ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ ∈ I → x₂ ∈ I → f a x₁ ≥ g x₂) ∧
  (∀ b : ℝ, b < a → ∃ x₁ x₂ : ℝ, x₁ ∈ I ∧ x₂ ∈ I ∧ f b x₁ < g x₂) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_is_one_l136_13685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_outer_circle_l136_13668

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- The area of a circle -/
noncomputable def Circle.area (c : Circle) : ℝ := Real.pi * c.radius^2

/-- Two concentric circles where the diameter of the inner circle
    equals the radius of the outer circle -/
structure ConcentricCircles where
  inner : Circle
  outer : Circle
  h : outer.radius = 2 * inner.radius

/-- Theorem stating that if the area of the inner circle is 16,
    then the area of the outer circle is 64 -/
theorem area_of_outer_circle (circles : ConcentricCircles) 
  (h : circles.inner.area = 16) : 
  circles.outer.area = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_outer_circle_l136_13668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_eq_2_sufficient_k_eq_2_not_necessary_k_eq_2_sufficient_not_necessary_l136_13663

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of line l₁ -/
noncomputable def m₁ : ℝ := -1/4

/-- The slope of line l₂ as a function of k -/
noncomputable def m₂ (k : ℝ) : ℝ := k^2

/-- k=2 is a sufficient condition for perpendicularity -/
theorem k_eq_2_sufficient : 
  perpendicular m₁ (m₂ 2) := by sorry

/-- k=2 is not a necessary condition for perpendicularity -/
theorem k_eq_2_not_necessary : 
  ∃ k, k ≠ 2 ∧ perpendicular m₁ (m₂ k) := by sorry

/-- k=2 is a sufficient but not necessary condition for perpendicularity -/
theorem k_eq_2_sufficient_not_necessary : 
  (perpendicular m₁ (m₂ 2)) ∧ (∃ k, k ≠ 2 ∧ perpendicular m₁ (m₂ k)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_eq_2_sufficient_k_eq_2_not_necessary_k_eq_2_sufficient_not_necessary_l136_13663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l136_13609

noncomputable def ellipse_C1 (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 = 1

noncomputable def ellipse_C2 (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

theorem ellipse_eccentricity_relation (a : ℝ) :
  a > 1 →
  ∃ (e₁ e₂ : ℝ),
    (∀ x y, ellipse_C1 a x y → e₁ = eccentricity a 1) ∧
    (∀ x y, ellipse_C2 x y → e₂ = eccentricity 2 1) ∧
    e₂ = Real.sqrt 3 * e₁ →
    a = 2 * Real.sqrt 3 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_relation_l136_13609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_sequence_exists_l136_13656

theorem no_infinite_sequence_exists : 
  ¬ (∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ 
    (∀ n, (a (n + 2) : ℝ) = (a (n + 1) : ℝ) + Real.sqrt ((a (n + 1) + a n) : ℝ))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_infinite_sequence_exists_l136_13656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l136_13666

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | (n + 2) => ((n + 2) * (sequence_a (n + 1))^2) / (2 * (sequence_a (n + 1))^2 + 4 * (n + 1) * (sequence_a (n + 1)) + (n + 1)^2)

theorem sequence_a_general_term (n : ℕ) :
  sequence_a n = n / (3^(2^(n-1)) - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_general_term_l136_13666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l136_13604

def S (n : ℕ) : ℚ := (3/2) * n^2 - (1/2) * n

def a (n : ℕ) : ℚ := 3 * n - 2

theorem sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = a (n + 1)) ∧
  (∃ lambda_max : ℚ, lambda_max = 28/3 ∧
    ∀ lambda : ℚ, (∀ n : ℕ, n ≥ 2 → a (n + 1) + lambda / a n ≥ lambda) → lambda ≤ lambda_max) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l136_13604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_and_blue_shoe_probability_l136_13690

/-- The probability of selecting a red shoe and a blue shoe from a collection of 10 pairs of shoes,
    where each pair is a different color, when randomly selecting 2 shoes without replacement. -/
theorem red_and_blue_shoe_probability (total_pairs : ℕ) (total_shoes : ℕ) 
  (h1 : total_pairs = 10)
  (h2 : total_shoes = 2 * total_pairs)
  (number_of_shoes : String → ℕ)
  (h3 : ∀ color, color ≠ "red" → color ≠ "blue" → number_of_shoes color = 2) :
  (1 : ℚ) / 95 = (number_of_shoes "red" / total_shoes) * (number_of_shoes "blue" / (total_shoes - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_and_blue_shoe_probability_l136_13690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_pass_time_l136_13693

/-- The time (in seconds) it takes for a car to pass a stationary object -/
noncomputable def passTime (carLength : ℝ) (carSpeed : ℝ) : ℝ :=
  carLength / (carSpeed * 1000 / 3600)

/-- Theorem: A 10-meter long car moving at 36 kmph takes 1 second to pass a telegraph post -/
theorem car_pass_time :
  let carLength : ℝ := 10
  let carSpeed : ℝ := 36
  passTime carLength carSpeed = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_pass_time_l136_13693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l136_13613

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangleConditions (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧  -- Sum of angles in a triangle
  t.a + t.c = 6 ∧
  (3 - Real.cos t.A) * Real.sin t.B = Real.sin t.A * (1 + Real.cos t.B)

-- Helper function to calculate area
noncomputable def area (t : Triangle) : Real :=
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangleConditions t) :
  t.b = 2 ∧ 
  ∀ (s : Triangle), triangleConditions s → area s ≤ 2 * Real.sqrt 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l136_13613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculation_l136_13640

noncomputable def initial_angle : ℝ := 35

noncomputable def complement (angle : ℝ) : ℝ := 90 - angle

noncomputable def supplement (angle : ℝ) : ℝ := 180 - angle

noncomputable def percent_of (percent : ℝ) (value : ℝ) : ℝ := (percent / 100) * value

theorem angle_calculation (angle : ℝ) (h : angle = initial_angle) :
  percent_of 10 (supplement (complement angle)) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_calculation_l136_13640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_alpha_l136_13646

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin x - Real.cos x

theorem min_value_sin_alpha (α : ℝ) (h : ∀ x, f x ≥ f α) : Real.sin α = -Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_alpha_l136_13646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l136_13675

/-- The distance of the race in yards -/
noncomputable def d : ℝ := sorry

/-- The speed of runner A -/
noncomputable def a : ℝ := sorry

/-- The speed of runner B -/
noncomputable def b : ℝ := sorry

/-- The speed of runner C -/
noncomputable def c : ℝ := sorry

/-- A can beat B by 25 yards -/
axiom h1 : d / a = (d - 25) / b

/-- B can beat C by 15 yards -/
axiom h2 : d / b = (d - 15) / c

/-- A can beat C by 35 yards -/
axiom h3 : d / a = (d - 35) / c

/-- The speeds are positive -/
axiom pos_speeds : a > 0 ∧ b > 0 ∧ c > 0

/-- The distance is positive -/
axiom pos_distance : d > 0

theorem race_distance : d = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_distance_l136_13675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l136_13687

noncomputable def f (x : ℝ) : ℝ :=
  if |Real.cos x| ≥ Real.sqrt 2 / 2 then Real.cos x else 0

theorem f_properties :
  (f (Real.pi / 3) = 0) ∧
  ({x : ℝ | f x ≤ Real.sin x ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi} = {x : ℝ | Real.pi / 4 ≤ x ∧ x ≤ 5 * Real.pi / 4}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l136_13687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_circle_l136_13692

/-- Given a circle with center (2,6) and a point (7,3) on the circle,
    the slope of the tangent line at (7,3) is 5/3 -/
theorem tangent_slope_of_circle (center : ℝ × ℝ) (point : ℝ × ℝ) :
  center = (2, 6) →
  point = (7, 3) →
  (point.2 - center.2) / (point.1 - center.1) = -3 / 5 →
  -1 / ((point.2 - center.2) / (point.1 - center.1)) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_of_circle_l136_13692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_q_l136_13682

theorem existence_of_prime_q (p : ℕ) (hp : Nat.Prime p) :
  ∃ q : ℕ, Nat.Prime q ∧ ∀ n : ℕ, ¬(q ∣ (n^p - p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_prime_q_l136_13682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_line_through_origin_with_distance_l136_13618

-- Define the types for points and lines
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the perpendicular relationship between lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define a function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a^2 + l.b^2)

-- Theorem 1
theorem perpendicular_line_through_point :
  let l1 : Line := { a := 3, b := 2, c := -1 }
  let l2 : Line := { a := 2, b := -3, c := 5 }
  let A : Point := { x := 2, y := 3 }
  perpendicular l1 l2 ∧ point_on_line A l2 := by sorry

-- Theorem 2
theorem line_through_origin_with_distance :
  let M : Point := { x := 5, y := 0 }
  let l : Line := { a := 3, b := 4, c := 0 }
  point_on_line { x := 0, y := 0 } l ∧ distance_point_to_line M l = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_line_through_origin_with_distance_l136_13618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_area_max_area_condition_l136_13603

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vectors m and n are parallel -/
def vectorsParallel (t : Triangle) : Prop :=
  (t.a * Real.cos t.B) = ((2 * t.c - t.b) * Real.cos t.A)

/-- The measure of angle A is π/3 -/
theorem angle_A_measure (t : Triangle) (h : vectorsParallel t) (ha : t.a = 4) : 
  t.A = π / 3 := by sorry

/-- The maximum area of the triangle is 4√3 -/
theorem max_area (t : Triangle) (h : vectorsParallel t) (ha : t.a = 4) : 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A ≤ 4 * Real.sqrt 3 := by sorry

/-- The maximum area is achieved when b = c = 4 -/
theorem max_area_condition (t : Triangle) (h : vectorsParallel t) (ha : t.a = 4) : 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 ↔ t.b = 4 ∧ t.c = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_area_max_area_condition_l136_13603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosBAD_equals_sqrt70_div_14_l136_13676

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let BC := Real.sqrt ((t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2)
  AB = 4 ∧ AC = 7 ∧ BC = 9

-- Define point D on BC
def pointDOnBC (t : Triangle) (D : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ D = (k * t.B.1 + (1 - k) * t.C.1, k * t.B.2 + (1 - k) * t.C.2)

-- Define AD bisects angle BAC
noncomputable def ADBisectsBAC (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AC := Real.sqrt ((t.C.1 - t.A.1)^2 + (t.C.2 - t.A.2)^2)
  let AD := Real.sqrt ((D.1 - t.A.1)^2 + (D.2 - t.A.2)^2)
  AB / AD = AC / AD

-- Define the area of triangle ABC
noncomputable def areaABC (t : Triangle) : ℝ :=
  let s := (4 + 7 + 9) / 2  -- semi-perimeter
  Real.sqrt (s * (s - 4) * (s - 7) * (s - 9))

-- Main theorem
theorem cosBAD_equals_sqrt70_div_14 (t : Triangle) (D : ℝ × ℝ) :
  isValidTriangle t → pointDOnBC t D → ADBisectsBAC t D → areaABC t = 12 →
  let AB := Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2)
  let AD := Real.sqrt ((D.1 - t.A.1)^2 + (D.2 - t.A.2)^2)
  (t.B.1 - t.A.1) * (D.1 - t.A.1) + (t.B.2 - t.A.2) * (D.2 - t.A.2) = AB * AD * (Real.sqrt 70 / 14) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosBAD_equals_sqrt70_div_14_l136_13676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_three_digit_numbers_l136_13637

def is_three_digit_number (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

def are_distinct_digits (a b c d e f : ℤ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def number_from_digits (a b c : ℤ) : ℤ := 100 * a + 10 * b + c

theorem smallest_difference_three_digit_numbers :
  ∃ (a b c d e f : ℤ),
    0 ≤ a ∧ a < 10 ∧ 0 ≤ b ∧ b < 10 ∧ 0 ≤ c ∧ c < 10 ∧
    0 ≤ d ∧ d < 10 ∧ 0 ≤ e ∧ e < 10 ∧ 0 ≤ f ∧ f < 10 ∧
    are_distinct_digits a b c d e f ∧
    is_three_digit_number (number_from_digits a b c) ∧
    is_three_digit_number (number_from_digits d e f) ∧
    (∀ (x y z u v w : ℤ),
      0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ 0 ≤ z ∧ z < 10 ∧
      0 ≤ u ∧ u < 10 ∧ 0 ≤ v ∧ v < 10 ∧ 0 ≤ w ∧ w < 10 →
      are_distinct_digits x y z u v w →
      is_three_digit_number (number_from_digits x y z) →
      is_three_digit_number (number_from_digits u v w) →
      number_from_digits x y z ≠ number_from_digits u v w →
      3 ≤ |number_from_digits x y z - number_from_digits u v w|) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_difference_three_digit_numbers_l136_13637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l136_13605

noncomputable def powerFunction (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_increasing (α : ℝ) (h : α ∈ ({1, 3, 1/2} : Set ℝ)) :
  StrictMono (powerFunction α) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_l136_13605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l136_13686

/-- The smaller angle between the hour and minute hands of a clock at 3:40 -/
noncomputable def clock_angle : ℝ := 50.0

/-- The number of degrees in a full circle -/
noncomputable def full_circle : ℝ := 360

/-- The number of hours on a clock face -/
def hours_on_clock : ℕ := 12

/-- The number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- The angle between hour and minute hands at 3:00 -/
noncomputable def angle_at_3 : ℝ := full_circle / 4

/-- The angle the minute hand moves per minute -/
noncomputable def minute_hand_speed : ℝ := full_circle / minutes_in_hour

/-- The angle the hour hand moves per minute -/
noncomputable def hour_hand_speed : ℝ := full_circle / (hours_on_clock * minutes_in_hour)

/-- The number of minutes past 3:00 -/
def minutes_past_3 : ℕ := 40

theorem clock_angle_at_3_40 : 
  clock_angle = full_circle - (angle_at_3 + (minute_hand_speed - hour_hand_speed) * minutes_past_3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l136_13686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_sum_squared_l136_13662

theorem multiples_sum_squared : 
  let a := Finset.card (Finset.filter (fun n => n > 0 ∧ n < 60 ∧ 12 ∣ n) (Finset.range 60))
  let b := Finset.card (Finset.filter (fun n => n > 0 ∧ n < 60 ∧ 4 ∣ n ∧ 3 ∣ n) (Finset.range 60))
  (a + b)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_sum_squared_l136_13662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l136_13669

/-- A line in a 2D plane --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, a * x + b * y + c = 0

/-- Check if a point lies on a line --/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- The x-intercept of a line --/
noncomputable def Line.xIntercept (l : Line) : ℝ := -l.c / l.a

/-- The y-intercept of a line --/
noncomputable def Line.yIntercept (l : Line) : ℝ := -l.c / l.b

/-- Theorem: A line passing through (1,2) with vertical intercept twice the horizontal intercept
    has equation 2x - y = 0 or 2x + y - 4 = 0 --/
theorem line_equation_theorem (l : Line) : 
  l.contains (1, 2) ∧ l.yIntercept = 2 * l.xIntercept →
  (l.a = 2 ∧ l.b = -1 ∧ l.c = 0) ∨ (l.a = 2 ∧ l.b = 1 ∧ l.c = -4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l136_13669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_minimum_value_l136_13608

-- Define the function f
noncomputable def f (a : ℝ) (m : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a + m

-- Define the function g
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := 2 * f x - f (x - 1)

theorem function_and_minimum_value 
  (a : ℝ) (m : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_ne_one : a ≠ 1) 
  (h_point_1 : f a m 8 = 2) 
  (h_symmetry : ∃ q : ℝ, f a m q = -1 ∧ 3 - 2 = 2 - q) :
  (∀ x > 0, f a m x = -1 + Real.log x / Real.log 2) ∧ 
  (∀ x > 1, g (f a m) x ≥ 1) ∧
  g (f a m) 2 = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_and_minimum_value_l136_13608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_increasing_condition_l136_13643

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x - m) / Real.log 0.5

-- Theorem for the first part
theorem domain_condition (m : ℝ) :
  (∀ x, f m x ∈ Set.univ) → m ∈ Set.Ioo (-4) 0 :=
sorry

-- Theorem for the second part
theorem increasing_condition (m : ℝ) :
  (∀ x ∈ Set.Ioo (-2) (-1/2), StrictMono (f m)) → m ∈ Set.Icc (-1) (1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_condition_increasing_condition_l136_13643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_fixed_point_l136_13606

/-- The complex number representing the center of rotation --/
noncomputable def d : ℂ := (16 * Real.sqrt 3) / 7 + (18 * Complex.I) / 7

/-- The rotation function --/
noncomputable def g (z : ℂ) : ℂ := ((1 - Complex.I * Real.sqrt 3) * z + (2 * Real.sqrt 3 + 12 * Complex.I)) / 3

/-- Theorem stating that d is the fixed point of g --/
theorem d_is_fixed_point : g d = d := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_d_is_fixed_point_l136_13606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l136_13622

-- Define the function f(x) = ln(x^2 - x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - x)

-- Define the domain of f
def domain (x : ℝ) : Prop := x > 1 ∨ x < 0

-- State the theorem
theorem monotonic_increase_interval :
  ∀ x y, domain x → domain y → x > 1 → y > 1 → x < y → f x < f y :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_l136_13622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_625_exponents_l136_13644

theorem value_of_625_exponents : (625 : ℝ) ^ (0.12 : ℝ) * (625 : ℝ) ^ (0.13 : ℝ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_625_exponents_l136_13644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_sin_reciprocal_x_intercepts_sin_reciprocal_specific_l136_13636

open Real

theorem x_intercepts_sin_reciprocal (a b : ℝ) (ha : a > 0) (hb : b > a) :
  let f := fun x => sin (1 / x)
  let num_intercepts := (⌊b / π⌋ : ℤ) - (⌊a / π⌋ : ℤ)
  (∃ n : ℤ, num_intercepts = n) ∧
  (∀ x ∈ Set.Ioo a b, f x = 0 ↔ ∃ k : ℕ, k > 0 ∧ x = 1 / (k * π)) :=
sorry

-- Specific case for the given problem
theorem x_intercepts_sin_reciprocal_specific :
  let f := fun x => sin (1 / x)
  let num_intercepts := (⌊100000 / π⌋ : ℤ) - (⌊10000 / π⌋ : ℤ)
  (∃ n : ℤ, num_intercepts = n) ∧
  (∀ x ∈ Set.Ioo 0.00001 0.0001, f x = 0 ↔ ∃ k : ℕ, k > 0 ∧ x = 1 / (k * π)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_sin_reciprocal_x_intercepts_sin_reciprocal_specific_l136_13636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l136_13651

theorem contrapositive_equivalence (m : ℕ+) :
  (¬(∃ x : ℝ, x^2 + x - m.val = 0) → m.val ≤ 0) ↔
  (m.val > 0 → ∃ x : ℝ, x^2 + x - m.val = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_equivalence_l136_13651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_value_l136_13680

/-- Define the recursive sum S --/
noncomputable def S : ℝ := 15 + (14 / 2) + (13 / 2^2) + (12 / 2^3) + (11 / 2^4) + (10 / 2^5) + (9 / 2^6) + (8 / 2^7)

/-- Theorem stating that S equals 28 + 1/128 --/
theorem sum_value : S = 28 + 1/128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_value_l136_13680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_living_expenses_percentage_l136_13612

noncomputable def monthly_income : ℝ := 1600
noncomputable def insurance_fraction : ℝ := 1/5
noncomputable def savings : ℝ := 80

theorem living_expenses_percentage :
  ∃ (living_expenses_fraction : ℝ),
    living_expenses_fraction * monthly_income +
    insurance_fraction * monthly_income +
    savings = monthly_income ∧
    living_expenses_fraction = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_living_expenses_percentage_l136_13612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_frac_part_l136_13699

noncomputable def floor (x : ℝ) := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

theorem largest_n_frac_part : 
  ∃ (n : ℝ), 
    (∀ (m : ℝ), (floor m : ℝ) / m = 2015 / 2016 → m ≤ n) ∧ 
    (floor n : ℝ) / n = 2015 / 2016 ∧ 
    frac n = 2014 / 2015 := by
  sorry

#check largest_n_frac_part

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_frac_part_l136_13699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_15_l136_13683

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x + 3 / x^2)^5

theorem constant_term_is_15 : 
  ∃ (g : ℝ → ℝ), ∀ x, x ≠ 0 → f x = 15 + g x ∧ Filter.Tendsto g (Filter.atTop) (nhds 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_is_15_l136_13683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l136_13670

/-- Calculates the present value given future value, interest rate, and time period -/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- Rounds a real number to the nearest cent -/
noncomputable def roundToCent (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

theorem investment_problem :
  let futureValue : ℝ := 600000
  let interestRate : ℝ := 0.04
  let years : ℕ := 8
  let result := roundToCent (presentValue futureValue interestRate years)
  result = 438447.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_l136_13670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_cosine_and_function_extrema_l136_13684

open Real

theorem alpha_cosine_and_function_extrema 
  (α : ℝ) 
  (h1 : sin (α + π/4) = 7 * sqrt 2 / 10) 
  (h2 : π/4 < α ∧ α < π/2) :
  (cos α = 3/5) ∧
  (∃ x, ∀ y, cos (2*y) + 5/2 * sin α * sin y ≤ cos (2*x) + 5/2 * sin α * sin x ∧
         cos (2*x) + 5/2 * sin α * sin x = 3/2) ∧
  (∃ x, ∀ y, cos (2*y) + 5/2 * sin α * sin y ≥ cos (2*x) + 5/2 * sin α * sin x ∧
         cos (2*x) + 5/2 * sin α * sin x = -3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_cosine_and_function_extrema_l136_13684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_unique_l136_13629

/-- The line passing through (5, 3, 2) in the direction (1, -1, 0) -/
def line (t : ℝ) : ℝ × ℝ × ℝ := (5 + t, 3 - t, 2)

/-- The plane 3x + y - 5z - 12 = 0 -/
def plane (p : ℝ × ℝ × ℝ) : Prop :=
  3 * p.1 + p.2.1 - 5 * p.2.2 - 12 = 0

/-- The point (7, 1, 2) -/
def intersection_point : ℝ × ℝ × ℝ := (7, 1, 2)

theorem intersection_unique :
  (∃! p, ∃ t, line t = p ∧ plane p) ∧
  (∃ t, line t = intersection_point ∧ plane intersection_point) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_unique_l136_13629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_jar_problem_l136_13679

theorem cookie_jar_problem (num_adults num_children : ℕ) 
  (adult_fraction : ℚ) (cookies_per_child : ℕ) : 
  num_adults = 2 →
  num_children = 4 →
  adult_fraction = 1/3 →
  cookies_per_child = 20 →
  ∃ total_cookies : ℕ, 
    (1 - adult_fraction) * total_cookies = num_children * cookies_per_child ∧
    total_cookies = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_jar_problem_l136_13679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l136_13673

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the constants a and b
noncomputable def a : ℝ := Real.log 2 / Real.log 5
noncomputable def b : ℝ := Real.log 3 / Real.log 8

-- State the theorem
theorem function_inequality
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x < deriv g x) :
  f a + g b > g a + f b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l136_13673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_syrup_problem_l136_13671

/-- Represents the state of the buckets --/
structure BucketState where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The set of allowed operations on the buckets --/
inductive Operation
  | PourAway (bucket : Fin 3)
  | PourAll (source target : Fin 3)
  | PourUntilEqual (source target other : Fin 3)

/-- Applies an operation to a bucket state --/
def applyOperation (op : Operation) (state : BucketState) : BucketState :=
  sorry

/-- Checks if a sequence of operations results in 10 litres of 30% syrup --/
def isValidSolution (n : ℝ) (ops : List Operation) : Prop :=
  sorry

/-- The set of valid n values --/
def validN : Set ℝ :=
  {n | ∃ k : ℕ, k ≥ 2 ∧ (n = 3 * k + 1 ∨ n = 3 * k + 2)}

/-- The main theorem --/
theorem syrup_problem (n : ℝ) :
  (∃ ops : List Operation, isValidSolution n ops) ↔ n ∈ validN :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_syrup_problem_l136_13671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l136_13631

def A : Set ℤ := {-2, -1, 0, 1, 2}

def B : Set ℤ := {x : ℤ | x > -1 ∧ x < 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l136_13631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l136_13647

-- Define the distance function between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Define the equation of the conic section
def conic_equation (x y : ℝ) : Prop :=
  distance x y 0 2 + distance x y 6 4 = 12

-- Define what an ellipse is
def is_ellipse (f : ℝ → ℝ → Prop) (f1x f1y f2x f2y : ℝ) (c : ℝ) : Prop :=
  ∀ x y, f x y ↔ distance x y f1x f1y + distance x y f2x f2y = c

-- State the theorem
theorem conic_is_ellipse :
  is_ellipse conic_equation 0 2 6 4 12 ∧
  distance 0 2 6 4 < 12 := by
  sorry

#check conic_is_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l136_13647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pqrs_l136_13610

/-- The area of a square given its vertices. -/
def area_square (A B C D : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if three points form an equilateral triangle. -/
def is_equilateral_triangle (A B C : ℝ × ℝ) : Prop := sorry

/-- Given a square WXYZ with an area of 36, and points P, Q, R, and S such that WPY, XQZ, YRW, and ZSX
    are equilateral triangles, the area of square PQRS is 72 + 36√3. -/
theorem area_of_pqrs (W X Y Z P Q R S : ℝ × ℝ) : 
  (area_square W X Y Z = 36) →
  (is_equilateral_triangle W P Y) →
  (is_equilateral_triangle X Q Z) →
  (is_equilateral_triangle Y R W) →
  (is_equilateral_triangle Z S X) →
  (area_square P Q R S = 72 + 36 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_pqrs_l136_13610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_problem_l136_13659

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the function g
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := (1 - 4*m) * Real.sqrt x

-- State the theorem
theorem exponential_function_problem (a : ℝ) (m : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≤ 4) →
  (∃ x ∈ Set.Icc (-2) 1, f a x = 4) →
  (∀ x ∈ Set.Icc (-2) 1, f a x ≥ m) →
  (∃ x ∈ Set.Icc (-2) 1, f a x = m) →
  (∀ x y, x < y → x ≥ 0 → g m y < g m x) →
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_problem_l136_13659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l136_13689

-- Define the nabla operation as noncomputable
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_calculation (h1 : 2 > 0) (h2 : 3 > 0) (h3 : 1 > 0) (h4 : 4 > 0) :
  nabla (nabla 2 3) (nabla 1 4) = 1 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_calculation_l136_13689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_tangent_sine_cosine_l136_13694

open Real MeasureTheory

theorem definite_integral_tangent_sine_cosine : 
  ∫ x in Set.Icc 0 (Real.arccos (Real.sqrt (2/3))), 
    (Real.tan x + 2) / (Real.sin x ^ 2 + 2 * Real.cos x ^ 2 - 3) = 
    -(Real.log 2 + Real.sqrt 2 * Real.pi) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_tangent_sine_cosine_l136_13694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l136_13667

-- Define the colors
inductive Color
| Blue
| Green
| Orange
| Purple
deriving BEq, DecidableEq

-- Define a type for the sequence of houses
def HouseSequence := List Color

-- Function to check if a sequence is valid
def is_valid_sequence (seq : HouseSequence) : Prop :=
  seq.length = 4 ∧ 
  seq.toFinset.card = 4 ∧
  (seq.indexOf Color.Purple < seq.indexOf Color.Green) ∧
  (seq.indexOf Color.Blue < seq.indexOf Color.Orange) ∧
  (seq.indexOf Color.Blue + 1 ≠ seq.indexOf Color.Orange) ∧
  (seq.indexOf Color.Orange + 1 ≠ seq.indexOf Color.Blue)

-- Theorem stating there are exactly 3 valid sequences
theorem valid_sequences_count :
  ∃ (valid_seqs : List HouseSequence),
    (∀ seq ∈ valid_seqs, is_valid_sequence seq) ∧
    (∀ seq, is_valid_sequence seq → seq ∈ valid_seqs) ∧
    valid_seqs.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequences_count_l136_13667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_market_value_theorem_l136_13649

/-- Represents a stock with its dividend rate and yield -/
structure Stock where
  dividend_rate : ℚ
  yield : ℚ

/-- Calculates the market value of a stock -/
def market_value (s : Stock) : ℚ := s.dividend_rate / s.yield

/-- Theorem stating the relationship between market value, dividend rate, and yield for a specific stock -/
theorem market_value_theorem (s : Stock) 
  (h1 : s.dividend_rate = 5 / 100)
  (h2 : s.yield = 10 / 100) : 
  market_value s = 1 / 2 := by
  unfold market_value
  rw [h1, h2]
  norm_num

/-- Example calculation -/
def example_stock : Stock := { dividend_rate := 5 / 100, yield := 10 / 100 }

#eval market_value example_stock

end NUMINAMATH_CALUDE_ERRORFEEDBACK_market_value_theorem_l136_13649
