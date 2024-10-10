import Mathlib

namespace intersection_implies_a_value_l1085_108525

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, a^2+1, 2*a-1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end intersection_implies_a_value_l1085_108525


namespace new_basis_from_old_l1085_108598

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem new_basis_from_old (a b c : V) 
  (h : LinearIndependent ℝ ![a, b, c]) 
  (h_span : Submodule.span ℝ {a, b, c} = ⊤) :
  LinearIndependent ℝ ![a + b, b + c, c + a] ∧ 
  Submodule.span ℝ {a + b, b + c, c + a} = ⊤ := by
sorry

end new_basis_from_old_l1085_108598


namespace ellipse_foci_distance_l1085_108532

theorem ellipse_foci_distance (x y : ℝ) :
  (x^2 / 45 + y^2 / 5 = 9) → (∃ f : ℝ, f = 12 * Real.sqrt 10 ∧ f = distance_between_foci) :=
by
  sorry

end ellipse_foci_distance_l1085_108532


namespace initial_pc_cost_l1085_108508

/-- Proves that the initial cost of a gaming PC is $1200, given the conditions of the video card upgrade and total spent. -/
theorem initial_pc_cost (old_card_sale : ℕ) (new_card_cost : ℕ) (total_spent : ℕ) 
  (h1 : old_card_sale = 300)
  (h2 : new_card_cost = 500)
  (h3 : total_spent = 1400) :
  total_spent - (new_card_cost - old_card_sale) = 1200 := by
  sorry

#check initial_pc_cost

end initial_pc_cost_l1085_108508


namespace greatest_unachievable_scores_l1085_108544

def score_system : List ℕ := [19, 9, 8]

def is_achievable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 19 * a + 9 * b + 8 * c

theorem greatest_unachievable_scores :
  (¬ is_achievable 31) ∧
  (¬ is_achievable 39) ∧
  (∀ m, m > 39 → is_achievable m) ∧
  (31 * 39 = 1209) := by
  sorry

#check greatest_unachievable_scores

end greatest_unachievable_scores_l1085_108544


namespace tourist_cyclist_speed_l1085_108595

/-- Given the conditions of a tourist and cyclist problem, prove their speeds -/
theorem tourist_cyclist_speed :
  -- Distance from A to B
  let distance : ℝ := 24

  -- Time difference between tourist and cyclist start
  let time_diff : ℝ := 4/3

  -- Time for cyclist to overtake tourist
  let overtake_time : ℝ := 1/2

  -- Time between first and second encounter
  let encounter_interval : ℝ := 3/2

  -- Speed of cyclist
  let v_cyclist : ℝ := 16.5

  -- Speed of tourist
  let v_tourist : ℝ := 4.5

  -- Equations based on the problem conditions
  (v_cyclist * overtake_time = v_tourist * (time_diff + overtake_time)) ∧
  (v_cyclist * 2 + v_tourist * (time_diff + overtake_time + encounter_interval) = 2 * distance)

  -- Conclusion: The speeds satisfy the equations
  → v_cyclist = 16.5 ∧ v_tourist = 4.5 := by
  sorry

end tourist_cyclist_speed_l1085_108595


namespace no_integer_cube_equals_3n2_plus_3n_plus_7_l1085_108588

theorem no_integer_cube_equals_3n2_plus_3n_plus_7 :
  ¬ ∃ (m n : ℤ), m^3 = 3*n^2 + 3*n + 7 := by
sorry

end no_integer_cube_equals_3n2_plus_3n_plus_7_l1085_108588


namespace calculation_proof_l1085_108510

theorem calculation_proof : (((15 - 2 + 4) / 1) / 2) * 8 = 68 := by
  sorry

end calculation_proof_l1085_108510


namespace sphere_radii_difference_l1085_108553

theorem sphere_radii_difference (R r : ℝ) : 
  R > r → 
  4 * Real.pi * R^2 - 4 * Real.pi * r^2 = 48 * Real.pi → 
  2 * Real.pi * R + 2 * Real.pi * r = 12 * Real.pi → 
  R - r = 2 :=
by sorry

end sphere_radii_difference_l1085_108553


namespace max_cube_sum_under_constraints_l1085_108590

theorem max_cube_sum_under_constraints {a b c d : ℝ} 
  (sum_squares : a^2 + b^2 + c^2 + d^2 = 20)
  (sum_linear : a + b + c + d = 10) :
  a^3 + b^3 + c^3 + d^3 ≤ 500 ∧ 
  ∃ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 = 20 ∧ 
                   x + y + z + w = 10 ∧ 
                   x^3 + y^3 + z^3 + w^3 = 500 :=
by sorry

end max_cube_sum_under_constraints_l1085_108590


namespace remainder_theorem_l1085_108515

def polynomial (x : ℝ) : ℝ := 5*x^5 - 12*x^4 + 3*x^3 - 7*x + 15

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + (-7) :=
by
  sorry

end remainder_theorem_l1085_108515


namespace count_five_digit_integers_l1085_108549

/-- The set of digits to be used -/
def digits : Multiset ℕ := {3, 3, 6, 6, 6, 7, 8, 8}

/-- The number of digits required for each integer -/
def required_digits : ℕ := 5

/-- The function to count valid integers -/
def count_valid_integers (d : Multiset ℕ) (r : ℕ) : ℕ :=
  (d.card.factorial) / ((d.count 3).factorial * (d.count 6).factorial * (d.count 8).factorial)

/-- The main theorem -/
theorem count_five_digit_integers : 
  count_valid_integers digits required_digits = 1680 :=
sorry

end count_five_digit_integers_l1085_108549


namespace one_third_percent_of_150_l1085_108568

theorem one_third_percent_of_150 : (1 / 3 * 1 / 100) * 150 = 0.5 := by
  sorry

end one_third_percent_of_150_l1085_108568


namespace minimum_translation_l1085_108542

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x + Real.cos x

theorem minimum_translation (a : ℝ) :
  (∀ x, f a (x - π/4) = f a (π/4 + (π/4 - x))) →
  ∃ φ : ℝ, φ > 0 ∧
    (∀ x, f a (x - φ) = f a (-x)) ∧
    (∀ ψ, ψ > 0 ∧ (∀ x, f a (x - ψ) = f a (-x)) → φ ≤ ψ) ∧
    φ = 3*π/4 :=
sorry

end minimum_translation_l1085_108542


namespace trevor_coin_count_l1085_108533

/-- Given that Trevor counted 29 quarters and has 48 more coins in total than quarters,
    prove that the total number of coins Trevor counted is 77. -/
theorem trevor_coin_count :
  let quarters : ℕ := 29
  let extra_coins : ℕ := 48
  quarters + extra_coins = 77
  := by sorry

end trevor_coin_count_l1085_108533


namespace inequality_proof_l1085_108529

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end inequality_proof_l1085_108529


namespace weekly_training_cost_l1085_108551

/-- Proves that the weekly training cost is $250, given the adoption fee, training duration, certification cost, insurance coverage, and total out-of-pocket cost. -/
theorem weekly_training_cost
  (adoption_fee : ℝ)
  (training_weeks : ℕ)
  (certification_cost : ℝ)
  (insurance_coverage : ℝ)
  (total_out_of_pocket : ℝ)
  (h1 : adoption_fee = 150)
  (h2 : training_weeks = 12)
  (h3 : certification_cost = 3000)
  (h4 : insurance_coverage = 0.9)
  (h5 : total_out_of_pocket = 3450)
  : ∃ (weekly_cost : ℝ),
    weekly_cost = 250 ∧
    total_out_of_pocket = adoption_fee + training_weeks * weekly_cost + (1 - insurance_coverage) * certification_cost :=
by sorry

end weekly_training_cost_l1085_108551


namespace set_A_is_empty_l1085_108516

def set_A : Set ℝ := {x : ℝ | x^2 + 2 = 0}
def set_B : Set ℝ := {0}
def set_C : Set ℝ := {x : ℝ | x > 8 ∨ x < 4}
def set_D : Set (Set ℝ) := {∅}

theorem set_A_is_empty : set_A = ∅ := by
  sorry

end set_A_is_empty_l1085_108516


namespace unique_perfect_square_l1085_108500

def f (k : ℕ) : ℕ := 2^k + 8*k + 5

theorem unique_perfect_square : ∃! k : ℕ, ∃ n : ℕ, f k = n^2 ∧ k = 2 := by
  sorry

end unique_perfect_square_l1085_108500


namespace set_partition_real_line_l1085_108567

theorem set_partition_real_line (m : ℝ) : 
  let A := {x : ℝ | x ≥ 3}
  let B := {x : ℝ | x < m}
  (A ∪ B = Set.univ) → (A ∩ B = ∅) → m = 3 := by
  sorry

end set_partition_real_line_l1085_108567


namespace jerry_field_hours_eq_96_l1085_108548

/-- The number of hours Jerry spends at the field watching his daughters play and practice -/
def jerry_field_hours : ℕ :=
  let num_daughters : ℕ := 2
  let games_per_daughter : ℕ := 8
  let practice_hours_per_game : ℕ := 4
  let game_duration_hours : ℕ := 2
  
  let game_hours_per_daughter : ℕ := games_per_daughter * game_duration_hours
  let practice_hours_per_daughter : ℕ := games_per_daughter * practice_hours_per_game
  
  num_daughters * (game_hours_per_daughter + practice_hours_per_daughter)

theorem jerry_field_hours_eq_96 : jerry_field_hours = 96 := by
  sorry

end jerry_field_hours_eq_96_l1085_108548


namespace election_win_margin_l1085_108522

theorem election_win_margin (total_votes : ℕ) (winner_votes : ℕ) : 
  winner_votes = (62 * total_votes) / 100 →
  winner_votes = 1054 →
  winner_votes - ((38 * total_votes) / 100) = 408 :=
by
  sorry

end election_win_margin_l1085_108522


namespace inequality_exists_n_l1085_108554

theorem inequality_exists_n : ∃ n : ℕ+, ∀ x : ℝ, x ≥ 0 → (x - 1) * (x^2005 - 2005*x^(n.val + 1) + 2005*x^n.val - 1) ≥ 0 := by
  sorry

end inequality_exists_n_l1085_108554


namespace arithmetic_sequence_10th_term_l1085_108517

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence :=
  (a : ℤ)  -- First term
  (d : ℤ)  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a + (n - 1) * seq.d

theorem arithmetic_sequence_10th_term
  (seq : ArithmeticSequence)
  (h4 : seq.nthTerm 4 = 23)
  (h8 : seq.nthTerm 8 = 55) :
  seq.nthTerm 10 = 71 := by
  sorry

#check arithmetic_sequence_10th_term

end arithmetic_sequence_10th_term_l1085_108517


namespace triangle_area_from_sides_and_median_l1085_108512

/-- Given a triangle PQR with side lengths and median, calculate its area -/
theorem triangle_area_from_sides_and_median 
  (PQ PR PM : ℝ) 
  (h_PQ : PQ = 8) 
  (h_PR : PR = 18) 
  (h_PM : PM = 12) : 
  ∃ (area : ℝ), area = Real.sqrt 2975 ∧ area > 0 := by
  sorry

end triangle_area_from_sides_and_median_l1085_108512


namespace geometric_sequence_property_l1085_108583

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence, if a_7 · a_19 = 8, then a_3 · a_23 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_prod : a 7 * a 19 = 8) : a 3 * a 23 = 8 := by
  sorry

end geometric_sequence_property_l1085_108583


namespace number_sequence_properties_l1085_108564

/-- Represents the sequence formed by concatenating numbers from 1 to 999 -/
def NumberSequence : Type := List Nat

/-- Constructs the NumberSequence -/
def createSequence : NumberSequence := sorry

/-- Counts the total number of digits in the sequence -/
def countDigits (seq : NumberSequence) : Nat := sorry

/-- Counts the occurrences of a specific digit in the sequence -/
def countDigitOccurrences (seq : NumberSequence) (digit : Nat) : Nat := sorry

/-- Finds the digit at a specific position in the sequence -/
def digitAtPosition (seq : NumberSequence) (position : Nat) : Nat := sorry

theorem number_sequence_properties (seq : NumberSequence) :
  seq = createSequence →
  (countDigits seq = 2889) ∧
  (countDigitOccurrences seq 1 = 300) ∧
  (digitAtPosition seq 2016 = 8) := by
  sorry

end number_sequence_properties_l1085_108564


namespace max_value_of_complex_sum_l1085_108596

theorem max_value_of_complex_sum (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z + 1 + Complex.I * Real.sqrt 3) ≤ 3 := by
sorry

end max_value_of_complex_sum_l1085_108596


namespace new_average_production_l1085_108577

/-- Given a company's production data, prove that the new average daily production is 45 units. -/
theorem new_average_production (n : ℕ) (past_average : ℝ) (today_production : ℝ) :
  n = 9 →
  past_average = 40 →
  today_production = 90 →
  (n * past_average + today_production) / (n + 1) = 45 := by
  sorry

end new_average_production_l1085_108577


namespace negation_of_universal_proposition_l1085_108534

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) ↔
  (∃ x : ℝ, x > 0 ∧ (x + 1) * Real.exp x ≤ 1) :=
by sorry

end negation_of_universal_proposition_l1085_108534


namespace no_real_solutions_l1085_108575

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (3 * x^2) / (x - 2) - (3 * x + 10) / 4 + (9 - 9 * x) / (x - 2) - 3 = 0

-- Theorem stating that the equation has no real solutions
theorem no_real_solutions : ¬∃ x : ℝ, original_equation x :=
sorry

end no_real_solutions_l1085_108575


namespace angle_through_point_l1085_108513

theorem angle_through_point (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α = 2 * Real.sqrt 5 / 5 ∧
  Real.cos α = -Real.sqrt 5 / 5 ∧
  Real.tan α = -2 ∧
  Real.tan (α - Real.pi/4) = 3 := by
sorry

end angle_through_point_l1085_108513


namespace tank_capacity_is_72_liters_l1085_108592

/-- The total capacity of a water tank in liters. -/
def tank_capacity : ℝ := 72

/-- The amount of water in the tank when it's 40% full, in liters. -/
def water_at_40_percent : ℝ := 0.4 * tank_capacity

/-- The amount of water in the tank when it's 10% empty, in liters. -/
def water_at_10_percent_empty : ℝ := 0.9 * tank_capacity

/-- Theorem stating that the tank capacity is 72 liters, given the condition. -/
theorem tank_capacity_is_72_liters :
  water_at_10_percent_empty - water_at_40_percent = 36 →
  tank_capacity = 72 :=
by sorry

end tank_capacity_is_72_liters_l1085_108592


namespace second_encounter_correct_l1085_108528

/-- Represents the highway with speed limit signs and monitoring devices -/
structure Highway where
  speed_limit_start : ℕ := 3
  speed_limit_interval : ℕ := 4
  monitoring_start : ℕ := 10
  monitoring_interval : ℕ := 9
  first_encounter : ℕ := 19

/-- The kilometer mark of the second simultaneous encounter -/
def second_encounter (h : Highway) : ℕ := 55

/-- Theorem stating that the second encounter occurs at 55 km -/
theorem second_encounter_correct (h : Highway) : 
  second_encounter h = 55 := by sorry

end second_encounter_correct_l1085_108528


namespace minimum_score_needed_l1085_108563

def current_scores : List ℕ := [90, 80, 70, 60, 85]
def score_count : ℕ := current_scores.length
def current_average : ℚ := (current_scores.sum : ℚ) / score_count
def target_increase : ℚ := 3
def new_score_count : ℕ := score_count + 1

theorem minimum_score_needed (x : ℕ) : 
  (((current_scores.sum + x) : ℚ) / new_score_count ≥ current_average + target_increase) ↔ 
  (x ≥ 95) :=
sorry

end minimum_score_needed_l1085_108563


namespace locus_of_P_l1085_108560

/-- The locus of point P given the conditions in the problem -/
theorem locus_of_P (F Q T P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  F = (2, 0) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (k : ℝ), y = k * (x - 2)) →
  Q.1 = 0 →
  Q ∈ l →
  (T.2 = 0 ∧ (Q.1 - T.1) * (F.1 - Q.1) = (F.2 - Q.2) * (Q.2 - T.2)) →
  (T.1 - Q.1)^2 + (T.2 - Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 →
  P.2^2 = 8 * P.1 :=
by sorry

end locus_of_P_l1085_108560


namespace circle_symmetry_about_origin_l1085_108524

/-- Given a circle with equation (x-1)^2+(y+2)^2=5, 
    prove that (x+1)^2+(y-2)^2=5 is its symmetric about the origin -/
theorem circle_symmetry_about_origin :
  let original_circle := (fun (x y : ℝ) => (x - 1)^2 + (y + 2)^2 = 5)
  let symmetric_circle := (fun (x y : ℝ) => (x + 1)^2 + (y - 2)^2 = 5)
  ∀ (x y : ℝ), original_circle (-x) (-y) ↔ symmetric_circle x y :=
by sorry

end circle_symmetry_about_origin_l1085_108524


namespace repeating_decimal_subtraction_l1085_108535

theorem repeating_decimal_subtraction : 
  let a : ℚ := 234 / 999
  let b : ℚ := 567 / 999
  let c : ℚ := 891 / 999
  a - b - c = -1224 / 999 := by sorry

end repeating_decimal_subtraction_l1085_108535


namespace direct_variation_problem_l1085_108561

-- Define the direct variation relationship
def direct_variation (y x : ℝ) := ∃ k : ℝ, y = k * x

-- State the theorem
theorem direct_variation_problem :
  ∀ y : ℝ → ℝ,
  (∀ x : ℝ, direct_variation (y x) x) →
  y 4 = 8 →
  y (-8) = -16 :=
by
  sorry

end direct_variation_problem_l1085_108561


namespace max_sum_cubes_l1085_108594

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  ∃ (M : ℝ), M = 5 * Real.sqrt 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 ≤ M ∧
  ∃ (a' b' c' d' e' : ℝ), a'^2 + b'^2 + c'^2 + d'^2 + e'^2 = 5 ∧
                           a'^3 + b'^3 + c'^3 + d'^3 + e'^3 = M :=
by
  sorry

end max_sum_cubes_l1085_108594


namespace xavier_yvonne_not_zelda_probability_l1085_108530

/-- The probability that Xavier and Yvonne solve a problem but Zelda does not, 
    given their individual probabilities of success. -/
theorem xavier_yvonne_not_zelda_probability 
  (p_xavier : ℚ) (p_yvonne : ℚ) (p_zelda : ℚ)
  (h_xavier : p_xavier = 1/6)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/32 :=
sorry

end xavier_yvonne_not_zelda_probability_l1085_108530


namespace rectangle_diagonal_l1085_108511

theorem rectangle_diagonal (x y a b : ℝ) (h1 : π * x^2 * y = a) (h2 : π * y^2 * x = b) :
  (x^2 + y^2).sqrt = ((a^2 + b^2) / (a * b)).sqrt * ((a * b) / π^2)^(1/6) := by
  sorry

end rectangle_diagonal_l1085_108511


namespace exists_fixed_point_l1085_108589

def S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 999}

def is_fixed_point (f : S → S) (a : S) : Prop := f a = a

def satisfies_condition (f : S → S) : Prop :=
  ∀ n : S, (f^[n + f n + 1] n = n) ∧ (f^[n * f n] n = n)

theorem exists_fixed_point (f : S → S) (h : satisfies_condition f) :
  ∃ a : S, is_fixed_point f a := by
  sorry

end exists_fixed_point_l1085_108589


namespace hexagon_longest_side_range_l1085_108541

/-- Given a hexagon formed by wrapping a line segment of length 20,
    the length of its longest side is between 10/3 and 10 (exclusive). -/
theorem hexagon_longest_side_range :
  ∀ x : ℝ,
    (∃ a b c d e f : ℝ,
      a + b + c + d + e + f = 20 ∧
      x = max a (max b (max c (max d (max e f)))) ∧
      a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0) →
    (10 / 3 : ℝ) ≤ x ∧ x < 10 :=
by sorry

end hexagon_longest_side_range_l1085_108541


namespace orthic_triangle_similarity_l1085_108576

/-- A triangle with angles A, B, and C in degrees -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180
  positive : 0 < A ∧ 0 < B ∧ 0 < C

/-- The orthic triangle of a given triangle -/
def orthicTriangle (t : Triangle) : Triangle where
  A := 180 - 2 * t.A
  B := 180 - 2 * t.B
  C := 180 - 2 * t.C
  sum_180 := sorry
  positive := sorry

/-- Two triangles are similar if their corresponding angles are equal -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.A = t2.A ∧ t1.B = t2.B ∧ t1.C = t2.C

theorem orthic_triangle_similarity (t : Triangle) 
  (h_not_right : t.A ≠ 90 ∧ t.B ≠ 90 ∧ t.C ≠ 90) :
  similar t (orthicTriangle t) ↔ t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end orthic_triangle_similarity_l1085_108576


namespace no_valid_n_l1085_108573

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end no_valid_n_l1085_108573


namespace fern_bushes_needed_l1085_108543

/-- The number of bushes needed to produce a given amount of perfume -/
def bushes_needed (petals_per_ounce : ℕ) (petals_per_rose : ℕ) (roses_per_bush : ℕ) 
                  (ounces_per_bottle : ℕ) (num_bottles : ℕ) : ℕ :=
  (petals_per_ounce * ounces_per_bottle * num_bottles) / (petals_per_rose * roses_per_bush)

/-- Theorem stating the number of bushes Fern needs to harvest -/
theorem fern_bushes_needed : 
  bushes_needed 320 8 12 12 20 = 800 := by
  sorry

end fern_bushes_needed_l1085_108543


namespace recurrence_sequence_b8_l1085_108514

/-- An increasing sequence of positive integers satisfying the given recurrence relation -/
def RecurrenceSequence (b : ℕ → ℕ) : Prop :=
  (∀ n, b n < b (n + 1)) ∧ 
  (∀ n, 1 ≤ n → b (n + 2) = b (n + 1) + b n)

/-- The theorem statement -/
theorem recurrence_sequence_b8 (b : ℕ → ℕ) 
  (h : RecurrenceSequence b) (h7 : b 7 = 198) : b 8 = 321 := by
  sorry

end recurrence_sequence_b8_l1085_108514


namespace distance_from_origin_l1085_108580

theorem distance_from_origin (x y : ℝ) (h1 : y = 15) 
  (h2 : Real.sqrt ((x - 2)^2 + (y - 8)^2) = 13) (h3 : x > 2) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (349 + 8 * Real.sqrt 30) := by
sorry

end distance_from_origin_l1085_108580


namespace playground_students_count_l1085_108531

/-- Represents the seating arrangement on the playground -/
structure PlaygroundSeating where
  left : Nat
  right : Nat
  front : Nat
  back : Nat

/-- Calculates the total number of students on the playground -/
def totalStudents (s : PlaygroundSeating) : Nat :=
  ((s.left + s.right - 1) * (s.front + s.back - 1))

/-- Theorem stating the total number of students on the playground -/
theorem playground_students_count (yujeong : PlaygroundSeating) 
  (h1 : yujeong.left = 12)
  (h2 : yujeong.right = 11)
  (h3 : yujeong.front = 18)
  (h4 : yujeong.back = 8) :
  totalStudents yujeong = 550 := by
  sorry

#check playground_students_count

end playground_students_count_l1085_108531


namespace train_passing_tree_l1085_108559

/-- Proves that a train of given length and speed takes a specific time to pass a tree -/
theorem train_passing_tree (train_length : ℝ) (train_speed_kmh : ℝ) (time : ℝ) :
  train_length = 280 →
  train_speed_kmh = 72 →
  time = train_length / (train_speed_kmh * (5/18)) →
  time = 14 := by
  sorry

#check train_passing_tree

end train_passing_tree_l1085_108559


namespace min_value_trig_expression_l1085_108571

open Real

theorem min_value_trig_expression (θ : ℝ) (h : 0 < θ ∧ θ < π / 2) :
  ∃ (min_val : ℝ), min_val = 9 * sqrt 3 ∧
  ∀ θ', 0 < θ' ∧ θ' < π / 2 →
    3 * sin θ' + 4 * (1 / cos θ') + 2 * sqrt 3 * tan θ' ≥ min_val :=
by sorry

end min_value_trig_expression_l1085_108571


namespace polygon_sides_l1085_108552

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 4 * 360 + 180) : n = 9 := by
  sorry

end polygon_sides_l1085_108552


namespace parabola_symmetric_points_l1085_108570

/-- A parabola with parameter p > 0 has two distinct points symmetrical with respect to the line x + y = 1 if and only if 0 < p < 2/3 -/
theorem parabola_symmetric_points (p : ℝ) :
  (p > 0) →
  (∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    (A.2)^2 = 2*p*A.1 ∧
    (B.2)^2 = 2*p*B.1 ∧
    (∃ (C : ℝ × ℝ),
      C.1 + C.2 = 1 ∧
      C.1 = (A.1 + B.1) / 2 ∧
      C.2 = (A.2 + B.2) / 2)) ↔
  (0 < p ∧ p < 2/3) :=
by sorry

end parabola_symmetric_points_l1085_108570


namespace product_sum_relation_l1085_108527

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 14 → b = 8 → b - a = 3 := by
  sorry

end product_sum_relation_l1085_108527


namespace trig_identity_l1085_108581

theorem trig_identity (α : Real) (h : Real.sin (α - π/12) = 1/3) : 
  Real.cos (α + 5*π/12) = -1/3 := by
  sorry

end trig_identity_l1085_108581


namespace area_circle_inscribed_equilateral_triangle_l1085_108539

theorem area_circle_inscribed_equilateral_triangle (p : ℝ) (h : p > 0) :
  ∃ (R : ℝ), R > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  p = 3 * s ∧
  R = s / Real.sqrt 3 ∧
  π * R^2 = π * p^2 / 27 :=
by sorry

end area_circle_inscribed_equilateral_triangle_l1085_108539


namespace identity_element_is_one_zero_l1085_108550

-- Define the operation ⊕
def oplus (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, a * d + b * c)

-- State the theorem
theorem identity_element_is_one_zero :
  (∀ a b : ℝ, oplus a b x y = (a, b)) → (x, y) = (1, 0) := by
  sorry

end identity_element_is_one_zero_l1085_108550


namespace solve_for_x_l1085_108599

theorem solve_for_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end solve_for_x_l1085_108599


namespace lucy_total_cost_l1085_108509

/-- The total cost Lucy paid for a lamp and a table, given specific pricing conditions. -/
theorem lucy_total_cost : 
  ∀ (lamp_original_price lamp_discounted_price table_price : ℝ),
  lamp_discounted_price = 20 →
  lamp_discounted_price = (1/5) * (0.6 * lamp_original_price) →
  table_price = 2 * lamp_original_price →
  lamp_discounted_price + table_price = 353.34 := by
  sorry

#check lucy_total_cost

end lucy_total_cost_l1085_108509


namespace dog_roaming_area_l1085_108579

/-- The area a dog can roam when tied to a circular pillar -/
theorem dog_roaming_area (leash_length : ℝ) (pillar_radius : ℝ) (roaming_area : ℝ) : 
  leash_length = 10 →
  pillar_radius = 2 →
  roaming_area = π * (leash_length + pillar_radius)^2 →
  roaming_area = 144 * π :=
by sorry

end dog_roaming_area_l1085_108579


namespace three_letter_words_with_E_count_l1085_108505

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}
def word_length : Nat := 3

def total_words : Nat := alphabet.card ^ word_length
def words_without_E : Nat := (alphabet.card - 1) ^ word_length

theorem three_letter_words_with_E_count :
  total_words - words_without_E = 61 := by
  sorry

end three_letter_words_with_E_count_l1085_108505


namespace pie_eating_contest_l1085_108557

/-- Pie-eating contest theorem -/
theorem pie_eating_contest 
  (adam bill sierra taylor : ℕ) -- Number of pies eaten by each participant
  (h1 : adam = bill + 3) -- Adam eats three more pies than Bill
  (h2 : sierra = 2 * bill) -- Sierra eats twice as many pies as Bill
  (h3 : taylor = (adam + bill + sierra) / 3) -- Taylor eats the average of Adam, Bill, and Sierra
  (h4 : sierra = 12) -- Sierra ate 12 pies
  : adam + bill + sierra + taylor = 36 ∧ adam + bill + sierra + taylor ≤ 50 := by
  sorry

end pie_eating_contest_l1085_108557


namespace sphere_surface_area_l1085_108565

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * π * r^2 = 36 * π := by
  sorry

end sphere_surface_area_l1085_108565


namespace determinant_evaluation_l1085_108521

theorem determinant_evaluation (x z : ℝ) : 
  Matrix.det !![1, x, z; 1, x + z, z; 1, x, x + z] = x * z + 2 * z^2 := by
  sorry

end determinant_evaluation_l1085_108521


namespace second_discount_percentage_l1085_108506

theorem second_discount_percentage
  (normal_price : ℝ)
  (first_discount_rate : ℝ)
  (final_price : ℝ)
  (h1 : normal_price = 174.99999999999997)
  (h2 : first_discount_rate = 0.1)
  (h3 : final_price = 126) :
  let price_after_first_discount := normal_price * (1 - first_discount_rate)
  let second_discount_rate := (price_after_first_discount - final_price) / price_after_first_discount
  second_discount_rate = 0.2 := by
sorry

#eval (174.99999999999997 * 0.9 - 126) / (174.99999999999997 * 0.9)

end second_discount_percentage_l1085_108506


namespace smallest_solution_of_equation_l1085_108572

theorem smallest_solution_of_equation (x : ℝ) : 
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → 
  (x ≥ 4 - Real.sqrt 2 ∧ 
   (1 / ((4 - Real.sqrt 2) - 3) + 1 / ((4 - Real.sqrt 2) - 5) = 4 / ((4 - Real.sqrt 2) - 4))) := by
  sorry

end smallest_solution_of_equation_l1085_108572


namespace quadratic_min_diff_l1085_108501

/-- The quadratic function f(x) = ax² - 2020x + 2021 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2020 * x + 2021

/-- The theorem stating that if the minimum difference between max and min values
    of f on any 2-unit interval is 2, then a must be 2 -/
theorem quadratic_min_diff (a : ℝ) (h_pos : a > 0) :
  (∀ t : ℝ, ∃ M N : ℝ,
    (∀ x ∈ Set.Icc (t - 1) (t + 1), f a x ≤ M) ∧
    (∀ x ∈ Set.Icc (t - 1) (t + 1), N ≤ f a x) ∧
    (∀ K L : ℝ,
      (∀ x ∈ Set.Icc (t - 1) (t + 1), f a x ≤ K) →
      (∀ x ∈ Set.Icc (t - 1) (t + 1), L ≤ f a x) →
      2 ≤ K - L)) →
  a = 2 :=
sorry

end quadratic_min_diff_l1085_108501


namespace hyperbola_trisect_foci_eccentricity_l1085_108582

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- Theorem: If the vertices of a hyperbola trisect the line segment between its foci,
    then its eccentricity is 3 -/
theorem hyperbola_trisect_foci_eccentricity (a b : ℝ) (h : Hyperbola a b) 
    (trisect : ∃ (c : ℝ), c = 3 * a) : eccentricity h = 3 := by sorry

end hyperbola_trisect_foci_eccentricity_l1085_108582


namespace sum_odd_implies_difference_odd_l1085_108536

theorem sum_odd_implies_difference_odd (a b : ℤ) : 
  Odd (a + b) → Odd (a - b) := by
  sorry

end sum_odd_implies_difference_odd_l1085_108536


namespace probability_four_ones_in_five_rolls_l1085_108555

/-- The probability of rolling a 1 on a fair six-sided die -/
def p_one : ℚ := 1/6

/-- The probability of not rolling a 1 on a fair six-sided die -/
def p_not_one : ℚ := 5/6

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll a 1 -/
def num_ones : ℕ := 4

/-- The number of ways to choose the positions for the non-1 roll -/
def num_arrangements : ℕ := 5

theorem probability_four_ones_in_five_rolls :
  num_arrangements * p_one^num_ones * p_not_one^(num_rolls - num_ones) = 25/7776 := by
  sorry

end probability_four_ones_in_five_rolls_l1085_108555


namespace bonnets_per_orphanage_l1085_108569

/-- The number of bonnets made on Monday -/
def monday_bonnets : ℕ := 10

/-- The number of bonnets made on Tuesday and Wednesday combined -/
def tuesday_wednesday_bonnets : ℕ := 2 * monday_bonnets

/-- The number of bonnets made on Thursday -/
def thursday_bonnets : ℕ := monday_bonnets + 5

/-- The number of bonnets made on Friday -/
def friday_bonnets : ℕ := thursday_bonnets - 5

/-- The total number of bonnets made -/
def total_bonnets : ℕ := monday_bonnets + tuesday_wednesday_bonnets + thursday_bonnets + friday_bonnets

/-- The number of orphanages -/
def num_orphanages : ℕ := 5

/-- Theorem stating the number of bonnets sent to each orphanage -/
theorem bonnets_per_orphanage : total_bonnets / num_orphanages = 11 := by
  sorry

end bonnets_per_orphanage_l1085_108569


namespace shanghai_score_is_75_l1085_108545

/-- The score of Yao Ming in the basketball game -/
def yao_ming_score : ℕ := 30

/-- The winning margin of the Shanghai team over the Beijing team -/
def shanghai_margin : ℕ := 10

/-- Calculates the total score of both teams based on Yao Ming's score -/
def total_score (yao_score : ℕ) : ℕ := 5 * yao_score - 10

/-- The score of the Shanghai team -/
def shanghai_score : ℕ := 75

/-- The score of the Beijing team -/
def beijing_score : ℕ := shanghai_score - shanghai_margin

theorem shanghai_score_is_75 :
  shanghai_score = 75 ∧
  shanghai_score - beijing_score = shanghai_margin ∧
  shanghai_score + beijing_score = total_score yao_ming_score :=
by sorry

end shanghai_score_is_75_l1085_108545


namespace prob_sum_15_equals_11_663_l1085_108504

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each rank (2 through 10) in a standard deck -/
def cards_per_rank : ℕ := 4

/-- The set of possible ranks that can sum to 15 -/
def valid_ranks : Finset ℕ := {6, 7, 8}

/-- The probability of selecting two number cards that sum to 15 from a standard deck -/
def prob_sum_15 : ℚ :=
  (cards_per_rank * cards_per_rank * 2 + cards_per_rank * (cards_per_rank - 1)) / (deck_size * (deck_size - 1))

theorem prob_sum_15_equals_11_663 : prob_sum_15 = 11 / 663 := by
  sorry

end prob_sum_15_equals_11_663_l1085_108504


namespace smallest_upper_bound_l1085_108593

/-- The set of functions satisfying the given conditions -/
def S : Set (ℕ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n, f (n + 1) ≥ f n ∧ f n ≥ (n / (n + 1 : ℝ)) * f (2 * n)}

/-- The smallest natural number M such that f(n) < M for all f ∈ S and n ∈ ℕ -/
theorem smallest_upper_bound : ∃! M : ℕ, 
  (∀ f ∈ S, ∀ n, f n < M) ∧ 
  (∀ M' : ℕ, (∀ f ∈ S, ∀ n, f n < M') → M ≤ M') :=
by
  use 10
  sorry

#check smallest_upper_bound

end smallest_upper_bound_l1085_108593


namespace number_problem_l1085_108585

theorem number_problem (x : ℝ) : 0.4 * x - 11 = 23 → x = 85 := by sorry

end number_problem_l1085_108585


namespace f_minimum_g_solution_set_l1085_108574

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem for the minimum value of f
theorem f_minimum : ∀ x : ℝ, f x ≥ -3 :=
sorry

-- Define the inequality function g
def g (x : ℝ) : ℝ := x^2 - 8*x + 15 + f x

-- Theorem for the solution set of g(x) ≤ 0
theorem g_solution_set : 
  ∀ x : ℝ, g x ≤ 0 ↔ 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6 :=
sorry

end f_minimum_g_solution_set_l1085_108574


namespace huahuan_initial_cards_l1085_108519

/-- Represents the card distribution among the three players -/
structure CardDistribution where
  huahuan : ℕ
  yingying : ℕ
  nini : ℕ

/-- Represents one round of operations -/
def performRound (dist : CardDistribution) : CardDistribution :=
  sorry

/-- Check if the distribution forms an arithmetic sequence -/
def isArithmeticSequence (dist : CardDistribution) : Prop :=
  dist.yingying - dist.huahuan = dist.nini - dist.yingying

/-- The main theorem -/
theorem huahuan_initial_cards 
  (initial : CardDistribution)
  (h1 : initial.huahuan + initial.yingying + initial.nini = 2712)
  (h2 : ∃ (final : CardDistribution), 
    (performRound^[50] initial = final) ∧ 
    (isArithmeticSequence final)) :
  initial.huahuan = 754 := by
  sorry


end huahuan_initial_cards_l1085_108519


namespace count_special_numbers_l1085_108587

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def satisfies_condition (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n = a + b^2 + c^3

theorem count_special_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_three_digit n ∧ satisfies_condition n) ∧
                    (∀ n, is_three_digit n → satisfies_condition n → n ∈ S) ∧
                    Finset.card S = 4 :=
sorry

end count_special_numbers_l1085_108587


namespace inequality_not_holding_l1085_108558

theorem inequality_not_holding (x y : ℝ) (h : x > y) : ¬(-2*x > -2*y) := by
  sorry

end inequality_not_holding_l1085_108558


namespace k_range_l1085_108507

theorem k_range (x y k : ℝ) : 
  3 * x + y = k + 1 →
  x + 3 * y = 3 →
  0 < x + y →
  x + y < 1 →
  -4 < k ∧ k < 0 := by
sorry

end k_range_l1085_108507


namespace selections_with_former_eq_2850_l1085_108503

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of coordinators to be selected -/
def num_coordinators : ℕ := 4

/-- The total number of members -/
def total_members : ℕ := 18

/-- The number of former coordinators -/
def former_coordinators : ℕ := 8

/-- The number of selections including at least one former coordinator -/
def selections_with_former : ℕ :=
  choose total_members num_coordinators - choose (total_members - former_coordinators) num_coordinators

theorem selections_with_former_eq_2850 : selections_with_former = 2850 := by sorry

end selections_with_former_eq_2850_l1085_108503


namespace red_balls_count_l1085_108562

theorem red_balls_count (total_balls : ℕ) (red_probability : ℚ) (h1 : total_balls = 20) (h2 : red_probability = 1/4) :
  (red_probability * total_balls : ℚ) = 5 := by
  sorry

end red_balls_count_l1085_108562


namespace alice_bracelet_profit_l1085_108538

def bracelet_profit (initial_bracelets : ℕ) (material_cost : ℚ) 
                    (given_away : ℕ) (selling_price : ℚ) : ℚ :=
  let remaining_bracelets := initial_bracelets - given_away
  let revenue := (remaining_bracelets : ℚ) * selling_price
  revenue - material_cost

theorem alice_bracelet_profit :
  bracelet_profit 52 3 8 (1/4) = 8 := by
  sorry

end alice_bracelet_profit_l1085_108538


namespace probability_of_event_a_l1085_108584

theorem probability_of_event_a 
  (prob_b : ℝ) 
  (prob_a_and_b : ℝ) 
  (prob_neither_a_nor_b : ℝ) 
  (h1 : prob_b = 0.40)
  (h2 : prob_a_and_b = 0.15)
  (h3 : prob_neither_a_nor_b = 0.5499999999999999) : 
  ∃ (prob_a : ℝ), prob_a = 0.20 := by
  sorry

#check probability_of_event_a

end probability_of_event_a_l1085_108584


namespace symmetry_problem_l1085_108556

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given point P -/
def P : Point3D := { x := -2, y := 1, z := 4 }

/-- Given point A -/
def A : Point3D := { x := 1, y := 0, z := 2 }

/-- Reflect a point about the xOy plane -/
def reflectXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

/-- Find the point symmetric to a given point about another point -/
def symmetricPoint (p q : Point3D) : Point3D :=
  { x := 2 * p.x - q.x,
    y := 2 * p.y - q.y,
    z := 2 * p.z - q.z }

theorem symmetry_problem :
  reflectXOY P = { x := -2, y := 1, z := -4 } ∧
  symmetricPoint P A = { x := -5, y := 2, z := 6 } := by
  sorry

end symmetry_problem_l1085_108556


namespace f_positive_before_root_l1085_108546

noncomputable def f (x : ℝ) : ℝ := (1/3)^x - Real.log x / Real.log 2

theorem f_positive_before_root (x₀ a : ℝ) 
  (h_root : f x₀ = 0)
  (h_decreasing : ∀ x y, x < y → f x > f y)
  (h_a_pos : 0 < a)
  (h_a_lt_x₀ : a < x₀) : 
  f a > 0 := by
  sorry

end f_positive_before_root_l1085_108546


namespace replaced_person_age_l1085_108537

theorem replaced_person_age 
  (n : ℕ) 
  (original_total_age : ℕ) 
  (new_person_age : ℕ) 
  (average_decrease : ℕ) :
  n = 10 →
  new_person_age = 10 →
  average_decrease = 3 →
  (original_total_age : ℚ) / n - average_decrease = 
    (original_total_age - (original_total_age / n * n - new_person_age) : ℚ) / n →
  (original_total_age / n * n - new_person_age : ℚ) / n = 40 :=
by sorry

end replaced_person_age_l1085_108537


namespace truck_speed_problem_l1085_108591

theorem truck_speed_problem (v : ℝ) : 
  v > 0 →  -- Truck speed is positive
  (60 * 4 = v * 5) →  -- Car catches up after 4 hours
  v = 48 := by
sorry

end truck_speed_problem_l1085_108591


namespace decimal_places_in_expression_l1085_108597

-- Define the original number
def original_number : ℝ := 3.456789

-- Define the expression
def expression : ℝ := ((10^4 : ℝ) * original_number)^9

-- Function to count decimal places
def count_decimal_places (x : ℝ) : ℕ :=
  sorry

-- Theorem stating that the number of decimal places in the expression is 2
theorem decimal_places_in_expression :
  count_decimal_places expression = 2 := by
  sorry

end decimal_places_in_expression_l1085_108597


namespace binomial_sum_theorem_l1085_108540

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the left-hand side of the first equation
def lhs1 (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the right-hand side of the first equation
def rhs1 (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the left-hand side of the second equation
def lhs2 (n : ℕ) : ℕ := sorry

-- Define the right-hand side of the second equation
def rhs2 (n : ℕ) : ℕ := sorry

-- State the theorem
theorem binomial_sum_theorem (n : ℕ) (hn : n ≥ 1) :
  (∀ x : ℝ, lhs1 n x = rhs1 n x) ∧ (lhs2 n = rhs2 n) := by sorry

end binomial_sum_theorem_l1085_108540


namespace intersection_points_on_circle_l1085_108518

-- Define the parabolas
def Parabola1 (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def Parabola2 (d e f : ℝ) (y : ℝ) : ℝ := d * y^2 + e * y + f

-- Define the intersection points
def IntersectionPoints (a b c d e f : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | Parabola1 a b c x = y ∧ Parabola2 d e f y = x}

-- Define a circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | (x - center.1)^2 + (y - center.2)^2 = radius^2}

-- Theorem statement
theorem intersection_points_on_circle 
  (a b c d e f : ℝ) (ha : a > 0) (hd : d > 0) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    IntersectionPoints a b c d e f ⊆ Circle center radius :=
sorry

end intersection_points_on_circle_l1085_108518


namespace common_divisors_84_90_l1085_108526

theorem common_divisors_84_90 : 
  (Finset.filter (λ x => x ∣ 84 ∧ x ∣ 90) (Finset.range (max 84 90 + 1))).card = 8 := by
  sorry

end common_divisors_84_90_l1085_108526


namespace simplify_expression_l1085_108547

theorem simplify_expression (x : ℝ) : (x + 15) + (150 * x + 20) = 151 * x + 35 := by
  sorry

end simplify_expression_l1085_108547


namespace inequality_proof_l1085_108520

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c = 1 → (2 * a * b) / (a + b) + (2 * b * c) / (b + c) + (2 * a * c) / (a + c) ≤ 1) := by
  sorry

end inequality_proof_l1085_108520


namespace fgh_supermarkets_l1085_108566

theorem fgh_supermarkets (total : ℕ) (difference : ℕ) (us_count : ℕ) : 
  total = 84 → difference = 10 → us_count = total / 2 + difference / 2 → us_count = 47 := by
  sorry

end fgh_supermarkets_l1085_108566


namespace binary_to_base5_conversion_l1085_108502

-- Define the binary number 1101₂
def binary_num : ℕ := 13

-- Define the base-5 number 23₅
def base5_num : ℕ := 2 * 5 + 3

-- Theorem stating the equality of the two representations
theorem binary_to_base5_conversion :
  binary_num = base5_num := by
  sorry

end binary_to_base5_conversion_l1085_108502


namespace population_change_l1085_108586

/-- The initial population of a village that underwent several population changes --/
def initial_population : ℕ :=
  -- Define the initial population (to be proved)
  6496

/-- The final population after a series of events --/
def final_population : ℕ :=
  -- Given final population
  4555

/-- Theorem stating the relationship between initial and final population --/
theorem population_change (P : ℕ) :
  P = initial_population →
  (1.10 : ℝ) * ((0.75 : ℝ) * ((0.85 : ℝ) * P)) = final_population := by
  sorry


end population_change_l1085_108586


namespace quadratic_factorization_l1085_108578

theorem quadratic_factorization (p q : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 3/2 ∧ 
   ∀ x : ℝ, 2*x^2 + p*x + q = 0 ↔ x = x₁ ∨ x = x₂) →
  ∀ x : ℝ, 2*x^2 + p*x + q = 0 ↔ (x + 2)*(2*x - 3) = 0 :=
by sorry

end quadratic_factorization_l1085_108578


namespace coefficient_proof_l1085_108523

theorem coefficient_proof (x : ℕ) (some_number : ℕ) :
  x = 13 →
  (2^x) - (2^(x-2)) = some_number * (2^11) →
  some_number = 3 := by
sorry

end coefficient_proof_l1085_108523
