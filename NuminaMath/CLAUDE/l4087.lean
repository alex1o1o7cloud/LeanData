import Mathlib

namespace min_value_theorem_min_value_achieved_l4087_408747

theorem min_value_theorem (a : ℝ) (h : a > 1) : (4 / (a - 1)) + a ≥ 6 := by
  sorry

theorem min_value_achieved (ε : ℝ) (h : ε > 0) : 
  ∃ a : ℝ, a > 1 ∧ (4 / (a - 1)) + a < 6 + ε := by
  sorry

end min_value_theorem_min_value_achieved_l4087_408747


namespace robotics_workshop_average_age_l4087_408767

theorem robotics_workshop_average_age (total_members : Nat) (overall_avg : Nat) 
  (num_girls num_boys num_adults : Nat) (avg_girls avg_boys : Nat) :
  total_members = 50 →
  overall_avg = 21 →
  num_girls = 25 →
  num_boys = 20 →
  num_adults = 5 →
  avg_girls = 18 →
  avg_boys = 20 →
  (total_members * overall_avg - num_girls * avg_girls - num_boys * avg_boys) / num_adults = 40 :=
by sorry

end robotics_workshop_average_age_l4087_408767


namespace remainder_problem_l4087_408737

theorem remainder_problem : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end remainder_problem_l4087_408737


namespace average_age_increase_proof_l4087_408720

/-- The initial number of men in a group where replacing two men with two women increases the average age by 2 years -/
def initial_men_count : ℕ := 8

theorem average_age_increase_proof :
  let men_removed_age_sum := 20 + 28
  let women_added_age_sum := 32 + 32
  let age_difference := women_added_age_sum - men_removed_age_sum
  let average_age_increase := 2
  initial_men_count * average_age_increase = age_difference := by
  sorry

#check average_age_increase_proof

end average_age_increase_proof_l4087_408720


namespace triangle_side_relation_l4087_408759

theorem triangle_side_relation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a^2 - 16*b^2 - c^2 + 6*a*b + 10*b*c = 0) :
  a + c = 2*b := by sorry

end triangle_side_relation_l4087_408759


namespace marathon_length_l4087_408711

/-- A marathon runner completes a race under specific conditions. -/
theorem marathon_length (initial_distance : ℝ) (initial_time : ℝ) (total_time : ℝ) 
  (pace_ratio : ℝ) (marathon_length : ℝ) : 
  initial_distance = 10 →
  initial_time = 1 →
  total_time = 3 →
  pace_ratio = 0.8 →
  marathon_length = initial_distance + 
    (total_time - initial_time) * (initial_distance / initial_time) * pace_ratio →
  marathon_length = 26 := by
  sorry

#check marathon_length

end marathon_length_l4087_408711


namespace find_number_l4087_408739

theorem find_number : ∃ n : ℤ, 695 - 329 = n - 254 ∧ n = 620 := by
  sorry

end find_number_l4087_408739


namespace sinusoid_amplitude_l4087_408710

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
if the function oscillates between 5 and -3, then the amplitude a is equal to 4.
-/
theorem sinusoid_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) → a = 4 := by
  sorry

end sinusoid_amplitude_l4087_408710


namespace max_pads_purchase_existence_of_max_purchase_l4087_408777

def cost_pin : ℕ := 2
def cost_pen : ℕ := 3
def cost_pad : ℕ := 9
def total_budget : ℕ := 60

def is_valid_purchase (pins pens pads : ℕ) : Prop :=
  pins ≥ 1 ∧ pens ≥ 1 ∧ pads ≥ 1 ∧
  cost_pin * pins + cost_pen * pens + cost_pad * pads = total_budget

theorem max_pads_purchase :
  ∀ pins pens pads : ℕ, is_valid_purchase pins pens pads → pads ≤ 5 :=
by sorry

theorem existence_of_max_purchase :
  ∃ pins pens : ℕ, is_valid_purchase pins pens 5 :=
by sorry

end max_pads_purchase_existence_of_max_purchase_l4087_408777


namespace pascal_triangle_row20_sum_l4087_408713

theorem pascal_triangle_row20_sum : Nat.choose 20 4 + Nat.choose 20 5 = 20349 := by sorry

end pascal_triangle_row20_sum_l4087_408713


namespace problem_statement_l4087_408744

theorem problem_statement (p q r u v w : ℝ) 
  (eq1 : 17 * u + q * v + r * w = 0)
  (eq2 : p * u + 29 * v + r * w = 0)
  (eq3 : p * u + q * v + 56 * w = 0)
  (h1 : p ≠ 17)
  (h2 : u ≠ 0) :
  p / (p - 17) + q / (q - 29) + r / (r - 56) = 0 := by
  sorry

end problem_statement_l4087_408744


namespace max_tickets_with_budget_l4087_408795

theorem max_tickets_with_budget (ticket_price : ℚ) (budget : ℚ) (max_tickets : ℕ) : 
  ticket_price = 15 → budget = 120 → max_tickets = 8 → 
  (∀ n : ℕ, n * ticket_price ≤ budget ↔ n ≤ max_tickets) := by
sorry

end max_tickets_with_budget_l4087_408795


namespace gcd_of_256_162_720_l4087_408701

theorem gcd_of_256_162_720 : Nat.gcd 256 (Nat.gcd 162 720) = 18 := by
  sorry

end gcd_of_256_162_720_l4087_408701


namespace first_term_range_l4087_408787

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = 1 / (2 - a n)

/-- The property that each term is greater than the previous one -/
def StrictlyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The main theorem stating the range of the first term -/
theorem first_term_range
  (a : ℕ → ℝ)
  (h_recurrence : RecurrenceSequence a)
  (h_increasing : StrictlyIncreasing a) :
  a 1 < 1 :=
sorry

end first_term_range_l4087_408787


namespace same_club_probability_l4087_408712

theorem same_club_probability :
  let num_students : ℕ := 2
  let num_clubs : ℕ := 3
  let total_outcomes : ℕ := num_clubs ^ num_students
  let favorable_outcomes : ℕ := num_clubs
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by
  sorry

end same_club_probability_l4087_408712


namespace angle_equality_in_triangle_l4087_408775

/-- Given an acute triangle ABC with its circumcircle, tangents at A and B intersecting at D,
    and M as the midpoint of AB, prove that ∠ACM = ∠BCD. -/
theorem angle_equality_in_triangle (A B C D M : ℂ) : 
  -- A, B, C are on the unit circle (representing the circumcircle)
  Complex.abs A = 1 ∧ Complex.abs B = 1 ∧ Complex.abs C = 1 →
  -- Triangle ABC is acute
  (0 < Real.cos (Complex.arg (B - A) - Complex.arg (C - A))) ∧
  (0 < Real.cos (Complex.arg (C - B) - Complex.arg (A - B))) ∧
  (0 < Real.cos (Complex.arg (A - C) - Complex.arg (B - C))) →
  -- D is the intersection of tangents at A and B
  D = (2 * A * B) / (A + B) →
  -- M is the midpoint of AB
  M = (A + B) / 2 →
  -- Conclusion: ∠ACM = ∠BCD
  Complex.arg ((M - C) / (A - C)) = Complex.arg ((B - C) / (D - C)) := by
  sorry

end angle_equality_in_triangle_l4087_408775


namespace helen_amy_height_difference_l4087_408758

/-- Given the heights of Angela, Amy, and the height difference between Angela and Helen,
    prove that Helen is 3 cm taller than Amy. -/
theorem helen_amy_height_difference
  (angela_height : ℕ)
  (amy_height : ℕ)
  (angela_helen_diff : ℕ)
  (h1 : angela_height = 157)
  (h2 : amy_height = 150)
  (h3 : angela_height = angela_helen_diff + helen_height)
  (helen_height : ℕ) :
  helen_height - amy_height = 3 :=
sorry

end helen_amy_height_difference_l4087_408758


namespace norris_balance_proof_l4087_408717

/-- Calculates the total savings with interest for Norris --/
def total_savings_with_interest (savings : List ℚ) (interest_rate : ℚ) : ℚ :=
  let base_savings := savings.sum
  let interest := 
    savings.take 4 -- Exclude January's savings from interest calculation
      |> List.scanl (λ acc x => acc + x) 0
      |> List.tail!
      |> List.map (λ x => x * interest_rate)
      |> List.sum
  base_savings + interest

/-- Calculates Norris's final balance --/
def norris_final_balance (savings : List ℚ) (interest_rate : ℚ) (loan_amount : ℚ) (repayment : ℚ) : ℚ :=
  total_savings_with_interest savings interest_rate + (loan_amount - repayment)

theorem norris_balance_proof (savings : List ℚ) (interest_rate : ℚ) (loan_amount : ℚ) (repayment : ℚ) :
  savings = [29, 25, 31, 35, 40] ∧ 
  interest_rate = 2 / 100 ∧
  loan_amount = 20 ∧
  repayment = 10 →
  norris_final_balance savings interest_rate loan_amount repayment = 175.76 := by
  sorry

end norris_balance_proof_l4087_408717


namespace largest_four_digit_congruent_to_17_mod_26_l4087_408792

theorem largest_four_digit_congruent_to_17_mod_26 : ∃ (n : ℕ), 
  (n ≤ 9999) ∧ 
  (n ≥ 1000) ∧
  (n % 26 = 17) ∧
  (∀ m : ℕ, (m ≤ 9999) → (m ≥ 1000) → (m % 26 = 17) → m ≤ n) ∧
  (n = 9978) := by
sorry

end largest_four_digit_congruent_to_17_mod_26_l4087_408792


namespace red_card_events_mutually_exclusive_not_opposite_l4087_408783

-- Define the set of cards
inductive Card : Type
  | Black : Card
  | Red : Card
  | White : Card

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the events
def EventAGetsRed (d : Distribution) : Prop := d Person.A = Card.Red
def EventBGetsRed (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem statement
theorem red_card_events_mutually_exclusive_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventAGetsRed d ∧ EventBGetsRed d)) ∧
  -- The events are not opposite (i.e., it's possible for neither to occur)
  (∃ d : Distribution, ¬EventAGetsRed d ∧ ¬EventBGetsRed d) :=
sorry

end red_card_events_mutually_exclusive_not_opposite_l4087_408783


namespace units_digit_of_power_l4087_408716

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The base number -/
def base : ℕ := 5689

/-- The exponent -/
def exponent : ℕ := 439

theorem units_digit_of_power : units_digit (base ^ exponent) = 9 := by
  sorry

end units_digit_of_power_l4087_408716


namespace total_cans_count_l4087_408766

-- Define the given conditions
def total_oil : ℕ := 290
def small_cans : ℕ := 10
def small_can_volume : ℕ := 8
def large_can_volume : ℕ := 15

-- State the theorem
theorem total_cans_count : 
  ∃ (large_cans : ℕ), 
    small_cans * small_can_volume + large_cans * large_can_volume = total_oil ∧
    small_cans + large_cans = 24 := by
  sorry

end total_cans_count_l4087_408766


namespace tractor_count_tractor_count_proof_l4087_408789

theorem tractor_count : ℝ → Prop :=
  fun T : ℝ =>
    let field_work : ℝ := T * 12
    let second_scenario_work : ℝ := 15 * 6.4
    (field_work = second_scenario_work) → T = 8

-- Proof
theorem tractor_count_proof : tractor_count 8 := by
  sorry

end tractor_count_tractor_count_proof_l4087_408789


namespace simplify_expression_simplify_and_evaluate_l4087_408731

-- Problem 1
theorem simplify_expression (a b : ℝ) : a + 2*b + 3*a - 2*b = 4*a := by sorry

-- Problem 2
theorem simplify_and_evaluate : (2*(2^2) - 3*2*1 + 8) - (5*2*1 - 4*(2^2) + 8) = 8 := by sorry

end simplify_expression_simplify_and_evaluate_l4087_408731


namespace bags_filled_on_sunday_l4087_408732

/-- Given the total number of cans collected, cans per bag, and bags filled on Saturday,
    calculate the number of bags filled on Sunday. -/
theorem bags_filled_on_sunday
  (total_cans : ℕ)
  (cans_per_bag : ℕ)
  (bags_on_saturday : ℕ)
  (h1 : total_cans = 63)
  (h2 : cans_per_bag = 9)
  (h3 : bags_on_saturday = 3) :
  total_cans / cans_per_bag - bags_on_saturday = 4 := by
  sorry

end bags_filled_on_sunday_l4087_408732


namespace complement_intersection_A_B_union_complement_B_A_l4087_408796

-- Define the sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- State the theorems
theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {x : ℝ | x ≥ 6 ∨ x < 3} := by sorry

theorem union_complement_B_A : 
  Bᶜ ∪ A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} := by sorry

end complement_intersection_A_B_union_complement_B_A_l4087_408796


namespace isaac_ribbon_length_l4087_408761

theorem isaac_ribbon_length :
  ∀ (total_parts : ℕ) (used_parts : ℕ) (unused_length : ℝ),
    total_parts = 6 →
    used_parts = 4 →
    unused_length = 10 →
    (unused_length / (total_parts - used_parts : ℝ)) * total_parts = 30 :=
by
  sorry

end isaac_ribbon_length_l4087_408761


namespace congruent_count_l4087_408727

theorem congruent_count : ∃ (n : ℕ), n = (Finset.filter (fun x => x > 0 ∧ x < 500 ∧ x % 9 = 4) (Finset.range 500)).card ∧ n = 56 := by
  sorry

end congruent_count_l4087_408727


namespace donuts_for_class_l4087_408700

theorem donuts_for_class (total_students : ℕ) (donut_likers_percentage : ℚ) (donuts_per_student : ℕ) : 
  total_students = 30 →
  donut_likers_percentage = 4/5 →
  donuts_per_student = 2 →
  (↑total_students * donut_likers_percentage * ↑donuts_per_student) / 12 = 4 := by
  sorry

end donuts_for_class_l4087_408700


namespace integer_polynomial_roots_l4087_408798

/-- A polynomial with integer coefficients of the form x^4 + a₃x³ + a₂x² + a₁x - 27 = 0 -/
def IntegerPolynomial (a₃ a₂ a₁ : ℤ) (x : ℤ) : ℤ :=
  x^4 + a₃*x^3 + a₂*x^2 + a₁*x - 27

/-- The set of possible integer roots of the polynomial -/
def PossibleRoots : Set ℤ :=
  {-27, -9, -3, -1, 1, 3, 9, 27}

theorem integer_polynomial_roots (a₃ a₂ a₁ : ℤ) :
  ∀ x : ℤ, (IntegerPolynomial a₃ a₂ a₁ x = 0) ↔ x ∈ PossibleRoots :=
sorry

end integer_polynomial_roots_l4087_408798


namespace at_least_one_vertex_inside_or_on_boundary_l4087_408786

structure CentrallySymmetricPolygon where
  vertices : Set (ℝ × ℝ)
  is_centrally_symmetric : ∃ (center : ℝ × ℝ), ∀ v ∈ vertices, 
    ∃ v' ∈ vertices, v' = (2 * center.1 - v.1, 2 * center.2 - v.2)

structure Polygon where
  vertices : Set (ℝ × ℝ)

def contained_in (T : Polygon) (M : CentrallySymmetricPolygon) : Prop :=
  ∀ v ∈ T.vertices, v ∈ M.vertices

def symmetric_image (T : Polygon) (P : ℝ × ℝ) : Polygon :=
  { vertices := {v' | ∃ v ∈ T.vertices, v' = (2 * P.1 - v.1, 2 * P.2 - v.2)} }

def vertex_inside_or_on_boundary (v : ℝ × ℝ) (M : CentrallySymmetricPolygon) : Prop :=
  v ∈ M.vertices

theorem at_least_one_vertex_inside_or_on_boundary 
  (M : CentrallySymmetricPolygon) (T : Polygon) (P : ℝ × ℝ) :
  contained_in T M →
  P ∈ {p | ∃ v ∈ T.vertices, p = v} →
  ∃ v ∈ (symmetric_image T P).vertices, vertex_inside_or_on_boundary v M :=
sorry

end at_least_one_vertex_inside_or_on_boundary_l4087_408786


namespace sherry_banana_bread_l4087_408799

/-- Calculates the number of bananas needed for a given number of loaves -/
def bananas_needed (total_loaves : ℕ) (loaves_per_batch : ℕ) (bananas_per_batch : ℕ) : ℕ :=
  (total_loaves / loaves_per_batch) * bananas_per_batch

theorem sherry_banana_bread (total_loaves : ℕ) (loaves_per_batch : ℕ) (bananas_per_batch : ℕ) 
  (h1 : total_loaves = 99)
  (h2 : loaves_per_batch = 3)
  (h3 : bananas_per_batch = 1) :
  bananas_needed total_loaves loaves_per_batch bananas_per_batch = 33 :=
by
  sorry

end sherry_banana_bread_l4087_408799


namespace height_to_hypotenuse_not_always_half_l4087_408725

theorem height_to_hypotenuse_not_always_half : ∃ (a b c h : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ h > 0 ∧
  a^2 + b^2 = c^2 ∧  -- right triangle condition
  h ≠ c / 2 ∧        -- height is not half of hypotenuse
  h * c = a * b      -- height formula
  := by sorry

end height_to_hypotenuse_not_always_half_l4087_408725


namespace geometric_sequence_sum_l4087_408704

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 1365/4096 := by
sorry

end geometric_sequence_sum_l4087_408704


namespace rectangle_to_square_area_ratio_l4087_408748

theorem rectangle_to_square_area_ratio :
  let large_square_side : ℝ := 50
  let grid_size : ℕ := 5
  let rectangle_rows : ℕ := 2
  let rectangle_cols : ℕ := 3
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_side : ℝ := large_square_side / grid_size
  let rectangle_area : ℝ := (rectangle_rows * small_square_side) * (rectangle_cols * small_square_side)
  rectangle_area / large_square_area = 6 / 25 := by
sorry

end rectangle_to_square_area_ratio_l4087_408748


namespace cake_muffin_mix_probability_l4087_408734

theorem cake_muffin_mix_probability (total_buyers : ℕ) (cake_buyers : ℕ) (muffin_buyers : ℕ) (both_buyers : ℕ)
  (h1 : total_buyers = 100)
  (h2 : cake_buyers = 50)
  (h3 : muffin_buyers = 40)
  (h4 : both_buyers = 19) :
  (total_buyers - (cake_buyers + muffin_buyers - both_buyers)) / total_buyers = 29 / 100 := by
  sorry

end cake_muffin_mix_probability_l4087_408734


namespace circle_max_min_value_l4087_408749

theorem circle_max_min_value (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S_max S_min : ℝ),
    (∀ S, S = 3*x - y → S ≤ S_max ∧ S ≥ S_min) ∧
    S_max = 5 + 2 * Real.sqrt 10 ∧
    S_min = 5 - 2 * Real.sqrt 10 :=
by sorry

end circle_max_min_value_l4087_408749


namespace sum_first_three_eq_18_l4087_408763

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ  -- First term
  d : ℕ  -- Common difference
  fifth_term_eq_15 : a + 4 * d = 15
  d_eq_3 : d = 3

/-- The sum of the first three terms of the arithmetic sequence -/
def sum_first_three (seq : ArithmeticSequence) : ℕ :=
  seq.a + (seq.a + seq.d) + (seq.a + 2 * seq.d)

/-- Theorem stating that the sum of the first three terms is 18 -/
theorem sum_first_three_eq_18 (seq : ArithmeticSequence) :
  sum_first_three seq = 18 := by
  sorry

#eval sum_first_three ⟨3, 3, rfl, rfl⟩

end sum_first_three_eq_18_l4087_408763


namespace square_sum_from_difference_and_product_l4087_408735

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 17) 
  (h2 : x * y = 6) : 
  x^2 + y^2 = 301 := by
sorry

end square_sum_from_difference_and_product_l4087_408735


namespace inclination_angle_of_negative_unit_slope_l4087_408781

theorem inclination_angle_of_negative_unit_slope (α : Real) : 
  (Real.tan α = -1) → (0 ≤ α) → (α < Real.pi) → (α = 3 * Real.pi / 4) := by
  sorry

end inclination_angle_of_negative_unit_slope_l4087_408781


namespace absolute_value_subtraction_l4087_408753

theorem absolute_value_subtraction : 4 - |(-3)| = 1 := by
  sorry

end absolute_value_subtraction_l4087_408753


namespace arithmetic_seq_first_term_arithmetic_seq_first_term_range_l4087_408709

/-- Arithmetic sequence with common difference -1 -/
def ArithmeticSeq (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => ArithmeticSeq a₁ n - 1

/-- Sum of first n terms of the arithmetic sequence -/
def SumSeq (a₁ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => SumSeq a₁ n + ArithmeticSeq a₁ (n + 1)

theorem arithmetic_seq_first_term (a₁ : ℝ) :
  SumSeq a₁ 5 = -5 → a₁ = 1 := by sorry

theorem arithmetic_seq_first_term_range (a₁ : ℝ) :
  (∀ n : ℕ, n > 0 → SumSeq a₁ n ≤ ArithmeticSeq a₁ n) → a₁ ≤ 0 := by sorry

end arithmetic_seq_first_term_arithmetic_seq_first_term_range_l4087_408709


namespace fraction_addition_l4087_408774

theorem fraction_addition : (11 : ℚ) / 12 + 7 / 15 = 83 / 60 := by
  sorry

end fraction_addition_l4087_408774


namespace union_of_A_and_B_complement_A_intersect_B_B_subset_A_iff_m_range_l4087_408738

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

-- Part 1
theorem union_of_A_and_B : A ∪ B 4 = {x | -2 ≤ x ∧ x < 5} := by sorry

theorem complement_A_intersect_B : (Set.univ \ A) ∩ B 4 = {x | 4 < x ∧ x < 5} := by sorry

-- Part 2
theorem B_subset_A_iff_m_range : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≤ 3 := by sorry

end union_of_A_and_B_complement_A_intersect_B_B_subset_A_iff_m_range_l4087_408738


namespace elizabeth_borrowed_53_cents_l4087_408730

/-- The amount Elizabeth borrowed from her neighbor -/
def amount_borrowed : ℕ := by sorry

theorem elizabeth_borrowed_53_cents :
  let pencil_cost : ℕ := 600  -- in cents
  let elizabeth_has : ℕ := 500  -- in cents
  let needs_more : ℕ := 47  -- in cents
  amount_borrowed = pencil_cost - elizabeth_has - needs_more :=
by sorry

end elizabeth_borrowed_53_cents_l4087_408730


namespace geometric_sequence_sum_l4087_408703

/-- Given a geometric sequence {a_n} where all terms are positive,
    if a_3 * a_5 + a_2 * a_10 + 2 * a_4 * a_6 = 100,
    then a_4 + a_6 = 10 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geo : ∀ n m : ℕ, a (n + m) = a n * (a 2) ^ (m - 1))
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_sum : a 3 * a 5 + a 2 * a 10 + 2 * a 4 * a 6 = 100) :
  a 4 + a 6 = 10 := by
  sorry

end geometric_sequence_sum_l4087_408703


namespace parallel_vectors_acute_angle_l4087_408746

/-- Given two vectors a and b that are parallel and α is an acute angle, prove that α = 45° -/
theorem parallel_vectors_acute_angle (α : Real) 
  (h_acute : 0 < α ∧ α < Real.pi / 2)
  (a : Fin 2 → Real)
  (b : Fin 2 → Real)
  (h_a : a = ![3/2, 1 + Real.sin α])
  (h_b : b = ![1 - Real.cos α, 1/3])
  (h_parallel : ∃ (k : Real), a = k • b) :
  α = Real.pi / 4 := by
sorry

end parallel_vectors_acute_angle_l4087_408746


namespace pizza_topping_distribution_l4087_408723

/-- Pizza topping distribution problem -/
theorem pizza_topping_distribution 
  (pepperoni : ℕ) 
  (ham : ℕ) 
  (sausage : ℕ) 
  (slices : ℕ) :
  pepperoni = 30 →
  ham = 2 * pepperoni →
  sausage = pepperoni + 12 →
  slices = 6 →
  (pepperoni + ham + sausage) / slices = 22 := by
  sorry

#check pizza_topping_distribution

end pizza_topping_distribution_l4087_408723


namespace number_multiplied_by_six_l4087_408705

theorem number_multiplied_by_six (n : ℚ) : n / 11 = 2 → n * 6 = 132 := by
  sorry

end number_multiplied_by_six_l4087_408705


namespace fraction_inequality_l4087_408765

theorem fraction_inequality (x y : ℝ) (h : x > y) : x / 5 > y / 5 := by
  sorry

end fraction_inequality_l4087_408765


namespace zeroth_power_of_nonzero_rational_l4087_408722

theorem zeroth_power_of_nonzero_rational (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zeroth_power_of_nonzero_rational_l4087_408722


namespace largest_n_divisible_by_seven_ninety_nine_thousand_nine_hundred_ninety_nine_is_largest_l4087_408785

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 100000 →
  (9 * (n - 3)^5 - 2 * n^3 + 17 * n - 33) % 7 = 0 →
  n ≤ 99999 :=
by sorry

theorem ninety_nine_thousand_nine_hundred_ninety_nine_is_largest :
  (9 * (99999 - 3)^5 - 2 * 99999^3 + 17 * 99999 - 33) % 7 = 0 ∧
  ∀ m : ℕ, m > 99999 → m < 100000 → (9 * (m - 3)^5 - 2 * m^3 + 17 * m - 33) % 7 ≠ 0 :=
by sorry

end largest_n_divisible_by_seven_ninety_nine_thousand_nine_hundred_ninety_nine_is_largest_l4087_408785


namespace hyperbola_asymptotes_l4087_408771

/-- Represents a hyperbola with equation x²/9 - y²/m = 1 -/
structure Hyperbola where
  m : ℝ

/-- Represents a line with equation x + y = 5 -/
def focus_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 5}

/-- Represents the asymptotes of a hyperbola -/
structure Asymptotes where
  slope : ℝ

theorem hyperbola_asymptotes (h : Hyperbola) (focus_on_line : ∃ p : ℝ × ℝ, p ∈ focus_line ∧ p.2 = 0) :
  Asymptotes.mk (4/3) = Asymptotes.mk (-4/3) :=
sorry

end hyperbola_asymptotes_l4087_408771


namespace circle_radius_is_five_l4087_408718

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x + y = 0

-- Define the circle passing through two points
def circle_passes_through (center_x center_y : ℝ) (point1_x point1_y point2_x point2_y : ℝ) : Prop :=
  (center_x - point1_x)^2 + (center_y - point1_y)^2 = (center_x - point2_x)^2 + (center_y - point2_y)^2

-- Theorem statement
theorem circle_radius_is_five :
  ∃ (center_x center_y : ℝ),
    line_equation center_x center_y ∧
    circle_passes_through center_x center_y 1 3 4 2 ∧
    ((center_x - 1)^2 + (center_y - 3)^2)^(1/2 : ℝ) = 5 :=
by
  sorry

end circle_radius_is_five_l4087_408718


namespace quadratic_inequality_l4087_408742

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |f a b c x| ≤ 1) →
  (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |2 * a * x + b| ≤ 4) := by
  sorry

end quadratic_inequality_l4087_408742


namespace fifth_equation_is_correct_l4087_408719

-- Define the sequence of equations
def equation (n : ℕ) : Prop :=
  match n with
  | 1 => 2^1 * 1 = 2
  | 2 => 2^2 * 1 * 3 = 3 * 4
  | 3 => 2^3 * 1 * 3 * 5 = 4 * 5 * 6
  | 5 => 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10
  | _ => True

-- Theorem statement
theorem fifth_equation_is_correct :
  equation 1 ∧ equation 2 ∧ equation 3 → equation 5 := by
  sorry

end fifth_equation_is_correct_l4087_408719


namespace yellow_balls_count_l4087_408726

theorem yellow_balls_count (Y : ℕ) : 
  (Y : ℝ) / (Y + 2) * ((Y - 1) / (Y + 1)) = 1 / 2 → Y = 5 := by
  sorry

end yellow_balls_count_l4087_408726


namespace complement_union_and_intersection_l4087_408728

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x < 9}

-- Define the complement of a set in ℝ
def complement (S : Set ℝ) : Set ℝ := {x : ℝ | x ∉ S}

-- State the theorem
theorem complement_union_and_intersection :
  (complement (A ∪ B) = {x : ℝ | x ≤ 2 ∨ x ≥ 9}) ∧
  (complement (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 5}) := by
  sorry

end complement_union_and_intersection_l4087_408728


namespace part_one_part_two_l4087_408776

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Part 1
theorem part_one : 
  (A (-1) ∪ B = {x | x < 2 ∨ x > 5}) ∧ 
  ((Set.univ \ A (-1)) ∩ B = {x | x < -2 ∨ x > 5}) := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, (A a ∩ B = ∅) ↔ (a ≥ 3 ∨ (-1/2 ≤ a ∧ a ≤ 2)) := by sorry

end part_one_part_two_l4087_408776


namespace arithmetic_operations_l4087_408755

theorem arithmetic_operations : 
  ((-9) + ((-4) * 5) = -29) ∧ 
  ((6 * (-2)) / (2/3) = -18) := by
sorry

end arithmetic_operations_l4087_408755


namespace paintings_distribution_l4087_408733

/-- Given a total number of paintings, number of rooms, and paintings kept in a private study,
    calculate the number of paintings placed in each room. -/
def paintings_per_room (total : ℕ) (rooms : ℕ) (kept : ℕ) : ℕ :=
  (total - kept) / rooms

/-- Theorem stating that given 47 total paintings, 6 rooms, and 5 paintings kept in a private study,
    the number of paintings placed in each room is 7. -/
theorem paintings_distribution :
  paintings_per_room 47 6 5 = 7 := by
  sorry

end paintings_distribution_l4087_408733


namespace pushup_percentage_l4087_408715

def jumping_jacks : ℕ := 12
def pushups : ℕ := 8
def situps : ℕ := 20

def total_exercises : ℕ := jumping_jacks + pushups + situps

theorem pushup_percentage :
  (pushups : ℚ) / (total_exercises : ℚ) * 100 = 20 := by
  sorry

end pushup_percentage_l4087_408715


namespace triangle_isosceles_or_right_l4087_408757

theorem triangle_isosceles_or_right 
  (A B C : ℝ) 
  (triangle_angles : A + B + C = π) 
  (angle_condition : Real.sin (A + B - C) = Real.sin (A - B + C)) : 
  (B = C) ∨ (B + C = π / 2) :=
sorry

end triangle_isosceles_or_right_l4087_408757


namespace stratified_sample_size_l4087_408768

/-- Represents the total number of schools of each type -/
structure SchoolCounts where
  universities : ℕ
  middleSchools : ℕ
  primarySchools : ℕ

/-- Calculates the total number of schools -/
def totalSchools (counts : SchoolCounts) : ℕ :=
  counts.universities + counts.middleSchools + counts.primarySchools

/-- Represents the sample size for middle schools -/
def middleSchoolSample : ℕ := 10

/-- Theorem: In a stratified sampling of schools, if 10 middle schools are sampled
    from a population with 20 universities, 200 middle schools, and 480 primary schools,
    then the total sample size is 35. -/
theorem stratified_sample_size 
  (counts : SchoolCounts) 
  (h1 : counts.universities = 20) 
  (h2 : counts.middleSchools = 200) 
  (h3 : counts.primarySchools = 480) :
  (middleSchoolSample : ℚ) / counts.middleSchools = 
  (35 : ℚ) / (totalSchools counts) := by
  sorry

end stratified_sample_size_l4087_408768


namespace correct_calculation_l4087_408702

theorem correct_calculation (x : ℚ) (h : x / 6 = 12) : x * 7 = 504 := by
  sorry

end correct_calculation_l4087_408702


namespace parallel_planes_condition_l4087_408745

structure GeometricSpace where
  Line : Type
  Plane : Type
  subset : Line → Plane → Prop
  parallel : Line → Plane → Prop
  plane_parallel : Plane → Plane → Prop

variable (S : GeometricSpace)

theorem parallel_planes_condition
  (a b : S.Line) (α β : S.Plane)
  (h1 : S.subset a α)
  (h2 : S.subset b β) :
  (∃ (α' β' : S.Plane), S.plane_parallel α' β' →
    (S.parallel a β' ∧ S.parallel b α')) ∧
  ¬(∀ (α' β' : S.Plane), S.parallel a β' ∧ S.parallel b α' →
    S.plane_parallel α' β') := by
  sorry

end parallel_planes_condition_l4087_408745


namespace remaining_funds_is_38817_l4087_408751

/-- Represents the family's financial situation and tax obligations -/
structure FamilyFinances where
  father_income : ℕ
  mother_income : ℕ
  grandmother_pension : ℕ
  mikhail_scholarship : ℕ
  tax_deduction_per_child : ℕ
  num_children : ℕ
  income_tax_rate : ℚ
  monthly_savings : ℕ
  monthly_household_expenses : ℕ
  apartment_area : ℕ
  apartment_cadastral_value : ℕ
  car1_horsepower : ℕ
  car1_months_owned : ℕ
  car2_horsepower : ℕ
  car2_months_registered : ℕ
  land_area : ℕ
  land_cadastral_value : ℕ
  tour_cost_per_person : ℕ
  num_people_for_tour : ℕ

/-- Calculates the remaining funds for additional expenses -/
def calculate_remaining_funds (f : FamilyFinances) : ℕ :=
  sorry

/-- Theorem stating that the remaining funds for additional expenses is 38817 rubles -/
theorem remaining_funds_is_38817 (f : FamilyFinances) 
  (h1 : f.father_income = 50000)
  (h2 : f.mother_income = 28000)
  (h3 : f.grandmother_pension = 15000)
  (h4 : f.mikhail_scholarship = 3000)
  (h5 : f.tax_deduction_per_child = 1400)
  (h6 : f.num_children = 2)
  (h7 : f.income_tax_rate = 13 / 100)
  (h8 : f.monthly_savings = 10000)
  (h9 : f.monthly_household_expenses = 65000)
  (h10 : f.apartment_area = 78)
  (h11 : f.apartment_cadastral_value = 6240000)
  (h12 : f.car1_horsepower = 106)
  (h13 : f.car1_months_owned = 3)
  (h14 : f.car2_horsepower = 122)
  (h15 : f.car2_months_registered = 8)
  (h16 : f.land_area = 10)
  (h17 : f.land_cadastral_value = 420300)
  (h18 : f.tour_cost_per_person = 17900)
  (h19 : f.num_people_for_tour = 5) :
  calculate_remaining_funds f = 38817 :=
by sorry

end remaining_funds_is_38817_l4087_408751


namespace bicycle_price_theorem_l4087_408794

/-- The price C pays for a bicycle, given the initial cost and two successive profit margins -/
def final_price (initial_cost : ℝ) (profit1 : ℝ) (profit2 : ℝ) : ℝ :=
  let price1 := initial_cost * (1 + profit1)
  price1 * (1 + profit2)

/-- Theorem stating that the final price of the bicycle is 225 -/
theorem bicycle_price_theorem :
  final_price 150 0.20 0.25 = 225 := by
  sorry

end bicycle_price_theorem_l4087_408794


namespace wendys_bake_sale_l4087_408769

/-- Wendy's bake sale problem -/
theorem wendys_bake_sale
  (cupcakes : ℕ)
  (cookies : ℕ)
  (leftover : ℕ)
  (h1 : cupcakes = 4)
  (h2 : cookies = 29)
  (h3 : leftover = 24) :
  cupcakes + cookies - leftover = 9 :=
by sorry

end wendys_bake_sale_l4087_408769


namespace area_in_three_triangles_l4087_408724

/-- Given a 6 by 8 rectangle with equilateral triangles on each side, 
    this function calculates the area of regions in exactly 3 of 4 triangles -/
def areaInThreeTriangles : ℝ := sorry

/-- The rectangle's width -/
def rectangleWidth : ℝ := 6

/-- The rectangle's length -/
def rectangleLength : ℝ := 8

/-- Theorem stating the area calculation -/
theorem area_in_three_triangles :
  areaInThreeTriangles = (288 - 154 * Real.sqrt 3) / 3 := by sorry

end area_in_three_triangles_l4087_408724


namespace equation_solution_l4087_408770

theorem equation_solution (t : ℝ) : 
  (Real.sqrt (3 * Real.sqrt (3 * t - 6)) = (8 - t) ^ (1/4)) ↔ 
  (t = (-43 + Real.sqrt 2321) / 2 ∨ t = (-43 - Real.sqrt 2321) / 2) :=
sorry

end equation_solution_l4087_408770


namespace variance_scaling_l4087_408741

variable {n : ℕ}
variable (a : Fin n → ℝ)

/-- The variance of a dataset -/
def variance (x : Fin n → ℝ) : ℝ := sorry

/-- The scaled dataset where each element is multiplied by 2 -/
def scaled_data (x : Fin n → ℝ) : Fin n → ℝ := λ i => 2 * x i

theorem variance_scaling (h : variance a = 4) : 
  variance (scaled_data a) = 16 := by sorry

end variance_scaling_l4087_408741


namespace largest_common_term_l4087_408780

def is_in_ap1 (a : ℕ) : Prop := ∃ k : ℕ, a = 4 + 5 * k

def is_in_ap2 (a : ℕ) : Prop := ∃ k : ℕ, a = 7 + 11 * k

def is_common_term (a : ℕ) : Prop := is_in_ap1 a ∧ is_in_ap2 a

theorem largest_common_term :
  ∃ a : ℕ, a = 984 ∧ is_common_term a ∧ a < 1000 ∧
  ∀ b : ℕ, is_common_term b ∧ b < 1000 → b ≤ a :=
sorry

end largest_common_term_l4087_408780


namespace selection_theorem_l4087_408706

/-- Represents the number of students who can play only chess -/
def chess_only : ℕ := 2

/-- Represents the number of students who can play only Go -/
def go_only : ℕ := 3

/-- Represents the number of students who can play both chess and Go -/
def both : ℕ := 4

/-- Represents the total number of students -/
def total_students : ℕ := chess_only + go_only + both

/-- Calculates the number of ways to select two students for chess and Go competitions -/
def selection_ways : ℕ :=
  chess_only * go_only +  -- One from chess_only, one from go_only
  both * go_only +        -- One from both for chess, one from go_only
  chess_only * both +     -- One from chess_only, one from both for Go
  (both * (both - 1)) / 2 -- Two from both (combination)

/-- Theorem stating that the number of ways to select students is 32 -/
theorem selection_theorem : selection_ways = 32 := by sorry

end selection_theorem_l4087_408706


namespace average_weight_of_removed_carrots_l4087_408784

/-- The average weight of 4 removed carrots given the following conditions:
    - There are initially 20 carrots, 10 apples, and 5 oranges
    - The total initial weight is 8.70 kg
    - After removal, there are 16 carrots and 8 apples
    - The average weight after removal is 206 grams
    - The average weight of an apple is 210 grams -/
theorem average_weight_of_removed_carrots :
  ∀ (total_weight : ℝ) 
    (initial_carrots initial_apples initial_oranges : ℕ)
    (remaining_carrots remaining_apples : ℕ)
    (avg_weight_after_removal avg_weight_apple : ℝ),
  total_weight = 8.70 ∧
  initial_carrots = 20 ∧
  initial_apples = 10 ∧
  initial_oranges = 5 ∧
  remaining_carrots = 16 ∧
  remaining_apples = 8 ∧
  avg_weight_after_removal = 206 ∧
  avg_weight_apple = 210 →
  (total_weight * 1000 - 
   (remaining_carrots + remaining_apples) * avg_weight_after_removal - 
   (initial_apples - remaining_apples) * avg_weight_apple) / 
   (initial_carrots - remaining_carrots) = 834 :=
by sorry

end average_weight_of_removed_carrots_l4087_408784


namespace range_of_a_l4087_408708

def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a : 
  (∃ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∃ a : ℝ, (a < 0 ∨ (1/4 < a ∧ a < 4)) ∧ 
            ∀ b : ℝ, (0 ≤ b ∧ b ≤ 1/4) → b ≠ a) :=
sorry

end range_of_a_l4087_408708


namespace smallest_y_value_l4087_408791

theorem smallest_y_value : ∃ y : ℝ, 
  (∀ z : ℝ, 3 * z^2 + 27 * z - 90 = z * (z + 15) → y ≤ z) ∧ 
  (3 * y^2 + 27 * y - 90 = y * (y + 15)) ∧ 
  y = -9 := by
  sorry

end smallest_y_value_l4087_408791


namespace inequality_solution_l4087_408790

theorem inequality_solution (x : ℝ) :
  x ≠ 1 ∧ x ≠ 2 →
  ((x^3 - 3*x^2 + 2*x) / (x^2 - 3*x + 2) ≤ 1 ↔ x < 1) :=
by sorry

end inequality_solution_l4087_408790


namespace total_pictures_l4087_408760

theorem total_pictures (randy peter quincy : ℕ) : 
  randy = 5 → 
  peter = randy + 3 → 
  quincy = peter + 20 → 
  randy + peter + quincy = 41 := by sorry

end total_pictures_l4087_408760


namespace return_speed_calculation_l4087_408788

/-- Given a round trip with the following conditions:
    - Total distance is 20 km (10 km each way)
    - Return speed is twice the outbound speed
    - Total travel time is 6 hours
    Prove that the return speed is 5 km/h -/
theorem return_speed_calculation (distance : ℝ) (total_time : ℝ) : 
  distance = 10 →
  total_time = 6 →
  ∃ (outbound_speed : ℝ),
    outbound_speed > 0 ∧
    distance / outbound_speed + distance / (2 * outbound_speed) = total_time →
    2 * outbound_speed = 5 := by
  sorry

#check return_speed_calculation

end return_speed_calculation_l4087_408788


namespace recurrence_relation_l4087_408782

def a : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 5
  | (n + 3) => (a (n + 2) * a (n + 1) - 2) / a n

def b (n : ℕ) : ℚ := a (2 * n)

theorem recurrence_relation (n : ℕ) :
  b (n + 2) - 4 * b (n + 1) + b n = 0 := by
  sorry

end recurrence_relation_l4087_408782


namespace compare_abc_l4087_408772

def tower_exp (base : ℕ) : ℕ → ℕ
| 0 => 1
| (n + 1) => base ^ (tower_exp base n)

def a : ℕ := tower_exp 3 25
def b : ℕ := tower_exp 4 20
def c : ℕ := 5^5

theorem compare_abc : c < a ∧ a < b := by
  sorry

end compare_abc_l4087_408772


namespace ellipse_focal_length_l4087_408740

/-- The focal length of an ellipse with equation x^2/25 + y^2/16 = 1 is 6 -/
theorem ellipse_focal_length : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 - b^2)
  2 * c = 6 := by sorry

end ellipse_focal_length_l4087_408740


namespace quadratic_roots_l4087_408797

theorem quadratic_roots (a b c : ℝ) (h : ∃ x₁ x₂ : ℝ, a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁^3 - x₂^3 = 2011) :
  ∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ a * y₁^2 + 2 * b * y₁ + 4 * c = 0 ∧ a * y₂^2 + 2 * b * y₂ + 4 * c = 0 :=
sorry

end quadratic_roots_l4087_408797


namespace divide_meter_into_hundred_parts_l4087_408743

theorem divide_meter_into_hundred_parts : 
  ∀ (total_length : ℝ) (num_parts : ℕ),
    total_length = 1 →
    num_parts = 100 →
    (total_length / num_parts : ℝ) = 1 / 100 := by
  sorry

end divide_meter_into_hundred_parts_l4087_408743


namespace electric_guitars_sold_l4087_408779

theorem electric_guitars_sold (total_guitars : ℕ) (total_revenue : ℕ) 
  (electric_price : ℕ) (acoustic_price : ℕ) :
  total_guitars = 9 →
  total_revenue = 3611 →
  electric_price = 479 →
  acoustic_price = 339 →
  ∃ (electric_sold : ℕ) (acoustic_sold : ℕ),
    electric_sold + acoustic_sold = total_guitars ∧
    electric_sold * electric_price + acoustic_sold * acoustic_price = total_revenue ∧
    electric_sold = 4 :=
by sorry

end electric_guitars_sold_l4087_408779


namespace spinner_probability_D_l4087_408754

-- Define the spinner with four regions
structure Spinner :=
  (A B C D : ℝ)

-- Define the properties of the spinner
def valid_spinner (s : Spinner) : Prop :=
  s.A = 1/4 ∧ s.B = 1/3 ∧ s.A + s.B + s.C + s.D = 1

-- Theorem statement
theorem spinner_probability_D (s : Spinner) 
  (h : valid_spinner s) : s.D = 1/4 := by
  sorry

end spinner_probability_D_l4087_408754


namespace complex_equation_imaginary_part_l4087_408778

theorem complex_equation_imaginary_part :
  ∀ (z : ℂ), (3 + 4*I) * z = 5 → z.im = -4/5 := by sorry

end complex_equation_imaginary_part_l4087_408778


namespace application_schemes_five_graduates_three_universities_l4087_408752

/-- The number of possible application schemes for high school graduates choosing universities. -/
def application_schemes (num_graduates : ℕ) (num_universities : ℕ) : ℕ :=
  num_universities ^ num_graduates

/-- Theorem: The number of possible application schemes for 5 high school graduates
    choosing from 3 universities, where each graduate can only fill in one preference,
    is equal to 3^5. -/
theorem application_schemes_five_graduates_three_universities :
  application_schemes 5 3 = 3^5 := by
  sorry

end application_schemes_five_graduates_three_universities_l4087_408752


namespace simplify_and_rationalize_l4087_408750

theorem simplify_and_rationalize : 
  (Real.sqrt 6 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 15) * (Real.sqrt 8 / Real.sqrt 14) = 
  (2 * Real.sqrt 7) / 7 := by sorry

end simplify_and_rationalize_l4087_408750


namespace equal_intercept_line_equation_characterization_l4087_408707

/-- A line passing through (1, 3) with equal x and y intercepts -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1, 3) -/
  point_condition : 3 = m * 1 + b
  /-- The line has equal x and y intercepts -/
  equal_intercepts : b ≠ 0 → -b / m = b

/-- The equation of a line with equal intercepts passing through (1, 3) -/
def equal_intercept_line_equation (l : EqualInterceptLine) : Prop :=
  (l.m = 3 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 4)

/-- Theorem stating that a line with equal intercepts passing through (1, 3) 
    must have the equation 3x - y = 0 or x + y - 4 = 0 -/
theorem equal_intercept_line_equation_characterization (l : EqualInterceptLine) :
  equal_intercept_line_equation l := by sorry

end equal_intercept_line_equation_characterization_l4087_408707


namespace x_gt_2_sufficient_not_necessary_for_x_neq_2_l4087_408729

theorem x_gt_2_sufficient_not_necessary_for_x_neq_2 :
  (∃ x : ℝ, x ≠ 2 ∧ ¬(x > 2)) ∧
  (∀ x : ℝ, x > 2 → x ≠ 2) :=
by sorry

end x_gt_2_sufficient_not_necessary_for_x_neq_2_l4087_408729


namespace stationery_sales_distribution_l4087_408756

theorem stationery_sales_distribution (pen_sales pencil_sales eraser_sales : ℝ) 
  (h_pen : pen_sales = 42)
  (h_pencil : pencil_sales = 25)
  (h_eraser : eraser_sales = 12)
  (h_total : pen_sales + pencil_sales + eraser_sales + (100 - pen_sales - pencil_sales - eraser_sales) = 100) :
  100 - pen_sales - pencil_sales - eraser_sales = 21 := by
sorry

end stationery_sales_distribution_l4087_408756


namespace scientific_notation_1200_l4087_408721

theorem scientific_notation_1200 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1200 = a * (10 : ℝ) ^ n ∧ a = 1.2 ∧ n = 3 := by
  sorry

end scientific_notation_1200_l4087_408721


namespace max_average_growth_rate_l4087_408736

theorem max_average_growth_rate
  (P₁ P₂ M : ℝ)
  (h_sum : P₁ + P₂ = M)
  (h_nonneg : 0 ≤ P₁ ∧ 0 ≤ P₂)
  (P : ℝ)
  (h_avg_growth : (1 + P)^2 = (1 + P₁) * (1 + P₂)) :
  P ≤ M / 2 :=
sorry

end max_average_growth_rate_l4087_408736


namespace candy_distribution_l4087_408764

theorem candy_distribution (total_candy : Nat) (num_friends : Nat) : 
  total_candy = 379 → num_friends = 6 → 
  ∃ (equal_distribution : Nat), 
    equal_distribution ≤ total_candy ∧ 
    equal_distribution.mod num_friends = 0 ∧
    ∀ n : Nat, n ≤ total_candy ∧ n.mod num_friends = 0 → n ≤ equal_distribution := by
  sorry

end candy_distribution_l4087_408764


namespace stratified_sampling_group_a_l4087_408793

/-- Calculates the number of cities to be selected from a group in stratified sampling -/
def stratifiedSampleSize (totalCities : ℕ) (groupSize : ℕ) (sampleSize : ℕ) : ℚ :=
  (groupSize : ℚ) * (sampleSize : ℚ) / (totalCities : ℚ)

/-- Theorem: In a stratified sampling of 6 cities from 24 total cities, 
    where 4 cities belong to group A, 1 city should be selected from group A -/
theorem stratified_sampling_group_a : 
  stratifiedSampleSize 24 4 6 = 1 := by
  sorry

end stratified_sampling_group_a_l4087_408793


namespace cubic_root_relation_l4087_408762

theorem cubic_root_relation (x₀ : ℝ) (z : ℝ) : 
  x₀^3 - x₀ - 1 = 0 →
  z = x₀^2 + 3 * x₀ + 1 →
  z^3 - 5*z^2 - 10*z - 11 = 0 := by
sorry

end cubic_root_relation_l4087_408762


namespace sphere_cylinder_equal_area_l4087_408714

theorem sphere_cylinder_equal_area (r : ℝ) : 
  (4 : ℝ) * Real.pi * r^2 = (2 : ℝ) * Real.pi * 4 * 8 → r = 4 := by
  sorry

end sphere_cylinder_equal_area_l4087_408714


namespace vector_operation_result_l4087_408773

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

variable (O A B C E : E)

theorem vector_operation_result :
  (A - B) - (C - B) + (O - E) - (O - C) = A - E := by sorry

end vector_operation_result_l4087_408773
