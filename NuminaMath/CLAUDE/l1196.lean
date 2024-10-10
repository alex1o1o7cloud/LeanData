import Mathlib

namespace arithmetic_sequence_cosine_ratio_l1196_119616

theorem arithmetic_sequence_cosine_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 9 - a 8) →  -- arithmetic sequence condition
  a 8 = 8 →                            -- given condition
  a 9 = 8 + π / 3 →                    -- given condition
  (Real.cos (a 5) + Real.cos (a 7)) / Real.cos (a 6) = 1 := by
    sorry

end arithmetic_sequence_cosine_ratio_l1196_119616


namespace sixth_term_is_thirteen_l1196_119672

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 7 ∧ 
  a 5 = a 2 + 6

/-- The 6th term of the arithmetic sequence is 13 -/
theorem sixth_term_is_thirteen 
  (a : ℕ → ℝ) 
  (h : ArithmeticSequence a) : 
  a 6 = 13 := by
sorry

end sixth_term_is_thirteen_l1196_119672


namespace abs_x_minus_one_leq_one_iff_x_leq_two_l1196_119643

theorem abs_x_minus_one_leq_one_iff_x_leq_two :
  ∀ x : ℝ, |x - 1| ≤ 1 ↔ x ≤ 2 := by sorry

end abs_x_minus_one_leq_one_iff_x_leq_two_l1196_119643


namespace prism_volume_l1196_119641

-- Define a right rectangular prism
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the volume of a rectangular prism
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

-- Define the face areas of a rectangular prism
def sideFaceArea (p : RectangularPrism) : ℝ := p.a * p.b
def frontFaceArea (p : RectangularPrism) : ℝ := p.b * p.c
def bottomFaceArea (p : RectangularPrism) : ℝ := p.a * p.c

-- Theorem: The volume of the prism is 12 cubic inches
theorem prism_volume (p : RectangularPrism) 
  (h1 : sideFaceArea p = 18) 
  (h2 : frontFaceArea p = 12) 
  (h3 : bottomFaceArea p = 8) : 
  volume p = 12 := by
  sorry


end prism_volume_l1196_119641


namespace secret_eggs_count_l1196_119631

/-- Given a jar with candy and secret eggs, calculate the number of secret eggs. -/
theorem secret_eggs_count (candy : ℝ) (total : ℕ) (h1 : candy = 3409.0) (h2 : total = 3554) :
  ↑total - candy = 145 :=
by sorry

end secret_eggs_count_l1196_119631


namespace intersection_equality_implies_a_values_l1196_119658

def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equality_implies_a_values (a : ℝ) :
  A ∩ B a = B a → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  sorry

end intersection_equality_implies_a_values_l1196_119658


namespace green_peaches_count_l1196_119699

/-- The number of green peaches in a basket, given the total number of peaches and the number of red peaches. -/
def num_green_peaches (total : ℕ) (red : ℕ) : ℕ :=
  total - red

/-- Theorem stating that there are 3 green peaches in the basket. -/
theorem green_peaches_count :
  let total := 16
  let red := 13
  num_green_peaches total red = 3 := by
  sorry

end green_peaches_count_l1196_119699


namespace correct_product_after_reversal_error_l1196_119676

-- Define a function to reverse digits of a two-digit number
def reverseDigits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

-- Define the theorem
theorem correct_product_after_reversal_error (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverseDigits a * b = 221) →  -- erroneous product is 221
  (a * b = 923) :=  -- correct product is 923
by sorry

end correct_product_after_reversal_error_l1196_119676


namespace hexagon_area_theorem_l1196_119647

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon given its vertices A and C -/
def hexagon_area (h : RegularHexagon) : ℝ :=
  sorry

theorem hexagon_area_theorem (h : RegularHexagon) 
  (h_A : h.A = (0, 0)) 
  (h_C : h.C = (8, 2)) : 
  hexagon_area h = 34 * Real.sqrt 3 := by
  sorry

end hexagon_area_theorem_l1196_119647


namespace tan_sum_range_l1196_119635

theorem tan_sum_range (m : ℝ) (α β : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ 
    m * x^2 - 2 * x * Real.sqrt (7 * m - 3) + 2 * m = 0 ∧
    m * y^2 - 2 * y * Real.sqrt (7 * m - 3) + 2 * m = 0 ∧
    x = Real.tan α ∧ y = Real.tan β) →
  ∃ (l u : ℝ), l = -(7 * Real.sqrt 3) / 3 ∧ u = -2 * Real.sqrt 2 ∧
    Real.tan (α + β) ∈ Set.Icc l u :=
sorry

end tan_sum_range_l1196_119635


namespace divisibility_properties_l1196_119604

theorem divisibility_properties (n : ℤ) :
  -- Part (a)
  (n = 3 → ∃ m₁ m₂ : ℤ, m₁ = -5 ∧ m₂ = 9 ∧
    ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 ↔ m = m₁ ∨ m = m₂) ∧
  -- Part (b)
  (∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0) ∧
  -- Part (c)
  (∃ k : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m ≤ k) :=
by sorry

end divisibility_properties_l1196_119604


namespace smallest_two_digit_multiple_of_17_l1196_119678

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, is_two_digit n ∧ 17 ∣ n → 34 ≤ n :=
by sorry

end smallest_two_digit_multiple_of_17_l1196_119678


namespace proposition_implication_l1196_119624

theorem proposition_implication (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  (¬ P 5) →
  (¬ P 4) :=
sorry

end proposition_implication_l1196_119624


namespace pauls_score_l1196_119679

theorem pauls_score (total_points cousin_points : ℕ) 
  (h1 : total_points = 5816)
  (h2 : cousin_points = 2713) :
  total_points - cousin_points = 3103 := by
  sorry

end pauls_score_l1196_119679


namespace base6_arithmetic_sum_l1196_119610

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- Calculates the number of terms in the arithmetic sequence --/
def numTerms (first last step : ℕ) : ℕ := (last - first) / step + 1

/-- Calculates the sum of an arithmetic sequence --/
def arithmeticSum (n first last : ℕ) : ℕ := n * (first + last) / 2

theorem base6_arithmetic_sum :
  let first := base6ToBase10 2
  let last := base6ToBase10 50
  let step := base6ToBase10 2
  let n := numTerms first last step
  let sum := arithmeticSum n first last
  base10ToBase6 sum = 1040 := by sorry

end base6_arithmetic_sum_l1196_119610


namespace negation_of_exists_greater_than_one_l1196_119612

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by sorry

end negation_of_exists_greater_than_one_l1196_119612


namespace ginas_account_fractions_l1196_119691

theorem ginas_account_fractions (betty_balance : ℝ) (gina_combined_balance : ℝ)
  (h1 : betty_balance = 3456)
  (h2 : gina_combined_balance = 1728) :
  ∃ (f1 f2 : ℝ), f1 + f2 = 1/2 ∧ f1 * betty_balance + f2 * betty_balance = gina_combined_balance :=
by sorry

end ginas_account_fractions_l1196_119691


namespace sum_of_squares_and_square_of_sum_l1196_119628

theorem sum_of_squares_and_square_of_sum : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end sum_of_squares_and_square_of_sum_l1196_119628


namespace not_enough_apples_for_pie_l1196_119633

theorem not_enough_apples_for_pie (tessa_initial : Real) (anita_gave : Real) (pie_requirement : Real) : 
  tessa_initial = 4.75 → anita_gave = 5.5 → pie_requirement = 12.25 → tessa_initial + anita_gave < pie_requirement :=
by
  sorry

end not_enough_apples_for_pie_l1196_119633


namespace roque_walking_time_l1196_119688

/-- The time it takes Roque to walk to work -/
def walking_time : ℝ := sorry

/-- The time it takes Roque to bike to work -/
def biking_time : ℝ := 1

/-- Number of times Roque walks to and from work per week -/
def walks_per_week : ℕ := 3

/-- Number of times Roque bikes to and from work per week -/
def bikes_per_week : ℕ := 2

/-- Total commuting time in a week -/
def total_commute_time : ℝ := 16

theorem roque_walking_time :
  walking_time = 2 :=
by
  have h1 : (2 * walking_time * walks_per_week) + (2 * biking_time * bikes_per_week) = total_commute_time := by sorry
  sorry

end roque_walking_time_l1196_119688


namespace find_C_l1196_119613

theorem find_C (A B C : ℤ) (h1 : A = 509) (h2 : A = B + 197) (h3 : C = B - 125) : C = 187 := by
  sorry

end find_C_l1196_119613


namespace inequality_equivalence_l1196_119697

theorem inequality_equivalence :
  ∀ y : ℝ, (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ ((7 ≤ y ∧ y ≤ 11) ∨ (-3 ≤ y ∧ y ≤ 1)) := by
  sorry

end inequality_equivalence_l1196_119697


namespace det_A_eq_neg_46_l1196_119668

def A : Matrix (Fin 3) (Fin 3) ℝ := !![2, -4, 5; 0, 6, -2; 3, -1, 2]

theorem det_A_eq_neg_46 : Matrix.det A = -46 := by
  sorry

end det_A_eq_neg_46_l1196_119668


namespace equation_solution_l1196_119690

theorem equation_solution : ∃! y : ℚ, 7 * (4 * y - 3) + 5 = 3 * (-2 + 8 * y) :=
  by sorry

end equation_solution_l1196_119690


namespace power_of_four_l1196_119622

theorem power_of_four (k : ℝ) : (4 : ℝ) ^ (2 * k + 2) = 400 → (4 : ℝ) ^ k = 5 := by
  sorry

end power_of_four_l1196_119622


namespace min_distance_to_line_l1196_119655

theorem min_distance_to_line (x y : ℝ) : 
  (x + 2)^2 + (y - 3)^2 = 1 → 
  ∃ (min : ℝ), min = 15 ∧ ∀ (a b : ℝ), (a + 2)^2 + (b - 3)^2 = 1 → |3*a + 4*b - 26| ≥ min :=
by sorry

end min_distance_to_line_l1196_119655


namespace polygon_angle_sum_l1196_119645

theorem polygon_angle_sum (n : ℕ) (x : ℝ) : 
  n ≥ 3 → 
  0 < x → 
  x < 180 → 
  (n - 2) * 180 + x = 1350 → 
  n = 9 ∧ x = 90 := by
sorry

end polygon_angle_sum_l1196_119645


namespace first_winner_of_both_prizes_l1196_119608

theorem first_winner_of_both_prizes (n : ℕ) : 
  (n % 5 = 0 ∧ n % 7 = 0) → n ≥ 35 :=
by sorry

end first_winner_of_both_prizes_l1196_119608


namespace factor_expression_l1196_119625

theorem factor_expression (y : ℝ) : y * (y + 3) + 2 * (y + 3) = (y + 2) * (y + 3) := by
  sorry

end factor_expression_l1196_119625


namespace f_properties_l1196_119614

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 - 18 * x + 5

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 12 * x^2 - 6 * x - 18

theorem f_properties :
  (f' (-1) = 0) ∧
  (f' (3/2) = 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) (3/2), f' x < 0) ∧
  (∀ x ∈ Set.Iic (-1 : ℝ), f' x > 0) ∧
  (∀ x ∈ Set.Ioi (3/2 : ℝ), f' x > 0) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ 16) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ -61/4) ∧
  (f (-1) = 16) ∧
  (f (3/2) = -61/4) :=
sorry

end f_properties_l1196_119614


namespace parallel_condition_perpendicular_condition_l1196_119677

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := (2*a + 1)*x + (a + 2)*y + 3 = 0
def l2 (a x y : ℝ) : Prop := (a - 1)*x - 2*y + 2 = 0

-- Define parallel and perpendicular conditions
def parallel (a : ℝ) : Prop := ∀ x y, l1 a x y ↔ ∃ k, l2 a (x + k * (2*a + 1)) (y + k * (a + 2))

def perpendicular (a : ℝ) : Prop := ∀ x1 y1 x2 y2, 
  l1 a x1 y1 → l2 a x2 y2 → (x2 - x1) * (2*a + 1) + (y2 - y1) * (a + 2) = 0

-- State the theorems
theorem parallel_condition : ∀ a : ℝ, parallel a ↔ a = 0 := by sorry

theorem perpendicular_condition : ∀ a : ℝ, perpendicular a ↔ a = -1 ∨ a = 5/2 := by sorry

end parallel_condition_perpendicular_condition_l1196_119677


namespace sqrt_n_divisors_characterization_l1196_119600

def has_sqrt_n_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).card = Nat.sqrt n

theorem sqrt_n_divisors_characterization :
  ∀ n : ℕ, has_sqrt_n_divisors n ↔ n = 1 ∨ n = 9 :=
by sorry

end sqrt_n_divisors_characterization_l1196_119600


namespace Fe2O3_weight_l1196_119618

/-- The atomic weight of iron in g/mol -/
def atomic_weight_Fe : ℝ := 55.845

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of iron atoms in Fe2O3 -/
def Fe_count : ℕ := 2

/-- The number of oxygen atoms in Fe2O3 -/
def O_count : ℕ := 3

/-- The number of moles of Fe2O3 -/
def moles_Fe2O3 : ℝ := 8

/-- The molecular weight of Fe2O3 in g/mol -/
def molecular_weight_Fe2O3 : ℝ := Fe_count * atomic_weight_Fe + O_count * atomic_weight_O

/-- The total weight of Fe2O3 in grams -/
def total_weight_Fe2O3 : ℝ := moles_Fe2O3 * molecular_weight_Fe2O3

theorem Fe2O3_weight : total_weight_Fe2O3 = 1277.496 := by sorry

end Fe2O3_weight_l1196_119618


namespace min_jumps_to_visit_all_l1196_119659

/-- Represents a jump on the circle -/
inductive Jump
| Two : Jump  -- Jump by 2 points
| Three : Jump  -- Jump by 3 points

/-- The number of points on the circle -/
def numPoints : Nat := 2016

/-- A sequence of jumps -/
def JumpSequence := List Jump

/-- Function to calculate the total distance covered by a sequence of jumps -/
def totalDistance (seq : JumpSequence) : Nat :=
  seq.foldl (fun acc jump => acc + match jump with
    | Jump.Two => 2
    | Jump.Three => 3) 0

/-- Predicate to check if a sequence of jumps visits all points and returns to start -/
def isValidSequence (seq : JumpSequence) : Prop :=
  totalDistance seq % numPoints = 0 ∧ seq.length ≥ numPoints

/-- The main theorem -/
theorem min_jumps_to_visit_all :
  ∃ (seq : JumpSequence), isValidSequence seq ∧ seq.length = 2017 ∧
  (∀ (other : JumpSequence), isValidSequence other → seq.length ≤ other.length) :=
sorry

end min_jumps_to_visit_all_l1196_119659


namespace train_crossing_pole_time_l1196_119656

/-- Proves that a train with a given length and speed takes a specific time to cross a pole -/
theorem train_crossing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 100 →
  crossing_time = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_pole_time

end train_crossing_pole_time_l1196_119656


namespace absolute_value_equals_sqrt_square_l1196_119673

theorem absolute_value_equals_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end absolute_value_equals_sqrt_square_l1196_119673


namespace no_solution_for_all_a_b_l1196_119692

theorem no_solution_for_all_a_b : ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧
  ¬∃ (x y : ℝ), (Real.tan (13 * x) * Real.tan (a * y) = 1) ∧
                (Real.tan (21 * x) * Real.tan (b * y) = 1) := by
  sorry

end no_solution_for_all_a_b_l1196_119692


namespace min_value_of_expression_lower_bound_is_tight_l1196_119670

theorem min_value_of_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 9 * Real.sqrt 5 / 5 := by sorry

theorem lower_bound_is_tight : 
  ∃ (x : ℝ), (x^2 + 9) / Real.sqrt (x^2 + 5) = 9 * Real.sqrt 5 / 5 := by sorry

end min_value_of_expression_lower_bound_is_tight_l1196_119670


namespace c_range_l1196_119652

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem c_range (c : ℝ) : c > 0 →
  (is_decreasing (fun x ↦ c^x) ∨ (∀ x ∈ Set.Icc 0 2, x + c > 2)) ∧
  ¬(is_decreasing (fun x ↦ c^x) ∧ (∀ x ∈ Set.Icc 0 2, x + c > 2)) →
  (0 < c ∧ c < 1) ∨ c > 2 :=
by sorry

end c_range_l1196_119652


namespace seminar_invitations_count_l1196_119695

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 6 teachers out of 10 for a seminar,
    where two specific teachers (A and B) cannot attend together -/
def seminar_invitations : ℕ :=
  2 * binomial 8 5 + binomial 8 6

theorem seminar_invitations_count : seminar_invitations = 140 := by
  sorry

end seminar_invitations_count_l1196_119695


namespace empty_solution_set_iff_a_in_range_l1196_119636

theorem empty_solution_set_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 3 > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) :=
sorry

end empty_solution_set_iff_a_in_range_l1196_119636


namespace negation_of_existence_negation_of_square_plus_one_less_than_zero_l1196_119648

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_square_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_square_plus_one_less_than_zero_l1196_119648


namespace parallel_vectors_m_value_l1196_119671

/-- Given two vectors a and b in R², if (a + b) is parallel to (m*a - b), then m = -1 -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (4, -1))
    (h2 : b = (-5, 2))
    (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (m • a - b)) : 
  m = -1 := by
  sorry

end parallel_vectors_m_value_l1196_119671


namespace solution_set_l1196_119619

theorem solution_set (x : ℝ) : 4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9 ↔ 63 / 26 < x ∧ x ≤ 28 / 11 := by
  sorry

end solution_set_l1196_119619


namespace range_of_f_l1196_119623

def f (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

theorem range_of_f :
  {y : ℝ | ∃ x ≥ 0, f x = y} = {y : ℝ | y ≥ 9} := by sorry

end range_of_f_l1196_119623


namespace olympic_rings_area_l1196_119665

/-- Olympic Ring -/
structure OlympicRing where
  outer_diameter : ℝ
  inner_diameter : ℝ

/-- Olympic Emblem -/
structure OlympicEmblem where
  rings : Fin 5 → OlympicRing
  hypotenuse : ℝ

/-- The area covered by the Olympic rings -/
def area_covered (e : OlympicEmblem) : ℝ :=
  sorry

/-- The theorem statement -/
theorem olympic_rings_area (e : OlympicEmblem)
  (h1 : ∀ i : Fin 5, (e.rings i).outer_diameter = 22)
  (h2 : ∀ i : Fin 5, (e.rings i).inner_diameter = 18)
  (h3 : e.hypotenuse = 24) :
  ∃ ε > 0, |area_covered e - 592| < ε :=
sorry

end olympic_rings_area_l1196_119665


namespace curve_is_line_segment_l1196_119607

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the parameter range
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 5}

-- Theorem statement
theorem curve_is_line_segment :
  ∃ (a b c : ℝ), ∀ t ∈ t_range, a * x t + b * y t + c = 0 ∧
  ∃ (x_min x_max : ℝ), (∀ t ∈ t_range, x_min ≤ x t ∧ x t ≤ x_max) ∧
  x_min < x_max :=
sorry

end curve_is_line_segment_l1196_119607


namespace combination_equality_l1196_119657

theorem combination_equality (x : ℕ) : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose x 4) ↔ x = 6 := by
  sorry

end combination_equality_l1196_119657


namespace arithmetic_sum_1_to_19_l1196_119651

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ aₙ : ℕ) (d : ℕ) : ℕ := 
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Proof that the sum of the arithmetic sequence 1, 3, 5, ..., 17, 19 is 100 -/
theorem arithmetic_sum_1_to_19 : arithmetic_sum 1 19 2 = 100 := by
  sorry

#eval arithmetic_sum 1 19 2

end arithmetic_sum_1_to_19_l1196_119651


namespace jakes_earnings_l1196_119694

/-- Jake's lawn mowing and flower planting problem -/
theorem jakes_earnings (mowing_time mowing_pay planting_time desired_rate : ℝ) 
  (h1 : mowing_time = 1)
  (h2 : mowing_pay = 15)
  (h3 : planting_time = 2)
  (h4 : desired_rate = 20) :
  let total_time := mowing_time + planting_time
  let total_desired_earnings := desired_rate * total_time
  let planting_charge := total_desired_earnings - mowing_pay
  planting_charge = 45 := by sorry

end jakes_earnings_l1196_119694


namespace area_between_circles_l1196_119685

/-- The area between a circumscribing circle and two externally tangent circles -/
theorem area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  let R := (r₁ + r₂) / 2
  π * R^2 - (π * r₁^2 + π * r₂^2) = 24 * π :=
by sorry

end area_between_circles_l1196_119685


namespace square_coverage_l1196_119650

/-- The smallest number of 3-by-4 rectangles needed to cover a square region exactly -/
def min_rectangles : ℕ := 12

/-- The side length of the square region -/
def square_side : ℕ := 12

/-- The width of each rectangle -/
def rectangle_width : ℕ := 3

/-- The height of each rectangle -/
def rectangle_height : ℕ := 4

theorem square_coverage :
  (square_side * square_side) = (min_rectangles * rectangle_width * rectangle_height) ∧
  (square_side % rectangle_width = 0) ∧
  (square_side % rectangle_height = 0) ∧
  ∀ n : ℕ, n < min_rectangles →
    (n * rectangle_width * rectangle_height) < (square_side * square_side) :=
by sorry

end square_coverage_l1196_119650


namespace min_neg_half_third_l1196_119602

theorem min_neg_half_third : min (-1/2 : ℚ) (-1/3) = -1/2 := by sorry

end min_neg_half_third_l1196_119602


namespace consecutive_negative_integers_product_l1196_119606

theorem consecutive_negative_integers_product (n : ℤ) :
  n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 2240 →
  |n - (n + 1)| = 1 :=
by sorry

end consecutive_negative_integers_product_l1196_119606


namespace circle_area_with_diameter_12_l1196_119693

theorem circle_area_with_diameter_12 (π : Real) (diameter : Real) (area : Real) :
  diameter = 12 →
  area = π * (diameter / 2)^2 →
  area = π * 36 :=
by sorry

end circle_area_with_diameter_12_l1196_119693


namespace function_symmetry_l1196_119660

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom functional_equation : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

-- State the theorem
theorem function_symmetry : ∀ x : ℝ, f (-x) = f x := by sorry

end function_symmetry_l1196_119660


namespace circle_equation_k_value_l1196_119644

theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 12*x + y^2 + 8*y - k = 0 ↔ (x + 6)^2 + (y + 4)^2 = 25) →
  k = -27 :=
by sorry

end circle_equation_k_value_l1196_119644


namespace gcd_5_factorial_7_factorial_l1196_119611

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_5_factorial_7_factorial : 
  Nat.gcd (factorial 5) (factorial 7) = factorial 5 := by sorry

end gcd_5_factorial_7_factorial_l1196_119611


namespace hybrid_car_percentage_l1196_119639

/-- Proves that the percentage of hybrid cars in a dealership is 60% -/
theorem hybrid_car_percentage
  (total_cars : ℕ)
  (hybrids_with_full_headlights : ℕ)
  (hybrid_one_headlight_percent : ℚ)
  (h1 : total_cars = 600)
  (h2 : hybrids_with_full_headlights = 216)
  (h3 : hybrid_one_headlight_percent = 40 / 100) :
  (hybrids_with_full_headlights / (1 - hybrid_one_headlight_percent) : ℚ) / total_cars = 60 / 100 :=
by sorry

end hybrid_car_percentage_l1196_119639


namespace triangle_ABC_properties_l1196_119689

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Given conditions
  (a * Real.cos B + b * Real.cos A = -2 * c * Real.cos C) →
  (c = Real.sqrt 7) →
  (b = 2) →
  -- Conclusions
  (C = 2 * Real.pi / 3) ∧
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 2) := by
  sorry

end triangle_ABC_properties_l1196_119689


namespace may_savings_l1196_119686

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (month 0)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end may_savings_l1196_119686


namespace least_reducible_fraction_l1196_119630

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(Nat.gcd (m - 17) (6 * m + 7) > 1)) ∧
  Nat.gcd (n - 17) (6 * n + 7) > 1 ∧
  n = 126 := by
sorry

end least_reducible_fraction_l1196_119630


namespace intersection_of_three_lines_l1196_119675

/-- If three lines ax + y + 1 = 0, y = 3x, and x + y = 4 intersect at one point, then a = -4 -/
theorem intersection_of_three_lines (a : ℝ) : 
  (∃! p : ℝ × ℝ, a * p.1 + p.2 + 1 = 0 ∧ p.2 = 3 * p.1 ∧ p.1 + p.2 = 4) →
  a = -4 := by
sorry

end intersection_of_three_lines_l1196_119675


namespace minimum_point_of_translated_graph_l1196_119627

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 3| + 5

-- State the theorem
theorem minimum_point_of_translated_graph :
  ∃! (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 5 ∧ x = 3 :=
sorry

end minimum_point_of_translated_graph_l1196_119627


namespace kids_played_correct_l1196_119662

/-- The number of kids Julia played with on each day --/
structure KidsPlayed where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the problem --/
def satisfiesConditions (k : KidsPlayed) : Prop :=
  k.tuesday = 14 ∧
  k.wednesday = k.tuesday + (k.tuesday / 4 + 1) ∧
  k.thursday = 2 * k.wednesday - 4 ∧
  k.monday = k.tuesday + 8

/-- The theorem to prove --/
theorem kids_played_correct : 
  ∃ (k : KidsPlayed), satisfiesConditions k ∧ 
    k.monday = 22 ∧ k.tuesday = 14 ∧ k.wednesday = 18 ∧ k.thursday = 32 := by
  sorry

end kids_played_correct_l1196_119662


namespace points_are_coplanar_l1196_119640

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem points_are_coplanar
  (h_nonzero : e₁ ≠ 0 ∧ e₂ ≠ 0)
  (h_not_collinear : ¬ ∃ (k : ℝ), e₁ = k • e₂)
  (h_AB : B - A = e₁ + e₂)
  (h_AC : C - A = -3 • e₁ + 7 • e₂)
  (h_AD : D - A = 2 • e₁ - 3 • e₂) :
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = d • (0 : V) :=
sorry

end points_are_coplanar_l1196_119640


namespace line_equal_intercepts_l1196_119653

/-- 
Given a line mx - y - 3 - m = 0, if its intercepts on the x-axis and y-axis are equal, 
then m = -3 or m = -1.
-/
theorem line_equal_intercepts (m : ℝ) : 
  (∃ (a : ℝ), a ≠ 0 ∧ m * a - 3 - m = 0 ∧ -3 - m = a) → 
  (m = -3 ∨ m = -1) :=
by sorry

end line_equal_intercepts_l1196_119653


namespace area_with_holes_formula_l1196_119687

/-- The area of a rectangle with holes -/
def area_with_holes (x : ℝ) : ℝ :=
  let large_rectangle_area := (x + 8) * (x + 6)
  let hole_area := (2 * x - 4) * (x - 3)
  let total_hole_area := 2 * hole_area
  large_rectangle_area - total_hole_area

/-- Theorem: The area of the rectangle with holes is equal to -3x^2 + 34x + 24 -/
theorem area_with_holes_formula (x : ℝ) :
  area_with_holes x = -3 * x^2 + 34 * x + 24 := by
  sorry

#check area_with_holes_formula

end area_with_holes_formula_l1196_119687


namespace simplify_expression_l1196_119696

theorem simplify_expression (a b : ℝ) : (15*a + 45*b) + (21*a + 32*b) - (12*a + 40*b) = 24*a + 37*b := by
  sorry

end simplify_expression_l1196_119696


namespace parabola_equation_l1196_119654

/-- A parabola with vertex (h, k) and vertical axis of symmetry has the form y = a(x-h)^2 + k -/
def is_vertical_parabola (a h k : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = a * (x - h)^2 + k

theorem parabola_equation (f : ℝ → ℝ) :
  (∀ x, f x = -3 * x^2 + 18 * x - 22) →
  is_vertical_parabola (-3) 3 5 f ∧
  f 2 = 2 :=
by sorry

end parabola_equation_l1196_119654


namespace extreme_value_implies_a_equals_5_l1196_119669

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a_equals_5 (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3) ∨ f a x ≥ f a (-3)) →
  a = 5 :=
by sorry

end extreme_value_implies_a_equals_5_l1196_119669


namespace square_root_of_factorial_fraction_l1196_119617

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 210) = 216 * Real.sqrt 3 := by
  sorry

end square_root_of_factorial_fraction_l1196_119617


namespace complex_equation_solution_l1196_119609

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l1196_119609


namespace job_completion_proof_l1196_119638

/-- The number of days it takes to complete the job with the initial number of machines -/
def initial_days : ℝ := 12

/-- The number of days it takes to complete the job after adding more machines -/
def new_days : ℝ := 8

/-- The number of additional machines added -/
def additional_machines : ℕ := 6

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 12

theorem job_completion_proof :
  ∀ (rate : ℝ),
  rate > 0 →
  (initial_machines : ℝ) * rate * initial_days = 1 →
  ((initial_machines : ℝ) + additional_machines) * rate * new_days = 1 →
  initial_machines = 12 := by
  sorry

end job_completion_proof_l1196_119638


namespace michaels_matchsticks_l1196_119666

theorem michaels_matchsticks (total : ℕ) : 
  (30 * 10 + 20 * 15 + 10 * 25 : ℕ) = (2 * total) / 3 → total = 1275 := by
  sorry

end michaels_matchsticks_l1196_119666


namespace square_root_x_minus_y_l1196_119698

theorem square_root_x_minus_y (x y : ℝ) (h : Real.sqrt (x - 2) + (y + 1)^2 = 0) : 
  (Real.sqrt (x - y))^2 = 3 := by
  sorry

end square_root_x_minus_y_l1196_119698


namespace height_difference_l1196_119626

-- Define the heights in inches
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height : ℕ := 3 * 12  -- 3 feet converted to inches

-- Theorem statement
theorem height_difference : carter_height - betty_height = 12 := by
  sorry

end height_difference_l1196_119626


namespace carriage_sharing_problem_l1196_119664

theorem carriage_sharing_problem (x : ℝ) : 
  (x > 0) →                            -- Ensure positive number of people
  (x / 3 + 2 = (x - 9) / 2) →           -- The equation to be proved
  (∃ n : ℕ, x = n) →                    -- Ensure x is a natural number
  (x / 3 + 2 : ℝ) = (x - 9) / 2 :=      -- The equation represents the problem
by
  sorry

end carriage_sharing_problem_l1196_119664


namespace fraction_equality_l1196_119629

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 12)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 := by
  sorry

end fraction_equality_l1196_119629


namespace square_sum_greater_than_one_l1196_119682

theorem square_sum_greater_than_one
  (x y z t : ℝ)
  (h : (x^2 + y^2 - 1) * (z^2 + t^2 - 1) > (x*z + y*t - 1)^2) :
  x^2 + y^2 > 1 := by
sorry

end square_sum_greater_than_one_l1196_119682


namespace complex_fraction_equality_l1196_119637

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 - a*b + b^2 = 0) : 
  (a^8 + b^8) / (a^2 + b^2)^4 = 2 := by
  sorry

end complex_fraction_equality_l1196_119637


namespace same_acquaintance_count_l1196_119642

theorem same_acquaintance_count (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n) ∧ f i = k ∧ f j = k) :=
by sorry

end same_acquaintance_count_l1196_119642


namespace water_fraction_proof_l1196_119601

def initial_water : ℚ := 18
def initial_total : ℚ := 20
def removal_amount : ℚ := 5
def num_iterations : ℕ := 3

def water_fraction_after_iterations : ℚ := 
  (initial_water / initial_total) * ((initial_total - removal_amount) / initial_total) ^ num_iterations

theorem water_fraction_proof : water_fraction_after_iterations = 243 / 640 := by
  sorry

end water_fraction_proof_l1196_119601


namespace set_union_condition_implies_m_geq_two_l1196_119632

theorem set_union_condition_implies_m_geq_two (m : ℝ) :
  let A : Set ℝ := {x | x ≥ 2}
  let B : Set ℝ := {x | x ≥ m}
  A ∪ B = A → m ≥ 2 := by
  sorry

end set_union_condition_implies_m_geq_two_l1196_119632


namespace sin_cos_equality_implies_pi_quarter_l1196_119680

theorem sin_cos_equality_implies_pi_quarter (x : Real) :
  x ∈ Set.Icc 0 Real.pi →
  Real.sin (x + Real.sin x) = Real.cos (x - Real.cos x) →
  x = Real.pi / 4 := by
sorry

end sin_cos_equality_implies_pi_quarter_l1196_119680


namespace total_blocks_l1196_119605

theorem total_blocks (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 18)
  (h2 : yellow = red + 7)
  (h3 : blue = red + 14) :
  red + yellow + blue = 75 := by
  sorry

end total_blocks_l1196_119605


namespace least_three_digit_multiple_l1196_119661

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 9 ∣ n ∧ 
  (∀ m : ℕ, (100 ≤ m ∧ m < 1000) ∧ 3 ∣ m ∧ 4 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  n = 108 := by
  sorry

end least_three_digit_multiple_l1196_119661


namespace largest_number_l1196_119621

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value of 85 in base 9 --/
def num1 : Nat := to_base_10 [5, 8] 9

/-- The value of 210 in base 6 --/
def num2 : Nat := to_base_10 [0, 1, 2] 6

/-- The value of 1000 in base 4 --/
def num3 : Nat := to_base_10 [0, 0, 0, 1] 4

/-- The value of 111111 in base 2 --/
def num4 : Nat := to_base_10 [1, 1, 1, 1, 1, 1] 2

theorem largest_number : num2 > num1 ∧ num2 > num3 ∧ num2 > num4 := by
  sorry

end largest_number_l1196_119621


namespace field_length_is_48_l1196_119634

-- Define the field and pond
def field_width : ℝ := sorry
def field_length : ℝ := 2 * field_width
def pond_side : ℝ := 8

-- Define the areas
def field_area : ℝ := field_length * field_width
def pond_area : ℝ := pond_side^2

-- State the theorem
theorem field_length_is_48 :
  field_length = 2 * field_width ∧
  pond_side = 8 ∧
  pond_area = (1/18) * field_area →
  field_length = 48 := by sorry

end field_length_is_48_l1196_119634


namespace solution_pairs_l1196_119649

theorem solution_pairs : 
  ∃ (S : Set (ℕ × ℕ)), 
    S = {(0, 0), (1, 0)} ∧ 
    ∀ (a b : ℕ) (x : ℝ), 
      (a, b) ∈ S ↔ 
        (-2 * (a : ℝ) + (b : ℝ)^2 = Real.cos (π * (a : ℝ) + (b : ℝ)^2) - 1 ∧
         (b : ℝ)^2 = Real.cos (2 * π * (a : ℝ) + (b : ℝ)^2) - 1) :=
by sorry

end solution_pairs_l1196_119649


namespace parabola_tangent_slope_l1196_119681

-- Define the parabola
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 9

-- Define the derivative of the parabola
def parabola_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem parabola_tangent_slope (a b : ℝ) :
  parabola a b 2 = -1 →
  parabola_derivative a b 2 = 1 →
  a = 3 ∧ b = -11 := by
sorry

end parabola_tangent_slope_l1196_119681


namespace vector_expression_simplification_l1196_119684

variable {V : Type*} [AddCommGroup V]

/-- For any four points A, B, C, D in a vector space,
    the expression AC - BD + CD - AB equals the zero vector. -/
theorem vector_expression_simplification
  (A B C D : V) : (C - A) - (D - B) + (D - C) - (B - A) = (0 : V) := by
  sorry

end vector_expression_simplification_l1196_119684


namespace sweater_price_after_discounts_l1196_119603

/-- Calculates the final price of an item after two successive discounts -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2)

/-- Theorem: The final price of a $240 sweater after 60% and 30% discounts is $67.20 -/
theorem sweater_price_after_discounts :
  finalPrice 240 0.6 0.3 = 67.2 := by
  sorry

end sweater_price_after_discounts_l1196_119603


namespace product_of_roots_l1196_119615

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = 14 → ∃ y : ℝ, (x + 2) * (x - 3) = 14 ∧ (x * y = -20) := by
  sorry

end product_of_roots_l1196_119615


namespace only_contrapositive_correct_l1196_119683

theorem only_contrapositive_correct (p q r : Prop) 
  (h : (p ∨ q) → ¬r) : 
  (¬((p ∨ q) → ¬r) ∧ 
   ¬(¬r → p) ∧ 
   ¬(r → ¬(p ∨ q)) ∧ 
   ((¬p ∧ ¬q) → r)) := by
  sorry

end only_contrapositive_correct_l1196_119683


namespace circumscribed_sphere_radius_l1196_119674

noncomputable def inscribed_sphere_radius : ℝ := Real.sqrt 6 - 1

theorem circumscribed_sphere_radius
  (inscribed_radius : ℝ)
  (h_inscribed_radius : inscribed_radius = inscribed_sphere_radius)
  (h_touching : inscribed_radius > 0) :
  ∃ (circumscribed_radius : ℝ),
    circumscribed_radius = 5 * (Real.sqrt 2 + 1) * inscribed_radius :=
by sorry

end circumscribed_sphere_radius_l1196_119674


namespace fraction_equality_l1196_119646

theorem fraction_equality : (12 : ℚ) / (8 * 75) = 1 / 50 := by
  sorry

end fraction_equality_l1196_119646


namespace average_salary_theorem_l1196_119620

def salary_A : ℕ := 9000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary_theorem :
  (total_salary : ℚ) / num_people = 8200 := by sorry

end average_salary_theorem_l1196_119620


namespace negative_two_less_than_negative_two_thirds_l1196_119667

theorem negative_two_less_than_negative_two_thirds : -2 < -(2/3) := by
  sorry

end negative_two_less_than_negative_two_thirds_l1196_119667


namespace samara_friends_average_alligators_l1196_119663

/-- Given a group of people searching for alligators, calculate the average number
    of alligators seen by friends, given the total number seen, the number seen by
    one person, and the number of friends. -/
def average_alligators_seen_by_friends 
  (total_alligators : ℕ) 
  (alligators_seen_by_one : ℕ) 
  (num_friends : ℕ) : ℚ :=
  (total_alligators - alligators_seen_by_one) / num_friends

/-- Prove that given the specific values from the problem, 
    the average number of alligators seen by each friend is 10. -/
theorem samara_friends_average_alligators :
  average_alligators_seen_by_friends 50 20 3 = 10 := by
  sorry

end samara_friends_average_alligators_l1196_119663
