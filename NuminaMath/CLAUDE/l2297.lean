import Mathlib

namespace subtract_to_one_l2297_229780

theorem subtract_to_one : ∃ x : ℤ, (-5) - x = 1 := by
  sorry

end subtract_to_one_l2297_229780


namespace bread_baking_pattern_l2297_229753

/-- A sequence of bread loaves baked over 6 days -/
def BreadSequence : Type := Fin 6 → ℕ

/-- The condition that the daily increase grows by 1 each day -/
def IncreasingDifference (s : BreadSequence) : Prop :=
  ∀ i : Fin 4, s (i + 1) - s i < s (i + 2) - s (i + 1)

/-- The known values for days 1, 2, 3, 4, and 6 -/
def KnownValues (s : BreadSequence) : Prop :=
  s 0 = 5 ∧ s 1 = 7 ∧ s 2 = 10 ∧ s 3 = 14 ∧ s 5 = 25

theorem bread_baking_pattern (s : BreadSequence) 
  (h1 : IncreasingDifference s) (h2 : KnownValues s) : s 4 = 19 := by
  sorry

end bread_baking_pattern_l2297_229753


namespace nested_radical_value_l2297_229794

/-- The value of the infinite nested radical √(6 + √(6 + √(6 + ...))) -/
noncomputable def nested_radical : ℝ :=
  Real.sqrt (6 + Real.sqrt (6 + Real.sqrt (6 + Real.sqrt 6)))

/-- The nested radical equals 3 -/
theorem nested_radical_value : nested_radical = 3 := by
  sorry

end nested_radical_value_l2297_229794


namespace jessa_cupcakes_l2297_229734

/-- The number of cupcakes needed for a given number of classes and students per class -/
def cupcakes_needed (num_classes : ℕ) (students_per_class : ℕ) : ℕ :=
  num_classes * students_per_class

theorem jessa_cupcakes : 
  let fourth_grade_cupcakes := cupcakes_needed 3 30
  let pe_class_cupcakes := cupcakes_needed 1 50
  fourth_grade_cupcakes + pe_class_cupcakes = 140 := by
  sorry

end jessa_cupcakes_l2297_229734


namespace factorization_left_to_right_l2297_229773

theorem factorization_left_to_right (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end factorization_left_to_right_l2297_229773


namespace orderedPartitions_of_five_l2297_229763

/-- The number of ordered partitions of a positive integer n into positive integers -/
def orderedPartitions (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: The number of ordered partitions of 5 is 16 -/
theorem orderedPartitions_of_five :
  orderedPartitions 5 = 16 := by
  sorry

end orderedPartitions_of_five_l2297_229763


namespace xyz_product_l2297_229711

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 5 * y = -20)
  (eq2 : y * z + 5 * z = -20)
  (eq3 : z * x + 5 * x = -20) :
  x * y * z = 100 := by
  sorry

end xyz_product_l2297_229711


namespace amanda_ticket_sales_l2297_229729

/-- The number of tickets Amanda needs to sell on the third day -/
def tickets_to_sell_day3 (total_goal : ℕ) (sold_day1 : ℕ) (sold_day2 : ℕ) : ℕ :=
  total_goal - (sold_day1 + sold_day2)

/-- Theorem stating that Amanda needs to sell 28 tickets on the third day -/
theorem amanda_ticket_sales : tickets_to_sell_day3 80 20 32 = 28 := by
  sorry

end amanda_ticket_sales_l2297_229729


namespace tournament_committee_count_l2297_229714

/-- The number of teams in the frisbee league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team for the committee -/
def host_committee_size : ℕ := 4

/-- The number of members selected from each non-host team for the committee -/
def non_host_committee_size : ℕ := 3

/-- The total size of the tournament committee -/
def total_committee_size : ℕ := 13

/-- The number of possible tournament committees -/
def num_possible_committees : ℕ := 3443073600

theorem tournament_committee_count :
  (num_teams * (Nat.choose team_size host_committee_size) * 
   (Nat.choose team_size non_host_committee_size) ^ (num_teams - 1)) = num_possible_committees :=
by sorry

end tournament_committee_count_l2297_229714


namespace parallelogram_area_32_22_l2297_229732

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_32_22 :
  parallelogram_area 32 22 = 704 := by
  sorry

end parallelogram_area_32_22_l2297_229732


namespace monomial_properties_l2297_229727

-- Define a monomial as a product of a coefficient and variables with non-negative integer exponents
structure Monomial (α : Type*) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the coefficient of a monomial
def coefficient {α : Type*} [CommRing α] (m : Monomial α) : α := m.coeff

-- Define the degree of a monomial
def degree {α : Type*} [CommRing α] (m : Monomial α) : Nat :=
  m.vars.foldl (fun acc (_, exp) => acc + exp) 0

-- The monomial -1/3 * x * y^2
def m : Monomial ℚ := ⟨-1/3, [(1, 1), (2, 2)]⟩

-- Theorem statement
theorem monomial_properties :
  coefficient m = -1/3 ∧ degree m = 3 := by sorry

end monomial_properties_l2297_229727


namespace negative_one_third_less_than_negative_point_three_l2297_229747

theorem negative_one_third_less_than_negative_point_three : -1/3 < -0.3 := by
  sorry

end negative_one_third_less_than_negative_point_three_l2297_229747


namespace min_length_line_segment_ellipse_l2297_229725

/-- The minimum length of a line segment AB on an ellipse -/
theorem min_length_line_segment_ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let ellipse := {p : ℝ × ℝ | (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1}
  ∃ (A B : ℝ × ℝ), A ∈ ellipse ∧ B ∈ ellipse ∧ 
    (A.1 * B.1 + A.2 * B.2 = 0) ∧  -- OA ⊥ OB
    ∀ (C D : ℝ × ℝ), C ∈ ellipse → D ∈ ellipse → (C.1 * D.1 + C.2 * D.2 = 0) →
      (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 ≤ (C.1 - D.1) ^ 2 + (C.2 - D.2) ^ 2 ∧
      (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = (2 * a * b * Real.sqrt (a ^ 2 + b ^ 2) / (a ^ 2 + b ^ 2)) ^ 2 :=
by sorry

end min_length_line_segment_ellipse_l2297_229725


namespace pure_imaginary_complex_number_l2297_229702

theorem pure_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (a^2 + 2*a - 3) + (a + 3)*Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = 1 := by
  sorry

end pure_imaginary_complex_number_l2297_229702


namespace intersection_M_N_l2297_229708

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 2}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 3} := by
  sorry

end intersection_M_N_l2297_229708


namespace divisibility_equivalence_l2297_229720

theorem divisibility_equivalence (n : ℕ+) :
  11 ∣ (n.val^5 + 5^n.val) ↔ 11 ∣ (n.val^5 * 5^n.val + 1) := by
  sorry

end divisibility_equivalence_l2297_229720


namespace johns_distance_conversion_l2297_229742

/-- Converts a base-8 number to base-10 --/
def base8_to_base10 (d₃ d₂ d₁ d₀ : ℕ) : ℕ :=
  d₃ * 8^3 + d₂ * 8^2 + d₁ * 8^1 + d₀ * 8^0

/-- John's weekly hiking distance in base 8 is 3762 --/
def johns_distance_base8 : ℕ × ℕ × ℕ × ℕ := (3, 7, 6, 2)

theorem johns_distance_conversion :
  let (d₃, d₂, d₁, d₀) := johns_distance_base8
  base8_to_base10 d₃ d₂ d₁ d₀ = 2034 :=
by sorry

end johns_distance_conversion_l2297_229742


namespace isosceles_max_angle_diff_l2297_229759

/-- An isosceles triangle has two equal angles -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sum_180 : a + b + c = 180
  isosceles : (a = b) ∨ (b = c) ∨ (a = c)

/-- Given an isosceles triangle with one angle 50°, prove that the maximum difference between the other two angles is 30° -/
theorem isosceles_max_angle_diff (t : IsoscelesTriangle) (h : t.a = 50 ∨ t.b = 50 ∨ t.c = 50) :
  ∃ (x y : ℝ), ((x = t.a ∧ y = t.b) ∨ (x = t.b ∧ y = t.c) ∨ (x = t.a ∧ y = t.c)) ∧
  (∀ (x' y' : ℝ), ((x' = t.a ∧ y' = t.b) ∨ (x' = t.b ∧ y' = t.c) ∨ (x' = t.a ∧ y' = t.c)) →
  |x' - y'| ≤ |x - y|) ∧ |x - y| = 30 :=
sorry

end isosceles_max_angle_diff_l2297_229759


namespace sequence_sum_l2297_229776

theorem sequence_sum (A B C D E F G H I : ℤ) : 
  E = 7 →
  A + B + C + D = 40 →
  B + C + D + E = 40 →
  C + D + E + F = 40 →
  D + E + F + G = 40 →
  E + F + G + H = 40 →
  F + G + H + I = 40 →
  A + I = 40 := by
sorry

end sequence_sum_l2297_229776


namespace first_number_in_ratio_l2297_229779

/-- Given two positive integers a and b with a ratio of 3:4 and LCM 180, prove that a = 45 -/
theorem first_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 3 / 4 → 
  Nat.lcm a b = 180 → 
  a = 45 := by
sorry

end first_number_in_ratio_l2297_229779


namespace equation_real_solutions_l2297_229731

theorem equation_real_solutions (a b x : ℝ) : 
  (∃ x : ℝ, Real.sqrt (2 * a + b + 2 * x) + Real.sqrt (10 * a + 9 * b - 6 * x) = 2 * Real.sqrt (2 * a + b - 2 * x)) ↔ 
  ((0 ≤ a ∧ -a ≤ b ∧ b ≤ 0 ∧ (x = Real.sqrt (a * (a + b)) ∨ x = -Real.sqrt (a * (a + b)))) ∨
   (a ≥ -8/9 * a ∧ -8/9 * a ≥ b ∧ b ≤ 0 ∧ x = -Real.sqrt (a * (a + b)))) :=
by sorry

end equation_real_solutions_l2297_229731


namespace pool_width_l2297_229707

/-- Proves the width of a rectangular pool given its draining rate, dimensions, initial capacity, and time to drain. -/
theorem pool_width
  (drain_rate : ℝ)
  (length depth : ℝ)
  (initial_capacity : ℝ)
  (drain_time : ℝ)
  (h1 : drain_rate = 60)
  (h2 : length = 150)
  (h3 : depth = 10)
  (h4 : initial_capacity = 0.8)
  (h5 : drain_time = 800) :
  ∃ (width : ℝ), width = 40 ∧ 
    drain_rate * drain_time = initial_capacity * (length * width * depth) :=
by sorry

end pool_width_l2297_229707


namespace female_officers_on_duty_percentage_l2297_229796

def total_on_duty : ℕ := 240
def female_ratio_on_duty : ℚ := 1/2
def total_female_officers : ℕ := 300

theorem female_officers_on_duty_percentage :
  (female_ratio_on_duty * total_on_duty) / total_female_officers * 100 = 40 := by
  sorry

end female_officers_on_duty_percentage_l2297_229796


namespace house_transaction_loss_l2297_229744

def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (gain_percent : ℝ) : ℝ :=
  let first_sale := initial_value * (1 - loss_percent)
  let second_sale := first_sale * (1 + gain_percent)
  initial_value - second_sale

theorem house_transaction_loss :
  house_transaction 9000 0.1 0.1 = 810 := by
  sorry

end house_transaction_loss_l2297_229744


namespace diophantine_equation_solutions_l2297_229728

theorem diophantine_equation_solutions :
  ∃ (S : Finset (ℤ × ℤ)), (∀ (p : ℤ × ℤ), p ∈ S → (p.1^2 + p.2^2 = 26 * p.1)) ∧ S.card ≥ 12 := by
  sorry

end diophantine_equation_solutions_l2297_229728


namespace triangle_rectangle_area_l2297_229756

theorem triangle_rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : 
  square_area = 1600 ∧ rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2/5) * circle_radius
  let triangle_height := 3 * circle_radius
  let triangle_area := (1/2) * rectangle_length * triangle_height
  let rectangle_area := rectangle_length * rectangle_breadth
  triangle_area + rectangle_area = 1120 := by
  sorry

end triangle_rectangle_area_l2297_229756


namespace smallest_x_for_simplified_fractions_l2297_229746

theorem smallest_x_for_simplified_fractions : ∃ (x : ℕ), x > 0 ∧
  (∀ (k : ℕ), k ≥ 1 ∧ k ≤ 40 → Nat.gcd (3*x + k) (k + 7) = 1) ∧
  (∀ (y : ℕ), y > 0 ∧ y < x → ∃ (k : ℕ), k ≥ 1 ∧ k ≤ 40 ∧ Nat.gcd (3*y + k) (k + 7) ≠ 1) ∧
  x = 5 :=
by sorry

end smallest_x_for_simplified_fractions_l2297_229746


namespace tony_packs_count_l2297_229777

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := 3

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of pens Kendra and Tony each keep for themselves -/
def pens_kept : ℕ := 2

/-- The number of friends who receive pens -/
def friends : ℕ := 14

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

theorem tony_packs_count :
  tony_packs * pens_per_pack + kendra_packs * pens_per_pack = 
  friends + 2 * pens_kept :=
by sorry

end tony_packs_count_l2297_229777


namespace geometric_sequence_sum_l2297_229785

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l2297_229785


namespace calculation_proof_l2297_229760

theorem calculation_proof : (0.0077 * 4.5) / (0.05 * 0.1 * 0.007) = 989.2857142857143 := by
  sorry

end calculation_proof_l2297_229760


namespace good_functions_count_l2297_229770

/-- A function f: ℤ → {1, 2, ..., n} is good if it satisfies the given condition -/
def IsGoodFunction (n : ℕ) (f : ℤ → Fin n) : Prop :=
  n ≥ 2 ∧ ∀ k : Fin (n-1), ∃ j : ℤ, ∀ m : ℤ,
    (f (m + j) : ℤ) ≡ (f (m + k) : ℤ) - (f m : ℤ) [ZMOD (n+1)]

/-- The number of good functions for a given n -/
def NumberOfGoodFunctions (n : ℕ) : ℕ := sorry

theorem good_functions_count (n : ℕ) :
  (n ≥ 2 ∧ NumberOfGoodFunctions n = n * Nat.totient n) ↔ Nat.Prime (n+1) :=
sorry

end good_functions_count_l2297_229770


namespace sin_plus_sqrt3_cos_l2297_229752

/-- Given an angle θ in the second quadrant such that tan(θ + π/3) = 1/2,
    prove that sin θ + √3 cos θ = -2√5/5 -/
theorem sin_plus_sqrt3_cos (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (θ + π/3) = 1/2) : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry


end sin_plus_sqrt3_cos_l2297_229752


namespace solution_set_part1_range_of_a_part2_l2297_229745

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - 1|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 3 x ≥ x + 9} = {x : ℝ | x < -11/3 ∨ x > 7} := by sorry

-- Part 2
theorem range_of_a_part2 :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 0 1, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 2 := by sorry

end solution_set_part1_range_of_a_part2_l2297_229745


namespace initial_crayons_count_l2297_229790

/-- 
Given a person who:
1. Has an initial number of crayons
2. Loses half of their crayons
3. Buys 20 new crayons
4. Ends up with 29 crayons total
This theorem proves that the initial number of crayons was 18.
-/
theorem initial_crayons_count (initial : ℕ) 
  (h1 : initial / 2 + 20 = 29) : initial = 18 := by
  sorry

end initial_crayons_count_l2297_229790


namespace min_value_expression_l2297_229791

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (m : ℝ), m = Real.sqrt 3 ∧ 
  (∀ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), 
    x^2 + y^2 + 1/x^2 + y/x ≥ m) ∧
  (∃ (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0), 
    x^2 + y^2 + 1/x^2 + y/x = m) := by
  sorry

end min_value_expression_l2297_229791


namespace problem_one_problem_two_l2297_229710

-- Problem 1
theorem problem_one : 4 * Real.sqrt 2 + Real.sqrt 8 - Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem_two : Real.sqrt (1 + 1/3) / Real.sqrt (2 + 1/3) * Real.sqrt (1 + 2/5) = 2 * Real.sqrt 5 / 5 := by
  sorry

end problem_one_problem_two_l2297_229710


namespace complex_fraction_simplification_l2297_229722

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the complex fraction simplifies to the given result -/
theorem complex_fraction_simplification :
  (3 * (1 + i)^2) / (i - 1) = 3 - 3*i :=
by
  sorry

end complex_fraction_simplification_l2297_229722


namespace square_diagonals_are_equal_l2297_229713

-- Define the basic shapes
class Square
class Parallelogram

-- Define the property of having equal diagonals
def has_equal_diagonals (α : Type*) := Prop

-- State the given conditions
axiom square_equal_diagonals : has_equal_diagonals Square
axiom parallelogram_equal_diagonals : has_equal_diagonals Parallelogram
axiom square_is_parallelogram : Square → Parallelogram

-- State the theorem to be proved
theorem square_diagonals_are_equal : has_equal_diagonals Square := by
  sorry

end square_diagonals_are_equal_l2297_229713


namespace sum_a_b_equals_31_l2297_229709

/-- The number of divisors of a positive integer -/
def num_divisors (x : ℕ+) : ℕ := sorry

/-- The product of the smallest ⌈n/2⌉ divisors of x -/
def f (x : ℕ+) : ℕ := sorry

/-- The least value of x such that f(x) is a multiple of x -/
def a : ℕ+ := sorry

/-- The least value of n such that there exists y with n factors and f(y) is a multiple of y -/
def b : ℕ := sorry

theorem sum_a_b_equals_31 : (a : ℕ) + b = 31 := by sorry

end sum_a_b_equals_31_l2297_229709


namespace anyas_age_l2297_229704

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem anyas_age :
  ∃ (age : ℕ), 
    110 ≤ sum_of_first_n age ∧ 
    sum_of_first_n age ≤ 130 ∧ 
    age = 15 := by
  sorry

end anyas_age_l2297_229704


namespace smallest_b_in_geometric_sequence_l2297_229757

theorem smallest_b_in_geometric_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  (∃ r : ℝ, 0 < r ∧ a * r = b ∧ b * r = c) →  -- geometric sequence condition
  a * b * c = 125 →  -- product condition
  ∀ x : ℝ, (0 < x ∧ ∃ y z : ℝ, 0 < y ∧ 0 < z ∧ 
    (∃ r : ℝ, 0 < r ∧ y * r = x ∧ x * r = z) ∧ 
    y * x * z = 125) → 
  5 ≤ x :=
by sorry

end smallest_b_in_geometric_sequence_l2297_229757


namespace negation_of_existence_negation_of_proposition_l2297_229754

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, 2 * x + 1 ≤ 0) ↔ (∀ x : ℝ, 2 * x + 1 > 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l2297_229754


namespace smallest_digit_divisible_by_7_l2297_229775

def is_divisible_by_7 (n : ℕ) : Prop := ∃ k, n = 7 * k

def number_with_digit (x : ℕ) : ℕ := 5200 + 10 * x + 4

theorem smallest_digit_divisible_by_7 :
  (∃ x : ℕ, x ≤ 9 ∧ is_divisible_by_7 (number_with_digit x)) ∧
  (∀ y : ℕ, y < 2 → ¬is_divisible_by_7 (number_with_digit y)) ∧
  is_divisible_by_7 (number_with_digit 2) :=
by sorry

end smallest_digit_divisible_by_7_l2297_229775


namespace fraction_units_and_exceed_l2297_229718

def fraction_units (numerator denominator : ℕ) : ℕ := numerator

def units_to_exceed (start target : ℕ) : ℕ :=
  if start ≥ target then 1 else target - start + 1

theorem fraction_units_and_exceed :
  (fraction_units 5 8 = 5) ∧
  (units_to_exceed 5 8 = 4) := by sorry

end fraction_units_and_exceed_l2297_229718


namespace mechanism_efficiency_problem_l2297_229741

theorem mechanism_efficiency_problem (t_combined t_partial t_remaining : ℝ) 
  (h_combined : t_combined = 30)
  (h_partial : t_partial = 6)
  (h_remaining : t_remaining = 40) :
  ∃ (t1 t2 : ℝ),
    t1 = 75 ∧ 
    t2 = 50 ∧ 
    (1 / t1 + 1 / t2 = 1 / t_combined) ∧
    (t_partial * (1 / t1 + 1 / t2) + t_remaining / t2 = 1) :=
by sorry

end mechanism_efficiency_problem_l2297_229741


namespace rectangle_division_theorem_l2297_229771

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

theorem rectangle_division_theorem :
  ∃ (original : Rectangle) (largest smallest : Rectangle),
    (∃ (other1 other2 : Rectangle),
      area original = area largest + area smallest + area other1 + area other2) ∧
    perimeter largest = 28 ∧
    perimeter smallest = 12 ∧
    area original = 96 :=
by sorry

end rectangle_division_theorem_l2297_229771


namespace vector_simplification_l2297_229787

variable {V : Type*} [AddCommGroup V]

variable (A B C M O : V)

theorem vector_simplification :
  (A - B + M - B) + (B - O + B - C) + (O - M) = A - C :=
by sorry

end vector_simplification_l2297_229787


namespace negation_of_universal_proposition_l2297_229733

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x < -1)) ↔ (∃ x : ℝ, x ≥ -1) := by
  sorry

end negation_of_universal_proposition_l2297_229733


namespace highest_power_of_1991_l2297_229701

theorem highest_power_of_1991 :
  let n : ℕ := 1990^(1991^1992) + 1992^(1991^1990)
  ∃ k : ℕ, k = 2 ∧ (1991 : ℕ)^k ∣ n ∧ ∀ m : ℕ, m > k → ¬((1991 : ℕ)^m ∣ n) := by
  sorry

end highest_power_of_1991_l2297_229701


namespace ten_attendants_used_both_l2297_229705

/-- The number of attendants who used both a pencil and a pen at a meeting -/
def attendants_using_both (pencil_users pen_users only_one_tool_users : ℕ) : ℕ :=
  (pencil_users + pen_users - only_one_tool_users) / 2

/-- Theorem stating that 10 attendants used both a pencil and a pen -/
theorem ten_attendants_used_both :
  attendants_using_both 25 15 20 = 10 := by
  sorry

end ten_attendants_used_both_l2297_229705


namespace circle_not_intersecting_diagonal_probability_l2297_229750

/-- The probability that a circle of radius 1 randomly placed inside a 15 × 36 rectangle
    does not intersect the diagonal of the rectangle -/
theorem circle_not_intersecting_diagonal_probability : ℝ := by
  -- Define the rectangle dimensions
  let rectangle_width : ℝ := 15
  let rectangle_height : ℝ := 36
  
  -- Define the circle radius
  let circle_radius : ℝ := 1
  
  -- Define the valid region for circle center
  let valid_region_width : ℝ := rectangle_width - 2 * circle_radius
  let valid_region_height : ℝ := rectangle_height - 2 * circle_radius
  
  -- Calculate the area of the valid region
  let valid_region_area : ℝ := valid_region_width * valid_region_height
  
  -- Define the safe area where the circle doesn't intersect the diagonal
  let safe_area : ℝ := 375
  
  -- Calculate the probability
  let probability : ℝ := safe_area / valid_region_area
  
  -- Prove that the probability equals 375/442
  sorry

#eval (375 : ℚ) / 442

end circle_not_intersecting_diagonal_probability_l2297_229750


namespace repeating_decimal_to_fraction_l2297_229764

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), 100 * x = 56 + x ∧ n * x = x) → x = 56 / 99 := by
  sorry

end repeating_decimal_to_fraction_l2297_229764


namespace triangle_existence_l2297_229784

/-- Represents a line in 2D space -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Theorem stating the existence of a triangle given specific conditions -/
theorem triangle_existence 
  (base_length : ℝ) 
  (base_direction : ℝ × ℝ) 
  (angle_difference : ℝ) 
  (third_vertex_line : Line) : 
  ∃ (t : Triangle), 
    -- The base of the triangle has the given length
    (Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = base_length) ∧
    -- The direction of the base matches the given direction
    ((t.B.1 - t.A.1, t.B.2 - t.A.2) = base_direction) ∧
    -- The difference between the base angles is as specified
    (∃ (α β : ℝ), α > β ∧ α - β = angle_difference) ∧
    -- The third vertex lies on the given line
    (∃ (k : ℝ), t.C = (third_vertex_line.point.1 + k * third_vertex_line.direction.1,
                       third_vertex_line.point.2 + k * third_vertex_line.direction.2)) := by
  sorry

end triangle_existence_l2297_229784


namespace smallest_sum_l2297_229774

theorem smallest_sum (A B C D : ℕ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = B - A ∧ B - A = d) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = B * r ∧ D = C * r) →  -- B, C, D form a geometric sequence
  C = (4 * B) / 3 →  -- C/B = 4/3
  (∀ A' B' C' D' : ℕ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = B' - A' ∧ B' - A' = d) →
    (∃ r : ℚ, C' = B' * r ∧ D' = C' * r) →
    C' = (4 * B') / 3 →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 43 :=
by sorry

end smallest_sum_l2297_229774


namespace slipper_cost_theorem_l2297_229739

/-- Calculate the total cost of slippers with embroidery and shipping --/
def calculate_slipper_cost (original_price : ℝ) (discount_rate : ℝ) 
  (embroidery_cost_multiple : ℝ) (num_initials : ℕ) (base_shipping : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let embroidery_cost := 2 * (embroidery_cost_multiple * num_initials)
  let total_cost := discounted_price + embroidery_cost + base_shipping
  total_cost

/-- Theorem stating the total cost of the slippers --/
theorem slipper_cost_theorem :
  calculate_slipper_cost 50 0.1 4.5 3 10 = 82 :=
by sorry

end slipper_cost_theorem_l2297_229739


namespace girls_math_questions_l2297_229721

def total_questions (fiona_per_hour shirley_per_hour kiana_per_hour : ℕ) (hours : ℕ) : ℕ :=
  (fiona_per_hour + shirley_per_hour + kiana_per_hour) * hours

theorem girls_math_questions :
  ∀ (fiona_per_hour : ℕ),
    fiona_per_hour = 36 →
    ∀ (shirley_per_hour : ℕ),
      shirley_per_hour = 2 * fiona_per_hour →
      ∀ (kiana_per_hour : ℕ),
        kiana_per_hour = (fiona_per_hour + shirley_per_hour) / 2 →
        total_questions fiona_per_hour shirley_per_hour kiana_per_hour 2 = 324 :=
by
  sorry

#eval total_questions 36 72 54 2

end girls_math_questions_l2297_229721


namespace closest_to_zero_minus_one_closest_l2297_229715

def integers : List ℤ := [-1, 2, -3, 4, -5]

theorem closest_to_zero (n : ℤ) (h : n ∈ integers) : 
  ∀ m ∈ integers, |n| ≤ |m| :=
by
  sorry

theorem minus_one_closest : 
  ∃ n ∈ integers, ∀ m ∈ integers, |n| ≤ |m| ∧ n = -1 :=
by
  sorry

end closest_to_zero_minus_one_closest_l2297_229715


namespace sqrt_sum_fractions_l2297_229748

theorem sqrt_sum_fractions : Real.sqrt ((1 : ℝ) / 8 + 1 / 18) = Real.sqrt 26 / 12 := by
  sorry

end sqrt_sum_fractions_l2297_229748


namespace integer_coordinates_cubic_l2297_229762

/-- A cubic function with integer coordinates for extrema and inflection point -/
structure IntegerCubic where
  n : ℤ
  p : ℤ
  c : ℤ

/-- The cubic function with the given coefficients -/
def cubic_function (f : IntegerCubic) (x : ℝ) : ℝ :=
  x^3 + 3 * f.n * x^2 + 3 * (f.n^2 - f.p^2) * x + f.c

/-- The first derivative of the cubic function -/
def cubic_derivative (f : IntegerCubic) (x : ℝ) : ℝ :=
  3 * x^2 + 6 * f.n * x + 3 * (f.n^2 - f.p^2)

/-- The second derivative of the cubic function -/
def cubic_second_derivative (f : IntegerCubic) (x : ℝ) : ℝ :=
  6 * x + 6 * f.n

/-- Theorem: The cubic function has integer coordinates for extrema and inflection point -/
theorem integer_coordinates_cubic (f : IntegerCubic) :
  ∃ (x1 x2 xi : ℤ),
    (cubic_derivative f x1 = 0 ∧ cubic_derivative f x2 = 0) ∧
    cubic_second_derivative f xi = 0 ∧
    (∀ x : ℤ, cubic_derivative f x = 0 → x = x1 ∨ x = x2) ∧
    (∀ x : ℤ, cubic_second_derivative f x = 0 → x = xi) :=
  sorry

end integer_coordinates_cubic_l2297_229762


namespace son_age_proof_l2297_229730

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 26 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 24 :=
by
  sorry

end son_age_proof_l2297_229730


namespace linear_function_derivative_l2297_229792

/-- Given a linear function f(x) = ax + 3 where f'(1) = 3, prove that a = 3 -/
theorem linear_function_derivative (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x + 3) ∧ (deriv f 1 = 3)) →
  a = 3 := by
  sorry

end linear_function_derivative_l2297_229792


namespace quadratic_factorization_sum_l2297_229782

theorem quadratic_factorization_sum (a b c : ℤ) :
  (∀ x, x^2 + 19*x + 88 = (x + a) * (x + b)) →
  (∀ x, x^2 - 23*x + 132 = (x - b) * (x - c)) →
  a + b + c = 31 := by
sorry

end quadratic_factorization_sum_l2297_229782


namespace fly_path_length_l2297_229781

theorem fly_path_length (r : ℝ) (path_end : ℝ) (h1 : r = 100) (h2 : path_end = 120) : 
  let diameter := 2 * r
  let chord := Real.sqrt (diameter^2 - path_end^2)
  diameter + chord + path_end = 480 := by sorry

end fly_path_length_l2297_229781


namespace right_triangle_ab_length_l2297_229737

/-- Given a right triangle ABC in the x-y plane with ∠B = 90°, 
    if the length of AC is 25 and the slope of AC is 4/3, 
    then the length of AB is 15. -/
theorem right_triangle_ab_length 
  (A B C : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (ac_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 25)
  (ac_slope : (C.2 - A.2) / (C.1 - A.1) = 4/3) :
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 15 :=
by sorry

end right_triangle_ab_length_l2297_229737


namespace line_parallel_to_parallel_plane_l2297_229766

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (containedIn : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (parallelLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (l : Line) (α β : Plane)
  (h1 : parallelPlanes α β)
  (h2 : containedIn l α) :
  parallelLineToPlane l β :=
sorry

end line_parallel_to_parallel_plane_l2297_229766


namespace sampling_more_suitable_for_large_population_l2297_229788

/-- Represents a survey method -/
inductive SurveyMethod
  | Comprehensive
  | Sampling

/-- Represents the characteristics of a survey -/
structure SurveyCharacteristics where
  populationSize : ℕ
  isSurveyingLargePopulation : Bool

/-- Determines the most suitable survey method based on survey characteristics -/
def mostSuitableSurveyMethod (characteristics : SurveyCharacteristics) : SurveyMethod :=
  if characteristics.isSurveyingLargePopulation then
    SurveyMethod.Sampling
  else
    SurveyMethod.Comprehensive

/-- Theorem: For a large population survey, sampling is more suitable than comprehensive -/
theorem sampling_more_suitable_for_large_population 
  (characteristics : SurveyCharacteristics) 
  (h : characteristics.isSurveyingLargePopulation = true) : 
  mostSuitableSurveyMethod characteristics = SurveyMethod.Sampling :=
by
  sorry

end sampling_more_suitable_for_large_population_l2297_229788


namespace max_servings_is_56_l2297_229712

/-- Ingredients required for one serving of salad -/
structure SaladServing where
  cucumbers : ℕ := 2
  tomatoes : ℕ := 2
  brynza : ℕ := 75  -- in grams
  peppers : ℕ := 1

/-- Ingredients available in the warehouse -/
structure Warehouse where
  cucumbers : ℕ := 117
  tomatoes : ℕ := 116
  brynza : ℕ := 4200  -- converted from 4.2 kg to grams
  peppers : ℕ := 60

/-- Calculate the maximum number of servings that can be made -/
def maxServings (w : Warehouse) (s : SaladServing) : ℕ :=
  min (w.cucumbers / s.cucumbers)
      (min (w.tomatoes / s.tomatoes)
           (min (w.brynza / s.brynza)
                (w.peppers / s.peppers)))

/-- Theorem: The maximum number of servings that can be made is 56 -/
theorem max_servings_is_56 (w : Warehouse) (s : SaladServing) :
  maxServings w s = 56 := by
  sorry

end max_servings_is_56_l2297_229712


namespace cube_inequality_l2297_229798

theorem cube_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x > a^y) : x^3 < y^3 := by
  sorry

end cube_inequality_l2297_229798


namespace simplify_expression_calculate_expression_calculate_profit_l2297_229724

-- Statement 1
theorem simplify_expression (a b : ℝ) :
  -3 * (a + b)^2 - 6 * (a + b)^2 + 8 * (a + b)^2 = -(a + b)^2 := by
  sorry

-- Statement 2
theorem calculate_expression (a b c d : ℝ) 
  (h1 : a - 2*b = 5) 
  (h2 : 2*b - c = -7) 
  (h3 : c - d = 12) :
  4*(a - c) + 4*(2*b - d) - 4*(2*b - c) = 40 := by
  sorry

-- Statement 3
theorem calculate_profit (initial_cost standard_price : ℝ) (sales : List ℝ) 
  (h1 : initial_cost = 400)
  (h2 : standard_price = 56)
  (h3 : sales = [-3, 7, -8, 9, -2, 0, -1, -6])
  (h4 : sales.length = 8) :
  (sales.sum + 8 * standard_price) - initial_cost = 44 := by
  sorry

end simplify_expression_calculate_expression_calculate_profit_l2297_229724


namespace logarithm_sum_equality_logarithm_product_equality_l2297_229740

-- Part 1
theorem logarithm_sum_equality : 2 * Real.log 10 / Real.log 2 + Real.log 0.04 / Real.log 2 = 2 := by sorry

-- Part 2
theorem logarithm_product_equality : 
  (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) * 
  (Real.log 5 / Real.log 3 + Real.log 5 / Real.log 9) * 
  (Real.log 2 / Real.log 5 + Real.log 2 / Real.log 25) = 15/8 := by sorry

end logarithm_sum_equality_logarithm_product_equality_l2297_229740


namespace inscribed_cube_edge_length_l2297_229793

theorem inscribed_cube_edge_length (S : Real) (r : Real) (x : Real) :
  S = 4 * Real.pi →  -- Surface area of the sphere
  S = 4 * Real.pi * r^2 →  -- Formula for surface area of a sphere
  x * Real.sqrt 3 = 2 * r →  -- Relationship between cube diagonal and sphere diameter
  x = 2 * Real.sqrt 3 / 3 :=  -- Edge length of the inscribed cube
by sorry

end inscribed_cube_edge_length_l2297_229793


namespace expression_value_l2297_229706

theorem expression_value : (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 := by
  sorry

end expression_value_l2297_229706


namespace combined_tennis_preference_l2297_229751

/-- Calculates the combined percentage of students preferring tennis across three schools -/
theorem combined_tennis_preference (north_students : ℕ) (north_tennis_pct : ℚ)
  (south_students : ℕ) (south_tennis_pct : ℚ)
  (valley_students : ℕ) (valley_tennis_pct : ℚ)
  (h1 : north_students = 1800)
  (h2 : north_tennis_pct = 25 / 100)
  (h3 : south_students = 3000)
  (h4 : south_tennis_pct = 50 / 100)
  (h5 : valley_students = 800)
  (h6 : valley_tennis_pct = 30 / 100) :
  (north_students * north_tennis_pct +
   south_students * south_tennis_pct +
   valley_students * valley_tennis_pct) /
  (north_students + south_students + valley_students) =
  39 / 100 := by
  sorry

end combined_tennis_preference_l2297_229751


namespace celeste_song_probability_l2297_229778

/-- Represents the collection of songs on Celeste's o-Pod -/
structure SongCollection where
  total_songs : Nat
  shortest_song : Nat
  song_increment : Nat
  favorite_song_length : Nat
  time_limit : Nat

/-- Calculates the probability of not hearing the entire favorite song 
    within the time limit for a given song collection -/
def probability_not_hearing_favorite (sc : SongCollection) : Rat :=
  1 - (Nat.factorial (sc.total_songs - 1) + Nat.factorial (sc.total_songs - 2)) / 
      Nat.factorial sc.total_songs

/-- The main theorem stating the probability for Celeste's specific case -/
theorem celeste_song_probability : 
  let sc : SongCollection := {
    total_songs := 12,
    shortest_song := 45,
    song_increment := 15,
    favorite_song_length := 240,
    time_limit := 300
  }
  probability_not_hearing_favorite sc = 10 / 11 := by
  sorry


end celeste_song_probability_l2297_229778


namespace ned_friend_games_l2297_229749

/-- The number of games Ned bought from his friend -/
def games_from_friend : ℕ := 50

/-- The number of games Ned bought at the garage sale -/
def garage_sale_games : ℕ := 27

/-- The number of games that didn't work -/
def non_working_games : ℕ := 74

/-- The number of good games Ned ended up with -/
def good_games : ℕ := 3

/-- Theorem stating that the number of games Ned bought from his friend is 50 -/
theorem ned_friend_games : 
  games_from_friend = 50 ∧
  games_from_friend + garage_sale_games = non_working_games + good_games :=
by sorry

end ned_friend_games_l2297_229749


namespace boys_pass_percentage_l2297_229789

theorem boys_pass_percentage (total_candidates : ℕ) (girls : ℕ) (girls_pass_rate : ℚ) (total_fail_rate : ℚ) :
  total_candidates = 2000 →
  girls = 900 →
  girls_pass_rate = 32 / 100 →
  total_fail_rate = 647 / 1000 →
  let boys := total_candidates - girls
  let total_pass_rate := 1 - total_fail_rate
  let total_pass := total_pass_rate * total_candidates
  let girls_pass := girls_pass_rate * girls
  let boys_pass := total_pass - girls_pass
  let boys_pass_rate := boys_pass / boys
  boys_pass_rate = 38 / 100 := by
sorry

end boys_pass_percentage_l2297_229789


namespace b_is_positive_l2297_229769

theorem b_is_positive (a b : ℝ) (h : ∀ x : ℝ, (x - a)^2 + b > 0) : b > 0 := by
  sorry

end b_is_positive_l2297_229769


namespace ones_digit_of_8_power_50_l2297_229772

theorem ones_digit_of_8_power_50 : 8^50 % 10 = 4 := by sorry

end ones_digit_of_8_power_50_l2297_229772


namespace max_value_mx_plus_ny_l2297_229703

theorem max_value_mx_plus_ny (a b : ℝ) (m n x y : ℝ) 
  (h1 : m^2 + n^2 = a) (h2 : x^2 + y^2 = b) :
  (∃ (k : ℝ), k = m*x + n*y ∧ ∀ (p q : ℝ), p^2 + q^2 = a → ∀ (r s : ℝ), r^2 + s^2 = b → 
    p*r + q*s ≤ k) → k = Real.sqrt (a*b) :=
sorry

end max_value_mx_plus_ny_l2297_229703


namespace odot_1_43_47_l2297_229765

/-- Custom operation ⊙ -/
def odot (a b c : ℤ) : ℤ := a * b * c + (a * b + b * c + c * a) - (a + b + c)

/-- Theorem stating that 1 ⊙ 43 ⊙ 47 = 4041 -/
theorem odot_1_43_47 : odot 1 43 47 = 4041 := by
  sorry

end odot_1_43_47_l2297_229765


namespace oplus_three_equals_fifteen_implies_a_equals_eleven_l2297_229799

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 3*a - 2*b^2

-- Theorem statement
theorem oplus_three_equals_fifteen_implies_a_equals_eleven :
  ∀ a : ℝ, oplus a 3 = 15 → a = 11 := by
sorry

end oplus_three_equals_fifteen_implies_a_equals_eleven_l2297_229799


namespace quadratic_minimum_l2297_229755

/-- The quadratic function f(x) = 3x^2 + 6x + 9 has its minimum value at x = -1 -/
theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + 6 * x + 9
  ∀ y : ℝ, f (-1) ≤ f y :=
by sorry

end quadratic_minimum_l2297_229755


namespace omm_moo_not_synonyms_l2297_229797

/-- Represents a word in the Ancient Tribe language --/
inductive Word
| empty : Word
| cons : Char → Word → Word

/-- Counts the number of occurrences of a given character in a word --/
def count_char (c : Char) : Word → Nat
| Word.empty => 0
| Word.cons x rest => (if x = c then 1 else 0) + count_char c rest

/-- Calculates the difference between the count of 'M's and 'O's in a word --/
def m_o_difference (w : Word) : Int :=
  (count_char 'M' w : Int) - (count_char 'O' w : Int)

/-- Defines when two words are synonyms --/
def are_synonyms (w1 w2 : Word) : Prop :=
  m_o_difference w1 = m_o_difference w2

/-- Represents the word OMM --/
def omm : Word := Word.cons 'O' (Word.cons 'M' (Word.cons 'M' Word.empty))

/-- Represents the word MOO --/
def moo : Word := Word.cons 'M' (Word.cons 'O' (Word.cons 'O' Word.empty))

/-- Theorem stating that OMM and MOO are not synonyms --/
theorem omm_moo_not_synonyms : ¬(are_synonyms omm moo) := by
  sorry

end omm_moo_not_synonyms_l2297_229797


namespace original_class_size_l2297_229783

theorem original_class_size (x : ℕ) : 
  (x > 0) →                        -- Ensure the class has at least one student
  (40 * x + 12 * 32) / (x + 12) = 36 →  -- New average age equation
  x = 12 :=
by sorry

end original_class_size_l2297_229783


namespace range_of_f_l2297_229761

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = {y | y ≥ 3} := by sorry

end range_of_f_l2297_229761


namespace root_sum_reciprocal_l2297_229767

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 24*x^2 + 98*x - 75

-- Define the theorem
theorem root_sum_reciprocal (p q r A B C : ℝ) :
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →  -- p, q, r are distinct
  (f p = 0 ∧ f q = 0 ∧ f r = 0) →  -- p, q, r are roots of f
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    5 / (s^3 - 24*s^2 + 98*s - 75) = A / (s-p) + B / (s-q) + C / (s-r)) →
  1/A + 1/B + 1/C = 256 :=
by sorry

end root_sum_reciprocal_l2297_229767


namespace star_polygon_n_value_l2297_229723

/-- Represents an n-pointed regular star polygon -/
structure StarPolygon (n : ℕ) where
  -- All 2n edges are congruent (implicit in the structure)
  -- Alternate angles A₁, A₂, ..., Aₙ are congruent (implicit)
  -- Alternate angles B₁, B₂, ..., Bₙ are congruent (implicit)
  angle_A : ℝ  -- Acute angle at each Aᵢ
  angle_B : ℝ  -- Acute angle at each Bᵢ
  angle_diff : angle_B = angle_A + 20  -- Angle difference condition
  sum_external : n * (angle_A + angle_B) = 360  -- Sum of external angles

/-- Theorem: For a star polygon satisfying the given conditions, n = 36 -/
theorem star_polygon_n_value :
  ∀ (n : ℕ) (s : StarPolygon n), n = 36 :=
by sorry

end star_polygon_n_value_l2297_229723


namespace smallest_number_with_remainders_l2297_229735

theorem smallest_number_with_remainders : ∃ (n : ℕ), n > 0 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  n % 8 = 7 ∧
  n % 9 = 8 ∧
  (∀ m : ℕ, m > 0 ∧
    m % 4 = 3 ∧
    m % 5 = 4 ∧
    m % 6 = 5 ∧
    m % 7 = 6 ∧
    m % 8 = 7 ∧
    m % 9 = 8 → m ≥ n) ∧
  n = 2519 :=
by
  sorry

end smallest_number_with_remainders_l2297_229735


namespace rahul_deepak_age_ratio_l2297_229716

/-- Given that Rahul will be 26 years old in 10 years and Deepak is currently 8 years old,
    prove that the ratio of Rahul's age to Deepak's age is 2:1. -/
theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    rahul_age + 10 = 26 →
    deepak_age = 8 →
    rahul_age / deepak_age = 2 := by
  sorry

end rahul_deepak_age_ratio_l2297_229716


namespace johnny_work_hours_l2297_229795

theorem johnny_work_hours (hourly_wage : ℝ) (total_earned : ℝ) (hours_worked : ℝ) : 
  hourly_wage = 2.35 →
  total_earned = 11.75 →
  hours_worked = total_earned / hourly_wage →
  hours_worked = 5 := by
sorry

end johnny_work_hours_l2297_229795


namespace elizabeth_haircut_l2297_229717

theorem elizabeth_haircut (first_day : ℝ) (second_day : ℝ) 
  (h1 : first_day = 0.38)
  (h2 : second_day = 0.5) :
  first_day + second_day = 0.88 := by
sorry

end elizabeth_haircut_l2297_229717


namespace unique_good_number_adjacent_to_power_of_two_l2297_229726

theorem unique_good_number_adjacent_to_power_of_two :
  ∃! n : ℕ, n > 0 ∧
  (∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a^b) ∧
  (∃ t : ℕ, t > 0 ∧ (n = 2^t + 1 ∨ n = 2^t - 1)) ∧
  n = 9 := by
sorry

end unique_good_number_adjacent_to_power_of_two_l2297_229726


namespace complex_number_problem_l2297_229743

variable (z : ℂ)

theorem complex_number_problem (h1 : ∃ (r : ℝ), z + 2*I = r) 
  (h2 : ∃ (s : ℝ), z / (2 - I) = s) : 
  z = 4 - 2*I ∧ Complex.abs (z / (1 + I)) = Real.sqrt 10 := by
  sorry

end complex_number_problem_l2297_229743


namespace sector_max_area_l2297_229736

/-- Given a sector with central angle α and radius R, if the perimeter of the sector
    is a constant C (C > 0), then the maximum area of the sector is C²/16. -/
theorem sector_max_area (α R C : ℝ) (h_pos : C > 0) :
  let perimeter := 2 * R + α * R
  let area := (1/2) * α * R^2
  (perimeter = C) → (∀ α' R', 2 * R' + α' * R' = C → (1/2) * α' * R'^2 ≤ C^2 / 16) ∧
  (∃ α' R', 2 * R' + α' * R' = C ∧ (1/2) * α' * R'^2 = C^2 / 16) :=
by sorry

end sector_max_area_l2297_229736


namespace square_fencing_cost_l2297_229758

/-- The number of sides in a square -/
def square_sides : ℕ := 4

/-- The cost of fencing each side of the square in dollars -/
def cost_per_side : ℕ := 79

/-- The total cost of fencing a square -/
def total_cost : ℕ := square_sides * cost_per_side

theorem square_fencing_cost : total_cost = 316 := by
  sorry

end square_fencing_cost_l2297_229758


namespace ticket_probabilities_l2297_229738

/-- Represents a group of tickets -/
structure TicketGroup where
  football : ℕ
  volleyball : ℕ

/-- The probability of drawing a football ticket from a group -/
def football_prob (group : TicketGroup) : ℚ :=
  group.football / (group.football + group.volleyball)

/-- The setup of the ticket drawing scenario -/
def ticket_scenario : Prop :=
  ∃ (group1 group2 : TicketGroup),
    group1.football = 6 ∧ group1.volleyball = 4 ∧
    group2.football = 4 ∧ group2.volleyball = 6

theorem ticket_probabilities (h : ticket_scenario) :
  ∃ (group1 group2 : TicketGroup),
    (football_prob group1 * football_prob group2 = 6/25) ∧
    (1 - (1 - football_prob group1) * (1 - football_prob group2) = 19/25) :=
by sorry

end ticket_probabilities_l2297_229738


namespace playground_area_l2297_229786

/-- Represents a rectangular landscape with a playground -/
structure Landscape where
  breadth : ℝ
  length : ℝ
  playgroundArea : ℝ

/-- The landscape satisfies the given conditions -/
def validLandscape (l : Landscape) : Prop :=
  l.length = 8 * l.breadth ∧
  l.length = 240 ∧
  l.playgroundArea = (1 / 6) * (l.length * l.breadth)

/-- Theorem: The playground area is 1200 square meters -/
theorem playground_area (l : Landscape) (h : validLandscape l) : 
  l.playgroundArea = 1200 := by
  sorry

end playground_area_l2297_229786


namespace positive_root_of_equation_l2297_229768

theorem positive_root_of_equation (x : ℝ) : 
  x > 0 ∧ (1/3) * (4*x^2 - 2) = (x^2 - 35*x - 7) * (x^2 + 20*x + 4) → 
  x = (35 + Real.sqrt 1257) / 2 := by
sorry

end positive_root_of_equation_l2297_229768


namespace shortest_altitude_of_triangle_l2297_229700

/-- The shortest altitude of a triangle with sides 9, 12, and 15 is 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 → 
  h * c = 2 * (a * b / 2) → 
  h ≤ a ∧ h ≤ b → 
  h = 7.2 := by sorry

end shortest_altitude_of_triangle_l2297_229700


namespace marias_car_trip_l2297_229719

theorem marias_car_trip (total_distance : ℝ) (remaining_distance : ℝ) 
  (h1 : total_distance = 400)
  (h2 : remaining_distance = 150) : 
  ∃ x : ℝ, x * total_distance + (1/4) * (total_distance - x * total_distance) = total_distance - remaining_distance ∧ x = 1/2 := by
sorry

end marias_car_trip_l2297_229719
