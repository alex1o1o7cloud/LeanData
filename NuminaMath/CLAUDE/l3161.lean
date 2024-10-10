import Mathlib

namespace four_digit_sum_11990_l3161_316107

def is_valid_digit (d : ℕ) : Prop := d > 0 ∧ d < 10

def distinct_digits (a b c d : ℕ) : Prop :=
  is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧ is_valid_digit d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def smallest_number (a b c d : ℕ) : ℕ := 1000 * a + 100 * b + 10 * c + d
def largest_number (a b c d : ℕ) : ℕ := 1000 * d + 100 * c + 10 * b + a

theorem four_digit_sum_11990 (a b c d : ℕ) :
  distinct_digits a b c d →
  (smallest_number a b c d + largest_number a b c d = 11990 ↔
   ((a = 1 ∧ b = 9 ∧ c = 9 ∧ d = 9) ∨ (a = 9 ∧ b = 9 ∧ c = 9 ∧ d = 1))) :=
by sorry

end four_digit_sum_11990_l3161_316107


namespace lamps_turned_on_l3161_316149

theorem lamps_turned_on (total_lamps : ℕ) (statement1 statement2 statement3 statement4 : Prop) :
  total_lamps = 10 →
  (statement1 ↔ (∃ x : ℕ, x = 5 ∧ x = total_lamps - (total_lamps - x))) →
  (statement2 ↔ ¬statement1) →
  (statement3 ↔ (∃ y : ℕ, y = 3 ∧ y = total_lamps - (total_lamps - y))) →
  (statement4 ↔ ∃ z : ℕ, z = total_lamps - (total_lamps - z) ∧ 2 ∣ z) →
  (statement1 ∨ statement2 ∨ statement3 ∨ statement4) →
  (statement1 → ¬statement2 ∧ ¬statement3 ∧ ¬statement4) →
  (statement2 → ¬statement1 ∧ ¬statement3 ∧ ¬statement4) →
  (statement3 → ¬statement1 ∧ ¬statement2 ∧ ¬statement4) →
  (statement4 → ¬statement1 ∧ ¬statement2 ∧ ¬statement3) →
  ∃ (lamps_on : ℕ), lamps_on = 9 ∧ lamps_on = total_lamps - (total_lamps - lamps_on) :=
by sorry

end lamps_turned_on_l3161_316149


namespace min_packages_for_scooter_l3161_316171

/-- The minimum number of packages to recover the cost of a scooter -/
def min_packages (scooter_cost : ℕ) (earning_per_package : ℕ) (fuel_cost : ℕ) : ℕ :=
  (scooter_cost + (earning_per_package - fuel_cost - 1)) / (earning_per_package - fuel_cost)

/-- Theorem stating the minimum number of packages needed to recover the scooter cost -/
theorem min_packages_for_scooter :
  min_packages 3200 15 4 = 291 :=
by sorry

end min_packages_for_scooter_l3161_316171


namespace tim_final_soda_cans_l3161_316160

/-- Calculates the final number of soda cans Tim has -/
def final_soda_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  let bought := remaining / 2
  remaining + bought

/-- Proves that Tim ends up with 24 cans of soda given the initial conditions -/
theorem tim_final_soda_cans : final_soda_cans 22 6 = 24 := by
  sorry

end tim_final_soda_cans_l3161_316160


namespace pages_in_harrys_book_l3161_316148

/-- Given that Selena's book has x pages and Harry's book has y fewer pages than half the number
    of pages of Selena's book, prove that the number of pages in Harry's book is equal to (x/2) - y. -/
theorem pages_in_harrys_book (x y : ℕ) : ℕ :=
  x / 2 - y

#check pages_in_harrys_book

end pages_in_harrys_book_l3161_316148


namespace f_composition_result_l3161_316137

-- Define the function f for complex numbers
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_composition_result : f (f (f (f (2 + I)))) = 3589 - 1984 * I := by
  sorry

end f_composition_result_l3161_316137


namespace expression_evaluation_l3161_316150

theorem expression_evaluation :
  let y : ℚ := 1/2
  (y + 1) * (y - 1) + (2*y - 1)^2 - 2*y*(2*y - 1) = -3/4 :=
by sorry

end expression_evaluation_l3161_316150


namespace roots_relationship_l3161_316185

/-- The polynomial h(x) = x^3 - 2x^2 - x + 2 -/
def h (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

/-- The polynomial j(x) = x^3 + bx^2 + cx + d -/
def j (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- The theorem stating the relationship between h and j -/
theorem roots_relationship (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ 
    h r₁ = 0 ∧ h r₂ = 0 ∧ h r₃ = 0) →
  (∀ x : ℝ, h x = 0 → j (x^3) b c d = 0) →
  b = 10 ∧ c = 31 ∧ d = 30 := by
sorry

end roots_relationship_l3161_316185


namespace fraction_equivalence_l3161_316198

theorem fraction_equivalence : 
  ∃ n : ℤ, (2 + n : ℚ) / (7 + n) = 3 / 4 ∧ n = 13 := by
  sorry

end fraction_equivalence_l3161_316198


namespace a_6_equals_12_l3161_316141

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_6_equals_12 
  (a : ℕ → ℚ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 9 = (1/2) * a 12 + 6) : 
  a 6 = 12 := by
  sorry

end a_6_equals_12_l3161_316141


namespace largest_multiple_of_12_with_5_hundreds_l3161_316128

theorem largest_multiple_of_12_with_5_hundreds : ∃! n : ℕ, 
  n % 12 = 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧
  (n / 100) % 10 = 5 ∧
  ∀ m : ℕ, m % 12 = 0 → 100 ≤ m → m < 1000 → (m / 100) % 10 = 5 → m ≤ n :=
by sorry

end largest_multiple_of_12_with_5_hundreds_l3161_316128


namespace min_value_theorem_l3161_316152

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : f a b 1 = 2) :
  ∃ (min_val : ℝ), min_val = 9 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a' + 4/b' ≥ min_val :=
sorry

end min_value_theorem_l3161_316152


namespace product_remainder_l3161_316115

theorem product_remainder (a b m : ℕ) (ha : a = 1488) (hb : b = 1977) (hm : m = 500) :
  (a * b) % m = 276 := by
  sorry

end product_remainder_l3161_316115


namespace shoe_percentage_gain_l3161_316106

/-- Prove that the percentage gain on the selling price of a shoe is approximately 16.67% -/
theorem shoe_percentage_gain :
  let manufacturing_cost : ℝ := 210
  let transportation_cost_per_100 : ℝ := 500
  let selling_price : ℝ := 258
  let total_cost : ℝ := manufacturing_cost + transportation_cost_per_100 / 100
  let gain : ℝ := selling_price - total_cost
  let percentage_gain : ℝ := gain / selling_price * 100
  ∃ ε > 0, abs (percentage_gain - 16.67) < ε :=
by sorry

end shoe_percentage_gain_l3161_316106


namespace rectangular_parallelepiped_dimensions_l3161_316110

def is_valid_dimension (x y z : ℕ) : Prop :=
  2 * (x * y + y * z + z * x) = x * y * z

def valid_dimensions : List (ℕ × ℕ × ℕ) :=
  [(6,6,6), (5,5,10), (4,8,8), (3,12,12), (3,7,42), (3,8,24), (3,9,18), (3,10,15), (4,5,20), (4,6,12)]

theorem rectangular_parallelepiped_dimensions (x y z : ℕ) :
  x > 0 ∧ y > 0 ∧ z > 0 → is_valid_dimension x y z → (x, y, z) ∈ valid_dimensions := by
  sorry

end rectangular_parallelepiped_dimensions_l3161_316110


namespace ratio_difference_bound_l3161_316157

theorem ratio_difference_bound (a : Fin 5 → ℝ) (h : ∀ i, a i > 0) :
  ∃ i j k l : Fin 5, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
  |a i / a j - a k / a l| < (1 : ℝ) / 2 := by
  sorry

end ratio_difference_bound_l3161_316157


namespace blue_pill_cost_proof_l3161_316116

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℝ := 23.5

/-- The cost of a red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days in the treatment period -/
def treatment_days : ℕ := 21

/-- The total cost of the treatment in dollars -/
def total_cost : ℝ := 945

theorem blue_pill_cost_proof :
  blue_pill_cost = 23.5 ∧
  red_pill_cost = blue_pill_cost - 2 ∧
  treatment_days = 21 ∧
  total_cost = 945 ∧
  treatment_days * (blue_pill_cost + red_pill_cost) = total_cost :=
by sorry

end blue_pill_cost_proof_l3161_316116


namespace alex_grocery_delivery_l3161_316133

/-- Represents the problem of calculating the total value of groceries Alex delivered --/
theorem alex_grocery_delivery 
  (savings : ℝ) 
  (car_cost : ℝ) 
  (trip_charge : ℝ) 
  (grocery_charge_percent : ℝ) 
  (num_trips : ℕ) 
  (h1 : savings = 14500) 
  (h2 : car_cost = 14600) 
  (h3 : trip_charge = 1.5) 
  (h4 : grocery_charge_percent = 0.05) 
  (h5 : num_trips = 40) 
  (h6 : savings + num_trips * trip_charge + grocery_charge_percent * (car_cost - savings - num_trips * trip_charge) / grocery_charge_percent ≥ car_cost) : 
  (car_cost - savings - num_trips * trip_charge) / grocery_charge_percent = 800 := by
  sorry

end alex_grocery_delivery_l3161_316133


namespace trigonometric_identities_l3161_316183

open Real

theorem trigonometric_identities (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : sin α = sqrt 5 / 5) : 
  sin (α + π/4) = 3 * sqrt 10 / 10 ∧ tan (2 * α) = 4/3 := by
sorry

end trigonometric_identities_l3161_316183


namespace divisibility_property_l3161_316143

theorem divisibility_property (a b : ℕ+) 
  (h : ∀ n : ℕ, n ≥ 1 → (a ^ n : ℕ) ∣ (b ^ (n + 1) : ℕ)) : 
  (a : ℕ) ∣ (b : ℕ) := by
sorry

end divisibility_property_l3161_316143


namespace rectangle_ratio_l3161_316199

/-- Given a rectangle with width 5 inches and area 100 square inches, 
    prove that the ratio of length to width is 4:1 -/
theorem rectangle_ratio (width : ℝ) (length : ℝ) (area : ℝ) :
  width = 5 →
  area = 100 →
  area = length * width →
  length / width = 4 := by
  sorry

end rectangle_ratio_l3161_316199


namespace no_solution_implies_m_geq_two_l3161_316139

theorem no_solution_implies_m_geq_two (m : ℝ) : 
  (∀ x : ℝ, ¬(2*x - 1 < 3 ∧ x > m)) → m ≥ 2 := by
  sorry

end no_solution_implies_m_geq_two_l3161_316139


namespace janes_age_problem_l3161_316122

theorem janes_age_problem :
  ∃ n : ℕ+, 
    (∃ x : ℕ+, n - 1 = x^3) ∧ 
    (∃ y : ℕ+, n + 4 = y^2) ∧ 
    n = 1332 := by
  sorry

end janes_age_problem_l3161_316122


namespace white_balls_count_l3161_316145

def total_balls : ℕ := 40
def red_frequency : ℚ := 15 / 100
def black_frequency : ℚ := 45 / 100

theorem white_balls_count :
  ∃ (white_balls : ℕ),
    white_balls = total_balls * (1 - red_frequency - black_frequency) := by
  sorry

end white_balls_count_l3161_316145


namespace problem_solution_l3161_316168

def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |x - a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 1 ↔ x ∈ Set.Ici (1/2)) ∧
  (Set.Iio 1 = {a : ℝ | ∀ x ≥ 0, f a x < 2}) := by
  sorry

end problem_solution_l3161_316168


namespace divisors_of_180_l3161_316109

theorem divisors_of_180 : Nat.card (Nat.divisors 180) = 18 := by
  sorry

end divisors_of_180_l3161_316109


namespace perpendicular_line_equation_l3161_316102

/-- Given a line L1 with equation x - 2y - 2 = 0 and a point P (5, 3),
    the line L2 passing through P and perpendicular to L1 has equation 2x + y - 13 = 0 -/
theorem perpendicular_line_equation (L1 : Set (ℝ × ℝ)) (P : ℝ × ℝ) :
  (L1 = {(x, y) | x - 2*y - 2 = 0}) →
  (P = (5, 3)) →
  (∃ L2 : Set (ℝ × ℝ), 
    (P ∈ L2) ∧ 
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w → 
      ∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q → 
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)) ∧
    (L2 = {(x, y) | 2*x + y - 13 = 0})) :=
by
  sorry

end perpendicular_line_equation_l3161_316102


namespace triangle_angle_calculation_l3161_316191

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C (in radians),
    prove that if b = 2, C = π/3, and c = √3, then B = π/2 -/
theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  b = 2 → C = π/3 → c = Real.sqrt 3 → B = π/2 := by
  sorry

end triangle_angle_calculation_l3161_316191


namespace divisibility_by_three_l3161_316182

theorem divisibility_by_three (B : Nat) : 
  B < 10 → (514 * 10 + B) % 3 = 0 ↔ B = 2 ∨ B = 5 ∨ B = 8 := by
  sorry

end divisibility_by_three_l3161_316182


namespace problem_1_problem_2_l3161_316178

/-- Given vectors in R^2 -/
def a : Fin 2 → ℝ := ![1, 0]
def b : Fin 2 → ℝ := ![2, 1]

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : Fin 2 → ℝ) : Prop :=
  ∃ t : ℝ, v = t • w ∨ w = t • v

theorem problem_1 (k : ℝ) :
  collinear (k • a - b) (a + 2 • b) ↔ k = -1/2 := by sorry

theorem problem_2 (m : ℝ) (A B C : Fin 2 → ℝ) :
  (B - A = 2 • a + 3 • b) →
  (C - B = a + m • b) →
  collinear (B - A) (C - B) →
  m = 3/2 := by sorry

end problem_1_problem_2_l3161_316178


namespace point_N_coordinates_l3161_316196

def M : ℝ × ℝ := (0, -1)

def N : ℝ × ℝ → Prop := fun p => p.1 - p.2 + 1 = 0

def perpendicular (p q r s : ℝ × ℝ) : Prop :=
  (q.1 - p.1) * (s.1 - r.1) + (q.2 - p.2) * (s.2 - r.2) = 0

def line_x_plus_2y_minus_3 (p : ℝ × ℝ) : Prop :=
  p.1 + 2 * p.2 - 3 = 0

theorem point_N_coordinates :
  ∃ n : ℝ × ℝ, N n ∧ perpendicular M n (0, 0) (1, -2) ∧ n = (2, 3) := by
  sorry

end point_N_coordinates_l3161_316196


namespace washing_machine_capacity_l3161_316172

theorem washing_machine_capacity 
  (shirts : ℕ) 
  (sweaters : ℕ) 
  (loads : ℕ) 
  (h1 : shirts = 43) 
  (h2 : sweaters = 2) 
  (h3 : loads = 9) : 
  (shirts + sweaters) / loads = 5 := by
  sorry

end washing_machine_capacity_l3161_316172


namespace bacteria_count_scientific_notation_l3161_316130

/-- The number of bacteria on a pair of unwashed hands -/
def bacteria_count : ℕ := 750000

/-- Scientific notation representation of the bacteria count -/
def scientific_notation : ℝ := 7.5 * (10 ^ 5)

theorem bacteria_count_scientific_notation : 
  (bacteria_count : ℝ) = scientific_notation := by
  sorry

end bacteria_count_scientific_notation_l3161_316130


namespace tan_product_equals_two_l3161_316156

theorem tan_product_equals_two :
  (∀ x y z : ℝ, x = 100 ∧ y = 35 ∧ z = 135 →
    Real.tan (z * π / 180) = -1 →
    (1 - Real.tan (x * π / 180)) * (1 - Real.tan (y * π / 180)) = 2) :=
by sorry

end tan_product_equals_two_l3161_316156


namespace arithmetic_mean_problem_l3161_316181

theorem arithmetic_mean_problem (a b c d : ℝ) 
  (h1 : (a + d) / 2 = 40)
  (h2 : (b + d) / 2 = 60)
  (h3 : (a + b) / 2 = 50)
  (h4 : (b + c) / 2 = 70) :
  c - a = 40 := by sorry

end arithmetic_mean_problem_l3161_316181


namespace white_white_overlapping_pairs_l3161_316124

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of overlapping pairs of each type when the figure is folded -/
structure OverlappingPairs where
  redRed : ℕ
  blueBlue : ℕ
  redWhite : ℕ

/-- The main theorem stating the number of white-white overlapping pairs -/
theorem white_white_overlapping_pairs
  (counts : TriangleCounts)
  (overlaps : OverlappingPairs)
  (h1 : counts.red = 4)
  (h2 : counts.blue = 6)
  (h3 : counts.white = 9)
  (h4 : overlaps.redRed = 3)
  (h5 : overlaps.blueBlue = 4)
  (h6 : overlaps.redWhite = 3) :
  counts.white - overlaps.redWhite = 6 :=
sorry

end white_white_overlapping_pairs_l3161_316124


namespace geometric_sequence_sum_l3161_316144

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence {a_n} where a_1 + a_3 = 1 and a_2 + a_4 = 2, 
    the sum of the 5th, 6th, 7th, and 8th terms equals 48. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
    (h_geometric : is_geometric_sequence a)
    (h_sum1 : a 1 + a 3 = 1)
    (h_sum2 : a 2 + a 4 = 2) :
    a 5 + a 6 + a 7 + a 8 = 48 := by
  sorry

end geometric_sequence_sum_l3161_316144


namespace unique_solution_l3161_316190

-- Define the functions f and p
def f (x : ℝ) : ℝ := |x + 1|

def p (x a : ℝ) : ℝ := |2*x + 5| + a

-- Define the set of values for 'a'
def A : Set ℝ := {-6.5, -5, 1.5}

-- State the theorem
theorem unique_solution (a : ℝ) :
  a ∈ A ↔ ∃! x : ℝ, x ≠ 1 ∧ x ≠ 2.5 ∧ f x = p x a :=
sorry

end unique_solution_l3161_316190


namespace cubic_minus_linear_at_five_l3161_316134

theorem cubic_minus_linear_at_five : 
  let x : ℝ := 5
  (x^3 - 3*x) = 110 := by sorry

end cubic_minus_linear_at_five_l3161_316134


namespace least_non_factor_non_prime_l3161_316159

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem least_non_factor_non_prime : 
  ∃ (n : ℕ), n > 0 ∧ ¬(factorial 30 % n = 0) ∧ ¬(is_prime n) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → (factorial 30 % m = 0) ∨ (is_prime m)) ∧ n = 961 := by
  sorry

end least_non_factor_non_prime_l3161_316159


namespace incorrect_statement_C_l3161_316170

theorem incorrect_statement_C : 
  (∀ x : ℝ, x > 0 → (∃ y : ℝ, y > 0 ∧ y^2 = x) ∧ (∃! y : ℝ, y > 0 ∧ y^2 = x)) ∧ 
  (∀ x : ℝ, x^3 < 0 → ∃ y : ℝ, y < 0 ∧ y^3 = x) ∧
  (∃ x : ℝ, x^(1/3) = x^(1/2)) ∧
  ¬(∃ y : ℝ, y ≠ 0 ∧ (y^2 = 81 → (y = 9 ∨ y = -9))) :=
by sorry

end incorrect_statement_C_l3161_316170


namespace fraction_problem_l3161_316175

theorem fraction_problem (numerator denominator : ℤ) (x : ℤ) : 
  denominator = numerator - 4 →
  denominator = 5 →
  numerator + x = 3 * denominator →
  x = 6 := by
sorry

end fraction_problem_l3161_316175


namespace repair_cost_calculation_l3161_316146

def purchase_price : ℕ := 12000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 27000
def profit_percentage : ℚ := 1/2

theorem repair_cost_calculation :
  ∃ (repair_cost : ℕ),
    selling_price = (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) ∧
    repair_cost = 5000 := by
  sorry

end repair_cost_calculation_l3161_316146


namespace trig_identity_l3161_316184

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_identity_l3161_316184


namespace town_population_l3161_316123

theorem town_population (growth_rate : ℝ) (future_population : ℕ) (present_population : ℕ) :
  growth_rate = 0.1 →
  future_population = 264 →
  present_population * (1 + growth_rate) = future_population →
  present_population = 240 := by
sorry

end town_population_l3161_316123


namespace at_least_one_true_l3161_316100

theorem at_least_one_true (p q : Prop) : ¬(¬(p ∨ q)) → (p ∨ q) := by sorry

end at_least_one_true_l3161_316100


namespace sixth_root_of_unity_product_l3161_316195

theorem sixth_root_of_unity_product (r : ℂ) (h1 : r^6 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 5 := by
  sorry

end sixth_root_of_unity_product_l3161_316195


namespace square_with_arcs_area_l3161_316125

/-- The area of the regions inside a square with side length 3 cm, 
    but outside two quarter-circle arcs from adjacent corners. -/
theorem square_with_arcs_area : 
  let square_side : ℝ := 3
  let quarter_circle_area := (π * square_side^2) / 4
  let triangle_area := (square_side^2) / 2
  let arc_area := 2 * (quarter_circle_area - triangle_area)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |arc_area - 5.1| < ε :=
sorry

end square_with_arcs_area_l3161_316125


namespace no_solution_for_a_l3161_316180

theorem no_solution_for_a (x a : ℝ) : x = 4 → 1 / (x + a) + 1 / (x - a) ≠ 1 / (x - a) := by
  sorry

end no_solution_for_a_l3161_316180


namespace least_pencils_l3161_316126

theorem least_pencils (p : ℕ) : p > 0 ∧ 
  p % 5 = 4 ∧ 
  p % 6 = 3 ∧ 
  p % 8 = 5 ∧ 
  (∀ q : ℕ, q > 0 ∧ q % 5 = 4 ∧ q % 6 = 3 ∧ q % 8 = 5 → p ≤ q) → 
  p = 69 := by
sorry

end least_pencils_l3161_316126


namespace adult_ticket_cost_is_correct_l3161_316158

/-- The cost of an adult ticket in dollars -/
def adult_ticket_cost : ℕ := 19

/-- The cost of a child ticket in dollars -/
def child_ticket_cost : ℕ := adult_ticket_cost - 6

/-- The total number of tickets -/
def total_tickets : ℕ := 5

/-- The number of adult tickets -/
def adult_tickets : ℕ := 2

/-- The number of child tickets -/
def child_tickets : ℕ := 3

/-- The total cost of all tickets in dollars -/
def total_cost : ℕ := 77

theorem adult_ticket_cost_is_correct : 
  adult_tickets * adult_ticket_cost + child_tickets * child_ticket_cost = total_cost :=
by sorry

end adult_ticket_cost_is_correct_l3161_316158


namespace set_a_contains_one_l3161_316153

theorem set_a_contains_one (a : ℝ) : 
  let A : Set ℝ := {a, a^2}
  1 ∈ A → a = -1 := by sorry

end set_a_contains_one_l3161_316153


namespace equation_solution_l3161_316194

theorem equation_solution : ∃ x : ℝ, 
  x = 625 ∧ 
  Real.sqrt (3 + Real.sqrt (4 + Real.sqrt x)) = (2 + Real.sqrt x) ^ (1/3 : ℝ) :=
by sorry

end equation_solution_l3161_316194


namespace rectangular_prism_inequality_l3161_316188

theorem rectangular_prism_inequality (a b c l : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ l > 0) 
  (h_diagonal : a^2 + b^2 + c^2 = l^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := by
  sorry

end rectangular_prism_inequality_l3161_316188


namespace weight_of_six_moles_of_compound_l3161_316147

/-- The weight of a given number of moles of a compound -/
def weight (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Proof that the weight of 6 moles of a compound with molecular weight 1404 is 8424 -/
theorem weight_of_six_moles_of_compound (molecular_weight : ℝ) 
  (h : molecular_weight = 1404) : weight 6 molecular_weight = 8424 := by
  sorry

end weight_of_six_moles_of_compound_l3161_316147


namespace remainder_5462_div_9_l3161_316114

theorem remainder_5462_div_9 : 5462 % 9 = 8 := by sorry

end remainder_5462_div_9_l3161_316114


namespace tax_to_savings_ratio_l3161_316186

/-- Esperanza's monthly finances -/
def monthly_finances (rent food mortgage savings tax gross_salary : ℚ) : Prop :=
  rent = 600 ∧
  food = (3/5) * rent ∧
  mortgage = 3 * food ∧
  savings = 2000 ∧
  gross_salary = 4840 ∧
  tax = gross_salary - (rent + food + mortgage + savings)

/-- The ratio of tax to savings is 2:5 -/
theorem tax_to_savings_ratio 
  (rent food mortgage savings tax gross_salary : ℚ) 
  (h : monthly_finances rent food mortgage savings tax gross_salary) : 
  tax / savings = 2/5 := by
  sorry

end tax_to_savings_ratio_l3161_316186


namespace gain_percent_calculation_l3161_316167

theorem gain_percent_calculation (cost_price selling_price : ℝ) :
  50 * cost_price = 45 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = 11.11 :=
by sorry

end gain_percent_calculation_l3161_316167


namespace pierre_ice_cream_scoops_l3161_316132

/-- 
Given:
- The cost of each scoop of ice cream
- The number of scoops Pierre's mom gets
- The total bill amount
Prove that Pierre gets 3 scoops of ice cream
-/
theorem pierre_ice_cream_scoops 
  (cost_per_scoop : ℕ) 
  (mom_scoops : ℕ) 
  (total_bill : ℕ) 
  (h1 : cost_per_scoop = 2)
  (h2 : mom_scoops = 4)
  (h3 : total_bill = 14) :
  ∃ (pierre_scoops : ℕ), 
    pierre_scoops = 3 ∧ 
    cost_per_scoop * (pierre_scoops + mom_scoops) = total_bill :=
by sorry

end pierre_ice_cream_scoops_l3161_316132


namespace ellipse_a_plus_k_eq_eight_l3161_316173

/-- An ellipse with given properties -/
structure Ellipse where
  -- Foci coordinates
  f1 : ℝ × ℝ := (1, 1)
  f2 : ℝ × ℝ := (1, 5)
  -- Point on the ellipse
  p : ℝ × ℝ := (-4, 3)
  -- Constants in the equation (x-h)^2/a^2 + (y-k)^2/b^2 = 1
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ
  -- Ensure a and b are positive
  ha : a > 0
  hb : b > 0
  -- Ensure the point p satisfies the equation
  heq : (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1

/-- Theorem stating that a + k = 8 for the given ellipse -/
theorem ellipse_a_plus_k_eq_eight (e : Ellipse) : e.a + e.k = 8 := by
  sorry

end ellipse_a_plus_k_eq_eight_l3161_316173


namespace sqrt_50_between_consecutive_integers_product_l3161_316163

theorem sqrt_50_between_consecutive_integers_product (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ b = a + 1 ∧ (a : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (b : ℝ) → a * b = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_product_l3161_316163


namespace sequence_periodicity_implies_zero_l3161_316174

theorem sequence_periodicity_implies_zero (a b c d : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) = a n + b n)
  (h2 : ∀ n, b (n + 1) = b n + c n)
  (h3 : ∀ n, c (n + 1) = c n + d n)
  (h4 : ∀ n, d (n + 1) = d n + a n)
  (h5 : ∃ k m : ℕ, k ≥ 1 ∧ m ≥ 1 ∧ 
    a (k + m) = a m ∧ 
    b (k + m) = b m ∧ 
    c (k + m) = c m ∧ 
    d (k + m) = d m) :
  a 2 = 0 ∧ b 2 = 0 ∧ c 2 = 0 ∧ d 2 = 0 := by
sorry

end sequence_periodicity_implies_zero_l3161_316174


namespace fractional_equation_solution_l3161_316117

theorem fractional_equation_solution : 
  ∃ x : ℝ, (1 / (x - 5) = 1) ∧ (x = 6) := by sorry

end fractional_equation_solution_l3161_316117


namespace manganese_percentage_after_iron_addition_l3161_316189

theorem manganese_percentage_after_iron_addition 
  (initial_mixture_mass : ℝ)
  (initial_manganese_percentage : ℝ)
  (added_iron_mass : ℝ)
  (h1 : initial_mixture_mass = 1)
  (h2 : initial_manganese_percentage = 20)
  (h3 : added_iron_mass = 1)
  : (initial_manganese_percentage / 100 * initial_mixture_mass) / 
    (initial_mixture_mass + added_iron_mass) * 100 = 10 := by
  sorry

#check manganese_percentage_after_iron_addition

end manganese_percentage_after_iron_addition_l3161_316189


namespace bridge_length_calculation_l3161_316118

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : 
  train_length = 360 ∧ 
  train_speed_kmh = 90 ∧ 
  time_to_pass = 20 → 
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 := by
  sorry

end bridge_length_calculation_l3161_316118


namespace binomial_7_2_l3161_316193

theorem binomial_7_2 : (7 : ℕ).choose 2 = 21 := by
  sorry

end binomial_7_2_l3161_316193


namespace largest_prime_divisor_xyxyxy_l3161_316135

/-- The largest prime divisor of a number in the form xyxyxy -/
theorem largest_prime_divisor_xyxyxy (x y : ℕ) (hx : x < 10) (hy : y < 10) :
  ∃ (p : ℕ), p.Prime ∧ p ∣ (100000 * x + 10000 * y + 1000 * x + 100 * y + 10 * x + y) ∧
  ∀ (q : ℕ), q.Prime → q ∣ (100000 * x + 10000 * y + 1000 * x + 100 * y + 10 * x + y) → q ≤ 97 :=
by sorry

end largest_prime_divisor_xyxyxy_l3161_316135


namespace tripodasaurus_count_l3161_316131

/-- A tripodasaurus is a creature with 3 legs and 1 head -/
structure Tripodasaurus where
  legs : Nat
  head : Nat

/-- A flock of tripodasauruses -/
structure Flock where
  count : Nat

/-- The total number of heads and legs in a flock -/
def totalHeadsAndLegs (f : Flock) : Nat :=
  f.count * (3 + 1)  -- 3 legs + 1 head per tripodasaurus

theorem tripodasaurus_count (f : Flock) :
  totalHeadsAndLegs f = 20 → f.count = 5 := by
  sorry

end tripodasaurus_count_l3161_316131


namespace union_complement_problem_l3161_316140

open Set

theorem union_complement_problem (A B : Set ℝ) 
  (hA : A = {x : ℝ | -2 ≤ x ∧ x ≤ 3})
  (hB : B = {x : ℝ | x < -1 ∨ 4 < x}) :
  A ∪ (univ \ B) = {x : ℝ | -2 ≤ x ∧ x ≤ 4} := by
  sorry

end union_complement_problem_l3161_316140


namespace zero_product_theorem_l3161_316112

theorem zero_product_theorem (x₁ x₂ x₃ x₄ : ℝ) 
  (sum_condition : x₁ + x₂ + x₃ + x₄ = 0)
  (power_sum_condition : x₁^7 + x₂^7 + x₃^7 + x₄^7 = 0) :
  x₄ * (x₄ + x₁) * (x₄ + x₂) * (x₄ + x₃) = 0 := by
sorry

end zero_product_theorem_l3161_316112


namespace alex_coin_distribution_l3161_316177

/-- The minimum number of additional coins needed for distribution --/
def min_additional_coins (n : ℕ) (initial_coins : ℕ) : ℕ :=
  (n * (n + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex's distribution --/
theorem alex_coin_distribution :
  min_additional_coins 20 192 = 18 := by
  sorry

end alex_coin_distribution_l3161_316177


namespace books_movies_difference_l3161_316129

theorem books_movies_difference : 
  ∀ (total_books total_movies : ℕ),
    total_books = 10 →
    total_movies = 6 →
    total_books - total_movies = 4 :=
by
  sorry

end books_movies_difference_l3161_316129


namespace symmetric_lines_l3161_316154

/-- Given two lines symmetric about y = x, if one line is y = 2x - 3, 
    then the other line is y = (1/2)x + (3/2) -/
theorem symmetric_lines (x y : ℝ) : 
  (y = 2 * x - 3) ↔ 
  (∃ (x' y' : ℝ), y' = (1/2) * x' + (3/2) ∧ 
    (x + x') / 2 = (y + y') / 2 ∧
    y = x) :=
by sorry

end symmetric_lines_l3161_316154


namespace system_solutions_system_no_solutions_l3161_316121

-- Define the system of equations
def system (x y k : ℝ) : Prop :=
  3 * x - 4 * y = 9 ∧ 6 * x - 8 * y = k

-- Theorem statement
theorem system_solutions (k : ℝ) :
  (∃ x y, system x y k) ↔ k = 18 :=
by sorry

-- Corollary for no solutions
theorem system_no_solutions (k : ℝ) :
  (¬ ∃ x y, system x y k) ↔ k ≠ 18 :=
by sorry

end system_solutions_system_no_solutions_l3161_316121


namespace symmetry_across_y_eq_neg_x_l3161_316192

/-- Given two lines in the xy-plane, this function checks if they are symmetrical across y = -x -/
def are_symmetrical_lines (line1 line2 : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line1 x y ↔ line2 y x

/-- The original line: √3x + y + 1 = 0 -/
def original_line (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y + 1 = 0

/-- The proposed symmetrical line: x + √3y - 1 = 0 -/
def symmetrical_line (x y : ℝ) : Prop :=
  x + Real.sqrt 3 * y - 1 = 0

/-- Theorem stating that the symmetrical_line is indeed symmetrical to the original_line across y = -x -/
theorem symmetry_across_y_eq_neg_x :
  are_symmetrical_lines original_line symmetrical_line :=
sorry

end symmetry_across_y_eq_neg_x_l3161_316192


namespace pascal_triangle_15th_row_5th_number_l3161_316155

theorem pascal_triangle_15th_row_5th_number :
  Nat.choose 15 4 = 1365 := by sorry

end pascal_triangle_15th_row_5th_number_l3161_316155


namespace chef_fries_problem_l3161_316162

/-- Given a chef making fries, prove the number of fries needed. -/
theorem chef_fries_problem (fries_per_potato : ℕ) (total_potatoes : ℕ) (leftover_potatoes : ℕ) :
  fries_per_potato = 25 →
  total_potatoes = 15 →
  leftover_potatoes = 7 →
  fries_per_potato * (total_potatoes - leftover_potatoes) = 200 := by
  sorry

end chef_fries_problem_l3161_316162


namespace probability_A_and_B_selected_l3161_316105

/-- The number of students in the group -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B -/
def prob_select_A_and_B : ℚ := 3 / 10

/-- Theorem stating that the probability of selecting both A and B
    when randomly choosing 3 students from a group of 5 students is 3/10 -/
theorem probability_A_and_B_selected :
  (Nat.choose (total_students - 2) (selected_students - 2) : ℚ) /
  (Nat.choose total_students selected_students : ℚ) = prob_select_A_and_B :=
sorry

end probability_A_and_B_selected_l3161_316105


namespace total_floor_area_square_slabs_l3161_316104

/-- Calculates the total floor area covered by square stone slabs. -/
theorem total_floor_area_square_slabs 
  (num_slabs : ℕ) 
  (slab_length : ℝ) 
  (h1 : num_slabs = 30)
  (h2 : slab_length = 200)
  : (num_slabs * (slab_length / 100)^2 : ℝ) = 120 := by
  sorry

#check total_floor_area_square_slabs

end total_floor_area_square_slabs_l3161_316104


namespace johns_hourly_rate_is_10_l3161_316120

/-- Calculates John's hourly rate when earning the performance bonus -/
def johnsHourlyRateWithBonus (basePay dayHours bonusPay bonusHours : ℚ) : ℚ :=
  (basePay + bonusPay) / (dayHours + bonusHours)

/-- Theorem: John's hourly rate with bonus is $10 per hour -/
theorem johns_hourly_rate_is_10 :
  johnsHourlyRateWithBonus 80 8 20 2 = 10 := by
  sorry

end johns_hourly_rate_is_10_l3161_316120


namespace N_subset_M_l3161_316176

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = -(p.1^2)}

-- State the theorem
theorem N_subset_M : N ⊆ M := by sorry

end N_subset_M_l3161_316176


namespace remainder_difference_l3161_316165

theorem remainder_difference (d : ℕ) (r : ℕ) (h1 : d > 1) : 
  (1059 % d = r ∧ 1417 % d = r ∧ 2312 % d = r) → d - r = 15 := by
  sorry

end remainder_difference_l3161_316165


namespace cos_two_pi_third_plus_two_alpha_l3161_316142

theorem cos_two_pi_third_plus_two_alpha (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end cos_two_pi_third_plus_two_alpha_l3161_316142


namespace T_eight_three_l3161_316164

def T (a b : ℤ) : ℤ := 4*a + 5*b - 1

theorem T_eight_three : T 8 3 = 46 := by sorry

end T_eight_three_l3161_316164


namespace tangent_line_slope_l3161_316179

/-- Given that the line y = kx is tangent to the curve y = x + exp(-x), prove that k = 1 - exp(1) -/
theorem tangent_line_slope (k : ℝ) : 
  (∃ x₀ : ℝ, k * x₀ = x₀ + Real.exp (-x₀) ∧ 
              k = 1 - Real.exp (-x₀)) → 
  k = 1 - Real.exp 1 := by
  sorry

end tangent_line_slope_l3161_316179


namespace quadratic_equation_problem_l3161_316161

theorem quadratic_equation_problem (a : ℝ) : 2 * (5 - a) * (6 + a) = 100 → a^2 + a + 1 = -19 := by
  sorry

end quadratic_equation_problem_l3161_316161


namespace opposite_reciprocal_calc_l3161_316151

theorem opposite_reciprocal_calc 
  (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 3) 
  (h4 : m < 0) : 
  m^3 + c*d + (a+b)/m = -26 := by
sorry

end opposite_reciprocal_calc_l3161_316151


namespace perimeter_of_square_d_l3161_316108

/-- Given two squares C and D, where C has a side length of 10 cm and D has an area
    that is half the area of C, the perimeter of D is 20√2 cm. -/
theorem perimeter_of_square_d (c d : Real) : 
  c = 10 →  -- side length of square C
  d ^ 2 = (c ^ 2) / 2 →  -- area of D is half the area of C
  4 * d = 20 * Real.sqrt 2 := by
sorry

end perimeter_of_square_d_l3161_316108


namespace unique_solution_l3161_316101

/-- Definition of the diamond operation -/
def diamond (a b c d : ℝ) : ℝ × ℝ :=
  (a * c - b * d, a * d + b * c)

/-- Theorem stating the unique solution to the equation -/
theorem unique_solution :
  ∀ x y : ℝ, diamond x 3 x y = (6, 0) ↔ x = 0 ∧ y = -2 := by
  sorry

end unique_solution_l3161_316101


namespace diamond_solution_l3161_316127

def diamond (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem diamond_solution (a : ℝ) : diamond a 3 = 6 → a = 8 := by
  sorry

end diamond_solution_l3161_316127


namespace min_waves_to_21_l3161_316111

/-- Represents the direction of a wand wave -/
inductive WaveDirection
  | Up
  | Down

/-- Calculates the number of open flowers after a single wave -/
def wave (n : ℕ) (d : WaveDirection) : ℕ :=
  match d with
  | WaveDirection.Up => if n > 0 then n - 1 else 0
  | WaveDirection.Down => 2 * n

/-- Calculates the number of open flowers after a sequence of waves -/
def waveSequence (initial : ℕ) (waves : List WaveDirection) : ℕ :=
  waves.foldl wave initial

/-- Checks if a sequence of waves results in the target number of flowers -/
def isValidSequence (initial target : ℕ) (waves : List WaveDirection) : Prop :=
  waveSequence initial waves = target

/-- Theorem: The minimum number of waves to reach 21 flowers from 3 flowers is 6 -/
theorem min_waves_to_21 :
  ∃ (waves : List WaveDirection),
    waves.length = 6 ∧
    isValidSequence 3 21 waves ∧
    ∀ (other : List WaveDirection),
      isValidSequence 3 21 other → waves.length ≤ other.length :=
by sorry

end min_waves_to_21_l3161_316111


namespace inequality_holds_l3161_316166

theorem inequality_holds (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end inequality_holds_l3161_316166


namespace coefficient_is_40_l3161_316197

/-- The coefficient of x^3y^2 in the expansion of (x-2y)^5 -/
def coefficient : ℤ := 
  (Nat.choose 5 2) * (-2)^2

/-- Theorem stating that the coefficient of x^3y^2 in (x-2y)^5 is 40 -/
theorem coefficient_is_40 : coefficient = 40 := by
  sorry

end coefficient_is_40_l3161_316197


namespace solution_set_of_inequality_l3161_316103

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_f_0 : f 0 = -2)
  (h_f_3 : f 3 = 2) :
  {x : ℝ | |f (x + 1)| ≥ 2} = {x : ℝ | x ≤ -1 ∨ x ≥ 2} :=
sorry

end solution_set_of_inequality_l3161_316103


namespace arithmetic_geometric_sequence_ratio_l3161_316169

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : ∀ n, a (n + 1) = a n + d) 
  (h3 : (a 5 - a 1) * (a 17 - a 5) = (a 5 - a 1)^2) : 
  a 5 / a 1 = 9 := by
sorry

end arithmetic_geometric_sequence_ratio_l3161_316169


namespace ratio_c_to_d_l3161_316138

theorem ratio_c_to_d (a b c d : ℝ) 
  (hab : a / b = 3 / 4)
  (hbc : b / c = 7 / 9)
  (had : a / d = 0.4166666666666667) :
  c / d = 5 / 7 := by
  sorry

end ratio_c_to_d_l3161_316138


namespace complex_power_simplification_l3161_316119

theorem complex_power_simplification :
  (Complex.exp (Complex.I * (123 * π / 180)))^25 = 
  -Complex.cos (15 * π / 180) - Complex.I * Complex.sin (15 * π / 180) := by
  sorry

end complex_power_simplification_l3161_316119


namespace triangle_similarity_l3161_316113

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

-- Define the perpendicular foot point
def perp_foot (P : Point) (line : Point × Point) : Point := sorry

-- Define the similarity relation between triangles
def similar (T1 T2 : Triangle) : Prop := sorry

-- Define the construction process
def construct_next_triangle (T : Triangle) (P : Point) : Triangle :=
  let B1 := perp_foot P (T.B, T.C)
  let B2 := perp_foot P (T.C, T.A)
  let B3 := perp_foot P (T.A, T.B)
  Triangle.mk B1 B2 B3

-- Theorem statement
theorem triangle_similarity 
  (A : Triangle) 
  (P : Point) 
  (h_interior : sorry) -- Assumption that P is interior to A
  : 
  let B := construct_next_triangle A P
  let C := construct_next_triangle B P
  let D := construct_next_triangle C P
  similar A D := by sorry

end triangle_similarity_l3161_316113


namespace seventh_term_value_l3161_316187

/-- The general term of the series at position n -/
def seriesTerm (n : ℕ) (a : ℝ) : ℝ := (-2)^n * a^(2*n - 1)

/-- The 7th term of the series -/
def seventhTerm (a : ℝ) : ℝ := seriesTerm 7 a

theorem seventh_term_value (a : ℝ) : seventhTerm a = -128 * a^13 := by sorry

end seventh_term_value_l3161_316187


namespace complex_sum_pure_imaginary_l3161_316136

theorem complex_sum_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (z₁ + z₂).re = 0 → a = -3 := by
sorry

end complex_sum_pure_imaginary_l3161_316136
