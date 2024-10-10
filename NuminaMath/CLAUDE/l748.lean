import Mathlib

namespace mass_calculation_l748_74884

/-- Given a concentration and volume, calculate the mass -/
def calculate_mass (C : ℝ) (V : ℝ) : ℝ := C * V

/-- Theorem stating that for given concentration and volume, the mass is 32 mg -/
theorem mass_calculation (C V : ℝ) (hC : C = 4) (hV : V = 8) :
  calculate_mass C V = 32 := by
  sorry

end mass_calculation_l748_74884


namespace three_fourths_of_forty_l748_74899

theorem three_fourths_of_forty : (3 / 4 : ℚ) * 40 = 30 := by
  sorry

end three_fourths_of_forty_l748_74899


namespace sqrt3_expression_equals_zero_l748_74862

theorem sqrt3_expression_equals_zero :
  Real.sqrt 3 * (1 - Real.sqrt 3) - |-(Real.sqrt 3)| + (27 : ℝ) ^ (1/3) = 0 := by
  sorry

end sqrt3_expression_equals_zero_l748_74862


namespace students_not_taking_languages_l748_74889

theorem students_not_taking_languages (total : ℕ) (french : ℕ) (spanish : ℕ) (both : ℕ) 
  (h1 : total = 28) 
  (h2 : french = 5) 
  (h3 : spanish = 10) 
  (h4 : both = 4) : 
  total - (french + spanish + both) = 9 := by
  sorry

end students_not_taking_languages_l748_74889


namespace smallest_valid_number_l748_74888

def is_valid (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 5 * k ∧ n % 3 = 1

theorem smallest_valid_number : (∀ m : ℕ, m > 0 ∧ m < 10 → ¬(is_valid m)) ∧ is_valid 10 := by
  sorry

end smallest_valid_number_l748_74888


namespace platform_length_l748_74879

/-- Given a train of length 300 meters that crosses a platform in 39 seconds
    and a signal pole in 18 seconds, the length of the platform is 350 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 39 →
  pole_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length / pole_time) * platform_time = train_length + platform_length :=
by sorry

end platform_length_l748_74879


namespace store_optimal_pricing_l748_74843

/-- Represents the store's product information and pricing strategy. -/
structure Store where
  purchase_price_A : ℝ
  purchase_price_B : ℝ
  retail_price_A : ℝ
  retail_price_B : ℝ
  daily_sales : ℝ
  price_decrease : ℝ

/-- Conditions for the store's pricing and sales. -/
def store_conditions (s : Store) : Prop :=
  s.purchase_price_A + s.purchase_price_B = 3 ∧
  s.retail_price_A = s.purchase_price_A + 1 ∧
  s.retail_price_B = 2 * s.purchase_price_B - 1 ∧
  3 * s.retail_price_A + 2 * s.retail_price_B = 12 ∧
  s.daily_sales = 500 ∧
  s.price_decrease > 0

/-- The profit function for the store. -/
def profit (s : Store) : ℝ :=
  (s.retail_price_A - s.price_decrease) * (s.daily_sales + 1000 * s.price_decrease) + s.retail_price_B * s.daily_sales - (s.purchase_price_A + s.purchase_price_B) * s.daily_sales

/-- Theorem stating the correct retail prices and optimal price decrease for maximum profit. -/
theorem store_optimal_pricing (s : Store) (h : store_conditions s) :
  s.retail_price_A = 2 ∧ s.retail_price_B = 3 ∧ profit s = 1000 ↔ s.price_decrease = 0.5 := by
  sorry


end store_optimal_pricing_l748_74843


namespace tan_2018pi_minus_alpha_l748_74864

theorem tan_2018pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo 0 (3 * π / 2)) 
  (h2 : Real.cos (3 * π / 2 - α) = Real.sqrt 3 / 2) : 
  Real.tan (2018 * π - α) = -Real.sqrt 3 := by
  sorry

end tan_2018pi_minus_alpha_l748_74864


namespace sum_equals_12x_l748_74863

theorem sum_equals_12x (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 3 * y - x) : 
  x + y + z = 12 * x := by
  sorry

end sum_equals_12x_l748_74863


namespace inequality_solution_l748_74844

theorem inequality_solution (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x * (x + 1) + 1) :
  {x : ℝ | f x < 0} = {x : ℝ | x < 1/a ∨ x > 1} ∩ {x : ℝ | a ≠ 0} :=
by sorry

end inequality_solution_l748_74844


namespace jewels_gain_is_3_25_l748_74854

/-- Calculates Jewel's total gain from selling magazines --/
def jewels_gain (num_magazines : ℕ) 
                (cost_per_magazine : ℚ) 
                (regular_price : ℚ) 
                (discount_percent : ℚ) 
                (num_regular_price : ℕ) : ℚ :=
  let total_cost := num_magazines * cost_per_magazine
  let revenue_regular := num_regular_price * regular_price
  let discounted_price := regular_price * (1 - discount_percent)
  let revenue_discounted := (num_magazines - num_regular_price) * discounted_price
  let total_revenue := revenue_regular + revenue_discounted
  total_revenue - total_cost

theorem jewels_gain_is_3_25 : 
  jewels_gain 10 3 (7/2) (1/10) 5 = 13/4 := by
  sorry

end jewels_gain_is_3_25_l748_74854


namespace wario_field_goals_l748_74851

theorem wario_field_goals 
  (missed_fraction : ℚ)
  (wide_right_fraction : ℚ)
  (wide_right_misses : ℕ)
  (h1 : missed_fraction = 1 / 4)
  (h2 : wide_right_fraction = 1 / 5)
  (h3 : wide_right_misses = 3) :
  ∃ (total_attempts : ℕ), 
    (↑wide_right_misses : ℚ) / wide_right_fraction / missed_fraction = total_attempts ∧ 
    total_attempts = 60 := by
sorry


end wario_field_goals_l748_74851


namespace sum_of_x_coordinates_l748_74869

/-- Triangle XYZ -/
structure TriangleXYZ where
  X : ℝ × ℝ
  Y : ℝ × ℝ := (0, 0)
  Z : ℝ × ℝ := (150, 0)
  area : ℝ := 1200

/-- Triangle XWV -/
structure TriangleXWV where
  X : ℝ × ℝ
  W : ℝ × ℝ := (500, 300)
  V : ℝ × ℝ := (510, 290)
  area : ℝ := 3600

/-- The theorem stating that the sum of all possible x-coordinates of X is 3200 -/
theorem sum_of_x_coordinates (triangle_xyz : TriangleXYZ) (triangle_xwv : TriangleXWV) 
  (h : triangle_xyz.X = triangle_xwv.X) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), (x₁ + x₂ + x₃ + x₄ = 3200 ∧ 
    (triangle_xyz.X.1 = x₁ ∨ triangle_xyz.X.1 = x₂ ∨ triangle_xyz.X.1 = x₃ ∨ triangle_xyz.X.1 = x₄)) :=
by sorry

end sum_of_x_coordinates_l748_74869


namespace jenny_recycling_problem_l748_74826

/-- The weight of each can in ounces -/
def can_weight : ℚ := 2

theorem jenny_recycling_problem :
  let total_weight : ℚ := 100
  let bottle_weight : ℚ := 6
  let num_cans : ℚ := 20
  let cents_per_bottle : ℚ := 10
  let cents_per_can : ℚ := 3
  let total_cents : ℚ := 160
  (total_weight - num_cans * can_weight) / bottle_weight * cents_per_bottle + num_cans * cents_per_can = total_cents :=
by sorry

end jenny_recycling_problem_l748_74826


namespace complex_root_modulus_one_l748_74876

theorem complex_root_modulus_one (n : ℕ) :
  (∃ z : ℂ, z^(n+1) - z^n - 1 = 0 ∧ Complex.abs z = 1) ↔ (∃ k : ℤ, n + 2 = 6 * k) :=
sorry

end complex_root_modulus_one_l748_74876


namespace tetrahedron_edge_assignment_l748_74816

/-- Represents a tetrahedron with face areas -/
structure Tetrahedron where
  s : ℝ  -- smallest face area
  S : ℝ  -- largest face area
  a : ℝ  -- area of another face
  b : ℝ  -- area of the fourth face
  h_s_smallest : s ≤ a ∧ s ≤ b ∧ s ≤ S
  h_S_largest : S ≥ a ∧ S ≥ b ∧ S ≥ s
  h_positive : s > 0 ∧ S > 0 ∧ a > 0 ∧ b > 0

/-- Represents the edge values of a tetrahedron -/
structure TetrahedronEdges where
  e1 : ℝ  -- edge common to smallest and largest face
  e2 : ℝ  -- edge of smallest face
  e3 : ℝ  -- edge of smallest face
  e4 : ℝ  -- edge of largest face
  e5 : ℝ  -- edge of largest face
  e6 : ℝ  -- remaining edge

/-- Checks if the edge values satisfy the face area conditions -/
def satisfies_conditions (t : Tetrahedron) (e : TetrahedronEdges) : Prop :=
  e.e1 ≥ 0 ∧ e.e2 ≥ 0 ∧ e.e3 ≥ 0 ∧ e.e4 ≥ 0 ∧ e.e5 ≥ 0 ∧ e.e6 ≥ 0 ∧
  e.e1 + e.e2 + e.e3 = t.s ∧
  e.e1 + e.e4 + e.e5 = t.S ∧
  e.e2 + e.e5 + e.e6 = t.a ∧
  e.e3 + e.e4 + e.e6 = t.b

theorem tetrahedron_edge_assignment (t : Tetrahedron) :
  ∃ e : TetrahedronEdges, satisfies_conditions t e := by
  sorry

end tetrahedron_edge_assignment_l748_74816


namespace sqrt_sum_equals_nine_sqrt_three_l748_74846

theorem sqrt_sum_equals_nine_sqrt_three : 
  Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 48 = 9 * Real.sqrt 3 := by
  sorry

end sqrt_sum_equals_nine_sqrt_three_l748_74846


namespace solve_for_y_l748_74850

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 := by
  sorry

end solve_for_y_l748_74850


namespace tan_sum_eq_two_l748_74800

theorem tan_sum_eq_two (x y : Real) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 2 := by
  sorry

end tan_sum_eq_two_l748_74800


namespace locus_and_line_theorem_l748_74894

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (3, 0)

-- Define the condition for P being outside C
def outside_circle (x y : ℝ) : Prop := x^2 + (y - 2)^2 > 1

-- Define the tangent condition (implicitly used in the problem)
def is_tangent (x y : ℝ) : Prop := 
  ∃ (t : ℝ), circle_C (x + t) (y + t) ∧ ¬(∃ (s : ℝ), s ≠ t ∧ circle_C (x + s) (y + s))

-- Define the PQ = √2 * PA condition
def pq_pa_relation (x y : ℝ) : Prop :=
  (x^2 + (y - 2)^2 - 1) = 2 * ((x - 3)^2 + y^2)

-- Define the locus of P
def locus_P (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 4*y + 15 = 0

-- Define the condition for line l
def line_l (x y : ℝ) : Prop := x = 3 ∨ 5*x - 12*y - 15 = 0

-- Main theorem
theorem locus_and_line_theorem :
  ∀ (x y : ℝ),
    outside_circle x y →
    is_tangent x y →
    pq_pa_relation x y →
    (locus_P x y ∧
     (∃ (m n : ℝ × ℝ),
       locus_P m.1 m.2 ∧
       locus_P n.1 n.2 ∧
       line_l m.1 m.2 ∧
       line_l n.1 n.2 ∧
       (m.1 - n.1)^2 + (m.2 - n.2)^2 = 64)) :=
by sorry

end locus_and_line_theorem_l748_74894


namespace fiftieth_term_is_247_l748_74847

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 50th term of the arithmetic sequence starting with 2 and common difference 5 is 247 -/
theorem fiftieth_term_is_247 : arithmetic_sequence 2 5 50 = 247 := by
  sorry

end fiftieth_term_is_247_l748_74847


namespace max_value_theorem_l748_74803

theorem max_value_theorem (a b c d e : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0)
  (sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ac + 3*b*c + 4*c*d + 6*c*e ≤ 252 * Real.sqrt 62 ∧
  (a = 2 ∧ b = 6 ∧ c = 6 * Real.sqrt 7 ∧ d = 8 ∧ e = 12) →
  ac + 3*b*c + 4*c*d + 6*c*e = 252 * Real.sqrt 62 := by
sorry

end max_value_theorem_l748_74803


namespace solve_for_d_l748_74827

theorem solve_for_d (n k c d : ℝ) (h : n = (2 * k * c * d) / (c + d)) (h_nonzero : 2 * k * c ≠ n) :
  d = (n * c) / (2 * k * c - n) := by
  sorry

end solve_for_d_l748_74827


namespace min_value_trig_expression_l748_74898

open Real

theorem min_value_trig_expression (x : ℝ) (h : 0 < x ∧ x < π / 2) :
  (sin x + 3 * (1 / sin x))^2 + (cos x + 3 * (1 / cos x))^2 ≥ 52 ∧
  ∃ y, 0 < y ∧ y < π / 2 ∧ (sin y + 3 * (1 / sin y))^2 + (cos y + 3 * (1 / cos y))^2 = 52 :=
by sorry

end min_value_trig_expression_l748_74898


namespace unique_digit_solution_l748_74858

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def to_two_digit_number (a b : ℕ) : ℕ := 10 * a + b

def to_eight_digit_number (c d e f g h i j : ℕ) : ℕ :=
  10000000 * c + 1000000 * d + 100000 * e + 10000 * f + 1000 * g + 100 * h + 10 * i + j

theorem unique_digit_solution :
  ∃! (A B C D : ℕ),
    is_single_digit A ∧
    is_single_digit B ∧
    is_single_digit C ∧
    is_single_digit D ∧
    A ^ (to_two_digit_number A B) = to_eight_digit_number C C B B D D C A :=
by
  sorry

end unique_digit_solution_l748_74858


namespace integral_of_4x_plus_7_cos_3x_l748_74873

theorem integral_of_4x_plus_7_cos_3x (x : ℝ) :
  deriv (fun x => (1/3) * (4*x + 7) * Real.sin (3*x) + (4/9) * Real.cos (3*x)) x
  = (4*x + 7) * Real.cos (3*x) := by
sorry

end integral_of_4x_plus_7_cos_3x_l748_74873


namespace semicircle_roll_path_length_l748_74874

theorem semicircle_roll_path_length (r : ℝ) (h : r = 4 / Real.pi) : 
  let semicircle_arc_length := r * Real.pi
  semicircle_arc_length = 4 :=
by sorry

end semicircle_roll_path_length_l748_74874


namespace athlete_b_more_stable_l748_74887

/-- Represents an athlete's assessment scores -/
structure AthleteScores where
  scores : Finset ℝ
  count : Nat
  avg : ℝ
  variance : ℝ

/-- Stability of an athlete's scores -/
def moreStable (a b : AthleteScores) : Prop :=
  a.variance < b.variance

theorem athlete_b_more_stable (a b : AthleteScores) 
  (h_count : a.count = 10 ∧ b.count = 10)
  (h_avg : a.avg = b.avg)
  (h_var_a : a.variance = 1.45)
  (h_var_b : b.variance = 0.85) :
  moreStable b a :=
sorry

end athlete_b_more_stable_l748_74887


namespace power_sum_theorem_l748_74856

theorem power_sum_theorem (m n : ℕ) (h1 : 2^m = 3) (h2 : 2^n = 2) : 2^(m+2*n) = 12 := by
  sorry

end power_sum_theorem_l748_74856


namespace geometric_sequence_a2_l748_74822

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a2 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/2 := by
sorry

end geometric_sequence_a2_l748_74822


namespace inverse_proportion_y_relationship_l748_74832

/-- Given points A, B, C on the inverse proportion function y = -2/x, 
    prove the relationship between their y-coordinates. -/
theorem inverse_proportion_y_relationship (y₁ y₂ y₃ : ℝ) : 
  y₁ = -2 / (-2) → y₂ = -2 / 2 → y₃ = -2 / 3 → y₂ < y₃ ∧ y₃ < y₁ := by
  sorry

#check inverse_proportion_y_relationship

end inverse_proportion_y_relationship_l748_74832


namespace carolyn_practice_time_l748_74865

/-- Calculates the total practice time for Carolyn in a month -/
def total_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (practice_days : ℕ) (weeks : ℕ) : ℕ :=
  let daily_total := piano_time + violin_multiplier * piano_time
  let weekly_total := daily_total * practice_days
  weekly_total * weeks

/-- Proves that Carolyn's total practice time in a month is 1920 minutes -/
theorem carolyn_practice_time :
  total_practice_time 20 3 6 4 = 1920 :=
by sorry

end carolyn_practice_time_l748_74865


namespace smallest_non_prime_non_square_with_large_factors_l748_74813

def is_prime (n : ℕ) : Prop := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def has_no_prime_factor_less_than (n m : ℕ) : Prop := sorry

theorem smallest_non_prime_non_square_with_large_factors : 
  ∃ (n : ℕ), n > 0 ∧ 
  ¬ is_prime n ∧ 
  ¬ is_perfect_square n ∧ 
  has_no_prime_factor_less_than n 50 ∧ 
  ∀ (m : ℕ), m > 0 ∧ 
    m < n ∧ 
    ¬ is_prime m ∧ 
    ¬ is_perfect_square m → 
    ¬ has_no_prime_factor_less_than m 50 :=
by
  use 3127
  sorry

#eval 3127

end smallest_non_prime_non_square_with_large_factors_l748_74813


namespace decode_sequence_is_palindrome_l748_74859

/-- Represents the mapping from indices to letters -/
def letter_mapping : Nat → Char
| 1 => 'A'
| 2 => 'E'
| 3 => 'B'
| 4 => 'Γ'
| 5 => 'Δ'
| 6 => 'E'
| 7 => 'E'
| 8 => 'E'
| 9 => '3'
| 10 => 'V'
| 11 => 'U'
| 12 => 'K'
| 13 => 'J'
| 14 => 'M'
| 15 => 'H'
| 16 => 'O'
| 17 => '4'
| 18 => 'P'
| 19 => 'C'
| 20 => 'T'
| 21 => 'y'
| 22 => 'Φ'
| 23 => 'X'
| 24 => '4'
| 25 => '4'
| 26 => 'W'
| 27 => 'M'
| 28 => 'b'
| 29 => 'b'
| 30 => 'b'
| 31 => '3'
| 32 => 'O'
| 33 => '夕'
| _ => ' '  -- Default case

/-- The sequence of numbers to be decoded -/
def encoded_sequence : List Nat := [1, 1, 3, 0, 1, 1, 1, 7, 1, 5, 3, 1, 5, 1, 3, 2, 3, 2, 1, 5, 3, 1, 1, 2, 3, 2, 6, 2, 6, 1, 4, 1, 1, 2, 7, 3, 1, 4, 1, 1, 9, 1, 5, 0, 4, 1, 4, 9]

/-- Function to decode the sequence -/
def decode (seq : List Nat) : String := sorry

/-- The expected decoded palindrome -/
def expected_palindrome : String := "голоден носитель лет и сон не долг"

/-- Theorem stating that decoding the sequence results in the expected palindrome -/
theorem decode_sequence_is_palindrome : decode encoded_sequence = expected_palindrome := by sorry

end decode_sequence_is_palindrome_l748_74859


namespace some_number_proof_l748_74880

theorem some_number_proof (x y : ℝ) (h1 : 5 * x + 3 = 10 * x - y) (h2 : x = 4) : y = 17 := by
  sorry

end some_number_proof_l748_74880


namespace trig_identity_l748_74806

theorem trig_identity (α β : ℝ) :
  (Real.sin (2 * α + β) / Real.sin α) - 2 * Real.cos (α + β) = Real.sin β / Real.sin α :=
by sorry

end trig_identity_l748_74806


namespace at_least_one_equation_has_two_roots_l748_74860

theorem at_least_one_equation_has_two_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hab : a ≠ b)
  (hbc : b ≠ c)
  (hac : a ≠ c) :
  ∃ (x y : ℝ),
    (x ≠ y ∧
      ((a * x^2 + 2 * b * x + c = 0 ∧ a * y^2 + 2 * b * y + c = 0) ∨
       (b * x^2 + 2 * c * x + a = 0 ∧ b * y^2 + 2 * c * y + a = 0) ∨
       (c * x^2 + 2 * a * x + b = 0 ∧ c * y^2 + 2 * a * y + b = 0))) :=
by
  sorry

end at_least_one_equation_has_two_roots_l748_74860


namespace functional_equation_solution_l748_74891

open Function Real

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x - f y) = 1 - x - y) →
  (∀ y : ℝ, f y = 1/2 - y) := by
sorry

end functional_equation_solution_l748_74891


namespace infinite_solutions_for_continuous_function_l748_74804

theorem infinite_solutions_for_continuous_function 
  (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_dom : ∀ x, x ≥ 1 → f x > 0) 
  (h_sol : ∀ a > 0, ∃ x ≥ 1, f x = a * x) : 
  ∀ a > 0, Set.Infinite {x | x ≥ 1 ∧ f x = a * x} :=
sorry

end infinite_solutions_for_continuous_function_l748_74804


namespace log_expression_equals_two_l748_74820

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_two :
  (log10 2)^2 + log10 2 * log10 50 + log10 25 = 2 := by sorry

end log_expression_equals_two_l748_74820


namespace lisa_photos_last_weekend_l748_74837

/-- Calculates the number of photos Lisa took last weekend based on given conditions --/
def photos_last_weekend (animal_photos : ℕ) (flower_multiplier : ℕ) (scenery_difference : ℕ) (weekend_difference : ℕ) : ℕ :=
  let flower_photos := animal_photos * flower_multiplier
  let scenery_photos := flower_photos - scenery_difference
  let total_this_weekend := animal_photos + flower_photos + scenery_photos
  total_this_weekend - weekend_difference

/-- Theorem stating that Lisa took 45 photos last weekend given the conditions --/
theorem lisa_photos_last_weekend :
  photos_last_weekend 10 3 10 15 = 45 := by
  sorry

#eval photos_last_weekend 10 3 10 15

end lisa_photos_last_weekend_l748_74837


namespace sin_600_degrees_l748_74812

theorem sin_600_degrees : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_600_degrees_l748_74812


namespace partnership_share_calculation_l748_74801

/-- Given a partnership where three partners invest different amounts and one partner's share is known, 
    calculate the share of another partner. -/
theorem partnership_share_calculation 
  (investment_a investment_b investment_c : ℕ)
  (duration : ℕ)
  (share_b : ℕ) 
  (h1 : investment_a = 11000)
  (h2 : investment_b = 15000)
  (h3 : investment_c = 23000)
  (h4 : duration = 8)
  (h5 : share_b = 3315) :
  (investment_a : ℚ) / (investment_a + investment_b + investment_c) * 
  (share_b : ℚ) * ((investment_a + investment_b + investment_c) : ℚ) / investment_b = 2421 :=
by sorry

end partnership_share_calculation_l748_74801


namespace min_value_in_region_l748_74870

-- Define the region
def in_region (x y : ℝ) : Prop :=
  y ≥ |x - 1| ∧ y ≤ 2

-- Define the function to be minimized
def f (x y : ℝ) : ℝ := 2*x - y

-- Theorem statement
theorem min_value_in_region :
  ∃ (min : ℝ), min = -4 ∧
  (∀ x y : ℝ, in_region x y → f x y ≥ min) ∧
  (∃ x y : ℝ, in_region x y ∧ f x y = min) :=
sorry

end min_value_in_region_l748_74870


namespace basic_astrophysics_degrees_l748_74824

/-- The total percentage allocated to categories other than basic astrophysics -/
def other_categories_percentage : ℝ := 98

/-- The total degrees in a circle -/
def circle_degrees : ℝ := 360

/-- The percentage allocated to basic astrophysics -/
def basic_astrophysics_percentage : ℝ := 100 - other_categories_percentage

theorem basic_astrophysics_degrees :
  (basic_astrophysics_percentage / 100) * circle_degrees = 7.2 := by sorry

end basic_astrophysics_degrees_l748_74824


namespace initial_customers_l748_74853

theorem initial_customers (remaining : ℕ) (left : ℕ) (initial : ℕ) : 
  remaining = 12 → left = 9 → initial = remaining + left → initial = 21 := by
  sorry

end initial_customers_l748_74853


namespace arithmetic_sequence_problem_l748_74807

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₅ = -1 and a₈ = 2,
    prove that the common difference is 1 and the first term is -5. -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) 
    (h_arith : is_arithmetic_sequence a)
    (h_a5 : a 5 = -1)
    (h_a8 : a 8 = 2) :
    (∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 1 ∧ a 1 = -5 :=
  sorry

end arithmetic_sequence_problem_l748_74807


namespace meaningful_sqrt_range_l748_74811

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end meaningful_sqrt_range_l748_74811


namespace average_of_wxz_l748_74823

variable (w x y z t : ℝ)

theorem average_of_wxz (h1 : 3/w + 3/x + 3/z = 3/(y + t))
                       (h2 : w*x*z = y + t)
                       (h3 : w*z + x*t + y*z = 3*w + 3*x + 3*z) :
  (w + x + z) / 3 = 1/6 := by
  sorry

end average_of_wxz_l748_74823


namespace expression_value_l748_74857

theorem expression_value :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - z^2 + 2*x*y + 10 = 10 := by
sorry

end expression_value_l748_74857


namespace ab_value_l748_74871

theorem ab_value (a b : ℕ+) (h1 : a + b = 24) (h2 : 2 * a * b + 10 * a = 3 * b + 222) : a * b = 108 := by
  sorry

end ab_value_l748_74871


namespace room_occupancy_l748_74835

theorem room_occupancy (people stools chairs : ℕ) : 
  people > stools ∧ 
  people > chairs ∧ 
  people < stools + chairs ∧ 
  2 * people + 3 * stools + 4 * chairs = 32 →
  people = 5 ∧ stools = 2 ∧ chairs = 4 := by
sorry

end room_occupancy_l748_74835


namespace power_of_power_l748_74825

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by sorry

end power_of_power_l748_74825


namespace dark_tile_fraction_is_one_fourth_l748_74892

/-- Represents a 4x4 tile pattern -/
structure TilePattern :=
  (darkTilesInRow : Fin 4 → Nat)
  (h_valid : ∀ i, darkTilesInRow i ≤ 4)

/-- The specific tile pattern described in the problem -/
def problemPattern : TilePattern :=
  { darkTilesInRow := λ i => if i.val < 2 then 2 else 0,
    h_valid := by sorry }

/-- The fraction of dark tiles in a given pattern -/
def darkTileFraction (pattern : TilePattern) : Rat :=
  (pattern.darkTilesInRow 0 + pattern.darkTilesInRow 1 + 
   pattern.darkTilesInRow 2 + pattern.darkTilesInRow 3) / 16

theorem dark_tile_fraction_is_one_fourth :
  darkTileFraction problemPattern = 1/4 := by sorry

end dark_tile_fraction_is_one_fourth_l748_74892


namespace repeating_decimal_sum_l748_74819

-- Define the repeating decimals as rational numbers
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Define the result of the operation
def result : ℚ := a - b + c

-- Theorem statement
theorem repeating_decimal_sum : result = 31 / 37 := by sorry

end repeating_decimal_sum_l748_74819


namespace sum_of_imaginary_parts_l748_74877

theorem sum_of_imaginary_parts (a c d e f : ℂ) : 
  (a + 2*Complex.I) + (c + d*Complex.I) + (e + f*Complex.I) = 4*Complex.I →
  e = -2*a - c →
  d + f = 2 :=
by sorry

end sum_of_imaginary_parts_l748_74877


namespace whitney_spent_440_l748_74828

/-- Calculates the total amount spent by Whitney on books and magazines. -/
def whitneyTotalSpent (whaleBooks fishBooks sharkBooks magazines : ℕ) 
  (whaleCost fishCost sharkCost magazineCost : ℕ) : ℕ :=
  whaleBooks * whaleCost + fishBooks * fishCost + sharkBooks * sharkCost + magazines * magazineCost

/-- Proves that Whitney spent $440 in total. -/
theorem whitney_spent_440 : 
  whitneyTotalSpent 15 12 5 8 14 13 10 3 = 440 := by
  sorry

end whitney_spent_440_l748_74828


namespace car_fuel_efficiency_l748_74845

def distance : ℝ := 120
def gasoline : ℝ := 6

theorem car_fuel_efficiency : distance / gasoline = 20 := by
  sorry

end car_fuel_efficiency_l748_74845


namespace minimum_parts_to_exceed_plan_l748_74855

def plan : ℕ := 40
def excess_percentage : ℚ := 47/100

theorem minimum_parts_to_exceed_plan : 
  ∀ n : ℕ, (n : ℚ) ≥ plan * (1 + excess_percentage) → n ≥ 59 :=
sorry

end minimum_parts_to_exceed_plan_l748_74855


namespace point_coordinate_sum_l748_74818

/-- Given two points A and B, where A is at (2, 1) and B is on the line y = 6,
    and the slope of segment AB is 4/5, prove that the sum of the x- and y-coordinates of B is 14.25 -/
theorem point_coordinate_sum (B : ℝ × ℝ) : 
  B.2 = 6 → -- B is on the line y = 6
  (B.2 - 1) / (B.1 - 2) = 4 / 5 → -- slope of AB is 4/5
  B.1 + B.2 = 14.25 := by sorry

end point_coordinate_sum_l748_74818


namespace work_completion_time_l748_74829

theorem work_completion_time (p q : ℕ) (work_left : ℚ) : 
  p = 15 → q = 20 → work_left = 8/15 → 
  (1 : ℚ) - (1/p + 1/q) * (days_worked : ℚ) = work_left → 
  days_worked = 4 := by
sorry

end work_completion_time_l748_74829


namespace equation_solution_l748_74875

theorem equation_solution : ∃ x : ℝ, 7 * x - 5 = 6 * x ∧ x = 5 := by
  sorry

end equation_solution_l748_74875


namespace brendan_afternoon_catch_brendan_fishing_proof_l748_74878

theorem brendan_afternoon_catch (morning_catch : ℕ) (thrown_back : ℕ) (dad_catch : ℕ) (total_catch : ℕ) : ℕ :=
  let kept_morning := morning_catch - thrown_back
  let afternoon_catch := total_catch - kept_morning - dad_catch
  afternoon_catch

theorem brendan_fishing_proof :
  let morning_catch := 8
  let thrown_back := 3
  let dad_catch := 13
  let total_catch := 23
  brendan_afternoon_catch morning_catch thrown_back dad_catch total_catch = 5 := by
  sorry

end brendan_afternoon_catch_brendan_fishing_proof_l748_74878


namespace percentage_comparison_l748_74805

theorem percentage_comparison (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : 0.02 * A > 0.03 * B) :
  0.05 * A > 0.07 * B := by
  sorry

end percentage_comparison_l748_74805


namespace tangent_perpendicular_implies_a_zero_l748_74896

/-- The function f(x) = (x+a)e^x --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * Real.exp x

/-- The derivative of f(x) --/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 
  (x + a + 1) * Real.exp x

theorem tangent_perpendicular_implies_a_zero (a : ℝ) : 
  (f_derivative a 0 = 1) → a = 0 := by
  sorry

end tangent_perpendicular_implies_a_zero_l748_74896


namespace expected_rounds_four_players_l748_74848

/-- Represents the expected number of rounds in a rock-paper-scissors game -/
def expected_rounds (n : ℕ) : ℚ :=
  match n with
  | 0 => 0
  | 1 => 0
  | 2 => 3/2
  | 3 => 9/4
  | 4 => 81/14
  | _ => 0  -- undefined for n > 4

/-- The rules of the rock-paper-scissors game -/
axiom game_rules : ∀ (n : ℕ), n > 0 → n ≤ 4 → 
  expected_rounds n = 
    if n = 1 then 0
    else if n = 2 then 3/2
    else if n = 3 then 9/4
    else 81/14

/-- The main theorem: expected number of rounds for 4 players is 81/14 -/
theorem expected_rounds_four_players :
  expected_rounds 4 = 81/14 :=
by
  exact game_rules 4 (by norm_num) (by norm_num)


end expected_rounds_four_players_l748_74848


namespace smallest_n_squares_average_is_square_l748_74861

theorem smallest_n_squares_average_is_square : 
  (∀ k : ℕ, k > 1 ∧ k < 337 → ¬ (∃ m : ℕ, (k + 1) * (2 * k + 1) / 6 = m^2)) ∧ 
  (∃ m : ℕ, (337 + 1) * (2 * 337 + 1) / 6 = m^2) := by
  sorry

end smallest_n_squares_average_is_square_l748_74861


namespace remainder_theorem_l748_74883

/-- A polynomial of the form Mx^4 + Nx^2 + Dx - 5 -/
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

theorem remainder_theorem (M N D : ℝ) :
  (∃ p : ℝ → ℝ, ∀ x, q M N D x = (x - 2) * p x + 15) →
  (∃ p : ℝ → ℝ, ∀ x, q M N D x = (x + 2) * p x + 15) :=
by sorry

end remainder_theorem_l748_74883


namespace binomial_expected_value_l748_74810

/-- A random variable following a binomial distribution with n trials and probability p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expected_value (X : BinomialDistribution) : ℝ := X.n * X.p

/-- Theorem: The expected value of X ~ B(6, 1/4) is 3/2 -/
theorem binomial_expected_value :
  let X : BinomialDistribution := { n := 6, p := 1/4, h_p := by norm_num }
  expected_value X = 3/2 := by
  sorry

end binomial_expected_value_l748_74810


namespace solve_equation_l748_74842

theorem solve_equation (m x : ℝ) : 
  (m * x + 1 = 2 * (m - x)) ∧ (|x + 2| = 0) → m = -|3/4| :=
by sorry

end solve_equation_l748_74842


namespace rectangular_field_area_l748_74834

/-- Proves that a rectangular field with width one-third of length and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_field_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  2 * (width + length) = 72 →
  width * length = 243 := by
sorry

end rectangular_field_area_l748_74834


namespace multiples_of_four_median_l748_74830

def first_seven_multiples_of_four : List ℕ := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem multiples_of_four_median (n : ℕ) :
  a ^ 2 - (b n) ^ 2 = 0 → n = 8 := by
  sorry

end multiples_of_four_median_l748_74830


namespace simple_interest_problem_l748_74802

/-- Given a principal P and an interest rate R, if increasing the rate by 6% 
    results in $90 more interest over 5 years, then P = $300. -/
theorem simple_interest_problem (P R : ℝ) : 
  (P * R * 5 / 100 + 90 = P * (R + 6) * 5 / 100) → P = 300 := by
  sorry

end simple_interest_problem_l748_74802


namespace dobarulho_solutions_l748_74895

def is_dobarulho (A B C D : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 8 ∧
  1 ≤ B ∧ B ≤ 9 ∧
  1 ≤ C ∧ C ≤ 9 ∧
  D > 1 ∧
  D ∣ (100*A + 10*B + C) ∧
  D ∣ (100*B + 10*C + A) ∧
  D ∣ (100*C + 10*A + B) ∧
  D ∣ (100*(A+1) + 10*C + B) ∧
  D ∣ (100*C + 10*B + (A+1)) ∧
  D ∣ (100*B + 10*(A+1) + C)

theorem dobarulho_solutions :
  ∀ A B C D : ℕ, is_dobarulho A B C D ↔ 
    ((A = 3 ∧ B = 7 ∧ C = 0 ∧ D = 37) ∨
     (A = 4 ∧ B = 8 ∧ C = 1 ∧ D = 37) ∨
     (A = 5 ∧ B = 9 ∧ C = 2 ∧ D = 37)) :=
by sorry

end dobarulho_solutions_l748_74895


namespace decimal_division_l748_74897

theorem decimal_division (x y : ℚ) (hx : x = 0.54) (hy : y = 0.006) : x / y = 90 := by
  sorry

end decimal_division_l748_74897


namespace maximize_profit_l748_74815

/-- The production volume that maximizes profit -/
def optimal_production_volume : ℝ := 6

/-- Sales revenue as a function of production volume -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Production cost as a function of production volume -/
def production_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit as a function of production volume -/
def profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem maximize_profit (x : ℝ) (h : x > 0) :
  profit x ≤ profit optimal_production_volume := by
  sorry

end maximize_profit_l748_74815


namespace inequality_proof_l748_74885

theorem inequality_proof (a b c : ℝ) (h1 : 0 < c) (h2 : c ≤ b) (h3 : b ≤ a) :
  (a^2 - b^2)/c + (c^2 - b^2)/a + (a^2 - c^2)/b ≥ 3*a - 4*b + c := by
  sorry

end inequality_proof_l748_74885


namespace robin_gum_packages_l748_74868

/-- Given that Robin has some packages of gum, with 7 pieces in each package,
    6 extra pieces, and 41 pieces in total, prove that Robin has 5 packages. -/
theorem robin_gum_packages : ∀ (p : ℕ), 
  (7 * p + 6 = 41) → p = 5 := by
  sorry

end robin_gum_packages_l748_74868


namespace valid_k_values_l748_74814

def A (n : ℕ) : ℕ := 1^n + 2^n + 3^n + 4^n

def ends_with_k_zeros (m k : ℕ) : Prop :=
  ∃ r : ℕ, m = r * 10^k ∧ r % 10 ≠ 0

theorem valid_k_values :
  {k : ℕ | ∃ n : ℕ, ends_with_k_zeros (A n) k} = {0, 1, 2} := by sorry

end valid_k_values_l748_74814


namespace parabola_line_intersection_l748_74821

/-- A parabola with vertex at origin and axis of symmetry along x-axis -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- A line with slope k passing through a fixed point -/
structure Line where
  k : ℝ
  fixed_point : ℝ × ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = k * x + (fixed_point.2 - k * fixed_point.1)

/-- The number of intersection points between a parabola and a line -/
def intersection_count (par : Parabola) (l : Line) : ℕ :=
  sorry

theorem parabola_line_intersection 
  (par : Parabola) 
  (h_par : par.eq (1/2) (-Real.sqrt 2))
  (l : Line)
  (h_line : l.fixed_point = (-2, 1)) :
  (intersection_count par l = 2) ↔ 
  (-1 < l.k ∧ l.k < 1/2 ∧ l.k ≠ 0) :=
sorry

end parabola_line_intersection_l748_74821


namespace first_half_speed_l748_74841

def total_distance : ℝ := 112
def total_time : ℝ := 5
def second_half_speed : ℝ := 24

theorem first_half_speed : 
  ∃ (v : ℝ), 
    v * (total_time - (total_distance / 2) / second_half_speed) = total_distance / 2 ∧ 
    v = 21 := by
  sorry

end first_half_speed_l748_74841


namespace incorrect_y_value_l748_74893

/-- Represents a quadratic function y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents a sequence of 7 x values with equal differences -/
structure XSequence where
  x : Fin 7 → ℝ
  increasing : ∀ i j, i < j → x i < x j
  equal_diff : ∀ i : Fin 6, x (i + 1) - x i = x 1 - x 0

/-- The given y values -/
def y_values : Fin 7 → ℝ := ![51, 107, 185, 285, 407, 549, 717]

/-- The theorem to prove -/
theorem incorrect_y_value (f : QuadraticFunction) (xs : XSequence) :
  (∀ i : Fin 7, i.val ≠ 5 → y_values i = f.a * (xs.x i)^2 + f.b * (xs.x i) + f.c) →
  y_values 5 ≠ f.a * (xs.x 5)^2 + f.b * (xs.x 5) + f.c ∧
  571 = f.a * (xs.x 5)^2 + f.b * (xs.x 5) + f.c := by
  sorry

end incorrect_y_value_l748_74893


namespace daves_rides_l748_74836

theorem daves_rides (total_rides : ℕ) (second_day_rides : ℕ) (first_day_rides : ℕ) :
  total_rides = 7 ∧ second_day_rides = 3 ∧ total_rides = first_day_rides + second_day_rides →
  first_day_rides = 4 := by
sorry

end daves_rides_l748_74836


namespace age_difference_proof_l748_74867

theorem age_difference_proof (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 245 →
  monica_age - patrick_age = 80 := by
  sorry

end age_difference_proof_l748_74867


namespace special_function_properties_l748_74852

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y - 1

theorem special_function_properties (f : ℝ → ℝ) 
  (h1 : special_function f) (h2 : f 1 = 4) : 
  f 0 = 1 ∧ ∀ n : ℕ, f n = (n + 1)^2 :=
by sorry

end special_function_properties_l748_74852


namespace count_valid_primes_l748_74809

/-- Convert a number from base p to base 10 --/
def to_base_10 (digits : List Nat) (p : Nat) : Nat :=
  digits.foldr (fun d acc => d + p * acc) 0

/-- Check if the equation holds for a given prime p --/
def equation_holds (p : Nat) : Prop :=
  to_base_10 [9, 7, 6] p + to_base_10 [5, 0, 7] p + to_base_10 [2, 3, 8] p =
  to_base_10 [4, 2, 9] p + to_base_10 [5, 9, 5] p + to_base_10 [6, 9, 7] p

/-- The main theorem --/
theorem count_valid_primes :
  ∃ (S : Finset Nat), S.card = 3 ∧ 
  (∀ p ∈ S, Nat.Prime p ∧ p < 10 ∧ equation_holds p) ∧
  (∀ p, Nat.Prime p → p < 10 → equation_holds p → p ∈ S) :=
sorry

end count_valid_primes_l748_74809


namespace sequence_difference_l748_74839

def sequence1 (n : ℕ) : ℤ := 2 * (-2)^(n - 1)
def sequence2 (n : ℕ) : ℤ := sequence1 n - 1
def sequence3 (n : ℕ) : ℤ := (-2)^n - sequence2 n

theorem sequence_difference : sequence1 7 - sequence2 7 + sequence3 7 = -254 := by
  sorry

end sequence_difference_l748_74839


namespace square_area_measurement_error_l748_74833

theorem square_area_measurement_error :
  let actual_length : ℝ := L
  let measured_side1 : ℝ := L * (1 + 0.02)
  let measured_side2 : ℝ := L * (1 - 0.03)
  let calculated_area : ℝ := measured_side1 * measured_side2
  let actual_area : ℝ := L * L
  let error : ℝ := actual_area - calculated_area
  let percentage_error : ℝ := (error / actual_area) * 100
  percentage_error = 1.06 := by
sorry

end square_area_measurement_error_l748_74833


namespace wine_purchase_additional_cost_l748_74881

/-- Represents the price changes and conditions for wine purchases over three months --/
structure WinePrices where
  initial_price : ℝ
  tariff_increase1 : ℝ
  tariff_increase2 : ℝ
  exchange_rate_change1 : ℝ
  exchange_rate_change2 : ℝ
  bulk_discount : ℝ
  bottles_per_month : ℕ

/-- Calculates the total additional cost of wine purchases over three months --/
def calculate_additional_cost (prices : WinePrices) : ℝ :=
  let month1_price := prices.initial_price * (1 + prices.exchange_rate_change1)
  let month2_price := prices.initial_price * (1 + prices.tariff_increase1) * (1 - prices.bulk_discount)
  let month3_price := prices.initial_price * (1 + prices.tariff_increase1 + prices.tariff_increase2) * (1 - prices.exchange_rate_change2)
  let total_cost := (month1_price + month2_price + month3_price) * prices.bottles_per_month
  let initial_total := prices.initial_price * prices.bottles_per_month * 3
  total_cost - initial_total

/-- Theorem stating that the additional cost of wine purchases over three months is $42.20 --/
theorem wine_purchase_additional_cost :
  let prices : WinePrices := {
    initial_price := 20,
    tariff_increase1 := 0.25,
    tariff_increase2 := 0.10,
    exchange_rate_change1 := 0.05,
    exchange_rate_change2 := 0.03,
    bulk_discount := 0.15,
    bottles_per_month := 5
  }
  calculate_additional_cost prices = 42.20 := by
  sorry


end wine_purchase_additional_cost_l748_74881


namespace estimate_sqrt_expression_l748_74808

theorem estimate_sqrt_expression :
  7 < Real.sqrt 32 * Real.sqrt (1/2) + Real.sqrt 12 ∧
  Real.sqrt 32 * Real.sqrt (1/2) + Real.sqrt 12 < 8 :=
by
  sorry

end estimate_sqrt_expression_l748_74808


namespace sum_equals_zero_l748_74882

def f (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_equals_zero :
  f (1/7) - f (2/7) + f (3/7) - f (4/7) + f (5/7) - f (6/7) = 0 := by
  sorry

end sum_equals_zero_l748_74882


namespace distance_to_school_prove_distance_to_school_l748_74838

theorem distance_to_school (time_with_traffic time_without_traffic : ℝ)
  (speed_difference : ℝ) (distance : ℝ) : Prop :=
  time_with_traffic = 20 / 60 →
  time_without_traffic = 15 / 60 →
  speed_difference = 15 →
  ∃ (speed_with_traffic : ℝ),
    distance = speed_with_traffic * time_with_traffic ∧
    distance = (speed_with_traffic + speed_difference) * time_without_traffic →
  distance = 15

-- The proof of the theorem
theorem prove_distance_to_school :
  ∀ (time_with_traffic time_without_traffic speed_difference distance : ℝ),
  distance_to_school time_with_traffic time_without_traffic speed_difference distance :=
by
  sorry

end distance_to_school_prove_distance_to_school_l748_74838


namespace photo_gallery_problem_l748_74866

/-- The total number of photos in a gallery after a two-day trip -/
def total_photos (initial : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  initial + first_day + second_day

/-- Theorem: Given the conditions of the photo gallery problem, the total number of photos is 920 -/
theorem photo_gallery_problem :
  let initial := 400
  let first_day := initial / 2
  let second_day := first_day + 120
  total_photos initial first_day second_day = 920 := by
  sorry

end photo_gallery_problem_l748_74866


namespace inequality_proof_root_inequality_l748_74872

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hne : ¬(a = b ∧ b = c)) : 
  a*(b^2 + c^2) + b*(c^2 + a^2) + c*(a^2 + b^2) > 6*a*b*c :=
sorry

theorem root_inequality : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 :=
sorry

end inequality_proof_root_inequality_l748_74872


namespace certain_number_problem_l748_74886

theorem certain_number_problem (x : ℝ) : 300 + (x * 8) = 340 → x = 5 := by
  sorry

end certain_number_problem_l748_74886


namespace negative_intervals_l748_74817

-- Define the expression
def f (x : ℝ) : ℝ := (x - 2) * (x + 2) * (x - 3)

-- Define the set of x for which f(x) is negative
def S : Set ℝ := {x | f x < 0}

-- State the theorem
theorem negative_intervals : S = Set.Iio (-2) ∪ Set.Ioo 2 3 := by sorry

end negative_intervals_l748_74817


namespace peanuts_remaining_l748_74890

def initial_peanuts : ℕ := 148
def bonita_eaten : ℕ := 29

theorem peanuts_remaining : 
  initial_peanuts - (initial_peanuts / 4) - bonita_eaten = 82 :=
by
  sorry

end peanuts_remaining_l748_74890


namespace quadratic_root_difference_l748_74840

theorem quadratic_root_difference (c : ℝ) : 
  (∃ x y : ℝ, x^2 + 7*x + c = 0 ∧ y^2 + 7*y + c = 0 ∧ |x - y| = Real.sqrt 85) → 
  c = -9 :=
by sorry

end quadratic_root_difference_l748_74840


namespace prob_two_threes_correct_l748_74831

/-- The probability of rolling exactly two 3s when rolling eight standard 6-sided dice -/
def prob_two_threes : ℚ :=
  (28 : ℚ) * 15625 / 559872

/-- The probability calculated using binomial distribution -/
def prob_two_threes_calc : ℚ :=
  (Nat.choose 8 2 : ℚ) * (1/6)^2 * (5/6)^6

theorem prob_two_threes_correct : prob_two_threes = prob_two_threes_calc := by
  sorry

end prob_two_threes_correct_l748_74831


namespace fraction_difference_l748_74849

theorem fraction_difference (a b : ℝ) : 
  a / (a + 1) - b / (b + 1) = (a - b) / ((a + 1) * (b + 1)) := by
  sorry

end fraction_difference_l748_74849
