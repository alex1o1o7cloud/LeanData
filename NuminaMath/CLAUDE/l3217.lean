import Mathlib

namespace initial_weavers_count_l3217_321746

/-- The number of mat-weavers initially weaving -/
def initial_weavers : ℕ := sorry

/-- The number of mats woven by the initial weavers -/
def initial_mats : ℕ := 4

/-- The number of days taken by the initial weavers -/
def initial_days : ℕ := 4

/-- The number of mat-weavers in the second scenario -/
def second_weavers : ℕ := 8

/-- The number of mats woven in the second scenario -/
def second_mats : ℕ := 16

/-- The number of days taken in the second scenario -/
def second_days : ℕ := 8

/-- The rate of weaving is consistent across both scenarios -/
axiom consistent_rate : 
  (initial_mats : ℚ) / (initial_weavers * initial_days) = 
  (second_mats : ℚ) / (second_weavers * second_days)

theorem initial_weavers_count : initial_weavers = 4 := by
  sorry

end initial_weavers_count_l3217_321746


namespace age_difference_is_six_l3217_321793

-- Define Claire's future age
def claire_future_age : ℕ := 20

-- Define the number of years until Claire reaches her future age
def years_until_future : ℕ := 2

-- Define Jessica's current age
def jessica_current_age : ℕ := 24

-- Theorem to prove
theorem age_difference_is_six :
  jessica_current_age - (claire_future_age - years_until_future) = 6 :=
by sorry

end age_difference_is_six_l3217_321793


namespace grunters_win_all_games_l3217_321740

/-- The number of games played between the Grunters and the Screamers -/
def num_games : ℕ := 6

/-- The probability of the Grunters winning a game that doesn't go to overtime -/
def p_win_no_overtime : ℝ := 0.6

/-- The probability of the Grunters winning a game that goes to overtime -/
def p_win_overtime : ℝ := 0.5

/-- The probability of a game going to overtime -/
def p_overtime : ℝ := 0.1

/-- The theorem stating the probability of the Grunters winning all games -/
theorem grunters_win_all_games : 
  (((1 - p_overtime) * p_win_no_overtime + p_overtime * p_win_overtime) ^ num_games : ℝ) = 
  (823543 : ℝ) / 10000000 := by sorry

end grunters_win_all_games_l3217_321740


namespace blackboard_numbers_theorem_l3217_321707

theorem blackboard_numbers_theorem (n : ℕ) (h_n : n > 3) 
  (numbers : Fin n → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j) 
  (h_bound : ∀ i, numbers i < Nat.factorial (n - 1)) :
  ∃ (i j k l : Fin n), i ≠ k ∧ j ≠ l ∧ numbers i > numbers j ∧ numbers k > numbers l ∧
    (numbers i / numbers j : ℕ) = (numbers k / numbers l : ℕ) :=
sorry

end blackboard_numbers_theorem_l3217_321707


namespace g_max_value_l3217_321725

/-- The function g(x) = 4x - x^4 -/
def g (x : ℝ) := 4 * x - x^4

/-- The maximum value of g(x) on the interval [0, 2] is 3 -/
theorem g_max_value : ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ ∀ x ∈ Set.Icc 0 2, g x ≤ g c ∧ g c = 3 := by
  sorry

end g_max_value_l3217_321725


namespace smallest_n_is_smallest_l3217_321731

/-- The smallest positive integer satisfying the given conditions -/
def smallest_n : ℕ := 46656

/-- n is divisible by 36 -/
axiom divisible_by_36 : smallest_n % 36 = 0

/-- n^2 is a perfect cube -/
axiom perfect_cube : ∃ k : ℕ, smallest_n^2 = k^3

/-- n^3 is a perfect square -/
axiom perfect_square : ∃ k : ℕ, smallest_n^3 = k^2

/-- Theorem stating that smallest_n is indeed the smallest positive integer satisfying all conditions -/
theorem smallest_n_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m % 36 = 0 ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2) → m ≥ smallest_n :=
sorry

end smallest_n_is_smallest_l3217_321731


namespace cylinder_triple_volume_radius_l3217_321790

/-- Theorem: Tripling the volume of a cylinder while keeping the same height results in a new radius that is √3 times the original radius. -/
theorem cylinder_triple_volume_radius (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  let v := π * r^2 * h
  let v_new := 3 * v
  let r_new := Real.sqrt ((3 * π * r^2 * h) / (π * h))
  r_new = r * Real.sqrt 3 := by
  sorry

end cylinder_triple_volume_radius_l3217_321790


namespace smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5_l3217_321701

theorem smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → ¬(18 ∣ m ∧ 30 < Real.sqrt m ∧ Real.sqrt m < 30.5)) ∧
            (18 ∣ n) ∧ (30 < Real.sqrt n) ∧ (Real.sqrt n < 30.5) ∧ n = 900 := by
  sorry

end smallest_integer_divisible_by_18_with_sqrt_between_30_and_30_5_l3217_321701


namespace infinite_solutions_for_diophantine_equation_l3217_321734

theorem infinite_solutions_for_diophantine_equation :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
  (∀ p ∈ S, Prime p ∧ 
  ∃ x y : ℤ, x^2 + x + 1 = p * y) := by
  sorry

end infinite_solutions_for_diophantine_equation_l3217_321734


namespace hypotenuse_length_of_isosceles_right_triangle_l3217_321706

def isosceles_right_triangle (a c : ℝ) : Prop :=
  a > 0 ∧ c > 0 ∧ c^2 = 2 * a^2

theorem hypotenuse_length_of_isosceles_right_triangle (a c : ℝ) :
  isosceles_right_triangle a c →
  2 * a + c = 8 + 8 * Real.sqrt 2 →
  c = 4 * Real.sqrt 2 := by
sorry

end hypotenuse_length_of_isosceles_right_triangle_l3217_321706


namespace cylinder_volume_l3217_321719

/-- Given a cylinder with lateral surface area 100π cm² and an inscribed rectangular solid
    with diagonal 10√2 cm, prove that the volume of the cylinder is 250π cm³. -/
theorem cylinder_volume (r h : ℝ) : 
  r > 0 → h > 0 →
  2 * Real.pi * r * h = 100 * Real.pi →
  4 * r^2 + h^2 = 200 →
  Real.pi * r^2 * h = 250 * Real.pi :=
by sorry

end cylinder_volume_l3217_321719


namespace min_values_l3217_321704

-- Define the equation
def equation (x y : ℝ) : Prop := Real.log (3 * x) + Real.log y = Real.log (x + y + 1)

-- Theorem statement
theorem min_values (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : equation x y) :
  (∀ a b, equation a b → x * y ≤ a * b) ∧
  (∀ a b, equation a b → x + y ≤ a + b) ∧
  (∀ a b, equation a b → 1 / x + 1 / y ≤ 1 / a + 1 / b) ∧
  x * y = 1 ∧ x + y = 2 ∧ 1 / x + 1 / y = 2 :=
sorry

end min_values_l3217_321704


namespace infinitely_many_nth_powers_l3217_321710

/-- An infinite arithmetic progression of positive integers -/
structure ArithmeticProgression :=
  (a : ℕ)  -- First term
  (d : ℕ)  -- Common difference

/-- Checks if a number is in the arithmetic progression -/
def ArithmeticProgression.contains (ap : ArithmeticProgression) (x : ℕ) : Prop :=
  ∃ k : ℕ, x = ap.a + k * ap.d

/-- Checks if a number is an nth power -/
def is_nth_power (x n : ℕ) : Prop :=
  ∃ m : ℕ, x = m^n

theorem infinitely_many_nth_powers
  (ap : ArithmeticProgression)
  (n : ℕ)
  (h : ∃ x : ℕ, ap.contains x ∧ is_nth_power x n) :
  ∀ N : ℕ, ∃ M : ℕ, M > N ∧ ap.contains M ∧ is_nth_power M n :=
sorry

end infinitely_many_nth_powers_l3217_321710


namespace smallest_non_prime_non_square_no_small_factors_l3217_321771

theorem smallest_non_prime_non_square_no_small_factors : ∃ n : ℕ,
  n = 5183 ∧
  (∀ m : ℕ, m < n →
    (Nat.Prime m → m ≥ 70) ∧
    (¬ Nat.Prime n) ∧
    (∀ k : ℕ, k * k ≠ n)) :=
by sorry

end smallest_non_prime_non_square_no_small_factors_l3217_321771


namespace quadratic_properties_l3217_321724

/-- Quadratic function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem quadratic_properties :
  (∃ (a b : ℝ), ∀ x, f x = (x - a)^2 + b ∧ a = 2 ∧ b = -1) ∧
  (f 1 = 0 ∧ f 3 = 0) :=
sorry

end quadratic_properties_l3217_321724


namespace project_hours_total_l3217_321727

/-- Represents the hours charged by Kate, Pat, and Mark to a project -/
structure ProjectHours where
  kate : ℕ
  pat : ℕ
  mark : ℕ

/-- Defines the conditions of the project hours -/
def validProjectHours (h : ProjectHours) : Prop :=
  h.pat = 2 * h.kate ∧
  h.pat = h.mark / 3 ∧
  h.mark = h.kate + 110

theorem project_hours_total (h : ProjectHours) (hValid : validProjectHours h) :
  h.kate + h.pat + h.mark = 198 := by
  sorry

end project_hours_total_l3217_321727


namespace B_proper_subset_A_l3217_321772

-- Define sets A and B
def A : Set ℝ := {x | x > (1/2)}
def B : Set ℝ := {x | x > 1}

-- Theorem statement
theorem B_proper_subset_A : B ⊂ A := by sorry

end B_proper_subset_A_l3217_321772


namespace same_color_sock_pairs_l3217_321753

def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem same_color_sock_pairs (white black red : ℕ) 
  (h_white : white = 5) 
  (h_black : black = 4) 
  (h_red : red = 3) : 
  (choose white 2) + (choose black 2) + (choose red 2) = 19 := by
  sorry

end same_color_sock_pairs_l3217_321753


namespace spinster_cat_problem_l3217_321733

theorem spinster_cat_problem (S C : ℕ) 
  (h1 : S * 9 = C * 2)  -- Ratio of spinsters to cats is 2:9
  (h2 : C = S + 63)     -- There are 63 more cats than spinsters
  : S = 18 := by        -- Prove that the number of spinsters is 18
sorry

end spinster_cat_problem_l3217_321733


namespace initial_to_doubled_ratio_l3217_321749

theorem initial_to_doubled_ratio (x : ℝ) : 3 * (2 * x + 8) = 84 → x / (2 * x) = 1 / 2 := by
  sorry

end initial_to_doubled_ratio_l3217_321749


namespace smaller_number_problem_l3217_321713

theorem smaller_number_problem (x y : ℝ) 
  (eq1 : 3 * x - y = 20) 
  (eq2 : x + y = 48) : 
  min x y = 17 := by
sorry

end smaller_number_problem_l3217_321713


namespace custom_operation_equation_l3217_321767

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a + 2 * b

-- State the theorem
theorem custom_operation_equation :
  ∃ x : ℝ, star 3 (star 4 x) = 6 ∧ x = -5/4 := by
  sorry

end custom_operation_equation_l3217_321767


namespace total_distance_to_grandma_l3217_321792

/-- The distance to Grandma's house -/
def distance_to_grandma (distance_to_pie_shop : ℕ) (distance_to_gas_station : ℕ) (remaining_distance : ℕ) : ℕ :=
  distance_to_pie_shop + distance_to_gas_station + remaining_distance

/-- Theorem: The total distance to Grandma's house is 78 miles -/
theorem total_distance_to_grandma : 
  distance_to_grandma 35 18 25 = 78 := by
  sorry

end total_distance_to_grandma_l3217_321792


namespace inequality_proof_l3217_321760

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) 
  (h_sum : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 := by
  sorry

end inequality_proof_l3217_321760


namespace binomial_coefficient_equation_solution_l3217_321796

theorem binomial_coefficient_equation_solution : 
  ∃! n : ℕ, (Nat.choose 25 n) + (Nat.choose 25 12) = (Nat.choose 26 13) ∧ n = 13 := by
  sorry

end binomial_coefficient_equation_solution_l3217_321796


namespace trigonometric_identity_l3217_321786

theorem trigonometric_identity (x : ℝ) :
  Real.sin (x + π / 3) + 2 * Real.sin (x - π / 3) - Real.sqrt 3 * Real.cos ((2 * π) / 3 - x) = 0 := by
  sorry

end trigonometric_identity_l3217_321786


namespace stratified_sample_size_l3217_321757

/-- Represents the number of students in each grade and the sample size for grade 10 -/
structure SchoolData where
  grade12 : ℕ
  grade11 : ℕ
  grade10 : ℕ
  sample10 : ℕ

/-- Calculates the total number of students sampled from the entire school using stratified sampling -/
def totalSampleSize (data : SchoolData) : ℕ :=
  (data.sample10 * (data.grade12 + data.grade11 + data.grade10)) / data.grade10

/-- Theorem stating that given the specific school data, the total sample size is 220 -/
theorem stratified_sample_size :
  let data := SchoolData.mk 700 700 800 80
  totalSampleSize data = 220 := by
  sorry

#check stratified_sample_size

end stratified_sample_size_l3217_321757


namespace smaller_angle_is_70_l3217_321715

/-- A parallelogram with one angle exceeding the other by 40 degrees -/
structure Parallelogram40 where
  -- The measure of the smaller angle
  small_angle : ℝ
  -- The measure of the larger angle
  large_angle : ℝ
  -- The larger angle exceeds the smaller by 40 degrees
  angle_difference : large_angle = small_angle + 40
  -- Adjacent angles are supplementary (sum to 180 degrees)
  supplementary : small_angle + large_angle = 180

/-- The smaller angle in a Parallelogram40 measures 70 degrees -/
theorem smaller_angle_is_70 (p : Parallelogram40) : p.small_angle = 70 := by
  sorry

end smaller_angle_is_70_l3217_321715


namespace graph_shift_l3217_321778

-- Define a generic function g
variable (g : ℝ → ℝ)

-- Define the shift transformation
def shift (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x => f (x - a)

-- Theorem statement
theorem graph_shift (a : ℝ) :
  ∀ x : ℝ, (shift g a) x = g (x - a) :=
by sorry

end graph_shift_l3217_321778


namespace dot_product_zero_l3217_321770

-- Define the circle
def Circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line on which P lies
def Line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define points A and B
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the dot product of two vectors
def dotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_zero (P : ℝ × ℝ) (h : Line P.1 P.2) :
  dotProduct (P.1 - A.1, P.2 - A.2) (P.1 - B.1, P.2 - B.2) = 0 := by
  sorry

end dot_product_zero_l3217_321770


namespace fourth_person_height_l3217_321732

/-- Given four people with heights in increasing order, prove that the fourth person is 84 inches tall -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- heights in increasing order
  h₂ - h₁ = 2 →                 -- difference between first and second
  h₃ - h₂ = 2 →                 -- difference between second and third
  h₄ - h₃ = 6 →                 -- difference between third and fourth
  (h₁ + h₂ + h₃ + h₄) / 4 = 78  -- average height
  → h₄ = 84 := by
sorry

end fourth_person_height_l3217_321732


namespace custom_mult_eleven_twelve_l3217_321779

/-- Custom multiplication operation for integers -/
def custom_mult (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that for y = 11, y * 12 = 110 under the custom multiplication -/
theorem custom_mult_eleven_twelve :
  let y : ℤ := 11
  custom_mult y 12 = 110 := by
  sorry

end custom_mult_eleven_twelve_l3217_321779


namespace min_value_greater_than_nine_l3217_321752

theorem min_value_greater_than_nine (a : ℝ) (h : a = 6) :
  ∀ x > a, x + 4 / (x - a) > 9 := by
  sorry

end min_value_greater_than_nine_l3217_321752


namespace abs_seven_minus_sqrt_two_l3217_321769

theorem abs_seven_minus_sqrt_two (h : Real.sqrt 2 < 7) : 
  |7 - Real.sqrt 2| = 7 - Real.sqrt 2 := by
  sorry

end abs_seven_minus_sqrt_two_l3217_321769


namespace max_median_amount_l3217_321739

/-- Represents the initial amounts of money for each person -/
def initial_amounts : List ℕ := [28, 72, 98]

/-- The total amount of money after pooling -/
def total_amount : ℕ := initial_amounts.sum

/-- The number of people -/
def num_people : ℕ := initial_amounts.length

theorem max_median_amount :
  ∃ (distribution : List ℕ),
    distribution.length = num_people ∧
    distribution.sum = total_amount ∧
    (∃ (median : ℕ), median ∈ distribution ∧ 
      (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
      (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2) ∧
    (∀ (other_distribution : List ℕ),
      other_distribution.length = num_people →
      other_distribution.sum = total_amount →
      (∃ (other_median : ℕ), other_median ∈ other_distribution ∧ 
        (other_distribution.filter (λ x => x ≤ other_median)).length ≥ num_people / 2 ∧
        (other_distribution.filter (λ x => x ≥ other_median)).length ≥ num_people / 2) →
      ∃ (median : ℕ), median ∈ distribution ∧ 
        (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
        (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2 ∧
        median ≥ other_median) ∧
    (∃ (median : ℕ), median ∈ distribution ∧ 
      (distribution.filter (λ x => x ≤ median)).length ≥ num_people / 2 ∧
      (distribution.filter (λ x => x ≥ median)).length ≥ num_people / 2 ∧
      median = 196) := by
  sorry


end max_median_amount_l3217_321739


namespace product_expansion_l3217_321776

theorem product_expansion (x : ℝ) (hx : x ≠ 0) :
  (3 / 4) * ((8 / x^2) + 5*x - 6) = 6 / x^2 + (15*x) / 4 - 4.5 := by
  sorry

end product_expansion_l3217_321776


namespace solution_set_for_a_equals_one_range_of_a_for_inclusion_l3217_321775

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

-- Theorem 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | |x + 2| + |x - 1| ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem range_of_a_for_inclusion :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x + |x - 1| ≤ 2) → -3/2 ≤ a ∧ a ≤ -1/2 := by sorry

end solution_set_for_a_equals_one_range_of_a_for_inclusion_l3217_321775


namespace solid_surface_area_l3217_321781

/-- The surface area of a solid composed of a cylinder topped with a hemisphere -/
theorem solid_surface_area (r h : ℝ) (hr : r = 1) (hh : h = 3) :
  2 * π * r * h + 2 * π * r^2 + 2 * π * r^2 = 12 * π := by
  sorry

end solid_surface_area_l3217_321781


namespace circle_diameter_endpoint_l3217_321716

/-- Given a circle with center (2,3) and one endpoint of a diameter at (-1,-1),
    the other endpoint of the diameter is at (5,7). -/
theorem circle_diameter_endpoint (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : 
  O = (2, 3) → A = (-1, -1) → 
  (O.1 - A.1 = B.1 - O.1 ∧ O.2 - A.2 = B.2 - O.2) → 
  B = (5, 7) := by
  sorry

end circle_diameter_endpoint_l3217_321716


namespace lcm_gcd_product_l3217_321797

theorem lcm_gcd_product (a b : ℕ) (ha : a = 30) (hb : b = 75) :
  Nat.lcm a b * Nat.gcd a b = a * b := by
  sorry

end lcm_gcd_product_l3217_321797


namespace overtime_rate_is_90_cents_l3217_321721

/-- Represents the worker's pay structure and work week --/
structure WorkerPay where
  ordinary_rate : ℚ
  total_hours : ℕ
  overtime_hours : ℕ
  total_pay : ℚ

/-- Calculates the overtime rate given the worker's pay structure --/
def overtime_rate (w : WorkerPay) : ℚ :=
  let ordinary_hours := w.total_hours - w.overtime_hours
  let ordinary_pay := (w.ordinary_rate * ordinary_hours : ℚ)
  let overtime_pay := w.total_pay - ordinary_pay
  overtime_pay / w.overtime_hours

/-- Theorem stating that the overtime rate is $0.90 per hour --/
theorem overtime_rate_is_90_cents (w : WorkerPay) 
  (h1 : w.ordinary_rate = 60 / 100)
  (h2 : w.total_hours = 50)
  (h3 : w.overtime_hours = 8)
  (h4 : w.total_pay = 3240 / 100) : 
  overtime_rate w = 90 / 100 := by
  sorry

#eval overtime_rate { 
  ordinary_rate := 60 / 100, 
  total_hours := 50, 
  overtime_hours := 8, 
  total_pay := 3240 / 100 
}

end overtime_rate_is_90_cents_l3217_321721


namespace inequality_proof_l3217_321738

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l3217_321738


namespace inequality_solution_l3217_321708

theorem inequality_solution (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  (2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5) < 1 / 24) ↔
  (x < 1 ∨ (4 < x ∧ x < 5) ∨ 6 < x) :=
sorry

end inequality_solution_l3217_321708


namespace min_sum_and_inequality_l3217_321717

-- Define the function f
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

-- State the theorem
theorem min_sum_and_inequality (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∀ x, f x a b ≥ 4) : 
  a + b ≥ 4 ∧ (a + b = 4 → 1/a + 4/b ≥ 9/4) := by
  sorry


end min_sum_and_inequality_l3217_321717


namespace quadratic_rewrite_l3217_321765

theorem quadratic_rewrite (x : ℝ) : ∃ (a b c : ℤ), 
  16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c ∧ a * b = -20 := by
sorry

end quadratic_rewrite_l3217_321765


namespace charlie_banana_consumption_l3217_321777

/-- Represents the daily banana consumption of Charlie the chimp over 7 days -/
def BananaSequence : Type := Fin 7 → ℚ

/-- The sum of bananas eaten over 7 days is 150 -/
def SumIs150 (seq : BananaSequence) : Prop :=
  (Finset.sum Finset.univ seq) = 150

/-- Each day's consumption is 4 more than the previous day -/
def ArithmeticProgression (seq : BananaSequence) : Prop :=
  ∀ i : Fin 6, seq (i.succ) = seq i + 4

/-- The theorem to be proved -/
theorem charlie_banana_consumption
  (seq : BananaSequence)
  (sum_cond : SumIs150 seq)
  (prog_cond : ArithmeticProgression seq) :
  seq 6 = 33 + 4/7 := by sorry

end charlie_banana_consumption_l3217_321777


namespace max_value_abc_l3217_321747

theorem max_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b * c * (a + b + c)) / ((a + b)^2 * (b + c)^3) ≤ 1 / 12 := by
  sorry

end max_value_abc_l3217_321747


namespace find_k_l3217_321754

theorem find_k (k : ℚ) (h : 64 / k = 4) : k = 16 := by
  sorry

end find_k_l3217_321754


namespace largest_x_floor_ratio_l3217_321782

theorem largest_x_floor_ratio : 
  ∀ x : ℝ, (↑(⌊x⌋) : ℝ) / x = 7 / 8 → x ≤ 48 / 7 := by
sorry

end largest_x_floor_ratio_l3217_321782


namespace calculation_proof_l3217_321714

theorem calculation_proof :
  (1/2 + (-2/3) - 4/7 + (-1/2) - 1/3 = -11/7) ∧
  (-7^2 + 2*(-3)^2 - (-6)/((-1/3)^2) = 23) := by
  sorry

end calculation_proof_l3217_321714


namespace inequality_solution_set_l3217_321700

theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 := by
  sorry

end inequality_solution_set_l3217_321700


namespace investment_ratio_a_to_b_l3217_321787

/-- Given the investment ratios and profit distribution, prove the ratio of investments between A and B -/
theorem investment_ratio_a_to_b :
  ∀ (a b c total_investment total_profit : ℚ),
  -- A and C invested in ratio 3:2
  a / c = 3 / 2 →
  -- Total investment
  total_investment = a + b + c →
  -- Total profit
  total_profit = 60000 →
  -- C's profit
  c / total_investment * total_profit = 20000 →
  -- Prove that A:B = 3:1
  a / b = 3 / 1 := by
sorry

end investment_ratio_a_to_b_l3217_321787


namespace outbound_time_calculation_l3217_321726

/-- The time taken for John to drive to the distant city -/
def outbound_time : ℝ := 30

/-- The time taken for John to return from the distant city -/
def return_time : ℝ := 5

/-- The speed increase on the return trip -/
def speed_increase : ℝ := 12

/-- The speed on the outbound trip -/
def outbound_speed : ℝ := 60

/-- The speed on the return trip -/
def return_speed : ℝ := outbound_speed + speed_increase

theorem outbound_time_calculation :
  outbound_time * outbound_speed = return_time * return_speed := by sorry

#check outbound_time_calculation

end outbound_time_calculation_l3217_321726


namespace train_problem_l3217_321703

/-- Calculates the number of people who got on a train given the initial count, 
    the number who got off, and the final count. -/
def peopleGotOn (initial : ℕ) (gotOff : ℕ) (final : ℕ) : ℕ :=
  final - (initial - gotOff)

theorem train_problem : peopleGotOn 78 27 63 = 12 := by
  sorry

end train_problem_l3217_321703


namespace smallest_cube_ending_584_l3217_321799

theorem smallest_cube_ending_584 :
  ∃ n : ℕ+, (n : ℤ)^3 ≡ 584 [ZMOD 1000] ∧
  ∀ m : ℕ+, (m : ℤ)^3 ≡ 584 [ZMOD 1000] → n ≤ m ∧ n = 34 :=
by sorry

end smallest_cube_ending_584_l3217_321799


namespace prob_valid_sequence_equals_377_4096_sum_numerator_denominator_l3217_321759

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Number of valid sequences of length n -/
def valid_sequences (n : ℕ) : ℕ := fib (n + 2)

/-- Total number of possible sequences of length n -/
def total_sequences (n : ℕ) : ℕ := 2^n

/-- The probability of a valid sequence of length 12 -/
def prob_valid_sequence : ℚ := (valid_sequences 12 : ℚ) / (total_sequences 12 : ℚ)

theorem prob_valid_sequence_equals_377_4096 :
  prob_valid_sequence = 377 / 4096 :=
sorry

theorem sum_numerator_denominator :
  377 + 4096 = 4473 :=
sorry

end prob_valid_sequence_equals_377_4096_sum_numerator_denominator_l3217_321759


namespace final_sum_after_operations_l3217_321722

theorem final_sum_after_operations (S a b : ℝ) (h : a + b = S) :
  3 * (a + 5) + 3 * (b + 7) = 3 * S + 36 := by
  sorry

end final_sum_after_operations_l3217_321722


namespace quadratic_inequality_l3217_321744

/-- Given a quadratic function f(x) = ax² + bx + c with a > 0, and roots α and β of f(x) = x 
    where 0 < α < β, prove that x < f(x) for all x such that 0 < x < α -/
theorem quadratic_inequality (a b c α β : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : a > 0)
  (h3 : f α = α)
  (h4 : f β = β)
  (h5 : 0 < α)
  (h6 : α < β) :
  ∀ x, 0 < x → x < α → x < f x :=
sorry

end quadratic_inequality_l3217_321744


namespace ellipse_condition_l3217_321763

/-- The equation represents an ellipse with foci on the x-axis -/
def is_ellipse_on_x_axis (k : ℝ) : Prop :=
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (2*k - 1) = 1) ∧
  (2 - k > 0) ∧ (2*k - 1 > 0) ∧ (2 - k > 2*k - 1)

theorem ellipse_condition (k : ℝ) :
  is_ellipse_on_x_axis k ↔ 1/2 < k ∧ k < 1 :=
sorry

end ellipse_condition_l3217_321763


namespace max_sin_x_value_l3217_321768

theorem max_sin_x_value (x y z : ℝ) 
  (h1 : Real.sin x = Real.cos y) 
  (h2 : Real.sin y = Real.cos z) 
  (h3 : Real.sin z = Real.cos x) : 
  ∃ (max_sin_x : ℝ), max_sin_x = Real.sqrt 2 / 2 ∧ 
    ∀ t, Real.sin t ≤ max_sin_x := by
  sorry

end max_sin_x_value_l3217_321768


namespace line_intersection_l3217_321723

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being contained in a plane
variable (contains : Plane → Line → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem line_intersection
  (a b c : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : contains α a)
  (h3 : contains β b)
  (h4 : c = intersect α β) :
  intersects c a ∨ intersects c b :=
sorry

end line_intersection_l3217_321723


namespace value_of_R_l3217_321718

theorem value_of_R : ∀ P Q R : ℚ, 
  P = 4014 / 2 →
  Q = P / 4 →
  R = P - Q →
  R = 1505.25 := by
sorry

end value_of_R_l3217_321718


namespace parabola_properties_l3217_321735

/-- Represents a parabola in a 2D Cartesian coordinate system -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * a * x

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in a 2D Cartesian coordinate system -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  equation : ℝ → ℝ → Prop := fun x y => a * x + b * y + c = 0

/-- Given a parabola C with vertex at (0, 0) and focus at (1, 0), 
    prove that its standard equation is y^2 = 4x and that for any two points 
    M and N on its directrix with y-coordinates y₁ and y₂ such that y₁y₂ = -4, 
    the line passing through the intersections of OM and ON with C 
    always contains the point (1, 0) -/
theorem parabola_properties (C : Parabola) 
  (h_vertex : C.equation 0 0)
  (h_focus : C.equation 1 0) :
  (C.equation = fun x y => y^2 = 4 * x) ∧
  (∀ y₁ y₂ : ℝ, y₁ * y₂ = -4 →
    ∃ (L : Line), 
      (∀ x y, L.equation x y → C.equation x y) ∧
      L.equation 1 0) := by
  sorry

end parabola_properties_l3217_321735


namespace roots_of_equation_l3217_321750

theorem roots_of_equation (x : ℝ) :
  x * (x - 3)^2 * (5 + x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := by
  sorry

end roots_of_equation_l3217_321750


namespace binomial_expansion_sum_l3217_321795

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a₀ + a₁ + a₃ = -39 := by
  sorry

end binomial_expansion_sum_l3217_321795


namespace triangle_vector_sum_l3217_321751

/-- Given a triangle ABC with points E and F, prove that x + y = -1/6 -/
theorem triangle_vector_sum (A B C E F : ℝ × ℝ) (x y : ℝ) : 
  (E - A : ℝ × ℝ) = (1/2 : ℝ) • (B - A) →
  (C - F : ℝ × ℝ) = 2 • (F - A) →
  (F - E : ℝ × ℝ) = x • (B - A) + y • (C - A) →
  x + y = -1/6 := by
  sorry

end triangle_vector_sum_l3217_321751


namespace ron_book_picks_l3217_321741

/-- Represents the number of times a person gets to pick a book in a year -/
def picks_per_year (total_members : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weeks_per_year / total_members

/-- The book club scenario -/
theorem ron_book_picks :
  let couples := 3
  let singles := 5
  let ron_and_wife := 2
  let total_members := couples * 2 + singles + ron_and_wife
  let weeks_per_year := 52
  picks_per_year total_members weeks_per_year = 4 := by
sorry

end ron_book_picks_l3217_321741


namespace rectangle_side_ratio_l3217_321743

theorem rectangle_side_ratio (a b c d : ℝ) (h1 : a * b / (c * d) = 0.16) (h2 : b / d = 2 / 5) :
  a / c = 0.4 := by
  sorry

end rectangle_side_ratio_l3217_321743


namespace prob_sum_six_is_five_thirty_sixths_l3217_321783

/-- The number of possible outcomes when throwing two dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum equals 6) when throwing two dice -/
def favorable_outcomes : ℕ := 5

/-- The probability of the sum of two fair dice equaling 6 -/
def prob_sum_six : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_six_is_five_thirty_sixths : 
  prob_sum_six = 5 / 36 := by sorry

end prob_sum_six_is_five_thirty_sixths_l3217_321783


namespace F_6_indeterminate_l3217_321755

theorem F_6_indeterminate (F : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (F k → F (k + 1)))
  (h2 : ¬ F 7) :
  (F 6 ∨ ¬ F 6) :=
sorry

end F_6_indeterminate_l3217_321755


namespace cylinder_lateral_area_l3217_321798

/-- The lateral area of a cylinder with base diameter and height both 4 cm is 16π cm² -/
theorem cylinder_lateral_area (π : ℝ) : 
  let base_diameter : ℝ := 4
  let height : ℝ := 4
  let lateral_area : ℝ := π * base_diameter * height
  lateral_area = 16 * π := by
sorry

end cylinder_lateral_area_l3217_321798


namespace discriminant_5x2_minus_8x_plus_1_l3217_321705

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of 5x^2 - 8x + 1 is 44 -/
theorem discriminant_5x2_minus_8x_plus_1 : discriminant 5 (-8) 1 = 44 := by
  sorry

end discriminant_5x2_minus_8x_plus_1_l3217_321705


namespace geometric_sequence_increasing_iff_first_three_l3217_321758

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- An increasing sequence -/
def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- First three terms increasing -/
def first_three_increasing (a : ℕ → ℝ) : Prop :=
  a 1 < a 2 ∧ a 2 < a 3

theorem geometric_sequence_increasing_iff_first_three (a : ℕ → ℝ) :
  geometric_sequence a →
  (increasing_sequence a ↔ first_three_increasing a) :=
sorry

end geometric_sequence_increasing_iff_first_three_l3217_321758


namespace four_digit_divisor_cyclic_iff_abab_l3217_321702

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def cyclic_shift (n : ℕ) : ℕ := 
  let d := n % 10
  let r := n / 10
  d * 1000 + r

def is_divisor_of_cyclic (n : ℕ) : Prop :=
  ∃ k, k * n = cyclic_shift n ∨ k * n = cyclic_shift (cyclic_shift n) ∨ k * n = cyclic_shift (cyclic_shift (cyclic_shift n))

def is_abab_form (n : ℕ) : Prop :=
  ∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ n = a * 1000 + b * 100 + a * 10 + b

theorem four_digit_divisor_cyclic_iff_abab (n : ℕ) :
  is_four_digit n ∧ is_divisor_of_cyclic n ↔ is_abab_form n := by sorry

end four_digit_divisor_cyclic_iff_abab_l3217_321702


namespace perpendicular_transitivity_l3217_321745

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- Define the statement
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (diff_lines : m ≠ n)
  (diff_planes : α ≠ β)
  (n_perp_α : perp n α)
  (n_perp_β : perp n β)
  (m_perp_β : perp m β) :
  perp m α :=
sorry

end perpendicular_transitivity_l3217_321745


namespace radius_of_circle_in_spherical_coordinates_l3217_321730

/-- The radius of the circle formed by points with spherical coordinates (2, θ, π/4) is √2 -/
theorem radius_of_circle_in_spherical_coordinates : 
  let ρ : ℝ := 2
  let φ : ℝ := π / 4
  Real.sqrt (ρ^2 * Real.sin φ^2) = Real.sqrt 2 := by
  sorry

end radius_of_circle_in_spherical_coordinates_l3217_321730


namespace fraction_to_decimal_l3217_321737

theorem fraction_to_decimal : (49 : ℚ) / (2^3 * 5^4) = 6.125 := by
  sorry

end fraction_to_decimal_l3217_321737


namespace divisibility_by_five_l3217_321774

theorem divisibility_by_five : ∃ k : ℤ, 3^444 + 4^333 = 5 * k := by
  sorry

end divisibility_by_five_l3217_321774


namespace lens_circumference_approx_l3217_321788

-- Define π as a constant (approximation)
def π : ℝ := 3.14159

-- Define the diameter of the lens
def d : ℝ := 10

-- Define the circumference calculation function
def circumference (diameter : ℝ) : ℝ := π * diameter

-- Theorem statement
theorem lens_circumference_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |circumference d - 31.42| < ε :=
sorry

end lens_circumference_approx_l3217_321788


namespace max_m_value_l3217_321784

-- Define the circle M
def circle_M (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + 5 = m}

-- Define points A and B
def point_A : ℝ × ℝ := (0, 4)
def point_B : ℝ × ℝ := (2, 0)

-- Define the property of right angle APB
def is_right_angle (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - point_A.1, P.2 - point_A.2)
  let BP := (P.1 - point_B.1, P.2 - point_B.2)
  AP.1 * BP.1 + AP.2 * BP.2 = 0

-- Theorem statement
theorem max_m_value :
  ∃ (m : ℝ), ∀ (m' : ℝ),
    (∃ (P : ℝ × ℝ), P ∈ circle_M m' ∧ is_right_angle P) →
    m' ≤ m ∧
    m = 45 :=
sorry

end max_m_value_l3217_321784


namespace fifteenth_digit_sum_one_ninth_one_eleventh_l3217_321720

/-- The decimal representation of a rational number -/
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

/-- The sum of decimal representations of two rational numbers -/
def sumDecimalRepresentations (q₁ q₂ : ℚ) : ℕ → ℕ := sorry

/-- The nth digit after the decimal point in a decimal representation -/
def nthDigitAfterDecimal (rep : ℕ → ℕ) (n : ℕ) : ℕ := sorry

theorem fifteenth_digit_sum_one_ninth_one_eleventh :
  nthDigitAfterDecimal (sumDecimalRepresentations (1/9) (1/11)) 15 = 1 := by sorry

end fifteenth_digit_sum_one_ninth_one_eleventh_l3217_321720


namespace equation_solutions_l3217_321762

theorem equation_solutions :
  (∃ x : ℝ, 3 * x + 6 = 31 - 2 * x ∧ x = 5) ∧
  (∃ x : ℝ, 1 - 8 * (1/4 + 0.5 * x) = 3 * (1 - 2 * x) ∧ x = 2) := by
  sorry

end equation_solutions_l3217_321762


namespace area_ratio_hexagon_octagon_l3217_321761

noncomputable def hexagon_area_between_circles (side_length : ℝ) : ℝ :=
  Real.pi * (11 / 3) * side_length^2

noncomputable def octagon_circumradius (side_length : ℝ) : ℝ :=
  side_length * (2 * Real.sqrt 2) / Real.sqrt (2 - Real.sqrt 2)

noncomputable def octagon_area_between_circles (side_length : ℝ) : ℝ :=
  Real.pi * ((octagon_circumradius side_length)^2 - (3 + 2 * Real.sqrt 2) * side_length^2)

theorem area_ratio_hexagon_octagon (side_length : ℝ) (h : side_length > 0) :
  hexagon_area_between_circles side_length / octagon_area_between_circles side_length =
  11 / (3 * ((octagon_circumradius 1)^2 - (3 + 2 * Real.sqrt 2))) :=
by sorry

end area_ratio_hexagon_octagon_l3217_321761


namespace group_size_proof_l3217_321791

/-- The number of people in a group where:
    1. The total weight increase is 2.5 kg times the number of people.
    2. The weight difference between the new person and the replaced person is 20 kg. -/
def number_of_people : ℕ := 8

theorem group_size_proof :
  ∃ (n : ℕ), n = number_of_people ∧ 
  (2.5 : ℝ) * n = (20 : ℝ) := by
  sorry

end group_size_proof_l3217_321791


namespace card_count_proof_l3217_321756

/-- The number of cards Sasha added to the box -/
def cards_added : ℕ := 48

/-- The fraction of cards Karen removed from what Sasha added -/
def removal_fraction : ℚ := 1 / 6

/-- The number of cards in the box after Sasha's and Karen's actions -/
def final_card_count : ℕ := 83

/-- The original number of cards in the box -/
def original_card_count : ℕ := 75

theorem card_count_proof :
  (cards_added : ℚ) - removal_fraction * cards_added + original_card_count = final_card_count :=
sorry

end card_count_proof_l3217_321756


namespace sean_houses_problem_l3217_321748

theorem sean_houses_problem (initial_houses : ℕ) : 
  initial_houses - 8 + 12 = 31 → initial_houses = 27 := by
  sorry

end sean_houses_problem_l3217_321748


namespace jerrys_shelf_l3217_321785

/-- The number of books on Jerry's shelf -/
def num_books : ℕ := 3

/-- The initial number of action figures -/
def initial_action_figures : ℕ := 4

/-- The number of action figures added -/
def added_action_figures : ℕ := 2

/-- The difference between action figures and books -/
def difference : ℕ := 3

theorem jerrys_shelf :
  num_books = 3 ∧
  initial_action_figures + added_action_figures = num_books + difference :=
sorry

end jerrys_shelf_l3217_321785


namespace emily_trivia_score_l3217_321728

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) : 
  first_round + 33 - 48 = 1 → first_round = 16 := by
  sorry

end emily_trivia_score_l3217_321728


namespace gary_paycheck_l3217_321766

/-- Calculates the total paycheck for an employee with overtime --/
def calculate_paycheck (normal_wage : ℚ) (total_hours : ℕ) (regular_hours : ℕ) (overtime_multiplier : ℚ) : ℚ :=
  let regular_pay := normal_wage * regular_hours
  let overtime_hours := total_hours - regular_hours
  let overtime_pay := normal_wage * overtime_multiplier * overtime_hours
  regular_pay + overtime_pay

/-- Gary's paycheck calculation --/
theorem gary_paycheck :
  let normal_wage : ℚ := 12
  let total_hours : ℕ := 52
  let regular_hours : ℕ := 40
  let overtime_multiplier : ℚ := 3/2
  calculate_paycheck normal_wage total_hours regular_hours overtime_multiplier = 696 := by
  sorry


end gary_paycheck_l3217_321766


namespace same_terminal_side_l3217_321789

/-- Two angles have the same terminal side if their difference is a multiple of 360 degrees -/
def SameTerminalSide (a b : ℝ) : Prop := ∃ k : ℤ, a - b = k * 360

/-- The angle -510 degrees -/
def angle1 : ℝ := -510

/-- The angle 210 degrees -/
def angle2 : ℝ := 210

/-- Theorem: angle1 and angle2 have the same terminal side -/
theorem same_terminal_side : SameTerminalSide angle1 angle2 := by sorry

end same_terminal_side_l3217_321789


namespace polygon_sides_l3217_321711

/-- A convex polygon with the sum of all angles except one equal to 2790° has 18 sides -/
theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  n ≥ 3 → -- convex polygon has at least 3 sides
  angle_sum = 2790 → -- sum of all angles except one is 2790°
  (n - 2) * 180 - angle_sum ≥ 0 → -- the missing angle is non-negative
  (n - 2) * 180 - angle_sum < 180 → -- the missing angle is less than 180°
  n = 18 := by
sorry

end polygon_sides_l3217_321711


namespace problem_statement_l3217_321742

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - a

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 1

theorem problem_statement (a : ℝ) :
  -- Part 1
  (f a 0 = f a 1) →
  (Set.Icc (-1) (1/2) = {x ∈ domain | |f a x - 1| < a * x + 3/4}) ∧
  -- Part 2
  (|a| ≤ 1) →
  (∀ x ∈ domain, |f a x| ≤ 5/4) :=
by sorry


end problem_statement_l3217_321742


namespace time_per_braid_l3217_321794

/-- The time it takes to braid one braid, given the number of dancers, braids per dancer, and total time -/
theorem time_per_braid (num_dancers : ℕ) (braids_per_dancer : ℕ) (total_time_minutes : ℕ) : 
  num_dancers = 8 → 
  braids_per_dancer = 5 → 
  total_time_minutes = 20 → 
  (total_time_minutes * 60) / (num_dancers * braids_per_dancer) = 30 := by
  sorry

#check time_per_braid

end time_per_braid_l3217_321794


namespace two_machines_total_copies_l3217_321712

/-- Represents a copy machine with a constant copying rate. -/
structure CopyMachine where
  rate : ℕ  -- Copies per minute

/-- Calculates the total number of copies made by a machine in a given time. -/
def copies_made (machine : CopyMachine) (minutes : ℕ) : ℕ :=
  machine.rate * minutes

/-- Represents the problem setup with two copy machines. -/
structure TwoMachineProblem where
  machine1 : CopyMachine
  machine2 : CopyMachine
  duration : ℕ  -- Duration in minutes

/-- The main theorem stating the total number of copies made by both machines. -/
theorem two_machines_total_copies (problem : TwoMachineProblem)
    (h1 : problem.machine1.rate = 40)
    (h2 : problem.machine2.rate = 55)
    (h3 : problem.duration = 30) :
    copies_made problem.machine1 problem.duration +
    copies_made problem.machine2 problem.duration = 2850 := by
  sorry

end two_machines_total_copies_l3217_321712


namespace melanie_plums_theorem_l3217_321780

/-- Represents the number of plums Melanie has -/
def plums_remaining (initial_plums : ℕ) (plums_given_away : ℕ) : ℕ :=
  initial_plums - plums_given_away

/-- Theorem stating that Melanie's remaining plums are correctly calculated -/
theorem melanie_plums_theorem (initial_plums : ℕ) (plums_given_away : ℕ) 
  (h : initial_plums ≥ plums_given_away) :
  plums_remaining initial_plums plums_given_away = initial_plums - plums_given_away :=
by sorry

end melanie_plums_theorem_l3217_321780


namespace sequence_problem_l3217_321709

theorem sequence_problem (a : ℕ → ℕ) (n : ℕ) : 
  a 1 = 2 ∧ 
  (∀ k ≥ 1, a (k + 1) = a k + 3) ∧ 
  a n = 2009 →
  n = 670 := by sorry

end sequence_problem_l3217_321709


namespace mike_investment_l3217_321736

/-- Prove that Mike's investment is $350 given the partnership conditions --/
theorem mike_investment (mary_investment : ℝ) (total_profit : ℝ) (profit_difference : ℝ) :
  mary_investment = 650 →
  total_profit = 2999.9999999999995 →
  profit_difference = 600 →
  ∃ (mike_investment : ℝ),
    mike_investment = 350 ∧
    (1/3 * total_profit / 2 + 2/3 * total_profit * mary_investment / (mary_investment + mike_investment) =
     1/3 * total_profit / 2 + 2/3 * total_profit * mike_investment / (mary_investment + mike_investment) + profit_difference) :=
by sorry

end mike_investment_l3217_321736


namespace initial_book_donations_l3217_321729

/-- Proves that the initial number of book donations is 300 given the conditions of the problem. -/
theorem initial_book_donations (
  people_donating : ℕ)
  (books_per_person : ℕ)
  (books_borrowed : ℕ)
  (remaining_books : ℕ)
  (h1 : people_donating = 10)
  (h2 : books_per_person = 5)
  (h3 : books_borrowed = 140)
  (h4 : remaining_books = 210) :
  people_donating * books_per_person + remaining_books + books_borrowed = 300 :=
by sorry


end initial_book_donations_l3217_321729


namespace flat_fee_is_40_l3217_321773

/-- A hotel pricing structure with a flat fee for the first night and a fixed amount for each additional night. -/
structure HotelPricing where
  flatFee : ℝ
  additionalNightFee : ℝ

/-- Calculate the total cost for a stay given the pricing structure and number of nights. -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.additionalNightFee * (nights - 1)

/-- The flat fee for the first night is $40 given the conditions. -/
theorem flat_fee_is_40 :
  ∃ (pricing : HotelPricing),
    totalCost pricing 4 = 195 ∧
    totalCost pricing 7 = 350 ∧
    pricing.flatFee = 40 := by
  sorry

end flat_fee_is_40_l3217_321773


namespace ceiling_minus_y_is_half_l3217_321764

theorem ceiling_minus_y_is_half (x : ℝ) (y : ℝ) 
  (h1 : ⌈x⌉ - ⌊x⌋ = 0) 
  (h2 : y = x + 1/2) : 
  ⌈y⌉ - y = 1/2 := by
  sorry

end ceiling_minus_y_is_half_l3217_321764
