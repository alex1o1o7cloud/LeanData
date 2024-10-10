import Mathlib

namespace arithmetic_operations_l2159_215958

theorem arithmetic_operations :
  (3 - (-2) = 5) ∧
  ((-4) * (-3) = 12) ∧
  (0 / (-3) = 0) ∧
  (|(-12)| + (-4) = 8) ∧
  ((3) - 14 - (-5) + (-16) = -22) ∧
  ((-5) / (-1/5) * (-5) = -125) ∧
  (-24 * ((-5/6) + (3/8) - (1/12)) = 13) ∧
  (3 * (-4) + 18 / (-6) - (-2) = -12) ∧
  ((-99 - 15/16) * 4 = -399 - 3/4) := by
sorry

#eval 3 - (-2)
#eval (-4) * (-3)
#eval 0 / (-3)
#eval |(-12)| + (-4)
#eval 3 - 14 - (-5) + (-16)
#eval (-5) / (-1/5) * (-5)
#eval -24 * ((-5/6) + (3/8) - (1/12))
#eval 3 * (-4) + 18 / (-6) - (-2)
#eval (-99 - 15/16) * 4

end arithmetic_operations_l2159_215958


namespace integral_one_plus_sin_over_pi_halves_l2159_215970

open Real MeasureTheory

theorem integral_one_plus_sin_over_pi_halves : 
  ∫ x in (-π/2)..(π/2), (1 + Real.sin x) = π := by sorry

end integral_one_plus_sin_over_pi_halves_l2159_215970


namespace coprime_condition_l2159_215915

theorem coprime_condition (a b c d : ℤ) (h : Nat.gcd a.natAbs b.natAbs = 1 ∧ 
  Nat.gcd c.natAbs d.natAbs = 1 ∧ Nat.gcd a.natAbs c.natAbs = 1) : 
  (∀ (p : ℕ), Nat.Prime p → p ∣ (a * d - b * c).natAbs → (p ∣ a.natAbs ∨ p ∣ c.natAbs)) ↔ 
  (∀ (n : ℤ), Nat.gcd (a * n + b).natAbs (c * n + d).natAbs = 1) := by
sorry

end coprime_condition_l2159_215915


namespace base8_237_equals_base10_159_l2159_215930

/-- Converts a three-digit number from base 8 to base 10 -/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The base 8 number 237 is equal to 159 in base 10 -/
theorem base8_237_equals_base10_159 : base8ToBase10 2 3 7 = 159 := by
  sorry

end base8_237_equals_base10_159_l2159_215930


namespace jelly_bean_distribution_l2159_215983

theorem jelly_bean_distribution (total_jelly_beans : ℕ) (remaining_jelly_beans : ℕ) (boy_girl_difference : ℕ) : 
  total_jelly_beans = 500 →
  remaining_jelly_beans = 10 →
  boy_girl_difference = 4 →
  ∃ (girls boys : ℕ),
    girls + boys = 32 ∧
    boys = girls + boy_girl_difference ∧
    girls * girls + boys * boys = total_jelly_beans - remaining_jelly_beans :=
by sorry

end jelly_bean_distribution_l2159_215983


namespace flower_shop_bouquets_l2159_215913

theorem flower_shop_bouquets (roses_per_bouquet : ℕ) 
  (rose_bouquets_sold daisy_bouquets_sold total_flowers : ℕ) :
  roses_per_bouquet = 12 →
  rose_bouquets_sold = 10 →
  daisy_bouquets_sold = 10 →
  total_flowers = 190 →
  total_flowers = roses_per_bouquet * rose_bouquets_sold + 
    (total_flowers - roses_per_bouquet * rose_bouquets_sold) →
  rose_bouquets_sold + daisy_bouquets_sold = 20 :=
by
  sorry

end flower_shop_bouquets_l2159_215913


namespace largest_n_binomial_sum_existence_n_6_largest_n_is_6_l2159_215978

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem existence_n_6 : 
  Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 6 :=
by sorry

theorem largest_n_is_6 : 
  ∃ (n : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) ∧ 
  (∀ (m : ℕ), (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ n) ∧
  n = 6 :=
by sorry

end largest_n_binomial_sum_existence_n_6_largest_n_is_6_l2159_215978


namespace complex_multiplication_result_l2159_215981

theorem complex_multiplication_result : 
  let i : ℂ := Complex.I
  (3 - 4*i) * (2 + 6*i) * (-1 + 2*i) = -50 + 50*i := by sorry

end complex_multiplication_result_l2159_215981


namespace circumscribed_circle_diameter_l2159_215948

theorem circumscribed_circle_diameter 
  (side : ℝ) (angle : ℝ) (h1 : side = 15) (h2 : angle = π / 4) :
  (side / Real.sin angle) = 15 * Real.sqrt 2 := by
  sorry

end circumscribed_circle_diameter_l2159_215948


namespace smaller_number_in_ratio_l2159_215963

theorem smaller_number_in_ratio (p q k : ℝ) (hp : p > 0) (hq : q > 0) : 
  p / q = 3 / 5 → p^2 + q^2 = 2 * k → min p q = 3 * Real.sqrt (k / 17) := by
  sorry

end smaller_number_in_ratio_l2159_215963


namespace main_theorem_l2159_215936

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f (f y))^2)

/-- The main theorem to prove -/
theorem main_theorem (f : ℝ → ℝ) (h : FunctionalEquation f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x :=
sorry

end main_theorem_l2159_215936


namespace distance_sf_to_atlantis_l2159_215941

theorem distance_sf_to_atlantis : 
  let sf : ℂ := 0
  let atlantis : ℂ := 1300 + 3120 * I
  Complex.abs (atlantis - sf) = 3380 := by
sorry

end distance_sf_to_atlantis_l2159_215941


namespace imaginary_part_of_complex_number_l2159_215917

theorem imaginary_part_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z := i^2 / (1 - i)
  (z.im : ℝ) = -1/2 := by sorry

end imaginary_part_of_complex_number_l2159_215917


namespace angle_problem_l2159_215956

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define complementary angles
def complementary (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes = 90 * 60

-- Define supplementary angles
def supplementary (a b : Angle) : Prop :=
  a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes = 180 * 60

-- State the theorem
theorem angle_problem (angle1 angle2 angle3 : Angle) :
  complementary angle1 angle2 →
  supplementary angle2 angle3 →
  angle1 = Angle.mk 67 12 →
  angle3 = Angle.mk 157 12 :=
by sorry

end angle_problem_l2159_215956


namespace custom_op_equality_l2159_215972

/-- Custom operation ⊗ -/
def custom_op (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality of the expression and its simplified form -/
theorem custom_op_equality (a b : ℝ) : 
  custom_op a b + custom_op (b - a) b = b^2 - b := by sorry

end custom_op_equality_l2159_215972


namespace total_dolls_l2159_215985

/-- The number of dolls each person has -/
structure DollCounts where
  vera : ℕ
  sophie : ℕ
  aida : ℕ

/-- The conditions of the doll distribution -/
def doll_distribution (d : DollCounts) : Prop :=
  d.vera = 20 ∧ d.sophie = 2 * d.vera ∧ d.aida = 2 * d.sophie

/-- The theorem stating the total number of dolls -/
theorem total_dolls (d : DollCounts) (h : doll_distribution d) : 
  d.vera + d.sophie + d.aida = 140 := by
  sorry

#check total_dolls

end total_dolls_l2159_215985


namespace apollonian_circle_m_range_l2159_215903

theorem apollonian_circle_m_range :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (2, 0)
  let C (m : ℝ) := {P : ℝ × ℝ | (P.1 - 2)^2 + (P.2 - m)^2 = 1/4}
  ∀ m > 0, (∃ P ∈ C m, dist P A = 2 * dist P B) →
    m ∈ Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 21 / 2) :=
by sorry


end apollonian_circle_m_range_l2159_215903


namespace product_markup_rate_l2159_215937

theorem product_markup_rate (selling_price : ℝ) (profit_rate : ℝ) (expense_rate : ℝ) (fixed_cost : ℝ) :
  selling_price = 10 ∧ 
  profit_rate = 0.20 ∧ 
  expense_rate = 0.30 ∧ 
  fixed_cost = 1 →
  let variable_cost := selling_price * (1 - profit_rate - expense_rate) - fixed_cost
  (selling_price - variable_cost) / variable_cost = 1.5 := by sorry

end product_markup_rate_l2159_215937


namespace range_of_a_l2159_215960

-- Define the quadratic equation
def quadratic_eq (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2*a + 6

-- Define the property of having one positive and one negative root
def has_pos_neg_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
  quadratic_eq a x₁ = 0 ∧ quadratic_eq a x₂ = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_pos_neg_roots a ↔ a < -3 :=
sorry

end range_of_a_l2159_215960


namespace remaining_money_l2159_215939

def salary : ℝ := 8123.08

theorem remaining_money (food_fraction : ℝ) (rent_fraction : ℝ) (clothes_fraction : ℝ)
  (h_food : food_fraction = 1/3)
  (h_rent : rent_fraction = 1/4)
  (h_clothes : clothes_fraction = 1/5) :
  let total_expenses := salary * (food_fraction + rent_fraction + clothes_fraction)
  ∃ ε > 0, |salary - total_expenses - 1759.00| < ε :=
by sorry

end remaining_money_l2159_215939


namespace company_attendees_l2159_215996

theorem company_attendees (total : ℕ) (other : ℕ) (h_total : total = 185) (h_other : other = 20) : 
  ∃ (a : ℕ), 
    a + (2 * a) + (a + 10) + (a + 5) + other = total ∧ 
    a = 30 := by
  sorry

end company_attendees_l2159_215996


namespace power_of_three_implies_large_prime_factor_l2159_215973

theorem power_of_three_implies_large_prime_factor (n : ℕ+) :
  (∃ k : ℕ, 125 * n + 22 = 3^k) →
  ∃ p : ℕ, p > 100 ∧ Prime p ∧ p ∣ (125 * n + 29) :=
by sorry

end power_of_three_implies_large_prime_factor_l2159_215973


namespace exists_special_quadratic_trinomial_l2159_215927

/-- A quadratic trinomial function -/
def QuadraticTrinomial := ℝ → ℝ

/-- The n-th composition of a function with itself -/
def compose_n_times (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0 => id
| n + 1 => f ∘ (compose_n_times f n)

/-- The number of distinct real roots of a function -/
noncomputable def num_distinct_real_roots (f : ℝ → ℝ) : ℕ := sorry

/-- The main theorem statement -/
theorem exists_special_quadratic_trinomial :
  ∃ (f : QuadraticTrinomial),
    ∀ (n : ℕ), num_distinct_real_roots (compose_n_times f n) = 2 * n :=
sorry

end exists_special_quadratic_trinomial_l2159_215927


namespace zero_exponent_is_one_l2159_215965

theorem zero_exponent_is_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zero_exponent_is_one_l2159_215965


namespace election_win_margin_l2159_215990

theorem election_win_margin :
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
    winner_votes = 837 →
    winner_votes = (62 * total_votes) / 100 →
    loser_votes = total_votes - winner_votes →
    winner_votes - loser_votes = 324 := by
  sorry

end election_win_margin_l2159_215990


namespace hannahs_brothers_l2159_215929

theorem hannahs_brothers (num_brothers : ℕ) : num_brothers = 3 :=
  by
  -- Hannah has some brothers
  have h1 : num_brothers > 0 := by sorry
  
  -- All her brothers are 8 years old
  let brother_age := 8
  
  -- Hannah is 48 years old
  let hannah_age := 48
  
  -- Hannah's age is twice the sum of her brothers' ages
  have h2 : hannah_age = 2 * (num_brothers * brother_age) := by sorry
  
  -- Proof that num_brothers = 3
  sorry

end hannahs_brothers_l2159_215929


namespace lcm_18_35_l2159_215959

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end lcm_18_35_l2159_215959


namespace consecutive_odd_numbers_l2159_215925

theorem consecutive_odd_numbers (n : ℕ) 
  (h_avg : (27 + 27 - 2 * (n - 1)) / 2 = 24) 
  (h_largest : 27 = 27 - 2 * (n - 1) + 2 * (n - 1)) : n = 4 := by
  sorry

end consecutive_odd_numbers_l2159_215925


namespace jake_brought_four_balloons_l2159_215975

/-- The number of balloons Allan brought -/
def allan_balloons : ℕ := 2

/-- The total number of balloons Allan and Jake had -/
def total_balloons : ℕ := 6

/-- The number of balloons Jake brought -/
def jake_balloons : ℕ := total_balloons - allan_balloons

theorem jake_brought_four_balloons : jake_balloons = 4 := by
  sorry

end jake_brought_four_balloons_l2159_215975


namespace unique_prime_ending_l2159_215982

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number (A : ℕ) : ℕ := 202100 + A

theorem unique_prime_ending :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) :=
sorry

end unique_prime_ending_l2159_215982


namespace rectangle_area_l2159_215992

/-- Theorem: Area of a rectangle with length 15 cm and width 0.9 times its length -/
theorem rectangle_area (length : ℝ) (width : ℝ) : 
  length = 15 →
  width = 0.9 * length →
  length * width = 202.5 := by
  sorry

end rectangle_area_l2159_215992


namespace exists_m_n_satisfying_equation_l2159_215991

-- Define the "*" operation
def star_op (a b : ℤ) : ℤ :=
  if a = 0 ∨ b = 0 then
    max (a^2) (b^2)
  else
    (if a * b > 0 then 1 else -1) * (a^2 + b^2)

-- Theorem statement
theorem exists_m_n_satisfying_equation :
  ∃ (m n : ℤ), star_op (m - 1) (n + 2) = -2 :=
sorry

end exists_m_n_satisfying_equation_l2159_215991


namespace complement_of_A_l2159_215901

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Theorem statement
theorem complement_of_A : 
  (U \ A) = {x : ℝ | x < -1 ∨ x ≥ 2} := by sorry

end complement_of_A_l2159_215901


namespace sqrt_two_minus_one_power_l2159_215974

theorem sqrt_two_minus_one_power (n : ℕ+) :
  ∃ (a b : ℤ) (m : ℕ),
    (Real.sqrt 2 - 1) ^ (n : ℝ) = b * Real.sqrt 2 - a ∧
    m = a ^ 2 * b ^ 2 + 1 ∧
    m > 1 ∧
    (Real.sqrt 2 - 1) ^ (n : ℝ) = Real.sqrt m - Real.sqrt (m - 1) := by
  sorry

end sqrt_two_minus_one_power_l2159_215974


namespace binary_1011011_equals_91_l2159_215984

def binary_to_decimal (binary_digits : List Bool) : ℕ :=
  binary_digits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1011011_equals_91 :
  binary_to_decimal [true, true, false, true, true, false, true] = 91 := by
  sorry

end binary_1011011_equals_91_l2159_215984


namespace quadratic_function_product_sign_l2159_215968

theorem quadratic_function_product_sign
  (a b c m n p x₁ x₂ : ℝ)
  (h_a_pos : a > 0)
  (h_roots : x₁ < x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0)
  (h_order : m < x₁ ∧ x₁ < n ∧ n < x₂ ∧ x₂ < p) :
  let f := fun x => a * x^2 + b * x + c
  f m * f n * f p < 0 :=
by sorry

end quadratic_function_product_sign_l2159_215968


namespace inverse_of_f_l2159_215943

def f (x : ℝ) : ℝ := 3 - 4 * x

theorem inverse_of_f :
  ∃ g : ℝ → ℝ, (∀ x, g (f x) = x) ∧ (∀ x, f (g x) = x) ∧ (∀ x, g x = (3 - x) / 4) :=
by sorry

end inverse_of_f_l2159_215943


namespace sum_of_roots_equals_negative_eight_l2159_215931

/-- An odd function satisfying f(x-4) = -f(x) -/
def OddPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x - 4) = -f x)

theorem sum_of_roots_equals_negative_eight
  (f : ℝ → ℝ) (m : ℝ) (x₁ x₂ x₃ x₄ : ℝ)
  (hf : OddPeriodicFunction f)
  (hm : m > 0)
  (h_roots : x₁ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₂ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₃ ∈ Set.Icc (-8 : ℝ) 8 ∧
             x₄ ∈ Set.Icc (-8 : ℝ) 8)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄)
  (h_eq : f x₁ = m ∧ f x₂ = m ∧ f x₃ = m ∧ f x₄ = m) :
  x₁ + x₂ + x₃ + x₄ = -8 := by
  sorry


end sum_of_roots_equals_negative_eight_l2159_215931


namespace ttakji_square_arrangement_l2159_215906

/-- The number of ttakjis on one side of the large square -/
def n : ℕ := 61

/-- The number of ttakjis on the perimeter of the large square -/
def perimeter_ttakjis : ℕ := 240

theorem ttakji_square_arrangement :
  (4 * n - 4 = perimeter_ttakjis) ∧ (n^2 = 3721) := by sorry

end ttakji_square_arrangement_l2159_215906


namespace quadratic_one_root_from_geometric_sequence_l2159_215932

/-- If a, b, c form a geometric sequence of real numbers, then ax^2 + bx + c has exactly one real root -/
theorem quadratic_one_root_from_geometric_sequence (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) → 
  ∃! x : ℝ, a * x^2 + b * x + c = 0 :=
sorry

end quadratic_one_root_from_geometric_sequence_l2159_215932


namespace pure_imaginary_condition_l2159_215969

theorem pure_imaginary_condition (a : ℝ) : 
  (∀ z : ℂ, z = (a^2 - 2*a - 3 : ℝ) + (a + 1 : ℝ) * I → z.re = 0 ∧ z.im ≠ 0) → 
  a = 3 :=
sorry

end pure_imaginary_condition_l2159_215969


namespace function_inequality_implies_a_bound_l2159_215905

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 2) - x^2

theorem function_inequality_implies_a_bound :
  ∀ a : ℝ,
  (∀ p q : ℝ, 0 < q ∧ q < p ∧ p < 1 →
    (f a (p + 1) - f a (q + 1)) / (p - q) > 2) →
  a ≥ 24 :=
sorry

end function_inequality_implies_a_bound_l2159_215905


namespace smallest_b_for_equation_exists_solution_unique_smallest_solution_l2159_215916

theorem smallest_b_for_equation (A B : ℕ) : 
  (360 / (A * A * A / B) = 5) → B ≥ 3 :=
by
  sorry

theorem exists_solution : 
  ∃ (A B : ℕ), (360 / (A * A * A / B) = 5) ∧ B = 3 :=
by
  sorry

theorem unique_smallest_solution (A B : ℕ) : 
  (360 / (A * A * A / B) = 5) → B ≥ 3 :=
by
  sorry

end smallest_b_for_equation_exists_solution_unique_smallest_solution_l2159_215916


namespace range_of_a_l2159_215961

def proposition_p (a x : ℝ) : Prop :=
  x^2 - 4*a*x + 3*a^2 < 0 ∧ a < 0

def proposition_q (x : ℝ) : Prop :=
  x^2 - x - 6 ≤ 0 ∨ x^2 + 2*x - 8 > 0

def negation_p_necessary_not_sufficient_for_negation_q (a : ℝ) : Prop :=
  ∀ x, ¬(proposition_q x) → ¬(proposition_p a x) ∧
  ∃ x, ¬(proposition_p a x) ∧ proposition_q x

theorem range_of_a :
  ∀ a : ℝ, negation_p_necessary_not_sufficient_for_negation_q a →
  (a < 0 ∧ a > -4) ∨ a ≤ -4 :=
sorry

end range_of_a_l2159_215961


namespace absolute_value_equation_l2159_215918

theorem absolute_value_equation (x : ℝ) (h : |2 - x| = 2 + |x|) : |2 - x| = 2 - x := by
  sorry

end absolute_value_equation_l2159_215918


namespace monster_hunt_l2159_215977

theorem monster_hunt (x : ℕ) : 
  (x + 2*x + 4*x + 8*x + 16*x = 62) → x = 2 := by
  sorry

end monster_hunt_l2159_215977


namespace y_divisibility_l2159_215953

def y : ℕ := 72 + 108 + 144 + 180 + 324 + 396 + 3600

theorem y_divisibility :
  (∃ k : ℕ, y = 6 * k) ∧
  (∃ k : ℕ, y = 12 * k) ∧
  (∃ k : ℕ, y = 18 * k) ∧
  (∃ k : ℕ, y = 36 * k) := by
  sorry

end y_divisibility_l2159_215953


namespace probability_at_least_four_out_of_five_l2159_215921

theorem probability_at_least_four_out_of_five (p : ℝ) (h : p = 4/5) :
  let binomial (n k : ℕ) := Nat.choose n k
  let prob_exactly (k : ℕ) := (binomial 5 k : ℝ) * p^k * (1 - p)^(5 - k)
  prob_exactly 4 + prob_exactly 5 = 2304/3125 := by
  sorry

end probability_at_least_four_out_of_five_l2159_215921


namespace prize_selection_ways_l2159_215924

/-- The number of ways to select prize winners from finalists -/
def select_winners (n : ℕ) : ℕ :=
  n * (n - 1).choose 2

/-- Theorem stating that selecting 1 first prize, 2 second prizes, and 3 third prizes
    from 6 finalists can be done in 60 ways -/
theorem prize_selection_ways : select_winners 6 = 60 := by
  sorry

end prize_selection_ways_l2159_215924


namespace gcd_lcm_sum_60_429_l2159_215979

theorem gcd_lcm_sum_60_429 : Nat.gcd 60 429 + Nat.lcm 60 429 = 8583 := by
  sorry

end gcd_lcm_sum_60_429_l2159_215979


namespace quadratic_equation_roots_l2159_215999

theorem quadratic_equation_roots (a b c : ℝ) (h1 : a ≠ 0) :
  let r1 := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r2 := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (a = 1 ∧ b = -6 ∧ c = -7) →
  (r1 + r2 = 6 ∧ r1 - r2 = 8) :=
by sorry

end quadratic_equation_roots_l2159_215999


namespace symbol_values_l2159_215919

theorem symbol_values (star circle ring : ℤ) 
  (h1 : star + ring = 46)
  (h2 : star + circle = 91)
  (h3 : circle + ring = 63) :
  star = 37 ∧ circle = 54 ∧ ring = 9 := by
  sorry

end symbol_values_l2159_215919


namespace complex_equation_solution_l2159_215904

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2 + Complex.I) :
  z = 1/2 + 3/2 * Complex.I := by
  sorry

end complex_equation_solution_l2159_215904


namespace perpendicular_parallel_implies_perpendicular_parallel_planes_line_in_plane_implies_parallel_parallel_lines_planes_implies_equal_angles_l2159_215989

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def angle_with_plane (l : Line) (p : Plane) : ℝ := sorry

-- Theorem statements
theorem perpendicular_parallel_implies_perpendicular 
  (m n : Line) (α : Plane) :
  perpendicular m α → parallel_lines n α → perpendicular m n := by sorry

theorem parallel_planes_line_in_plane_implies_parallel 
  (m : Line) (α β : Plane) :
  parallel_planes α β → line_in_plane m α → parallel_lines m β := by sorry

theorem parallel_lines_planes_implies_equal_angles 
  (m n : Line) (α β : Plane) :
  parallel_lines m n → parallel_planes α β → 
  angle_with_plane m α = angle_with_plane n β := by sorry

end perpendicular_parallel_implies_perpendicular_parallel_planes_line_in_plane_implies_parallel_parallel_lines_planes_implies_equal_angles_l2159_215989


namespace carol_initial_peanuts_l2159_215980

/-- The number of peanuts Carol initially collected -/
def initial_peanuts : ℕ := sorry

/-- The number of peanuts Carol's father gave her -/
def fathers_peanuts : ℕ := 5

/-- The total number of peanuts Carol has after receiving peanuts from her father -/
def total_peanuts : ℕ := 7

/-- Theorem: Carol initially collected 2 peanuts -/
theorem carol_initial_peanuts : 
  initial_peanuts + fathers_peanuts = total_peanuts ∧ initial_peanuts = 2 :=
by sorry

end carol_initial_peanuts_l2159_215980


namespace quadratic_and_slope_l2159_215951

-- Define the quadratic polynomial
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def passes_through (a b c : ℝ) : Prop :=
  quadratic a b c 1 = -2 ∧
  quadratic a b c 2 = 4 ∧
  quadratic a b c 3 = 10

-- Define the slope of the tangent line
def tangent_slope (a b c : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

-- Theorem statement
theorem quadratic_and_slope :
  ∃ a b c : ℝ,
    passes_through a b c ∧
    (∀ x : ℝ, quadratic a b c x = 6 * x - 8) ∧
    tangent_slope a b c 2 = 6 := by sorry

end quadratic_and_slope_l2159_215951


namespace claras_weight_l2159_215911

theorem claras_weight (alice_weight clara_weight : ℝ) 
  (h1 : alice_weight + clara_weight = 240)
  (h2 : clara_weight - alice_weight = (2/3) * clara_weight) : 
  clara_weight = 180 := by
sorry

end claras_weight_l2159_215911


namespace length_of_AC_l2159_215947

/-- Given a quadrilateral ABCD with specific side lengths, prove the length of AC -/
theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 12) (h2 : DC = 15) (h3 : AD = 9) :
  ∃ (AC : ℝ), AC^2 = 585 := by sorry

end length_of_AC_l2159_215947


namespace cube_prime_factorization_l2159_215945

theorem cube_prime_factorization (x y : ℕ+) (p : ℕ) :
  (x + y) * (x^2 + 9*y) = p^3 ∧ Nat.Prime p ↔ (x = 2 ∧ y = 5) ∨ (x = 4 ∧ y = 1) := by
  sorry

end cube_prime_factorization_l2159_215945


namespace power_of_nine_l2159_215935

theorem power_of_nine (n : ℕ) (h : 3^(2*n) = 81) : 9^(n+1) = 729 := by
  sorry

end power_of_nine_l2159_215935


namespace quadratic_real_roots_condition_l2159_215938

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, (k - 2) * x^2 - 2 * k * x + k = 6) ↔ (k ≥ 3/2 ∧ k ≠ 2) :=
by sorry

end quadratic_real_roots_condition_l2159_215938


namespace product_of_cubes_l2159_215966

theorem product_of_cubes (r s : ℝ) (hr : r > 0) (hs : s > 0) 
  (h1 : r^3 + s^3 = 1) (h2 : r^6 + s^6 = 15/16) : r * s = (1/48)^(1/3) := by
  sorry

end product_of_cubes_l2159_215966


namespace seedling_probability_l2159_215987

def total_seedlings : ℕ := 14
def selechenskaya_seedlings : ℕ := 6
def vologda_seedlings : ℕ := 8
def selected_seedlings : ℕ := 3

theorem seedling_probability :
  (Nat.choose selechenskaya_seedlings selected_seedlings : ℚ) / 
  (Nat.choose total_seedlings selected_seedlings : ℚ) = 5 / 91 := by
  sorry

end seedling_probability_l2159_215987


namespace max_consecutive_integers_sum_l2159_215942

theorem max_consecutive_integers_sum (k : ℕ) : 
  (∃ n : ℕ, (k : ℤ) * (2 * n + k - 1) = 2 * 3^8) →
  k ≤ 108 :=
by sorry

end max_consecutive_integers_sum_l2159_215942


namespace cubic_polynomial_factor_property_l2159_215914

/-- Given a cubic polynomial 2x³ - hx + k where x + 2 and x - 1 are factors, 
    prove that |2h-3k| = 0 -/
theorem cubic_polynomial_factor_property (h k : ℝ) : 
  (∀ x, (x + 2) * (x - 1) ∣ (2 * x^3 - h * x + k)) → 
  |2 * h - 3 * k| = 0 := by
  sorry

end cubic_polynomial_factor_property_l2159_215914


namespace sum_of_seven_smallest_multiples_of_12_l2159_215993

theorem sum_of_seven_smallest_multiples_of_12 : 
  (Finset.range 7).sum (λ n => 12 * (n + 1)) = 336 := by
  sorry

end sum_of_seven_smallest_multiples_of_12_l2159_215993


namespace only_vinyl_chloride_and_benzene_planar_l2159_215909

/-- Represents an organic compound -/
inductive OrganicCompound
| Propylene
| VinylChloride
| Benzene
| Toluene

/-- Predicate to check if all atoms in a compound are on the same plane -/
def all_atoms_on_same_plane (c : OrganicCompound) : Prop :=
  match c with
  | OrganicCompound.Propylene => False
  | OrganicCompound.VinylChloride => True
  | OrganicCompound.Benzene => True
  | OrganicCompound.Toluene => False

/-- Theorem stating that only vinyl chloride and benzene have all atoms on the same plane -/
theorem only_vinyl_chloride_and_benzene_planar :
  ∀ c : OrganicCompound, all_atoms_on_same_plane c ↔ (c = OrganicCompound.VinylChloride ∨ c = OrganicCompound.Benzene) :=
by
  sorry


end only_vinyl_chloride_and_benzene_planar_l2159_215909


namespace tree_cutting_theorem_l2159_215954

/-- The number of trees James cuts per day -/
def james_trees_per_day : ℕ := 20

/-- The number of days James works alone -/
def james_solo_days : ℕ := 2

/-- The number of days the brothers help -/
def brother_help_days : ℕ := 3

/-- The number of brothers helping -/
def num_brothers : ℕ := 2

/-- The percentage reduction in trees cut by brothers compared to James -/
def brother_reduction_percent : ℚ := 20 / 100

/-- The total number of trees cut down -/
def total_trees_cut : ℕ := 136

theorem tree_cutting_theorem :
  james_trees_per_day * james_solo_days + 
  (james_trees_per_day * (1 - brother_reduction_percent) * num_brothers * brother_help_days).floor = 
  total_trees_cut :=
sorry

end tree_cutting_theorem_l2159_215954


namespace equal_area_triangles_l2159_215944

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem equal_area_triangles :
  triangleArea 20 20 24 = triangleArea 20 20 32 := by
  sorry

end equal_area_triangles_l2159_215944


namespace expression_evaluation_l2159_215967

theorem expression_evaluation : 15 - 6 / (-2) + |3| * (-5) = 3 := by
  sorry

end expression_evaluation_l2159_215967


namespace x_less_than_y_less_than_zero_l2159_215907

theorem x_less_than_y_less_than_zero (x y : ℝ) 
  (h1 : 2 * x - 3 * y > 6 * x) 
  (h2 : 3 * x - 4 * y < 2 * y - x) : 
  x < y ∧ y < 0 := by
  sorry

end x_less_than_y_less_than_zero_l2159_215907


namespace smallest_n_same_last_two_digits_l2159_215926

theorem smallest_n_same_last_two_digits : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬(107 * m ≡ m [ZMOD 100])) ∧ 
  (107 * n ≡ n [ZMOD 100]) ∧
  n = 50 := by
sorry

end smallest_n_same_last_two_digits_l2159_215926


namespace train_speed_l2159_215900

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  bridge_length = 240.03 →
  crossing_time = 30 →
  (((train_length + bridge_length) / crossing_time) * 3.6) = 45.0036 := by
  sorry

end train_speed_l2159_215900


namespace reciprocal_roots_implies_p_zero_l2159_215971

-- Define the quadratic equation
def quadratic (p : ℝ) (x : ℝ) : ℝ := 2 * x^2 + p * x + 4

-- Define the condition for reciprocal roots
def has_reciprocal_roots (p : ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ 0 ∧ s ≠ 0 ∧ r * s = 1 ∧
  quadratic p r = 0 ∧ quadratic p s = 0

-- Theorem statement
theorem reciprocal_roots_implies_p_zero :
  has_reciprocal_roots p → p = 0 := by sorry

end reciprocal_roots_implies_p_zero_l2159_215971


namespace flower_cost_proof_l2159_215922

/-- Proves that if Lilly saves $2 per day for 22 days and can buy 11 flowers with her savings, then each flower costs $4. -/
theorem flower_cost_proof (days : ℕ) (daily_savings : ℚ) (num_flowers : ℕ) 
  (h1 : days = 22) 
  (h2 : daily_savings = 2) 
  (h3 : num_flowers = 11) : 
  (days * daily_savings) / num_flowers = 4 := by
  sorry

#check flower_cost_proof

end flower_cost_proof_l2159_215922


namespace madison_distance_l2159_215908

/-- Represents the travel from Gardensquare to Madison -/
structure Journey where
  time : ℝ
  speed : ℝ
  mapScale : ℝ

/-- Calculates the distance on the map given a journey -/
def mapDistance (j : Journey) : ℝ :=
  j.time * j.speed * j.mapScale

/-- Theorem stating that the distance on the map is 5 inches -/
theorem madison_distance (j : Journey) 
  (h1 : j.time = 5)
  (h2 : j.speed = 60)
  (h3 : j.mapScale = 0.016666666666666666) : 
  mapDistance j = 5 := by
  sorry

end madison_distance_l2159_215908


namespace fraction_woodwind_brass_this_year_l2159_215955

-- Define the fractions of students for each instrument last year
def woodwind_last_year : ℚ := 1/2
def brass_last_year : ℚ := 2/5
def percussion_last_year : ℚ := 1 - (woodwind_last_year + brass_last_year)

-- Define the fractions of students who left for each instrument
def woodwind_left : ℚ := 1/2
def brass_left : ℚ := 1/4
def percussion_left : ℚ := 0

-- Calculate the fractions of students for each instrument this year
def woodwind_this_year : ℚ := woodwind_last_year * (1 - woodwind_left)
def brass_this_year : ℚ := brass_last_year * (1 - brass_left)
def percussion_this_year : ℚ := percussion_last_year * (1 - percussion_left)

-- Theorem to prove
theorem fraction_woodwind_brass_this_year :
  woodwind_this_year + brass_this_year = 11/20 := by sorry

end fraction_woodwind_brass_this_year_l2159_215955


namespace circle_chord_tangent_relation_l2159_215950

/-- Given a circle with diameter AB and radius r, chord BF extended to meet
    the tangent at A at point C, and point E on BC extended such that BE = DC,
    prove that h = √(r² - d²), where d is the distance from E to the tangent at B
    and h is the distance from E to the diameter AB. -/
theorem circle_chord_tangent_relation (r d h : ℝ) : h = Real.sqrt (r^2 - d^2) :=
sorry

end circle_chord_tangent_relation_l2159_215950


namespace intersection_nonempty_intersection_equals_B_l2159_215949

-- Define sets A and B
def A : Set ℝ := {x : ℝ | (x + 1) * (4 - x) ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 2}

-- Theorem 1
theorem intersection_nonempty (a : ℝ) : 
  (A ∩ B a).Nonempty ↔ -1/2 ≤ a ∧ a ≤ 2 :=
sorry

-- Theorem 2
theorem intersection_equals_B (a : ℝ) :
  A ∩ B a = B a ↔ a ≥ 2 ∨ a ≤ -3 :=
sorry

end intersection_nonempty_intersection_equals_B_l2159_215949


namespace least_subtraction_for_divisibility_l2159_215912

theorem least_subtraction_for_divisibility :
  ∃ (n : ℕ), n ≤ 5 ∧ (∀ m : ℕ, m < n → ¬(37 ∣ (5000 - m))) ∧ (37 ∣ (5000 - n)) := by
  sorry

end least_subtraction_for_divisibility_l2159_215912


namespace line_tangent_to_circle_l2159_215933

/-- The value of c for which the line x - y + c = 0 is tangent to the circle (x - 1)^2 + y^2 = 2 -/
theorem line_tangent_to_circle (x y : ℝ) :
  (∃! p : ℝ × ℝ, (p.1 - p.2 + c = 0) ∧ ((p.1 - 1)^2 + p.2^2 = 2)) →
  c = -1 + Real.sqrt 2 :=
sorry

end line_tangent_to_circle_l2159_215933


namespace inheritance_tax_problem_l2159_215946

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 15000) → x = 41379 := by
  sorry

end inheritance_tax_problem_l2159_215946


namespace baby_shower_parking_lot_wheels_l2159_215986

/-- Calculates the total number of car wheels in a parking lot --/
def total_wheels (guest_cars : ℕ) (parent_cars : ℕ) (wheels_per_car : ℕ) : ℕ :=
  (guest_cars + parent_cars) * wheels_per_car

/-- Theorem statement for the baby shower parking lot problem --/
theorem baby_shower_parking_lot_wheels : 
  total_wheels 10 2 4 = 48 := by
sorry

end baby_shower_parking_lot_wheels_l2159_215986


namespace xiaoming_money_problem_l2159_215910

theorem xiaoming_money_problem (price_left : ℕ) (price_right : ℕ) :
  price_right = price_left - 1 →
  12 * price_left = 14 * price_right →
  12 * price_left = 84 :=
by sorry

end xiaoming_money_problem_l2159_215910


namespace abs_equation_quadratic_coefficients_l2159_215940

theorem abs_equation_quadratic_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = -8 ∧ c = 7 := by
sorry

end abs_equation_quadratic_coefficients_l2159_215940


namespace quadratic_inequality_l2159_215952

/-- A quadratic function of the form (x + m - 3)(x - m) + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := (x + m - 3) * (x - m) + 3

theorem quadratic_inequality (m x₁ x₂ : ℝ) (h₁ : x₁ < x₂) (h₂ : x₁ + x₂ < 3) :
  f m x₁ > f m x₂ := by
  sorry

end quadratic_inequality_l2159_215952


namespace largest_share_is_18000_l2159_215964

/-- Represents the profit share of a partner -/
structure Share where
  ratio : Nat
  amount : Nat

/-- Calculates the largest share given a total profit and a list of ratios -/
def largest_share (total_profit : Nat) (ratios : List Nat) : Nat :=
  let sum_ratios := ratios.sum
  let part_value := total_profit / sum_ratios
  (ratios.maximum.getD 0) * part_value

/-- The theorem stating that the largest share is $18,000 -/
theorem largest_share_is_18000 :
  largest_share 48000 [1, 2, 3, 4, 6] = 18000 := by
  sorry

end largest_share_is_18000_l2159_215964


namespace fraction_equality_l2159_215988

theorem fraction_equality (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x - y) / (x + y) = -1 / Real.sqrt 3 := by
sorry

end fraction_equality_l2159_215988


namespace right_triangle_area_l2159_215923

theorem right_triangle_area (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b = 24 →
  c = 24 →
  a^2 + c^2 = (24 + b)^2 →
  (1/2) * a * c = 216 :=
by sorry

end right_triangle_area_l2159_215923


namespace solution_set_for_a_eq_1_range_of_a_l2159_215995

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_for_a_eq_1_range_of_a_l2159_215995


namespace bush_height_after_two_years_l2159_215976

def bush_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

theorem bush_height_after_two_years 
  (h : bush_height 1 5 = 81) : 
  bush_height 1 2 = 3 := by
  sorry

end bush_height_after_two_years_l2159_215976


namespace profitable_after_three_years_l2159_215994

/-- Represents the financial data for the communication equipment --/
structure EquipmentData where
  initialInvestment : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualProfit : ℕ

/-- Calculates the cumulative profit after a given number of years --/
def cumulativeProfit (data : EquipmentData) (years : ℕ) : ℤ :=
  (data.annualProfit * years : ℤ) - 
  (data.initialInvestment + data.firstYearExpenses * years + 
   data.annualExpenseIncrease * (years * (years - 1) / 2) : ℤ)

/-- Theorem stating that the equipment becomes profitable after 3 years --/
theorem profitable_after_three_years (data : EquipmentData) 
  (h1 : data.initialInvestment = 980000)
  (h2 : data.firstYearExpenses = 120000)
  (h3 : data.annualExpenseIncrease = 40000)
  (h4 : data.annualProfit = 500000) :
  cumulativeProfit data 3 > 0 ∧ cumulativeProfit data 2 ≤ 0 := by
  sorry

#check profitable_after_three_years

end profitable_after_three_years_l2159_215994


namespace quartic_roots_l2159_215962

theorem quartic_roots : 
  let f : ℝ → ℝ := λ x ↦ 3*x^4 + 2*x^3 - 8*x^2 + 2*x + 3
  ∃ (a b c d : ℝ), 
    (a = (1 - Real.sqrt 43 + 2*Real.sqrt 34) / 6) ∧
    (b = (1 - Real.sqrt 43 - 2*Real.sqrt 34) / 6) ∧
    (c = (1 + Real.sqrt 43 + 2*Real.sqrt 34) / 6) ∧
    (d = (1 + Real.sqrt 43 - 2*Real.sqrt 34) / 6) ∧
    (∀ x : ℝ, f x = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
by sorry

end quartic_roots_l2159_215962


namespace product_mod_seventeen_l2159_215957

theorem product_mod_seventeen :
  (3001 * 3002 * 3003 * 3004 * 3005) % 17 = 12 := by
  sorry

end product_mod_seventeen_l2159_215957


namespace expression_value_l2159_215934

theorem expression_value (a b : ℝ) (h : a * b > 0) :
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = 3 ∨
  (a / abs a) + (b / abs b) + ((a * b) / abs (a * b)) = -1 :=
by sorry

end expression_value_l2159_215934


namespace pure_imaginary_complex_number_l2159_215997

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 1) : ℂ).re = 0 ∧ (a^2 - 4*a + 3 + Complex.I * (a - 1) : ℂ).im ≠ 0 → a = 3 :=
by sorry

end pure_imaginary_complex_number_l2159_215997


namespace num_terms_xyz_4_is_15_l2159_215920

/-- The number of terms in the expansion of (x+y+z)^4 -/
def num_terms_xyz_4 : ℕ := sorry

/-- Theorem stating that the number of terms in (x+y+z)^4 is 15 -/
theorem num_terms_xyz_4_is_15 : num_terms_xyz_4 = 15 := by sorry

end num_terms_xyz_4_is_15_l2159_215920


namespace divisor_of_sum_l2159_215902

theorem divisor_of_sum (n : ℕ) (a : ℕ) (d : ℕ) : 
  n = 425897 → a = 7 → d = 7 → (n + a) % d = 0 := by
  sorry

end divisor_of_sum_l2159_215902


namespace arithmetic_sequence_common_difference_l2159_215998

theorem arithmetic_sequence_common_difference
  (a : ℚ)  -- first term
  (aₙ : ℚ) -- last term
  (S : ℚ)  -- sum of all terms
  (h1 : a = 3)
  (h2 : aₙ = 50)
  (h3 : S = 318) :
  ∃ (n : ℕ) (d : ℚ), n > 1 ∧ d = 47 / 11 ∧ aₙ = a + (n - 1) * d ∧ S = (n / 2) * (a + aₙ) :=
sorry

end arithmetic_sequence_common_difference_l2159_215998


namespace first_account_interest_rate_l2159_215928

/-- Proves that the interest rate of the first account is 0.02 given the problem conditions --/
theorem first_account_interest_rate :
  ∀ (r : ℝ),
    r > 0 →
    r < 1 →
    1000 * r + 1800 * 0.04 = 92 →
    r = 0.02 := by
  sorry

end first_account_interest_rate_l2159_215928
