import Mathlib

namespace product_of_max_min_sum_l2067_206746

theorem product_of_max_min_sum (x y z : ℝ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  (4 : ℝ)^(Real.sqrt (5*x + 9*y + 4*z)) - 68 * 2^(Real.sqrt (5*x + 9*y + 4*z)) + 256 = 0 → 
  ∃ (min_sum max_sum : ℝ), 
    (∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
      (4 : ℝ)^(Real.sqrt (5*a + 9*b + 4*c)) - 68 * 2^(Real.sqrt (5*a + 9*b + 4*c)) + 256 = 0 → 
      min_sum ≤ a + b + c ∧ a + b + c ≤ max_sum) ∧
    min_sum * max_sum = 4 := by
  sorry

end product_of_max_min_sum_l2067_206746


namespace population_increase_rate_l2067_206738

theorem population_increase_rate 
  (initial_population : ℕ) 
  (final_population : ℕ) 
  (increase_rate : ℚ) : 
  initial_population = 240 →
  final_population = 264 →
  increase_rate = (final_population - initial_population : ℚ) / initial_population * 100 →
  increase_rate = 10 := by
  sorry

end population_increase_rate_l2067_206738


namespace root_intervals_l2067_206707

noncomputable def f (x : ℝ) : ℝ :=
  if x > -2 then Real.exp (x + 1) - 2
  else Real.exp (-x - 3) - 2

theorem root_intervals (e : ℝ) (h_e : e = Real.exp 1) :
  {k : ℤ | ∃ x : ℝ, f x = 0 ∧ k - 1 < x ∧ x < k} = {-4, 0} := by
  sorry

end root_intervals_l2067_206707


namespace monomial_exponent_sum_l2067_206783

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m n : ℕ) : Prop := m = 3 ∧ n = 2

theorem monomial_exponent_sum (m n : ℕ) (h : like_terms m n) : m + n = 5 := by
  sorry

end monomial_exponent_sum_l2067_206783


namespace cube_sum_reciprocal_l2067_206718

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^3 + 1/x^3 = 18 := by
  sorry

end cube_sum_reciprocal_l2067_206718


namespace fraction_meaningful_iff_not_one_l2067_206781

theorem fraction_meaningful_iff_not_one (x : ℝ) :
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by
  sorry

end fraction_meaningful_iff_not_one_l2067_206781


namespace yangmei_sales_l2067_206754

/-- Yangmei sales problem -/
theorem yangmei_sales (total_weight : ℕ) (round_weight round_price square_weight square_price : ℕ) 
  (h_total : total_weight = 1000)
  (h_round : round_weight = 8 ∧ round_price = 160)
  (h_square : square_weight = 18 ∧ square_price = 270) :
  (∃ a : ℕ, a * round_price + a * square_price = 8600 → a = 20) ∧
  (∃ x y : ℕ, x * round_price + y * square_price = 16760 ∧ 
              x * round_weight + y * square_weight = total_weight →
              x = 44 ∧ y = 36) ∧
  (∃ b : ℕ, b > 0 ∧ 
            (∃ m n : ℕ, (m + b) * round_weight + n * square_weight = total_weight ∧
                        m * round_price + n * square_price = 16760) →
            b = 9 ∨ b = 18) := by
  sorry

end yangmei_sales_l2067_206754


namespace unique_solution_implies_a_value_l2067_206768

theorem unique_solution_implies_a_value (a : ℝ) : 
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) → a = 2019 :=
by sorry

end unique_solution_implies_a_value_l2067_206768


namespace complex_bound_l2067_206737

theorem complex_bound (z : ℂ) (h : Complex.abs (z + z⁻¹) = 1) :
  (Real.sqrt 5 - 1) / 2 ≤ Complex.abs z ∧ Complex.abs z ≤ (Real.sqrt 5 + 1) / 2 := by
  sorry

end complex_bound_l2067_206737


namespace medal_award_combinations_l2067_206720

/-- The number of sprinters --/
def total_sprinters : ℕ := 10

/-- The number of American sprinters --/
def american_sprinters : ℕ := 4

/-- The number of non-American sprinters --/
def non_american_sprinters : ℕ := total_sprinters - american_sprinters

/-- The number of medals to be awarded --/
def medals : ℕ := 4

/-- The maximum number of Americans that can win medals --/
def max_american_winners : ℕ := 2

/-- The function to calculate the number of ways medals can be awarded --/
def ways_to_award_medals : ℕ := sorry

/-- Theorem stating that the number of ways to award medals is 6600 --/
theorem medal_award_combinations : ways_to_award_medals = 6600 := by sorry

end medal_award_combinations_l2067_206720


namespace album_distribution_ways_l2067_206709

/-- The number of ways to distribute albums to friends -/
def distribute_albums (photo_albums : ℕ) (stamp_albums : ℕ) (friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of ways to distribute the albums -/
theorem album_distribution_ways :
  distribute_albums 2 3 4 = 10 := by sorry

end album_distribution_ways_l2067_206709


namespace fly_distance_from_ceiling_l2067_206787

theorem fly_distance_from_ceiling (z : ℝ) : 
  3^2 + 4^2 + z^2 = 6^2 → z = Real.sqrt 11 := by
  sorry

end fly_distance_from_ceiling_l2067_206787


namespace recipe_flour_amount_l2067_206714

/-- The amount of flour Mary put in -/
def flour_added : ℝ := 7.5

/-- The amount of excess flour added -/
def excess_flour : ℝ := 0.8

/-- The amount of flour the recipe wants -/
def recipe_flour : ℝ := flour_added - excess_flour

theorem recipe_flour_amount : recipe_flour = 6.7 := by
  sorry

end recipe_flour_amount_l2067_206714


namespace min_value_theorem_l2067_206750

def f (x : ℝ) := 45 * |2*x - 1|

def g (x : ℝ) := f x + f (x - 1)

theorem min_value_theorem (a m n : ℝ) :
  (∀ x, g x ≥ a) →
  m > 0 →
  n > 0 →
  m + n = a →
  (∀ p q, p > 0 → q > 0 → p + q = a → 4/m + 1/n ≤ 4/p + 1/q) →
  4/m + 1/n = 9/2 :=
sorry

end min_value_theorem_l2067_206750


namespace sum_of_roots_l2067_206766

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 := by
sorry

end sum_of_roots_l2067_206766


namespace waiter_new_customers_l2067_206790

theorem waiter_new_customers 
  (initial_customers : ℕ) 
  (left_customers : ℕ) 
  (final_customers : ℕ) 
  (h1 : initial_customers = 14) 
  (h2 : left_customers = 3) 
  (h3 : final_customers = 50) : 
  final_customers - (initial_customers - left_customers) = 39 := by
  sorry

end waiter_new_customers_l2067_206790


namespace extra_money_is_seven_l2067_206726

/-- The amount of extra money given by an appreciative customer to Hillary at a flea market. -/
def extra_money (price_per_craft : ℕ) (crafts_sold : ℕ) (deposited : ℕ) (remaining : ℕ) : ℕ :=
  (deposited + remaining) - (price_per_craft * crafts_sold)

/-- Theorem stating that the extra money given to Hillary is 7 dollars. -/
theorem extra_money_is_seven :
  extra_money 12 3 18 25 = 7 := by
  sorry

end extra_money_is_seven_l2067_206726


namespace hyperbola_equation_l2067_206723

theorem hyperbola_equation (a b : ℝ) (h1 : b > 0) (h2 : a > 0) (h3 : ∃ n : ℕ, a = n) 
  (h4 : (a^2 + b^2) / a^2 = 7/4) (h5 : a^2 + b^2 ≤ 20) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((x^2 - 4*y^2/3 = 1) ∨ (x^2/4 - y^2/3 = 1) ∨ (x^2/9 - 4*y^2/27 = 1)) :=
by sorry

end hyperbola_equation_l2067_206723


namespace area_of_triangle_FOH_l2067_206732

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- Theorem about the area of triangle FOH in a trapezoid -/
theorem area_of_triangle_FOH (t : Trapezoid) 
  (h1 : t.base1 = 40)
  (h2 : t.base2 = 50)
  (h3 : t.area = 900) : 
  ∃ (area_FOH : ℝ), abs (area_FOH - 400/9) < 0.01 := by
  sorry

#check area_of_triangle_FOH

end area_of_triangle_FOH_l2067_206732


namespace smallest_quadratic_root_l2067_206749

theorem smallest_quadratic_root : 
  let f : ℝ → ℝ := λ y => 4 * y^2 - 7 * y + 3
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → y ≤ z ∧ y = 3/4 := by
  sorry

end smallest_quadratic_root_l2067_206749


namespace arithmetic_sequence_ratio_l2067_206769

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n = (n : ℚ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) →
  (∀ n : ℕ, n > 0 → T n = (n : ℚ) / 2 * (2 * b 1 + (n - 1) * (b 2 - b 1))) →
  (∀ n : ℕ, n > 0 → S n / T n = (n : ℚ) / (2 * n + 1)) →
  (a 5 / b 5 = 9 / 19) :=
by sorry

end arithmetic_sequence_ratio_l2067_206769


namespace equation_solution_l2067_206704

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x - 3

-- Theorem statement
theorem equation_solution :
  ∃ x : ℝ, 2 * (f x) - 11 = f (x - 2) ∧ x = 5 := by
  sorry

end equation_solution_l2067_206704


namespace equation_solutions_l2067_206752

theorem equation_solutions : 
  (∃ x₁ x₂ : ℝ, (∀ x : ℝ, (2*x - 1)^2 = (3 - x)^2 ↔ x = x₁ ∨ x = x₂) ∧ x₁ = -2 ∧ x₂ = 4/3) ∧
  (∃ y₁ y₂ : ℝ, (∀ x : ℝ, x^2 - Real.sqrt 3 * x - 1/4 = 0 ↔ x = y₁ ∨ x = y₂) ∧ 
    y₁ = (Real.sqrt 3 + 2)/2 ∧ y₂ = (Real.sqrt 3 - 2)/2) :=
by sorry

end equation_solutions_l2067_206752


namespace greatest_second_term_arithmetic_sequence_l2067_206751

theorem greatest_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
    a > 0 →
    d > 0 →
    a + (a + d) + (a + 2*d) + (a + 3*d) = 80 →
    ∀ (b e : ℕ),
      b > 0 →
      e > 0 →
      b + (b + e) + (b + 2*e) + (b + 3*e) = 80 →
      a + d ≤ 19 :=
by sorry

end greatest_second_term_arithmetic_sequence_l2067_206751


namespace log_product_equality_l2067_206799

theorem log_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (Real.log x^2 / Real.log y^8) * (Real.log y^3 / Real.log x^7) *
  (Real.log x^4 / Real.log y^5) * (Real.log y^5 / Real.log x^4) *
  (Real.log x^7 / Real.log y^3) * (Real.log y^8 / Real.log x^2) =
  28/3 * (Real.log x / Real.log y) := by
  sorry

end log_product_equality_l2067_206799


namespace average_scores_is_68_l2067_206702

def scores : List ℝ := [50, 60, 70, 80, 80]

theorem average_scores_is_68 : (scores.sum / scores.length) = 68 := by
  sorry

end average_scores_is_68_l2067_206702


namespace parallelogram_base_l2067_206798

/-- 
Given a parallelogram with area 612 square centimeters and height 18 cm, 
prove that its base is 34 cm.
-/
theorem parallelogram_base (area height : ℝ) (h1 : area = 612) (h2 : height = 18) :
  area / height = 34 := by
  sorry

end parallelogram_base_l2067_206798


namespace polynomial_division_remainder_l2067_206772

theorem polynomial_division_remainder
  (dividend : Polynomial ℤ)
  (divisor : Polynomial ℤ)
  (h_dividend : dividend = 3 * X^6 - 2 * X^4 + 5 * X^2 - 9)
  (h_divisor : divisor = X^2 + 3 * X + 2) :
  ∃ (q : Polynomial ℤ), dividend = q * divisor + (-174 * X - 177) :=
by sorry

end polynomial_division_remainder_l2067_206772


namespace vector_magnitude_l2067_206742

theorem vector_magnitude (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  (a.1^2 + a.2^2 = 4) →
  (b.1^2 + b.2^2 = 1) →
  (a.1 * b.1 + a.2 * b.2 = 2 * Real.cos angle) →
  ((a.1 - 2*b.1)^2 + (a.2 - 2*b.2)^2 = 4) :=
by
  sorry

#check vector_magnitude

end vector_magnitude_l2067_206742


namespace common_solution_y_values_l2067_206794

theorem common_solution_y_values (x y : ℝ) : 
  (x^2 + y^2 - 3 = 0 ∧ x^2 - 4*y + 6 = 0) →
  (y = -2 + Real.sqrt 13 ∨ y = -2 - Real.sqrt 13) :=
by sorry

end common_solution_y_values_l2067_206794


namespace polynomial_remainder_l2067_206784

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^3 - x + 4) % (x - 2) = 50 := by
  sorry

end polynomial_remainder_l2067_206784


namespace sum_of_digits_7_pow_23_l2067_206763

/-- Returns the ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Returns the tens digit of a natural number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- The sum of the tens digit and the ones digit of 7^23 is 7 -/
theorem sum_of_digits_7_pow_23 :
  tensDigit (7^23) + onesDigit (7^23) = 7 := by
  sorry

end sum_of_digits_7_pow_23_l2067_206763


namespace simplify_nested_roots_l2067_206727

theorem simplify_nested_roots (x : ℝ) :
  (((x ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 2 + (((x ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 2 = 2 * x :=
by sorry

end simplify_nested_roots_l2067_206727


namespace ten_steps_climb_l2067_206776

/-- Number of ways to climb n steps when allowed to take 1, 2, or 3 steps at a time -/
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | k + 3 => climbStairs (k + 2) + climbStairs (k + 1) + climbStairs k

/-- Theorem stating that there are 274 ways to climb 10 steps -/
theorem ten_steps_climb : climbStairs 10 = 274 := by
  sorry


end ten_steps_climb_l2067_206776


namespace pen_ratio_theorem_l2067_206706

/-- Represents the number of pens bought by each person -/
structure PenPurchase where
  julia : ℕ
  dorothy : ℕ
  robert : ℕ

/-- Represents the given conditions of the problem -/
def ProblemConditions (p : PenPurchase) : Prop :=
  p.dorothy = p.julia / 2 ∧
  p.robert = 4 ∧
  p.julia + p.dorothy + p.robert = 22

theorem pen_ratio_theorem (p : PenPurchase) :
  ProblemConditions p → p.julia / p.robert = 3 := by
  sorry

#check pen_ratio_theorem

end pen_ratio_theorem_l2067_206706


namespace new_shoes_cost_approx_l2067_206778

/-- Cost of repairing used shoes in dollars -/
def repair_cost : ℝ := 13.50

/-- Duration of repaired shoes in years -/
def repaired_duration : ℝ := 1

/-- Duration of new shoes in years -/
def new_duration : ℝ := 2

/-- Percentage increase in average cost per year of new shoes compared to repaired shoes -/
def percentage_increase : ℝ := 0.1852

/-- Cost of purchasing new shoes -/
def new_shoes_cost : ℝ := 2 * (repair_cost + percentage_increase * repair_cost)

theorem new_shoes_cost_approx :
  ∃ ε > 0, |new_shoes_cost - 32| < ε :=
sorry

end new_shoes_cost_approx_l2067_206778


namespace divisible_by_64_l2067_206712

theorem divisible_by_64 (n : ℕ) (h : n > 0) : ∃ k : ℤ, 3^(2*n + 2) - 8*n - 9 = 64*k := by
  sorry

end divisible_by_64_l2067_206712


namespace brocard_and_interior_angle_bound_l2067_206795

/-- The Brocard angle of a triangle -/
def brocardAngle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (A B C : ℝ × ℝ) : ℝ := sorry

/-- Check if a point is inside or on the boundary of a triangle -/
def isInsideOrOnBoundary (M A B C : ℝ × ℝ) : Prop := sorry

theorem brocard_and_interior_angle_bound (A B C M : ℝ × ℝ) :
  isInsideOrOnBoundary M A B C →
  min (brocardAngle A B C) (min (angle A B M) (min (angle B C M) (angle C A M))) ≤ Real.pi / 6 := by
  sorry

end brocard_and_interior_angle_bound_l2067_206795


namespace lcm_of_three_numbers_specific_lcm_l2067_206770

theorem lcm_of_three_numbers (a b c : ℕ) (hcf : ℕ) (h_hcf : Nat.gcd a (Nat.gcd b c) = hcf) :
  Nat.lcm a (Nat.lcm b c) = a * b * c / hcf :=
by sorry

theorem specific_lcm :
  Nat.lcm 136 (Nat.lcm 144 168) = 411264 :=
by
  have h_hcf : Nat.gcd 136 (Nat.gcd 144 168) = 8 := by sorry
  exact lcm_of_three_numbers 136 144 168 8 h_hcf

end lcm_of_three_numbers_specific_lcm_l2067_206770


namespace interest_rate_calculation_l2067_206758

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (interest : ℝ)
  (h1 : principal = 1100)
  (h2 : time = 8)
  (h3 : interest = principal - 572) :
  ∃ (rate : ℝ), simple_interest principal rate time = interest ∧ rate = 0.06 := by
  sorry

end interest_rate_calculation_l2067_206758


namespace candy_problem_l2067_206793

theorem candy_problem (x : ℚ) : 
  (((3/4 * x - 3) * 3/4 - 5) = 10) → x = 336 := by
  sorry

end candy_problem_l2067_206793


namespace deposit_difference_approximately_219_01_l2067_206779

-- Constants
def initial_deposit : ℝ := 10000
def a_interest_rate : ℝ := 0.0288
def b_interest_rate : ℝ := 0.0225
def tax_rate : ℝ := 0.20
def years : ℕ := 5

-- A's total amount after 5 years
def a_total : ℝ := initial_deposit + initial_deposit * a_interest_rate * (1 - tax_rate) * years

-- B's total amount after 5 years (compound interest)
def b_total : ℝ := initial_deposit * (1 + b_interest_rate * (1 - tax_rate)) ^ years

-- Theorem statement
theorem deposit_difference_approximately_219_01 :
  ∃ ε > 0, ε < 0.005 ∧ |a_total - b_total - 219.01| < ε :=
sorry

end deposit_difference_approximately_219_01_l2067_206779


namespace seating_arrangements_l2067_206711

def number_of_people : ℕ := 10
def table_seats : ℕ := 8

def alice_bob_block : ℕ := 1
def other_individuals : ℕ := table_seats - 2

def ways_to_choose : ℕ := Nat.choose number_of_people table_seats
def ways_to_arrange_units : ℕ := Nat.factorial (other_individuals + alice_bob_block - 1)
def ways_to_arrange_alice_bob : ℕ := 2

theorem seating_arrangements :
  ways_to_choose * ways_to_arrange_units * ways_to_arrange_alice_bob = 64800 :=
sorry

end seating_arrangements_l2067_206711


namespace estimate_total_children_l2067_206753

theorem estimate_total_children (k m n : ℕ) (h1 : k > 0) (h2 : m > 0) (h3 : n > 0) (h4 : n ≤ m) (h5 : n ≤ k) :
  ∃ (total : ℚ), total = k * (m / n) :=
sorry

end estimate_total_children_l2067_206753


namespace range_of_a_l2067_206705

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → a ∈ Set.Ioo (-4) 2 := by
  sorry

end range_of_a_l2067_206705


namespace tangent_slope_implies_function_value_l2067_206739

open Real

theorem tangent_slope_implies_function_value (x₀ : ℝ) (h : x₀ > 0) : 
  let f : ℝ → ℝ := λ x ↦ log x + 2 * x
  (deriv f x₀ = 3) → f x₀ = 2 := by
sorry

end tangent_slope_implies_function_value_l2067_206739


namespace sequence_sum_times_three_l2067_206735

theorem sequence_sum_times_three (seq : List Nat) : 
  seq = [82, 84, 86, 88, 90, 92, 94, 96, 98, 100] →
  3 * (seq.sum) = 2730 := by
  sorry

end sequence_sum_times_three_l2067_206735


namespace fraction_division_l2067_206744

theorem fraction_division (x : ℝ) (hx : x ≠ 0) :
  (3 / 8) / (5 * x / 12) = 9 / (10 * x) := by
  sorry

end fraction_division_l2067_206744


namespace dog_walker_base_charge_l2067_206708

/-- Represents the earnings of a dog walker given their base charge per dog and walking durations. -/
def dog_walker_earnings (base_charge : ℝ) : ℝ :=
  (base_charge + 10 * 1) +  -- One dog for 10 minutes
  (2 * base_charge + 2 * 7 * 1) +  -- Two dogs for 7 minutes each
  (3 * base_charge + 3 * 9 * 1)  -- Three dogs for 9 minutes each

/-- Theorem stating that if a dog walker earns $171 with the given walking schedule, 
    their base charge per dog must be $20. -/
theorem dog_walker_base_charge : 
  ∃ (x : ℝ), dog_walker_earnings x = 171 → x = 20 :=
sorry

end dog_walker_base_charge_l2067_206708


namespace cake_area_l2067_206760

/-- Represents the size of a piece of cake in inches -/
def piece_size : ℝ := 2

/-- Represents the number of pieces that can be cut from the cake -/
def num_pieces : ℕ := 100

/-- Calculates the area of a single piece of cake -/
def piece_area : ℝ := piece_size * piece_size

/-- Theorem: The total area of the cake is 400 square inches -/
theorem cake_area : piece_area * num_pieces = 400 := by
  sorry

end cake_area_l2067_206760


namespace absolute_value_inequality_l2067_206774

theorem absolute_value_inequality (x : ℝ) : 
  |x^2 - 3*x| > 4 ↔ x < -1 ∨ x > 4 := by
  sorry

end absolute_value_inequality_l2067_206774


namespace length_of_AB_l2067_206771

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

def C₂ (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ C₂ A.1 A.2 ∧ C₂ B.1 B.2

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24 / 7 := by sorry

end length_of_AB_l2067_206771


namespace randy_pictures_l2067_206796

theorem randy_pictures (peter_pictures quincy_pictures randy_pictures total_pictures : ℕ) :
  peter_pictures = 8 →
  quincy_pictures = peter_pictures + 20 →
  total_pictures = 41 →
  total_pictures = peter_pictures + quincy_pictures + randy_pictures →
  randy_pictures = 5 := by
sorry

end randy_pictures_l2067_206796


namespace chord_length_squared_l2067_206734

/-- Given three circles with radii 6, 9, and 15, where the circles with radii 6 and 9
    are externally tangent to each other and internally tangent to the circle with radius 15,
    this theorem states that the square of the length of the chord of the circle with radius 15,
    which is a common external tangent to the other two circles, is equal to 692.64. -/
theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ = 6) (h₂ : r₂ = 9) (h₃ : R = 15)
  (h₄ : r₁ + r₂ = R - r₁ - r₂) : -- Condition for external tangency of smaller circles and internal tangency with larger circle
  (2 * R * ((r₁ * r₂) / (r₁ + r₂)))^2 = 692.64 := by
  sorry

end chord_length_squared_l2067_206734


namespace sunny_lead_in_new_race_l2067_206721

/-- Represents the race conditions and results -/
structure RaceData where
  initial_race_length : ℝ
  initial_sunny_lead : ℝ
  new_race_length : ℝ
  sunny_speed_increase : ℝ
  windy_speed_decrease : ℝ
  sunny_initial_lag : ℝ

/-- Calculates Sunny's lead at the end of the new race -/
def calculate_sunny_lead (data : RaceData) : ℝ :=
  sorry

/-- Theorem stating that given the race conditions, Sunny's lead at the end of the new race is 106.25 meters -/
theorem sunny_lead_in_new_race (data : RaceData) 
  (h1 : data.initial_race_length = 400)
  (h2 : data.initial_sunny_lead = 50)
  (h3 : data.new_race_length = 500)
  (h4 : data.sunny_speed_increase = 0.1)
  (h5 : data.windy_speed_decrease = 0.1)
  (h6 : data.sunny_initial_lag = 50) :
  calculate_sunny_lead data = 106.25 :=
sorry

end sunny_lead_in_new_race_l2067_206721


namespace solve_for_y_l2067_206775

theorem solve_for_y (x y : ℝ) (h1 : x^(2*y) = 64) (h2 : x = 8) : y = 1 := by
  sorry

end solve_for_y_l2067_206775


namespace exhibition_ticket_sales_l2067_206717

/-- Calculates the total worth of tickets sold over a period of days -/
def totalWorth (averageTicketsPerDay : ℕ) (numDays : ℕ) (ticketPrice : ℕ) : ℕ :=
  averageTicketsPerDay * numDays * ticketPrice

theorem exhibition_ticket_sales :
  let averageTicketsPerDay : ℕ := 80
  let numDays : ℕ := 3
  let ticketPrice : ℕ := 4
  totalWorth averageTicketsPerDay numDays ticketPrice = 960 := by
sorry

end exhibition_ticket_sales_l2067_206717


namespace right_triangle_and_symmetric_circle_l2067_206748

/-- Given a right triangle OAB in a rectangular coordinate system where:
  - O is the origin (0, 0)
  - A is the right-angle vertex at (4, -3)
  - |AB| = 2|OA|
  - The y-coordinate of B is positive
This theorem proves the coordinates of B and the equation of a symmetric circle. -/
theorem right_triangle_and_symmetric_circle :
  ∃ (B : ℝ × ℝ),
    let O : ℝ × ℝ := (0, 0)
    let A : ℝ × ℝ := (4, -3)
    -- B is in the first quadrant
    B.1 > 0 ∧ B.2 > 0 ∧
    -- OA ⟂ AB (right angle at A)
    (B.1 - A.1) * A.1 + (B.2 - A.2) * A.2 = 0 ∧
    -- |AB| = 2|OA|
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 4 * (A.1^2 + A.2^2) ∧
    -- B has coordinates (10, 5)
    B = (10, 5) ∧
    -- The equation of the symmetric circle
    ∀ (x y : ℝ),
      (x^2 - 6*x + y^2 + 2*y = 0) ↔
      ((x - 1)^2 + (y - 3)^2 = 10) :=
by sorry

end right_triangle_and_symmetric_circle_l2067_206748


namespace equation_solutions_parabola_properties_l2067_206729

-- Part 1: Equation solving
def equation (x : ℝ) : Prop := (x - 9)^2 = 2 * (x - 9)

theorem equation_solutions : 
  ∃ (x₁ x₂ : ℝ), x₁ = 9 ∧ x₂ = 11 ∧ equation x₁ ∧ equation x₂ ∧
  ∀ (x : ℝ), equation x → x = x₁ ∨ x = x₂ :=
sorry

-- Part 2: Parabola function
def parabola (x y : ℝ) : Prop := y = -x^2 - 6*x - 7

theorem parabola_properties :
  (parabola (-3) 2) ∧ (parabola (-1) (-2)) ∧
  ∀ (x y : ℝ), y = -(x + 3)^2 + 2 ↔ parabola x y :=
sorry

end equation_solutions_parabola_properties_l2067_206729


namespace dog_cost_l2067_206701

/-- The cost of a dog given the current money and additional money needed -/
theorem dog_cost (current_money additional_money : ℕ) :
  current_money = 34 →
  additional_money = 13 →
  current_money + additional_money = 47 :=
by sorry

end dog_cost_l2067_206701


namespace first_month_sale_is_800_l2067_206745

/-- Calculates the first month's sale given the sales of the following months and the average -/
def first_month_sale (sales : List ℕ) (average : ℕ) : ℕ :=
  6 * average - sales.sum

/-- Proves that the first month's sale is 800 given the problem conditions -/
theorem first_month_sale_is_800 :
  let sales : List ℕ := [900, 1000, 700, 800, 900]
  let average : ℕ := 850
  first_month_sale sales average = 800 := by
    sorry

end first_month_sale_is_800_l2067_206745


namespace intersection_point_satisfies_equations_intersection_point_unique_l2067_206756

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (5/3, 7/3)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 10 * x - 5 * y = 5

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := 8 * x + 2 * y = 18

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ (x y : ℚ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_satisfies_equations_intersection_point_unique_l2067_206756


namespace parabola_focus_on_line_l2067_206710

/-- The value of p for a parabola y^2 = 2px whose focus lies on 2x + y - 2 = 0 -/
theorem parabola_focus_on_line : ∃ (p : ℝ), 
  p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x ∧ 2*x + y - 2 = 0 ∧ x = p/2 ∧ y = 0) →
  p = 2 := by
  sorry

end parabola_focus_on_line_l2067_206710


namespace cork_mass_proof_l2067_206782

/-- The density of platinum in kg/m^3 -/
def platinum_density : ℝ := 2.15e4

/-- The density of cork wood in kg/m^3 -/
def cork_density : ℝ := 2.4e2

/-- The density of the combined system in kg/m^3 -/
def system_density : ℝ := 4.8e2

/-- The mass of the piece of platinum in kg -/
def platinum_mass : ℝ := 86.94

/-- The mass of the piece of cork wood in kg -/
def cork_mass : ℝ := 85

theorem cork_mass_proof :
  ∃ (cork_volume platinum_volume : ℝ),
    cork_volume > 0 ∧
    platinum_volume > 0 ∧
    cork_density = cork_mass / cork_volume ∧
    platinum_density = platinum_mass / platinum_volume ∧
    system_density = (cork_mass + platinum_mass) / (cork_volume + platinum_volume) :=
by sorry

end cork_mass_proof_l2067_206782


namespace billboard_count_l2067_206786

theorem billboard_count (B : ℕ) : 
  (B + 20 + 23) / 3 = 20 → B = 17 := by
  sorry

end billboard_count_l2067_206786


namespace solution_pairs_l2067_206700

theorem solution_pairs (x y : ℝ) : 
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := by
  sorry

end solution_pairs_l2067_206700


namespace problem_statement_l2067_206730

theorem problem_statement (a b : ℝ) (h : 2 * a - b + 3 = 0) :
  2 * (2 * a + b) - 4 * b = -6 := by
  sorry

end problem_statement_l2067_206730


namespace scooter_initial_value_l2067_206792

/-- The depreciation rate of the scooter's value each year -/
def depreciation_rate : ℚ := 3/4

/-- The number of years of depreciation -/
def years : ℕ := 4

/-- The value of the scooter after 4 years in rupees -/
def final_value : ℚ := 12656.25

/-- The initial value of the scooter in rupees -/
def initial_value : ℚ := 30000

/-- Theorem stating that given the depreciation rate, number of years, and final value,
    the initial value of the scooter can be calculated -/
theorem scooter_initial_value :
  initial_value * depreciation_rate ^ years = final_value := by
  sorry

end scooter_initial_value_l2067_206792


namespace find_divisor_l2067_206759

def nearest_number : ℕ := 3108
def original_number : ℕ := 3105

theorem find_divisor : 
  (nearest_number - original_number = 3) →
  (∃ d : ℕ, d > 1 ∧ nearest_number % d = 0 ∧ 
   ∀ n : ℕ, n > original_number ∧ n < nearest_number → n % d ≠ 0) →
  (∃ d : ℕ, d = 3 ∧ nearest_number % d = 0) :=
by sorry

end find_divisor_l2067_206759


namespace discount_percentage_proof_l2067_206719

theorem discount_percentage_proof (num_people : ℕ) (savings_per_person : ℝ) (final_price : ℝ) :
  num_people = 3 →
  savings_per_person = 4 →
  final_price = 48 →
  let total_savings := num_people * savings_per_person
  let original_price := final_price + total_savings
  let discount_percentage := (total_savings / original_price) * 100
  discount_percentage = 20 := by
  sorry

end discount_percentage_proof_l2067_206719


namespace difference_of_fractions_of_6000_l2067_206731

theorem difference_of_fractions_of_6000 : 
  (1 / 10 : ℚ) * 6000 - (1 / 1000 : ℚ) * 6000 = 594 := by
  sorry

end difference_of_fractions_of_6000_l2067_206731


namespace square_sum_from_sum_and_product_l2067_206757

theorem square_sum_from_sum_and_product (x y : ℚ) 
  (h1 : x + y = 5/6) (h2 : x * y = 7/36) : x^2 + y^2 = 11/36 := by
  sorry

end square_sum_from_sum_and_product_l2067_206757


namespace counterexample_exists_l2067_206762

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end counterexample_exists_l2067_206762


namespace no_base_for_131_square_l2067_206736

theorem no_base_for_131_square (b : ℕ) : b > 3 → ¬∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end no_base_for_131_square_l2067_206736


namespace tank_capacity_theorem_l2067_206724

/-- Represents a tank with a leak and an inlet pipe. -/
structure Tank where
  capacity : ℝ
  leak_empty_time : ℝ
  inlet_rate : ℝ
  combined_empty_time : ℝ

/-- Theorem stating the relationship between tank properties and its capacity. -/
theorem tank_capacity_theorem (t : Tank) 
  (h1 : t.leak_empty_time = 6)
  (h2 : t.inlet_rate = 2.5 * 60)
  (h3 : t.combined_empty_time = 8) :
  t.capacity = 3600 / 7 := by
  sorry

#check tank_capacity_theorem

end tank_capacity_theorem_l2067_206724


namespace average_age_is_35_l2067_206703

/-- Represents the ages of John, Mary, and Tonya -/
structure Ages where
  john : ℕ
  mary : ℕ
  tonya : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  ages.john = 2 * ages.mary ∧
  2 * ages.john = ages.tonya ∧
  ages.tonya = 60

/-- The average age of John, Mary, and Tonya -/
def average_age (ages : Ages) : ℚ :=
  (ages.john + ages.mary + ages.tonya : ℚ) / 3

/-- Theorem stating that the average age is 35 given the conditions -/
theorem average_age_is_35 (ages : Ages) (h : satisfies_conditions ages) :
  average_age ages = 35 := by
  sorry

end average_age_is_35_l2067_206703


namespace apple_arrangements_l2067_206788

def word : String := "APPLE"

def letter_count : Nat := word.length

def letter_frequencies : List (Char × Nat) := [('A', 1), ('P', 2), ('L', 1), ('E', 1)]

/-- The number of distinct arrangements of the letters in the word "APPLE" -/
def distinct_arrangements : Nat := 60

/-- Theorem stating that the number of distinct arrangements of the letters in "APPLE" is 60 -/
theorem apple_arrangements :
  distinct_arrangements = 60 :=
by sorry

end apple_arrangements_l2067_206788


namespace fibonacci_like_sequence_roots_l2067_206733

def fibonacci_like_sequence (F : ℕ → ℝ) : Prop :=
  F 0 = 2 ∧ F 1 = 3 ∧ ∀ n, F (n + 1) * F (n - 1) - F n ^ 2 = (-1) ^ n * 2

def has_exponential_form (F : ℕ → ℝ) (r₁ r₂ : ℝ) : Prop :=
  ∃ a b : ℝ, ∀ n, F n = a * r₁ ^ n + b * r₂ ^ n

theorem fibonacci_like_sequence_roots 
  (F : ℕ → ℝ) (r₁ r₂ : ℝ) 
  (h₁ : fibonacci_like_sequence F) 
  (h₂ : has_exponential_form F r₁ r₂) : 
  |r₁ - r₂| = Real.sqrt 17 / 2 := by sorry

end fibonacci_like_sequence_roots_l2067_206733


namespace players_who_quit_l2067_206764

theorem players_who_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) 
  (h1 : initial_players = 8)
  (h2 : lives_per_player = 3)
  (h3 : total_lives = 15) :
  initial_players - (total_lives / lives_per_player) = 3 :=
by sorry

end players_who_quit_l2067_206764


namespace min_reciprocal_sum_l2067_206722

theorem min_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (sum_eq : x + y + z = 3) (z_eq : z = 1) :
  1/x + 1/y + 1/z ≥ 3 ∧ (1/x + 1/y + 1/z = 3 ↔ x = 1 ∧ y = 1) := by
  sorry

end min_reciprocal_sum_l2067_206722


namespace train_passing_time_l2067_206797

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : 
  train_length = 80 → 
  train_speed_kmph = 36 → 
  (train_length / (train_speed_kmph * 1000 / 3600)) = 8 := by
  sorry

end train_passing_time_l2067_206797


namespace work_completion_time_l2067_206725

/-- The time taken by A, B, and C to complete a work given their pairwise completion times -/
theorem work_completion_time 
  (time_AB : ℝ) 
  (time_BC : ℝ) 
  (time_AC : ℝ) 
  (h_AB : time_AB = 8) 
  (h_BC : time_BC = 12) 
  (h_AC : time_AC = 8) : 
  (1 / (1 / time_AB + 1 / time_BC + 1 / time_AC)) = 6 := by
  sorry

end work_completion_time_l2067_206725


namespace modulus_two_plus_i_sixth_l2067_206747

/-- The modulus of (2 + i)^6 is equal to 125 -/
theorem modulus_two_plus_i_sixth : Complex.abs ((2 : ℂ) + Complex.I) ^ 6 = 125 := by
  sorry

end modulus_two_plus_i_sixth_l2067_206747


namespace even_pairs_ge_odd_pairs_l2067_206761

/-- A sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Count the number of (1,0) pairs with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : ℕ := sorry

/-- Count the number of (1,0) pairs with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : ℕ := sorry

/-- Theorem: In any binary sequence, the number of (1,0) pairs with even number
    of digits between them is greater than or equal to the number of (1,0) pairs
    with odd number of digits between them -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq := by sorry

end even_pairs_ge_odd_pairs_l2067_206761


namespace sector_angle_l2067_206780

/-- Given a circular sector with circumference 4 and area 1, prove that its central angle is 2 radians -/
theorem sector_angle (r : ℝ) (l : ℝ) (α : ℝ) 
  (h_circumference : 2 * r + l = 4)
  (h_area : (1 / 2) * l * r = 1) :
  α = 2 :=
sorry

end sector_angle_l2067_206780


namespace three_black_reachable_l2067_206743

structure UrnState :=
  (black : ℕ)
  (white : ℕ)

def initial_state : UrnState :=
  ⟨100, 120⟩

inductive Operation
  | replace_3b_with_2b
  | replace_2b1w_with_1b1w
  | replace_1b2w_with_2w
  | replace_3w_with_1b1w

def apply_operation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.replace_3b_with_2b => ⟨state.black - 1, state.white⟩
  | Operation.replace_2b1w_with_1b1w => ⟨state.black - 1, state.white⟩
  | Operation.replace_1b2w_with_2w => ⟨state.black - 1, state.white⟩
  | Operation.replace_3w_with_1b1w => ⟨state.black + 1, state.white - 2⟩

def reachable (target : UrnState) : Prop :=
  ∃ (n : ℕ) (ops : Fin n → Operation),
    (List.foldl apply_operation initial_state (List.ofFn ops)) = target

theorem three_black_reachable :
  reachable ⟨3, 120⟩ :=
sorry

end three_black_reachable_l2067_206743


namespace stratified_sampling_best_l2067_206773

/-- Represents different sampling methods -/
inductive SamplingMethod
| Lottery
| RandomNumberTable
| Systematic
| Stratified

/-- Represents product quality classes -/
inductive ProductClass
| FirstClass
| SecondClass
| Defective

/-- Represents a collection of products with their quantities -/
structure ProductCollection :=
  (total : ℕ)
  (firstClass : ℕ)
  (secondClass : ℕ)
  (defective : ℕ)

/-- Determines the most appropriate sampling method for quality analysis -/
def bestSamplingMethod (products : ProductCollection) (sampleSize : ℕ) : SamplingMethod :=
  sorry

/-- Theorem stating that stratified sampling is the best method for the given conditions -/
theorem stratified_sampling_best :
  let products : ProductCollection := {
    total := 40,
    firstClass := 10,
    secondClass := 25,
    defective := 5
  }
  let sampleSize := 8
  bestSamplingMethod products sampleSize = SamplingMethod.Stratified :=
by sorry

end stratified_sampling_best_l2067_206773


namespace correct_product_l2067_206767

/-- Given two positive integers a and b, where a is a two-digit number,
    if reversing the digits of a and multiplying by b results in 172,
    then the correct product of a and b is 136. -/
theorem correct_product (a b : ℕ) : 
  (a ≥ 10 ∧ a ≤ 99) →  -- a is a two-digit number
  (b > 0) →  -- b is positive
  (((a % 10) * 10 + (a / 10)) * b = 172) →  -- reversing digits of a and multiplying by b gives 172
  (a * b = 136) :=
by sorry

end correct_product_l2067_206767


namespace max_value_expression_l2067_206789

theorem max_value_expression (x y : ℝ) : 
  (Real.sqrt (3 - Real.sqrt 2) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 1) *
  (3 + 2 * Real.sqrt (7 - Real.sqrt 2) * Real.cos y - Real.cos (2 * y)) ≤ 10 := by
  sorry

end max_value_expression_l2067_206789


namespace reciprocal_of_negative_fraction_l2067_206791

theorem reciprocal_of_negative_fraction (n : ℕ) (h : n ≠ 0) :
  ((-1 : ℚ) / n)⁻¹ = -n := by
  sorry

end reciprocal_of_negative_fraction_l2067_206791


namespace sqrt_sum_2160_l2067_206785

theorem sqrt_sum_2160 (a b : ℕ+) : 
  a < b → 
  (a.val : ℝ).sqrt + (b.val : ℝ).sqrt = Real.sqrt 2160 → 
  a ∈ ({15, 60, 135, 240, 375} : Set ℕ+) := by
sorry

end sqrt_sum_2160_l2067_206785


namespace juan_lunch_time_l2067_206740

/-- The number of pages in Juan's book -/
def book_pages : ℕ := 4000

/-- The number of pages Juan reads per hour -/
def pages_per_hour : ℕ := 250

/-- The time it takes Juan to read the entire book, in hours -/
def reading_time : ℚ := book_pages / pages_per_hour

/-- The time it takes Juan to grab lunch from his office and back, in hours -/
def lunch_time : ℚ := reading_time / 2

theorem juan_lunch_time : lunch_time = 8 := by
  sorry

end juan_lunch_time_l2067_206740


namespace safe_combinations_l2067_206715

def digits : Finset Nat := {1, 3, 5}

theorem safe_combinations : Fintype.card (Equiv.Perm digits) = 6 := by
  sorry

end safe_combinations_l2067_206715


namespace choose_leaders_count_l2067_206765

/-- Represents the number of members in each category -/
structure ClubMembers where
  senior_boys : Nat
  junior_boys : Nat
  senior_girls : Nat
  junior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_leaders (members : ClubMembers) : Nat :=
  let boys_combinations := members.senior_boys * members.junior_boys * 2
  let girls_combinations := members.senior_girls * members.junior_girls * 2
  boys_combinations + girls_combinations

/-- Theorem stating the number of ways to choose leaders under given conditions -/
theorem choose_leaders_count (members : ClubMembers) 
  (h1 : members.senior_boys = 6)
  (h2 : members.junior_boys = 6)
  (h3 : members.senior_girls = 6)
  (h4 : members.junior_girls = 6) :
  choose_leaders members = 144 := by
  sorry

#eval choose_leaders ⟨6, 6, 6, 6⟩

end choose_leaders_count_l2067_206765


namespace binomial_expansion_coefficient_l2067_206741

theorem binomial_expansion_coefficient (x : ℝ) : 
  let expansion := (x - 2 / Real.sqrt x) ^ 7
  ∃ (a b c : ℝ), expansion = a*x + 560*x + b*x^2 + c :=
by sorry

end binomial_expansion_coefficient_l2067_206741


namespace consecutive_odd_numbers_sum_l2067_206755

theorem consecutive_odd_numbers_sum (n1 n2 n3 : ℕ) : 
  (n1 % 2 = 1) →  -- n1 is odd
  (n2 = n1 + 2) →  -- n2 is the next consecutive odd number
  (n3 = n2 + 2) →  -- n3 is the next consecutive odd number after n2
  (n3 = 27) →      -- the largest number is 27
  (n1 + n2 + n3 ≠ 72) :=  -- their sum cannot be 72
by sorry

end consecutive_odd_numbers_sum_l2067_206755


namespace integer_roots_of_polynomials_l2067_206777

def poly1 (x : ℤ) : ℤ := 2 * x^3 - 3 * x^2 - 11 * x + 6
def poly2 (x : ℤ) : ℤ := x^4 + 4 * x^3 - 9 * x^2 - 16 * x + 20

theorem integer_roots_of_polynomials :
  (∀ x : ℤ, poly1 x = 0 ↔ x = -2 ∨ x = 3) ∧
  (∀ x : ℤ, poly2 x = 0 ↔ x = 1 ∨ x = 2 ∨ x = -2 ∨ x = -5) :=
sorry

end integer_roots_of_polynomials_l2067_206777


namespace graph_equation_two_lines_l2067_206728

theorem graph_equation_two_lines (x y : ℝ) :
  (x - y)^2 = x^2 + y^2 ↔ x = 0 ∨ y = 0 := by
  sorry

end graph_equation_two_lines_l2067_206728


namespace mouse_jump_distance_l2067_206713

theorem mouse_jump_distance (grasshopper_jump frog_jump mouse_jump : ℕ) :
  grasshopper_jump = 25 →
  frog_jump = grasshopper_jump + 32 →
  mouse_jump = frog_jump - 26 →
  mouse_jump = 31 := by sorry

end mouse_jump_distance_l2067_206713


namespace smallest_number_in_specific_integer_set_l2067_206716

theorem smallest_number_in_specific_integer_set :
  ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    (a + b + c : ℚ) / 3 = 30 →
    b = 29 →
    max a (max b c) = b + 4 →
    min a (min b c) = 28 :=
by sorry

end smallest_number_in_specific_integer_set_l2067_206716
