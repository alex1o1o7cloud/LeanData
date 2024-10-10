import Mathlib

namespace composition_computation_stages_l3583_358350

/-- A structure representing a linear function f(x) = px + q -/
structure LinearFunction where
  p : ℝ
  q : ℝ

/-- Computes the composition of n linear functions -/
def compose_linear_functions (fs : List LinearFunction) (x : ℝ) : ℝ := sorry

/-- Represents a computation stage -/
inductive Stage
  | init : Stage
  | next : Stage → Stage

/-- Counts the number of stages -/
def stage_count : Stage → Nat
  | Stage.init => 0
  | Stage.next s => stage_count s + 1

/-- Theorem stating that the composition can be computed in no more than 30 stages -/
theorem composition_computation_stages
  (fs : List LinearFunction)
  (h_length : fs.length = 1000)
  (x₀ : ℝ) :
  ∃ (s : Stage), stage_count s ≤ 30 ∧ compose_linear_functions fs x₀ = sorry :=
by sorry

end composition_computation_stages_l3583_358350


namespace inequality_implication_l3583_358302

theorem inequality_implication (a b : ℝ) (h : a < b) : 1 - a > 1 - b := by
  sorry

end inequality_implication_l3583_358302


namespace sum_of_coefficients_l3583_358303

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (1 - 2*x)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ + 7*a₇ + 8*a₈ = 16 :=
by
  sorry

end sum_of_coefficients_l3583_358303


namespace crate_stacking_probability_l3583_358357

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of ways to arrange n crates with 3 possible orientations each -/
def totalArrangements (n : ℕ) : ℕ :=
  3^n

/-- Calculates the number of ways to arrange crates to reach a specific height -/
def validArrangements (n : ℕ) (target_height : ℕ) : ℕ :=
  sorry  -- Placeholder for the actual calculation

/-- The probability of achieving the target height -/
def probability (n : ℕ) (target_height : ℕ) : ℚ :=
  (validArrangements n target_height : ℚ) / (totalArrangements n : ℚ)

theorem crate_stacking_probability :
  let crate_dims : CrateDimensions := ⟨3, 5, 7⟩
  let num_crates : ℕ := 10
  let target_height : ℕ := 43
  probability num_crates target_height = 10 / 6561 := by
  sorry

end crate_stacking_probability_l3583_358357


namespace ellipse_max_value_l3583_358380

theorem ellipse_max_value (x y : ℝ) :
  x^2 / 9 + y^2 = 1 → x + 3 * y ≤ 3 * Real.sqrt 2 := by sorry

end ellipse_max_value_l3583_358380


namespace solve_for_A_l3583_358328

theorem solve_for_A : ∃ A : ℚ, 80 - (5 - (6 + A * (7 - 8 - 5))) = 89 ∧ A = -4/3 := by
  sorry

end solve_for_A_l3583_358328


namespace count_non_zero_area_triangles_l3583_358393

/-- The total number of dots in the grid -/
def total_dots : ℕ := 17

/-- The number of collinear dots in each direction (horizontal and vertical) -/
def collinear_dots : ℕ := 9

/-- The number of ways to choose 3 dots from the total dots -/
def total_combinations : ℕ := Nat.choose total_dots 3

/-- The number of ways to choose 3 collinear dots -/
def collinear_combinations : ℕ := Nat.choose collinear_dots 3

/-- The number of lines with collinear dots (horizontal and vertical) -/
def collinear_lines : ℕ := 2

/-- The number of triangles with non-zero area -/
def non_zero_area_triangles : ℕ := total_combinations - collinear_lines * collinear_combinations

theorem count_non_zero_area_triangles : non_zero_area_triangles = 512 := by
  sorry

end count_non_zero_area_triangles_l3583_358393


namespace linear_equation_condition_l3583_358340

theorem linear_equation_condition (m : ℝ) :
  (∃ x, (3*m - 1)*x + 9 = 0) ∧ (∀ x y, (3*m - 1)*x + 9 = 0 ∧ (3*m - 1)*y + 9 = 0 → x = y) →
  m ≠ 1/3 := by
sorry

end linear_equation_condition_l3583_358340


namespace tub_capacity_l3583_358343

/-- Calculates the capacity of a tub given specific filling conditions -/
theorem tub_capacity 
  (flow_rate : ℕ) 
  (escape_rate : ℕ) 
  (cycle_time : ℕ) 
  (total_time : ℕ) 
  (h1 : flow_rate = 12)
  (h2 : escape_rate = 1)
  (h3 : cycle_time = 2)
  (h4 : total_time = 24) :
  (total_time / cycle_time) * (flow_rate - escape_rate - escape_rate) = 120 :=
by sorry

end tub_capacity_l3583_358343


namespace min_even_integers_l3583_358398

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b = 29 → 
  a + b + c + d = 47 → 
  a + b + c + d + e + f + g = 66 → 
  (∃ (count : ℕ), count ≥ 1 ∧ 
    count = (if Even a then 1 else 0) + 
            (if Even b then 1 else 0) + 
            (if Even c then 1 else 0) + 
            (if Even d then 1 else 0) + 
            (if Even e then 1 else 0) + 
            (if Even f then 1 else 0) + 
            (if Even g then 1 else 0) ∧
    ∀ (other_count : ℕ), 
      other_count = (if Even a then 1 else 0) + 
                    (if Even b then 1 else 0) + 
                    (if Even c then 1 else 0) + 
                    (if Even d then 1 else 0) + 
                    (if Even e then 1 else 0) + 
                    (if Even f then 1 else 0) + 
                    (if Even g then 1 else 0) →
      count ≤ other_count) :=
by sorry

end min_even_integers_l3583_358398


namespace dinitrogen_pentoxide_weight_l3583_358370

/-- The atomic weight of Nitrogen in g/mol -/
def atomic_weight_N : ℝ := 14.01

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Nitrogen atoms in Dinitrogen pentoxide -/
def N_count : ℕ := 2

/-- The number of Oxygen atoms in Dinitrogen pentoxide -/
def O_count : ℕ := 5

/-- The molecular weight of Dinitrogen pentoxide in g/mol -/
def molecular_weight_N2O5 : ℝ := N_count * atomic_weight_N + O_count * atomic_weight_O

theorem dinitrogen_pentoxide_weight :
  molecular_weight_N2O5 = 108.02 := by
  sorry

end dinitrogen_pentoxide_weight_l3583_358370


namespace min_oranges_in_new_box_l3583_358324

theorem min_oranges_in_new_box (m n x : ℕ) : 
  m + n ≤ 60 →
  59 * m = 60 * n + x →
  x > 0 →
  (∀ y : ℕ, y < x → ¬(∃ m' n' : ℕ, m' + n' ≤ 60 ∧ 59 * m' = 60 * n' + y)) →
  x = 30 :=
by sorry

end min_oranges_in_new_box_l3583_358324


namespace smallest_positive_period_cos_l3583_358368

/-- The smallest positive period of cos(π/3 - 2x/5) is 5π -/
theorem smallest_positive_period_cos (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (π/3 - 2*x/5)
  ∃ (T : ℝ), T > 0 ∧ (∀ t, f (t + T) = f t) ∧ 
  (∀ S, S > 0 ∧ (∀ t, f (t + S) = f t) → T ≤ S) ∧
  T = 5*π :=
by sorry

end smallest_positive_period_cos_l3583_358368


namespace norm_scalar_multiple_l3583_358383

variable (n : ℕ)
variable (v : Fin n → ℝ)

theorem norm_scalar_multiple
  (h : ‖v‖ = 6) :
  ‖(5 : ℝ) • v‖ = 30 := by
sorry

end norm_scalar_multiple_l3583_358383


namespace inequality_proof_l3583_358311

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  Real.sqrt (16 * a^2 + 9) + Real.sqrt (16 * b^2 + 9) + Real.sqrt (16 * c^2 + 9) ≥ 3 + 4 * (a + b + c) :=
by sorry

end inequality_proof_l3583_358311


namespace sandy_average_book_price_l3583_358382

theorem sandy_average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) : 
  books1 = 65 → 
  books2 = 55 → 
  price1 = 1480 → 
  price2 = 920 → 
  (price1 + price2) / (books1 + books2 : ℚ) = 20 := by
sorry

end sandy_average_book_price_l3583_358382


namespace min_value_a_plus_4b_l3583_358323

theorem min_value_a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b) :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = x + y → a + 4 * b ≤ x + 4 * y ∧ 
  (a + 4 * b = 9 ↔ a = 3 ∧ b = 3/2) := by
  sorry

end min_value_a_plus_4b_l3583_358323


namespace perpendicular_length_l3583_358353

/-- A parallelogram with a diagonal of length 'd' and area 'a' has a perpendicular of length 'h' dropped on that diagonal. -/
structure Parallelogram where
  d : ℝ  -- length of the diagonal
  a : ℝ  -- area of the parallelogram
  h : ℝ  -- length of the perpendicular dropped on the diagonal

/-- The area of a parallelogram is equal to the product of its diagonal and the perpendicular dropped on that diagonal. -/
axiom area_formula (p : Parallelogram) : p.a = p.d * p.h

/-- For a parallelogram with a diagonal of 30 meters and an area of 600 square meters, 
    the length of the perpendicular dropped on the diagonal is 20 meters. -/
theorem perpendicular_length : 
  ∀ (p : Parallelogram), p.d = 30 → p.a = 600 → p.h = 20 := by
  sorry


end perpendicular_length_l3583_358353


namespace triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral_l3583_358358

open Real

/-- Represents a triangle with sides a, b, c and angles α, β, γ -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : α + β + γ = π

/-- The sides of the triangle form an arithmetic sequence -/
def SidesArithmeticSequence (t : Triangle) : Prop :=
  2 / t.b = 1 / t.a + 1 / t.c

/-- The angles of the triangle form an arithmetic sequence -/
def AnglesArithmeticSequence (t : Triangle) : Prop :=
  2 * t.β = t.α + t.γ

/-- A triangle is equilateral if all its sides are equal -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

theorem triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral
  (t : Triangle)
  (h_sides : SidesArithmeticSequence t)
  (h_angles : AnglesArithmeticSequence t) :
  IsEquilateral t :=
sorry

end triangle_with_arithmetic_angles_and_reciprocal_sides_is_equilateral_l3583_358358


namespace expression_evaluation_l3583_358339

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_evaluation_l3583_358339


namespace complex_sum_equality_l3583_358349

theorem complex_sum_equality : 
  5 * Complex.exp (Complex.I * Real.pi / 12) + 5 * Complex.exp (Complex.I * 13 * Real.pi / 24) 
  = 10 * Real.cos (11 * Real.pi / 48) * Complex.exp (Complex.I * 5 * Real.pi / 16) := by
  sorry

end complex_sum_equality_l3583_358349


namespace stan_magician_payment_l3583_358304

def magician_payment (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_per_week * num_weeks

theorem stan_magician_payment :
  magician_payment 60 3 7 2 = 2520 := by
  sorry

end stan_magician_payment_l3583_358304


namespace rectangular_solid_volume_l3583_358342

theorem rectangular_solid_volume (a b c : ℝ) 
  (h_top : a * b = 15)
  (h_front : b * c = 10)
  (h_side : c * a = 6) :
  a * b * c = 30 := by
sorry

end rectangular_solid_volume_l3583_358342


namespace gcd_from_lcm_and_ratio_l3583_358392

theorem gcd_from_lcm_and_ratio (A B : ℕ+) : 
  A.lcm B = 240 → A.val * 6 = B.val * 5 → A.gcd B = 8 := by sorry

end gcd_from_lcm_and_ratio_l3583_358392


namespace trigonometric_expression_value_l3583_358394

theorem trigonometric_expression_value (θ : Real) (h : Real.tan θ = 3) :
  (Real.sin (3 * Real.pi / 2 + θ) + 2 * Real.cos (Real.pi - θ)) /
  (Real.sin (Real.pi / 2 - θ) - Real.sin (Real.pi - θ)) = 3 / 2 := by
  sorry

end trigonometric_expression_value_l3583_358394


namespace cot_thirty_degrees_l3583_358316

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end cot_thirty_degrees_l3583_358316


namespace ellipse_max_b_l3583_358337

/-- Given an ellipse x^2 + y^2/b^2 = 1 where 0 < b < 1, with foci F1 and F2 at distance 2c apart,
    if there exists a point P on the ellipse such that the distance from P to the line x = 1/c
    is the arithmetic mean of |PF1| and |PF2|, then the maximum value of b is √3/2. -/
theorem ellipse_max_b (b c : ℝ) (h1 : 0 < b) (h2 : b < 1) :
  (∃ (x y : ℝ), x^2 + y^2/b^2 = 1 ∧
    ∃ (PF1 PF2 : ℝ), |x - 1/c| = (PF1 + PF2)/2 ∧
      ∃ (c_foci : ℝ), c_foci = 2*c) →
  b ≤ Real.sqrt 3 / 2 :=
by sorry

end ellipse_max_b_l3583_358337


namespace linear_function_through_points_l3583_358385

/-- A linear function passing through point A(1, -1) -/
def f (a : ℝ) (x : ℝ) : ℝ := -x + a

theorem linear_function_through_points :
  ∃ (a : ℝ), f a 1 = -1 ∧ f a (-2) = 4 := by
  sorry

end linear_function_through_points_l3583_358385


namespace watermelon_total_sold_l3583_358341

def watermelon_problem (customers_one : Nat) (customers_three : Nat) (customers_two : Nat) : Nat :=
  customers_one * 1 + customers_three * 3 + customers_two * 2

theorem watermelon_total_sold :
  watermelon_problem 17 3 10 = 46 := by
  sorry

end watermelon_total_sold_l3583_358341


namespace fold_square_crease_l3583_358361

/-- Given a square ABCD with side length 18 cm, if point B is folded to point E on AD
    such that DE = 6 cm, and the resulting crease intersects AB at point F,
    then the length of FB is 13 cm. -/
theorem fold_square_crease (A B C D E F : ℝ × ℝ) : 
  -- Square ABCD with side length 18
  (A = (0, 0) ∧ B = (18, 0) ∧ C = (18, 18) ∧ D = (0, 18)) →
  -- E is on AD and DE = 6
  (E.1 = 0 ∧ E.2 = 12) →
  -- F is on AB
  (F.2 = 0) →
  -- F is on the perpendicular bisector of BE
  (F.2 - 6 = (3/2) * (F.1 - 9)) →
  -- The length of FB is 13
  Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2) = 13 :=
by sorry

end fold_square_crease_l3583_358361


namespace age_difference_l3583_358364

theorem age_difference (li_age zhang_age jung_age : ℕ) : 
  li_age = 12 →
  zhang_age = 2 * li_age →
  jung_age = 26 →
  jung_age - zhang_age = 2 := by
  sorry

end age_difference_l3583_358364


namespace letter_lock_unsuccessful_attempts_l3583_358317

/-- A letter lock with a given number of rings and letters per ring -/
structure LetterLock where
  num_rings : ℕ
  letters_per_ring : ℕ

/-- The number of distinct unsuccessful attempts for a given letter lock -/
def unsuccessfulAttempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem: For a letter lock with 3 rings and 6 letters per ring, 
    the number of distinct unsuccessful attempts is 215 -/
theorem letter_lock_unsuccessful_attempts :
  ∃ (lock : LetterLock), lock.num_rings = 3 ∧ lock.letters_per_ring = 6 ∧ 
  unsuccessfulAttempts lock = 215 := by
  sorry

end letter_lock_unsuccessful_attempts_l3583_358317


namespace five_from_second_row_wading_l3583_358363

/-- Represents the beach scenario with people in rows and some wading in the water -/
structure BeachScenario where
  initial_first_row : ℕ
  initial_second_row : ℕ
  third_row : ℕ
  first_row_wading : ℕ
  remaining_on_beach : ℕ

/-- Calculates the number of people from the second row who joined those wading in the water -/
def second_row_wading (scenario : BeachScenario) : ℕ :=
  scenario.initial_first_row + scenario.initial_second_row + scenario.third_row
  - scenario.first_row_wading - scenario.remaining_on_beach

/-- Theorem stating that 5 people from the second row joined those wading in the water -/
theorem five_from_second_row_wading (scenario : BeachScenario)
  (h1 : scenario.initial_first_row = 24)
  (h2 : scenario.initial_second_row = 20)
  (h3 : scenario.third_row = 18)
  (h4 : scenario.first_row_wading = 3)
  (h5 : scenario.remaining_on_beach = 54) :
  second_row_wading scenario = 5 := by
  sorry

end five_from_second_row_wading_l3583_358363


namespace seventh_roots_of_unity_polynomial_factorization_l3583_358348

theorem seventh_roots_of_unity_polynomial_factorization (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = (x^2 + b₁*x + c₁)*(x^2 + b₂*x + c₂)*(x^2 + b₃*x + c₃)) :
  b₁*c₁ + b₂*c₂ + b₃*c₃ = 1 := by
sorry

end seventh_roots_of_unity_polynomial_factorization_l3583_358348


namespace no_natural_number_power_of_two_l3583_358334

theorem no_natural_number_power_of_two : 
  ∀ n : ℕ, ¬∃ k : ℕ, n^2012 - 1 = 2^k := by
  sorry

end no_natural_number_power_of_two_l3583_358334


namespace scenic_spot_assignment_l3583_358336

/-- The number of scenic spots -/
def num_spots : ℕ := 3

/-- The number of people -/
def num_people : ℕ := 4

/-- The total number of possible assignments without restrictions -/
def total_assignments : ℕ := num_spots ^ num_people

/-- The number of assignments where A and B are in the same spot -/
def restricted_assignments : ℕ := num_spots * (num_spots ^ (num_people - 2))

/-- The number of valid assignments where A and B are not in the same spot -/
def valid_assignments : ℕ := total_assignments - restricted_assignments

theorem scenic_spot_assignment :
  valid_assignments = 54 := by sorry

end scenic_spot_assignment_l3583_358336


namespace octopus_equality_month_l3583_358389

theorem octopus_equality_month : 
  (∀ k : ℕ, k < 4 → 3^(k + 1) ≠ 15 * 5^k) ∧ 
  3^(4 + 1) = 15 * 5^4 := by
  sorry

end octopus_equality_month_l3583_358389


namespace max_value_x_2y_plus_1_l3583_358384

theorem max_value_x_2y_plus_1 (x y : ℝ) 
  (hx : |x - 1| ≤ 1) 
  (hy : |y - 2| ≤ 1) : 
  |x - 2*y + 1| ≤ 5 := by sorry

end max_value_x_2y_plus_1_l3583_358384


namespace total_sum_lent_total_sum_lent_proof_l3583_358388

/-- Proves that the total sum lent is 2795 rupees given the problem conditions -/
theorem total_sum_lent : ℕ → Prop := fun total_sum =>
  ∃ (first_part second_part : ℕ),
    -- The sum is divided into two parts
    total_sum = first_part + second_part ∧
    -- Interest on first part for 8 years at 3% per annum equals interest on second part for 3 years at 5% per annum
    (first_part * 3 * 8) = (second_part * 5 * 3) ∧
    -- The second part is Rs. 1720
    second_part = 1720 ∧
    -- The total sum lent is 2795 rupees
    total_sum = 2795

/-- The proof of the theorem -/
theorem total_sum_lent_proof : total_sum_lent 2795 := by
  sorry

end total_sum_lent_total_sum_lent_proof_l3583_358388


namespace prime_factorial_divisibility_l3583_358390

theorem prime_factorial_divisibility (p k n : ℕ) (hp : Prime p) :
  p^k ∣ n! → (p!)^k ∣ n! := by
  sorry

end prime_factorial_divisibility_l3583_358390


namespace concert_friends_count_l3583_358396

theorem concert_friends_count : 
  ∀ (P : ℝ), P > 0 → 
  ∃ (F : ℕ), 
    (F : ℝ) * P = ((F + 1 : ℕ) : ℝ) * P * (1 - 0.25) ∧ 
    F = 3 := by
  sorry

end concert_friends_count_l3583_358396


namespace sum_of_products_l3583_358375

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 36) :
  x*y + y*z + x*z = 18 := by
sorry

end sum_of_products_l3583_358375


namespace unit_circle_trig_values_l3583_358313

theorem unit_circle_trig_values :
  ∀ y : ℝ,
  ((-Real.sqrt 3 / 2) ^ 2 + y ^ 2 = 1) →
  ∃ θ : ℝ,
  (0 < θ ∧ θ < 2 * Real.pi) ∧
  (Real.sin θ = y ∧ Real.cos θ = -Real.sqrt 3 / 2) ∧
  (y = 1 / 2 ∨ y = -1 / 2) :=
by sorry

end unit_circle_trig_values_l3583_358313


namespace garden_fence_columns_l3583_358362

theorem garden_fence_columns (S C : ℕ) : 
  S * C + (S - 1) / 2 = 1223 → 
  S = 2 * C + 5 → 
  C = 23 := by sorry

end garden_fence_columns_l3583_358362


namespace multiplication_sum_equality_l3583_358367

theorem multiplication_sum_equality : 45 * 25 + 55 * 45 + 20 * 45 = 4500 := by
  sorry

end multiplication_sum_equality_l3583_358367


namespace geometric_sequence_product_l3583_358305

theorem geometric_sequence_product (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h2 : a 2 = 2) (h3 : a 6 = 8) : a 3 * a 4 * a 5 = 64 := by
  sorry

end geometric_sequence_product_l3583_358305


namespace functional_equation_solution_l3583_358355

/-- A function from rational numbers to rational numbers -/
def RationalFunction := ℚ → ℚ

/-- The functional equation property -/
def SatisfiesEquation (f : RationalFunction) : Prop :=
  ∀ x y : ℚ, f (x + y) + f (x - y) = 2 * f x + 2 * f y

/-- The theorem statement -/
theorem functional_equation_solution :
  ∀ f : RationalFunction, SatisfiesEquation f →
  ∃ a : ℚ, ∀ x : ℚ, f x = a * x^2 := by sorry

end functional_equation_solution_l3583_358355


namespace smallest_year_after_2000_with_digit_sum_15_l3583_358381

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

theorem smallest_year_after_2000_with_digit_sum_15 :
  (∀ y : ℕ, 2000 < y ∧ y < 2049 → sumOfDigits y ≠ 15) ∧ 
  2000 < 2049 ∧ 
  sumOfDigits 2049 = 15 := by
sorry

end smallest_year_after_2000_with_digit_sum_15_l3583_358381


namespace sum_of_arithmetic_sequence_l3583_358331

theorem sum_of_arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) :
  a₁ = 107 →
  d = 10 →
  aₙ = 447 →
  n = (aₙ - a₁) / d + 1 →
  (n : ℝ) / 2 * (a₁ + aₙ) = 9695 :=
by sorry

end sum_of_arithmetic_sequence_l3583_358331


namespace factorial_simplification_l3583_358377

theorem factorial_simplification : (13 * 12 * 11 * 10 * 9 * Nat.factorial 8) / (10 * 9 * Nat.factorial 8 + 3 * 9 * Nat.factorial 8) = 1320 := by
  sorry

end factorial_simplification_l3583_358377


namespace sum_ages_theorem_l3583_358359

/-- The sum of Josiah's and Hans' ages after 3 years, given their current ages -/
def sum_ages_after_3_years (hans_current_age : ℕ) (josiah_current_age : ℕ) : ℕ :=
  (hans_current_age + 3) + (josiah_current_age + 3)

/-- Theorem stating the sum of Josiah's and Hans' ages after 3 years -/
theorem sum_ages_theorem (hans_current_age : ℕ) (josiah_current_age : ℕ) 
  (h1 : hans_current_age = 15)
  (h2 : josiah_current_age = 3 * hans_current_age) :
  sum_ages_after_3_years hans_current_age josiah_current_age = 66 := by
sorry

end sum_ages_theorem_l3583_358359


namespace indira_cricket_time_l3583_358356

def total_minutes : ℕ := 15000

def amaya_cricket_minutes : ℕ := 75 * 5 * 4
def amaya_basketball_minutes : ℕ := 45 * 2 * 4
def sean_cricket_minutes : ℕ := 65 * 14
def sean_basketball_minutes : ℕ := 55 * 4

def amaya_sean_total : ℕ := amaya_cricket_minutes + amaya_basketball_minutes + sean_cricket_minutes + sean_basketball_minutes

theorem indira_cricket_time :
  total_minutes - amaya_sean_total = 12010 :=
sorry

end indira_cricket_time_l3583_358356


namespace expand_expression_l3583_358397

theorem expand_expression (y : ℝ) : (7 * y + 12) * (3 * y) = 21 * y^2 + 36 * y := by
  sorry

end expand_expression_l3583_358397


namespace abs_diff_neg_self_l3583_358344

theorem abs_diff_neg_self (m : ℝ) (h : m < 0) : |m - (-m)| = -2*m := by
  sorry

end abs_diff_neg_self_l3583_358344


namespace simplify_expression_l3583_358329

theorem simplify_expression (z : ℝ) : (3 - 5*z^2) - (4*z^2 + 2*z - 5) = 8 - 9*z^2 - 2*z := by
  sorry

end simplify_expression_l3583_358329


namespace expression_evaluation_l3583_358308

theorem expression_evaluation :
  let x : ℚ := -2
  (1 - 1 / (1 - x)) / (x^2 / (x^2 - 1)) = 1/2 :=
by sorry

end expression_evaluation_l3583_358308


namespace positive_integer_solution_iff_n_eq_three_l3583_358399

theorem positive_integer_solution_iff_n_eq_three (n : ℕ) :
  (∃ (x y z : ℕ+), x^2 + y^2 + z^2 = n * x * y * z) ↔ n = 3 :=
sorry

end positive_integer_solution_iff_n_eq_three_l3583_358399


namespace magnitude_of_vector_difference_l3583_358373

/-- Given two vectors in ℝ³, prove that the magnitude of their difference is 3 -/
theorem magnitude_of_vector_difference (a b : ℝ × ℝ × ℝ) :
  a = (1, 0, 2) → b = (0, 1, 2) →
  ‖a - 2 • b‖ = 3 := by sorry

end magnitude_of_vector_difference_l3583_358373


namespace class_transfer_problem_l3583_358369

/-- Proof of the class transfer problem -/
theorem class_transfer_problem :
  -- Define the total number of students
  ∀ (total : ℕ),
  -- Define the number of students transferred from A to B
  ∀ (transfer_a_to_b : ℕ),
  -- Define the number of students transferred from B to C
  ∀ (transfer_b_to_c : ℕ),
  -- Condition: total students is 92
  total = 92 →
  -- Condition: 5 students transferred from A to B
  transfer_a_to_b = 5 →
  -- Condition: 32 students transferred from B to C
  transfer_b_to_c = 32 →
  -- Condition: After transfers, students in A = 3 * students in B
  ∃ (final_a final_b : ℕ),
    final_a = 3 * final_b ∧
    final_a + final_b = total - transfer_b_to_c →
  -- Conclusion: Originally 45 students in A and 47 in B
  ∃ (original_a original_b : ℕ),
    original_a = 45 ∧
    original_b = 47 ∧
    original_a + original_b = total :=
by sorry

end class_transfer_problem_l3583_358369


namespace regular_polygon_sides_l3583_358345

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 18 → n * exterior_angle = 360 → n = 20 := by
  sorry

end regular_polygon_sides_l3583_358345


namespace croissant_count_is_two_l3583_358330

/-- Represents the number of items bought at each price point -/
structure ItemCounts where
  expensive : ℕ
  cheap : ℕ

/-- Calculates the total cost given the number of items at each price point -/
def totalCost (counts : ItemCounts) : ℚ :=
  1.5 * counts.expensive + 1.2 * counts.cheap

/-- Checks if a rational number is a whole number -/
def isWholeNumber (q : ℚ) : Prop :=
  ∃ n : ℤ, q = n

/-- The main theorem to be proved -/
theorem croissant_count_is_two :
  ∀ counts : ItemCounts,
    counts.expensive + counts.cheap = 7 →
    isWholeNumber (totalCost counts) →
    counts.expensive = 2 :=
by sorry

end croissant_count_is_two_l3583_358330


namespace expression_evaluation_l3583_358387

theorem expression_evaluation :
  let x : ℝ := 3 * Real.sqrt 3 + 2 * Real.sqrt 2
  let y : ℝ := 3 * Real.sqrt 3 - 2 * Real.sqrt 2
  ((x * (x + y) + 2 * y * (x + y)) / (x * y * (x + 2 * y))) / ((x * y) / (x + 2 * y)) = 108 := by
  sorry

end expression_evaluation_l3583_358387


namespace total_gas_cost_l3583_358352

-- Define the given parameters
def miles_per_gallon : ℝ := 50
def miles_per_day : ℝ := 75
def price_per_gallon : ℝ := 3
def number_of_days : ℝ := 10

-- Define the theorem
theorem total_gas_cost : 
  (number_of_days * miles_per_day / miles_per_gallon) * price_per_gallon = 45 :=
by
  sorry


end total_gas_cost_l3583_358352


namespace lucky_lila_coincidence_l3583_358306

theorem lucky_lila_coincidence (a b c d e f : ℚ) : 
  a = 2 → b = 3 → c = 4 → d = 5 → f = 6 →
  (a * b * c * d * e / f = a * (b - (c * (d - (e / f))))) →
  e = -51 / 28 := by
sorry

end lucky_lila_coincidence_l3583_358306


namespace almond_butter_servings_l3583_358371

def container_amount : ℚ := 34 + 3/5
def serving_size : ℚ := 5 + 1/2

theorem almond_butter_servings :
  (container_amount / serving_size : ℚ) = 6 + 21/55 := by
  sorry

end almond_butter_servings_l3583_358371


namespace find_number_l3583_358346

theorem find_number : ∃ n : ℝ, 7 * n - 15 = 2 * n + 10 ∧ n = 5 := by
  sorry

end find_number_l3583_358346


namespace timothy_read_300_pages_l3583_358312

/-- The total number of pages Timothy read in a week -/
def total_pages_read : ℕ :=
  let monday_tuesday := 2 * 45
  let wednesday := 50
  let thursday_to_saturday := 3 * 40
  let sunday := 25 + 15
  monday_tuesday + wednesday + thursday_to_saturday + sunday

/-- Theorem stating that Timothy read 300 pages in total -/
theorem timothy_read_300_pages : total_pages_read = 300 := by
  sorry

end timothy_read_300_pages_l3583_358312


namespace book_arrangement_count_l3583_358366

theorem book_arrangement_count :
  let math_books : ℕ := 4
  let english_books : ℕ := 6
  let particular_english_book : ℕ := 1
  let math_block_arrangements : ℕ := Nat.factorial math_books
  let english_block_arrangements : ℕ := Nat.factorial (english_books - particular_english_book)
  let block_arrangements : ℕ := 1  -- Only one way to arrange the two blocks due to the particular book constraint
  block_arrangements * math_block_arrangements * english_block_arrangements = 2880
  := by sorry

end book_arrangement_count_l3583_358366


namespace geometric_sequence_common_ratio_l3583_358301

/-- Given a geometric sequence {a_n} where a_{2020} = 8a_{2017}, prove that the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- Definition of geometric sequence
  a 2020 = 8 * a 2017 →         -- Given condition
  q = 2 :=                      -- Conclusion to prove
by
  sorry


end geometric_sequence_common_ratio_l3583_358301


namespace sum_frequencies_equals_total_data_l3583_358347

/-- Represents a frequency distribution table -/
structure FrequencyDistributionTable where
  groups : List ℕ  -- List of frequencies for each group
  total_data : ℕ   -- Total number of data points

/-- 
Theorem: In a frequency distribution table, the sum of the frequencies 
of all groups is equal to the total number of data points.
-/
theorem sum_frequencies_equals_total_data (table : FrequencyDistributionTable) : 
  table.groups.sum = table.total_data := by
  sorry


end sum_frequencies_equals_total_data_l3583_358347


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l3583_358354

/-- Given a natural number, returns the sum of its digits. -/
def digitSum (n : ℕ) : ℕ := sorry

/-- Returns true if the number is a three-digit number. -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, isThreeDigit n → n % 9 = 0 → digitSum n = 27 → n ≤ 999 := by sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l3583_358354


namespace unique_counterexample_l3583_358360

-- Define geometric figures
inductive GeometricFigure
| Line
| Plane

-- Define spatial relationships
def perpendicular (a b : GeometricFigure) : Prop := sorry
def parallel (a b : GeometricFigure) : Prop := sorry

-- Define the proposition
def proposition (x y z : GeometricFigure) : Prop :=
  (perpendicular x y ∧ parallel y z) → perpendicular x z

-- Theorem statement
theorem unique_counterexample :
  ∀ x y z : GeometricFigure,
    ¬proposition x y z ↔ 
      x = GeometricFigure.Line ∧ 
      y = GeometricFigure.Line ∧ 
      z = GeometricFigure.Plane :=
sorry

end unique_counterexample_l3583_358360


namespace sixth_term_seq1_sixth_term_seq2_l3583_358391

-- Define the first sequence
def seq1 (n : ℕ) : ℕ := 3 * n

-- Define the second sequence
def seq2 (n : ℕ) : ℕ := n * n

-- Theorem for the first sequence
theorem sixth_term_seq1 : seq1 5 = 15 := by sorry

-- Theorem for the second sequence
theorem sixth_term_seq2 : seq2 6 = 36 := by sorry

end sixth_term_seq1_sixth_term_seq2_l3583_358391


namespace polynomial_division_theorem_l3583_358310

noncomputable def polynomial_remainder 
  (p : ℝ → ℝ) (a b : ℝ) (h : a ≠ b) : ℝ × ℝ × ℝ :=
  let r := (p a - p b - (b - a) * (deriv p a)) / ((b - a) * (b + a))
  let d := deriv p a - 2 * r * a
  let e := p a - r * a^2 - d * a
  (r, d, e)

theorem polynomial_division_theorem 
  (p : ℝ → ℝ) (a b : ℝ) (h : a ≠ b) :
  ∃ (q : ℝ → ℝ) (r d e : ℝ),
    (∀ x, p x = q x * (x - a)^2 * (x - b) + r * x^2 + d * x + e) ∧
    (r, d, e) = polynomial_remainder p a b h :=
sorry

end polynomial_division_theorem_l3583_358310


namespace shortest_distance_to_parabola_l3583_358333

/-- The parabola defined by x = 2y^2 -/
def parabola (y : ℝ) : ℝ := 2 * y^2

/-- The point from which we measure the distance -/
def point : ℝ × ℝ := (8, 14)

/-- The shortest distance between the point and the parabola -/
def shortest_distance : ℝ := 26

/-- Theorem stating that the shortest distance between the point (8,14) and the parabola x = 2y^2 is 26 -/
theorem shortest_distance_to_parabola :
  ∃ (y : ℝ), 
    shortest_distance = 
      Real.sqrt ((parabola y - point.1)^2 + (y - point.2)^2) ∧
    ∀ (z : ℝ), 
      Real.sqrt ((parabola z - point.1)^2 + (z - point.2)^2) ≥ shortest_distance :=
by
  sorry


end shortest_distance_to_parabola_l3583_358333


namespace harry_terry_calculation_harry_terry_calculation_proof_l3583_358376

theorem harry_terry_calculation : ℤ → ℤ → Prop :=
  fun (H T : ℤ) =>
    (H = 8 - (2 + 5)) ∧ (T = 8 - 2 + 5) → H - T = -10

-- The proof is omitted
theorem harry_terry_calculation_proof : harry_terry_calculation 1 11 := by
  sorry

end harry_terry_calculation_harry_terry_calculation_proof_l3583_358376


namespace equal_numbers_l3583_358395

theorem equal_numbers (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv.Perm (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
sorry

end equal_numbers_l3583_358395


namespace not_all_zero_deriv_is_critical_point_l3583_358374

open Set
open Function
open Filter

/-- A point x₀ is a critical point of a differentiable function f if f'(x₀) = 0 
    and f'(x) changes sign in any neighborhood of x₀. -/
def IsCriticalPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  DifferentiableAt ℝ f x₀ ∧ 
  (deriv f) x₀ = 0 ∧
  ∀ ε > 0, ∃ x₁ x₂, x₁ < x₀ ∧ x₀ < x₂ ∧ 
    abs (x₁ - x₀) < ε ∧ abs (x₂ - x₀) < ε ∧
    (deriv f) x₁ * (deriv f) x₂ < 0

/-- The statement "For all differentiable functions f, if f'(x₀) = 0, 
    then x₀ is a critical point of f" is false. -/
theorem not_all_zero_deriv_is_critical_point :
  ¬ (∀ (f : ℝ → ℝ) (x₀ : ℝ), DifferentiableAt ℝ f x₀ → (deriv f) x₀ = 0 → IsCriticalPoint f x₀) :=
by sorry

end not_all_zero_deriv_is_critical_point_l3583_358374


namespace smallest_n_l3583_358365

/-- Given a positive integer k, N is the smallest positive integer such that
    there exists a set of 2k + 1 distinct positive integers whose sum is greater than N,
    but the sum of any k-element subset is at most N/2 -/
theorem smallest_n (k : ℕ+) : ∃ (N : ℕ),
  N = 2 * k.val^3 + 3 * k.val^2 + 3 * k.val ∧
  (∃ (S : Finset ℕ),
    S.card = 2 * k.val + 1 ∧
    (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y) ∧
    (S.sum id > N) ∧
    (∀ (T : Finset ℕ), T ⊆ S → T.card = k.val → T.sum id ≤ N / 2)) ∧
  (∀ (M : ℕ), M < N →
    ¬∃ (S : Finset ℕ),
      S.card = 2 * k.val + 1 ∧
      (∀ (x y : ℕ), x ∈ S → y ∈ S → x ≠ y) ∧
      (S.sum id > M) ∧
      (∀ (T : Finset ℕ), T ⊆ S → T.card = k.val → T.sum id ≤ M / 2)) :=
by sorry


end smallest_n_l3583_358365


namespace cyclic_product_sum_theorem_l3583_358372

/-- A permutation of (1, 2, 3, 4, 5, 6) -/
def Permutation := Fin 6 → Fin 6

/-- The cyclic product sum for a given permutation -/
def cyclicProductSum (p : Permutation) : ℕ :=
  (p 0) * (p 1) + (p 1) * (p 2) + (p 2) * (p 3) + (p 3) * (p 4) + (p 4) * (p 5) + (p 5) * (p 0)

/-- Predicate to check if a function is a valid permutation of (1, 2, 3, 4, 5, 6) -/
def isValidPermutation (p : Permutation) : Prop :=
  Function.Injective p ∧ Function.Surjective p

/-- The maximum value of the cyclic product sum -/
def M : ℕ := 79

/-- The number of permutations that achieve the maximum value -/
def N : ℕ := 12

theorem cyclic_product_sum_theorem :
  (∀ p : Permutation, isValidPermutation p → cyclicProductSum p ≤ M) ∧
  (∃! (s : Finset Permutation), s.card = N ∧ 
    ∀ p ∈ s, isValidPermutation p ∧ cyclicProductSum p = M) :=
sorry

end cyclic_product_sum_theorem_l3583_358372


namespace smallest_6digit_binary_palindrome_4digit_other_base_l3583_358338

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Counts the number of digits in a number in a given base -/
def digitCount (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_6digit_binary_palindrome_4digit_other_base :
  ∀ n : ℕ,
  isPalindrome n 2 →
  digitCount n 2 = 6 →
  (∃ b : ℕ, b > 2 ∧ isPalindrome (baseConvert n 2 b) b ∧ digitCount (baseConvert n 2 b) b = 4) →
  n ≥ 33 := by sorry

end smallest_6digit_binary_palindrome_4digit_other_base_l3583_358338


namespace equivalent_operations_l3583_358307

theorem equivalent_operations (x : ℝ) : 
  (x * (5/6)) / (2/7) = x * (35/12) :=
by sorry

end equivalent_operations_l3583_358307


namespace urn_probability_theorem_l3583_358326

/-- Represents the number of balls of each color in the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents one draw operation -/
inductive DrawResult
| Red
| Blue

/-- Represents a sequence of 6 draw operations -/
def DrawSequence := Vector DrawResult 6

/-- Initial state of the urn -/
def initial_state : UrnState := ⟨1, 2⟩

/-- Final number of balls in the urn after 6 operations -/
def final_ball_count : ℕ := 8

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of sequences that result in 4 red and 4 blue balls -/
def favorable_sequence_count : ℕ :=
  sorry

/-- Theorem: The probability of having 4 red and 4 blue balls after 6 operations is 5/14 -/
theorem urn_probability_theorem :
  (favorable_sequence_count : ℚ) * sequence_probability (Vector.replicate 6 DrawResult.Red) = 5/14 :=
sorry

end urn_probability_theorem_l3583_358326


namespace total_distance_is_202_l3583_358379

/-- Represents the driving data for a single day -/
structure DailyDrive where
  hours : Float
  speed : Float

/-- Calculates the distance traveled in a day given the driving data -/
def distanceTraveled (drive : DailyDrive) : Float :=
  drive.hours * drive.speed

/-- The week's driving schedule -/
def weekSchedule : List DailyDrive := [
  { hours := 3, speed := 12 },    -- Monday
  { hours := 3.5, speed := 8 },   -- Tuesday
  { hours := 2.5, speed := 12 },  -- Wednesday
  { hours := 4, speed := 6 },     -- Thursday
  { hours := 2, speed := 12 },    -- Friday
  { hours := 3, speed := 15 },    -- Saturday
  { hours := 1.5, speed := 10 }   -- Sunday
]

/-- Theorem: The total distance traveled during the week is 202 km -/
theorem total_distance_is_202 :
  (weekSchedule.map distanceTraveled).sum = 202 := by
  sorry

end total_distance_is_202_l3583_358379


namespace problem_statement_l3583_358309

theorem problem_statement (m n : ℝ) (a b : ℝ) 
  (h1 : m + n = 9)
  (h2 : 0 < a ∧ 0 < b)
  (h3 : a^2 + b^2 = 9) : 
  (∀ x : ℝ, |x - m| + |x + n| ≥ 9) ∧ 
  (a + b) * (a^3 + b^3) ≥ 81 := by
  sorry

end problem_statement_l3583_358309


namespace last_digit_fibonacci_mod12_l3583_358319

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def fibonacci_mod12 (n : ℕ) : ℕ := fibonacci n % 12

def digit_appears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ n ∧ fibonacci_mod12 k = d

theorem last_digit_fibonacci_mod12 :
  ∀ d : ℕ, d < 12 →
    (digit_appears d 21 → digit_appears 11 22) ∧
    (¬ digit_appears 11 21) ∧
    digit_appears 11 22 :=
by sorry

end last_digit_fibonacci_mod12_l3583_358319


namespace is_center_of_hyperbola_l3583_358300

/-- The equation of the hyperbola -/
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 36 * x - 16 * y^2 + 128 * y - 400 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, 4)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_center_of_hyperbola :
  ∀ (x y : ℝ), hyperbola_eq x y ↔ hyperbola_eq (x - hyperbola_center.1) (y - hyperbola_center.2) :=
sorry

end is_center_of_hyperbola_l3583_358300


namespace two_digit_number_sum_l3583_358314

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a + b) →
  (10 * a + b) + (10 * b + a) = 99 := by
sorry

end two_digit_number_sum_l3583_358314


namespace probability_theorem_l3583_358351

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def valid_pair (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧ 1 ≤ b ∧ b ≤ 20 ∧ a ≠ b ∧
  is_odd (a * b) ∧ (is_prime a ∨ is_prime b)

def total_pairs : ℕ := Nat.choose 20 2

def valid_pairs : ℕ := 42

theorem probability_theorem : 
  (valid_pairs : ℚ) / total_pairs = 21 / 95 :=
sorry

end probability_theorem_l3583_358351


namespace range_of_m_l3583_358378

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) →
  (∀ x : ℝ, (2 / (x + 1) > 1) → x ≤ 1) →
  m ≤ -1 :=
by sorry

end range_of_m_l3583_358378


namespace rohan_age_multiple_l3583_358320

def rohan_current_age : ℕ := 25

def rohan_past_age : ℕ := rohan_current_age - 15

def rohan_future_age : ℕ := rohan_current_age + 15

theorem rohan_age_multiple : 
  ∃ (x : ℚ), rohan_future_age = x * rohan_past_age ∧ x = 4 := by
sorry

end rohan_age_multiple_l3583_358320


namespace district_a_schools_l3583_358318

/-- Represents the three types of schools in Veenapaniville -/
inductive SchoolType
  | Public
  | Parochial
  | PrivateIndependent

/-- Represents the three districts in Veenapaniville -/
inductive District
  | A
  | B
  | C

/-- The total number of high schools in Veenapaniville -/
def totalSchools : Nat := 50

/-- The number of public schools in Veenapaniville -/
def publicSchools : Nat := 25

/-- The number of parochial schools in Veenapaniville -/
def parochialSchools : Nat := 16

/-- The number of private independent schools in Veenapaniville -/
def privateIndependentSchools : Nat := 9

/-- The number of high schools in District B -/
def districtBSchools : Nat := 17

/-- The number of private independent schools in District B -/
def districtBPrivateIndependentSchools : Nat := 2

/-- Function to calculate the number of schools in District C -/
def districtCSchools : Nat := 3 * (min publicSchools (min parochialSchools privateIndependentSchools))

/-- Theorem stating that the number of high schools in District A is 6 -/
theorem district_a_schools :
  totalSchools - (districtBSchools + districtCSchools) = 6 := by
  sorry


end district_a_schools_l3583_358318


namespace closest_integer_to_35_4_l3583_358335

theorem closest_integer_to_35_4 : ∀ n : ℤ, |n - (35 : ℚ) / 4| ≥ |9 - (35 : ℚ) / 4| := by
  sorry

end closest_integer_to_35_4_l3583_358335


namespace dislike_tv_and_books_l3583_358322

theorem dislike_tv_and_books (total_population : ℕ) 
  (tv_dislike_percentage : ℚ) (book_dislike_percentage : ℚ) :
  total_population = 800 →
  tv_dislike_percentage = 25 / 100 →
  book_dislike_percentage = 15 / 100 →
  (tv_dislike_percentage * total_population : ℚ) * book_dislike_percentage = 30 := by
sorry

end dislike_tv_and_books_l3583_358322


namespace log_relationship_l3583_358321

theorem log_relationship (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  6 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 12 * (Real.log x)^2 / (Real.log a * Real.log b) →
  (a = b^(5/3) ∨ a = b^(3/5)) :=
by sorry

end log_relationship_l3583_358321


namespace A_power_101_l3583_358327

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]]

theorem A_power_101 :
  A ^ 101 = ![![0, 1, 0],
              ![0, 0, 1],
              ![1, 0, 0]] := by
  sorry

end A_power_101_l3583_358327


namespace spending_difference_l3583_358332

/- Define the prices and quantities -/
def basketball_price : ℝ := 29
def basketball_quantity : ℕ := 10
def baseball_price : ℝ := 2.5
def baseball_quantity : ℕ := 14
def baseball_bat_price : ℝ := 18

/- Define the total spending for each coach -/
def coach_A_spending : ℝ := basketball_price * basketball_quantity
def coach_B_spending : ℝ := baseball_price * baseball_quantity + baseball_bat_price

/- Theorem statement -/
theorem spending_difference :
  coach_A_spending - coach_B_spending = 237 := by
  sorry

end spending_difference_l3583_358332


namespace roots_of_equation_correct_description_l3583_358325

theorem roots_of_equation (x : ℝ) : 
  (x^2 + 4) * (x^2 - 4) = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem correct_description : 
  ∀ x : ℝ, (x^2 + 4) * (x^2 - 4) = 0 → 
  ∃ y : ℝ, y = 2 ∨ y = -2 ∧ (y^2 + 4) * (y^2 - 4) = 0 :=
by sorry

end roots_of_equation_correct_description_l3583_358325


namespace infinitely_many_primes_6n_plus_5_l3583_358386

theorem infinitely_many_primes_6n_plus_5 :
  ∀ k : ℕ, ∃ p : ℕ, p > k ∧ Prime p ∧ ∃ n : ℕ, p = 6 * n + 5 := by
  sorry

end infinitely_many_primes_6n_plus_5_l3583_358386


namespace average_income_problem_l3583_358315

/-- Given the average incomes of different pairs of individuals and the income of one individual,
    prove that the average income of a specific pair is as stated. -/
theorem average_income_problem (M N O : ℕ) : 
  (M + N) / 2 = 5050 →
  (M + O) / 2 = 5200 →
  M = 4000 →
  (N + O) / 2 = 6250 := by
sorry

end average_income_problem_l3583_358315
