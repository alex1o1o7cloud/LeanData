import Mathlib

namespace fiona_received_59_l1858_185882

theorem fiona_received_59 (Dan_riddles : ℕ) (Andy_riddles : ℕ) (Bella_riddles : ℕ) (Emma_riddles : ℕ) (Fiona_riddles : ℕ)
  (h1 : Dan_riddles = 21)
  (h2 : Andy_riddles = Dan_riddles + 12)
  (h3 : Bella_riddles = Andy_riddles - 7)
  (h4 : Emma_riddles = Bella_riddles / 2)
  (h5 : Fiona_riddles = Andy_riddles + Bella_riddles) :
  Fiona_riddles = 59 :=
by
  sorry

end fiona_received_59_l1858_185882


namespace no_integer_b_for_four_integer_solutions_l1858_185815

theorem no_integer_b_for_four_integer_solutions :
  ∀ (b : ℤ), ¬ ∃ x1 x2 x3 x4 : ℤ, 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (∀ x : ℤ, (x^2 + b*x + 1 ≤ 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)) :=
by sorry

end no_integer_b_for_four_integer_solutions_l1858_185815


namespace math_problem_l1858_185847

theorem math_problem
  (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ)
  (h1 : x₁ + 4 * x₂ + 9 * x₃ + 16 * x₄ + 25 * x₅ + 36 * x₆ + 49 * x₇ = 1)
  (h2 : 4 * x₁ + 9 * x₂ + 16 * x₃ + 25 * x₄ + 36 * x₅ + 49 * x₆ + 64 * x₇ = 12)
  (h3 : 9 * x₁ + 16 * x₂ + 25 * x₃ + 36 * x₄ + 49 * x₅ + 64 * x₆ + 81 * x₇ = 123) :
  16 * x₁ + 25 * x₂ + 36 * x₃ + 49 * x₄ + 64 * x₅ + 81 * x₆ + 100 * x₇ = 334 := by
  sorry

end math_problem_l1858_185847


namespace overlap_area_of_parallelogram_l1858_185884

theorem overlap_area_of_parallelogram (w1 w2 : ℝ) (β : ℝ) (hβ : β = 30) (hw1 : w1 = 2) (hw2 : w2 = 1) : 
  (w1 * (w2 / Real.sin (β * Real.pi / 180))) = 4 :=
by
  sorry

end overlap_area_of_parallelogram_l1858_185884


namespace sampling_method_is_systematic_l1858_185833

-- Definition of the conditions
def factory_produces_product := True  -- Assuming the factory is always producing
def uses_conveyor_belt := True  -- Assuming the conveyor belt is always in use
def samples_taken_every_10_minutes := True  -- Sampling at specific intervals

-- Definition corresponding to the systematic sampling
def systematic_sampling := True

-- Theorem: Prove that given the conditions, the sampling method is systematic sampling.
theorem sampling_method_is_systematic :
  factory_produces_product → uses_conveyor_belt → samples_taken_every_10_minutes → systematic_sampling :=
by
  intros _ _ _
  trivial

end sampling_method_is_systematic_l1858_185833


namespace triangle_shape_l1858_185861

-- Define the sides of the triangle and the angles
variables {a b c : ℝ}
variables {A B C : ℝ} 
-- Assume that angles are in radians and 0 < A, B, C < π
-- Also assume that the sum of angles in the triangle is π
axiom angle_sum_triangle : A + B + C = Real.pi

-- Given condition
axiom given_condition : a^2 * Real.cos A * Real.sin B = b^2 * Real.sin A * Real.cos B

-- Conclusion: The shape of triangle ABC is either isosceles or right triangle
theorem triangle_shape : 
  (A = B) ∨ (A + B = (Real.pi / 2)) := 
by sorry

end triangle_shape_l1858_185861


namespace unattainable_y_l1858_185869

theorem unattainable_y (x : ℝ) (h : x ≠ -4 / 3) :
  ¬ ∃ y : ℝ, y = (2 - x) / (3 * x + 4) ∧ y = -1 / 3 :=
by
  sorry

end unattainable_y_l1858_185869


namespace fraction_of_males_l1858_185877

theorem fraction_of_males (M F : ℚ) (h1 : M + F = 1)
  (h2 : (3/4) * M + (5/6) * F = 7/9) :
  M = 2/3 :=
by sorry

end fraction_of_males_l1858_185877


namespace find_m_value_l1858_185851

-- Definitions from conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (2, -4)
def OA := (A.1 - O.1, A.2 - O.2)
def AB := (B.1 - A.1, B.2 - A.2)

-- Defining the vector OP with the given expression
def OP (m : ℝ) := (2 * OA.1 + m * AB.1, 2 * OA.2 + m * AB.2)

-- The point P is on the y-axis if the x-coordinate of OP is zero
theorem find_m_value : ∃ m : ℝ, OP m = (0, (OP m).2) ∧ m = 2 / 3 :=
by { 
  -- sorry is added to skip the proof itself
  sorry 
}

end find_m_value_l1858_185851


namespace triangle_area_is_4_l1858_185808

variable {PQ RS : ℝ} -- lengths of PQ and RS respectively
variable {area_PQRS area_PQS : ℝ} -- areas of the trapezoid and triangle respectively

-- Given conditions
@[simp]
def trapezoid_area_is_12 (area_PQRS : ℝ) : Prop :=
  area_PQRS = 12

@[simp]
def RS_is_twice_PQ (PQ RS : ℝ) : Prop :=
  RS = 2 * PQ

-- To prove: the area of triangle PQS is 4 given the conditions
theorem triangle_area_is_4 (h1 : trapezoid_area_is_12 area_PQRS)
                          (h2 : RS_is_twice_PQ PQ RS)
                          (h3 : area_PQRS = 3 * area_PQS) : area_PQS = 4 :=
by
  sorry

end triangle_area_is_4_l1858_185808


namespace find_theta_l1858_185839

variable (x : ℝ) (θ : ℝ) (k : ℤ)

def condition := (3 - 3^(-|x - 3|))^2 = 3 - Real.cos θ

theorem find_theta (h : condition x θ) : ∃ k : ℤ, θ = (2 * k + 1) * Real.pi :=
by
  sorry

end find_theta_l1858_185839


namespace profit_function_correct_l1858_185855

-- Definitions based on Conditions
def selling_price {R : Type*} [LinearOrderedField R] : R := 45
def profit_max {R : Type*} [LinearOrderedField R] : R := 450
def price_no_sales {R : Type*} [LinearOrderedField R] : R := 60
def quadratic_profit {R : Type*} [LinearOrderedField R] (x : R) : R := -2 * (x - 30) * (x - 60)

-- The statement we need to prove.
theorem profit_function_correct {R : Type*} [LinearOrderedField R] :
  quadratic_profit (selling_price : R) = profit_max ∧ quadratic_profit (price_no_sales : R) = 0 := 
sorry

end profit_function_correct_l1858_185855


namespace binomial_identity_l1858_185862

theorem binomial_identity (n k : ℕ) (h1 : 0 < k) (h2 : k < n)
    (h3 : Nat.choose n (k-1) + Nat.choose n (k+1) = 2 * Nat.choose n k) :
  ∃ c : ℤ, k = (c^2 + c - 2) / 2 ∧ n = c^2 - 2 := sorry

end binomial_identity_l1858_185862


namespace grown_ups_in_milburg_l1858_185870

def total_population : ℕ := 8243
def number_of_children : ℕ := 2987

theorem grown_ups_in_milburg : total_population - number_of_children = 5256 :=
by {
  sorry
}

end grown_ups_in_milburg_l1858_185870


namespace randy_initial_amount_l1858_185889

theorem randy_initial_amount (spend_per_trip: ℤ) (trips_per_month: ℤ) (dollars_left_after_year: ℤ) (total_month_months: ℤ := 12):
  (spend_per_trip = 2 ∧ trips_per_month = 4 ∧ dollars_left_after_year = 104) → spend_per_trip * trips_per_month * total_month_months + dollars_left_after_year = 200 := 
by
  sorry

end randy_initial_amount_l1858_185889


namespace total_cost_is_63_l1858_185805

-- Define the original price, markdown percentage, and sales tax percentage
def original_price : ℝ := 120
def markdown_percentage : ℝ := 0.50
def sales_tax_percentage : ℝ := 0.05

-- Calculate the reduced price
def reduced_price : ℝ := original_price * (1 - markdown_percentage)

-- Calculate the sales tax on the reduced price
def sales_tax : ℝ := reduced_price * sales_tax_percentage

-- Calculate the total cost
noncomputable def total_cost : ℝ := reduced_price + sales_tax

-- Theorem stating that the total cost of the aquarium is $63
theorem total_cost_is_63 : total_cost = 63 := by
  sorry

end total_cost_is_63_l1858_185805


namespace starling_nests_flying_condition_l1858_185875

theorem starling_nests_flying_condition (n : ℕ) (h1 : n ≥ 3)
  (h2 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → A ≠ B)
  (h3 : ∀ (A B : Finset ℕ), A.card = n → B.card = n → 
  (∃ d1 d2 : ℝ, d1 < d2 ∧ d1 < d2 → d1 > d2)) : n = 3 :=
by
  sorry

end starling_nests_flying_condition_l1858_185875


namespace possible_values_for_n_l1858_185853

theorem possible_values_for_n (n : ℕ) (h1 : ∀ a b c : ℤ, (a = n-1) ∧ (b = n) ∧ (c = n+1) → 
    (∃ f g : ℤ, f = 2*a - b ∧ g = 2*b - a)) 
    (h2 : ∃ a b c : ℤ, (a = 0 ∨ b = 0 ∨ c = 0) ∧ (a + b + c = 0)) : 
    ∃ k : ℕ, n = 3^k := 
sorry

end possible_values_for_n_l1858_185853


namespace spring_length_5kg_weight_l1858_185814

variable {x y : ℝ}

-- Given conditions
def spring_length_no_weight : y = 6 := sorry
def spring_length_4kg_weight : y = 7.2 := sorry

-- The problem: to find the length of the spring for 5 kilograms
theorem spring_length_5kg_weight :
  (∃ (k b : ℝ), (∀ x, y = k * x + b) ∧ (b = 6) ∧ (4 * k + b = 7.2)) →
  y = 0.3 * 5 + 6 :=
  sorry

end spring_length_5kg_weight_l1858_185814


namespace probability_one_solve_l1858_185893

variables {p1 p2 : ℝ}

theorem probability_one_solve (h1 : 0 ≤ p1 ∧ p1 ≤ 1) (h2 : 0 ≤ p2 ∧ p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) := 
sorry

end probability_one_solve_l1858_185893


namespace quadratic_solution_difference_l1858_185846

theorem quadratic_solution_difference (x : ℝ) :
  ∀ x : ℝ, (x^2 - 5*x + 15 = x + 55) → (∃ a b : ℝ, a ≠ b ∧ x^2 - 6*x - 40 = 0 ∧ abs (a - b) = 14) :=
by
  sorry

end quadratic_solution_difference_l1858_185846


namespace cos_A_of_triangle_l1858_185885

theorem cos_A_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : b = Real.sqrt 2 * c)
  (h2 : Real.sin A + Real.sqrt 2 * Real.sin C = 2 * Real.sin B)
  (h3 : a = Real.sin A / Real.sin A * b) -- Sine rule used implicitly

: Real.cos A = Real.sqrt 2 / 4 := by
  -- proof will be skipped, hence 'sorry' included
  sorry

end cos_A_of_triangle_l1858_185885


namespace angle_in_fourth_quadrant_l1858_185859

-- Define the main condition converting the angle to the range [0, 360)
def reducedAngle (θ : ℤ) : ℤ := (θ % 360 + 360) % 360

-- State the theorem proving the angle of -390° is in the fourth quadrant
theorem angle_in_fourth_quadrant (θ : ℤ) (h : θ = -390) : 270 ≤ reducedAngle θ ∧ reducedAngle θ < 360 := by
  sorry

end angle_in_fourth_quadrant_l1858_185859


namespace episodes_per_wednesday_l1858_185865

theorem episodes_per_wednesday :
  ∀ (W : ℕ), (∃ (n_episodes : ℕ) (n_mondays : ℕ) (n_weeks : ℕ), 
    n_episodes = 201 ∧ n_mondays = 67 ∧ n_weeks = 67 
    ∧ n_weeks * W + n_mondays = n_episodes) 
    → W = 2 :=
by
  intro W
  rintro ⟨n_episodes, n_mondays, n_weeks, h1, h2, h3, h4⟩
  -- proof would go here
  sorry

end episodes_per_wednesday_l1858_185865


namespace greatest_q_minus_r_l1858_185871

theorem greatest_q_minus_r : 
  ∃ (q r : ℕ), 1013 = 23 * q + r ∧ q > 0 ∧ r > 0 ∧ (q - r = 39) := 
by
  sorry

end greatest_q_minus_r_l1858_185871


namespace possible_values_for_p_l1858_185888

-- Definitions for the conditions
variables {a b c p : ℝ}

-- Assumptions
def distinct (a b c : ℝ) := ¬(a = b) ∧ ¬(b = c) ∧ ¬(c = a)
def main_eq (a b c p : ℝ) := a + (1 / b) = p ∧ b + (1 / c) = p ∧ c + (1 / a) = p

-- Theorem statement
theorem possible_values_for_p (h1 : distinct a b c) (h2 : main_eq a b c p) : p = 1 ∨ p = -1 := 
sorry

end possible_values_for_p_l1858_185888


namespace inscribed_circle_radius_range_l1858_185800

noncomputable def r_range (AD DB : ℝ) (angle_A : ℝ) : Set ℝ :=
  { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 }

theorem inscribed_circle_radius_range (AD DB : ℝ) (angle_A : ℝ) (h1 : AD = 2 * Real.sqrt 3) 
    (h2 : DB = Real.sqrt 3) (h3 : angle_A > 60) : 
    r_range AD DB angle_A = { r | 0 < r ∧ r < (-3 + Real.sqrt 33) / 2 } :=
by
  sorry

end inscribed_circle_radius_range_l1858_185800


namespace sum_of_digits_of_N_l1858_185873

-- The total number of coins
def total_coins : ℕ := 3081

-- Setting up the equation N^2 = 3081
def N : ℕ := 55 -- Since 55^2 is closest to 3081 and sqrt(3081) ≈ 55

-- Proving the sum of the digits of N is 10
theorem sum_of_digits_of_N : (5 + 5) = 10 :=
by
  sorry

end sum_of_digits_of_N_l1858_185873


namespace bucket_capacity_l1858_185822

theorem bucket_capacity (jack_buckets_per_trip : ℕ)
                        (jill_buckets_per_trip : ℕ)
                        (jack_trip_ratio : ℝ)
                        (jill_trips : ℕ)
                        (tank_capacity : ℝ)
                        (bucket_capacity : ℝ)
                        (h1 : jack_buckets_per_trip = 2)
                        (h2 : jill_buckets_per_trip = 1)
                        (h3 : jack_trip_ratio = 3 / 2)
                        (h4 : jill_trips = 30)
                        (h5 : tank_capacity = 600) :
  bucket_capacity = 5 :=
by 
  sorry

end bucket_capacity_l1858_185822


namespace simplify_inverse_sum_l1858_185849

variable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem simplify_inverse_sum :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) :=
by sorry

end simplify_inverse_sum_l1858_185849


namespace earnings_per_weed_is_six_l1858_185821

def flower_bed_weeds : ℕ := 11
def vegetable_patch_weeds : ℕ := 14
def grass_weeds : ℕ := 32
def grass_weeds_half : ℕ := grass_weeds / 2
def soda_cost : ℕ := 99
def money_left : ℕ := 147
def total_weeds : ℕ := flower_bed_weeds + vegetable_patch_weeds + grass_weeds_half
def total_money : ℕ := money_left + soda_cost

theorem earnings_per_weed_is_six :
  total_money / total_weeds = 6 :=
by
  sorry

end earnings_per_weed_is_six_l1858_185821


namespace Ayla_call_duration_l1858_185811

theorem Ayla_call_duration
  (charge_per_minute : ℝ)
  (monthly_bill : ℝ)
  (customers_per_week : ℕ)
  (weeks_in_month : ℕ)
  (calls_duration : ℝ)
  (h_charge : charge_per_minute = 0.05)
  (h_bill : monthly_bill = 600)
  (h_customers : customers_per_week = 50)
  (h_weeks_in_month : weeks_in_month = 4)
  (h_calls_duration : calls_duration = (monthly_bill / charge_per_minute) / (customers_per_week * weeks_in_month)) :
  calls_duration = 60 :=
by 
  sorry

end Ayla_call_duration_l1858_185811


namespace domain_of_f_l1858_185858

noncomputable def f (x : ℝ) : ℝ := (Real.log (2 * x - 1)) / Real.sqrt (x + 1)

theorem domain_of_f :
  {x : ℝ | 2 * x - 1 > 0 ∧ x + 1 ≥ 0} = {x : ℝ | x > 1/2} :=
by
  sorry

end domain_of_f_l1858_185858


namespace quintuple_sum_not_less_than_l1858_185841

theorem quintuple_sum_not_less_than (a : ℝ) : 5 * (a + 3) ≥ 6 :=
by
  -- Insert proof here
  sorry

end quintuple_sum_not_less_than_l1858_185841


namespace find_abc_l1858_185887

noncomputable def log (x : ℝ) : ℝ := sorry -- Replace sorry with an actual implementation of log function if needed

theorem find_abc (a b c : ℝ) 
    (h1 : 1 ≤ a) 
    (h2 : 1 ≤ b) 
    (h3 : 1 ≤ c)
    (h4 : a * b * c = 10)
    (h5 : a^(log a) * b^(log b) * c^(log c) ≥ 10) :
    (a = 1 ∧ b = 10 ∧ c = 1) ∨ (a = 10 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 10) := 
by
  sorry

end find_abc_l1858_185887


namespace grade_more_problems_l1858_185895

theorem grade_more_problems (worksheets_total problems_per_worksheet worksheets_graded: ℕ)
  (h1 : worksheets_total = 9)
  (h2 : problems_per_worksheet = 4)
  (h3 : worksheets_graded = 5):
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 16 :=
by
  sorry

end grade_more_problems_l1858_185895


namespace missing_condition_l1858_185860

theorem missing_condition (x y : ℕ) (h1 : y = 2 * x + 9) (h2 : y = 3 * (x - 2)) :
  true := -- The equivalent mathematical statement asserts the correct missing condition.
sorry

end missing_condition_l1858_185860


namespace exponential_monotone_l1858_185819

theorem exponential_monotone {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a < b :=
sorry

end exponential_monotone_l1858_185819


namespace angles_bisectors_l1858_185856

theorem angles_bisectors (k : ℤ) : 
    ∃ α : ℤ, α = k * 180 + 135 
  -> 
    (α = (2 * k) * 180 + 135 ∨ α = (2 * k + 1) * 180 + 135) 
  := sorry

end angles_bisectors_l1858_185856


namespace factor_difference_of_squares_196_l1858_185824

theorem factor_difference_of_squares_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end factor_difference_of_squares_196_l1858_185824


namespace arnold_total_protein_l1858_185848

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein_l1858_185848


namespace factorize_expression1_factorize_expression2_l1858_185872

variable {R : Type*} [CommRing R]

theorem factorize_expression1 (x y : R) : x^2 + 2 * x + 1 - y^2 = (x + y + 1) * (x - y + 1) :=
  sorry

theorem factorize_expression2 (m n p : R) : m^2 - n^2 - 2 * n * p - p^2 = (m + n + p) * (m - n - p) :=
  sorry

end factorize_expression1_factorize_expression2_l1858_185872


namespace fraction_to_decimal_l1858_185854

theorem fraction_to_decimal : (7 / 16 : ℚ) = 0.4375 :=
by
  sorry

end fraction_to_decimal_l1858_185854


namespace mandy_difference_of_cinnamon_and_nutmeg_l1858_185892

theorem mandy_difference_of_cinnamon_and_nutmeg :
  let cinnamon := 0.6666666666666666
  let nutmeg := 0.5
  let difference := cinnamon - nutmeg
  difference = 0.1666666666666666 :=
by
  sorry

end mandy_difference_of_cinnamon_and_nutmeg_l1858_185892


namespace find_f_13_l1858_185801

def periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f (x + p) = f x

theorem find_f_13 (f : ℝ → ℝ) 
  (h_period : periodic f 1.5) 
  (h_val : f 1 = 20) 
  : f 13 = 20 :=
by
  sorry

end find_f_13_l1858_185801


namespace Jordan_length_is_8_l1858_185890

-- Definitions of the conditions given in the problem
def Carol_length := 5
def Carol_width := 24
def Jordan_width := 15

-- Definition to calculate the area of Carol's rectangle
def Carol_area : ℕ := Carol_length * Carol_width

-- Definition to calculate the length of Jordan's rectangle
def Jordan_length (area : ℕ) (width : ℕ) : ℕ := area / width

-- Proposition to prove the length of Jordan's rectangle
theorem Jordan_length_is_8 : Jordan_length Carol_area Jordan_width = 8 :=
by
  -- skipping the proof
  sorry

end Jordan_length_is_8_l1858_185890


namespace part1_part2_l1858_185899

def A (x : ℝ) : Prop := x ^ 2 - 2 * x - 8 < 0
def B (x : ℝ) : Prop := x ^ 2 + 2 * x - 3 > 0
def C (a : ℝ) (x : ℝ) : Prop := x ^ 2 - 3 * a * x + 2 * a ^ 2 < 0

theorem part1 : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x < 4} := 
by sorry

theorem part2 (a : ℝ) : {x : ℝ | C a x} ⊆ {x : ℝ | A x} ∩ {x : ℝ | B x} ↔ (a = 0 ∨ (1 ≤ a ∧ a ≤ 2)) := 
by sorry

end part1_part2_l1858_185899


namespace graph_transformation_l1858_185842

theorem graph_transformation (a b c : ℝ) (h1 : c = 1) (h2 : a + b + c = -2) (h3 : a - b + c = 2) :
  (∀ x, cx^2 + 2 * bx + a = (x - 2)^2 - 5) := 
sorry

end graph_transformation_l1858_185842


namespace first_step_of_testing_circuit_broken_l1858_185809

-- Definitions based on the problem
def circuit_broken : Prop := true
def binary_search_method : Prop := true
def test_first_step_at_midpoint : Prop := true

-- The theorem stating the first step in testing a broken circuit using the binary search method
theorem first_step_of_testing_circuit_broken (h1 : circuit_broken) (h2 : binary_search_method) :
  test_first_step_at_midpoint :=
sorry

end first_step_of_testing_circuit_broken_l1858_185809


namespace new_student_weight_l1858_185820

theorem new_student_weight :
  let avg_weight_29 := 28
  let num_students_29 := 29
  let avg_weight_30 := 27.4
  let num_students_30 := 30
  let total_weight_29 := avg_weight_29 * num_students_29
  let total_weight_30 := avg_weight_30 * num_students_30
  let new_student_weight := total_weight_30 - total_weight_29
  new_student_weight = 10 :=
by
  sorry

end new_student_weight_l1858_185820


namespace time_to_eat_potatoes_l1858_185844

theorem time_to_eat_potatoes (rate : ℕ → ℕ → ℝ) (potatoes : ℕ → ℕ → ℝ) 
    (minutes : ℕ) (hours : ℝ) (total_potatoes : ℕ) : 
    rate 3 20 = 9 / 1 -> potatoes 27 9 = 3 := 
by
  intro h1
  -- You can add intermediate steps here as optional comments for clarity during proof construction
  /- 
  Given: 
  rate 3 20 = 9 -> Jason's rate of eating potatoes is 9 potatoes per hour
  time = potatoes / rate -> 27 potatoes / 9 potatoes/hour = 3 hours
  -/
  sorry

end time_to_eat_potatoes_l1858_185844


namespace arithmetic_sequence_problem_l1858_185818

variable {a : ℕ → ℝ} {d : ℝ} -- Declare the sequence and common difference

-- Define the arithmetic sequence property
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  a 5 + a 10 = 12 ∧ arithmetic_sequence a d

-- Main theorem statement
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h : given_conditions a d) :
  3 * a 7 + a 9 = 24 :=
sorry

end arithmetic_sequence_problem_l1858_185818


namespace days_until_see_grandma_l1858_185868

def hours_in_a_day : ℕ := 24
def hours_until_see_grandma : ℕ := 48

theorem days_until_see_grandma : hours_until_see_grandma / hours_in_a_day = 2 := by
  sorry

end days_until_see_grandma_l1858_185868


namespace sales_fifth_month_l1858_185837

theorem sales_fifth_month (s1 s2 s3 s4 s6 s5 : ℝ) (target_avg total_sales : ℝ)
  (h1 : s1 = 4000)
  (h2 : s2 = 6524)
  (h3 : s3 = 5689)
  (h4 : s4 = 7230)
  (h6 : s6 = 12557)
  (h_avg : target_avg = 7000)
  (h_total_sales : total_sales = 42000) :
  s5 = 6000 :=
by
  sorry

end sales_fifth_month_l1858_185837


namespace short_haired_girls_l1858_185874

def total_people : ℕ := 55
def boys : ℕ := 30
def total_girls : ℕ := total_people - boys
def girls_with_long_hair : ℕ := (3 / 5) * total_girls
def girls_with_short_hair : ℕ := total_girls - girls_with_long_hair

theorem short_haired_girls :
  girls_with_short_hair = 10 := sorry

end short_haired_girls_l1858_185874


namespace smallest_circle_radius_polygonal_chain_l1858_185879

theorem smallest_circle_radius_polygonal_chain (l : ℝ) (hl : l = 1) : ∃ (r : ℝ), r = 0.5 := 
sorry

end smallest_circle_radius_polygonal_chain_l1858_185879


namespace geometric_sequence_26th_term_l1858_185896

noncomputable def r : ℝ := (8 : ℝ)^(1/6)

noncomputable def a (n : ℕ) (a₁ : ℝ) (r : ℝ) : ℝ := a₁ * r^(n - 1)

theorem geometric_sequence_26th_term :
  (a 26 (a 14 10 r) r = 640) :=
by
  have h₁ : a 14 10 r = 10 := sorry
  have h₂ : r^6 = 8 := sorry
  sorry

end geometric_sequence_26th_term_l1858_185896


namespace box_area_relation_l1858_185804

theorem box_area_relation (a b c : ℕ) (h : a = b + c + 10) :
  (a * b) * (b * c) * (c * a) = (2 * (b + c) + 10)^2 := 
sorry

end box_area_relation_l1858_185804


namespace distinct_solutions_sub_l1858_185826

open Nat Real

theorem distinct_solutions_sub (p q : Real) (hpq_distinct : p ≠ q) (h_eqn_p : (p - 4) * (p + 4) = 17 * p - 68) (h_eqn_q : (q - 4) * (q + 4) = 17 * q - 68) (h_p_gt_q : p > q) : p - q = 9 := 
sorry

end distinct_solutions_sub_l1858_185826


namespace total_squares_l1858_185876

theorem total_squares (num_groups : ℕ) (squares_per_group : ℕ) (total : ℕ) 
  (h1 : num_groups = 5) (h2 : squares_per_group = 5) (h3 : total = num_groups * squares_per_group) : 
  total = 25 :=
by
  rw [h1, h2] at h3
  exact h3

end total_squares_l1858_185876


namespace coeff_of_nxy_n_l1858_185829

theorem coeff_of_nxy_n {n : ℕ} (degree_eq : 1 + n = 10) : n = 9 :=
by
  sorry

end coeff_of_nxy_n_l1858_185829


namespace students_in_second_class_l1858_185898

-- Definitions based on the conditions
def students_first_class : ℕ := 30
def avg_mark_first_class : ℕ := 40
def avg_mark_second_class : ℕ := 80
def combined_avg_mark : ℕ := 65

-- Proposition to prove
theorem students_in_second_class (x : ℕ) 
  (h1 : students_first_class * avg_mark_first_class + x * avg_mark_second_class = (students_first_class + x) * combined_avg_mark) : 
  x = 50 :=
sorry

end students_in_second_class_l1858_185898


namespace value_of_a_if_lines_are_parallel_l1858_185812

theorem value_of_a_if_lines_are_parallel (a : ℝ) :
  (∀ (x y : ℝ), x + a*y - 7 = 0 → (a+1)*x + 2*y - 14 = 0) → a = -2 :=
sorry

end value_of_a_if_lines_are_parallel_l1858_185812


namespace sequence_bounded_l1858_185810

theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n)
  (h_dep : ∀ k n m l, k + n = m + l → (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ m M : ℝ, ∀ n, m ≤ a n ∧ a n ≤ M :=
sorry

end sequence_bounded_l1858_185810


namespace find_ax5_plus_by5_l1858_185840

theorem find_ax5_plus_by5 (a b x y : ℝ)
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := 
sorry

end find_ax5_plus_by5_l1858_185840


namespace Mike_and_Sarah_missed_days_l1858_185863

theorem Mike_and_Sarah_missed_days :
  ∀ (V M S : ℕ), V + M + S = 17 → V + M = 14 → V = 5 → M + S = 12 :=
by
  intros V M S h1 h2 h3
  sorry

end Mike_and_Sarah_missed_days_l1858_185863


namespace min_sum_of_box_dimensions_l1858_185883

theorem min_sum_of_box_dimensions :
  ∃ (x y z : ℕ), x * y * z = 2541 ∧ (y = x + 3 ∨ x = y + 3) ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 38 :=
sorry

end min_sum_of_box_dimensions_l1858_185883


namespace simplify_expression_l1858_185831

theorem simplify_expression (x y : ℝ) : ((3 * x + 22) + (150 * y + 22)) = (3 * x + 150 * y + 44) :=
by
  sorry

end simplify_expression_l1858_185831


namespace evaluate_expression_l1858_185825

theorem evaluate_expression : (1 / (2 + (1 / (3 + (1 / 4))))) = (13 / 30) :=
by
  sorry

end evaluate_expression_l1858_185825


namespace enemies_left_undefeated_l1858_185830

theorem enemies_left_undefeated (points_per_enemy points_earned total_enemies : ℕ) 
  (h1 : points_per_enemy = 3)
  (h2 : total_enemies = 6)
  (h3 : points_earned = 12) : 
  (total_enemies - points_earned / points_per_enemy) = 2 :=
by
  sorry

end enemies_left_undefeated_l1858_185830


namespace complement_of_A_in_U_l1858_185828

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U : U \ A = {2, 4} := 
by
  sorry

end complement_of_A_in_U_l1858_185828


namespace proof_problem_l1858_185864

-- Definitions for the conditions and the events in the problem
def P_A : ℚ := 2 / 3
def P_B : ℚ := 1 / 4
def P_not_any_module : ℚ := 1 - (P_A + P_B)

-- Definition for the binomial coefficient
def C (n k : ℕ) := Nat.choose n k

-- Definition for the event where at least 3 out of 4 students have taken "Selected Topics in Geometric Proofs"
def P_at_least_three_taken : ℚ := 
  C 4 3 * (P_A ^ 3) * ((1 - P_A) ^ 1) + C 4 4 * (P_A ^ 4)

-- The main theorem to prove
theorem proof_problem : 
  P_not_any_module = 1 / 12 ∧ P_at_least_three_taken = 16 / 27 :=
by
  sorry

end proof_problem_l1858_185864


namespace kite_minimum_area_correct_l1858_185843

noncomputable def minimumKiteAreaAndSum (r : ℕ) (OP : ℕ) (h₁ : r = 60) (h₂ : OP < r) : ℕ × ℝ :=
  let d₁ := 2 * r
  let d₂ := 2 * Real.sqrt (r^2 - OP^2)
  let area := (d₁ * d₂) / 2
  (120 + 119, area)

theorem kite_minimum_area_correct {r OP : ℕ} (h₁ : r = 60) (h₂ : OP < r) :
  minimumKiteAreaAndSum r OP h₁ h₂ = (239, 120 * Real.sqrt 119) :=
by simp [minimumKiteAreaAndSum, h₁, h₂] ; sorry

end kite_minimum_area_correct_l1858_185843


namespace total_barking_dogs_eq_l1858_185886

-- Definitions
def initial_barking_dogs : ℕ := 30
def additional_barking_dogs : ℕ := 10

-- Theorem to prove the total number of barking dogs
theorem total_barking_dogs_eq :
  initial_barking_dogs + additional_barking_dogs = 40 :=
by
  sorry

end total_barking_dogs_eq_l1858_185886


namespace material_needed_for_second_type_l1858_185857

namespace CherylProject

def first_material := 5 / 9
def leftover_material := 1 / 3
def total_material_used := 5 / 9

theorem material_needed_for_second_type :
  0.8888888888888889 - (5 / 9 : ℝ) = 0.3333333333333333 := by
  sorry

end CherylProject

end material_needed_for_second_type_l1858_185857


namespace proof1_proof2_l1858_185834

noncomputable def a (n : ℕ) : ℝ := (n^2 + 1) * 3^n

def recurrence_relation : Prop :=
  ∀ n, a (n + 3) - 9 * a (n + 2) + 27 * a (n + 1) - 27 * a n = 0

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n, a n * x^n

def series_evaluation (x : ℝ) : Prop :=
  series_sum x = (1 - 3*x + 18*x^2) / (1 - 3*x)^3

theorem proof1 : recurrence_relation := 
  by sorry

theorem proof2 : ∀ x : ℝ, series_evaluation x := 
  by sorry

end proof1_proof2_l1858_185834


namespace solve_for_y_l1858_185802

theorem solve_for_y : ∃ y : ℝ, (5 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + y^(1/3)) ↔ y = 1000 := by
  sorry

end solve_for_y_l1858_185802


namespace vendor_profit_l1858_185835

theorem vendor_profit {s₁ s₂ c₁ c₂ : ℝ} (h₁ : s₁ = 80) (h₂ : s₂ = 80) (profit₁ : s₁ = c₁ * 1.60) (loss₂ : s₂ = c₂ * 0.80) 
: (s₁ + s₂) - (c₁ + c₂) = 10 := by 
  sorry

end vendor_profit_l1858_185835


namespace statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l1858_185823

-- Definitions of conditions
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℝ)
  (angles : Fin n → ℝ)

def circumscribed (P : Polygon n) : Prop := sorry -- Definition of circumscribed
def inscribed (P : Polygon n) : Prop := sorry -- Definition of inscribed
def equal_sides (P : Polygon n) : Prop := ∀ i j, P.sides i = P.sides j
def equal_angles (P : Polygon n) : Prop := ∀ i j, P.angles i = P.angles j

-- The statements to be proved
theorem statement_I : ∀ P : Polygon n, circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_II : ∃ P : Polygon n, inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_III : ∃ P : Polygon n, circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_IV : ∀ P : Polygon n, inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_V : ∀ (P : Polygon 5), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VI : ∀ (P : Polygon 6), circumscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VII : ∀ (P : Polygon 5), inscribed P → equal_sides P → equal_angles P := sorry

theorem statement_VIII : ∃ (P : Polygon 6), inscribed P ∧ equal_sides P ∧ ¬equal_angles P := sorry

theorem statement_IX : ∀ (P : Polygon 5), circumscribed P → equal_angles P → equal_sides P := sorry

theorem statement_X : ∃ (P : Polygon 6), circumscribed P ∧ equal_angles P ∧ ¬equal_sides P := sorry

theorem statement_XI : ∀ (P : Polygon 5), inscribed P → equal_angles P → equal_sides P := sorry

theorem statement_XII : ∀ (P : Polygon 6), inscribed P → equal_angles P → equal_sides P := sorry

end statement_I_statement_II_statement_III_statement_IV_statement_V_statement_VI_statement_VII_statement_VIII_statement_IX_statement_X_statement_XI_statement_XII_l1858_185823


namespace average_ABC_is_3_l1858_185878

theorem average_ABC_is_3
  (A B C : ℝ)
  (h1 : 2003 * C - 4004 * A = 8008)
  (h2 : 2003 * B + 6006 * A = 10010)
  (h3 : B = 2 * A - 6) : 
  (A + B + C) / 3 = 3 :=
sorry

end average_ABC_is_3_l1858_185878


namespace john_order_cost_l1858_185881

-- Definitions from the problem conditions
def discount_rate : ℝ := 0.10
def item_price : ℝ := 200
def num_items : ℕ := 7
def discount_threshold : ℝ := 1000

-- Final proof statement
theorem john_order_cost : 
  (num_items * item_price) - 
  (if (num_items * item_price) > discount_threshold then 
    discount_rate * ((num_items * item_price) - discount_threshold) 
  else 0) = 1360 := 
sorry

end john_order_cost_l1858_185881


namespace max_hedgehogs_l1858_185867

theorem max_hedgehogs (S : ℕ) (n : ℕ) (hS : S = 65) (hn : ∀ m, m > n → (m * (m + 1)) / 2 > S) :
  n = 10 := 
sorry

end max_hedgehogs_l1858_185867


namespace inequality_cannot_hold_l1858_185894

theorem inequality_cannot_hold (a b : ℝ) (ha : a < b) (hb : b < 0) : a^3 ≤ b^3 :=
by
  sorry

end inequality_cannot_hold_l1858_185894


namespace complex_computation_l1858_185832

theorem complex_computation : 3 * 3^4 - 27^63 / 27^61 = -486 := by
  sorry

end complex_computation_l1858_185832


namespace sum_of_interior_angles_heptagon_l1858_185806

theorem sum_of_interior_angles_heptagon (n : ℕ) (h : n = 7) : (n - 2) * 180 = 900 := by
  sorry

end sum_of_interior_angles_heptagon_l1858_185806


namespace D_neither_sufficient_nor_necessary_for_A_l1858_185880

theorem D_neither_sufficient_nor_necessary_for_A 
  (A B C D : Prop) 
  (h1 : A → B) 
  (h2 : ¬(B → A)) 
  (h3 : B ↔ C) 
  (h4 : C → D) 
  (h5 : ¬(D → C)) 
  :
  ¬(D → A) ∧ ¬(A → D) :=
by 
  sorry

end D_neither_sufficient_nor_necessary_for_A_l1858_185880


namespace least_value_of_p_plus_q_l1858_185813

theorem least_value_of_p_plus_q (p q : ℕ) (hp : 1 < p) (hq : 1 < q) (h : 17 * (p + 1) = 28 * (q + 1)) : p + q = 135 :=
  sorry

end least_value_of_p_plus_q_l1858_185813


namespace binomial_30_3_l1858_185807

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l1858_185807


namespace f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l1858_185817

noncomputable def f : ℕ → ℕ → ℕ
| 0, y => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_2_2_eq_7 : f 2 2 = 7 :=
sorry

theorem f_3_3_eq_61 : f 3 3 = 61 :=
sorry

theorem f_4_4_can_be_evaluated : ∃ n, f 4 4 = n :=
sorry

end f_2_2_eq_7_f_3_3_eq_61_f_4_4_can_be_evaluated_l1858_185817


namespace arithmetic_sequence_terms_sum_l1858_185852

variable (a_n : ℕ → ℕ)
variable (S_n : ℕ → ℕ)

-- Definitions based on given problem conditions
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) := 
  ∀ n, S n = n * (a 1 + a n) / 2

axiom Sn_2017 : S_n 2017 = 4034

-- Goal: a_3 + a_1009 + a_2015 = 6
theorem arithmetic_sequence_terms_sum :
  arithmetic_sequence a_n →
  sum_first_n_terms S_n a_n →
  S_n 2017 = 4034 → 
  a_n 3 + a_n 1009 + a_n 2015 = 6 :=
by
  intros
  sorry

end arithmetic_sequence_terms_sum_l1858_185852


namespace square_side_length_equals_5_sqrt_pi_l1858_185845

theorem square_side_length_equals_5_sqrt_pi :
  ∃ s : ℝ, ∃ r : ℝ, (r = 5) ∧ (s = 2 * r) ∧ (s ^ 2 = 25 * π) ∧ (s = 5 * Real.sqrt π) :=
by
  sorry

end square_side_length_equals_5_sqrt_pi_l1858_185845


namespace expression_divisibility_l1858_185816

theorem expression_divisibility (x y : ℝ) : 
  ∃ P : ℝ, (x^2 - x * y + y^2)^3 + (x^2 + x * y + y^2)^3 = (2 * x^2 + 2 * y^2) * P := 
by 
  sorry

end expression_divisibility_l1858_185816


namespace value_of_composition_l1858_185897

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x - 1

theorem value_of_composition : g (f (1 + 2 * g 3)) = 12 := by
  sorry

end value_of_composition_l1858_185897


namespace symmetric_points_on_parabola_l1858_185850

theorem symmetric_points_on_parabola {a b m n : ℝ}
  (hA : m = a^2 - 2*a - 2)
  (hB : m = b^2 - 2*b - 2)
  (hP : n = (a + b)^2 - 2*(a + b) - 2)
  (h_symmetry : (a + b) / 2 = 1) :
  n = -2 :=
by {
  -- Proof omitted
  sorry
}

end symmetric_points_on_parabola_l1858_185850


namespace difference_of_digits_l1858_185866

theorem difference_of_digits (A B : ℕ) (h1 : 6 * 10 + A - (B * 10 + 2) = 36) (h2 : A ≠ B) : A - B = 5 :=
sorry

end difference_of_digits_l1858_185866


namespace total_value_after_3_years_l1858_185827

noncomputable def value_after_years (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

def machine1_initial_value : ℝ := 2500
def machine1_depreciation_rate : ℝ := 0.05
def machine2_initial_value : ℝ := 3500
def machine2_depreciation_rate : ℝ := 0.07
def machine3_initial_value : ℝ := 4500
def machine3_depreciation_rate : ℝ := 0.04
def years : ℕ := 3

theorem total_value_after_3_years :
  value_after_years machine1_initial_value machine1_depreciation_rate years +
  value_after_years machine2_initial_value machine2_depreciation_rate years +
  value_after_years machine3_initial_value machine3_depreciation_rate years = 8940 :=
by
  sorry

end total_value_after_3_years_l1858_185827


namespace linear_function_quadrants_l1858_185838

theorem linear_function_quadrants (k : ℝ) :
  (∀ x y : ℝ, y = (k + 1) * x + k - 2 → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) ∨ (x > 0 ∧ y < 0))) ↔ (-1 < k ∧ k < 2) := 
sorry

end linear_function_quadrants_l1858_185838


namespace negation_of_positive_x2_plus_2_l1858_185891

theorem negation_of_positive_x2_plus_2 (h : ∀ x : ℝ, x^2 + 2 > 0) : ¬ (∀ x : ℝ, x^2 + 2 > 0) = False := 
by
  sorry

end negation_of_positive_x2_plus_2_l1858_185891


namespace meal_cost_l1858_185803

/-- 
    Define the cost of a meal consisting of one sandwich, one cup of coffee, and one piece of pie 
    given the costs of two different meals.
-/
theorem meal_cost (s c p : ℝ) (h1 : 2 * s + 5 * c + p = 5) (h2 : 3 * s + 8 * c + p = 7) :
    s + c + p = 3 :=
by
  sorry

end meal_cost_l1858_185803


namespace calculate_f_sum_l1858_185836

noncomputable def f (n : ℕ) := Real.log (3 * n^2) / Real.log 3003

theorem calculate_f_sum :
  f 7 + f 11 + f 13 = 2 :=
by
  sorry

end calculate_f_sum_l1858_185836
