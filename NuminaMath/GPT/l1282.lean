import Mathlib

namespace real_inequality_l1282_128239

theorem real_inequality (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + a * c + b * c := by
  sorry

end real_inequality_l1282_128239


namespace trigonometric_identity_l1282_128202

open Real

noncomputable def acute (x : ℝ) := 0 < x ∧ x < π / 2

theorem trigonometric_identity 
  {α β : ℝ} (hα : acute α) (hβ : acute β) (h : cos α > sin β) :
  α + β < π / 2 :=
sorry

end trigonometric_identity_l1282_128202


namespace zookeeper_feeding_problem_l1282_128215

noncomputable def feeding_ways : ℕ :=
  sorry

theorem zookeeper_feeding_problem :
  feeding_ways = 2880 := 
sorry

end zookeeper_feeding_problem_l1282_128215


namespace no_two_exact_cubes_between_squares_l1282_128299

theorem no_two_exact_cubes_between_squares :
  ∀ (n a b : ℤ), ¬ (n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2) :=
by
  intros n a b
  sorry

end no_two_exact_cubes_between_squares_l1282_128299


namespace ratio_son_grandson_l1282_128240

-- Define the conditions
variables (Markus_age Son_age Grandson_age : ℕ)
axiom Markus_twice_son : Markus_age = 2 * Son_age
axiom sum_ages : Markus_age + Son_age + Grandson_age = 140
axiom Grandson_age_20 : Grandson_age = 20

-- Define the goal to prove
theorem ratio_son_grandson : (Son_age : ℚ) / Grandson_age = 2 :=
by
  sorry

end ratio_son_grandson_l1282_128240


namespace chosen_number_is_5_l1282_128231

theorem chosen_number_is_5 (x : ℕ) (h_pos : x > 0)
  (h_eq : ((10 * x + 5 - x^2) / x) - x = 1) : x = 5 :=
by
  sorry

end chosen_number_is_5_l1282_128231


namespace bus_ride_cost_l1282_128235

/-- The cost of a bus ride from town P to town Q, given that the cost of a train ride is $2.35 more 
    than a bus ride, and the combined cost of one train ride and one bus ride is $9.85. -/
theorem bus_ride_cost (B : ℝ) (h1 : ∃T, T = B + 2.35) (h2 : ∃T, T + B = 9.85) : B = 3.75 :=
by
  obtain ⟨T1, hT1⟩ := h1
  obtain ⟨T2, hT2⟩ := h2
  simp only [hT1, add_right_inj] at hT2
  sorry

end bus_ride_cost_l1282_128235


namespace parallel_lines_a_value_l1282_128251

theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, ((a + 1) * x + 3 * y + 3 = 0) → (x + (a - 1) * y + 1 = 0)) → a = -2 :=
by
  sorry

end parallel_lines_a_value_l1282_128251


namespace no_solution_of_fractional_equation_l1282_128283

theorem no_solution_of_fractional_equation (x : ℝ) : ¬ (x - 8) / (x - 7) - 8 = 1 / (7 - x) := 
sorry

end no_solution_of_fractional_equation_l1282_128283


namespace repeating_decimal_fraction_difference_l1282_128263

theorem repeating_decimal_fraction_difference :
  ∀ (F : ℚ),
  F = 817 / 999 → (999 - 817 = 182) :=
by
  sorry

end repeating_decimal_fraction_difference_l1282_128263


namespace profit_without_discount_l1282_128291

theorem profit_without_discount
  (CP SP_with_discount : ℝ) 
  (H1 : CP = 100) -- Assume cost price is 100
  (H2 : SP_with_discount = CP + 0.216 * CP) -- Selling price with discount
  (H3 : SP_with_discount = 0.95 * SP_without_discount) -- SP with discount is 95% of SP without discount
  : (SP_without_discount - CP) / CP * 100 = 28 := 
by
  -- proof goes here
  sorry

end profit_without_discount_l1282_128291


namespace visiting_plans_correct_l1282_128218

-- Define the number of students
def num_students : ℕ := 4

-- Define the number of places to visit
def num_places : ℕ := 3

-- Define the total number of visiting plans without any restrictions
def total_visiting_plans : ℕ := num_places ^ num_students

-- Define the number of visiting plans where no one visits Haxi Station
def no_haxi_visiting_plans : ℕ := (num_places - 1) ^ num_students

-- Define the number of visiting plans where Haxi Station has at least one visitor
def visiting_plans_with_haxi : ℕ := total_visiting_plans - no_haxi_visiting_plans

-- Prove that the number of different visiting plans with at least one student visiting Haxi Station is 65
theorem visiting_plans_correct : visiting_plans_with_haxi = 65 := by
  -- Omitted proof
  sorry

end visiting_plans_correct_l1282_128218


namespace problem_1_problem_2_problem_3_problem_4_l1282_128230

theorem problem_1 : 42.67 - (12.67 - 2.87) = 32.87 :=
by sorry

theorem problem_2 : (4.8 - 4.8 * (3.2 - 2.7)) / 0.24 = 10 :=
by sorry

theorem problem_3 : 4.31 * 0.57 + 0.43 * 4.31 - 4.31 = 0 :=
by sorry

theorem problem_4 : 9.99 * 222 + 3.33 * 334 = 3330 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l1282_128230


namespace trigonometric_identity_l1282_128249

-- Define the problem conditions and formulas
variables (α : Real) (h : Real.cos (Real.pi / 6 + α) = Real.sqrt 3 / 3)

-- State the theorem
theorem trigonometric_identity : Real.cos (5 * Real.pi / 6 - α) = - (Real.sqrt 3 / 3) :=
by
  -- Placeholder for the proof
  sorry

end trigonometric_identity_l1282_128249


namespace molecular_weight_compound_l1282_128227

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def num_H : ℝ := 1
def num_Br : ℝ := 1
def num_O : ℝ := 3

def molecular_weight (num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O : ℝ) : ℝ :=
  (num_H * atomic_weight_H) + (num_Br * atomic_weight_Br) + (num_O * atomic_weight_O)

theorem molecular_weight_compound : molecular_weight num_H num_Br num_O atomic_weight_H atomic_weight_Br atomic_weight_O = 128.91 :=
by
  sorry

end molecular_weight_compound_l1282_128227


namespace f_is_odd_f_is_decreasing_range_of_m_l1282_128253

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (1 + 2^x)

-- Prove that f(x) is an odd function
theorem f_is_odd (x : ℝ) : f (-x) = - f x := by
  sorry

-- Prove that f(x) is decreasing on ℝ
theorem f_is_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := by
  sorry

-- Prove the range of m if f(m-1) + f(2m-1) > 0
theorem range_of_m (m : ℝ) (h : f (m - 1) + f (2 * m - 1) > 0) : m < 2 / 3 := by
  sorry

end f_is_odd_f_is_decreasing_range_of_m_l1282_128253


namespace proposition_p_and_not_q_l1282_128296

theorem proposition_p_and_not_q (P Q : Prop) 
  (h1 : P ∨ Q) 
  (h2 : ¬ (P ∧ Q)) : (P ↔ ¬ Q) :=
sorry

end proposition_p_and_not_q_l1282_128296


namespace evaluate_expression_l1282_128272

theorem evaluate_expression (x : ℝ) : x * (x * (x * (x - 3) - 5) + 12) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 12 * x + 2 :=
by
  sorry

end evaluate_expression_l1282_128272


namespace count_integers_in_interval_l1282_128241

theorem count_integers_in_interval : 
  ∃ (n : ℕ), (∀ (x : ℤ), (-2 ≤ x ∧ x ≤ 8 → ∃ (k : ℕ), k < n ∧ x = -2 + k)) ∧ n = 11 := 
by
  sorry

end count_integers_in_interval_l1282_128241


namespace new_average_production_l1282_128254

theorem new_average_production (n : ℕ) (average_past today : ℕ) (h₁ : average_past = 70) (h₂ : today = 90) (h₃ : n = 3) : 
  (average_past * n + today) / (n + 1) = 75 := by
  sorry

end new_average_production_l1282_128254


namespace part1_part2_l1282_128290

-- Part 1
theorem part1 : (9 / 4) ^ (1 / 2) - (-2.5) ^ 0 - (8 / 27) ^ (2 / 3) + (3 / 2) ^ (-2) = 1 / 2 := 
by sorry

-- Part 2
theorem part2 (lg : ℝ → ℝ) -- Assuming a hypothetical lg function for demonstration
  (lg_prop1 : lg 10 = 1)
  (lg_prop2 : ∀ x y, lg (x * y) = lg x + lg y) :
  (lg 5) ^ 2 + lg 2 * lg 50 = 1 := 
by sorry

end part1_part2_l1282_128290


namespace maximum_value_fraction_l1282_128298

theorem maximum_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) ≤ 2 / 3 :=
sorry

end maximum_value_fraction_l1282_128298


namespace distance_center_is_12_l1282_128236

-- Define the side length of the square and the radius of the circle
def side_length_square : ℝ := 5
def radius_circle : ℝ := 1

-- The center path forms a smaller square inside the original square
-- with side length 3 units
def side_length_smaller_square : ℝ := side_length_square - 2 * radius_circle

-- The perimeter of the smaller square, which is the path length that
-- the center of the circle travels
def distance_center_travel : ℝ := 4 * side_length_smaller_square

-- Prove that the distance traveled by the center of the circle is 12 units
theorem distance_center_is_12 : distance_center_travel = 12 := by
  -- the proof is skipped
  sorry

end distance_center_is_12_l1282_128236


namespace largest_mersenne_prime_is_127_l1282_128275

noncomputable def largest_mersenne_prime_less_than_500 : ℕ :=
  127

theorem largest_mersenne_prime_is_127 :
  ∃ p : ℕ, Nat.Prime p ∧ (2^p - 1) = largest_mersenne_prime_less_than_500 ∧ 2^p - 1 < 500 := 
by 
  -- The largest Mersenne prime less than 500 is 127
  use 7
  sorry

end largest_mersenne_prime_is_127_l1282_128275


namespace ratio_of_areas_l1282_128295

theorem ratio_of_areas (r : ℝ) (w_smaller : ℝ) (h_smaller : ℝ) (h_semi : ℝ) :
  (5 / 4) * 40 = r + 40 →
  h_semi = 20 →
  w_smaller = 5 →
  h_smaller = 20 →
  2 * w_smaller * h_smaller / ((1 / 2) * π * h_semi^2) = 1 / π :=
by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_l1282_128295


namespace alpha_necessary_but_not_sufficient_for_beta_l1282_128288

theorem alpha_necessary_but_not_sufficient_for_beta 
  (a b : ℝ) (hα : b * (b - a) ≤ 0) (hβ : a / b ≥ 1) : 
  (b * (b - a) ≤ 0) ↔ (a / b ≥ 1) := 
sorry

end alpha_necessary_but_not_sufficient_for_beta_l1282_128288


namespace price_of_each_shirt_is_15_30_l1282_128232

theorem price_of_each_shirt_is_15_30:
  ∀ (shorts_price : ℝ) (num_shorts : ℕ) (shirt_num : ℕ) (total_paid : ℝ) (discount : ℝ),
  shorts_price = 15 →
  num_shorts = 3 →
  shirt_num = 5 →
  total_paid = 117 →
  discount = 0.10 →
  (total_paid - (num_shorts * shorts_price - discount * (num_shorts * shorts_price))) / shirt_num = 15.30 :=
by 
  sorry

end price_of_each_shirt_is_15_30_l1282_128232


namespace triangle_PQR_area_l1282_128278

/-- Given a triangle PQR where PQ = 4 miles, PR = 2 miles, and PQ is along Pine Street
and PR is along Quail Road, and there is a sub-triangle PQS within PQR
with PS = 2 miles along Summit Avenue and QS = 3 miles along Pine Street,
prove that the area of triangle PQR is 4 square miles --/
theorem triangle_PQR_area :
  ∀ (PQ PR PS QS : ℝ),
    PQ = 4 → PR = 2 → PS = 2 → QS = 3 →
    (1/2) * PQ * PR = 4 :=
by
  intros PQ PR PS QS hpq hpr hps hqs
  rw [hpq, hpr]
  norm_num
  done

end triangle_PQR_area_l1282_128278


namespace geometric_sequence_property_l1282_128280

variable (a : ℕ → ℤ)
-- Assume the sequence is geometric with ratio r
variable (r : ℤ)

-- Define the sequence a_n as a geometric sequence
def geometric_sequence (a : ℕ → ℤ) (r : ℤ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * r

-- Given condition: a_4 + a_8 = -2
axiom condition : a 4 + a 8 = -2

theorem geometric_sequence_property
  (h : geometric_sequence a r) : a 6 * (a 2 + 2 * a 6 + a 10) = 4 :=
sorry

end geometric_sequence_property_l1282_128280


namespace math_problem_l1282_128211

def f (x : ℝ) : ℝ := sorry

theorem math_problem (n s : ℕ)
  (h1 : f 1 = 2)
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y))
  (hn : n = 1)
  (hs : s = 6) :
  n * s = 6 := by
  sorry

end math_problem_l1282_128211


namespace lord_moneybag_l1282_128287

theorem lord_moneybag (n : ℕ) (hlow : 300 ≤ n) (hhigh : n ≤ 500)
           (h6 : 6 ∣ n) (h5 : 5 ∣ (n - 1)) (h4 : 4 ∣ (n - 2)) 
           (h3 : 3 ∣ (n - 3)) (h2 : 2 ∣ (n - 4)) (hprime : Nat.Prime (n - 5)) :
  n = 426 := by
  sorry

end lord_moneybag_l1282_128287


namespace cauliflower_production_diff_l1282_128292

theorem cauliflower_production_diff
  (area_this_year : ℕ)
  (area_last_year : ℕ)
  (side_this_year : ℕ)
  (side_last_year : ℕ)
  (H1 : side_this_year * side_this_year = area_this_year)
  (H2 : side_last_year * side_last_year = area_last_year)
  (H3 : side_this_year = side_last_year + 1)
  (H4 : area_this_year = 12544) :
  area_this_year - area_last_year = 223 :=
by
  sorry

end cauliflower_production_diff_l1282_128292


namespace triangle_perfect_square_l1282_128213

theorem triangle_perfect_square (a b c : ℤ) (h : ∃ h₁ h₂ h₃ : ℤ, (1/2) * a * h₁ = (1/2) * b * h₂ ∧ (1/2) * b * h₂ = (1/2) * c * h₃ ∧ (h₁ = h₂ + h₃)) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_perfect_square_l1282_128213


namespace common_root_solutions_l1282_128242

theorem common_root_solutions (a : ℝ) (b : ℝ) :
  (a^2 * b^2 + a * b - 1 = 0) ∧ (b^2 - a * b - a^2 = 0) →
  a = (-1 + Real.sqrt 5) / 2 ∨ a = (-1 - Real.sqrt 5) / 2 ∨
  a = (1 + Real.sqrt 5) / 2 ∨ a = (1 - Real.sqrt 5) / 2 :=
by
  intro h
  sorry

end common_root_solutions_l1282_128242


namespace min_value_fraction_l1282_128260

theorem min_value_fraction (a b : ℝ) (h : x^2 - 3*x + a*b < 0 ∧ 1 < x ∧ x < 2) (h1 : a > b) : 
  (∃ minValue : ℝ, minValue = 4 ∧ ∀ a b : ℝ, a > b → minValue ≤ (a^2 + b^2) / (a - b)) := 
sorry

end min_value_fraction_l1282_128260


namespace car_average_speed_l1282_128297

def average_speed (speed1 speed2 : ℕ) (time1 time2 : ℕ) : ℕ := 
  (speed1 * time1 + speed2 * time2) / (time1 + time2)

theorem car_average_speed :
  average_speed 60 90 (1/3) (2/3) = 80 := 
by 
  sorry

end car_average_speed_l1282_128297


namespace total_amount_is_correct_l1282_128228

-- Definitions based on the conditions
def share_a (x : ℕ) : ℕ := 2 * x
def share_b (x : ℕ) : ℕ := 4 * x
def share_c (x : ℕ) : ℕ := 5 * x
def share_d (x : ℕ) : ℕ := 4 * x

-- Condition: combined share of a and b is 1800
def combined_share_of_ab (x : ℕ) : Prop := share_a x + share_b x = 1800

-- Theorem we want to prove: Total amount given to all children is $4500
theorem total_amount_is_correct (x : ℕ) (h : combined_share_of_ab x) : 
  share_a x + share_b x + share_c x + share_d x = 4500 := sorry

end total_amount_is_correct_l1282_128228


namespace number_of_two_digit_primes_with_ones_digit_three_l1282_128279

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop :=
  n % 10 = 3

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_ones_digit_three :
  ∃! s : Finset ℕ, (∀ n ∈ s, is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n) ∧ s.card = 6 :=
by
  sorry

end number_of_two_digit_primes_with_ones_digit_three_l1282_128279


namespace solve_inequality_l1282_128250

theorem solve_inequality (x : ℝ) : -3 * x^2 + 8 * x + 1 < 0 ↔ x ∈ Set.Ioo (-1 / 3 : ℝ) 1 :=
sorry

end solve_inequality_l1282_128250


namespace sampling_interval_divisor_l1282_128201

theorem sampling_interval_divisor (P : ℕ) (hP : P = 524) (k : ℕ) (hk : k ∣ P) : k = 4 :=
by
  sorry

end sampling_interval_divisor_l1282_128201


namespace num_of_lists_is_correct_l1282_128224

theorem num_of_lists_is_correct :
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  total_lists = 50625 :=
by
  let num_balls := 15
  let num_selections := 4
  let total_lists := num_balls ^ num_selections
  show total_lists = 50625
  sorry

end num_of_lists_is_correct_l1282_128224


namespace neg_exists_exp_l1282_128203

theorem neg_exists_exp (p : Prop) :
  (¬ (∃ x : ℝ, Real.exp x < 0)) = (∀ x : ℝ, Real.exp x ≥ 0) :=
by
  sorry

end neg_exists_exp_l1282_128203


namespace factorization_eq_l1282_128255

theorem factorization_eq (x : ℝ) : 
  -3 * x^3 + 12 * x^2 - 12 * x = -3 * x * (x - 2)^2 :=
by
  sorry

end factorization_eq_l1282_128255


namespace chloe_cherries_l1282_128229

noncomputable def cherries_received (x y : ℝ) : Prop :=
  x = y + 8 ∧ y = x / 3

theorem chloe_cherries : ∃ (x : ℝ), ∀ (y : ℝ), cherries_received x y → x = 12 := 
by
  sorry

end chloe_cherries_l1282_128229


namespace x_fifth_power_sum_l1282_128243

theorem x_fifth_power_sum (x : ℝ) (h : x + 1 / x = -5) : x^5 + 1 / x^5 = -2525 := by
  sorry

end x_fifth_power_sum_l1282_128243


namespace simplify_expression_l1282_128208

theorem simplify_expression (p q r s : ℝ) (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) :
    (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end simplify_expression_l1282_128208


namespace problem_statements_l1282_128266

noncomputable def f (x : ℕ) : ℕ := x % 2
noncomputable def g (x : ℕ) : ℕ := x % 3

theorem problem_statements (x : ℕ) : (f (2 * x) = 0) ∧ (f x + f (x + 3) = 1) :=
by
  sorry

end problem_statements_l1282_128266


namespace sphere_volume_given_surface_area_l1282_128220

theorem sphere_volume_given_surface_area (r : ℝ) (V : ℝ) (S : ℝ)
  (hS : S = 36 * Real.pi)
  (h_surface_area : 4 * Real.pi * r^2 = S)
  (h_volume : V = (4/3) * Real.pi * r^3) : V = 36 * Real.pi := by
  sorry

end sphere_volume_given_surface_area_l1282_128220


namespace sum_of_divisors_85_l1282_128285

theorem sum_of_divisors_85 : (1 + 5 + 17 + 85 = 108) := by
  sorry

end sum_of_divisors_85_l1282_128285


namespace garden_area_maximal_l1282_128238

/-- Given a garden with sides 20 meters, 16 meters, 12 meters, and 10 meters, 
    prove that the area is approximately 194.4 square meters. -/
theorem garden_area_maximal (a b c d : ℝ) (h1 : a = 20) (h2 : b = 16) (h3 : c = 12) (h4 : d = 10) :
    ∃ A : ℝ, abs (A - 194.4) < 0.1 :=
by
  sorry

end garden_area_maximal_l1282_128238


namespace mass_percentage_of_Cl_in_NaOCl_l1282_128256

theorem mass_percentage_of_Cl_in_NaOCl :
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  100 * (Cl_mass / NaOCl_mass) = 47.6 := 
by
  let Na_mass := 22.99
  let O_mass := 16.00
  let Cl_mass := 35.45
  let NaOCl_mass := Na_mass + O_mass + Cl_mass
  sorry

end mass_percentage_of_Cl_in_NaOCl_l1282_128256


namespace a_2_value_l1282_128209

theorem a_2_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) (x : ℝ) :
  x^3 + x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 +
  a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^10 → 
  a2 = 42 :=
by
  sorry

end a_2_value_l1282_128209


namespace angle_measure_l1282_128210

theorem angle_measure (x : ℝ) 
  (h1 : 5 * x + 12 = 180 - x) : x = 28 := by
  sorry

end angle_measure_l1282_128210


namespace tangent_line_at_five_l1282_128268

variable {f : ℝ → ℝ}

theorem tangent_line_at_five 
  (h_tangent : ∀ x, f x = -x + 8)
  (h_tangent_deriv : deriv f 5 = -1) :
  f 5 = 3 ∧ deriv f 5 = -1 :=
by sorry

end tangent_line_at_five_l1282_128268


namespace xy_sum_l1282_128271

variable (x y : ℚ)

theorem xy_sum : (1/x + 1/y = 4) → (1/x - 1/y = -6) → x + y = -4/5 := by
  intros h1 h2
  sorry

end xy_sum_l1282_128271


namespace sqrt_112_consecutive_integers_product_l1282_128221

theorem sqrt_112_consecutive_integers_product : 
  (∃ (a b : ℕ), a * a < 112 ∧ 112 < b * b ∧ b = a + 1 ∧ a * b = 110) :=
by 
  use 10, 11
  repeat { sorry }

end sqrt_112_consecutive_integers_product_l1282_128221


namespace simplify_product_of_fractions_l1282_128223

theorem simplify_product_of_fractions :
  (252 / 21) * (7 / 168) * (12 / 4) = 3 / 2 :=
by
  sorry

end simplify_product_of_fractions_l1282_128223


namespace tenth_term_is_26_l1282_128222

-- Definitions used from the conditions
def first_term : ℤ := 8
def common_difference : ℤ := 2
def term_number : ℕ := 10

-- Define the formula for the nth term of an arithmetic progression
def nth_term (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Proving that the 10th term is 26 given the conditions
theorem tenth_term_is_26 : nth_term first_term common_difference term_number = 26 := by
  sorry

end tenth_term_is_26_l1282_128222


namespace pool_ratio_l1282_128264

theorem pool_ratio 
  (total_pools : ℕ)
  (ark_athletic_wear_pools : ℕ)
  (total_pools_eq : total_pools = 800)
  (ark_athletic_wear_pools_eq : ark_athletic_wear_pools = 200)
  : ((total_pools - ark_athletic_wear_pools) / ark_athletic_wear_pools) = 3 :=
by
  sorry

end pool_ratio_l1282_128264


namespace negation_proof_l1282_128246

theorem negation_proof :
  ¬ (∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  -- Proof to be filled
  sorry

end negation_proof_l1282_128246


namespace initial_ratio_milk_water_l1282_128274

-- Define the initial conditions
variables (M W : ℕ) (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4)

-- State the theorem to prove the initial ratio of milk to water
theorem initial_ratio_milk_water (h_volume : M + W = 115) (h_ratio : M / (W + 46) = 3 / 4) :
  (M * 2 = W * 3) :=
by
  sorry

end initial_ratio_milk_water_l1282_128274


namespace rice_difference_on_15th_and_first_10_squares_l1282_128262

-- Definitions
def grains_on_square (k : ℕ) : ℕ := 3^k

def sum_first_n_squares (n : ℕ) : ℕ := 
  (3 * (3^n - 1)) / (3 - 1)

-- Theorem statement
theorem rice_difference_on_15th_and_first_10_squares :
  grains_on_square 15 - sum_first_n_squares 10 = 14260335 :=
by
  sorry

end rice_difference_on_15th_and_first_10_squares_l1282_128262


namespace opposite_of_3_l1282_128284

theorem opposite_of_3 : -3 = -3 := 
by
  -- sorry is added to skip the proof as per instructions
  sorry

end opposite_of_3_l1282_128284


namespace maximal_regions_convex_quadrilaterals_l1282_128200

theorem maximal_regions_convex_quadrilaterals (n : ℕ) (hn : n ≥ 1) : 
  ∃ a_n : ℕ, a_n = 4*n^2 - 4*n + 2 :=
by
  sorry

end maximal_regions_convex_quadrilaterals_l1282_128200


namespace totalStudents_l1282_128273

-- Define the number of seats per ride
def seatsPerRide : ℕ := 15

-- Define the number of empty seats per ride
def emptySeatsPerRide : ℕ := 3

-- Define the number of rides taken
def ridesTaken : ℕ := 18

-- Define the number of students per ride
def studentsPerRide (seats : ℕ) (empty : ℕ) : ℕ := seats - empty

-- Calculate the total number of students
theorem totalStudents : studentsPerRide seatsPerRide emptySeatsPerRide * ridesTaken = 216 :=
by
  sorry

end totalStudents_l1282_128273


namespace boys_passed_percentage_l1282_128217

theorem boys_passed_percentage
  (total_candidates : ℝ)
  (total_girls : ℝ)
  (failed_percentage : ℝ)
  (girls_passed_percentage : ℝ)
  (boys_passed_percentage : ℝ) :
  total_candidates = 2000 →
  total_girls = 900 →
  failed_percentage = 70.2 →
  girls_passed_percentage = 32 →
  boys_passed_percentage = 28 :=
by
  sorry

end boys_passed_percentage_l1282_128217


namespace part1_part2_l1282_128248
noncomputable def equation1 (x k : ℝ) := 3 * (2 * x - 1) = k + 2 * x
noncomputable def equation2 (x k : ℝ) := (x - k) / 2 = x + 2 * k

theorem part1 (x k : ℝ) (h1 : equation1 4 k) : equation2 x k ↔ x = -65 := sorry

theorem part2 (x k : ℝ) (h1 : equation1 x k) (h2 : equation2 x k) : k = -1 / 7 := sorry

end part1_part2_l1282_128248


namespace jessa_cupcakes_l1282_128219

-- Define the number of classes and students
def fourth_grade_classes : ℕ := 3
def students_per_fourth_grade_class : ℕ := 30
def pe_classes : ℕ := 1
def students_per_pe_class : ℕ := 50

-- Calculate the total number of cupcakes needed
def total_cupcakes_needed : ℕ :=
  (fourth_grade_classes * students_per_fourth_grade_class) +
  (pe_classes * students_per_pe_class)

-- Statement to prove
theorem jessa_cupcakes : total_cupcakes_needed = 140 :=
by
  sorry

end jessa_cupcakes_l1282_128219


namespace complement_M_in_U_l1282_128286

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_M_in_U :
  U \ M = {3, 5, 6} :=
by sorry

end complement_M_in_U_l1282_128286


namespace find_staff_age_l1282_128293

theorem find_staff_age (n_students : ℕ) (avg_age_students : ℕ) (avg_age_with_staff : ℕ) (total_students : ℕ) :
  n_students = 32 →
  avg_age_students = 16 →
  avg_age_with_staff = 17 →
  total_students = 33 →
  (33 * 17 - 32 * 16) = 49 :=
by
  intros
  sorry

end find_staff_age_l1282_128293


namespace interest_rate_borrowed_l1282_128206

variables {P : Type} [LinearOrderedField P]

def borrowed_amount : P := 9000
def lent_interest_rate : P := 0.06
def gain_per_year : P := 180
def per_cent : P := 100

theorem interest_rate_borrowed (r : P) (h : borrowed_amount * lent_interest_rate - gain_per_year = borrowed_amount * r) : 
  r = 0.04 :=
by sorry

end interest_rate_borrowed_l1282_128206


namespace isle_of_unluckiness_l1282_128277

-- Definitions:
def is_knight (i : ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, k = i * n / 100 ∧ k > 0

-- Main statement:
theorem isle_of_unluckiness (n : ℕ) (h : n ∈ [1, 2, 4, 5, 10, 20, 25, 50, 100]) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ is_knight i n := by
  sorry

end isle_of_unluckiness_l1282_128277


namespace einstein_birth_weekday_l1282_128289

-- Defining the reference day of the week for 31 May 2006
def reference_date := 31
def reference_month := 5
def reference_year := 2006
def reference_weekday := 3  -- Wednesday

-- Defining Albert Einstein's birth date
def einstein_birth_day := 14
def einstein_birth_month := 3
def einstein_birth_year := 1879

-- Defining the calculation of weekday
def weekday_from_reference(reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year : Nat) : Nat :=
  let days_from_reference_to_birth := 46464  -- Total days calculated in solution
  (reference_weekday - (days_from_reference_to_birth % 7) + 7) % 7

-- Stating the theorem
theorem einstein_birth_weekday : weekday_from_reference reference_day reference_weekday einstein_birth_day einstein_birth_month einstein_birth_year = 5 :=
by
  -- Proof omitted
  sorry

end einstein_birth_weekday_l1282_128289


namespace smallest_n_for_terminating_fraction_l1282_128247

-- Define what it means for a number to be a product of only prime factors of 2 and 5
def isTerminatingDenominator (d : ℕ) : Prop := ∃ (a b : ℕ), d = 2^a * 5^b

-- The main statement to prove
theorem smallest_n_for_terminating_fraction : 
  ∃ (n : ℕ), 0 < n ∧ isTerminatingDenominator (n + 150) ∧ 
  ∀ (m : ℕ), (0 < m → isTerminatingDenominator (m + 150) → n ≤ m)
:= sorry

end smallest_n_for_terminating_fraction_l1282_128247


namespace probability_good_or_excellent_l1282_128204

noncomputable def P_H1 : ℚ := 5 / 21
noncomputable def P_H2 : ℚ := 10 / 21
noncomputable def P_H3 : ℚ := 6 / 21

noncomputable def P_A_given_H1 : ℚ := 1
noncomputable def P_A_given_H2 : ℚ := 1
noncomputable def P_A_given_H3 : ℚ := 1 / 3

noncomputable def P_A : ℚ := 
  P_H1 * P_A_given_H1 + 
  P_H2 * P_A_given_H2 + 
  P_H3 * P_A_given_H3

theorem probability_good_or_excellent : P_A = 17 / 21 :=
by
  sorry

end probability_good_or_excellent_l1282_128204


namespace complex_division_l1282_128259

def i_units := Complex.I

def numerator := (3 : ℂ) + i_units
def denominator := (1 : ℂ) + i_units
def expected_result := (2 : ℂ) - i_units

theorem complex_division :
  numerator / denominator = expected_result :=
by sorry

end complex_division_l1282_128259


namespace fraction_home_l1282_128257

-- Defining the conditions
def fractionFun := 5 / 13
def fractionYouth := 4 / 13

-- Stating the theorem to be proven
theorem fraction_home : 1 - (fractionFun + fractionYouth) = 4 / 13 := by
  sorry

end fraction_home_l1282_128257


namespace population_after_panic_l1282_128237

noncomputable def original_population : ℕ := 7200
def first_event_loss (population : ℕ) : ℕ := population * 10 / 100
def after_first_event (population : ℕ) : ℕ := population - first_event_loss population
def second_event_loss (population : ℕ) : ℕ := population * 25 / 100
def after_second_event (population : ℕ) : ℕ := population - second_event_loss population

theorem population_after_panic : after_second_event (after_first_event original_population) = 4860 := sorry

end population_after_panic_l1282_128237


namespace no_perfect_squares_in_seq_l1282_128270

def seq (x : ℕ → ℤ) : Prop :=
  x 0 = 1 ∧ x 1 = 3 ∧ ∀ n : ℕ, 0 < n → x (n + 1) = 6 * x n - x (n - 1)

theorem no_perfect_squares_in_seq (x : ℕ → ℤ) (n : ℕ) (h_seq : seq x) :
  ¬ ∃ k : ℤ, k * k = x (n + 1) :=
by
  sorry

end no_perfect_squares_in_seq_l1282_128270


namespace log_fraction_eq_l1282_128281

variable (a b : ℝ)
axiom h1 : a = Real.logb 3 5
axiom h2 : b = Real.logb 5 7

theorem log_fraction_eq : Real.logb 15 (49 / 45) = (2 * (a * b) - a - 2) / (1 + a) :=
by sorry

end log_fraction_eq_l1282_128281


namespace trading_cards_initial_total_l1282_128225

theorem trading_cards_initial_total (x : ℕ) 
  (h1 : ∃ d : ℕ, d = (1 / 3 : ℕ) * x)
  (h2 : ∃ n1 : ℕ, n1 = (1 / 5 : ℕ) * (1 / 3 : ℕ) * x)
  (h3 : ∃ n2 : ℕ, n2 = (1 / 3 : ℕ) * ((1 / 5 : ℕ) * (1 / 3 : ℕ) * x))
  (h4 : ∃ n3 : ℕ, n3 = (1 / 2 : ℕ) * (2 / 45 : ℕ) * x)
  (h5 : (1 / 15 : ℕ) * x + (2 / 45 : ℕ) * x + (1 / 45 : ℕ) * x = 850) :
  x = 6375 := 
sorry

end trading_cards_initial_total_l1282_128225


namespace cost_of_baseball_cards_l1282_128233

variables (cost_football cost_pokemon total_spent cost_baseball : ℝ)
variable (h1 : cost_football = 2 * 2.73)
variable (h2 : cost_pokemon = 4.01)
variable (h3 : total_spent = 18.42)
variable (total_cost_football_pokemon : ℝ)
variable (h4 : total_cost_football_pokemon = cost_football + cost_pokemon)

theorem cost_of_baseball_cards
  (h : cost_baseball = total_spent - total_cost_football_pokemon) : 
  cost_baseball = 8.95 :=
by
  sorry

end cost_of_baseball_cards_l1282_128233


namespace binom_10_3_eq_120_l1282_128207

theorem binom_10_3_eq_120 : Nat.choose 10 3 = 120 := by
  sorry

end binom_10_3_eq_120_l1282_128207


namespace race_times_l1282_128282

theorem race_times (x y : ℕ) (h1 : 5 * x + 1 = 4 * y) (h2 : 5 * y - 8 = 4 * x) :
  5 * x = 15 ∧ 5 * y = 20 :=
by
  sorry

end race_times_l1282_128282


namespace Eve_total_running_distance_l1282_128212

def Eve_walked_distance := 0.6

def Eve_ran_distance := Eve_walked_distance + 0.1

theorem Eve_total_running_distance : Eve_ran_distance = 0.7 := 
by sorry

end Eve_total_running_distance_l1282_128212


namespace g_of_1001_l1282_128214

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) + x = x * g y + g x
axiom g_of_1 : g 1 = -3

theorem g_of_1001 : g 1001 = -2001 := 
by sorry

end g_of_1001_l1282_128214


namespace total_workers_calculation_l1282_128234

theorem total_workers_calculation :
  ∀ (N : ℕ), 
  (∀ (total_avg_salary : ℕ) (techs_salary : ℕ) (nontech_avg_salary : ℕ),
    total_avg_salary = 8000 → 
    techs_salary = 7 * 20000 → 
    nontech_avg_salary = 6000 →
    8000 * (7 + N) = 7 * 20000 + N * 6000 →
    (7 + N) = 49) :=
by
  intros
  sorry

end total_workers_calculation_l1282_128234


namespace added_amount_l1282_128294

theorem added_amount (x y : ℕ) (h1 : x = 17) (h2 : 3 * (2 * x + y) = 117) : y = 5 :=
by
  sorry

end added_amount_l1282_128294


namespace even_func_monotonic_on_negative_interval_l1282_128245

variable {α : Type*} [LinearOrderedField α]
variable {f : α → α}

theorem even_func_monotonic_on_negative_interval 
  (h_even : ∀ x : α, f (-x) = f x)
  (h_mon_incr : ∀ x y : α, x < y → (x < 0 ∧ y ≤ 0) → f x < f y) :
  f 2 < f (-3 / 2) :=
sorry

end even_func_monotonic_on_negative_interval_l1282_128245


namespace total_sales_is_10400_l1282_128226

-- Define the conditions
def tough_week_sales : ℝ := 800
def good_week_sales : ℝ := 2 * tough_week_sales
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the total sales function
def total_sales (good_sales : ℝ) (tough_sales : ℝ) (good_weeks : ℕ) (tough_weeks : ℕ) : ℝ :=
  good_weeks * good_sales + tough_weeks * tough_sales

-- Prove that the total sales is $10400
theorem total_sales_is_10400 : total_sales good_week_sales tough_week_sales good_weeks tough_weeks = 10400 := 
by
  sorry

end total_sales_is_10400_l1282_128226


namespace total_time_l1282_128269

theorem total_time {minutes seconds : ℕ} (hmin : minutes = 3450) (hsec : seconds = 7523) :
  ∃ h m s : ℕ, h = 59 ∧ m = 35 ∧ s = 23 :=
by
  sorry

end total_time_l1282_128269


namespace probability_of_draw_l1282_128205

-- Define the probabilities as given conditions
def P_A : ℝ := 0.4
def P_A_not_losing : ℝ := 0.9

-- Define the probability of drawing
def P_draw : ℝ :=
  P_A_not_losing - P_A

-- State the theorem to be proved
theorem probability_of_draw : P_draw = 0.5 := by
  sorry

end probability_of_draw_l1282_128205


namespace arithmetic_evaluation_l1282_128252

theorem arithmetic_evaluation : (10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3) = -4 :=
by
  sorry

end arithmetic_evaluation_l1282_128252


namespace simplify_product_of_fractions_l1282_128267

theorem simplify_product_of_fractions :
  8 * (15 / 4) * (-28 / 45) = -56 / 3 := by
  sorry

end simplify_product_of_fractions_l1282_128267


namespace maximum_area_rectangular_backyard_l1282_128244

theorem maximum_area_rectangular_backyard (x : ℕ) (y : ℕ) (h_perimeter : 2 * (x + y) = 100) : 
  x * y ≤ 625 :=
by
  sorry

end maximum_area_rectangular_backyard_l1282_128244


namespace f_at_2_f_pos_solution_set_l1282_128261

variable (a : ℝ)

def f (x : ℝ) : ℝ := x^2 - (3 - a) * x + 2 * (1 - a)

-- Question (I)
theorem f_at_2 : f a 2 = 0 := by sorry

-- Question (II)
theorem f_pos_solution_set :
  (∀ x, (a < -1 → (f a x > 0 ↔ (x < 2 ∨ 1 - a < x))) ∧
       (a = -1 → ¬(f a x > 0)) ∧
       (a > -1 → (f a x > 0 ↔ (1 - a < x ∧ x < 2)))) := 
by sorry

end f_at_2_f_pos_solution_set_l1282_128261


namespace sum_of_squares_bounds_l1282_128216

theorem sum_of_squares_bounds (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 10) : 
  (x^2 + y^2 ≤ 100) ∧ (x^2 + y^2 ≥ 50) :=
by 
  sorry

end sum_of_squares_bounds_l1282_128216


namespace sufficient_but_not_necessary_condition_l1282_128265

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x > 1 → x^2 > 1) ∧ ¬(x^2 > 1 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l1282_128265


namespace num_trucks_washed_l1282_128276

theorem num_trucks_washed (total_revenue cars_revenue suvs_revenue truck_charge : ℕ) 
  (h_total : total_revenue = 100)
  (h_cars : cars_revenue = 7 * 5)
  (h_suvs : suvs_revenue = 5 * 7)
  (h_truck_charge : truck_charge = 6) : 
  ∃ T : ℕ, (total_revenue - suvs_revenue - cars_revenue) / truck_charge = T := 
by {
  use 5,
  sorry
}

end num_trucks_washed_l1282_128276


namespace scientific_notation_11580000_l1282_128258

theorem scientific_notation_11580000 :
  (11580000 : ℝ) = 1.158 * 10^7 :=
sorry

end scientific_notation_11580000_l1282_128258
