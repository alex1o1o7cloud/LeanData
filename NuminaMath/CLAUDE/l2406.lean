import Mathlib

namespace function_value_2007_l2406_240634

def is_multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, f (x + y) = f x * f y

theorem function_value_2007 (f : ℕ+ → ℕ+) 
  (h_mult : is_multiplicative f) (h_base : f 1 = 2) : 
  f 2007 = 2^2007 := by
  sorry

end function_value_2007_l2406_240634


namespace unique_five_digit_number_l2406_240666

/-- A five-digit number is a natural number between 10000 and 99999 inclusive. -/
def FiveDigitNumber (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- The property that defines our target number. -/
def SatisfiesCondition (x : ℕ) : Prop :=
  FiveDigitNumber x ∧ 7 * 10^5 + x = 5 * (10 * x + 7)

theorem unique_five_digit_number : 
  ∃! x : ℕ, SatisfiesCondition x ∧ x = 14285 :=
sorry

end unique_five_digit_number_l2406_240666


namespace min_value_reciprocal_sum_l2406_240662

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 1) :
  (1/a + 1/b) ≥ 4*Real.sqrt 3 + 8 :=
sorry

end min_value_reciprocal_sum_l2406_240662


namespace sufficient_but_not_necessary_l2406_240610

theorem sufficient_but_not_necessary (a : ℝ) (h : a > 0) : 
  (a ≥ 2 → a ≥ 1) ∧ ¬(a ≥ 1 → a ≥ 2) :=
by sorry

end sufficient_but_not_necessary_l2406_240610


namespace min_value_of_z_l2406_240609

-- Define the variables and the objective function
variables (x y : ℝ)
def z (x y : ℝ) : ℝ := 2 * x + y

-- State the theorem
theorem min_value_of_z (hx : x ≥ 1) (hxy : x + y ≤ 3) (hxy2 : x - 2 * y - 3 ≤ 0) :
  ∃ (x₀ y₀ : ℝ), x₀ ≥ 1 ∧ x₀ + y₀ ≤ 3 ∧ x₀ - 2 * y₀ - 3 ≤ 0 ∧
  ∀ (x y : ℝ), x ≥ 1 → x + y ≤ 3 → x - 2 * y - 3 ≤ 0 → z x₀ y₀ ≤ z x y ∧ z x₀ y₀ = 1 :=
by sorry

end min_value_of_z_l2406_240609


namespace normal_curve_properties_l2406_240670

/- Normal distribution density function -/
noncomputable def normal_pdf (μ σ : ℝ) (x : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(1/2) * ((x - μ) / σ)^2)

theorem normal_curve_properties (μ σ : ℝ) (h : σ > 0) :
  /- 1. Symmetry about x = μ -/
  (∀ x : ℝ, normal_pdf μ σ (μ + x) = normal_pdf μ σ (μ - x)) ∧
  /- 2. Always positive -/
  (∀ x : ℝ, normal_pdf μ σ x > 0) ∧
  /- 3. Global maximum at x = μ -/
  (∀ x : ℝ, normal_pdf μ σ x ≤ normal_pdf μ σ μ) ∧
  /- 4. Axis of symmetry is x = μ -/
  (∀ x : ℝ, normal_pdf μ σ (μ + x) = normal_pdf μ σ (μ - x)) ∧
  /- 5. σ determines the spread -/
  (∀ σ₁ σ₂ : ℝ, σ₁ ≠ σ₂ → ∃ x : ℝ, normal_pdf μ σ₁ x ≠ normal_pdf μ σ₂ x) ∧
  /- 6. Larger σ, flatter and wider curve -/
  (∀ σ₁ σ₂ : ℝ, σ₁ > σ₂ → ∃ x : ℝ, x ≠ μ ∧ normal_pdf μ σ₁ x > normal_pdf μ σ₂ x) ∧
  /- 7. Smaller σ, taller and slimmer curve -/
  (∀ σ₁ σ₂ : ℝ, σ₁ < σ₂ → normal_pdf μ σ₁ μ > normal_pdf μ σ₂ μ) :=
by sorry

end normal_curve_properties_l2406_240670


namespace pet_shop_dogs_l2406_240637

/-- Given a pet shop with dogs, bunnies, and birds in the ratio 3:9:11,
    and a total of 816 animals, prove that there are 105 dogs. -/
theorem pet_shop_dogs (total : ℕ) (h_total : total = 816) :
  let ratio_sum := 3 + 9 + 11
  let part_size := total / ratio_sum
  let dogs := 3 * part_size
  dogs = 105 := by
  sorry

#check pet_shop_dogs

end pet_shop_dogs_l2406_240637


namespace dilation_transforms_line_l2406_240622

-- Define the original line
def original_line (x y : ℝ) : Prop := x + y = 1

-- Define the transformed line
def transformed_line (x y : ℝ) : Prop := 2*x + 3*y = 6

-- Define the dilation transformation
def dilation (x y : ℝ) : ℝ × ℝ := (2*x, 3*y)

-- Theorem statement
theorem dilation_transforms_line :
  ∀ x y : ℝ, original_line x y → transformed_line (dilation x y).1 (dilation x y).2 := by
  sorry

end dilation_transforms_line_l2406_240622


namespace wheel_probabilities_l2406_240690

theorem wheel_probabilities :
  ∀ (p_C p_D : ℚ),
    (1 : ℚ)/3 + (1 : ℚ)/4 + p_C + p_D = 1 →
    p_C = 2 * p_D →
    p_C = (5 : ℚ)/18 ∧ p_D = (5 : ℚ)/36 :=
by
  sorry

end wheel_probabilities_l2406_240690


namespace simplify_and_evaluate_evaluate_at_two_l2406_240657

theorem simplify_and_evaluate (a : ℝ) (h : a ≠ 1) :
  (1 - 2 / (a + 1)) / ((a^2 - 2*a + 1) / (a + 1)) = 1 / (a - 1) :=
sorry

theorem evaluate_at_two :
  (1 - 2 / (2 + 1)) / ((2^2 - 2*2 + 1) / (2 + 1)) = 1 :=
sorry

end simplify_and_evaluate_evaluate_at_two_l2406_240657


namespace unique_root_in_interval_l2406_240635

open Complex

theorem unique_root_in_interval : ∃! x : ℝ, 0 ≤ x ∧ x < 2 * π ∧
  2 + exp (I * x) - 2 * exp (2 * I * x) + exp (3 * I * x) = 0 := by
  sorry

end unique_root_in_interval_l2406_240635


namespace parabola_vertex_l2406_240687

def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 4*y + 2*x + 8 = 0

def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x' y' : ℝ), eq x' y' → (x' - x)^2 + (y' - y)^2 ≥ 0

theorem parabola_vertex :
  is_vertex (-2) 2 parabola_equation :=
sorry

end parabola_vertex_l2406_240687


namespace diamond_three_eight_l2406_240649

-- Define the ◇ operation
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y + 2

-- Theorem statement
theorem diamond_three_eight : diamond 3 8 = 62 := by
  sorry

end diamond_three_eight_l2406_240649


namespace greatest_multiple_24_unique_digits_remainder_l2406_240621

/-- 
M is the greatest integer multiple of 24 with no two digits being the same.
-/
def M : ℕ := sorry

/-- 
A function that checks if a natural number has all unique digits.
-/
def has_unique_digits (n : ℕ) : Prop := sorry

theorem greatest_multiple_24_unique_digits_remainder (h1 : M % 24 = 0) 
  (h2 : has_unique_digits M) 
  (h3 : ∀ k : ℕ, k > M → k % 24 = 0 → ¬(has_unique_digits k)) : 
  M % 1000 = 720 := by sorry

end greatest_multiple_24_unique_digits_remainder_l2406_240621


namespace sum_congruence_l2406_240654

theorem sum_congruence : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := by
  sorry

end sum_congruence_l2406_240654


namespace box_side_area_l2406_240684

theorem box_side_area (L W H : ℕ) : 
  L * H = (L * W) / 2 →  -- front area is half of top area
  L * W = (3/2) * (H * W) →  -- top area is 1.5 times side area
  3 * H = 2 * L →  -- length to height ratio is 3:2
  L * W * H = 3000 →  -- volume is 3000
  H * W = 200 :=  -- side area is 200
by sorry

end box_side_area_l2406_240684


namespace ceiling_of_negative_three_point_seven_l2406_240641

theorem ceiling_of_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_of_negative_three_point_seven_l2406_240641


namespace multiply_add_theorem_l2406_240633

theorem multiply_add_theorem : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end multiply_add_theorem_l2406_240633


namespace equation_has_two_distinct_real_roots_l2406_240693

-- Define the custom operation ⊗
def otimes (a b : ℝ) : ℝ := b^2 - a*b

-- Theorem statement
theorem equation_has_two_distinct_real_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ otimes (m - 2) x₁ = m ∧ otimes (m - 2) x₂ = m :=
by sorry

end equation_has_two_distinct_real_roots_l2406_240693


namespace hyperbola_eccentricity_l2406_240664

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : ∃ (M N Q : ℝ × ℝ),
  let F1 := (-c, 0)
  let F2 := (c, 0)
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  M ∈ C ∧ N ∈ C ∧
  (N.1 - M.1) * 2 * c = 0 ∧
  (N.2 - M.2) * (F2.1 - F1.1) = 0 ∧
  (F2.1 - F1.1) = 4 * Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2) ∧
  Q ∈ C ∧
  (Q.1 - F1.1)^2 + (Q.2 - F1.2)^2 = (N.1 - Q.1)^2 + (N.2 - Q.2)^2 →
  c^2 / a^2 = 6 :=
by sorry

end hyperbola_eccentricity_l2406_240664


namespace tan_sin_inequality_l2406_240667

open Real

theorem tan_sin_inequality (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (tan x - x) / (x - sin x) > 2 ∧
  ∃ (a : ℝ), a = 3 ∧ ∀ b > a, ∃ y ∈ Set.Ioo 0 (π/2), tan y + 2 * sin y - b * y ≤ 0 :=
by sorry

end tan_sin_inequality_l2406_240667


namespace existence_of_a_and_b_l2406_240627

/-- The number of positive divisors of a natural number -/
noncomputable def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Main theorem -/
theorem existence_of_a_and_b (k l c : ℕ+) :
  ∃ (a b : ℕ+),
    (b - a = c * Nat.gcd a b) ∧
    (tau a * tau (b / Nat.gcd a b) * l = tau b * tau (a / Nat.gcd a b) * k) := by
  sorry

end existence_of_a_and_b_l2406_240627


namespace fraction_difference_l2406_240604

theorem fraction_difference (n : ℝ) : 
  let simplified := (n * (n + 3)) / (n^2 + 3*n + 1)
  (n^2 + 3*n + 1) - (n^2 + 3*n) = 1 := by
sorry

end fraction_difference_l2406_240604


namespace arithmetic_geometric_mean_problem_l2406_240611

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) : 
  x^2 + y^2 = 1400 := by
sorry

end arithmetic_geometric_mean_problem_l2406_240611


namespace jeremy_tylenol_duration_l2406_240631

/-- Calculates the duration in days for which Jeremy takes Tylenol -/
def tylenol_duration (dose_mg : ℕ) (dose_interval_hours : ℕ) (total_pills : ℕ) (mg_per_pill : ℕ) : ℕ :=
  let total_mg := total_pills * mg_per_pill
  let total_doses := total_mg / dose_mg
  let total_hours := total_doses * dose_interval_hours
  total_hours / 24

/-- Theorem stating that Jeremy takes Tylenol for 14 days -/
theorem jeremy_tylenol_duration :
  tylenol_duration 1000 6 112 500 = 14 := by
  sorry

end jeremy_tylenol_duration_l2406_240631


namespace find_x_and_y_l2406_240697

theorem find_x_and_y :
  ∀ x y : ℝ,
  x > y →
  x + y = 55 →
  x - y = 15 →
  x = 35 ∧ y = 20 := by
sorry

end find_x_and_y_l2406_240697


namespace watch_payment_l2406_240669

theorem watch_payment (original_price : ℚ) (discount_rate : ℚ) (dime_value : ℚ) (quarter_value : ℚ) :
  original_price = 15 →
  discount_rate = 1/5 →
  dime_value = 1/10 →
  quarter_value = 1/4 →
  ∃ (num_dimes num_quarters : ℕ),
    (num_dimes : ℚ) = 2 * (num_quarters : ℚ) ∧
    (original_price * (1 - discount_rate) = dime_value * num_dimes + quarter_value * num_quarters) ∧
    num_dimes = 52 :=
by sorry

end watch_payment_l2406_240669


namespace discount_difference_l2406_240644

/-- The cover price of the book in cents -/
def cover_price : ℕ := 3000

/-- The percentage discount as a fraction -/
def percent_discount : ℚ := 1/4

/-- The fixed discount in cents -/
def fixed_discount : ℕ := 500

/-- Applies the percentage discount followed by the fixed discount -/
def percent_then_fixed (price : ℕ) : ℚ :=
  (price : ℚ) * (1 - percent_discount) - fixed_discount

/-- Applies the fixed discount followed by the percentage discount -/
def fixed_then_percent (price : ℕ) : ℚ :=
  ((price : ℚ) - fixed_discount) * (1 - percent_discount)

theorem discount_difference :
  fixed_then_percent cover_price - percent_then_fixed cover_price = 125 := by
  sorry

end discount_difference_l2406_240644


namespace equation1_solutions_equation2_solutions_l2406_240626

-- Define the equations
def equation1 (x : ℝ) : Prop := (x + 1)^2 = 4
def equation2 (x : ℝ) : Prop := x^2 + 4*x - 5 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ equation1 x1 ∧ equation1 x2 ∧ 
  (∀ x : ℝ, equation1 x → x = x1 ∨ x = x2) ∧
  x1 = 1 ∧ x2 = -3 :=
sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ equation2 x1 ∧ equation2 x2 ∧ 
  (∀ x : ℝ, equation2 x → x = x1 ∨ x = x2) ∧
  x1 = -5 ∧ x2 = 1 :=
sorry

end equation1_solutions_equation2_solutions_l2406_240626


namespace count_integers_with_digit_sum_two_is_correct_l2406_240665

/-- The count of positive integers between 10^7 and 10^8 whose sum of digits is equal to 2 -/
def count_integers_with_digit_sum_two : ℕ :=
  let lower_bound := 10^7
  let upper_bound := 10^8 - 1
  let digit_sum := 2
  28  -- The actual count, to be proven

/-- Theorem stating that the count of positive integers between 10^7 and 10^8
    whose sum of digits is equal to 2 is correct -/
theorem count_integers_with_digit_sum_two_is_correct :
  count_integers_with_digit_sum_two = 
    (Finset.filter (fun n => (Finset.sum (Finset.range 8) (fun i => (n / 10^i) % 10) = 2))
      (Finset.range (10^8 - 10^7))).card :=
by
  sorry

#eval count_integers_with_digit_sum_two

end count_integers_with_digit_sum_two_is_correct_l2406_240665


namespace sqrt_equation_solutions_l2406_240696

theorem sqrt_equation_solutions :
  {x : ℝ | Real.sqrt (4 * x - 3) + 10 / Real.sqrt (4 * x - 3) = 7} = {7/4, 7} := by
  sorry

end sqrt_equation_solutions_l2406_240696


namespace simplify_expression_l2406_240672

theorem simplify_expression : (256 : ℝ) ^ (1/4) * (144 : ℝ) ^ (1/2) = 48 := by
  sorry

end simplify_expression_l2406_240672


namespace unique_solution_quadratic_l2406_240643

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (k + 2) * x^2 - 2 * (k - 1) * x + k + 1 = 0) ↔ (k = -1/5 ∨ k = -2) := by
  sorry

end unique_solution_quadratic_l2406_240643


namespace min_data_for_plan_y_effectiveness_l2406_240692

/-- Represents the cost in cents for Plan X given the data usage in MB -/
def cost_plan_x (data : ℕ) : ℕ := 20 * data

/-- Represents the cost in cents for Plan Y given the data usage in MB -/
def cost_plan_y (data : ℕ) : ℕ := 1500 + 10 * data

/-- Proves that 151 MB is the minimum amount of data that makes Plan Y more cost-effective -/
theorem min_data_for_plan_y_effectiveness : 
  ∀ d : ℕ, d ≥ 151 ↔ cost_plan_y d < cost_plan_x d :=
by sorry

end min_data_for_plan_y_effectiveness_l2406_240692


namespace ship_passengers_l2406_240603

theorem ship_passengers (total : ℝ) (round_trip_with_car : ℝ) 
  (h1 : 0 < total) 
  (h2 : 0 ≤ round_trip_with_car) 
  (h3 : round_trip_with_car ≤ total) 
  (h4 : round_trip_with_car / total = 0.2 * (round_trip_with_car / 0.2) / total) : 
  round_trip_with_car / 0.2 = total := by
sorry

end ship_passengers_l2406_240603


namespace solution_set_part1_a_range_part2_l2406_240602

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 + |2*x - 4| + a

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | (f x (-3)) > x^2 + |x|} = {x : ℝ | x < 1/3 ∨ x > 7} := by sorry

-- Part 2
theorem a_range_part2 :
  (∀ x : ℝ, f x a ≥ 0) → a ≥ -3 := by sorry

end solution_set_part1_a_range_part2_l2406_240602


namespace first_movie_duration_l2406_240608

/-- Represents the duration of a movie marathon with three movies --/
structure MovieMarathon where
  first_movie : ℝ
  second_movie : ℝ
  third_movie : ℝ

/-- Defines the conditions of the movie marathon --/
def valid_marathon (m : MovieMarathon) : Prop :=
  m.second_movie = 1.5 * m.first_movie ∧
  m.third_movie = m.first_movie + m.second_movie - 1 ∧
  m.first_movie + m.second_movie + m.third_movie = 9

theorem first_movie_duration :
  ∀ m : MovieMarathon, valid_marathon m → m.first_movie = 2 := by
  sorry

end first_movie_duration_l2406_240608


namespace percent_relation_l2406_240614

theorem percent_relation (x y z w : ℝ) 
  (hx : x = 1.2 * y) 
  (hy : y = 0.7 * z) 
  (hw : w = 1.5 * z) : 
  x / w = 0.56 := by
  sorry

end percent_relation_l2406_240614


namespace ellipse_dot_product_range_l2406_240605

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the left focus -/
def F₁ : ℝ × ℝ := (-1, 0)

/-- Definition of the right focus -/
def F₂ : ℝ × ℝ := (1, 0)

/-- Definition of a point being on a line through F₂ -/
def is_on_line_through_F₂ (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k * (x - F₂.1)

/-- The dot product of vectors F₁M and F₁N -/
def F₁M_dot_F₁N (M N : ℝ × ℝ) : ℝ :=
  (M.1 - F₁.1) * (N.1 - F₁.1) + (M.2 - F₁.2) * (N.2 - F₁.2)

/-- The main theorem -/
theorem ellipse_dot_product_range :
  ∀ M N : ℝ × ℝ,
  is_on_ellipse M.1 M.2 →
  is_on_ellipse N.1 N.2 →
  is_on_line_through_F₂ M.1 M.2 →
  is_on_line_through_F₂ N.1 N.2 →
  M ≠ N →
  -1 ≤ F₁M_dot_F₁N M N ∧ F₁M_dot_F₁N M N ≤ 7/2 :=
sorry

end ellipse_dot_product_range_l2406_240605


namespace express_y_in_terms_of_x_l2406_240615

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := by
  sorry

end express_y_in_terms_of_x_l2406_240615


namespace gcd_459_357_l2406_240616

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l2406_240616


namespace new_players_count_l2406_240691

theorem new_players_count (returning_players : ℕ) (groups : ℕ) (players_per_group : ℕ) :
  returning_players = 6 →
  groups = 9 →
  players_per_group = 6 →
  groups * players_per_group - returning_players = 48 := by
sorry

end new_players_count_l2406_240691


namespace medical_team_composition_l2406_240636

theorem medical_team_composition (total : ℕ) 
  (female_nurses male_nurses female_doctors male_doctors : ℕ) :
  total = 13 →
  female_nurses + male_nurses + female_doctors + male_doctors = total →
  female_nurses + male_nurses ≥ female_doctors + male_doctors →
  male_doctors > female_nurses →
  female_nurses > male_nurses →
  female_doctors ≥ 1 →
  female_nurses = 4 ∧ male_nurses = 3 ∧ female_doctors = 1 ∧ male_doctors = 5 :=
by sorry

end medical_team_composition_l2406_240636


namespace bookstore_sales_amount_l2406_240646

theorem bookstore_sales_amount (total_calculators : ℕ) (price1 price2 : ℕ) (quantity1 : ℕ) :
  total_calculators = 85 →
  price1 = 15 →
  price2 = 67 →
  quantity1 = 35 →
  (quantity1 * price1 + (total_calculators - quantity1) * price2 : ℕ) = 3875 := by
  sorry

end bookstore_sales_amount_l2406_240646


namespace nikanor_lost_second_match_l2406_240642

/-- Represents a player in the knock-out table tennis game -/
inductive Player : Type
| Nikanor : Player
| Philemon : Player
| Agathon : Player

/-- Represents the state of the game after each match -/
structure GameState :=
  (matches_played : Nat)
  (nikanor_matches : Nat)
  (philemon_matches : Nat)
  (agathon_matches : Nat)
  (last_loser : Player)

/-- The rules of the knock-out table tennis game -/
def game_rules (state : GameState) : Prop :=
  state.matches_played = (state.nikanor_matches + state.philemon_matches + state.agathon_matches) / 2 ∧
  state.nikanor_matches + state.philemon_matches + state.agathon_matches = state.matches_played * 2 ∧
  state.nikanor_matches ≤ state.matches_played ∧
  state.philemon_matches ≤ state.matches_played ∧
  state.agathon_matches ≤ state.matches_played

/-- The final state of the game -/
def final_state : GameState :=
  { matches_played := 21
  , nikanor_matches := 10
  , philemon_matches := 15
  , agathon_matches := 17
  , last_loser := Player.Nikanor }

/-- Theorem stating that Nikanor lost the second match -/
theorem nikanor_lost_second_match :
  game_rules final_state →
  final_state.last_loser = Player.Nikanor :=
by sorry

end nikanor_lost_second_match_l2406_240642


namespace absolute_value_inequality_l2406_240650

theorem absolute_value_inequality (x y z : ℝ) :
  (|x + y + z| + |x*y + y*z + z*x| + |x*y*z| ≤ 1) →
  (max (|x|) (max (|y|) (|z|)) ≤ 1) := by
  sorry

end absolute_value_inequality_l2406_240650


namespace largest_percentage_increase_l2406_240679

def students : Fin 7 → ℕ
  | 0 => 80  -- 2010
  | 1 => 85  -- 2011
  | 2 => 88  -- 2012
  | 3 => 90  -- 2013
  | 4 => 95  -- 2014
  | 5 => 100 -- 2015
  | 6 => 120 -- 2016

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 6 := 5 -- Represents 2015 to 2016

theorem largest_percentage_increase :
  ∀ i : Fin 6, percentageIncrease (students i) (students (i.succ)) ≤ 
    percentageIncrease (students largestIncreaseYears) (students (largestIncreaseYears.succ)) := by
  sorry

end largest_percentage_increase_l2406_240679


namespace M_divisible_by_51_l2406_240699

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Definition of the function that concatenates numbers from 1 to n
  sorry

theorem M_divisible_by_51 :
  ∃ M : ℕ, M = concatenate_numbers 50 ∧ M % 51 = 0 :=
sorry

end M_divisible_by_51_l2406_240699


namespace factorization_xy_plus_3x_l2406_240619

theorem factorization_xy_plus_3x (x y : ℝ) : x * y + 3 * x = x * (y + 3) := by
  sorry

end factorization_xy_plus_3x_l2406_240619


namespace average_daily_high_temp_l2406_240660

def daily_highs : List ℝ := [51, 63, 59, 56, 47, 64, 52]

theorem average_daily_high_temp : 
  (daily_highs.sum / daily_highs.length : ℝ) = 56 := by
  sorry

end average_daily_high_temp_l2406_240660


namespace distribute_5_3_l2406_240613

/-- The number of ways to distribute n indistinguishable objects into k distinct containers,
    with each container containing at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

theorem distribute_5_3 : distribute 5 3 = 150 := by
  sorry

end distribute_5_3_l2406_240613


namespace floor_plus_self_eq_seventeen_fourths_l2406_240612

theorem floor_plus_self_eq_seventeen_fourths (x : ℝ) : 
  ⌊x⌋ + x = 17/4 → x = 9/4 := by
  sorry

end floor_plus_self_eq_seventeen_fourths_l2406_240612


namespace inequality_equivalence_l2406_240694

theorem inequality_equivalence (x : ℝ) :
  |x + 1| + 2 * |x - 1| < 3 * x + 5 ↔ x > -1/2 := by sorry

end inequality_equivalence_l2406_240694


namespace ratio_sum_problem_l2406_240661

theorem ratio_sum_problem (w x y : ℝ) (hw_x : w / x = 1 / 6) (hw_y : w / y = 1 / 5) :
  (x + y) / y = 11 / 5 := by
  sorry

end ratio_sum_problem_l2406_240661


namespace partners_capital_time_l2406_240618

/-- A proof that under given business conditions, A's capital was used for 15 months -/
theorem partners_capital_time (C P : ℝ) : 
  C > 0 → P > 0 →
  let a_capital := C / 4
  let b_capital := 3 * C / 4
  let b_time := 10
  let b_profit := 2 * P / 3
  let a_profit := P / 3
  ∃ (a_time : ℝ),
    a_time * a_capital / (b_time * b_capital) = a_profit / b_profit ∧
    a_time = 15 :=
by sorry

end partners_capital_time_l2406_240618


namespace factor_sum_l2406_240647

theorem factor_sum (P Q : ℚ) : 
  (∃ b c : ℚ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + P*X^3 + Q*X^2 + 45*X - 14) →
  P + Q = 260/7 := by
sorry

end factor_sum_l2406_240647


namespace smallest_fraction_between_l2406_240676

theorem smallest_fraction_between (p q : ℕ+) : 
  (3 : ℚ) / 5 < (p : ℚ) / q ∧ 
  (p : ℚ) / q < (2 : ℚ) / 3 ∧ 
  (∀ (p' q' : ℕ+), (3 : ℚ) / 5 < (p' : ℚ) / q' ∧ (p' : ℚ) / q' < (2 : ℚ) / 3 → q ≤ q') →
  q - p = 3 := by
sorry

end smallest_fraction_between_l2406_240676


namespace f_inequality_solution_f_plus_abs_inequality_l2406_240659

def f (x : ℝ) := |3 * x + 1| - |x - 4|

theorem f_inequality_solution (x : ℝ) :
  f x < 0 ↔ -5/2 < x ∧ x < 3/4 := by sorry

theorem f_plus_abs_inequality (m : ℝ) :
  (∀ x : ℝ, f x + 4 * |x - 4| > m) ↔ m < 15 := by sorry

end f_inequality_solution_f_plus_abs_inequality_l2406_240659


namespace line_equations_l2406_240600

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in 2D space using slope-intercept form
structure Line2D where
  slope : ℝ
  intercept : ℝ

def isParallel (l1 l2 : Line2D) : Prop :=
  l1.slope = l2.slope

def isPerpendicular (l1 l2 : Line2D) : Prop :=
  l1.slope * l2.slope = -1

def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem line_equations (P : Point2D) (l : Line2D) (given_line : Line2D) :
  P.x = -2 ∧ P.y = 1 ∧ given_line.slope = 1/2 ∧ given_line.intercept = -1/2 →
  (isParallel l given_line → l.slope = 1/2 ∧ l.intercept = 3/2) ∧
  (isPerpendicular l given_line → l.slope = -2 ∧ l.intercept = -5/2) :=
sorry

end line_equations_l2406_240600


namespace inequality_solution_set_l2406_240695

theorem inequality_solution_set :
  {x : ℝ | x - 3 / x > 2} = {x : ℝ | -1 < x ∧ x < 0 ∨ x > 3} := by
  sorry

end inequality_solution_set_l2406_240695


namespace arithmetic_sequence_terms_l2406_240651

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  commonDiff : ℕ
  numTerms : ℕ
  sumOddTerms : ℕ
  sumEvenTerms : ℕ
  isEven : Even numTerms
  diffIs2 : commonDiff = 2

/-- Theorem stating the number of terms in the sequence with given conditions -/
theorem arithmetic_sequence_terms
  (seq : ArithmeticSequence)
  (h1 : seq.sumOddTerms = 15)
  (h2 : seq.sumEvenTerms = 35) :
  seq.numTerms = 20 := by
  sorry

#check arithmetic_sequence_terms

end arithmetic_sequence_terms_l2406_240651


namespace tunnel_length_l2406_240652

theorem tunnel_length : 
  ∀ (initial_speed : ℝ),
  initial_speed > 0 →
  (400 + 8600) / initial_speed = 10 →
  (400 + 8600) / (initial_speed + 0.1 * 60) = 9 :=
by sorry

end tunnel_length_l2406_240652


namespace complement_union_theorem_l2406_240629

def U : Set Nat := {0, 2, 4, 6, 8, 10}
def A : Set Nat := {2, 4, 6}
def B : Set Nat := {1}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 1, 8, 10} := by sorry

end complement_union_theorem_l2406_240629


namespace square_difference_l2406_240678

theorem square_difference (n : ℕ) (h : (n + 1)^2 = n^2 + 2*n + 1) :
  (n - 1)^2 = n^2 - (2*n - 1) :=
by sorry

end square_difference_l2406_240678


namespace solution_set_f_geq_1_max_value_f_minus_quadratic_range_of_m_l2406_240617

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem for the solution set of f(x) ≥ 1
theorem solution_set_f_geq_1 : 
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 1} := by sorry

-- Theorem for the maximum value of f(x) - x^2 + x
theorem max_value_f_minus_quadratic :
  ∃ (x : ℝ), ∀ (y : ℝ), f y - y^2 + y ≤ f x - x^2 + x ∧ f x - x^2 + x = 5/4 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ (x : ℝ), f x ≥ x^2 - x + m) ↔ m ≤ 5/4 := by sorry

end solution_set_f_geq_1_max_value_f_minus_quadratic_range_of_m_l2406_240617


namespace smallest_multiple_of_6_and_15_l2406_240688

theorem smallest_multiple_of_6_and_15 (c : ℕ) :
  (c > 0 ∧ 6 ∣ c ∧ 15 ∣ c) → c ≥ 30 :=
by
  sorry

end smallest_multiple_of_6_and_15_l2406_240688


namespace sophia_ate_five_percent_l2406_240685

-- Define the percentages eaten by each person
def caden_percent : ℝ := 20
def zoe_percent : ℝ := caden_percent + 0.5 * caden_percent
def noah_percent : ℝ := zoe_percent + 0.5 * zoe_percent
def sophia_percent : ℝ := 100 - (caden_percent + zoe_percent + noah_percent)

-- Theorem statement
theorem sophia_ate_five_percent :
  sophia_percent = 5 := by
  sorry

end sophia_ate_five_percent_l2406_240685


namespace problem_statement_l2406_240689

theorem problem_statement :
  (∀ x : ℝ, x^4 - x^3 - x + 1 ≥ 0) ∧
  (1 + 1 + 1 = 3 ∧ 1^3 + 1^3 + 1^3 = 1^4 + 1^4 + 1^4) := by
  sorry

end problem_statement_l2406_240689


namespace speed_upstream_l2406_240668

theorem speed_upstream (boat_speed : ℝ) (current_speed : ℝ) (h1 : boat_speed = 60) (h2 : current_speed = 17) :
  boat_speed - current_speed = 43 := by
  sorry

end speed_upstream_l2406_240668


namespace ticket_sales_total_l2406_240673

/-- Calculates the total amount collected from ticket sales -/
def totalAmountCollected (adultPrice studentPrice : ℚ) (totalTickets studentTickets : ℕ) : ℚ :=
  let adultTickets := totalTickets - studentTickets
  adultPrice * adultTickets + studentPrice * studentTickets

/-- Proves that the total amount collected from ticket sales is 222.50 -/
theorem ticket_sales_total : 
  totalAmountCollected 4 (5/2) 59 9 = 445/2 := by
  sorry

end ticket_sales_total_l2406_240673


namespace negative_square_cubed_l2406_240677

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l2406_240677


namespace divisible_by_18_sqrt_between_30_and_30_5_l2406_240630

theorem divisible_by_18_sqrt_between_30_and_30_5 :
  ∃ (n : ℕ), n > 0 ∧ n % 18 = 0 ∧ 30 < Real.sqrt n ∧ Real.sqrt n < 30.5 ∧
  (n = 900 ∨ n = 918) := by
  sorry

end divisible_by_18_sqrt_between_30_and_30_5_l2406_240630


namespace soccer_stars_games_l2406_240607

theorem soccer_stars_games (wins losses draws : ℕ) 
  (h1 : wins = 14)
  (h2 : losses = 2)
  (h3 : 3 * wins + draws = 46) :
  wins + losses + draws = 20 := by
sorry

end soccer_stars_games_l2406_240607


namespace solve_equation_l2406_240645

theorem solve_equation (y : ℝ) : (4 / 7) * (1 / 5) * y - 2 = 10 → y = 105 := by
  sorry

end solve_equation_l2406_240645


namespace equal_distribution_l2406_240686

def total_amount : ℕ := 42900
def num_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem equal_distribution :
  total_amount / num_persons = amount_per_person := by
  sorry

end equal_distribution_l2406_240686


namespace area_of_triangle_ABF_l2406_240656

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define the focus F
def F : ℝ × ℝ := (-2, 0)

-- Define the intersection line
def intersection_line (x : ℝ) : Prop := x = 2

-- Define the points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (2, -3)

-- Theorem statement
theorem area_of_triangle_ABF :
  hyperbola A.1 A.2 ∧
  hyperbola B.1 B.2 ∧
  intersection_line A.1 ∧
  intersection_line B.1 →
  (1/2 : ℝ) * |A.2 - B.2| * |A.1 - F.1| = 12 :=
sorry

end area_of_triangle_ABF_l2406_240656


namespace student_number_problem_l2406_240606

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 112 → x = 125 := by
  sorry

end student_number_problem_l2406_240606


namespace exponent_sum_l2406_240628

theorem exponent_sum (m n : ℕ) (h : 2^m * 2^n = 16) : m + n = 4 := by
  sorry

end exponent_sum_l2406_240628


namespace evelyn_skittles_l2406_240683

def skittles_problem (starting_skittles shared_skittles : ℕ) : ℕ :=
  starting_skittles - shared_skittles

theorem evelyn_skittles :
  skittles_problem 76 72 = 4 := by
  sorry

end evelyn_skittles_l2406_240683


namespace car_count_on_river_road_l2406_240655

theorem car_count_on_river_road (buses cars bikes : ℕ) : 
  (3 : ℕ) * cars = 7 * buses →
  (3 : ℕ) * bikes = 10 * buses →
  cars = buses + 90 →
  bikes = buses + 140 →
  cars = 150 := by
sorry

end car_count_on_river_road_l2406_240655


namespace intersection_on_ellipse_l2406_240620

/-- Ellipse C with given properties -/
structure EllipseC where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The eccentricity of ellipse C is √3/2 -/
axiom eccentricity (C : EllipseC) : (Real.sqrt (C.a^2 - C.b^2)) / C.a = Real.sqrt 3 / 2

/-- A circle centered at the origin with diameter equal to the minor axis of C
    is tangent to the line x - y + 2 = 0 -/
axiom circle_tangent (C : EllipseC) : C.b = Real.sqrt 2

/-- Point on ellipse C -/
def on_ellipse (C : EllipseC) (x y : ℝ) : Prop :=
  x^2 / C.a^2 + y^2 / C.b^2 = 1

/-- Theorem: If M and N are symmetric points on C, and T is the intersection of PM and QN,
    then T lies on the ellipse C -/
theorem intersection_on_ellipse (C : EllipseC) (x₀ y₀ x y : ℝ) :
  on_ellipse C x₀ y₀ →
  on_ellipse C (-x₀) y₀ →
  x = 2 * x₀ * (y - 1) →
  y = (3 * y₀ - 4) / (2 * y₀ - 3) →
  on_ellipse C x y := by
  sorry

end intersection_on_ellipse_l2406_240620


namespace max_sum_distance_from_line_max_sum_distance_from_line_tight_l2406_240674

theorem max_sum_distance_from_line (x₁ y₁ x₂ y₂ : ℝ) :
  x₁^2 + y₁^2 = 1 →
  x₂^2 + y₂^2 = 1 →
  (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 →
  |x₁ + y₁ - 1| + |x₂ + y₂ - 1| ≤ 2 + Real.sqrt 6 :=
by sorry

theorem max_sum_distance_from_line_tight :
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    x₁^2 + y₁^2 = 1 ∧
    x₂^2 + y₂^2 = 1 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 1 ∧
    |x₁ + y₁ - 1| + |x₂ + y₂ - 1| = 2 + Real.sqrt 6 :=
by sorry

end max_sum_distance_from_line_max_sum_distance_from_line_tight_l2406_240674


namespace no_divisible_by_99_ab32_l2406_240653

theorem no_divisible_by_99_ab32 : ∀ a b : ℕ, 
  0 ≤ a ∧ a ≤ 9 → 0 ≤ b ∧ b ≤ 9 → 
  ¬(∃ k : ℕ, 1000 * a + 100 * b + 32 = 99 * k) := by
sorry

end no_divisible_by_99_ab32_l2406_240653


namespace rope_cutting_problem_l2406_240675

theorem rope_cutting_problem (initial_length : ℝ) : 
  initial_length / 2 / 2 / 5 = 5 → initial_length = 100 := by
  sorry

end rope_cutting_problem_l2406_240675


namespace consecutive_integers_product_plus_one_is_square_l2406_240682

theorem consecutive_integers_product_plus_one_is_square (n : ℤ) :
  ∃ m : ℤ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m ^ 2 := by
  sorry

end consecutive_integers_product_plus_one_is_square_l2406_240682


namespace soup_donation_theorem_l2406_240640

theorem soup_donation_theorem (shelters cans_per_person total_cans : ℕ) 
  (h1 : shelters = 6)
  (h2 : cans_per_person = 10)
  (h3 : total_cans = 1800) :
  total_cans / (shelters * cans_per_person) = 30 := by
  sorry

end soup_donation_theorem_l2406_240640


namespace fraction_division_l2406_240601

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := by
  sorry

end fraction_division_l2406_240601


namespace edward_earnings_l2406_240681

/-- Calculate Edward's earnings for a week --/
theorem edward_earnings (regular_rate : ℝ) (regular_hours overtime_hours : ℕ) : 
  regular_rate = 7 →
  regular_hours = 40 →
  overtime_hours = 5 →
  let overtime_rate := 2 * regular_rate
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := overtime_rate * overtime_hours
  regular_earnings + overtime_earnings = 350 := by sorry

end edward_earnings_l2406_240681


namespace train_speed_problem_l2406_240639

/-- Proves that given two trains of equal length 37.5 meters, where the faster train travels
    at 46 km/hr and passes the slower train in 27 seconds, the speed of the slower train is 36 km/hr. -/
theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) 
    (h1 : train_length = 37.5)
    (h2 : faster_speed = 46)
    (h3 : passing_time = 27) :
  ∃ slower_speed : ℝ, 
    slower_speed > 0 ∧ 
    slower_speed < faster_speed ∧
    2 * train_length = (faster_speed - slower_speed) * (5/18) * passing_time ∧
    slower_speed = 36 := by
  sorry

end train_speed_problem_l2406_240639


namespace slow_clock_theorem_l2406_240658

/-- Represents a clock with a specific overlap time between its minute and hour hands. -/
structure Clock where
  overlap_time : ℕ  -- Time in minutes between each overlap of minute and hour hands

/-- The number of overlaps in a 24-hour period for any clock -/
def num_overlaps : ℕ := 22

/-- Calculates the length of a 24-hour period for a given clock in minutes -/
def period_length (c : Clock) : ℕ :=
  num_overlaps * c.overlap_time

/-- The length of a standard 24-hour period in minutes -/
def standard_period : ℕ := 24 * 60

/-- Theorem stating that a clock with 66-minute overlaps is 12 minutes slower over 24 hours -/
theorem slow_clock_theorem (c : Clock) (h : c.overlap_time = 66) :
  period_length c - standard_period = 12 := by
  sorry


end slow_clock_theorem_l2406_240658


namespace max_parts_two_planes_l2406_240663

-- Define a plane in 3D space
def Plane3D : Type := Unit

-- Define the possible relationships between two planes
inductive PlaneRelationship
| Parallel
| Intersecting

-- Function to calculate the number of parts based on the relationship
def numParts (rel : PlaneRelationship) : ℕ :=
  match rel with
  | PlaneRelationship.Parallel => 3
  | PlaneRelationship.Intersecting => 4

-- Theorem statement
theorem max_parts_two_planes (p1 p2 : Plane3D) :
  ∃ (rel : PlaneRelationship), ∀ (r : PlaneRelationship), numParts rel ≥ numParts r :=
sorry

end max_parts_two_planes_l2406_240663


namespace complement_intersection_empty_l2406_240648

def S : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_empty :
  (S \ M) ∩ (S \ N) = ∅ := by sorry

end complement_intersection_empty_l2406_240648


namespace transformations_of_g_reflection_about_y_axis_horizontal_stretch_horizontal_shift_l2406_240680

-- Define the original function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the transformed function g
noncomputable def g (x : ℝ) : ℝ := f ((1 - x) / 2)

-- Theorem stating the transformations
theorem transformations_of_g (x : ℝ) :
  g x = f (-(x - 1) / 2) :=
sorry

-- Theorem stating the reflection about y-axis
theorem reflection_about_y_axis (x : ℝ) :
  g (-x + 1) = f x :=
sorry

-- Theorem stating the horizontal stretch
theorem horizontal_stretch (x : ℝ) :
  g (2 * x + 1) = f x :=
sorry

-- Theorem stating the horizontal shift
theorem horizontal_shift (x : ℝ) :
  g (x + 1) = f (x / 2) :=
sorry

end transformations_of_g_reflection_about_y_axis_horizontal_stretch_horizontal_shift_l2406_240680


namespace z_range_l2406_240632

theorem z_range (x y : ℝ) (hx : x ≥ 0) (hxy : y ≥ x) (hsum : 4 * x + 3 * y ≤ 12) :
  let z := (x + 2 * y + 3) / (x + 1)
  2 ≤ z ∧ z ≤ 6 :=
by sorry

end z_range_l2406_240632


namespace tree_planting_problem_l2406_240623

theorem tree_planting_problem (x : ℝ) : 
  (∀ y : ℝ, y = x + 5 → 60 / y = 45 / x) → x = 15 := by
  sorry

end tree_planting_problem_l2406_240623


namespace career_preference_theorem_l2406_240698

/-- Represents the number of degrees in a circle that should be allocated to a career
    preference in a class with a given boy-to-girl ratio and preference ratios. -/
def career_preference_degrees (boy_ratio girl_ratio : ℚ) 
                               (boy_preference girl_preference : ℚ) : ℚ :=
  let total_parts := boy_ratio + girl_ratio
  let boy_parts := (boy_preference * boy_ratio) / total_parts
  let girl_parts := (girl_preference * girl_ratio) / total_parts
  let preference_fraction := (boy_parts + girl_parts) / 1
  preference_fraction * 360

/-- Theorem stating that for a class with a 2:3 ratio of boys to girls, 
    where 1/3 of boys and 2/3 of girls prefer a career, 
    192 degrees should be used to represent this preference in a circle graph. -/
theorem career_preference_theorem : 
  career_preference_degrees 2 3 (1/3) (2/3) = 192 := by
  sorry

end career_preference_theorem_l2406_240698


namespace bread_loaves_from_flour_l2406_240671

/-- Given 5 cups of flour and requiring 2.5 cups per loaf, prove that 2 loaves can be baked. -/
theorem bread_loaves_from_flour (total_flour : ℝ) (flour_per_loaf : ℝ) (h1 : total_flour = 5) (h2 : flour_per_loaf = 2.5) :
  total_flour / flour_per_loaf = 2 := by
sorry

end bread_loaves_from_flour_l2406_240671


namespace exam_thresholds_l2406_240625

theorem exam_thresholds (T : ℝ) 
  (hA : 0.25 * T + 30 = 130) 
  (hB : 0.35 * T - 10 = 130) 
  (hC : 0.40 * T = 160) : 
  (130 : ℝ) = 130 ∧ (160 : ℝ) = 160 := by
  sorry

end exam_thresholds_l2406_240625


namespace g_range_contains_pi_quarters_l2406_240638

open Real

noncomputable def g (x : ℝ) : ℝ := arctan x + arctan ((x - 1) / (x + 1)) + arctan (1 / x)

theorem g_range_contains_pi_quarters :
  ∃ (x : ℝ), g x = π / 4 ∨ g x = 5 * π / 4 :=
sorry

end g_range_contains_pi_quarters_l2406_240638


namespace inverse_variation_solution_l2406_240624

/-- Inverse variation constant -/
def k : ℝ := 9

/-- The relation between x and y -/
def inverse_variation (x y : ℝ) : Prop := x = k / (y ^ 2)

theorem inverse_variation_solution :
  ∀ x y : ℝ,
  inverse_variation x y →
  inverse_variation 1 3 →
  inverse_variation 0.1111111111111111 y →
  y = 9 := by
sorry

end inverse_variation_solution_l2406_240624
