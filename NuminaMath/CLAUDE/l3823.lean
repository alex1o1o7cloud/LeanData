import Mathlib

namespace inequality_proof_l3823_382358

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b + b / c + c / a)^2 ≥ (3 / 2) * ((a + b) / c + (b + c) / a + (c + a) / b) := by
  sorry

end inequality_proof_l3823_382358


namespace factor_implies_c_value_l3823_382370

theorem factor_implies_c_value (c : ℚ) : 
  (∀ x : ℚ, (x - 3) ∣ (c * x^3 - 6 * x^2 - c * x + 10)) → c = 11/6 := by
  sorry

end factor_implies_c_value_l3823_382370


namespace max_min_values_l3823_382399

theorem max_min_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 4 * a + b = 3) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 3 → b - 1 / a ≥ y - 1 / x) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 4 * x + y = 3 → 1 / (3 * a + 1) + 1 / (a + b) ≤ 1 / (3 * x + 1) + 1 / (x + y)) :=
by sorry

end max_min_values_l3823_382399


namespace max_x_plus_z_l3823_382315

theorem max_x_plus_z (x y z t : ℝ) 
  (h1 : x^2 + y^2 = 4)
  (h2 : z^2 + t^2 = 9)
  (h3 : x*t + y*z = 6) :
  x + z ≤ Real.sqrt 13 ∧ ∃ x y z t, x^2 + y^2 = 4 ∧ z^2 + t^2 = 9 ∧ x*t + y*z = 6 ∧ x + z = Real.sqrt 13 := by
  sorry

#check max_x_plus_z

end max_x_plus_z_l3823_382315


namespace cost_price_calculation_l3823_382366

/-- Given a discount, profit percentage, and markup percentage, 
    calculate the cost price of an item. -/
theorem cost_price_calculation 
  (discount : ℝ) 
  (profit_percentage : ℝ) 
  (markup_percentage : ℝ) 
  (h1 : discount = 45)
  (h2 : profit_percentage = 0.20)
  (h3 : markup_percentage = 0.45) :
  ∃ (cost_price : ℝ), 
    cost_price * (1 + markup_percentage) - discount = cost_price * (1 + profit_percentage) ∧ 
    cost_price = 180 := by
  sorry


end cost_price_calculation_l3823_382366


namespace wendy_album_pics_l3823_382352

def pictures_per_album (phone_pics camera_pics num_albums : ℕ) : ℕ :=
  (phone_pics + camera_pics) / num_albums

theorem wendy_album_pics : pictures_per_album 22 2 4 = 6 := by
  sorry

end wendy_album_pics_l3823_382352


namespace percentage_problem_l3823_382374

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.3 * x = 120 := by
  sorry

end percentage_problem_l3823_382374


namespace new_girl_weight_l3823_382348

/-- Proves that the weight of a new girl is 80 kg given the conditions of the problem -/
theorem new_girl_weight (n : ℕ) (initial_weight total_weight : ℝ) :
  n = 20 →
  initial_weight = 40 →
  (total_weight - initial_weight + 80) / n = total_weight / n + 2 →
  80 = total_weight - initial_weight + 40 :=
by sorry

end new_girl_weight_l3823_382348


namespace horner_method_v3_l3823_382312

def f (x : ℤ) (a b : ℤ) : ℤ := x^5 + a*x^4 - b*x^2 + 1

def horner_v3 (a b : ℤ) : ℤ :=
  let x := -1
  let v0 := 1
  let v1 := v0 * x + a
  let v2 := v1 * x + 0
  v2 * x - b

theorem horner_method_v3 :
  horner_v3 47 37 = 9 := by sorry

end horner_method_v3_l3823_382312


namespace cara_catches_47_l3823_382356

/-- The number of animals Martha's cat catches -/
def martha_animals : ℕ := 3 + 7

/-- The number of animals Cara's cat catches -/
def cara_animals : ℕ := 5 * martha_animals - 3

/-- Theorem stating that Cara's cat catches 47 animals -/
theorem cara_catches_47 : cara_animals = 47 := by
  sorry

end cara_catches_47_l3823_382356


namespace sphere_quarter_sphere_radius_l3823_382311

theorem sphere_quarter_sphere_radius (r : ℝ) (h : r = 2 * Real.rpow 4 (1/3)) :
  ∃ R : ℝ, (4/3 * Real.pi * R^3 = 1/3 * Real.pi * r^3) ∧ R = 2 := by
  sorry

end sphere_quarter_sphere_radius_l3823_382311


namespace mark_parking_tickets_l3823_382369

theorem mark_parking_tickets :
  ∀ (mark_speeding mark_parking sarah_speeding sarah_parking : ℕ),
  mark_speeding + mark_parking + sarah_speeding + sarah_parking = 24 →
  mark_parking = 2 * sarah_parking →
  mark_speeding = sarah_speeding →
  sarah_speeding = 6 →
  mark_parking = 8 :=
by
  sorry

end mark_parking_tickets_l3823_382369


namespace empty_set_subset_of_all_l3823_382351

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end empty_set_subset_of_all_l3823_382351


namespace certain_number_proof_l3823_382338

theorem certain_number_proof : ∃ n : ℕ, n * 40 = 173 * 240 ∧ n = 1038 := by
  sorry

end certain_number_proof_l3823_382338


namespace log_equation_solution_l3823_382394

theorem log_equation_solution (m n : ℝ) (b : ℝ) (h : m > 0) (h' : n > 0) :
  Real.log m^2 / Real.log 10 = b - Real.log n^3 / Real.log 10 →
  m = (10^b / n^3)^(1/2) := by
  sorry

end log_equation_solution_l3823_382394


namespace sum_of_integers_l3823_382392

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 289) (h2 : x * y = 120) : x + y = 23 := by
  sorry

end sum_of_integers_l3823_382392


namespace curve_C₂_equation_l3823_382301

-- Define the parabola C₁
def C₁ (x y : ℝ) : Prop := y = (1/20) * x^2

-- Define the focus F of C₁
def F : ℝ × ℝ := (0, 5)

-- Define point E symmetric to F with respect to the origin
def E : ℝ × ℝ := (0, -5)

-- Define the property of points on C₂
def on_C₂ (x y : ℝ) : Prop :=
  abs (Real.sqrt ((x - E.1)^2 + (y - E.2)^2) - Real.sqrt ((x - F.1)^2 + (y - F.2)^2)) = 6

-- Theorem statement
theorem curve_C₂_equation :
  ∀ x y : ℝ, on_C₂ x y ↔ y^2/9 - x^2/16 = 1 :=
sorry

end curve_C₂_equation_l3823_382301


namespace solution_set_for_a_l3823_382319

/-- The solution set for parameter a in the given equation with domain restrictions -/
theorem solution_set_for_a (x a : ℝ) : 
  x ≠ 2 → x ≠ 6 → a - 7*x + 39 ≥ 0 →
  (x^2 - 4*x - 21 + ((|x-2|)/(x-2) + (|x-6|)/(x-6) + a)^2 = 0) →
  a ∈ Set.Ioo (-5) (-4) ∪ Set.Ioo (-3) 3 ∪ Set.Ico 5 7 :=
sorry

end solution_set_for_a_l3823_382319


namespace egg_processing_plant_l3823_382387

theorem egg_processing_plant (E : ℕ) : 
  (∃ A R : ℕ, 
    E = A + R ∧ 
    A = 388 * (R / 12) ∧
    (A + 37) / R = 405 / 3) →
  E = 125763 := by
sorry

end egg_processing_plant_l3823_382387


namespace isabel_weekly_run_distance_l3823_382372

/-- Calculates the total distance run in a week given a circuit length, 
    morning runs, afternoon runs, and number of days. -/
def total_distance_run (circuit_length : ℕ) (morning_runs : ℕ) (afternoon_runs : ℕ) (days : ℕ) : ℕ :=
  (circuit_length * (morning_runs + afternoon_runs) * days)

/-- Proves that running a 365-meter circuit 7 times in the morning and 3 times 
    in the afternoon for 7 days results in a total distance of 25550 meters. -/
theorem isabel_weekly_run_distance :
  total_distance_run 365 7 3 7 = 25550 := by
  sorry

#eval total_distance_run 365 7 3 7

end isabel_weekly_run_distance_l3823_382372


namespace abhay_sameer_speed_difference_l3823_382302

/-- Prove that when Abhay doubles his speed, he takes 1 hour less than Sameer to cover 18 km,
    given that Abhay's original speed is 3 km/h and he initially takes 2 hours more than Sameer. -/
theorem abhay_sameer_speed_difference (distance : ℝ) (abhay_speed : ℝ) (sameer_speed : ℝ) :
  distance = 18 →
  abhay_speed = 3 →
  distance / abhay_speed = distance / sameer_speed + 2 →
  distance / (2 * abhay_speed) = distance / sameer_speed - 1 :=
by sorry

end abhay_sameer_speed_difference_l3823_382302


namespace multiplication_equalities_l3823_382341

theorem multiplication_equalities : 
  (50 * 6 = 300) ∧ (5 * 60 = 300) ∧ (4 * 300 = 1200) := by
  sorry

end multiplication_equalities_l3823_382341


namespace expression_equals_power_of_seven_l3823_382310

theorem expression_equals_power_of_seven : 
  6 * (7 + 1) * (7^2 + 1) * (7^4 + 1) * (7^8 + 1) + 1 = 7^16 := by sorry

end expression_equals_power_of_seven_l3823_382310


namespace f_x1_gt_f_x2_l3823_382391

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Define the theorem
theorem f_x1_gt_f_x2 
  (h_even : is_even f) 
  (h_incr : is_increasing_on_pos f) 
  (x₁ x₂ : ℝ) 
  (h_x1_neg : x₁ < 0) 
  (h_x2_pos : x₂ > 0) 
  (h_abs : abs x₁ > abs x₂) : 
  f x₁ > f x₂ := by
  sorry

end f_x1_gt_f_x2_l3823_382391


namespace counterexample_necessity_l3823_382368

-- Define the concept of a mathematical statement
def MathStatement : Type := String

-- Define the concept of a proof method
inductive ProofMethod
| Direct : ProofMethod
| Counterexample : ProofMethod
| Other : ProofMethod

-- Define a property of mathematical statements
def CanBeProvedDirectly (s : MathStatement) : Prop := sorry

-- Define the theorem to be proved
theorem counterexample_necessity (s : MathStatement) :
  ¬(∀ s, ¬(CanBeProvedDirectly s) → (∀ m : ProofMethod, m = ProofMethod.Counterexample)) :=
sorry

end counterexample_necessity_l3823_382368


namespace strawberry_picking_problem_l3823_382384

/-- Calculates the number of pounds of strawberries picked given the problem conditions -/
def strawberries_picked (entrance_fee : ℚ) (price_per_pound : ℚ) (num_people : ℕ) (total_paid : ℚ) : ℚ :=
  (total_paid + num_people * entrance_fee) / price_per_pound

/-- Theorem stating that under the given conditions, 7 pounds of strawberries were picked -/
theorem strawberry_picking_problem :
  let entrance_fee : ℚ := 4
  let price_per_pound : ℚ := 20
  let num_people : ℕ := 3
  let total_paid : ℚ := 128
  strawberries_picked entrance_fee price_per_pound num_people total_paid = 7 := by
  sorry


end strawberry_picking_problem_l3823_382384


namespace solve_for_a_l3823_382305

theorem solve_for_a (a : ℝ) : (1 + 2 * a = -3) → a = -2 := by
  sorry

end solve_for_a_l3823_382305


namespace triangle_abc_proof_l3823_382328

theorem triangle_abc_proof (A B C : Real) (a b c : Real) : 
  A = π / 6 →
  (1 + Real.sqrt 3) * c = 2 * b →
  c * b * Real.cos C = 1 + Real.sqrt 3 →
  C = π / 4 ∧ 
  a = Real.sqrt 2 ∧ 
  b = 1 + Real.sqrt 3 ∧ 
  c = 2 := by
  sorry

end triangle_abc_proof_l3823_382328


namespace polynomial_factorization_l3823_382325

def polynomial (x y k : ℤ) : ℤ := x^2 + 5*x*y + x + k*y - k

def is_factorable (k : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ), ∀ (x y : ℤ),
    polynomial x y k = (a*x + b*y + c) * (d*x + e*y + f)

theorem polynomial_factorization (k : ℤ) :
  is_factorable k ↔ k = 0 ∨ k = 15 ∨ k = -15 := by sorry

end polynomial_factorization_l3823_382325


namespace expression_independent_of_a_l3823_382363

theorem expression_independent_of_a (a : ℝ) : 7 + a - (8 * a - (a + 5 - (4 - 6 * a))) = 8 := by
  sorry

end expression_independent_of_a_l3823_382363


namespace star_equality_implies_x_equals_nine_l3823_382382

/-- Binary operation ⋆ on pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) :=
  fun (a, b) (c, d) => (a - c, b + d)

/-- Theorem stating that if (6,5) ⋆ (2,3) = (x,y) ⋆ (5,4), then x = 9 -/
theorem star_equality_implies_x_equals_nine :
  ∀ x y : ℤ, star (6, 5) (2, 3) = star (x, y) (5, 4) → x = 9 :=
by
  sorry


end star_equality_implies_x_equals_nine_l3823_382382


namespace parallelogram_side_sum_l3823_382375

theorem parallelogram_side_sum (x y : ℝ) : 
  12 = 10 * y - 2 ∧ 15 = 3 * x + 6 → x + y = 4.4 := by
  sorry

end parallelogram_side_sum_l3823_382375


namespace inequality_system_solution_range_l3823_382379

theorem inequality_system_solution_range (k : ℝ) : 
  (∃! x : ℤ, (x^2 - 2*x - 8 > 0) ∧ (2*x^2 + (2*k + 7)*x + 7*k < 0)) ↔ 
  (k ∈ Set.Icc (-5 : ℝ) 3 ∪ Set.Ioc 4 5) :=
sorry

end inequality_system_solution_range_l3823_382379


namespace inverse_matrices_solution_l3823_382357

def matrix1 (a : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![5, -9; a, 12]
def matrix2 (b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![12, b; 3, 5]

theorem inverse_matrices_solution (a b : ℝ) :
  (matrix1 a) * (matrix2 b) = 1 → a = -3 ∧ b = 9 := by
  sorry

end inverse_matrices_solution_l3823_382357


namespace sum_of_absolute_coefficients_l3823_382317

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| = 4^7 := by
sorry

end sum_of_absolute_coefficients_l3823_382317


namespace tangent_range_l3823_382385

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Define the circle C
def C (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y + 2*k - 1 = 0

-- Define the condition for two tangents
def has_two_tangents (P : ℝ × ℝ) (C : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ k, (P.1^2 + P.2^2 + 2*P.1 + P.2 + 2*k - 1 > 0) ∧
       (4 + 1 - 4*(2*k - 1) > 0)

-- Theorem statement
theorem tangent_range :
  has_two_tangents P C → ∃ k, -4 < k ∧ k < 9/8 :=
sorry

end tangent_range_l3823_382385


namespace no_simultaneous_perfect_squares_l3823_382381

theorem no_simultaneous_perfect_squares (n : ℕ+) :
  ¬∃ (a b : ℕ+), ((n + 1) * 2^n.val = a^2) ∧ ((n + 3) * 2^(n.val + 2) = b^2) := by
  sorry

end no_simultaneous_perfect_squares_l3823_382381


namespace chips_juice_weight_difference_l3823_382344

/-- Given that 2 bags of chips weigh 800 g and 5 bags of chips and 4 bottles of juice
    together weigh 2200 g, prove that a bag of chips is 350 g heavier than a bottle of juice. -/
theorem chips_juice_weight_difference :
  (∀ (chips_weight bottle_weight : ℕ),
    2 * chips_weight = 800 →
    5 * chips_weight + 4 * bottle_weight = 2200 →
    chips_weight - bottle_weight = 350) :=
by sorry

end chips_juice_weight_difference_l3823_382344


namespace lucy_fish_count_l3823_382332

/-- The number of fish Lucy needs to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := 280

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := total_fish - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end lucy_fish_count_l3823_382332


namespace expression_simplification_l3823_382383

theorem expression_simplification (a b : ℝ) (h1 : a = 1) (h2 : b = -2) :
  ((a - 2*b)^2 - (a - 2*b)*(a + 2*b) - 4*b) / (-2*b) = 6 := by
  sorry

end expression_simplification_l3823_382383


namespace negative_a_sufficient_not_necessary_l3823_382327

/-- Represents a quadratic equation ax² + 2x + 1 = 0 -/
structure QuadraticEquation (a : ℝ) where
  eq : ∀ x : ℝ, a * x^2 + 2 * x + 1 = 0 → x ∈ {x | a * x^2 + 2 * x + 1 = 0}

/-- Predicate indicating if an equation has at least one negative root -/
def has_negative_root (eq : QuadraticEquation a) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

/-- The main theorem to prove -/
theorem negative_a_sufficient_not_necessary (a : ℝ) :
  (a < 0 → ∀ eq : QuadraticEquation a, has_negative_root eq) ∧
  (∃ a : ℝ, a ≥ 0 ∧ ∃ eq : QuadraticEquation a, has_negative_root eq) :=
sorry

end negative_a_sufficient_not_necessary_l3823_382327


namespace hiker_count_l3823_382308

theorem hiker_count : ∃ (n m : ℕ), n > 13 ∧ n = 23 ∧ m > 0 ∧ 
  2 * m ≡ 1 [MOD n] ∧ 3 * m ≡ 13 [MOD n] := by
  sorry

end hiker_count_l3823_382308


namespace magician_numbers_l3823_382340

theorem magician_numbers : ∃! (a b : ℕ), 
  a * b = 2280 ∧ 
  a + b < 100 ∧ 
  a + b > 9 ∧ 
  Odd (a + b) ∧ 
  a = 40 ∧ 
  b = 57 := by sorry

end magician_numbers_l3823_382340


namespace table_runner_coverage_l3823_382337

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) (coverage_percentage : ℝ) (two_layer_area : ℝ) :
  total_runner_area = 204 →
  table_area = 175 →
  coverage_percentage = 0.8 →
  two_layer_area = 24 →
  ∃ (one_layer_area three_layer_area : ℝ),
    one_layer_area + two_layer_area + three_layer_area = coverage_percentage * table_area ∧
    one_layer_area + 2 * two_layer_area + 3 * three_layer_area = total_runner_area ∧
    three_layer_area = 20 :=
by sorry

end table_runner_coverage_l3823_382337


namespace estimate_sum_approximately_equal_500_l3823_382321

def round_to_nearest_hundred (n : ℕ) : ℕ :=
  (n + 50) / 100 * 100

def approximately_equal (a b : ℕ) : Prop :=
  round_to_nearest_hundred a = round_to_nearest_hundred b

theorem estimate_sum_approximately_equal_500 :
  approximately_equal (208 + 298) 500 := by sorry

end estimate_sum_approximately_equal_500_l3823_382321


namespace pet_store_cats_l3823_382378

theorem pet_store_cats (siamese : ℝ) (house : ℝ) (added : ℝ) : 
  siamese = 13.0 → house = 5.0 → added = 10.0 → 
  siamese + house + added = 28.0 :=
by
  sorry

end pet_store_cats_l3823_382378


namespace smallest_sum_of_distinct_squares_l3823_382313

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- The theorem statement -/
theorem smallest_sum_of_distinct_squares (a b c d : ℕ) :
  isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectSquare c ∧ isPerfectSquare d ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a ^ b = c ^ d →
  305 ≤ a + b + c + d :=
sorry

end smallest_sum_of_distinct_squares_l3823_382313


namespace sum_odd_integers_mod_12_l3823_382336

/-- The sum of the first n odd positive integers -/
def sum_odd_integers (n : ℕ) : ℕ := n * n

/-- The theorem stating that the remainder when the sum of the first 10 odd positive integers 
    is divided by 12 is equal to 4 -/
theorem sum_odd_integers_mod_12 : sum_odd_integers 10 % 12 = 4 := by
  sorry

end sum_odd_integers_mod_12_l3823_382336


namespace sum_of_d_μ_M_equals_59_22_l3823_382390

-- Define the data distribution
def data_distribution : List (Nat × Nat) :=
  (List.range 28).map (fun x => (x + 1, 24)) ++
  [(29, 22), (30, 22), (31, 14)]

-- Define the total number of data points
def total_count : Nat :=
  data_distribution.foldl (fun acc (_, count) => acc + count) 0

-- Define the median of modes
def d : ℝ := 14.5

-- Define the median of the entire dataset
def M : ℝ := 29

-- Define the mean of the entire dataset
noncomputable def μ : ℝ :=
  let sum := data_distribution.foldl (fun acc (value, count) => acc + value * count) 0
  (sum : ℝ) / total_count

-- Theorem statement
theorem sum_of_d_μ_M_equals_59_22 :
  d + μ + M = 59.22 := by sorry

end sum_of_d_μ_M_equals_59_22_l3823_382390


namespace root_product_equality_l3823_382395

theorem root_product_equality : 
  (16 : ℝ) ^ (1/5) * (64 : ℝ) ^ (1/6) = 2 * (16 : ℝ) ^ (1/5) :=
by sorry

end root_product_equality_l3823_382395


namespace triangle_problem_l3823_382359

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with side lengths a, b, c opposite to angles A, B, C
  c * Real.cos A - 2 * b * Real.cos B + a * Real.cos C = 0 →
  a + c = 13 →
  c > a →
  a * c * Real.cos B = 20 →
  B = Real.pi / 3 ∧ Real.sin A = 5 * Real.sqrt 3 / 14 := by
  sorry

end triangle_problem_l3823_382359


namespace tan_fifteen_equals_sqrt_three_l3823_382386

theorem tan_fifteen_equals_sqrt_three : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end tan_fifteen_equals_sqrt_three_l3823_382386


namespace age_problem_l3823_382307

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 32 → 
  b = 12 :=
by sorry

end age_problem_l3823_382307


namespace remainder_theorem_l3823_382397

theorem remainder_theorem (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 19) → 
  (∃ m : ℤ, N = 13 * m + 6) :=
sorry

end remainder_theorem_l3823_382397


namespace sum_from_true_discount_and_simple_interest_l3823_382345

/-- Given a sum, time, and rate, if the true discount is 80 and the simple interest is 88, then the sum is 880. -/
theorem sum_from_true_discount_and_simple_interest
  (S T R : ℝ) 
  (h1 : S > 0) 
  (h2 : T > 0) 
  (h3 : R > 0) 
  (h4 : (S * R * T) / 100 = 88) 
  (h5 : S - S / (1 + R * T / 100) = 80) : 
  S = 880 := by
sorry

end sum_from_true_discount_and_simple_interest_l3823_382345


namespace star_equation_solution_l3823_382377

def star (a b : ℕ) : ℕ := a^b + a*b

theorem star_equation_solution :
  ∀ a b : ℕ, 
  a ≥ 2 → b ≥ 2 → 
  star a b = 24 → 
  a + b = 6 := by sorry

end star_equation_solution_l3823_382377


namespace triangle_area_theorem_l3823_382360

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 1/2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (side_a : a = Real.sqrt 19)
  (side_b : b = 5)
  (angle_condition : f A = 0)

theorem triangle_area_theorem (t : Triangle) :
  is_monotone_increasing f (π/2) π ∧ 
  (1/2 * t.b * Real.sqrt (19 - t.b^2 + 2*t.b*Real.sqrt 19 * Real.cos t.A)) = 15 * Real.sqrt 3 / 4 :=
sorry

end triangle_area_theorem_l3823_382360


namespace second_candidate_percentage_l3823_382365

theorem second_candidate_percentage (total_marks : ℝ) (passing_marks : ℝ) 
  (first_candidate_percentage : ℝ) (first_candidate_deficit : ℝ) 
  (second_candidate_excess : ℝ) : 
  passing_marks = 160 ∧ 
  first_candidate_percentage = 0.20 ∧ 
  first_candidate_deficit = 40 ∧ 
  second_candidate_excess = 20 ∧
  first_candidate_percentage * total_marks = passing_marks - first_candidate_deficit →
  (passing_marks + second_candidate_excess) / total_marks = 0.30 := by
  sorry

end second_candidate_percentage_l3823_382365


namespace babylonian_square_58_l3823_382316

-- Define the pattern function
def babylonian_square (n : Nat) : Nat × Nat :=
  let square := n * n
  let quotient := square / 60
  let remainder := square % 60
  if remainder = 0 then (quotient - 1, 60) else (quotient, remainder)

-- Theorem statement
theorem babylonian_square_58 : babylonian_square 58 = (56, 4) := by
  sorry

end babylonian_square_58_l3823_382316


namespace symmetric_point_proof_l3823_382367

/-- Given a point (0, 2) and a line x + y - 1 = 0, prove that (-1, 1) is the symmetric point --/
theorem symmetric_point_proof (P : ℝ × ℝ) (P' : ℝ × ℝ) (l : ℝ → ℝ → Prop) :
  P = (0, 2) →
  (∀ x y, l x y ↔ x + y - 1 = 0) →
  P' = (-1, 1) →
  (∀ x y, l ((P.1 + x) / 2) ((P.2 + y) / 2) ↔ l x y) →
  (P'.1 - P.1) * (P'.1 - P.1) + (P'.2 - P.2) * (P'.2 - P.2) =
    ((0 : ℝ) - P.1) * ((0 : ℝ) - P.1) + ((0 : ℝ) - P.2) * ((0 : ℝ) - P.2) :=
by sorry


end symmetric_point_proof_l3823_382367


namespace horner_method_v4_l3823_382371

def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

def horner_v4 (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  let v₀ := a₆
  let v₁ := v₀ * x + a₅
  let v₂ := v₁ * x + a₄
  let v₃ := v₂ * x + a₃
  v₃ * x + a₂

theorem horner_method_v4 :
  horner_v4 3 5 6 20 (-8) 35 12 (-2) = -16 :=
by sorry

end horner_method_v4_l3823_382371


namespace tom_reading_speed_increase_l3823_382380

/-- The factor by which Tom's reading speed increased -/
def reading_speed_increase_factor (normal_speed : ℕ) (increased_pages : ℕ) (hours : ℕ) : ℚ :=
  (increased_pages : ℚ) / ((normal_speed * hours) : ℚ)

/-- Theorem stating that Tom's reading speed increased by a factor of 3 -/
theorem tom_reading_speed_increase :
  reading_speed_increase_factor 12 72 2 = 3 := by
  sorry

end tom_reading_speed_increase_l3823_382380


namespace points_per_enemy_l3823_382304

theorem points_per_enemy (total_enemies : ℕ) (enemies_not_destroyed : ℕ) (total_points : ℕ) : 
  total_enemies = 8 →
  enemies_not_destroyed = 6 →
  total_points = 10 →
  (total_points : ℚ) / (total_enemies - enemies_not_destroyed : ℚ) = 5 := by
  sorry

end points_per_enemy_l3823_382304


namespace union_A_B_complement_A_intersect_B_range_of_a_l3823_382355

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorems to be proved
theorem union_A_B : A ∪ B = {x | 3 ≤ x ∧ x ≤ 10} := by sorry

theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x | 7 ≤ x ∧ x ≤ 10} := by sorry

theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) : a > 3 := by sorry

end union_A_B_complement_A_intersect_B_range_of_a_l3823_382355


namespace triangle_problem_l3823_382389

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Given conditions
  A = π / 4 →
  b = Real.sqrt 6 →
  (1 / 2) * b * c * Real.sin A = (3 + Real.sqrt 3) / 2 →
  -- Definitions from cosine rule
  a = Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos A)) →
  Real.cos B = (a^2 + c^2 - b^2) / (2*a*c) →
  -- Conclusion
  c = 1 + Real.sqrt 3 ∧ B = π / 3 := by
sorry

end triangle_problem_l3823_382389


namespace problem_solution_l3823_382346

def f (m : ℝ) (x : ℝ) : ℝ := |x - 2| - m

theorem problem_solution :
  (∃ m : ℝ, ∀ x : ℝ, f m (x + 2) ≤ 0 ↔ x ∈ Set.Icc (-1) 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a^2 + b^2 + c^2 = 1 →
    a + 2*b + 3*c ≤ Real.sqrt 14) ∧
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 1 ∧
    a + 2*b + 3*c = Real.sqrt 14) := by
  sorry

end problem_solution_l3823_382346


namespace tree_height_problem_l3823_382329

/-- Given two trees where one is 20 feet taller than the other and their heights
    are in the ratio 2:3, prove that the height of the taller tree is 60 feet. -/
theorem tree_height_problem (h : ℝ) (h_positive : h > 0) : 
  (h - 20) / h = 2 / 3 → h = 60 := by
  sorry

end tree_height_problem_l3823_382329


namespace floor_product_eq_42_l3823_382334

theorem floor_product_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 :=
sorry

end floor_product_eq_42_l3823_382334


namespace multiply_three_point_six_by_half_l3823_382350

theorem multiply_three_point_six_by_half : 3.6 * 0.5 = 1.8 := by
  sorry

end multiply_three_point_six_by_half_l3823_382350


namespace sixth_score_for_target_mean_l3823_382393

def david_scores : List ℝ := [85, 88, 90, 82, 94]
def target_mean : ℝ := 90

theorem sixth_score_for_target_mean :
  ∃ (x : ℝ), (david_scores.sum + x) / 6 = target_mean ∧ x = 101 := by
sorry

end sixth_score_for_target_mean_l3823_382393


namespace largest_prime_factor_of_expression_l3823_382324

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (9^3 + 8^5 - 4^5) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (9^3 + 8^5 - 4^5) → q ≤ p :=
by sorry

end largest_prime_factor_of_expression_l3823_382324


namespace arithmetic_mean_of_normal_distribution_l3823_382309

-- Define the arithmetic mean and standard deviation
variable (μ : ℝ) -- arithmetic mean
def σ : ℝ := 1.5 -- standard deviation

-- Define the relationship between the mean, standard deviation, and the given value
def value_two_std_below_mean : ℝ := μ - 2 * σ

-- State the theorem
theorem arithmetic_mean_of_normal_distribution :
  value_two_std_below_mean = 12 → μ = 15 := by
  sorry

end arithmetic_mean_of_normal_distribution_l3823_382309


namespace laptop_lighter_than_tote_l3823_382396

/-- Represents the weights of various items in pounds -/
structure Weights where
  karens_tote : ℝ
  kevins_empty_briefcase : ℝ
  kevins_umbrella : ℝ
  kevins_laptop : ℝ
  kevins_work_papers : ℝ

/-- Conditions given in the problem -/
def problem_conditions (w : Weights) : Prop :=
  w.karens_tote = 8 ∧
  w.karens_tote = 2 * w.kevins_empty_briefcase ∧
  w.kevins_umbrella = w.kevins_empty_briefcase / 2 ∧
  w.kevins_empty_briefcase + w.kevins_laptop + w.kevins_work_papers + w.kevins_umbrella = 2 * w.karens_tote ∧
  w.kevins_work_papers = (w.kevins_empty_briefcase + w.kevins_laptop + w.kevins_work_papers) / 6

theorem laptop_lighter_than_tote (w : Weights) (h : problem_conditions w) :
  w.kevins_laptop < w.karens_tote ∧ w.karens_tote - w.kevins_laptop = 1/3 := by
  sorry

#check laptop_lighter_than_tote

end laptop_lighter_than_tote_l3823_382396


namespace popsicle_bottle_cost_l3823_382353

/-- Represents the cost of popsicle supplies and production -/
structure PopsicleSupplies where
  total_budget : ℚ
  mold_cost : ℚ
  stick_pack_cost : ℚ
  sticks_per_pack : ℕ
  popsicles_per_bottle : ℕ
  remaining_sticks : ℕ

/-- Calculates the cost of each bottle of juice -/
def bottle_cost (supplies : PopsicleSupplies) : ℚ :=
  let money_for_juice := supplies.total_budget - supplies.mold_cost - supplies.stick_pack_cost
  let used_sticks := supplies.sticks_per_pack - supplies.remaining_sticks
  let bottles_used := used_sticks / supplies.popsicles_per_bottle
  money_for_juice / bottles_used

/-- Theorem stating that given the conditions, the cost of each bottle is $2 -/
theorem popsicle_bottle_cost :
  let supplies := PopsicleSupplies.mk 10 3 1 100 20 40
  bottle_cost supplies = 2 := by
  sorry

end popsicle_bottle_cost_l3823_382353


namespace arrangement_count_l3823_382318

/-- The number of representatives in unit A -/
def unitA : ℕ := 7

/-- The number of representatives in unit B -/
def unitB : ℕ := 3

/-- The total number of elements to arrange (treating unit B as one element) -/
def totalElements : ℕ := unitA + 1

/-- The number of possible arrangements -/
def numArrangements : ℕ := (Nat.factorial totalElements) * (Nat.factorial unitB)

theorem arrangement_count : numArrangements = 241920 := by sorry

end arrangement_count_l3823_382318


namespace systematic_sampling_theorem_l3823_382333

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  sample_size : Nat
  start : Nat
  interval : Nat

/-- Generates the sequence of selected student numbers -/
def generate_sequence (s : SystematicSampling) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.interval)

/-- Checks if all numbers in the sequence are valid student numbers -/
def valid_sequence (s : SystematicSampling) (seq : List Nat) : Prop :=
  seq.all (fun n => 1 ≤ n ∧ n ≤ s.total_students)

theorem systematic_sampling_theorem (s : SystematicSampling) :
  s.total_students = 60 →
  s.sample_size = 5 →
  s.start = 6 →
  s.interval = 12 →
  generate_sequence s = [6, 18, 30, 42, 54] ∧
  valid_sequence s (generate_sequence s) := by
  sorry

#eval generate_sequence { total_students := 60, sample_size := 5, start := 6, interval := 12 }

end systematic_sampling_theorem_l3823_382333


namespace probability_seven_white_three_black_l3823_382349

/-- The probability of drawing first a black ball and then a white ball from a bag -/
def probability_black_then_white (white_balls black_balls : ℕ) : ℚ :=
  let total_balls := white_balls + black_balls
  let prob_black_first := black_balls / total_balls
  let prob_white_second := white_balls / (total_balls - 1)
  prob_black_first * prob_white_second

/-- Theorem stating the probability of drawing first a black ball and then a white ball
    from a bag containing 7 white balls and 3 black balls is 7/30 -/
theorem probability_seven_white_three_black :
  probability_black_then_white 7 3 = 7 / 30 := by
  sorry

end probability_seven_white_three_black_l3823_382349


namespace inequality_preservation_l3823_382342

theorem inequality_preservation (m n : ℝ) (h1 : m < n) (h2 : n < 0) :
  m + 2 < n + 2 := by
  sorry

end inequality_preservation_l3823_382342


namespace root_relationship_l3823_382362

def P (x : ℝ) : ℝ := x^3 - 2*x + 1

def Q (x : ℝ) : ℝ := x^3 - 4*x^2 + 4*x - 1

theorem root_relationship (r : ℝ) : P r = 0 → Q (r^2) = 0 := by
  sorry

end root_relationship_l3823_382362


namespace tan_sum_reciprocal_l3823_382314

theorem tan_sum_reciprocal (a b : ℝ) 
  (h1 : (Real.sin a / Real.cos b) + (Real.sin b / Real.cos a) = 2)
  (h2 : (Real.cos a / Real.sin b) + (Real.cos b / Real.sin a) = 4) :
  (Real.tan a / Real.tan b) + (Real.tan b / Real.tan a) = 44/5 := by
  sorry

end tan_sum_reciprocal_l3823_382314


namespace vector_calculation_l3823_382364

/-- Given vectors a and b in ℝ², prove that 2a - b equals (5, 7) -/
theorem vector_calculation (a b : ℝ × ℝ) 
  (ha : a = (2, 4)) (hb : b = (-1, 1)) : 
  (2 : ℝ) • a - b = (5, 7) := by sorry

end vector_calculation_l3823_382364


namespace angle_c_measure_l3823_382361

/-- Given a triangle ABC where the sum of angles A and B is 110°, prove that the measure of angle C is 70°. -/
theorem angle_c_measure (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := by
  sorry

end angle_c_measure_l3823_382361


namespace clock_face_partition_l3823_382343

noncomputable def clockFaceAreas (r : ℝ) : (ℝ × ℝ × ℝ × ℝ) :=
  let t₁ := (Real.pi + 2 * Real.sqrt 3 - 6) / 12 * r^2
  let t₂ := (Real.pi - Real.sqrt 3) / 6 * r^2
  let t₃ := (7 * Real.pi + 2 * Real.sqrt 3 - 6) / 12 * r^2
  (t₁, t₂, t₂, t₃)

theorem clock_face_partition (r : ℝ) (h : r > 0) :
  let (t₁, t₂, t₂', t₃) := clockFaceAreas r
  t₁ + t₂ + t₂' + t₃ = Real.pi * r^2 ∧
  t₂ = t₂' ∧
  t₁ > 0 ∧ t₂ > 0 ∧ t₃ > 0 :=
by sorry

end clock_face_partition_l3823_382343


namespace arrangement_theorem_l3823_382376

/-- The number of ways to arrange 5 people in a row with two specific people not adjacent -/
def arrangement_count : ℕ := 72

/-- The number of people to be arranged -/
def total_people : ℕ := 5

/-- The number of people who can be freely arranged -/
def free_people : ℕ := total_people - 2

/-- The number of positions where the two specific people can be inserted -/
def insertion_positions : ℕ := free_people + 1

theorem arrangement_theorem :
  arrangement_count = (free_people.factorial) * (insertion_positions.factorial / 2) :=
sorry

end arrangement_theorem_l3823_382376


namespace company_manager_fraction_l3823_382339

/-- Given a company with female managers, total female employees, and the condition that
    the fraction of managers is the same for all employees and male employees,
    prove that the fraction of employees who are managers is 0.4 -/
theorem company_manager_fraction (total_female_employees : ℕ) (female_managers : ℕ)
    (h1 : female_managers = 200)
    (h2 : total_female_employees = 500)
    (h3 : ∃ (f : ℚ), f * (total_female_employees : ℚ) = (female_managers : ℚ) ∧
                     f * ((total_female_employees : ℚ) - (female_managers : ℚ)) = 
                     (female_managers : ℚ) * ((total_female_employees : ℚ) / (female_managers : ℚ) - 1)) :
  ∃ (f : ℚ), f = 0.4 ∧ 
    f * (total_female_employees : ℚ) = (female_managers : ℚ) ∧
    f * ((total_female_employees : ℚ) - (female_managers : ℚ)) = 
    (female_managers : ℚ) * ((total_female_employees : ℚ) / (female_managers : ℚ) - 1) := by
  sorry


end company_manager_fraction_l3823_382339


namespace complex_fraction_simplification_l3823_382320

theorem complex_fraction_simplification :
  (5 - 7 * Complex.I) / (2 - 3 * Complex.I) = 31/13 + (1/13) * Complex.I :=
by sorry

end complex_fraction_simplification_l3823_382320


namespace possible_distances_andrey_gleb_l3823_382354

/-- Represents the position of a house on a straight street. -/
structure HousePosition where
  position : ℝ

/-- Represents the configuration of houses on the street. -/
structure StreetConfiguration where
  andrey : HousePosition
  borya : HousePosition
  vova : HousePosition
  gleb : HousePosition

/-- The distance between two house positions. -/
def distance (a b : HousePosition) : ℝ :=
  |a.position - b.position|

/-- Theorem stating the possible distances between Andrey's and Gleb's houses. -/
theorem possible_distances_andrey_gleb (config : StreetConfiguration) :
  (distance config.andrey config.borya = 600) →
  (distance config.vova config.gleb = 600) →
  (distance config.andrey config.gleb = 3 * distance config.borya config.vova) →
  (distance config.andrey config.gleb = 900 ∨ distance config.andrey config.gleb = 1800) :=
by sorry

end possible_distances_andrey_gleb_l3823_382354


namespace girls_count_in_school_l3823_382398

/-- Proves that in a school with a given total number of students and a ratio of boys to girls,
    the number of girls is as calculated. -/
theorem girls_count_in_school (total : ℕ) (boys_ratio girls_ratio : ℕ) 
    (h_total : total = 480) 
    (h_ratio : boys_ratio = 3 ∧ girls_ratio = 5) : 
    (girls_ratio * total) / (boys_ratio + girls_ratio) = 300 := by
  sorry

end girls_count_in_school_l3823_382398


namespace pam_has_ten_bags_l3823_382323

/-- Represents the number of apples in each of Gerald's bags -/
def geralds_bag_count : ℕ := 40

/-- Represents the ratio of apples in Pam's bags to Gerald's bags -/
def pam_to_gerald_ratio : ℕ := 3

/-- Represents the total number of apples Pam has -/
def pam_total_apples : ℕ := 1200

/-- Calculates the number of bags Pam has -/
def pam_bag_count : ℕ := pam_total_apples / (geralds_bag_count * pam_to_gerald_ratio)

theorem pam_has_ten_bags : pam_bag_count = 10 := by
  sorry

end pam_has_ten_bags_l3823_382323


namespace f_strictly_increasing_l3823_382326

def f (x : ℝ) : ℝ := (x + 1)^2 + 1

theorem f_strictly_increasing : 
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry

end f_strictly_increasing_l3823_382326


namespace a_eq_one_sufficient_not_necessary_l3823_382322

def M : Set ℝ := {1, 2}
def N (a : ℝ) : Set ℝ := {a^2}

theorem a_eq_one_sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → N a ⊆ M) ∧
  (∃ a : ℝ, a ≠ 1 ∧ N a ⊆ M) :=
by sorry

end a_eq_one_sufficient_not_necessary_l3823_382322


namespace complex_magnitude_problem_l3823_382331

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 - 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
sorry

end complex_magnitude_problem_l3823_382331


namespace absolute_value_inequality_l3823_382300

theorem absolute_value_inequality (x : ℝ) :
  (|x + 1| - |x - 3| ≥ 2) ↔ (x ≥ 2) := by
  sorry

end absolute_value_inequality_l3823_382300


namespace jordan_empty_boxes_l3823_382373

/-- A structure representing the distribution of items in boxes -/
structure BoxDistribution where
  total : ℕ
  pencils : ℕ
  pens : ℕ
  markers : ℕ
  pencils_and_pens : ℕ
  pencils_and_markers : ℕ
  pens_and_markers : ℕ

/-- The number of boxes with no items, given a box distribution -/
def empty_boxes (d : BoxDistribution) : ℕ :=
  d.total - (d.pencils + d.pens + d.markers - d.pencils_and_pens - d.pencils_and_markers - d.pens_and_markers)

/-- The specific box distribution from the problem -/
def jordan_boxes : BoxDistribution :=
  { total := 15
  , pencils := 8
  , pens := 5
  , markers := 3
  , pencils_and_pens := 2
  , pencils_and_markers := 1
  , pens_and_markers := 1 }

/-- Theorem stating that the number of empty boxes in Jordan's distribution is 3 -/
theorem jordan_empty_boxes :
    empty_boxes jordan_boxes = 3 := by
  sorry


end jordan_empty_boxes_l3823_382373


namespace problem_1_problem_2_l3823_382388

-- Problem 1
theorem problem_1 (a : ℝ) (h : Real.sqrt a + 1 / Real.sqrt a = 3) :
  (a^2 + 1/a^2 + 3) / (4*a + 1/(4*a)) = 10 * Real.sqrt 5 := by sorry

-- Problem 2
theorem problem_2 :
  (1 - Real.log 3 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 18 / Real.log 6) * (Real.log 6 / Real.log 4) = 1 := by sorry

end problem_1_problem_2_l3823_382388


namespace part_one_part_two_l3823_382303

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := abs x * (x + a)

-- Part I
theorem part_one (a : ℝ) (h : ∀ x, f a x = -f a (-x)) : a = 0 := by
  sorry

-- Part II
theorem part_two (b : ℝ) (h1 : b > 0) 
  (h2 : ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-b) b, f 0 x ≤ max ∧ min ≤ f 0 x) ∧ max - min = b) :
  b = 2 := by
  sorry

end part_one_part_two_l3823_382303


namespace probability_two_white_and_one_white_one_red_l3823_382330

/-- Represents the color of a ball -/
inductive Color
  | White
  | Red

/-- Represents a bag of balls -/
structure Bag :=
  (total : Nat)
  (white : Nat)
  (red : Nat)
  (h_total : total = white + red)

/-- Calculates the probability of drawing two balls of a specific color combination -/
def probability_draw_two (bag : Bag) (first second : Color) : Rat :=
  sorry

theorem probability_two_white_and_one_white_one_red 
  (bag : Bag)
  (h_total : bag.total = 6)
  (h_white : bag.white = 4)
  (h_red : bag.red = 2) :
  (probability_draw_two bag Color.White Color.White = 2/5) ∧
  (probability_draw_two bag Color.White Color.Red = 8/15) :=
sorry

end probability_two_white_and_one_white_one_red_l3823_382330


namespace odd_function_property_l3823_382335

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- Main theorem -/
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : IsOdd f) 
  (h_even : IsEven (fun x ↦ f (x + 2))) 
  (h_f_neg_one : f (-1) = -1) : 
  f 2017 + f 2016 = 1 := by
  sorry

end odd_function_property_l3823_382335


namespace quadrilateral_diagonal_sum_lower_bound_l3823_382347

theorem quadrilateral_diagonal_sum_lower_bound (x y : ℝ) (α : ℝ) :
  x > 0 → y > 0 → 0 < α → α < π →
  x * y * Real.sin α = 2 →
  x + y ≥ 2 * Real.sqrt 2 := by
sorry

end quadrilateral_diagonal_sum_lower_bound_l3823_382347


namespace fish_population_estimate_l3823_382306

/-- Estimate the fish population in a pond given tagging and sampling data --/
theorem fish_population_estimate
  (initial_tagged : ℕ)
  (august_sample : ℕ)
  (august_tagged : ℕ)
  (left_pond_ratio : ℚ)
  (new_fish_ratio : ℚ)
  (h_initial_tagged : initial_tagged = 50)
  (h_august_sample : august_sample = 80)
  (h_august_tagged : august_tagged = 4)
  (h_left_pond : left_pond_ratio = 3/10)
  (h_new_fish : new_fish_ratio = 45/100)
  (h_representative_sample : True)  -- Assuming the sample is representative
  (h_negligible_tag_loss : True)    -- Assuming tag loss is negligible
  : ↑initial_tagged * (august_sample * (1 - new_fish_ratio)) / august_tagged = 550 := by
  sorry


end fish_population_estimate_l3823_382306
