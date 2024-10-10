import Mathlib

namespace polynomial_fits_data_l454_45415

def f (x : ℝ) : ℝ := x^3 + 2*x^2 + x + 1

theorem polynomial_fits_data : 
  f 1 = 5 ∧ f 2 = 15 ∧ f 3 = 35 ∧ f 4 = 69 ∧ f 5 = 119 := by
  sorry

end polynomial_fits_data_l454_45415


namespace combination_5_choose_3_l454_45431

/-- The number of combinations of n things taken k at a time -/
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Proof that C(5,3) equals 10 -/
theorem combination_5_choose_3 : combination 5 3 = 10 := by
  sorry

end combination_5_choose_3_l454_45431


namespace swimming_pool_width_l454_45424

/-- Represents the dimensions and area of a rectangular swimming pool -/
structure SwimmingPool where
  length : ℝ
  width : ℝ
  area : ℝ

/-- Theorem: Given a rectangular swimming pool with area 143.2 m² and length 4 m, its width is 35.8 m -/
theorem swimming_pool_width (pool : SwimmingPool) 
  (h_area : pool.area = 143.2)
  (h_length : pool.length = 4)
  (h_rectangle : pool.area = pool.length * pool.width) : 
  pool.width = 35.8 := by
  sorry

end swimming_pool_width_l454_45424


namespace last_digit_of_large_prime_l454_45445

theorem last_digit_of_large_prime (n : ℕ) (h : n = 859433) :
  (2^n - 1) % 10 = 1 :=
by
  sorry

end last_digit_of_large_prime_l454_45445


namespace group_sum_difference_l454_45488

/-- S_n represents the sum of the n-th group in a sequence where
    the n-th group contains n consecutive natural numbers starting from
    n(n-1)/2 + 1 -/
def S (n : ℕ) : ℕ := n * (n^2 + 1) / 2

/-- The theorem states that S_16 - S_4 - S_1 = 2021 -/
theorem group_sum_difference : S 16 - S 4 - S 1 = 2021 := by
  sorry

end group_sum_difference_l454_45488


namespace exists_points_with_midpoint_l454_45400

/-- Definition of the hyperbola -/
def is_on_hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2/9 = 1

/-- Definition of midpoint -/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂)/2 ∧ y₀ = (y₁ + y₂)/2

/-- Theorem statement -/
theorem exists_points_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_on_hyperbola x₁ y₁ ∧
    is_on_hyperbola x₂ y₂ ∧
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
by sorry

end exists_points_with_midpoint_l454_45400


namespace water_removal_proof_l454_45481

/-- Represents the fraction of water remaining after n steps -/
def remainingWater (n : ℕ) : ℚ :=
  2 / (n + 2 : ℚ)

/-- The number of steps after which one eighth of the water remains -/
def stepsToOneEighth : ℕ := 14

theorem water_removal_proof :
  remainingWater stepsToOneEighth = 1/8 :=
sorry

end water_removal_proof_l454_45481


namespace initial_bird_families_l454_45402

theorem initial_bird_families (flew_away left_now : ℕ) 
  (h1 : flew_away = 27) 
  (h2 : left_now = 14) : 
  flew_away + left_now = 41 := by
  sorry

end initial_bird_families_l454_45402


namespace min_value_reciprocal_sum_l454_45441

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 4 * y = Real.log 2) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_reciprocal_sum_l454_45441


namespace min_value_reciprocal_sum_l454_45417

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 :=
sorry

end min_value_reciprocal_sum_l454_45417


namespace sum_of_repeating_decimals_l454_45473

-- Define the repeating decimals
def repeating_2 : ℚ := 2/9
def repeating_03 : ℚ := 1/33

-- Theorem statement
theorem sum_of_repeating_decimals : 
  repeating_2 + repeating_03 = 25/99 := by sorry

end sum_of_repeating_decimals_l454_45473


namespace fractional_equation_solution_l454_45475

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 3) :
  (2 - x) / (x - 3) + 1 / (3 - x) = 1 ↔ x = 2 := by
sorry

end fractional_equation_solution_l454_45475


namespace money_division_l454_45495

theorem money_division (total : ℝ) (a b c : ℝ) 
  (h_total : total = 392)
  (h_a : a = b / 2)
  (h_b : b = c / 2)
  (h_sum : a + b + c = total) : c = 224 := by
  sorry

end money_division_l454_45495


namespace correct_num_tripodasauruses_l454_45444

/-- Represents the number of tripodasauruses in a flock -/
def num_tripodasauruses : ℕ := 5

/-- Represents the number of legs a tripodasaurus has -/
def legs_per_tripodasaurus : ℕ := 3

/-- Represents the number of heads a tripodasaurus has -/
def heads_per_tripodasaurus : ℕ := 1

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 20

/-- Theorem stating that the number of tripodasauruses in the flock is correct -/
theorem correct_num_tripodasauruses : 
  num_tripodasauruses * (legs_per_tripodasaurus + heads_per_tripodasaurus) = total_heads_and_legs :=
by sorry

end correct_num_tripodasauruses_l454_45444


namespace probability_consecutive_days_l454_45499

-- Define the number of days
def total_days : ℕ := 10

-- Define the number of days to be selected
def selected_days : ℕ := 3

-- Define the number of ways to select 3 consecutive days
def consecutive_selections : ℕ := total_days - selected_days + 1

-- Define the total number of ways to select 3 days from 10 days
def total_selections : ℕ := Nat.choose total_days selected_days

-- Theorem statement
theorem probability_consecutive_days :
  (consecutive_selections : ℚ) / total_selections = 1 / 15 :=
sorry

end probability_consecutive_days_l454_45499


namespace completing_square_equivalence_l454_45484

theorem completing_square_equivalence (x : ℝ) :
  x^2 - 6*x + 1 = 0 ↔ (x - 3)^2 = 8 := by sorry

end completing_square_equivalence_l454_45484


namespace log_base_1024_integer_count_l454_45436

theorem log_base_1024_integer_count :
  ∃! (S : Finset ℕ+), 
    (∀ b ∈ S, ∃ n : ℕ+, (b : ℝ) ^ (n : ℝ) = 1024) ∧ 
    (∀ b : ℕ+, (∃ n : ℕ+, (b : ℝ) ^ (n : ℝ) = 1024) → b ∈ S) ∧
    S.card = 4 :=
by sorry

end log_base_1024_integer_count_l454_45436


namespace integer_solutions_of_equation_l454_45487

theorem integer_solutions_of_equation :
  ∀ m n : ℤ, m^2 + 2*m = n^4 + 20*n^3 + 104*n^2 + 40*n + 2003 →
  ((m = 128 ∧ n = 7) ∨ (m = 128 ∧ n = -17)) := by
sorry

end integer_solutions_of_equation_l454_45487


namespace min_reciprocal_sum_l454_45463

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 65 / 6 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 65 / 6 :=
by sorry

end min_reciprocal_sum_l454_45463


namespace image_square_characterization_l454_45458

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ := (x^2 - y^2, x*y)

-- Define the unit square
def unit_square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the image of the unit square
def image_square : Set (ℝ × ℝ) := {p | ∃ q ∈ unit_square, transform q.1 q.2 = p}

-- Define the boundary curves
def curve_OC : Set (ℝ × ℝ) := {p | ∃ y ∈ Set.Icc 0 1, p = (-y^2, 0)}
def curve_OA : Set (ℝ × ℝ) := {p | ∃ x ∈ Set.Icc 0 1, p = (x^2, 0)}
def curve_AB : Set (ℝ × ℝ) := {p | ∃ y ∈ Set.Icc 0 1, p = (1 - y^2, y)}
def curve_BC : Set (ℝ × ℝ) := {p | ∃ x ∈ Set.Icc 0 1, p = (x^2 - 1, x)}

-- Define the boundary of the image
def image_boundary : Set (ℝ × ℝ) := curve_OC ∪ curve_OA ∪ curve_AB ∪ curve_BC

-- Theorem statement
theorem image_square_characterization :
  image_square = {p | p ∈ image_boundary ∨ (∃ q ∈ image_boundary, p.1 < q.1 ∧ p.2 < q.2)} := by
  sorry

end image_square_characterization_l454_45458


namespace hyperbola_axis_ratio_l454_45405

/-- For a hyperbola with equation x^2 - my^2 = 1, if the length of the imaginary axis
    is three times the length of the real axis, then m = 1/9 -/
theorem hyperbola_axis_ratio (m : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (∀ (x y : ℝ), x^2 - m*y^2 = 1 ↔ (x/a)^2 - (y/b)^2 = 1) ∧
    b = 3*a) →
  m = 1/9 :=
sorry

end hyperbola_axis_ratio_l454_45405


namespace weight_of_new_person_l454_45464

theorem weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 66 →
  avg_increase = 2.5 →
  ∃ (new_weight : ℝ), new_weight = replaced_weight + initial_count * avg_increase :=
by
  sorry

end weight_of_new_person_l454_45464


namespace triangle_determinant_zero_l454_45432

theorem triangle_determinant_zero (A B C : Real) 
  (h_triangle : A + B + C = Real.pi) : 
  let matrix : Matrix (Fin 3) (Fin 3) Real := 
    ![![Real.cos A ^ 2, Real.tan A, 1],
      ![Real.cos B ^ 2, Real.tan B, 1],
      ![Real.cos C ^ 2, Real.tan C, 1]]
  Matrix.det matrix = 0 := by
  sorry

end triangle_determinant_zero_l454_45432


namespace unique_number_property_l454_45467

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def remove_digit (n : ℕ) (pos : Fin 5) : ℕ :=
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  let removed := digits.removeNth pos
  removed.foldl (fun acc d => acc * 10 + d) 0

theorem unique_number_property :
  ∃! n : ℕ, is_five_digit n ∧
    ∃ pos : Fin 5, n + remove_digit n pos = 54321 := by sorry

end unique_number_property_l454_45467


namespace factor_into_sqrt_l454_45418

theorem factor_into_sqrt (a b : ℝ) (h : a < b) :
  (a - b) * Real.sqrt (-1 / (a - b)) = -Real.sqrt (b - a) := by
  sorry

end factor_into_sqrt_l454_45418


namespace division_simplification_l454_45494

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a^2 / (a / 2) = 12 * a := by
  sorry

end division_simplification_l454_45494


namespace divisibility_criterion_l454_45455

theorem divisibility_criterion (a b c : ℤ) (d : ℤ) (h1 : d = 10*c + 1) (h2 : ∃ k, a - b*c = d*k) : 
  ∃ m, 10*a + b = d*m :=
sorry

end divisibility_criterion_l454_45455


namespace inverse_function_relation_l454_45462

/-- Given a function h and its inverse f⁻¹, prove the relation between a and b --/
theorem inverse_function_relation (a b : ℝ) :
  (∀ x, 3 * x - 6 = (Function.invFun (fun x => a * x + b)) x - 2) →
  3 * a + 4 * b = 19 / 3 := by
  sorry

end inverse_function_relation_l454_45462


namespace eds_pets_l454_45407

/-- The number of pets Ed has -/
def total_pets (dogs cats : ℕ) : ℕ :=
  let fish := 2 * (dogs + cats)
  let birds := dogs * cats
  dogs + cats + fish + birds

/-- Theorem stating the total number of Ed's pets -/
theorem eds_pets : total_pets 2 3 = 21 := by
  sorry

end eds_pets_l454_45407


namespace b_55_mod_55_eq_zero_l454_45410

/-- The integer obtained by writing all the integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The remainder when b₅₅ is divided by 55 is 0 -/
theorem b_55_mod_55_eq_zero : b 55 % 55 = 0 := by
  sorry

end b_55_mod_55_eq_zero_l454_45410


namespace rectangle_dimensions_l454_45489

theorem rectangle_dimensions (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (harea : x * y = 36) (hperim : 2 * x + 2 * y = 30) :
  (x = 12 ∧ y = 3) ∨ (x = 3 ∧ y = 12) := by
  sorry

end rectangle_dimensions_l454_45489


namespace infinitely_many_fantastic_triplets_l454_45446

/-- Definition of a fantastic triplet -/
def is_fantastic_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ k : ℚ, b = k * a ∧ c = k * b) ∧
  (∃ d : ℤ, b + 1 - a = d ∧ c - (b + 1) = d)

/-- There exist infinitely many fantastic triplets -/
theorem infinitely_many_fantastic_triplets :
  ∀ i : ℕ, ∃ a b c : ℕ,
    is_fantastic_triplet a b c ∧
    a = 2^(2*i+1) ∧
    b = 2^(2*i+1) + 2^i ∧
    c = 2^(2*i+1) + 2^(i+2) + 2 :=
by sorry

end infinitely_many_fantastic_triplets_l454_45446


namespace bernardo_wins_game_l454_45422

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem bernardo_wins_game :
  ∃ N : ℕ,
    N = 32 ∧
    16 * N + 1400 < 2000 ∧
    16 * N + 1500 ≥ 2000 ∧
    sum_of_digits N = 5 ∧
    ∀ m : ℕ, m < N →
      ¬(16 * m + 1400 < 2000 ∧
        16 * m + 1500 ≥ 2000 ∧
        sum_of_digits m = 5) :=
by sorry

end bernardo_wins_game_l454_45422


namespace product_no_x3_x2_terms_l454_45491

theorem product_no_x3_x2_terms (p q : ℝ) : 
  (∀ x : ℝ, (x^2 + p*x + 8) * (x^2 - 3*x + q) = x^4 + (p*q - 24)*x + 8*q) → 
  p = 3 ∧ q = 1 := by
sorry

end product_no_x3_x2_terms_l454_45491


namespace complex_magnitude_equation_l454_45456

theorem complex_magnitude_equation (t : ℝ) : 
  (t > 0 ∧ Complex.abs (t + 3 * Complex.I * Real.sqrt 2) * Complex.abs (8 - 3 * Complex.I) = 40) ↔ 
  t = Real.sqrt (286 / 73) := by
sorry

end complex_magnitude_equation_l454_45456


namespace g_monotone_decreasing_iff_a_in_range_l454_45401

/-- The function g(x) defined as ax³ + 2(1-a)x² - 3ax -/
def g (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 2 * (1 - a) * x^2 - 3 * a * x

/-- g(x) is monotonically decreasing in the interval (-∞, a/3) if and only if -1 ≤ a ≤ 0 -/
theorem g_monotone_decreasing_iff_a_in_range (a : ℝ) :
  (∀ x y, x < y → y < a/3 → g a x > g a y) ↔ -1 ≤ a ∧ a ≤ 0 :=
sorry

end g_monotone_decreasing_iff_a_in_range_l454_45401


namespace rajs_house_area_l454_45479

/-- The total area of Raj's house given the specified room dimensions and counts -/
theorem rajs_house_area : 
  let bedroom_count : ℕ := 4
  let bedroom_side : ℕ := 11
  let bathroom_count : ℕ := 2
  let bathroom_length : ℕ := 8
  let bathroom_width : ℕ := 6
  let kitchen_area : ℕ := 265
  
  bedroom_count * (bedroom_side * bedroom_side) +
  bathroom_count * (bathroom_length * bathroom_width) +
  kitchen_area +
  kitchen_area = 1110 := by
sorry

end rajs_house_area_l454_45479


namespace exp_25pi_i_div_2_equals_i_l454_45425

theorem exp_25pi_i_div_2_equals_i :
  Complex.exp (25 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end exp_25pi_i_div_2_equals_i_l454_45425


namespace printing_task_theorem_l454_45443

/-- Represents the printing task -/
structure PrintingTask where
  totalPages : ℕ
  printerATime : ℕ
  printerBExtraRate : ℕ

/-- Calculates the time taken for both printers to complete the task together -/
def timeTakenTogether (task : PrintingTask) : ℚ :=
  (task.totalPages : ℚ) * (task.printerATime : ℚ) / (task.totalPages + task.printerATime * task.printerBExtraRate : ℚ)

/-- Theorem stating that for the given conditions, the time taken is (35 * 60) / 430 minutes -/
theorem printing_task_theorem (task : PrintingTask) 
  (h1 : task.totalPages = 35)
  (h2 : task.printerATime = 60)
  (h3 : task.printerBExtraRate = 6) : 
  timeTakenTogether task = 35 * 60 / 430 := by
  sorry

#eval timeTakenTogether { totalPages := 35, printerATime := 60, printerBExtraRate := 6 }

end printing_task_theorem_l454_45443


namespace polynomial_equality_l454_45466

theorem polynomial_equality (a b c : ℝ) : 
  ((a - b) - c = a - b - c) ∧ 
  (a - (b + c) = a - b - c) ∧ 
  (-(b + c - a) = a - b - c) ∧ 
  (a - (b - c) ≠ a - b - c) :=
sorry

end polynomial_equality_l454_45466


namespace sufficient_not_necessary_l454_45420

-- Define the concept of a function being even or odd
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the condition that both f and g are either odd or even
def BothEvenOrOdd (f g : ℝ → ℝ) : Prop :=
  (IsEven f ∧ IsEven g) ∨ (IsOdd f ∧ IsOdd g)

-- Define the property that the product of f and g is even
def ProductIsEven (f g : ℝ → ℝ) : Prop :=
  IsEven (fun x ↦ f x * g x)

-- Theorem statement
theorem sufficient_not_necessary (f g : ℝ → ℝ) :
  (BothEvenOrOdd f g → ProductIsEven f g) ∧
  ¬(ProductIsEven f g → BothEvenOrOdd f g) := by
  sorry


end sufficient_not_necessary_l454_45420


namespace color_assignment_l454_45440

-- Define the colors
inductive Color
| White
| Red
| Blue

-- Define the friends
inductive Friend
| Tamara
| Valya
| Lida

-- Define a function to assign colors to dresses
def dress : Friend → Color := sorry

-- Define a function to assign colors to shoes
def shoes : Friend → Color := sorry

-- Define the theorem
theorem color_assignment :
  -- Tamara's dress and shoes match
  (dress Friend.Tamara = shoes Friend.Tamara) ∧
  -- Valya wore white shoes
  (shoes Friend.Valya = Color.White) ∧
  -- Neither Lida's dress nor her shoes were red
  (dress Friend.Lida ≠ Color.Red ∧ shoes Friend.Lida ≠ Color.Red) ∧
  -- All friends have different dress colors
  (dress Friend.Tamara ≠ dress Friend.Valya ∧
   dress Friend.Tamara ≠ dress Friend.Lida ∧
   dress Friend.Valya ≠ dress Friend.Lida) ∧
  -- All friends have different shoe colors
  (shoes Friend.Tamara ≠ shoes Friend.Valya ∧
   shoes Friend.Tamara ≠ shoes Friend.Lida ∧
   shoes Friend.Valya ≠ shoes Friend.Lida) →
  -- The only valid assignment is:
  (dress Friend.Tamara = Color.Red ∧ shoes Friend.Tamara = Color.Red) ∧
  (dress Friend.Valya = Color.Blue ∧ shoes Friend.Valya = Color.White) ∧
  (dress Friend.Lida = Color.White ∧ shoes Friend.Lida = Color.Blue) :=
by
  sorry

end color_assignment_l454_45440


namespace empty_set_proof_l454_45409

theorem empty_set_proof : {x : ℝ | x^2 + x + 1 = 0} = ∅ := by
  sorry

end empty_set_proof_l454_45409


namespace fraction_addition_l454_45460

theorem fraction_addition : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end fraction_addition_l454_45460


namespace complex_equation_solution_l454_45427

theorem complex_equation_solution (z : ℂ) :
  (z - 2) * (1 + Complex.I) = 1 - Complex.I → z = 2 - Complex.I := by
  sorry

end complex_equation_solution_l454_45427


namespace divisible_by_three_l454_45404

theorem divisible_by_three (n : ℕ) : 
  (3 ∣ n * 2^n + 1) ↔ (n % 3 = 1 ∨ n % 3 = 2) := by
  sorry

end divisible_by_three_l454_45404


namespace geometric_sequence_sum_l454_45419

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  (a 1 + a 3 = 8) ∧
  (a 5 + a 7 = 4)

/-- The sum of specific terms in the geometric sequence equals 3 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 9 + a 11 + a 13 + a 15 = 3 := by
  sorry

end geometric_sequence_sum_l454_45419


namespace colors_drying_time_l454_45439

/-- Represents the time in minutes for a laundry load -/
structure LaundryTime where
  washing : ℕ
  drying : ℕ

/-- The total time for all three loads of laundry -/
def total_time : ℕ := 344

/-- The laundry time for the whites -/
def whites : LaundryTime := { washing := 72, drying := 50 }

/-- The laundry time for the darks -/
def darks : LaundryTime := { washing := 58, drying := 65 }

/-- The washing time for the colors -/
def colors_washing : ℕ := 45

/-- The theorem stating that the drying time for colors is 54 minutes -/
theorem colors_drying_time : 
  total_time - (whites.washing + whites.drying + darks.washing + darks.drying + colors_washing) = 54 := by
  sorry

end colors_drying_time_l454_45439


namespace movie_theater_receipts_l454_45465

/-- 
Given a movie theater with the following conditions:
- Child ticket price is $4.50
- Adult ticket price is $6.75
- There are 20 more children than adults
- There are 48 children at the matinee

Prove that the total receipts for today's matinee is $405.
-/
theorem movie_theater_receipts : 
  let child_price : ℚ := 4.5
  let adult_price : ℚ := 6.75
  let child_count : ℕ := 48
  let adult_count : ℕ := child_count - 20
  let total_receipts : ℚ := child_price * child_count + adult_price * adult_count
  total_receipts = 405 := by sorry

end movie_theater_receipts_l454_45465


namespace cosine_difference_l454_45478

theorem cosine_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end cosine_difference_l454_45478


namespace ratio_determination_l454_45435

/-- Given constants a and b, and unknowns x and y, the equation
    ax³ + bx²y + bxy² + ay³ = 0 can be transformed into a polynomial
    equation in terms of t, where t = x/y. -/
theorem ratio_determination (a b x y : ℝ) :
  ∃ t, t = x / y ∧ a * t^3 + b * t^2 + b * t + a = 0 :=
by sorry

end ratio_determination_l454_45435


namespace plane_equation_correct_l454_45416

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the coefficients of a plane equation Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- The plane equation we want to prove -/
def targetEquation : PlaneEquation :=
  { A := 15, B := 7, C := 17, D := -26 }

/-- The three given points -/
def p1 : Point3D := { x := 2, y := -3, z := 1 }
def p2 : Point3D := { x := -1, y := 1, z := 2 }
def p3 : Point3D := { x := 4, y := 0, z := -2 }

theorem plane_equation_correct :
  (pointOnPlane p1 targetEquation) ∧
  (pointOnPlane p2 targetEquation) ∧
  (pointOnPlane p3 targetEquation) ∧
  (targetEquation.A > 0) ∧
  (Nat.gcd (Nat.gcd (Int.natAbs targetEquation.A) (Int.natAbs targetEquation.B))
           (Nat.gcd (Int.natAbs targetEquation.C) (Int.natAbs targetEquation.D)) = 1) :=
by sorry


end plane_equation_correct_l454_45416


namespace empty_solution_set_implies_a_range_l454_45412

theorem empty_solution_set_implies_a_range :
  (∀ x : ℝ, |x - 1| - |x - 2| ≤ 1) →
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  a ∈ Set.Iio (-1) ∪ Set.Ioi 0 := by
  sorry

end empty_solution_set_implies_a_range_l454_45412


namespace tax_savings_proof_l454_45403

def original_tax_rate : ℚ := 40 / 100
def new_tax_rate : ℚ := 33 / 100
def annual_income : ℚ := 45000

def differential_savings : ℚ := original_tax_rate * annual_income - new_tax_rate * annual_income

theorem tax_savings_proof : differential_savings = 3150 := by
  sorry

end tax_savings_proof_l454_45403


namespace max_value_complex_l454_45438

theorem max_value_complex (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z^3 + 3*z + Complex.I*2) ≤ 3 * Real.sqrt 3 := by
  sorry

end max_value_complex_l454_45438


namespace coin_flip_probability_l454_45474

theorem coin_flip_probability : 
  let n : ℕ := 12  -- total number of flips
  let k : ℕ := 9   -- number of heads we want
  let p : ℚ := 1/2 -- probability of heads for a fair coin
  Nat.choose n k * p^k * (1-p)^(n-k) = 55/1024 := by sorry

end coin_flip_probability_l454_45474


namespace eulers_formula_l454_45498

/-- A convex polyhedron is represented by its number of vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Euler's formula for convex polyhedra -/
theorem eulers_formula (p : ConvexPolyhedron) : p.vertices + p.faces = p.edges + 2 := by
  sorry

end eulers_formula_l454_45498


namespace point_on_y_axis_m_zero_l454_45453

/-- A point P with coordinates (x, y) lies on the y-axis if and only if x = 0 -/
def lies_on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0

/-- The theorem states that if a point P(m,2) lies on the y-axis, then m = 0 -/
theorem point_on_y_axis_m_zero (m : ℝ) :
  lies_on_y_axis (m, 2) → m = 0 := by
  sorry

end point_on_y_axis_m_zero_l454_45453


namespace square_sum_zero_implies_both_zero_l454_45471

theorem square_sum_zero_implies_both_zero (a b : ℝ) : a^2 + b^2 = 0 → a = 0 ∧ b = 0 := by
  sorry

end square_sum_zero_implies_both_zero_l454_45471


namespace football_players_count_l454_45434

theorem football_players_count (cricket_players hockey_players softball_players total_players : ℕ) 
  (h1 : cricket_players = 22)
  (h2 : hockey_players = 15)
  (h3 : softball_players = 19)
  (h4 : total_players = 77) :
  total_players - (cricket_players + hockey_players + softball_players) = 21 := by
  sorry

end football_players_count_l454_45434


namespace right_triangle_arctan_sum_l454_45469

/-- In a right-angled triangle ABC, the sum of arctan(b/(a+c)) and arctan(c/(a+b)) is equal to π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let triangle_abc : (ℝ × ℝ × ℝ) := (a, b, c)
  b^2 + c^2 = a^2 →
  Real.arctan (b / (a + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l454_45469


namespace max_elevation_l454_45452

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ t : ℝ, ∀ u : ℝ, s u ≤ s t ∧ s t = 500 := by
  sorry

end max_elevation_l454_45452


namespace store_discount_percentage_l454_45470

/-- Represents the pricing strategy and profit of a store selling turtleneck sweaters -/
theorem store_discount_percentage (C : ℝ) (D : ℝ) : 
  C > 0 → -- Cost price is positive
  (1.20 * C) * 1.25 * (1 - D / 100) = 1.35 * C → -- February selling price equals 35% profit
  D = 10 := by
  sorry

end store_discount_percentage_l454_45470


namespace pythagorean_proof_depends_on_parallel_postulate_l454_45423

-- Define Euclidean geometry
class EuclideanGeometry where
  -- Assume the existence of parallel postulate
  parallel_postulate : Prop

-- Define the concept of a direct proof of the Pythagorean theorem
class PythagoreanProof (E : EuclideanGeometry) where
  -- The proof uses similarity of triangles
  uses_triangle_similarity : Prop
  -- The proof uses equivalency of areas
  uses_area_equivalence : Prop

-- Theorem statement
theorem pythagorean_proof_depends_on_parallel_postulate 
  (E : EuclideanGeometry) 
  (P : PythagoreanProof E) : 
  E.parallel_postulate → 
  (P.uses_triangle_similarity ∨ P.uses_area_equivalence) → 
  -- The proof depends on the parallel postulate
  Prop :=
sorry

end pythagorean_proof_depends_on_parallel_postulate_l454_45423


namespace solution_set_quadratic_inequality_l454_45447

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - 3*x - 4 ≤ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 4} := by sorry

end solution_set_quadratic_inequality_l454_45447


namespace triangle_max_third_side_l454_45437

theorem triangle_max_third_side (a b : ℝ) (ha : a = 5) (hb : b = 10) :
  ∃ (c : ℕ), c = 14 ∧ 
  (∀ (x : ℕ), x > c → ¬(a + b > x ∧ b + x > a ∧ x + a > b)) :=
by sorry

end triangle_max_third_side_l454_45437


namespace integer_pair_solution_l454_45426

theorem integer_pair_solution (m n : ℤ) :
  (m - n)^2 = 4 * m * n / (m + n - 1) →
  ∃ k : ℕ, k ≠ 1 ∧
    ((m = (k^2 + k) / 2 ∧ n = (k^2 - k) / 2) ∨
     (m = (k^2 - k) / 2 ∧ n = (k^2 + k) / 2)) :=
by sorry

end integer_pair_solution_l454_45426


namespace max_d_value_l454_45430

def a (n : ℕ+) : ℕ := 100 + 2 * n ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ k : ℕ+, d k = 49) ∧ (∀ n : ℕ+, d n ≤ 49) :=
sorry

end max_d_value_l454_45430


namespace relationship_abcd_l454_45472

theorem relationship_abcd (a b c d : ℝ) :
  (a + 2*b) / (2*b + c) = (c + 2*d) / (2*d + a) →
  (a = c ∨ a + c + 2*(b + d) = 0) :=
by sorry

end relationship_abcd_l454_45472


namespace function_value_at_e_l454_45490

open Real

theorem function_value_at_e (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f x = 2 * (deriv f 1) * log x + x) →
  f (exp 1) = -2 + exp 1 := by
sorry

end function_value_at_e_l454_45490


namespace penny_draw_probability_l454_45457

/-- The number of shiny pennies in the box -/
def shiny_pennies : ℕ := 5

/-- The number of dull pennies in the box -/
def dull_pennies : ℕ := 3

/-- The total number of pennies in the box -/
def total_pennies : ℕ := shiny_pennies + dull_pennies

/-- The probability of needing more than five draws to get the fourth shiny penny -/
def probability : ℚ := 31 / 56

theorem penny_draw_probability :
  probability = (Nat.choose 5 3 * Nat.choose 3 1 + Nat.choose 5 0 * Nat.choose 3 3) / Nat.choose total_pennies shiny_pennies ∧
  probability.num + probability.den = 87 := by sorry

end penny_draw_probability_l454_45457


namespace circle_symmetry_theorem_l454_45483

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 4*y = 0

-- Define the line of symmetry
def symmetry_line (k b : ℝ) (x y : ℝ) : Prop := y = k*x + b

-- Define the symmetric circle centered at the origin
def symmetric_circle (x y : ℝ) : Prop := x^2 + y^2 = 20

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  symmetric_circle A.1 A.2 ∧ symmetric_circle B.1 B.2

-- Define the angle ACB
def angle_ACB (A B : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_symmetry_theorem :
  ∃ (k b : ℝ) (A B : ℝ × ℝ),
    (∀ x y : ℝ, circle_C x y ↔ symmetric_circle (2*x - k*y + b) (2*y + k*x - k*b)) →
    k = 2 ∧ b = 5 ∧
    intersection_points A B ∧
    angle_ACB A B = 120 := by sorry

end circle_symmetry_theorem_l454_45483


namespace expression_equals_eight_l454_45485

theorem expression_equals_eight :
  ((18^18 / 18^17)^3 * 9^3) / 3^6 = 8 := by
  sorry

end expression_equals_eight_l454_45485


namespace oil_weight_in_salad_dressing_salad_dressing_oil_weight_l454_45413

/-- Calculates the weight of oil per ml in a salad dressing mixture --/
theorem oil_weight_in_salad_dressing 
  (bowl_capacity : ℝ) 
  (oil_proportion : ℝ) 
  (vinegar_proportion : ℝ) 
  (vinegar_weight : ℝ) 
  (total_weight : ℝ) : ℝ :=
  let oil_volume := bowl_capacity * oil_proportion
  let vinegar_volume := bowl_capacity * vinegar_proportion
  let vinegar_total_weight := vinegar_volume * vinegar_weight
  let oil_total_weight := total_weight - vinegar_total_weight
  oil_total_weight / oil_volume

/-- Proves that the weight of oil in the given salad dressing mixture is 5 g/ml --/
theorem salad_dressing_oil_weight :
  oil_weight_in_salad_dressing 150 (2/3) (1/3) 4 700 = 5 := by
  sorry

end oil_weight_in_salad_dressing_salad_dressing_oil_weight_l454_45413


namespace quadratic_inequality_solution_range_l454_45411

theorem quadratic_inequality_solution_range (k : ℝ) :
  (k > 0) →
  (∃ x : ℝ, x^2 - 8*x + k < 0) ↔ (k < 16) :=
sorry

end quadratic_inequality_solution_range_l454_45411


namespace right_triangle_side_length_l454_45450

/-- In a right triangle LMN, given cos M and the length of LM, we can determine the length of LN. -/
theorem right_triangle_side_length 
  (L M N : ℝ × ℝ) 
  (right_angle_M : (N.1 - M.1) * (L.2 - M.2) = (L.1 - M.1) * (N.2 - M.2)) 
  (cos_M : Real.cos (Real.arctan ((L.2 - M.2) / (L.1 - M.1))) = 3/5) 
  (LM_length : Real.sqrt ((L.1 - M.1)^2 + (L.2 - M.2)^2) = 15) :
  Real.sqrt ((L.1 - N.1)^2 + (L.2 - N.2)^2) = 9 := by
    sorry


end right_triangle_side_length_l454_45450


namespace remainder_9_1995_mod_7_l454_45428

theorem remainder_9_1995_mod_7 : 9^1995 % 7 = 1 := by
  sorry

end remainder_9_1995_mod_7_l454_45428


namespace return_speed_l454_45433

/-- Given two towns and a person's travel speeds, calculate the return speed -/
theorem return_speed (d : ℝ) (v_xy v_total : ℝ) (h1 : v_xy = 54) (h2 : v_total = 43.2) :
  let v_yx := 2 * v_total * v_xy / (2 * v_xy - v_total)
  v_yx = 36 := by sorry

end return_speed_l454_45433


namespace recurring_decimal_to_fraction_l454_45429

theorem recurring_decimal_to_fraction : 
  (0.3 : ℚ) + (23 : ℚ) / 99 = 527 / 990 := by sorry

end recurring_decimal_to_fraction_l454_45429


namespace distance_point_to_line_l454_45480

/-- The distance from a point to a vertical line -/
def distance_point_to_vertical_line (point : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |point.1 - line_x|

/-- Theorem: The distance from point (1, 2) to the line x = -2 is 3 -/
theorem distance_point_to_line : distance_point_to_vertical_line (1, 2) (-2) = 3 := by
  sorry

end distance_point_to_line_l454_45480


namespace three_dollar_two_l454_45406

-- Define the custom operation $
def dollar (a b : ℕ) : ℕ := a^2 * (b + 1) + a * b

-- Theorem statement
theorem three_dollar_two : dollar 3 2 = 33 := by
  sorry

end three_dollar_two_l454_45406


namespace students_enjoying_both_sports_l454_45468

theorem students_enjoying_both_sports 
  (total : ℕ) 
  (running : ℕ) 
  (basketball : ℕ) 
  (neither : ℕ) 
  (h1 : total = 38) 
  (h2 : running = 21) 
  (h3 : basketball = 15) 
  (h4 : neither = 10) :
  running + basketball - (total - neither) = 8 :=
by sorry

end students_enjoying_both_sports_l454_45468


namespace perpendicular_vectors_m_collinear_vectors_k_l454_45493

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 3)
def c (m : ℝ) : ℝ × ℝ := (-2, m)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector addition for 2D vectors
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication for 2D vectors
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define collinearity for 2D vectors
def collinear (v w : ℝ × ℝ) : Prop := ∃ (k : ℝ), v = scalar_mult k w

-- Theorem 1
theorem perpendicular_vectors_m (m : ℝ) : 
  dot_product a (vector_add b (c m)) = 0 → m = -1 := by sorry

-- Theorem 2
theorem collinear_vectors_k (k : ℝ) :
  collinear (vector_add (scalar_mult k a) b) (vector_add (scalar_mult 2 a) (scalar_mult (-1) b)) → k = -2 := by sorry

end perpendicular_vectors_m_collinear_vectors_k_l454_45493


namespace money_problem_l454_45497

theorem money_problem (c d : ℝ) 
  (h1 : 3 * c - 2 * d < 30)
  (h2 : 4 * c + d = 60) :
  c < 150 / 11 ∧ d > 60 / 11 := by
  sorry

end money_problem_l454_45497


namespace rectangle_area_transformation_l454_45442

theorem rectangle_area_transformation (A : ℝ) : 
  (12 + 3) * (12 - A) = 120 → A = 4 := by sorry

end rectangle_area_transformation_l454_45442


namespace larger_number_twice_smaller_l454_45486

theorem larger_number_twice_smaller (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a - b/2 = 3 * (b - b/2)) : a = 2 * b := by
  sorry

end larger_number_twice_smaller_l454_45486


namespace range_of_a_l454_45421

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l454_45421


namespace newspaper_price_calculation_l454_45448

/-- The price of each Wednesday, Thursday, and Friday edition of the newspaper -/
def weekday_price : ℚ := 1/2

theorem newspaper_price_calculation :
  let weeks : ℕ := 8
  let weekday_editions_per_week : ℕ := 3
  let sunday_price : ℚ := 2
  let total_spent : ℚ := 28
  weekday_price = (total_spent - (sunday_price * weeks)) / (weekday_editions_per_week * weeks) :=
by
  sorry

#eval weekday_price

end newspaper_price_calculation_l454_45448


namespace merchant_markup_percentage_l454_45492

/-- The percentage of the list price at which goods should be marked to achieve
    the desired profit and discount conditions. -/
theorem merchant_markup_percentage 
  (list_price : ℝ) 
  (purchase_discount : ℝ) 
  (selling_discount : ℝ) 
  (profit_percentage : ℝ) 
  (h1 : purchase_discount = 0.2)
  (h2 : selling_discount = 0.2)
  (h3 : profit_percentage = 0.2)
  : ∃ (markup_percentage : ℝ),
    markup_percentage = 1.25 ∧ 
    (1 - purchase_discount) * list_price = 
    (1 - profit_percentage) * ((1 - selling_discount) * (markup_percentage * list_price)) :=
by sorry

end merchant_markup_percentage_l454_45492


namespace exponent_calculation_l454_45482

theorem exponent_calculation (a : ℝ) : (-a)^10 / (-a)^3 = -a^7 := by sorry

end exponent_calculation_l454_45482


namespace abc_value_l454_45459

theorem abc_value (a b c : ℂ) 
  (eq1 : a * b + 5 * b = -20)
  (eq2 : b * c + 5 * c = -20)
  (eq3 : c * a + 5 * a = -20) :
  a * b * c = 100 := by
  sorry

end abc_value_l454_45459


namespace factor_probability_l454_45414

/-- The number of consecutive natural numbers in the set -/
def n : ℕ := 120

/-- The factorial we're considering -/
def f : ℕ := 5

/-- The number of factors of f! -/
def num_factors : ℕ := 16

/-- The probability of selecting a factor of f! from the set of n consecutive natural numbers -/
def probability : ℚ := num_factors / n

theorem factor_probability : probability = 2 / 15 := by
  sorry

end factor_probability_l454_45414


namespace sum_of_abs_sum_and_diff_lt_two_l454_45408

theorem sum_of_abs_sum_and_diff_lt_two (a b : ℝ) : 
  (|a| < 1) → (|b| < 1) → (|a + b| + |a - b| < 2) := by
sorry

end sum_of_abs_sum_and_diff_lt_two_l454_45408


namespace smallest_n_is_five_l454_45461

/-- A triple of positive integers (x, y, z) such that x + y = 3z -/
structure SpecialTriple where
  x : ℕ+
  y : ℕ+
  z : ℕ+
  sum_condition : x + y = 3 * z

/-- The property that a positive integer n satisfies the condition -/
def SatisfiesCondition (n : ℕ+) : Prop :=
  ∃ (triples : Fin n → SpecialTriple),
    (∀ i j, i ≠ j → (triples i).x ≠ (triples j).x ∧ (triples i).y ≠ (triples j).y ∧ (triples i).z ≠ (triples j).z) ∧
    (∀ k : ℕ+, k ≤ 3*n → ∃ i, (triples i).x = k ∨ (triples i).y = k ∨ (triples i).z = k)

theorem smallest_n_is_five :
  SatisfiesCondition 5 ∧ ∀ m : ℕ+, m < 5 → ¬SatisfiesCondition m :=
by sorry

end smallest_n_is_five_l454_45461


namespace sequence_formulas_correct_l454_45449

def sequence1 (n : ℕ) : ℚ := 1 / (n * (n + 1))

def sequence2 (n : ℕ) : ℕ := 2^(n - 1)

def sequence3 (n : ℕ) : ℚ := 4 / (3 * n + 2)

theorem sequence_formulas_correct :
  (∀ n : ℕ, n > 0 → sequence1 n = 1 / (n * (n + 1))) ∧
  (∀ n : ℕ, n > 0 → sequence2 n = 2^(n - 1)) ∧
  (∀ n : ℕ, n > 0 → sequence3 n = 4 / (3 * n + 2)) :=
by sorry

end sequence_formulas_correct_l454_45449


namespace store_prices_existence_l454_45476

theorem store_prices_existence (S : ℕ) (h : S ≥ 100) :
  ∃ (T C B P : ℕ), T > C ∧ C > B ∧ T + C + B = S ∧ T * C * B = P ∧
  ∃ (T' C' B' : ℕ), (T', C', B') ≠ (T, C, B) ∧
    T' > C' ∧ C' > B' ∧ T' + C' + B' = S ∧ T' * C' * B' = P :=
by sorry

end store_prices_existence_l454_45476


namespace ian_hourly_rate_l454_45451

/-- Represents Ian's survey work and earnings -/
structure SurveyWork where
  hours_worked : ℕ
  money_left : ℕ
  spend_ratio : ℚ

/-- Calculates Ian's hourly rate given his survey work details -/
def hourly_rate (work : SurveyWork) : ℚ :=
  (work.money_left / (1 - work.spend_ratio)) / work.hours_worked

/-- Theorem stating that Ian's hourly rate is $18 -/
theorem ian_hourly_rate :
  let work : SurveyWork := {
    hours_worked := 8,
    money_left := 72,
    spend_ratio := 1/2
  }
  hourly_rate work = 18 := by sorry

end ian_hourly_rate_l454_45451


namespace neighborhood_cable_cost_l454_45477

/-- Represents the neighborhood cable layout problem -/
structure NeighborhoodCable where
  east_west_streets : Nat
  east_west_length : Nat
  north_south_streets : Nat
  north_south_length : Nat
  cable_per_mile : Nat
  cable_cost_per_mile : Nat

/-- Calculates the total cost of cable for the neighborhood -/
def total_cable_cost (n : NeighborhoodCable) : Nat :=
  let total_street_length := n.east_west_streets * n.east_west_length + n.north_south_streets * n.north_south_length
  let total_cable_length := total_street_length * n.cable_per_mile
  total_cable_length * n.cable_cost_per_mile

/-- The theorem stating the total cost of cable for the given neighborhood -/
theorem neighborhood_cable_cost :
  let n : NeighborhoodCable := {
    east_west_streets := 18,
    east_west_length := 2,
    north_south_streets := 10,
    north_south_length := 4,
    cable_per_mile := 5,
    cable_cost_per_mile := 2000
  }
  total_cable_cost n = 760000 := by
  sorry

end neighborhood_cable_cost_l454_45477


namespace sum_of_two_numbers_l454_45454

theorem sum_of_two_numbers (x : ℤ) : 
  x + 35 = 62 → x = 27 := by
  sorry

end sum_of_two_numbers_l454_45454


namespace unique_a_in_A_l454_45496

def A (a : ℝ) : Set ℝ := {a + 2, (a + 1)^2, a^2 + 3*a + 3}

theorem unique_a_in_A : ∃! a : ℝ, 1 ∈ A a := by sorry

end unique_a_in_A_l454_45496
