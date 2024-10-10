import Mathlib

namespace alice_stops_l3421_342143

/-- Represents the coefficients of a quadratic equation ax² + bx + c = 0 -/
structure QuadraticCoeffs where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Transformation rule for the quadratic coefficients -/
def transform (q : QuadraticCoeffs) : QuadraticCoeffs :=
  { a := q.b + q.c
  , b := q.c + q.a
  , c := q.a + q.b }

/-- Sequence of quadratic coefficients after n transformations -/
def coeff_seq (q₀ : QuadraticCoeffs) : ℕ → QuadraticCoeffs
  | 0 => q₀
  | n + 1 => transform (coeff_seq q₀ n)

/-- Predicate to check if a quadratic equation has real roots -/
def has_real_roots (q : QuadraticCoeffs) : Prop :=
  q.b ^ 2 ≥ 4 * q.a * q.c

/-- Main theorem: Alice will stop after a finite number of moves -/
theorem alice_stops (q₀ : QuadraticCoeffs)
  (h₁ : (q₀.a + q₀.c) * q₀.b > 0) :
  ∃ k : ℕ, ¬(has_real_roots (coeff_seq q₀ k)) := by
  sorry

end alice_stops_l3421_342143


namespace matrix_power_2023_l3421_342147

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 3, 1]

theorem matrix_power_2023 : 
  A ^ 2023 = !![1, 0; 6069, 1] := by sorry

end matrix_power_2023_l3421_342147


namespace function_decreasing_condition_l3421_342130

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- State the theorem
theorem function_decreasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ici 2, ∀ y ∈ Set.Ici 2, x < y → f a x > f a y) ↔ a ≤ -1/2 := by
  sorry

end function_decreasing_condition_l3421_342130


namespace triangle_count_on_circle_l3421_342135

theorem triangle_count_on_circle (n : ℕ) (h : n = 10) : 
  (n.choose 3) = 120 := by
  sorry

end triangle_count_on_circle_l3421_342135


namespace arcsin_sqrt2_over_2_l3421_342121

theorem arcsin_sqrt2_over_2 : Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end arcsin_sqrt2_over_2_l3421_342121


namespace second_number_proof_l3421_342149

theorem second_number_proof (x : ℕ) : 
  (∃ k : ℕ, 60 = 18 * k + 6) →
  (∃ m : ℕ, x = 18 * m + 10) →
  (∀ d : ℕ, d > 18 → (d ∣ 60 ∧ d ∣ x) → False) →
  x > 60 →
  x = 64 := by
sorry

end second_number_proof_l3421_342149


namespace unique_digit_divisibility_l3421_342142

def is_divisible (n m : ℕ) : Prop := ∃ k : ℕ, n = m * k

theorem unique_digit_divisibility :
  ∃! B : ℕ,
    B < 10 ∧
    let number := 658274 * 10 + B
    is_divisible number 2 ∧
    is_divisible number 4 ∧
    is_divisible number 5 ∧
    is_divisible number 7 ∧
    is_divisible number 8 :=
by sorry

end unique_digit_divisibility_l3421_342142


namespace team_selection_ways_l3421_342165

def boys : ℕ := 10
def girls : ℕ := 10
def team_size : ℕ := 8
def boys_in_team : ℕ := team_size / 2
def girls_in_team : ℕ := team_size / 2

theorem team_selection_ways : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 44100 := by
  sorry

end team_selection_ways_l3421_342165


namespace man_speed_in_still_water_l3421_342113

/-- Represents the speed of a swimmer in still water and the speed of the stream -/
structure SwimmingSpeed where
  manSpeed : ℝ  -- Speed of the man in still water
  streamSpeed : ℝ  -- Speed of the stream

/-- Calculates the effective speed for downstream swimming -/
def downstreamSpeed (s : SwimmingSpeed) : ℝ := s.manSpeed + s.streamSpeed

/-- Calculates the effective speed for upstream swimming -/
def upstreamSpeed (s : SwimmingSpeed) : ℝ := s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 5 km/h -/
theorem man_speed_in_still_water :
  ∀ s : SwimmingSpeed,
    (downstreamSpeed s * 4 = 24) →  -- Downstream condition
    (upstreamSpeed s * 5 = 20) →    -- Upstream condition
    s.manSpeed = 5 := by
  sorry

end man_speed_in_still_water_l3421_342113


namespace double_age_in_two_years_l3421_342139

/-- Calculates the number of years until a man's age is twice his son's age -/
def years_until_double_age (man_age_difference : ℕ) (son_current_age : ℕ) : ℕ :=
  let man_current_age := son_current_age + man_age_difference
  2 * son_current_age + 2 - man_current_age

theorem double_age_in_two_years 
  (man_age_difference : ℕ) 
  (son_current_age : ℕ) 
  (h1 : man_age_difference = 25) 
  (h2 : son_current_age = 23) : 
  years_until_double_age man_age_difference son_current_age = 2 := by
sorry

#eval years_until_double_age 25 23

end double_age_in_two_years_l3421_342139


namespace system_solution_value_l3421_342157

theorem system_solution_value (a b x y : ℝ) : 
  x = 2 ∧ 
  y = 1 ∧ 
  a * x + b * y = 5 ∧ 
  b * x + a * y = 1 → 
  3 - a - b = 1 := by
sorry

end system_solution_value_l3421_342157


namespace negation_of_existence_proposition_l3421_342198

theorem negation_of_existence_proposition :
  (¬ ∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x^2 - x + c ≠ 0) :=
by sorry

end negation_of_existence_proposition_l3421_342198


namespace circle_containment_l3421_342162

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point is inside a circle if its distance from the center is less than the radius -/
def is_inside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

/-- A circle contains another circle's center if the center is inside the circle -/
def contains_center (c1 c2 : Circle) : Prop :=
  is_inside c2.center c1

theorem circle_containment (circles : Fin 6 → Circle) (O : ℝ × ℝ)
  (h : ∀ i : Fin 6, is_inside O (circles i)) :
  ∃ i j : Fin 6, i ≠ j ∧ contains_center (circles i) (circles j) := by
  sorry

end circle_containment_l3421_342162


namespace oil_water_ratio_l3421_342189

/-- Represents the capacity and contents of a bottle -/
structure Bottle where
  capacity : ℝ
  oil : ℝ
  water : ℝ

/-- The problem setup -/
def bottleProblem (C_A : ℝ) : Prop :=
  ∃ (A B C D : Bottle),
    A.capacity = C_A ∧
    A.oil = C_A / 2 ∧
    A.water = C_A / 2 ∧
    B.capacity = 2 * C_A ∧
    B.oil = C_A / 2 ∧
    B.water = 3 * C_A / 2 ∧
    C.capacity = 3 * C_A ∧
    C.oil = C_A ∧
    C.water = 2 * C_A ∧
    D.capacity = 4 * C_A ∧
    D.oil = 0 ∧
    D.water = 0

/-- The theorem to prove -/
theorem oil_water_ratio (C_A : ℝ) (h : C_A > 0) :
  bottleProblem C_A →
  ∃ (D_final : Bottle),
    D_final.capacity = 4 * C_A ∧
    D_final.oil = 2 * C_A ∧
    D_final.water = 3.7 * C_A :=
by
  sorry

#check oil_water_ratio

end oil_water_ratio_l3421_342189


namespace remainder_three_power_2023_mod_5_l3421_342195

theorem remainder_three_power_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end remainder_three_power_2023_mod_5_l3421_342195


namespace sqrt_expression_equals_three_l3421_342182

theorem sqrt_expression_equals_three :
  (Real.sqrt 2 + 1)^2 - Real.sqrt 18 + 2 * Real.sqrt (1/2) = 3 := by
  sorry

end sqrt_expression_equals_three_l3421_342182


namespace max_sum_squares_l3421_342152

theorem max_sum_squares (m n : ℕ) : 
  m ∈ Finset.range 1982 →
  n ∈ Finset.range 1982 →
  ((n^2 : ℤ) - m*n - m^2)^2 = 1 →
  m^2 + n^2 ≤ 3524578 :=
by sorry

end max_sum_squares_l3421_342152


namespace linear_function_property_l3421_342191

theorem linear_function_property :
  ∀ x y : ℝ, y = -2 * x + 1 → x > (1/2 : ℝ) → y < 0 := by
  sorry

end linear_function_property_l3421_342191


namespace time_to_draw_picture_l3421_342123

/-- Proves that the time to draw each picture is 2 hours -/
theorem time_to_draw_picture (num_pictures : ℕ) (coloring_ratio : ℚ) (total_time : ℚ) :
  num_pictures = 10 →
  coloring_ratio = 7/10 →
  total_time = 34 →
  ∃ (draw_time : ℚ), draw_time = 2 ∧ num_pictures * draw_time * (1 + coloring_ratio) = total_time :=
by sorry

end time_to_draw_picture_l3421_342123


namespace linda_remaining_candies_l3421_342163

-- Define the initial number of candies Linda has
def initial_candies : ℝ := 34.0

-- Define the number of candies Linda gave away
def candies_given : ℝ := 28.0

-- Define the number of candies Linda has left
def remaining_candies : ℝ := initial_candies - candies_given

-- Theorem statement
theorem linda_remaining_candies :
  remaining_candies = 6.0 := by sorry

end linda_remaining_candies_l3421_342163


namespace f_not_bounded_on_neg_reals_a_range_when_f_bounded_l3421_342193

-- Define the function f(x) = 1 + x + ax^2
def f (a : ℝ) (x : ℝ) : ℝ := 1 + x + a * x^2

-- Part 1: f(x) is not bounded on (-∞, 0) when a = -1
theorem f_not_bounded_on_neg_reals :
  ¬ ∃ (M : ℝ), ∀ (x : ℝ), x < 0 → |f (-1) x| ≤ M :=
sorry

-- Part 2: If |f(x)| ≤ 3 for all x ∈ [1, 4], then a ∈ [-1/2, -1/8]
theorem a_range_when_f_bounded (a : ℝ) :
  (∀ x, x ∈ Set.Icc 1 4 → |f a x| ≤ 3) →
  a ∈ Set.Icc (-1/2) (-1/8) :=
sorry

end f_not_bounded_on_neg_reals_a_range_when_f_bounded_l3421_342193


namespace fraction_evaluation_l3421_342109

theorem fraction_evaluation : (1/4 - 1/6) / (1/3 - 1/4) = 1 := by
  sorry

end fraction_evaluation_l3421_342109


namespace violin_enjoyment_misreporting_l3421_342170

/-- Represents the student population at Peculiar Academy -/
def TotalStudents : ℝ := 100

/-- Fraction of students who enjoy playing the violin -/
def EnjoyViolin : ℝ := 0.4

/-- Fraction of students who do not enjoy playing the violin -/
def DislikeViolin : ℝ := 0.6

/-- Fraction of violin-enjoying students who accurately state they enjoy it -/
def AccurateEnjoy : ℝ := 0.7

/-- Fraction of violin-enjoying students who falsely claim they do not enjoy it -/
def FalseDislike : ℝ := 0.3

/-- Fraction of violin-disliking students who correctly claim they dislike it -/
def AccurateDislike : ℝ := 0.8

/-- Fraction of violin-disliking students who mistakenly say they like it -/
def FalseLike : ℝ := 0.2

theorem violin_enjoyment_misreporting :
  let enjoy_but_say_dislike := EnjoyViolin * FalseDislike * TotalStudents
  let total_say_dislike := EnjoyViolin * FalseDislike * TotalStudents + DislikeViolin * AccurateDislike * TotalStudents
  enjoy_but_say_dislike / total_say_dislike = 1 / 5 := by
  sorry

end violin_enjoyment_misreporting_l3421_342170


namespace stratified_sampling_proportionality_l3421_342153

/-- Represents the number of students selected in stratified sampling -/
structure StratifiedSample where
  total : ℕ
  first_year : ℕ
  second_year : ℕ
  selected_first : ℕ
  selected_second : ℕ

/-- Checks if the stratified sample maintains proportionality -/
def is_proportional (s : StratifiedSample) : Prop :=
  s.selected_first * s.second_year = s.selected_second * s.first_year

theorem stratified_sampling_proportionality :
  ∀ s : StratifiedSample,
    s.total = 70 →
    s.first_year = 30 →
    s.second_year = 40 →
    s.selected_first = 6 →
    s.selected_second = 8 →
    is_proportional s :=
  sorry

end stratified_sampling_proportionality_l3421_342153


namespace katherines_bananas_l3421_342146

/-- Given Katherine's fruit inventory, calculate the number of bananas -/
theorem katherines_bananas (apples pears bananas total : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  total = apples + pears + bananas →
  total = 21 →
  bananas = 5 := by
sorry

end katherines_bananas_l3421_342146


namespace unique_solution_proof_l3421_342129

/-- The positive value of m for which the quadratic equation 4x^2 + mx + 4 = 0 has exactly one real solution -/
def unique_solution_m : ℝ := 8

/-- The quadratic equation 4x^2 + mx + 4 = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  4 * x^2 + m * x + 4 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  m^2 - 4 * 4 * 4

theorem unique_solution_proof :
  unique_solution_m > 0 ∧
  discriminant unique_solution_m = 0 ∧
  ∀ m : ℝ, m > 0 → discriminant m = 0 → m = unique_solution_m :=
by sorry

end unique_solution_proof_l3421_342129


namespace vector_parallelism_l3421_342181

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 1]
def b (x : ℝ) : Fin 2 → ℝ := ![2, x]

-- Define the condition for parallel vectors
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

-- State the theorem
theorem vector_parallelism (x : ℝ) :
  parallel (λ i => a i + b x i) (λ i => a i - b x i) → x = 2 := by
  sorry

end vector_parallelism_l3421_342181


namespace weight_loss_program_result_l3421_342104

/-- Calculates the final weight after a weight loss program -/
def final_weight (initial_weight : ℕ) (loss_rate1 : ℕ) (weeks1 : ℕ) (loss_rate2 : ℕ) (weeks2 : ℕ) : ℕ :=
  initial_weight - (loss_rate1 * weeks1 + loss_rate2 * weeks2)

/-- Theorem stating that the final weight after the given weight loss program is 222 pounds -/
theorem weight_loss_program_result :
  final_weight 250 3 4 2 8 = 222 := by
  sorry

end weight_loss_program_result_l3421_342104


namespace questionnaire_responses_l3421_342134

theorem questionnaire_responses (response_rate : ℝ) (min_questionnaires : ℝ) : 
  response_rate = 0.62 → min_questionnaires = 483.87 → 
  ⌊(⌈min_questionnaires⌉ : ℝ) * response_rate⌋ = 300 := by
sorry

end questionnaire_responses_l3421_342134


namespace division_with_special_remainder_l3421_342164

theorem division_with_special_remainder :
  ∃! (n : ℕ), n > 0 ∧ 
    ∃ (k m : ℕ), 
      180 = n * k + m ∧ 
      4 * m = k ∧ 
      m < n ∧ 
      n = 11 := by
  sorry

end division_with_special_remainder_l3421_342164


namespace parallel_vectors_x_value_l3421_342114

/-- Two vectors in R² are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Given vectors a and b, prove that if they are parallel, then x = 6 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  are_parallel a b → x = 6 :=
by
  sorry


end parallel_vectors_x_value_l3421_342114


namespace equation_solutions_l3421_342185

theorem equation_solutions : 
  (∃ x : ℝ, 2 * x + 62 = 248 ∧ x = 93) ∧
  (∃ x : ℝ, x - 12.7 = 2.7 ∧ x = 15.4) ∧
  (∃ x : ℝ, x / 5 = 0.16 ∧ x = 0.8) ∧
  (∃ x : ℝ, 7 * x + 2 * x = 6.3 ∧ x = 0.7) :=
by sorry

end equation_solutions_l3421_342185


namespace larger_number_proof_l3421_342199

theorem larger_number_proof (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := by
  sorry

end larger_number_proof_l3421_342199


namespace range_of_a_l3421_342122

/-- The function f(x) = x^2 - 2x --/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 --/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The closed interval [-1, 2] --/
def I : Set ℝ := Set.Icc (-1) 2

theorem range_of_a :
  ∀ a : ℝ, (a > 0 ∧
    (∀ x₁ ∈ I, ∃ x₀ ∈ I, g a x₁ = f x₀)) ↔
    (a ∈ Set.Ioo 0 (1/2)) :=
sorry

end range_of_a_l3421_342122


namespace special_arrangement_count_l3421_342106

/-- The number of ways to arrange n people in a row --/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k items from n items --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row,
    with the elderly people next to each other but not at the ends --/
def specialArrangement : ℕ :=
  choose 5 2 * linearArrangements 4 * 2

theorem special_arrangement_count : specialArrangement = 960 := by
  sorry

end special_arrangement_count_l3421_342106


namespace geometric_sequence_a4_l3421_342101

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a4 (a : ℕ → ℝ) :
  is_geometric_sequence a → a 2 = 4 → a 6 = 16 → a 4 = 8 :=
by sorry

end geometric_sequence_a4_l3421_342101


namespace multiple_solutions_and_no_solution_for_2891_l3421_342119

def equation (x y n : ℤ) : Prop := x^3 - 3*x*y^2 + y^3 = n

theorem multiple_solutions_and_no_solution_for_2891 :
  (∀ n : ℤ, (∃ x y : ℤ, equation x y n) → 
    (∃ a b c d : ℤ, equation a b n ∧ equation c d n ∧ 
      (a, b) ≠ (x, y) ∧ (c, d) ≠ (x, y) ∧ (a, b) ≠ (c, d))) ∧
  (¬ ∃ x y : ℤ, equation x y 2891) :=
by sorry

end multiple_solutions_and_no_solution_for_2891_l3421_342119


namespace deposit_growth_condition_l3421_342172

theorem deposit_growth_condition 
  (X r s : ℝ) 
  (h_X_pos : X > 0) 
  (h_s_bound : s < 20) :
  X * (1 + r / 100) * (1 - s / 100) > X ↔ r > 100 * s / (100 - s) := by
  sorry

end deposit_growth_condition_l3421_342172


namespace volleyball_team_selection_l3421_342177

theorem volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 880 :=
by sorry

end volleyball_team_selection_l3421_342177


namespace investment_timing_l3421_342184

/-- Given two investments A and B, where A invests for the full year and B invests for part of the year,
    prove that B's investment starts 6 months after A's if their total investment-months are equal. -/
theorem investment_timing (a_amount : ℝ) (b_amount : ℝ) (total_months : ℕ) (x : ℝ) :
  a_amount > 0 →
  b_amount > 0 →
  total_months = 12 →
  a_amount * total_months = b_amount * (total_months - x) →
  x = 6 := by
  sorry

end investment_timing_l3421_342184


namespace ferris_wheel_large_seats_undetermined_l3421_342115

structure FerrisWheel where
  smallSeats : Nat
  smallSeatCapacity : Nat
  largeSeatCapacity : Nat
  peopleOnSmallSeats : Nat

theorem ferris_wheel_large_seats_undetermined (fw : FerrisWheel)
  (h1 : fw.smallSeats = 2)
  (h2 : fw.smallSeatCapacity = 14)
  (h3 : fw.largeSeatCapacity = 54)
  (h4 : fw.peopleOnSmallSeats = 28) :
  ∀ n : Nat, ∃ m : Nat, m ≠ n ∧ 
    (∃ totalSeats totalCapacity : Nat,
      totalSeats = fw.smallSeats + n ∧
      totalCapacity = fw.smallSeats * fw.smallSeatCapacity + m * fw.largeSeatCapacity) :=
sorry

end ferris_wheel_large_seats_undetermined_l3421_342115


namespace odd_function_equivalence_l3421_342186

theorem odd_function_equivalence (f : ℝ → ℝ) : 
  (∀ x, f x + f (-x) = 0) ↔ (∀ x, f (-x) = -f x) :=
sorry

end odd_function_equivalence_l3421_342186


namespace polynomial_factorization_l3421_342136

theorem polynomial_factorization (x : ℝ) : 
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 9) = 
  (x^2 + 6*x + 3) * (x^2 + 6*x + 12) := by
  sorry

end polynomial_factorization_l3421_342136


namespace barbara_shopping_cost_l3421_342158

/-- The amount Barbara spent on goods other than tuna and water -/
def other_goods_cost (tuna_packs : ℕ) (tuna_price : ℚ) (water_bottles : ℕ) (water_price : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid - (tuna_packs : ℚ) * tuna_price - (water_bottles : ℚ) * water_price

/-- Theorem stating that Barbara spent $40 on goods other than tuna and water -/
theorem barbara_shopping_cost :
  other_goods_cost 5 2 4 (3/2) 56 = 40 := by
  sorry

end barbara_shopping_cost_l3421_342158


namespace tan_two_theta_l3421_342187

theorem tan_two_theta (θ : Real) 
  (h1 : π / 2 < θ ∧ θ < π) -- θ is an obtuse angle
  (h2 : Real.cos (2 * θ) - Real.sin (2 * θ) = (Real.cos θ)^2) :
  Real.tan (2 * θ) = 4 / 3 := by
sorry

end tan_two_theta_l3421_342187


namespace N_smallest_with_digit_sum_2021_sum_of_digits_N_plus_2021_l3421_342173

/-- The smallest positive integer whose digits sum to 2021 -/
def N : ℕ := sorry

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem stating the property of N -/
theorem N_smallest_with_digit_sum_2021 :
  (∀ m : ℕ, m < N → sum_of_digits m ≠ 2021) ∧
  sum_of_digits N = 2021 := by sorry

/-- Main theorem to prove -/
theorem sum_of_digits_N_plus_2021 :
  sum_of_digits (N + 2021) = 10 := by sorry

end N_smallest_with_digit_sum_2021_sum_of_digits_N_plus_2021_l3421_342173


namespace geometric_sum_equals_5592404_l3421_342144

/-- The sum of a geometric series with 11 terms, first term 4, and common ratio 4 -/
def geometricSum : ℕ :=
  4 * (1 - 4^11) / (1 - 4)

/-- Theorem stating that the geometric sum is equal to 5592404 -/
theorem geometric_sum_equals_5592404 : geometricSum = 5592404 := by
  sorry

end geometric_sum_equals_5592404_l3421_342144


namespace parallel_lines_theorem_l3421_342180

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel_line_line (l₁ l₂ : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect_planes (p₁ p₂ : Plane) (l : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem parallel_lines_theorem 
  (a b c : Line) 
  (α β : Plane) 
  (h_non_overlapping_lines : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_non_overlapping_planes : α ≠ β)
  (h1 : parallel_line_plane a α)
  (h2 : intersect_planes α β b)
  (h3 : line_in_plane a β) :
  parallel_line_line a b :=
sorry

end parallel_lines_theorem_l3421_342180


namespace contrapositive_real_roots_l3421_342117

theorem contrapositive_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + x - a ≠ 0) → a < 0 := by
  sorry

end contrapositive_real_roots_l3421_342117


namespace consecutive_integers_sum_l3421_342103

theorem consecutive_integers_sum (x : ℤ) : 
  x * (x + 1) * (x + 2) = 384 → x + (x + 1) + (x + 2) = 24 := by
sorry

end consecutive_integers_sum_l3421_342103


namespace interval_of_increase_l3421_342125

def f (x : ℝ) := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem interval_of_increase (x : ℝ) :
  StrictMonoOn f (Set.Iio (-2) ∪ Set.Ioi 1) :=
sorry

end interval_of_increase_l3421_342125


namespace chord_intersection_lengths_l3421_342116

theorem chord_intersection_lengths (r : ℝ) (AK CH : ℝ) :
  r = 7 →
  AK = 3 →
  CH = 12 →
  let KB := 2 * r - AK
  ∃ (CK KH : ℝ),
    CK + KH = CH ∧
    AK * KB = CK * KH ∧
    AK = 3 ∧
    KB = 11 := by
  sorry

end chord_intersection_lengths_l3421_342116


namespace cyclists_problem_l3421_342188

/-- The problem of two cyclists traveling between Huntington and Montauk -/
theorem cyclists_problem (x y : ℝ) : 
  (y = x + 6) →                   -- Y is 6 mph faster than X
  (80 / x = (80 + 16) / y) →      -- Time taken by X equals time taken by Y
  (x = 12) :=                     -- X's speed is 12 mph
by sorry

end cyclists_problem_l3421_342188


namespace fraction_evaluation_l3421_342132

theorem fraction_evaluation (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  ((1 / y^2) / (1 / x^2))^2 = 256 / 625 := by
  sorry

end fraction_evaluation_l3421_342132


namespace factorization_of_2a_squared_minus_8_l3421_342155

theorem factorization_of_2a_squared_minus_8 (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) := by
  sorry

end factorization_of_2a_squared_minus_8_l3421_342155


namespace total_revenue_is_628_l3421_342197

/-- Represents the characteristics of a pie type -/
structure PieType where
  slices_per_pie : ℕ
  price_per_slice : ℕ
  pies_sold : ℕ

/-- Calculates the revenue for a single pie type -/
def revenue_for_pie_type (pie : PieType) : ℕ :=
  pie.slices_per_pie * pie.price_per_slice * pie.pies_sold

/-- Defines the pumpkin pie -/
def pumpkin_pie : PieType :=
  { slices_per_pie := 8, price_per_slice := 5, pies_sold := 4 }

/-- Defines the custard pie -/
def custard_pie : PieType :=
  { slices_per_pie := 6, price_per_slice := 6, pies_sold := 5 }

/-- Defines the apple pie -/
def apple_pie : PieType :=
  { slices_per_pie := 10, price_per_slice := 4, pies_sold := 3 }

/-- Defines the pecan pie -/
def pecan_pie : PieType :=
  { slices_per_pie := 12, price_per_slice := 7, pies_sold := 2 }

/-- Theorem stating that the total revenue from all pie sales is $628 -/
theorem total_revenue_is_628 :
  revenue_for_pie_type pumpkin_pie +
  revenue_for_pie_type custard_pie +
  revenue_for_pie_type apple_pie +
  revenue_for_pie_type pecan_pie = 628 := by
  sorry

end total_revenue_is_628_l3421_342197


namespace total_seashells_is_58_l3421_342190

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The total number of seashells found -/
def total_seashells : ℕ := tom_seashells + fred_seashells

/-- Theorem stating that the total number of seashells found is 58 -/
theorem total_seashells_is_58 : total_seashells = 58 := by
  sorry

end total_seashells_is_58_l3421_342190


namespace lucas_sequence_property_l3421_342196

def L : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => L (n + 1) + L n

theorem lucas_sequence_property (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  p ∣ (L (2 * k) - 2) → p ∣ (L (2 * k + 1) - 1) :=
by sorry

end lucas_sequence_property_l3421_342196


namespace school_network_connections_l3421_342151

/-- The number of connections in a network of switches where each switch connects to a fixed number of others -/
def connections (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a network of 30 switches, where each switch connects to exactly 4 others, there are 60 connections -/
theorem school_network_connections :
  connections 30 4 = 60 := by
  sorry

end school_network_connections_l3421_342151


namespace total_distance_walked_l3421_342154

def first_part : ℝ := 0.75
def second_part : ℝ := 0.25

theorem total_distance_walked : first_part + second_part = 1 := by
  sorry

end total_distance_walked_l3421_342154


namespace cookie_price_equality_l3421_342167

/-- The radius of Art's circular cookies -/
def art_radius : ℝ := 2

/-- The side length of Roger's square cookies -/
def roger_side : ℝ := 4

/-- The number of cookies Art makes from one batch -/
def art_cookie_count : ℕ := 9

/-- The price of one of Art's cookies in cents -/
def art_cookie_price : ℕ := 50

/-- The price of one of Roger's cookies in cents -/
def roger_cookie_price : ℕ := 64

theorem cookie_price_equality :
  let art_total_area := art_cookie_count * Real.pi * art_radius^2
  let roger_cookie_area := roger_side^2
  let roger_cookie_count := art_total_area / roger_cookie_area
  art_cookie_count * art_cookie_price = ⌊roger_cookie_count⌋ * roger_cookie_price :=
sorry

end cookie_price_equality_l3421_342167


namespace dandelion_color_change_l3421_342112

/-- The number of dandelions that turned white in the first two days -/
def dandelions_turned_white_first_two_days : ℕ := 25

/-- The number of dandelions that will turn white on the fourth day -/
def dandelions_turn_white_fourth_day : ℕ := 9

/-- The total number of dandelions that have turned or will turn white over the four-day period -/
def total_white_dandelions : ℕ := dandelions_turned_white_first_two_days + dandelions_turn_white_fourth_day

theorem dandelion_color_change :
  total_white_dandelions = 34 := by sorry

end dandelion_color_change_l3421_342112


namespace sqrt_expressions_l3421_342107

theorem sqrt_expressions :
  (∀ x y z : ℝ, x = 8 ∧ y = 2 ∧ z = 18 → Real.sqrt x + Real.sqrt y - Real.sqrt z = 0) ∧
  (∀ a : ℝ, a = 3 → (Real.sqrt a - 2)^2 = 7 - 4 * Real.sqrt a) := by
  sorry

end sqrt_expressions_l3421_342107


namespace cricket_average_proof_l3421_342111

def average_runs (total_runs : ℕ) (innings : ℕ) : ℚ :=
  (total_runs : ℚ) / (innings : ℚ)

theorem cricket_average_proof 
  (initial_innings : ℕ) 
  (next_innings_runs : ℕ) 
  (average_increase : ℚ) :
  initial_innings = 10 →
  next_innings_runs = 74 →
  average_increase = 4 →
  ∃ (initial_total_runs : ℕ),
    average_runs (initial_total_runs + next_innings_runs) (initial_innings + 1) =
    average_runs initial_total_runs initial_innings + average_increase →
    average_runs initial_total_runs initial_innings = 30 := by
  sorry

end cricket_average_proof_l3421_342111


namespace quadratic_equation_game_l3421_342124

/-- Represents a strategy for playing the quadratic equation game -/
def Strategy := Nat → Nat → ℝ

/-- Represents the outcome of a game given two strategies -/
def GameOutcome (n : Nat) (s1 s2 : Strategy) : Nat := sorry

/-- The maximum number of equations without roots that Player 1 can guarantee -/
def MaxRootlessEquations (n : Nat) : Nat := (n + 1) / 2

theorem quadratic_equation_game (n : Nat) (h : Odd n) :
  ∃ (s1 : Strategy), ∀ (s2 : Strategy), GameOutcome n s1 s2 ≥ MaxRootlessEquations n :=
sorry

end quadratic_equation_game_l3421_342124


namespace arrangement_count_correct_l3421_342166

/-- The number of ways to arrange 4 students into 2 out of 6 classes, with 2 students in each chosen class -/
def arrangementCount : ℕ :=
  (Nat.choose 6 2 * Nat.factorial 2 * Nat.choose 4 2) / 2

/-- Theorem stating that the arrangement count is correct -/
theorem arrangement_count_correct :
  arrangementCount = (Nat.choose 6 2 * Nat.factorial 2 * Nat.choose 4 2) / 2 := by
  sorry

#eval arrangementCount

end arrangement_count_correct_l3421_342166


namespace intersection_line_l3421_342131

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 3)^2 = 100

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 6)^2 = 121

/-- Theorem stating that the line passing through the intersection points of the two circles
    has the equation x - y = -17/9 -/
theorem intersection_line : ∃ (x y : ℝ), 
  circle1 x y ∧ circle2 x y ∧ (x - y = -17/9) := by
  sorry

end intersection_line_l3421_342131


namespace emily_sees_emerson_time_l3421_342148

def emily_speed : ℝ := 15
def emerson_speed : ℝ := 9
def initial_distance : ℝ := 1
def final_distance : ℝ := 1

theorem emily_sees_emerson_time : 
  let relative_speed := emily_speed - emerson_speed
  let time_to_catch := initial_distance / relative_speed
  let time_to_lose_sight := final_distance / relative_speed
  let total_time := time_to_catch + time_to_lose_sight
  (total_time * 60) = 20 := by sorry

end emily_sees_emerson_time_l3421_342148


namespace rectangle_area_l3421_342161

/-- A rectangle with perimeter 40 and length twice its width has area 800/9 -/
theorem rectangle_area (w : ℝ) (h1 : w > 0) (h2 : 6 * w = 40) : w * (2 * w) = 800 / 9 := by
  sorry

end rectangle_area_l3421_342161


namespace negation_of_universal_positive_cubic_l3421_342176

theorem negation_of_universal_positive_cubic (x : ℝ) :
  (¬ ∀ x ≥ 0, x^3 + x > 0) ↔ (∃ x ≥ 0, x^3 + x ≤ 0) := by
  sorry

end negation_of_universal_positive_cubic_l3421_342176


namespace average_assembly_rate_l3421_342174

/-- Represents the car assembly problem with the given conditions -/
def CarAssemblyProblem (x : ℝ) : Prop :=
  let original_plan := 21
  let assembled_before_order := 6
  let additional_order := 5
  let increased_rate := x + 2
  (original_plan / x) - (assembled_before_order / x) - 
    ((original_plan - assembled_before_order + additional_order) / increased_rate) = 1

/-- Theorem stating that the average daily assembly rate after the additional order is 5 cars per day -/
theorem average_assembly_rate : ∃ x : ℝ, CarAssemblyProblem x ∧ x + 2 = 5 := by
  sorry

end average_assembly_rate_l3421_342174


namespace sampling_survey_more_appropriate_for_city_air_quality_l3421_342171

-- Define the city and survey types
def City : Type := Unit
def ComprehensiveSurvey : Type := Unit
def SamplingSurvey : Type := Unit

-- Define the properties of the city and surveys
def has_vast_area (c : City) : Prop := sorry
def has_varying_conditions (c : City) : Prop := sorry
def is_comprehensive (s : ComprehensiveSurvey) : Prop := sorry
def is_strategically_sampled (s : SamplingSurvey) : Prop := sorry

-- Define the concept of feasibility and appropriateness
def is_feasible (c : City) (s : ComprehensiveSurvey) : Prop := sorry
def is_appropriate (c : City) (s : SamplingSurvey) : Prop := sorry

-- Theorem stating that sampling survey is more appropriate for air quality testing in a city
theorem sampling_survey_more_appropriate_for_city_air_quality 
  (c : City) (comp_survey : ComprehensiveSurvey) (samp_survey : SamplingSurvey) :
  has_vast_area c →
  has_varying_conditions c →
  is_comprehensive comp_survey →
  is_strategically_sampled samp_survey →
  ¬(is_feasible c comp_survey) →
  is_appropriate c samp_survey :=
by sorry

end sampling_survey_more_appropriate_for_city_air_quality_l3421_342171


namespace bob_candy_count_l3421_342178

/-- Calculates Bob's share of candies given a total amount and a ratio --/
def bobShare (total : ℕ) (samRatio : ℕ) (bobRatio : ℕ) : ℕ :=
  (total / (samRatio + bobRatio)) * bobRatio

/-- The total number of candies Bob received --/
def bobTotalCandies : ℕ :=
  bobShare 45 2 3 + bobShare 60 3 1 + (45 / 2)

theorem bob_candy_count : bobTotalCandies = 64 := by
  sorry

#eval bobTotalCandies

end bob_candy_count_l3421_342178


namespace student_count_l3421_342138

/-- Given a student's position from both ends of a line, calculate the total number of students -/
theorem student_count (right_rank left_rank : ℕ) (h1 : right_rank = 13) (h2 : left_rank = 8) :
  right_rank + left_rank - 1 = 20 := by
  sorry

end student_count_l3421_342138


namespace charges_needed_equals_total_rooms_l3421_342128

def battery_duration : ℕ := 10
def vacuum_time_per_room : ℕ := 8
def num_bedrooms : ℕ := 3
def num_kitchen : ℕ := 1
def num_living_room : ℕ := 1
def num_dining_room : ℕ := 1
def num_office : ℕ := 1
def num_bathrooms : ℕ := 2

def total_rooms : ℕ := num_bedrooms + num_kitchen + num_living_room + num_dining_room + num_office + num_bathrooms

theorem charges_needed_equals_total_rooms :
  battery_duration > vacuum_time_per_room ∧
  battery_duration < 2 * vacuum_time_per_room →
  total_rooms = total_rooms :=
by sorry

end charges_needed_equals_total_rooms_l3421_342128


namespace log_equality_l3421_342183

theorem log_equality (y : ℝ) (k : ℝ) 
  (h1 : Real.log 3 / Real.log 8 = y)
  (h2 : Real.log 243 / Real.log 2 = k * y) : 
  k = 15 := by
sorry

end log_equality_l3421_342183


namespace earth_inhabitable_fraction_l3421_342140

/-- The fraction of Earth's surface that humans can inhabit -/
def inhabitable_fraction : ℚ := 1/4

theorem earth_inhabitable_fraction :
  (earth_land_fraction : ℚ) = 1/3 →
  (habitable_land_fraction : ℚ) = 3/4 →
  inhabitable_fraction = earth_land_fraction * habitable_land_fraction :=
by sorry

end earth_inhabitable_fraction_l3421_342140


namespace complex_magnitude_equation_l3421_342110

theorem complex_magnitude_equation (t : ℝ) (h : t > 0) :
  Complex.abs (8 + 2 * t * Complex.I) = 12 → t = 2 * Real.sqrt 5 := by
sorry

end complex_magnitude_equation_l3421_342110


namespace deck_size_l3421_342141

theorem deck_size (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/6 →
  r + b = 12 := by
sorry

end deck_size_l3421_342141


namespace journey_distance_l3421_342102

theorem journey_distance (train_fraction : ℚ) (bus_fraction : ℚ) (walk_distance : ℝ) :
  train_fraction = 3/5 →
  bus_fraction = 7/20 →
  walk_distance = 6.5 →
  1 - (train_fraction + bus_fraction) = walk_distance / 130 →
  130 = (walk_distance * 20 : ℝ) :=
by sorry

end journey_distance_l3421_342102


namespace num_unique_labelings_eq_30_l3421_342120

/-- A cube is a three-dimensional object with 6 faces. -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- A labeling of a cube is valid if it uses the numbers 1 to 6 exactly once each. -/
def is_valid_labeling (c : Cube) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 6 → n + 1 ∈ Finset.image c.faces Finset.univ) ∧
  (∀ f₁ f₂ : Fin 6, f₁ ≠ f₂ → c.faces f₁ ≠ c.faces f₂)

/-- Two labelings are equivalent up to rotation if they can be transformed into each other by rotating the cube. -/
def equivalent_up_to_rotation (c₁ c₂ : Cube) : Prop :=
  ∃ (perm : Equiv.Perm (Fin 6)), ∀ (f : Fin 6), c₁.faces f = c₂.faces (perm f)

/-- The number of unique labelings of a cube up to rotation -/
def num_unique_labelings : ℕ := sorry

theorem num_unique_labelings_eq_30 : num_unique_labelings = 30 := by
  sorry

end num_unique_labelings_eq_30_l3421_342120


namespace triangle_angle_A_l3421_342127

/-- Given a triangle ABC where C = π/3, b = √6, and c = 3, prove that A = 5π/12 -/
theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) : 
  C = π/3 → b = Real.sqrt 6 → c = 3 → 
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  A = 5*π/12 := by
  sorry

end triangle_angle_A_l3421_342127


namespace diamond_equation_solution_l3421_342160

/-- Custom binary operation ◇ -/
def diamond (a b : ℚ) : ℚ := a * b + 3 * b - 2 * a

/-- Theorem stating that if 4 ◇ y = 50, then y = 58/7 -/
theorem diamond_equation_solution :
  ∀ y : ℚ, diamond 4 y = 50 → y = 58 / 7 := by
  sorry

end diamond_equation_solution_l3421_342160


namespace solve_sales_problem_l3421_342145

def sales_problem (sales1 sales2 sales4 sales5 desired_average : ℕ) : Prop :=
  let total_months : ℕ := 5
  let known_sales : ℕ := sales1 + sales2 + sales4 + sales5
  let total_desired : ℕ := desired_average * total_months
  let sales3 : ℕ := total_desired - known_sales
  sales3 = 7570 ∧ 
  (sales1 + sales2 + sales3 + sales4 + sales5) / total_months = desired_average

theorem solve_sales_problem : 
  sales_problem 5420 5660 6350 6500 6300 := by
  sorry

end solve_sales_problem_l3421_342145


namespace lampshire_parade_group_size_l3421_342175

theorem lampshire_parade_group_size (n : ℕ) : 
  (∃ k : ℕ, n = 30 * k) →
  (30 * n) % 31 = 7 →
  (30 * n) % 17 = 0 →
  30 * n < 1500 →
  (∀ m : ℕ, 
    (∃ j : ℕ, m = 30 * j) →
    (30 * m) % 31 = 7 →
    (30 * m) % 17 = 0 →
    30 * m < 1500 →
    30 * m ≤ 30 * n) →
  30 * n = 1020 :=
by sorry

end lampshire_parade_group_size_l3421_342175


namespace opposite_numbers_solution_l3421_342168

theorem opposite_numbers_solution (x : ℚ) : (2 * x - 3 = -(1 - 4 * x)) → x = -1 := by
  sorry

end opposite_numbers_solution_l3421_342168


namespace hyperbola_asymptotes_l3421_342156

/-- Given a hyperbola with equation x²/9 - y²/b² = 1 and foci at (-5,0) and (5,0),
    prove that its asymptotes have the equation 4x ± 3y = 0 -/
theorem hyperbola_asymptotes (b : ℝ) (h1 : b > 0) (h2 : 9 + b^2 = 25) :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x y : ℝ), (x^2 / 9 - y^2 / b^2 = 1) → 
   ((4*x + 3*y = 0) ∨ (4*x - 3*y = 0))) :=
sorry

end hyperbola_asymptotes_l3421_342156


namespace max_points_difference_between_adjacent_teams_l3421_342100

/-- Represents a football league with the given properties -/
structure FootballLeague where
  num_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculates the maximum points a team can achieve in the league -/
def max_points (league : FootballLeague) : Nat :=
  (league.num_teams - 1) * 2 * league.points_for_win

/-- Calculates the minimum points a team can achieve in the league -/
def min_points (league : FootballLeague) : Nat :=
  (league.num_teams - 1) * 2 * league.points_for_draw

/-- Theorem stating the maximum points difference between adjacent teams -/
theorem max_points_difference_between_adjacent_teams 
  (league : FootballLeague) 
  (h1 : league.num_teams = 12)
  (h2 : league.points_for_win = 2)
  (h3 : league.points_for_draw = 1)
  (h4 : league.points_for_loss = 0) :
  max_points league - min_points league = 24 := by
  sorry


end max_points_difference_between_adjacent_teams_l3421_342100


namespace vector_inequality_l3421_342133

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Given two non-zero vectors a and b satisfying |a + b| = |b|, prove |2b| > |a + 2b| -/
theorem vector_inequality (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (h : ‖a + b‖ = ‖b‖) :
  ‖(2 : ℝ) • b‖ > ‖a + (2 : ℝ) • b‖ := by sorry

end vector_inequality_l3421_342133


namespace completing_square_equivalence_l3421_342118

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
sorry

end completing_square_equivalence_l3421_342118


namespace infinite_series_equality_l3421_342194

theorem infinite_series_equality (p q : ℝ) 
  (h : ∑' n, p / q^n = 5) :
  ∑' n, p / (p^2 + q)^n = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) := by
  sorry

end infinite_series_equality_l3421_342194


namespace joans_kittens_l3421_342169

theorem joans_kittens (given_away : ℕ) (remaining : ℕ) (original : ℕ) : 
  given_away = 2 → remaining = 6 → original = given_away + remaining :=
by
  sorry

end joans_kittens_l3421_342169


namespace unique_solution_l3421_342179

theorem unique_solution (x y z : ℝ) :
  (Real.sqrt (x^3 - y) = z - 1) ∧
  (Real.sqrt (y^3 - z) = x - 1) ∧
  (Real.sqrt (z^3 - x) = y - 1) →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end unique_solution_l3421_342179


namespace three_quarters_difference_l3421_342137

theorem three_quarters_difference (n : ℕ) (h : n = 76) : n - (3 * n / 4) = 19 := by
  sorry

end three_quarters_difference_l3421_342137


namespace melanie_plums_l3421_342159

/-- The number of plums Melanie has after giving some away -/
def plums_remaining (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Melanie has 4 plums after initially picking 7 and giving 3 away -/
theorem melanie_plums : plums_remaining 7 3 = 4 := by
  sorry

end melanie_plums_l3421_342159


namespace quadratic_function_properties_l3421_342126

/-- A quadratic function satisfying certain conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a ≠ 0 ∧
    (∀ x, f x = a * x^2 + b * x + c) ∧
    f (-1) = 0 ∧
    (∀ x, x ≤ f x ∧ f x ≤ (x^2 + 1) / 2) ∧
    {x : ℝ | |f x| < 1} = Set.Ioo (-1) 3

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties (f : ℝ → ℝ) (h : QuadraticFunction f) :
  (∀ x, f x = (1/4) * (x + 1)^2) ∧
  (∃ a : ℝ, (a > 0 ∧ a < 1/2) ∨ (a < 0 ∧ a > -1/2)) :=
by sorry

end quadratic_function_properties_l3421_342126


namespace smallest_three_digit_multiple_of_17_l3421_342105

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  n % 17 = 0 ∧ 
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ m % 17 = 0 → n ≤ m) ∧
  n = 102 := by
  sorry

end smallest_three_digit_multiple_of_17_l3421_342105


namespace ratio_ties_to_losses_l3421_342192

def total_games : ℕ := 56
def losses : ℕ := 12
def wins : ℕ := 38

def ties : ℕ := total_games - (losses + wins)

theorem ratio_ties_to_losses :
  (ties : ℚ) / losses = 1 / 2 := by sorry

end ratio_ties_to_losses_l3421_342192


namespace cake_frosting_theorem_l3421_342150

/-- Represents a person who can frost cakes -/
structure FrostingPerson where
  name : String
  frostingTime : ℕ

/-- Represents the cake frosting problem -/
structure CakeFrostingProblem where
  people : List FrostingPerson
  numCakes : ℕ
  passingTime : ℕ

/-- Calculates the minimum time to frost all cakes -/
def minFrostingTime (problem : CakeFrostingProblem) : ℕ :=
  sorry

theorem cake_frosting_theorem (problem : CakeFrostingProblem) :
  problem.people = [
    { name := "Ann", frostingTime := 8 },
    { name := "Bob", frostingTime := 6 },
    { name := "Carol", frostingTime := 10 }
  ] ∧
  problem.numCakes = 10 ∧
  problem.passingTime = 1
  →
  minFrostingTime problem = 116 := by
  sorry

end cake_frosting_theorem_l3421_342150


namespace gcd_lcm_product_24_36_l3421_342108

theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by sorry

end gcd_lcm_product_24_36_l3421_342108
