import Mathlib

namespace only_coordinates_specific_l3588_358854

/-- Represents a location description --/
inductive LocationDescription
  | CinemaRow (row : Nat)
  | StreetAddress (street : String) (city : String)
  | Direction (angle : Float) (direction : String)
  | Coordinates (longitude : Float) (latitude : Float)

/-- Determines if a location description provides a specific, unique location --/
def isSpecificLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.Coordinates _ _ => True
  | _ => False

/-- Theorem stating that only the coordinates option provides a specific location --/
theorem only_coordinates_specific (desc : LocationDescription) :
  isSpecificLocation desc ↔ ∃ (long lat : Float), desc = LocationDescription.Coordinates long lat :=
sorry

#check only_coordinates_specific

end only_coordinates_specific_l3588_358854


namespace complex_expression_evaluation_l3588_358841

theorem complex_expression_evaluation :
  (7 - 3*Complex.I) - 3*(2 + 4*Complex.I) + (1 + 2*Complex.I) = 2 - 13*Complex.I :=
by sorry

end complex_expression_evaluation_l3588_358841


namespace geometric_sequence_minimum_value_l3588_358875

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_geom : GeometricSequence a)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exists : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1 ∧
    ∀ k l : ℕ, k > 0 → l > 0 → (a k * a l).sqrt = 4 * a 1 →
      1 / m + 4 / n ≤ 1 / k + 4 / l) ∧
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ (a m * a n).sqrt = 4 * a 1 ∧
    1 / m + 4 / n = 3 / 2) :=
by sorry

end geometric_sequence_minimum_value_l3588_358875


namespace min_distance_between_curves_l3588_358864

/-- The minimum distance between a point on y = x^2 + 2 and a point on y = √(x - 2) -/
theorem min_distance_between_curves : ∃ (d : ℝ),
  d = (7 * Real.sqrt 2) / 4 ∧
  ∀ (xP yP xQ yQ : ℝ),
    yP = xP^2 + 2 →
    yQ = Real.sqrt (xQ - 2) →
    d ≤ Real.sqrt ((xP - xQ)^2 + (yP - yQ)^2) :=
by sorry

end min_distance_between_curves_l3588_358864


namespace cylinder_lateral_surface_area_l3588_358842

theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let circumference := 2 * Real.pi * r
  let height := circumference
  let lateral_area := circumference * height
  lateral_area = 4 * Real.pi * S := by
sorry

end cylinder_lateral_surface_area_l3588_358842


namespace neighbor_oranges_correct_l3588_358805

/-- The number of kilograms of oranges added for the neighbor -/
def neighbor_oranges : ℕ := 25

/-- The initial purchase of oranges in kilograms -/
def initial_purchase : ℕ := 10

/-- The total quantity of oranges bought over three weeks in kilograms -/
def total_quantity : ℕ := 75

/-- The quantity of oranges bought in each of the next two weeks -/
def next_weeks_purchase : ℕ := 2 * initial_purchase

theorem neighbor_oranges_correct :
  (initial_purchase + neighbor_oranges) + next_weeks_purchase + next_weeks_purchase = total_quantity :=
by sorry

end neighbor_oranges_correct_l3588_358805


namespace power_sum_l3588_358836

theorem power_sum (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 2) : a^(m+n) = 6 := by
  sorry

end power_sum_l3588_358836


namespace permutations_of_polarized_l3588_358837

theorem permutations_of_polarized (n : ℕ) (h : n = 9) :
  Nat.factorial n = 362880 := by
  sorry

end permutations_of_polarized_l3588_358837


namespace last_positive_term_is_six_l3588_358817

/-- Represents an arithmetic sequence with a given start and common difference. -/
structure ArithmeticSequence where
  start : ℤ
  diff : ℤ

/-- Calculates the nth term of an arithmetic sequence. -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.start + (n - 1 : ℤ) * seq.diff

/-- Theorem: The last term greater than 0 in the sequence (72, 61, 50, ...) is 6. -/
theorem last_positive_term_is_six :
  let seq := ArithmeticSequence.mk 72 (-11)
  ∃ n : ℕ, 
    (nthTerm seq n = 6) ∧ 
    (nthTerm seq n > 0) ∧ 
    (nthTerm seq (n + 1) ≤ 0) :=
by sorry

#check last_positive_term_is_six

end last_positive_term_is_six_l3588_358817


namespace regular_hexagon_area_l3588_358891

/-- The area of a regular hexagon with vertices A(0,0) and C(4,6) is 78√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (4, 6)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let hexagon_area : ℝ := 6 * (Real.sqrt 3 / 4 * AC^2)
  hexagon_area = 78 * Real.sqrt 3 := by
sorry


end regular_hexagon_area_l3588_358891


namespace sequence_inequality_l3588_358857

theorem sequence_inequality (a : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, a n ≥ 0)
  (h2 : ∀ n : ℕ, a n + a (2*n) ≥ 3*n)
  (h3 : ∀ n : ℕ, a (n+1) + n ≤ 2 * Real.sqrt (a n * (n+1))) :
  ∀ n : ℕ, a n ≥ n := by
  sorry

end sequence_inequality_l3588_358857


namespace gcd_228_1995_decimal_to_ternary_l3588_358807

-- Problem 1: GCD of 228 and 1995
theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by sorry

-- Problem 2: Convert 104 to base 3
theorem decimal_to_ternary :
  ∃ (a b c d e : Nat),
    104 = a * 3^4 + b * 3^3 + c * 3^2 + d * 3^1 + e * 3^0 ∧
    a = 1 ∧ b = 0 ∧ c = 2 ∧ d = 1 ∧ e = 2 := by sorry

end gcd_228_1995_decimal_to_ternary_l3588_358807


namespace city_population_l3588_358859

theorem city_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 85 / 100 →
  partial_population = 85000 →
  (percentage * total_population : ℚ) = partial_population →
  total_population = 100000 := by
sorry

end city_population_l3588_358859


namespace cos_squared_plus_sin_minus_one_range_l3588_358800

theorem cos_squared_plus_sin_minus_one_range :
  ∀ x : ℝ, -2 ≤ (Real.cos x)^2 + Real.sin x - 1 ∧ (Real.cos x)^2 + Real.sin x - 1 ≤ 1/4 := by
  sorry

end cos_squared_plus_sin_minus_one_range_l3588_358800


namespace subtracted_number_l3588_358833

theorem subtracted_number (x y : ℤ) : x = 48 → 5 * x - y = 102 → y = 138 := by
  sorry

end subtracted_number_l3588_358833


namespace geometric_sequence_common_ratio_l3588_358847

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = q * a n) 
  (S : ℕ → ℝ) 
  (h2 : ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) 
  (h3 : 2 * S 4 = S 5 + S 6) :
  q = -2 := by
sorry

end geometric_sequence_common_ratio_l3588_358847


namespace equation_solution_l3588_358838

theorem equation_solution :
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2 / 11 := by
  sorry

end equation_solution_l3588_358838


namespace sally_rum_amount_l3588_358889

theorem sally_rum_amount (x : ℝ) : 
  (∀ (max_rum : ℝ), max_rum = 3 * x) →   -- Maximum amount is 3 times what Sally gave
  (∀ (earlier_rum : ℝ), earlier_rum = 12) →  -- Don already had 12 oz
  (∀ (remaining_rum : ℝ), remaining_rum = 8) →  -- Don can still have 8 oz
  (x + 12 + 8 = 3 * x) →  -- Total amount equals maximum healthy amount
  x = 10 := by
sorry

end sally_rum_amount_l3588_358889


namespace binomial_coefficient_17_8_l3588_358899

theorem binomial_coefficient_17_8 (h1 : Nat.choose 15 6 = 5005) 
                                  (h2 : Nat.choose 15 7 = 6435) 
                                  (h3 : Nat.choose 15 8 = 6435) : 
  Nat.choose 17 8 = 24310 := by
  sorry

end binomial_coefficient_17_8_l3588_358899


namespace tea_in_milk_equals_milk_in_tea_l3588_358862

/-- Represents the contents of a cup --/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- Represents the state of both cups --/
structure CupState where
  tea_cup : Cup
  milk_cup : Cup

/-- Initial state of the cups --/
def initial_state : CupState :=
  { tea_cup := { tea := 5, milk := 0 },
    milk_cup := { tea := 0, milk := 5 } }

/-- State after transferring milk to tea cup --/
def after_milk_transfer (state : CupState) : CupState :=
  { tea_cup := { tea := state.tea_cup.tea, milk := state.tea_cup.milk + 1 },
    milk_cup := { tea := state.milk_cup.tea, milk := state.milk_cup.milk - 1 } }

/-- State after transferring mixture back to milk cup --/
def after_mixture_transfer (state : CupState) : CupState :=
  let total_in_tea_cup := state.tea_cup.tea + state.tea_cup.milk
  let tea_fraction := state.tea_cup.tea / total_in_tea_cup
  let milk_fraction := state.tea_cup.milk / total_in_tea_cup
  { tea_cup := { tea := state.tea_cup.tea - tea_fraction, 
                 milk := state.tea_cup.milk - milk_fraction },
    milk_cup := { tea := state.milk_cup.tea + tea_fraction, 
                  milk := state.milk_cup.milk + milk_fraction } }

/-- Final state after both transfers --/
def final_state : CupState :=
  after_mixture_transfer (after_milk_transfer initial_state)

theorem tea_in_milk_equals_milk_in_tea :
  final_state.milk_cup.tea = final_state.tea_cup.milk := by
  sorry

end tea_in_milk_equals_milk_in_tea_l3588_358862


namespace san_antonio_bound_bus_encounters_l3588_358856

-- Define the time type (in minutes since midnight)
def Time := ℕ

-- Define the bus schedules
def austin_to_san_antonio_schedule (t : Time) : Prop :=
  ∃ n : ℕ, t = 360 + 120 * n

def san_antonio_to_austin_schedule (t : Time) : Prop :=
  ∃ n : ℕ, t = 390 + 60 * n

-- Define the travel time
def travel_time : ℕ := 360  -- 6 hours in minutes

-- Define the function to count encounters
def count_encounters (start_time : Time) : ℕ :=
  sorry  -- Implementation details omitted

-- Theorem statement
theorem san_antonio_bound_bus_encounters :
  ∀ (start_time : Time),
    san_antonio_to_austin_schedule start_time →
    count_encounters start_time = 2 :=
by sorry

end san_antonio_bound_bus_encounters_l3588_358856


namespace sum_of_ages_l3588_358823

theorem sum_of_ages (tom_age antonette_age : ℝ) : 
  tom_age = 40.5 → 
  antonette_age = 13.5 → 
  tom_age = 3 * antonette_age → 
  tom_age + antonette_age = 54 := by
sorry

end sum_of_ages_l3588_358823


namespace ellipse_equation_l3588_358835

theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0)
  (h4 : c / a = Real.sqrt 3 / 2)
  (h5 : a - c = 2 - Real.sqrt 3)
  (h6 : b^2 = a^2 - c^2) :
  ∃ (x y : ℝ), y^2 / 4 + x^2 = 1 ∧ y^2 / a^2 + x^2 / b^2 = 1 :=
by sorry

end ellipse_equation_l3588_358835


namespace employee_pay_l3588_358849

/-- Given two employees X and Y, proves that Y's weekly pay is 150 units -/
theorem employee_pay (total_pay x y : ℝ) : 
  total_pay = x + y → 
  x = 1.2 * y → 
  total_pay = 330 → 
  y = 150 := by sorry

end employee_pay_l3588_358849


namespace angle_C_in_triangle_l3588_358832

theorem angle_C_in_triangle (A B C : ℝ) (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
    (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) : C = π / 6 := by
  sorry

end angle_C_in_triangle_l3588_358832


namespace length_width_difference_approx_l3588_358839

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  length_gt_width : length > width
  area_eq : area = length * width

/-- The difference between length and width of a rectangular field -/
def length_width_difference (field : RectangularField) : ℝ :=
  field.length - field.width

theorem length_width_difference_approx 
  (field : RectangularField) 
  (h_area : field.area = 171) 
  (h_length : field.length = 19.13) : 
  ∃ ε > 0, |length_width_difference field - 10.19| < ε :=
sorry

end length_width_difference_approx_l3588_358839


namespace polynomial_remainder_l3588_358834

theorem polynomial_remainder (s : ℝ) : (s^10 + 1) % (s - 2) = 1025 := by
  sorry

end polynomial_remainder_l3588_358834


namespace pencil_cost_l3588_358884

/-- If 120 pencils cost $40, then 3600 pencils will cost $1200. -/
theorem pencil_cost (cost_120 : ℕ) (pencils : ℕ) :
  cost_120 = 40 ∧ pencils = 3600 → pencils * cost_120 / 120 = 1200 := by
  sorry

end pencil_cost_l3588_358884


namespace percentage_to_pass_l3588_358820

/-- Given a test with maximum marks, a student's score, and the margin by which they failed,
    calculate the percentage of marks needed to pass the test. -/
theorem percentage_to_pass (max_marks student_score fail_margin : ℕ) : 
  max_marks = 200 → 
  student_score = 80 → 
  fail_margin = 40 → 
  (student_score + fail_margin) / max_marks * 100 = 60 := by
  sorry

end percentage_to_pass_l3588_358820


namespace positive_solution_form_l3588_358890

theorem positive_solution_form (x : ℝ) (a b : ℕ+) :
  x^2 + 14*x = 82 →
  x > 0 →
  x = Real.sqrt a - b →
  a + b = 138 :=
by
  sorry

end positive_solution_form_l3588_358890


namespace min_value_of_2a_plus_b_l3588_358898

theorem min_value_of_2a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a - 2*a*b + b = 0) :
  (∀ x y : ℝ, 0 < x → 0 < y → x - 2*x*y + y = 0 → 2*x + y ≥ 2*a + b) →
  2*a + b = 3/2 + Real.sqrt 2 :=
sorry

end min_value_of_2a_plus_b_l3588_358898


namespace derivative_at_one_l3588_358826

/-- Given a function f(x) = x³ - 2f'(1)x, prove that f'(1) = 1 -/
theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) : 
  deriv f 1 = 1 := by
  sorry

end derivative_at_one_l3588_358826


namespace point_on_unit_circle_l3588_358865

theorem point_on_unit_circle (x : ℝ) (θ : ℝ) :
  (∃ (M : ℝ × ℝ), M = (x, 1) ∧ M.1 = x * Real.cos θ ∧ M.2 = x * Real.sin θ) →
  Real.cos θ = (Real.sqrt 2 / 2) * x →
  x = -1 ∨ x = 0 ∨ x = 1 := by
sorry

end point_on_unit_circle_l3588_358865


namespace sum_seven_consecutive_integers_l3588_358885

theorem sum_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end sum_seven_consecutive_integers_l3588_358885


namespace orthocenter_quadrilateral_congruence_l3588_358848

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : Point) : Prop := sorry

/-- The orthocenter of a triangle is the point where the three altitudes of the triangle intersect. -/
def Orthocenter (H A B C : Point) : Prop := sorry

/-- Two quadrilaterals are congruent if they have the same shape and size. -/
def CongruentQuadrilaterals (A B C D A' B' C' D' : Point) : Prop := sorry

theorem orthocenter_quadrilateral_congruence 
  (A B C D A' B' C' D' : Point) :
  CyclicQuadrilateral A B C D →
  Orthocenter A' B C D →
  Orthocenter B' A C D →
  Orthocenter C' A B D →
  Orthocenter D' A B C →
  CongruentQuadrilaterals A B C D A' B' C' D' :=
sorry

end orthocenter_quadrilateral_congruence_l3588_358848


namespace geometric_sequence_common_ratio_l3588_358873

/-- A geometric sequence with first four terms 25, -50, 100, -200 has a common ratio of -2 -/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℚ), 
    a 0 = 25 ∧ a 1 = -50 ∧ a 2 = 100 ∧ a 3 = -200 →
    ∃ (r : ℚ), r = -2 ∧ ∀ (n : ℕ), a (n + 1) = r * a n :=
by sorry

end geometric_sequence_common_ratio_l3588_358873


namespace sum_of_integers_ending_in_3_l3588_358887

theorem sum_of_integers_ending_in_3 :
  let first_term : ℕ := 103
  let last_term : ℕ := 493
  let common_difference : ℕ := 10
  let n : ℕ := (last_term - first_term) / common_difference + 1
  let sum : ℕ := n * (first_term + last_term) / 2
  sum = 11920 := by sorry

end sum_of_integers_ending_in_3_l3588_358887


namespace power_digit_cycle_l3588_358893

theorem power_digit_cycle (n : ℤ) (k : ℕ) : n^(k+4) ≡ n^k [ZMOD 10] := by
  sorry

end power_digit_cycle_l3588_358893


namespace multiply_fractions_l3588_358802

theorem multiply_fractions : (2 * (1/3)) * (3 * (1/2)) = 1 := by
  sorry

end multiply_fractions_l3588_358802


namespace C_power_50_l3588_358879

def C : Matrix (Fin 2) (Fin 2) ℤ :=
  !![3, 1;
    -4, -1]

theorem C_power_50 :
  C^50 = !![101, 50;
            -200, -99] := by
  sorry

end C_power_50_l3588_358879


namespace inequality_equivalence_l3588_358860

theorem inequality_equivalence (x : ℝ) (h : x ≠ 4) :
  (x^2 - 16) / (x - 4) ≤ 0 ↔ x ≤ -4 := by
  sorry

end inequality_equivalence_l3588_358860


namespace train_length_l3588_358808

/-- Calculates the length of a train given the bridge length, train speed, and time to pass the bridge. -/
theorem train_length (bridge_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) :
  bridge_length = 160 ∧ 
  train_speed_kmh = 40 ∧ 
  time_to_pass = 25.2 →
  (train_speed_kmh * 1000 / 3600) * time_to_pass - bridge_length = 120 :=
by
  sorry

#check train_length

end train_length_l3588_358808


namespace problem_solution_l3588_358828

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem problem_solution :
  (∀ x : ℝ, 0 < x → x < Real.exp 1 → (deriv f) x > 0) ∧
  (∀ x : ℝ, x > Real.exp 1 → (deriv f) x < 0) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → Real.log x + 1 / x > a) ↔ a < 1) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = (1/6) * x₀ - (5/6) / x₀ + 2/3 ∧
    (deriv f) x₀ = (1/3) * x₀ + 2/3) := by sorry

end problem_solution_l3588_358828


namespace horner_method_eval_l3588_358804

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def f (x : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + 5 * x^3 - 2.5 * x^2 + 1.5 * x - 0.7

theorem horner_method_eval :
  f 4 = horner_eval [3, -2, 5, -2.5, 1.5, -0.7] 4 ∧
  horner_eval [3, -2, 5, -2.5, 1.5, -0.7] 4 = 2845.3 := by
  sorry

end horner_method_eval_l3588_358804


namespace gcd_72_108_150_l3588_358880

theorem gcd_72_108_150 : Nat.gcd 72 (Nat.gcd 108 150) = 6 := by
  sorry

end gcd_72_108_150_l3588_358880


namespace correct_average_equals_initial_l3588_358822

theorem correct_average_equals_initial (n : ℕ) (initial_avg : ℚ) 
  (correct1 incorrect1 correct2 incorrect2 : ℚ) : 
  n = 15 → 
  initial_avg = 37 → 
  correct1 = 64 → 
  incorrect1 = 52 → 
  correct2 = 27 → 
  incorrect2 = 39 → 
  (n * initial_avg - incorrect1 - incorrect2 + correct1 + correct2) / n = initial_avg := by
  sorry

end correct_average_equals_initial_l3588_358822


namespace smallest_s_plus_d_l3588_358816

theorem smallest_s_plus_d : ∀ s d : ℕ+,
  (1 : ℚ) / s + (1 : ℚ) / (2 * s) + (1 : ℚ) / (3 * s) = (1 : ℚ) / (d^2 - 2*d) →
  ∀ s' d' : ℕ+,
  (1 : ℚ) / s' + (1 : ℚ) / (2 * s') + (1 : ℚ) / (3 * s') = (1 : ℚ) / (d'^2 - 2*d') →
  (s + d : ℕ) ≤ (s' + d' : ℕ) →
  (s + d : ℕ) = 50 :=
sorry

end smallest_s_plus_d_l3588_358816


namespace odd_plus_even_combination_l3588_358868

theorem odd_plus_even_combination (p q : ℤ) 
  (h_p : ∃ k, p = 2 * k + 1) 
  (h_q : ∃ m, q = 2 * m) : 
  ∃ n, 3 * p + 2 * q = 2 * n + 1 := by
sorry

end odd_plus_even_combination_l3588_358868


namespace ellipse_dimensions_l3588_358863

/-- An ellipse with foci F₁ and F₂, and a point P on the ellipse. -/
structure Ellipse (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  on_ellipse : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1
  perpendicular : (P.1 - F₁.1) * (P.2 - F₂.2) + (P.2 - F₁.2) * (P.1 - F₂.1) = 0
  triangle_area : abs ((F₁.1 - P.1) * (F₂.2 - P.2) - (F₁.2 - P.2) * (F₂.1 - P.1)) / 2 = 9
  triangle_perimeter : dist P F₁ + dist P F₂ + dist F₁ F₂ = 18

/-- The theorem stating that under the given conditions, a = 5 and b = 3 -/
theorem ellipse_dimensions (E : Ellipse a b) : a = 5 ∧ b = 3 := by
  sorry

end ellipse_dimensions_l3588_358863


namespace sqrt_three_squared_l3588_358801

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end sqrt_three_squared_l3588_358801


namespace tan_value_for_special_condition_l3588_358852

theorem tan_value_for_special_condition (a : Real) 
  (h1 : 0 < a ∧ a < π / 2) 
  (h2 : Real.sin a ^ 2 + Real.cos (2 * a) = 1) : 
  Real.tan a = 0 := by
sorry

end tan_value_for_special_condition_l3588_358852


namespace tan_roots_sum_l3588_358894

theorem tan_roots_sum (α β : Real) : 
  (∃ x y : Real, x^2 + 3 * Real.sqrt 3 * x + 4 = 0 ∧ y^2 + 3 * Real.sqrt 3 * y + 4 = 0 ∧ 
   x = Real.tan α ∧ y = Real.tan β) →
  α > -π/2 ∧ α < π/2 ∧ β > -π/2 ∧ β < π/2 →
  α + β = π/3 ∨ α + β = -2*π/3 := by
  sorry

end tan_roots_sum_l3588_358894


namespace pencil_count_l3588_358815

/-- The total number of pencils after multiplication and addition -/
def total_pencils (initial : ℕ) (factor : ℕ) (additional : ℕ) : ℕ :=
  initial * factor + additional

/-- Theorem stating that the total number of pencils is 153 -/
theorem pencil_count : total_pencils 27 4 45 = 153 := by
  sorry

end pencil_count_l3588_358815


namespace ring_weight_sum_l3588_358897

/-- The weight of the orange ring in ounces -/
def orange_ring : ℚ := 0.08

/-- The weight of the purple ring in ounces -/
def purple_ring : ℚ := 0.33

/-- The weight of the white ring in ounces -/
def white_ring : ℚ := 0.42

/-- The total weight of all rings in ounces -/
def total_weight : ℚ := orange_ring + purple_ring + white_ring

theorem ring_weight_sum :
  total_weight = 0.83 := by sorry

end ring_weight_sum_l3588_358897


namespace circle_through_three_points_l3588_358895

/-- The equation of a circle passing through three given points -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x + 6*y - 12 = 0

/-- Point A coordinates -/
def point_A : ℝ × ℝ := (5, 1)

/-- Point B coordinates -/
def point_B : ℝ × ℝ := (6, 0)

/-- Point C coordinates -/
def point_C : ℝ × ℝ := (-1, 1)

/-- Theorem stating that the given equation represents the unique circle passing through the three points -/
theorem circle_through_three_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 ∧
  (∀ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 ↔ circle_equation x y) →
    D = -4 ∧ E = 6 ∧ F = -12) :=
by sorry

end circle_through_three_points_l3588_358895


namespace nelly_painting_bid_l3588_358809

/-- The amount Nelly paid for the painting -/
def nellys_bid (joes_bid : ℕ) : ℕ :=
  3 * joes_bid + 2000

/-- Theorem stating Nelly's final bid given Joe's bid -/
theorem nelly_painting_bid :
  let joes_bid : ℕ := 160000
  nellys_bid joes_bid = 482000 := by
  sorry

end nelly_painting_bid_l3588_358809


namespace clearance_sale_gain_percentage_shopkeeper_clearance_sale_gain_l3588_358827

/-- Calculates the gain percentage during a clearance sale -/
theorem clearance_sale_gain_percentage 
  (original_price : ℝ) 
  (original_gain_percent : ℝ) 
  (discount_percent : ℝ) : ℝ :=
  let cost_price := original_price / (1 + original_gain_percent / 100)
  let discounted_price := original_price * (1 - discount_percent / 100)
  let new_gain := discounted_price - cost_price
  let new_gain_percent := (new_gain / cost_price) * 100
  new_gain_percent

/-- The gain percentage during the clearance sale is approximately 21.5% -/
theorem shopkeeper_clearance_sale_gain :
  abs (clearance_sale_gain_percentage 30 35 10 - 21.5) < 0.1 :=
by sorry

end clearance_sale_gain_percentage_shopkeeper_clearance_sale_gain_l3588_358827


namespace ball_drawing_probability_l3588_358811

theorem ball_drawing_probability : 
  let total_balls : ℕ := 25
  let black_balls : ℕ := 10
  let white_balls : ℕ := 10
  let red_balls : ℕ := 5
  let drawn_balls : ℕ := 4

  let probability : ℚ := 
    (Nat.choose black_balls 2 * Nat.choose white_balls 2 + 
     Nat.choose black_balls 2 * Nat.choose red_balls 2 + 
     Nat.choose white_balls 2 * Nat.choose red_balls 2) / 
    Nat.choose total_balls drawn_balls

  probability = 195 / 841 := by
sorry

end ball_drawing_probability_l3588_358811


namespace mode_median_mean_relationship_l3588_358877

def dataset : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode (data : List ℕ) : ℕ := sorry

def median (data : List ℕ) : ℚ := sorry

def mean (data : List ℕ) : ℚ := sorry

theorem mode_median_mean_relationship :
  let m := mode dataset
  let med := median dataset
  let μ := mean dataset
  (m : ℚ) > med ∧ med > μ := by sorry

end mode_median_mean_relationship_l3588_358877


namespace m_range_theorem_l3588_358871

/-- Proposition P: The equation x²/(2m) + y²/(9-m) = 1 represents an ellipse with foci on the y-axis -/
def P (m : ℝ) : Prop :=
  0 < m ∧ m < 3

/-- Proposition Q: The eccentricity e of the hyperbola y²/5 - x²/m = 1 is within the range (√6/2, √2) -/
def Q (m : ℝ) : Prop :=
  m > 0 ∧ 5/2 < m ∧ m < 5

/-- The set of valid m values -/
def M : Set ℝ :=
  {m | (0 < m ∧ m ≤ 5/2) ∨ (3 ≤ m ∧ m < 5)}

theorem m_range_theorem :
  ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) → m ∈ M :=
sorry

end m_range_theorem_l3588_358871


namespace largest_changeable_digit_is_nine_l3588_358882

/-- The original incorrect sum --/
def original_sum : ℕ := 2436

/-- The correct sum of the addends --/
def correct_sum : ℕ := 731 + 962 + 843

/-- The difference between the correct sum and the original sum --/
def difference : ℕ := correct_sum - original_sum

/-- The largest digit in the hundreds place of the addends --/
def largest_hundreds_digit : ℕ := max (731 / 100) (max (962 / 100) (843 / 100))

theorem largest_changeable_digit_is_nine :
  largest_hundreds_digit = 9 ∧ difference = 100 :=
sorry

end largest_changeable_digit_is_nine_l3588_358882


namespace ramanujan_hardy_complex_numbers_l3588_358806

theorem ramanujan_hardy_complex_numbers :
  ∀ (z w : ℂ),
  z * w = 40 - 24 * I →
  w = 4 + 4 * I →
  z = 2 - 8 * I :=
by sorry

end ramanujan_hardy_complex_numbers_l3588_358806


namespace c_payment_l3588_358866

def work_rate (days : ℕ) : ℚ := 1 / days

def total_payment : ℕ := 3200

def completion_days : ℕ := 3

theorem c_payment (a_days b_days : ℕ) (ha : a_days = 6) (hb : b_days = 8) : 
  let a_rate := work_rate a_days
  let b_rate := work_rate b_days
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * completion_days
  let c_work := 1 - ab_work
  c_work * total_payment = 400 := by sorry

end c_payment_l3588_358866


namespace footprint_calculation_l3588_358878

/-- Calculates the total number of footprints left by three creatures on their respective planets -/
theorem footprint_calculation (pogo_rate : ℕ) (grimzi_rate : ℕ) (zeb_rate : ℕ)
  (pogo_distance : ℕ) (grimzi_distance : ℕ) (zeb_distance : ℕ)
  (total_distance : ℕ) :
  pogo_rate = 4 ∧ 
  grimzi_rate = 3 ∧ 
  zeb_rate = 5 ∧
  pogo_distance = 1 ∧ 
  grimzi_distance = 6 ∧ 
  zeb_distance = 8 ∧
  total_distance = 6000 →
  pogo_rate * total_distance + 
  (total_distance / grimzi_distance) * grimzi_rate + 
  (total_distance / zeb_distance) * zeb_rate = 30750 := by
sorry


end footprint_calculation_l3588_358878


namespace additive_inverse_solution_equal_surds_solution_l3588_358843

-- Part 1
theorem additive_inverse_solution (x : ℝ) : 
  x^2 + 3*x - 6 = -((-x + 1)) → x = -1 + Real.sqrt 6 ∨ x = -1 - Real.sqrt 6 := by sorry

-- Part 2
theorem equal_surds_solution (m : ℝ) :
  Real.sqrt (m^2 - 6) = Real.sqrt (6*m + 1) → m = 7 := by sorry

end additive_inverse_solution_equal_surds_solution_l3588_358843


namespace clock_synchronization_l3588_358824

theorem clock_synchronization (arthur_gain oleg_gain cycle : ℕ) 
  (h1 : arthur_gain = 15)
  (h2 : oleg_gain = 12)
  (h3 : cycle = 720) :
  let sync_days := Nat.lcm (cycle / arthur_gain) (cycle / oleg_gain)
  sync_days = 240 ∧ 
  ∀ k : ℕ, k < sync_days → ¬(arthur_gain * k % cycle = 0 ∧ oleg_gain * k % cycle = 0) := by
  sorry

end clock_synchronization_l3588_358824


namespace valid_three_digit_numbers_count_l3588_358850

/-- The count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with two adjacent identical digits -/
def numbers_with_two_adjacent_identical_digits : ℕ := 153

/-- The count of valid three-digit numbers according to the problem conditions -/
def valid_three_digit_numbers : ℕ := total_three_digit_numbers - numbers_with_two_adjacent_identical_digits

theorem valid_three_digit_numbers_count :
  valid_three_digit_numbers = 747 := by
  sorry

end valid_three_digit_numbers_count_l3588_358850


namespace arithmetic_sequence_8th_term_l3588_358883

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_8th_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_4th : a 4 = 23)
  (h_6th : a 6 = 47) :
  a 8 = 71 := by
sorry

end arithmetic_sequence_8th_term_l3588_358883


namespace total_marbles_l3588_358872

/-- Represents the number of marbles each boy has -/
structure MarbleDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The ratio of marbles between the three boys -/
def marbleRatio : MarbleDistribution := ⟨5, 2, 3⟩

/-- The number of additional marbles the first boy has -/
def additionalMarbles : ℕ := 3

/-- The number of marbles the middle (second) boy has -/
def middleBoyMarbles : ℕ := 12

/-- The theorem stating the total number of marbles -/
theorem total_marbles :
  ∃ (x : ℕ),
    x * marbleRatio.second = middleBoyMarbles ∧
    (x * marbleRatio.first + additionalMarbles) +
    (x * marbleRatio.second) +
    (x * marbleRatio.third) = 63 := by
  sorry


end total_marbles_l3588_358872


namespace min_xy_value_l3588_358886

theorem min_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 8) :
  (x : ℚ) * y ≥ 128 :=
sorry

end min_xy_value_l3588_358886


namespace exterior_angle_square_octagon_is_135_l3588_358825

/-- The exterior angle formed by a square and a regular octagon sharing a common side -/
def exterior_angle_square_octagon : ℝ := 135

/-- Theorem: The exterior angle formed by a square and a regular octagon sharing a common side is 135 degrees -/
theorem exterior_angle_square_octagon_is_135 :
  exterior_angle_square_octagon = 135 := by
  sorry

end exterior_angle_square_octagon_is_135_l3588_358825


namespace quadratic_equation_solutions_l3588_358829

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3 := by sorry

end quadratic_equation_solutions_l3588_358829


namespace segment_transformation_midpoint_l3588_358803

/-- Given a segment with endpoints (3, -2) and (9, 6), when translated 4 units left and 2 units down,
    then rotated 90° counterclockwise about its midpoint, the resulting segment has a midpoint at (2, 0) -/
theorem segment_transformation_midpoint : 
  let s₁_start : ℝ × ℝ := (3, -2)
  let s₁_end : ℝ × ℝ := (9, 6)
  let translate : ℝ × ℝ := (-4, -2)
  let s₁_midpoint := ((s₁_start.1 + s₁_end.1) / 2, (s₁_start.2 + s₁_end.2) / 2)
  let s₂_midpoint := (s₁_midpoint.1 + translate.1, s₁_midpoint.2 + translate.2)
  s₂_midpoint = (2, 0) := by
  sorry

end segment_transformation_midpoint_l3588_358803


namespace max_abs_z5_l3588_358819

theorem max_abs_z5 (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h1 : Complex.abs z₁ ≤ 1)
  (h2 : Complex.abs z₂ ≤ 1)
  (h3 : Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h4 : Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂))
  (h5 : Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄)) :
  Complex.abs z₅ ≤ Real.sqrt 3 ∧ ∃ z₁ z₂ z₃ z₄ z₅, 
    Complex.abs z₁ ≤ 1 ∧ 
    Complex.abs z₂ ≤ 1 ∧
    Complex.abs (2 * z₃ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₄ - (z₁ + z₂)) ≤ Complex.abs (z₁ - z₂) ∧
    Complex.abs (2 * z₅ - (z₃ + z₄)) ≤ Complex.abs (z₃ - z₄) ∧
    Complex.abs z₅ = Real.sqrt 3 :=
by
  sorry

end max_abs_z5_l3588_358819


namespace construct_m_is_perfect_square_l3588_358870

/-- The number of 1's in the sequence -/
def num_ones : ℕ := 1997

/-- The number of 2's in the sequence -/
def num_twos : ℕ := 1998

/-- Constructs the number m as described in the problem -/
def construct_m : ℕ :=
  let n := (10^num_ones * (10^num_ones - 1) + 2 * (10^num_ones - 1)) / 9
  10 * n + 25

/-- Theorem stating that the constructed number m is a perfect square -/
theorem construct_m_is_perfect_square : ∃ k : ℕ, construct_m = k^2 := by
  sorry

end construct_m_is_perfect_square_l3588_358870


namespace fraction_sum_equation_l3588_358876

theorem fraction_sum_equation (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
  x / y = (8 + Real.sqrt 46) / 6 := by
sorry

end fraction_sum_equation_l3588_358876


namespace only_solutions_l3588_358896

/-- A function from nonnegative integers to nonnegative integers -/
def NonNegIntFunction := ℕ → ℕ

/-- The property that f(f(f(n))) = f(n+1) + 1 for all n -/
def SatisfiesEquation (f : NonNegIntFunction) : Prop :=
  ∀ n, f (f (f n)) = f (n + 1) + 1

/-- The first solution function: f(n) = n + 1 -/
def Solution1 : NonNegIntFunction :=
  λ n => n + 1

/-- The second solution function: 
    f(n) = n + 1 if n ≡ 0 (mod 4) or n ≡ 2 (mod 4),
    f(n) = n + 5 if n ≡ 1 (mod 4),
    f(n) = n - 3 if n ≡ 3 (mod 4) -/
def Solution2 : NonNegIntFunction :=
  λ n => match n % 4 with
    | 0 | 2 => n + 1
    | 1 => n + 5
    | 3 => n - 3
    | _ => n  -- This case is unreachable, but needed for exhaustiveness

/-- The main theorem: Solution1 and Solution2 are the only functions satisfying the equation -/
theorem only_solutions (f : NonNegIntFunction) :
  SatisfiesEquation f ↔ (f = Solution1 ∨ f = Solution2) := by
  sorry

end only_solutions_l3588_358896


namespace arithmetic_sequence_ratio_l3588_358845

theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ)
  (h_arithmetic_a : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_arithmetic_b : ∀ n, b (n + 1) - b n = b 2 - b 1)
  (h_sum_S : ∀ n, S n = n * (a 1 + a n) / 2)
  (h_sum_T : ∀ n, T n = n * (b 1 + b n) / 2)
  (h_ratio : ∀ n, S n / T n = (n + 3) / (2 * n + 1)) :
  a 6 / b 6 = 14 / 23 := by
sorry

end arithmetic_sequence_ratio_l3588_358845


namespace movie_ratio_proof_l3588_358810

theorem movie_ratio_proof (total : ℕ) (dvd : ℕ) (bluray : ℕ) :
  total = 378 →
  dvd + bluray = total →
  dvd / (bluray - 4) = 9 / 2 →
  (dvd : ℚ) / bluray = 51 / 12 :=
by
  sorry

end movie_ratio_proof_l3588_358810


namespace exam_survey_analysis_l3588_358881

structure SurveyData where
  total_candidates : Nat
  sample_size : Nat

def sampling_survey_method (data : SurveyData) : Prop :=
  data.sample_size < data.total_candidates

def is_population (data : SurveyData) (n : Nat) : Prop :=
  n = data.total_candidates

def is_sample (data : SurveyData) (n : Nat) : Prop :=
  n = data.sample_size

theorem exam_survey_analysis (data : SurveyData)
  (h1 : data.total_candidates = 60000)
  (h2 : data.sample_size = 1000) :
  ∃ (correct_statements : Finset (Fin 4)),
    correct_statements.card = 2 ∧
    (1 ∈ correct_statements ↔ sampling_survey_method data) ∧
    (2 ∈ correct_statements ↔ is_population data data.total_candidates) ∧
    (3 ∈ correct_statements ↔ is_sample data data.sample_size) ∧
    (4 ∈ correct_statements ↔ data.sample_size = 1000) :=
sorry

end exam_survey_analysis_l3588_358881


namespace monotonicity_range_and_minimum_value_l3588_358814

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - log x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := exp x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * exp (a * x - 1) - 2 * a * x + f a x

theorem monotonicity_range_and_minimum_value 
  (h1 : ∀ x, x > 0)
  (h2 : ∀ a, a < 0) :
  (∃ S : Set ℝ, S = { a | ∀ x ∈ (Set.Ioo 0 (log 3)), 
    (Monotone (f a) ↔ Monotone (F a)) ∧ S = Set.Iic (-3)}) ∧
  (∀ a ∈ Set.Iic (-1 / (exp 2)), 
    IsMinOn (g a) (Set.Ioi 0) 0) := by sorry

end monotonicity_range_and_minimum_value_l3588_358814


namespace dantes_age_l3588_358830

theorem dantes_age (cooper dante maria : ℕ) : 
  cooper + dante + maria = 31 →
  cooper = dante / 2 →
  maria = dante + 1 →
  dante = 12 := by
sorry

end dantes_age_l3588_358830


namespace distribute_5_to_3_l3588_358812

/-- The number of ways to distribute n volunteers to k venues --/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 volunteers to 3 venues --/
theorem distribute_5_to_3 : distribute 5 3 = 150 := by sorry

end distribute_5_to_3_l3588_358812


namespace initial_medium_size_shoes_l3588_358888

/-- Given a shoe shop's inventory and sales data, prove the initial number of medium-size shoes. -/
theorem initial_medium_size_shoes
  (large_size : Nat) -- Initial number of large-size shoes
  (small_size : Nat) -- Initial number of small-size shoes
  (sold : Nat) -- Number of shoes sold
  (remaining : Nat) -- Number of shoes remaining after sale
  (h1 : large_size = 22)
  (h2 : small_size = 24)
  (h3 : sold = 83)
  (h4 : remaining = 13)
  (h5 : ∃ M : Nat, large_size + M + small_size = sold + remaining) :
  ∃ M : Nat, M = 26 ∧ large_size + M + small_size = sold + remaining :=
by sorry


end initial_medium_size_shoes_l3588_358888


namespace mary_balloon_count_l3588_358844

/-- The number of yellow balloons each person has -/
structure BalloonCount where
  fred : ℕ
  sam : ℕ
  mary : ℕ

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- The actual balloon count for Fred, Sam, and Mary -/
def actual_count : BalloonCount where
  fred := 5
  sam := 6
  mary := 7

/-- Theorem stating that Mary has 7 yellow balloons -/
theorem mary_balloon_count :
  ∀ (count : BalloonCount),
    count.fred = actual_count.fred →
    count.sam = actual_count.sam →
    count.fred + count.sam + count.mary = total_balloons →
    count.mary = actual_count.mary :=
by
  sorry

end mary_balloon_count_l3588_358844


namespace marble_redistribution_l3588_358892

/-- Represents the number of marbles each person has -/
structure MarbleDistribution :=
  (person1 : ℕ)
  (person2 : ℕ)
  (person3 : ℕ)
  (person4 : ℕ)

/-- The theorem statement -/
theorem marble_redistribution 
  (initial : MarbleDistribution)
  (h1 : initial.person1 = 14)
  (h2 : initial.person2 = 19)
  (h3 : initial.person3 = 7)
  (h4 : ∀ (final : MarbleDistribution), 
    final.person1 + final.person2 + final.person3 + final.person4 = 
    initial.person1 + initial.person2 + initial.person3 + initial.person4)
  (h5 : ∀ (final : MarbleDistribution), 
    final.person1 = final.person2 ∧ 
    final.person2 = final.person3 ∧ 
    final.person3 = final.person4 ∧
    final.person4 = 15) :
  initial.person4 = 20 := by
sorry


end marble_redistribution_l3588_358892


namespace cubes_arrangement_theorem_l3588_358818

/-- Represents the colors used to paint the cubes -/
inductive Color
  | White
  | Black
  | Red

/-- Represents a cube with 6 faces, each painted with a color -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Represents the set of 16 cubes -/
def CubeSet := Fin 16 → Cube

/-- Represents an arrangement of the 16 cubes -/
structure Arrangement :=
  (placement : Fin 16 → Fin 3 × Fin 3 × Fin 3)
  (orientation : Fin 16 → Fin 6)

/-- Predicate to check if an arrangement shows only one color -/
def ShowsOnlyOneColor (cs : CubeSet) (arr : Arrangement) (c : Color) : Prop :=
  ∀ i : Fin 16, (cs i).faces (arr.orientation i) = c

/-- Theorem stating that it's possible to arrange the cubes to show only one color -/
theorem cubes_arrangement_theorem (cs : CubeSet) :
  ∃ (arr : Arrangement) (c : Color), ShowsOnlyOneColor cs arr c :=
sorry

end cubes_arrangement_theorem_l3588_358818


namespace cos_thirty_degrees_l3588_358867

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end cos_thirty_degrees_l3588_358867


namespace orange_juice_production_l3588_358831

/-- The amount of oranges (in million tons) used for juice production -/
def juice_production (total : ℝ) (export_percent : ℝ) (juice_percent : ℝ) : ℝ :=
  total * (1 - export_percent) * juice_percent

/-- Theorem stating the amount of oranges used for juice production -/
theorem orange_juice_production :
  let total := 8
  let export_percent := 0.25
  let juice_percent := 0.60
  juice_production total export_percent juice_percent = 3.6 := by
sorry

#eval juice_production 8 0.25 0.60

end orange_juice_production_l3588_358831


namespace number_problem_l3588_358861

theorem number_problem (x : ℚ) : (3 / 4 : ℚ) * x = x - 19 → x = 76 := by
  sorry

end number_problem_l3588_358861


namespace max_change_percentage_l3588_358855

theorem max_change_percentage (initial_yes initial_no final_yes final_no fixed_mindset_ratio : ℚ)
  (h1 : initial_yes + initial_no = 1)
  (h2 : final_yes + final_no = 1)
  (h3 : initial_yes = 2/5)
  (h4 : initial_no = 3/5)
  (h5 : final_yes = 4/5)
  (h6 : final_no = 1/5)
  (h7 : fixed_mindset_ratio = 1/5) :
  let fixed_mindset := fixed_mindset_ratio * initial_no
  let max_change := final_yes - initial_yes
  max_change ≤ initial_no - fixed_mindset ∧ max_change = 2/5 := by
  sorry

end max_change_percentage_l3588_358855


namespace marbles_remainder_l3588_358853

theorem marbles_remainder (n m k : ℤ) : (8*n + 5 + 7*m + 2 + 7*k + 4) % 7 = 4 := by
  sorry

end marbles_remainder_l3588_358853


namespace jamal_grade_jamal_grade_is_108_l3588_358858

theorem jamal_grade (total_students : ℕ) (absent_students : ℕ) (first_day_average : ℕ) 
  (new_average : ℕ) (taqeesha_score : ℕ) : ℕ :=
  let students_first_day := total_students - absent_students
  let total_score_first_day := students_first_day * first_day_average
  let total_score_all := total_students * new_average
  let combined_score_absent := total_score_all - total_score_first_day
  combined_score_absent - taqeesha_score

theorem jamal_grade_is_108 :
  jamal_grade 30 2 85 86 92 = 108 := by
  sorry

end jamal_grade_jamal_grade_is_108_l3588_358858


namespace alicia_local_tax_l3588_358813

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate percentage. -/
def local_tax_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate / 100)

/-- Theorem stating that for an hourly wage of $25 and a 2% tax rate, the local tax amount is 50 cents. -/
theorem alicia_local_tax : local_tax_cents 25 2 = 50 := by
  sorry

end alicia_local_tax_l3588_358813


namespace point_inside_circle_l3588_358846

/-- Definition of a circle with center (a, b) and radius r -/
def Circle (a b r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - b)^2 = r^2}

/-- Definition of a point being inside a circle -/
def InsideCircle (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  (p.1 - 2)^2 + (p.2 - 3)^2 < 4

/-- The main theorem -/
theorem point_inside_circle :
  let c : Set (ℝ × ℝ) := Circle 2 3 2
  let p : ℝ × ℝ := (1, 2)
  InsideCircle p c := by sorry

end point_inside_circle_l3588_358846


namespace race_total_length_l3588_358821

/-- The total length of a race with four parts -/
def race_length (part1 part2 part3 part4 : ℝ) : ℝ :=
  part1 + part2 + part3 + part4

theorem race_total_length :
  race_length 15.5 21.5 21.5 16 = 74.5 := by
  sorry

end race_total_length_l3588_358821


namespace laborer_income_l3588_358840

/-- The monthly income of a laborer given certain expenditure and savings conditions -/
theorem laborer_income (
  average_expenditure : ℝ)
  (reduced_expenditure : ℝ)
  (months_initial : ℕ)
  (months_reduced : ℕ)
  (savings : ℝ)
  (h1 : average_expenditure = 90)
  (h2 : reduced_expenditure = 60)
  (h3 : months_initial = 6)
  (h4 : months_reduced = 4)
  (h5 : savings = 30)
  : ∃ (income : ℝ) (debt : ℝ),
    income * months_initial = average_expenditure * months_initial - debt ∧
    income * months_reduced = reduced_expenditure * months_reduced + debt + savings ∧
    income = 81 := by
  sorry

end laborer_income_l3588_358840


namespace rectangle_fourth_vertex_l3588_358869

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a rectangle by its four vertices
structure Rectangle where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define the property of being a rectangle
def isRectangle (rect : Rectangle) : Prop :=
  -- Add properties that define a rectangle, such as perpendicular sides and equal diagonals
  sorry

-- Theorem statement
theorem rectangle_fourth_vertex 
  (rect : Rectangle)
  (h1 : isRectangle rect)
  (h2 : rect.A = ⟨1, 1⟩)
  (h3 : rect.B = ⟨3, 1⟩)
  (h4 : rect.C = ⟨3, 5⟩) :
  rect.D = ⟨1, 5⟩ := by
  sorry

end rectangle_fourth_vertex_l3588_358869


namespace smallest_a_value_l3588_358851

theorem smallest_a_value (a b : ℕ) (h : b^3 = 1176*a) : 
  (∀ x : ℕ, x < a → ¬∃ y : ℕ, y^3 = 1176*x) → a = 63 := by
  sorry

end smallest_a_value_l3588_358851


namespace arithmetic_sequence_sum_2019_l3588_358874

-- Define the arithmetic sequence and its sum
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)
def S (a₁ d : ℤ) (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

-- State the theorem
theorem arithmetic_sequence_sum_2019 (a₁ d : ℤ) :
  a₁ = -2017 →
  (S a₁ d 2017 / 2017 - S a₁ d 2015 / 2015 = 2) →
  S a₁ d 2019 = 2019 := by
  sorry


end arithmetic_sequence_sum_2019_l3588_358874
