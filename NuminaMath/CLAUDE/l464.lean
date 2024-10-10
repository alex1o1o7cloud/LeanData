import Mathlib

namespace blue_parrots_count_l464_46485

/-- The number of blue parrots on Bird Island --/
def blue_parrots : ℕ := 38

/-- The total number of parrots on Bird Island after new arrivals --/
def total_parrots : ℕ := 150

/-- The fraction of red parrots --/
def red_fraction : ℚ := 1/2

/-- The fraction of green parrots --/
def green_fraction : ℚ := 1/4

/-- The number of new parrots that arrived --/
def new_parrots : ℕ := 30

theorem blue_parrots_count :
  blue_parrots = total_parrots - (red_fraction * total_parrots).floor - (green_fraction * total_parrots).floor :=
by sorry

end blue_parrots_count_l464_46485


namespace specific_pentagon_area_l464_46475

/-- Represents a pentagon that can be decomposed into two triangles and a trapezoid -/
structure DecomposablePentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  trapezoid_area : ℝ
  decomposable : side1 > 0 ∧ side2 > 0 ∧ side3 > 0 ∧ side4 > 0 ∧ side5 > 0

/-- Calculate the area of a decomposable pentagon -/
def area (p : DecomposablePentagon) : ℝ :=
  p.triangle1_area + p.triangle2_area + p.trapezoid_area

/-- Theorem stating that a specific pentagon has an area of 848 square units -/
theorem specific_pentagon_area :
  ∃ (p : DecomposablePentagon),
    p.side1 = 18 ∧ p.side2 = 22 ∧ p.side3 = 30 ∧ p.side4 = 26 ∧ p.side5 = 22 ∧
    area p = 848 := by
  sorry


end specific_pentagon_area_l464_46475


namespace quadratic_inequality_solution_l464_46431

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -14 := by
sorry

end quadratic_inequality_solution_l464_46431


namespace angle_AEC_measure_l464_46423

-- Define the angles in the triangle
def angle_ABE' : ℝ := 150
def angle_BAC : ℝ := 108

-- Define the property of supplementary angles
def supplementary (a b : ℝ) : Prop := a + b = 180

-- Theorem statement
theorem angle_AEC_measure :
  ∀ angle_ABE angle_AEC,
  supplementary angle_ABE angle_ABE' →
  angle_ABE + angle_BAC + angle_AEC = 180 →
  angle_AEC = 42 := by
    sorry

end angle_AEC_measure_l464_46423


namespace product_of_numbers_l464_46453

theorem product_of_numbers (x y : ℝ) : x + y = 40 ∧ x - y = 16 → x * y = 336 := by
  sorry

end product_of_numbers_l464_46453


namespace smaller_number_in_sum_l464_46473

theorem smaller_number_in_sum (x y : ℕ) : 
  x + y = 84 → y = 3 * x → x = 21 := by
  sorry

end smaller_number_in_sum_l464_46473


namespace greatest_integer_radius_of_circle_l464_46455

theorem greatest_integer_radius_of_circle (r : ℝ) : 
  (π * r^2 < 100 * π) → (∀ n : ℕ, n > 9 → π * (n : ℝ)^2 ≥ 100 * π) :=
by sorry

end greatest_integer_radius_of_circle_l464_46455


namespace probability_of_e_in_theorem_l464_46496

theorem probability_of_e_in_theorem :
  let total_letters : ℕ := 7
  let number_of_e : ℕ := 2
  let probability : ℚ := number_of_e / total_letters
  probability = 2 / 7 := by sorry

end probability_of_e_in_theorem_l464_46496


namespace first_transfer_amount_l464_46432

/-- Proves that the amount of the first bank transfer is approximately $91.18 given the initial and final balances and service charge. -/
theorem first_transfer_amount (initial_balance : ℝ) (final_balance : ℝ) (service_charge_rate : ℝ) :
  initial_balance = 400 →
  final_balance = 307 →
  service_charge_rate = 0.02 →
  ∃ (transfer_amount : ℝ), 
    initial_balance - (transfer_amount * (1 + service_charge_rate)) = final_balance ∧
    (transfer_amount ≥ 91.17 ∧ transfer_amount ≤ 91.19) :=
by sorry

end first_transfer_amount_l464_46432


namespace typist_salary_problem_l464_46477

theorem typist_salary_problem (x : ℝ) : 
  (x * 1.1 * 0.95 = 6270) → x = 6000 := by
  sorry

end typist_salary_problem_l464_46477


namespace tangent_line_equation_max_integer_k_l464_46469

noncomputable section

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := (Real.log x + k) / Real.exp x

-- Define the derivative of f
def f_derivative (k : ℝ) (x : ℝ) : ℝ := (1 - k*x - x * Real.log x) / (x * Real.exp x)

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  k = 2 →
  y = f 2 x →
  x = 1 →
  (x + Real.exp y - 3 = 0) :=
sorry

-- Theorem for the maximum integer value of k
theorem max_integer_k (k : ℤ) :
  (∀ x > 1, x * Real.exp x * f_derivative k x + (2 * ↑k - 1) * x < 1 + ↑k) →
  k ≤ 3 :=
sorry

end

end tangent_line_equation_max_integer_k_l464_46469


namespace square_pyramid_frustum_volume_fraction_l464_46440

/-- The volume of a square pyramid frustum as a fraction of the original pyramid --/
theorem square_pyramid_frustum_volume_fraction 
  (base_edge : ℝ) 
  (altitude : ℝ) 
  (h_base : base_edge = 40) 
  (h_alt : altitude = 18) :
  let original_volume := (1/3) * base_edge^2 * altitude
  let small_base_edge := (1/5) * base_edge
  let small_altitude := (1/5) * altitude
  let small_volume := (1/3) * small_base_edge^2 * small_altitude
  let frustum_volume := original_volume - small_volume
  frustum_volume / original_volume = 2383 / 2400 := by
sorry

end square_pyramid_frustum_volume_fraction_l464_46440


namespace value_is_appropriate_for_project_assessment_other_terms_not_appropriate_l464_46433

-- Define the possible options
inductive ProjectAssessmentTerm
  | Price
  | Value
  | Cost
  | Expense

-- Define a function that determines if a term is appropriate for project assessment
def isAppropriateForProjectAssessment (term : ProjectAssessmentTerm) : Prop :=
  match term with
  | ProjectAssessmentTerm.Value => True
  | _ => False

-- Theorem stating that "Value" is the appropriate term
theorem value_is_appropriate_for_project_assessment :
  isAppropriateForProjectAssessment ProjectAssessmentTerm.Value :=
by sorry

-- Theorem stating that other terms are not appropriate
theorem other_terms_not_appropriate (term : ProjectAssessmentTerm) :
  term ≠ ProjectAssessmentTerm.Value →
  ¬(isAppropriateForProjectAssessment term) :=
by sorry

end value_is_appropriate_for_project_assessment_other_terms_not_appropriate_l464_46433


namespace equation_solutions_l464_46472

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(1, 2, 26), (1, 8, 8), (2, 2, 19), (2, 4, 12), (2, 5, 10), (4, 4, 8)}

def satisfies_equation (triple : ℕ × ℕ × ℕ) : Prop :=
  let (x, y, z) := triple
  x * y + y * z + z * x = 80 ∧ x ≤ y ∧ y ≤ z

theorem equation_solutions :
  ∀ (x y z : ℕ), satisfies_equation (x, y, z) ↔ (x, y, z) ∈ solution_set :=
by sorry

end equation_solutions_l464_46472


namespace function_bounds_l464_46428

theorem function_bounds 
  (f : ℕ+ → ℕ+) 
  (h_increasing : ∀ n m : ℕ+, n < m → f n < f m) 
  (k : ℕ+) 
  (h_composition : ∀ n : ℕ+, f (f n) = k * n) :
  ∀ n : ℕ+, (2 * k * n : ℚ) / (k + 1) ≤ f n ∧ (f n : ℚ) ≤ ((k + 1) * n) / 2 :=
sorry

end function_bounds_l464_46428


namespace angle_C_measure_l464_46427

/-- Given a triangle ABC where sin²A - sin²C = (sin A - sin B) sin B, prove that the measure of angle C is π/3 -/
theorem angle_C_measure (A B C : ℝ) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 - Real.sin C ^ 2 = (Real.sin A - Real.sin B) * Real.sin B) : 
  C = π / 3 := by
  sorry

end angle_C_measure_l464_46427


namespace intersection_complement_equals_one_l464_46404

universe u

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem intersection_complement_equals_one : M ∩ (U \ N) = {1} := by sorry

end intersection_complement_equals_one_l464_46404


namespace division_practice_time_l464_46470

-- Define the given conditions
def total_training_time : ℕ := 5 * 60  -- 5 hours in minutes
def training_days : ℕ := 10
def daily_multiplication_time : ℕ := 10

-- Define the theorem
theorem division_practice_time :
  (total_training_time - training_days * daily_multiplication_time) / training_days = 20 := by
  sorry

end division_practice_time_l464_46470


namespace average_first_100_odd_numbers_l464_46452

theorem average_first_100_odd_numbers : 
  let n := 100
  let nth_odd (k : ℕ) := 2 * k - 1
  let first_odd := nth_odd 1
  let last_odd := nth_odd n
  let sum := (n / 2) * (first_odd + last_odd)
  sum / n = 100 := by
sorry

end average_first_100_odd_numbers_l464_46452


namespace student_count_l464_46412

/-- Given a group of students where replacing one student changes the average weight,
    this theorem proves the total number of students. -/
theorem student_count
  (avg_decrease : ℝ)  -- The decrease in average weight
  (old_weight : ℝ)    -- Weight of the replaced student
  (new_weight : ℝ)    -- Weight of the new student
  (h1 : avg_decrease = 5)  -- The average weight decreases by 5 kg
  (h2 : old_weight = 86)   -- The replaced student weighs 86 kg
  (h3 : new_weight = 46)   -- The new student weighs 46 kg
  : ℕ := by
  sorry

end student_count_l464_46412


namespace ellipse_major_axis_length_l464_46492

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- An ellipse with axes parallel to coordinate axes -/
structure Ellipse where
  center : Point
  a : ℝ  -- semi-major axis
  b : ℝ  -- semi-minor axis

/-- Check if a point lies on an ellipse -/
def onEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.a^2 + (p.y - e.center.y)^2 / e.b^2 = 1

theorem ellipse_major_axis_length :
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨2, 2⟩
  let p3 : Point := ⟨-2, 2⟩
  let p4 : Point := ⟨4, 0⟩
  let p5 : Point := ⟨4, 4⟩
  ∃ (e : Ellipse),
    (¬ collinear p1 p2 p3 ∧ ¬ collinear p1 p2 p4 ∧ ¬ collinear p1 p2 p5 ∧
     ¬ collinear p1 p3 p4 ∧ ¬ collinear p1 p3 p5 ∧ ¬ collinear p1 p4 p5 ∧
     ¬ collinear p2 p3 p4 ∧ ¬ collinear p2 p3 p5 ∧ ¬ collinear p2 p4 p5 ∧
     ¬ collinear p3 p4 p5) →
    (onEllipse p1 e ∧ onEllipse p2 e ∧ onEllipse p3 e ∧ onEllipse p4 e ∧ onEllipse p5 e) →
    2 * e.a = 4 :=
by
  sorry


end ellipse_major_axis_length_l464_46492


namespace calendar_date_theorem_l464_46482

/-- Represents a monthly calendar with dates behind letters --/
structure MonthlyCalendar where
  C : ℤ  -- Date behind C
  A : ℤ  -- Date behind A
  B : ℤ  -- Date behind B
  Q : ℤ  -- Date behind Q

/-- Theorem: The difference between dates behind C and Q equals the sum of dates behind A and B --/
theorem calendar_date_theorem (cal : MonthlyCalendar) 
  (hC : cal.C = x)
  (hA : cal.A = x + 2)
  (hB : cal.B = x + 14)
  (hQ : cal.Q = -x - 16)
  : cal.C - cal.Q = cal.A + cal.B :=
by sorry

end calendar_date_theorem_l464_46482


namespace olympic_torch_relay_schemes_l464_46426

/-- The number of segments in the Olympic torch relay -/
def num_segments : ℕ := 6

/-- The number of torchbearers -/
def num_torchbearers : ℕ := 6

/-- The number of choices for the first torchbearer -/
def first_choices : ℕ := 3

/-- The number of choices for the last torchbearer -/
def last_choices : ℕ := 2

/-- The number of choices for each middle segment -/
def middle_choices : ℕ := num_torchbearers

/-- The number of middle segments -/
def num_middle_segments : ℕ := num_segments - 2

/-- The total number of different relay schemes -/
def total_schemes : ℕ := first_choices * (middle_choices ^ num_middle_segments) * last_choices

theorem olympic_torch_relay_schemes :
  total_schemes = 7776 := by
  sorry

end olympic_torch_relay_schemes_l464_46426


namespace ellipse_eccentricity_l464_46435

/-- An ellipse with parametric equations x = 3cos(φ) and y = 5sin(φ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ φ, x φ = 3 * Real.cos φ
  h_y : ∀ φ, y φ = 5 * Real.sin φ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating that the eccentricity of the given ellipse is 4/5 -/
theorem ellipse_eccentricity (e : Ellipse) : eccentricity e = 4/5 :=
  sorry

end ellipse_eccentricity_l464_46435


namespace remainder_r17_plus_1_div_r_plus_1_l464_46486

theorem remainder_r17_plus_1_div_r_plus_1 (r : ℤ) : (r^17 + 1) % (r + 1) = 0 := by
  sorry

end remainder_r17_plus_1_div_r_plus_1_l464_46486


namespace digit_mean_is_four_point_five_l464_46451

/-- The period length of the repeating decimal expansion of 1/(98^2) -/
def period_length : ℕ := 9604

/-- The sum of all digits in one period of the repeating decimal expansion of 1/(98^2) -/
def digit_sum : ℕ := 432180

/-- The mean of the digits in one complete period of the repeating decimal expansion of 1/(98^2) -/
def digit_mean : ℚ := digit_sum / period_length

theorem digit_mean_is_four_point_five :
  digit_mean = 4.5 := by sorry

end digit_mean_is_four_point_five_l464_46451


namespace counterexample_exists_l464_46444

theorem counterexample_exists : ∃ a : ℝ, (|a - 1| > 1) ∧ (a ≤ 2) := by
  sorry

end counterexample_exists_l464_46444


namespace evaluate_64_to_5_6th_power_l464_46465

theorem evaluate_64_to_5_6th_power : (64 : ℝ) ^ (5/6) = 32 := by
  sorry

end evaluate_64_to_5_6th_power_l464_46465


namespace factorization_equivalence_l464_46461

theorem factorization_equivalence (x y : ℝ) : -(2*x - y) * (2*x + y) = -4*x^2 + y^2 := by
  sorry

end factorization_equivalence_l464_46461


namespace first_class_product_rate_l464_46407

/-- Given a product with a pass rate and a rate of first-class products among qualified products,
    prove that the overall rate of first-class products is their product. -/
theorem first_class_product_rate
  (pass_rate : ℝ)
  (first_class_rate_among_qualified : ℝ)
  (h1 : 0 ≤ pass_rate ∧ pass_rate ≤ 1)
  (h2 : 0 ≤ first_class_rate_among_qualified ∧ first_class_rate_among_qualified ≤ 1) :
  pass_rate * first_class_rate_among_qualified =
  pass_rate * first_class_rate_among_qualified :=
by sorry

end first_class_product_rate_l464_46407


namespace octagon_area_in_square_l464_46464

/-- The area of a regular octagon inscribed in a square -/
theorem octagon_area_in_square (s : ℝ) (h : s = 4 + 2 * Real.sqrt 2) :
  let octagon_area := s^2 - 8
  octagon_area = 16 + 16 * Real.sqrt 2 := by sorry

end octagon_area_in_square_l464_46464


namespace equation_solutions_l464_46401

theorem equation_solutions : 
  ∀ m n : ℕ, 20^m - 10*m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2) := by
  sorry

end equation_solutions_l464_46401


namespace linear_congruence_intercepts_l464_46462

/-- Proves the properties of x-intercept and y-intercept for the linear congruence equation 5x ≡ 3y + 2 (mod 27) -/
theorem linear_congruence_intercepts :
  ∃ (x₀ y₀ : ℕ),
    x₀ < 27 ∧
    y₀ < 27 ∧
    (5 * x₀) % 27 = 2 ∧
    (3 * y₀) % 27 = 25 ∧
    x₀ + y₀ = 40 := by
  sorry

end linear_congruence_intercepts_l464_46462


namespace correct_quotient_proof_l464_46417

theorem correct_quotient_proof (D : ℕ) : 
  D % 21 = 0 →  -- The remainder is 0 when divided by 21
  D / 12 = 42 →  -- Dividing by 12 (incorrect divisor) yields 42
  D / 21 = 24  -- The correct quotient when dividing by 21 is 24
:= by sorry

end correct_quotient_proof_l464_46417


namespace cyclic_pentagon_area_diagonal_ratio_l464_46480

/-- A cyclic pentagon is a pentagon inscribed in a circle -/
structure CyclicPentagon where
  vertices : Fin 5 → ℝ × ℝ
  center : ℝ × ℝ
  radius : ℝ
  is_cyclic : ∀ i : Fin 5, dist (vertices i) center = radius

/-- The area of a cyclic pentagon -/
def area (p : CyclicPentagon) : ℝ := sorry

/-- The sum of the diagonals of a cyclic pentagon -/
def sum_diagonals (p : CyclicPentagon) : ℝ := sorry

/-- The theorem stating that the ratio of a cyclic pentagon's area to the sum of its diagonals
    is not greater than a quarter of its circumradius -/
theorem cyclic_pentagon_area_diagonal_ratio (p : CyclicPentagon) :
  area p / sum_diagonals p ≤ p.radius / 4 := by sorry

end cyclic_pentagon_area_diagonal_ratio_l464_46480


namespace ratio_HD_HA_is_5_11_l464_46408

/-- A triangle with sides of lengths 13, 14, and 15 -/
structure Triangle :=
  (a b c : ℝ)
  (side_a : a = 13)
  (side_b : b = 14)
  (side_c : c = 15)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The altitude from vertex A to the side of length 14 -/
def altitude_AD (t : Triangle) : ℝ := sorry

/-- The ratio of HD to HA -/
def ratio_HD_HA (t : Triangle) : ℚ := sorry

/-- Theorem: The ratio HD:HA is 5:11 -/
theorem ratio_HD_HA_is_5_11 (t : Triangle) : 
  ratio_HD_HA t = 5 / 11 := by sorry

end ratio_HD_HA_is_5_11_l464_46408


namespace inequality_solution_l464_46441

theorem inequality_solution (a : ℝ) (h : a ≠ 0) :
  let f := fun x => x^2 - 5*a*x + 6*a^2
  (∀ x, f x > 0 ↔ (a > 0 ∧ (x < 2*a ∨ x > 3*a)) ∨ (a < 0 ∧ (x < 3*a ∨ x > 2*a))) :=
by sorry

end inequality_solution_l464_46441


namespace square_ratio_bounds_l464_46402

theorem square_ratio_bounds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ∃ (m M : ℝ), 
    (0 ≤ m) ∧ 
    (M ≤ 1) ∧ 
    (∀ z w : ℝ, z ≠ 0 → w ≠ 0 → m ≤ ((|z + w| / (|z| + |w|))^2) ∧ ((|z + w| / (|z| + |w|))^2) ≤ M) ∧
    (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ((|a + b| / (|a| + |b|))^2) = m) ∧
    (∃ c d : ℝ, c ≠ 0 ∧ d ≠ 0 ∧ ((|c + d| / (|c| + |d|))^2) = M) ∧
    (M - m = 1) :=
by sorry

end square_ratio_bounds_l464_46402


namespace lottery_ratio_l464_46483

def lottery_problem (lottery_winnings : ℕ) (savings : ℕ) (fun_money : ℕ) : Prop :=
  let taxes := lottery_winnings / 2
  let after_taxes := lottery_winnings - taxes
  let investment := savings / 5
  let student_loans := after_taxes - (savings + investment + fun_money)
  (lottery_winnings = 12006 ∧ savings = 1000 ∧ fun_money = 2802) →
  (student_loans : ℚ) / after_taxes = 1 / 3

theorem lottery_ratio : 
  lottery_problem 12006 1000 2802 :=
sorry

end lottery_ratio_l464_46483


namespace complex_multiplication_l464_46446

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l464_46446


namespace chord_line_equation_l464_46439

/-- The equation of a line containing a chord of a parabola -/
theorem chord_line_equation (x y : ℝ → ℝ) :
  (∀ t : ℝ, (y t)^2 = -8 * (x t)) →  -- parabola equation
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    (x t₁ + x t₂) / 2 = -1 ∧ 
    (y t₁ + y t₂) / 2 = 1) →  -- midpoint condition
  ∃ a b c : ℝ, a ≠ 0 ∧ 
    (∀ t : ℝ, a * (x t) + b * (y t) + c = 0) ∧ 
    (4 * a = -b ∧ 3 * a = -c) :=  -- line equation
by sorry

end chord_line_equation_l464_46439


namespace megan_country_albums_l464_46474

theorem megan_country_albums :
  ∀ (country_albums pop_albums total_songs songs_per_album : ℕ),
    pop_albums = 8 →
    songs_per_album = 7 →
    total_songs = 70 →
    total_songs = country_albums * songs_per_album + pop_albums * songs_per_album →
    country_albums = 2 := by
  sorry

end megan_country_albums_l464_46474


namespace digit_difference_in_base_d_l464_46450

/-- Represents a digit in a given base -/
def Digit (d : ℕ) := {n : ℕ // n < d}

/-- Converts a two-digit number in base d to its decimal representation -/
def toDecimal (d : ℕ) (tens : Digit d) (ones : Digit d) : ℕ :=
  d * tens.val + ones.val

theorem digit_difference_in_base_d 
  (d : ℕ) (hd : d > 8) 
  (C : Digit d) (D : Digit d) 
  (h : toDecimal d C D + toDecimal d C C = d * d + 5 * d + 3) :
  C.val - D.val = 1 := by
sorry

end digit_difference_in_base_d_l464_46450


namespace friendly_match_schemes_l464_46442

/-- The number of ways to form two teams from teachers and students -/
def formTeams (numTeachers numStudents : ℕ) : ℕ :=
  let teacherCombinations := 1 -- Always select both teachers
  let studentCombinations := numStudents.choose 3
  let studentDistributions := 3 -- Ways to distribute 3 students into 2 teams
  teacherCombinations * studentCombinations * studentDistributions

/-- Theorem stating the number of ways to form teams in the given scenario -/
theorem friendly_match_schemes :
  formTeams 2 4 = 12 := by
  sorry

end friendly_match_schemes_l464_46442


namespace transmission_time_is_three_minutes_l464_46421

/-- The number of blocks to be sent -/
def num_blocks : ℕ := 150

/-- The number of chunks per block -/
def chunks_per_block : ℕ := 256

/-- The transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- The time it takes to send all blocks in minutes -/
def transmission_time : ℚ :=
  (num_blocks * chunks_per_block : ℚ) / transmission_rate / 60

theorem transmission_time_is_three_minutes :
  transmission_time = 3 := by
  sorry

end transmission_time_is_three_minutes_l464_46421


namespace cookies_in_fridge_l464_46419

/-- The number of cookies Uncle Jude baked -/
def total_cookies : ℕ := 1024

/-- The number of cookies given to Tim -/
def tim_cookies : ℕ := 48

/-- The number of cookies given to Mike -/
def mike_cookies : ℕ := 58

/-- The number of cookies given to Sarah -/
def sarah_cookies : ℕ := 78

/-- The number of cookies given to Anna -/
def anna_cookies : ℕ := 2 * (tim_cookies + mike_cookies) - sarah_cookies / 2

/-- The number of cookies Uncle Jude put in the fridge -/
def fridge_cookies : ℕ := total_cookies - (tim_cookies + mike_cookies + sarah_cookies + anna_cookies)

theorem cookies_in_fridge : fridge_cookies = 667 := by
  sorry

end cookies_in_fridge_l464_46419


namespace square_area_equal_perimeter_l464_46415

theorem square_area_equal_perimeter (a b c s : ℝ) : 
  a = 6 → b = 8 → c = 10 → -- Triangle side lengths
  a^2 + b^2 = c^2 →        -- Right-angled triangle condition
  4 * s = a + b + c →      -- Equal perimeter condition
  s^2 = 36 :=              -- Square area
by sorry

end square_area_equal_perimeter_l464_46415


namespace second_class_size_l464_46414

theorem second_class_size (students1 : ℕ) (avg1 : ℚ) (avg2 : ℚ) (avg_total : ℚ) :
  students1 = 25 →
  avg1 = 40 →
  avg2 = 60 →
  avg_total = 50.90909090909091 →
  ∃ students2 : ℕ, 
    students2 = 30 ∧
    (students1 * avg1 + students2 * avg2) / (students1 + students2 : ℚ) = avg_total :=
by sorry

end second_class_size_l464_46414


namespace problem_solution_l464_46499

theorem problem_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : Real.sqrt x / Real.sqrt y - Real.sqrt y / Real.sqrt x = 7/12)
  (h2 : x - y = 7) :
  x + y = 25 := by
  sorry

end problem_solution_l464_46499


namespace wallpapering_solution_l464_46456

/-- Represents the number of days needed to complete the wallpapering job -/
structure WallpaperingJob where
  worker1 : ℝ  -- Days needed for worker 1 to complete the job alone
  worker2 : ℝ  -- Days needed for worker 2 to complete the job alone

/-- The wallpapering job satisfies the given conditions -/
def satisfies_conditions (job : WallpaperingJob) : Prop :=
  -- Worker 1 needs 3 days more than Worker 2
  job.worker1 = job.worker2 + 3 ∧
  -- The combined work of both workers in 7 days equals the whole job
  (7 / job.worker1) + (5.5 / job.worker2) = 1

/-- The theorem stating the solution to the wallpapering problem -/
theorem wallpapering_solution :
  ∃ (job : WallpaperingJob), satisfies_conditions job ∧ job.worker1 = 14 ∧ job.worker2 = 11 := by
  sorry


end wallpapering_solution_l464_46456


namespace rectangle_area_rectangle_area_proof_l464_46445

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := circle_radius / 4
  let rectangle_area : ℝ := rectangle_length * rectangle_breadth
  rectangle_area

theorem rectangle_area_proof (h1 : square_area = 784) (h2 : rectangle_breadth = 5) :
  rectangle_area square_area rectangle_breadth = 35 := by
  sorry

end rectangle_area_rectangle_area_proof_l464_46445


namespace prob_second_good_is_five_ninths_l464_46458

/-- Represents the number of good transistors initially in the box -/
def initial_good : ℕ := 6

/-- Represents the number of bad transistors initially in the box -/
def initial_bad : ℕ := 4

/-- Represents the total number of transistors initially in the box -/
def initial_total : ℕ := initial_good + initial_bad

/-- Represents the probability of selecting a good transistor as the second one,
    given that the first one selected was good -/
def prob_second_good : ℚ := (initial_good - 1) / (initial_total - 1)

theorem prob_second_good_is_five_ninths :
  prob_second_good = 5 / 9 := by sorry

end prob_second_good_is_five_ninths_l464_46458


namespace solution_set_properties_l464_46488

def M (k : ℝ) : Set ℝ :=
  {x : ℝ | (k^2 + 2*k - 3)*x^2 + (k + 3)*x - 1 > 0}

theorem solution_set_properties (k : ℝ) :
  (M k = ∅ → k ∈ Set.Icc (-3 : ℝ) (1/5)) ∧
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ M k = Set.Ioo a b → k ∈ Set.Ioo (1/5 : ℝ) 1) :=
sorry

end solution_set_properties_l464_46488


namespace land_to_cabin_ratio_example_l464_46467

/-- Given a total cost and cabin cost, calculate the ratio of land cost to cabin cost -/
def land_to_cabin_ratio (total_cost cabin_cost : ℕ) : ℚ :=
  (total_cost - cabin_cost) / cabin_cost

/-- Theorem: The ratio of land cost to cabin cost is 4 when the total cost is $30,000 and the cabin cost is $6,000 -/
theorem land_to_cabin_ratio_example : land_to_cabin_ratio 30000 6000 = 4 := by
  sorry

end land_to_cabin_ratio_example_l464_46467


namespace triangle_abc_properties_l464_46405

def triangle_abc (a b c A B C : ℝ) : Prop :=
  b = c * (2 * Real.sin A + Real.cos A) ∧ 
  a = Real.sqrt 2 ∧ 
  B = 3 * Real.pi / 4

theorem triangle_abc_properties (a b c A B C : ℝ) 
  (h : triangle_abc a b c A B C) :
  Real.sin C = Real.sqrt 5 / 5 ∧ 
  (1/2) * a * c * Real.sin B = 1 := by
sorry

end triangle_abc_properties_l464_46405


namespace hyperbola_eccentricity_l464_46457

/-- The eccentricity of a hyperbola with the given properties is √3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ (c x y n : ℝ),
  -- Hyperbola equation
  x^2 / a^2 - y^2 / b^2 = 1 ∧
  -- M is on the hyperbola
  c^2 / a^2 - (b^2 / a)^2 / b^2 = 1 ∧
  -- F is a focus
  c^2 = a^2 + b^2 ∧
  -- M is center of circle
  (x - c)^2 + (y - n)^2 = (b^2 / a)^2 ∧
  -- Circle tangent to x-axis at F
  n = b^2 / a ∧
  -- Circle intersects y-axis
  c^2 + n^2 = (2 * n)^2 ∧
  -- MPQ is equilateral
  c^2 = 3 * n^2 →
  -- Eccentricity is √3
  c / a = Real.sqrt 3 := by
  sorry

end hyperbola_eccentricity_l464_46457


namespace simplify_and_evaluate_l464_46479

theorem simplify_and_evaluate : 
  let x : ℝ := 1
  let y : ℝ := -2
  7 * x * y - 2 * (5 * x * y - 2 * x^2 * y) + 3 * x * y = -8 := by
sorry

end simplify_and_evaluate_l464_46479


namespace f_x_plus_2_l464_46489

/-- Given a function f where f(x) = x(x-1)/2, prove that f(x+2) = (x+2)(x+1)/2 -/
theorem f_x_plus_2 (f : ℝ → ℝ) (h : ∀ x, f x = x * (x - 1) / 2) :
  ∀ x, f (x + 2) = (x + 2) * (x + 1) / 2 := by
  sorry

end f_x_plus_2_l464_46489


namespace white_bellied_minnows_count_l464_46478

/-- Proves the number of white-bellied minnows in a pond given the percentages of red, green, and white-bellied minnows and the number of red-bellied minnows. -/
theorem white_bellied_minnows_count 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (red_count : ℕ) 
  (h_red_percent : red_percent = 40 / 100)
  (h_green_percent : green_percent = 30 / 100)
  (h_red_count : red_count = 20)
  : ∃ (total : ℕ) (white_count : ℕ),
    red_percent * total = red_count ∧
    (1 - red_percent - green_percent) * total = white_count ∧
    white_count = 15 := by
  sorry

#check white_bellied_minnows_count

end white_bellied_minnows_count_l464_46478


namespace equation_solution_l464_46434

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.4 * x = 2.1 * x - 30) ∧ (x = -35 / 1.3) := by
  sorry

end equation_solution_l464_46434


namespace line_inclination_angle_ratio_l464_46448

theorem line_inclination_angle_ratio (θ : Real) : 
  (2 : Real) * Real.tan θ + 1 = 0 →
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 1/3 := by
  sorry

end line_inclination_angle_ratio_l464_46448


namespace factorization_xy_squared_minus_x_l464_46495

theorem factorization_xy_squared_minus_x (x y : ℝ) : x * y^2 - x = x * (y - 1) * (y + 1) := by
  sorry

end factorization_xy_squared_minus_x_l464_46495


namespace rock_paper_scissors_wins_l464_46425

/-- Represents the outcome of a single round --/
inductive RoundResult
| Win
| Lose
| Tie

/-- Represents a player's position and game results --/
structure PlayerState :=
  (position : Int)
  (wins : Nat)
  (losses : Nat)
  (ties : Nat)

/-- Updates a player's state based on the round result --/
def updatePlayerState (state : PlayerState) (result : RoundResult) : PlayerState :=
  match result with
  | RoundResult.Win => { state with position := state.position + 3, wins := state.wins + 1 }
  | RoundResult.Lose => { state with position := state.position - 2, losses := state.losses + 1 }
  | RoundResult.Tie => { state with position := state.position + 1, ties := state.ties + 1 }

/-- Represents the state of the game --/
structure GameState :=
  (playerA : PlayerState)
  (playerB : PlayerState)
  (rounds : Nat)

/-- Updates the game state based on the round result for Player A --/
def updateGameState (state : GameState) (result : RoundResult) : GameState :=
  { state with
    playerA := updatePlayerState state.playerA result,
    playerB := updatePlayerState state.playerB (match result with
      | RoundResult.Win => RoundResult.Lose
      | RoundResult.Lose => RoundResult.Win
      | RoundResult.Tie => RoundResult.Tie),
    rounds := state.rounds + 1 }

/-- The main theorem to prove --/
theorem rock_paper_scissors_wins
  (initialDistance : Nat)
  (totalRounds : Nat)
  (finalPositionA : Int)
  (finalPositionB : Int)
  (h1 : initialDistance = 30)
  (h2 : totalRounds = 15)
  (h3 : finalPositionA = 17)
  (h4 : finalPositionB = 2) :
  ∃ (gameResults : List RoundResult),
    let finalState := gameResults.foldl updateGameState
      { playerA := ⟨0, 0, 0, 0⟩,
        playerB := ⟨initialDistance, 0, 0, 0⟩,
        rounds := 0 }
    finalState.rounds = totalRounds ∧
    finalState.playerA.position = finalPositionA ∧
    finalState.playerB.position = finalPositionB ∧
    finalState.playerA.wins = 7 :=
sorry

end rock_paper_scissors_wins_l464_46425


namespace board_game_change_l464_46443

theorem board_game_change (num_games : ℕ) (game_cost : ℕ) (payment : ℕ) (change_bill : ℕ) : 
  num_games = 8 →
  game_cost = 18 →
  payment = 200 →
  change_bill = 10 →
  (payment - num_games * game_cost) / change_bill = 5 := by
sorry

end board_game_change_l464_46443


namespace digit_sum_property_l464_46422

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem digit_sum_property (M : ℕ) :
  (∀ k : ℕ, k > 0 → k ≤ M → S (M * k) = S M) ↔
  ∃ n : ℕ, n > 0 ∧ M = 10^n - 1 :=
sorry

end digit_sum_property_l464_46422


namespace min_value_expression_l464_46400

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (heq : a * b = 1 / 2) :
  (4 * a^2 + b^2 + 3) / (2 * a - b) ≥ 2 * Real.sqrt 5 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ > b₀ ∧ a₀ * b₀ = 1 / 2 ∧
    (4 * a₀^2 + b₀^2 + 3) / (2 * a₀ - b₀) = 2 * Real.sqrt 5 :=
by sorry

end min_value_expression_l464_46400


namespace problem_solution_l464_46487

theorem problem_solution (m n : ℝ) (h : |m - n - 5| + (2*m + n - 4)^2 = 0) : 
  3*m + n = 7 := by
  sorry

end problem_solution_l464_46487


namespace bed_weight_difference_bed_weight_difference_proof_l464_46476

theorem bed_weight_difference : ℝ → ℝ → Prop :=
  fun single_bed_weight double_bed_weight =>
    (5 * single_bed_weight = 50) →
    (2 * single_bed_weight + 4 * double_bed_weight = 100) →
    (double_bed_weight - single_bed_weight = 10)

-- The proof is omitted
theorem bed_weight_difference_proof : ∃ (s d : ℝ), bed_weight_difference s d :=
  sorry

end bed_weight_difference_bed_weight_difference_proof_l464_46476


namespace expression_simplification_l464_46403

theorem expression_simplification (y : ℝ) : 7*y + 8 - 2*y + 15 = 5*y + 23 := by
  sorry

end expression_simplification_l464_46403


namespace shorter_side_length_l464_46424

-- Define the rectangle
def Rectangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a ≥ b

-- Theorem statement
theorem shorter_side_length (a b : ℝ) 
  (h_rect : Rectangle a b) 
  (h_perim : 2 * a + 2 * b = 62) 
  (h_area : a * b = 240) : 
  b = 15 := by
  sorry

end shorter_side_length_l464_46424


namespace sum_equals_60_l464_46498

/-- An arithmetic sequence with specific terms. -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a3_eq_4 : a 3 = 4
  a101_eq_36 : a 101 = 36

/-- The sum of specific terms in the arithmetic sequence equals 60. -/
theorem sum_equals_60 (seq : ArithmeticSequence) : seq.a 9 + seq.a 52 + seq.a 95 = 60 := by
  sorry

end sum_equals_60_l464_46498


namespace bakery_items_l464_46420

theorem bakery_items (total : ℕ) (bread_rolls : ℕ) (bagels : ℕ) (croissants : ℕ)
  (h1 : total = 90)
  (h2 : bread_rolls = 49)
  (h3 : bagels = 22)
  (h4 : total = bread_rolls + croissants + bagels) :
  croissants = 19 := by
sorry

end bakery_items_l464_46420


namespace largest_divisor_of_expression_l464_46466

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (12*x + 2)*(12*x + 6)*(12*x + 10)*(6*x + 3) = 864 * k) ∧
  (∀ (m : ℤ), m > 864 → ∃ (y : ℤ), Odd y ∧ ¬∃ (l : ℤ), (12*y + 2)*(12*y + 6)*(12*y + 10)*(6*y + 3) = m * l) :=
by sorry

end largest_divisor_of_expression_l464_46466


namespace hyperbola_k_range_l464_46490

-- Define the hyperbola equation
def is_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (|k| - 2) + y^2 / (5 - k) = 1

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (-2 < k ∧ k < 2) ∨ k > 5

-- Theorem stating the relationship between the hyperbola equation and the range of k
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ k_range k :=
sorry

end hyperbola_k_range_l464_46490


namespace interest_group_count_l464_46449

/-- The number of students who joined at least one interest group -/
def students_in_interest_groups (science_tech : ℕ) (speech : ℕ) (both : ℕ) : ℕ :=
  science_tech + speech - both

theorem interest_group_count : 
  students_in_interest_groups 65 35 20 = 80 := by
sorry

end interest_group_count_l464_46449


namespace point_in_same_region_l464_46413

/-- The line equation -/
def line_equation (x y : ℝ) : ℝ := 3*x + 2*y + 5

/-- Definition of being in the same region -/
def same_region (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (line_equation x₁ y₁ > 0 ∧ line_equation x₂ y₂ > 0) ∨
  (line_equation x₁ y₁ < 0 ∧ line_equation x₂ y₂ < 0)

/-- Theorem stating that (-3,4) is in the same region as (0,0) -/
theorem point_in_same_region : same_region (-3) 4 0 0 := by
  sorry

end point_in_same_region_l464_46413


namespace square_position_after_2023_transformations_l464_46438

-- Define a square as a list of four vertices
def Square := List Char

-- Define the transformations
def rotate90CW (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [d, a, b, c]
  | _ => s

def reflectVertical (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [c, b, a, d]
  | _ => s

def rotate180 (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [c, d, a, b]
  | _ => s

-- Define the sequence of transformations
def transform (s : Square) (n : Nat) : Square :=
  match n % 3 with
  | 0 => rotate180 s
  | 1 => rotate90CW s
  | _ => reflectVertical s

-- Main theorem
theorem square_position_after_2023_transformations (initial : Square) :
  initial = ['A', 'B', 'C', 'D'] →
  (transform initial 2023) = ['C', 'B', 'A', 'D'] := by
  sorry


end square_position_after_2023_transformations_l464_46438


namespace equation_solution_l464_46481

theorem equation_solution (z : ℝ) (some_number : ℝ) :
  (14 * (-1 + z) + some_number = -14 * (1 - z) - 10) →
  some_number = -10 := by
sorry

end equation_solution_l464_46481


namespace factorization_equality_l464_46497

theorem factorization_equality (a : ℝ) : 2 * a^2 - 2 * a + (1/2 : ℝ) = 2 * (a - 1/2)^2 := by
  sorry

end factorization_equality_l464_46497


namespace koshchey_chest_count_l464_46409

/-- Represents the number of chests Koshchey has -/
structure KoshcheyChests where
  large : ℕ
  medium : ℕ
  small : ℕ
  empty : ℕ

/-- The total number of chests Koshchey has -/
def total_chests (k : KoshcheyChests) : ℕ :=
  k.large + k.medium + k.small

/-- Koshchey's chest configuration satisfies the problem conditions -/
def is_valid_configuration (k : KoshcheyChests) : Prop :=
  k.large = 11 ∧
  k.empty = 102 ∧
  ∃ (x : ℕ), x ≤ k.large ∧ k.medium = 8 * x

theorem koshchey_chest_count (k : KoshcheyChests) 
  (h : is_valid_configuration k) : total_chests k = 115 :=
by sorry

end koshchey_chest_count_l464_46409


namespace probability_one_red_one_green_l464_46459

def total_marbles : ℕ := 4 + 6 + 11

def prob_red_then_green : ℚ :=
  (4 : ℚ) / total_marbles * 6 / (total_marbles - 1)

def prob_green_then_red : ℚ :=
  (6 : ℚ) / total_marbles * 4 / (total_marbles - 1)

theorem probability_one_red_one_green :
  prob_red_then_green + prob_green_then_red = 4 / 35 := by
  sorry

end probability_one_red_one_green_l464_46459


namespace specific_tetrahedron_volume_l464_46494

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  pq : ℝ
  pr : ℝ
  ps : ℝ
  qr : ℝ
  qs : ℝ
  rs : ℝ

/-- Calculates the volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 18 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    pq := 6,
    pr := 4,
    ps := 5,
    qr := 5,
    qs := 7,
    rs := 8
  }
  tetrahedronVolume t = 18 := by
  sorry

end specific_tetrahedron_volume_l464_46494


namespace clothing_factory_production_adjustment_l464_46454

/-- Represents the scenario of a clothing factory adjusting its production rate -/
theorem clothing_factory_production_adjustment 
  (total_pieces : ℕ) 
  (original_rate : ℕ) 
  (days_earlier : ℕ) 
  (x : ℝ) 
  (h1 : total_pieces = 720)
  (h2 : original_rate = 48)
  (h3 : days_earlier = 5) :
  (total_pieces : ℝ) / original_rate - total_pieces / (x + original_rate) = days_earlier :=
by sorry

end clothing_factory_production_adjustment_l464_46454


namespace even_function_implies_m_equals_two_l464_46468

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (m-1)x^2 + (m-2)x + (m^2 - 7m + 12) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 + (m - 2) * x + (m^2 - 7*m + 12)

theorem even_function_implies_m_equals_two :
  ∀ m : ℝ, IsEven (f m) → m = 2 := by
  sorry

end even_function_implies_m_equals_two_l464_46468


namespace min_sum_squares_given_cubic_constraint_l464_46493

/-- Given real numbers x, y, and z satisfying x^3 + y^3 + z^3 - 3xyz = 1,
    the sum of their squares x^2 + y^2 + z^2 is always greater than or equal to 1 -/
theorem min_sum_squares_given_cubic_constraint (x y z : ℝ) 
    (h : x^3 + y^3 + z^3 - 3*x*y*z = 1) : 
    x^2 + y^2 + z^2 ≥ 1 := by
  sorry

#check min_sum_squares_given_cubic_constraint

end min_sum_squares_given_cubic_constraint_l464_46493


namespace factorial_plus_twelve_square_l464_46463

theorem factorial_plus_twelve_square (m n : ℕ) : m.factorial + 12 = n^2 ↔ m = 4 ∧ n = 6 := by
  sorry

end factorial_plus_twelve_square_l464_46463


namespace quadratic_points_range_l464_46484

/-- Given a quadratic function f(x) = -x^2 - 2x + 3, prove that if (a, m) and (a+2, n) are points on the graph of f, and m ≥ n, then a ≥ -2. -/
theorem quadratic_points_range (a m n : ℝ) : 
  (m = -a^2 - 2*a + 3) → 
  (n = -(a+2)^2 - 2*(a+2) + 3) → 
  (m ≥ n) → 
  (a ≥ -2) := by
sorry

end quadratic_points_range_l464_46484


namespace carpenter_woodblocks_l464_46406

theorem carpenter_woodblocks (total_needed : ℕ) (current_logs : ℕ) (additional_logs : ℕ) 
  (h1 : total_needed = 80)
  (h2 : current_logs = 8)
  (h3 : additional_logs = 8) :
  (total_needed / (current_logs + additional_logs) : ℕ) = 5 := by
  sorry

end carpenter_woodblocks_l464_46406


namespace building_height_calculation_l464_46430

/-- Given a building and a pole, calculate the height of the building using similar triangles. -/
theorem building_height_calculation (building_shadow : ℝ) (pole_height : ℝ) (pole_shadow : ℝ)
  (h_building_shadow : building_shadow = 20)
  (h_pole_height : pole_height = 2)
  (h_pole_shadow : pole_shadow = 3) :
  (pole_height / pole_shadow) * building_shadow = 40 / 3 := by
  sorry

end building_height_calculation_l464_46430


namespace f_extrema_l464_46416

def f (x : ℝ) := x^2 - 2*x

theorem f_extrema :
  ∀ x ∈ Set.Icc (-1 : ℝ) 5,
    -1 ≤ f x ∧ f x ≤ 15 ∧
    (∃ x₁ ∈ Set.Icc (-1 : ℝ) 5, f x₁ = -1) ∧
    (∃ x₂ ∈ Set.Icc (-1 : ℝ) 5, f x₂ = 15) :=
by
  sorry

end f_extrema_l464_46416


namespace cary_earnings_l464_46436

/-- Calculates the total net earnings over three years for an employee named Cary --/
def total_net_earnings (initial_wage : ℚ) : ℚ :=
  let year1_base_wage := initial_wage
  let year1_hours := 40 * 50
  let year1_gross := year1_hours * year1_base_wage + 500
  let year1_net := year1_gross * (1 - 0.2)

  let year2_base_wage := year1_base_wage * 1.2 * 0.75
  let year2_regular_hours := 40 * 51
  let year2_overtime_hours := 10 * 51
  let year2_gross := year2_regular_hours * year2_base_wage + 
                     year2_overtime_hours * (year2_base_wage * 1.5) - 300
  let year2_net := year2_gross * (1 - 0.22)

  let year3_base_wage := year2_base_wage * 1.1
  let year3_hours := 40 * 50
  let year3_gross := year3_hours * year3_base_wage + 1000
  let year3_net := year3_gross * (1 - 0.18)

  year1_net + year2_net + year3_net

/-- Theorem stating that Cary's total net earnings over three years equals $52,913.10 --/
theorem cary_earnings : total_net_earnings 10 = 52913.1 := by
  sorry

end cary_earnings_l464_46436


namespace arithmetic_sequence_formula_l464_46418

def arithmetic_sequence (a : ℝ) (n : ℕ) : ℝ := a + (n - 1) * (a + 1 - (a - 1))

theorem arithmetic_sequence_formula (a : ℝ) :
  (arithmetic_sequence a 1 = a - 1) ∧
  (arithmetic_sequence a 2 = a + 1) ∧
  (arithmetic_sequence a 3 = 2 * a + 3) →
  ∀ n : ℕ, arithmetic_sequence a n = 2 * n - 3 :=
by
  sorry

#check arithmetic_sequence_formula

end arithmetic_sequence_formula_l464_46418


namespace exactly_two_successes_out_of_three_l464_46471

/-- The probability of making a successful shot -/
def p : ℚ := 2 / 3

/-- The number of attempts -/
def n : ℕ := 3

/-- The number of successful shots -/
def k : ℕ := 2

/-- The probability of making exactly k successful shots out of n attempts -/
def probability_k_successes : ℚ := 
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem exactly_two_successes_out_of_three : 
  probability_k_successes = 4 / 9 := by
  sorry

end exactly_two_successes_out_of_three_l464_46471


namespace zero_in_interval_one_two_l464_46429

noncomputable def f (x : ℝ) := Real.exp x + 2 * x - 6

theorem zero_in_interval_one_two :
  ∃ z ∈ Set.Ioo 1 2, f z = 0 := by
  sorry

end zero_in_interval_one_two_l464_46429


namespace tv_screen_area_difference_l464_46460

theorem tv_screen_area_difference : 
  let diagonal_large : ℝ := 22
  let diagonal_small : ℝ := 20
  let area_large := diagonal_large ^ 2
  let area_small := diagonal_small ^ 2
  area_large - area_small = 84 := by
  sorry

end tv_screen_area_difference_l464_46460


namespace cos_two_pi_thirds_plus_two_alpha_l464_46437

theorem cos_two_pi_thirds_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos ((2 * π) / 3 + 2 * α) = -7 / 9 := by
  sorry

end cos_two_pi_thirds_plus_two_alpha_l464_46437


namespace parabola_constant_l464_46411

theorem parabola_constant (c : ℝ) : 
  (∃ (x y : ℝ), y = x^2 - c ∧ x = 3 ∧ y = 8) → c = 1 := by
  sorry

end parabola_constant_l464_46411


namespace product_of_solutions_abs_value_equation_l464_46410

theorem product_of_solutions_abs_value_equation :
  ∃ (x₁ x₂ : ℝ), (|x₁| = 3 * (|x₁| - 2) ∧ |x₂| = 3 * (|x₂| - 2) ∧ x₁ ≠ x₂) ∧ x₁ * x₂ = -9 :=
by sorry

end product_of_solutions_abs_value_equation_l464_46410


namespace tangent_line_equation_l464_46447

/-- The function f(x) = x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := 2 * x

/-- The point of tangency -/
def tangent_point : ℝ := 1

/-- The slope of the tangent line at x = 1 -/
def tangent_slope : ℝ := f_deriv tangent_point

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := -(tangent_slope * tangent_point - f tangent_point)

theorem tangent_line_equation :
  ∀ x y : ℝ, y = tangent_slope * x + y_intercept ↔ y = 2 * x - 2 :=
by sorry

end tangent_line_equation_l464_46447


namespace complex_number_location_l464_46491

theorem complex_number_location (z : ℂ) (h : z * (1 + Complex.I) * (-2 * Complex.I) = z) :
  Complex.re z > 0 ∧ Complex.im z < 0 :=
sorry

end complex_number_location_l464_46491
