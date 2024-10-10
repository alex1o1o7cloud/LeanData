import Mathlib

namespace number_of_divisors_3003_l2105_210594

theorem number_of_divisors_3003 : Finset.card (Nat.divisors 3003) = 16 := by
  sorry

end number_of_divisors_3003_l2105_210594


namespace intersection_complement_when_m_2_union_equals_B_iff_l2105_210566

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 6}

theorem intersection_complement_when_m_2 :
  A ∩ (Bᶜ 2) = {x | -1 ≤ x ∧ x < 2} := by sorry

theorem union_equals_B_iff (m : ℝ) :
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 := by sorry

end intersection_complement_when_m_2_union_equals_B_iff_l2105_210566


namespace largest_x_absolute_value_equation_l2105_210520

theorem largest_x_absolute_value_equation :
  ∃ (x : ℝ), x = 17 ∧ |2*x - 4| = 30 ∧ ∀ (y : ℝ), |2*y - 4| = 30 → y ≤ x :=
by sorry

end largest_x_absolute_value_equation_l2105_210520


namespace digit_1234_is_4_l2105_210508

/-- The number of digits in the representation of an integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The nth digit in the decimal expansion of x -/
def nth_digit (x : ℝ) (n : ℕ) : ℕ := sorry

/-- The number formed by concatenating the decimal representations of integers from 1 to n -/
def concat_integers (n : ℕ) : ℝ := sorry

theorem digit_1234_is_4 :
  let x := concat_integers 500
  nth_digit x 1234 = 4 := by sorry

end digit_1234_is_4_l2105_210508


namespace tangent_parallel_point_l2105_210558

theorem tangent_parallel_point (x y : ℝ) : 
  y = Real.exp x → -- Point A (x, y) is on the curve y = e^x
  (Real.exp x) = 1 → -- Tangent at A is parallel to x - y + 3 = 0 (slope = 1)
  x = 0 ∧ y = 1 := by
sorry

end tangent_parallel_point_l2105_210558


namespace fraction_evaluation_l2105_210559

theorem fraction_evaluation (a b : ℚ) (ha : a = 5) (hb : b = -2) : 
  5 / (a + b) = 5 / 3 := by sorry

end fraction_evaluation_l2105_210559


namespace calculate_expression_l2105_210598

theorem calculate_expression : 15 * 30 + 45 * 15 = 1125 := by
  sorry

end calculate_expression_l2105_210598


namespace ellipse_condition_l2105_210583

/-- The equation represents an ellipse with foci on the x-axis -/
def is_ellipse_on_x_axis (n : ℝ) : Prop :=
  2 - n > 0 ∧ n + 1 > 0 ∧ 2 - n > n + 1

/-- The condition -1 < n < 2 is sufficient but not necessary for the equation to represent an ellipse with foci on the x-axis -/
theorem ellipse_condition (n : ℝ) :
  ((-1 < n ∧ n < 2) → is_ellipse_on_x_axis n) ∧
  ¬(is_ellipse_on_x_axis n → (-1 < n ∧ n < 2)) :=
sorry

end ellipse_condition_l2105_210583


namespace ceiling_minus_x_eq_half_l2105_210505

theorem ceiling_minus_x_eq_half (x : ℝ) (h : x - ⌊x⌋ = 1/2) : ⌈x⌉ - x = 1/2 := by
  sorry

end ceiling_minus_x_eq_half_l2105_210505


namespace square_root_meaningful_l2105_210518

theorem square_root_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 5) → x ≥ 5 :=
by sorry

end square_root_meaningful_l2105_210518


namespace ap_terms_count_l2105_210551

theorem ap_terms_count (n : ℕ) (a d : ℝ) : 
  n % 2 = 0 ∧ 
  n > 0 ∧
  (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 30 ∧ 
  (n / 2 : ℝ) * (2 * a + n * d) = 36 ∧ 
  a + (n - 1) * d - a = 15 → 
  n = 6 := by sorry

end ap_terms_count_l2105_210551


namespace fraction_subtraction_l2105_210576

theorem fraction_subtraction : 
  (1 + 4 + 7) / (2 + 5 + 8) - (2 + 5 + 8) / (1 + 4 + 7) = -9 / 20 := by
  sorry

end fraction_subtraction_l2105_210576


namespace otimes_neg_two_neg_one_l2105_210549

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a^2 - |b|

-- Theorem to prove
theorem otimes_neg_two_neg_one : otimes (-2) (-1) = 3 := by
  sorry

end otimes_neg_two_neg_one_l2105_210549


namespace triangle_area_condition_l2105_210513

/-- The area of the triangle formed by the line x - 2y + 2m = 0 and the coordinate axes is not less than 1 if and only if m ∈ (-∞, -1] ∪ [1, +∞) -/
theorem triangle_area_condition (m : ℝ) : 
  (∃ (x y : ℝ), x - 2*y + 2*m = 0 ∧ 
   (1/2) * |x| * |y| ≥ 1) ↔ 
  (m ≤ -1 ∨ m ≥ 1) :=
by sorry

end triangle_area_condition_l2105_210513


namespace g_18_value_l2105_210587

-- Define the properties of g
def is_valid_g (g : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, g (n + 1) > g n) ∧ 
  (∀ m n : ℕ+, g (m * n) = g m * g n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → g m = n ^ 2 ∨ g n = m ^ 2)

-- State the theorem
theorem g_18_value (g : ℕ+ → ℕ+) (h : is_valid_g g) : g 18 = 104976 := by
  sorry

end g_18_value_l2105_210587


namespace division_problem_l2105_210547

theorem division_problem (L S Q : ℕ) : 
  L - S = 1515 →
  L = 1600 →
  L = Q * S + 15 →
  Q = 18 := by
sorry

end division_problem_l2105_210547


namespace paul_reading_theorem_l2105_210542

/-- Calculates the total number of books read given a weekly reading rate and number of weeks -/
def total_books_read (books_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  books_per_week * num_weeks

/-- Proves that reading 4 books per week for 5 weeks results in 20 books read -/
theorem paul_reading_theorem : 
  total_books_read 4 5 = 20 := by
  sorry

end paul_reading_theorem_l2105_210542


namespace complex_number_equality_l2105_210565

theorem complex_number_equality (a b : ℝ) (i : ℂ) : 
  i * i = -1 →
  (a - 2 * i) * i = b - i →
  (a + b * i : ℂ) = -1 + 2 * i := by
sorry

end complex_number_equality_l2105_210565


namespace olga_aquarium_fish_count_l2105_210501

/-- The number of fish in Olga's aquarium -/
def fish_count (yellow blue green : ℕ) : ℕ := yellow + blue + green

/-- Theorem stating the total number of fish in Olga's aquarium -/
theorem olga_aquarium_fish_count :
  ∀ (yellow blue green : ℕ),
    yellow = 12 →
    blue = yellow / 2 →
    green = yellow * 2 →
    fish_count yellow blue green = 42 :=
by
  sorry

#check olga_aquarium_fish_count

end olga_aquarium_fish_count_l2105_210501


namespace largest_square_from_wire_l2105_210524

/-- Given a wire of length 28 centimeters forming the largest possible square,
    the length of one side of the square is 7 centimeters. -/
theorem largest_square_from_wire (wire_length : ℝ) (side_length : ℝ) :
  wire_length = 28 →
  side_length * 4 = wire_length →
  side_length = 7 := by sorry

end largest_square_from_wire_l2105_210524


namespace sum_of_remaining_digits_l2105_210595

theorem sum_of_remaining_digits 
  (total_count : Nat) 
  (known_count : Nat) 
  (total_average : ℚ) 
  (known_average : ℚ) 
  (h1 : total_count = 20) 
  (h2 : known_count = 14) 
  (h3 : total_average = 500) 
  (h4 : known_average = 390) :
  (total_count : ℚ) * total_average - (known_count : ℚ) * known_average = 4540 := by
  sorry

end sum_of_remaining_digits_l2105_210595


namespace at_least_two_same_connections_l2105_210536

-- Define the type for interns
def Intern : Type := ℕ

-- Define the knowing relation
def knows : Intern → Intern → Prop := sorry

-- The number of interns
def num_interns : ℕ := 80

-- The knowing relation is symmetric
axiom knows_symmetric : ∀ (a b : Intern), knows a b ↔ knows b a

-- Function to count how many interns a given intern knows
def num_known (i : Intern) : ℕ := sorry

-- Theorem statement
theorem at_least_two_same_connections : 
  ∃ (i j : Intern), i ≠ j ∧ num_known i = num_known j :=
sorry

end at_least_two_same_connections_l2105_210536


namespace scientific_notation_3462_23_l2105_210507

theorem scientific_notation_3462_23 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 3462.23 = a * (10 : ℝ) ^ n ∧ a = 3.46223 ∧ n = 3 := by
  sorry

end scientific_notation_3462_23_l2105_210507


namespace maria_carrots_l2105_210593

def total_carrots (initial : ℕ) (thrown_out : ℕ) (additional : ℕ) : ℕ :=
  initial - thrown_out + additional

theorem maria_carrots : total_carrots 48 11 15 = 52 := by
  sorry

end maria_carrots_l2105_210593


namespace constant_term_expansion_l2105_210556

/-- The constant term in the expansion of (x^2 + 1/x^3)^5 -/
def constant_term : ℕ := 10

/-- The binomial coefficient C(5,2) -/
def C_5_2 : ℕ := Nat.choose 5 2

theorem constant_term_expansion :
  constant_term = C_5_2 :=
sorry

end constant_term_expansion_l2105_210556


namespace det_trig_matrix_zero_l2105_210546

theorem det_trig_matrix_zero (a c : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1, Real.sin (a + c), Real.sin a; 
                                        Real.sin (a + c), 1, Real.sin c; 
                                        Real.sin a, Real.sin c, 1]
  Matrix.det M = 0 := by
  sorry

end det_trig_matrix_zero_l2105_210546


namespace crunch_difference_l2105_210579

/-- Given that Zachary did 17 crunches and David did 4 crunches,
    prove that David did 13 less crunches than Zachary. -/
theorem crunch_difference (zachary_crunches : ℕ) (david_crunches : ℕ)
  (h1 : zachary_crunches = 17)
  (h2 : david_crunches = 4) :
  zachary_crunches - david_crunches = 13 := by
  sorry

end crunch_difference_l2105_210579


namespace afternoon_campers_calculation_l2105_210560

-- Define the number of campers who went rowing in the morning
def morning_campers : ℝ := 15.5

-- Define the total number of campers who went rowing that day
def total_campers : ℝ := 32.75

-- Define the number of campers who went rowing in the afternoon
def afternoon_campers : ℝ := total_campers - morning_campers

-- Theorem to prove
theorem afternoon_campers_calculation :
  afternoon_campers = 17.25 := by sorry

end afternoon_campers_calculation_l2105_210560


namespace cubic_polynomial_interpolation_l2105_210564

-- Define the set of cubic polynomials over ℝ
def CubicPolynomial : Type := ℝ → ℝ

-- Define the property of being a cubic polynomial
def IsCubicPolynomial (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x : ℝ, p x = a * x^3 + b * x^2 + c * x + d

-- Theorem statement
theorem cubic_polynomial_interpolation
  (P Q R : CubicPolynomial)
  (hP : IsCubicPolynomial P)
  (hQ : IsCubicPolynomial Q)
  (hR : IsCubicPolynomial R)
  (h_order : ∀ x : ℝ, P x ≤ Q x ∧ Q x ≤ R x)
  (h_equal : ∃ x₀ : ℝ, P x₀ = R x₀) :
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ ∀ x : ℝ, Q x = k * P x + (1 - k) * R x :=
sorry

end cubic_polynomial_interpolation_l2105_210564


namespace repel_creatures_l2105_210582

/-- Represents the number of cloves needed to repel creatures -/
def cloves_needed (vampires wights vampire_bats : ℕ) : ℕ :=
  let vampires_cloves := (3 * vampires + 1) / 2
  let wights_cloves := wights
  let bats_cloves := (3 * vampire_bats + 7) / 8
  vampires_cloves + wights_cloves + bats_cloves

/-- Theorem stating the number of cloves needed to repel specific numbers of creatures -/
theorem repel_creatures : cloves_needed 30 12 40 = 72 := by
  sorry

end repel_creatures_l2105_210582


namespace pokemon_card_difference_l2105_210548

theorem pokemon_card_difference : ∀ (orlando_cards : ℕ),
  orlando_cards > 6 →
  6 + orlando_cards + 3 * orlando_cards = 38 →
  orlando_cards - 6 = 2 := by
sorry

end pokemon_card_difference_l2105_210548


namespace inequality_proof_l2105_210525

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b * c) + b / (a * c) + c / (a * b) ≥ 2 / a + 2 / b - 2 / c := by
  sorry

end inequality_proof_l2105_210525


namespace survey_questions_l2105_210531

-- Define the number of questions per survey
def questionsPerSurvey : ℕ := sorry

-- Define the payment per question
def paymentPerQuestion : ℚ := 1/5

-- Define the number of surveys completed on Monday
def mondaySurveys : ℕ := 3

-- Define the number of surveys completed on Tuesday
def tuesdaySurveys : ℕ := 4

-- Define the total earnings
def totalEarnings : ℚ := 14

-- Theorem statement
theorem survey_questions :
  questionsPerSurvey * (mondaySurveys + tuesdaySurveys : ℚ) * paymentPerQuestion = totalEarnings ∧
  questionsPerSurvey = 10 := by
  sorry

end survey_questions_l2105_210531


namespace wild_sorghum_and_corn_different_species_reproductive_isolation_wild_sorghum_corn_l2105_210522

/-- Represents a plant species -/
structure Species where
  name : String
  chromosomes : Nat

/-- Defines reproductive isolation between two species -/
def reproductiveIsolation (s1 s2 : Species) : Prop :=
  s1.chromosomes ≠ s2.chromosomes

/-- Defines whether two species are the same -/
def sameSpecies (s1 s2 : Species) : Prop :=
  s1.chromosomes = s2.chromosomes ∧ ¬reproductiveIsolation s1 s2

/-- Wild sorghum species -/
def wildSorghum : Species :=
  { name := "Wild Sorghum", chromosomes := 22 }

/-- Corn species -/
def corn : Species :=
  { name := "Corn", chromosomes := 20 }

/-- Theorem stating that wild sorghum and corn are not the same species -/
theorem wild_sorghum_and_corn_different_species :
  ¬sameSpecies wildSorghum corn :=
by
  sorry

/-- Theorem stating that there is reproductive isolation between wild sorghum and corn -/
theorem reproductive_isolation_wild_sorghum_corn :
  reproductiveIsolation wildSorghum corn :=
by
  sorry

end wild_sorghum_and_corn_different_species_reproductive_isolation_wild_sorghum_corn_l2105_210522


namespace sin_negative_thirty_degrees_l2105_210540

theorem sin_negative_thirty_degrees :
  let θ : Real := 30 * Real.pi / 180
  (∀ x, Real.sin (-x) = -Real.sin x) →  -- sine is an odd function
  Real.sin θ = 1/2 →                    -- sin 30° = 1/2
  Real.sin (-θ) = -1/2 := by
    sorry

end sin_negative_thirty_degrees_l2105_210540


namespace perpendicular_bisector_y_intercept_range_l2105_210506

/-- Given two distinct points on a parabola y = 2x², prove that the y-intercept of their perpendicular bisector with slope 2 is greater than 9/32. -/
theorem perpendicular_bisector_y_intercept_range 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h_distinct : (x₁, y₁) ≠ (x₂, y₂))
  (h_parabola₁ : y₁ = 2 * x₁^2)
  (h_parabola₂ : y₂ = 2 * x₂^2)
  (b : ℝ) 
  (h_perpendicular_bisector : ∃ (m : ℝ), 
    y₁ = -1/(2*m) * x₁ + b + 1/(4*m) ∧ 
    y₂ = -1/(2*m) * x₂ + b + 1/(4*m) ∧ 
    m = 2) : 
  b > 9/32 := by
  sorry

end perpendicular_bisector_y_intercept_range_l2105_210506


namespace triangle_perimeter_l2105_210580

theorem triangle_perimeter (a : ℕ) (h1 : 2 < a) (h2 : a < 8) (h3 : Even a) :
  2 + 6 + a = 14 :=
sorry

end triangle_perimeter_l2105_210580


namespace brian_read_chapters_l2105_210570

/-- The number of chapters Brian read -/
def total_chapters (book1 book2 book3 book4 : ℕ) : ℕ :=
  book1 + book2 + book3 + book4

/-- The theorem stating the total number of chapters Brian read -/
theorem brian_read_chapters : ∃ (book4 : ℕ),
  let book1 := 20
  let book2 := 15
  let book3 := 15
  book4 = (book1 + book2 + book3) / 2 ∧
  total_chapters book1 book2 book3 book4 = 75 := by
  sorry

end brian_read_chapters_l2105_210570


namespace min_triangle_area_l2105_210528

/-- Given a line that passes through (1,2) and intersects positive semi-axes, 
    prove that the minimum area of the triangle formed is 4 -/
theorem min_triangle_area (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_line : 1/m + 2/n = 1) : 
  ∃ (A B : ℝ × ℝ), 
    A.1 > 0 ∧ A.2 = 0 ∧ 
    B.1 = 0 ∧ B.2 > 0 ∧
    (∀ (x y : ℝ), x/m + y/n = 1 → (x = A.1 ∧ y = 0) ∨ (x = 0 ∧ y = B.2)) ∧
    (∀ (C : ℝ × ℝ), C.1 > 0 ∧ C.2 > 0 ∧ C.1/m + C.2/n = 1 → 
      1/2 * A.1 * B.2 ≥ 4) :=
sorry

end min_triangle_area_l2105_210528


namespace other_root_of_quadratic_l2105_210569

theorem other_root_of_quadratic (m : ℝ) :
  (3 * (1 : ℝ)^2 + m * 1 = 5) →
  (3 * (-5/3 : ℝ)^2 + m * (-5/3) = 5) ∧
  (∀ x : ℝ, 3 * x^2 + m * x = 5 → x = 1 ∨ x = -5/3) :=
by sorry

end other_root_of_quadratic_l2105_210569


namespace complex_modulus_equation_l2105_210517

theorem complex_modulus_equation : ∃ (n : ℝ), n > 0 ∧ Complex.abs (5 + n * Complex.I) = 5 * Real.sqrt 26 ∧ n = 25 := by
  sorry

end complex_modulus_equation_l2105_210517


namespace platform_length_l2105_210512

/-- Given a train of length 300 meters that crosses a signal pole in 20 seconds
    and a platform in 39 seconds, the length of the platform is 285 meters. -/
theorem platform_length (train_length : ℝ) (pole_time : ℝ) (platform_time : ℝ) :
  train_length = 300 →
  pole_time = 20 →
  platform_time = 39 →
  ∃ platform_length : ℝ,
    platform_length = 285 ∧
    train_length / pole_time * platform_time = train_length + platform_length :=
by
  sorry


end platform_length_l2105_210512


namespace smallest_n_dividing_m_pow_n_minus_one_l2105_210574

theorem smallest_n_dividing_m_pow_n_minus_one (m : ℕ) (h_m_odd : Odd m) (h_m_gt_1 : m > 1) :
  (∀ n : ℕ, n > 0 → (2^1989 ∣ m^n - 1)) ↔ n ≥ 2^1987 :=
by sorry

end smallest_n_dividing_m_pow_n_minus_one_l2105_210574


namespace unique_sequence_exists_l2105_210537

def sequence_condition (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 = a n * a (n + 2) - 1

theorem unique_sequence_exists : ∃! a : ℕ → ℕ, sequence_condition a := by
  sorry

end unique_sequence_exists_l2105_210537


namespace clock_correction_time_l2105_210597

/-- The number of minutes in 12 hours -/
def minutes_in_12_hours : ℕ := 12 * 60

/-- The number of minutes the clock gains per day -/
def minutes_gained_per_day : ℕ := 3

/-- The minimum number of days for the clock to show the correct time again -/
def min_days_to_correct_time : ℕ := minutes_in_12_hours / minutes_gained_per_day

theorem clock_correction_time :
  min_days_to_correct_time = 240 :=
sorry

end clock_correction_time_l2105_210597


namespace trigonometric_equation_solution_l2105_210532

theorem trigonometric_equation_solution (k : ℤ) :
  let x₁ := π / 60 + k * π / 10
  let x₂ := -π / 24 - k * π / 4
  (∀ x, x = x₁ ∨ x = x₂ → (Real.sin (3 * x) + Real.sqrt 3 * Real.cos (3 * x))^2 - 2 * Real.cos (14 * x) = 2) := by
  sorry

end trigonometric_equation_solution_l2105_210532


namespace integral_f_equals_344_over_15_l2105_210584

-- Define the function to be integrated
def f (x : ℝ) : ℝ := (x^2 + 2*x - 3) * (4*x^2 - x + 1)

-- State the theorem
theorem integral_f_equals_344_over_15 : 
  ∫ x in (0)..(2), f x = 344 / 15 := by sorry

end integral_f_equals_344_over_15_l2105_210584


namespace fraction_subtraction_l2105_210535

theorem fraction_subtraction : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end fraction_subtraction_l2105_210535


namespace find_divisor_l2105_210596

theorem find_divisor (dividend quotient : ℕ) (h1 : dividend = 62976) (h2 : quotient = 123) :
  dividend / quotient = 512 := by
  sorry

end find_divisor_l2105_210596


namespace another_max_occurrence_sequence_l2105_210519

/-- Represents a circular strip of zeros and ones -/
def CircularStrip := List Bool

/-- Counts the number of occurrences of a sequence in a circular strip -/
def count_occurrences (strip : CircularStrip) (seq : List Bool) : Nat :=
  sorry

/-- The sequence with the maximum number of occurrences -/
def max_seq (n : Nat) : List Bool :=
  [true, true] ++ List.replicate (n - 2) false

/-- The sequence with the minimum number of occurrences -/
def min_seq (n : Nat) : List Bool :=
  List.replicate (n - 2) false ++ [true, true]

theorem another_max_occurrence_sequence 
  (n : Nat) 
  (h_n : n > 5) 
  (strip : CircularStrip) 
  (h_strip : strip.length > 0) 
  (h_max : ∀ seq : List Bool, seq.length = n → 
    count_occurrences strip seq ≤ count_occurrences strip (max_seq n)) 
  (h_min : count_occurrences strip (min_seq n) < count_occurrences strip (max_seq n)) :
  ∃ seq : List Bool, 
    seq.length = n ∧ 
    seq ≠ max_seq n ∧ 
    count_occurrences strip seq = count_occurrences strip (max_seq n) :=
  sorry

end another_max_occurrence_sequence_l2105_210519


namespace solve_for_y_l2105_210573

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end solve_for_y_l2105_210573


namespace train_crossing_time_l2105_210578

/-- Proves that a train of given length, passing a platform of given length in a given time,
    will take a specific time to cross a tree. -/
theorem train_crossing_time
  (train_length : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_length = 1200)
  (h2 : platform_length = 700)
  (h3 : platform_crossing_time = 190)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 120 :=
by sorry

#check train_crossing_time

end train_crossing_time_l2105_210578


namespace polygon_sides_l2105_210581

theorem polygon_sides (interior_angle : ℝ) (h : interior_angle = 140) :
  (360 : ℝ) / (180 - interior_angle) = 9 := by
  sorry

end polygon_sides_l2105_210581


namespace expression_simplification_l2105_210515

theorem expression_simplification (a b : ℝ) 
  (h : (a - 2)^2 + Real.sqrt (b + 1) = 0) :
  (a^2 - 2*a*b + b^2) / (a^2 - b^2) / ((a^2 - a*b) / a) - 2 / (a + b) = -1 := by
  sorry

end expression_simplification_l2105_210515


namespace employee_salaries_calculation_l2105_210552

/-- Given a total revenue and a ratio for division between employee salaries and stock purchases,
    calculate the amount spent on employee salaries. -/
def calculate_employee_salaries (total_revenue : ℚ) (salary_ratio stock_ratio : ℕ) : ℚ :=
  (salary_ratio : ℚ) / ((salary_ratio : ℚ) + (stock_ratio : ℚ)) * total_revenue

/-- Theorem stating that given a total revenue of 3000 and a division ratio of 4:11
    for employee salaries to stock purchases, the amount spent on employee salaries is 800. -/
theorem employee_salaries_calculation :
  calculate_employee_salaries 3000 4 11 = 800 := by
  sorry

#eval calculate_employee_salaries 3000 4 11

end employee_salaries_calculation_l2105_210552


namespace sum_of_squares_lower_bound_l2105_210521

theorem sum_of_squares_lower_bound (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 := by
sorry

end sum_of_squares_lower_bound_l2105_210521


namespace range_of_a_l2105_210592

theorem range_of_a (a : ℝ) : 
  (∃ x₀ : ℝ, 2 * x₀^2 + (a - 1) * x₀ + 1/2 ≤ 0) ↔ 
  a ≤ -1 ∨ a ≥ 3 :=
by sorry

end range_of_a_l2105_210592


namespace right_triangle_identification_l2105_210504

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  ¬(is_right_triangle 6 15 17) ∧
  ¬(is_right_triangle 7 12 15) ∧
  is_right_triangle 7 24 25 ∧
  ¬(is_right_triangle 13 15 20) :=
by sorry

end right_triangle_identification_l2105_210504


namespace actual_average_height_l2105_210555

/-- Represents the average height calculation problem in a class --/
structure HeightProblem where
  totalStudents : ℕ
  initialAverage : ℚ
  incorrectHeights : List ℚ
  actualHeights : List ℚ

/-- Calculates the actual average height given the problem data --/
def calculateActualAverage (problem : HeightProblem) : ℚ :=
  let initialTotal := problem.initialAverage * problem.totalStudents
  let heightDifference := (problem.incorrectHeights.sum - problem.actualHeights.sum)
  let correctedTotal := initialTotal - heightDifference
  correctedTotal / problem.totalStudents

/-- The theorem stating that the actual average height is 164.5 cm --/
theorem actual_average_height
  (problem : HeightProblem)
  (h1 : problem.totalStudents = 50)
  (h2 : problem.initialAverage = 165)
  (h3 : problem.incorrectHeights = [150, 175, 190])
  (h4 : problem.actualHeights = [135, 170, 185]) :
  calculateActualAverage problem = 164.5 := by
  sorry


end actual_average_height_l2105_210555


namespace sum_x_y_z_l2105_210557

def x : ℕ := (List.range 11).map (· + 30) |>.sum

def y : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 = 0) |>.length

def z : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 ≠ 0) |>.prod

theorem sum_x_y_z : x + y + z = 51768016 := by
  sorry

end sum_x_y_z_l2105_210557


namespace g_satisfies_conditions_l2105_210526

/-- A monic polynomial of degree 3 satisfying specific conditions -/
def g (x : ℝ) : ℝ := x^3 + 4*x^2 + 3*x + 6

/-- Theorem stating that g satisfies the given conditions -/
theorem g_satisfies_conditions :
  (∀ x, g x = x^3 + 4*x^2 + 3*x + 6) ∧
  g 0 = 6 ∧
  g 1 = 14 ∧
  g (-1) = 6 :=
by sorry

end g_satisfies_conditions_l2105_210526


namespace point_outside_circle_l2105_210550

/-- Given a circle with center O and radius 3, and a point P such that OP = 5,
    prove that P is outside the circle. -/
theorem point_outside_circle (O P : ℝ × ℝ) (r : ℝ) (h1 : r = 3) (h2 : dist O P = 5) :
  dist O P > r := by
  sorry

end point_outside_circle_l2105_210550


namespace show_episodes_count_l2105_210599

/-- The number of episodes watched on Mondays each week -/
def monday_episodes : ℕ := 1

/-- The number of episodes watched on Wednesdays each week -/
def wednesday_episodes : ℕ := 2

/-- The number of weeks it takes to watch the whole series -/
def total_weeks : ℕ := 67

/-- The total number of episodes in the show -/
def total_episodes : ℕ := 201

theorem show_episodes_count : 
  monday_episodes + wednesday_episodes * total_weeks = total_episodes := by
  sorry

end show_episodes_count_l2105_210599


namespace smallest_block_volume_l2105_210543

theorem smallest_block_volume (a b c : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 240 → 
  a * b * c ≥ 385 ∧ 
  ∃ (a₀ b₀ c₀ : ℕ), (a₀ - 1) * (b₀ - 1) * (c₀ - 1) = 240 ∧ a₀ * b₀ * c₀ = 385 :=
by sorry

end smallest_block_volume_l2105_210543


namespace parabola_directrix_l2105_210554

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop :=
  y = (x^2 - 8*x + 12) / 16

/-- The directrix equation -/
def directrix_eq (y : ℝ) : Prop :=
  y = -5/4

/-- Theorem: The directrix of the given parabola is y = -5/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_eq x y → ∃ d : ℝ, directrix_eq d ∧ 
  (∀ p q : ℝ, parabola_eq p q → (p - x)^2 + (q - y)^2 = (q - d)^2) :=
sorry

end parabola_directrix_l2105_210554


namespace sine_midpoint_inequality_l2105_210511

theorem sine_midpoint_inequality (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : x₁ < π) (h₃ : 0 < x₂) (h₄ : x₂ < π) (h₅ : x₁ ≠ x₂) : 
  (Real.sin x₁ + Real.sin x₂) / 2 < Real.sin ((x₁ + x₂) / 2) := by
  sorry

end sine_midpoint_inequality_l2105_210511


namespace school_girls_count_l2105_210589

theorem school_girls_count (boys : ℕ) (girls : ℝ) : 
  boys = 387 →
  girls = boys + 0.54 * boys →
  ⌊girls + 0.5⌋ = 596 := by
  sorry

end school_girls_count_l2105_210589


namespace newspaper_conference_max_both_l2105_210509

theorem newspaper_conference_max_both (total : ℕ) (writers : ℕ) (editors : ℕ) (neither : ℕ) (both : ℕ) :
  total = 90 →
  writers = 45 →
  editors > 38 →
  neither = 2 * both →
  total = writers + editors + neither - both →
  both ≤ 4 ∧ (∃ (e : ℕ), editors = 38 + e ∧ both = 4) :=
by sorry

end newspaper_conference_max_both_l2105_210509


namespace logarithm_equation_solution_l2105_210503

theorem logarithm_equation_solution (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (∀ x : ℝ, x > 0 → 5 * (Real.log x / Real.log a)^2 + 2 * (Real.log x / Real.log b)^2 = 
    (10 * (Real.log x)^2) / (Real.log a * Real.log b) + (Real.log x)^2) →
  b = a^(2 / (5 + Real.sqrt 17)) ∨ b = a^(2 / (5 - Real.sqrt 17)) :=
by sorry

end logarithm_equation_solution_l2105_210503


namespace nut_mixture_price_l2105_210529

/-- Calculates the selling price per pound of a nut mixture --/
theorem nut_mixture_price
  (cashew_price : ℝ)
  (brazil_price : ℝ)
  (total_weight : ℝ)
  (cashew_weight : ℝ)
  (h1 : cashew_price = 6.75)
  (h2 : brazil_price = 5.00)
  (h3 : total_weight = 50)
  (h4 : cashew_weight = 20)
  : (cashew_weight * cashew_price + (total_weight - cashew_weight) * brazil_price) / total_weight = 5.70 := by
  sorry

end nut_mixture_price_l2105_210529


namespace function_properties_l2105_210568

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  Real.cos x * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * (Real.sin x)^2

theorem function_properties (a : ℝ) :
  f a (Real.pi / 12) = 0 →
  (∃ T > 0, ∀ x, f a (x + T) = f a x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f a (y + S) ≠ f a y) ∧
  T = Real.pi ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x ≤ Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x ≥ -2) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x = Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc (-Real.pi/6) (Real.pi/4), f a x = -2) :=
by sorry

end function_properties_l2105_210568


namespace max_u_coordinate_is_two_l2105_210500

-- Define the transformation
def transform (x y : ℝ) : ℝ × ℝ :=
  (x^2 + y^2, x - y)

-- Define the unit square vertices
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Define the set of points in the unit square
def unitSquare : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Theorem: The maximum u-coordinate of the transformed unit square is 2
theorem max_u_coordinate_is_two :
  ∃ (p : ℝ × ℝ), p ∈ unitSquare ∧
    (∀ (q : ℝ × ℝ), q ∈ unitSquare →
      (transform p.1 p.2).1 ≥ (transform q.1 q.2).1) ∧
    (transform p.1 p.2).1 = 2 :=
  sorry

end max_u_coordinate_is_two_l2105_210500


namespace range_of_a_l2105_210534

-- Define the sets A and B
def A : Set ℝ := {x | x < 3}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∪ B a = Set.univ → a < 3 := by
  sorry

end range_of_a_l2105_210534


namespace absolute_value_equation_solution_difference_l2105_210588

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ + 3| = 15) ∧ 
  (|x₂ + 3| = 15) ∧ 
  (x₁ ≠ x₂) ∧
  (x₁ - x₂ = 30 ∨ x₂ - x₁ = 30) := by
  sorry

end absolute_value_equation_solution_difference_l2105_210588


namespace min_sum_of_squares_min_sum_of_squares_value_l2105_210544

theorem min_sum_of_squares (a b c d : ℤ) : 
  a^2 ≠ b^2 → a^2 ≠ c^2 → a^2 ≠ d^2 → b^2 ≠ c^2 → b^2 ≠ d^2 → c^2 ≠ d^2 →
  (a*b + c*d)^2 + (a*d - b*c)^2 = 2004 →
  ∀ (w x y z : ℤ), w^2 ≠ x^2 → w^2 ≠ y^2 → w^2 ≠ z^2 → x^2 ≠ y^2 → x^2 ≠ z^2 → y^2 ≠ z^2 →
  (w*x + y*z)^2 + (w*z - x*y)^2 = 2004 →
  a^2 + b^2 + c^2 + d^2 ≤ w^2 + x^2 + y^2 + z^2 :=
by sorry

theorem min_sum_of_squares_value (a b c d : ℤ) : 
  a^2 ≠ b^2 → a^2 ≠ c^2 → a^2 ≠ d^2 → b^2 ≠ c^2 → b^2 ≠ d^2 → c^2 ≠ d^2 →
  (a*b + c*d)^2 + (a*d - b*c)^2 = 2004 →
  ∃ (x y : ℤ), x^2 + y^2 = 2004 ∧ a^2 + b^2 + c^2 + d^2 = 2 * (x + y) :=
by sorry

end min_sum_of_squares_min_sum_of_squares_value_l2105_210544


namespace quadratic_function_properties_l2105_210539

-- Define the quadratic function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 15

-- Define g(x) in terms of f(x) and a
def g (a x : ℝ) : ℝ := (2 - 2*a)*x - f x

-- Theorem statement
theorem quadratic_function_properties :
  -- f(x) has vertex (1, 16)
  (f 1 = 16 ∧ ∀ x, f x ≤ f 1) ∧
  -- The roots of f(x) are 8 units apart
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 8) →
  -- 1. f(x) = -x^2 + 2x + 15
  (∀ x, f x = -x^2 + 2*x + 15) ∧
  -- 2. g(x) is monotonically increasing on [0, 2] iff a ≤ 0
  (∀ a, (∀ x₁ x₂, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2 → g a x₁ < g a x₂) ↔ a ≤ 0) ∧
  -- 3. Minimum value of g(x) on [0, 2]
  (∀ a, (∃ m, ∀ x, 0 ≤ x ∧ x ≤ 2 → m ≤ g a x ∧
    ((a > 2 → m = -4*a - 11) ∧
     (a < 0 → m = -15) ∧
     (0 ≤ a ∧ a ≤ 2 → m = -a^2 - 15)))) :=
by sorry

end quadratic_function_properties_l2105_210539


namespace sin_540_plus_alpha_implies_cos_alpha_minus_270_l2105_210575

theorem sin_540_plus_alpha_implies_cos_alpha_minus_270
  (α : Real)
  (h : Real.sin (540 * Real.pi / 180 + α) = -4/5) :
  Real.cos (α - 270 * Real.pi / 180) = -4/5 := by
  sorry

end sin_540_plus_alpha_implies_cos_alpha_minus_270_l2105_210575


namespace function_inequality_l2105_210553

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x > 0, deriv f x + f x / x > 0) :
  ∀ a b, a > 0 → b > 0 → a > b → a * f a > b * f b := by
  sorry

end function_inequality_l2105_210553


namespace stratified_sampling_third_year_l2105_210541

theorem stratified_sampling_third_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (h1 : total_students = 2400) 
  (h2 : sample_size = 120) 
  (h3 : first_year = 760) 
  (h4 : second_year = 840) : 
  (total_students - first_year - second_year) * sample_size / total_students = 40 := by
  sorry

end stratified_sampling_third_year_l2105_210541


namespace cos_A_right_triangle_l2105_210514

theorem cos_A_right_triangle (adjacent hypotenuse : ℝ) 
  (h1 : adjacent = 5)
  (h2 : hypotenuse = 13)
  (h3 : adjacent > 0)
  (h4 : hypotenuse > 0)
  (h5 : adjacent < hypotenuse) : 
  Real.cos (Real.arccos (adjacent / hypotenuse)) = 5 / 13 := by
sorry

end cos_A_right_triangle_l2105_210514


namespace perpendicular_vector_implies_y_equals_five_l2105_210545

/-- Given points A and B, and vector a, proves that if AB is perpendicular to a, then y = 5 -/
theorem perpendicular_vector_implies_y_equals_five (A B : ℝ × ℝ) (a : ℝ × ℝ) :
  A = (10, 1) →
  B.1 = 2 →
  a = (1, 2) →
  (B.1 - A.1) * a.1 + (B.2 - A.2) * a.2 = 0 →
  B.2 = 5 := by
  sorry

end perpendicular_vector_implies_y_equals_five_l2105_210545


namespace sin_pi_minus_alpha_l2105_210502

theorem sin_pi_minus_alpha (α : Real) :
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ x = r * Real.cos α ∧ y = r * Real.sin α ∧ r > 0) →
  Real.sin (Real.pi - α) = 3/5 := by
  sorry

end sin_pi_minus_alpha_l2105_210502


namespace largest_radius_is_61_l2105_210533

/-- A circle containing specific points and the unit circle -/
structure SpecialCircle where
  center : ℝ × ℝ
  radius : ℝ
  contains_points : center.1^2 + 11^2 = radius^2
  contains_unit_circle : ∀ (x y : ℝ), x^2 + y^2 < 1 → 
    (x - center.1)^2 + (y - center.2)^2 < radius^2

/-- The largest possible radius of a SpecialCircle is 61 -/
theorem largest_radius_is_61 : 
  (∃ (c : SpecialCircle), true) → 
  (∀ (c : SpecialCircle), c.radius ≤ 61) ∧ 
  (∃ (c : SpecialCircle), c.radius = 61) :=
sorry

end largest_radius_is_61_l2105_210533


namespace profit_percentage_calculation_l2105_210567

theorem profit_percentage_calculation (cost_price selling_price : ℝ) :
  cost_price = 500 ∧ selling_price = 800 →
  (selling_price - cost_price) / cost_price * 100 = 60 := by
  sorry

end profit_percentage_calculation_l2105_210567


namespace prove_initial_stock_l2105_210590

-- Define the total number of books sold
def books_sold : ℕ := 272

-- Define the percentage of books sold as a rational number
def percentage_sold : ℚ := 19.42857142857143 / 100

-- Define the initial stock of books
def initial_stock : ℕ := 1400

-- Theorem statement
theorem prove_initial_stock : 
  (books_sold : ℚ) / initial_stock = percentage_sold :=
by sorry

end prove_initial_stock_l2105_210590


namespace triangle_inequality_l2105_210561

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b + c)^2 < 4 * (a * b + b * c + c * a) := by
  sorry

end triangle_inequality_l2105_210561


namespace range_of_a_l2105_210586

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  2 * x + y ≥ 4 ∧ x - y ≥ 1 ∧ x - 2 * y ≤ 2

-- Define the function z
def z (a x y : ℝ) : ℝ := a * x + y

-- Define the minimum point
def min_point : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ,
  (∀ x y : ℝ, feasible_region x y → z a x y ≥ z a (min_point.1) (min_point.2)) →
  (∃ x y : ℝ, feasible_region x y ∧ z a x y = z a (min_point.1) (min_point.2) → (x, y) = min_point) →
  -1/2 < a ∧ a < 2 :=
sorry

end range_of_a_l2105_210586


namespace linear_equation_implies_k_equals_one_l2105_210591

/-- A function that represents the linearity condition of an equation -/
def is_linear_equation (k : ℝ) : Prop :=
  (k + 1 ≠ 0) ∧ (|k| = 1)

/-- Theorem stating that if (k+1)x + 8y^|k| + 3 = 0 is a linear equation in x and y, then k = 1 -/
theorem linear_equation_implies_k_equals_one :
  is_linear_equation k → k = 1 := by sorry

end linear_equation_implies_k_equals_one_l2105_210591


namespace printer_task_time_l2105_210577

/-- Given two printers A and B, this theorem proves the time taken to complete a task together -/
theorem printer_task_time (pages : ℕ) (time_A : ℕ) (rate_diff : ℕ) : 
  pages = 480 → 
  time_A = 60 → 
  rate_diff = 4 → 
  (pages : ℚ) / ((pages : ℚ) / time_A + ((pages : ℚ) / time_A + rate_diff)) = 24 := by
  sorry

#check printer_task_time

end printer_task_time_l2105_210577


namespace fraction_subtraction_l2105_210523

theorem fraction_subtraction :
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by sorry

end fraction_subtraction_l2105_210523


namespace smallest_a_l2105_210527

-- Define the polynomial
def P (a b x : ℤ) : ℤ := x^3 - a*x^2 + b*x - 1806

-- Define the property of having three positive integer roots
def has_three_positive_integer_roots (a b : ℤ) : Prop :=
  ∃ (x y z : ℤ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  P a b x = 0 ∧ P a b y = 0 ∧ P a b z = 0

-- State the theorem
theorem smallest_a :
  ∃ (a : ℤ), has_three_positive_integer_roots a (a*56 - 1806) ∧
  (∀ (a' : ℤ), has_three_positive_integer_roots a' (a'*56 - 1806) → a ≤ a') :=
sorry

end smallest_a_l2105_210527


namespace doctor_selection_ways_l2105_210572

/-- The number of ways to choose a team of doctors from internists and surgeons --/
def choose_doctors (internists surgeons team_size : ℕ) : ℕ :=
  Nat.choose (internists + surgeons) team_size -
  (Nat.choose internists team_size + Nat.choose surgeons team_size)

/-- Theorem stating the number of ways to choose 4 doctors from 5 internists and 6 surgeons --/
theorem doctor_selection_ways :
  choose_doctors 5 6 4 = 310 := by
  sorry

end doctor_selection_ways_l2105_210572


namespace eight_last_to_appear_l2105_210530

def tribonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 3
  | 2 => 4
  | n + 3 => tribonacci n + tribonacci (n + 1) + tribonacci (n + 2)

def lastDigit (n : ℕ) : ℕ := n % 10

def digitAppears (d : ℕ) (n : ℕ) : Prop :=
  ∃ k, k ≤ n ∧ lastDigit (tribonacci k) = d

theorem eight_last_to_appear :
  ∃ N, ∀ n, n ≥ N → 
    (∀ d, d ≠ 8 → digitAppears d n) ∧
    ¬(digitAppears 8 n) ∧
    digitAppears 8 (n + 1) := by sorry

end eight_last_to_appear_l2105_210530


namespace inverse_proportion_problem_l2105_210538

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h_prop : InverselyProportional x y) 
  (h_init : x 8 = 40 ∧ y 8 = 8) :
  x 10 = 32 ∧ y 10 = 10 := by
  sorry

end inverse_proportion_problem_l2105_210538


namespace no_two_right_angles_l2105_210585

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_180 : A + B + C = 180

-- Define a right angle
def is_right_angle (angle : ℝ) : Prop := angle = 90

-- Theorem statement
theorem no_two_right_angles (t : Triangle) : 
  ¬(is_right_angle t.A ∧ is_right_angle t.B) ∧ 
  ¬(is_right_angle t.B ∧ is_right_angle t.C) ∧ 
  ¬(is_right_angle t.A ∧ is_right_angle t.C) :=
sorry

end no_two_right_angles_l2105_210585


namespace f_comp_three_roots_l2105_210571

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- The number of distinct real roots of f_comp -/
noncomputable def num_distinct_roots (c : ℝ) : ℕ := sorry

theorem f_comp_three_roots :
  ∀ c : ℝ, num_distinct_roots c = 3 ↔ c = (11 - Real.sqrt 13) / 2 :=
by sorry

end f_comp_three_roots_l2105_210571


namespace smallest_consecutive_integer_l2105_210563

theorem smallest_consecutive_integer (a b c d : ℕ) : 
  a > 0 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ a * b * c * d = 1680 → a = 5 :=
by sorry

end smallest_consecutive_integer_l2105_210563


namespace valid_assignment_probability_l2105_210562

/-- A regular dodecahedron with 12 numbered faces -/
structure NumberedDodecahedron :=
  (assignment : Fin 12 → Fin 12)
  (injective : Function.Injective assignment)

/-- Two numbers are consecutive if they differ by 1 or are 1 and 12 -/
def consecutive (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 0 ∧ b = 11) ∨ (a = 11 ∧ b = 0)

/-- The set of all possible numbered dodecahedrons -/
def allAssignments : Finset NumberedDodecahedron := sorry

/-- The set of valid assignments where no consecutive numbers are on adjacent faces -/
def validAssignments : Finset NumberedDodecahedron := sorry

/-- The probability of a valid assignment -/
def validProbability : ℚ := (validAssignments.card : ℚ) / (allAssignments.card : ℚ)

/-- The main theorem stating that the probability is 1/100 -/
theorem valid_assignment_probability :
  validProbability = 1 / 100 := by sorry

end valid_assignment_probability_l2105_210562


namespace tan_double_angle_l2105_210510

theorem tan_double_angle (x : ℝ) (h : Real.tan (Real.pi - x) = 3 / 4) : 
  Real.tan (2 * x) = -24 / 7 := by
  sorry

end tan_double_angle_l2105_210510


namespace identity_function_satisfies_equation_l2105_210516

theorem identity_function_satisfies_equation (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) →
  (∀ x : ℝ, f x = x) :=
by sorry

end identity_function_satisfies_equation_l2105_210516
